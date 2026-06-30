//! Write-ahead log for crash recovery.
//!
//! Each operation is appended to the WAL (and fsynced) *before* it is applied
//! to a dataset's segment file. A separate checkpoint file records the highest
//! sequence number that has been durably applied to a segment. On startup,
//! entries with `seq > checkpoint` are replayed (idempotently, deduped by id),
//! then the WAL is truncated.
//!
//! WAL entry layout: `[u32 entry_len][u64 seq][u32 dataset_len][dataset][frame_payload]`.

use crate::format::{decode_frame, encode_frame, Frame};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use tensorus_core::error::{Result, TensorusError};

/// A replayable WAL entry.
pub(crate) struct WalEntry {
    pub seq: u64,
    pub dataset: String,
    pub frame: Frame,
}

/// Append-only write-ahead log plus its checkpoint sidecar.
pub(crate) struct Wal {
    file: File,
    checkpoint_path: PathBuf,
}

impl Wal {
    /// Open (creating if needed) the WAL and checkpoint files under `wal_dir`.
    pub fn open(wal_dir: &Path) -> Result<Wal> {
        std::fs::create_dir_all(wal_dir)?;
        let path = wal_dir.join("wal.log");
        let checkpoint_path = wal_dir.join("checkpoint");
        // Read+write (not append): Windows rejects `set_len` on an append-only
        // handle, and recovery needs to truncate the WAL. `truncate(false)`
        // preserves existing entries on open so they can be replayed. Appends
        // seek to end.
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)?;
        Ok(Wal {
            file,
            checkpoint_path,
        })
    }

    /// Append an entry to the OS (write-through, no fsync). The bytes survive a
    /// process crash via the OS page cache; [`Wal::sync`] forces them to stable
    /// storage for power-loss durability (group commit).
    pub fn append(&mut self, seq: u64, dataset: &str, frame: &Frame) -> Result<()> {
        let payload = encode_frame(frame)?;
        let ds = dataset.as_bytes();
        let entry_len = 8 + 4 + ds.len() + payload.len();
        let mut buf = Vec::with_capacity(4 + entry_len);
        buf.extend_from_slice(&(entry_len as u32).to_le_bytes());
        buf.extend_from_slice(&seq.to_le_bytes());
        buf.extend_from_slice(&(ds.len() as u32).to_le_bytes());
        buf.extend_from_slice(ds);
        buf.extend_from_slice(&payload);
        self.file.seek(SeekFrom::End(0))?;
        self.file.write_all(&buf)?;
        Ok(())
    }

    /// Force previously appended entries to stable storage.
    pub fn sync(&mut self) -> Result<()> {
        self.file.flush()?;
        self.file
            .sync_data()
            .map_err(|e| TensorusError::Storage(format!("wal fsync failed: {e}")))?;
        Ok(())
    }

    /// Read every complete entry, ignoring a torn trailing entry.
    pub fn scan(&mut self) -> Result<Vec<WalEntry>> {
        self.file.seek(SeekFrom::Start(0))?;
        let mut bytes = Vec::new();
        self.file.read_to_end(&mut bytes)?;
        let mut entries = Vec::new();
        let mut pos = 0usize;
        while pos + 4 <= bytes.len() {
            let entry_len =
                u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
                    as usize;
            let body_start = pos + 4;
            let body_end = match body_start.checked_add(entry_len) {
                Some(e) => e,
                None => break,
            };
            if body_end > bytes.len() {
                break; // torn tail
            }
            let body = &bytes[body_start..body_end];
            if body.len() < 12 {
                break;
            }
            let seq = u64::from_le_bytes([
                body[0], body[1], body[2], body[3], body[4], body[5], body[6], body[7],
            ]);
            let ds_len = u32::from_le_bytes([body[8], body[9], body[10], body[11]]) as usize;
            let ds_start = 12;
            let ds_end = ds_start + ds_len;
            if ds_end > body.len() {
                break;
            }
            let dataset = match std::str::from_utf8(&body[ds_start..ds_end]) {
                Ok(s) => s.to_string(),
                Err(_) => break,
            };
            match decode_frame(&body[ds_end..]) {
                Ok(frame) => entries.push(WalEntry {
                    seq,
                    dataset,
                    frame,
                }),
                Err(_) => break,
            }
            pos = body_end;
        }
        Ok(entries)
    }

    /// Read the durable checkpoint sequence (0 if none yet).
    pub fn read_checkpoint(&self) -> Result<u64> {
        match std::fs::read(&self.checkpoint_path) {
            Ok(b) if b.len() >= 8 => Ok(u64::from_le_bytes([
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
            ])),
            Ok(_) => Ok(0),
            Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => Ok(0),
            Err(e) => Err(e.into()),
        }
    }

    /// Atomically advance the checkpoint (write to a temp file then rename).
    pub fn write_checkpoint(&self, seq: u64) -> Result<()> {
        let tmp = self.checkpoint_path.with_extension("tmp");
        {
            let mut f = File::create(&tmp)?;
            f.write_all(&seq.to_le_bytes())?;
            f.sync_data()
                .map_err(|e| TensorusError::Storage(format!("checkpoint fsync failed: {e}")))?;
        }
        std::fs::rename(&tmp, &self.checkpoint_path)?;
        Ok(())
    }

    /// Empty the WAL and reset the checkpoint. Called once recovery has folded
    /// all pending entries into the segments.
    pub fn truncate(&mut self) -> Result<()> {
        self.file.set_len(0)?;
        self.file.seek(SeekFrom::Start(0))?;
        self.file.sync_all().ok();
        self.write_checkpoint(0)?;
        Ok(())
    }
}
