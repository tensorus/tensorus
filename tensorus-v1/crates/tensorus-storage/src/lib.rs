//! # tensorus-storage
//!
//! Durable storage engine implementing [`tensorus_core::Storage`].
//!
//! ## Design note: Lance substitution
//!
//! The plan specifies the [Lance](https://lancedb.github.io/lance/) columnar
//! format for the warm tier. Lance pulls in DataFusion, `object_store`, a large
//! Arrow surface, and a build-time `protoc` dependency; building it reliably on
//! the `x86_64-pc-windows-gnu` toolchain available in this environment is high
//! risk and very slow. Following the plan's guidance ("If blocked on a
//! dependency ... implement a simpler alternative that maintains the same
//! interface"), this crate provides a self-contained, crash-safe storage engine
//! behind the identical [`Storage`] trait:
//!
//! - **Append-only segment files** per dataset (`{data_dir}/datasets/{name}/segment.dat`),
//!   the durable record of committed data — conceptually the role Lance plays.
//! - **Write-ahead log** (the `wal` module) with checkpointing for crash recovery.
//! - **In-memory hot map** serving reads (the "hot tier"); the on-disk segment
//!   is the source of truth on restart.
//!
//! The trait boundary is unchanged, so a Lance-backed implementation can be
//! dropped in later without touching callers.

#![forbid(unsafe_code)]

pub mod compression;
mod format;
mod segment;
pub mod tiering;
mod wal;

pub use compression::{compress, compress_with, decompress, Codec, Compressed};
pub use tiering::{ColdStore, LocalColdStore, Policy, TieredStore, TieringConfig};

use crate::format::{Frame, StoredRecord};
use crate::wal::Wal;
use async_trait::async_trait;
use chrono::{TimeZone, Utc};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tensorus_core::error::{Result, TensorusError};
use tensorus_core::traits::Storage;
use tensorus_core::types::{Metadata, TensorDescriptor, TensorId, TensorRecord};

/// Force an fsync after this many buffered write operations (group commit).
const SYNC_EVERY: u64 = 512;

/// A replicated operation from the change-data-capture log, in commit order by
/// `seq`. Used for single-leader async replication: a follower pulls ops with
/// [`FileStorage::changes_since`] and applies them via
/// [`FileStorage::apply_replicated`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplOp {
    /// Insert/overwrite a record (carries the full record, including its id).
    Put { seq: u64, record: Box<TensorRecord> },
    /// Delete a record by id.
    Delete {
        seq: u64,
        dataset: String,
        id: TensorId,
    },
}

impl ReplOp {
    /// The commit sequence number of this operation.
    pub fn seq(&self) -> u64 {
        match self {
            ReplOp::Put { seq, .. } | ReplOp::Delete { seq, .. } => *seq,
        }
    }
}

/// In-memory + on-disk state for a single dataset.
struct Dataset {
    seg: File,
    records: HashMap<TensorId, StoredRecord>,
    order: Vec<TensorId>,
}

impl Dataset {
    fn apply_put(&mut self, rec: StoredRecord) -> Result<()> {
        segment::write_frame(&mut self.seg, &Frame::Put(rec.clone()))?;
        if self.records.insert(rec.id, rec.clone()).is_none() {
            self.order.push(rec.id);
        }
        Ok(())
    }

    fn apply_del(&mut self, id: TensorId) -> Result<bool> {
        segment::write_frame(&mut self.seg, &Frame::Del(id))?;
        let existed = self.records.remove(&id).is_some();
        if existed {
            self.order.retain(|x| *x != id);
        }
        Ok(existed)
    }

    fn sync(&mut self) -> Result<()> {
        use std::io::Write;
        self.seg.flush()?;
        self.seg
            .sync_data()
            .map_err(|e| TensorusError::Storage(format!("segment fsync failed: {e}")))?;
        Ok(())
    }
}

struct Inner {
    datasets_dir: PathBuf,
    datasets: RwLock<HashMap<String, Arc<Mutex<Dataset>>>>,
    wal: Mutex<Wal>,
    /// Append-only change-data-capture log for replication (never truncated).
    replog: Mutex<Wal>,
    /// Serializes writers so WAL sequence order matches checkpoint order.
    write_lock: Mutex<()>,
    next_seq: AtomicU64,
    unsynced: AtomicU64,
}

/// A durable, crash-safe tensor store.
#[derive(Clone)]
pub struct FileStorage {
    inner: Arc<Inner>,
}

fn open_segment(dir: &Path) -> Result<File> {
    let path = dir.join("segment.dat");
    let f = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(&path)?;
    Ok(f)
}

fn now_us() -> i64 {
    Utc::now().timestamp_micros()
}

fn stored_to_record(rec: &StoredRecord, dataset: &str) -> TensorRecord {
    let created_at = Utc
        .timestamp_micros(rec.created_at_us)
        .single()
        .unwrap_or_else(Utc::now);
    TensorRecord {
        id: rec.id,
        dataset: dataset.to_string(),
        descriptor: rec.descriptor.clone(),
        metadata: rec.metadata.clone(),
        created_at,
        version: rec.version,
        data: rec.data.clone(),
    }
}

impl FileStorage {
    /// Open (or create) a store rooted at `data_dir` with its WAL under
    /// `wal_dir`, replaying any pending WAL entries.
    pub fn open(data_dir: impl AsRef<Path>, wal_dir: impl AsRef<Path>) -> Result<Self> {
        let datasets_dir = data_dir.as_ref().join("datasets");
        std::fs::create_dir_all(&datasets_dir)?;

        let mut datasets = Self::load_datasets(&datasets_dir)?;
        let mut wal = Wal::open(wal_dir.as_ref())?;

        // Replay WAL entries that were not yet folded into the segments.
        let checkpoint = wal.read_checkpoint()?;
        let entries = wal.scan()?;
        let mut max_seq = checkpoint;
        for entry in &entries {
            max_seq = max_seq.max(entry.seq);
            if entry.seq <= checkpoint {
                continue;
            }
            let ds = datasets
                .entry(entry.dataset.clone())
                .or_insert_with(|| {
                    let dir = datasets_dir.join(&entry.dataset);
                    let _ = std::fs::create_dir_all(&dir);
                    Arc::new(Mutex::new(Dataset {
                        seg: open_segment(&dir).expect("open segment during recovery"),
                        records: HashMap::new(),
                        order: Vec::new(),
                    }))
                })
                .clone();
            let mut guard = ds.lock();
            match &entry.frame {
                Frame::Put(rec) => {
                    // Idempotent: if the segment already had this record (crash
                    // after segment write but before checkpoint), skip.
                    if !guard.records.contains_key(&rec.id) {
                        guard.apply_put(rec.clone())?;
                    }
                }
                Frame::Del(id) => {
                    if guard.records.contains_key(id) {
                        guard.apply_del(*id)?;
                    }
                }
            }
        }

        // Fold complete: durably flush segments, then reset the WAL.
        for ds in datasets.values() {
            ds.lock().sync()?;
        }
        wal.truncate()?;

        // The replication log lives alongside the data and is never truncated.
        let replog = Wal::open(&data_dir.as_ref().join("replog"))?;

        let inner = Inner {
            datasets_dir,
            datasets: RwLock::new(datasets),
            wal: Mutex::new(wal),
            replog: Mutex::new(replog),
            write_lock: Mutex::new(()),
            next_seq: AtomicU64::new(max_seq + 1),
            unsynced: AtomicU64::new(0),
        };
        Ok(FileStorage {
            inner: Arc::new(inner),
        })
    }

    fn load_datasets(datasets_dir: &Path) -> Result<HashMap<String, Arc<Mutex<Dataset>>>> {
        let mut map = HashMap::new();
        for entry in std::fs::read_dir(datasets_dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            let dir = entry.path();
            let seg_path = dir.join("segment.dat");
            let bytes = std::fs::read(&seg_path).unwrap_or_default();
            let scan = segment::scan_segment(&bytes);
            // Truncate a torn trailing frame if present.
            if scan.valid_len < bytes.len() as u64 {
                let f = OpenOptions::new().write(true).open(&seg_path)?;
                f.set_len(scan.valid_len)?;
            }
            let mut records = HashMap::new();
            let mut order = Vec::new();
            for frame in scan.frames {
                match frame {
                    Frame::Put(rec) => {
                        let id = rec.id;
                        if records.insert(id, rec).is_none() {
                            order.push(id);
                        }
                    }
                    Frame::Del(id) => {
                        if records.remove(&id).is_some() {
                            order.retain(|x| *x != id);
                        }
                    }
                }
            }
            let seg = open_segment(&dir)?;
            map.insert(
                name,
                Arc::new(Mutex::new(Dataset {
                    seg,
                    records,
                    order,
                })),
            );
        }
        Ok(map)
    }

    fn get_or_create_dataset(&self, name: &str) -> Result<Arc<Mutex<Dataset>>> {
        if let Some(ds) = self.inner.datasets.read().get(name) {
            return Ok(ds.clone());
        }
        let mut w = self.inner.datasets.write();
        if let Some(ds) = w.get(name) {
            return Ok(ds.clone());
        }
        let dir = self.inner.datasets_dir.join(name);
        std::fs::create_dir_all(&dir)?;
        let ds = Arc::new(Mutex::new(Dataset {
            seg: open_segment(&dir)?,
            records: HashMap::new(),
            order: Vec::new(),
        }));
        w.insert(name.to_string(), ds.clone());
        Ok(ds)
    }

    /// Force all buffered writes to stable storage and advance the checkpoint.
    /// Durability is guaranteed for everything inserted before this returns.
    pub fn flush(&self) -> Result<()> {
        let _g = self.inner.write_lock.lock();
        self.sync_all_locked()
    }

    /// Create a consistent on-disk snapshot of every dataset under `dest`.
    ///
    /// Holds the writer lock so no insert/delete can interleave, durably flushes,
    /// then copies each dataset's segment file to
    /// `{dest}/datasets/{name}/segment.dat`. Reads are not blocked. The snapshot
    /// is self-contained: restore by opening it with [`FileStorage::open`]
    /// (pointing at a fresh, empty WAL directory).
    pub fn snapshot(&self, dest: impl AsRef<Path>) -> Result<()> {
        let dest = dest.as_ref();
        let dest_datasets = dest.join("datasets");
        std::fs::create_dir_all(&dest_datasets)?;
        // Hold the writer lock across flush + copy for a consistent point-in-time
        // image (writers block briefly; readers are unaffected).
        let _g = self.inner.write_lock.lock();
        self.sync_all_locked()?;
        let names: Vec<String> = self.inner.datasets.read().keys().cloned().collect();
        for name in names {
            let src = self.inner.datasets_dir.join(&name).join("segment.dat");
            let dst_dir = dest_datasets.join(&name);
            std::fs::create_dir_all(&dst_dir)?;
            if src.exists() {
                std::fs::copy(&src, dst_dir.join("segment.dat"))?;
            }
        }
        Ok(())
    }

    /// Replication: return committed operations with `seq > since`, in commit
    /// order, up to `limit`. Followers poll this to stay current. (Scans the
    /// change-log; a production deployment would index it by `seq`.)
    pub fn changes_since(&self, since: u64, limit: usize) -> Result<Vec<ReplOp>> {
        let entries = self.inner.replog.lock().scan()?;
        let mut out = Vec::new();
        for e in entries {
            if e.seq <= since {
                continue;
            }
            let op = match e.frame {
                Frame::Put(rec) => ReplOp::Put {
                    seq: e.seq,
                    record: Box::new(stored_to_record(&rec, &e.dataset)),
                },
                Frame::Del(id) => ReplOp::Delete {
                    seq: e.seq,
                    dataset: e.dataset,
                    id,
                },
            };
            out.push(op);
            if out.len() >= limit {
                break;
            }
        }
        Ok(out)
    }

    /// The highest sequence number recorded in the change-log (a follower's
    /// target), or 0 if empty.
    pub fn replication_head(&self) -> Result<u64> {
        Ok(self
            .inner
            .replog
            .lock()
            .scan()?
            .iter()
            .map(|e| e.seq)
            .max()
            .unwrap_or(0))
    }

    /// Replication: apply an operation received from a leader, preserving the
    /// original id and sequence. Idempotent re-application is safe. Does not
    /// re-record into this node's change-log (followers are leaves).
    pub fn apply_replicated(&self, op: ReplOp) -> Result<()> {
        match op {
            ReplOp::Put { seq, record } => {
                let record = *record;
                let dataset = record.dataset.clone();
                let ds = self.get_or_create_dataset(&dataset)?;
                let _g = self.inner.write_lock.lock();
                let stored = StoredRecord {
                    id: record.id,
                    created_at_us: record.created_at.timestamp_micros(),
                    version: record.version,
                    descriptor: record.descriptor,
                    metadata: record.metadata,
                    data: record.data,
                };
                self.inner
                    .wal
                    .lock()
                    .append(seq, &dataset, &Frame::Put(stored.clone()))?;
                ds.lock().apply_put(stored)?;
                self.inner.next_seq.fetch_max(seq + 1, Ordering::SeqCst);
                self.maybe_sync()?;
            }
            ReplOp::Delete { seq, dataset, id } => {
                let ds = self.get_or_create_dataset(&dataset)?;
                let _g = self.inner.write_lock.lock();
                self.inner
                    .wal
                    .lock()
                    .append(seq, &dataset, &Frame::Del(id))?;
                ds.lock().apply_del(id)?;
                self.inner.next_seq.fetch_max(seq + 1, Ordering::SeqCst);
                self.maybe_sync()?;
            }
        }
        Ok(())
    }

    /// Insert a full record preserving its id (used by restore/import). Assigns a
    /// fresh local sequence and records it to the change-log so followers also
    /// receive the restored data.
    pub fn ingest(&self, record: TensorRecord) -> Result<()> {
        let dataset = record.dataset.clone();
        let ds = self.get_or_create_dataset(&dataset)?;
        let _g = self.inner.write_lock.lock();
        let seq = self.inner.next_seq.fetch_add(1, Ordering::SeqCst);
        let stored = StoredRecord {
            id: record.id,
            created_at_us: record.created_at.timestamp_micros(),
            version: record.version,
            descriptor: record.descriptor,
            metadata: record.metadata,
            data: record.data,
        };
        let put = Frame::Put(stored.clone());
        self.inner.wal.lock().append(seq, &dataset, &put)?;
        self.inner.replog.lock().append(seq, &dataset, &put)?;
        ds.lock().apply_put(stored)?;
        self.maybe_sync()?;
        Ok(())
    }

    /// Remove a dataset entirely: drop its in-memory state and delete its
    /// on-disk directory. (Per-tensor deletes are what propagate to followers;
    /// the directory drop is a local operation.)
    pub fn delete_dataset(&self, name: &str) -> Result<()> {
        let removed = self.inner.datasets.write().remove(name);
        if removed.is_some() {
            let dir = self.inner.datasets_dir.join(name);
            let _ = std::fs::remove_dir_all(&dir);
        }
        Ok(())
    }

    fn sync_all_locked(&self) -> Result<()> {
        for ds in self.inner.datasets.read().values() {
            ds.lock().sync()?;
        }
        {
            let mut wal = self.inner.wal.lock();
            wal.sync()?;
            let committed = self.inner.next_seq.load(Ordering::SeqCst).saturating_sub(1);
            wal.write_checkpoint(committed)?;
        }
        self.inner.replog.lock().sync()?;
        self.inner.unsynced.store(0, Ordering::SeqCst);
        Ok(())
    }

    fn maybe_sync(&self) -> Result<()> {
        if self.inner.unsynced.fetch_add(1, Ordering::SeqCst) + 1 >= SYNC_EVERY {
            self.sync_all_locked()?;
        }
        Ok(())
    }
}

#[async_trait]
impl Storage for FileStorage {
    async fn insert(
        &self,
        dataset: &str,
        data: &[u8],
        mut descriptor: TensorDescriptor,
        metadata: Metadata,
    ) -> Result<TensorId> {
        let ds = self.get_or_create_dataset(dataset)?;
        let _g = self.inner.write_lock.lock();

        let id = TensorId::new();
        descriptor.tensor_id = id;
        let rec = StoredRecord {
            id,
            created_at_us: now_us(),
            version: 1,
            descriptor,
            metadata,
            data: data.to_vec(),
        };

        let seq = self.inner.next_seq.fetch_add(1, Ordering::SeqCst);
        // WAL first, then segment (write-ahead ordering); also record in the
        // replication change-log.
        let put = Frame::Put(rec.clone());
        self.inner.wal.lock().append(seq, dataset, &put)?;
        self.inner.replog.lock().append(seq, dataset, &put)?;
        ds.lock().apply_put(rec)?;
        // Group-commit check runs while the writer lock is still held so the
        // checkpoint never outruns a segment write that is still in flight.
        self.maybe_sync()?;
        Ok(id)
    }

    async fn get(&self, dataset: &str, id: TensorId) -> Result<TensorRecord> {
        let ds = self
            .inner
            .datasets
            .read()
            .get(dataset)
            .cloned()
            .ok_or_else(|| TensorusError::NotFound(format!("dataset '{dataset}'")))?;
        let guard = ds.lock();
        guard
            .records
            .get(&id)
            .map(|r| stored_to_record(r, dataset))
            .ok_or_else(|| TensorusError::NotFound(format!("tensor '{id}' in '{dataset}'")))
    }

    async fn scan(&self, dataset: &str, limit: usize, offset: usize) -> Result<Vec<TensorRecord>> {
        let ds = self
            .inner
            .datasets
            .read()
            .get(dataset)
            .cloned()
            .ok_or_else(|| TensorusError::NotFound(format!("dataset '{dataset}'")))?;
        let guard = ds.lock();
        let out = guard
            .order
            .iter()
            .skip(offset)
            .take(limit)
            .filter_map(|id| guard.records.get(id).map(|r| stored_to_record(r, dataset)))
            .collect();
        Ok(out)
    }

    async fn delete(&self, dataset: &str, id: TensorId) -> Result<()> {
        let ds = match self.inner.datasets.read().get(dataset).cloned() {
            Some(d) => d,
            None => return Ok(()),
        };
        let _g = self.inner.write_lock.lock();
        let seq = self.inner.next_seq.fetch_add(1, Ordering::SeqCst);
        let del = Frame::Del(id);
        self.inner.wal.lock().append(seq, dataset, &del)?;
        self.inner.replog.lock().append(seq, dataset, &del)?;
        ds.lock().apply_del(id)?;
        self.maybe_sync()?;
        Ok(())
    }

    async fn create_dataset(&self, name: &str) -> Result<()> {
        self.get_or_create_dataset(name)?;
        Ok(())
    }

    async fn list_datasets(&self) -> Result<Vec<String>> {
        let mut names: Vec<String> = self.inner.datasets.read().keys().cloned().collect();
        names.sort();
        Ok(names)
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        // Best-effort durable flush on shutdown.
        for ds in self.datasets.read().values() {
            let _ = ds.lock().sync();
        }
        let mut wal = self.wal.lock();
        let _ = wal.sync();
        let committed = self.next_seq.load(Ordering::SeqCst).saturating_sub(1);
        let _ = wal.write_checkpoint(committed);
        drop(wal);
        let _ = self.replog.lock().sync();
    }
}

#[cfg(test)]
mod tests;
