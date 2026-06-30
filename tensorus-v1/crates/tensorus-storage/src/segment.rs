//! Length-prefixed frame I/O over a single append-only segment file.
//!
//! Each record on disk is `[u32 payload_len][payload]`. The length prefix lets
//! recovery detect a torn (partially written) trailing frame after a crash and
//! truncate the file back to the last complete frame.

use crate::format::{decode_frame, encode_frame, Frame};
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use tensorus_core::error::Result;

/// Append a frame and return the byte offset at which it begins.
pub(crate) fn write_frame(file: &mut File, frame: &Frame) -> Result<u64> {
    let payload = encode_frame(frame)?;
    let offset = file.seek(SeekFrom::End(0))?;
    let len = payload.len() as u32;
    file.write_all(&len.to_le_bytes())?;
    file.write_all(&payload)?;
    Ok(offset)
}

/// Outcome of scanning a segment file on startup.
pub(crate) struct ScanResult {
    pub frames: Vec<Frame>,
    /// Number of bytes occupied by complete frames. If smaller than the file
    /// length, the tail is torn and should be truncated to this value.
    pub valid_len: u64,
}

/// Read every complete frame from a segment, stopping at the first torn or
/// malformed frame.
pub(crate) fn scan_segment(bytes: &[u8]) -> ScanResult {
    let mut frames = Vec::new();
    let mut pos: usize = 0;
    loop {
        if pos + 4 > bytes.len() {
            break;
        }
        let len = u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
            as usize;
        let payload_start = pos + 4;
        let payload_end = match payload_start.checked_add(len) {
            Some(e) => e,
            None => break,
        };
        if payload_end > bytes.len() {
            break; // torn trailing frame
        }
        match decode_frame(&bytes[payload_start..payload_end]) {
            Ok(frame) => {
                frames.push(frame);
                pos = payload_end;
            }
            Err(_) => break, // corrupt frame: stop and treat the rest as torn
        }
    }
    ScanResult {
        frames,
        valid_len: pos as u64,
    }
}
