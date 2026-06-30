//! On-disk binary frame format shared by the segment files and the WAL.
//!
//! A *frame* is the unit of durable change. It is either a `Put` (a full
//! record) or a `Del` (a tombstone identified by id). Frames are encoded as a
//! self-describing payload; the segment/WAL writers prepend a `u32` length so
//! torn (partially written) trailing frames can be detected and discarded on
//! recovery.
//!
//! Layout of a `Put` payload:
//! ```text
//! [u8 op=0][16B id][i64 created_at_us][u64 version]
//! [u32 desc_len][desc_json][u32 meta_len][meta_json][u64 data_len][data]
//! ```
//! Layout of a `Del` payload: `[u8 op=1][16B id]`.

use tensorus_core::error::{Result, TensorusError};
use tensorus_core::types::{Metadata, TensorDescriptor, TensorId};

const OP_PUT: u8 = 0;
const OP_DEL: u8 = 1;

/// A fully decoded stored record (without its dataset name, which is implied by
/// the file it lives in).
#[derive(Debug, Clone)]
pub(crate) struct StoredRecord {
    pub id: TensorId,
    pub created_at_us: i64,
    pub version: u64,
    pub descriptor: TensorDescriptor,
    pub metadata: Metadata,
    pub data: Vec<u8>,
}

/// A durable change recorded in a segment or the WAL.
//
// `Put` is much larger than `Del`; we accept the size difference because frames
// are short-lived (constructed per write, dropped after encode/apply) and
// boxing every record would add an allocation on the hot insert path.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub(crate) enum Frame {
    Put(StoredRecord),
    Del(TensorId),
}

/// Encode a frame's payload (without the outer length prefix).
pub(crate) fn encode_frame(frame: &Frame) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    match frame {
        Frame::Put(rec) => {
            buf.push(OP_PUT);
            buf.extend_from_slice(rec.id.as_bytes());
            buf.extend_from_slice(&rec.created_at_us.to_le_bytes());
            buf.extend_from_slice(&rec.version.to_le_bytes());

            let desc = serde_json::to_vec(&rec.descriptor)?;
            buf.extend_from_slice(&(desc.len() as u32).to_le_bytes());
            buf.extend_from_slice(&desc);

            let meta = serde_json::to_vec(&rec.metadata)?;
            buf.extend_from_slice(&(meta.len() as u32).to_le_bytes());
            buf.extend_from_slice(&meta);

            buf.extend_from_slice(&(rec.data.len() as u64).to_le_bytes());
            buf.extend_from_slice(&rec.data);
        }
        Frame::Del(id) => {
            buf.push(OP_DEL);
            buf.extend_from_slice(id.as_bytes());
        }
    }
    Ok(buf)
}

/// Decode a frame payload produced by [`encode_frame`].
pub(crate) fn decode_frame(bytes: &[u8]) -> Result<Frame> {
    let mut r = Reader::new(bytes);
    let op = r.read_u8()?;
    match op {
        OP_PUT => {
            let id = TensorId::from_bytes(r.read_array16()?);
            let created_at_us = r.read_i64()?;
            let version = r.read_u64()?;
            let desc_len = r.read_u32()? as usize;
            let desc_bytes = r.read_bytes(desc_len)?;
            let descriptor: TensorDescriptor = serde_json::from_slice(desc_bytes)?;
            let meta_len = r.read_u32()? as usize;
            let meta_bytes = r.read_bytes(meta_len)?;
            let metadata: Metadata = serde_json::from_slice(meta_bytes)?;
            let data_len = r.read_u64()? as usize;
            let data = r.read_bytes(data_len)?.to_vec();
            Ok(Frame::Put(StoredRecord {
                id,
                created_at_us,
                version,
                descriptor,
                metadata,
                data,
            }))
        }
        OP_DEL => {
            let id = TensorId::from_bytes(r.read_array16()?);
            Ok(Frame::Del(id))
        }
        other => Err(TensorusError::Storage(format!(
            "unknown frame op byte {other}"
        ))),
    }
}

/// A bounds-checked little-endian reader over a byte slice.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| TensorusError::Storage("frame length overflow".into()))?;
        if end > self.buf.len() {
            return Err(TensorusError::Storage(format!(
                "frame truncated: needed {n} bytes at offset {}, have {}",
                self.pos,
                self.buf.len() - self.pos
            )));
        }
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_u32(&mut self) -> Result<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    fn read_array16(&mut self) -> Result<[u8; 16]> {
        let b = self.read_bytes(16)?;
        let mut arr = [0u8; 16];
        arr.copy_from_slice(b);
        Ok(arr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorus_core::types::{DType, Shape};

    fn sample_record() -> StoredRecord {
        let id = TensorId::new();
        let descriptor = TensorDescriptor::empty(id, Shape::new(vec![2, 2]), DType::Float32);
        StoredRecord {
            id,
            created_at_us: 1_700_000_000_000_000,
            version: 1,
            descriptor,
            metadata: serde_json::json!({"name": "w"}),
            data: vec![1, 2, 3, 4, 5, 6, 7, 8],
        }
    }

    #[test]
    fn put_frame_roundtrip() {
        let rec = sample_record();
        let frame = Frame::Put(rec.clone());
        let bytes = encode_frame(&frame).unwrap();
        let decoded = decode_frame(&bytes).unwrap();
        match decoded {
            Frame::Put(d) => {
                assert_eq!(d.id, rec.id);
                assert_eq!(d.version, rec.version);
                assert_eq!(d.data, rec.data);
                assert_eq!(d.metadata, rec.metadata);
            }
            _ => panic!("expected put"),
        }
    }

    #[test]
    fn del_frame_roundtrip() {
        let id = TensorId::new();
        let bytes = encode_frame(&Frame::Del(id)).unwrap();
        match decode_frame(&bytes).unwrap() {
            Frame::Del(d) => assert_eq!(d, id),
            _ => panic!("expected del"),
        }
    }

    #[test]
    fn truncated_frame_errors() {
        let bytes = encode_frame(&Frame::Put(sample_record())).unwrap();
        // Lop off the tail to simulate a torn write.
        assert!(decode_frame(&bytes[..bytes.len() - 3]).is_err());
    }
}
