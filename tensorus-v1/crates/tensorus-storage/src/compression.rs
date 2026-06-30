//! Adaptive, content-aware tensor compression.
//!
//! A tensor is analyzed and routed to the codec best suited to its structure:
//!
//! | Content        | Codec        | Lossy? | Notes |
//! |----------------|--------------|--------|-------|
//! | Sparse (>80% 0)| [`Codec::Coo`] | no   | delta-varint indices + DEFLATE |
//! | Low-rank 2-D   | [`Codec::LowRankSvd`] | yes (bounded) | truncated SVD |
//! | Dense/embedding| [`Codec::Sq8`] | yes (bounded) | 8-bit scalar quantization (4x) |
//! | Dense (lossless)| [`Codec::Deflate`] | no | general-purpose, ratio depends on entropy |
//!
//! ## Substitution note
//!
//! The plan specifies Zstd for the dense lossless path. To avoid a C build
//! dependency this uses DEFLATE via the pure-Rust `flate2`/`miniz_oxide`
//! backend; the [`Codec`] abstraction makes Zstd a drop-in replacement.
// TODO: SIMD dequantization (AVX-512/NEON) and a GPU INT8/INT4 -> FP32 kernel.

use flate2::read::DeflateDecoder;
use flate2::write::DeflateEncoder;
use flate2::Compression as DeflateLevel;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use tensorus_core::error::{Result, TensorusError};

/// The codec used to encode a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Codec {
    /// 8-bit scalar quantization (lossy, ~4x).
    Sq8,
    /// Coordinate-list sparse encoding (lossless).
    Coo,
    /// General-purpose DEFLATE over the raw bytes (lossless).
    Deflate,
    /// Truncated SVD for low-rank matrices (lossy, bounded).
    LowRankSvd,
}

/// A compressed tensor plus the metadata needed to reconstruct it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Compressed {
    pub codec: Codec,
    pub shape: Vec<u64>,
    pub num_elements: usize,
    /// Size of the original `f32` data in bytes.
    pub original_bytes: usize,
    /// Maximum absolute reconstruction error (`0.0` for lossless codecs).
    pub max_error: f32,
    pub payload: Vec<u8>,
}

impl Compressed {
    /// Compression ratio (original size / payload size).
    pub fn ratio(&self) -> f64 {
        if self.payload.is_empty() {
            return 0.0;
        }
        self.original_bytes as f64 / self.payload.len() as f64
    }
}

/// Sparsity threshold above which COO encoding is chosen.
const SPARSE_THRESHOLD: f64 = 0.8;
/// Energy fraction the top singular values must capture to be deemed low-rank.
const LOWRANK_ENERGY: f64 = 0.99;
/// Maximum element count for which auto-classification runs an SVD probe.
const LOWRANK_PROBE_LIMIT: usize = 65_536;

/// Analyze a tensor and pick the best codec.
pub fn classify(data: &[f32], shape: &[u64]) -> Codec {
    let n = data.len();
    if n == 0 {
        return Codec::Deflate;
    }
    let zeros = data.iter().filter(|&&x| x == 0.0).count();
    let sparsity = zeros as f64 / n as f64;
    if sparsity > SPARSE_THRESHOLD {
        return Codec::Coo;
    }
    if shape.len() == 2 && n <= LOWRANK_PROBE_LIMIT {
        let rows = shape[0] as usize;
        let cols = shape[1] as usize;
        if rows > 0 && cols > 0 {
            let m = DMatrix::<f64>::from_row_slice(rows, cols, &to_f64(data));
            let svals = m.singular_values();
            let total: f64 = svals.iter().map(|s| s * s).sum();
            if total > 0.0 {
                let k = (rows.min(cols) / 4).max(1);
                let kept: f64 = svals.iter().take(k).map(|s| s * s).sum();
                if kept / total > LOWRANK_ENERGY {
                    return Codec::LowRankSvd;
                }
            }
        }
    }
    Codec::Sq8
}

fn to_f64(data: &[f32]) -> Vec<f64> {
    data.iter().map(|&x| x as f64).collect()
}

/// Compress using the automatically selected codec.
pub fn compress(data: &[f32], shape: &[u64]) -> Result<Compressed> {
    compress_with(data, shape, classify(data, shape))
}

/// Compress using an explicit codec.
pub fn compress_with(data: &[f32], shape: &[u64], codec: Codec) -> Result<Compressed> {
    let original_bytes = data.len() * 4;
    let (payload, max_error) = match codec {
        Codec::Sq8 => sq8_compress(data),
        Codec::Coo => (coo_compress(data)?, 0.0),
        Codec::Deflate => (deflate(&f32_to_le(data))?, 0.0),
        Codec::LowRankSvd => lowrank_compress(data, shape)?,
    };
    Ok(Compressed {
        codec,
        shape: shape.to_vec(),
        num_elements: data.len(),
        original_bytes,
        max_error,
        payload,
    })
}

/// Reconstruct the tensor from its compressed form.
pub fn decompress(c: &Compressed) -> Result<Vec<f32>> {
    match c.codec {
        Codec::Sq8 => Ok(sq8_decompress(&c.payload)),
        Codec::Coo => coo_decompress(&c.payload, c.num_elements),
        Codec::Deflate => le_to_f32(&inflate(&c.payload)?),
        Codec::LowRankSvd => lowrank_decompress(&c.payload),
    }
}

// --- SQ8 ---------------------------------------------------------------------

fn sq8_compress(data: &[f32]) -> (Vec<u8>, f32) {
    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let scale = if max > min { (max - min) / 255.0 } else { 1.0 };
    let mut payload = Vec::with_capacity(8 + data.len());
    payload.extend_from_slice(&min.to_le_bytes());
    payload.extend_from_slice(&scale.to_le_bytes());
    for &x in data {
        let q = ((x - min) / scale).round().clamp(0.0, 255.0) as u8;
        payload.push(q);
    }
    (payload, scale / 2.0)
}

fn sq8_decompress(payload: &[u8]) -> Vec<f32> {
    let min = f32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let scale = f32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]);
    // TODO: SIMD — this scalar dequant loop vectorizes well but an explicit
    // AVX-512/NEON kernel would exceed 2 GB/s comfortably.
    payload[8..]
        .iter()
        .map(|&c| min + c as f32 * scale)
        .collect()
}

// --- COO (sparse) ------------------------------------------------------------

fn write_varint(buf: &mut Vec<u8>, mut v: u32) {
    loop {
        let mut byte = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if v == 0 {
            break;
        }
    }
}

fn read_varint(buf: &[u8], pos: &mut usize) -> Result<u32> {
    let mut result = 0u32;
    let mut shift = 0;
    loop {
        if *pos >= buf.len() {
            return Err(TensorusError::Storage("varint truncated".into()));
        }
        let byte = buf[*pos];
        *pos += 1;
        result |= ((byte & 0x7f) as u32) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    Ok(result)
}

fn coo_compress(data: &[f32]) -> Result<Vec<u8>> {
    let nonzeros: Vec<(u32, f32)> = data
        .iter()
        .enumerate()
        .filter(|(_, &v)| v != 0.0)
        .map(|(i, &v)| (i as u32, v))
        .collect();
    let mut raw = Vec::new();
    raw.extend_from_slice(&(nonzeros.len() as u32).to_le_bytes());
    // Delta-encoded ascending indices as varints (compresses well under DEFLATE).
    let mut prev = 0u32;
    for &(i, _) in &nonzeros {
        write_varint(&mut raw, i - prev);
        prev = i;
    }
    for &(_, v) in &nonzeros {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    deflate(&raw)
}

fn coo_decompress(payload: &[u8], n: usize) -> Result<Vec<f32>> {
    let raw = inflate(payload)?;
    if raw.len() < 4 {
        return Err(TensorusError::Storage("coo payload too small".into()));
    }
    let nnz = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize;
    let mut pos = 4usize;
    let mut indices = Vec::with_capacity(nnz);
    let mut prev = 0u32;
    for _ in 0..nnz {
        let delta = read_varint(&raw, &mut pos)?;
        prev += delta;
        indices.push(prev);
    }
    let mut out = vec![0.0f32; n];
    for &idx in &indices {
        let v = f32::from_le_bytes([raw[pos], raw[pos + 1], raw[pos + 2], raw[pos + 3]]);
        pos += 4;
        out[idx as usize] = v;
    }
    Ok(out)
}

// --- DEFLATE (general lossless) ---------------------------------------------

fn f32_to_le(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &x in data {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    bytes
}

fn le_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(TensorusError::Storage(
            "deflate payload not 4-aligned".into(),
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn deflate(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut enc = DeflateEncoder::new(Vec::new(), DeflateLevel::new(6));
    enc.write_all(bytes)?;
    enc.finish()
        .map_err(|e| TensorusError::Storage(e.to_string()))
}

fn inflate(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut dec = DeflateDecoder::new(bytes);
    let mut out = Vec::new();
    dec.read_to_end(&mut out)?;
    Ok(out)
}

// --- Low-rank SVD ------------------------------------------------------------

fn lowrank_compress(data: &[f32], shape: &[u64]) -> Result<(Vec<u8>, f32)> {
    if shape.len() != 2 {
        return Err(TensorusError::InvalidArgument(
            "low-rank codec requires a 2-D tensor".into(),
        ));
    }
    let rows = shape[0] as usize;
    let cols = shape[1] as usize;
    let m = DMatrix::<f64>::from_row_slice(rows, cols, &to_f64(data));
    let svd = m.clone().svd(true, true);
    let u = svd.u.as_ref().expect("u");
    let vt = svd.v_t.as_ref().expect("v_t");
    let svals = &svd.singular_values;
    let full = svals.len();
    // Choose the smallest rank capturing LOWRANK_ENERGY of the spectrum.
    let total: f64 = svals.iter().map(|s| s * s).sum();
    let mut rank = full;
    if total > 0.0 {
        let mut acc = 0.0;
        for (i, s) in svals.iter().enumerate() {
            acc += s * s;
            if acc / total >= LOWRANK_ENERGY {
                rank = i + 1;
                break;
            }
        }
    }

    let mut payload = Vec::new();
    payload.extend_from_slice(&(rank as u32).to_le_bytes());
    payload.extend_from_slice(&(rows as u32).to_le_bytes());
    payload.extend_from_slice(&(cols as u32).to_le_bytes());
    for c in 0..rank {
        for r in 0..rows {
            payload.extend_from_slice(&(u[(r, c)] as f32).to_le_bytes());
        }
    }
    for s in svals.iter().take(rank) {
        payload.extend_from_slice(&(*s as f32).to_le_bytes());
    }
    for c in 0..cols {
        for r in 0..rank {
            payload.extend_from_slice(&(vt[(r, c)] as f32).to_le_bytes());
        }
    }

    // Exact max element error of the truncated reconstruction.
    let recon = reconstruct_lowrank(u, svals, vt, rank, rows, cols);
    let mut max_err = 0.0f32;
    for (orig, rec) in data.iter().zip(recon.iter()) {
        max_err = max_err.max((orig - rec).abs());
    }
    Ok((payload, max_err))
}

fn reconstruct_lowrank(
    u: &DMatrix<f64>,
    svals: &nalgebra::DVector<f64>,
    vt: &DMatrix<f64>,
    rank: usize,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let mut acc = 0.0f64;
            for l in 0..rank {
                acc += u[(r, l)] * svals[l] * vt[(l, c)];
            }
            out[r * cols + c] = acc as f32;
        }
    }
    out
}

fn lowrank_decompress(payload: &[u8]) -> Result<Vec<f32>> {
    let rd = |p: &[u8], o: usize| u32::from_le_bytes([p[o], p[o + 1], p[o + 2], p[o + 3]]);
    let rank = rd(payload, 0) as usize;
    let rows = rd(payload, 4) as usize;
    let cols = rd(payload, 8) as usize;
    let mut pos = 12usize;
    let read_f32 = |p: &[u8], pos: &mut usize| {
        let v = f32::from_le_bytes([p[*pos], p[*pos + 1], p[*pos + 2], p[*pos + 3]]);
        *pos += 4;
        v
    };
    let mut u = DMatrix::<f64>::zeros(rows, rank);
    for c in 0..rank {
        for r in 0..rows {
            u[(r, c)] = read_f32(payload, &mut pos) as f64;
        }
    }
    let mut svals = nalgebra::DVector::<f64>::zeros(rank);
    for l in 0..rank {
        svals[l] = read_f32(payload, &mut pos) as f64;
    }
    let mut vt = DMatrix::<f64>::zeros(rank, cols);
    for c in 0..cols {
        for r in 0..rank {
            vt[(r, c)] = read_f32(payload, &mut pos) as f64;
        }
    }
    Ok(reconstruct_lowrank(&u, &svals, &vt, rank, rows, cols))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn rng() -> StdRng {
        StdRng::seed_from_u64(123)
    }

    #[test]
    fn sq8_roundtrip_and_ratio() {
        let mut r = rng();
        let data: Vec<f32> = (0..4096).map(|_| r.gen_range(-3.0..3.0)).collect();
        let c = compress_with(&data, &[4096], Codec::Sq8).unwrap();
        assert!(c.ratio() >= 3.9, "sq8 ratio {}", c.ratio());
        let back = decompress(&c).unwrap();
        for (o, b) in data.iter().zip(&back) {
            assert!((o - b).abs() <= c.max_error + 1e-6, "error exceeds bound");
        }
    }

    #[test]
    fn coo_sparse_lossless_and_ratio() {
        let mut r = rng();
        let n = 10_000;
        let mut data = vec![0.0f32; n];
        // ~5% nonzero (95% zeros).
        for _ in 0..(n / 20) {
            data[r.gen_range(0..n)] = r.gen_range(-1.0..1.0);
        }
        let c = compress_with(&data, &[n as u64], Codec::Coo).unwrap();
        let back = decompress(&c).unwrap();
        assert_eq!(back, data, "COO must be lossless");
        println!("COO ratio at 95% sparsity = {:.1}x", c.ratio());
        assert!(c.ratio() >= 8.0, "coo ratio {}", c.ratio());
        assert_eq!(c.max_error, 0.0);
    }

    #[test]
    fn deflate_lossless_on_compressible() {
        // Repetitive data compresses well and round-trips exactly.
        let data: Vec<f32> = (0..4096).map(|i| (i % 8) as f32).collect();
        let c = compress_with(&data, &[4096], Codec::Deflate).unwrap();
        let back = decompress(&c).unwrap();
        assert_eq!(back, data);
        assert_eq!(c.max_error, 0.0);
        println!("DEFLATE ratio on structured data = {:.1}x", c.ratio());
        assert!(c.ratio() >= 4.0);
    }

    #[test]
    fn lowrank_compresses_low_rank_matrix() {
        // Rank-1 matrix: outer product of two vectors.
        let rows = 64;
        let cols = 64;
        let mut r = rng();
        let u: Vec<f32> = (0..rows).map(|_| r.gen_range(-1.0..1.0)).collect();
        let v: Vec<f32> = (0..cols).map(|_| r.gen_range(-1.0..1.0)).collect();
        let mut data = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[i * cols + j] = u[i] * v[j];
            }
        }
        let c = compress_with(&data, &[rows as u64, cols as u64], Codec::LowRankSvd).unwrap();
        println!(
            "low-rank ratio = {:.1}x, max_error = {}",
            c.ratio(),
            c.max_error
        );
        assert!(c.ratio() >= 8.0, "low-rank ratio {}", c.ratio());
        assert!(c.max_error < 1e-3, "reconstruction error too high");
        let back = decompress(&c).unwrap();
        for (o, b) in data.iter().zip(&back) {
            assert!((o - b).abs() < 1e-3);
        }
    }

    #[test]
    fn classify_routes_correctly() {
        // Sparse.
        let mut sparse = vec![0.0f32; 1000];
        sparse[3] = 1.0;
        sparse[700] = 2.0;
        assert_eq!(classify(&sparse, &[1000]), Codec::Coo);

        // Low-rank 2-D.
        let rows = 32;
        let cols = 32;
        let mut lr = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                lr[i * cols + j] = (i + 1) as f32 * (j + 1) as f32; // rank 1
            }
        }
        assert_eq!(
            classify(&lr, &[rows as u64, cols as u64]),
            Codec::LowRankSvd
        );

        // Dense full-rank vector -> Sq8.
        let mut r = rng();
        let dense: Vec<f32> = (0..1000).map(|_| r.gen_range(0.5..3.0)).collect();
        assert_eq!(classify(&dense, &[1000]), Codec::Sq8);
    }

    #[test]
    #[ignore]
    fn sq8_decompression_throughput() {
        let mut r = rng();
        let n = 4_000_000;
        let data: Vec<f32> = (0..n).map(|_| r.gen_range(-1.0..1.0)).collect();
        let c = compress_with(&data, &[n as u64], Codec::Sq8).unwrap();
        let start = std::time::Instant::now();
        let iters = 20;
        let mut sink = 0.0f32;
        for _ in 0..iters {
            let back = decompress(&c).unwrap();
            sink += back[0];
        }
        let elapsed = start.elapsed();
        let bytes = (n * 4 * iters) as f64; // output bytes produced
        let gbps = bytes / elapsed.as_secs_f64() / 1e9;
        println!("SQ8 dequant throughput = {gbps:.2} GB/s (sink={sink})");
    }
}
