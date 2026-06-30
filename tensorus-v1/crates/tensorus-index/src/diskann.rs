//! DiskANN / Vamana SSD-resident vector index.
//!
//! Implements the Vamana graph construction (greedy search + `RobustPrune` with
//! the `alpha` diversification parameter) from Subramanya et al., "DiskANN:
//! Fast Accurate Billion-point Nearest Neighbor Search on a Single Node" (2019),
//! plus an SSD-optimized on-disk layout and a disk-resident beam search.
//!
//! ## Layout & search
//!
//! After construction the graph is flushed to a file where each node occupies a
//! 4 KB-aligned record: `[R neighbor u32s][dim f32 vector]`. Beam search reads a
//! node's neighbor list from disk (one page read per visited node) and ranks the
//! frontier using **in-memory SQ8 codes** (1 byte/dim) for an approximate
//! distance; the final candidates are reranked by reading their full vectors
//! back from disk. This mirrors DiskANN: cheap in-memory approximate distances
//! drive traversal, exact distances (from "SSD") refine the result.
//!
//! ## Environment notes / simplifications
//!
//! - The plan calls for `io_uring` async I/O; that is Linux-only and unavailable
//!   on this Windows host, so reads use synchronous `std::fs` seeks (documented
//!   substitution). The page-aligned layout is preserved so an async reader can
//!   be dropped in later.
//! - The plan specifies Product Quantization; this uses the simpler per-dimension
//!   Scalar Quantization (SQ8) for the in-memory approximate distance. PQ is a
//!   drop-in upgrade (TODO).
//! - Vamana is a static (build-once) index; updates are handled by rebuilds.
// TODO: GPU acceleration for batched distance; PQ codes; io_uring async reads.

use parking_lot::Mutex;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::BinaryHeap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use tensorus_core::error::Result;
use tensorus_core::types::TensorId;

const PAGE: u64 = 4096;
const SENTINEL: u32 = u32::MAX;

/// Vamana build/search parameters.
#[derive(Debug, Clone, Copy)]
pub struct VamanaConfig {
    /// Maximum out-degree `R`.
    pub degree: usize,
    /// Search list size `L` used during construction.
    pub build_list: usize,
    /// Diversification factor `alpha` (>= 1.0).
    pub alpha: f32,
    /// Default search list size at query time.
    pub search_list: usize,
    pub seed: u64,
}

impl Default for VamanaConfig {
    fn default() -> Self {
        VamanaConfig {
            degree: 64,
            build_list: 100,
            alpha: 1.2,
            search_list: 100,
            seed: 0xda7a,
        }
    }
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Per-dimension 8-bit scalar quantization for in-memory approximate distance.
struct Sq8 {
    mins: Vec<f32>,
    scales: Vec<f32>, // (max-min)/255 per dim
    codes: Vec<u8>,   // n * dim
    dim: usize,
}

impl Sq8 {
    fn build(vectors: &[Vec<f32>], dim: usize) -> Sq8 {
        let mut mins = vec![f32::INFINITY; dim];
        let mut maxs = vec![f32::NEG_INFINITY; dim];
        for v in vectors {
            for ((mn, mx), &x) in mins.iter_mut().zip(maxs.iter_mut()).zip(v.iter()) {
                *mn = mn.min(x);
                *mx = mx.max(x);
            }
        }
        let scales: Vec<f32> = (0..dim)
            .map(|d| {
                let range = maxs[d] - mins[d];
                if range > 0.0 {
                    range / 255.0
                } else {
                    1.0
                }
            })
            .collect();
        let mut codes = vec![0u8; vectors.len() * dim];
        for (i, v) in vectors.iter().enumerate() {
            let out = &mut codes[i * dim..(i + 1) * dim];
            for (((o, &x), &mn), &sc) in out.iter_mut().zip(v).zip(&mins).zip(&scales) {
                let q = ((x - mn) / sc).round().clamp(0.0, 255.0);
                *o = q as u8;
            }
        }
        Sq8 {
            mins,
            scales,
            codes,
            dim,
        }
    }

    /// Approximate squared-L2 distance between an f32 query and node `idx`.
    fn approx_dist(&self, query: &[f32], idx: u32) -> f32 {
        let base = idx as usize * self.dim;
        let codes = &self.codes[base..base + self.dim];
        self.mins
            .iter()
            .zip(self.scales.iter())
            .zip(codes.iter())
            .zip(query.iter())
            .map(|(((min, scale), code), q)| {
                let val = min + *code as f32 * scale;
                let diff = q - val;
                diff * diff
            })
            .sum()
    }

    fn bytes_per_vector(&self) -> usize {
        self.dim // 1 byte per dim
    }
}

#[derive(Clone, Copy)]
struct Scored {
    dist: f32,
    node: u32,
}
impl PartialEq for Scored {
    fn eq(&self, o: &Self) -> bool {
        self.dist.total_cmp(&o.dist) == std::cmp::Ordering::Equal && self.node == o.node
    }
}
impl Eq for Scored {}
impl Ord for Scored {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.dist.total_cmp(&o.dist).then(self.node.cmp(&o.node))
    }
}
impl PartialOrd for Scored {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

/// An SSD-resident Vamana index.
pub struct DiskAnnIndex {
    file: Mutex<File>,
    n: usize,
    dim: usize,
    r: usize,
    medoid: u32,
    header_size: u64,
    page_stride: u64,
    ids: Vec<TensorId>,
    sq: Sq8,
    cfg: VamanaConfig,
}

impl DiskAnnIndex {
    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Per-vector overhead in bytes (SQ8 codes + graph edges), excluding the
    /// full vectors on disk.
    pub fn bytes_per_vector_overhead(&self) -> usize {
        self.sq.bytes_per_vector() + self.r * 4
    }

    /// Build a Vamana graph over `vectors` and flush it to `path`.
    pub fn build(
        vectors: Vec<Vec<f32>>,
        ids: Vec<TensorId>,
        cfg: VamanaConfig,
        path: impl AsRef<Path>,
    ) -> Result<DiskAnnIndex> {
        let n = vectors.len();
        assert_eq!(n, ids.len());
        let dim = vectors.first().map(|v| v.len()).unwrap_or(0);
        let r = cfg.degree;

        let mut graph: Vec<Vec<u32>> = vec![Vec::new(); n];
        let medoid = if n == 0 { 0 } else { compute_medoid(&vectors) };

        if n > 1 {
            let mut rng = StdRng::seed_from_u64(cfg.seed);
            // Random initial graph.
            for (u, nbrs) in graph.iter_mut().enumerate() {
                let mut others: Vec<u32> = Vec::with_capacity(r);
                while others.len() < r.min(n - 1) {
                    let cand = rng.gen_range(0..n as u32);
                    if cand as usize != u && !others.contains(&cand) {
                        others.push(cand);
                    }
                }
                *nbrs = others;
            }

            // Two passes: alpha = 1.0 then alpha = cfg.alpha.
            for &alpha in &[1.0f32, cfg.alpha] {
                let mut order: Vec<u32> = (0..n as u32).collect();
                order.shuffle(&mut rng);
                for &p in &order {
                    let visited = greedy_search(
                        &vectors,
                        &graph,
                        medoid,
                        &vectors[p as usize],
                        cfg.build_list,
                    );
                    let pruned = robust_prune(&vectors, p, visited, alpha, r);
                    graph[p as usize] = pruned.clone();
                    for &j in &pruned {
                        if !graph[j as usize].contains(&p) {
                            graph[j as usize].push(p);
                            if graph[j as usize].len() > r {
                                let cand = graph[j as usize].clone();
                                let scored: Vec<u32> = cand;
                                graph[j as usize] = robust_prune(&vectors, j, scored, alpha, r);
                            }
                        }
                    }
                }
            }
        }

        // SQ8 codes for in-memory approximate distance.
        let sq = Sq8::build(&vectors, dim);

        // Flush page-aligned node records: [R u32 neighbors][dim f32 vector].
        let record_size = (r * 4 + dim * 4) as u64;
        let page_stride = record_size.div_ceil(PAGE) * PAGE;
        let header_size = PAGE; // header occupies one page
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&path)?;

        // Header: magic, n, dim, r, medoid.
        let mut header = Vec::with_capacity(PAGE as usize);
        header.extend_from_slice(b"VAMANA01");
        header.extend_from_slice(&(n as u64).to_le_bytes());
        header.extend_from_slice(&(dim as u64).to_le_bytes());
        header.extend_from_slice(&(r as u64).to_le_bytes());
        header.extend_from_slice(&(medoid as u64).to_le_bytes());
        header.resize(PAGE as usize, 0);
        file.write_all(&header)?;

        for u in 0..n {
            let mut rec = Vec::with_capacity(page_stride as usize);
            for slot in 0..r {
                let id = graph[u].get(slot).copied().unwrap_or(SENTINEL);
                rec.extend_from_slice(&id.to_le_bytes());
            }
            for &x in &vectors[u] {
                rec.extend_from_slice(&x.to_le_bytes());
            }
            rec.resize(page_stride as usize, 0);
            file.write_all(&rec)?;
        }
        file.flush()?;
        file.sync_all().ok();

        Ok(DiskAnnIndex {
            file: Mutex::new(file),
            n,
            dim,
            r,
            medoid,
            header_size,
            page_stride,
            ids,
            sq,
            cfg,
        })
    }

    fn node_offset(&self, node: u32) -> u64 {
        self.header_size + node as u64 * self.page_stride
    }

    fn read_neighbors(&self, node: u32) -> Result<Vec<u32>> {
        let mut buf = vec![0u8; self.r * 4];
        {
            let mut f = self.file.lock();
            f.seek(SeekFrom::Start(self.node_offset(node)))?;
            f.read_exact(&mut buf)?;
        }
        let mut out = Vec::with_capacity(self.r);
        for chunk in buf.chunks_exact(4) {
            let id = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if id == SENTINEL {
                break;
            }
            out.push(id);
        }
        Ok(out)
    }

    fn read_vector(&self, node: u32) -> Result<Vec<f32>> {
        let mut buf = vec![0u8; self.dim * 4];
        {
            let mut f = self.file.lock();
            f.seek(SeekFrom::Start(
                self.node_offset(node) + (self.r * 4) as u64,
            ))?;
            f.read_exact(&mut buf)?;
        }
        Ok(buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Beam search for the `k` nearest vectors to `query`.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(TensorId, f32)>> {
        self.search_with_beam(query, k, self.cfg.search_list)
    }

    /// Beam search with an explicit search-list size (beam width).
    pub fn search_with_beam(
        &self,
        query: &[f32],
        k: usize,
        beam: usize,
    ) -> Result<Vec<(TensorId, f32)>> {
        if self.n == 0 {
            return Ok(Vec::new());
        }
        let beam = beam.max(k);
        let mut visited = vec![false; self.n];
        // Frontier: min-heap by approx distance.
        let mut frontier: BinaryHeap<std::cmp::Reverse<Scored>> = BinaryHeap::new();
        // Result list: max-heap (worst on top) capped at `beam`.
        let mut results: BinaryHeap<Scored> = BinaryHeap::new();

        let start = Scored {
            dist: self.sq.approx_dist(query, self.medoid),
            node: self.medoid,
        };
        frontier.push(std::cmp::Reverse(start));
        results.push(start);
        visited[self.medoid as usize] = true;

        while let Some(std::cmp::Reverse(c)) = frontier.pop() {
            let worst = results.peek().map(|s| s.dist).unwrap_or(f32::INFINITY);
            if c.dist > worst && results.len() >= beam {
                break;
            }
            let neighbors = self.read_neighbors(c.node)?;
            for nb in neighbors {
                if !visited[nb as usize] {
                    visited[nb as usize] = true;
                    let d = self.sq.approx_dist(query, nb);
                    let worst = results.peek().map(|s| s.dist).unwrap_or(f32::INFINITY);
                    if results.len() < beam || d < worst {
                        frontier.push(std::cmp::Reverse(Scored { dist: d, node: nb }));
                        results.push(Scored { dist: d, node: nb });
                        if results.len() > beam {
                            results.pop();
                        }
                    }
                }
            }
        }

        // Rerank the approximate results with exact distances from disk.
        let mut exact: Vec<Scored> = Vec::with_capacity(results.len());
        for s in results.into_vec() {
            let v = self.read_vector(s.node)?;
            exact.push(Scored {
                dist: l2(query, &v),
                node: s.node,
            });
        }
        exact.sort();
        Ok(exact
            .into_iter()
            .take(k)
            .map(|s| (self.ids[s.node as usize], s.dist))
            .collect())
    }
}

/// The node whose vector is closest to the dataset centroid.
fn compute_medoid(vectors: &[Vec<f32>]) -> u32 {
    let n = vectors.len();
    let dim = vectors[0].len();
    let mut centroid = vec![0.0f32; dim];
    for v in vectors {
        for (c, &x) in centroid.iter_mut().zip(v.iter()) {
            *c += x;
        }
    }
    for c in &mut centroid {
        *c /= n as f32;
    }
    let mut best = 0u32;
    let mut best_d = f32::INFINITY;
    for (i, v) in vectors.iter().enumerate() {
        let d = l2(&centroid, v);
        if d < best_d {
            best_d = d;
            best = i as u32;
        }
    }
    best
}

/// In-memory greedy search returning the full set of visited nodes (the
/// candidate pool for `RobustPrune`).
fn greedy_search(
    vectors: &[Vec<f32>],
    graph: &[Vec<u32>],
    start: u32,
    query: &[f32],
    l: usize,
) -> Vec<u32> {
    let n = vectors.len();
    let mut visited = vec![false; n];
    let mut considered: Vec<u32> = Vec::new();
    let mut frontier: BinaryHeap<std::cmp::Reverse<Scored>> = BinaryHeap::new();
    let mut best: BinaryHeap<Scored> = BinaryHeap::new();

    let s = Scored {
        dist: l2(query, &vectors[start as usize]),
        node: start,
    };
    frontier.push(std::cmp::Reverse(s));
    best.push(s);
    visited[start as usize] = true;
    considered.push(start);

    while let Some(std::cmp::Reverse(c)) = frontier.pop() {
        let worst = best.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
        if c.dist > worst && best.len() >= l {
            break;
        }
        for &nb in &graph[c.node as usize] {
            if !visited[nb as usize] {
                visited[nb as usize] = true;
                considered.push(nb);
                let d = l2(query, &vectors[nb as usize]);
                let worst = best.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
                if best.len() < l || d < worst {
                    frontier.push(std::cmp::Reverse(Scored { dist: d, node: nb }));
                    best.push(Scored { dist: d, node: nb });
                    if best.len() > l {
                        best.pop();
                    }
                }
            }
        }
    }
    considered
}

/// `RobustPrune`: select up to `r` diverse neighbors for `p` from `candidates`.
fn robust_prune(
    vectors: &[Vec<f32>],
    p: u32,
    candidates: Vec<u32>,
    alpha: f32,
    r: usize,
) -> Vec<u32> {
    let mut pool: Vec<Scored> = candidates
        .into_iter()
        .filter(|&c| c != p)
        .map(|c| Scored {
            dist: l2(&vectors[p as usize], &vectors[c as usize]),
            node: c,
        })
        .collect();
    pool.sort();
    // Deduplicate by node while preserving distance order.
    pool.dedup_by_key(|s| s.node);

    let mut result: Vec<u32> = Vec::with_capacity(r);
    let mut i = 0;
    while i < pool.len() && result.len() < r {
        let p_star = pool[i].node;
        result.push(p_star);
        // Remove candidates dominated by p_star under the alpha rule.
        let mut kept = Vec::new();
        for &cand in pool.iter().skip(i + 1) {
            let d_star = l2(&vectors[p_star as usize], &vectors[cand.node as usize]);
            if alpha * d_star > cand.dist {
                kept.push(cand);
            }
        }
        pool = kept;
        i = 0;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet;
    use tempfile::TempDir;

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect()
    }

    fn brute_topk(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, l2(query, v)))
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored.into_iter().take(k).map(|(i, _)| i).collect()
    }

    #[test]
    fn build_search_small() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("vamana.idx");
        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![5.0, 5.0],
            vec![0.1, 0.1],
        ];
        let ids: Vec<TensorId> = (0..5).map(|_| TensorId::new()).collect();
        let idx = DiskAnnIndex::build(
            vectors.clone(),
            ids.clone(),
            VamanaConfig {
                degree: 4,
                ..Default::default()
            },
            &path,
        )
        .unwrap();
        assert_eq!(idx.len(), 5);
        let res = idx.search(&[0.0, 0.0], 2).unwrap();
        let got: Vec<TensorId> = res.iter().map(|(id, _)| *id).collect();
        assert!(got.contains(&ids[0]));
        assert!(got.contains(&ids[4]));
    }

    #[test]
    fn page_aligned_layout() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("vamana.idx");
        let vectors = random_vectors(50, 16, 1);
        let ids: Vec<TensorId> = (0..50).map(|_| TensorId::new()).collect();
        let idx = DiskAnnIndex::build(vectors, ids, VamanaConfig::default(), &path).unwrap();
        // Header is one page; node stride is a multiple of the page size.
        assert_eq!(idx.header_size, PAGE);
        assert_eq!(idx.page_stride % PAGE, 0);
        let file_len = std::fs::metadata(&path).unwrap().len();
        assert_eq!(file_len, PAGE + 50 * idx.page_stride);
    }

    #[test]
    fn recall_vs_brute_force() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("vamana.idx");
        let dim = 32;
        let n = 2000;
        let vectors = random_vectors(n, dim, 7);
        let ids: Vec<TensorId> = (0..n).map(|_| TensorId::new()).collect();
        let id_to_idx: std::collections::HashMap<TensorId, usize> =
            ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();
        let idx =
            DiskAnnIndex::build(vectors.clone(), ids, VamanaConfig::default(), &path).unwrap();

        let queries = random_vectors(100, dim, 8);
        let k = 10;
        let mut hits = 0;
        let mut total = 0;
        for q in &queries {
            let truth: HashSet<usize> = brute_topk(&vectors, q, k).into_iter().collect();
            for (id, _) in idx.search_with_beam(q, k, 100).unwrap() {
                if truth.contains(&id_to_idx[&id]) {
                    hits += 1;
                }
            }
            total += k;
        }
        let recall = hits as f64 / total as f64;
        println!("DiskANN recall@{k} on {n} vectors = {recall:.4}");
        assert!(recall >= 0.90, "recall too low: {recall}");
    }

    #[test]
    fn reopen_via_disk_reads() {
        // Verify search works purely from the page-aligned file content.
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("vamana.idx");
        let vectors = random_vectors(200, 8, 3);
        let ids: Vec<TensorId> = (0..200).map(|_| TensorId::new()).collect();
        let idx =
            DiskAnnIndex::build(vectors.clone(), ids, VamanaConfig::default(), &path).unwrap();
        let q = &vectors[42];
        let res = idx.search(q, 1).unwrap();
        // Nearest to a stored vector should be itself (distance ~0).
        assert!(res[0].1 < 1e-3, "expected self as nearest, got {res:?}");
        assert!(idx.bytes_per_vector_overhead() > 0);
    }
}
