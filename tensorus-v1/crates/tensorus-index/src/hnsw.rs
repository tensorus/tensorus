//! In-memory HNSW (Hierarchical Navigable Small World) vector index.
//!
//! A custom Rust implementation (no external ANN crate) giving full control
//! over the memory layout and metric. Implements the construction and search of
//! Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search
//! using Hierarchical Navigable Small World graphs" (2018), including the
//! neighbor-selection heuristic.
//!
//! Defaults: `M = 16`, `ef_construction = 200`, `ef_search = 100`. Supports
//! cosine, L2, and dot-product metrics, plus dynamic insert and (tombstone)
//! delete.
//!
//! ## Performance note
//!
//! This is a correct, full-featured implementation but the distance kernel is
//! scalar. Measured on 10k 128-dim vectors (release): search ~0.3-0.6 ms/query
//! (meets the < 1 ms target) and recall@10 ≈ 0.95 on low-dimensional / clustered
//! data (≈ 0.83 on uniform-random 128-dim, an adversarial case). Insert
//! throughput (~1.6k/sec with `ef_construction = 200`) is well below the
//! 50k/sec target; closing that gap requires the SIMD/flat-layout work noted in
//! the TODO below. Numbers are documented in the Task 20 benchmark report.
// TODO: GPU acceleration — batch distance computation in a CUDA kernel; SIMD
// (AVX-512/NEON) for the CPU distance hot loop.

use parking_lot::RwLock;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use tensorus_core::types::TensorId;

/// Distance metric used by the index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Squared Euclidean distance.
    L2,
    /// Cosine distance (`1 - cosine_similarity`); vectors are normalized on
    /// insert/search.
    Cosine,
    /// Negative inner product (so that a larger dot product is "closer").
    Dot,
}

impl Metric {
    /// Normalize a query/insert vector as required by the metric.
    fn prepare(&self, v: &[f32]) -> Vec<f32> {
        match self {
            Metric::Cosine => {
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    v.iter().map(|x| x / norm).collect()
                } else {
                    v.to_vec()
                }
            }
            _ => v.to_vec(),
        }
    }

    /// Distance between two already-prepared vectors (smaller is closer).
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::L2 => a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum(),
            Metric::Cosine => 1.0 - a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>(),
            Metric::Dot => -a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>(),
        }
    }
}

/// Configuration for an HNSW index.
#[derive(Debug, Clone, Copy)]
pub struct HnswConfig {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub metric: Metric,
    pub seed: u64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            metric: Metric::Cosine,
            seed: 0x5eed,
        }
    }
}

/// A `(distance, node)` pair ordered by distance (with `total_cmp`, so it is a
/// total order over finite floats).
#[derive(Debug, Clone, Copy)]
struct Cand {
    dist: f32,
    node: u32,
}

impl PartialEq for Cand {
    fn eq(&self, other: &Self) -> bool {
        self.dist.total_cmp(&other.dist) == Ordering::Equal && self.node == other.node
    }
}
impl Eq for Cand {}
impl Ord for Cand {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then(self.node.cmp(&other.node))
    }
}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct Node {
    vector: Vec<f32>,
    id: TensorId,
    deleted: bool,
    /// Neighbor lists, indexed by layer (`neighbors.len()` = node's top level + 1).
    neighbors: Vec<Vec<u32>>,
}

struct Graph {
    nodes: Vec<Node>,
    entry: Option<u32>,
    max_level: usize,
    id_map: HashMap<TensorId, u32>,
    rng: StdRng,
    cfg: HnswConfig,
    ml: f64,
}

/// An HNSW vector index.
pub struct Hnsw {
    inner: RwLock<Graph>,
}

impl Hnsw {
    /// Create an index with the given configuration.
    pub fn new(cfg: HnswConfig) -> Self {
        let ml = 1.0 / (cfg.m as f64).ln();
        Hnsw {
            inner: RwLock::new(Graph {
                nodes: Vec::new(),
                entry: None,
                max_level: 0,
                id_map: HashMap::new(),
                rng: StdRng::seed_from_u64(cfg.seed),
                cfg,
                ml,
            }),
        }
    }

    /// Create an index with default configuration for the given metric.
    pub fn with_metric(metric: Metric) -> Self {
        Self::new(HnswConfig {
            metric,
            ..Default::default()
        })
    }

    /// Number of live (non-deleted) vectors.
    pub fn len(&self) -> usize {
        self.inner.read().id_map.len()
    }

    /// Whether the index holds no live vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert a vector under `id`. Re-inserting an existing id is ignored.
    pub fn insert(&self, id: TensorId, vector: &[f32]) {
        let mut g = self.inner.write();
        g.insert(id, vector);
    }

    /// Return the `k` nearest `(id, distance)` pairs to `query`.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(TensorId, f32)> {
        let g = self.inner.read();
        g.search(query, k)
    }

    /// Tombstone-delete a vector (it is excluded from results; edges remain for
    /// navigation).
    pub fn delete(&self, id: TensorId) {
        let mut g = self.inner.write();
        if let Some(&idx) = g.id_map.get(&id) {
            g.nodes[idx as usize].deleted = true;
            g.id_map.remove(&id);
        }
    }
}

impl Graph {
    fn sample_level(&mut self) -> usize {
        let u: f64 = self.rng.gen_range(f64::MIN_POSITIVE..1.0);
        (-u.ln() * self.ml).floor() as usize
    }

    fn query_distance(&self, q: &[f32], node: u32) -> f32 {
        self.cfg
            .metric
            .distance(q, &self.nodes[node as usize].vector)
    }

    fn node_distance(&self, a: u32, b: u32) -> f32 {
        self.cfg.metric.distance(
            &self.nodes[a as usize].vector,
            &self.nodes[b as usize].vector,
        )
    }

    fn insert(&mut self, id: TensorId, vector: &[f32]) {
        if self.id_map.contains_key(&id) {
            return;
        }
        let v = self.cfg.metric.prepare(vector);
        let level = self.sample_level();
        let node_idx = self.nodes.len() as u32;
        self.nodes.push(Node {
            vector: v.clone(),
            id,
            deleted: false,
            neighbors: vec![Vec::new(); level + 1],
        });
        self.id_map.insert(id, node_idx);

        let entry = match self.entry {
            Some(e) => e,
            None => {
                self.entry = Some(node_idx);
                self.max_level = level;
                return;
            }
        };

        // Reusable visited buffer (epoch-stamped) shared across this insert's
        // search_layer calls — avoids per-call hash-set allocation.
        let mut visited = vec![0u32; self.nodes.len()];
        let mut epoch = 0u32;

        // Phase 1: greedily descend to the insertion level with ef = 1.
        let mut cur = entry;
        let mut l = self.max_level;
        while l > level {
            epoch += 1;
            let nearest = self.search_layer(&v, &[cur], 1, l, &mut visited, epoch);
            if let Some(best) = nearest.into_iter().min() {
                cur = best.node;
            }
            l -= 1;
        }

        // Phase 2: connect from min(level, max_level) down to 0.
        let m = self.cfg.m;
        let mut ep = vec![cur];
        let top = level.min(self.max_level);
        for l in (0..=top).rev() {
            epoch += 1;
            let found =
                self.search_layer(&v, &ep, self.cfg.ef_construction, l, &mut visited, epoch);
            let next_ep: Vec<u32> = found.iter().map(|c| c.node).collect();
            let selected = self.select_neighbors(found, m);
            let m_max = if l == 0 { 2 * m } else { m };

            for s in &selected {
                self.add_edge(node_idx, s.node, l);
                self.add_edge(s.node, node_idx, l);
                if self.nodes[s.node as usize].neighbors[l].len() > m_max {
                    self.prune(s.node, l, m_max);
                }
            }
            ep = if next_ep.is_empty() {
                vec![cur]
            } else {
                next_ep
            };
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry = Some(node_idx);
        }
    }

    fn add_edge(&mut self, from: u32, to: u32, level: usize) {
        let list = &mut self.nodes[from as usize].neighbors[level];
        if !list.contains(&to) {
            list.push(to);
        }
    }

    /// Re-select a node's neighbor list at `level` down to `m_max` using the
    /// heuristic.
    fn prune(&mut self, node: u32, level: usize, m_max: usize) {
        let current: Vec<u32> = self.nodes[node as usize].neighbors[level].clone();
        let cands: Vec<Cand> = current
            .iter()
            .map(|&n| Cand {
                dist: self.node_distance(node, n),
                node: n,
            })
            .collect();
        let selected = self.select_neighbors(cands, m_max);
        self.nodes[node as usize].neighbors[level] = selected.into_iter().map(|c| c.node).collect();
    }

    /// SELECT-NEIGHBORS-HEURISTIC: prefer candidates closer to the query than to
    /// any already-selected neighbor (diversity), filling up to `m` with the
    /// remaining closest if needed. Each candidate's `dist` is its distance to
    /// the query, so the query vector itself is not needed here.
    fn select_neighbors(&self, mut cands: Vec<Cand>, m: usize) -> Vec<Cand> {
        cands.sort();
        let mut result: Vec<Cand> = Vec::with_capacity(m);
        for c in &cands {
            if result.len() >= m {
                break;
            }
            let diverse = result
                .iter()
                .all(|r| self.node_distance(c.node, r.node) >= c.dist);
            if diverse {
                result.push(*c);
            }
        }
        if result.len() < m {
            for c in &cands {
                if result.len() >= m {
                    break;
                }
                if !result.iter().any(|r| r.node == c.node) {
                    result.push(*c);
                }
            }
        }
        result
    }

    /// Greedy best-first search within one layer, returning up to `ef` closest
    /// nodes found. `visited` is an epoch-stamped scratch buffer (a node is
    /// "seen" iff `visited[node] == epoch`).
    fn search_layer(
        &self,
        q: &[f32],
        entry_points: &[u32],
        ef: usize,
        level: usize,
        visited: &mut [u32],
        epoch: u32,
    ) -> Vec<Cand> {
        let mut candidates: BinaryHeap<std::cmp::Reverse<Cand>> = BinaryHeap::new();
        let mut results: BinaryHeap<Cand> = BinaryHeap::new();

        for &ep in entry_points {
            if visited[ep as usize] != epoch {
                visited[ep as usize] = epoch;
                let d = self.query_distance(q, ep);
                candidates.push(std::cmp::Reverse(Cand { dist: d, node: ep }));
                results.push(Cand { dist: d, node: ep });
            }
        }

        while let Some(std::cmp::Reverse(c)) = candidates.pop() {
            let worst = results.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
            if c.dist > worst && results.len() >= ef {
                break;
            }
            let neighbors = &self.nodes[c.node as usize].neighbors;
            if level >= neighbors.len() {
                continue;
            }
            for &n in &neighbors[level] {
                if visited[n as usize] != epoch {
                    visited[n as usize] = epoch;
                    let d = self.query_distance(q, n);
                    let worst = results.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
                    if results.len() < ef || d < worst {
                        candidates.push(std::cmp::Reverse(Cand { dist: d, node: n }));
                        results.push(Cand { dist: d, node: n });
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }
        results.into_vec()
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(TensorId, f32)> {
        let entry = match self.entry {
            Some(e) => e,
            None => return Vec::new(),
        };
        let q = self.cfg.metric.prepare(query);
        let mut visited = vec![0u32; self.nodes.len()];
        let mut epoch = 0u32;
        let mut cur = entry;
        let mut l = self.max_level;
        while l >= 1 {
            epoch += 1;
            let nearest = self.search_layer(&q, &[cur], 1, l, &mut visited, epoch);
            if let Some(best) = nearest.into_iter().min() {
                cur = best.node;
            }
            l -= 1;
        }
        let ef = self.cfg.ef_search.max(k);
        epoch += 1;
        let mut found = self.search_layer(&q, &[cur], ef, 0, &mut visited, epoch);
        found.sort();
        found
            .into_iter()
            .filter(|c| !self.nodes[c.node as usize].deleted)
            .take(k)
            .map(|c| (self.nodes[c.node as usize].id, c.dist))
            .collect()
    }
}

#[async_trait::async_trait]
impl tensorus_core::traits::VectorIndex for Hnsw {
    async fn insert(&self, id: TensorId, vector: &[f32]) -> tensorus_core::Result<()> {
        Hnsw::insert(self, id, vector);
        Ok(())
    }

    async fn search(&self, query: &[f32], k: usize) -> tensorus_core::Result<Vec<(TensorId, f32)>> {
        Ok(Hnsw::search(self, query, k))
    }

    async fn delete(&self, id: TensorId) -> tensorus_core::Result<()> {
        Hnsw::delete(self, id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet;

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect()
    }

    fn brute_force_topk(
        vectors: &[Vec<f32>],
        query: &[f32],
        k: usize,
        metric: Metric,
    ) -> Vec<usize> {
        let qp = metric.prepare(query);
        let mut scored: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, metric.distance(&qp, &metric.prepare(v))))
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored.into_iter().take(k).map(|(i, _)| i).collect()
    }

    #[test]
    fn basic_insert_search() {
        let hnsw = Hnsw::with_metric(Metric::L2);
        let ids: Vec<TensorId> = (0..5).map(|_| TensorId::new()).collect();
        hnsw.insert(ids[0], &[0.0, 0.0]);
        hnsw.insert(ids[1], &[1.0, 0.0]);
        hnsw.insert(ids[2], &[0.0, 1.0]);
        hnsw.insert(ids[3], &[5.0, 5.0]);
        hnsw.insert(ids[4], &[0.1, 0.1]);
        assert_eq!(hnsw.len(), 5);
        let res = hnsw.search(&[0.0, 0.0], 2);
        let got: Vec<TensorId> = res.iter().map(|(id, _)| *id).collect();
        // Nearest to origin are ids[0] and ids[4].
        assert!(got.contains(&ids[0]));
        assert!(got.contains(&ids[4]));
    }

    #[test]
    fn delete_excludes_from_results() {
        let hnsw = Hnsw::with_metric(Metric::L2);
        let a = TensorId::new();
        let b = TensorId::new();
        hnsw.insert(a, &[0.0, 0.0]);
        hnsw.insert(b, &[10.0, 10.0]);
        hnsw.delete(a);
        assert_eq!(hnsw.len(), 1);
        let res = hnsw.search(&[0.0, 0.0], 5);
        assert!(res.iter().all(|(id, _)| *id != a));
    }

    #[test]
    fn recall_on_synthetic() {
        let dim = 32;
        let n = 3000;
        let vectors = random_vectors(n, dim, 1);
        let hnsw = Hnsw::with_metric(Metric::L2);
        let ids: Vec<TensorId> = (0..n).map(|_| TensorId::new()).collect();
        for i in 0..n {
            hnsw.insert(ids[i], &vectors[i]);
        }

        let queries = random_vectors(100, dim, 2);
        let k = 10;
        let mut hits = 0usize;
        let mut total = 0usize;
        let id_to_idx: HashMap<TensorId, usize> =
            ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();
        for q in &queries {
            let truth: HashSet<usize> = brute_force_topk(&vectors, q, k, Metric::L2)
                .into_iter()
                .collect();
            let got = hnsw.search(q, k);
            for (id, _) in got {
                if truth.contains(&id_to_idx[&id]) {
                    hits += 1;
                }
            }
            total += k;
        }
        let recall = hits as f64 / total as f64;
        println!("HNSW recall@{k} on {n} vectors = {recall:.4}");
        assert!(recall >= 0.95, "recall too low: {recall}");
    }

    #[test]
    fn cosine_metric_works() {
        let hnsw = Hnsw::with_metric(Metric::Cosine);
        let a = TensorId::new();
        let b = TensorId::new();
        let c = TensorId::new();
        hnsw.insert(a, &[1.0, 0.0, 0.0]);
        hnsw.insert(b, &[0.9, 0.1, 0.0]); // similar direction to a
        hnsw.insert(c, &[0.0, 0.0, 1.0]); // orthogonal
        let res = hnsw.search(&[1.0, 0.0, 0.0], 2);
        let got: Vec<TensorId> = res.iter().map(|(id, _)| *id).collect();
        assert_eq!(got[0], a);
        assert!(got.contains(&b));
    }

    #[test]
    #[ignore]
    fn hnsw_50k_benchmark() {
        let dim = 128;
        let n = 10_000;
        let vectors = random_vectors(n, dim, 10);
        let hnsw = Hnsw::with_metric(Metric::L2);
        let ids: Vec<TensorId> = (0..n).map(|_| TensorId::new()).collect();

        let t = std::time::Instant::now();
        for i in 0..n {
            hnsw.insert(ids[i], &vectors[i]);
        }
        let build = t.elapsed();
        println!(
            "HNSW built {n} x {dim}-dim in {:?} ({:.0} inserts/sec)",
            build,
            n as f64 / build.as_secs_f64()
        );

        let queries = random_vectors(1000, dim, 11);
        let t = std::time::Instant::now();
        let mut found = 0;
        for q in &queries {
            found += hnsw.search(q, 10).len();
        }
        let per = t.elapsed().as_micros() as f64 / queries.len() as f64;
        println!("HNSW search avg {per:.1} us/query (found={found})");

        // Recall check.
        let id_to_idx: HashMap<TensorId, usize> =
            ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();
        let mut hits = 0;
        for q in queries.iter().take(100) {
            let truth: HashSet<usize> = brute_force_topk(&vectors, q, 10, Metric::L2)
                .into_iter()
                .collect();
            for (id, _) in hnsw.search(q, 10) {
                if truth.contains(&id_to_idx[&id]) {
                    hits += 1;
                }
            }
        }
        println!("HNSW recall@10 = {:.4}", hits as f64 / 1000.0);
    }
}
