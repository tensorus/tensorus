//! PGM-Index: a piecewise geometric (learned) index for sub-microsecond
//! numeric range and point queries over sorted keys.
//!
//! A PGM-Index approximates the mapping `key -> position` in a sorted array
//! with a sequence of linear models ("segments"), each guaranteed to predict a
//! key's position within a bounded error `epsilon`. A lookup evaluates the
//! relevant segment's line (O(1)) and then performs a binary search in the
//! `2*epsilon` window around the prediction.
//!
//! ## Segmentation
//!
//! Segments are built with a streaming O(n) algorithm. Anchoring each segment's
//! line at its first point `(x0, y0)`, every later point `(xj, j)` constrains
//! the feasible slope to the interval `[(j-ε-y0)/(xj-x0), (j+ε-y0)/(xj-x0)]`.
//! The segment is extended while the running intersection of these intervals is
//! non-empty; the chosen slope (interval midpoint) then keeps every member
//! within `ε` of its true position. This is slightly more conservative than the
//! optimal convex-hull method (it may emit a few more segments) but is simple
//! and provably correct.
//!
//! ## Correctness
//!
//! Every windowed search is validated against the true sorted order; on the
//! rare miss (e.g. a run of duplicate keys longer than `ε`) it falls back to a
//! full binary search, so results are always exact.
//!
//! Reference: Ferragina & Vinciguerra, "The PGM-Index," PVLDB (2020).

/// Default maximum prediction error (in positions).
pub const DEFAULT_EPSILON: usize = 64;

/// A linear model covering a contiguous run of sorted keys.
#[derive(Debug, Clone, Copy)]
struct Segment {
    /// First key covered by this segment.
    first_key: f64,
    slope: f64,
    intercept: f64,
}

/// A learned index over `(key, payload)` pairs.
#[derive(Debug, Clone)]
pub struct PgmIndex {
    keys: Vec<f64>,
    payloads: Vec<u64>,
    segments: Vec<Segment>,
    epsilon: usize,
}

impl PgmIndex {
    /// Build an index from `(key, payload)` pairs using [`DEFAULT_EPSILON`].
    pub fn build(entries: Vec<(f64, u64)>) -> Self {
        Self::build_with_epsilon(entries, DEFAULT_EPSILON)
    }

    /// Build an index with an explicit error bound.
    pub fn build_with_epsilon(mut entries: Vec<(f64, u64)>, epsilon: usize) -> Self {
        entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let keys: Vec<f64> = entries.iter().map(|e| e.0).collect();
        let payloads: Vec<u64> = entries.iter().map(|e| e.1).collect();
        let segments = Self::build_segments(&keys, epsilon.max(1));
        PgmIndex {
            keys,
            payloads,
            segments,
            epsilon: epsilon.max(1),
        }
    }

    fn build_segments(keys: &[f64], epsilon: usize) -> Vec<Segment> {
        let n = keys.len();
        let eps = epsilon as f64;
        let mut segments = Vec::new();
        let mut i = 0usize;
        while i < n {
            let x0 = keys[i];
            let y0 = i as f64;
            let mut lo = f64::NEG_INFINITY;
            let mut hi = f64::INFINITY;
            let mut j = i + 1;
            while j < n {
                let xj = keys[j];
                let yj = j as f64;
                let dx = xj - x0;
                if dx == 0.0 {
                    // Duplicate key: prediction at x0 is y0 for any slope.
                    if (yj - y0).abs() > eps {
                        break;
                    }
                } else {
                    let lo_j = (yj - eps - y0) / dx;
                    let hi_j = (yj + eps - y0) / dx;
                    let new_lo = lo.max(lo_j);
                    let new_hi = hi.min(hi_j);
                    if new_lo > new_hi {
                        break;
                    }
                    lo = new_lo;
                    hi = new_hi;
                }
                j += 1;
            }
            let slope = match (lo.is_finite(), hi.is_finite()) {
                (true, true) => (lo + hi) / 2.0,
                (true, false) => lo,
                (false, true) => hi,
                (false, false) => 0.0,
            };
            let intercept = y0 - slope * x0;
            segments.push(Segment {
                first_key: x0,
                slope,
                intercept,
            });
            i = j;
        }
        segments
    }

    /// Number of indexed entries.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Number of linear segments (a measure of model compactness).
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    fn predict(&self, q: f64) -> usize {
        if self.keys.is_empty() {
            return 0;
        }
        let idx = self.segments.partition_point(|s| s.first_key <= q);
        let si = idx.saturating_sub(1);
        let s = &self.segments[si];
        let p = s.slope * q + s.intercept;
        if p <= 0.0 {
            0
        } else if p >= (self.keys.len() - 1) as f64 {
            self.keys.len() - 1
        } else {
            p as usize
        }
    }

    /// First index whose key is `>= q` (a la `lower_bound`), accelerated by the
    /// learned model with a binary-search fallback for correctness.
    fn lower_bound(&self, q: f64) -> usize {
        let n = self.keys.len();
        if n == 0 {
            return 0;
        }
        let p = self.predict(q);
        let lo = p.saturating_sub(self.epsilon + 1);
        let hi = (p + self.epsilon + 2).min(n);
        let idx = lo + self.keys[lo..hi].partition_point(|&k| k < q);
        let valid = (idx == 0 || self.keys[idx - 1] < q) && (idx == n || self.keys[idx] >= q);
        if valid {
            idx
        } else {
            self.keys.partition_point(|&k| k < q)
        }
    }

    /// First index whose key is `> q` (a la `upper_bound`).
    fn upper_bound(&self, q: f64) -> usize {
        let n = self.keys.len();
        if n == 0 {
            return 0;
        }
        let p = self.predict(q);
        let lo = p.saturating_sub(self.epsilon + 1);
        let hi = (p + self.epsilon + 2).min(n);
        let idx = lo + self.keys[lo..hi].partition_point(|&k| k <= q);
        let valid = (idx == 0 || self.keys[idx - 1] <= q) && (idx == n || self.keys[idx] > q);
        if valid {
            idx
        } else {
            self.keys.partition_point(|&k| k <= q)
        }
    }

    /// All payloads whose key equals `q` exactly.
    pub fn lookup(&self, q: f64) -> Vec<u64> {
        let start = self.lower_bound(q);
        let mut out = Vec::new();
        let mut i = start;
        while i < self.keys.len() && self.keys[i] == q {
            out.push(self.payloads[i]);
            i += 1;
        }
        out
    }

    /// All payloads whose key lies in the inclusive range `[min, max]`.
    pub fn range(&self, min: f64, max: f64) -> Vec<u64> {
        if min > max {
            return Vec::new();
        }
        let start = self.lower_bound(min);
        let end = self.upper_bound(max);
        self.payloads[start..end].to_vec()
    }

    /// Count of payloads in the inclusive range `[min, max]` (cheaper than
    /// materializing them).
    pub fn range_count(&self, min: f64, max: f64) -> usize {
        if min > max {
            return 0;
        }
        self.upper_bound(max) - self.lower_bound(min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn brute_range(entries: &[(f64, u64)], min: f64, max: f64) -> Vec<u64> {
        let mut v: Vec<u64> = entries
            .iter()
            .filter(|(k, _)| *k >= min && *k <= max)
            .map(|(_, p)| *p)
            .collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn empty_index() {
        let idx = PgmIndex::build(vec![]);
        assert!(idx.is_empty());
        assert_eq!(idx.lookup(1.0), Vec::<u64>::new());
        assert_eq!(idx.range(0.0, 10.0), Vec::<u64>::new());
    }

    #[test]
    fn point_lookup_with_duplicates() {
        let entries = vec![
            (1.0, 10),
            (2.0, 20),
            (2.0, 21),
            (2.0, 22),
            (5.0, 50),
            (9.0, 90),
        ];
        let idx = PgmIndex::build(entries);
        let mut got = idx.lookup(2.0);
        got.sort_unstable();
        assert_eq!(got, vec![20, 21, 22]);
        assert_eq!(idx.lookup(1.0), vec![10]);
        assert_eq!(idx.lookup(3.0), Vec::<u64>::new());
        assert_eq!(idx.lookup(9.0), vec![90]);
    }

    #[test]
    fn range_query_basic() {
        let entries: Vec<(f64, u64)> = (0..1000).map(|i| (i as f64, i as u64)).collect();
        let idx = PgmIndex::build(entries);
        let mut got = idx.range(100.0, 200.0);
        got.sort_unstable();
        let expected: Vec<u64> = (100..=200).collect();
        assert_eq!(got, expected);
        assert_eq!(idx.range_count(100.0, 200.0), 101);
    }

    #[test]
    fn correctness_vs_linear_scan_random() {
        let mut rng = StdRng::seed_from_u64(42);
        // 20k random keys in [0, 10_000) -> guaranteed duplicates.
        let entries: Vec<(f64, u64)> = (0..20_000u64)
            .map(|i| (rng.gen_range(0..10_000) as f64, i))
            .collect();
        let idx = PgmIndex::build_with_epsilon(entries.clone(), 64);

        for _ in 0..500 {
            let a = rng.gen_range(0..10_000) as f64;
            let b = rng.gen_range(0..10_000) as f64;
            let (min, max) = if a <= b { (a, b) } else { (b, a) };
            let mut got = idx.range(min, max);
            got.sort_unstable();
            let expected = brute_range(&entries, min, max);
            assert_eq!(got, expected, "range [{min}, {max}] mismatch");
            assert_eq!(idx.range_count(min, max), expected.len());
        }

        // Exact point lookups for every distinct key.
        for k in 0..10_000u64 {
            let mut got = idx.lookup(k as f64);
            got.sort_unstable();
            let mut expected: Vec<u64> = entries
                .iter()
                .filter(|(key, _)| *key == k as f64)
                .map(|(_, p)| *p)
                .collect();
            expected.sort_unstable();
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn lookups_after_duplicate_run_longer_than_epsilon() {
        // 500 copies of the same key (>> epsilon) exercises the fallback path.
        let mut entries: Vec<(f64, u64)> = (0..500).map(|i| (7.0, i)).collect();
        entries.push((1.0, 1000));
        entries.push((9.0, 1001));
        let idx = PgmIndex::build_with_epsilon(entries, 16);
        assert_eq!(idx.lookup(7.0).len(), 500);
        assert_eq!(idx.lookup(1.0), vec![1000]);
        assert_eq!(idx.range_count(1.0, 7.0), 501);
    }

    /// Large-scale latency benchmark. Ignored by default (slow to build in
    /// debug); run with `cargo test --release -- --ignored pgm_10m`.
    #[test]
    #[ignore]
    fn pgm_10m_latency() {
        let mut rng = StdRng::seed_from_u64(7);
        let n = 10_000_000u64;
        let entries: Vec<(f64, u64)> = (0..n).map(|i| (rng.gen_range(0.0..1.0e9), i)).collect();
        let build_start = std::time::Instant::now();
        let idx = PgmIndex::build_with_epsilon(entries, 64);
        println!(
            "built {} keys into {} segments in {:?}",
            idx.len(),
            idx.segment_count(),
            build_start.elapsed()
        );

        // Point lookup latency (median over many probes).
        let probes: Vec<f64> = (0..10_000).map(|_| rng.gen_range(0.0..1.0e9)).collect();
        let t = std::time::Instant::now();
        let mut sink = 0usize;
        for &q in &probes {
            sink += idx.range_count(q, q + 1.0);
        }
        let per = t.elapsed().as_nanos() as f64 / probes.len() as f64;
        println!("point/narrow-range query avg {per:.1} ns (sink={sink})");

        // Range query latency.
        let t = std::time::Instant::now();
        let mut total = 0usize;
        for _ in 0..1_000 {
            let lo = rng.gen_range(0.0..1.0e9);
            total += idx.range_count(lo, lo + 1.0e6);
        }
        let per = t.elapsed().as_nanos() as f64 / 1_000.0;
        println!("range_count avg {per:.1} ns (total={total})");
    }
}
