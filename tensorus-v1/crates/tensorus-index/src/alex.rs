//! ALEX-style dynamic learned index supporting concurrent inserts and lookups.
//!
//! Structure:
//! - An **internal routing node** maps a key to a leaf. Routing uses a sorted
//!   array of per-leaf minimum keys (`pivots`); the routing decision is a single
//!   `partition_point`.
//! - **Leaves** are model-accelerated sorted arrays with capacity headroom (a
//!   pragmatic stand-in for ALEX's gapped array): a per-leaf linear model
//!   predicts a key's slot, and a bounded local search around the prediction
//!   resolves it exactly (with a binary-search fallback for guaranteed
//!   correctness). Inserts shift within the (small) leaf; when a leaf exceeds
//!   `MAX_LEAF` entries it **splits** and a new pivot is published.
//!
//! Concurrency: lookups and non-splitting inserts take a shared read lock on the
//! routing node plus a per-leaf mutex, so operations on different leaves proceed
//! in parallel. The rare split takes the routing write lock briefly.
//!
//! Reference: Ding et al., "ALEX: An Updatable Adaptive Learned Index," SIGMOD (2020).

use parking_lot::{Mutex, RwLock};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Maximum entries in a leaf before it splits.
const MAX_LEAF: usize = 256;
/// Target fill fraction; leaves are built with `len / DENSITY` capacity.
const DENSITY: f64 = 0.7;

/// A model-accelerated sorted leaf.
struct Leaf {
    keys: Vec<f64>,
    payloads: Vec<u64>,
    slope: f64,
    intercept: f64,
}

impl Leaf {
    fn new() -> Self {
        let mut keys = Vec::with_capacity((MAX_LEAF as f64 / DENSITY) as usize);
        keys.clear();
        Leaf {
            keys,
            payloads: Vec::with_capacity((MAX_LEAF as f64 / DENSITY) as usize),
            slope: 0.0,
            intercept: 0.0,
        }
    }

    fn from_sorted(keys: Vec<f64>, payloads: Vec<u64>) -> Self {
        let mut leaf = Leaf {
            keys,
            payloads,
            slope: 0.0,
            intercept: 0.0,
        };
        leaf.refit();
        leaf
    }

    fn len(&self) -> usize {
        self.keys.len()
    }

    fn first_key(&self) -> f64 {
        *self.keys.first().unwrap_or(&f64::NEG_INFINITY)
    }

    /// Refit the linear model by interpolating between the first and last key.
    fn refit(&mut self) {
        let n = self.keys.len();
        if n <= 1 {
            self.slope = 0.0;
            self.intercept = 0.0;
            return;
        }
        let (k0, kn) = (self.keys[0], self.keys[n - 1]);
        if kn > k0 {
            self.slope = (n - 1) as f64 / (kn - k0);
            self.intercept = -self.slope * k0;
        } else {
            self.slope = 0.0;
            self.intercept = 0.0;
        }
    }

    fn predict(&self, key: f64) -> usize {
        let n = self.keys.len();
        if n == 0 {
            return 0;
        }
        let p = self.slope * key + self.intercept;
        if p <= 0.0 {
            0
        } else if p >= (n - 1) as f64 {
            n - 1
        } else {
            p as usize
        }
    }

    /// First index with `keys[idx] >= key`, model-accelerated with fallback.
    fn lower_bound(&self, key: f64) -> usize {
        let n = self.keys.len();
        if n == 0 {
            return 0;
        }
        let p = self.predict(key);
        let lo = p.saturating_sub(MAX_LEAF / 4 + 1);
        let hi = (p + MAX_LEAF / 4 + 2).min(n);
        let idx = lo + self.keys[lo..hi].partition_point(|&k| k < key);
        let valid = (idx == 0 || self.keys[idx - 1] < key) && (idx == n || self.keys[idx] >= key);
        if valid {
            idx
        } else {
            self.keys.partition_point(|&k| k < key)
        }
    }

    fn insert(&mut self, key: f64, payload: u64) {
        let pos = self.lower_bound(key);
        self.keys.insert(pos, key);
        self.payloads.insert(pos, payload);
        // Occasionally refit so predictions track growth.
        if self.keys.len().is_power_of_two() {
            self.refit();
        }
    }

    fn lookup(&self, key: f64, out: &mut Vec<u64>) {
        let mut i = self.lower_bound(key);
        while i < self.keys.len() && self.keys[i] == key {
            out.push(self.payloads[i]);
            i += 1;
        }
    }

    fn range(&self, min: f64, max: f64, out: &mut Vec<u64>) {
        let mut i = self.lower_bound(min);
        while i < self.keys.len() && self.keys[i] <= max {
            out.push(self.payloads[i]);
            i += 1;
        }
    }

    /// Split into the upper half at a *key boundary* (so equal keys are never
    /// spread across leaves), returning the new right-hand leaf. Returns `None`
    /// when the leaf is a single run of one key and cannot be split cleanly.
    fn split_off(&mut self) -> Option<Leaf> {
        let n = self.keys.len();
        // Search outward from the midpoint for a boundary between distinct keys.
        let half = n / 2;
        let mut mid = half;
        while mid < n && self.keys[mid] == self.keys[mid - 1] {
            mid += 1;
        }
        if mid >= n {
            mid = half;
            while mid > 0 && self.keys[mid] == self.keys[mid - 1] {
                mid -= 1;
            }
        }
        if mid == 0 || mid >= n {
            return None; // entire leaf is one key run
        }
        let right_keys = self.keys.split_off(mid);
        let right_payloads = self.payloads.split_off(mid);
        self.refit();
        Some(Leaf::from_sorted(right_keys, right_payloads))
    }
}

struct Routing {
    /// Minimum key routed to each leaf (sorted ascending; `pivots[i]` is the
    /// lower bound of `leaves[i]`).
    pivots: Vec<f64>,
    leaves: Vec<Arc<Mutex<Leaf>>>,
}

/// A concurrent, updatable learned index.
pub struct AlexIndex {
    routing: RwLock<Routing>,
    len: AtomicUsize,
}

impl Default for AlexIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl AlexIndex {
    /// Create an empty index with a single leaf.
    pub fn new() -> Self {
        AlexIndex {
            routing: RwLock::new(Routing {
                pivots: vec![f64::NEG_INFINITY],
                leaves: vec![Arc::new(Mutex::new(Leaf::new()))],
            }),
            len: AtomicUsize::new(0),
        }
    }

    /// Bulk-load from `(key, payload)` pairs, partitioning into balanced leaves.
    pub fn bulk_load(mut entries: Vec<(f64, u64)>) -> Self {
        entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        if entries.is_empty() {
            return Self::new();
        }
        let chunk = ((MAX_LEAF as f64) * DENSITY) as usize;
        let chunk = chunk.max(1);
        let mut pivots = Vec::new();
        let mut leaves = Vec::new();
        let total = entries.len();
        let mut i = 0;
        while i < total {
            let end = (i + chunk).min(total);
            let keys: Vec<f64> = entries[i..end].iter().map(|e| e.0).collect();
            let payloads: Vec<u64> = entries[i..end].iter().map(|e| e.1).collect();
            pivots.push(if i == 0 { f64::NEG_INFINITY } else { keys[0] });
            leaves.push(Arc::new(Mutex::new(Leaf::from_sorted(keys, payloads))));
            i = end;
        }
        AlexIndex {
            routing: RwLock::new(Routing { pivots, leaves }),
            len: AtomicUsize::new(total),
        }
    }

    /// Total number of entries.
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// Whether the index holds no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of leaves (grows as the index splits).
    pub fn leaf_count(&self) -> usize {
        self.routing.read().leaves.len()
    }

    fn route(pivots: &[f64], key: f64) -> usize {
        pivots.partition_point(|&p| p <= key).saturating_sub(1)
    }

    /// Insert a `(key, payload)` pair. Thread-safe.
    pub fn insert(&self, key: f64, payload: u64) {
        let needs_split = {
            let routing = self.routing.read();
            let idx = Self::route(&routing.pivots, key);
            let leaf_arc = routing.leaves[idx].clone();
            let mut leaf = leaf_arc.lock();
            leaf.insert(key, payload);
            leaf.len() > MAX_LEAF
        };
        self.len.fetch_add(1, Ordering::Relaxed);
        if needs_split {
            self.try_split(key);
        }
    }

    fn try_split(&self, key: f64) {
        let mut routing = self.routing.write();
        let idx = Self::route(&routing.pivots, key);
        let (new_pivot, right_leaf) = {
            let mut leaf = routing.leaves[idx].lock();
            if leaf.len() <= MAX_LEAF {
                return; // already handled by another thread
            }
            match leaf.split_off() {
                Some(right) => (right.first_key(), Arc::new(Mutex::new(right))),
                None => return, // single-key run; cannot split cleanly
            }
        };
        routing.pivots.insert(idx + 1, new_pivot);
        routing.leaves.insert(idx + 1, right_leaf);
    }

    /// All payloads whose key equals `key` exactly.
    pub fn lookup(&self, key: f64) -> Vec<u64> {
        let routing = self.routing.read();
        let idx = Self::route(&routing.pivots, key);
        let mut out = Vec::new();
        // Equal keys may straddle a leaf boundary after a split; scan forward
        // while a leaf can still contain the key.
        let mut i = idx;
        while i < routing.leaves.len() {
            if i > idx && routing.pivots[i] > key {
                break;
            }
            routing.leaves[i].lock().lookup(key, &mut out);
            i += 1;
        }
        out
    }

    /// All payloads whose key lies in the inclusive range `[min, max]`.
    pub fn range(&self, min: f64, max: f64) -> Vec<u64> {
        if min > max {
            return Vec::new();
        }
        let routing = self.routing.read();
        let start = Self::route(&routing.pivots, min);
        let mut out = Vec::new();
        let mut i = start;
        while i < routing.leaves.len() {
            if routing.pivots[i] > max {
                break;
            }
            routing.leaves[i].lock().range(min, max, &mut out);
            i += 1;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashMap;

    #[test]
    fn empty_then_insert() {
        let idx = AlexIndex::new();
        assert!(idx.is_empty());
        idx.insert(3.0, 30);
        idx.insert(1.0, 10);
        idx.insert(2.0, 20);
        assert_eq!(idx.len(), 3);
        assert_eq!(idx.lookup(2.0), vec![20]);
        assert_eq!(idx.lookup(5.0), Vec::<u64>::new());
        let mut r = idx.range(1.0, 2.0);
        r.sort_unstable();
        assert_eq!(r, vec![10, 20]);
    }

    #[test]
    fn splits_and_stays_correct() {
        let idx = AlexIndex::new();
        // Insert far more than one leaf to force multiple splits.
        for i in 0..5_000u64 {
            idx.insert(i as f64, i);
        }
        assert_eq!(idx.len(), 5_000);
        assert!(idx.leaf_count() > 1, "expected splits");
        for i in 0..5_000u64 {
            assert_eq!(idx.lookup(i as f64), vec![i]);
        }
        assert_eq!(idx.range_count_helper(100.0, 199.0), 100);
    }

    impl AlexIndex {
        fn range_count_helper(&self, min: f64, max: f64) -> usize {
            self.range(min, max).len()
        }
    }

    #[test]
    fn mixed_insert_lookup_with_duplicates() {
        let mut rng = StdRng::seed_from_u64(11);
        let idx = AlexIndex::new();
        let mut reference: HashMap<i64, Vec<u64>> = HashMap::new();
        for payload in 0..10_000u64 {
            let key = rng.gen_range(0..2_000) as i64;
            idx.insert(key as f64, payload);
            reference.entry(key).or_default().push(payload);
            // Interleave a lookup of a previously inserted key.
            if payload % 3 == 0 {
                let probe = rng.gen_range(0..2_000) as i64;
                let mut got = idx.lookup(probe as f64);
                got.sort_unstable();
                let mut exp = reference.get(&probe).cloned().unwrap_or_default();
                exp.sort_unstable();
                assert_eq!(got, exp, "lookup({probe}) mismatch mid-workload");
            }
        }
        // Final full verification.
        for (k, mut exp) in reference {
            exp.sort_unstable();
            let mut got = idx.lookup(k as f64);
            got.sort_unstable();
            assert_eq!(got, exp);
        }
    }

    #[test]
    fn bulk_load_then_query() {
        let entries: Vec<(f64, u64)> = (0..3_000u64).map(|i| ((i * 2) as f64, i)).collect();
        let idx = AlexIndex::bulk_load(entries);
        assert_eq!(idx.len(), 3_000);
        assert_eq!(idx.lookup(10.0), vec![5]);
        assert_eq!(idx.lookup(11.0), Vec::<u64>::new());
        assert_eq!(idx.range(0.0, 18.0).len(), 10);
    }

    #[test]
    fn concurrent_inserts_and_lookups() {
        use std::thread;
        let idx = Arc::new(AlexIndex::new());
        let n_threads = 8;
        let per = 5_000u64;

        let mut handles = Vec::new();
        for t in 0..n_threads {
            let idx = idx.clone();
            handles.push(thread::spawn(move || {
                let base = t * per;
                for i in 0..per {
                    let key = base + i;
                    idx.insert(key as f64, key);
                    // Concurrent reads of our own just-inserted key.
                    let got = idx.lookup(key as f64);
                    assert!(got.contains(&key), "missing own key {key}");
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(idx.len(), (n_threads * per) as usize);
        // Every key from every thread is present exactly once.
        for t in 0..n_threads {
            for i in 0..per {
                let key = t * per + i;
                assert_eq!(idx.lookup(key as f64), vec![key]);
            }
        }
    }

    #[test]
    #[ignore]
    fn alex_insert_throughput() {
        let mut rng = StdRng::seed_from_u64(99);
        let keys: Vec<f64> = (0..1_000_000).map(|_| rng.gen_range(0.0..1.0e9)).collect();
        let idx = AlexIndex::new();
        let start = std::time::Instant::now();
        for (i, &k) in keys.iter().enumerate() {
            idx.insert(k, i as u64);
        }
        let elapsed = start.elapsed();
        let per_sec = 1_000_000.0 / elapsed.as_secs_f64();
        println!(
            "ALEX inserted 1M random keys in {:?} ({:.0}/sec), {} leaves",
            elapsed,
            per_sec,
            idx.leaf_count()
        );
    }
}
