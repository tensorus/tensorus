//! Tiered storage: a byte-budgeted in-memory **hot** cache backed by a durable
//! **cold** store, with access tracking and LRU/LFU eviction.
//!
//! Reads promote items into the hot tier; when the hot tier exceeds its byte
//! budget the configured policy evicts the least-valuable items (which remain
//! safely in the cold tier, so nothing is ever lost).
//!
//! ## Substitution note
//!
//! The plan specifies the `object_store` crate (S3/GCS/Azure) for the cold tier.
//! To keep the dependency surface light, the cold tier is abstracted behind the
//! [`ColdStore`] trait with a filesystem implementation ([`LocalColdStore`])
//! that "simulates S3" exactly as the acceptance criteria describe; an
//! `object_store`-backed implementation is a drop-in behind the same trait.
// TODO: background demotion thread (currently eviction is synchronous, which is
// O(evicted) per write and does not block reads); GPU HBM pinning for the
// hottest tensors.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tensorus_core::error::{Result, TensorusError};
use tensorus_core::types::TensorId;

/// Eviction policy for the hot tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// Evict the least-recently-used item.
    Lru,
    /// Evict the least-frequently-used item.
    Lfu,
}

/// Tiering configuration.
#[derive(Debug, Clone, Copy)]
pub struct TieringConfig {
    /// Maximum bytes resident in the hot tier.
    pub hot_max_bytes: usize,
    pub policy: Policy,
}

impl Default for TieringConfig {
    fn default() -> Self {
        TieringConfig {
            hot_max_bytes: 1 << 30, // 1 GiB
            policy: Policy::Lfu,
        }
    }
}

/// A durable cold tier.
pub trait ColdStore: Send + Sync {
    fn put(&self, id: TensorId, bytes: &[u8]) -> Result<()>;
    fn get(&self, id: TensorId) -> Result<Option<Vec<u8>>>;
    fn delete(&self, id: TensorId) -> Result<()>;
}

/// Filesystem cold store (one file per tensor). Stands in for an S3/GCS object
/// store.
pub struct LocalColdStore {
    dir: PathBuf,
}

impl LocalColdStore {
    pub fn new(dir: impl AsRef<Path>) -> Result<Self> {
        std::fs::create_dir_all(dir.as_ref())?;
        Ok(LocalColdStore {
            dir: dir.as_ref().to_path_buf(),
        })
    }

    fn path(&self, id: TensorId) -> PathBuf {
        self.dir.join(format!("{id}.bin"))
    }
}

impl ColdStore for LocalColdStore {
    fn put(&self, id: TensorId, bytes: &[u8]) -> Result<()> {
        std::fs::write(self.path(id), bytes)?;
        Ok(())
    }

    fn get(&self, id: TensorId) -> Result<Option<Vec<u8>>> {
        match std::fs::read(self.path(id)) {
            Ok(b) => Ok(Some(b)),
            Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn delete(&self, id: TensorId) -> Result<()> {
        match std::fs::remove_file(self.path(id)) {
            Ok(()) => Ok(()),
            Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
    }
}

struct HotEntry {
    bytes: Vec<u8>,
    /// Access frequency (LFU).
    count: u64,
    /// Logical clock of last access (LRU).
    last: u64,
}

struct HotTier {
    entries: HashMap<TensorId, HotEntry>,
    total_bytes: usize,
}

/// A two-tier store: hot (DRAM, budgeted) over cold (durable).
pub struct TieredStore<C: ColdStore> {
    cold: C,
    hot: Mutex<HotTier>,
    cfg: TieringConfig,
    clock: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl<C: ColdStore> TieredStore<C> {
    /// Create a tiered store over the given cold backend.
    pub fn new(cold: C, cfg: TieringConfig) -> Self {
        TieredStore {
            cold,
            hot: Mutex::new(HotTier {
                entries: HashMap::new(),
                total_bytes: 0,
            }),
            cfg,
            clock: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    fn tick(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::Relaxed)
    }

    /// Store a tensor: write-through to cold, then cache hot (evicting as
    /// needed). Durability is provided by the cold tier.
    pub fn put(&self, id: TensorId, bytes: Vec<u8>) -> Result<()> {
        self.cold.put(id, &bytes)?;
        let now = self.tick();
        let mut hot = self.hot.lock();
        self.insert_hot(&mut hot, id, bytes, now);
        self.evict(&mut hot);
        Ok(())
    }

    fn insert_hot(&self, hot: &mut HotTier, id: TensorId, bytes: Vec<u8>, now: u64) {
        let len = bytes.len();
        if let Some(prev) = hot.entries.insert(
            id,
            HotEntry {
                bytes,
                count: 1,
                last: now,
            },
        ) {
            hot.total_bytes -= prev.bytes.len();
        }
        hot.total_bytes += len;
    }

    /// Fetch a tensor, promoting it into the hot tier. Returns `NotFound` if it
    /// exists in neither tier.
    pub fn get(&self, id: TensorId) -> Result<Vec<u8>> {
        {
            let mut hot = self.hot.lock();
            if let Some(entry) = hot.entries.get_mut(&id) {
                entry.count += 1;
                entry.last = self.clock.fetch_add(1, Ordering::Relaxed);
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(entry.bytes.clone());
            }
        }
        // Cold fetch (promotion) without holding the hot lock during I/O.
        self.misses.fetch_add(1, Ordering::Relaxed);
        let bytes = self
            .cold
            .get(id)?
            .ok_or_else(|| TensorusError::NotFound(format!("tensor {id} in tiered store")))?;
        let now = self.tick();
        let mut hot = self.hot.lock();
        self.insert_hot(&mut hot, id, bytes.clone(), now);
        self.evict(&mut hot);
        Ok(bytes)
    }

    /// Delete from both tiers.
    pub fn delete(&self, id: TensorId) -> Result<()> {
        {
            let mut hot = self.hot.lock();
            if let Some(e) = hot.entries.remove(&id) {
                hot.total_bytes -= e.bytes.len();
            }
        }
        self.cold.delete(id)
    }

    /// Evict victims until the hot tier is within budget.
    fn evict(&self, hot: &mut HotTier) {
        while hot.total_bytes > self.cfg.hot_max_bytes && !hot.entries.is_empty() {
            let victim = match self.cfg.policy {
                Policy::Lru => hot
                    .entries
                    .iter()
                    .min_by_key(|(_, e)| e.last)
                    .map(|(id, _)| *id),
                Policy::Lfu => hot
                    .entries
                    .iter()
                    .min_by_key(|(_, e)| (e.count, e.last))
                    .map(|(id, _)| *id),
            };
            match victim {
                Some(id) => {
                    if let Some(e) = hot.entries.remove(&id) {
                        hot.total_bytes -= e.bytes.len();
                    }
                }
                None => break,
            }
        }
    }

    /// Whether a tensor is currently resident in the hot tier.
    pub fn is_hot(&self, id: TensorId) -> bool {
        self.hot.lock().entries.contains_key(&id)
    }

    /// Current hot-tier byte occupancy.
    pub fn hot_bytes(&self) -> usize {
        self.hot.lock().total_bytes
    }

    /// Number of items resident in the hot tier.
    pub fn hot_len(&self) -> usize {
        self.hot.lock().entries.len()
    }

    /// Hit rate over all `get` calls so far.
    pub fn hit_rate(&self) -> f64 {
        let h = self.hits.load(Ordering::Relaxed) as f64;
        let m = self.misses.load(Ordering::Relaxed) as f64;
        if h + m == 0.0 {
            0.0
        } else {
            h / (h + m)
        }
    }

    /// Reset the hit/miss counters (e.g. after a cache warm-up phase).
    pub fn reset_stats(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use tempfile::TempDir;

    fn store(tmp: &TempDir, budget: usize, policy: Policy) -> TieredStore<LocalColdStore> {
        let cold = LocalColdStore::new(tmp.path().join("cold")).unwrap();
        TieredStore::new(
            cold,
            TieringConfig {
                hot_max_bytes: budget,
                policy,
            },
        )
    }

    fn payload(i: usize) -> Vec<u8> {
        (i as u64).to_le_bytes().to_vec()
    }

    #[test]
    fn budget_is_respected() {
        let tmp = TempDir::new().unwrap();
        let s = store(&tmp, 8 * 10, Policy::Lru); // room for 10 items of 8 bytes
        let ids: Vec<TensorId> = (0..100).map(|_| TensorId::new()).collect();
        for (i, id) in ids.iter().enumerate() {
            s.put(*id, payload(i)).unwrap();
        }
        assert!(s.hot_bytes() <= 8 * 10, "hot bytes {}", s.hot_bytes());
        assert!(s.hot_len() <= 10);
    }

    #[test]
    fn no_data_loss_after_eviction() {
        let tmp = TempDir::new().unwrap();
        let s = store(&tmp, 8 * 5, Policy::Lru);
        let ids: Vec<TensorId> = (0..200).map(|_| TensorId::new()).collect();
        for (i, id) in ids.iter().enumerate() {
            s.put(*id, payload(i)).unwrap();
        }
        // Every tensor remains retrievable (from hot or cold).
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(s.get(*id).unwrap(), payload(i));
        }
    }

    #[test]
    fn promotion_from_cold() {
        let tmp = TempDir::new().unwrap();
        let s = store(&tmp, 8 * 3, Policy::Lru);
        let target = TensorId::new();
        s.put(target, payload(42)).unwrap();
        // Push the target out of the hot tier.
        for i in 0..50 {
            s.put(TensorId::new(), payload(i)).unwrap();
        }
        assert!(!s.is_hot(target), "target should have been evicted");
        let got = s.get(target).unwrap();
        assert_eq!(got, payload(42));
        assert!(s.is_hot(target), "get should promote into hot tier");
    }

    #[test]
    fn missing_tensor_errors() {
        let tmp = TempDir::new().unwrap();
        let s = store(&tmp, 1024, Policy::Lfu);
        assert!(s.get(TensorId::new()).is_err());
    }

    #[test]
    fn delete_removes_from_both_tiers() {
        let tmp = TempDir::new().unwrap();
        let s = store(&tmp, 1024, Policy::Lfu);
        let id = TensorId::new();
        s.put(id, payload(1)).unwrap();
        s.delete(id).unwrap();
        assert!(s.get(id).is_err());
        assert!(!s.is_hot(id));
    }

    #[test]
    fn zipfian_hit_rate_above_90_percent() {
        let tmp = TempDir::new().unwrap();
        let n = 5_000usize;
        // Hot budget holds ~1000 of 5000 items (8 bytes each).
        let s = store(&tmp, 8 * 1_000, Policy::Lfu);
        let ids: Vec<TensorId> = (0..n).map(|_| TensorId::new()).collect();
        for (i, id) in ids.iter().enumerate() {
            s.put(*id, payload(i)).unwrap();
        }

        let mut rng = StdRng::seed_from_u64(2024);
        // Skewed (Zipfian-style) workload: a small hot set of 500 items receives
        // ~95% of traffic, the rest is spread across the whole dataset. The hot
        // set fits comfortably in the 1000-item budget.
        const HOT_SET: usize = 500;
        let sample = |rng: &mut StdRng| -> usize {
            if rng.gen_range(0.0..1.0) < 0.95 {
                rng.gen_range(0..HOT_SET)
            } else {
                rng.gen_range(0..n)
            }
        };

        // Warm up the cache, then measure steady-state hit rate.
        for _ in 0..20_000 {
            let i = sample(&mut rng);
            let _ = s.get(ids[i]).unwrap();
        }
        s.reset_stats();
        for _ in 0..50_000 {
            let i = sample(&mut rng);
            let _ = s.get(ids[i]).unwrap();
        }
        let hit_rate = s.hit_rate();
        println!("Zipfian-style hot-tier hit rate = {:.3}", hit_rate);
        assert!(hit_rate > 0.90, "hit rate too low: {hit_rate}");
    }

    #[test]
    fn lfu_keeps_frequent_items() {
        let tmp = TempDir::new().unwrap();
        let s = store(&tmp, 8 * 2, Policy::Lfu); // only 2 hot slots
        let a = TensorId::new();
        let b = TensorId::new();
        let c = TensorId::new();
        s.put(a, payload(1)).unwrap();
        s.put(b, payload(2)).unwrap();
        // Access `a` many times to raise its frequency.
        for _ in 0..10 {
            let _ = s.get(a).unwrap();
        }
        // Inserting c should evict b (lower frequency), not a.
        s.put(c, payload(3)).unwrap();
        assert!(s.is_hot(a), "frequently-used a should stay hot");
        assert!(!s.is_hot(b), "infrequent b should be evicted");
    }
}
