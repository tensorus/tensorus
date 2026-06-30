# Service: `tensorus-index`

**Indexing structures.** Five index types covering numeric property queries,
vector similarity, and metadata search. `#![forbid(unsafe_code)]`.

- **Depends on:** `tensorus-core`, `parking_lot`, `rand`, `serde`, `serde_json`,
  `async-trait`.
- **Used by:** `tensorus-search` (HNSW), and intended as the backing store for
  the API's property/vector search.

| Module | Type | Purpose | Reference |
|--------|------|---------|-----------|
| `pgm` | [`PgmIndex`](#pgm-learned-index) | static learned index for numeric range/point queries | |
| `alex` | [`AlexIndex`](#alex-dynamic-learned-index) | dynamic (updatable) learned index | |
| `hnsw` | [`Hnsw`](#hnsw) | in-memory ANN vector index | implements `VectorIndex` |
| `diskann` | [`DiskAnnIndex`](#diskann--vamana) | SSD-resident ANN (Vamana graph) | |
| `metadata` | [`MetadataIndex`](#metadata-index) | full-text + structured metadata search | |

---

## PGM learned index

A Piecewise Geometric Model index: approximates the `key → position` mapping in
a sorted array with linear segments, each bounded by an error `epsilon`. A lookup
evaluates a segment's line (O(1)) then does a bounded local binary search.

```rust
pub const DEFAULT_EPSILON: usize = 64;

pub struct PgmIndex { /* … */ }
impl PgmIndex {
    pub fn build(entries: Vec<(f64, u64)>) -> Self;                        // ε = 64
    pub fn build_with_epsilon(entries: Vec<(f64, u64)>, epsilon: usize) -> Self;
    pub fn lookup(&self, q: f64) -> Vec<u64>;            // all payloads with key == q
    pub fn range(&self, min: f64, max: f64) -> Vec<u64>; // payloads with key in [min,max]
    pub fn range_count(&self, min: f64, max: f64) -> usize;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn segment_count(&self) -> usize; // model compactness
}
```

- Entries are `(key, payload)`; `payload` is an opaque `u64` (e.g. a row id).
- **Correctness guarantee:** every windowed search is validated against the true
  sorted order and falls back to a full binary search on the rare miss (e.g. a
  run of duplicate keys longer than `epsilon`), so results are always exact.
- **Use for:** `frobenius_norm`, `rank`, `condition_number`, `sparsity`,
  timestamps — any sorted numeric property.

**Measured (10M uniform keys):** 957 segments; point/narrow-range query ~308 ns;
`range_count` ~299 ns — within the `<500 ns` / `<10 µs` targets.

---

## ALEX dynamic learned index

An updatable learned index supporting **concurrent** inserts and lookups. An
internal routing node (sorted pivots) routes a key to a leaf; leaves are
model-accelerated sorted arrays that **split** when they overflow. Per-leaf locks
let operations on different leaves proceed in parallel.

```rust
pub struct AlexIndex { /* … */ }
impl AlexIndex {
    pub fn new() -> Self;                                  // empty
    pub fn bulk_load(entries: Vec<(f64, u64)>) -> Self;    // balanced leaves
    pub fn insert(&self, key: f64, payload: u64);          // &self (thread-safe)
    pub fn lookup(&self, key: f64) -> Vec<u64>;
    pub fn range(&self, min: f64, max: f64) -> Vec<u64>;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn leaf_count(&self) -> usize;
}
```

- Splits occur at a **key boundary**, so equal keys are never spread across
  leaves (lookups for a key touch exactly one leaf).
- Lookups and non-splitting inserts take a shared read lock on the routing node
  plus a per-leaf mutex; the rare split takes the routing write lock briefly.

**Measured:** 1M random inserts in ~394 ms (**~2.54M inserts/sec**), exceeding
the >500K/sec target; correct under concurrent and mixed workloads.

---

## HNSW

A custom in-memory Hierarchical Navigable Small World graph (Malkov & Yashunin),
including the neighbor-selection heuristic. Implements
[`VectorIndex`](./tensorus-core.md#vectorindex).

```rust
pub enum Metric { L2, Cosine, Dot }

pub struct HnswConfig {
    pub m: usize,                 // default 16
    pub ef_construction: usize,   // default 200
    pub ef_search: usize,         // default 100
    pub metric: Metric,           // default Cosine
    pub seed: u64,
}

pub struct Hnsw { /* … */ }
impl Hnsw {
    pub fn new(cfg: HnswConfig) -> Self;
    pub fn with_metric(metric: Metric) -> Self;   // defaults otherwise
    pub fn insert(&self, id: TensorId, vector: &[f32]);
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(TensorId, f32)>; // (id, distance)
    pub fn delete(&self, id: TensorId);            // tombstone
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}
```

- **Metrics:** `L2` (squared Euclidean), `Cosine` (vectors normalized on
  insert/search; distance `1 − cos`), `Dot` (negative inner product).
- Thread-safe via an internal `RwLock`; `delete` is a tombstone (excluded from
  results, edges kept for navigation).
- Also usable through the async `VectorIndex` trait.

**Measured (10k × 128-dim, release):** search ~0.3–0.6 ms/query (meets `<1 ms`);
recall@10 ≈ 0.95 on low-dimensional/clustered data (≈ 0.83 on adversarial
uniform-random 128-dim). Insert throughput (~1.6K/sec at `ef_construction=200`)
is below the 50K/sec target — the distance kernel is scalar; SIMD/flat-layout
work (marked `// TODO`) would close the gap. See [BENCHMARKS](../../BENCHMARKS.md).

---

## DiskANN / Vamana

An SSD-resident index: a Vamana graph (greedy search + `RobustPrune` with the
`alpha` diversification factor) flushed to a 4 KB-page-aligned file. Beam search
reads a node's neighbor list from disk and ranks the frontier with in-memory SQ8
codes, then reranks the top candidates with full vectors from disk.

```rust
pub struct VamanaConfig {
    pub degree: usize,       // R, default 64
    pub build_list: usize,   // L during build, default 100
    pub alpha: f32,          // diversification, default 1.2
    pub search_list: usize,  // default beam width, default 100
    pub seed: u64,
}

pub struct DiskAnnIndex { /* … */ }
impl DiskAnnIndex {
    pub fn build(vectors: Vec<Vec<f32>>, ids: Vec<TensorId>,
                 cfg: VamanaConfig, path: impl AsRef<Path>) -> Result<DiskAnnIndex>;
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(TensorId, f32)>>;
    pub fn search_with_beam(&self, query: &[f32], k: usize, beam: usize) -> Result<Vec<(TensorId, f32)>>;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn bytes_per_vector_overhead(&self) -> usize; // SQ8 codes + graph edges
}
```

- **Static (build-once):** updates are handled by rebuilds, as in DiskANN.
- **Substitutions:** SQ8 in place of Product Quantization; synchronous `std::fs`
  reads in place of `io_uring` (Linux-only). The page-aligned layout is preserved
  so an async reader can drop in.

**Measured:** recall@10 = 1.0 on 2,000 synthetic vectors (target ≥0.90);
page-aligned file layout verified.

---

## Metadata index

A self-contained inverted index for full-text search over string fields plus
structured equality/range/text filters and faceting. Thread-safe.

```rust
pub struct Document { pub id: TensorId, pub dataset: String, pub fields: serde_json::Map<String, Value> }
impl Document { pub fn new(id: TensorId, dataset: impl Into<String>, fields: Map<String,Value>) -> Self; }

pub enum Filter {
    Eq(String, serde_json::Value),     // field == value
    Range(String, f64, f64),           // numeric field in [min,max]
    TextMatch(String, String),         // a field's tokens contain token
}

pub struct MetadataQuery { pub text: Option<String>, pub filters: Vec<Filter>, pub limit: usize }
impl MetadataQuery {
    pub fn text(q: impl Into<String>) -> Self;        // full-text only
    pub fn filtered(filters: Vec<Filter>) -> Self;    // structured only
    pub fn with_limit(self, limit: usize) -> Self;
}

pub struct MetadataIndex { /* … */ }
impl MetadataIndex {
    pub fn new() -> Self;
    pub fn insert(&self, doc: Document);              // insert or replace
    pub fn delete(&self, id: TensorId);
    pub fn search(&self, query: &MetadataQuery) -> Vec<(TensorId, f32)>; // ranked (id, score)
    pub fn facet_counts(&self, field: &str) -> HashMap<String, usize>;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}
```

- `Document::new` injects `dataset` as a queryable field automatically.
- Full-text matching tokenizes on non-alphanumeric boundaries, lowercased; the
  score is the number of matched query tokens plus a small term-frequency bonus
  (`0.0` for pure structured queries). `filters` combine with AND.
- `limit > 0` caps results (top-k via partial selection).
- **Substitution:** an in-house inverted index in place of Tantivy.

**Measured:** ~604 µs per query over 100k documents (target `<5 ms`).

---

## Choosing an index

| You have… | Use |
|-----------|-----|
| a sorted numeric property, read-heavy | `PgmIndex` |
| a numeric property with frequent inserts | `AlexIndex` |
| dense embedding vectors, in RAM | `Hnsw` |
| billions of vectors, SSD-resident | `DiskAnnIndex` |
| names/descriptions/structured metadata | `MetadataIndex` |
| structural tensor similarity | see [`tensorus-search`](./tensorus-search.md) |
