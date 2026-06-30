# Service: `tensorus-storage`

**Durable storage engine.** Implements [`tensorus_core::Storage`] with a
self-contained, crash-safe engine, plus adaptive compression and hot/warm/cold
tiering. `#![forbid(unsafe_code)]`.

- **Depends on:** `tensorus-core`, `tokio`, `chrono`, `parking_lot`, `serde_json`,
  `nalgebra` (low-rank codec), `flate2` (DEFLATE), `uuid`.
- **Used by:** `tensorus-api`, `tensorus-python`.

> **Lance substitution.** The plan specifies the Lance columnar format for the
> warm tier. Lance pulls in DataFusion, `object_store`, a large Arrow surface,
> and a build-time `protoc` dependency. To keep the build feasible on the
> pure-Rust gnu toolchain, this crate provides an equivalent engine behind the
> identical [`Storage`] trait, so a Lance backend can be dropped in later without
> touching callers.

---

## 1. `FileStorage` — the engine

```rust
pub struct FileStorage { /* … */ }

impl FileStorage {
    pub fn open(data_dir: impl AsRef<Path>, wal_dir: impl AsRef<Path>) -> Result<Self>;
    pub fn flush(&self) -> Result<()>;
}
// plus the full async `Storage` trait impl (insert/get/scan/delete/create_dataset/list_datasets)
```

| Method | Description |
|--------|-------------|
| `open(data_dir, wal_dir)` | Open or create a store; loads segments and **replays the WAL** (crash recovery). |
| `flush()` | Force all buffered writes to stable storage and advance the checkpoint. Durability is guaranteed for everything inserted before it returns. |

`FileStorage` is `Clone` (an `Arc` handle) and `Send + Sync`. On drop, it
best-effort fsyncs and checkpoints.

### On-disk layout

```
{data_dir}/datasets/{name}/segment.dat   # append-only frame log per dataset
{wal_dir}/wal.log                         # write-ahead log
{wal_dir}/checkpoint                      # highest seq folded into segments
```

### Three-part design

- **Segment files** — the durable, append-only record of committed `Put`/`Del`
  frames per dataset.
- **Write-ahead log (WAL)** — every op is appended here (with a sequence number
  and target dataset) *before* it is applied to a segment.
- **In-memory hot map** — a `HashMap<TensorId, StoredRecord>` per dataset plus an
  insertion-order list; serves all reads (the "hot tier").

---

## 2. Frame format

Each durable change is a *frame*, written as `[u32 payload_len][payload]`. The
length prefix lets recovery detect a torn (partially written) trailing frame.

**`Put` payload:**
```
[u8 op=0][16B id][i64 created_at_us][u64 version]
[u32 desc_len][descriptor JSON][u32 meta_len][metadata JSON][u64 data_len][data bytes]
```
**`Del` payload:** `[u8 op=1][16B id]`.

Descriptor and metadata are stored as JSON; tensor bytes are raw little-endian.

---

## 3. Durability & recovery protocol

**Write path** (`insert`/`delete`), serialized by a single writer lock so the
WAL sequence order matches the checkpoint order:

1. Allocate `seq` (monotonic).
2. Append the frame to the **WAL** (write-through to the OS).
3. Append the frame to the dataset **segment** and update the in-memory map.
4. **Group commit:** every `SYNC_EVERY` (512) writes — and on `flush()`/drop —
   fsync segments + WAL, then write the checkpoint = latest committed `seq`.

This means a value returned from `insert` is durable against a **process crash**
immediately (OS page cache) and against **power loss** after the next
`flush()`/group-commit.

**Recovery** (`open`):

1. Load each dataset's segment, rebuilding the in-memory map; if the segment has
   a torn trailing frame, truncate it to the last complete frame.
2. Read the `checkpoint` (0 if absent).
3. Replay WAL entries with `seq > checkpoint`, **idempotently** (skip a `Put`
   whose id already exists — i.e. a crash after segment write but before
   checkpoint).
4. fsync segments and **truncate** the WAL.

The recovery path is covered by a white-box test that crafts a WAL entry beyond
the checkpoint and verifies it is replayed.

### Performance

Inserting **10,000** shape-`(128,)` `Float32` tensors takes **~252 ms** in
release (~40K inserts/sec) — well under the 2 s task target. (The plan's
>100K/sec headline target needs the columnar/batched Lance path; see
[BENCHMARKS](../../BENCHMARKS.md).)

---

## 4. Adaptive compression (`compression` module)

Content-aware codecs that pick the best encoding per tensor.

```rust
pub enum Codec { Sq8, Coo, Deflate, LowRankSvd }

pub struct Compressed {
    pub codec: Codec,
    pub shape: Vec<u64>,
    pub num_elements: usize,
    pub original_bytes: usize,
    pub max_error: f32,   // 0.0 for lossless codecs
    pub payload: Vec<u8>,
}
impl Compressed { pub fn ratio(&self) -> f64; }   // original / payload size

pub fn classify(data: &[f32], shape: &[u64]) -> Codec;
pub fn compress(data: &[f32], shape: &[u64]) -> Result<Compressed>;       // auto-classify
pub fn compress_with(data: &[f32], shape: &[u64], codec: Codec) -> Result<Compressed>;
pub fn decompress(c: &Compressed) -> Result<Vec<f32>>;
```

| Codec | Lossy? | Best for | Method |
|-------|--------|----------|--------|
| `Sq8` | yes (bounded) | dense / embeddings | 8-bit scalar quantization (global min/scale), ~4× |
| `Coo` | no | sparse (>80% zeros) | coordinate list, delta-varint indices + DEFLATE |
| `Deflate` | no | dense, low-entropy | DEFLATE over the raw bytes |
| `LowRankSvd` | yes (bounded) | low-rank 2-D | truncated SVD capturing ≥99% spectral energy |

`classify` chooses: `sparsity > 0.8 → Coo`; else a small 2-D matrix whose top
quarter of singular values hold ≥99% energy → `LowRankSvd`; otherwise `Sq8`.
Lossy codecs record `max_error` (the exact maximum absolute reconstruction
error). Substitution note: `Deflate` stands in for the plan's Zstd to avoid a C
dependency.

**Measured:** SQ8 ~4× (3.0 GB/s dequant), COO 17.3× at 95% sparsity (lossless),
DEFLATE 210× on structured data, low-rank 31× on a rank-1 64×64 matrix.

---

## 5. Tiered storage (`tiering` module)

A byte-budgeted in-memory **hot** cache over a durable **cold** store, with
access tracking and LRU/LFU eviction. Nothing is ever lost — evicted items
remain in the cold tier.

```rust
pub enum Policy { Lru, Lfu }

pub struct TieringConfig { pub hot_max_bytes: usize, pub policy: Policy }

pub trait ColdStore: Send + Sync {
    fn put(&self, id: TensorId, bytes: &[u8]) -> Result<()>;
    fn get(&self, id: TensorId) -> Result<Option<Vec<u8>>>;
    fn delete(&self, id: TensorId) -> Result<()>;
}

pub struct LocalColdStore { /* filesystem, one file per id */ }
impl LocalColdStore { pub fn new(dir: impl AsRef<Path>) -> Result<Self>; }

pub struct TieredStore<C: ColdStore> { /* … */ }
impl<C: ColdStore> TieredStore<C> {
    pub fn new(cold: C, cfg: TieringConfig) -> Self;
    pub fn put(&self, id: TensorId, bytes: Vec<u8>) -> Result<()>; // write-through
    pub fn get(&self, id: TensorId) -> Result<Vec<u8>>;            // promotes to hot
    pub fn delete(&self, id: TensorId) -> Result<()>;
    pub fn is_hot(&self, id: TensorId) -> bool;
    pub fn hot_bytes(&self) -> usize;
    pub fn hot_len(&self) -> usize;
    pub fn hit_rate(&self) -> f64;
    pub fn reset_stats(&self);
}
```

- **`put`** writes through to cold (durable) then caches hot, evicting to stay
  within `hot_max_bytes`.
- **`get`** serves from hot (a hit) or fetches from cold and promotes (a miss).
- **Eviction** picks the least-recently-used (LRU) or least-frequently-used
  (LFU) victim.

`LocalColdStore` simulates an S3/GCS object store on the local filesystem; the
[`ColdStore`] trait lets an `object_store`-backed implementation drop in.

**Measured:** ~96% hot-tier hit rate on a skewed (Zipfian-style) workload with a
budget covering the hot set.

---

## Usage

```rust
use std::sync::Arc;
use tensorus_core::{Storage, types::{TensorDescriptor, Shape, DType, TensorId}};
use tensorus_storage::FileStorage;

# async fn demo() -> tensorus_core::Result<()> {
let store = FileStorage::open("./data", "./data/wal")?;
store.create_dataset("weights").await?;

let desc = TensorDescriptor::empty(TensorId::new(), Shape::new(vec![2,2]), DType::Float32);
let bytes: Vec<u8> = [1.0f32, 0.0, 0.0, 1.0].iter().flat_map(|x| x.to_le_bytes()).collect();
let id = store.insert("weights", &bytes, desc, serde_json::json!({"name":"I2"})).await?;

let rec = store.get("weights", id).await?;
store.flush()?; // durability barrier
# Ok(()) }
```

[`tensorus_core::Storage`]: ./tensorus-core.md#storage
[`Storage`]: ./tensorus-core.md#storage
[`ColdStore`]: #5-tiered-storage-tiering-module
