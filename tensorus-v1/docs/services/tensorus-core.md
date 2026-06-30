# Service: `tensorus-core`

**Foundation crate.** Defines the shared vocabulary — value types, errors, and
the traits that decouple every other layer. It has no dependency on any other
Tensorus crate and is `#![forbid(unsafe_code)]`.

- **Depends on:** `uuid`, `serde`, `serde_json`, `chrono`, `thiserror`,
  `async-trait`, `arrow-array`/`arrow-buffer`/`arrow-schema`.
- **Depended on by:** every other crate.

## Modules

| Module | Contents |
|--------|----------|
| `types` | [`TensorId`], [`DType`], [`Shape`], [`TensorDescriptor`], [`TensorData`], [`TensorRecord`], `Metadata` |
| `error` | `TensorusError`, `Result<T>` |
| `traits` | `Storage`, `Index`, `VectorIndex`, `SearchEngine` |
| `arrow_bridge` | `tensor_to_arrow`, `arrow_to_tensor`, `pack_fixed_size_list` |

The value types and errors are fully described in [Data Model](../data-model.md).
This page focuses on the **traits**, which are the seams of the system.

---

## Traits

All trait methods are `async` (via `async-trait`) and return
`tensorus_core::Result`. Implementors must be `Send + Sync`.

### Storage

Durable CRUD over datasets of tensors. Implemented by
[`tensorus-storage::FileStorage`](./tensorus-storage.md).

```rust
#[async_trait]
pub trait Storage: Send + Sync {
    async fn insert(&self, dataset: &str, data: &[u8],
                    descriptor: TensorDescriptor, metadata: Metadata) -> Result<TensorId>;
    async fn get(&self, dataset: &str, id: TensorId) -> Result<TensorRecord>;
    async fn scan(&self, dataset: &str, limit: usize, offset: usize) -> Result<Vec<TensorRecord>>;
    async fn delete(&self, dataset: &str, id: TensorId) -> Result<()>;
    async fn create_dataset(&self, name: &str) -> Result<()>;
    async fn list_datasets(&self) -> Result<Vec<String>>;
}
```

| Method | Semantics |
|--------|-----------|
| `insert` | Allocate a fresh `TensorId`, persist the record, return the id. The implementation stamps `descriptor.tensor_id`. |
| `get` | Returns `NotFound` if the dataset or id is absent. |
| `scan` | Pages records in insertion order via `limit`/`offset`. |
| `delete` | Idempotent — deleting a missing id (or dataset) is `Ok(())`. |
| `create_dataset` | Idempotent — creating an existing dataset is `Ok(())`. |
| `list_datasets` | Sorted dataset names. |

### Index

A property index mapping descriptor fields to ids (backed by learned indexes /
bitmaps). Defines the contract that the learned indexes in
[`tensorus-index`](./tensorus-index.md) satisfy.

```rust
#[async_trait]
pub trait Index: Send + Sync {
    async fn insert(&self, id: TensorId, descriptor: &TensorDescriptor) -> Result<()>;
    async fn lookup_range(&self, field: &str, min: f64, max: f64) -> Result<Vec<TensorId>>;
    async fn lookup_exact(&self, field: &str, value: &serde_json::Value) -> Result<Vec<TensorId>>;
    async fn delete(&self, id: TensorId) -> Result<()>;
}
```

### VectorIndex

Approximate-nearest-neighbour search. Implemented by
[`tensorus-index::Hnsw`](./tensorus-index.md#hnsw).

```rust
#[async_trait]
pub trait VectorIndex: Send + Sync {
    async fn insert(&self, id: TensorId, vector: &[f32]) -> Result<()>;
    async fn search(&self, query: &[f32], k: usize) -> Result<Vec<(TensorId, f32)>>;
    async fn delete(&self, id: TensorId) -> Result<()>;
}
```

`search` returns `(id, distance)` pairs, nearest first (smaller distance = closer).

### SearchEngine

A higher-level engine that combines property, vector, and structural similarity.

```rust
#[async_trait]
pub trait SearchEngine: Send + Sync {
    async fn search_similar(&self, dataset: &str, query: &[f32],
                            metric: &str, k: usize) -> Result<Vec<(TensorId, f32)>>;
}
```

`metric` is a string such as `"cosine"`, `"l2"`, or `"contraction"`.

---

## Arrow bridge

Zero-copy interchange of `Float32` payloads with Apache Arrow. See
[Data Model → Arrow bridge](../data-model.md#serialization--the-arrow-bridge).

## Constants

```rust
pub const VERSION: &str; // = CARGO_PKG_VERSION ("1.0.0")
```

Surfaced via the API `/health` endpoint and the Python `__version__`.

[`TensorId`]: ../data-model.md#tensorid
[`DType`]: ../data-model.md#dtype
[`Shape`]: ../data-model.md#shape
[`TensorDescriptor`]: ../data-model.md#tensordescriptor
[`TensorData`]: ../data-model.md#tensordata
[`TensorRecord`]: ../data-model.md#tensorrecord
