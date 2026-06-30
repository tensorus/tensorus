//! Cross-cutting traits that decouple the storage, index, and search layers
//! from their concrete implementations.

use crate::error::Result;
use crate::types::{Metadata, TensorDescriptor, TensorId, TensorRecord};
use async_trait::async_trait;

/// Persistent tensor storage: durable CRUD over datasets of tensors.
#[async_trait]
pub trait Storage: Send + Sync {
    /// Insert a tensor's raw bytes, descriptor, and metadata into a dataset,
    /// returning its freshly allocated identifier.
    async fn insert(
        &self,
        dataset: &str,
        data: &[u8],
        descriptor: TensorDescriptor,
        metadata: Metadata,
    ) -> Result<TensorId>;

    /// Fetch a single record by identifier.
    async fn get(&self, dataset: &str, id: TensorId) -> Result<TensorRecord>;

    /// Page through a dataset's records.
    async fn scan(&self, dataset: &str, limit: usize, offset: usize) -> Result<Vec<TensorRecord>>;

    /// Remove a record. Idempotent: deleting a missing id is not an error.
    async fn delete(&self, dataset: &str, id: TensorId) -> Result<()>;

    /// Create an empty dataset.
    async fn create_dataset(&self, name: &str) -> Result<()>;

    /// List all dataset names.
    async fn list_datasets(&self) -> Result<Vec<String>>;
}

/// A property index mapping descriptor fields to tensor identifiers. Backed by
/// learned indexes (PGM/ALEX) and bitmaps for booleans.
#[async_trait]
pub trait Index: Send + Sync {
    /// Index a tensor's descriptor fields.
    async fn insert(&self, id: TensorId, descriptor: &TensorDescriptor) -> Result<()>;

    /// All identifiers whose `field` falls within `[min, max]`.
    async fn lookup_range(&self, field: &str, min: f64, max: f64) -> Result<Vec<TensorId>>;

    /// All identifiers whose `field` exactly equals `value`.
    async fn lookup_exact(&self, field: &str, value: &serde_json::Value) -> Result<Vec<TensorId>>;

    /// Remove a tensor from the index.
    async fn delete(&self, id: TensorId) -> Result<()>;
}

/// An approximate-nearest-neighbour vector index (HNSW/DiskANN).
#[async_trait]
pub trait VectorIndex: Send + Sync {
    /// Insert a vector under a tensor identifier.
    async fn insert(&self, id: TensorId, vector: &[f32]) -> Result<()>;

    /// Return the `k` nearest identifiers to `query` with their distances.
    async fn search(&self, query: &[f32], k: usize) -> Result<Vec<(TensorId, f32)>>;

    /// Remove a vector.
    async fn delete(&self, id: TensorId) -> Result<()>;
}

/// A higher-level search engine combining property, vector, and structural
/// similarity into ranked results.
#[async_trait]
pub trait SearchEngine: Send + Sync {
    /// Rank `k` tensors by similarity to the given query payload under a named
    /// metric (e.g. "cosine", "l2", "contraction").
    async fn search_similar(
        &self,
        dataset: &str,
        query: &[f32],
        metric: &str,
        k: usize,
    ) -> Result<Vec<(TensorId, f32)>>;
}
