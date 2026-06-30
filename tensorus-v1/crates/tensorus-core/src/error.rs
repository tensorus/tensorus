//! Error types for Tensorus.

use thiserror::Error;

/// The canonical result type used across all Tensorus crates.
pub type Result<T> = std::result::Result<T, TensorusError>;

/// Top-level error enum spanning every subsystem.
#[derive(Debug, Error)]
pub enum TensorusError {
    /// A requested entity (tensor, dataset, index) does not exist.
    #[error("not found: {0}")]
    NotFound(String),

    /// An entity already exists and cannot be created again.
    #[error("already exists: {0}")]
    AlreadyExists(String),

    /// The caller supplied an argument that violates a precondition.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// A shape/dimension mismatch between two tensors or an operation's
    /// expectation.
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// A storage-layer failure (Lance/WAL/filesystem).
    #[error("storage error: {0}")]
    Storage(String),

    /// An index-layer failure (PGM/ALEX/HNSW/metadata).
    #[error("index error: {0}")]
    Index(String),

    /// A compute-layer failure (descriptor/decomposition).
    #[error("compute error: {0}")]
    Compute(String),

    /// A search-layer failure.
    #[error("search error: {0}")]
    Search(String),

    /// An AI/LLM provider failure.
    #[error("provider error: {0}")]
    Provider(String),

    /// Serialization or deserialization failed.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// An I/O failure surfaced from the standard library.
    #[error("io error: {0}")]
    Io(String),

    /// A catch-all for unexpected internal failures.
    #[error("internal error: {0}")]
    Internal(String),
}

impl From<std::io::Error> for TensorusError {
    fn from(e: std::io::Error) -> Self {
        TensorusError::Io(e.to_string())
    }
}

impl From<serde_json::Error> for TensorusError {
    fn from(e: serde_json::Error) -> Self {
        TensorusError::Serialization(e.to_string())
    }
}

impl From<uuid::Error> for TensorusError {
    fn from(e: uuid::Error) -> Self {
        TensorusError::InvalidArgument(format!("invalid uuid: {e}"))
    }
}
