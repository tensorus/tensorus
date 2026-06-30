//! # tensorus-index
//!
//! Indexing structures for Tensorus:
//! - Learned numeric indexes: [`pgm::PgmIndex`] (static) and ALEX (dynamic).
//! - Vector indexes: HNSW (in-memory) and DiskANN/Vamana (SSD).
//! - Metadata index for full-text and structured filters.

#![forbid(unsafe_code)]

pub mod alex;
pub mod diskann;
pub mod hnsw;
pub mod metadata;
pub mod pgm;

pub use alex::AlexIndex;
pub use diskann::{DiskAnnIndex, VamanaConfig};
pub use hnsw::{Hnsw, HnswConfig, Metric};
pub use metadata::{Document, Filter, MetadataIndex, MetadataQuery};
pub use pgm::PgmIndex;
