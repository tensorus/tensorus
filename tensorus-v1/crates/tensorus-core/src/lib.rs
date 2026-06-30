//! # tensorus-core
//!
//! Foundational types, traits, and errors shared by every Tensorus crate.
//!
//! - [`types`] defines the value types: [`types::TensorId`], [`types::DType`],
//!   [`types::Shape`], [`types::TensorDescriptor`], [`types::TensorData`], and
//!   [`types::TensorRecord`].
//! - [`traits`] defines the storage/index/search abstractions that decouple
//!   layers.
//! - [`error`] defines the crate-wide [`error::TensorusError`] and
//!   [`error::Result`].
//! - [`arrow_bridge`] converts between tensor payloads and Apache Arrow arrays.

#![forbid(unsafe_code)]

pub mod arrow_bridge;
pub mod error;
pub mod traits;
pub mod types;

pub use error::{Result, TensorusError};
pub use traits::{Index, SearchEngine, Storage, VectorIndex};
pub use types::{DType, Metadata, Shape, TensorData, TensorDescriptor, TensorId, TensorRecord};

/// The crate (and product) version, surfaced via the API health endpoint.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
