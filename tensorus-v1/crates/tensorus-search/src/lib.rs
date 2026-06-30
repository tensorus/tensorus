//! # tensorus-search
//!
//! Hybrid search combining semantic and computational relevance, plus the
//! novel tensor contraction similarity engine (Tucker sketch + Grassmannian
//! distance).

#![forbid(unsafe_code)]

pub mod contraction;

pub use contraction::{
    contraction_similarity, grassmannian_distance, sketch_similarity, tucker_sketch,
    ContractionIndex, ContractionSketch, DenseTensor,
};
