//! # tensorus-compute
//!
//! CPU implementations of tensor descriptor computation and the linear-algebra
//! primitives the rest of the system relies on. GPU acceleration points are
//! marked with `// TODO: GPU acceleration`.

#![forbid(unsafe_code)]

mod descriptor;

pub use descriptor::compute_descriptor;

use tensorus_core::error::Result;
use tensorus_core::types::{TensorData, TensorDescriptor};

/// Convenience wrapper computing a descriptor directly from a [`TensorData`]
/// payload (currently `Float32` only).
pub fn descriptor_from_tensor_data(td: &TensorData) -> Result<TensorDescriptor> {
    let values = td.as_f32()?;
    Ok(compute_descriptor(&values, td.shape.dims(), td.dtype))
}
