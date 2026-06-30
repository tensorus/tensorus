//! Conversions between Tensorus tensor payloads and Apache Arrow arrays.
//!
//! Arrow is the zero-copy interchange format used between components and at the
//! Python boundary. For `Float32` tensors we expose the data as an
//! [`arrow_array::Float32Array`], and batches of equal-length tensors as a
//! [`arrow_array::FixedSizeListArray`] (the layout the plan specifies for the
//! Lance schema).

use crate::error::{Result, TensorusError};
use crate::types::{DType, Shape, TensorData};
use arrow_array::{Array, FixedSizeListArray, Float32Array};
use arrow_buffer::Buffer;
use arrow_schema::{DataType, Field};
use std::sync::Arc;

/// Convert a `Float32` [`TensorData`] into an Arrow [`Float32Array`] without
/// copying element-by-element (the byte buffer is reinterpreted).
pub fn tensor_to_arrow(td: &TensorData) -> Result<Float32Array> {
    if td.dtype != DType::Float32 {
        return Err(TensorusError::InvalidArgument(format!(
            "arrow bridge currently supports Float32 only, got {:?}",
            td.dtype
        )));
    }
    let buffer = Buffer::from_vec(td.data.clone());
    let arr = Float32Array::new(buffer.into(), None);
    Ok(arr)
}

/// Convert an Arrow [`Float32Array`] back into a flat `Float32` [`TensorData`]
/// with the supplied shape.
pub fn arrow_to_tensor(arr: &Float32Array, shape: Shape) -> Result<TensorData> {
    if arr.len() as u64 != shape.num_elements() {
        return Err(TensorusError::DimensionMismatch(format!(
            "arrow array length {} does not match shape {:?}",
            arr.len(),
            shape.dims()
        )));
    }
    TensorData::from_f32(arr.values(), shape)
}

/// Pack a batch of equal-length `Float32` tensors into a single
/// [`FixedSizeListArray`] of `list_size` elements each. This is the columnar
/// layout used for the Lance `tensor_data` column.
pub fn pack_fixed_size_list(tensors: &[TensorData], list_size: i32) -> Result<FixedSizeListArray> {
    let mut values: Vec<f32> = Vec::new();
    for td in tensors {
        let v = td.as_f32()?;
        if v.len() as i32 != list_size {
            return Err(TensorusError::DimensionMismatch(format!(
                "tensor of length {} does not match fixed list size {}",
                v.len(),
                list_size
            )));
        }
        values.extend_from_slice(&v);
    }
    let value_array = Arc::new(Float32Array::from(values)) as Arc<dyn Array>;
    let field = Arc::new(Field::new("item", DataType::Float32, false));
    FixedSizeListArray::try_new(field, list_size, value_array, None)
        .map_err(|e| TensorusError::Internal(format!("failed to build FixedSizeListArray: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float32_roundtrip_through_arrow() {
        let values = vec![1.5f32, -2.0, 3.25, 0.0];
        let td = TensorData::from_f32(&values, Shape::new(vec![2, 2])).unwrap();
        let arr = tensor_to_arrow(&td).unwrap();
        assert_eq!(arr.len(), 4);
        let back = arrow_to_tensor(&arr, Shape::new(vec![2, 2])).unwrap();
        assert_eq!(back.as_f32().unwrap(), values);
    }

    #[test]
    fn pack_batch_into_fixed_size_list() {
        let a = TensorData::from_f32(&[1.0, 2.0, 3.0], Shape::new(vec![3])).unwrap();
        let b = TensorData::from_f32(&[4.0, 5.0, 6.0], Shape::new(vec![3])).unwrap();
        let list = pack_fixed_size_list(&[a, b], 3).unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list.value_length(), 3);
    }

    #[test]
    fn mismatched_list_size_errors() {
        let a = TensorData::from_f32(&[1.0, 2.0], Shape::new(vec![2])).unwrap();
        assert!(pack_fixed_size_list(&[a], 3).is_err());
    }
}
