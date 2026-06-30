//! Fundamental value types: identifiers, dtypes, shapes, descriptors, records.

use crate::error::{Result, TensorusError};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use uuid::Uuid;

/// Unique tensor identifier. Uses UUID v7 so identifiers are time-ordered,
/// which keeps inserts roughly sequential in sorted index structures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorId(pub Uuid);

impl TensorId {
    /// Allocate a fresh, time-ordered identifier.
    pub fn new() -> Self {
        TensorId(Uuid::now_v7())
    }

    /// Construct from raw bytes (e.g. when reading from storage).
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        TensorId(Uuid::from_bytes(bytes))
    }

    /// Borrow the underlying 16 bytes.
    pub fn as_bytes(&self) -> &[u8; 16] {
        self.0.as_bytes()
    }
}

impl Default for TensorId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TensorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for TensorId {
    type Err = TensorusError;
    fn from_str(s: &str) -> Result<Self> {
        Ok(TensorId(Uuid::parse_str(s)?))
    }
}

/// Element data type of a tensor. The discriminant values are stable and used
/// directly on the wire (protobuf `uint32 dtype`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum DType {
    Float16 = 0,
    Float32 = 1,
    Float64 = 2,
    Int8 = 3,
    Int16 = 4,
    Int32 = 5,
    Int64 = 6,
    UInt8 = 7,
    UInt16 = 8,
    UInt32 = 9,
    UInt64 = 10,
    Bool = 11,
    Complex64 = 12,
    Complex128 = 13,
}

impl DType {
    /// Size of a single element in bytes.
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::Bool | DType::Int8 | DType::UInt8 => 1,
            DType::Float16 | DType::Int16 | DType::UInt16 => 2,
            DType::Float32 | DType::Int32 | DType::UInt32 => 4,
            DType::Float64 | DType::Int64 | DType::UInt64 | DType::Complex64 => 8,
            DType::Complex128 => 16,
        }
    }

    /// Whether this dtype is a floating-point real type.
    pub fn is_float(&self) -> bool {
        matches!(self, DType::Float16 | DType::Float32 | DType::Float64)
    }

    /// Convert the stable wire discriminant back into a `DType`.
    pub fn from_u8(v: u8) -> Result<Self> {
        Ok(match v {
            0 => DType::Float16,
            1 => DType::Float32,
            2 => DType::Float64,
            3 => DType::Int8,
            4 => DType::Int16,
            5 => DType::Int32,
            6 => DType::Int64,
            7 => DType::UInt8,
            8 => DType::UInt16,
            9 => DType::UInt32,
            10 => DType::UInt64,
            11 => DType::Bool,
            12 => DType::Complex64,
            13 => DType::Complex128,
            other => {
                return Err(TensorusError::InvalidArgument(format!(
                    "unknown dtype discriminant {other}"
                )))
            }
        })
    }

    /// The stable wire discriminant.
    pub fn as_u8(&self) -> u8 {
        *self as u8
    }
}

/// The shape of a tensor as a list of dimension sizes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape(pub Vec<u64>);

impl Shape {
    /// Build a shape from any slice of dimension sizes.
    pub fn new(dims: impl Into<Vec<u64>>) -> Self {
        Shape(dims.into())
    }

    /// Number of dimensions (the tensor "order").
    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// Total number of scalar elements (product of dimensions).
    pub fn num_elements(&self) -> u64 {
        self.0.iter().product()
    }

    /// True for 2-D tensors whose two dimensions are equal.
    pub fn is_square_matrix(&self) -> bool {
        self.0.len() == 2 && self.0[0] == self.0[1]
    }

    /// Borrow the dimensions slice.
    pub fn dims(&self) -> &[u64] {
        &self.0
    }
}

/// The mathematical "fingerprint" of a tensor: the set of scalar, matrix, and
/// boolean properties that make tensors queryable by structure. This is the
/// core innovation carried forward from v0.1 and the basis for learned-index
/// keys.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorDescriptor {
    pub tensor_id: TensorId,
    pub shape: Shape,
    pub dtype: DType,
    pub num_elements: u64,

    // Scalar properties
    pub frobenius_norm: f64,
    pub l1_norm: f64,
    pub l_inf_norm: f64,
    pub mean: f64,
    pub std_dev: f64,
    /// Fraction of elements equal to zero, in `[0, 1]`.
    pub sparsity: f64,

    // Matrix properties (None for non-2-D tensors)
    pub rank: Option<u32>,
    pub trace: Option<f64>,
    pub determinant: Option<f64>,
    pub condition_number: Option<f64>,
    pub max_eigenvalue: Option<f64>,
    pub min_eigenvalue: Option<f64>,

    // Boolean properties
    pub is_symmetric: bool,
    pub is_positive_definite: bool,
    pub is_orthogonal: bool,
    pub is_sparse: bool,
    pub is_square: bool,
    pub is_diagonal: bool,
}

impl TensorDescriptor {
    /// Create a descriptor with all numeric fields zeroed and booleans false.
    /// Used as a starting point by the compute engine and in tests.
    pub fn empty(tensor_id: TensorId, shape: Shape, dtype: DType) -> Self {
        let num_elements = shape.num_elements();
        TensorDescriptor {
            tensor_id,
            shape,
            dtype,
            num_elements,
            frobenius_norm: 0.0,
            l1_norm: 0.0,
            l_inf_norm: 0.0,
            mean: 0.0,
            std_dev: 0.0,
            sparsity: 0.0,
            rank: None,
            trace: None,
            determinant: None,
            condition_number: None,
            max_eigenvalue: None,
            min_eigenvalue: None,
            is_symmetric: false,
            is_positive_definite: false,
            is_orthogonal: false,
            is_sparse: false,
            is_square: false,
            is_diagonal: false,
        }
    }

    /// Look up a numeric property by name, used by the learned-index layer to
    /// build per-field indexes. Returns `None` for unknown or absent fields.
    pub fn numeric_field(&self, field: &str) -> Option<f64> {
        match field {
            "frobenius_norm" => Some(self.frobenius_norm),
            "l1_norm" => Some(self.l1_norm),
            "l_inf_norm" => Some(self.l_inf_norm),
            "mean" => Some(self.mean),
            "std_dev" => Some(self.std_dev),
            "sparsity" => Some(self.sparsity),
            "num_elements" => Some(self.num_elements as f64),
            "rank" => self.rank.map(|r| r as f64),
            "trace" => self.trace,
            "determinant" => self.determinant,
            "condition_number" => self.condition_number,
            "max_eigenvalue" => self.max_eigenvalue,
            "min_eigenvalue" => self.min_eigenvalue,
            _ => None,
        }
    }

    /// Look up a boolean property by name.
    pub fn bool_field(&self, field: &str) -> Option<bool> {
        match field {
            "is_symmetric" => Some(self.is_symmetric),
            "is_positive_definite" => Some(self.is_positive_definite),
            "is_orthogonal" => Some(self.is_orthogonal),
            "is_sparse" => Some(self.is_sparse),
            "is_square" => Some(self.is_square),
            "is_diagonal" => Some(self.is_diagonal),
            _ => None,
        }
    }
}

/// Arbitrary user metadata attached to a tensor (free-form JSON).
pub type Metadata = serde_json::Value;

/// Raw tensor payload: little-endian bytes plus the shape and dtype needed to
/// interpret them. Mirrors the protobuf `TensorData` message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorData {
    pub data: Vec<u8>,
    pub shape: Shape,
    pub dtype: DType,
}

impl TensorData {
    /// Build from raw bytes, validating that the byte length matches the
    /// product of dimensions times the element size.
    pub fn new(data: Vec<u8>, shape: Shape, dtype: DType) -> Result<Self> {
        let expected = shape.num_elements() as usize * dtype.size_in_bytes();
        if data.len() != expected {
            return Err(TensorusError::DimensionMismatch(format!(
                "byte length {} does not match shape {:?} dtype {:?} (expected {})",
                data.len(),
                shape.dims(),
                dtype,
                expected
            )));
        }
        Ok(TensorData { data, shape, dtype })
    }

    /// Construct directly from an `f32` slice.
    pub fn from_f32(values: &[f32], shape: Shape) -> Result<Self> {
        if values.len() as u64 != shape.num_elements() {
            return Err(TensorusError::DimensionMismatch(format!(
                "value count {} does not match shape {:?}",
                values.len(),
                shape.dims()
            )));
        }
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Ok(TensorData {
            data: bytes,
            shape,
            dtype: DType::Float32,
        })
    }

    /// Decode the payload as `f32` values. Only valid when `dtype` is
    /// `Float32`.
    pub fn as_f32(&self) -> Result<Vec<f32>> {
        if self.dtype != DType::Float32 {
            return Err(TensorusError::InvalidArgument(format!(
                "cannot read dtype {:?} as f32",
                self.dtype
            )));
        }
        Ok(self
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Number of scalar elements.
    pub fn num_elements(&self) -> u64 {
        self.shape.num_elements()
    }
}

/// A fully materialized tensor record as stored and returned by the storage
/// layer. Carries the raw payload alongside its descriptor and metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorRecord {
    pub id: TensorId,
    pub dataset: String,
    pub descriptor: TensorDescriptor,
    pub metadata: Metadata,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub version: u64,
    /// Raw little-endian tensor bytes. Interpret using `descriptor.dtype` and
    /// `descriptor.shape`.
    pub data: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_id_roundtrip() {
        let id = TensorId::new();
        let s = id.to_string();
        let parsed: TensorId = s.parse().unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn dtype_discriminant_roundtrip() {
        for v in 0u8..=13 {
            let dt = DType::from_u8(v).unwrap();
            assert_eq!(dt.as_u8(), v);
        }
        assert!(DType::from_u8(99).is_err());
    }

    #[test]
    fn shape_helpers() {
        let s = Shape::new(vec![3, 3]);
        assert_eq!(s.ndim(), 2);
        assert_eq!(s.num_elements(), 9);
        assert!(s.is_square_matrix());
        assert!(!Shape::new(vec![2, 3]).is_square_matrix());
    }

    #[test]
    fn tensor_data_f32_roundtrip() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let td = TensorData::from_f32(&values, Shape::new(vec![2, 2])).unwrap();
        assert_eq!(td.as_f32().unwrap(), values);
        assert_eq!(td.num_elements(), 4);
    }

    #[test]
    fn tensor_data_length_validation() {
        let bad = TensorData::new(vec![0u8; 3], Shape::new(vec![2, 2]), DType::Float32);
        assert!(bad.is_err());
    }

    #[test]
    fn descriptor_field_lookup() {
        let mut d =
            TensorDescriptor::empty(TensorId::new(), Shape::new(vec![2, 2]), DType::Float32);
        d.frobenius_norm = 5.0;
        d.is_symmetric = true;
        assert_eq!(d.numeric_field("frobenius_norm"), Some(5.0));
        assert_eq!(d.bool_field("is_symmetric"), Some(true));
        assert_eq!(d.numeric_field("nonexistent"), None);
    }

    #[test]
    fn descriptor_serde_roundtrip() {
        let d = TensorDescriptor::empty(TensorId::new(), Shape::new(vec![4]), DType::Float64);
        let json = serde_json::to_string(&d).unwrap();
        let back: TensorDescriptor = serde_json::from_str(&json).unwrap();
        assert_eq!(d, back);
    }
}
