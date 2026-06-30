# Data Model

All shared value types live in `tensorus-core` (module `types`) and are
re-exported at the crate root. They are the vocabulary every service speaks.

- [`TensorId`](#tensorid)
- [`DType`](#dtype)
- [`Shape`](#shape)
- [`TensorDescriptor`](#tensordescriptor)
- [`TensorData`](#tensordata)
- [`TensorRecord`](#tensorrecord)
- [`Metadata`](#metadata)

---

## TensorId

A unique tensor identifier wrapping a UUID. **Version 7** UUIDs are used so ids
are time-ordered — newly inserted tensors land roughly sequentially in sorted
index structures.

```rust
pub struct TensorId(pub Uuid);
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new() -> TensorId` | Allocate a fresh time-ordered (UUID v7) id |
| `from_bytes` | `fn from_bytes(bytes: [u8; 16]) -> TensorId` | Build from raw 16 bytes (e.g. read from storage) |
| `as_bytes` | `fn as_bytes(&self) -> &[u8; 16]` | Borrow the underlying bytes |

Also implements `Display` (canonical hyphenated form) and
`FromStr` (`"…".parse::<TensorId>()`), `Default` (= `new()`), plus
`Copy`/`Eq`/`Hash`/`Serialize`/`Deserialize`.

---

## DType

The element data type. The discriminant values are stable and used directly on
the wire (protobuf `uint32 dtype`).

```rust
#[repr(u8)]
pub enum DType {
    Float16=0, Float32=1, Float64=2,
    Int8=3, Int16=4, Int32=5, Int64=6,
    UInt8=7, UInt16=8, UInt32=9, UInt64=10,
    Bool=11, Complex64=12, Complex128=13,
}
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `size_in_bytes` | `fn size_in_bytes(&self) -> usize` | Bytes per element (1–16) |
| `is_float` | `fn is_float(&self) -> bool` | True for `Float16/32/64` |
| `from_u8` | `fn from_u8(v: u8) -> Result<DType>` | Decode the wire discriminant (errors on unknown) |
| `as_u8` | `fn as_u8(&self) -> u8` | The wire discriminant |

> The compute, storage, search, and Python layers currently materialize tensor
> values as **`Float32`**. Other dtypes are representable in the model and on the
> wire; non-`Float32` payloads are stored faithfully but the descriptor/SDK
> helpers operate on `Float32`.

---

## Shape

A tensor's dimensions.

```rust
pub struct Shape(pub Vec<u64>);
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(dims: impl Into<Vec<u64>>) -> Shape` | Construct from any dim list |
| `ndim` | `fn ndim(&self) -> usize` | Number of dimensions (tensor order) |
| `num_elements` | `fn num_elements(&self) -> u64` | Product of dimensions |
| `is_square_matrix` | `fn is_square_matrix(&self) -> bool` | True for 2-D with equal dims |
| `dims` | `fn dims(&self) -> &[u64]` | Borrow the dimension slice |

---

## TensorDescriptor

The mathematical "fingerprint" of a tensor and the basis for property queries
and learned-index keys. This is the core innovation carried from v0.1.

```rust
pub struct TensorDescriptor {
    pub tensor_id: TensorId,
    pub shape: Shape,
    pub dtype: DType,
    pub num_elements: u64,

    // Scalar properties (always present)
    pub frobenius_norm: f64,
    pub l1_norm: f64,
    pub l_inf_norm: f64,
    pub mean: f64,
    pub std_dev: f64,        // sample (N-1) std
    pub sparsity: f64,       // fraction of zeros in [0,1]

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
    pub is_sparse: bool,        // sparsity > 0.5
    pub is_square: bool,
    pub is_diagonal: bool,
}
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `empty` | `fn empty(id: TensorId, shape: Shape, dtype: DType) -> TensorDescriptor` | All numerics zeroed, booleans false |
| `numeric_field` | `fn numeric_field(&self, field: &str) -> Option<f64>` | Look up a numeric property by name (used by learned indexes) |
| `bool_field` | `fn bool_field(&self, field: &str) -> Option<bool>` | Look up a boolean property by name |

**Recognized field names** (`numeric_field`): `frobenius_norm`, `l1_norm`,
`l_inf_norm`, `mean`, `std_dev`, `sparsity`, `num_elements`, `rank`, `trace`,
`determinant`, `condition_number`, `max_eigenvalue`, `min_eigenvalue`.
**(`bool_field`)**: `is_symmetric`, `is_positive_definite`, `is_orthogonal`,
`is_sparse`, `is_square`, `is_diagonal`.

How these are computed (semantics, tolerances) is documented in
[compute](./services/tensorus-compute.md).

---

## TensorData

A raw tensor payload: little-endian bytes plus the shape and dtype needed to
interpret them. Mirrors the protobuf `TensorData` message.

```rust
pub struct TensorData {
    pub data: Vec<u8>,
    pub shape: Shape,
    pub dtype: DType,
}
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(data: Vec<u8>, shape: Shape, dtype: DType) -> Result<TensorData>` | Validate byte length matches `num_elements × size_in_bytes` |
| `from_f32` | `fn from_f32(values: &[f32], shape: Shape) -> Result<TensorData>` | Build a `Float32` payload from a slice |
| `as_f32` | `fn as_f32(&self) -> Result<Vec<f32>>` | Decode as `f32` (errors unless dtype is `Float32`) |
| `num_elements` | `fn num_elements(&self) -> u64` | `shape.num_elements()` |

---

## TensorRecord

A fully materialized stored record returned by the storage layer.

```rust
pub struct TensorRecord {
    pub id: TensorId,
    pub dataset: String,
    pub descriptor: TensorDescriptor,
    pub metadata: Metadata,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub version: u64,
    pub data: Vec<u8>,   // raw little-endian bytes; interpret via descriptor
}
```

`data` is the raw payload; interpret it with `descriptor.dtype` and
`descriptor.shape` (for `Float32`, read 4-byte little-endian chunks).

---

## Metadata

```rust
pub type Metadata = serde_json::Value;
```

Arbitrary user JSON attached to a tensor (e.g. `{"name": "layer1.weight",
"layer": 7}`). It is stored verbatim and returned on reads; the metadata index
([`tensorus-index::metadata`](./services/tensorus-index.md#metadata-index)) can
tokenize and filter on it.

---

## Serialization & the Arrow bridge

All value types derive `serde::{Serialize, Deserialize}`. Descriptors and
metadata are persisted as JSON inside storage frames.

For zero-copy interchange, `tensorus-core::arrow_bridge` converts `Float32`
payloads to/from Apache Arrow:

| Function | Signature |
|----------|-----------|
| `tensor_to_arrow` | `fn tensor_to_arrow(td: &TensorData) -> Result<arrow_array::Float32Array>` |
| `arrow_to_tensor` | `fn arrow_to_tensor(arr: &Float32Array, shape: Shape) -> Result<TensorData>` |
| `pack_fixed_size_list` | `fn pack_fixed_size_list(tensors: &[TensorData], list_size: i32) -> Result<FixedSizeListArray>` |

`FixedSizeListArray` is the columnar layout used to batch equal-length tensors.

---

## Errors

Every fallible operation returns `tensorus_core::Result<T>` =
`std::result::Result<T, TensorusError>`.

```rust
pub enum TensorusError {
    NotFound(String),
    AlreadyExists(String),
    InvalidArgument(String),
    DimensionMismatch(String),
    Storage(String),
    Index(String),
    Compute(String),
    Search(String),
    Provider(String),
    Serialization(String),
    Io(String),
    Internal(String),
}
```

`From` conversions are provided for `std::io::Error`, `serde_json::Error`, and
`uuid::Error`. The API maps these to HTTP status codes (see
[api-reference](./api-reference.md#error-mapping)).
