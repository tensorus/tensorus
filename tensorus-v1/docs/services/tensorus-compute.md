# Service: `tensorus-compute`

**Descriptor computation.** Given a tensor's raw `f32` values, shape, and dtype,
it computes the full [`TensorDescriptor`](../data-model.md#tensordescriptor):
norms, statistics, and (for 2-D tensors) matrix properties via SVD /
eigendecomposition. All computation is done in `f64` regardless of the stored
dtype.

- **Depends on:** `tensorus-core`, `ndarray`, `nalgebra`.
- **Used by:** `tensorus-api` (on insert), `tensorus-python`, `tensorus-search`.
- GPU acceleration points are marked `// TODO: GPU acceleration`.

## Public API

```rust
pub fn compute_descriptor(data: &[f32], shape: &[u64], dtype: DType) -> TensorDescriptor;
pub fn descriptor_from_tensor_data(td: &TensorData) -> Result<TensorDescriptor>;
```

- `compute_descriptor` is the core entry point.
- `descriptor_from_tensor_data` decodes a `Float32` [`TensorData`] and delegates.

The returned descriptor has a freshly generated `tensor_id` (callers that store
the tensor overwrite it with the storage-assigned id).

## Computed properties & semantics

The implementation is a faithful port of the v0.1 `TensorDescriptor.from_tensor`
and targets agreement within `1e-6`.

### Always computed (any rank)

| Property | Definition |
|----------|------------|
| `frobenius_norm` | `sqrt(Σ xᵢ²)` |
| `l1_norm` | `Σ |xᵢ|` |
| `l_inf_norm` | `max |xᵢ|` |
| `mean` | `Σ xᵢ / N` |
| `std_dev` | sample standard deviation (denominator `N-1`; `0` if `N ≤ 1`) |
| `sparsity` | fraction of elements exactly equal to `0`, in `[0,1]` |
| `is_sparse` | `sparsity > 0.5` |
| `num_elements` | `N` |

### 2-D tensors only

Computed from an `nalgebra::DMatrix<f64>` built row-major from the data.

| Property | Definition / method |
|----------|---------------------|
| `rank` | count of singular values `> 1e-10` (SVD) |
| `condition_number` | `σ_max / σ_min` when `σ_min > 1e-10`, else `None` |
| `is_square` | `rows == cols` |

### Square 2-D tensors only

| Property | Definition / method |
|----------|---------------------|
| `trace` | `Σ Aᵢᵢ` |
| `determinant` | `nalgebra` determinant |
| `is_symmetric` | `Aᵢⱼ ≈ Aⱼᵢ` under `allclose` (rtol `1e-5`, atol `1e-8`) |
| `is_diagonal` | every off-diagonal entry `|Aᵢⱼ| ≤ 1e-8` |
| `is_orthogonal` | `A·Aᵀ ≈ I` under the same tolerances |
| `max_eigenvalue` / `min_eigenvalue` | symmetric → real spectrum via Hermitian solver; otherwise the real parts of the complex spectrum |
| `is_positive_definite` | symmetric **and** all eigenvalues `> 0` |

For non-2-D tensors, all `Option`/matrix fields are `None`/`false`.

## Performance note

Computing the descriptor of a **(256, 256)** general (non-symmetric) matrix takes
~65 ms in release on the reference machine — above the plan's `< 10 ms` target.
The dominant cost is the general (non-symmetric) eigendecomposition: `nalgebra`'s
pure-Rust QR eigensolver is slower than the LAPACK/cuSOLVER the target assumes.
Symmetric matrices (Hermitian solver) and norm/stat-only paths are much faster.
Correctness — the hard requirement — is met. See [BENCHMARKS](../../BENCHMARKS.md).

## Example

```rust
use tensorus_compute::compute_descriptor;
use tensorus_core::types::DType;

// 2x2 identity matrix.
let d = compute_descriptor(&[1.0, 0.0, 0.0, 1.0], &[2, 2], DType::Float32);
assert!(d.is_symmetric && d.is_positive_definite && d.is_diagonal && d.is_orthogonal);
assert_eq!(d.rank, Some(2));
assert_eq!(d.trace, Some(2.0));
```

[`TensorData`]: ../data-model.md#tensordata
