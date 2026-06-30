//! TensorDescriptor computation (CPU path).
//!
//! Ports v0.1's `TensorDescriptor.from_tensor()` to Rust, computing a tensor's
//! mathematical "fingerprint": norms, statistics, and (for 2-D tensors) matrix
//! properties via SVD / eigendecomposition. Results are designed to match the
//! v0.1 PyTorch implementation within a tolerance of 1e-6.
//!
//! All computation is performed in `f64` for accuracy regardless of the stored
//! dtype.
// TODO: GPU acceleration — batch SVD/eigendecomposition via cuSOLVER and norms
// via cuBLAS for large or numerous tensors.

use nalgebra::DMatrix;
use tensorus_core::types::{DType, Shape, TensorDescriptor, TensorId};

/// `torch.allclose` semantics: `|a - b| <= atol + rtol * |b|`.
fn close(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

const RTOL: f64 = 1e-5;
const ATOL: f64 = 1e-8;
/// Singular values below this are treated as zero (matches v0.1).
const SV_EPS: f64 = 1e-10;

/// Compute the full descriptor for a tensor given its raw `f32` values
/// (row-major), shape, and dtype.
pub fn compute_descriptor(data: &[f32], shape: &[u64], dtype: DType) -> TensorDescriptor {
    let id = TensorId::new();
    let shape_vec: Vec<u64> = shape.to_vec();
    let n = data.len();
    let mut desc = TensorDescriptor::empty(id, Shape::new(shape_vec), dtype);
    desc.num_elements = n as u64;

    if n == 0 {
        return desc;
    }

    // Work in f64.
    let values: Vec<f64> = data.iter().map(|&x| x as f64).collect();

    // Norms.
    let mut sum_sq = 0.0;
    let mut sum_abs = 0.0;
    let mut max_abs = 0.0_f64;
    let mut sum = 0.0;
    let mut num_zeros = 0u64;
    for &v in &values {
        sum_sq += v * v;
        sum_abs += v.abs();
        if v.abs() > max_abs {
            max_abs = v.abs();
        }
        sum += v;
        if v == 0.0 {
            num_zeros += 1;
        }
    }
    desc.frobenius_norm = sum_sq.sqrt();
    desc.l1_norm = sum_abs;
    desc.l_inf_norm = max_abs;

    // Statistics.
    let mean = sum / n as f64;
    desc.mean = mean;
    desc.std_dev = if n > 1 {
        let var = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
        var.sqrt()
    } else {
        0.0
    };

    // Sparsity.
    desc.sparsity = num_zeros as f64 / n as f64;
    desc.is_sparse = desc.sparsity > 0.5;

    // Matrix properties (2-D only).
    if shape.len() == 2 {
        let rows = shape[0] as usize;
        let cols = shape[1] as usize;
        desc.is_square = rows == cols;
        let m = DMatrix::<f64>::from_row_slice(rows, cols, &values);

        // Rank + condition number from singular values (any 2-D matrix).
        if rows > 0 && cols > 0 {
            let mut svs: Vec<f64> = m.clone().singular_values().iter().copied().collect();
            svs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let rank = svs.iter().filter(|&&s| s > SV_EPS).count() as u32;
            desc.rank = Some(rank);
            if let (Some(&s_max), Some(&s_min)) = (svs.first(), svs.last()) {
                if s_min > SV_EPS {
                    desc.condition_number = Some(s_max / s_min);
                }
            }
        }

        if desc.is_square && rows > 0 {
            desc.trace = Some(m.trace());

            // Symmetry: compare against the transpose elementwise.
            let mut symmetric = true;
            'sym: for i in 0..rows {
                for j in 0..rows {
                    if !close(m[(i, j)], m[(j, i)], RTOL, ATOL) {
                        symmetric = false;
                        break 'sym;
                    }
                }
            }
            desc.is_symmetric = symmetric;

            // Diagonal: every off-diagonal entry is ~0 (atol bound; rtol*0 = 0).
            let mut diagonal = true;
            'diag: for i in 0..rows {
                for j in 0..rows {
                    if i != j && m[(i, j)].abs() > ATOL {
                        diagonal = false;
                        break 'diag;
                    }
                }
            }
            desc.is_diagonal = diagonal;

            desc.determinant = Some(m.determinant());

            // Eigenvalues: symmetric -> real spectrum via Hermitian solver;
            // otherwise take the real parts of the complex spectrum.
            let eigs: Vec<f64> = if symmetric {
                m.clone().symmetric_eigenvalues().iter().copied().collect()
            } else {
                m.clone()
                    .complex_eigenvalues()
                    .iter()
                    .map(|c| c.re)
                    .collect()
            };
            if !eigs.is_empty() {
                let max_e = eigs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let min_e = eigs.iter().copied().fold(f64::INFINITY, f64::min);
                desc.max_eigenvalue = Some(max_e);
                desc.min_eigenvalue = Some(min_e);
                if symmetric && min_e > 0.0 {
                    desc.is_positive_definite = true;
                }
            }

            // Orthogonality: t @ t^T == I.
            let product = &m * m.transpose();
            let mut orthogonal = true;
            'orth: for i in 0..rows {
                for j in 0..rows {
                    let target = if i == j { 1.0 } else { 0.0 };
                    if !close(product[(i, j)], target, RTOL, ATOL) {
                        orthogonal = false;
                        break 'orth;
                    }
                }
            }
            desc.is_orthogonal = orthogonal;
        }
    }

    desc
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn vector_norms_and_stats() {
        let d = compute_descriptor(&[3.0, 4.0], &[2], DType::Float32);
        assert_relative_eq!(d.frobenius_norm, 5.0, epsilon = 1e-6);
        assert_relative_eq!(d.l1_norm, 7.0, epsilon = 1e-6);
        assert_relative_eq!(d.l_inf_norm, 4.0, epsilon = 1e-6);
        assert_relative_eq!(d.mean, 3.5, epsilon = 1e-6);
        // unbiased std of [3,4] = sqrt(0.5)
        assert_relative_eq!(d.std_dev, 0.5f64.sqrt(), epsilon = 1e-6);
        assert!(d.rank.is_none());
        assert!(!d.is_square);
    }

    #[test]
    fn identity_matrix_properties() {
        let data = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let d = compute_descriptor(&data, &[3, 3], DType::Float32);
        assert_relative_eq!(d.frobenius_norm, 3.0f64.sqrt(), epsilon = 1e-6);
        assert_eq!(d.rank, Some(3));
        assert_relative_eq!(d.trace.unwrap(), 3.0, epsilon = 1e-6);
        assert_relative_eq!(d.determinant.unwrap(), 1.0, epsilon = 1e-6);
        assert_relative_eq!(d.condition_number.unwrap(), 1.0, epsilon = 1e-6);
        assert!(d.is_symmetric);
        assert!(d.is_diagonal);
        assert!(d.is_orthogonal);
        assert!(d.is_positive_definite);
        assert!(d.is_square);
        // 6 of 9 entries are zero -> sparse.
        assert!(d.is_sparse);
        assert_relative_eq!(d.max_eigenvalue.unwrap(), 1.0, epsilon = 1e-6);
        assert_relative_eq!(d.min_eigenvalue.unwrap(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn symmetric_positive_definite() {
        // [[2,1],[1,2]] -> eigenvalues 1 and 3.
        let data = [2.0, 1.0, 1.0, 2.0];
        let d = compute_descriptor(&data, &[2, 2], DType::Float32);
        assert!(d.is_symmetric);
        assert!(!d.is_diagonal);
        assert!(!d.is_orthogonal);
        assert!(d.is_positive_definite);
        assert_relative_eq!(d.trace.unwrap(), 4.0, epsilon = 1e-6);
        assert_relative_eq!(d.determinant.unwrap(), 3.0, epsilon = 1e-6);
        assert_relative_eq!(d.min_eigenvalue.unwrap(), 1.0, epsilon = 1e-5);
        assert_relative_eq!(d.max_eigenvalue.unwrap(), 3.0, epsilon = 1e-5);
        assert_relative_eq!(d.frobenius_norm, 10.0f64.sqrt(), epsilon = 1e-6);
    }

    #[test]
    fn nonsymmetric_matrix() {
        // [[1,2],[3,4]] -> det = -2, trace = 5, not symmetric.
        let data = [1.0, 2.0, 3.0, 4.0];
        let d = compute_descriptor(&data, &[2, 2], DType::Float32);
        assert!(!d.is_symmetric);
        assert!(!d.is_positive_definite);
        assert_eq!(d.rank, Some(2));
        assert_relative_eq!(d.trace.unwrap(), 5.0, epsilon = 1e-6);
        assert_relative_eq!(d.determinant.unwrap(), -2.0, epsilon = 1e-5);
    }

    #[test]
    fn diagonal_non_identity() {
        let data = [5.0, 0.0, 0.0, 3.0];
        let d = compute_descriptor(&data, &[2, 2], DType::Float32);
        assert!(d.is_diagonal);
        assert!(d.is_symmetric);
        assert!(!d.is_orthogonal);
        assert!(d.is_positive_definite);
        assert_relative_eq!(d.min_eigenvalue.unwrap(), 3.0, epsilon = 1e-6);
        assert_relative_eq!(d.max_eigenvalue.unwrap(), 5.0, epsilon = 1e-6);
    }

    #[test]
    fn rank_deficient_matrix() {
        // [[1,2],[2,4]] is rank 1.
        let data = [1.0, 2.0, 2.0, 4.0];
        let d = compute_descriptor(&data, &[2, 2], DType::Float32);
        assert_eq!(d.rank, Some(1));
        // s_min ~ 0 -> condition number left unset.
        assert!(d.condition_number.is_none());
    }

    #[test]
    fn non_square_2d_has_rank_only() {
        // 2x3 matrix: rank computed, square-only props stay unset.
        let data = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let d = compute_descriptor(&data, &[2, 3], DType::Float32);
        assert!(!d.is_square);
        assert_eq!(d.rank, Some(2));
        assert!(d.trace.is_none());
        assert!(d.determinant.is_none());
    }

    #[test]
    fn empty_tensor() {
        let d = compute_descriptor(&[], &[0], DType::Float32);
        assert_eq!(d.num_elements, 0);
        assert_eq!(d.frobenius_norm, 0.0);
    }

    #[test]
    fn three_d_tensor_no_matrix_props() {
        let data = vec![1.0f32; 24];
        let d = compute_descriptor(&data, &[2, 3, 4], DType::Float32);
        assert!(d.rank.is_none());
        assert!(!d.is_square);
        assert_eq!(d.num_elements, 24);
    }

    /// Deterministic, generally non-symmetric 256x256 matrix used by the
    /// correctness and timing tests below.
    fn sample_256x256() -> Vec<f32> {
        let n = 256usize;
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let x = (i as f32 * 0.013 + j as f32 * 0.027).sin() + (i as f32 - j as f32) * 0.001;
                data[i * n + j] = x;
            }
        }
        data
    }

    #[test]
    fn descriptor_256x256_correctness() {
        let data = sample_256x256();
        let d = compute_descriptor(&data, &[256, 256], DType::Float32);
        assert_eq!(d.shape.dims(), &[256, 256]);
        assert!(d.rank.is_some());
        assert!(d.is_square);
    }

    /// Wall-clock performance check. Ignored by default because the plan target
    /// (< 10ms) only holds for an optimized build; debug builds run ~50x slower.
    /// Run with: `cargo test -p tensorus-compute --release -- --ignored`.
    #[test]
    #[ignore = "performance benchmark; run in release with --ignored"]
    fn descriptor_256x256_timing() {
        let data = sample_256x256();
        let start = std::time::Instant::now();
        let d = compute_descriptor(&data, &[256, 256], DType::Float32);
        let elapsed = start.elapsed();
        println!("compute_descriptor(256x256) took {elapsed:?}");
        assert!(d.rank.is_some());
        // Generous bound vs the < 10ms release target, to absorb CI variance.
        assert!(elapsed.as_millis() < 500, "too slow: {elapsed:?}");
    }
}
