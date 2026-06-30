//! Tensor contraction similarity: a structure-aware similarity metric for
//! tensors, built from truncated Tucker sketches, Grassmannian (principal-angle)
//! distances between mode subspaces, and a Procrustes-aligned core distance.
//!
//! Unlike cosine similarity on flattened tensors (which destroys multilinear
//! structure), contraction similarity compares the *subspace structure* of each
//! tensor mode, so two tensors that share mode subspaces are recognized as
//! similar even when their raw values differ.
//!
//! See §7 of the implementation plan and:
//! - Kolda & Bader, "Tensor Decompositions and Applications," SIAM Review (2009).
//! - Edelman et al., "The Geometry of Algorithms with Orthogonality
//!   Constraints," SIAM J. Matrix Anal. Appl. (1998).
// TODO: GPU acceleration — batch SVD via cuSOLVER for the per-mode factors.

use nalgebra::DMatrix;
use std::collections::{HashMap, HashSet};
use tensorus_core::types::TensorId;
use tensorus_index::hnsw::{Hnsw, HnswConfig, Metric};

/// A dense N-dimensional tensor (row-major, `f64` for numerical work).
#[derive(Debug, Clone)]
pub struct DenseTensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

impl DenseTensor {
    /// Build from `f32` data and a `u64` shape.
    pub fn from_f32(data: &[f32], shape: &[u64]) -> Self {
        DenseTensor {
            data: data.iter().map(|&x| x as f64).collect(),
            shape: shape.iter().map(|&d| d as usize).collect(),
        }
    }

    fn ndim(&self) -> usize {
        self.shape.len()
    }

    fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Row-major strides.
    fn strides(&self) -> Vec<usize> {
        let n = self.ndim();
        let mut s = vec![1usize; n];
        for i in (0..n.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * self.shape[i + 1];
        }
        s
    }

    /// Column strides for mode-`mode` unfolding (mode 0 varies fastest among the
    /// non-`mode` axes; Kolda & Bader convention).
    fn col_strides(&self, mode: usize) -> Vec<usize> {
        let n = self.ndim();
        let mut cs = vec![0usize; n];
        let mut mult = 1usize;
        for (m, slot) in cs.iter_mut().enumerate() {
            if m == mode {
                continue;
            }
            *slot = mult;
            mult *= self.shape[m];
        }
        cs
    }

    fn decode(flat: usize, strides: &[usize], shape: &[usize]) -> Vec<usize> {
        strides
            .iter()
            .zip(shape)
            .map(|(&st, &sh)| (flat / st) % sh)
            .collect()
    }

    /// Mode-`mode` matricization: shape `(I_mode, prod(other dims))`.
    fn unfold(&self, mode: usize) -> DMatrix<f64> {
        let rows = self.shape[mode];
        let cols = self.numel() / rows.max(1);
        let mut m = DMatrix::zeros(rows, cols);
        let strides = self.strides();
        let col_strides = self.col_strides(mode);
        for (flat, &val) in self.data.iter().enumerate() {
            let idx = Self::decode(flat, &strides, &self.shape);
            let r = idx[mode];
            let c: usize = (0..self.ndim())
                .filter(|&x| x != mode)
                .map(|x| idx[x] * col_strides[x])
                .sum();
            m[(r, c)] = val;
        }
        m
    }

    /// Inverse of [`unfold`]: build a tensor of `new_shape` from a mode-`mode`
    /// matricization.
    fn fold(p: &DMatrix<f64>, new_shape: &[usize], mode: usize) -> DenseTensor {
        let t = DenseTensor {
            data: vec![0.0; new_shape.iter().product()],
            shape: new_shape.to_vec(),
        };
        let strides = t.strides();
        let col_strides = t.col_strides(mode);
        let mut data = t.data;
        for (flat, slot) in data.iter_mut().enumerate() {
            let idx = Self::decode(flat, &strides, new_shape);
            let r = idx[mode];
            let c: usize = (0..new_shape.len())
                .filter(|&x| x != mode)
                .map(|x| idx[x] * col_strides[x])
                .sum();
            *slot = p[(r, c)];
        }
        DenseTensor {
            data,
            shape: new_shape.to_vec(),
        }
    }

    /// Mode-`mode` product with matrix `a` of shape `(out, I_mode)`; the result's
    /// `mode` dimension becomes `out`.
    pub fn mode_product(&self, a: &DMatrix<f64>, mode: usize) -> DenseTensor {
        let mk = self.unfold(mode);
        let p = a * mk;
        let mut new_shape = self.shape.clone();
        new_shape[mode] = a.nrows();
        DenseTensor::fold(&p, &new_shape, mode)
    }

    /// Frobenius norm of the elementwise difference with `other` (shapes must
    /// match).
    fn frobenius_diff(&self, other: &DenseTensor) -> f64 {
        self.data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }
}

/// A compact Tucker sketch: per-mode orthonormal factor bases plus the core.
#[derive(Debug, Clone)]
pub struct ContractionSketch {
    /// Factor matrices `U_k` of shape `(I_k, r_k)` (orthonormal columns).
    pub factors: Vec<DMatrix<f64>>,
    /// Core tensor of shape `(r_0, r_1, ..., r_{N-1})`.
    pub core: DenseTensor,
    /// Original tensor shape.
    pub shape: Vec<usize>,
}

impl ContractionSketch {
    /// Flatten factor `mode` (row-major) to an `f32` vector for indexing.
    pub fn factor_vector(&self, mode: usize) -> Vec<f32> {
        let f = &self.factors[mode];
        let mut v = Vec::with_capacity(f.nrows() * f.ncols());
        for r in 0..f.nrows() {
            for c in 0..f.ncols() {
                v.push(f[(r, c)] as f32);
            }
        }
        v
    }
}

/// Compute a truncated Tucker sketch with per-mode rank `min(rank, I_k)`.
pub fn tucker_sketch(data: &[f32], shape: &[u64], rank: usize) -> ContractionSketch {
    let tensor = DenseTensor::from_f32(data, shape);
    let n = tensor.ndim();
    let mut factors = Vec::with_capacity(n);
    for k in 0..n {
        let mk = tensor.unfold(k);
        let svd = mk.svd(true, false);
        let u = svd.u.expect("left singular vectors requested");
        let r_k = rank.min(tensor.shape[k]).min(u.ncols()).max(1);
        let factor = u.columns(0, r_k).into_owned();
        factors.push(factor);
    }
    // Core = T contracted with each U_k^T.
    let mut core = tensor.clone();
    for (k, factor) in factors.iter().enumerate() {
        core = core.mode_product(&factor.transpose(), k);
    }
    ContractionSketch {
        factors,
        core,
        shape: tensor.shape,
    }
}

/// Grassmannian distance between two subspaces given orthonormal bases (columns
/// of `ua`, `ub`): `sqrt(sum theta_i^2)` over principal angles.
pub fn grassmannian_distance(ua: &DMatrix<f64>, ub: &DMatrix<f64>) -> f64 {
    let r = ua.ncols().min(ub.ncols());
    let a = ua.columns(0, r).into_owned();
    let b = ub.columns(0, r).into_owned();
    let cross = a.transpose() * b; // (r x r)
    let svd = cross.svd(false, false);
    svd.singular_values
        .iter()
        .map(|&s| {
            let c = s.clamp(-1.0, 1.0);
            let angle = c.acos();
            angle * angle
        })
        .sum::<f64>()
        .sqrt()
}

/// Core distance after Procrustes alignment of factor subspaces.
fn aligned_core_distance(a: &ContractionSketch, b: &ContractionSketch) -> f64 {
    let mut core_b_aligned = b.core.clone();
    for (mode, (ua, ub)) in a.factors.iter().zip(&b.factors).enumerate() {
        let m = ua.transpose() * ub; // (r x r)
        let svd = m.svd(true, true);
        let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
        let rot = u * vt; // nearest orthogonal matrix
        core_b_aligned = core_b_aligned.mode_product(&rot, mode);
    }
    a.core.frobenius_diff(&core_b_aligned)
}

/// Default weighting of mode vs. core distance.
pub const MODE_WEIGHT: f64 = 0.6;
pub const CORE_WEIGHT: f64 = 0.4;

/// Contraction similarity between two pre-computed sketches, in `(0, 1]`.
/// Returns `0.0` for shape-incompatible tensors.
pub fn sketch_similarity(a: &ContractionSketch, b: &ContractionSketch) -> f64 {
    if a.shape != b.shape {
        return 0.0;
    }
    let mode_dists: Vec<f64> = a
        .factors
        .iter()
        .zip(&b.factors)
        .map(|(ua, ub)| grassmannian_distance(ua, ub))
        .collect();
    let avg_mode = if mode_dists.is_empty() {
        0.0
    } else {
        mode_dists.iter().sum::<f64>() / mode_dists.len() as f64
    };
    let core_dist = aligned_core_distance(a, b);
    let total = MODE_WEIGHT * avg_mode + CORE_WEIGHT * core_dist;
    (-total).exp()
}

/// Convenience: contraction similarity directly from two tensors.
pub fn contraction_similarity(
    a_data: &[f32],
    a_shape: &[u64],
    b_data: &[f32],
    b_shape: &[u64],
    rank: usize,
) -> f64 {
    if a_shape != b_shape {
        return 0.0;
    }
    let sa = tucker_sketch(a_data, a_shape, rank);
    let sb = tucker_sketch(b_data, b_shape, rank);
    sketch_similarity(&sa, &sb)
}

/// A multi-probe contraction-similarity index: one HNSW per tensor mode keyed by
/// flattened factor vectors, plus stored sketches for exact reranking.
///
/// All indexed tensors must share the same shape (factors must live in the same
/// ambient spaces to be comparable).
pub struct ContractionIndex {
    ndim: usize,
    rank: usize,
    shape: Option<Vec<u64>>,
    mode_indexes: Vec<Hnsw>,
    sketches: HashMap<TensorId, ContractionSketch>,
}

impl ContractionIndex {
    /// Create an index for `ndim`-mode tensors sketched at the given `rank`.
    pub fn new(ndim: usize, rank: usize) -> Self {
        let mode_indexes = (0..ndim)
            .map(|_| {
                Hnsw::new(HnswConfig {
                    metric: Metric::L2,
                    ..Default::default()
                })
            })
            .collect();
        ContractionIndex {
            ndim,
            rank,
            shape: None,
            mode_indexes,
            sketches: HashMap::new(),
        }
    }

    /// Number of indexed tensors.
    pub fn len(&self) -> usize {
        self.sketches.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.sketches.is_empty()
    }

    /// Insert a tensor. Returns an error string if the shape is incompatible.
    pub fn insert(&mut self, id: TensorId, data: &[f32], shape: &[u64]) -> Result<(), String> {
        if shape.len() != self.ndim {
            return Err(format!(
                "expected {}-mode tensor, got {}",
                self.ndim,
                shape.len()
            ));
        }
        match &self.shape {
            Some(s) if s != shape => {
                return Err(format!("index shape {s:?} does not match {shape:?}"));
            }
            None => self.shape = Some(shape.to_vec()),
            _ => {}
        }
        let sketch = tucker_sketch(data, shape, self.rank);
        for mode in 0..self.ndim {
            self.mode_indexes[mode].insert(id, &sketch.factor_vector(mode));
        }
        self.sketches.insert(id, sketch);
        Ok(())
    }

    /// Find the `k` most structurally similar tensors to `query`.
    pub fn search(&self, data: &[f32], shape: &[u64], k: usize) -> Vec<(TensorId, f64)> {
        if Some(shape.to_vec()) != self.shape {
            return Vec::new();
        }
        let query = tucker_sketch(data, shape, self.rank);
        // Gather candidates from every mode index (union).
        let probe = (k * 8).max(32);
        let mut candidates: HashSet<TensorId> = HashSet::new();
        for mode in 0..self.ndim {
            for (id, _) in self.mode_indexes[mode].search(&query.factor_vector(mode), probe) {
                candidates.insert(id);
            }
        }
        // Exact rerank by contraction similarity.
        let mut scored: Vec<(TensorId, f64)> = candidates
            .into_iter()
            .filter_map(|id| {
                self.sketches
                    .get(&id)
                    .map(|s| (id, sketch_similarity(&query, s)))
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.as_bytes().cmp(b.0.as_bytes())));
        scored.truncate(k);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Orthonormalize a random `rows x cols` matrix via SVD (columns of U).
    fn random_orthonormal(rows: usize, cols: usize, rng: &mut StdRng) -> DMatrix<f64> {
        let m = DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(-1.0..1.0));
        let svd = m.svd(true, false);
        svd.u.unwrap().columns(0, cols).into_owned()
    }

    /// Reconstruct a full tensor from a core and factor matrices.
    fn reconstruct(core: &DenseTensor, factors: &[DMatrix<f64>]) -> DenseTensor {
        let mut t = core.clone();
        for (k, f) in factors.iter().enumerate() {
            t = t.mode_product(f, k);
        }
        t
    }

    fn to_f32(t: &DenseTensor) -> Vec<f32> {
        t.data.iter().map(|&x| x as f32).collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    }

    #[test]
    fn unfold_fold_roundtrip() {
        // 2x3x2 tensor; unfold each mode and fold back.
        let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        let t = DenseTensor {
            data: data.clone(),
            shape: vec![2, 3, 2],
        };
        for mode in 0..3 {
            let m = t.unfold(mode);
            let back = DenseTensor::fold(&m, &t.shape, mode);
            assert_eq!(back.data, data, "mode {mode} roundtrip failed");
        }
    }

    #[test]
    fn self_similarity_is_one() {
        let mut rng = StdRng::seed_from_u64(1);
        let shape = [4u64, 5, 3];
        let data: Vec<f32> = (0..60).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let sim = contraction_similarity(&data, &shape, &data, &shape, 8);
        assert!((sim - 1.0).abs() < 1e-6, "self-similarity = {sim}");
    }

    #[test]
    fn shape_mismatch_is_zero() {
        let a = vec![1.0f32; 24];
        let b = vec![1.0f32; 24];
        assert_eq!(
            contraction_similarity(&a, &[2, 3, 4], &b, &[4, 3, 2], 8),
            0.0
        );
    }

    #[test]
    fn shared_subspace_more_similar_than_random() {
        let mut rng = StdRng::seed_from_u64(7);
        let dims = [6usize, 6, 4];
        let ranks = [3usize, 3, 2];
        let shape_u64 = [6u64, 6, 4];

        // Shared orthonormal factors.
        let factors: Vec<DMatrix<f64>> = (0..3)
            .map(|k| random_orthonormal(dims[k], ranks[k], &mut rng))
            .collect();
        let core_shape = ranks.to_vec();
        let make_core = |rng: &mut StdRng| DenseTensor {
            data: (0..ranks.iter().product::<usize>())
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect(),
            shape: core_shape.clone(),
        };

        let a = reconstruct(&make_core(&mut rng), &factors);
        let b = reconstruct(&make_core(&mut rng), &factors); // same subspaces, different core
        let c_data: Vec<f32> = (0..a.data.len())
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        let a_f = to_f32(&a);
        let b_f = to_f32(&b);

        let sim_ab = contraction_similarity(&a_f, &shape_u64, &b_f, &shape_u64, 8);
        let sim_ac = contraction_similarity(&a_f, &shape_u64, &c_data, &shape_u64, 8);

        println!(
            "contraction: sim(A,B shared-subspace)={sim_ab:.3} sim(A,C random)={sim_ac:.3}; \
             cosine: cos(A,B)={:.3} cos(A,C)={:.3}",
            cosine(&a_f, &b_f),
            cosine(&a_f, &c_data)
        );
        assert!(
            sim_ab > sim_ac,
            "shared-subspace tensor should be more similar: {sim_ab} vs {sim_ac}"
        );
    }

    #[test]
    fn index_finds_structurally_similar() {
        let mut rng = StdRng::seed_from_u64(3);
        let shape = [5u64, 5, 3];
        let mut index = ContractionIndex::new(3, 8);

        // Insert several random tensors and remember one "target".
        let mut target_id = TensorId::new();
        let mut target_data = Vec::new();
        for i in 0..20 {
            let data: Vec<f32> = (0..75).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let id = TensorId::new();
            index.insert(id, &data, &shape).unwrap();
            if i == 10 {
                target_id = id;
                target_data = data;
            }
        }
        assert_eq!(index.len(), 20);

        // Query with the target itself -> it should rank first with sim ~1.
        let res = index.search(&target_data, &shape, 3);
        assert!(!res.is_empty());
        assert_eq!(res[0].0, target_id);
        assert!(res[0].1 > 0.99, "self should score ~1, got {}", res[0].1);
    }

    #[test]
    fn cosine_misses_shared_column_space() {
        // A = u v^T and B = u w^T with v ⊥ w. The two matrices share their
        // column space (span(u)) but their flattened forms are orthogonal, so
        // cosine similarity is 0 — it "misses" the shared structure, while
        // contraction similarity recognizes it.
        let d = 4usize;
        let mut u = vec![1.0f32; d];
        let norm = (d as f32).sqrt();
        for x in &mut u {
            *x /= norm;
        }
        let v = [1.0f32, 0.0, 0.0, 0.0]; // e0
        let w = [0.0f32, 1.0, 0.0, 0.0]; // e1, orthogonal to v

        let mut a = vec![0.0f32; d * d];
        let mut b = vec![0.0f32; d * d];
        for i in 0..d {
            for j in 0..d {
                a[i * d + j] = u[i] * v[j];
                b[i * d + j] = u[i] * w[j];
            }
        }

        let shape = [d as u64, d as u64];
        // rank-1 sketch isolates the meaningful subspace (the data is rank-1).
        let sim = contraction_similarity(&a, &shape, &b, &shape, 1);
        let cos = cosine(&a, &b);
        println!("shared column space: contraction={sim:.3}, cosine={cos:.4}");

        assert!(
            cos.abs() < 1e-6,
            "cosine should be ~0 (orthogonal), got {cos}"
        );
        assert!(
            sim > 0.4,
            "contraction should recognize shared column space, got {sim}"
        );
    }

    #[test]
    #[ignore]
    fn sketch_timing_64x64x3() {
        let mut rng = StdRng::seed_from_u64(9);
        let shape = [64u64, 64, 3];
        let data: Vec<f32> = (0..64 * 64 * 3).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let start = std::time::Instant::now();
        let _ = tucker_sketch(&data, &shape, 8);
        println!("tucker_sketch(64x64x3) took {:?}", start.elapsed());
    }
}
