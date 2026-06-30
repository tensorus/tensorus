# Service: `tensorus-search`

**Tensor contraction similarity engine** — a structure-aware similarity metric
for tensors, plus a multi-probe index built on it. `#![forbid(unsafe_code)]`.

- **Depends on:** `tensorus-core`, `tensorus-index` (HNSW), `tensorus-compute`,
  `nalgebra`, `serde`.

## Why not cosine similarity?

Flattening a tensor to a 1-D vector and taking cosine similarity destroys
multilinear structure. Two `(3,3,64)` convolution filters can have identical
cosine similarity yet detect completely different spatial patterns; and two
tensors that *share* a subspace can look orthogonal once flattened. Contraction
similarity compares the **subspace structure** of each tensor mode instead.

This is demonstrated by a test where `A = u·vᵀ` and `B = u·wᵀ` (with `v ⊥ w`)
share a column space: contraction similarity ≈ **0.624** while cosine = **0.0**.

---

## The algorithm

Given a tensor `T` of shape `(I₁,…,I_N)`:

1. **Tucker sketch.** For each mode `k`, unfold `T` to a matrix `M_k`, take the
   top-`r_k` left singular vectors `U_k` (orthonormal basis of the mode-`k`
   subspace, `r_k = min(rank, I_k)`). The **core** is `G = T ×₁U₁ᵀ ×₂…×_N U_Nᵀ`.
   The sketch is `(G, [U₁,…,U_N])` — typically 10–100× smaller than `T`.
2. **Grassmannian distance** between two tensors' mode-`k` subspaces: the
   principal angles `θᵢ` come from the SVD of `U_aᵀU_b`; the distance is
   `sqrt(Σ θᵢ²)`.
3. **Aligned core distance.** Align factors via orthogonal Procrustes, transform
   `B`'s core into `A`'s coordinate system, take the Frobenius distance of the
   cores.
4. **Combine:** `similarity = exp(−(0.6·mean_mode_distance + 0.4·core_distance))`,
   in `(0, 1]`. Shape-incompatible tensors score `0.0`.

---

## Public API

```rust
// One-shot similarity between two tensors (returns 0.0 on shape mismatch).
pub fn contraction_similarity(a_data: &[f32], a_shape: &[u64],
                              b_data: &[f32], b_shape: &[u64], rank: usize) -> f64;

// Lower-level building blocks.
pub fn tucker_sketch(data: &[f32], shape: &[u64], rank: usize) -> ContractionSketch;
pub fn grassmannian_distance(ua: &DMatrix<f64>, ub: &DMatrix<f64>) -> f64;
pub fn sketch_similarity(a: &ContractionSketch, b: &ContractionSketch) -> f64;

pub struct ContractionSketch { pub factors: Vec<DMatrix<f64>>, pub core: DenseTensor, pub shape: Vec<usize> }
impl ContractionSketch { pub fn factor_vector(&self, mode: usize) -> Vec<f32>; }

pub struct DenseTensor { pub data: Vec<f64>, pub shape: Vec<usize> }
impl DenseTensor {
    pub fn from_f32(data: &[f32], shape: &[u64]) -> Self;
    pub fn mode_product(&self, a: &DMatrix<f64>, mode: usize) -> DenseTensor;
}

pub const MODE_WEIGHT: f64 = 0.6;
pub const CORE_WEIGHT: f64 = 0.4;
```

## Contraction index

A multi-probe index: one HNSW per tensor mode, keyed by the flattened mode
factor vectors, plus stored sketches for exact reranking. All indexed tensors
must share the same shape (so factors live in comparable ambient spaces).

```rust
pub struct ContractionIndex { /* … */ }
impl ContractionIndex {
    pub fn new(ndim: usize, rank: usize) -> Self;
    pub fn insert(&mut self, id: TensorId, data: &[f32], shape: &[u64]) -> Result<(), String>;
    pub fn search(&self, data: &[f32], shape: &[u64], k: usize) -> Vec<(TensorId, f64)>;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}
```

`search` probes every mode index, **unions** the candidates, then reranks them
by exact `sketch_similarity`, returning the top-`k` `(id, similarity)` pairs
(higher = more similar).

## Performance

Tucker sketch of a **(64, 64, 3)** tensor: ~4.2 ms (target `<10 ms`). Self-
similarity is `1.0`; structurally-related tensors score higher than random ones.

## Example

```rust
use tensorus_search::{contraction_similarity, ContractionIndex};
use tensorus_core::types::TensorId;

let a = [1.0f32, 0.0, 0.0, 1.0];           // 2x2 identity
let shape = [2u64, 2];
assert!((contraction_similarity(&a, &shape, &a, &shape, 2) - 1.0).abs() < 1e-6);

let mut idx = ContractionIndex::new(2, 8);
idx.insert(TensorId::new(), &a, &shape).unwrap();
let hits = idx.search(&a, &shape, 1);      // [(id, ~1.0)]
```

## GPU note

SVD of the per-mode factors is the hot path; `// TODO: GPU acceleration` marks
where batched cuSOLVER SVD would slot in.
