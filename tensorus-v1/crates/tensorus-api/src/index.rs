//! In-memory secondary indexes that make the running server a real
//! tensor/vector database rather than a CRUD store.
//!
//! For each dataset the [`IndexManager`] maintains:
//! - a **vector index** ([`Hnsw`]) over the flattened `f32` payload, for
//!   approximate nearest-neighbour similarity search;
//! - a **contraction index** ([`ContractionIndex`]) over same-shape tensors, for
//!   structure-aware similarity;
//! - a **property index** ([`PropertyIndex`]) — learned ([`AlexIndex`]) numeric
//!   indexes plus boolean posting sets — so descriptor predicates resolve
//!   without a full scan.
//!
//! The indexes are in-memory and rebuilt from durable storage on startup
//! (see [`crate::service::TensorService::recover`]). Vector and contraction
//! deletes are handled by HNSW tombstones and by filtering search results
//! against the live set, respectively (the contraction index has no delete).
//!
//! ## Dimension and shape rules
//!
//! A dataset's vector index adopts the dimension of the **first** indexed tensor
//! (Pinecone/Weaviate-style fixed-dimension collections). Tensors whose
//! flattened length differs are still stored and remain property-searchable, but
//! are skipped from the vector index. The contraction index likewise locks to the
//! shape of the first matrix/tensor (`ndim >= 2`) it sees.

use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use tensorus_core::types::{TensorDescriptor, TensorId};
use tensorus_index::{AlexIndex, Hnsw, HnswConfig, Metric};
use tensorus_search::ContractionIndex;

/// Numeric descriptor fields that get a learned index. Must match
/// [`TensorDescriptor::numeric_field`].
const NUMERIC_FIELDS: &[&str] = &[
    "frobenius_norm",
    "l1_norm",
    "l_inf_norm",
    "mean",
    "std_dev",
    "sparsity",
    "num_elements",
    "rank",
    "trace",
    "determinant",
    "condition_number",
    "max_eigenvalue",
    "min_eigenvalue",
];

/// Boolean descriptor fields that get a posting set. Must match
/// [`TensorDescriptor::bool_field`].
const BOOL_FIELDS: &[&str] = &[
    "is_symmetric",
    "is_positive_definite",
    "is_orthogonal",
    "is_sparse",
    "is_square",
    "is_diagonal",
];

/// Parse a metric name (case-insensitive). Accepts Pinecone/Weaviate spellings.
pub fn metric_from_str(s: &str) -> Option<Metric> {
    match s.trim().to_lowercase().as_str() {
        "cosine" => Some(Metric::Cosine),
        "l2" | "euclidean" | "euclid" => Some(Metric::L2),
        "dot" | "dotproduct" | "dot_product" | "inner_product" | "ip" => Some(Metric::Dot),
        _ => None,
    }
}

/// The canonical name for a metric (used for persistence and responses).
pub fn metric_name(m: Metric) -> &'static str {
    match m {
        Metric::Cosine => "cosine",
        Metric::L2 => "l2",
        Metric::Dot => "dot",
    }
}

/// Convert an index distance (smaller = closer) into a similarity score
/// (larger = more similar), matching the metric's semantics.
pub fn similarity(metric: Metric, distance: f32) -> f32 {
    match metric {
        // distance = 1 - cos  =>  similarity = cos
        Metric::Cosine => 1.0 - distance,
        // distance = squared L2  =>  bounded (0, 1]
        Metric::L2 => 1.0 / (1.0 + distance),
        // distance = -dot  =>  similarity = dot
        Metric::Dot => -distance,
    }
}

/// Comparison operator for a numeric predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumCmp {
    Eq,
    Gt,
    Lt,
    Ge,
    Le,
}

/// A numeric descriptor predicate.
#[derive(Debug, Clone)]
pub struct NumPred {
    pub field: String,
    pub cmp: NumCmp,
    pub value: f64,
}

/// A boolean descriptor predicate.
#[derive(Debug, Clone)]
pub struct BoolPred {
    pub field: String,
    pub value: bool,
}

/// A single property predicate (AND-combined by the search).
#[derive(Debug, Clone)]
pub enum PropPred {
    Num(NumPred),
    Bool(BoolPred),
}

impl PropPred {
    /// Exact evaluation against a descriptor (the verification step).
    fn eval(&self, d: &TensorDescriptor) -> bool {
        match self {
            PropPred::Num(np) => match d.numeric_field(&np.field) {
                Some(v) => match np.cmp {
                    NumCmp::Eq => (v - np.value).abs() <= 1e-9 * np.value.abs().max(1.0),
                    NumCmp::Gt => v > np.value,
                    NumCmp::Lt => v < np.value,
                    NumCmp::Ge => v >= np.value,
                    NumCmp::Le => v <= np.value,
                },
                None => false,
            },
            PropPred::Bool(bp) => d.bool_field(&bp.field) == Some(bp.value),
        }
    }
}

fn intersect(candidate: &mut Option<HashSet<u64>>, set: HashSet<u64>) {
    match candidate {
        Some(existing) => existing.retain(|x| set.contains(x)),
        None => *candidate = Some(set),
    }
}

/// Mutable interior state of a [`PropertyIndex`].
struct PropInner {
    /// Ordinal -> tensor id (append-only; ordinals are stable).
    ids: Vec<TensorId>,
    /// Ordinal -> descriptor (`None` once deleted).
    descriptors: Vec<Option<TensorDescriptor>>,
    /// Tensor id -> ordinal, for deletes and liveness checks.
    id_to_ord: HashMap<TensorId, u64>,
    /// Per-field set of ordinals whose boolean value is `true`.
    bool_true: HashMap<&'static str, HashSet<u64>>,
}

/// A property index: learned numeric indexes plus boolean posting sets over a
/// stable ordinal space, with descriptors retained for exact verification.
pub struct PropertyIndex {
    inner: RwLock<PropInner>,
    /// Field -> learned index mapping value to ordinal. `AlexIndex` is
    /// internally synchronized, so it lives outside the lock.
    numeric: HashMap<&'static str, AlexIndex>,
}

impl Default for PropertyIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl PropertyIndex {
    pub fn new() -> Self {
        let numeric = NUMERIC_FIELDS
            .iter()
            .map(|&f| (f, AlexIndex::new()))
            .collect();
        let bool_true = BOOL_FIELDS.iter().map(|&f| (f, HashSet::new())).collect();
        PropertyIndex {
            inner: RwLock::new(PropInner {
                ids: Vec::new(),
                descriptors: Vec::new(),
                id_to_ord: HashMap::new(),
                bool_true,
            }),
            numeric,
        }
    }

    /// Number of live (non-deleted) descriptors.
    pub fn len(&self) -> usize {
        self.inner.read().id_to_ord.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether a tensor is currently live in this index.
    pub fn contains(&self, id: TensorId) -> bool {
        self.inner.read().id_to_ord.contains_key(&id)
    }

    /// The stored descriptor for a live tensor, if any.
    pub fn descriptor(&self, id: TensorId) -> Option<TensorDescriptor> {
        let inner = self.inner.read();
        inner
            .id_to_ord
            .get(&id)
            .and_then(|&ord| inner.descriptors.get(ord as usize).cloned().flatten())
    }

    /// Index a descriptor. Re-inserting an existing id is ignored.
    pub fn insert(&self, id: TensorId, descriptor: &TensorDescriptor) {
        let ord = {
            let mut inner = self.inner.write();
            if inner.id_to_ord.contains_key(&id) {
                return;
            }
            let ord = inner.ids.len() as u64;
            inner.ids.push(id);
            inner.descriptors.push(Some(descriptor.clone()));
            inner.id_to_ord.insert(id, ord);
            for &f in BOOL_FIELDS {
                if descriptor.bool_field(f) == Some(true) {
                    inner.bool_true.get_mut(f).expect("known field").insert(ord);
                }
            }
            ord
        };
        for &f in NUMERIC_FIELDS {
            if let Some(v) = descriptor.numeric_field(f) {
                if v.is_finite() {
                    self.numeric[f].insert(v, ord);
                }
            }
        }
    }

    /// Remove a tensor from the index (idempotent).
    pub fn delete(&self, id: TensorId) {
        let mut inner = self.inner.write();
        if let Some(ord) = inner.id_to_ord.remove(&id) {
            if let Some(slot) = inner.descriptors.get_mut(ord as usize) {
                *slot = None;
            }
            for set in inner.bool_true.values_mut() {
                set.remove(&ord);
            }
        }
    }

    /// Find up to `limit` tensors satisfying all predicates (AND), in insertion
    /// order. Indexed predicates narrow the candidate set; every candidate is
    /// then verified exactly against its descriptor.
    pub fn search(&self, preds: &[PropPred], limit: usize) -> Vec<(TensorId, TensorDescriptor)> {
        let inner = self.inner.read();

        // Build the candidate ordinal set from indexable predicates.
        let mut candidate: Option<HashSet<u64>> = None;
        for p in preds {
            match p {
                PropPred::Num(np) => {
                    if let Some(alex) = self.numeric.get(np.field.as_str()) {
                        let ords = match np.cmp {
                            NumCmp::Eq => alex.lookup(np.value),
                            NumCmp::Gt | NumCmp::Ge => alex.range(np.value, f64::INFINITY),
                            NumCmp::Lt | NumCmp::Le => alex.range(f64::NEG_INFINITY, np.value),
                        };
                        intersect(&mut candidate, ords.into_iter().collect());
                    }
                }
                PropPred::Bool(bp) if bp.value => {
                    if let Some(set) = inner.bool_true.get(bp.field.as_str()) {
                        intersect(&mut candidate, set.iter().copied().collect());
                    }
                }
                // `field = false` is not selective enough to index; verify only.
                PropPred::Bool(_) => {}
            }
        }

        let ordinals: Vec<u64> = match candidate {
            Some(set) => {
                let mut v: Vec<u64> = set.into_iter().collect();
                v.sort_unstable();
                v
            }
            None => (0..inner.ids.len() as u64).collect(),
        };

        let mut out = Vec::new();
        for ord in ordinals {
            if let Some(Some(d)) = inner.descriptors.get(ord as usize) {
                if preds.iter().all(|p| p.eval(d)) {
                    out.push((inner.ids[ord as usize], d.clone()));
                    if out.len() >= limit {
                        break;
                    }
                }
            }
        }
        out
    }
}

/// Shape-locked contraction index state.
struct ContractionState {
    shape: Vec<u64>,
    index: ContractionIndex,
}

/// All secondary indexes for a single dataset.
pub struct DatasetIndexes {
    metric: Metric,
    contraction_rank: usize,
    vector: Hnsw,
    /// Established vector dimension (`None` until the first vector is indexed).
    vector_dim: RwLock<Option<usize>>,
    contraction: Mutex<Option<ContractionState>>,
    props: PropertyIndex,
}

impl DatasetIndexes {
    fn new(metric: Metric, contraction_rank: usize) -> Self {
        DatasetIndexes {
            metric,
            contraction_rank,
            vector: Hnsw::new(HnswConfig {
                metric,
                ..Default::default()
            }),
            vector_dim: RwLock::new(None),
            contraction: Mutex::new(None),
            props: PropertyIndex::new(),
        }
    }

    /// The vector metric this dataset's index was built with.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// The established vector dimension, if any vectors have been indexed.
    pub fn vector_dim(&self) -> Option<usize> {
        *self.vector_dim.read()
    }

    /// Number of live property-indexed tensors.
    pub fn len(&self) -> usize {
        self.props.len()
    }

    pub fn is_empty(&self) -> bool {
        self.props.is_empty()
    }

    /// The property index (for property/NQL search).
    pub fn props(&self) -> &PropertyIndex {
        &self.props
    }

    fn on_insert(&self, id: TensorId, data: &[f32], shape: &[u64], descriptor: &TensorDescriptor) {
        self.props.insert(id, descriptor);

        let index_vector = {
            let mut dim = self.vector_dim.write();
            match *dim {
                None => {
                    *dim = Some(data.len());
                    true
                }
                Some(d) => d == data.len(),
            }
        };
        if index_vector {
            self.vector.insert(id, data);
        } else {
            tracing::debug!(
                tensor = %id,
                "skipping vector index: dimension {} != dataset dimension {:?}",
                data.len(),
                self.vector_dim()
            );
        }

        self.maybe_index_contraction(id, data, shape);
    }

    fn maybe_index_contraction(&self, id: TensorId, data: &[f32], shape: &[u64]) {
        if shape.len() < 2 {
            return; // contraction similarity is meaningful for matrices/tensors
        }
        let mut guard = self.contraction.lock();
        match guard.as_mut() {
            None => {
                let mut index = ContractionIndex::new(shape.len(), self.contraction_rank);
                let _ = index.insert(id, data, shape);
                *guard = Some(ContractionState {
                    shape: shape.to_vec(),
                    index,
                });
            }
            Some(state) if state.shape == shape => {
                let _ = state.index.insert(id, data, shape);
            }
            Some(_) => { /* different shape than the locked one; skip */ }
        }
    }

    fn on_delete(&self, id: TensorId) {
        self.props.delete(id);
        self.vector.delete(id);
        // ContractionIndex has no delete; search filters against the live set.
    }

    /// Nearest-neighbour vector search. Returns `(id, distance)` pairs (smaller
    /// distance = closer). Errors if the query dimension does not match the
    /// dataset's established dimension.
    pub fn search_vector(&self, query: &[f32], k: usize) -> Result<Vec<(TensorId, f32)>, String> {
        match *self.vector_dim.read() {
            None => Ok(Vec::new()),
            Some(d) if d == query.len() => Ok(self.vector.search(query, k)),
            Some(d) => Err(format!(
                "query dimension {} does not match dataset vector dimension {}",
                query.len(),
                d
            )),
        }
    }

    /// Structure-aware contraction search over same-shape tensors. Deleted
    /// tensors are filtered out using the live property set.
    pub fn search_contraction(
        &self,
        data: &[f32],
        shape: &[u64],
        k: usize,
    ) -> Vec<(TensorId, f64)> {
        let guard = self.contraction.lock();
        match guard.as_ref() {
            Some(state) if state.shape == shape => {
                // Over-fetch so live-filtering still returns up to k results.
                let raw = state
                    .index
                    .search(data, shape, k.saturating_mul(2).max(k + 8));
                raw.into_iter()
                    .filter(|(id, _)| self.props.contains(*id))
                    .take(k)
                    .collect()
            }
            _ => Vec::new(),
        }
    }
}

/// Owns the secondary indexes for every dataset and (optionally) persists each
/// dataset's vector metric so it survives a restart.
pub struct IndexManager {
    datasets: RwLock<HashMap<String, Arc<DatasetIndexes>>>,
    contraction_rank: usize,
    /// Path to the metric-config sidecar (`None` = in-memory only, for tests).
    config_path: Option<PathBuf>,
    /// Dataset -> metric name, mirrored to `config_path` when set.
    config: Mutex<HashMap<String, String>>,
}

impl IndexManager {
    /// In-memory manager (no metric persistence); used by tests.
    pub fn in_memory(contraction_rank: usize) -> Self {
        IndexManager {
            datasets: RwLock::new(HashMap::new()),
            contraction_rank,
            config_path: None,
            config: Mutex::new(HashMap::new()),
        }
    }

    /// Manager that persists dataset metrics to `config_path`, loading any
    /// existing config.
    pub fn with_persistence(contraction_rank: usize, config_path: PathBuf) -> Self {
        let config = std::fs::read(&config_path)
            .ok()
            .and_then(|b| serde_json::from_slice::<HashMap<String, String>>(&b).ok())
            .unwrap_or_default();
        IndexManager {
            datasets: RwLock::new(HashMap::new()),
            contraction_rank,
            config_path: Some(config_path),
            config: Mutex::new(config),
        }
    }

    fn persist_config(&self) {
        if let Some(path) = &self.config_path {
            let snapshot = self.config.lock().clone();
            if let Ok(bytes) = serde_json::to_vec_pretty(&snapshot) {
                if let Some(parent) = path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let _ = std::fs::write(path, bytes);
            }
        }
    }

    /// The configured (or default cosine) metric for a dataset.
    pub fn metric_for(&self, dataset: &str) -> Metric {
        self.config
            .lock()
            .get(dataset)
            .and_then(|s| metric_from_str(s))
            .unwrap_or(Metric::Cosine)
    }

    /// Record a dataset's chosen vector metric and create its (empty) indexes.
    pub fn create_dataset(&self, name: &str, metric: Metric) {
        {
            let mut cfg = self.config.lock();
            cfg.insert(name.to_string(), metric_name(metric).to_string());
        }
        self.persist_config();
        let mut datasets = self.datasets.write();
        datasets
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(DatasetIndexes::new(metric, self.contraction_rank)));
    }

    /// Get a dataset's indexes, creating them (with the configured/default
    /// metric) if absent.
    fn get_or_create(&self, name: &str) -> Arc<DatasetIndexes> {
        if let Some(di) = self.datasets.read().get(name) {
            return di.clone();
        }
        let metric = self.metric_for(name);
        let mut datasets = self.datasets.write();
        datasets
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(DatasetIndexes::new(metric, self.contraction_rank)))
            .clone()
    }

    /// Get a dataset's indexes if they exist.
    pub fn dataset(&self, name: &str) -> Option<Arc<DatasetIndexes>> {
        self.datasets.read().get(name).cloned()
    }

    /// Index a freshly inserted tensor across all secondary indexes.
    pub fn on_insert(
        &self,
        dataset: &str,
        id: TensorId,
        data: &[f32],
        shape: &[u64],
        descriptor: &TensorDescriptor,
    ) {
        self.get_or_create(dataset)
            .on_insert(id, data, shape, descriptor);
    }

    /// Index only a tensor's descriptor (for non-`Float32` tensors that cannot
    /// be vector/contraction indexed but remain property-searchable).
    pub fn index_descriptor_only(
        &self,
        dataset: &str,
        id: TensorId,
        descriptor: &TensorDescriptor,
    ) {
        self.get_or_create(dataset).props().insert(id, descriptor);
    }

    /// Remove a tensor from all secondary indexes.
    pub fn on_delete(&self, dataset: &str, id: TensorId) {
        if let Some(di) = self.dataset(dataset) {
            di.on_delete(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorus_core::types::{DType, Shape};

    fn desc(norm: f64, symmetric: bool, rank: Option<u32>) -> TensorDescriptor {
        let mut d =
            TensorDescriptor::empty(TensorId::new(), Shape::new(vec![2, 2]), DType::Float32);
        d.frobenius_norm = norm;
        d.is_symmetric = symmetric;
        d.rank = rank;
        d
    }

    #[test]
    fn property_index_range_and_bool() {
        let idx = PropertyIndex::new();
        let a = TensorId::new();
        let b = TensorId::new();
        let c = TensorId::new();
        idx.insert(a, &desc(2.0, true, Some(2)));
        idx.insert(b, &desc(8.0, true, Some(2)));
        idx.insert(c, &desc(10.0, false, Some(1)));

        // norm >= 5 -> b, c
        let res = idx.search(
            &[PropPred::Num(NumPred {
                field: "frobenius_norm".into(),
                cmp: NumCmp::Ge,
                value: 5.0,
            })],
            100,
        );
        let ids: HashSet<TensorId> = res.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, HashSet::from([b, c]));

        // symmetric AND norm > 5 -> only b
        let res = idx.search(
            &[
                PropPred::Bool(BoolPred {
                    field: "is_symmetric".into(),
                    value: true,
                }),
                PropPred::Num(NumPred {
                    field: "frobenius_norm".into(),
                    cmp: NumCmp::Gt,
                    value: 5.0,
                }),
            ],
            100,
        );
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, b);
    }

    #[test]
    fn property_index_delete_excludes() {
        let idx = PropertyIndex::new();
        let a = TensorId::new();
        idx.insert(a, &desc(8.0, true, Some(2)));
        assert!(idx.contains(a));
        idx.delete(a);
        assert!(!idx.contains(a));
        let res = idx.search(
            &[PropPred::Bool(BoolPred {
                field: "is_symmetric".into(),
                value: true,
            })],
            100,
        );
        assert!(res.is_empty());
    }

    #[test]
    fn vector_index_adopts_first_dim_and_skips_mismatch() {
        let di = DatasetIndexes::new(Metric::L2, 8);
        let a = TensorId::new();
        let b = TensorId::new();
        di.on_insert(a, &[1.0, 0.0], &[2], &desc(1.0, false, None));
        assert_eq!(di.vector_dim(), Some(2));
        // Mismatched dimension: stored in props but skipped from the vector index.
        di.on_insert(b, &[1.0, 0.0, 0.0], &[3], &desc(1.0, false, None));
        assert!(di.props().contains(b));

        let res = di.search_vector(&[1.0, 0.0], 5).unwrap();
        assert!(res.iter().any(|(id, _)| *id == a));
        assert!(res.iter().all(|(id, _)| *id != b));

        // Wrong query dimension errors.
        assert!(di.search_vector(&[1.0, 0.0, 0.0], 5).is_err());
    }

    #[test]
    fn contraction_index_locks_shape_and_filters_deletes() {
        let di = DatasetIndexes::new(Metric::Cosine, 4);
        let a = TensorId::new();
        let data_a: Vec<f32> = (0..16).map(|i| i as f32).collect();
        di.on_insert(a, &data_a, &[4, 4], &desc(1.0, false, None));
        // Same-shape query finds a.
        let res = di.search_contraction(&data_a, &[4, 4], 3);
        assert!(res.iter().any(|(id, _)| *id == a));
        // After delete, a is filtered out.
        di.on_delete(a);
        let res = di.search_contraction(&data_a, &[4, 4], 3);
        assert!(res.iter().all(|(id, _)| *id != a));
    }

    #[test]
    fn metric_parsing_roundtrip() {
        assert_eq!(metric_from_str("cosine"), Some(Metric::Cosine));
        assert_eq!(metric_from_str("Euclidean"), Some(Metric::L2));
        assert_eq!(metric_from_str("dotProduct"), Some(Metric::Dot));
        assert_eq!(metric_from_str("nonsense"), None);
        assert_eq!(metric_name(Metric::Cosine), "cosine");
    }

    #[test]
    fn manager_default_metric_and_create() {
        let mgr = IndexManager::in_memory(8);
        assert_eq!(mgr.metric_for("unknown"), Metric::Cosine);
        mgr.create_dataset("vecs", Metric::L2);
        assert_eq!(mgr.metric_for("vecs"), Metric::L2);
        let id = TensorId::new();
        mgr.on_insert("vecs", id, &[1.0, 2.0, 3.0], &[3], &desc(1.0, false, None));
        let di = mgr.dataset("vecs").unwrap();
        assert_eq!(di.vector_dim(), Some(3));
        assert_eq!(di.metric(), Metric::L2);
    }
}
