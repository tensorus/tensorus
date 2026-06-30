//! `TensorService`: the transport-agnostic business logic shared by the REST and
//! gRPC surfaces. It wraps the storage engine, computes a tensor's descriptor on
//! insert, and maintains the secondary indexes ([`IndexManager`]) that power
//! vector, contraction, and property search.

use crate::index::{
    metric_from_str, metric_name, similarity, BoolPred, DatasetIndexes, IndexManager, NumCmp,
    NumPred, PropPred,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tensorus_ai::{Cmp, Predicate, QueryContext, QueryRow};
use tensorus_compute::compute_descriptor;
use tensorus_core::error::{Result, TensorusError};
use tensorus_core::types::{DType, Shape, TensorData, TensorDescriptor, TensorId, TensorRecord};
use tensorus_core::Storage;
use tensorus_index::Metric;
use tensorus_storage::FileStorage;

/// Default Tucker-sketch rank for contraction indexing (matches
/// `tensorus.toml`'s `index.tensor_contraction.default_sketch_rank`).
const DEFAULT_CONTRACTION_RANK: usize = 8;

/// Request body for inserting a tensor (`Float32`).
#[derive(Debug, Clone, Deserialize)]
pub struct InsertRequest {
    pub data: Vec<f32>,
    pub shape: Vec<u64>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// Property-search query (mirrors the gRPC `PropertySearchRequest`).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct PropertyQuery {
    pub min_norm: Option<f64>,
    pub max_norm: Option<f64>,
    pub is_symmetric: Option<bool>,
    pub is_positive_definite: Option<bool>,
    pub rank: Option<u32>,
    pub max_condition_number: Option<f64>,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    100
}

/// Health snapshot.
#[derive(Debug, Clone, Serialize)]
pub struct HealthInfo {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
}

/// A vector-similarity search hit.
#[derive(Debug, Clone, Serialize)]
pub struct SimilarHit {
    pub tensor_id: TensorId,
    /// Raw index distance (smaller is closer).
    pub distance: f32,
    /// Metric-appropriate similarity score (larger is more similar).
    pub score: f32,
    pub descriptor: TensorDescriptor,
    pub metadata: serde_json::Value,
}

/// A contraction-similarity search hit.
#[derive(Debug, Clone, Serialize)]
pub struct ContractionHit {
    pub tensor_id: TensorId,
    /// Structural similarity in `(0, 1]` (larger is more similar).
    pub score: f64,
    pub descriptor: TensorDescriptor,
    pub metadata: serde_json::Value,
}

/// The core service.
#[derive(Clone)]
pub struct TensorService {
    storage: Arc<FileStorage>,
    indexes: Arc<IndexManager>,
    started: Instant,
}

impl TensorService {
    /// Build a service with in-memory secondary indexes (no metric
    /// persistence). Indexes populate as tensors are inserted.
    pub fn new(storage: Arc<FileStorage>) -> Self {
        TensorService {
            storage,
            indexes: Arc::new(IndexManager::in_memory(DEFAULT_CONTRACTION_RANK)),
            started: Instant::now(),
        }
    }

    /// Build a service whose per-dataset vector metrics persist to
    /// `index_config_path` (so they survive a restart).
    pub fn with_index_persistence(storage: Arc<FileStorage>, index_config_path: PathBuf) -> Self {
        TensorService {
            storage,
            indexes: Arc::new(IndexManager::with_persistence(
                DEFAULT_CONTRACTION_RANK,
                index_config_path,
            )),
            started: Instant::now(),
        }
    }

    /// Rebuild the in-memory secondary indexes from durable storage. Call once
    /// at startup, after opening storage.
    #[tracing::instrument(skip(self))]
    pub async fn recover(&self) -> Result<()> {
        let datasets = self.storage.list_datasets().await?;
        let mut total = 0usize;
        for ds in &datasets {
            self.indexes.create_dataset(ds, self.indexes.metric_for(ds));
            let mut offset = 0;
            const PAGE: usize = 1000;
            loop {
                let batch = self.storage.scan(ds, PAGE, offset).await?;
                let n = batch.len();
                if n == 0 {
                    break;
                }
                for rec in &batch {
                    self.index_record(ds, rec);
                    total += 1;
                }
                offset += n;
                if n < PAGE {
                    break;
                }
            }
        }
        tracing::info!(
            datasets = datasets.len(),
            tensors = total,
            "index recovery complete"
        );
        Ok(())
    }

    /// Index a single stored record into the secondary indexes.
    fn index_record(&self, dataset: &str, rec: &TensorRecord) {
        if rec.descriptor.dtype == DType::Float32 {
            let data: Vec<f32> = rec
                .data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            self.indexes.on_insert(
                dataset,
                rec.id,
                &data,
                rec.descriptor.shape.dims(),
                &rec.descriptor,
            );
        } else {
            // Non-f32 tensors are still property-searchable by descriptor.
            self.indexes
                .index_descriptor_only(dataset, rec.id, &rec.descriptor);
        }
    }

    pub async fn create_dataset(&self, name: &str) -> Result<()> {
        self.storage.create_dataset(name).await?;
        self.indexes
            .create_dataset(name, self.indexes.metric_for(name));
        Ok(())
    }

    /// Create a dataset whose vector index uses the given metric name
    /// (`cosine` | `l2`/`euclidean` | `dot`).
    pub async fn create_dataset_with_metric(&self, name: &str, metric: &str) -> Result<()> {
        let metric = metric_from_str(metric)
            .ok_or_else(|| TensorusError::InvalidArgument(format!("unknown metric '{metric}'")))?;
        self.storage.create_dataset(name).await?;
        self.indexes.create_dataset(name, metric);
        Ok(())
    }

    pub async fn list_datasets(&self) -> Result<Vec<String>> {
        self.storage.list_datasets().await
    }

    /// The vector metric configured for a dataset.
    pub fn dataset_metric(&self, dataset: &str) -> Metric {
        self.indexes.metric_for(dataset)
    }

    /// Insert a `Float32` tensor, computing its descriptor and updating indexes.
    #[tracing::instrument(skip(self, data, metadata), fields(dataset = %dataset, n = data.len()))]
    pub async fn insert(
        &self,
        dataset: &str,
        data: &[f32],
        shape: &[u64],
        metadata: serde_json::Value,
    ) -> Result<(TensorId, TensorDescriptor)> {
        let td = TensorData::from_f32(data, Shape::new(shape.to_vec()))?;
        let mut descriptor = compute_descriptor(data, shape, DType::Float32);
        let id = self
            .storage
            .insert(dataset, &td.data, descriptor.clone(), metadata)
            .await?;
        descriptor.tensor_id = id;
        self.indexes
            .on_insert(dataset, id, data, shape, &descriptor);
        Ok((id, descriptor))
    }

    pub async fn get(&self, dataset: &str, id: TensorId) -> Result<TensorRecord> {
        self.storage.get(dataset, id).await
    }

    pub async fn delete(&self, dataset: &str, id: TensorId) -> Result<()> {
        self.storage.delete(dataset, id).await?;
        self.indexes.on_delete(dataset, id);
        Ok(())
    }

    pub async fn scan(
        &self,
        dataset: &str,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<TensorRecord>> {
        self.storage.scan(dataset, limit, offset).await
    }

    /// Resolve a dataset's secondary indexes, creating an empty index set if the
    /// dataset exists in storage but has not been indexed yet. Errors with
    /// `NotFound` if the dataset does not exist at all.
    async fn require_dataset_index(&self, dataset: &str) -> Result<Arc<DatasetIndexes>> {
        if let Some(di) = self.indexes.dataset(dataset) {
            return Ok(di);
        }
        if self
            .storage
            .list_datasets()
            .await?
            .iter()
            .any(|d| d == dataset)
        {
            self.indexes
                .create_dataset(dataset, self.indexes.metric_for(dataset));
            return Ok(self.indexes.dataset(dataset).expect("just created"));
        }
        Err(TensorusError::NotFound(format!("dataset '{dataset}'")))
    }

    /// Property search, backed by the per-dataset learned/boolean indexes
    /// (falling back to verification, never a full storage scan).
    #[tracing::instrument(skip(self, q), fields(dataset = %dataset))]
    pub async fn search_by_property(
        &self,
        dataset: &str,
        q: &PropertyQuery,
    ) -> Result<Vec<(TensorId, TensorDescriptor)>> {
        let di = self.require_dataset_index(dataset).await?;
        let preds = property_query_preds(q);
        Ok(di.props().search(&preds, q.limit))
    }

    /// Approximate nearest-neighbour vector search over the flattened tensor
    /// payload. `metric_override`, if given, must match the dataset's index
    /// metric.
    #[tracing::instrument(skip(self, query), fields(dataset = %dataset, dim = query.len(), k))]
    pub async fn search_similar(
        &self,
        dataset: &str,
        query: &[f32],
        k: usize,
        metric_override: Option<&str>,
    ) -> Result<Vec<SimilarHit>> {
        let di = self.require_dataset_index(dataset).await?;
        if let Some(m) = metric_override {
            let requested = metric_from_str(m)
                .ok_or_else(|| TensorusError::InvalidArgument(format!("unknown metric '{m}'")))?;
            if requested != di.metric() {
                return Err(TensorusError::InvalidArgument(format!(
                    "dataset '{dataset}' vector index uses metric '{}'; \
                     recreate the dataset to change it",
                    metric_name(di.metric())
                )));
            }
        }
        let hits = di
            .search_vector(query, k)
            .map_err(TensorusError::InvalidArgument)?;
        let metric = di.metric();
        let mut out = Vec::with_capacity(hits.len());
        for (id, distance) in hits {
            let (descriptor, metadata) = match self.storage.get(dataset, id).await {
                Ok(rec) => (rec.descriptor, rec.metadata),
                Err(_) => match di.props().descriptor(id) {
                    Some(d) => (d, Value::Null),
                    None => continue,
                },
            };
            out.push(SimilarHit {
                tensor_id: id,
                distance,
                score: similarity(metric, distance),
                descriptor,
                metadata,
            });
        }
        Ok(out)
    }

    /// Structure-aware contraction similarity search over same-shape tensors.
    #[tracing::instrument(skip(self, data), fields(dataset = %dataset, k))]
    pub async fn search_contraction(
        &self,
        dataset: &str,
        data: &[f32],
        shape: &[u64],
        k: usize,
    ) -> Result<Vec<ContractionHit>> {
        let di = self.require_dataset_index(dataset).await?;
        let hits = di.search_contraction(data, shape, k);
        let mut out = Vec::with_capacity(hits.len());
        for (id, score) in hits {
            if let Ok(rec) = self.storage.get(dataset, id).await {
                out.push(ContractionHit {
                    tensor_id: id,
                    score,
                    descriptor: rec.descriptor,
                    metadata: rec.metadata,
                });
            }
        }
        Ok(out)
    }

    pub fn health(&self) -> HealthInfo {
        HealthInfo {
            status: "healthy".to_string(),
            version: tensorus_core::VERSION.to_string(),
            uptime_seconds: self.started.elapsed().as_secs(),
        }
    }

    /// Map a [`TensorusError`] to an HTTP-ish status code.
    pub fn status_code(err: &TensorusError) -> u16 {
        match err {
            TensorusError::NotFound(_) => 404,
            TensorusError::AlreadyExists(_) => 409,
            TensorusError::InvalidArgument(_) | TensorusError::DimensionMismatch(_) => 400,
            _ => 500,
        }
    }
}

/// Translate the fixed REST property query into index predicates.
fn property_query_preds(q: &PropertyQuery) -> Vec<PropPred> {
    let mut v = Vec::new();
    if let Some(min) = q.min_norm {
        v.push(PropPred::Num(NumPred {
            field: "frobenius_norm".into(),
            cmp: NumCmp::Ge,
            value: min,
        }));
    }
    if let Some(max) = q.max_norm {
        v.push(PropPred::Num(NumPred {
            field: "frobenius_norm".into(),
            cmp: NumCmp::Le,
            value: max,
        }));
    }
    if let Some(sym) = q.is_symmetric {
        v.push(PropPred::Bool(BoolPred {
            field: "is_symmetric".into(),
            value: sym,
        }));
    }
    if let Some(pd) = q.is_positive_definite {
        v.push(PropPred::Bool(BoolPred {
            field: "is_positive_definite".into(),
            value: pd,
        }));
    }
    if let Some(rank) = q.rank {
        v.push(PropPred::Num(NumPred {
            field: "rank".into(),
            cmp: NumCmp::Eq,
            value: rank as f64,
        }));
    }
    if let Some(maxc) = q.max_condition_number {
        v.push(PropPred::Num(NumPred {
            field: "condition_number".into(),
            cmp: NumCmp::Le,
            value: maxc,
        }));
    }
    v
}

/// Translate an NQL predicate into an index predicate (numeric or boolean).
fn nql_predicate_to_proppred(p: &Predicate) -> Option<PropPred> {
    if let Some(b) = p.value.as_bool() {
        return Some(PropPred::Bool(BoolPred {
            field: p.field.clone(),
            value: b,
        }));
    }
    if let Some(v) = p.value.as_f64() {
        let cmp = match p.cmp {
            Cmp::Eq => NumCmp::Eq,
            Cmp::Gt => NumCmp::Gt,
            Cmp::Lt => NumCmp::Lt,
            Cmp::Ge => NumCmp::Ge,
            Cmp::Le => NumCmp::Le,
        };
        return Some(PropPred::Num(NumPred {
            field: p.field.clone(),
            cmp,
            value: v,
        }));
    }
    None
}

/// Bridges NQL [`QueryPlan`](tensorus_ai::QueryPlan) execution to storage + indexes.
#[async_trait]
impl QueryContext for TensorService {
    async fn scan(
        &self,
        dataset: &str,
        limit: usize,
    ) -> std::result::Result<Vec<QueryRow>, String> {
        let recs = self
            .storage
            .scan(dataset, limit, 0)
            .await
            .map_err(|e| e.to_string())?;
        Ok(recs
            .into_iter()
            .map(|r| QueryRow {
                id: r.id,
                score: 1.0,
                metadata: r.metadata,
            })
            .collect())
    }

    async fn property_search(
        &self,
        dataset: &str,
        predicates: &[Predicate],
        limit: usize,
    ) -> std::result::Result<Vec<QueryRow>, String> {
        let di = self
            .indexes
            .dataset(dataset)
            .ok_or_else(|| format!("unknown dataset '{dataset}'"))?;
        let preds: Vec<PropPred> = predicates
            .iter()
            .filter_map(nql_predicate_to_proppred)
            .collect();
        let hits = di.props().search(&preds, limit);
        Ok(hits
            .into_iter()
            .map(|(id, _d)| QueryRow {
                id,
                score: 1.0,
                metadata: Value::Null,
            })
            .collect())
    }

    async fn vector_search(
        &self,
        dataset: &str,
        query: &[f32],
        k: usize,
    ) -> std::result::Result<Vec<QueryRow>, String> {
        let di = self
            .indexes
            .dataset(dataset)
            .ok_or_else(|| format!("unknown dataset '{dataset}'"))?;
        let metric = di.metric();
        let hits = di.search_vector(query, k)?;
        Ok(hits
            .into_iter()
            .map(|(id, distance)| QueryRow {
                id,
                score: similarity(metric, distance) as f64,
                metadata: Value::Null,
            })
            .collect())
    }

    async fn aggregate(
        &self,
        dataset: &str,
        function: &str,
        field: &str,
    ) -> std::result::Result<Vec<QueryRow>, String> {
        let recs = self
            .storage
            .scan(dataset, usize::MAX, 0)
            .await
            .map_err(|e| e.to_string())?;
        let vals: Vec<f64> = recs
            .iter()
            .filter_map(|r| r.descriptor.numeric_field(field))
            .collect();
        let value = match function {
            "count" => recs.len() as f64,
            "min" => vals.iter().copied().fold(f64::INFINITY, f64::min),
            "max" => vals.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            "sum" => vals.iter().sum(),
            "avg" | "mean" => {
                if vals.is_empty() {
                    0.0
                } else {
                    vals.iter().sum::<f64>() / vals.len() as f64
                }
            }
            other => return Err(format!("unsupported aggregate function '{other}'")),
        };
        Ok(vec![QueryRow {
            id: TensorId::new(),
            score: value,
            metadata: serde_json::json!({"function": function, "field": field, "value": value}),
        }])
    }
}
