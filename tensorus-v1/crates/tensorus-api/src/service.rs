//! `TensorService`: the transport-agnostic business logic shared by the REST and
//! gRPC surfaces. It wraps the storage engine and computes a tensor's descriptor
//! on insert.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tensorus_compute::compute_descriptor;
use tensorus_core::error::{Result, TensorusError};
use tensorus_core::types::{DType, Shape, TensorData, TensorDescriptor, TensorId, TensorRecord};
use tensorus_core::Storage;
use tensorus_storage::FileStorage;

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

/// The core service.
#[derive(Clone)]
pub struct TensorService {
    storage: Arc<FileStorage>,
    started: Instant,
}

impl TensorService {
    pub fn new(storage: Arc<FileStorage>) -> Self {
        TensorService {
            storage,
            started: Instant::now(),
        }
    }

    pub async fn create_dataset(&self, name: &str) -> Result<()> {
        self.storage.create_dataset(name).await
    }

    pub async fn list_datasets(&self) -> Result<Vec<String>> {
        self.storage.list_datasets().await
    }

    /// Insert a `Float32` tensor, computing its descriptor.
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
        Ok((id, descriptor))
    }

    pub async fn get(&self, dataset: &str, id: TensorId) -> Result<TensorRecord> {
        self.storage.get(dataset, id).await
    }

    pub async fn delete(&self, dataset: &str, id: TensorId) -> Result<()> {
        self.storage.delete(dataset, id).await
    }

    pub async fn scan(
        &self,
        dataset: &str,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<TensorRecord>> {
        self.storage.scan(dataset, limit, offset).await
    }

    /// Property search: scan the dataset and filter by the descriptor predicates.
    ///
    /// This is the index-free reference path; the learned indexes in
    /// `tensorus-index` accelerate the same predicates in production.
    #[tracing::instrument(skip(self, q), fields(dataset = %dataset))]
    pub async fn search_by_property(
        &self,
        dataset: &str,
        q: &PropertyQuery,
    ) -> Result<Vec<(TensorId, TensorDescriptor)>> {
        let records = self.storage.scan(dataset, usize::MAX, 0).await?;
        let mut out = Vec::new();
        for r in records {
            let d = &r.descriptor;
            if let Some(min) = q.min_norm {
                if d.frobenius_norm < min {
                    continue;
                }
            }
            if let Some(max) = q.max_norm {
                if d.frobenius_norm > max {
                    continue;
                }
            }
            if let Some(sym) = q.is_symmetric {
                if d.is_symmetric != sym {
                    continue;
                }
            }
            if let Some(pd) = q.is_positive_definite {
                if d.is_positive_definite != pd {
                    continue;
                }
            }
            if let Some(rank) = q.rank {
                if d.rank != Some(rank) {
                    continue;
                }
            }
            if let Some(maxc) = q.max_condition_number {
                match d.condition_number {
                    Some(c) if c <= maxc => {}
                    _ => continue,
                }
            }
            out.push((r.id, r.descriptor));
            if out.len() >= q.limit {
                break;
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
