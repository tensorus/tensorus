//! PyO3 bindings: the `tensorus` native extension module.
//!
//! Exposes a `Tensorus` class backed by the Rust storage + compute engines.
//! numpy arrays are read zero-copy into Rust slices on insert and rebuilt on
//! get. A small Tokio runtime bridges the async storage API into Python's
//! synchronous calls.
// `useless_conversion` fires inside pyo3's `#[pymethods]` macro expansion (it
// inserts `.into()` on the error type); allow it for this generated surface.
#![allow(clippy::useless_conversion)]

use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::str::FromStr;
use std::sync::Arc;
use tensorus_compute::compute_descriptor;
use tensorus_core::types::{DType, Shape, TensorData, TensorId};
use tensorus_core::Storage;
use tensorus_storage::FileStorage;

fn err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Handle to a Tensorus instance.
#[pyclass]
pub struct Tensorus {
    storage: Arc<FileStorage>,
    rt: Arc<tokio::runtime::Runtime>,
}

impl Tensorus {
    fn open(data_dir: std::path::PathBuf) -> PyResult<Self> {
        let wal = data_dir.join("wal");
        let storage = FileStorage::open(&data_dir, &wal).map_err(err)?;
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(err)?;
        Ok(Tensorus {
            storage: Arc::new(storage),
            rt: Arc::new(rt),
        })
    }
}

#[pymethods]
impl Tensorus {
    /// Open an ephemeral instance in a fresh temp directory.
    #[staticmethod]
    fn memory() -> PyResult<Self> {
        let dir = std::env::temp_dir().join(format!("tensorus-{}", TensorId::new()));
        Self::open(dir)
    }

    /// Open a local instance rooted at `data_dir` (defaults to `./data`).
    #[staticmethod]
    #[pyo3(signature = (data_dir=None))]
    fn local(data_dir: Option<String>) -> PyResult<Self> {
        Self::open(std::path::PathBuf::from(
            data_dir.unwrap_or_else(|| "./data".to_string()),
        ))
    }

    /// Connect to a Tensorus instance. A `file://` or bare path opens locally;
    /// remote `grpc://` URIs require the gRPC client (not yet wired here).
    #[staticmethod]
    fn connect(uri: &str) -> PyResult<Self> {
        let path = uri.strip_prefix("file://").unwrap_or(uri);
        if path.starts_with("grpc://") || path.starts_with("http") {
            return Err(PyRuntimeError::new_err(
                "remote connections require the gRPC client (not available in this build)",
            ));
        }
        Self::open(std::path::PathBuf::from(path))
    }

    /// Tensorus version string.
    #[staticmethod]
    fn version() -> &'static str {
        tensorus_core::VERSION
    }

    /// Create a dataset.
    fn create_dataset(&self, name: &str) -> PyResult<()> {
        self.rt
            .block_on(self.storage.create_dataset(name))
            .map_err(err)
    }

    /// List datasets.
    fn list_datasets(&self) -> PyResult<Vec<String>> {
        self.rt.block_on(self.storage.list_datasets()).map_err(err)
    }

    /// Insert a numpy `float32` array, returning its id. Optional `metadata` is
    /// a JSON string (the high-level SDK serializes a dict for you).
    #[pyo3(signature = (dataset, array, metadata=None))]
    fn insert(
        &self,
        dataset: &str,
        array: PyReadonlyArrayDyn<'_, f32>,
        metadata: Option<String>,
    ) -> PyResult<String> {
        // Logical (C-order) flattening regardless of memory layout.
        let data: Vec<f32> = array.as_array().iter().copied().collect();
        let shape: Vec<u64> = array.shape().iter().map(|&d| d as u64).collect();
        let td = TensorData::from_f32(&data, Shape::new(shape.clone())).map_err(err)?;
        let descriptor = compute_descriptor(&data, &shape, DType::Float32);
        let meta = match metadata {
            Some(s) => serde_json::from_str(&s).map_err(err)?,
            None => serde_json::Value::Null,
        };
        let id = self
            .rt
            .block_on(self.storage.insert(dataset, &td.data, descriptor, meta))
            .map_err(err)?;
        Ok(id.to_string())
    }

    /// Fetch a tensor by id as a numpy `float32` array.
    fn get<'py>(
        &self,
        py: Python<'py>,
        dataset: &str,
        id: &str,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let tid = TensorId::from_str(id).map_err(err)?;
        let rec = self
            .rt
            .block_on(self.storage.get(dataset, tid))
            .map_err(err)?;
        let data: Vec<f32> = rec
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let shape: Vec<usize> = rec.descriptor.shape.0.iter().map(|&d| d as usize).collect();
        let arr = data.into_pyarray_bound(py);
        let reshaped = arr
            .reshape(shape)
            .map_err(|e| PyRuntimeError::new_err(format!("reshape failed: {e}")))?;
        Ok(reshaped)
    }

    /// Delete a tensor.
    fn delete(&self, dataset: &str, id: &str) -> PyResult<()> {
        let tid = TensorId::from_str(id).map_err(err)?;
        self.rt
            .block_on(self.storage.delete(dataset, tid))
            .map_err(err)
    }
}

/// The `tensorus._native` Python extension module.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tensorus>()?;
    m.add("__version__", tensorus_core::VERSION)?;
    Ok(())
}
