//! Cross-crate integration tests exercising full user journeys across the
//! storage, compute, index, search, AI, and API layers.

use std::sync::Arc;
use tempfile::TempDir;

use tensorus_api::{build_app, ApiConfig, AppState, PropertyQuery, TensorService};
use tensorus_core::types::{Shape, TensorId};
use tensorus_core::Storage;
use tensorus_storage::{
    compress_with, decompress, Codec, FileStorage, LocalColdStore, Policy, TieredStore,
    TieringConfig,
};

fn storage(tmp: &TempDir) -> Arc<FileStorage> {
    Arc::new(FileStorage::open(tmp.path().join("data"), tmp.path().join("wal")).unwrap())
}

// --- storage + compute -------------------------------------------------------

#[tokio::test]
async fn journey_storage_crud_with_descriptor() {
    let tmp = TempDir::new().unwrap();
    let svc = TensorService::new(storage(&tmp));
    svc.create_dataset("weights").await.unwrap();
    // 2x2 identity is symmetric & positive-definite.
    let (id, desc) = svc
        .insert("weights", &[1.0, 0.0, 0.0, 1.0], &[2, 2], serde_json::json!({}))
        .await
        .unwrap();
    assert!(desc.is_symmetric && desc.is_positive_definite);
    let rec = svc.get("weights", id).await.unwrap();
    assert_eq!(rec.descriptor.shape, Shape::new(vec![2, 2]));
    assert_eq!(svc.scan("weights", 10, 0).await.unwrap().len(), 1);
}

#[tokio::test]
async fn journey_property_search() {
    let tmp = TempDir::new().unwrap();
    let svc = TensorService::new(storage(&tmp));
    svc.create_dataset("d").await.unwrap();
    // Symmetric (norm ~2.83) and a non-symmetric matrix.
    svc.insert("d", &[2.0, 0.0, 0.0, 2.0], &[2, 2], serde_json::json!({}))
        .await
        .unwrap();
    svc.insert("d", &[1.0, 2.0, 3.0, 4.0], &[2, 2], serde_json::json!({}))
        .await
        .unwrap();
    let q = PropertyQuery {
        is_symmetric: Some(true),
        ..Default::default()
    };
    let res = svc.search_by_property("d", &q).await.unwrap();
    assert_eq!(res.len(), 1);
    assert!(res[0].1.is_symmetric);
}

// --- REST API ----------------------------------------------------------------

#[tokio::test]
async fn journey_rest_end_to_end() {
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    let tmp = TempDir::new().unwrap();
    let svc = TensorService::new(storage(&tmp));
    let app = build_app(AppState::new(
        svc,
        ApiConfig {
            api_key: Some("k".into()),
            ..Default::default()
        },
    ));

    let send = |app: axum::Router, method: &str, uri: String, body: Option<serde_json::Value>| {
        let b = Request::builder().method(method).uri(uri).header("x-api-key", "k");
        let req = match body {
            Some(j) => b
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&j).unwrap()))
                .unwrap(),
            None => b.body(Body::empty()).unwrap(),
        };
        async move { app.oneshot(req).await.unwrap() }
    };

    assert_eq!(
        send(app.clone(), "POST", "/datasets".into(), Some(serde_json::json!({"name":"w"})))
            .await
            .status(),
        StatusCode::OK
    );
    let resp = send(
        app.clone(),
        "POST",
        "/datasets/w/tensors".into(),
        Some(serde_json::json!({"data":[1.0,0.0,0.0,1.0],"shape":[2,2]})),
    )
    .await;
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let j: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let id = j["tensor_id"].as_str().unwrap().to_string();

    // Property search finds the symmetric tensor.
    let resp = send(
        app.clone(),
        "POST",
        "/datasets/w/search/property".into(),
        Some(serde_json::json!({"is_symmetric": true})),
    )
    .await;
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let arr: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(arr.as_array().unwrap().len(), 1);

    // Get round-trips data.
    let resp = send(app.clone(), "GET", format!("/datasets/w/tensors/{id}"), None).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn journey_rest_auth_required() {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;
    let tmp = TempDir::new().unwrap();
    let svc = TensorService::new(storage(&tmp));
    let app = build_app(AppState::new(
        svc,
        ApiConfig {
            api_key: Some("secret".into()),
            ..Default::default()
        },
    ));
    let resp = app
        .oneshot(Request::builder().method("GET").uri("/datasets").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

// --- indexes -----------------------------------------------------------------

#[test]
fn journey_pgm_range() {
    use tensorus_index::PgmIndex;
    let entries: Vec<(f64, u64)> = (0..1000).map(|i| (i as f64, i as u64)).collect();
    let idx = PgmIndex::build(entries);
    assert_eq!(idx.range_count(100.0, 199.0), 100);
}

#[test]
fn journey_alex_insert_lookup() {
    use tensorus_index::AlexIndex;
    let idx = AlexIndex::new();
    for i in 0..2000u64 {
        idx.insert(i as f64, i);
    }
    assert_eq!(idx.lookup(1234.0), vec![1234]);
}

#[test]
fn journey_metadata_search() {
    use tensorus_index::{Document, MetadataIndex, MetadataQuery};
    let idx = MetadataIndex::new();
    let mut fields = serde_json::Map::new();
    fields.insert("name".into(), serde_json::json!("attention weight matrix"));
    idx.insert(Document::new(TensorId::new(), "m", fields));
    assert_eq!(idx.search(&MetadataQuery::text("attention")).len(), 1);
}

#[test]
fn journey_hnsw_nearest() {
    use tensorus_index::{Hnsw, Metric};
    let h = Hnsw::with_metric(Metric::L2);
    let near = TensorId::new();
    h.insert(near, &[0.0, 0.0]);
    h.insert(TensorId::new(), &[9.0, 9.0]);
    let res = h.search(&[0.1, 0.1], 1);
    assert_eq!(res[0].0, near);
}

#[test]
fn journey_diskann_build_search() {
    use tensorus_index::{DiskAnnIndex, VamanaConfig};
    let tmp = TempDir::new().unwrap();
    let vectors = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![5.0, 5.0]];
    let ids: Vec<TensorId> = (0..3).map(|_| TensorId::new()).collect();
    let idx = DiskAnnIndex::build(
        vectors,
        ids.clone(),
        VamanaConfig {
            degree: 2,
            ..Default::default()
        },
        tmp.path().join("v.idx"),
    )
    .unwrap();
    let res = idx.search(&[0.0, 0.0], 1).unwrap();
    assert_eq!(res[0].0, ids[0]);
}

// --- search ------------------------------------------------------------------

#[test]
fn journey_contraction_similarity() {
    use tensorus_search::contraction_similarity;
    let a = vec![1.0f32, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0];
    let shape = [2u64, 2, 2];
    assert!((contraction_similarity(&a, &shape, &a, &shape, 4) - 1.0).abs() < 1e-6);
}

// --- compression -------------------------------------------------------------

#[test]
fn journey_compression_roundtrip_all_codecs() {
    let dense: Vec<f32> = (0..256).map(|i| (i % 17) as f32 * 0.5).collect();
    for codec in [Codec::Sq8, Codec::Deflate] {
        let c = compress_with(&dense, &[256], codec).unwrap();
        let back = decompress(&c).unwrap();
        assert_eq!(back.len(), dense.len());
    }
    // Sparse via COO is lossless.
    let mut sparse = vec![0.0f32; 1000];
    sparse[10] = 1.0;
    sparse[900] = 2.0;
    let c = compress_with(&sparse, &[1000], Codec::Coo).unwrap();
    assert_eq!(decompress(&c).unwrap(), sparse);
}

// --- tiering -----------------------------------------------------------------

#[test]
fn journey_tiering_no_data_loss() {
    let tmp = TempDir::new().unwrap();
    let cold = LocalColdStore::new(tmp.path().join("cold")).unwrap();
    let store = TieredStore::new(
        cold,
        TieringConfig {
            hot_max_bytes: 32,
            policy: Policy::Lru,
        },
    );
    let ids: Vec<TensorId> = (0..50).map(|_| TensorId::new()).collect();
    for (i, id) in ids.iter().enumerate() {
        store.put(*id, vec![i as u8; 8]).unwrap();
    }
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(store.get(*id).unwrap(), vec![i as u8; 8]);
    }
}

// --- AI: router, NQL over real storage, agent, optimizer ---------------------

#[tokio::test]
async fn journey_llm_router_fallback() {
    use tensorus_ai::{InferenceConfig, LlmProvider, LlmRouter, MockProvider, RoutingStrategy};
    let providers: Vec<Arc<dyn LlmProvider>> = vec![
        Arc::new(MockProvider::always_fail("a")),
        Arc::new(MockProvider::always("b", "ok")),
    ];
    let router = LlmRouter::new(providers, RoutingStrategy::LocalFirst, 0.0);
    let out = router.complete(&[], &InferenceConfig::default()).await.unwrap();
    assert_eq!(out.text, "ok");
}

/// NQL parsed by a (mock) LLM and executed against the REAL storage engine.
#[tokio::test]
async fn journey_nql_over_real_storage() {
    use async_trait::async_trait;
    use tensorus_ai::{
        Cmp, LlmRouter, MockProvider, Nql, Predicate, QueryContext, QueryRow, RoutingStrategy,
    };

    struct StorageCtx {
        storage: Arc<FileStorage>,
    }
    #[async_trait]
    impl QueryContext for StorageCtx {
        async fn scan(&self, ds: &str, limit: usize) -> Result<Vec<QueryRow>, String> {
            let recs = self.storage.scan(ds, limit, 0).await.map_err(|e| e.to_string())?;
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
            ds: &str,
            preds: &[Predicate],
            limit: usize,
        ) -> Result<Vec<QueryRow>, String> {
            let recs = self.storage.scan(ds, usize::MAX, 0).await.map_err(|e| e.to_string())?;
            let pass = |d: &tensorus_core::types::TensorDescriptor| {
                preds.iter().all(|p| match (&p.field[..], p.value.as_bool(), p.value.as_f64()) {
                    ("is_symmetric", Some(b), _) => d.is_symmetric == b,
                    ("frobenius_norm", _, Some(v)) => match p.cmp {
                        Cmp::Gt => d.frobenius_norm > v,
                        Cmp::Ge => d.frobenius_norm >= v,
                        Cmp::Lt => d.frobenius_norm < v,
                        Cmp::Le => d.frobenius_norm <= v,
                        Cmp::Eq => (d.frobenius_norm - v).abs() < 1e-9,
                    },
                    _ => false,
                })
            };
            Ok(recs
                .into_iter()
                .filter(|r| pass(&r.descriptor))
                .take(limit)
                .map(|r| QueryRow {
                    id: r.id,
                    score: 1.0,
                    metadata: r.metadata,
                })
                .collect())
        }
        async fn vector_search(&self, _: &str, _: &[f32], _: usize) -> Result<Vec<QueryRow>, String> {
            Ok(vec![])
        }
        async fn aggregate(&self, _: &str, _: &str, _: &str) -> Result<Vec<QueryRow>, String> {
            Ok(vec![])
        }
    }

    let tmp = TempDir::new().unwrap();
    let store = storage(&tmp);
    let svc = TensorService::new(store.clone());
    svc.create_dataset("layers").await.unwrap();
    // Symmetric matrix with norm ~5.83 and a non-symmetric one.
    svc.insert("layers", &[3.0, 1.0, 1.0, 4.0], &[2, 2], serde_json::json!({}))
        .await
        .unwrap();
    svc.insert("layers", &[0.1, 0.0, 0.0, 0.1], &[2, 2], serde_json::json!({}))
        .await
        .unwrap();

    let provider = MockProvider::scripted(
        "mock",
        vec![Ok(r#"{"op":"index_lookup","dataset":"layers","predicates":[{"field":"is_symmetric","cmp":"eq","value":true},{"field":"frobenius_norm","cmp":"gt","value":2.0}]}"#.to_string())],
    );
    let router = LlmRouter::new(
        vec![Arc::new(provider)],
        RoutingStrategy::Fixed { provider_index: 0 },
        0.0,
    );
    let nql = Nql::new(router);
    let ctx = StorageCtx { storage: store };
    let result = nql
        .query("find symmetric matrices with norm > 2", &ctx, 2)
        .await
        .unwrap();
    // Only the symmetric norm~5.83 matrix qualifies (the other has norm 0.2).
    assert_eq!(result.rows.len(), 1);
}

#[tokio::test]
async fn journey_react_agent_single_tool() {
    use tensorus_ai::{
        AgentConfig, AgentOutcome, FnTool, LlmRouter, MockProvider, ReActAgent, RoutingStrategy,
        Tool, ToolRegistry,
    };
    let mut reg = ToolRegistry::new();
    reg.register(Arc::new(FnTool::new(
        "tensor_compute",
        "compute",
        serde_json::json!({}),
        |_| Ok("rank=2".to_string()),
    )) as Arc<dyn Tool>);
    let provider = MockProvider::scripted(
        "mock",
        vec![
            Ok(r#"{"thought":"compute it","action":{"tool":"tensor_compute","args":{}}}"#.to_string()),
            Ok(r#"{"thought":"done","final_answer":"the rank is 2"}"#.to_string()),
        ],
    );
    let router = Arc::new(LlmRouter::new(
        vec![Arc::new(provider)],
        RoutingStrategy::Fixed { provider_index: 0 },
        0.0,
    ));
    let agent = ReActAgent::new(router, reg, AgentConfig::default());
    match agent.run("compute the rank").await {
        AgentOutcome::Success { answer, .. } => assert!(answer.contains("rank")),
        other => panic!("expected success, got {other:?}"),
    }
}

#[test]
fn journey_optimizer_suggests_index() {
    use std::collections::HashSet;
    use tensorus_ai::{Action, AutoOptimizer, OptimizerConfig};
    let mut opt = AutoOptimizer::new(OptimizerConfig::default());
    for _ in 0..30 {
        opt.observe("frobenius_norm", 42.0);
    }
    let action = opt.suggest(&HashSet::new()).unwrap();
    assert_eq!(action, Action::CreateIndex("frobenius_norm".to_string()));
}
