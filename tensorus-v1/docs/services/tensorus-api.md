# Service: `tensorus-api`

**The network-facing service.** A transport-agnostic `TensorService` — which
wraps durable storage *and* the in-memory secondary indexes — fronted by an Axum
REST surface (auth, rate limiting, metrics) and a runnable server binary.
`#![forbid(unsafe_code)]`.

- **Depends on:** `tensorus-core`, `tensorus-storage`, `tensorus-compute`,
  `tensorus-index`, `tensorus-search`, `tensorus-ai`, `axum`, `tokio` (incl.
  `net`), `tracing`, `tracing-subscriber`, `serde`/`serde_json`, `parking_lot`,
  `rand` + `sha2` (API-key generation/hashing).
- **Binary:** `tensorus-server`.

| Module | Contents |
|--------|----------|
| `service` | `TensorService` (core logic), `QueryContext`/`ScopedContext`, DTOs |
| `index` | `IndexManager`, `DatasetIndexes`, `PropertyIndex`, metric helpers |
| `tenancy` | `TenantRegistry`, `Principal`/`Scope`/`Role`, `Quota`, hashed keys |
| `rest` | `build_app`, `AppState`, `ApiConfig`, `LlmConfig`, handlers, middleware |
| `tools` | `default_tool_registry` — async agent tools backed by the service |
| `llm_http` | `HttpTransport` — `http://` client implementing `tensorus_ai::HttpClient` |
| `telemetry` | `init_tracing`, `Histogram` |

See the wire-level endpoint contract in [API Reference](../api-reference.md).

---

## 1. `TensorService` (`service`)

The transport-agnostic business logic shared by REST and (optional) gRPC. It
wraps `FileStorage`, computes a descriptor on insert, and maintains the
per-dataset secondary indexes through an `IndexManager`.

```rust
pub struct TensorService { /* storage + Arc<IndexManager> + start time */ }
impl TensorService {
    pub fn new(storage: Arc<FileStorage>) -> Self;                 // in-memory indexes
    pub fn with_index_persistence(storage: Arc<FileStorage>, index_config_path: PathBuf) -> Self;
    pub async fn recover(&self) -> Result<()>;                     // rebuild indexes from storage

    pub async fn create_dataset(&self, name: &str) -> Result<()>;
    pub async fn create_dataset_with_metric(&self, name: &str, metric: &str) -> Result<()>;
    pub async fn list_datasets(&self) -> Result<Vec<String>>;
    pub fn dataset_metric(&self, dataset: &str) -> Metric;

    pub async fn insert(&self, dataset: &str, data: &[f32], shape: &[u64], metadata: Value)
        -> Result<(TensorId, TensorDescriptor)>;
    pub async fn get(&self, dataset: &str, id: TensorId) -> Result<TensorRecord>;
    pub async fn delete(&self, dataset: &str, id: TensorId) -> Result<()>;
    pub async fn scan(&self, dataset: &str, limit: usize, offset: usize) -> Result<Vec<TensorRecord>>;

    pub async fn search_by_property(&self, dataset: &str, q: &PropertyQuery)
        -> Result<Vec<(TensorId, TensorDescriptor)>>;
    pub async fn search_similar(&self, dataset: &str, query: &[f32], k: usize, metric_override: Option<&str>)
        -> Result<Vec<SimilarHit>>;
    pub async fn search_contraction(&self, dataset: &str, data: &[f32], shape: &[u64], k: usize)
        -> Result<Vec<ContractionHit>>;

    pub fn health(&self) -> HealthInfo;
    pub fn status_code(err: &TensorusError) -> u16;  // HTTP-ish status mapping
}
```

- `insert` builds a `Float32` payload, computes its descriptor via
  `tensorus-compute`, stores it durably, then updates the dataset's vector,
  contraction, and property indexes.
- `delete` removes the tensor from storage and from the indexes (HNSW tombstone +
  property removal; the contraction index filters deleted ids at query time).
- `search_by_property` is **index-backed**: numeric predicates resolve through
  per-field ALEX indexes and booleans through posting sets, then every candidate
  is verified exactly. No full storage scan.
- `recover` rebuilds the in-memory indexes by paging through storage; the server
  calls it once at startup.
- `TensorService` also implements `tensorus_ai::QueryContext` (`scan`,
  `property_search`, `vector_search`, `aggregate`) so the NQL engine executes
  plans directly against storage + indexes.

### DTOs

```rust
pub struct InsertRequest { pub data: Vec<f32>, pub shape: Vec<u64>, pub metadata: serde_json::Value }

pub struct PropertyQuery {
    pub min_norm: Option<f64>, pub max_norm: Option<f64>,
    pub is_symmetric: Option<bool>, pub is_positive_definite: Option<bool>,
    pub rank: Option<u32>, pub max_condition_number: Option<f64>,
    pub limit: usize,            // default 100
}

pub struct SimilarHit {                 // vector search
    pub tensor_id: TensorId, pub distance: f32, pub score: f32,
    pub descriptor: TensorDescriptor, pub metadata: serde_json::Value,
}
pub struct ContractionHit {             // structural search
    pub tensor_id: TensorId, pub score: f64,
    pub descriptor: TensorDescriptor, pub metadata: serde_json::Value,
}
pub struct HealthInfo { pub status: String, pub version: String, pub uptime_seconds: u64 }
```

---

## 2. Secondary indexes (`index`)

`IndexManager` owns the per-dataset `DatasetIndexes` and (optionally) persists
each dataset's vector metric to a JSON sidecar so it survives a restart.

```rust
pub struct IndexManager { /* … */ }
impl IndexManager {
    pub fn in_memory(contraction_rank: usize) -> Self;                       // tests
    pub fn with_persistence(contraction_rank: usize, config_path: PathBuf) -> Self;
    pub fn create_dataset(&self, name: &str, metric: Metric);
    pub fn metric_for(&self, dataset: &str) -> Metric;                        // default cosine
    pub fn dataset(&self, name: &str) -> Option<Arc<DatasetIndexes>>;
    pub fn on_insert(&self, dataset: &str, id: TensorId, data: &[f32], shape: &[u64], d: &TensorDescriptor);
    pub fn on_delete(&self, dataset: &str, id: TensorId);
}
```

Each `DatasetIndexes` holds:

| Index | Type | Notes |
|-------|------|-------|
| Vector | `tensorus_index::Hnsw` | metric fixed at creation; dimension adopted from the first indexed tensor; mismatched dims are stored but skipped |
| Contraction | `tensorus_search::ContractionIndex` | lazily created, locked to the first `ndim ≥ 2` shape; deletes handled by live-filtering results |
| Property | `PropertyIndex` | per-numeric-field `AlexIndex` + per-boolean posting sets over a stable ordinal space; descriptors retained for exact verification |

Helpers: `metric_from_str` (`cosine`/`l2`/`euclidean`/`dot`/…), `metric_name`,
and `similarity(metric, distance)` (maps an index distance to a higher-is-better
score).

---

## 3. REST surface (`rest`)

```rust
pub struct ApiConfig {
    pub api_key: Option<String>, // None disables auth (dev only)
    pub rate_capacity: f64,      // token-bucket burst, default 1000
    pub rate_per_sec: f64,       // sustained refill rate, default 1000
}

pub struct LlmConfig {           // enables /query and /agent
    pub providers: Vec<Arc<dyn LlmProvider>>,
    pub strategy: RoutingStrategy,
    pub budget_per_hour: f64,
    pub max_correction: usize,   // NQL self-correction rounds
    pub agent_max_steps: usize,
}

pub struct AppState { /* … */ }
impl AppState {
    pub fn new(service: TensorService, config: ApiConfig) -> Self;
    pub fn with_llm(self, llm: LlmConfig) -> Self;   // opt-in LLM endpoints
}

pub fn build_app(state: AppState) -> axum::Router;
```

`build_app` returns the full router:

| Method & path | Auth | Handler |
|---------------|------|---------|
| `POST /datasets` | ✓ | create dataset (optional `metric`) |
| `GET /datasets` | ✓ | list datasets |
| `POST /datasets/{ds}/tensors` | ✓ | insert tensor |
| `GET /datasets/{ds}/tensors?limit&offset` | ✓ | scan |
| `GET /datasets/{ds}/tensors/{id}` | ✓ | get tensor |
| `DELETE /datasets/{ds}/tensors/{id}` | ✓ | delete tensor |
| `POST /datasets/{ds}/search/property` | ✓ | index-backed property search |
| `POST /datasets/{ds}/search/similar` | ✓ | HNSW vector search |
| `POST /datasets/{ds}/search/contraction` | ✓ | contraction similarity search |
| `POST /query` | ✓ | NQL natural-language query (needs LLM) |
| `POST /agent` | ✓ | ReAct agent run (needs LLM) |
| `POST/GET /admin/tenants` | ✓ (system) | create / list tenants |
| `POST/GET /admin/tenants/{t}/keys` | ✓ (system or tenant admin) | issue / list API keys |
| `DELETE /admin/keys/{id}` | ✓ (system or tenant admin) | revoke a key |
| `GET /admin/tenants/{t}/usage` | ✓ (system or tenant admin) | usage vs quota |
| `POST /admin/snapshot` | ✓ (system) | back up all data |
| `GET /health` | — | health probe |
| `GET /metrics` | — | Prometheus metrics |

**Middleware** (protected routes): a token-bucket rate limiter (429 when
exhausted), then authentication, which resolves the caller to a
[`Principal`](#multi-tenancy--rbac) stored in the request extensions. After the
handler runs, latency is recorded into a histogram and a structured `tracing`
event is emitted. `/query` and `/agent` return **503** when no `LlmConfig` is
attached. `/health` and `/metrics` bypass auth.

Request/response JSON shapes are in [API Reference](../api-reference.md).

### Multi-tenancy & RBAC

`AppState::with_tenancy(registry, admin_key)` switches authentication to the
`tenancy` module's `TenantRegistry`. Each request becomes a `Principal { scope,
role }`:

- `Scope::Unscoped` — legacy single-key mode (full access, no dataset prefixing).
- `Scope::System` — the bootstrap `admin_key`: control plane only (`/admin/*`),
  no tenant data access.
- `Scope::Tenant(t)` — a tenant key; data operations are transparently scoped to
  the storage namespace `{t}.{dataset}` (handlers call `resolve_ds`).

Roles (`read_only` < `read_write` < `admin`) gate writes and key management.
`TenantRegistry` persists tenants and **SHA-256-hashed** keys to
`{data_dir}/control/registry.json`; plaintext keys are shown once at issuance.
Per-tenant `Quota` (max datasets / max tensors; `0` = unlimited) is enforced on
create/insert via in-memory counters that are seeded from storage at startup.
The `ScopedContext` decorator applies the same `{t}.{dataset}` prefixing to NQL
queries and agent tools so the LLM surface is tenant-isolated too.

---

## 4. Agent tools (`tools`) & LLM transport (`llm_http`)

- `default_tool_registry(service)` builds a `tensorus_ai::ToolRegistry` of async
  tools — `list_datasets`, `tensor_scan`, `tensor_search`, `tensor_aggregate`,
  `tensor_get` — each calling the async service. (These implement the async
  `Tool` trait directly; `FnTool` is synchronous and can't reach async storage.)
- `HttpTransport` implements `tensorus_ai::HttpClient` over `tokio::net::TcpStream`
  with no external HTTP crate: it writes an HTTP/1.1 `POST` (`Connection: close`)
  and parses the response (status line, `Content-Length` **and**
  `Transfer-Encoding: chunked` bodies). It is **`http://` only** — a pure-Rust
  HTTPS stack needs a C/assembly crypto backend, which this build avoids, so
  cloud HTTPS providers require a TLS-enabled build. Local Ollama/vLLM over
  `http://localhost` is the primary path.

---

## 5. Telemetry (`telemetry`)

```rust
pub fn init_tracing(json: bool);   // install a tracing subscriber (idempotent)

pub struct Histogram { /* … */ }
impl Histogram {
    pub fn new_default() -> Self;                 // request-latency ms buckets
    pub fn with_bounds(bounds: Vec<f64>) -> Self;
    pub fn observe(&self, ms: f64);
    pub fn count(&self) -> u64;
    pub fn sum_ms(&self) -> f64;
    pub fn render(&self, name: &str) -> String;   // Prometheus exposition format
}
```

- `init_tracing(false)` installs a pretty stdout subscriber; `init_tracing(true)`
  emits structured JSON logs (filter via `RUST_LOG`).
- `/metrics` renders request/error/rate-limit counters plus the latency
  histogram (`tensorus_request_latency_ms_bucket/_sum/_count`).
- **Substitution:** an OpenTelemetry/OTLP exporter is additive via
  `tracing-opentelemetry`; it is omitted here to avoid the heavy OTel tree.

---

## 6. Server binary (`tensorus-server`)

A runnable REST server configured by environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `TENSORUS_DATA_DIR` | `./data` | storage root (WAL under `{dir}/wal`, index metric sidecar under `{dir}/indexes`, control plane under `{dir}/control`) |
| `TENSORUS_API_KEY` | *(unset)* | legacy single key; **unset disables auth (dev only)** |
| `TENSORUS_ADMIN_KEY` | *(unset)* | enables **multi-tenancy**; bootstrap system/control-plane key |
| `TENSORUS_HOST` | `0.0.0.0` | bind host |
| `TENSORUS_REST_PORT` | `8080` | bind port |
| `TENSORUS_LOG_JSON` | `false` | `true` → JSON logs |
| `RUST_LOG` | `info` | tracing filter |
| `TENSORUS_LLM_BASE_URL` | *(unset)* | OpenAI-compatible base URL (HTTP) → enables `/query`, `/agent` |
| `TENSORUS_LLM_MODEL` | *(unset)* | model name |
| `TENSORUS_LLM_API_KEY` | *(unset)* | optional bearer token |

At startup the server opens storage, constructs the service with metric
persistence, and calls `recover()` to rebuild the in-memory indexes before
serving. If `TENSORUS_LLM_BASE_URL`/`MODEL` are set, it attaches an
`OpenAiCompatProvider` over `HttpTransport` (LocalFirst routing).

```bash
cargo run -p tensorus-api --bin tensorus-server
# or via the container image / Helm chart (see deploy/README.md)
```

## 7. gRPC (optional)

The protobuf service is defined at `proto/tensorus.proto` (dataset CRUD, tensor
CRUD, property/similarity/contraction search, NQL `Query`, `RunAgent`, `Health`).
A tonic server would mirror the REST handlers over the same `TensorService`.
tonic's build needs `protoc`, so the gRPC server is gated behind an optional
`grpc` feature (not built by default in this environment); REST is the tested
primary and now exposes the full search/NQL/agent surface.
See [API Reference → gRPC](../api-reference.md#grpc-service).

## Testing

The REST layer is tested in-process with `tower::ServiceExt::oneshot` (no socket
needed). Beyond CRUD/auth/metrics, the integration suite covers: vector search
ranking and dimension-mismatch 400s; contraction search self-match; index-backed
property range queries; index recovery after a simulated restart; NQL `/query`
and ReAct `/agent` driven by a mock LLM (plus the 503 when unconfigured); and an
`HttpTransport` round-trip against a local one-shot TCP server. `index` and
`llm_http` add unit tests for the property/vector/contraction indexes and the
HTTP response parser.
