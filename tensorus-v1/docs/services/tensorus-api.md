# Service: `tensorus-api`

**The network-facing service.** A transport-agnostic `TensorService` wrapped by
an Axum REST surface (auth, rate limiting, metrics) and a runnable server
binary. `#![forbid(unsafe_code)]`.

- **Depends on:** `tensorus-core`, `tensorus-storage`, `tensorus-compute`,
  `axum`, `tokio`, `tracing`, `tracing-subscriber`, `serde`/`serde_json`,
  `parking_lot`.
- **Binary:** `tensorus-server`.

| Module | Contents |
|--------|----------|
| `service` | `TensorService` (core logic) + DTOs |
| `rest` | `build_app`, `AppState`, `ApiConfig`, handlers, middleware |
| `telemetry` | `init_tracing`, `Histogram` |

See the wire-level endpoint contract in [API Reference](../api-reference.md).

---

## 1. `TensorService` (`service`)

The transport-agnostic business logic shared by REST and (optional) gRPC. It
wraps `FileStorage` and computes a descriptor on insert.

```rust
pub struct TensorService { /* … */ }
impl TensorService {
    pub fn new(storage: Arc<FileStorage>) -> Self;

    pub async fn create_dataset(&self, name: &str) -> Result<()>;
    pub async fn list_datasets(&self) -> Result<Vec<String>>;
    pub async fn insert(&self, dataset: &str, data: &[f32], shape: &[u64], metadata: Value)
        -> Result<(TensorId, TensorDescriptor)>;
    pub async fn get(&self, dataset: &str, id: TensorId) -> Result<TensorRecord>;
    pub async fn delete(&self, dataset: &str, id: TensorId) -> Result<()>;
    pub async fn scan(&self, dataset: &str, limit: usize, offset: usize) -> Result<Vec<TensorRecord>>;
    pub async fn search_by_property(&self, dataset: &str, q: &PropertyQuery)
        -> Result<Vec<(TensorId, TensorDescriptor)>>;
    pub fn health(&self) -> HealthInfo;
    pub fn status_code(err: &TensorusError) -> u16;  // HTTP-ish status mapping
}
```

- `insert` builds a `Float32` payload, computes its descriptor via
  `tensorus-compute`, and stores it; it is instrumented with a `tracing` span.
- `search_by_property` scans the dataset and filters by descriptor predicates
  (the index-free reference path; learned indexes accelerate the same predicates
  in production). It too is `tracing`-instrumented.

### DTOs

```rust
pub struct InsertRequest { pub data: Vec<f32>, pub shape: Vec<u64>, pub metadata: serde_json::Value }

pub struct PropertyQuery {
    pub min_norm: Option<f64>,
    pub max_norm: Option<f64>,
    pub is_symmetric: Option<bool>,
    pub is_positive_definite: Option<bool>,
    pub rank: Option<u32>,
    pub max_condition_number: Option<f64>,
    pub limit: usize,            // default 100
}

pub struct HealthInfo { pub status: String, pub version: String, pub uptime_seconds: u64 }
```

---

## 2. REST surface (`rest`)

```rust
pub struct ApiConfig {
    pub api_key: Option<String>, // None disables auth (dev only)
    pub rate_capacity: f64,      // token-bucket burst, default 1000
    pub rate_per_sec: f64,       // sustained refill rate, default 1000
}

pub struct AppState { /* … */ }
impl AppState { pub fn new(service: TensorService, config: ApiConfig) -> Self; }

pub fn build_app(state: AppState) -> axum::Router;
```

`build_app` returns the full router:

| Method & path | Auth | Handler |
|---------------|------|---------|
| `POST /datasets` | ✓ | create dataset |
| `GET /datasets` | ✓ | list datasets |
| `POST /datasets/{ds}/tensors` | ✓ | insert tensor |
| `GET /datasets/{ds}/tensors?limit&offset` | ✓ | scan |
| `GET /datasets/{ds}/tensors/{id}` | ✓ | get tensor |
| `DELETE /datasets/{ds}/tensors/{id}` | ✓ | delete tensor |
| `POST /datasets/{ds}/search/property` | ✓ | property search |
| `GET /health` | — | health probe |
| `GET /metrics` | — | Prometheus metrics |

**Middleware** (applied to the protected routes): a token-bucket rate limiter
(429 when exhausted), then API-key auth (401 if missing/wrong; key read from the
`x-api-key` header or `Authorization: Bearer …`). After the handler runs, the
request latency is recorded into a histogram and a structured `tracing` event is
emitted. `/health` and `/metrics` bypass auth.

Request/response JSON shapes are in [API Reference](../api-reference.md).

---

## 3. Telemetry (`telemetry`)

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
  emits structured JSON logs (filter via `RUST_LOG`). Spans/events from every
  crate flow to whichever subscriber is installed — satisfying "traces appear in
  a collector (or stdout)".
- `/metrics` renders request/error/rate-limit counters plus the latency
  histogram (`tensorus_request_latency_ms_bucket/_sum/_count`).
- **Substitution:** an OpenTelemetry/OTLP exporter is additive via
  `tracing-opentelemetry`; it is omitted here to avoid the heavy OTel tree.

---

## 4. Server binary (`tensorus-server`)

A runnable REST server configured by environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `TENSORUS_DATA_DIR` | `./data` | storage root (WAL under `{dir}/wal`) |
| `TENSORUS_API_KEY` | *(unset)* | required key; **unset disables auth (dev only)** |
| `TENSORUS_HOST` | `0.0.0.0` | bind host |
| `TENSORUS_REST_PORT` | `8080` | bind port |
| `TENSORUS_LOG_JSON` | `false` | `true` → JSON logs |
| `RUST_LOG` | `info` | tracing filter |

```bash
cargo run -p tensorus-api --bin tensorus-server
# or via the container image / Helm chart (see docs/deployment via deploy/README.md)
```

If `TENSORUS_API_KEY` is unset, the server logs a warning that authentication is
disabled.

## 5. gRPC (optional)

The protobuf service is defined at `proto/tensorus.proto` (dataset CRUD, tensor
CRUD, property/similarity/contraction search, NQL `Query`, `RunAgent`, `Health`).
A tonic server mirrors the REST handlers over the same `TensorService`. tonic's
build needs `protoc`, so the gRPC server is gated behind an optional `grpc`
feature (not built by default in this environment); REST is the tested primary.
See [API Reference → gRPC](../api-reference.md#grpc-service).

## Testing

The REST layer is tested in-process with `tower::ServiceExt::oneshot` (no socket
needed): health is public and 200, missing/wrong key is 401, and a full
create → insert → property-search → get → delete → 404 journey passes. `/metrics`
includes the latency histogram after a request.
