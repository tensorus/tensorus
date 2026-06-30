# Configuration

There are two configuration surfaces:

1. **`tensorus.toml`** — the canonical, full configuration template. It documents
   every tunable and is rendered into the Kubernetes `ConfigMap` by the Helm
   chart.
2. **Environment variables** — what the `tensorus-server` binary reads today to
   wire up the REST service.

> Status: the server binary currently consumes the **environment variables**
> below (plus sensible defaults). The richer `tensorus.toml` schema is the target
> surface and is shipped/rendered by deployment; index/AI/tiering tunables there
> map onto the library APIs (`HnswConfig`, `VamanaConfig`, `TieringConfig`,
> `OptimizerConfig`, `LlmRouter`, etc.) when those subsystems are wired into the
> server.

---

## `tensorus.toml`

```toml
[server]
host = "0.0.0.0"
grpc_port = 9090
rest_port = 8080

[storage]
backend = "lance"          # logical backend name; this build uses the segment+WAL engine
data_dir = "./data"
wal_dir = "./data/wal"

[storage.tiering]
hot_max_bytes = 1_073_741_824   # 1 GiB hot-tier budget
warm_path = "./data/warm"
cold_backend = "local"          # local filesystem cold tier (S3/GCS via object_store later)
cold_path = "./data/cold"

[index]
learned_index_epsilon = 64      # PGM epsilon
hnsw_m = 16                     # HNSW M
hnsw_ef_construction = 200
hnsw_ef_search = 100

[index.tensor_contraction]
default_sketch_rank = 8         # Tucker sketch per-mode rank

[ai]
default_strategy = "local_first"   # LLM routing strategy
budget_per_hour_usd = 1.0
max_retries = 3

[ai.providers.mock]
type = "openai_compat"
base_url = "http://localhost:11434/v1"
model = "qwen2.5:7b"

[compute]
gpu_enabled = false

[observability]
log_level = "info"
log_format = "pretty"   # "json" in production
metrics_enabled = true
```

### Section reference

| Section / key | Maps to | Notes |
|---------------|---------|-------|
| `server.host` / `rest_port` | server bind address | REST listener |
| `server.grpc_port` | gRPC listener | optional `grpc` feature |
| `storage.data_dir` / `wal_dir` | `FileStorage::open` | segment + WAL roots |
| `storage.tiering.hot_max_bytes` | `TieringConfig.hot_max_bytes` | hot-tier byte budget |
| `storage.tiering.cold_*` | `LocalColdStore` / object store | cold tier |
| `index.learned_index_epsilon` | `PgmIndex` epsilon (`DEFAULT_EPSILON`) | PGM error bound |
| `index.hnsw_*` | `HnswConfig` | M / ef_construction / ef_search |
| `index.tensor_contraction.default_sketch_rank` | `tucker_sketch(..., rank)` | sketch rank |
| `ai.default_strategy` | `RoutingStrategy` | `local_first`, `cost_optimized`, `latency_optimized`, `fixed` |
| `ai.budget_per_hour_usd` | `LlmRouter` budget | 0 disables enforcement |
| `ai.max_retries` | `complete_structured` retries | structured-output retries |
| `ai.providers.*` | `LlmProvider` instances | provider type + endpoint + model |
| `compute.gpu_enabled` | GPU paths | CPU-only in this build |
| `observability.log_format` | `init_tracing(json)` | `pretty` or `json` |
| `observability.metrics_enabled` | `/metrics` | Prometheus exposition |

---

## Environment variables (server binary)

Read by `tensorus-server` (see [api → server binary](./services/tensorus-api.md#4-server-binary-tensorus-server)):

| Variable | Default | Meaning |
|----------|---------|---------|
| `TENSORUS_DATA_DIR` | `./data` | storage root; WAL is `{dir}/wal` |
| `TENSORUS_API_KEY` | *(unset)* | required API key; **unset disables auth (dev only)** |
| `TENSORUS_HOST` | `0.0.0.0` | bind host |
| `TENSORUS_REST_PORT` | `8080` | bind port |
| `TENSORUS_LOG_JSON` | `false` | `true` → JSON structured logs |
| `RUST_LOG` | `info` | tracing filter (e.g. `tensorus_api=debug,info`) |

### Example

```bash
export TENSORUS_DATA_DIR=/var/lib/tensorus
export TENSORUS_API_KEY=$(openssl rand -hex 16)
export TENSORUS_LOG_JSON=true
export RUST_LOG=info
cargo run -p tensorus-api --bin tensorus-server
```

---

## Programmatic configuration

When embedding the engine as a library, configure the components directly:

```rust
use tensorus_api::{ApiConfig, AppState, TensorService, build_app};
use tensorus_storage::FileStorage;
use std::sync::Arc;

let storage = Arc::new(FileStorage::open("./data", "./data/wal")?);
let app = build_app(AppState::new(
    TensorService::new(storage),
    ApiConfig { api_key: Some("secret".into()), rate_capacity: 1000.0, rate_per_sec: 1000.0 },
));
```

Index and AI components take their own config structs: `HnswConfig`,
`VamanaConfig`, `TieringConfig`, `OptimizerConfig`, and `LlmRouter::new(providers,
strategy, budget_per_hour)`. See each [service reference](./services/).

---

## Kubernetes

The Helm chart renders `tensorus.toml` into a `ConfigMap` and injects the API key
from a `Secret`. See `deploy/README.md` and the chart values in
`deploy/helm/tensorus/values.yaml`.
