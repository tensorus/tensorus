# Tensorus v1.0

An agentic, **tensor-native database**: tensors are first-class citizens,
queryable by their mathematical properties (norm, rank, eigenvalues, symmetry,
sparsity) and by structural similarity, with a built-in AI layer (model-agnostic
LLM router, neural query language, and ReAct agent).

This is the ground-up Rust rewrite of the v0.1 prototype.

## Workspace layout

| Crate | Responsibility |
|-------|----------------|
| `tensorus-core` | Types (`TensorId`, `DType`, `Shape`, `TensorDescriptor`), traits, errors, Arrow bridge |
| `tensorus-storage` | Crash-safe segment+WAL storage, adaptive compression, hot/warm/cold tiering |
| `tensorus-compute` | `TensorDescriptor` computation (norms, rank, eigenvalues, properties) |
| `tensorus-index` | Learned indexes (PGM, ALEX), HNSW + DiskANN/Vamana vector indexes, metadata index |
| `tensorus-search` | Tensor **contraction similarity** (Tucker sketch + Grassmannian distance) |
| `tensorus-ai` | LLM router, Neural Query Language (NQL), ReAct agent, auto-optimizer |
| `tensorus-api` | Axum REST API (auth, rate limiting, metrics) + server binary |
| `tensorus-python` | PyO3 bindings + the `tensorus` Python SDK |

## Documentation

Full documentation lives in [`docs/`](./docs/):

- [Documentation index](./docs/README.md)
- [Architecture](./docs/architecture.md) · [Data model](./docs/data-model.md) · [API reference](./docs/api-reference.md) · [Configuration](./docs/configuration.md)
- Per-service references: [core](./docs/services/tensorus-core.md) · [compute](./docs/services/tensorus-compute.md) · [storage](./docs/services/tensorus-storage.md) · [index](./docs/services/tensorus-index.md) · [search](./docs/services/tensorus-search.md) · [ai](./docs/services/tensorus-ai.md) · [api](./docs/services/tensorus-api.md) · [python](./docs/services/tensorus-python.md)
- Generate API-level (rustdoc) docs with `cargo doc --workspace --no-deps --open`.

## Quickstart (Rust)

```rust
use std::sync::Arc;
use tensorus_api::{TensorService};
use tensorus_storage::FileStorage;

# async fn demo() -> Result<(), Box<dyn std::error::Error>> {
let storage = Arc::new(FileStorage::open("./data", "./data/wal")?);
let svc = TensorService::new(storage);
svc.create_dataset("weights").await?;

// Insert a 2x2 identity matrix; its descriptor is computed automatically.
let (id, descriptor) = svc
    .insert("weights", &[1.0, 0.0, 0.0, 1.0], &[2, 2], serde_json::json!({"name": "I2"}))
    .await?;
assert!(descriptor.is_symmetric && descriptor.is_positive_definite);

let record = svc.get("weights", id).await?;
println!("frobenius norm = {}", record.descriptor.frobenius_norm);
# Ok(()) }
```

## Quickstart (Python)

```python
import numpy as np
from tensorus import Tensorus

ts = Tensorus.memory()
ts.create_dataset("weights")
tid = ts.insert("weights", np.eye(3, dtype=np.float32), metadata={"name": "I3"})
print(ts.get("weights", tid))
```

Build the extension with `maturin develop --features python` from `python/`
(see `python/README.md` for the toolchain note).

## Run the server

```bash
cargo run -p tensorus-api --bin tensorus-server
# then, in another shell:
curl localhost:8080/health
curl -H "x-api-key: $TENSORUS_API_KEY" \
  -d '{"data":[1,0,0,1],"shape":[2,2]}' -H 'content-type: application/json' \
  localhost:8080/datasets/weights/tensors
```

REST endpoints: `POST/GET /datasets` (optional per-dataset vector `metric`),
`POST/GET/DELETE /datasets/{ds}/tensors[/{id}]`,
`POST /datasets/{ds}/search/property` (index-backed),
`POST /datasets/{ds}/search/similar` (HNSW vector search),
`POST /datasets/{ds}/search/contraction` (structural similarity),
`POST /query` (NQL) and `POST /agent` (ReAct) when an LLM is configured,
`GET /health`, `GET /metrics`. With `TENSORUS_ADMIN_KEY` set, **multi-tenancy** is
enabled: per-tenant scoped API keys with `read_only`/`read_write`/`admin` roles,
isolated dataset namespaces, quotas, and a `/admin/*` control plane
(create tenants, issue/revoke keys, usage, delete-tenant, `/admin/snapshot`
backup and `/admin/restore`). HNSW vector graphs are **persisted** (loaded on
restart instead of rebuilt), and a single-leader **replication** change-log
(`/replication/changes`) lets followers mirror a leader for read-scaling/standby.
See the [API reference](./docs/api-reference.md).

## Build, test, lint

```bash
cargo build --workspace
cargo test  --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo bench -p tensorus-api          # Criterion micro-benchmarks
cargo doc   --workspace --no-deps    # API docs from doc comments
```

## Deployment

A `Dockerfile` and a Helm chart (`deploy/helm/tensorus`) are provided; see
`deploy/README.md`.

## Performance

See [`BENCHMARKS.md`](./BENCHMARKS.md) for measured numbers versus the plan's
targets, including where this CPU-only, pure-Rust build meets or trails them.

## Implementation notes

Several components document pragmatic, interface-preserving substitutions made to
fit a pure-Rust, CPU-only build environment (e.g. a custom segment+WAL store in
place of Lance, an in-house inverted index in place of Tantivy, SQ8 in place of
Product Quantization, `std::fs` reads in place of `io_uring`). Each is documented
at the top of the relevant module, and the trait boundaries are unchanged so the
specified backends can be dropped in. GPU acceleration points are marked with
`// TODO: GPU acceleration`.
