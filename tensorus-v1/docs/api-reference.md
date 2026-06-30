# API Reference

The network surface is provided by [`tensorus-api`](./services/tensorus-api.md).
The **REST** API is the tested primary; a **gRPC** service is defined and gated
behind an optional feature.

- Base URL: `http://{host}:{rest_port}` (default `0.0.0.0:8080`).
- Content type: `application/json` for all request/response bodies.
- All values are `Float32` tensors; `data` is a flat row-major array, `shape` its
  dimensions.

---

## Authentication

Protected endpoints require an API key (when the server is configured with one),
supplied via either header:

```
x-api-key: <key>
# or
Authorization: Bearer <key>
```

- Missing/incorrect key → **401 Unauthorized**.
- `/health` and `/metrics` are **public** (no key required).
- If the server is started without a key (`TENSORUS_API_KEY` unset), auth is
  disabled (development only) and a warning is logged.

## Rate limiting

A token bucket (defaults: capacity 1000, refill 1000/s) guards protected routes.
When exhausted → **429 Too Many Requests**.

## Error mapping

Errors return `{"error": "<message>"}` with status:

| `TensorusError` | HTTP |
|-----------------|------|
| `NotFound` | 404 |
| `AlreadyExists` | 409 |
| `InvalidArgument`, `DimensionMismatch` | 400 |
| everything else | 500 |

---

## Endpoints

### `POST /datasets` — create a dataset

Request (`metric` optional; the vector index uses it):
```json
{ "name": "weights", "metric": "cosine" }
```
`metric` is one of `cosine` (default), `l2`/`euclidean`, or `dot`. It is fixed
for the dataset's vector index (Pinecone/Weaviate-style) and persisted across
restarts.

Response `200`:
```json
{ "created": true, "name": "weights", "metric": "cosine" }
```
Idempotent (creating an existing dataset succeeds).

### `GET /datasets` — list datasets

Response `200`:
```json
["bert", "resnet50", "weights"]
```

### `POST /datasets/{ds}/tensors` — insert a tensor

Request:
```json
{ "data": [1.0, 0.0, 0.0, 1.0], "shape": [2, 2], "metadata": { "name": "I2" } }
```
Response `200`:
```json
{
  "tensor_id": "019f17e3-2662-73d2-b8fa-1339a86295d2",
  "descriptor": {
    "tensor_id": "019f17e3-2662-73d2-b8fa-1339a86295d2",
    "shape": [2, 2],
    "dtype": "Float32",
    "num_elements": 4,
    "frobenius_norm": 1.4142135623730951,
    "l1_norm": 2.0,
    "l_inf_norm": 1.0,
    "mean": 0.5,
    "std_dev": 0.5773502691896257,
    "sparsity": 0.5,
    "rank": 2,
    "trace": 2.0,
    "determinant": 1.0,
    "condition_number": 1.0,
    "max_eigenvalue": 1.0,
    "min_eigenvalue": 1.0,
    "is_symmetric": true,
    "is_positive_definite": true,
    "is_orthogonal": true,
    "is_sparse": false,
    "is_square": true,
    "is_diagonal": true
  }
}
```
The descriptor is computed server-side on insert. `400` if the data length does
not match `shape`.

> Note: `is_sparse` is `sparsity > 0.5`; for the identity above `sparsity` is
> exactly `0.5`, so `is_sparse` is `false`. `Option` fields (`rank`, `trace`,
> `determinant`, `condition_number`, `max_eigenvalue`, `min_eigenvalue`) are
> `null` for non-2-D or singular cases.

### `GET /datasets/{ds}/tensors?limit&offset` — scan

Query params: `limit` (default 100), `offset` (default 0). Returns records in
insertion order:
```json
[
  { "tensor_id": "…", "descriptor": { … }, "metadata": { "name": "I2" } }
]
```

### `GET /datasets/{ds}/tensors/{id}` — get a tensor

Response `200`:
```json
{
  "tensor_id": "…",
  "shape": [2, 2],
  "data": [1.0, 0.0, 0.0, 1.0],
  "descriptor": { … },
  "metadata": { "name": "I2" }
}
```
`404` if the dataset or id is unknown; `400` if the id is malformed.

### `DELETE /datasets/{ds}/tensors/{id}` — delete a tensor

Response `200`:
```json
{ "deleted": true }
```
Idempotent.

### `POST /datasets/{ds}/search/property` — property search

Request (all fields optional; `limit` defaults to 100):
```json
{
  "min_norm": 5.0,
  "max_norm": 10.0,
  "is_symmetric": true,
  "is_positive_definite": true,
  "rank": 3,
  "max_condition_number": 100.0,
  "limit": 50
}
```
Response `200` — matching records (descriptor + metadata):
```json
[
  { "tensor_id": "…", "descriptor": { … }, "metadata": null }
]
```
Predicates combine with AND. `min_norm`/`max_norm` bound `frobenius_norm`. This
is **index-backed**: numeric predicates resolve through per-field learned (ALEX)
indexes and booleans through posting sets, with an exact verification pass — not
a full scan.

### `POST /datasets/{ds}/search/similar` — vector similarity search

Approximate nearest-neighbour search over the flattened tensor payload, served
by the per-dataset HNSW index.

Request (`vector` is the query embedding; `data` is accepted as an alias; `k`
defaults to 10):
```json
{ "vector": [0.12, 0.04, -0.98, …], "k": 10 }
```
Response `200` — ranked hits (best first):
```json
[
  {
    "tensor_id": "…",
    "distance": 0.0193,
    "score": 0.9807,
    "descriptor": { … },
    "metadata": { "name": "doc-42" }
  }
]
```
- `distance` is the raw index distance (smaller is closer); `score` is the
  metric-appropriate similarity (larger is more similar): `cosine → 1 − distance`,
  `l2 → 1/(1+distance)`, `dot → dot product`.
- The query dimension must match the dataset's vector dimension (adopted from the
  first indexed tensor) → **400** otherwise.
- An optional `"metric"` field must match the dataset's index metric, else **400**
  (recreate the dataset to change the metric).
- **404** if the dataset does not exist.

### `POST /datasets/{ds}/search/contraction` — structural similarity search

Tensor-contraction similarity (Tucker sketch + Grassmannian distance) over
same-shape tensors — finds structurally similar tensors that cosine similarity on
the flattened form misses.

Request:
```json
{ "data": [ … ], "shape": [64, 64, 3], "k": 10 }
```
Response `200` — hits ranked by structural similarity in `(0, 1]`:
```json
[
  { "tensor_id": "…", "score": 0.97, "descriptor": { … }, "metadata": { … } }
]
```
- The contraction index locks to the shape of the first matrix/tensor (`ndim ≥ 2`)
  it sees; queries of other shapes return an empty list.
- **400** if `data` length ≠ product of `shape`.

### `POST /query` — neural query language (NQL)

Natural-language query → LLM-produced `QueryPlan` → optimized → executed against
storage + indexes, with self-correction on execution errors. **Requires an LLM to
be configured** (see the server's `TENSORUS_LLM_*` variables); returns **503** if
not.

Request (`dataset` optional, prepended as context):
```json
{ "query": "find symmetric matrices with norm > 5", "dataset": "layers" }
```
Response `200`:
```json
{
  "plan": "{\"op\":\"index_lookup\",\"dataset\":\"layers\",\"predicates\":[…]}",
  "rows": [ { "id": "…", "score": 1.0, "metadata": null } ],
  "count": 1
}
```
- `plan` is the executed `QueryPlan` (JSON). Plan ops: `scan`, `index_lookup`,
  `vector_search`, `aggregate`.
- **422** if the query cannot be planned/executed after the configured number of
  self-correction rounds.

### `POST /agent` — ReAct agent

Runs an autonomous Thought → Action → Observation loop with tools backed by this
service (`list_datasets`, `tensor_scan`, `tensor_search`, `tensor_aggregate`,
`tensor_get`). **Requires an LLM** (503 otherwise).

Request (`max_steps` optional, default 10):
```json
{ "task": "find the largest tensor in dataset layers and report its rank", "max_steps": 10 }
```
Response `200`:
```json
{
  "status": "success",
  "answer": "The largest tensor … has rank 3.",
  "steps": [
    { "thought": "…", "tool": "tensor_aggregate", "args": { … }, "observation": "{…}" }
  ]
}
```
`status` is one of `success`, `max_steps_reached`, `timed_out`, `budget_exceeded`.

### `GET /health` — health probe (public)

Response `200`:
```json
{ "status": "healthy", "version": "1.0.0", "uptime_seconds": 42 }
```

### `GET /metrics` — Prometheus metrics (public)

Response `200`, `text/plain`:
```
# TYPE tensorus_requests_total counter
tensorus_requests_total 17
# TYPE tensorus_errors_total counter
tensorus_errors_total 0
# TYPE tensorus_rate_limited_total counter
tensorus_rate_limited_total 0
# TYPE tensorus_request_latency_ms histogram
tensorus_request_latency_ms_bucket{le="0.5"} 12
tensorus_request_latency_ms_bucket{le="1"} 15
…
tensorus_request_latency_ms_bucket{le="+Inf"} 17
tensorus_request_latency_ms_sum 23.4
tensorus_request_latency_ms_count 17
```

---

## curl walkthrough

```bash
KEY=secret
BASE=http://localhost:8080

curl -X POST $BASE/datasets -H "x-api-key: $KEY" \
     -H 'content-type: application/json' -d '{"name":"weights"}'

ID=$(curl -s -X POST $BASE/datasets/weights/tensors -H "x-api-key: $KEY" \
     -H 'content-type: application/json' \
     -d '{"data":[1,0,0,1],"shape":[2,2],"metadata":{"name":"I2"}}' \
     | python -c "import sys,json;print(json.load(sys.stdin)['tensor_id'])")

curl -s $BASE/datasets/weights/tensors/$ID -H "x-api-key: $KEY"

curl -s -X POST $BASE/datasets/weights/search/property -H "x-api-key: $KEY" \
     -H 'content-type: application/json' -d '{"is_symmetric":true}'

# Vector similarity search (k nearest by the dataset's metric)
curl -s -X POST $BASE/datasets/weights/search/similar -H "x-api-key: $KEY" \
     -H 'content-type: application/json' -d '{"vector":[1,0,0,1],"k":5}'

# Structural (contraction) similarity over same-shape tensors
curl -s -X POST $BASE/datasets/weights/search/contraction -H "x-api-key: $KEY" \
     -H 'content-type: application/json' -d '{"data":[1,0,0,1],"shape":[2,2],"k":5}'

# Natural-language query (requires an LLM endpoint configured on the server)
curl -s -X POST $BASE/query -H "x-api-key: $KEY" \
     -H 'content-type: application/json' \
     -d '{"query":"symmetric matrices with norm > 1","dataset":"weights"}'

curl -s $BASE/health
curl -s $BASE/metrics
```

---

## Configuring the LLM endpoints (`/query`, `/agent`)

These endpoints are enabled when the server is started with an OpenAI-compatible
endpoint (e.g. a local [Ollama](https://ollama.com) or vLLM server over HTTP):

| Variable | Example | Meaning |
|----------|---------|---------|
| `TENSORUS_LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible base URL (HTTP) |
| `TENSORUS_LLM_MODEL` | `qwen2.5:7b` | model name |
| `TENSORUS_LLM_API_KEY` | *(unset)* | optional bearer token |

When unset, `/query` and `/agent` return **503**; all other endpoints work
normally. The built-in HTTP transport is plain-`http` only (a local-first design);
HTTPS/cloud providers require a TLS-enabled build.

---

## gRPC service

The protobuf contract is at `proto/tensorus.proto`, package `tensorus.v1`. A
tonic server mirrors the REST handlers over the same `TensorService`; it is gated
behind the optional `grpc` feature (tonic's build needs `protoc`, unavailable in
this environment).

```protobuf
service TensorusService {
    // Dataset operations
    rpc CreateDataset(CreateDatasetRequest) returns (CreateDatasetResponse);
    rpc ListDatasets(ListDatasetsRequest) returns (ListDatasetsResponse);
    rpc DeleteDataset(DeleteDatasetRequest) returns (DeleteDatasetResponse);

    // Tensor CRUD
    rpc InsertTensor(InsertTensorRequest) returns (InsertTensorResponse);
    rpc GetTensor(GetTensorRequest) returns (GetTensorResponse);
    rpc DeleteTensor(DeleteTensorRequest) returns (DeleteTensorResponse);
    rpc ScanTensors(ScanTensorsRequest) returns (stream TensorRecord);

    // Search
    rpc SearchByProperty(PropertySearchRequest) returns (SearchResponse);
    rpc SearchSimilar(SimilaritySearchRequest) returns (SearchResponse);
    rpc SearchContraction(ContractionSearchRequest) returns (SearchResponse);

    // NQL
    rpc Query(NQLRequest) returns (QueryResponse);

    // Agent (server-streamed Thought/Action/Observation events)
    rpc RunAgent(AgentRequest) returns (stream AgentEvent);

    // Health
    rpc Health(HealthRequest) returns (HealthResponse);
}
```

Key messages: `TensorData { bytes data; repeated uint64 shape; uint32 dtype }`,
`TensorDescriptor { … }` (mirrors the [data model](./data-model.md#tensordescriptor)),
`PropertySearchRequest { dataset; optional min_norm/max_norm/is_symmetric/…; uint32 limit }`,
`SearchResponse { repeated SearchResult results; uint32 total_count; double latency_ms }`,
`NQLRequest { query; dataset }` / `QueryResponse { results; executed_plan; latency_ms }`,
`AgentEvent { event_type; content; uint32 step }`. The `dtype` field is the
stable `DType` discriminant (see [data model](./data-model.md#dtype)).
