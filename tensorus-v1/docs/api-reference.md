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

Request:
```json
{ "name": "weights" }
```
Response `200`:
```json
{ "created": true, "name": "weights" }
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
Predicates combine with AND. `min_norm`/`max_norm` bound `frobenius_norm`.

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

curl -s $BASE/health
curl -s $BASE/metrics
```

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
