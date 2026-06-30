# Tensorus v1.0 — Complete Serving Guide

**A step-by-step guide for configuring, building, and launching the Tensorus v1.0 commercial service**

This guide is designed for novice engineers who want to deploy Tensorus v1.0 from scratch. We'll cover everything from prerequisites to production deployment.

---

## Table of Contents

1. [What is Tensorus v1.0?](#what-is-tensorus-v10)
2. [Prerequisites](#prerequisites)
3. [Understanding the Architecture](#understanding-the-architecture)
4. [Quick Start (Local Development)](#quick-start-local-development)
5. [Configuration Guide](#configuration-guide)
6. [Building from Source](#building-from-source)
7. [Running the Server](#running-the-server)
8. [API Verification](#api-verification)
9. [Docker Deployment](#docker-deployment)
10. [Kubernetes Deployment](#kubernetes-deployment)
11. [Production Hardening](#production-hardening)
12. [Troubleshooting](#troubleshooting)
13. [Next Steps](#next-steps)

---

## What is Tensorus v1.0?

Tensorus v1.0 is a tensor-native database with the following capabilities:

- **Tensor-first storage**: Store tensors in their native shape
- **Mathematical property queries**: Search by norm, rank, eigenvalues, symmetry, sparsity
- **Vector similarity search**: HNSW-based approximate nearest neighbor search
- **Structural similarity**: Tensor contraction similarity for finding structurally similar tensors
- **AI-powered queries**: Neural Query Language (NQL) and ReAct agents
- **Enterprise features**: Multi-tenancy, RBAC, quotas, authentication, metrics

The server binary is called **`tensorus-server`** and provides a REST API on port 8080 by default.

---

## Prerequisites

Before you begin, ensure you have the following installed:

### Required

- **Rust toolchain** (stable, 1.70+)
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  source $HOME/.cargo/env
  rustc --version  # Verify installation
  ```

- **Git**
  ```bash
  git --version  # Verify installation
  ```

- **curl** (for testing)
  ```bash
  curl --version  # Verify installation
  ```

### Optional (for advanced deployments)

- **Docker** (for containerized deployment)
  ```bash
  docker --version
  ```

- **Kubernetes + Helm** (for cluster deployment)
  ```bash
  kubectl version --client
  helm version
  ```

- **Ollama or vLLM** (for LLM-powered `/query` and `/agent` endpoints)
  ```bash
  # Example: Install Ollama
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull qwen2.5:7b
  ```

---

## Understanding the Architecture

Tensorus v1.0 is a Rust workspace with the following crates:

| Crate | Purpose |
|-------|---------|
| `tensorus-core` | Core types, traits, errors |
| `tensorus-storage` | Crash-safe storage with WAL |
| `tensorus-compute` | Descriptor computation (norms, eigenvalues) |
| `tensorus-index` | Learned indexes (PGM, ALEX, HNSW) |
| `tensorus-search` | Tensor contraction similarity |
| `tensorus-ai` | LLM router, NQL, ReAct agent |
| `tensorus-api` | REST API server (⭐ this is what you'll run) |
| `tensorus-python` | Python SDK (PyO3 bindings) |

The **`tensorus-api`** crate contains the server binary that you'll deploy.

---

## Quick Start (Local Development)

The fastest way to get started:

```bash
# 1. Clone the repository
git clone https://github.com/tensorus/tensorus.git
cd tensorus/tensorus-v1

# 2. Generate an API key
export TENSORUS_API_KEY=$(openssl rand -hex 16)
echo "Your API key: $TENSORUS_API_KEY"  # Save this!

# 3. Build and run
cargo run -p tensorus-api --bin tensorus-server
```

In another terminal:

```bash
# 4. Test the health endpoint
curl http://localhost:8080/health
# Expected: {"status":"ok"}

# 5. Test authenticated API
export KEY=<your-api-key-from-step-2>
curl -H "x-api-key: $KEY" \
  -X POST http://localhost:8080/datasets \
  -H "content-type: application/json" \
  -d '{"name":"test","metric":"cosine"}'
# Expected: {"created":true,"name":"test","metric":"cosine"}
```

✅ If you see these responses, Tensorus is running!

---

## Configuration Guide

Tensorus can be configured via **environment variables**. Here's what each variable controls:

### Core Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TENSORUS_DATA_DIR` | `./data` | Directory for all persistent data |
| `TENSORUS_HOST` | `0.0.0.0` | Server bind address |
| `TENSORUS_REST_PORT` | `8080` | REST API port |
| `RUST_LOG` | `info` | Log level (`debug`, `info`, `warn`, `error`) |
| `TENSORUS_LOG_JSON` | `false` | Enable JSON structured logging (recommended for production) |

### Authentication Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `TENSORUS_API_KEY` | *(unset)* | **Single-key auth mode**: One key for all operations |
| `TENSORUS_ADMIN_KEY` | *(unset)* | **Multi-tenant mode**: Bootstrap admin key for control plane |

**Important**: If neither key is set, authentication is **disabled** (development only!).

### LLM Configuration (Optional)

These enable the `/query` (NQL) and `/agent` (ReAct) endpoints:

| Variable | Default | Description |
|----------|---------|-------------|
| `TENSORUS_LLM_BASE_URL` | *(unset)* | OpenAI-compatible API base URL (e.g., `http://localhost:11434/v1`) |
| `TENSORUS_LLM_MODEL` | *(unset)* | Model name (e.g., `qwen2.5:7b`) |
| `TENSORUS_LLM_API_KEY` | *(unset)* | Optional bearer token for LLM endpoint |

**Note**: Without these, `/query` and `/agent` return 503. All other endpoints work normally.

### Data Directory Layout

Everything is stored under `TENSORUS_DATA_DIR`:

```
data/
├── datasets/
│   └── {name}/
│       └── segment.dat          # Durable tensor records
├── wal/                          # Write-ahead log (crash recovery)
├── replog/                       # Replication change log
├── indexes/
│   ├── metrics.json              # Vector index configurations
│   └── vectors/{name}.hnsw       # Persisted HNSW graphs
└── control/
    └── registry.json             # Multi-tenant registry
```

### Example Configurations

#### Development (no auth, local)
```bash
export TENSORUS_DATA_DIR=./data
export RUST_LOG=debug
```

#### Production (single-key auth)
```bash
export TENSORUS_DATA_DIR=/var/lib/tensorus
export TENSORUS_API_KEY=$(openssl rand -hex 32)
export TENSORUS_LOG_JSON=true
export RUST_LOG=info
```

#### Production (multi-tenant with LLM)
```bash
export TENSORUS_DATA_DIR=/var/lib/tensorus
export TENSORUS_ADMIN_KEY=$(openssl rand -hex 32)
export TENSORUS_LLM_BASE_URL=http://localhost:11434/v1
export TENSORUS_LLM_MODEL=qwen2.5:7b
export TENSORUS_LOG_JSON=true
export RUST_LOG=info
```

---

## Building from Source

### Development Build

```bash
cd tensorus/tensorus-v1
cargo build --workspace
```

Binary location: `target/debug/tensorus-server`

### Production Build (Optimized)

```bash
cd tensorus/tensorus-v1
cargo build --release -p tensorus-api --bin tensorus-server
```

Binary location: `target/release/tensorus-server`

The release build is **significantly faster** and recommended for production.

### Verify the Build

```bash
./target/release/tensorus-server --help
# Expected: Usage information (or the binary runs if no --help is implemented)
```

### Run Tests

```bash
cargo test --workspace
```

### Run Linter

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

---

## Running the Server

### Option 1: Using Cargo (Development)

```bash
# Set environment variables first
export TENSORUS_API_KEY=$(openssl rand -hex 16)
export TENSORUS_DATA_DIR=./data

# Run the server
cargo run -p tensorus-api --bin tensorus-server
```

### Option 2: Run the Binary Directly (Production)

```bash
# Build first
cargo build --release -p tensorus-api --bin tensorus-server

# Set environment variables
export TENSORUS_DATA_DIR=/var/lib/tensorus
export TENSORUS_API_KEY=your-secret-key-here
export TENSORUS_LOG_JSON=true

# Create data directory
mkdir -p /var/lib/tensorus

# Run the binary
./target/release/tensorus-server
```

### Option 3: Run as a Background Service

```bash
# Using nohup
nohup ./target/release/tensorus-server > tensorus.log 2>&1 &

# Or using systemd (create /etc/systemd/system/tensorus.service)
```

Example systemd service file:

```ini
[Unit]
Description=Tensorus v1.0 Server
After=network.target

[Service]
Type=simple
User=tensorus
Group=tensorus
WorkingDirectory=/opt/tensorus
Environment="TENSORUS_DATA_DIR=/var/lib/tensorus"
Environment="TENSORUS_API_KEY=your-key-here"
Environment="TENSORUS_LOG_JSON=true"
Environment="RUST_LOG=info"
ExecStart=/opt/tensorus/tensorus-server
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable tensorus
sudo systemctl start tensorus
sudo systemctl status tensorus
```

### Server Startup Logs

When the server starts successfully, you should see:

```
INFO tensorus_api: Starting Tensorus server on 0.0.0.0:8080
INFO tensorus_api: Data directory: ./data
INFO tensorus_api: API authentication: enabled
INFO tensorus_api: Multi-tenancy: disabled
INFO tensorus_api: LLM endpoints: disabled
```

---

## API Verification

Once the server is running, verify each component:

### 1. Health Check (Public, No Auth)

```bash
curl http://localhost:8080/health
```

Expected response:
```json
{"status":"ok"}
```

### 2. Metrics (Public, No Auth)

```bash
curl http://localhost:8080/metrics
```

Expected: Prometheus-format metrics

### 3. Create a Dataset

```bash
export BASE=http://localhost:8080
export KEY=your-api-key

curl -X POST "$BASE/datasets" \
  -H "x-api-key: $KEY" \
  -H "content-type: application/json" \
  -d '{"name":"weights","metric":"cosine"}'
```

Expected response:
```json
{"created":true,"name":"weights","metric":"cosine"}
```

### 4. Insert a Tensor

```bash
curl -X POST "$BASE/datasets/weights/tensors" \
  -H "x-api-key: $KEY" \
  -H "content-type: application/json" \
  -d '{"data":[1,0,0,1],"shape":[2,2],"metadata":{"name":"I2"}}'
```

Expected response (abbreviated):
```json
{
  "tensor_id":"019f...",
  "descriptor":{
    "shape":[2,2],
    "frobenius_norm":1.414,
    "is_symmetric":true,
    "is_positive_definite":true
  }
}
```

### 5. List Datasets

```bash
curl -H "x-api-key: $KEY" "$BASE/datasets"
```

Expected response:
```json
["weights"]
```

### 6. Property Search

```bash
curl -X POST "$BASE/datasets/weights/search/property" \
  -H "x-api-key: $KEY" \
  -H "content-type: application/json" \
  -d '{"is_symmetric":true,"limit":10}'
```

Expected: Array of matching tensors

### 7. Vector Similarity Search

```bash
curl -X POST "$BASE/datasets/weights/search/similar" \
  -H "x-api-key: $KEY" \
  -H "content-type: application/json" \
  -d '{"vector":[1,0,0,1],"k":5}'
```

Expected: Array of similar tensors ranked by distance

### 8. LLM Endpoints (if configured)

```bash
# NQL Query
curl -X POST "$BASE/query" \
  -H "x-api-key: $KEY" \
  -H "content-type: application/json" \
  -d '{"query":"find symmetric matrices","dataset":"weights"}'

# ReAct Agent
curl -X POST "$BASE/agent" \
  -H "x-api-key: $KEY" \
  -H "content-type: application/json" \
  -d '{"goal":"analyze tensor properties","dataset":"weights"}'
```

Expected: AI-generated responses (503 if LLM not configured)

---

## Docker Deployment

### Build the Docker Image

From the `tensorus-v1/` directory:

```bash
docker build -t tensorus:1.0.0 .
```

This creates a multi-stage build:
1. Builds the Rust binary in a `rust:1-bookworm` builder container
2. Ships the binary on `debian:bookworm-slim` as a non-root user

### Run the Container

```bash
docker run -d \
  --name tensorus \
  -p 8080:8080 \
  -e TENSORUS_API_KEY=$(openssl rand -hex 16) \
  -e TENSORUS_LOG_JSON=true \
  -v tensorus_data:/data \
  tensorus:1.0.0
```

### Verify the Container

```bash
docker ps | grep tensorus
docker logs tensorus
curl http://localhost:8080/health
```

### Docker Compose (Recommended)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  tensorus:
    build: .
    image: tensorus:1.0.0
    container_name: tensorus
    ports:
      - "8080:8080"
    environment:
      - TENSORUS_DATA_DIR=/data
      - TENSORUS_API_KEY=${TENSORUS_API_KEY}
      - TENSORUS_LOG_JSON=true
      - RUST_LOG=info
    volumes:
      - tensorus_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  tensorus_data:
    driver: local
```

Run:

```bash
export TENSORUS_API_KEY=$(openssl rand -hex 16)
echo "API Key: $TENSORUS_API_KEY"  # Save this!
docker-compose up -d
docker-compose logs -f
```

### Push to Registry

```bash
# Tag for your registry
docker tag tensorus:1.0.0 ghcr.io/your-org/tensorus:1.0.0

# Push
docker push ghcr.io/your-org/tensorus:1.0.0
```

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (minikube, kind, GKE, EKS, AKS, etc.)
- `kubectl` configured
- Helm 3.x installed

### Helm Chart Installation

The Helm chart is located in `deploy/helm/tensorus/`.

#### 1. Review the Values

```bash
cat deploy/helm/tensorus/values.yaml
```

Key values to customize:
- `image.repository` — your Docker image
- `image.tag` — image version
- `auth.apiKey` — API authentication key
- `persistence.size` — storage size for data
- `resources` — CPU/memory limits

#### 2. Generate an API Key

```bash
export TENSORUS_API_KEY=$(openssl rand -hex 32)
echo "API Key: $TENSORUS_API_KEY"  # Save this securely!
```

#### 3. Install the Chart

```bash
cd deploy/helm/tensorus

helm install tensorus . \
  --set image.repository=ghcr.io/your-org/tensorus \
  --set image.tag=1.0.0 \
  --set auth.apiKey=$TENSORUS_API_KEY \
  --create-namespace \
  --namespace tensorus
```

#### 4. Verify the Deployment

```bash
# Check pods
kubectl get pods -n tensorus
# Expected: tensorus-0  1/1  Running

# Check services
kubectl get svc -n tensorus

# View logs
kubectl logs -n tensorus tensorus-0 -f

# Port forward for testing
kubectl port-forward -n tensorus svc/tensorus 8080:8080
```

Test from your local machine:

```bash
curl http://localhost:8080/health
```

#### 5. Enable Optional Features

**Autoscaling (HPA):**

```bash
helm upgrade tensorus . \
  --set autoscaling.enabled=true \
  --set autoscaling.minReplicas=2 \
  --set autoscaling.maxReplicas=10 \
  --set autoscaling.targetCPUUtilizationPercentage=70 \
  -n tensorus
```

**Ingress (external access):**

```bash
helm upgrade tensorus . \
  --set ingress.enabled=true \
  --set ingress.className=nginx \
  --set ingress.hosts[0].host=tensorus.yourdomain.com \
  --set ingress.hosts[0].paths[0].path=/ \
  --set ingress.hosts[0].paths[0].pathType=Prefix \
  -n tensorus
```

**Prometheus Monitoring:**

```bash
helm upgrade tensorus . \
  --set serviceMonitor.enabled=true \
  --set serviceMonitor.interval=30s \
  -n tensorus
```

#### 6. Upgrade the Deployment

```bash
# Update values and upgrade
helm upgrade tensorus . \
  --set image.tag=1.0.1 \
  -n tensorus

# Or upgrade with new values file
helm upgrade tensorus . -f custom-values.yaml -n tensorus
```

#### 7. Uninstall

```bash
helm uninstall tensorus -n tensorus
kubectl delete namespace tensorus
```

### Multi-Tenant Setup in Kubernetes

For production multi-tenant deployments:

```bash
# Generate admin key
export ADMIN_KEY=$(openssl rand -hex 32)

# Install with multi-tenancy
helm install tensorus . \
  --set image.repository=ghcr.io/your-org/tensorus \
  --set auth.apiKey="" \
  --set env.adminKey=$ADMIN_KEY \
  -n tensorus
```

Then use the admin key to create tenant-scoped keys via the `/admin/*` endpoints.

---

## Production Hardening

### Security Checklist

- ✅ **Always enable authentication** (`TENSORUS_API_KEY` or `TENSORUS_ADMIN_KEY`)
- ✅ **Use strong, randomly generated keys** (32+ hex characters)
- ✅ **Enable TLS** (use ingress/reverse proxy with certificates)
- ✅ **Rotate API keys regularly**
- ✅ **Use multi-tenancy** for customer isolation
- ✅ **Set resource limits** (CPU, memory, storage quotas)
- ✅ **Enable rate limiting** (built-in token bucket at 1000 req/s by default)

### Observability

#### Structured Logging

```bash
export TENSORUS_LOG_JSON=true
export RUST_LOG=info
```

Logs will be in JSON format, ready for aggregation (Elasticsearch, Loki, etc.).

#### Metrics Collection

Scrape the `/metrics` endpoint with Prometheus:

```yaml
scrape_configs:
  - job_name: 'tensorus'
    static_configs:
      - targets: ['tensorus:8080']
    metrics_path: '/metrics'
```

Key metrics to monitor:
- `http_requests_total` — request count
- `http_request_duration_seconds` — latency histogram
- `tensorus_storage_size_bytes` — storage usage
- `tensorus_index_size_bytes` — index memory

#### Health Monitoring

Configure liveness and readiness probes:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 10
```

### Backup and Recovery

#### Snapshot (System Admin Key Required)

```bash
curl -X POST http://localhost:8080/admin/snapshot \
  -H "x-api-key: $TENSORUS_ADMIN_KEY"
```

Response:
```json
{"path":"/data/snapshots/snapshot-2026-06-30-12-00-00.tar.gz"}
```

#### Backup Strategy

1. **Snapshot the data directory** (via `/admin/snapshot` API)
2. **Copy persistent files**:
   - `{data_dir}/datasets/` — all tensor data
   - `{data_dir}/indexes/` — HNSW graphs and configurations
   - `{data_dir}/control/` — tenant registry
3. **Store snapshots** in durable object storage (S3, GCS, Azure Blob)

#### Restore

```bash
curl -X POST http://localhost:8080/admin/restore \
  -H "x-api-key: $TENSORUS_ADMIN_KEY" \
  -H "content-type: application/json" \
  -d '{"snapshot_path":"/data/snapshots/snapshot-2026-06-30-12-00-00.tar.gz"}'
```

### Scaling Considerations

#### Vertical Scaling

Increase CPU/memory for:
- Larger tensor computations
- More HNSW index capacity
- Higher query throughput

Recommended starting point: 4 vCPU, 8GB RAM

#### Horizontal Scaling (Read Replicas)

The server supports **single-leader replication**:

1. **Leader**: Handles all writes, exposes `/replication/changes` endpoint
2. **Followers**: Replicate change log, serve read-only queries

Configure followers:

```bash
export TENSORUS_REPLICATION_LEADER=http://leader:8080
export TENSORUS_REPLICATION_API_KEY=$LEADER_KEY
```

#### Storage Tiering

Tensorus supports hot/warm/cold storage tiers:

- **Hot**: In-memory/fast SSD (recent, frequently accessed)
- **Warm**: Local SSD (older, occasionally accessed)
- **Cold**: Object storage (archival)

Configure in `tensorus.toml` or programmatically via `TieringConfig`.

### Network Configuration

#### Recommended Architecture

```
Internet
  ↓
[TLS Termination / Ingress]
  ↓
[API Gateway / Load Balancer]
  ↓
[Tensorus Server(s)]
  ↓
[Persistent Storage]
```

#### Firewall Rules

- **8080/tcp** — REST API (internal only)
- **9090/tcp** — gRPC (if enabled)
- Block all other ports

---

## Troubleshooting

### Common Issues

#### 1. Server Won't Start

**Symptoms**: Binary exits immediately, no logs

**Possible causes**:
- Data directory doesn't exist or lacks permissions
- Port 8080 already in use

**Solutions**:
```bash
# Check if port is in use
lsof -i :8080

# Create data directory with proper permissions
mkdir -p /var/lib/tensorus
chown tensorus:tensorus /var/lib/tensorus

# Try a different port
export TENSORUS_REST_PORT=8081
```

#### 2. 401 Unauthorized

**Symptoms**: All API calls return 401

**Possible causes**:
- Missing API key
- Wrong API key
- Key not in correct header

**Solutions**:
```bash
# Verify API key is set
echo $TENSORUS_API_KEY

# Use correct header format
curl -H "x-api-key: $KEY" ...
# OR
curl -H "Authorization: ******" ...
```

#### 3. 503 Service Unavailable on /query or /agent

**Symptoms**: Only `/query` and `/agent` return 503

**Possible causes**:
- LLM environment variables not set

**Solutions**:
```bash
# Set LLM configuration
export TENSORUS_LLM_BASE_URL=http://localhost:11434/v1
export TENSORUS_LLM_MODEL=qwen2.5:7b

# Restart server
```

#### 4. Data Lost After Restart

**Symptoms**: Datasets/tensors disappear on server restart

**Possible causes**:
- `TENSORUS_DATA_DIR` pointing to ephemeral storage
- Data directory not mounted in Docker

**Solutions**:
```bash
# Use persistent directory
export TENSORUS_DATA_DIR=/var/lib/tensorus

# For Docker, use volume
docker run -v tensorus_data:/data ...
```

#### 5. High Memory Usage

**Symptoms**: Server uses excessive RAM

**Possible causes**:
- Large HNSW indexes in memory
- Many datasets with vector indexes

**Solutions**:
- Enable index persistence (HNSW graphs saved to disk)
- Adjust tiering configuration (move cold data to object storage)
- Increase server memory or use horizontal scaling

#### 6. Slow Query Performance

**Symptoms**: Searches taking too long

**Possible causes**:
- Indexes not built
- Inefficient query patterns
- Need to tune HNSW parameters

**Solutions**:
```bash
# Check index status in logs (RUST_LOG=debug)
export RUST_LOG=tensorus_index=debug

# Tune HNSW in tensorus.toml:
# hnsw_ef_search = 200  # Higher = more accurate, slower
```

### Debugging Tips

#### Enable Debug Logging

```bash
export RUST_LOG=debug
cargo run -p tensorus-api --bin tensorus-server
```

Or for specific modules:

```bash
export RUST_LOG=tensorus_api=debug,tensorus_storage=info
```

#### Check Storage State

```bash
# List datasets on disk
ls -lh $TENSORUS_DATA_DIR/datasets/

# Check WAL
ls -lh $TENSORUS_DATA_DIR/wal/

# Check indexes
ls -lh $TENSORUS_DATA_DIR/indexes/
```

#### Test with Minimal Config

```bash
# Start with no auth (testing only!)
unset TENSORUS_API_KEY
unset TENSORUS_ADMIN_KEY
export TENSORUS_DATA_DIR=/tmp/tensorus-test
cargo run -p tensorus-api --bin tensorus-server
```

---

## Next Steps

### Learn More

- **[Architecture documentation](./architecture.md)** — understand the system design
- **[API reference](./api-reference.md)** — complete endpoint documentation
- **[Configuration guide](./configuration.md)** — all config options
- **[Service references](./services/)** — deep dive into each crate

### Python SDK

To use Tensorus from Python:

```bash
cd python/
maturin develop --features python

# Then in Python:
import numpy as np
from tensorus import Tensorus

ts = Tensorus.connect("http://localhost:8080", api_key="your-key")
ts.create_dataset("test")
tid = ts.insert("test", np.eye(3))
print(ts.get("test", tid))
```

See [`python/README.md`](../python/README.md) for details.

### Contributing

Want to contribute? See [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

### Support

- **Issues**: [GitHub Issues](https://github.com/tensorus/tensorus/issues)
- **Discussions**: [GitHub Discussions](https://github.com/tensorus/tensorus/discussions)

---

## Summary

You should now be able to:

✅ Install prerequisites  
✅ Configure Tensorus via environment variables  
✅ Build the server from source  
✅ Run the server locally  
✅ Verify the API works  
✅ Deploy with Docker  
✅ Deploy to Kubernetes with Helm  
✅ Harden for production  
✅ Monitor and troubleshoot  

**Quick reference for experienced users:**

```bash
# Local dev
export TENSORUS_API_KEY=$(openssl rand -hex 16)
cargo run -p tensorus-api --bin tensorus-server

# Production
docker build -t tensorus:1.0.0 .
docker run -d -p 8080:8080 \
  -e TENSORUS_API_KEY=<key> \
  -v data:/data tensorus:1.0.0

# Kubernetes
helm install tensorus ./deploy/helm/tensorus \
  --set auth.apiKey=<key>
```

Happy serving! 🚀
