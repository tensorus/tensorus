# Deploying Tensorus

## Container image

```bash
# From tensorus-v1/
docker build -t ghcr.io/tensorus/tensorus:1.0.0 .
docker push ghcr.io/tensorus/tensorus:1.0.0
```

The multi-stage `Dockerfile` builds the `tensorus-server` binary (the Axum REST
API) and ships it on `debian:bookworm-slim` as a non-root user, listening on
`:8080`.

## Helm chart

```bash
helm lint ./deploy/helm/tensorus
helm install tensorus ./deploy/helm/tensorus \
  --set image.repository=ghcr.io/tensorus/tensorus \
  --set auth.apiKey=$(openssl rand -hex 16)

# Optional features:
helm upgrade tensorus ./deploy/helm/tensorus \
  --set autoscaling.enabled=true \
  --set ingress.enabled=true \
  --set serviceMonitor.enabled=true \
  --set gpu.nvidiaCount=1
```

The chart renders: a `StatefulSet` (with a per-replica PVC for the warm tier), a
`ConfigMap` (`tensorus.toml`), a `Secret` (API key), `ClusterIP` + headless
`Service`s, and optional `HorizontalPodAutoscaler`, `Ingress`, and
`ServiceMonitor`. Liveness/readiness probes hit the unauthenticated `/health`
endpoint; Prometheus scrapes `/metrics`.

## Verification status

`helm lint` and `helm template` pass and render correct manifests (validated).
A full `helm install` into `kind`/`minikube` and a live image build were **not**
run in the development environment (no cluster / not exercised here); the chart
and Dockerfile are authored to the documented conventions.

## Architecture note

The current server is a single binary that owns its storage. The plan's fully
disaggregated topology (separate stateless API `Deployment` + stateful index
`StatefulSet` + object-store cold tier) is a forward step on the same chart
structure.
