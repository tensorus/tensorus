"""Shared single-process slots for selected scientific HTTP reads."""
from tensorus.request_work_admission import MAX_REQUEST_WORK_SLOTS
from tensorus.similarity_admission import ROUTES as SIMILARITY_ROUTES
MAX_SLOTS = MAX_REQUEST_WORK_SLOTS
ROUTES = (
    "GET /v1/versions/{version_id}/preview",
    "POST /v1/versions/{version_id}/slice",
    "POST /v1/versions/{version_id}/axis-slice",
    "POST /v1/versions/{version_id}/coordinate-slice",
    "POST /v1/versions/{version_id}/slice/npy",
    "GET /v1/versions/{version_id}/npy",
    *(f"POST {path}" for path in SIMILARITY_ROUTES),
)
EVALUATION_ROUTES = (
    "POST /v1/versions/{version_id}/evaluate",
    "POST /v1/versions/{version_id}/validity-statistics",
    "POST /v1/plans/{plan_id}/evaluate",
)
ROUTES_V2 = (*ROUTES, *EVALUATION_ROUTES)
EVIDENCE_ROUTES = (
    "POST /v1/reference-evaluations",
    "POST /v1/reference-evaluations/{evaluation_id}/approve",
    "POST /v1/data-products/{name}/similarity-evaluations",
    "POST /v1/product-similarity-evaluations/{evaluation_id}/approve",
)
ROUTES_V3 = (*ROUTES_V2, *EVIDENCE_ROUTES)
RELEASE_ROUTES = (
    "POST /v1/datasets/{dataset}/releases",
    "POST /v1/releases/{release_id}/approve",
    "POST /v1/data-products/{name}/releases",
    "POST /v1/product-releases/{release_id}/approve",
)
ROUTES_V4 = (*ROUTES_V3, *RELEASE_ROUTES)
RECOVERY_ROUTES = (
    "POST /v1/maintenance/verify-objects",
    "GET /v1/maintenance/recovery-inventory",
    "POST /v1/maintenance/recovery-inventory/verify-batch",
    "GET /v1/maintenance/recovery-inventory/partition",
    "POST /v1/maintenance/recovery-inventory/verify-partition-batch",
)
ROUTES_V5 = (*ROUTES_V4, *RECOVERY_ROUTES)
PROVENANCE_ROUTES = (
    "GET /v1/plans/{plan_id}/receipt",
    "GET /v1/plans/{plan_id}/outcome",
    "GET /v1/fitted-preprocessing/{state_version_id}",
    "POST /v1/fitted-preprocessing/{state_version_id}/review-imported",
    "POST /v1/validity-bindings/{binding_id}/explain-apply-preprocessing",
    "POST /v1/validity-bindings/{binding_id}/complete-transform",
    "GET /v1/releases/{release_id}/portable",
    "POST /v1/datasets/{dataset}/release-history/export",
    "POST /v1/datasets/{dataset}/release-history/imports",
    "POST /v1/datasets/{dataset}/release-history/bundle-imports",
    "POST /v1/data-products/{name}/release-imports",
    "POST /v1/data-products/{name}/source-evaluation-bindings",
)
ROUTES_V6 = (*ROUTES_V5, *PROVENANCE_ROUTES)
INGESTION_ROUTES = (
    "PUT /v1/datasets/{dataset}/tensors/{name}",
    "PUT /v1/datasets/{dataset}/tensors/{name}/npy",
    "PUT /v1/datasets/{dataset}/tensors/{name}/conditional-npy",
    "POST /v1/uploads/{upload_id}/finalize",
)
ROUTES_V7 = (*ROUTES_V6, *INGESTION_ROUTES)
PRODUCT_RELEASE_READ_ROUTES = (
    "GET /v1/data-products/{name}/releases",
    "GET /v1/product-releases/{release_id}",
    "GET /v1/product-releases/{release_id}/portable",
    "GET /v1/data-products/{name}/release-imports",
    "GET /v1/product-release-imports/{import_id}",
)
PRODUCT_DESCRIPTOR_READ_ROUTES = (
    "GET /v1/data-products",
    "GET /v1/data-products/{name}",
    "GET /v1/data-products/{name}/revisions",
)
DATASET_READ_FENCE_ROUTES = ("GET /v1/datasets", "GET /v1/datasets/{dataset}")
SNAPSHOT_READ_FENCE_ROUTES = (
    "GET /v1/datasets/{dataset}/snapshots",
    "GET /v1/snapshots/{snapshot_id}",
    "GET /v1/snapshots/{left}/diff/{right}",
)
ROUTES_V8 = (*ROUTES_V7, *PRODUCT_RELEASE_READ_ROUTES)
QUALITY_RELEASE_ROUTES = (
    "GET /v1/datasets/{dataset}/releases",
    "GET /v1/releases/{release_id}",
    "POST /v1/releases/{release_id}/retirement-review",
    "POST /v1/releases/{release_id}/retire",
    "GET /v1/quality-release-retirements",
    "GET /v1/quality-release-retirements/{release_id}",
)
ROUTES_V9 = (*ROUTES_V8, *QUALITY_RELEASE_ROUTES)
DISCOVERY_ROUTES = (
    "GET /v1/datasets",
    "GET /v1/datasets/{dataset}/tensors/{name}/versions",
    "POST /v1/query",
    "POST /v1/discovery/semantic-index/query",
    "POST /v1/discovery/semantic-index/snapshot-page",
    "GET /v1/datasets/{dataset}/snapshots",
    "GET /v1/snapshots/{snapshot_id}",
    "GET /v1/snapshots/{left}/diff/{right}",
)
ROUTES_V10 = (*ROUTES_V9, *DISCOVERY_ROUTES)
PLAN_STATUS_ROUTES = (
    "GET /v1/plans",
    "GET /v1/plans/{plan_id}",
    "GET /v1/plans/{plan_id}/capacity-review",
    "GET /v1/plans/{plan_id}/price-scenario",
    "GET /v1/plans/{plan_id}/output-reservations",
    "GET /v1/plans/{plan_id}/worker-attempts",
    "GET /v1/plans/{plan_id}/worker-claims",
    "GET /v1/plans/{plan_id}/usage-observations",
    "GET /v1/plans/{plan_id}/process-usage-observations",
    "GET /v1/plans/{plan_id}/usage-handoff",
    "GET /v1/plans/{plan_id}/adjusted-usage-handoff",
    "GET /v1/plans/{plan_id}/usage-adjustments",
)
ROUTES_V11 = (*ROUTES_V10, *PLAN_STATUS_ROUTES)
CREDENTIAL_INVENTORY_ROUTES = ("GET /v1/keys", "GET /v1/keys/pages")
WORKSPACE_OVERVIEW_ROUTES = ("GET /v1/stats", "GET /v1/insights")
TENANT_STAGING_STATUS_ROUTES = (
    "GET /v1/maintenance/staging-reservations",
    "GET /v1/maintenance/staging-work-quota",
    "GET /v1/maintenance/staging-input-quota",
)
WORKFLOW_INVENTORY_ROUTES = (
    "GET /v1/workflows",
    "GET /v1/workflows/{name}",
    "GET /v1/workflows/{name}/versions",
)
HISTORY_READ_ROUTES = (
    "GET /v1/release-history/imports/{import_id}",
    "GET /v1/datasets/{dataset}/release-history/imports/by-name/{name}",
    "POST /v1/datasets/{dataset}/release-history/inspect",
    "POST /v1/datasets/{dataset}/release-history/scan",
    "POST /v1/datasets/{dataset}/release-history/scan-readable",
    "POST /v1/datasets/{dataset}/release-history/index/lookup",
    "POST /v1/datasets/{dataset}/release-history/index/review",
    "POST /v1/datasets/{dataset}/release-history/index/diagnose",
)
