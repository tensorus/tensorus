//! Axum REST surface for Tensorus, with API-key auth, token-bucket rate
//! limiting, health, and Prometheus-format metrics.
//!
//! Handlers are transport-thin wrappers over [`TensorService`]; the gRPC surface
//! (the `grpc` module, optional) mirrors them over the same service.

use crate::service::{
    ContractionHit, HealthInfo, InsertRequest, PropertyQuery, ScopedContext, SimilarHit,
    TensorService,
};
use crate::telemetry::Histogram;
use crate::tenancy::{
    valid_slug, KeyMeta, Principal, Quota, QuotaError, Role, Scope, TenantInfo, TenantRegistry,
};
use crate::tools::default_tool_registry;
use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::{Request, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Extension, Json, Router,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tensorus_ai::{
    AgentConfig, AgentOutcome, LlmProvider, LlmRouter, Nql, ReActAgent, RoutingStrategy,
};
use tensorus_core::error::TensorusError;
use tensorus_core::types::TensorId;

/// Configuration for the optional LLM-backed endpoints (`/query`, `/agent`).
///
/// Cloned per request to build a fresh [`LlmRouter`]; the providers themselves
/// are shared via `Arc`.
#[derive(Clone)]
pub struct LlmConfig {
    /// Ordered candidate providers.
    pub providers: Vec<Arc<dyn LlmProvider>>,
    /// How the router ranks providers.
    pub strategy: RoutingStrategy,
    /// Per-hour USD budget (`<= 0` disables enforcement).
    pub budget_per_hour: f64,
    /// Maximum NQL self-correction rounds.
    pub max_correction: usize,
    /// Maximum ReAct agent steps.
    pub agent_max_steps: usize,
}

impl LlmConfig {
    fn router(&self) -> LlmRouter {
        LlmRouter::new(
            self.providers.clone(),
            self.strategy.clone(),
            self.budget_per_hour,
        )
    }
}

/// API configuration.
#[derive(Debug, Clone)]
pub struct ApiConfig {
    /// Required API key; `None` disables auth (development only).
    pub api_key: Option<String>,
    /// Token-bucket capacity (burst).
    pub rate_capacity: f64,
    /// Tokens refilled per second (sustained rate).
    pub rate_per_sec: f64,
}

impl Default for ApiConfig {
    fn default() -> Self {
        ApiConfig {
            api_key: None,
            rate_capacity: 1000.0,
            rate_per_sec: 1000.0,
        }
    }
}

struct TokenBucket {
    tokens: f64,
    last: Instant,
    capacity: f64,
    per_sec: f64,
}

impl TokenBucket {
    fn try_acquire(&mut self) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last).as_secs_f64();
        self.last = now;
        self.tokens = (self.tokens + elapsed * self.per_sec).min(self.capacity);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

#[derive(Default)]
struct Metrics {
    requests: AtomicU64,
    errors: AtomicU64,
    rejected: AtomicU64,
    latency: Histogram,
}

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    service: Arc<TensorService>,
    config: Arc<ApiConfig>,
    bucket: Arc<Mutex<TokenBucket>>,
    metrics: Arc<Metrics>,
    llm: Option<Arc<LlmConfig>>,
    tenancy: Option<Arc<TenantRegistry>>,
    admin_key: Option<String>,
}

impl AppState {
    pub fn new(service: TensorService, config: ApiConfig) -> Self {
        let bucket = TokenBucket {
            tokens: config.rate_capacity,
            last: Instant::now(),
            capacity: config.rate_capacity,
            per_sec: config.rate_per_sec,
        };
        AppState {
            service: Arc::new(service),
            config: Arc::new(config),
            bucket: Arc::new(Mutex::new(bucket)),
            metrics: Arc::new(Metrics::default()),
            llm: None,
            tenancy: None,
            admin_key: None,
        }
    }

    /// Enable the LLM-backed endpoints (`/query`, `/agent`).
    pub fn with_llm(mut self, llm: LlmConfig) -> Self {
        self.llm = Some(Arc::new(llm));
        self
    }

    /// Enable multi-tenancy: keys resolve to tenant principals via `registry`,
    /// and `admin_key` is the bootstrap system (control-plane) key. This
    /// supersedes `ApiConfig.api_key` for authentication.
    pub fn with_tenancy(
        mut self,
        registry: Arc<TenantRegistry>,
        admin_key: Option<String>,
    ) -> Self {
        self.tenancy = Some(registry);
        self.admin_key = admin_key;
        self
    }
}

/// Build the full router (protected API routes + public health/metrics).
pub fn build_app(state: AppState) -> Router {
    let protected = Router::new()
        .route("/datasets", post(create_dataset).get(list_datasets))
        .route("/datasets/:ds/tensors", post(insert_tensor).get(scan))
        .route(
            "/datasets/:ds/tensors/:id",
            get(get_tensor).delete(delete_tensor),
        )
        .route("/datasets/:ds/search/property", post(search_property))
        .route("/datasets/:ds/search/similar", post(search_similar))
        .route("/datasets/:ds/search/contraction", post(search_contraction))
        .route("/query", post(nql_query))
        .route("/agent", post(run_agent))
        // Control plane (multi-tenant mode only; system/tenant-admin guarded).
        .route(
            "/admin/tenants",
            post(admin_create_tenant).get(admin_list_tenants),
        )
        .route(
            "/admin/tenants/:tenant/keys",
            post(admin_issue_key).get(admin_list_keys),
        )
        .route("/admin/tenants/:tenant/usage", get(admin_tenant_usage))
        .route("/admin/keys/:id", axum::routing::delete(admin_revoke_key))
        .route("/admin/snapshot", post(admin_snapshot))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_and_rate_limit,
        ))
        .with_state(state.clone());

    let public = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .with_state(state);

    Router::new().merge(protected).merge(public)
}

/// Auth + rate-limit middleware for protected routes. Resolves the caller to a
/// [`Principal`] and stores it in the request extensions for handlers.
async fn auth_and_rate_limit(
    State(state): State<AppState>,
    mut req: Request<Body>,
    next: Next,
) -> Response {
    state.metrics.requests.fetch_add(1, Ordering::Relaxed);
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    // Rate limit.
    if !state.bucket.lock().try_acquire() {
        state.metrics.rejected.fetch_add(1, Ordering::Relaxed);
        return (StatusCode::TOO_MANY_REQUESTS, "rate limit exceeded").into_response();
    }

    // Authenticate → Principal.
    let principal = match &state.tenancy {
        // Multi-tenant: bootstrap admin key → system; otherwise resolve a tenant key.
        Some(registry) => match extract_api_key(&req) {
            Some(k) if state.admin_key.as_deref() == Some(k.as_str()) => Principal::system(),
            Some(k) => match registry.resolve(&k) {
                Some(p) => p,
                None => return (StatusCode::UNAUTHORIZED, "invalid API key").into_response(),
            },
            None => return (StatusCode::UNAUTHORIZED, "missing API key").into_response(),
        },
        // Legacy single-key mode.
        None => {
            if let Some(expected) = &state.config.api_key {
                if extract_api_key(&req).as_deref() != Some(expected.as_str()) {
                    return (StatusCode::UNAUTHORIZED, "missing or invalid API key")
                        .into_response();
                }
            }
            Principal::unscoped()
        }
    };
    req.extensions_mut().insert(principal);

    let start = Instant::now();
    let resp = next.run(req).await;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    state.metrics.latency.observe(elapsed_ms);
    tracing::info!(
        target: "tensorus_api",
        %method,
        path = %path,
        status = resp.status().as_u16(),
        latency_ms = elapsed_ms,
        "request"
    );
    resp
}

fn extract_api_key(req: &Request<Body>) -> Option<String> {
    if let Some(v) = req.headers().get("x-api-key") {
        return v.to_str().ok().map(|s| s.to_string());
    }
    if let Some(v) = req.headers().get("authorization") {
        if let Ok(s) = v.to_str() {
            if let Some(token) = s.strip_prefix("Bearer ") {
                return Some(token.to_string());
            }
        }
    }
    None
}

/// A JSON error response.
struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(serde_json::json!({"error": self.1}))).into_response()
    }
}

impl From<TensorusError> for ApiError {
    fn from(e: TensorusError) -> Self {
        let code = StatusCode::from_u16(TensorService::status_code(&e))
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        ApiError(code, e.to_string())
    }
}

fn parse_id(s: &str) -> Result<TensorId, ApiError> {
    TensorId::from_str(s)
        .map_err(|_| ApiError(StatusCode::BAD_REQUEST, format!("invalid id '{s}'")))
}

/// The tenant a principal is scoped to, if any.
fn tenant_of(principal: &Principal) -> Option<&str> {
    match &principal.scope {
        Scope::Tenant(t) => Some(t),
        _ => None,
    }
}

/// Map a user-facing dataset name to its storage key, enforcing scope rules.
/// System (control-plane) principals cannot touch tenant data.
fn resolve_ds(principal: &Principal, ds: &str) -> Result<String, ApiError> {
    match &principal.scope {
        Scope::Unscoped => Ok(ds.to_string()),
        Scope::System => Err(ApiError(
            StatusCode::FORBIDDEN,
            "system key cannot access tenant data; use a tenant-scoped key".into(),
        )),
        Scope::Tenant(t) => {
            if !valid_slug(ds) {
                return Err(ApiError(
                    StatusCode::BAD_REQUEST,
                    format!("invalid dataset name '{ds}' (1-63 lowercase alphanumeric or hyphen)"),
                ));
            }
            Ok(format!("{t}.{ds}"))
        }
    }
}

fn require_write(principal: &Principal) -> Result<(), ApiError> {
    if principal.can_write() {
        Ok(())
    } else {
        Err(ApiError(
            StatusCode::FORBIDDEN,
            "read-only key cannot perform writes".into(),
        ))
    }
}

fn quota_to_error(e: QuotaError) -> ApiError {
    let msg = match e {
        QuotaError::Datasets(m) => format!("dataset quota exceeded (max {m})"),
        QuotaError::Tensors(m) => format!("tensor quota exceeded (max {m})"),
    };
    ApiError(StatusCode::TOO_MANY_REQUESTS, msg)
}

/// Require the bootstrap system principal (control plane). Also requires that
/// multi-tenancy is enabled.
fn require_system(state: &AppState, principal: &Principal) -> Result<(), ApiError> {
    if state.tenancy.is_none() {
        return Err(ApiError(
            StatusCode::NOT_FOUND,
            "multi-tenancy is not enabled".into(),
        ));
    }
    if matches!(principal.scope, Scope::System | Scope::Unscoped) {
        Ok(())
    } else {
        Err(ApiError(
            StatusCode::FORBIDDEN,
            "requires the system admin key".into(),
        ))
    }
}

/// Require system, or a tenant admin acting on its own tenant.
fn require_tenant_admin(
    state: &AppState,
    principal: &Principal,
    tenant: &str,
) -> Result<(), ApiError> {
    if state.tenancy.is_none() {
        return Err(ApiError(
            StatusCode::NOT_FOUND,
            "multi-tenancy is not enabled".into(),
        ));
    }
    match &principal.scope {
        Scope::System | Scope::Unscoped => Ok(()),
        Scope::Tenant(t) if t == tenant && principal.is_admin() => Ok(()),
        _ => Err(ApiError(
            StatusCode::FORBIDDEN,
            "requires the system admin key or this tenant's admin key".into(),
        )),
    }
}

// --- request/response bodies ---

#[derive(Deserialize)]
struct CreateDatasetBody {
    name: String,
    /// Optional vector metric: `cosine` (default) | `l2`/`euclidean` | `dot`.
    #[serde(default)]
    metric: Option<String>,
}

#[derive(Serialize)]
struct InsertResponse {
    tensor_id: String,
    descriptor: tensorus_core::types::TensorDescriptor,
}

#[derive(Serialize)]
struct GetResponse {
    tensor_id: String,
    shape: Vec<u64>,
    data: Vec<f32>,
    descriptor: tensorus_core::types::TensorDescriptor,
    metadata: serde_json::Value,
}

#[derive(Serialize)]
struct RecordSummary {
    tensor_id: String,
    descriptor: tensorus_core::types::TensorDescriptor,
    metadata: serde_json::Value,
}

#[derive(Deserialize)]
struct ScanParams {
    #[serde(default = "scan_default_limit")]
    limit: usize,
    #[serde(default)]
    offset: usize,
}

fn scan_default_limit() -> usize {
    100
}

// --- handlers ---

async fn create_dataset(
    State(state): State<AppState>,
    Extension(principal): Extension<Principal>,
    Json(body): Json<CreateDatasetBody>,
) -> Result<Json<serde_json::Value>, ApiError> {
    require_write(&principal)?;
    let key = resolve_ds(&principal, &body.name)?;

    // Reserve a dataset slot against the tenant quota, but only for a genuinely
    // new dataset (create is idempotent).
    let reserved_tenant = match (&state.tenancy, tenant_of(&principal)) {
        (Some(reg), Some(tenant)) if !state.service.has_dataset(&key) => {
            reg.try_add_dataset(tenant).map_err(quota_to_error)?;
            Some(tenant.to_string())
        }
        _ => None,
    };

    let result = match &body.metric {
        Some(m) => state
            .service
            .create_dataset_with_metric(&key, m)
            .await
            .map(|_| m.clone()),
        None => state
            .service
            .create_dataset(&key)
            .await
            .map(|_| crate::index::metric_name(state.service.dataset_metric(&key)).to_string()),
    };

    let metric = match result {
        Ok(m) => m,
        Err(e) => {
            // Roll back the quota reservation on failure.
            if let (Some(reg), Some(tenant)) = (&state.tenancy, &reserved_tenant) {
                reg.remove_dataset(tenant);
            }
            return Err(e.into());
        }
    };
    Ok(Json(
        serde_json::json!({"created": true, "name": body.name, "metric": metric}),
    ))
}

async fn list_datasets(
    State(state): State<AppState>,
    Extension(principal): Extension<Principal>,
) -> Result<Json<Vec<String>>, ApiError> {
    let all = state.service.list_datasets().await?;
    let visible = match &principal.scope {
        Scope::Unscoped => all,
        Scope::System => {
            return Err(ApiError(
                StatusCode::FORBIDDEN,
                "system key cannot list tenant datasets; use /admin/tenants".into(),
            ))
        }
        Scope::Tenant(t) => {
            let prefix = format!("{t}.");
            all.into_iter()
                .filter_map(|k| k.strip_prefix(&prefix).map(|s| s.to_string()))
                .collect()
        }
    };
    Ok(Json(visible))
}

async fn insert_tensor(
    State(state): State<AppState>,
    Extension(principal): Extension<Principal>,
    Path(ds): Path<String>,
    Json(req): Json<InsertRequest>,
) -> Result<Json<InsertResponse>, ApiError> {
    require_write(&principal)?;
    let key = resolve_ds(&principal, &ds)?;

    // Reserve a tensor slot against the tenant quota up-front.
    if let (Some(reg), Some(tenant)) = (&state.tenancy, tenant_of(&principal)) {
        reg.try_add_tensor(tenant).map_err(quota_to_error)?;
    }

    let inserted = state
        .service
        .insert(&key, &req.data, &req.shape, req.metadata)
        .await;
    match inserted {
        Ok((id, descriptor)) => Ok(Json(InsertResponse {
            tensor_id: id.to_string(),
            descriptor,
        })),
        Err(e) => {
            if let (Some(reg), Some(tenant)) = (&state.tenancy, tenant_of(&principal)) {
                reg.remove_tensor(tenant);
            }
            state.metrics.errors.fetch_add(1, Ordering::Relaxed);
            Err(e.into())
        }
    }
}

async fn get_tensor(
    State(state): State<AppState>,
    Extension(principal): Extension<Principal>,
    Path((ds, id)): Path<(String, String)>,
) -> Result<Json<GetResponse>, ApiError> {
    let key = resolve_ds(&principal, &ds)?;
    let id = parse_id(&id)?;
    let rec = state.service.get(&key, id).await?;
    let data: Vec<f32> = rec
        .data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(Json(GetResponse {
        tensor_id: rec.id.to_string(),
        shape: rec.descriptor.shape.0.clone(),
        data,
        descriptor: rec.descriptor,
        metadata: rec.metadata,
    }))
}

async fn delete_tensor(
    State(state): State<AppState>,
    Extension(principal): Extension<Principal>,
    Path((ds, id)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, ApiError> {
    require_write(&principal)?;
    let key = resolve_ds(&principal, &ds)?;
    let id = parse_id(&id)?;
    let existed = state.service.has_tensor(&key, id);
    state.service.delete(&key, id).await?;
    if existed {
        if let (Some(reg), Some(tenant)) = (&state.tenancy, tenant_of(&principal)) {
            reg.remove_tensor(tenant);
        }
    }
    Ok(Json(serde_json::json!({"deleted": true})))
}

async fn scan(
    State(state): State<AppState>,
    Extension(principal): Extension<Principal>,
    Path(ds): Path<String>,
    Query(params): Query<ScanParams>,
) -> Result<Json<Vec<RecordSummary>>, ApiError> {
    let key = resolve_ds(&principal, &ds)?;
    let records = state
        .service
        .scan(&key, params.limit, params.offset)
        .await?;
    Ok(Json(
        records
            .into_iter()
            .map(|r| RecordSummary {
                tensor_id: r.id.to_string(),
                descriptor: r.descriptor,
                metadata: r.metadata,
            })
            .collect(),
    ))
}

async fn search_property(
    State(state): State<AppState>,
    Extension(principal): Extension<Principal>,
    Path(ds): Path<String>,
    Json(q): Json<PropertyQuery>,
) -> Result<Json<Vec<RecordSummary>>, ApiError> {
    let key = resolve_ds(&principal, &ds)?;
    let results = state.service.search_by_property(&key, &q).await?;
    Ok(Json(
        results
            .into_iter()
            .map(|(id, descriptor)| RecordSummary {
                tensor_id: id.to_string(),
                descriptor,
                metadata: serde_json::Value::Null,
            })
            .collect(),
    ))
}

fn default_k() -> usize {
    10
}

#[derive(Deserialize)]
struct SimilarBody {
    /// The query embedding/vector.
    #[serde(default)]
    vector: Option<Vec<f32>>,
    /// Alternative to `vector`: a flattened tensor payload.
    #[serde(default)]
    data: Option<Vec<f32>>,
    #[serde(default = "default_k")]
    k: usize,
    /// Optional metric; must match the dataset's index metric if provided.
    #[serde(default)]
    metric: Option<String>,
}

async fn search_similar(
    State(state): State<AppState>,
    Extension(principal): Extension<Principal>,
    Path(ds): Path<String>,
    Json(body): Json<SimilarBody>,
) -> Result<Json<Vec<SimilarHit>>, ApiError> {
    let key = resolve_ds(&principal, &ds)?;
    let query = body.vector.or(body.data).ok_or_else(|| {
        ApiError(
            StatusCode::BAD_REQUEST,
            "missing 'vector' (or 'data') in request body".into(),
        )
    })?;
    if query.is_empty() {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "empty query vector".into(),
        ));
    }
    let hits = state
        .service
        .search_similar(&key, &query, body.k, body.metric.as_deref())
        .await?;
    Ok(Json(hits))
}

#[derive(Deserialize)]
struct ContractionBody {
    data: Vec<f32>,
    shape: Vec<u64>,
    #[serde(default = "default_k")]
    k: usize,
}

async fn search_contraction(
    State(state): State<AppState>,
    Extension(principal): Extension<Principal>,
    Path(ds): Path<String>,
    Json(body): Json<ContractionBody>,
) -> Result<Json<Vec<ContractionHit>>, ApiError> {
    let key = resolve_ds(&principal, &ds)?;
    let expected: u64 = body.shape.iter().product();
    if expected as usize != body.data.len() {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            format!(
                "data length {} does not match shape product {}",
                body.data.len(),
                expected
            ),
        ));
    }
    let hits = state
        .service
        .search_contraction(&key, &body.data, &body.shape, body.k)
        .await?;
    Ok(Json(hits))
}

#[derive(Deserialize)]
struct QueryBody {
    query: String,
    /// Optional dataset hint, prepended to the natural-language query.
    #[serde(default)]
    dataset: Option<String>,
}

/// Resolve the principal's tenant for LLM data access, rejecting the
/// control-plane (system) principal.
fn llm_tenant(principal: &Principal) -> Result<Option<String>, ApiError> {
    match &principal.scope {
        Scope::Unscoped => Ok(None),
        Scope::Tenant(t) => Ok(Some(t.clone())),
        Scope::System => Err(ApiError(
            StatusCode::FORBIDDEN,
            "system key cannot run queries/agents; use a tenant-scoped key".into(),
        )),
    }
}

async fn nql_query(
    State(state): State<AppState>,
    Extension(principal): Extension<Principal>,
    Json(body): Json<QueryBody>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let llm = state.llm.as_ref().ok_or_else(llm_unconfigured)?;
    let tenant = llm_tenant(&principal)?;
    let nl = match &body.dataset {
        Some(ds) => format!("In dataset '{ds}': {}", body.query),
        None => body.query.clone(),
    };
    let nql = Nql::new(llm.router());
    let ctx = ScopedContext::new(state.service.clone(), tenant);
    match nql.query(&nl, &ctx, llm.max_correction).await {
        Ok(res) => Ok(Json(serde_json::json!({
            "plan": res.plan_json,
            "rows": res.rows,
            "count": res.rows.len(),
        }))),
        Err(e) => Err(ApiError(StatusCode::UNPROCESSABLE_ENTITY, e)),
    }
}

#[derive(Deserialize)]
struct AgentBody {
    task: String,
    #[serde(default)]
    max_steps: Option<usize>,
}

async fn run_agent(
    State(state): State<AppState>,
    Extension(principal): Extension<Principal>,
    Json(body): Json<AgentBody>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let llm = state.llm.as_ref().ok_or_else(llm_unconfigured)?;
    let tenant = llm_tenant(&principal)?;
    let router = Arc::new(llm.router());
    let tools = default_tool_registry(state.service.clone(), tenant);
    let cfg = AgentConfig {
        max_steps: body.max_steps.unwrap_or(llm.agent_max_steps),
        timeout: Duration::from_secs(60),
    };
    let agent = ReActAgent::new(router, tools, cfg);
    let outcome = agent.run(&body.task).await;
    Ok(Json(agent_outcome_json(&outcome)))
}

fn llm_unconfigured() -> ApiError {
    ApiError(
        StatusCode::SERVICE_UNAVAILABLE,
        "LLM is not configured; set TENSORUS_LLM_BASE_URL and TENSORUS_LLM_MODEL".into(),
    )
}

fn steps_to_json(steps: &[tensorus_ai::agent::Step]) -> serde_json::Value {
    serde_json::Value::Array(
        steps
            .iter()
            .map(|s| {
                serde_json::json!({
                    "thought": s.thought,
                    "tool": s.tool,
                    "args": s.args,
                    "observation": s.observation,
                })
            })
            .collect(),
    )
}

fn agent_outcome_json(outcome: &AgentOutcome) -> serde_json::Value {
    match outcome {
        AgentOutcome::Success { answer, steps } => serde_json::json!({
            "status": "success", "answer": answer, "steps": steps_to_json(steps),
        }),
        AgentOutcome::MaxStepsReached { steps } => serde_json::json!({
            "status": "max_steps_reached", "steps": steps_to_json(steps),
        }),
        AgentOutcome::TimedOut { steps } => serde_json::json!({
            "status": "timed_out", "steps": steps_to_json(steps),
        }),
        AgentOutcome::BudgetExceeded { steps } => serde_json::json!({
            "status": "budget_exceeded", "steps": steps_to_json(steps),
        }),
    }
}

// --- admin / control-plane handlers ---

fn registry(state: &AppState) -> Result<&Arc<TenantRegistry>, ApiError> {
    state
        .tenancy
        .as_ref()
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "multi-tenancy is not enabled".into()))
}

#[derive(Deserialize)]
struct CreateTenantBody {
    id: String,
    #[serde(default)]
    max_datasets: usize,
    #[serde(default)]
    max_tensors: usize,
}

async fn admin_create_tenant(
    State(state): State<AppState>,
    Extension(principal): Extension<Principal>,
    Json(body): Json<CreateTenantBody>,
) -> Result<Json<serde_json::Value>, ApiError> {
    require_system(&state, &principal)?;
    let reg = registry(&state)?;
    let info = reg
        .create_tenant(
            &body.id,
            Quota {
                max_datasets: body.max_datasets,
                max_tensors: body.max_tensors,
            },
        )
        .map_err(|e| ApiError(StatusCode::BAD_REQUEST, e))?;
    Ok(Json(serde_json::json!({
        "created": true,
        "id": info.id,
        "quota": info.quota,
        "created_at": info.created_at,
    })))
}

async fn admin_list_tenants(
    State(state): State<AppState>,
    Extension(principal): Extension<Principal>,
) -> Result<Json<Vec<TenantInfo>>, ApiError> {
    require_system(&state, &principal)?;
    Ok(Json(registry(&state)?.list_tenants()))
}

#[derive(Deserialize)]
struct IssueKeyBody {
    role: String,
    #[serde(default)]
    name: Option<String>,
}

async fn admin_issue_key(
    State(state): State<AppState>,
    Extension(principal): Extension<Principal>,
    Path(tenant): Path<String>,
    Json(body): Json<IssueKeyBody>,
) -> Result<Json<serde_json::Value>, ApiError> {
    require_tenant_admin(&state, &principal, &tenant)?;
    let reg = registry(&state)?;
    let role = Role::parse(&body.role).ok_or_else(|| {
        ApiError(
            StatusCode::BAD_REQUEST,
            format!("invalid role '{}'", body.role),
        )
    })?;
    let name = body.name.unwrap_or_else(|| "key".to_string());
    let (id, key) = reg
        .issue_key(&tenant, role, &name)
        .map_err(|e| ApiError(StatusCode::BAD_REQUEST, e))?;
    Ok(Json(serde_json::json!({
        "id": id,
        "key": key,
        "tenant": tenant,
        "role": role.as_str(),
        "name": name,
        "warning": "store this key now; it is shown only once and cannot be retrieved again",
    })))
}

async fn admin_list_keys(
    State(state): State<AppState>,
    Extension(principal): Extension<Principal>,
    Path(tenant): Path<String>,
) -> Result<Json<Vec<KeyMeta>>, ApiError> {
    require_tenant_admin(&state, &principal, &tenant)?;
    Ok(Json(registry(&state)?.list_keys(&tenant)))
}

async fn admin_tenant_usage(
    State(state): State<AppState>,
    Extension(principal): Extension<Principal>,
    Path(tenant): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    require_tenant_admin(&state, &principal, &tenant)?;
    let reg = registry(&state)?;
    if !reg.tenant_exists(&tenant) {
        return Err(ApiError(
            StatusCode::NOT_FOUND,
            format!("unknown tenant '{tenant}'"),
        ));
    }
    let (datasets, tensors) = reg.usage(&tenant);
    let quota = reg.quota(&tenant).unwrap_or_default();
    Ok(Json(serde_json::json!({
        "tenant": tenant,
        "datasets": datasets,
        "tensors": tensors,
        "quota": quota,
    })))
}

async fn admin_revoke_key(
    State(state): State<AppState>,
    Extension(principal): Extension<Principal>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let reg = registry(&state)?;
    let owner = reg
        .key_tenant(&id)
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, format!("unknown key id '{id}'")))?;
    require_tenant_admin(&state, &principal, &owner)?;
    reg.revoke_key(&id)
        .map_err(|e| ApiError(StatusCode::NOT_FOUND, e))?;
    Ok(Json(serde_json::json!({"revoked": true, "id": id})))
}

#[derive(Deserialize)]
struct SnapshotBody {
    dest: String,
}

async fn admin_snapshot(
    State(state): State<AppState>,
    Extension(principal): Extension<Principal>,
    Json(body): Json<SnapshotBody>,
) -> Result<Json<serde_json::Value>, ApiError> {
    require_system(&state, &principal)?;
    state.service.snapshot(&body.dest).await?;
    Ok(Json(
        serde_json::json!({"snapshot": true, "dest": body.dest}),
    ))
}

async fn health(State(state): State<AppState>) -> Json<HealthInfo> {
    Json(state.service.health())
}

async fn metrics(State(state): State<AppState>) -> Response {
    let m = &state.metrics;
    let body = format!(
        "# HELP tensorus_requests_total Total API requests.\n\
         # TYPE tensorus_requests_total counter\n\
         tensorus_requests_total {}\n\
         # HELP tensorus_errors_total Total request errors.\n\
         # TYPE tensorus_errors_total counter\n\
         tensorus_errors_total {}\n\
         # HELP tensorus_rate_limited_total Total rate-limited requests.\n\
         # TYPE tensorus_rate_limited_total counter\n\
         tensorus_rate_limited_total {}\n\
         # HELP tensorus_request_latency_ms Request latency in milliseconds.\n\
         {}",
        m.requests.load(Ordering::Relaxed),
        m.errors.load(Ordering::Relaxed),
        m.rejected.load(Ordering::Relaxed),
        m.latency.render("tensorus_request_latency_ms"),
    );
    (StatusCode::OK, body).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use tempfile::TempDir;
    use tower::ServiceExt;

    async fn test_app(api_key: Option<&str>) -> (Router, TempDir) {
        let tmp = TempDir::new().unwrap();
        let storage = Arc::new(
            tensorus_storage::FileStorage::open(tmp.path().join("data"), tmp.path().join("wal"))
                .unwrap(),
        );
        let service = TensorService::new(storage);
        let config = ApiConfig {
            api_key: api_key.map(|s| s.to_string()),
            ..Default::default()
        };
        (build_app(AppState::new(service, config)), tmp)
    }

    fn req(
        method: &str,
        uri: &str,
        key: Option<&str>,
        body: Option<serde_json::Value>,
    ) -> Request<Body> {
        let mut b = Request::builder().method(method).uri(uri);
        if let Some(k) = key {
            b = b.header("x-api-key", k);
        }
        match body {
            Some(j) => b
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&j).unwrap()))
                .unwrap(),
            None => b.body(Body::empty()).unwrap(),
        }
    }

    async fn body_json(resp: Response) -> serde_json::Value {
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
    }

    #[tokio::test]
    async fn health_is_public_and_ok() {
        let (app, _t) = test_app(Some("secret")).await;
        let resp = app
            .oneshot(req("GET", "/health", None, None))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let j = body_json(resp).await;
        assert_eq!(j["status"], "healthy");
    }

    #[tokio::test]
    async fn unauthorized_without_key() {
        let (app, _t) = test_app(Some("secret")).await;
        let resp = app
            .oneshot(req(
                "POST",
                "/datasets",
                None,
                Some(serde_json::json!({"name": "d"})),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn wrong_key_rejected() {
        let (app, _t) = test_app(Some("secret")).await;
        let resp = app
            .oneshot(req(
                "POST",
                "/datasets",
                Some("nope"),
                Some(serde_json::json!({"name": "d"})),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn full_crud_flow() {
        let (app, _t) = test_app(Some("secret")).await;
        let key = Some("secret");

        // Create dataset.
        let resp = app
            .clone()
            .oneshot(req(
                "POST",
                "/datasets",
                key,
                Some(serde_json::json!({"name": "weights"})),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Insert a 2x2 identity matrix.
        let insert = serde_json::json!({
            "data": [1.0, 0.0, 0.0, 1.0],
            "shape": [2, 2],
            "metadata": {"name": "I2"}
        });
        let resp = app
            .clone()
            .oneshot(req("POST", "/datasets/weights/tensors", key, Some(insert)))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let j = body_json(resp).await;
        let id = j["tensor_id"].as_str().unwrap().to_string();
        assert!(j["descriptor"]["is_symmetric"].as_bool().unwrap());

        // Get it back.
        let resp = app
            .clone()
            .oneshot(req(
                "GET",
                &format!("/datasets/weights/tensors/{id}"),
                key,
                None,
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let j = body_json(resp).await;
        assert_eq!(j["data"], serde_json::json!([1.0, 0.0, 0.0, 1.0]));
        assert_eq!(j["metadata"]["name"], "I2");

        // Property search: symmetric matrices.
        let resp = app
            .clone()
            .oneshot(req(
                "POST",
                "/datasets/weights/search/property",
                key,
                Some(serde_json::json!({"is_symmetric": true})),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let j = body_json(resp).await;
        assert_eq!(j.as_array().unwrap().len(), 1);

        // Delete.
        let resp = app
            .clone()
            .oneshot(req(
                "DELETE",
                &format!("/datasets/weights/tensors/{id}"),
                key,
                None,
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Gone now -> 404.
        let resp = app
            .oneshot(req(
                "GET",
                &format!("/datasets/weights/tensors/{id}"),
                key,
                None,
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn metrics_endpoint_ok() {
        let (app, _t) = test_app(None).await;
        // Make a request first so the latency histogram records something.
        let _ = app
            .clone()
            .oneshot(req("GET", "/datasets", None, None))
            .await
            .unwrap();
        let resp = app
            .oneshot(req("GET", "/metrics", None, None))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains("tensorus_requests_total"));
        assert!(text.contains("tensorus_request_latency_ms_bucket"));
        assert!(text.contains("tensorus_request_latency_ms_count"));
    }
}
