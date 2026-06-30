# Service: `tensorus-ai`

**The intelligence layer.** A model-agnostic LLM router with pluggable
providers, the Neural Query Language (NQL), a ReAct agent orchestrator, and a
workload auto-optimizer. `#![forbid(unsafe_code)]`.

- **Depends on:** `tensorus-core`, `serde`, `serde_json`, `async-trait`,
  `parking_lot`, `tokio`, `thiserror`.
- Networking is abstracted behind an [`HttpClient`](#httpclient) trait, so the
  whole layer is testable (and builds) without a real HTTP/TLS stack.

| Module | Contents |
|--------|----------|
| `llm` | message/response types, `LlmProvider` + `HttpClient` traits, `LlmError` |
| `providers` | `MockProvider`, `OpenAiCompatProvider`, `AnthropicProvider` |
| `router` | `LlmRouter`, `RoutingStrategy`, structured output, budget |
| `nql` | `QueryPlan` IR, parser, optimizer, executor, `Nql` |
| `agent` | `ReActAgent`, `Tool`/`ToolRegistry`, guardrails |
| `optimizer` | `WorkloadProfiler`, `AutoOptimizer` (UCB1 bandit) |

---

## 1. LLM abstractions (`llm`)

```rust
pub enum Role { System, User, Assistant, Tool }

pub struct Message { pub role: Role, pub content: String }
impl Message {
    pub fn system(c: impl Into<String>) -> Message;
    pub fn user(c: impl Into<String>) -> Message;
    pub fn assistant(c: impl Into<String>) -> Message;
}

pub struct InferenceConfig { pub temperature: f32, pub max_tokens: usize } // default 0.0 / 1024
pub struct Usage { pub input_tokens: u32, pub output_tokens: u32 }
pub struct CompletionResponse { pub text: String, pub usage: Usage, pub latency_ms: f64, pub provider: String }

pub struct ProviderCapabilities {
    pub max_context_tokens: usize,
    pub supports_structured_output: bool,
    pub typical_latency_ms: f64,
    pub cost_per_1k_input_tokens: f64,
    pub cost_per_1k_output_tokens: f64,
    pub quality_score: f64,   // 0..1
    pub is_local: bool,
}
impl ProviderCapabilities { pub fn cost(&self, usage: &Usage) -> f64; }

pub enum LlmError { Provider(String), Transport(String), ValidationFailed(String),
                    AllProvidersFailed, BudgetExceeded, MaxRetries(String) }
```

### `LlmProvider`

```rust
#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> &ProviderCapabilities;
    async fn complete(&self, messages: &[Message], config: &InferenceConfig)
        -> Result<CompletionResponse, LlmError>;
}
```

### `HttpClient`

The transport seam that lets HTTP providers be implemented and tested without a
concrete networking stack:

```rust
#[async_trait]
pub trait HttpClient: Send + Sync {
    async fn post_json(&self, url: &str, headers: &[(String, String)], body: serde_json::Value)
        -> Result<serde_json::Value, LlmError>;
}
```

A production deployment wires a `reqwest`/`hyper` implementation; tests use a
mock that returns canned JSON.

---

## 2. Providers (`providers`)

```rust
// Deterministic provider for tests/wiring.
impl MockProvider {
    pub fn always(name: &str, text: &str) -> Self;                       // fixed response
    pub fn always_fail(name: &str) -> Self;                              // always errors
    pub fn scripted(name: &str, script: Vec<Result<String, LlmError>>) -> Self;
    pub fn with_caps(self, caps: ProviderCapabilities) -> Self;          // override cost/quality/locality
}

// OpenAI /chat/completions schema (also vLLM, Ollama OpenAI-compat).
impl<H: HttpClient> OpenAiCompatProvider<H> {
    pub fn new(http: H, base_url: &str, model: &str, api_key: Option<String>) -> Self;
    pub fn with_caps(self, caps: ProviderCapabilities) -> Self;
}

// Anthropic /v1/messages schema (splits the system prompt out of the message list).
impl<H: HttpClient> AnthropicProvider<H> {
    pub fn new(http: H, base_url: &str, model: &str, api_key: &str) -> Self;
}
```

The HTTP providers build the request body and parse the response (content +
token usage) generically over `HttpClient`.

---

## 3. Router (`router`)

```rust
pub enum RoutingStrategy {
    CostOptimized { min_quality: f64 }, // cheapest provider above a quality floor
    LatencyOptimized,                   // lowest typical latency
    LocalFirst,                         // prefer local, then cheaper
    Fixed { provider_index: usize },    // always this provider
}

pub struct LlmRouter { /* … */ }
impl LlmRouter {
    pub fn new(providers: Vec<Arc<dyn LlmProvider>>, strategy: RoutingStrategy, budget_per_hour: f64) -> Self;
    pub async fn complete(&self, messages: &[Message], config: &InferenceConfig)
        -> Result<CompletionResponse, LlmError>;
    pub async fn complete_structured<T: DeserializeOwned>(
        &self, messages: &[Message], schema_hint: &str,
        config: &InferenceConfig, max_retries: usize) -> Result<T, LlmError>;
    pub fn spent(&self) -> f64; // USD spent this hour
}

pub fn parse_json<T: DeserializeOwned>(text: &str) -> Result<T, String>;
```

- **`complete`** ranks providers per the strategy, then tries them in order with
  **fallback** until one succeeds; it records cost against a per-hour budget
  (`budget_per_hour ≤ 0` disables enforcement; an exhausted budget returns
  `BudgetExceeded`).
- **`complete_structured::<T>`** asks for JSON, validates it by deserializing
  into `T` (the Rust type *is* the schema), and **retries with error feedback**
  on invalid output up to `max_retries`. `parse_json` tolerates surrounding prose
  / markdown fences by slicing the outermost `{…}`.

**Measured:** routing/fallback overhead `<5 ms` per call (mock providers).

---

## 4. Neural Query Language (`nql`)

Natural language → `QueryPlan` IR → optimization → execution, with
self-correction.

```rust
pub enum Cmp { Eq, Gt, Lt, Ge, Le }
pub struct Predicate { pub field: String, pub cmp: Cmp, pub value: serde_json::Value }

// Tagged by "op" in JSON: "scan" | "index_lookup" | "vector_search" | "aggregate".
pub enum QueryPlan {
    Scan { dataset: String, limit: Option<usize> },
    IndexLookup { dataset: String, predicates: Vec<Predicate>, limit: Option<usize> },
    VectorSearch { dataset: String, k: usize, query: Option<Vec<f32>> },
    Aggregate { dataset: String, function: String, field: String },
}
impl QueryPlan { pub fn dataset(&self) -> &str; }

pub fn optimize(plan: QueryPlan, default_scan_limit: usize) -> QueryPlan;

pub struct QueryRow { pub id: TensorId, pub score: f64, pub metadata: serde_json::Value }

#[async_trait]
pub trait QueryContext: Send + Sync {
    async fn scan(&self, dataset: &str, limit: usize) -> Result<Vec<QueryRow>, String>;
    async fn property_search(&self, dataset: &str, predicates: &[Predicate], limit: usize) -> Result<Vec<QueryRow>, String>;
    async fn vector_search(&self, dataset: &str, query: &[f32], k: usize) -> Result<Vec<QueryRow>, String>;
    async fn aggregate(&self, dataset: &str, function: &str, field: &str) -> Result<Vec<QueryRow>, String>;
}
pub async fn execute<C: QueryContext>(plan: &QueryPlan, ctx: &C) -> Result<Vec<QueryRow>, String>;

pub struct NqlResult { pub plan: QueryPlan, pub rows: Vec<QueryRow>, pub plan_json: String }

pub struct Nql { /* … */ }
impl Nql {
    pub fn new(router: LlmRouter) -> Self;
    pub async fn parse(&self, nl: &str) -> Result<QueryPlan, LlmError>;
    pub async fn query<C: QueryContext>(&self, nl: &str, ctx: &C, max_correction: usize)
        -> Result<NqlResult, String>;
}
```

- The **parser** sends the NL query with a schema hint and few-shot examples to
  the router's `complete_structured`, yielding a validated `QueryPlan`.
- The **optimizer** orders predicates by selectivity (equality first) and caps
  unbounded scans.
- The **executor** dispatches the plan to a `QueryContext` — implemented by the
  serving layer over storage + indexes (the integration tests implement one
  directly over `FileStorage`).
- **Self-correction:** if execution fails, the error is fed back to the LLM to
  revise the plan, up to `max_correction` times.

Parses `"get all tensors from dataset X"` → `Scan`, and `"find symmetric
matrices with norm > 5"` → `IndexLookup` with two predicates.

---

## 5. ReAct agent (`agent`)

A Thought → Action → Observation loop driven by the router, with a pluggable
tool registry and guardrails.

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn schema(&self) -> serde_json::Value;            // JSON Schema for args (advisory)
    async fn execute(&self, args: serde_json::Value) -> Result<String, String>;
}

pub struct FnTool { /* closure-backed Tool */ }
impl FnTool {
    pub fn new(name: &str, description: &str, schema: serde_json::Value,
               func: impl Fn(serde_json::Value) -> Result<String, String> + Send + Sync + 'static) -> Self;
}

pub struct ToolRegistry { /* … */ }
impl ToolRegistry { pub fn new() -> Self; pub fn register(&mut self, tool: Arc<dyn Tool>); }

pub struct Step { pub thought: String, pub tool: Option<String>, pub args: Option<Value>, pub observation: Option<String> }

pub enum AgentOutcome {
    Success { answer: String, steps: Vec<Step> },
    MaxStepsReached { steps: Vec<Step> },
    TimedOut { steps: Vec<Step> },
    BudgetExceeded { steps: Vec<Step> },
}

pub struct AgentConfig { pub max_steps: usize, pub timeout: Duration } // default 10 / 60s

pub struct ReActAgent { /* … */ }
impl ReActAgent {
    pub fn new(router: Arc<LlmRouter>, tools: ToolRegistry, cfg: AgentConfig) -> Self;
    pub async fn run(&self, task: &str) -> AgentOutcome;
}
```

Each step the model returns a JSON `Decision` — either
`{thought, action: {tool, args}}` or `{thought, final_answer}`. The agent
executes the named tool (observations are fed back), and stops on a final
answer, at `max_steps`, on `timeout`, or when the router's budget is exhausted.
Tools concrete to the database (`tensor_store`, `tensor_search`,
`tensor_compute`, `nql_query`) are registered by the serving layer, which has
access to storage/indexes/NQL.

---

## 6. Auto-optimizer (`optimizer`)

Self-tuning: profiles the workload, generates candidate tuning actions, and
selects among them with a **UCB1 multi-armed bandit**, subject to safety
constraints (cooldown + action budget) to avoid oscillation.

```rust
pub struct QueryObservation { pub field: String, pub latency_ms: f64 }

pub struct WorkloadProfiler { /* ring buffer */ }
impl WorkloadProfiler {
    pub fn new(capacity: usize) -> Self;
    pub fn record(&mut self, field: &str, latency_ms: f64);
    pub fn field_frequency(&self) -> HashMap<String, usize>;
    pub fn hot_fields(&self, k: usize) -> Vec<String>;
    pub fn mean_latency(&self, field: &str) -> Option<f64>;
}

pub enum Action { CreateIndex(String), NoOp }

pub struct OptimizerConfig { pub top_k: usize, pub exploration: f64, pub cooldown_iters: u64, pub action_budget: u64 }

pub struct AutoOptimizer { pub profiler: WorkloadProfiler, /* … */ }
impl AutoOptimizer {
    pub fn new(cfg: OptimizerConfig) -> Self;
    pub fn observe(&mut self, field: &str, latency_ms: f64);
    pub fn suggest(&self, indexed: &HashSet<String>) -> Option<Action>;
    pub fn apply(&mut self, action: Action, reward: f64);
}
```

- `suggest` proposes `CreateIndex` for hot, not-yet-indexed fields (plus `NoOp`),
  ranked by UCB1 (unexplored arms first), skipping actions on cooldown and
  honoring the action budget.
- `apply` records the reward (e.g. latency saved) and starts the cooldown.

Over a simulated workload it identifies the frequently-queried field, builds its
index, and drives average latency down — without oscillating.

---

## Worked example (router + structured output)

```rust
use std::sync::Arc;
use serde::Deserialize;
use tensorus_ai::{LlmRouter, RoutingStrategy, MockProvider, Message, InferenceConfig, LlmProvider};

#[derive(Deserialize)]
struct Plan { op: String, limit: u32 }

# async fn demo() {
let providers: Vec<Arc<dyn LlmProvider>> = vec![
    Arc::new(MockProvider::always("local", r#"{"op":"scan","limit":10}"#)),
];
let router = LlmRouter::new(providers, RoutingStrategy::LocalFirst, 0.0);
let plan: Plan = router
    .complete_structured(&[Message::user("plan it")], "{op:string,limit:int}", &InferenceConfig::default(), 3)
    .await.unwrap();
assert_eq!(plan.op, "scan");
# }
```

[`HttpClient`]: #httpclient
