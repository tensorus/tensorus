//! Intelligent LLM router: ranks providers by a routing strategy, tries them in
//! order with fallback, tracks a spend budget, and produces validated
//! structured output with retry.

use crate::llm::{CompletionResponse, InferenceConfig, LlmError, LlmProvider, Message};
use parking_lot::Mutex;
use serde::de::DeserializeOwned;
use std::sync::Arc;
use std::time::Instant;

/// How the router picks among eligible providers.
#[derive(Debug, Clone)]
pub enum RoutingStrategy {
    /// Cheapest provider whose quality is at least `min_quality`.
    CostOptimized { min_quality: f64 },
    /// Lowest typical latency.
    LatencyOptimized,
    /// Prefer local providers, then by cost.
    LocalFirst,
    /// Always use the provider at this index.
    Fixed { provider_index: usize },
}

/// Tracks spend against a per-hour budget.
struct BudgetTracker {
    limit_per_hour: f64,
    spent: f64,
    window_start: Instant,
}

impl BudgetTracker {
    fn new(limit_per_hour: f64) -> Self {
        BudgetTracker {
            limit_per_hour,
            spent: 0.0,
            window_start: Instant::now(),
        }
    }

    fn roll_window(&mut self) {
        if self.window_start.elapsed().as_secs_f64() >= 3600.0 {
            self.spent = 0.0;
            self.window_start = Instant::now();
        }
    }

    fn has_budget(&mut self) -> bool {
        self.roll_window();
        self.limit_per_hour <= 0.0 || self.spent < self.limit_per_hour
    }

    fn record(&mut self, cost: f64) {
        self.spent += cost;
    }
}

/// A model-agnostic router over a set of providers.
pub struct LlmRouter {
    providers: Vec<Arc<dyn LlmProvider>>,
    strategy: RoutingStrategy,
    budget: Mutex<BudgetTracker>,
}

impl LlmRouter {
    /// Build a router. `budget_per_hour <= 0` disables budget enforcement.
    pub fn new(
        providers: Vec<Arc<dyn LlmProvider>>,
        strategy: RoutingStrategy,
        budget_per_hour: f64,
    ) -> Self {
        LlmRouter {
            providers,
            strategy,
            budget: Mutex::new(BudgetTracker::new(budget_per_hour)),
        }
    }

    /// USD spent in the current hour window.
    pub fn spent(&self) -> f64 {
        self.budget.lock().spent
    }

    /// Provider indices in the order the strategy prefers them.
    fn ranked(&self) -> Vec<usize> {
        let mut idxs: Vec<usize> = (0..self.providers.len()).collect();
        match &self.strategy {
            RoutingStrategy::Fixed { provider_index } => {
                if *provider_index < self.providers.len() {
                    vec![*provider_index]
                } else {
                    vec![]
                }
            }
            RoutingStrategy::CostOptimized { min_quality } => {
                idxs.retain(|&i| self.providers[i].capabilities().quality_score >= *min_quality);
                idxs.sort_by(|&a, &b| {
                    let ca = self.providers[a].capabilities();
                    let cb = self.providers[b].capabilities();
                    ca.cost_per_1k_output_tokens
                        .total_cmp(&cb.cost_per_1k_output_tokens)
                });
                idxs
            }
            RoutingStrategy::LatencyOptimized => {
                idxs.sort_by(|&a, &b| {
                    self.providers[a]
                        .capabilities()
                        .typical_latency_ms
                        .total_cmp(&self.providers[b].capabilities().typical_latency_ms)
                });
                idxs
            }
            RoutingStrategy::LocalFirst => {
                idxs.sort_by(|&a, &b| {
                    let ca = self.providers[a].capabilities();
                    let cb = self.providers[b].capabilities();
                    // local (true) first, then cheaper.
                    cb.is_local.cmp(&ca.is_local).then(
                        ca.cost_per_1k_output_tokens
                            .total_cmp(&cb.cost_per_1k_output_tokens),
                    )
                });
                idxs
            }
        }
    }

    /// Complete a prompt, trying ranked providers until one succeeds.
    pub async fn complete(
        &self,
        messages: &[Message],
        config: &InferenceConfig,
    ) -> Result<CompletionResponse, LlmError> {
        if !self.budget.lock().has_budget() {
            return Err(LlmError::BudgetExceeded);
        }
        let order = self.ranked();
        if order.is_empty() {
            return Err(LlmError::AllProvidersFailed);
        }
        for idx in order {
            let provider = &self.providers[idx];
            match provider.complete(messages, config).await {
                Ok(resp) => {
                    let cost = provider.capabilities().cost(&resp.usage);
                    self.budget.lock().record(cost);
                    return Ok(resp);
                }
                Err(_) => continue,
            }
        }
        Err(LlmError::AllProvidersFailed)
    }

    /// Complete and deserialize into `T`, retrying with error feedback on
    /// invalid output. `schema_hint` is appended to the system context to guide
    /// the model. The Rust type `T` is the schema.
    pub async fn complete_structured<T: DeserializeOwned>(
        &self,
        messages: &[Message],
        schema_hint: &str,
        config: &InferenceConfig,
        max_retries: usize,
    ) -> Result<T, LlmError> {
        let mut convo = messages.to_vec();
        if !schema_hint.is_empty() {
            convo.insert(
                0,
                Message::system(format!(
                    "Respond with a single JSON object matching this schema: {schema_hint}. \
                     Output only JSON, no prose."
                )),
            );
        }
        let mut last_err = String::new();
        for attempt in 0..=max_retries {
            if attempt > 0 {
                convo.push(Message::user(format!(
                    "Your previous response was not valid: {last_err}. Return only valid JSON."
                )));
            }
            let resp = self.complete(&convo, config).await?;
            match parse_json::<T>(&resp.text) {
                Ok(value) => return Ok(value),
                Err(e) => {
                    last_err = e;
                    convo.push(Message::assistant(resp.text));
                }
            }
        }
        Err(LlmError::MaxRetries(last_err))
    }
}

/// Extract and deserialize a JSON object from model text (tolerates surrounding
/// prose or markdown fences by slicing the outermost `{...}`).
pub fn parse_json<T: DeserializeOwned>(text: &str) -> Result<T, String> {
    let trimmed = text.trim();
    let slice = match (trimmed.find('{'), trimmed.rfind('}')) {
        (Some(s), Some(e)) if e >= s => &trimmed[s..=e],
        _ => trimmed,
    };
    serde_json::from_str::<T>(slice).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ProviderCapabilities;
    use crate::providers::MockProvider;
    use serde::Deserialize;

    fn caps(quality: f64, cost: f64, local: bool, latency: f64) -> ProviderCapabilities {
        ProviderCapabilities {
            max_context_tokens: 8192,
            supports_structured_output: true,
            typical_latency_ms: latency,
            cost_per_1k_input_tokens: cost,
            cost_per_1k_output_tokens: cost,
            quality_score: quality,
            is_local: local,
        }
    }

    #[tokio::test]
    async fn fallback_to_next_provider() {
        let providers: Vec<Arc<dyn LlmProvider>> = vec![
            Arc::new(MockProvider::always_fail("p0")),
            Arc::new(MockProvider::always("p1", "recovered")),
        ];
        let router = LlmRouter::new(providers, RoutingStrategy::Fixed { provider_index: 0 }, 0.0);
        // Fixed{0} only tries p0 which fails -> AllProvidersFailed.
        assert!(router
            .complete(&[], &InferenceConfig::default())
            .await
            .is_err());

        // With a strategy that ranks both, fallback reaches p1.
        let providers: Vec<Arc<dyn LlmProvider>> = vec![
            Arc::new(MockProvider::always_fail("p0")),
            Arc::new(MockProvider::always("p1", "recovered")),
        ];
        let router = LlmRouter::new(providers, RoutingStrategy::LatencyOptimized, 0.0);
        let out = router
            .complete(&[], &InferenceConfig::default())
            .await
            .unwrap();
        assert_eq!(out.text, "recovered");
    }

    #[tokio::test]
    async fn cost_optimized_filters_and_orders() {
        let providers: Vec<Arc<dyn LlmProvider>> = vec![
            Arc::new(
                MockProvider::always("cheap-low-quality", "A")
                    .with_caps(caps(0.5, 0.01, false, 100.0)),
            ),
            Arc::new(MockProvider::always("mid", "B").with_caps(caps(0.85, 0.20, false, 100.0))),
            Arc::new(
                MockProvider::always("expensive", "C").with_caps(caps(0.95, 1.0, false, 100.0)),
            ),
        ];
        let router = LlmRouter::new(
            providers,
            RoutingStrategy::CostOptimized { min_quality: 0.8 },
            0.0,
        );
        // Filters out the low-quality one; cheapest remaining (quality>=0.8) is "mid".
        let out = router
            .complete(&[], &InferenceConfig::default())
            .await
            .unwrap();
        assert_eq!(out.text, "B");
    }

    #[tokio::test]
    async fn local_first_prefers_local() {
        let providers: Vec<Arc<dyn LlmProvider>> = vec![
            Arc::new(MockProvider::always("api", "API").with_caps(caps(0.95, 0.5, false, 50.0))),
            Arc::new(MockProvider::always("local", "LOCAL").with_caps(caps(0.8, 0.0, true, 200.0))),
        ];
        let router = LlmRouter::new(providers, RoutingStrategy::LocalFirst, 0.0);
        let out = router
            .complete(&[], &InferenceConfig::default())
            .await
            .unwrap();
        assert_eq!(out.text, "LOCAL");
    }

    #[tokio::test]
    async fn structured_output_retries_then_succeeds() {
        #[derive(Debug, Deserialize, PartialEq)]
        struct Plan {
            op: String,
            limit: u32,
        }
        // First response invalid, second valid JSON.
        let provider = MockProvider::scripted(
            "m",
            vec![
                Ok("not json at all".to_string()),
                Ok(r#"{"op": "scan", "limit": 10}"#.to_string()),
            ],
        );
        let router = LlmRouter::new(
            vec![Arc::new(provider)],
            RoutingStrategy::Fixed { provider_index: 0 },
            0.0,
        );
        let plan: Plan = router
            .complete_structured(
                &[Message::user("plan it")],
                "{op: string, limit: int}",
                &InferenceConfig::default(),
                3,
            )
            .await
            .unwrap();
        assert_eq!(
            plan,
            Plan {
                op: "scan".into(),
                limit: 10
            }
        );
    }

    #[tokio::test]
    async fn structured_output_extracts_from_prose() {
        #[derive(Debug, Deserialize)]
        struct R {
            x: i32,
        }
        let provider = MockProvider::always("m", "Sure! Here you go:\n```json\n{\"x\": 7}\n```");
        let router = LlmRouter::new(
            vec![Arc::new(provider)],
            RoutingStrategy::Fixed { provider_index: 0 },
            0.0,
        );
        let r: R = router
            .complete_structured(&[Message::user("q")], "", &InferenceConfig::default(), 1)
            .await
            .unwrap();
        assert_eq!(r.x, 7);
    }

    #[tokio::test]
    async fn budget_enforced() {
        let provider = MockProvider::always("m", "hi").with_caps(caps(0.9, 100.0, false, 10.0)); // very expensive
        let router = LlmRouter::new(
            vec![Arc::new(provider)],
            RoutingStrategy::Fixed { provider_index: 0 },
            0.001, // tiny budget
        );
        // First call succeeds and blows the budget.
        let _ = router
            .complete(&[Message::user("aaaa")], &InferenceConfig::default())
            .await;
        assert!(router.spent() > 0.0);
        // Second call is rejected.
        let res = router
            .complete(&[Message::user("bbbb")], &InferenceConfig::default())
            .await;
        assert!(matches!(res, Err(LlmError::BudgetExceeded)));
    }

    #[tokio::test]
    async fn routing_overhead_is_small() {
        let providers: Vec<Arc<dyn LlmProvider>> = (0..8)
            .map(|i| Arc::new(MockProvider::always(&format!("p{i}"), "ok")) as Arc<dyn LlmProvider>)
            .collect();
        let router = LlmRouter::new(providers, RoutingStrategy::LocalFirst, 0.0);
        let start = Instant::now();
        for _ in 0..1000 {
            let _ = router
                .complete(&[], &InferenceConfig::default())
                .await
                .unwrap();
        }
        let per = start.elapsed().as_secs_f64() * 1000.0 / 1000.0;
        println!("router overhead per call (mock): {per:.4} ms");
        assert!(per < 5.0, "router overhead too high: {per} ms");
    }
}
