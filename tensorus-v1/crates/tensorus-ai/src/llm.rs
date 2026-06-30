//! Model-agnostic LLM abstractions: messages, the [`LlmProvider`] trait, an
//! [`HttpClient`] seam for provider implementations, and error types.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Conversation role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    /// Wire string used by OpenAI/Anthropic-style APIs.
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// A single chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Message {
            role: Role::System,
            content: content.into(),
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Message {
            role: Role::User,
            content: content.into(),
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Message {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// Inference parameters.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub temperature: f32,
    pub max_tokens: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        InferenceConfig {
            temperature: 0.0,
            max_tokens: 1024,
        }
    }
}

/// Token usage for a completion.
#[derive(Debug, Clone, Copy, Default)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// A provider's completion result.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub text: String,
    pub usage: Usage,
    pub latency_ms: f64,
    pub provider: String,
}

/// Static description of a provider used for routing decisions.
#[derive(Debug, Clone)]
pub struct ProviderCapabilities {
    pub max_context_tokens: usize,
    pub supports_structured_output: bool,
    pub typical_latency_ms: f64,
    pub cost_per_1k_input_tokens: f64,
    pub cost_per_1k_output_tokens: f64,
    /// Benchmarked quality in `[0, 1]`.
    pub quality_score: f64,
    /// Whether the provider runs locally (free, preferred by `LocalFirst`).
    pub is_local: bool,
}

impl ProviderCapabilities {
    /// Estimated USD cost of a completion given its usage.
    pub fn cost(&self, usage: &Usage) -> f64 {
        (usage.input_tokens as f64 / 1000.0) * self.cost_per_1k_input_tokens
            + (usage.output_tokens as f64 / 1000.0) * self.cost_per_1k_output_tokens
    }
}

/// Errors surfaced by providers and the router.
#[derive(Debug, Clone, thiserror::Error)]
pub enum LlmError {
    #[error("provider error: {0}")]
    Provider(String),
    #[error("transport error: {0}")]
    Transport(String),
    #[error("structured-output validation failed: {0}")]
    ValidationFailed(String),
    #[error("all providers failed")]
    AllProvidersFailed,
    #[error("budget exceeded")]
    BudgetExceeded,
    #[error("max retries exceeded: {0}")]
    MaxRetries(String),
}

/// A unified LLM provider.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Stable provider name (for logs and routing).
    fn name(&self) -> &str;

    /// Static capabilities for routing/cost decisions.
    fn capabilities(&self) -> &ProviderCapabilities;

    /// Basic chat completion.
    async fn complete(
        &self,
        messages: &[Message],
        config: &InferenceConfig,
    ) -> Result<CompletionResponse, LlmError>;
}

/// An HTTP transport seam so providers can be implemented and tested without a
/// concrete networking stack. Production wires a `reqwest`/`hyper`-backed
/// implementation; tests use a mock.
#[async_trait]
pub trait HttpClient: Send + Sync {
    /// POST a JSON body and return the parsed JSON response.
    async fn post_json(
        &self,
        url: &str,
        headers: &[(String, String)],
        body: serde_json::Value,
    ) -> Result<serde_json::Value, LlmError>;
}
