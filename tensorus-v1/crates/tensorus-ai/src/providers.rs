//! LLM provider implementations.
//!
//! - [`MockProvider`] returns scripted responses for deterministic tests.
//! - [`OpenAiCompatProvider`] speaks the OpenAI `/chat/completions` schema (also
//!   works with vLLM and Ollama's OpenAI-compatible endpoint).
//! - [`AnthropicProvider`] speaks the Anthropic `/v1/messages` schema.
//!
//! The HTTP providers are generic over an [`HttpClient`] so their request
//! building and response parsing are tested without networking.

use crate::llm::{
    CompletionResponse, HttpClient, InferenceConfig, LlmError, LlmProvider, Message,
    ProviderCapabilities, Role, Usage,
};
use async_trait::async_trait;
use parking_lot::Mutex;
use std::collections::VecDeque;

/// A provider that replays scripted results, for tests.
pub struct MockProvider {
    name: String,
    caps: ProviderCapabilities,
    scripted: Mutex<VecDeque<Result<String, LlmError>>>,
    /// If the script is exhausted, repeat this response forever.
    default: Option<String>,
}

impl MockProvider {
    /// A provider that always returns `text`.
    pub fn always(name: &str, text: &str) -> Self {
        MockProvider {
            name: name.to_string(),
            caps: default_caps(),
            scripted: Mutex::new(VecDeque::new()),
            default: Some(text.to_string()),
        }
    }

    /// A provider that always fails.
    pub fn always_fail(name: &str) -> Self {
        let mut q = VecDeque::new();
        // Empty script + no default => every call fails.
        q.clear();
        MockProvider {
            name: name.to_string(),
            caps: default_caps(),
            scripted: Mutex::new(q),
            default: None,
        }
    }

    /// A provider that replays the given sequence of results.
    pub fn scripted(name: &str, script: Vec<Result<String, LlmError>>) -> Self {
        MockProvider {
            name: name.to_string(),
            caps: default_caps(),
            scripted: Mutex::new(script.into()),
            default: None,
        }
    }

    /// Override capabilities (cost, quality, locality).
    pub fn with_caps(mut self, caps: ProviderCapabilities) -> Self {
        self.caps = caps;
        self
    }
}

fn default_caps() -> ProviderCapabilities {
    ProviderCapabilities {
        max_context_tokens: 8192,
        supports_structured_output: true,
        typical_latency_ms: 1.0,
        cost_per_1k_input_tokens: 0.0,
        cost_per_1k_output_tokens: 0.0,
        quality_score: 0.8,
        is_local: true,
    }
}

#[async_trait]
impl LlmProvider for MockProvider {
    fn name(&self) -> &str {
        &self.name
    }
    fn capabilities(&self) -> &ProviderCapabilities {
        &self.caps
    }
    async fn complete(
        &self,
        messages: &[Message],
        _config: &InferenceConfig,
    ) -> Result<CompletionResponse, LlmError> {
        let text = {
            let mut q = self.scripted.lock();
            match q.pop_front() {
                Some(Ok(t)) => t,
                Some(Err(e)) => return Err(e),
                None => match &self.default {
                    Some(t) => t.clone(),
                    None => return Err(LlmError::Provider("mock script exhausted".into())),
                },
            }
        };
        let input_tokens = messages.iter().map(|m| m.content.len() as u32 / 4).sum();
        Ok(CompletionResponse {
            usage: Usage {
                input_tokens,
                output_tokens: text.len() as u32 / 4,
            },
            text,
            latency_ms: self.caps.typical_latency_ms,
            provider: self.name.clone(),
        })
    }
}

/// OpenAI-compatible chat provider.
pub struct OpenAiCompatProvider<H: HttpClient> {
    http: H,
    base_url: String,
    model: String,
    api_key: Option<String>,
    caps: ProviderCapabilities,
}

impl<H: HttpClient> OpenAiCompatProvider<H> {
    pub fn new(http: H, base_url: &str, model: &str, api_key: Option<String>) -> Self {
        OpenAiCompatProvider {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key,
            caps: ProviderCapabilities {
                max_context_tokens: 128_000,
                supports_structured_output: true,
                typical_latency_ms: 600.0,
                cost_per_1k_input_tokens: 0.15,
                cost_per_1k_output_tokens: 0.60,
                quality_score: 0.9,
                is_local: false,
            },
        }
    }

    pub fn with_caps(mut self, caps: ProviderCapabilities) -> Self {
        self.caps = caps;
        self
    }
}

#[async_trait]
impl<H: HttpClient> LlmProvider for OpenAiCompatProvider<H> {
    fn name(&self) -> &str {
        &self.model
    }
    fn capabilities(&self) -> &ProviderCapabilities {
        &self.caps
    }
    async fn complete(
        &self,
        messages: &[Message],
        config: &InferenceConfig,
    ) -> Result<CompletionResponse, LlmError> {
        let body = serde_json::json!({
            "model": self.model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "messages": messages.iter().map(|m| serde_json::json!({
                "role": m.role.as_str(),
                "content": m.content,
            })).collect::<Vec<_>>(),
        });
        let mut headers = vec![("Content-Type".to_string(), "application/json".to_string())];
        if let Some(key) = &self.api_key {
            headers.push(("Authorization".to_string(), format!("Bearer {key}")));
        }
        let start = std::time::Instant::now();
        let resp = self
            .http
            .post_json(
                &format!("{}/chat/completions", self.base_url),
                &headers,
                body,
            )
            .await?;
        let text = resp
            .pointer("/choices/0/message/content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| LlmError::Provider("missing choices[0].message.content".into()))?
            .to_string();
        let usage = Usage {
            input_tokens: resp
                .pointer("/usage/prompt_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            output_tokens: resp
                .pointer("/usage/completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
        };
        Ok(CompletionResponse {
            text,
            usage,
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            provider: self.model.clone(),
        })
    }
}

/// Anthropic Claude provider (`/v1/messages`).
pub struct AnthropicProvider<H: HttpClient> {
    http: H,
    base_url: String,
    model: String,
    api_key: String,
    caps: ProviderCapabilities,
}

impl<H: HttpClient> AnthropicProvider<H> {
    pub fn new(http: H, base_url: &str, model: &str, api_key: &str) -> Self {
        AnthropicProvider {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key: api_key.to_string(),
            caps: ProviderCapabilities {
                max_context_tokens: 200_000,
                supports_structured_output: true,
                typical_latency_ms: 800.0,
                cost_per_1k_input_tokens: 0.80,
                cost_per_1k_output_tokens: 4.0,
                quality_score: 0.95,
                is_local: false,
            },
        }
    }
}

#[async_trait]
impl<H: HttpClient> LlmProvider for AnthropicProvider<H> {
    fn name(&self) -> &str {
        &self.model
    }
    fn capabilities(&self) -> &ProviderCapabilities {
        &self.caps
    }
    async fn complete(
        &self,
        messages: &[Message],
        config: &InferenceConfig,
    ) -> Result<CompletionResponse, LlmError> {
        // Anthropic takes the system prompt separately from the message list.
        let system: String = messages
            .iter()
            .filter(|m| m.role == Role::System)
            .map(|m| m.content.clone())
            .collect::<Vec<_>>()
            .join("\n");
        let msgs: Vec<_> = messages
            .iter()
            .filter(|m| m.role != Role::System)
            .map(|m| serde_json::json!({"role": m.role.as_str(), "content": m.content}))
            .collect();
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "system": system,
            "messages": msgs,
        });
        let headers = vec![
            ("Content-Type".to_string(), "application/json".to_string()),
            ("x-api-key".to_string(), self.api_key.clone()),
            ("anthropic-version".to_string(), "2023-06-01".to_string()),
        ];
        let start = std::time::Instant::now();
        let resp = self
            .http
            .post_json(&format!("{}/v1/messages", self.base_url), &headers, body)
            .await?;
        let text = resp
            .pointer("/content/0/text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| LlmError::Provider("missing content[0].text".into()))?
            .to_string();
        let usage = Usage {
            input_tokens: resp
                .pointer("/usage/input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            output_tokens: resp
                .pointer("/usage/output_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
        };
        Ok(CompletionResponse {
            text,
            usage,
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            provider: self.model.clone(),
        })
    }
}

#[cfg(test)]
pub(crate) mod test_http {
    use super::*;

    /// A mock HTTP client that returns a fixed JSON response.
    pub struct MockHttpClient {
        pub response: serde_json::Value,
        pub last_body: Mutex<Option<serde_json::Value>>,
    }

    impl MockHttpClient {
        pub fn new(response: serde_json::Value) -> Self {
            MockHttpClient {
                response,
                last_body: Mutex::new(None),
            }
        }
    }

    #[async_trait]
    impl HttpClient for MockHttpClient {
        async fn post_json(
            &self,
            _url: &str,
            _headers: &[(String, String)],
            body: serde_json::Value,
        ) -> Result<serde_json::Value, LlmError> {
            *self.last_body.lock() = Some(body);
            Ok(self.response.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_http::MockHttpClient;
    use super::*;

    #[tokio::test]
    async fn openai_compat_parses_response() {
        let resp = serde_json::json!({
            "choices": [{"message": {"role": "assistant", "content": "hello world"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3}
        });
        let http = MockHttpClient::new(resp);
        let provider =
            OpenAiCompatProvider::new(http, "http://localhost:11434/v1", "qwen2.5", None);
        let out = provider
            .complete(&[Message::user("hi")], &InferenceConfig::default())
            .await
            .unwrap();
        assert_eq!(out.text, "hello world");
        assert_eq!(out.usage.input_tokens, 12);
        assert_eq!(out.usage.output_tokens, 3);
        assert_eq!(provider.name(), "qwen2.5");
    }

    #[tokio::test]
    async fn openai_compat_sends_expected_body() {
        let resp = serde_json::json!({
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        });
        let http = MockHttpClient::new(resp);
        let provider = OpenAiCompatProvider::new(http, "http://x/v1", "m", Some("secret".into()));
        let _ = provider
            .complete(
                &[Message::system("sys"), Message::user("u")],
                &InferenceConfig::default(),
            )
            .await
            .unwrap();
        let body = provider.http.last_body.lock().clone().unwrap();
        assert_eq!(body["model"], "m");
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["content"], "u");
    }

    #[tokio::test]
    async fn anthropic_parses_and_splits_system() {
        let resp = serde_json::json!({
            "content": [{"type": "text", "text": "claude says hi"}],
            "usage": {"input_tokens": 20, "output_tokens": 5}
        });
        let http = MockHttpClient::new(resp);
        let provider =
            AnthropicProvider::new(http, "https://api.anthropic.com", "claude-3-5-haiku", "k");
        let out = provider
            .complete(
                &[Message::system("be brief"), Message::user("hi")],
                &InferenceConfig::default(),
            )
            .await
            .unwrap();
        assert_eq!(out.text, "claude says hi");
        let body = provider.http.last_body.lock().clone().unwrap();
        assert_eq!(body["system"], "be brief");
        assert_eq!(body["messages"].as_array().unwrap().len(), 1); // system filtered out
    }

    #[tokio::test]
    async fn mock_provider_scripts_and_fails() {
        let p = MockProvider::scripted(
            "m",
            vec![Err(LlmError::Provider("boom".into())), Ok("second".into())],
        );
        assert!(p.complete(&[], &InferenceConfig::default()).await.is_err());
        let out = p.complete(&[], &InferenceConfig::default()).await.unwrap();
        assert_eq!(out.text, "second");
    }
}
