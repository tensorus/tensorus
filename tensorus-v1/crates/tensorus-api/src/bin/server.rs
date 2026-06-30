//! Tensorus REST API server binary.
//!
//! Configured via environment variables:
//! - `TENSORUS_DATA_DIR`  storage root (default `./data`)
//! - `TENSORUS_API_KEY`   required API key (unset = auth disabled, dev only)
//! - `TENSORUS_HOST`      bind host (default `0.0.0.0`)
//! - `TENSORUS_REST_PORT` bind port (default `8080`)
//! - `TENSORUS_LOG_JSON`  `true` for JSON logs (default pretty)
//! - `RUST_LOG`           tracing filter (default `info`)
//!
//! LLM-backed endpoints (`/query`, `/agent`) are enabled when an OpenAI-compatible
//! endpoint is configured (e.g. a local Ollama/vLLM server over HTTP):
//! - `TENSORUS_LLM_BASE_URL`  e.g. `http://localhost:11434/v1`
//! - `TENSORUS_LLM_MODEL`     e.g. `qwen2.5:7b`
//! - `TENSORUS_LLM_API_KEY`   optional bearer token

use std::sync::Arc;
use tensorus_ai::{LlmProvider, OpenAiCompatProvider, RoutingStrategy};
use tensorus_api::{
    build_app, init_tracing, ApiConfig, AppState, HttpTransport, LlmConfig, TensorService,
};
use tensorus_storage::FileStorage;

/// Build the optional LLM configuration from environment variables.
fn llm_config_from_env() -> Option<LlmConfig> {
    let base_url = std::env::var("TENSORUS_LLM_BASE_URL").ok()?;
    let model = std::env::var("TENSORUS_LLM_MODEL").ok()?;
    let api_key = std::env::var("TENSORUS_LLM_API_KEY").ok();
    let provider = OpenAiCompatProvider::new(HttpTransport::new(), &base_url, &model, api_key);
    tracing::info!(%base_url, %model, "LLM endpoints enabled (/query, /agent)");
    Some(LlmConfig {
        providers: vec![Arc::new(provider) as Arc<dyn LlmProvider>],
        strategy: RoutingStrategy::LocalFirst,
        budget_per_hour: 0.0,
        max_correction: 3,
        agent_max_steps: 10,
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let json_logs = std::env::var("TENSORUS_LOG_JSON").as_deref() == Ok("true");
    init_tracing(json_logs);

    let data_dir = std::env::var("TENSORUS_DATA_DIR").unwrap_or_else(|_| "./data".to_string());
    let host = std::env::var("TENSORUS_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port: u16 = std::env::var("TENSORUS_REST_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);
    let api_key = std::env::var("TENSORUS_API_KEY").ok();
    if api_key.is_none() {
        tracing::warn!(
            "TENSORUS_API_KEY is not set: authentication is DISABLED (development only). \
             Set it to require an API key on all data endpoints."
        );
    }

    let wal_dir = format!("{data_dir}/wal");
    let storage = Arc::new(FileStorage::open(&data_dir, &wal_dir)?);
    let index_config = format!("{data_dir}/indexes/metrics.json");
    let service = TensorService::with_index_persistence(storage, index_config.into());

    // Rebuild in-memory secondary indexes from durable storage.
    tracing::info!("recovering secondary indexes from storage...");
    service.recover().await?;

    let config = ApiConfig {
        api_key,
        rate_capacity: 1000.0,
        rate_per_sec: 1000.0,
    };
    let mut state = AppState::new(service, config);
    if let Some(llm) = llm_config_from_env() {
        state = state.with_llm(llm);
    }
    let app = build_app(state);

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!(%addr, "Tensorus REST API listening");
    axum::serve(listener, app).await?;
    Ok(())
}
