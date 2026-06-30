//! Tensorus REST API server binary.
//!
//! Configured via environment variables:
//! - `TENSORUS_DATA_DIR`  storage root (default `./data`)
//! - `TENSORUS_API_KEY`   required API key (unset = auth disabled, dev only)
//! - `TENSORUS_HOST`      bind host (default `0.0.0.0`)
//! - `TENSORUS_REST_PORT` bind port (default `8080`)
//! - `TENSORUS_LOG_JSON`  `true` for JSON logs (default pretty)
//! - `RUST_LOG`           tracing filter (default `info`)

use std::sync::Arc;
use tensorus_api::{build_app, init_tracing, ApiConfig, AppState, TensorService};
use tensorus_storage::FileStorage;

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
    let service = TensorService::new(storage);
    let config = ApiConfig {
        api_key,
        rate_capacity: 1000.0,
        rate_per_sec: 1000.0,
    };
    let app = build_app(AppState::new(service, config));

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!(%addr, "Tensorus REST API listening");
    axum::serve(listener, app).await?;
    Ok(())
}
