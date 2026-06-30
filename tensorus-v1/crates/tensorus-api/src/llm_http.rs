//! A dependency-free HTTP/1.1 client implementing [`tensorus_ai::HttpClient`],
//! so the LLM router can talk to real model servers (Ollama, vLLM, or any
//! OpenAI-compatible endpoint) over plain HTTP.
//!
//! ## Scope and TLS
//!
//! This transport speaks `http://` only. A pure-Rust HTTPS stack requires a TLS
//! implementation whose crypto backend (`ring`/`aws-lc`) pulls in a C/assembly
//! toolchain, which this build environment deliberately avoids. Local inference
//! servers (the `local_first` default) are reached over `http://localhost`, so
//! `http` is sufficient for the primary path; cloud HTTPS providers require a
//! TLS-enabled build (a future opt-in feature). An `https://` URL returns a
//! clear [`LlmError::Transport`] rather than failing obscurely.
//!
//! The request uses `Connection: close` (one-shot completions need no
//! keep-alive) and the reader consumes to EOF; the response parser handles both
//! `Content-Length` and `Transfer-Encoding: chunked` bodies.

use async_trait::async_trait;
use std::time::Duration;
use tensorus_ai::{HttpClient, LlmError};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// HTTP/1.1 transport for LLM providers (plain `http://`).
#[derive(Debug, Clone)]
pub struct HttpTransport {
    timeout: Duration,
}

impl Default for HttpTransport {
    fn default() -> Self {
        HttpTransport::new()
    }
}

impl HttpTransport {
    /// A transport with the default 60s request timeout.
    pub fn new() -> Self {
        HttpTransport {
            timeout: Duration::from_secs(60),
        }
    }

    /// A transport with a custom request timeout.
    pub fn with_timeout(timeout: Duration) -> Self {
        HttpTransport { timeout }
    }
}

struct Target {
    host: String,
    port: u16,
    path: String,
}

impl Target {
    fn host_header(&self) -> String {
        if self.port == 80 {
            self.host.clone()
        } else {
            format!("{}:{}", self.host, self.port)
        }
    }
}

fn parse_url(url: &str) -> Result<Target, LlmError> {
    let rest = if let Some(r) = url.strip_prefix("http://") {
        r
    } else if url.starts_with("https://") {
        return Err(LlmError::Transport(
            "HTTPS is not supported by the built-in HTTP transport (no TLS in this build); \
             use an http:// endpoint such as a local Ollama/vLLM server"
                .into(),
        ));
    } else {
        return Err(LlmError::Transport(format!(
            "unsupported URL scheme: {url}"
        )));
    };
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) => (
            h.to_string(),
            p.parse::<u16>()
                .map_err(|_| LlmError::Transport(format!("invalid port in '{authority}'")))?,
        ),
        None => (authority.to_string(), 80),
    };
    if host.is_empty() {
        return Err(LlmError::Transport("empty host in URL".into()));
    }
    Ok(Target {
        host,
        port,
        path: path.to_string(),
    })
}

fn build_request(target: &Target, headers: &[(String, String)], body: &[u8]) -> Vec<u8> {
    let mut req = format!("POST {} HTTP/1.1\r\n", target.path);
    req.push_str(&format!("Host: {}\r\n", target.host_header()));
    let mut has_content_type = false;
    for (k, v) in headers {
        let kl = k.to_lowercase();
        // We manage these ourselves.
        if kl == "host" || kl == "content-length" || kl == "connection" {
            continue;
        }
        if kl == "content-type" {
            has_content_type = true;
        }
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    if !has_content_type {
        req.push_str("Content-Type: application/json\r\n");
    }
    req.push_str(&format!("Content-Length: {}\r\n", body.len()));
    req.push_str("Connection: close\r\n\r\n");
    let mut bytes = req.into_bytes();
    bytes.extend_from_slice(body);
    bytes
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn parse_status(line: &str) -> Result<u16, LlmError> {
    // e.g. "HTTP/1.1 200 OK"
    let mut parts = line.split_whitespace();
    let _http = parts.next();
    let code = parts
        .next()
        .ok_or_else(|| LlmError::Transport(format!("bad status line: '{line}'")))?;
    code.parse::<u16>()
        .map_err(|_| LlmError::Transport(format!("bad status code: '{line}'")))
}

fn dechunk(mut body: &[u8]) -> Result<Vec<u8>, LlmError> {
    let mut out = Vec::new();
    while let Some(nl) = find_subslice(body, b"\r\n") {
        let size_line = String::from_utf8_lossy(&body[..nl]);
        // Strip any chunk extensions after ';'.
        let size_hex = size_line.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_hex, 16)
            .map_err(|_| LlmError::Transport(format!("bad chunk size: '{size_hex}'")))?;
        let data_start = nl + 2;
        if size == 0 {
            break;
        }
        let data_end = data_start + size;
        if data_end > body.len() {
            return Err(LlmError::Transport("truncated chunk body".into()));
        }
        out.extend_from_slice(&body[data_start..data_end]);
        let next = data_end + 2; // skip trailing CRLF
        if next > body.len() {
            break;
        }
        body = &body[next..];
    }
    Ok(out)
}

/// Parse a raw HTTP response into JSON. Separated from socket I/O so it can be
/// unit-tested with fixtures.
fn parse_response(raw: &[u8]) -> Result<serde_json::Value, LlmError> {
    let split = find_subslice(raw, b"\r\n\r\n").ok_or_else(|| {
        LlmError::Transport("malformed HTTP response (no header terminator)".into())
    })?;
    let header_str = String::from_utf8_lossy(&raw[..split]);
    let body = &raw[split + 4..];

    let mut lines = header_str.split("\r\n");
    let status = parse_status(
        lines
            .next()
            .ok_or_else(|| LlmError::Transport("empty response".into()))?,
    )?;

    let mut chunked = false;
    let mut content_length: Option<usize> = None;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            let kl = k.trim().to_lowercase();
            let vl = v.trim();
            if kl == "transfer-encoding" && vl.to_lowercase().contains("chunked") {
                chunked = true;
            } else if kl == "content-length" {
                content_length = vl.parse::<usize>().ok();
            }
        }
    }

    let decoded = if chunked {
        dechunk(body)?
    } else if let Some(n) = content_length {
        body[..n.min(body.len())].to_vec()
    } else {
        body.to_vec()
    };

    if status >= 400 {
        let snippet: String = String::from_utf8_lossy(&decoded)
            .chars()
            .take(300)
            .collect();
        return Err(LlmError::Provider(format!("HTTP {status}: {snippet}")));
    }
    serde_json::from_slice(&decoded)
        .map_err(|e| LlmError::Transport(format!("invalid JSON response: {e}")))
}

#[async_trait]
impl HttpClient for HttpTransport {
    async fn post_json(
        &self,
        url: &str,
        headers: &[(String, String)],
        body: serde_json::Value,
    ) -> Result<serde_json::Value, LlmError> {
        let target = parse_url(url)?;
        let body_bytes =
            serde_json::to_vec(&body).map_err(|e| LlmError::Transport(e.to_string()))?;
        let request = build_request(&target, headers, &body_bytes);

        let exchange = async {
            let mut stream = TcpStream::connect((target.host.as_str(), target.port))
                .await
                .map_err(|e| LlmError::Transport(format!("connect to {url} failed: {e}")))?;
            stream
                .write_all(&request)
                .await
                .map_err(|e| LlmError::Transport(format!("write failed: {e}")))?;
            stream
                .flush()
                .await
                .map_err(|e| LlmError::Transport(format!("flush failed: {e}")))?;
            let mut buf = Vec::with_capacity(4096);
            stream
                .read_to_end(&mut buf)
                .await
                .map_err(|e| LlmError::Transport(format!("read failed: {e}")))?;
            Ok::<Vec<u8>, LlmError>(buf)
        };

        let raw = tokio::time::timeout(self.timeout, exchange)
            .await
            .map_err(|_| LlmError::Transport("request timed out".into()))??;
        parse_response(&raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_url_variants() {
        let t = parse_url("http://localhost:11434/v1/chat/completions").unwrap();
        assert_eq!(t.host, "localhost");
        assert_eq!(t.port, 11434);
        assert_eq!(t.path, "/v1/chat/completions");
        assert_eq!(t.host_header(), "localhost:11434");

        let t = parse_url("http://example.com").unwrap();
        assert_eq!(t.port, 80);
        assert_eq!(t.path, "/");
        assert_eq!(t.host_header(), "example.com");

        assert!(parse_url("https://api.openai.com/v1").is_err());
        assert!(parse_url("ftp://x").is_err());
    }

    #[test]
    fn build_request_sets_managed_headers() {
        let target = parse_url("http://h:8080/p").unwrap();
        let headers = vec![("Authorization".to_string(), "Bearer k".to_string())];
        let req = build_request(&target, &headers, b"{}");
        let text = String::from_utf8(req).unwrap();
        assert!(text.starts_with("POST /p HTTP/1.1\r\n"));
        assert!(text.contains("Host: h:8080\r\n"));
        assert!(text.contains("Authorization: Bearer k\r\n"));
        assert!(text.contains("Content-Type: application/json\r\n"));
        assert!(text.contains("Content-Length: 2\r\n"));
        assert!(text.contains("Connection: close\r\n"));
        assert!(text.ends_with("\r\n\r\n{}"));
    }

    #[test]
    fn parse_content_length_response() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\n\r\n{\"ok\":true}xx";
        let v = parse_response(raw).unwrap();
        assert_eq!(v["ok"], true);
    }

    #[test]
    fn parse_chunked_response() {
        // Two chunks "{"x":" and "1}" then terminator.
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\n{\"x\":\r\n2\r\n1}\r\n0\r\n\r\n";
        let v = parse_response(raw).unwrap();
        assert_eq!(v["x"], 1);
    }

    #[test]
    fn parse_error_status_is_provider_error() {
        let raw =
            b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 21\r\n\r\n{\"error\":\"bad key\"}xx";
        let err = parse_response(raw).unwrap_err();
        assert!(matches!(err, LlmError::Provider(_)));
    }
}
