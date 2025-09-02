import os
from fastapi.testclient import TestClient
from tensorus.api import app


def test_security_headers_default(monkeypatch):
    monkeypatch.delenv("TENSORUS_X_FRAME_OPTIONS", raising=False)
    monkeypatch.delenv("TENSORUS_CONTENT_SECURITY_POLICY", raising=False)
    with TestClient(app) as client:
        resp = client.get("/")
    assert resp.headers.get("X-Frame-Options") == "SAMEORIGIN"
    assert resp.headers.get("Content-Security-Policy") == "default-src 'self'"


def test_security_headers_custom(monkeypatch):
    monkeypatch.setenv("TENSORUS_X_FRAME_OPTIONS", "ALLOW-FROM https://example.com")
    policy = "default-src 'self'; script-src 'self' https://cdn.example.com"
    monkeypatch.setenv("TENSORUS_CONTENT_SECURITY_POLICY", policy)
    with TestClient(app) as client:
        resp = client.get("/")
    assert resp.headers["X-Frame-Options"] == "ALLOW-FROM https://example.com"
    assert resp.headers["Content-Security-Policy"] == policy


def test_security_headers_omitted(monkeypatch):
    monkeypatch.setenv("TENSORUS_X_FRAME_OPTIONS", "NONE")
    monkeypatch.setenv("TENSORUS_CONTENT_SECURITY_POLICY", "")
    with TestClient(app) as client:
        resp = client.get("/")
    assert "X-Frame-Options" not in resp.headers
    assert "Content-Security-Policy" not in resp.headers
