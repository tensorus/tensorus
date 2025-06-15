import pytest
from fastapi import FastAPI, Depends, HTTPException, Security as FastAPISecurity # Renamed to avoid clash
from fastapi.testclient import TestClient
from fastapi.security.api_key import APIKeyHeader
from typing import Optional, Dict, Any
from jose import jwt, jwk
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from unittest.mock import patch, ANY

from tensorus.config import settings as global_settings # The global settings instance
from tensorus.api.security import verify_api_key, api_key_header_auth # The dependency to test

# --- Test App Setup ---
# We need a minimal FastAPI app to test the dependency in context.

# This is the header object instance used by the dependency
# It's defined in security.py, but we might need to re-initialize it if settings.API_KEY_HEADER_NAME changes
# For these tests, we'll assume settings.API_KEY_HEADER_NAME is fixed or monkeypatched globally.

def create_test_app_with_protected_route():
    app = FastAPI()

    # This matches how it's used in endpoints.py
    @app.get("/protected")
    async def protected_route(api_key: str = FastAPISecurity(verify_api_key)):
        return {"message": "Access granted", "api_key_used": api_key}

    @app.get("/unprotected")
    async def unprotected_route():
        return {"message": "Access granted freely"}
    return app

# --- Fixtures ---

@pytest.fixture
def test_app_client():
    """Provides a TestClient for an app with the protected route."""
    app = create_test_app_with_protected_route()
    with TestClient(app) as client:
        yield client

# --- Tests for verify_api_key ---

def test_verify_api_key_valid_key(test_app_client: TestClient, monkeypatch):
    valid_keys = ["testkey1", "testkey2"]
    header_name = "X-TEST-API-KEY"

    monkeypatch.setattr(global_settings, 'VALID_API_KEYS', valid_keys)
    monkeypatch.setattr(global_settings, 'API_KEY_HEADER_NAME', header_name)
    # Important: The APIKeyHeader instance `api_key_header_auth` in security.py is created at import time.
    # If settings.API_KEY_HEADER_NAME is changed by monkeypatch, that instance won't pick it up.
    # We need to monkeypatch the `api_key_header_auth.name` directly or re-import security after patching settings.
    # The simplest way for testing is to ensure `api_key_header_auth.name` reflects the patched setting.
    monkeypatch.setattr(api_key_header_auth, 'name', header_name)


    headers = {header_name: "testkey1"}
    response = test_app_client.get("/protected", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"message": "Access granted", "api_key_used": "testkey1"}

def test_verify_api_key_invalid_key(test_app_client: TestClient, monkeypatch):
    valid_keys = ["testkey1"]
    header_name = "X-TEST-API-KEY"
    monkeypatch.setattr(global_settings, 'VALID_API_KEYS', valid_keys)
    monkeypatch.setattr(global_settings, 'API_KEY_HEADER_NAME', header_name)
    monkeypatch.setattr(api_key_header_auth, 'name', header_name)

    headers = {header_name: "wrongkey"}
    response = test_app_client.get("/protected", headers=headers)
    assert response.status_code == 401 # Default FastAPI error for Depends raising HTTPException
    assert response.json() == {"detail": "Invalid API Key"}


def test_verify_api_key_missing_key(test_app_client: TestClient, monkeypatch):
    valid_keys = ["testkey1"]
    header_name = "X-TEST-API-KEY" # Does not matter much as no header will be sent
    monkeypatch.setattr(global_settings, 'VALID_API_KEYS', valid_keys)
    monkeypatch.setattr(global_settings, 'API_KEY_HEADER_NAME', header_name)
    monkeypatch.setattr(api_key_header_auth, 'name', header_name)

    response = test_app_client.get("/protected") # No API key header sent
    assert response.status_code == 401
    assert response.json() == {"detail": "Missing API Key"}


def test_verify_api_key_no_keys_configured(test_app_client: TestClient, monkeypatch):
    # If VALID_API_KEYS is empty, no key is valid.
    header_name = "X-TEST-API-KEY"
    monkeypatch.setattr(global_settings, 'VALID_API_KEYS', []) # No keys configured
    monkeypatch.setattr(global_settings, 'API_KEY_HEADER_NAME', header_name)
    monkeypatch.setattr(api_key_header_auth, 'name', header_name)

    headers = {header_name: "anykey"} # Send some key
    response = test_app_client.get("/protected", headers=headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid API Key"} # Because "anykey" is not in []

    # Test missing key when no keys are configured
    response_missing = test_app_client.get("/protected")
    assert response_missing.status_code == 401
    assert response_missing.json() == {"detail": "Missing API Key"}


def test_unprotected_route_accessible(test_app_client: TestClient):
    response = test_app_client.get("/unprotected")
    assert response.status_code == 200
    assert response.json() == {"message": "Access granted freely"}

# To address the issue of `api_key_header_auth` being initialized with old settings values,
# a more robust way is to have the dependency itself fetch `settings.API_KEY_HEADER_NAME` dynamically
# or re-initialize `api_key_header_auth` within the test after monkeypatching settings.
# The current monkeypatch of `api_key_header_auth.name` is a direct workaround.

# --- Tests for verify_jwt_token ---

# Helper to create a test app with a JWT protected route
def create_test_app_with_jwt_route():
    app = FastAPI()
    from tensorus.api.security import verify_jwt_token # Import here to use patched settings if any

    @app.get("/jwt_protected")
    async def jwt_protected_route(token_payload: Dict[str, Any] = FastAPISecurity(verify_jwt_token)):
        return {"message": "JWT Access granted", "claims": token_payload}
    return app


def generate_token_and_jwks(sub: str = "jwt_user", issuer: str = "https://issuer.test", audience: str = "test_aud"):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv_pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    pub_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    jwk_obj = jwk.construct(pub_pem, algorithm="RS256")
    jwk_dict = jwk_obj.to_dict()
    jwk_dict["kid"] = "test-key"
    jwks = {"keys": [jwk_dict]}
    token = jwt.encode({"sub": sub, "iss": issuer, "aud": audience}, priv_pem, algorithm="RS256", headers={"kid": "test-key"})
    return token, jwks

@pytest.fixture
def jwt_test_app_client():
    app = create_test_app_with_jwt_route()
    with TestClient(app) as client:
        yield client

def test_verify_jwt_disabled_dev_mode_no_token(jwt_test_app_client: TestClient, monkeypatch):
    monkeypatch.setattr(global_settings, 'AUTH_JWT_ENABLED', False)
    monkeypatch.setattr(global_settings, 'AUTH_DEV_MODE_ALLOW_DUMMY_JWT', True)

    # Even in dev dummy mode, if JWT is expected (by endpoint using the dependency) but not provided,
    # and JWT is globally disabled, the current verify_jwt_token logic raises 503 (service unavailable)
    # because it's seen as a misconfiguration: endpoint wants JWT but system says JWT is off.
    # If a token *were* provided, it would return dummy claims.
    response = jwt_test_app_client.get("/jwt_protected")
    assert response.status_code == 503
    assert "JWT authentication is not enabled" in response.json()["detail"]

def test_verify_jwt_disabled_dev_mode_with_token(jwt_test_app_client: TestClient, monkeypatch):
    monkeypatch.setattr(global_settings, 'AUTH_JWT_ENABLED', False)
    monkeypatch.setattr(global_settings, 'AUTH_DEV_MODE_ALLOW_DUMMY_JWT', True)

    headers = {"Authorization": "Bearer anydummytoken"}
    response = jwt_test_app_client.get("/jwt_protected", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "JWT Access granted"
    assert data["claims"]["sub"] == "dummy_jwt_user_jwt_disabled_but_dev_mode"

def test_verify_jwt_disabled_no_dev_mode(jwt_test_app_client: TestClient, monkeypatch):
    monkeypatch.setattr(global_settings, 'AUTH_JWT_ENABLED', False)
    monkeypatch.setattr(global_settings, 'AUTH_DEV_MODE_ALLOW_DUMMY_JWT', False)

    headers = {"Authorization": "Bearer anytoken"} # Token is irrelevant here
    response = jwt_test_app_client.get("/jwt_protected", headers=headers)
    assert response.status_code == 503
    assert "JWT authentication is not enabled" in response.json()["detail"]


def test_verify_jwt_enabled_dev_mode_dummy_token(jwt_test_app_client: TestClient, monkeypatch):
    monkeypatch.setattr(global_settings, 'AUTH_JWT_ENABLED', True)
    monkeypatch.setattr(global_settings, 'AUTH_DEV_MODE_ALLOW_DUMMY_JWT', True)

    headers = {"Authorization": "Bearer sometokenstring"}
    response = jwt_test_app_client.get("/jwt_protected", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "JWT Access granted"
    assert data["claims"]["sub"] == "dummy_jwt_user"
    assert data["claims"]["token_value"] == "sometokenstring"

def test_verify_jwt_enabled_dev_mode_no_token(jwt_test_app_client: TestClient, monkeypatch):
    monkeypatch.setattr(global_settings, 'AUTH_JWT_ENABLED', True)
    monkeypatch.setattr(global_settings, 'AUTH_DEV_MODE_ALLOW_DUMMY_JWT', True)
    # oauth2_scheme has auto_error=False, so dependency receives None if no token
    # verify_jwt_token then raises 401 if token is None and JWT is enabled.

    response = jwt_test_app_client.get("/jwt_protected")
    assert response.status_code == 401
    assert "Not authenticated via JWT (No token provided)" in response.json()["detail"]


@patch('tensorus.api.security.log_audit_event')
@patch('requests.get')
def test_verify_jwt_enabled_prod_mode_valid_token(mock_get, mock_audit, jwt_test_app_client: TestClient, monkeypatch):
    monkeypatch.setattr(global_settings, 'AUTH_JWT_ENABLED', True)
    monkeypatch.setattr(global_settings, 'AUTH_DEV_MODE_ALLOW_DUMMY_JWT', False)
    monkeypatch.setattr(global_settings, 'AUTH_JWT_JWKS_URI', "http://dummy.jwks/uri")
    monkeypatch.setattr(global_settings, 'AUTH_JWT_ISSUER', "https://issuer.test")
    monkeypatch.setattr(global_settings, 'AUTH_JWT_AUDIENCE', "test_aud")

    token, jwks = generate_token_and_jwks(issuer="https://issuer.test", audience="test_aud")
    mock_get.return_value.json.return_value = jwks

    headers = {"Authorization": f"Bearer {token}"}
    response = jwt_test_app_client.get("/jwt_protected", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["claims"]["sub"] == "jwt_user"
    mock_audit.assert_any_call("JWT_VALIDATION_SUCCESS", user="jwt_user", details={"issuer": "https://issuer.test", "aud": "test_aud"})

def test_verify_jwt_enabled_prod_mode_no_token(jwt_test_app_client: TestClient, monkeypatch):
    monkeypatch.setattr(global_settings, 'AUTH_JWT_ENABLED', True)
    monkeypatch.setattr(global_settings, 'AUTH_DEV_MODE_ALLOW_DUMMY_JWT', False)

    response = jwt_test_app_client.get("/jwt_protected") # No token
    assert response.status_code == 401
    assert "Not authenticated via JWT (No token provided)" in response.json()["detail"]


@patch('tensorus.api.security.log_audit_event')
@patch('requests.get')
def test_verify_jwt_enabled_prod_mode_invalid_token(mock_get, mock_audit, jwt_test_app_client: TestClient, monkeypatch):
    monkeypatch.setattr(global_settings, 'AUTH_JWT_ENABLED', True)
    monkeypatch.setattr(global_settings, 'AUTH_DEV_MODE_ALLOW_DUMMY_JWT', False)
    monkeypatch.setattr(global_settings, 'AUTH_JWT_JWKS_URI', "http://dummy.jwks/uri")
    monkeypatch.setattr(global_settings, 'AUTH_JWT_ISSUER', "https://issuer.test")
    monkeypatch.setattr(global_settings, 'AUTH_JWT_AUDIENCE', "test_aud")

    token, _ = generate_token_and_jwks(issuer="https://issuer.test", audience="test_aud")
    # JWKS returned does not contain the signing key
    _, wrong_jwks = generate_token_and_jwks(issuer="https://issuer.test", audience="test_aud")
    mock_get.return_value.json.return_value = wrong_jwks

    headers = {"Authorization": f"Bearer {token}"}
    response = jwt_test_app_client.get("/jwt_protected", headers=headers)
    assert response.status_code == 401
    mock_audit.assert_any_call("JWT_VALIDATION_FAILED", details=ANY)
