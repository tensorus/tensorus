import pytest
from fastapi import FastAPI, Depends, HTTPException, Security as FastAPISecurity # Renamed to avoid clash
from fastapi.testclient import TestClient
from fastapi.security.api_key import APIKeyHeader
from typing import Optional

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

# Example of testing the dependency directly (not via TestClient)
# This requires careful mocking of what FastAPI's Security(...) does.
# from fastapi.exceptions import HTTPException as FastApiHTTPException
#
# async def run_verify_dependency(api_key_value: Optional[str], header_name_in_security: str):
#     # Simulate how FastAPI calls the dependency with Security wrapper
#     # This is a simplified simulation.
#     # The actual `Security(api_key_header_auth)` part is complex.
#     # It's usually better to test via TestClient.
#     temp_header_scheme = APIKeyHeader(name=header_name_in_security, auto_error=False)
#
#     # This is not how `Security` works. `Security` uses the scheme to extract the key.
#     # The dependency `verify_api_key` *receives* the extracted key.
#     # So, we should call verify_api_key directly with the extracted value.
#
#     # Correct way to test dependency in isolation:
#     # Call verify_api_key as if FastAPI has already extracted the key using the scheme.
#     try:
#         return await verify_api_key(api_key_value)
#     except FastApiHTTPException: # Catch FastAPI's HTTPException
#         raise
#
# @pytest.mark.asyncio
# async def test_verify_api_key_direct_valid(monkeypatch):
#     monkeypatch.setattr(global_settings, 'VALID_API_KEYS', ["directkey"])
#     # No need to patch API_KEY_HEADER_NAME for direct call, as extraction is skipped
#
#     result = await run_verify_dependency("directkey", "any_header_name_for_scheme")
#     assert result == "directkey"
#
# @pytest.mark.asyncio
# async def test_verify_api_key_direct_invalid(monkeypatch):
#     monkeypatch.setattr(global_settings, 'VALID_API_KEYS', ["directkey"])
#     with pytest.raises(HTTPException) as excinfo:
#         await run_verify_dependency("wrongkey", "any_header_name_for_scheme")
#     assert excinfo.value.status_code == 401
#     assert excinfo.value.detail == "Invalid API Key"
#
# @pytest.mark.asyncio
# async def test_verify_api_key_direct_missing(monkeypatch):
#     monkeypatch.setattr(global_settings, 'VALID_API_KEYS', ["directkey"])
#     with pytest.raises(HTTPException) as excinfo:
#         await run_verify_dependency(None, "any_header_name_for_scheme") # Simulate key not found by scheme
#     assert excinfo.value.status_code == 401
#     assert excinfo.value.detail == "Missing API Key"

# The TestClient approach is generally preferred as it tests the dependency in its actual FastAPI integration context.
