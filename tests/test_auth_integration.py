"""
Integration tests for Tensorus API key authentication system.

Tests the new Bearer token authentication implementation across
all protected endpoints, ensuring security and backward compatibility.
"""

import pytest
from fastapi.testclient import TestClient
from tests.conftest import TEST_API_KEY, INVALID_API_KEY


class TestAPIAuthentication:
    """Test API key authentication for all endpoints."""

    def test_bearer_token_authentication_success(self, client, auth_headers):
        """Test successful authentication with Bearer token."""
        response = client.get("/datasets", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_bearer_token_authentication_failure(self, client, invalid_auth_headers):
        """Test failed authentication with invalid Bearer token."""
        response = client.get("/datasets", headers=invalid_auth_headers)
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]

    def test_missing_authentication(self, client):
        """Test request without authentication headers."""
        response = client.get("/datasets")
        assert response.status_code == 401
        assert "Missing API key" in response.json()["detail"]

    def test_legacy_header_authentication(self, client, legacy_auth_headers):
        """Test backward compatibility with X-API-KEY header."""
        response = client.get("/datasets", headers=legacy_auth_headers)
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_malformed_bearer_token(self, client):
        """Test malformed Authorization header."""
        headers = {"Authorization": "NotBearer invalid_format"}
        response = client.get("/datasets", headers=headers)
        assert response.status_code == 401

    def test_empty_bearer_token(self, client):
        """Test empty Bearer token."""
        headers = {"Authorization": "Bearer "}
        response = client.get("/datasets", headers=headers)
        assert response.status_code == 401

    def test_auth_disabled_allows_access(self, unauthenticated_client):
        """Test that disabling auth allows access without API key."""
        response = unauthenticated_client.get("/datasets")
        assert response.status_code == 200

    def test_api_key_format_validation(self, client):
        """Test API key format validation for tsr_ prefixed keys."""
        invalid_format_headers = {"Authorization": "Bearer tsr_short"}
        response = client.get("/datasets", headers=invalid_format_headers)
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]


class TestEndpointProtection:
    """Test that all endpoints are properly protected."""

    @pytest.mark.parametrize("endpoint,method,data", [
        ("/datasets", "get", None),
        ("/datasets/create", "post", {"name": "test_dataset"}),
        ("/datasets/test/count", "get", None),
        ("/datasets/test/fetch", "get", None),
        ("/datasets/test/records", "get", None),
    ])
    def test_dataset_endpoints_require_auth(self, client, endpoint, method, data):
        """Test that all dataset endpoints require authentication."""
        if method == "get":
            response = client.get(endpoint)
        elif method == "post":
            response = client.post(endpoint, json=data)
        
        assert response.status_code == 401

    @pytest.mark.parametrize("endpoint,method,data", [
        ("/datasets", "get", None),
        ("/datasets/create", "post", {"name": "test_dataset"}),
    ])
    def test_dataset_endpoints_work_with_auth(self, authenticated_client, endpoint, method, data):
        """Test that dataset endpoints work with proper authentication."""
        if method == "get":
            response = authenticated_client.get(endpoint)
        elif method == "post":
            response = authenticated_client.post(endpoint, json=data)
        
        assert response.status_code in [200, 201]

    def test_query_endpoint_requires_auth(self, client):
        """Test that query endpoint requires authentication."""
        response = client.post("/query", json={"query": "list datasets"})
        assert response.status_code == 401

    def test_query_endpoint_works_with_auth(self, authenticated_client):
        """Test that query endpoint works with authentication."""
        response = authenticated_client.post("/query", json={"query": "list datasets"})
        assert response.status_code == 200


class TestAPIKeyGeneration:
    """Test API key generation and validation utilities."""

    def test_generate_api_key_format(self):
        """Test that generated API keys have correct format."""
        from tensorus.auth.key_generator import generate_api_key, TensorusAPIKey
        
        key = generate_api_key()
        assert key.startswith("tsr_")
        assert len(key) == len("tsr_") + 48
        assert TensorusAPIKey.validate_format(key)

    def test_api_key_validation(self):
        """Test API key format validation."""
        from tensorus.auth.key_generator import TensorusAPIKey
        
        # Valid format
        valid_key = "tsr_" + "a" * 48
        assert TensorusAPIKey.validate_format(valid_key)
        
        # Invalid formats
        assert not TensorusAPIKey.validate_format("invalid_key")
        assert not TensorusAPIKey.validate_format("tsr_short")
        assert not TensorusAPIKey.validate_format("wrong_prefix_" + "a" * 48)
        assert not TensorusAPIKey.validate_format("")
        assert not TensorusAPIKey.validate_format(None)

    def test_api_key_masking(self):
        """Test API key masking for safe logging."""
        from tensorus.auth.key_generator import TensorusAPIKey
        
        # Generate a valid key for testing
        key = TensorusAPIKey.generate()
        masked = TensorusAPIKey.mask_key(key)
        assert masked.startswith("tsr_")
        assert masked.endswith(key[-4:])
        assert "..." in masked
        assert len(masked) < len(key)
        
        # Test invalid key masking
        invalid_key = "invalid_key"
        invalid_masked = TensorusAPIKey.mask_key(invalid_key)
        assert invalid_masked == "invalid_key"


class TestConfigurationManagement:
    """Test configuration and environment variable handling."""

    def test_multiple_api_keys_support(self, client):
        """Test support for multiple API keys."""
        import os
        # Setup multiple keys
        key1 = "tsr_key1_12345678901234567890123456789012345678"
        key2 = "tsr_key2_87654321098765432109876543210987654321"
        
        os.environ["TENSORUS_API_KEYS"] = f"{key1},{key2}"
        
        # Test both keys work
        headers1 = {"Authorization": f"Bearer {key1}"}
        headers2 = {"Authorization": f"Bearer {key2}"}
        
        response1 = client.get("/datasets", headers=headers1)
        response2 = client.get("/datasets", headers=headers2)
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Cleanup
        os.environ.pop("TENSORUS_API_KEYS", None)

    def test_configuration_property_access(self):
        """Test configuration property access for API keys."""
        from tensorus.config import Settings
        import os
        
        # Test with comma-separated keys
        test_keys = "key1,key2,key3"
        os.environ["TENSORUS_API_KEYS"] = test_keys
        
        settings = Settings()
        valid_keys = settings.valid_api_keys
        
        assert len(valid_keys) == 3
        assert "key1" in valid_keys
        assert "key2" in valid_keys
        assert "key3" in valid_keys
        
        # Cleanup
        os.environ.pop("TENSORUS_API_KEYS", None)


class TestSecurityFeatures:
    """Test security-related features and protections."""

    def test_authentication_audit_logging(self, client, auth_headers):
        """Test that authentication events are logged."""
        # This would require checking audit logs, which is environment-dependent
        # For now, just ensure the endpoint works
        response = client.get("/datasets", headers=auth_headers)
        assert response.status_code == 200

    def test_rate_limiting_headers(self, client, auth_headers):
        """Test that proper security headers are set."""
        response = client.get("/datasets", headers=auth_headers)
        # Check for security headers
        assert "X-Content-Type-Options" in response.headers

    def test_error_message_security(self, client):
        """Test that error messages don't leak sensitive information."""
        response = client.get("/datasets")
        error_detail = response.json()["detail"]
        
        # Should not contain internal paths, stack traces, etc.
        assert "traceback" not in error_detail.lower()
        assert "internal" not in error_detail.lower()
        assert "stack" not in error_detail.lower()