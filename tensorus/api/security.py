"""
Enhanced Tensorus API Security Module

Implements industry-standard Bearer token authentication following patterns from
OpenAI, Pinecone, and other major AI/ML services.

Key Features:
- Bearer token authentication (Authorization: Bearer <token>)
- Secure API key format validation (tsr_ prefix)
- Backward compatibility with existing API keys
- Comprehensive audit logging
- JWT authentication support (optional)
"""

from fastapi import Security, HTTPException, status, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any, List
import logging

from tensorus.config import settings
from tensorus.audit import log_audit_event
from jose import jwt, JWTError
import requests

logger = logging.getLogger(__name__)


class MutableAPIKeyHeader(APIKeyHeader):
    """APIKeyHeader that allows updating the header name for testing."""

    @property
    def name(self) -> str:  # type: ignore[override]
        return self.model.name

    @name.setter
    def name(self, value: str) -> None:  # type: ignore[override]
        self.model.name = value


# --- Enhanced Bearer Token Authentication ---
bearer_scheme = HTTPBearer(auto_error=False)

# Legacy API key header support for backward compatibility
api_key_header_auth = MutableAPIKeyHeader(name="X-API-KEY", auto_error=False)


async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    legacy_api_key: Optional[str] = Security(api_key_header_auth)
) -> str:
    """
    Verify API key using Bearer token authentication (OpenAI/Pinecone style).
    
    Supports both:
    1. Bearer token: Authorization: Bearer tsr_...
    2. Legacy header: X-API-KEY: tsr_... (backward compatibility)
    
    Args:
        credentials: Bearer token from Authorization header
        legacy_api_key: API key from X-API-KEY header (legacy)
        
    Returns:
        str: The validated API key
        
    Raises:
        HTTPException: 401 if authentication fails, 503 if auth not configured
    """
    if not settings.AUTH_ENABLED:
        # Authentication disabled - allow all requests
        logger.debug("API authentication disabled, allowing request")
        return "auth_disabled"
    
    # Get list of valid API keys
    valid_keys = settings.valid_api_keys
    if not valid_keys:
        logger.error("API authentication enabled but no valid API keys configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication not configured"
        )
    
    # Extract API key from Bearer token or legacy header
    api_key = None
    auth_method = None
    
    if credentials and credentials.scheme.lower() == "bearer":
        api_key = credentials.credentials
        auth_method = "bearer"
    elif legacy_api_key:
        api_key = legacy_api_key
        auth_method = "legacy"
    
    if not api_key:
        logger.warning("API request missing authentication")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Use 'Authorization: Bearer <api_key>' header.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Validate API key format (if it's a new format key)
    try:
        from tensorus.auth.key_generator import TensorusAPIKey
        
        if api_key.startswith("tsr_") and not TensorusAPIKey.validate_format(api_key):
            logger.warning(f"Invalid API key format: {TensorusAPIKey.mask_key(api_key)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key format",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except ImportError:
        # Auth module not available, skip format validation
        pass
    
    # Validate API key against configured keys
    if api_key not in valid_keys:
        logger.warning(f"Invalid API key attempt: {_mask_key(api_key)} via {auth_method}")
        log_audit_event(
            action="API_AUTH_FAILED",
            details={
                "method": auth_method,
                "key_prefix": api_key[:7] if len(api_key) > 7 else "short_key",
                "reason": "invalid_key"
            }
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Success - log and return
    logger.debug(f"API authentication successful via {auth_method}: {_mask_key(api_key)}")
    log_audit_event(
        action="API_AUTH_SUCCESS",
        user=_mask_key(api_key),
        details={"method": auth_method}
    )
    
    return api_key


async def verify_api_key_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    legacy_api_key: Optional[str] = Security(api_key_header_auth)
) -> Optional[str]:
    """
    Optional API key verification for public endpoints.
    
    Returns:
        Optional[str]: The API key if provided and valid, None if not provided
        
    Raises:
        HTTPException: Only if key is provided but invalid
    """
    if not settings.AUTH_ENABLED:
        return None
    
    # If no authentication provided, return None (allowed for public endpoints)
    if not credentials and not legacy_api_key:
        return None
    
    # If authentication provided, it must be valid
    return await verify_api_key(credentials, legacy_api_key)


def _mask_key(key: str) -> str:
    """Mask API key for safe logging"""
    if not key:
        return "empty_key"
    
    try:
        from tensorus.auth.key_generator import TensorusAPIKey
        return TensorusAPIKey.mask_key(key)
    except ImportError:
        # Fallback masking
        if len(key) <= 8:
            return "short_key"
        return f"{key[:7]}...{key[-4:]}"


# --- Legacy API Key Authentication (Backward Compatibility) ---
async def verify_api_key_legacy(api_key: Optional[str] = Security(api_key_header_auth)):
    """
    Legacy API key verification for backward compatibility.
    
    This function maintains the original behavior for existing endpoints
    that haven't been migrated to Bearer token authentication yet.
    """
    if not settings.valid_api_keys:
        # If no API keys are configured, treat as no valid keys configured.
        pass
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key"
        )
    
    valid_keys = settings.valid_api_keys
    if not valid_keys or api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    
    return api_key


# --- JWT Token Authentication (Enhanced) ---
oauth2_scheme = HTTPBearer(auto_error=False)


async def verify_jwt_token(token: Optional[HTTPAuthorizationCredentials] = Security(oauth2_scheme)) -> Dict[str, Any]:
    """
    Enhanced JWT token verification with proper error handling.
    
    This implementation maintains the existing JWT functionality while
    improving error messages and security.
    """
    if not settings.AUTH_JWT_ENABLED:
        if settings.AUTH_DEV_MODE_ALLOW_DUMMY_JWT and token:
            return {
                "sub": "dummy_jwt_user_jwt_disabled_but_dev_mode",
                "username": "dummy_dev_jwt",
                "token_type": "dummy_bearer_dev"
            }

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JWT authentication is not enabled or configured for this service."
        )

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated via JWT (No token provided)"
        )

    if settings.AUTH_DEV_MODE_ALLOW_DUMMY_JWT:
        return {
            "sub": "dummy_jwt_user",
            "username": "dummy_jwt_user", 
            "token_type": "dummy_bearer",
            "token_value": token.credentials
        }

    if not settings.AUTH_JWT_JWKS_URI:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JWKS URI not configured"
        )

    try:
        jwks_data = requests.get(settings.AUTH_JWT_JWKS_URI).json()
    except Exception as e:
        log_audit_event("JWT_VALIDATION_FAILED", details={"error": f"Failed fetching JWKS: {e}"})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to fetch JWKS"
        )

    unverified_header = jwt.get_unverified_header(token.credentials)
    rsa_key = None
    for key in jwks_data.get("keys", []):
        if key.get("kid") == unverified_header.get("kid"):
            rsa_key = key
            break

    if rsa_key is None:
        log_audit_event("JWT_VALIDATION_FAILED", details={"error": "kid not found"})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token header"
        )

    try:
        claims = jwt.decode(
            token.credentials,
            rsa_key,
            algorithms=[settings.AUTH_JWT_ALGORITHM],
            issuer=settings.AUTH_JWT_ISSUER,
            audience=settings.AUTH_JWT_AUDIENCE,
        )
    except JWTError as e:
        log_audit_event("JWT_VALIDATION_FAILED", details={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid JWT token"
        )

    log_audit_event(
        "JWT_VALIDATION_SUCCESS",
        user=claims.get("sub"),
        details={"issuer": claims.get("iss"), "aud": claims.get("aud")}
    )
    return claims


# --- Development and Testing Utilities ---
def get_valid_api_keys() -> List[str]:
    """Get list of valid API keys for development/testing"""
    return settings.valid_api_keys


def is_auth_enabled() -> bool:
    """Check if authentication is enabled"""
    return settings.AUTH_ENABLED


# Example usage:
# @app.get("/protected", dependencies=[Depends(verify_api_key)])
# async def protected_endpoint():
#     return {"message": "This endpoint requires authentication"}
#
# @app.get("/optional")
# async def optional_auth_endpoint(api_key: Optional[str] = Depends(verify_api_key_optional)):
#     if api_key:
#         return {"message": "Authenticated user", "key": _mask_key(api_key)}
#     return {"message": "Anonymous user"}