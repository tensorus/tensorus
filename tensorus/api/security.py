from fastapi import Security, HTTPException, status, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any # Added Dict, Any for JWT payload

from tensorus.config import settings
from tensorus.audit import log_audit_event
from jose import jwt, JWTError
import requests


class MutableAPIKeyHeader(APIKeyHeader):
    """APIKeyHeader that allows updating the header name for testing."""

    @property
    def name(self) -> str:  # type: ignore[override]
        return self.model.name

    @name.setter
    def name(self, value: str) -> None:  # type: ignore[override]
        self.model.name = value

# --- API Key Authentication ---
api_key_header_auth = MutableAPIKeyHeader(name=settings.API_KEY_HEADER_NAME, auto_error=False)

async def verify_api_key(api_key: Optional[str] = Security(api_key_header_auth)):
    """
    Verifies the API key provided in the request header.
    Raises HTTPException if the API key is missing or invalid.
    Returns the API key string if valid.
    """
    if not settings.VALID_API_KEYS:
        # If no API keys are configured, treat as no valid keys configured.
        # Endpoints depending on this will be inaccessible unless keys are provided.
        pass
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key"
        )
    if not settings.VALID_API_KEYS or api_key not in settings.VALID_API_KEYS:
        # If list is empty OR key is not in the list
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key


# --- JWT Token Authentication (Conceptual) ---
oauth2_scheme = HTTPBearer(auto_error=False) # auto_error=False means it won't raise error if token is missing

async def verify_jwt_token(token: Optional[HTTPAuthorizationCredentials] = Security(oauth2_scheme)) -> Dict[str, Any]:
    """
    Conceptual JWT token verification dependency.
    - If JWT auth is disabled, denies access if an endpoint specifically requires it (unless in dev dummy mode).
    - If enabled and in dev dummy mode, allows any bearer token string.
    - If enabled and not in dev dummy mode, raises 501 Not Implemented (actual validation needed here).
    """
    if not settings.AUTH_JWT_ENABLED:
        # If JWT auth is globally disabled:
        # If an endpoint *still* tries to use this JWT verifier, it means it expects JWT.
        # So, deny access because the system isn't configured for it.
        # However, if AUTH_DEV_MODE_ALLOW_DUMMY_JWT is true, we might let it pass for local dev convenience
        # even if AUTH_JWT_ENABLED is false (treating dummy mode as an override).
        if settings.AUTH_DEV_MODE_ALLOW_DUMMY_JWT and token: # Token provided, dummy mode on
             return {"sub": "dummy_jwt_user_jwt_disabled_but_dev_mode", "username": "dummy_dev_jwt", "token_type": "dummy_bearer_dev"}

        # Standard behavior: if JWT is not enabled, this verifier should fail if called.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, # Or 403 Forbidden
            detail="JWT authentication is not enabled or configured for this service."
        )

    # JWT Auth is enabled, proceed.
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated via JWT (No token provided)"
        )

    if settings.AUTH_DEV_MODE_ALLOW_DUMMY_JWT:
        # In dev dummy mode, allow any token value.
        return {"sub": "dummy_jwt_user", "username": "dummy_jwt_user", "token_type": "dummy_bearer", "token_value": token.credentials}

    if not settings.AUTH_JWT_JWKS_URI:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="JWKS URI not configured")

    try:
        jwks_data = requests.get(settings.AUTH_JWT_JWKS_URI).json()
    except Exception as e:  # pragma: no cover - network issues
        log_audit_event("JWT_VALIDATION_FAILED", details={"error": f"Failed fetching JWKS: {e}"})
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Unable to fetch JWKS")

    unverified_header = jwt.get_unverified_header(token.credentials)
    rsa_key = None
    for key in jwks_data.get("keys", []):
        if key.get("kid") == unverified_header.get("kid"):
            rsa_key = key
            break

    if rsa_key is None:
        log_audit_event("JWT_VALIDATION_FAILED", details={"error": "kid not found"})
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token header")

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
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid JWT token")

    log_audit_event("JWT_VALIDATION_SUCCESS", user=claims.get("sub"), details={"issuer": claims.get("iss"), "aud": claims.get("aud")})
    return claims

# Example of how to use it in an endpoint:
# from fastapi import Depends
# from .security import verify_api_key
#
# @router.post("/some_protected_route", dependencies=[Depends(verify_api_key)])
# async def protected_route_function():
#     # ...
#
# Or if you need the key value:
# @router.post("/another_route")
# async def another_route_function(api_key: str = Depends(verify_api_key)):
#     # api_key variable now holds the validated key
#     # ...
