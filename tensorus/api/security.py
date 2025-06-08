from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from typing import Optional

from tensorus.config import settings

api_key_header_auth = APIKeyHeader(name=settings.API_KEY_HEADER_NAME, auto_error=False)

async def verify_api_key(api_key: Optional[str] = Security(api_key_header_auth)):
    """
    Verifies the API key provided in the request header.
    Raises HTTPException if the API key is missing or invalid.
    Returns the API key string if valid.
    """
    if not settings.VALID_API_KEYS:
        # If no API keys are configured, allow access (development mode or no auth intended)
        # Or, to enforce keys always, raise HTTPException here or configure a default deny key.
        # For this exercise, if list is empty, we'll consider it as "no auth configured".
        # However, a production system should probably deny if the list is empty but auth is intended.
        # Let's change this to: if VALID_API_KEYS is empty, it means no keys are valid (unless specific "allow all" logic).
        # For this subtask, if VALID_API_KEYS is empty, let's make it so NO key is valid, enforcing configuration.
        # To disable auth, one would comment out the `Depends(verify_api_key)` from endpoints.
        # If settings.VALID_API_KEYS is an empty list, no key will ever be valid.
        pass # Fall through to checks below. If list is empty, key will not be in it.


    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key"
        )

    if api_key not in settings.VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key # Return the key itself, can be used for logging user/actor.

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
