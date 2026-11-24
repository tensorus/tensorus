"""Client-held, issuer-bound identities for retrying agent-grant issuance."""
import hashlib
import math
from datetime import datetime
from typing import Literal
from uuid import UUID
from uuid import uuid5
from pydantic import field_validator
from tensorus.models import Model
FORMAT = "tensorus-agent-grant-client-key-v1"
MAX_RESPONSE_BYTES = 16 * 1024
TIMESTAMP_TOLERANCE_SECONDS = 0.000001
def require_grant_lifetime(created_at: str, expires_at: float, lifetime: int) -> None:
    try:
        created = datetime.fromisoformat(created_at)
        if created.utcoffset() is None or not math.isfinite(expires_at):
            raise ValueError("Invalid grant timestamps")
        if abs(expires_at - created.timestamp() - lifetime) > TIMESTAMP_TOLERANCE_SECONDS:
            raise ValueError("Grant lifetime differs from the request")
    except (TypeError, OverflowError, OSError) as exc:
        raise ValueError("Invalid grant timestamps") from exc
class AgentGrantRetryCapabilities(Model):
    format: Literal["tensorus-agent-grant-client-key-v1"]
    credential_option: Literal["client_api_key"]
    max_response_bytes: Literal[16_384]
    client_held_secret: Literal[True]
    requires_unscoped_admin: Literal[True]
    exact_retry: Literal[True]
    issuer_bound: Literal[True]
    lifetime_pinned: Literal[True]
    status_snapshot_may_change: Literal[True]
    revoked_reactivated: Literal[False]
    randomness_verified: Literal[False]

    @field_validator(
        "client_held_secret",
        "requires_unscoped_admin",
        "exact_retry",
        "issuer_bound",
        "lifetime_pinned",
        "status_snapshot_may_change",
        "revoked_reactivated",
        "randomness_verified",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Agent grant retry flags must be booleans")
        return value
def grant_id(tenant_id: str, issuer_key_id: str, token: str) -> str:
    digest = hashlib.sha256(token.encode("ascii")).hexdigest()
    return str(uuid5(UUID(tenant_id), f"{FORMAT}:{issuer_key_id}:{digest}"))
