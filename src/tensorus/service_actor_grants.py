"""Administrator-issued, membership-fenced service actor grants."""
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import SecretStr
from pydantic import StrictInt
from pydantic import field_validator
from pydantic import model_validator
from tensorus.models import AgentGrantCreate
from tensorus.models import Model
from tensorus.models import MultiDatasetAgentGrantCreate
from tensorus.models import Role
BINDING_FORMAT: Literal["tensorus-service-actor-grant-binding-v1"] = (
    "tensorus-service-actor-grant-binding-v1"
)
STATUS_FORMAT: Literal["tensorus-service-actor-recipient-status-v1"] = (
    "tensorus-service-actor-recipient-status-v1"
)
CAPACITY_FORMAT: Literal["tensorus-service-actor-recipient-capacity-v1"] = (
    "tensorus-service-actor-recipient-capacity-v1"
)
REVIEW_FORMAT: Literal["tensorus-service-actor-recipient-review-v1"] = (
    "tensorus-service-actor-recipient-review-v1"
)
MAX_TRANSITION_GRANTS: Literal[20_000] = 20_000
MAX_CAPACITY_RESPONSE_BYTES: Literal[4096] = 4_096
MAX_REVIEW_RESPONSE_BYTES: Literal[16384] = 16_384
HASH = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
GENERATION = Annotated[StrictInt, Field(ge=1, le=2**63 - 1)]
PROPOSALS = Annotated[StrictInt, Field(ge=1, le=100_000)]
UNITS = Annotated[StrictInt, Field(ge=1, le=1_000_000_000_000)]
class RecipientReview(Model):
    membership_id: UUID
    expected_actor_id: UUID
    expected_generation: GENERATION
    expected_policy_hash: HASH
class RecipientAgentGrantCreate(AgentGrantCreate):
    client_api_key: SecretStr
    expected_actor_id: UUID
    expected_generation: GENERATION
    expected_policy_hash: HASH
    confirmation: Literal["GRANT"]
class RecipientMultiDatasetAgentGrantCreate(MultiDatasetAgentGrantCreate):
    client_api_key: SecretStr
    expected_actor_id: UUID
    expected_generation: GENERATION
    expected_policy_hash: HASH
    confirmation: Literal["GRANT"]
class ActorGrantBinding(Model):
    format: Literal["tensorus-service-actor-grant-binding-v1"] = BINDING_FORMAT
    grant_id: UUID
    membership_id: UUID
    actor_id: UUID
    membership_generation: GENERATION
    membership_role: Role
    identity_policy_hash: HASH
    audience: Annotated[str, Field(strict=True, min_length=1, max_length=256)]
    scope_hash: HASH
    max_plan_proposals: PROPOSALS | None
    max_estimated_execution_units: UNITS | None
    max_direct_ingest_bytes: UNITS | None
class RecipientActor(Model):
    actor_id: UUID
    membership_id: UUID
    display_name: Annotated[str, Field(strict=True, min_length=1, max_length=128)]
    membership_generation: GENERATION
    identity_policy_hash: HASH
class RecipientStatus(Model):
    format: Literal["tensorus-service-actor-recipient-status-v1"] = STATUS_FORMAT
    tenant_id: UUID
    key_id: UUID
    recipient: RecipientActor | None
class RecipientBindingReviewCapabilities(Model):
    format: Literal["tensorus-service-actor-recipient-review-v1"] = REVIEW_FORMAT
    max_response_bytes: Literal[16384] = MAX_REVIEW_RESPONSE_BYTES
    active_unscoped_administrator_required: Literal[True] = True
    expired_and_revoked_included: Literal[True] = True
    historical_binding_only: Literal[True] = True
    scope_recomputed: Literal[False] = False
    current_authorization_proven: Literal[False] = False
    response_atomic_with_changes: Literal[False] = False

    @field_validator("max_response_bytes", mode="before")
    @classmethod
    def exact_limit(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Recipient review limit must be an integer")
        return value

    @field_validator(
        "active_unscoped_administrator_required",
        "expired_and_revoked_included",
        "historical_binding_only",
        "scope_recomputed",
        "current_authorization_proven",
        "response_atomic_with_changes",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Recipient review flags must be booleans")
        return value
class RecipientBindingReview(Model):
    format: Literal["tensorus-service-actor-recipient-review-v1"] = REVIEW_FORMAT
    tenant_id: UUID
    key_id: UUID
    key_name: Annotated[str, Field(strict=True, min_length=1, max_length=128)]
    expires_at: float | None
    revoked_at: str | None
    historical_binding: ActorGrantBinding | None
    historical_issuer_key_id: Annotated[str, Field(strict=True, min_length=1, max_length=36)] | None
    current_identity_policy_hash: HASH | None
    current_audience: Annotated[str, Field(strict=True, min_length=1, max_length=256)]
    current_authorization_proven: Literal[False] = False
    scope_recomputed: Literal[False] = False

    @field_validator("revoked_at")
    @classmethod
    def valid_revocation_time(cls, value: str | None) -> str | None:
        from tensorus.organizations import _timestamp

        return _timestamp(value) if value is not None else None

    @field_validator("current_authorization_proven", "scope_recomputed", mode="before")
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Recipient review flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if (self.historical_binding is None) != (self.historical_issuer_key_id is None):
            raise ValueError("Recipient review issuer evidence is inconsistent")
        if self.historical_binding is not None and (
            self.historical_binding.grant_id != self.key_id or self.expires_at is None
        ):
            raise ValueError("Recipient review binding differs from the credential")
        return self
class RecipientCapacityCapabilities(Model):
    format: Literal["tensorus-service-actor-recipient-capacity-v1"] = CAPACITY_FORMAT
    unrevoked_limit: Annotated[StrictInt, Field(ge=1, le=MAX_TRANSITION_GRANTS)]
    transition_recipient_limit: Literal[20_000] = MAX_TRANSITION_GRANTS
    max_response_bytes: Literal[4096] = MAX_CAPACITY_RESPONSE_BYTES
    expired_unrevoked_counted: Literal[True] = True
    tenant_lock_serializes_participants: Literal[True] = True
    exact_retry_allowed_at_limit: Literal[True] = True
    status_reserves_capacity: Literal[False] = False
    old_or_unconfigured_writers_fenced: Literal[False] = False
    database_work_hard_bounded: Literal[False] = False

    @model_validator(mode="before")
    @classmethod
    def exact_fields(cls, value: Any) -> Any:
        if isinstance(value, dict):
            for name in (
                "expired_unrevoked_counted",
                "tenant_lock_serializes_participants",
                "exact_retry_allowed_at_limit",
                "status_reserves_capacity",
                "old_or_unconfigured_writers_fenced",
                "database_work_hard_bounded",
            ):
                if name in value and type(value[name]) is not bool:
                    raise ValueError("Recipient capacity flags must be booleans")
            for name in ("transition_recipient_limit", "max_response_bytes"):
                if name in value and type(value[name]) is not int:
                    raise ValueError("Recipient capacity limits must be integers")
        return value
class RecipientCapacityStatus(Model):
    format: Literal["tensorus-service-actor-recipient-capacity-v1"] = CAPACITY_FORMAT
    tenant_id: UUID
    unrevoked_limit: Annotated[StrictInt, Field(ge=1, le=MAX_TRANSITION_GRANTS)]
    unrevoked_at_least: Annotated[StrictInt, Field(ge=0, le=MAX_TRANSITION_GRANTS + 1)]
    count_is_lower_bound: bool
    available_slots: Annotated[StrictInt, Field(ge=0, le=MAX_TRANSITION_GRANTS)]
    reserves_capacity: Literal[False] = False

    @model_validator(mode="before")
    @classmethod
    def exact_flags(cls, value: Any) -> Any:
        if isinstance(value, dict):
            for name in ("count_is_lower_bound", "reserves_capacity"):
                if name in value and type(value[name]) is not bool:
                    raise ValueError("Recipient capacity status flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent(self) -> "RecipientCapacityStatus":
        if self.count_is_lower_bound != (
            self.unrevoked_at_least > MAX_TRANSITION_GRANTS
        ) or self.available_slots != max(0, self.unrevoked_limit - self.unrevoked_at_least):
            raise ValueError("Recipient capacity status is inconsistent")
        return self
class RecipientCapabilities(Model):
    format: Literal["tensorus-service-actor-recipient-grant-v1"] = (
        "tensorus-service-actor-recipient-grant-v1"
    )
    max_request_bytes: Annotated[StrictInt, Field(ge=1, le=16_384)]
    max_response_bytes: Literal[16_384] = 16_384
    max_status_bytes: Literal[4096] = 4_096
    max_recipient_grants_per_departure: Literal[20_000] = 20_000
    max_recipient_grants_per_session_transition: Literal[20_000] = 20_000
    max_recipient_grants_per_detachment: Literal[20_000] = 20_000
    active_recipient_session_required_for_issue: Literal[True] = True
    recipient_membership_generation_fenced: Literal[True] = True
    recipient_policy_pinned: Literal[True] = True
    scope_and_allowances_pinned: Literal[True] = True
    exact_client_key_retry: Literal[True] = True
    unscoped_administrator_required: Literal[True] = True
    recipient_session_lifetime_bound: Literal[False] = False
    sender_constrained: Literal[False] = False
    approved_plan_automatically: Literal[False] = False

    @model_validator(mode="before")
    @classmethod
    def exact_fields(cls, value: Any) -> Any:
        if isinstance(value, dict):
            for name in (
                "active_recipient_session_required_for_issue",
                "recipient_membership_generation_fenced",
                "recipient_policy_pinned",
                "scope_and_allowances_pinned",
                "exact_client_key_retry",
                "unscoped_administrator_required",
                "recipient_session_lifetime_bound",
                "sender_constrained",
                "approved_plan_automatically",
            ):
                if name in value and type(value[name]) is not bool:
                    raise ValueError("Recipient capability flags must be booleans")
            for name in (
                "max_response_bytes",
                "max_status_bytes",
                "max_recipient_grants_per_departure",
                "max_recipient_grants_per_session_transition",
                "max_recipient_grants_per_detachment",
            ):
                if name in value and type(value[name]) is not int:
                    raise ValueError("Recipient capability limits must be integers")
        return value
