"""Actor-bound retries for reviewed workspace membership changes."""
import re
from typing import Annotated
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import StrictInt
from pydantic import field_validator
from pydantic import model_validator
from tensorus.control_request_limits import MAX_CONTROL_REQUEST_BYTES
from tensorus.control_request_limits import WORKSPACE_MEMBER_TRANSITION_PATH
from tensorus.models import Model
from tensorus.models import Role
from tensorus.service_actors import ServiceActorMember
FORMAT: Literal["tensorus-workspace-member-transition-v1"] = (
    "tensorus-workspace-member-transition-v1"
)
PATH: Literal["/v1/workspace/member-transitions"] = WORKSPACE_MEMBER_TRANSITION_PATH
MAX_RESPONSE_BYTES: Literal[16384] = 16384
Status = Literal["invited", "active", "suspended", "removed"]
class MemberTransition(Model):
    request_id: UUID
    member_id: UUID
    expected_issuer: Annotated[str, Field(strict=True, min_length=1, max_length=1024)]
    expected_subject: Annotated[str, Field(strict=True, pattern=r"^[!-~]{1,255}$")]
    expected_generation: Annotated[StrictInt, Field(ge=1, le=2**63 - 2)]
    expected_role: Role
    expected_status: Status
    role: Role
    status: Status
    confirmation: Literal["CHANGE"]

    @field_validator("request_id", "member_id")
    @classmethod
    def non_nil_id(cls, value: UUID) -> UUID:
        if value.int == 0:
            raise ValueError("Non-nil request and membership IDs are required")
        return value

    @field_validator("expected_subject")
    @classmethod
    def exact_subject(cls, value: str) -> str:
        if not re.fullmatch(r"[!-~]{1,255}", value):
            raise ValueError("OIDC subject must be printable ASCII without whitespace")
        return value
class MemberTransitionReceipt(Model):
    format: Literal["tensorus-workspace-member-transition-v1"] = FORMAT
    request_id: UUID
    previous_role: Role
    previous_status: Status
    requested_role: Role
    requested_status: Status
    committed_generation: Annotated[StrictInt, Field(ge=2, le=2**63 - 1)]
    member: ServiceActorMember
    replayed: bool

    @field_validator("replayed", mode="before")
    @classmethod
    def exact_replayed(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Transition replay evidence must be boolean")
        return value

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.request_id.int == 0 or self.member.generation < self.committed_generation:
            raise ValueError("Transition receipt generations are inconsistent")
        if self.member.generation == self.committed_generation and (
            self.member.role != self.requested_role or self.member.status != self.requested_status
        ):
            raise ValueError("Transition receipt state is inconsistent")
        return self
class MemberTransitionCapabilities(Model):
    format: Literal["tensorus-workspace-member-transition-v1"] = FORMAT
    path: Literal["/v1/workspace/member-transitions"] = PATH
    max_request_bytes: Annotated[StrictInt, Field(ge=1, le=MAX_CONTROL_REQUEST_BYTES)]
    max_response_bytes: Literal[16384] = MAX_RESPONSE_BYTES
    active_unscoped_administrator_required: Literal[True] = True
    reviewed_identity_and_generation_required: Literal[True] = True
    exact_retry_same_actor: Literal[True] = True
    retry_returns_current_membership: Literal[True] = True
    retry_reapplies_transition: Literal[False] = False
    revoked_actor_can_retry: Literal[False] = False
    provider_sessions_revoked: Literal[False] = False
    exported_copies_revoked: Literal[False] = False
    evidence_independently_authenticated: Literal[False] = False

    @field_validator("max_response_bytes", mode="before")
    @classmethod
    def exact_limit(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Transition response limit must be an integer")
        return value

    @field_validator(
        "active_unscoped_administrator_required",
        "reviewed_identity_and_generation_required",
        "exact_retry_same_actor",
        "retry_returns_current_membership",
        "retry_reapplies_transition",
        "revoked_actor_can_retry",
        "provider_sessions_revoked",
        "exported_copies_revoked",
        "evidence_independently_authenticated",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Transition capability flags must be booleans")
        return value
