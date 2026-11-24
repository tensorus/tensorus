"""Provider-neutral copies of observed usage and administrator assertions."""
import hashlib
from typing import Annotated
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from tensorus.arrays import canonical_hash
from tensorus.arrays import canonical_json
from tensorus.arrays import decode_unique_json
from tensorus.contracts import ReleaseHash
from tensorus.errors import TensorusError
from tensorus.models import Model
from tensorus.plan_usage_adjustments import AdjustmentReport
from tensorus.plan_usage_handoff import HandoffDocument
FORMAT: Literal["tensorus-plan-adjusted-usage-handoff-v1"] = (
    "tensorus-plan-adjusted-usage-handoff-v1"
)
DOCUMENT_FORMAT: Literal["tensorus-plan-adjusted-usage-handoff-document-v1"] = (
    "tensorus-plan-adjusted-usage-handoff-document-v1"
)
ASSERTION_ID_FORMAT: Literal["tensorus-plan-usage-assertion-identity-v1"] = (
    "tensorus-plan-usage-assertion-identity-v1"
)
MAX_DOCUMENT_BYTES: Literal[393216] = 393216
MAX_RESPONSE_BYTES: Literal[786432] = 786432
MAX_ASSERTIONS: Literal[32] = 32
_TERMINAL = {"succeeded", "failed", "cancelled"}
class AssertionIdentity(Model):
    request_id: UUID
    event_id: ReleaseHash
    payload_sha256: ReleaseHash
def _assertion_identities(report: AdjustmentReport) -> list[AssertionIdentity]:
    return [
        AssertionIdentity(
            request_id=event.request.request_id,
            event_id=canonical_hash(
                {
                    "format": ASSERTION_ID_FORMAT,
                    "tenant_id": str(event.tenant_id),
                    "plan_id": str(event.plan_id),
                    "plan_hash": event.request.expected_plan_hash,
                    "request_id": str(event.request.request_id),
                }
            ),
            payload_sha256=hashlib.sha256(
                canonical_json(event.model_dump(mode="json")).encode("utf-8")
            ).hexdigest(),
        )
        for event in report.adjustments
    ]
class AdjustedHandoffDocument(Model):
    format: Literal["tensorus-plan-adjusted-usage-handoff-document-v1"] = DOCUMENT_FORMAT
    original: HandoffDocument
    assertions: AdjustmentReport
    assertion_identities: list[AssertionIdentity] = Field(max_length=MAX_ASSERTIONS)
    snapshots_atomic: Literal[False] = False
    assertions_independently_verified: Literal[False] = False
    automatic_reconciliation: Literal[False] = False
    provider_submission: Literal[False] = False
    charges_created: Literal[False] = False
    billing_authority: Literal[False] = False
    restored_sources_fenced: Literal[False] = False
    missing_observations_are_zero: Literal[False] = False

    @field_validator(
        "snapshots_atomic",
        "assertions_independently_verified",
        "automatic_reconciliation",
        "provider_submission",
        "charges_created",
        "billing_authority",
        "restored_sources_fenced",
        "missing_observations_are_zero",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Adjusted usage handoff flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_evidence(self) -> Self:
        if self.original.usage.plan_status not in _TERMINAL or self.original.usage.model_dump(
            exclude={"observed_at"}
        ) != self.assertions.usage.model_dump(exclude={"observed_at"}):
            raise ValueError("Observed usage and assertion sources do not match")
        expected = _assertion_identities(self.assertions)
        if self.assertion_identities != expected or len(
            {item.event_id for item in self.assertion_identities}
        ) != len(expected):
            raise ValueError("Usage assertion identities or payloads are inconsistent")
        return self
class AdjustedUsageHandoff(Model):
    format: Literal["tensorus-plan-adjusted-usage-handoff-v1"] = FORMAT
    document_json: Annotated[str, Field(strict=True, min_length=1, max_length=MAX_DOCUMENT_BYTES)]
    document_sha256: ReleaseHash

    @model_validator(mode="after")
    def checked_document(self) -> Self:
        raw = self.document_json.encode("utf-8")
        if len(raw) > MAX_DOCUMENT_BYTES:
            raise ValueError("Adjusted usage handoff document exceeds its byte limit")
        try:
            decoded = decode_unique_json(raw)
            checked = AdjustedHandoffDocument.model_validate_json(raw, strict=True)
            if (
                canonical_json(decoded) != self.document_json
                or canonical_json(checked.model_dump(mode="json")) != self.document_json
                or hashlib.sha256(raw).hexdigest() != self.document_sha256
            ):
                raise ValueError("Adjusted usage handoff document is not exact")
        except (TensorusError, TypeError, ValueError, UnicodeError, RecursionError) as exc:
            raise ValueError("Adjusted usage handoff document is invalid") from exc
        return self
class AdjustedHandoffCapabilities(Model):
    format: Literal["tensorus-plan-adjusted-usage-handoff-v1"] = FORMAT
    document_format: Literal["tensorus-plan-adjusted-usage-handoff-document-v1"] = DOCUMENT_FORMAT
    assertion_identity_format: Literal["tensorus-plan-usage-assertion-identity-v1"] = (
        ASSERTION_ID_FORMAT
    )
    authority: Literal["active_unscoped_tenant_admin"] = "active_unscoped_tenant_admin"
    max_document_bytes: Literal[393216] = MAX_DOCUMENT_BYTES
    max_response_bytes: Literal[786432] = MAX_RESPONSE_BYTES
    max_assertions: Literal[32] = MAX_ASSERTIONS
    terminal_plans_only: Literal[True] = True
    snapshots_atomic: Literal[False] = False
    assertions_independently_verified: Literal[False] = False
    automatic_reconciliation: Literal[False] = False
    provider_submission: Literal[False] = False
    charges_created: Literal[False] = False
    billing_authority: Literal[False] = False
    restored_sources_fenced: Literal[False] = False
    missing_observations_are_zero: Literal[False] = False

    @field_validator("max_document_bytes", "max_response_bytes", "max_assertions", mode="before")
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Adjusted usage handoff limits must be integers")
        return value

    @field_validator(
        "terminal_plans_only",
        "snapshots_atomic",
        "assertions_independently_verified",
        "automatic_reconciliation",
        "provider_submission",
        "charges_created",
        "billing_authority",
        "restored_sources_fenced",
        "missing_observations_are_zero",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Adjusted usage handoff flags must be booleans")
        return value
