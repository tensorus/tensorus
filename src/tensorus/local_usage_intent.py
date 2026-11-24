"""Catalog-audited intent before local fixture file publication."""
from typing import Annotated
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from tensorus.arrays import canonical_hash
from tensorus.local_usage_delivery import MAX_RESPONSE_BYTES
from tensorus.local_usage_delivery import FixtureDeliveryRequest
from tensorus.local_usage_delivery import FixtureReceipt
from tensorus.local_usage_delivery import FixtureReviewDocument
from tensorus.local_usage_delivery import review_matches_request
from tensorus.models import Model
from tensorus.plan_usage_adjustments import _timestamp
FORMAT: Literal["tensorus-local-usage-fixture-intent-v1"] = "tensorus-local-usage-fixture-intent-v1"
CAPABILITY_FORMAT: Literal["tensorus-local-usage-fixture-intents-v1"] = (
    "tensorus-local-usage-fixture-intents-v1"
)
MAX_INTENTS: Literal[1000] = 1000
class FixtureIntent(Model):
    format: Literal["tensorus-local-usage-fixture-intent-v1"] = FORMAT
    tenant_id: UUID
    actor_id: Annotated[str, Field(strict=True, min_length=1, max_length=36)]
    request: FixtureDeliveryRequest
    review: FixtureReviewDocument
    recorded_at: str
    files_checked: Literal[False] = False
    completion_assessed: Literal[False] = False
    prior_partial_attributed: Literal[False] = False
    source_authenticity_verified: Literal[False] = False
    authorizes_delivery: Literal[False] = False
    billing_authority: Literal[False] = False

    _recorded_at = field_validator("recorded_at")(_timestamp)

    @field_validator("actor_id")
    @classmethod
    def checked_actor(cls, value: str) -> str:
        return FixtureReceipt.checked_actor(value)

    @field_validator(
        "files_checked",
        "completion_assessed",
        "prior_partial_attributed",
        "source_authenticity_verified",
        "authorizes_delivery",
        "billing_authority",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Fixture intent flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if (
            not review_matches_request(self.review, self.request, self.tenant_id)
            or self.request.expected_review_sha256
            != canonical_hash(self.review.model_dump(mode="json"))
            or self.review.event_count == 0
            or (self.review.unknown_coverage and not self.request.acknowledge_unknown)
        ):
            raise ValueError("Fixture intent does not match the reviewed delivery")
        return self
class FixtureIntentCapabilities(Model):
    format: Literal["tensorus-local-usage-fixture-intents-v1"] = CAPABILITY_FORMAT
    intent_format: Literal["tensorus-local-usage-fixture-intent-v1"] = FORMAT
    authority: Literal["active_unscoped_tenant_admin"] = "active_unscoped_tenant_admin"
    max_intents_per_tenant: Literal[1000] = MAX_INTENTS
    max_response_bytes: Literal[262144] = MAX_RESPONSE_BYTES
    catalog_intent_before_new_files: Literal[True] = True
    request_and_actor_bound: Literal[True] = True
    prior_partial_attributed: Literal[False] = False
    completion_assessed: Literal[False] = False
    source_authenticity_verified: Literal[False] = False
    authorizes_delivery: Literal[False] = False
    billing_authority: Literal[False] = False

    @field_validator("max_intents_per_tenant", "max_response_bytes", mode="before")
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Fixture intent limits must be integers")
        return value

    @field_validator(
        "catalog_intent_before_new_files",
        "request_and_actor_bound",
        "prior_partial_attributed",
        "completion_assessed",
        "source_authenticity_verified",
        "authorizes_delivery",
        "billing_authority",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Fixture intent capability flags must be booleans")
        return value
