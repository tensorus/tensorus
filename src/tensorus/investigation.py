"""Bounded, read-only model-assisted investigation; plans require separate proposal and approval."""
from typing import Literal
from pydantic import field_validator
from tensorus.model_provider_http import MAX_RESPONSE_BYTES
from tensorus.models import Model
MAX_INVENTORY: Literal[50] = 50
MAX_DECISIONS: Literal[4] = 4
MAX_PROVIDER_RESPONSE_BYTES: Literal[131072] = MAX_RESPONSE_BYTES
class InvestigationCapabilities(Model):
    format: Literal["tensorus-model-investigation-v1"] = "tensorus-model-investigation-v1"
    route: Literal["/v1/agent/investigate"] = "/v1/agent/investigate"
    max_inventory: Literal[50] = MAX_INVENTORY
    max_decisions: Literal[4] = MAX_DECISIONS
    max_provider_response_bytes: Literal[131072] = MAX_PROVIDER_RESPONSE_BYTES
    explicit_observation_consent_required: Literal[True] = True
    statistics_scope: Literal["raw_stored_values"] = "raw_stored_values"
    automatic_validity_applied: Literal[False] = False
    scoped_agents_permitted: Literal[False] = False
    creates_plan: Literal[False] = False
    approves_or_executes_plan: Literal[False] = False
    provider_identity_verified: Literal[False] = False
    provider_charges_bounded: Literal[False] = False
    response_size_hard_bounded: Literal[False] = False
    end_to_end_deadline_hard_bounded: Literal[False] = False

    @field_validator("max_inventory", "max_decisions", "max_provider_response_bytes", mode="before")
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Investigation limits must be integers")
        return value

    @field_validator(
        "explicit_observation_consent_required",
        "automatic_validity_applied",
        "scoped_agents_permitted",
        "creates_plan",
        "approves_or_executes_plan",
        "provider_identity_verified",
        "provider_charges_bounded",
        "response_size_hard_bounded",
        "end_to_end_deadline_hard_bounded",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Investigation flags must be booleans")
        return value
