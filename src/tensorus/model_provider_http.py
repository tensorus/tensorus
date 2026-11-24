"""Shared, bounded HTTP response handling for external-model calls."""
from typing import Annotated
from typing import Literal
from pydantic import Field
from pydantic import StrictInt
from pydantic import field_validator
from tensorus.models import Model
MAX_RESPONSE_BYTES: Literal[131072] = 131072
MAX_TIMEOUT_SECONDS = 300
class ModelProviderTimeoutCapabilities(Model):
    format: Literal["tensorus-model-provider-cooperative-timeout-v1"] = (
        "tensorus-model-provider-cooperative-timeout-v1"
    )
    seconds_per_call: Annotated[StrictInt, Field(ge=1, le=MAX_TIMEOUT_SECONDS)]
    covered_calls: Literal["goal_planning_and_investigation"] = "goal_planning_and_investigation"
    asynchronous_network_cancellation: Literal[True] = True
    existing_inactivity_timeouts_apply: Literal[True] = True
    synchronous_caller_required: Literal[True] = True
    catalog_and_validation_covered: Literal[False] = False
    multi_call_journey_covered: Literal[False] = False
    hard_deadline: Literal[False] = False
    resolver_threads_terminated: Literal[False] = False
    remote_processing_terminated: Literal[False] = False
    provider_charges_bounded: Literal[False] = False
    fleet_configuration_verified: Literal[False] = False

    @field_validator("seconds_per_call", mode="before")
    @classmethod
    def exact_seconds(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Model provider timeout must be an integer")
        return value

    @field_validator(
        "asynchronous_network_cancellation",
        "existing_inactivity_timeouts_apply",
        "synchronous_caller_required",
        "catalog_and_validation_covered",
        "multi_call_journey_covered",
        "hard_deadline",
        "resolver_threads_terminated",
        "remote_processing_terminated",
        "provider_charges_bounded",
        "fleet_configuration_verified",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Model provider timeout flags must be booleans")
        return value
