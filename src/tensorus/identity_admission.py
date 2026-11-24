"""Optional process-local admission for OIDC provider calls."""
from typing import Annotated
from typing import Literal
from typing import Self
from pydantic import Field
from pydantic import StrictInt
from pydantic import field_validator
from pydantic import model_validator
from tensorus.models import Model
from tensorus.provider_admission import MAX_SLOTS
FORMAT: Literal["tensorus-identity-provider-admission-v1"] = (
    "tensorus-identity-provider-admission-v1"
)
BROWSER_SEPARATE_LIMIT: Literal[4] = 4
class IdentityProviderAdmissionCapabilities(Model):
    format: Literal["tensorus-identity-provider-admission-v1"] = FORMAT
    slots: Annotated[StrictInt, Field(ge=1, le=MAX_SLOTS)]
    global_request_slots: Annotated[StrictInt, Field(ge=4, le=1024)]
    browser_separate_limit: Literal[4] = BROWSER_SEPARATE_LIMIT
    shared_assertion_and_browser: Literal[True] = True
    before_provider_calls: Literal[True] = True
    after_body_staging: Literal[True] = True
    single_api_process: Literal[True] = True
    fair_queue: Literal[False] = False
    physical_memory_reserved: Literal[False] = False
    provider_hard_deadline: Literal[False] = False
    response_delivery_covered: Literal[False] = False
    denials_in_process_metrics: Literal[True] = True
    counters_durable: Literal[False] = False

    @field_validator("slots", "global_request_slots", "browser_separate_limit", mode="before")
    @classmethod
    def exact_integers(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Identity provider admission limits must be integers")
        return value

    @field_validator(
        "shared_assertion_and_browser",
        "before_provider_calls",
        "after_body_staging",
        "single_api_process",
        "fair_queue",
        "physical_memory_reserved",
        "provider_hard_deadline",
        "response_delivery_covered",
        "denials_in_process_metrics",
        "counters_durable",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Identity provider admission flags must be booleans")
        return value

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        if self.slots >= self.global_request_slots:
            raise ValueError("Identity provider slots must be fewer than active request slots")
        return self
