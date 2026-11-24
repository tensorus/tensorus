"""Optional process-local admission for external-model HTTP journeys."""
from typing import Annotated
from typing import Literal
from typing import Self
from pydantic import Field
from pydantic import StrictInt
from pydantic import field_validator
from pydantic import model_validator
from tensorus.models import Model
from tensorus.provider_admission import MAX_SLOTS
FORMAT: Literal["tensorus-model-provider-admission-v1"] = "tensorus-model-provider-admission-v1"
TENANT_FORMAT: Literal["tensorus-model-provider-tenant-admission-v1"] = (
    "tensorus-model-provider-tenant-admission-v1"
)
class ModelProviderAdmissionCapabilities(Model):
    format: Literal["tensorus-model-provider-admission-v1"] = FORMAT
    slots: Annotated[StrictInt, Field(ge=1, le=MAX_SLOTS)]
    global_request_slots: Annotated[StrictInt, Field(ge=4, le=1024)]
    shared_investigation_and_planning: Literal[True] = True
    after_authentication: Literal[True] = True
    after_body_staging: Literal[True] = True
    before_inventory: Literal[True] = True
    holds_handler: Literal[True] = True
    single_api_process: Literal[True] = True
    fair_queue: Literal[False] = False
    physical_memory_reserved: Literal[False] = False
    provider_hard_deadline: Literal[False] = False
    response_delivery_covered: Literal[False] = False
    denials_in_process_metrics: Literal[True] = True
    counters_durable: Literal[False] = False

    @field_validator("slots", "global_request_slots", mode="before")
    @classmethod
    def exact_integers(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Model provider admission limits must be integers")
        return value

    @field_validator(
        "shared_investigation_and_planning",
        "after_authentication",
        "after_body_staging",
        "before_inventory",
        "holds_handler",
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
            raise ValueError("Model provider admission flags must be booleans")
        return value

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        if self.slots >= self.global_request_slots:
            raise ValueError("Model provider slots must be fewer than active request slots")
        return self
class ModelProviderTenantAdmissionCapabilities(Model):
    format: Literal["tensorus-model-provider-tenant-admission-v1"] = TENANT_FORMAT
    tenant_slots: Annotated[StrictInt, Field(ge=1, le=MAX_SLOTS)]
    shared_slots: Annotated[StrictInt, Field(ge=1, le=MAX_SLOTS)]
    global_request_slots: Annotated[StrictInt, Field(ge=4, le=1024)]
    shared_investigation_and_planning: Literal[True] = True
    catalog_workspace_identity: Literal[True] = True
    after_authentication: Literal[True] = True
    after_body_staging: Literal[True] = True
    before_inventory: Literal[True] = True
    holds_handler: Literal[True] = True
    single_api_process: Literal[True] = True
    fair_queue: Literal[False] = False
    physical_memory_reserved: Literal[False] = False
    provider_hard_deadline: Literal[False] = False
    response_delivery_covered: Literal[False] = False
    denials_in_process_metrics: Literal[True] = True
    tenant_labels_in_metrics: Literal[False] = False
    counters_durable: Literal[False] = False

    @field_validator("tenant_slots", "shared_slots", "global_request_slots", mode="before")
    @classmethod
    def exact_integers(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Model provider tenant admission limits must be integers")
        return value

    @field_validator(
        "shared_investigation_and_planning",
        "catalog_workspace_identity",
        "after_authentication",
        "after_body_staging",
        "before_inventory",
        "holds_handler",
        "single_api_process",
        "fair_queue",
        "physical_memory_reserved",
        "provider_hard_deadline",
        "response_delivery_covered",
        "denials_in_process_metrics",
        "tenant_labels_in_metrics",
        "counters_durable",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Model provider tenant admission flags must be booleans")
        return value

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        if self.tenant_slots > self.shared_slots or self.shared_slots >= self.global_request_slots:
            raise ValueError("Model provider tenant admission limits are inconsistent")
        return self
