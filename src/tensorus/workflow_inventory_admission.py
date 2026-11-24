"""Capability for opt-in workflow metadata read concurrency admission."""

from typing import Literal

from pydantic import Field, field_validator

from tensorus.models import Model
from tensorus.scientific_read_admission import WORKFLOW_INVENTORY_ROUTES


class WorkflowInventoryAdmissionCapabilities(Model):
    format: Literal["tensorus-workflow-inventory-admission-v1"] = (
        "tensorus-workflow-inventory-admission-v1"
    )
    routes: list[str] = Field(default_factory=lambda: list(WORKFLOW_INVENTORY_ROUTES))
    after_authentication_and_unscoped_check: Literal[True] = True
    shared_scientific_read_pool: Literal[True] = True
    single_api_process: Literal[True] = True
    active_until_response_and_handler_finish: Literal[True] = True
    mutations_and_preflights_admitted: Literal[False] = False
    individual_response_bytes_bounded: Literal[False] = False
    fair_queue: Literal[False] = False
    database_work_hard_bounded: Literal[False] = False
    physical_memory_reserved: Literal[False] = False

    @field_validator("routes")
    @classmethod
    def exact_routes(cls, value: list[str]) -> list[str]:
        if value != list(WORKFLOW_INVENTORY_ROUTES):
            raise ValueError("Workflow inventory admission routes are inconsistent")
        return value

    @field_validator(
        "after_authentication_and_unscoped_check",
        "shared_scientific_read_pool",
        "single_api_process",
        "active_until_response_and_handler_finish",
        "mutations_and_preflights_admitted",
        "individual_response_bytes_bounded",
        "fair_queue",
        "database_work_hard_bounded",
        "physical_memory_reserved",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Workflow inventory admission flags must be booleans")
        return value
