"""Capabilities for workspace aggregate response fencing and admission."""

from typing import Literal

from pydantic import Field, field_validator

from tensorus.models import Model
from tensorus.scientific_read_admission import WORKSPACE_OVERVIEW_ROUTES


class WorkspaceOverviewResponseFenceCapabilities(Model):
    format: Literal["tensorus-workspace-overview-response-fence-v1"] = (
        "tensorus-workspace-overview-response-fence-v1"
    )
    routes: list[str] = Field(default_factory=lambda: list(WORKSPACE_OVERVIEW_ROUTES))
    unscoped_reader_required: Literal[True] = True
    caller_rechecked_after_serialization: Literal[True] = True
    response_delivery_atomic: Literal[False] = False
    aggregate_snapshot_retained: Literal[False] = False

    @field_validator("routes")
    @classmethod
    def exact_routes(cls, value: list[str]) -> list[str]:
        if value != list(WORKSPACE_OVERVIEW_ROUTES):
            raise ValueError("Workspace overview response fence routes are inconsistent")
        return value

    @field_validator(
        "unscoped_reader_required",
        "caller_rechecked_after_serialization",
        "response_delivery_atomic",
        "aggregate_snapshot_retained",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Workspace overview response fence flags must be booleans")
        return value


class WorkspaceOverviewAdmissionCapabilities(Model):
    format: Literal["tensorus-workspace-overview-admission-v1"] = (
        "tensorus-workspace-overview-admission-v1"
    )
    routes: list[str] = Field(default_factory=lambda: list(WORKSPACE_OVERVIEW_ROUTES))
    after_authentication_and_unscoped_check: Literal[True] = True
    shared_scientific_read_pool: Literal[True] = True
    single_api_process: Literal[True] = True
    active_until_response_and_handler_finish: Literal[True] = True
    post_serialization_recheck_admitted: Literal[True] = True
    individual_response_bytes_bounded: Literal[False] = False
    fair_queue: Literal[False] = False
    database_work_hard_bounded: Literal[False] = False
    physical_memory_reserved: Literal[False] = False

    @field_validator("routes")
    @classmethod
    def exact_routes(cls, value: list[str]) -> list[str]:
        if value != list(WORKSPACE_OVERVIEW_ROUTES):
            raise ValueError("Workspace overview admission routes are inconsistent")
        return value

    @field_validator(
        "after_authentication_and_unscoped_check",
        "shared_scientific_read_pool",
        "single_api_process",
        "active_until_response_and_handler_finish",
        "post_serialization_recheck_admitted",
        "individual_response_bytes_bounded",
        "fair_queue",
        "database_work_hard_bounded",
        "physical_memory_reserved",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Workspace overview admission flags must be booleans")
        return value
