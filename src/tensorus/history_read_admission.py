"""Capability for optional retained-history read concurrency admission."""

from typing import Literal

from pydantic import Field, field_validator

from tensorus.history_summary_policy import HISTORY_SUMMARY_ROUTE
from tensorus.models import Model
from tensorus.scientific_read_admission import HISTORY_READ_ROUTES


class HistoryReadAdmissionCapabilities(Model):
    format: Literal["tensorus-history-read-admission-v1"] = "tensorus-history-read-admission-v1"
    routes: list[str] = Field(default_factory=lambda: list(HISTORY_READ_ROUTES))
    after_authentication_and_coarse_authorization: Literal[True] = True
    resource_authorization_in_handler: Literal[True] = True
    shared_scientific_read_pool: Literal[True] = True
    single_api_process: Literal[True] = True
    active_until_response_and_handler_finish: Literal[True] = True
    mutations_admitted: Literal[False] = False
    individual_response_bytes_bounded: Literal[False] = False
    fair_queue: Literal[False] = False
    database_work_hard_bounded: Literal[False] = False
    physical_memory_reserved: Literal[False] = False

    @field_validator("routes")
    @classmethod
    def exact_routes(cls, value: list[str]) -> list[str]:
        if value != list(HISTORY_READ_ROUTES):
            raise ValueError("History read admission routes are inconsistent")
        return value

    @field_validator(
        "after_authentication_and_coarse_authorization",
        "resource_authorization_in_handler",
        "shared_scientific_read_pool",
        "single_api_process",
        "active_until_response_and_handler_finish",
        "mutations_admitted",
        "individual_response_bytes_bounded",
        "fair_queue",
        "database_work_hard_bounded",
        "physical_memory_reserved",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("History read admission flags must be booleans")
        return value


class HistoryListAdmissionCapabilities(Model):
    format: Literal["tensorus-history-list-admission-v1"] = "tensorus-history-list-admission-v1"
    route: Literal["GET /v1/datasets/{dataset}/release-history/imports"] = HISTORY_SUMMARY_ROUTE
    after_authentication_and_coarse_authorization: Literal[True] = True
    resource_authorization_in_handler: Literal[True] = True
    shared_scientific_read_pool: Literal[True] = True
    single_api_process: Literal[True] = True
    active_until_response_and_handler_finish: Literal[True] = True
    final_caller_recheck: Literal[True] = True
    does_not_enable_scoped_filter: Literal[True] = True
    mutations_admitted: Literal[False] = False
    individual_response_bytes_bounded: Literal[False] = False
    fair_queue: Literal[False] = False
    database_work_hard_bounded: Literal[False] = False
    physical_memory_reserved: Literal[False] = False

    @field_validator(
        "after_authentication_and_coarse_authorization",
        "resource_authorization_in_handler",
        "shared_scientific_read_pool",
        "single_api_process",
        "active_until_response_and_handler_finish",
        "final_caller_recheck",
        "does_not_enable_scoped_filter",
        "mutations_admitted",
        "individual_response_bytes_bounded",
        "fair_queue",
        "database_work_hard_bounded",
        "physical_memory_reserved",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("History list admission flags must be booleans")
        return value
