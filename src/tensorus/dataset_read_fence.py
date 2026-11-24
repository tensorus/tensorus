"""Opt-in response rechecks and shared admission for dataset descriptors."""

from typing import Literal

from pydantic import Field, field_validator

from tensorus.models import Model
from tensorus.scientific_read_admission import DATASET_READ_FENCE_ROUTES


class DatasetReadFenceCapabilities(Model):
    format: Literal["tensorus-dataset-read-response-fence-v1"] = (
        "tensorus-dataset-read-response-fence-v1"
    )
    routes: list[str] = Field(default_factory=lambda: list(DATASET_READ_FENCE_ROUTES))
    rechecks_authority_and_captured_response: Literal[True] = True
    shared_scientific_read_pool: Literal[True] = True
    single_api_process: Literal[True] = True
    active_until_response_and_handler_finish: Literal[True] = True
    rereads_tensor_payloads: Literal[False] = False
    other_dataset_routes_covered: Literal[False] = False
    writes_admitted: Literal[False] = False
    individual_response_bytes_bounded: Literal[False] = False
    database_work_hard_bounded: Literal[False] = False
    physical_memory_reserved: Literal[False] = False
    fair_queue: Literal[False] = False
    atomic_with_delivery: Literal[False] = False

    @field_validator("routes")
    @classmethod
    def exact_routes(cls, value: list[str]) -> list[str]:
        if value != list(DATASET_READ_FENCE_ROUTES):
            raise ValueError("Dataset read response fence routes are inconsistent")
        return value

    @field_validator(
        "rechecks_authority_and_captured_response",
        "shared_scientific_read_pool",
        "single_api_process",
        "active_until_response_and_handler_finish",
        "rereads_tensor_payloads",
        "other_dataset_routes_covered",
        "writes_admitted",
        "individual_response_bytes_bounded",
        "database_work_hard_bounded",
        "physical_memory_reserved",
        "fair_queue",
        "atomic_with_delivery",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Dataset read response fence flags must be booleans")
        return value
