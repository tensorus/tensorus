"""Capability for optional product-descriptor HTTP read admission."""

from typing import Literal

from pydantic import Field, field_validator

from tensorus.models import Model
from tensorus.scientific_read_admission import PRODUCT_DESCRIPTOR_READ_ROUTES


class ProductDescriptorAdmissionCapabilities(Model):
    format: Literal["tensorus-product-descriptor-admission-v1"] = (
        "tensorus-product-descriptor-admission-v1"
    )
    routes: list[str] = Field(default_factory=lambda: list(PRODUCT_DESCRIPTOR_READ_ROUTES))
    after_authentication: Literal[True] = True
    resource_authorization_in_handler: Literal[True] = True
    shared_scientific_read_pool: Literal[True] = True
    single_api_process: Literal[True] = True
    active_until_response_and_handler_finish: Literal[True] = True
    post_serialization_recheck_admitted: Literal[True] = True
    mutations_admitted: Literal[False] = False
    individual_response_bytes_bounded: Literal[False] = False
    fair_queue: Literal[False] = False
    database_work_hard_bounded: Literal[False] = False
    physical_memory_reserved: Literal[False] = False

    @field_validator("routes")
    @classmethod
    def exact_routes(cls, value: list[str]) -> list[str]:
        if value != list(PRODUCT_DESCRIPTOR_READ_ROUTES):
            raise ValueError("Product descriptor admission routes are inconsistent")
        return value

    @field_validator(
        "after_authentication",
        "resource_authorization_in_handler",
        "shared_scientific_read_pool",
        "single_api_process",
        "active_until_response_and_handler_finish",
        "post_serialization_recheck_admitted",
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
            raise ValueError("Product descriptor admission flags must be booleans")
        return value
