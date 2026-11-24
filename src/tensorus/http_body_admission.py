"""Logical-byte admission for reserved HTTP request bodies."""
from typing import Annotated
from typing import Literal
from typing import Self
from pydantic import Field
from pydantic import StrictInt
from pydantic import field_validator
from pydantic import model_validator
from tensorus.models import Model
FORMAT: Literal["tensorus-http-body-admission-v1"] = "tensorus-http-body-admission-v1"
MAX_POOL_BYTES = 2**50
BodyMethod = Literal["POST", "PUT", "PATCH"]
def _methods() -> list[BodyMethod]:
    return ["POST", "PUT", "PATCH"]
class HttpBodyAdmissionCapabilities(Model):
    format: Literal["tensorus-http-body-admission-v1"] = FORMAT
    pool_bytes: Annotated[StrictInt, Field(ge=1024, le=MAX_POOL_BYTES)]
    max_body_bytes: Annotated[StrictInt, Field(ge=1024, le=3 * 1024**3)]
    methods: list[BodyMethod] = Field(default_factory=_methods)
    before_authentication_and_receive: Literal[True] = True
    unknown_length_reserves_route_limit: Literal[True] = True
    staged_size_reduces_reservation: Literal[True] = True
    staging_io_and_asgi_task_covered: Literal[True] = True
    reserved_staging_required: Literal[True] = True
    single_api_process: Literal[True] = True
    per_workspace_fairness: Literal[False] = False
    physical_memory_reserved: Literal[False] = False
    process_memory_bounded: Literal[False] = False
    proxy_or_asgi_buffers_bounded: Literal[False] = False
    workers_or_other_methods_covered: Literal[False] = False
    counters_durable: Literal[False] = False

    @field_validator(
        "before_authentication_and_receive",
        "unknown_length_reserves_route_limit",
        "staged_size_reduces_reservation",
        "staging_io_and_asgi_task_covered",
        "reserved_staging_required",
        "single_api_process",
        "per_workspace_fairness",
        "physical_memory_reserved",
        "process_memory_bounded",
        "proxy_or_asgi_buffers_bounded",
        "workers_or_other_methods_covered",
        "counters_durable",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("HTTP body admission flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.pool_bytes < self.max_body_bytes or self.methods != ["POST", "PUT", "PATCH"]:
            raise ValueError("HTTP body admission limits or methods are inconsistent")
        return self
