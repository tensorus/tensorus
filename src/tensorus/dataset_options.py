"""Compact, paged dataset choices; this evidence does not authorize actions."""

from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import Field, StrictBool, StrictInt, model_validator

from tensorus.models import Model, Name

MAX_PAGE_SIZE: Literal[25] = 25
MAX_OFFSET = 1_000_000
MAX_RESPONSE_BYTES: Literal[65536] = 65_536


class DatasetOptionsCapabilities(Model):
    format: Literal["tensorus-dataset-options-v1"] = "tensorus-dataset-options-v1"
    max_page_size: Literal[25] = MAX_PAGE_SIZE
    max_response_bytes: Literal[65536] = MAX_RESPONSE_BYTES
    shared_admission_enabled: StrictBool
    response_rechecks_enabled: StrictBool
    includes_specs: Literal[False] = False
    snapshot_across_pages: Literal[False] = False
    authorizes_actions: Literal[False] = False

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.response_rechecks_enabled and not self.shared_admission_enabled:
            raise ValueError("Dataset option response rechecks require shared admission")
        return self


class DatasetOption(Model):
    id: UUID
    name: Name
    title: Annotated[str, Field(strict=True, max_length=256)]


class DatasetOptionsPage(Model):
    format: Literal["tensorus-dataset-options-page-v1"] = "tensorus-dataset-options-page-v1"
    tenant_id: UUID
    items: list[DatasetOption] = Field(max_length=MAX_PAGE_SIZE)
    total: Annotated[StrictInt, Field(ge=0, le=2**53 - 1)]
    limit: Annotated[StrictInt, Field(ge=1, le=MAX_PAGE_SIZE)]
    offset: Annotated[StrictInt, Field(ge=0, le=MAX_OFFSET)]

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if (
            len(self.items) != min(self.limit, max(0, self.total - self.offset))
            or len({item.id for item in self.items}) != len(self.items)
            or len({item.name for item in self.items}) != len(self.items)
        ):
            raise ValueError("Dataset option page is inconsistent")
        return self
