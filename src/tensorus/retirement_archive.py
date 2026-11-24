"""Export selected retirement receipts without changing retained audit history."""
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import model_validator
from tensorus.models import Model
MAX_RETIREMENT_ARCHIVE_RECEIPTS = 128
class RetirementArchiveSelection(Model):
    snapshot_ids: list[UUID] = Field(min_length=1, max_length=MAX_RETIREMENT_ARCHIVE_RECEIPTS)

    @model_validator(mode="after")
    def unique_snapshots(self) -> Self:
        if len(set(self.snapshot_ids)) != len(self.snapshot_ids):
            raise ValueError("Retirement archive snapshot IDs must be unique")
        return self
