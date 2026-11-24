"""A bounded, file-backed local usage receiver; it never contacts a billing service."""
from typing import Literal
from typing import Self
from pydantic import model_validator
from tensorus.contracts import ReleaseHash
from tensorus.models import Model
MAX_EVENT_BYTES: Literal[8192] = 8192
MAX_BATCH_EVENTS: Literal[288] = 288
MAX_TENANT_ENTRIES: Literal[4096] = 4096
MAX_INSPECTION_EVENTS: Literal[64] = 64
class FixtureManifestEntry(Model):
    event_id: ReleaseHash
    file_sha256: ReleaseHash
class FixtureFileObservation(Model):
    event_id: ReleaseHash
    expected_payload_sha256: ReleaseHash
    state: Literal["absent", "payload_matches", "payload_differs", "invalid"]
    file_sha256: ReleaseHash | None
    observed_payload_sha256: ReleaseHash | None
    observed_kind: (
        Literal["logical_output", "wall_attempt", "process_attempt", "assertion", "ingestion"]
        | None
    )

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.state == "absent":
            valid = (
                self.file_sha256 is None
                and self.observed_payload_sha256 is None
                and self.observed_kind is None
            )
        elif self.state == "invalid":
            valid = (
                self.file_sha256 is not None
                and self.observed_payload_sha256 is None
                and self.observed_kind is None
            )
        else:
            valid = (
                self.file_sha256 is not None
                and self.observed_payload_sha256 is not None
                and self.observed_kind is not None
                and (self.observed_payload_sha256 == self.expected_payload_sha256)
                == (self.state == "payload_matches")
            )
        if not valid:
            raise ValueError("Local fixture file observation is inconsistent")
        return self
