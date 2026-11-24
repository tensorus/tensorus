"""Reconciled, catalog-derived property discovery for immutable tensor versions."""
import base64
import binascii
from typing import Literal
from uuid import UUID
from pydantic import Field
from pydantic import ValidationError
from tensorus.arrays import canonical_json
from tensorus.arrays import decode_unique_json
from tensorus.contracts import ReleaseHash
from tensorus.errors import TensorusError
from tensorus.models import Model
from tensorus.models import Name
INDEX_FORMAT = "tensorus-semantic-property-index-v1"
SNAPSHOT_CURSOR_FORMAT: Literal["tensorus-indexed-snapshot-cursor-v1"] = (
    "tensorus-indexed-snapshot-cursor-v1"
)
MAX_INDEX_RECONCILIATION = 10_000
STREAMED_EVALUATION_FORMAT = "tensorus-indexed-stream-evaluation-v1"
INDEXED_PROPERTIES = (
    "value_unit",
    "model_id",
    "model_version",
    "preprocessing_id",
    "preprocessing_version",
)
class SnapshotCursor(Model):
    format: Literal["tensorus-indexed-snapshot-cursor-v1"] = SNAPSHOT_CURSOR_FORMAT
    query_hash: ReleaseHash
    snapshot_id: UUID
    offset: int = Field(strict=True, ge=1, le=MAX_INDEX_RECONCILIATION)
    total: int = Field(strict=True, ge=1, le=MAX_INDEX_RECONCILIATION)
    last_version_id: UUID
    last_dataset: Name
    last_name: Name
def encode_snapshot_cursor(cursor: SnapshotCursor) -> str:
    encoded = canonical_json(cursor.model_dump(mode="json")).encode()
    return base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")
def decode_snapshot_cursor(value: str) -> SnapshotCursor:
    if not value or len(value) > 2048 or not value.isascii():
        raise TensorusError("Invalid snapshot cursor")
    try:
        encoded = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
        cursor = SnapshotCursor.model_validate(decode_unique_json(encoded))
        if encode_snapshot_cursor(cursor) != value or cursor.offset >= cursor.total:
            raise ValueError("Noncanonical or completed cursor")
        return cursor
    except (binascii.Error, UnicodeDecodeError, ValidationError, ValueError, RecursionError) as exc:
        raise TensorusError("Invalid snapshot cursor") from exc
