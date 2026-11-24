"""Capability and bounds for opt-in scoped imported-history summaries."""

from typing import Annotated, Literal

from pydantic import Field, StrictInt, field_validator

from tensorus.models import Model

MAX_SCOPED_HISTORY_CANDIDATES = 64
MAX_SELECTED_HISTORY_BYTES = 8 * 1024**2
MAX_SELECTED_HISTORY_MAPPINGS = 20_000
HISTORY_SUMMARY_ROUTE: Literal["GET /v1/datasets/{dataset}/release-history/imports"] = (
    "GET /v1/datasets/{dataset}/release-history/imports"
)


class ScopedHistorySummaryPolicyCapabilities(Model):
    format: Literal["tensorus-scoped-history-summary-policy-v1"] = (
        "tensorus-scoped-history-summary-policy-v1"
    )
    route: Literal["GET /v1/datasets/{dataset}/release-history/imports"] = HISTORY_SUMMARY_ROUTE
    candidate_limit: Annotated[
        StrictInt, Field(ge=MAX_SCOPED_HISTORY_CANDIDATES, le=MAX_SCOPED_HISTORY_CANDIDATES)
    ] = MAX_SCOPED_HISTORY_CANDIDATES
    stored_request_bytes_limit: Annotated[
        StrictInt, Field(ge=MAX_SELECTED_HISTORY_BYTES, le=MAX_SELECTED_HISTORY_BYTES)
    ] = MAX_SELECTED_HISTORY_BYTES
    mapping_limit: Annotated[
        StrictInt, Field(ge=MAX_SELECTED_HISTORY_MAPPINGS, le=MAX_SELECTED_HISTORY_MAPPINGS)
    ] = MAX_SELECTED_HISTORY_MAPPINGS
    scoped_items_require_detail_access: Literal[True] = True
    scoped_total_accessible_only: Literal[True] = True
    unscoped_summary_contents_unchanged: Literal[True] = True
    readable_scan_blocked_for_scoped: Literal[True] = True
    shared_scientific_read_pool: Literal[True] = True
    final_caller_recheck: Literal[True] = True
    per_process_configuration: Literal[True] = True
    physical_memory_reserved: Literal[False] = False
    delivery_atomic: Literal[False] = False
    other_processes_fenced: Literal[False] = False

    @field_validator(
        "scoped_items_require_detail_access",
        "scoped_total_accessible_only",
        "unscoped_summary_contents_unchanged",
        "readable_scan_blocked_for_scoped",
        "shared_scientific_read_pool",
        "final_caller_recheck",
        "per_process_configuration",
        "physical_memory_reserved",
        "delivery_atomic",
        "other_processes_fenced",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Scoped history summary policy flags must be booleans")
        return value
