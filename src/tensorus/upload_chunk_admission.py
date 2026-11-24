"""Capability for request-lifetime admission of resumable upload chunks."""

from typing import Literal

from pydantic import field_validator

from tensorus.models import Model

ROUTE: Literal["PUT /v1/uploads/{upload_id}/chunks/{offset}"] = (
    "PUT /v1/uploads/{upload_id}/chunks/{offset}"
)


class UploadChunkAdmissionCapabilities(Model):
    format: Literal["tensorus-upload-chunk-admission-v1"] = "tensorus-upload-chunk-admission-v1"
    route: Literal["PUT /v1/uploads/{upload_id}/chunks/{offset}"] = ROUTE
    after_authentication: Literal[True] = True
    ingest_action_checked_before_admission: Literal[True] = True
    request_body_staged_before_admission: Literal[True] = True
    shared_scientific_read_pool: Literal[True] = True
    single_api_process: Literal[True] = True
    active_until_response_and_handler_finish: Literal[True] = True
    inflight_chunk_work_holds_slot: Literal[True] = True
    exact_chunk_retry_supported: Literal[True] = True
    response_loss_may_follow_commit: Literal[True] = True
    fair_queue: Literal[False] = False
    physical_memory_reserved: Literal[False] = False
    database_work_hard_bounded: Literal[False] = False
    all_upload_paths_covered: Literal[False] = False

    @field_validator(
        "after_authentication",
        "ingest_action_checked_before_admission",
        "request_body_staged_before_admission",
        "shared_scientific_read_pool",
        "single_api_process",
        "active_until_response_and_handler_finish",
        "inflight_chunk_work_holds_slot",
        "exact_chunk_retry_supported",
        "response_loss_may_follow_commit",
        "fair_queue",
        "physical_memory_reserved",
        "database_work_hard_bounded",
        "all_upload_paths_covered",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Upload chunk admission flags must be booleans")
        return value
