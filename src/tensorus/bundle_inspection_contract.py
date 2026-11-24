"""Wire format and resource ceilings for optional bundle progress inspection."""

from typing import Any, Literal, Self
from uuid import UUID

from pydantic import Field, StrictBool, StrictInt, ValidationError, field_validator, model_validator

from tensorus.arrays import canonical_json
from tensorus.errors import TensorusError
from tensorus.models import Model, WriteContractBinding

FORMAT = "tensorus-bundle-restore-inspection-v1"
PATH = "/v1/restore-bundles/inspect-progress"
MEDIA_TYPE = "application/vnd.tensorus.bundle-inspection"
MAGIC = b"TRBI1\n"
APPLY_FORMAT = "tensorus-bundle-restore-apply-v1"
APPLY_PATH = "/v1/restore-bundles/apply"
APPLY_MEDIA_TYPE = "application/vnd.tensorus.bundle-restore-apply"
APPLY_MAGIC = b"TRBA1\n"
PREPARE_FORMAT = "tensorus-bundle-restore-preparation-v1"
PREPARE_RESULT_FORMAT = "tensorus-bundle-restore-preparation-result-v1"
PREPARE_PATH = "/v1/restore-bundles/prepare"
PREPARE_MEDIA_TYPE = "application/vnd.tensorus.bundle-restore-preparation"
PREPARE_MAGIC = b"TRBR1\n"
MAX_PREPARE_HEADER_BYTES = 1024**2
MAX_HEADER_BYTES = 9 * 1024**2
MAX_BODY_BYTES = 40 * 1024**2
MAX_RESPONSE_BYTES = 32 * 1024**2
MAX_INNER_RESPONSE_BYTES = 18 * 1024**2
MAX_INNER_TOTAL_BYTES = 1024**3
MAX_INNER_REQUEST_BYTES = 16 * 1024
MAX_APPLY_REQUEST_BYTES = 16 * 1024**2 + 64 * 1024
MAX_APPLY_TOTAL_REQUEST_BYTES = 64 * 1024**2
MAX_INNER_REQUESTS = 4096
SOFT_DEADLINE_SECONDS = 120


class BundleInspectionCapabilities(Model):
    format: Literal["tensorus-bundle-restore-inspection-v1"]
    path: Literal["/v1/restore-bundles/inspect-progress"]
    media_type: Literal["application/vnd.tensorus.bundle-inspection"]
    max_body_bytes: StrictInt = Field(ge=1024, le=MAX_BODY_BYTES)
    max_header_bytes: StrictInt
    max_product_bundle_bytes: StrictInt = Field(ge=1, le=MAX_BODY_BYTES)
    max_history_bundle_bytes: StrictInt = Field(ge=1, le=MAX_BODY_BYTES)
    max_response_bytes: StrictInt
    max_internal_requests: StrictInt
    max_internal_response_bytes: StrictInt
    max_internal_total_bytes: StrictInt
    soft_deadline_seconds: StrictInt
    active_per_process: Literal[1]
    requires_unscoped_writer: Literal[True]
    reserved_request_staging_required: Literal[True]
    internal_requests_use_existing_admission: Literal[True]
    payload_verified_when_observed: Literal[True]
    point_in_time_atomic: Literal[False]
    retry_guaranteed: Literal[False]
    source_claims_authenticated: Literal[False]
    approval_transferred: Literal[False]
    physical_memory_reserved: Literal[False]
    hard_deadline: Literal[False]

    @field_validator(
        "requires_unscoped_writer",
        "reserved_request_staging_required",
        "internal_requests_use_existing_admission",
        "payload_verified_when_observed",
        "point_in_time_atomic",
        "retry_guaranteed",
        "source_claims_authenticated",
        "approval_transferred",
        "physical_memory_reserved",
        "hard_deadline",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Bundle inspection capability flags must be booleans")
        return value

    @field_validator("active_per_process", mode="before")
    @classmethod
    def exact_concurrency(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Bundle inspection concurrency must be an integer")
        return value

    @model_validator(mode="after")
    def fixed_limits(self) -> Self:
        if (
            self.max_header_bytes != MAX_HEADER_BYTES
            or self.max_response_bytes != MAX_RESPONSE_BYTES
            or self.max_internal_requests != MAX_INNER_REQUESTS
            or self.max_internal_response_bytes != MAX_INNER_RESPONSE_BYTES
            or self.max_internal_total_bytes != MAX_INNER_TOTAL_BYTES
            or self.soft_deadline_seconds != SOFT_DEADLINE_SECONDS
        ):
            raise ValueError("Bundle inspection limits are inconsistent")
        return self


class BundleHeadPreflightCapabilities(Model):
    format: Literal["tensorus-bundle-restore-head-preflight-v1"]
    path: Literal["/v1/restore-bundles/inspect-progress"]
    envelope_field: Literal["include_head_preflight"]
    shared_inspection_slot: Literal[True]
    includes_tombstoned_heads: Literal[True]
    server_apply_checks_heads: StrictBool
    point_in_time_atomic: Literal[False]
    retry_guaranteed: Literal[False]
    authorizes_restore: Literal[False]

    @field_validator(
        "shared_inspection_slot",
        "includes_tombstoned_heads",
        "server_apply_checks_heads",
        "point_in_time_atomic",
        "retry_guaranteed",
        "authorizes_restore",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Bundle head preflight capability flags must be booleans")
        return value


class BundlePayloadObservationCapabilities(Model):
    format: Literal["tensorus-bundle-restore-payload-observation-v1"] = (
        "tensorus-bundle-restore-payload-observation-v1"
    )
    path: Literal["/v1/restore-bundles/inspect-progress"] = "/v1/restore-bundles/inspect-progress"
    envelope_field: Literal["payloads_only"] = "payloads_only"
    shared_inspection_slot: Literal[True] = True
    requires_original_review: Literal[True] = True
    payloads_verified_when_observed: Literal[True] = True
    retained_records_checked: Literal[False] = False
    completion_assessed: Literal[False] = False
    heads_checked: Literal[False] = False
    names_checked: Literal[False] = False
    point_in_time_atomic: Literal[False] = False
    authorizes_restore: Literal[False] = False

    @field_validator(
        "shared_inspection_slot",
        "requires_original_review",
        "payloads_verified_when_observed",
        "retained_records_checked",
        "completion_assessed",
        "heads_checked",
        "names_checked",
        "point_in_time_atomic",
        "authorizes_restore",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Bundle payload observation flags must be booleans")
        return value


class HistoryReleaseObservationCapabilities(Model):
    format: Literal["tensorus-history-restore-release-observation-v1"] = (
        "tensorus-history-restore-release-observation-v1"
    )
    path: Literal["/v1/restore-bundles/inspect-progress"] = "/v1/restore-bundles/inspect-progress"
    envelope_field: Literal["history_release_observation"] = "history_release_observation"
    shared_inspection_slot: Literal[True] = True
    requires_original_review: Literal[True] = True
    independent_release_checks: Literal[True] = True
    conflicting_imports_reported: Literal[True] = True
    snapshots_checked_only_for_matching_imports: Literal[True] = True
    heads_checked: Literal[False] = False
    names_reserved: Literal[False] = False
    point_in_time_atomic: Literal[False] = False
    retry_guaranteed: Literal[False] = False
    authorizes_restore: Literal[False] = False
    source_claims_authenticated: Literal[False] = False

    @field_validator(
        "shared_inspection_slot",
        "requires_original_review",
        "independent_release_checks",
        "conflicting_imports_reported",
        "snapshots_checked_only_for_matching_imports",
        "heads_checked",
        "names_reserved",
        "point_in_time_atomic",
        "retry_guaranteed",
        "authorizes_restore",
        "source_claims_authenticated",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("History release observation flags must be booleans")
        return value


class HistoryRestoreSelectionCapabilities(Model):
    format: Literal["tensorus-history-restore-selection-v1"] = (
        "tensorus-history-restore-selection-v1"
    )
    review_path: Literal["/v1/restore-bundles/inspect-progress"] = (
        "/v1/restore-bundles/inspect-progress"
    )
    apply_path: Literal["/v1/restore-bundles/apply"] = "/v1/restore-bundles/apply"
    release_id_field: Literal["history_selection_release_id"] = "history_selection_release_id"
    selection_hash_field: Literal["expected_selection_hash"] = "expected_selection_hash"
    server_apply_enabled: StrictBool
    requires_original_review: Literal[True] = True
    requires_selection_confirmation: Literal[True] = True
    requires_claim_confirmation: Literal[True] = True
    selected_payloads_and_names_checked: Literal[True] = True
    other_releases_written: Literal[False] = False
    names_reserved: Literal[False] = False
    point_in_time_atomic: Literal[False] = False
    source_claims_authenticated: Literal[False] = False

    @field_validator(
        "server_apply_enabled",
        "requires_original_review",
        "requires_selection_confirmation",
        "requires_claim_confirmation",
        "selected_payloads_and_names_checked",
        "other_releases_written",
        "names_reserved",
        "point_in_time_atomic",
        "source_claims_authenticated",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Selected history capability flags must be booleans")
        return value


class HistoryIndependentSelectionCapabilities(Model):
    format: Literal["tensorus-history-restore-independent-selection-v1"] = (
        "tensorus-history-restore-independent-selection-v1"
    )
    review_path: Literal["/v1/restore-bundles/inspect-progress"] = (
        "/v1/restore-bundles/inspect-progress"
    )
    apply_path: Literal["/v1/restore-bundles/apply"] = "/v1/restore-bundles/apply"
    release_id_field: Literal["history_independent_selection_release_id"] = (
        "history_independent_selection_release_id"
    )
    selection_hash_field: Literal["expected_selection_hash"] = "expected_selection_hash"
    server_apply_enabled: StrictBool
    requires_original_review: Literal[True] = True
    requires_selection_confirmation: Literal[True] = True
    requires_claim_confirmation: Literal[True] = True
    selected_payloads_and_names_checked: Literal[True] = True
    source_archive_fully_verified: Literal[True] = True
    unselected_receipts_checked: Literal[False] = False
    other_releases_written: Literal[False] = False
    names_reserved: Literal[False] = False
    point_in_time_atomic: Literal[False] = False
    source_claims_authenticated: Literal[False] = False

    @field_validator(
        "server_apply_enabled",
        "requires_original_review",
        "requires_selection_confirmation",
        "requires_claim_confirmation",
        "selected_payloads_and_names_checked",
        "source_archive_fully_verified",
        "unselected_receipts_checked",
        "other_releases_written",
        "names_reserved",
        "point_in_time_atomic",
        "source_claims_authenticated",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Independent history capability flags must be booleans")
        return value


class HistoryIndependentObservationCapabilities(Model):
    format: Literal["tensorus-history-restore-independent-observation-v1"] = (
        "tensorus-history-restore-independent-observation-v1"
    )
    review_path: Literal["/v1/restore-bundles/inspect-progress"] = (
        "/v1/restore-bundles/inspect-progress"
    )
    release_id_field: Literal["history_independent_observation_release_id"] = (
        "history_independent_observation_release_id"
    )
    shared_inspection_slot: Literal[True] = True
    requires_original_review: Literal[True] = True
    selected_payloads_and_names_checked: Literal[True] = True
    conflicting_imports_reported: Literal[True] = True
    matching_snapshots_checked: Literal[True] = True
    unselected_receipts_checked: Literal[False] = False
    names_reserved: Literal[False] = False
    point_in_time_atomic: Literal[False] = False
    read_only: Literal[True] = True
    authorizes_restore: Literal[False] = False
    source_claims_authenticated: Literal[False] = False

    @field_validator(
        "shared_inspection_slot",
        "requires_original_review",
        "selected_payloads_and_names_checked",
        "conflicting_imports_reported",
        "matching_snapshots_checked",
        "unselected_receipts_checked",
        "names_reserved",
        "point_in_time_atomic",
        "read_only",
        "authorizes_restore",
        "source_claims_authenticated",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Independent history observation flags must be booleans")
        return value


class HistoryPayloadSelectionCapabilities(Model):
    format: Literal["tensorus-history-payload-selection-v1"] = (
        "tensorus-history-payload-selection-v1"
    )
    review_path: Literal["/v1/restore-bundles/inspect-progress"] = (
        "/v1/restore-bundles/inspect-progress"
    )
    apply_path: Literal["/v1/restore-bundles/apply"] = "/v1/restore-bundles/apply"
    release_id_field: Literal["history_payload_selection_release_id"] = (
        "history_payload_selection_release_id"
    )
    selection_hash_field: Literal["expected_selection_hash"] = "expected_selection_hash"
    server_apply_enabled: StrictBool
    requires_original_review: Literal[True] = True
    requires_selection_confirmation: Literal[True] = True
    requires_claim_confirmation: Literal[True] = True
    selected_prefixes_and_validity_only: Literal[True] = True
    source_archive_fully_verified: Literal[True] = True
    history_imported: Literal[False] = False
    point_in_time_atomic: Literal[False] = False
    retry_guaranteed: Literal[False] = False
    source_claims_authenticated: Literal[False] = False

    @field_validator(
        "server_apply_enabled",
        "requires_original_review",
        "requires_selection_confirmation",
        "requires_claim_confirmation",
        "selected_prefixes_and_validity_only",
        "source_archive_fully_verified",
        "history_imported",
        "point_in_time_atomic",
        "retry_guaranteed",
        "source_claims_authenticated",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Selected history payload capability flags must be booleans")
        return value


class ProductPayloadSelectionCapabilities(Model):
    format: Literal["tensorus-product-payload-selection-v1"] = (
        "tensorus-product-payload-selection-v1"
    )
    review_path: Literal["/v1/restore-bundles/inspect-progress"] = (
        "/v1/restore-bundles/inspect-progress"
    )
    apply_path: Literal["/v1/restore-bundles/apply"] = "/v1/restore-bundles/apply"
    role_field: Literal["product_payload_selection_role"] = "product_payload_selection_role"
    selection_hash_field: Literal["expected_selection_hash"] = "expected_selection_hash"
    server_apply_enabled: StrictBool
    requires_original_review: Literal[True] = True
    requires_selection_confirmation: Literal[True] = True
    requires_claim_confirmation: Literal[True] = True
    selected_role_and_mask_only: Literal[True] = True
    source_archive_fully_verified: Literal[True] = True
    unselected_receipts_checked: Literal[False] = False
    local_product_created: Literal[False] = False
    point_in_time_atomic: Literal[False] = False
    retry_guaranteed: Literal[False] = False
    source_claims_authenticated: Literal[False] = False

    @field_validator(
        "server_apply_enabled",
        "requires_original_review",
        "requires_selection_confirmation",
        "requires_claim_confirmation",
        "selected_role_and_mask_only",
        "source_archive_fully_verified",
        "unselected_receipts_checked",
        "local_product_created",
        "point_in_time_atomic",
        "retry_guaranteed",
        "source_claims_authenticated",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Selected product payload capability flags must be booleans")
        return value


class NameOccupancyCapabilities(Model):
    format: Literal["tensorus-bundle-restore-name-occupancy-v1"] = (
        "tensorus-bundle-restore-name-occupancy-v1"
    )
    path: Literal["/v1/restore-bundles/name-occupancy"] = "/v1/restore-bundles/name-occupancy"
    max_history_releases: Literal[8] = 8
    requires_unscoped_writer: Literal[True] = True
    includes_retired_product_names: Literal[True] = True
    point_in_time_atomic: Literal[False] = False
    names_reserved: Literal[False] = False

    @field_validator(
        "requires_unscoped_writer",
        "includes_retired_product_names",
        "point_in_time_atomic",
        "names_reserved",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Bundle name capability flags must be booleans")
        return value

    @field_validator("max_history_releases", mode="before")
    @classmethod
    def exact_limit(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Bundle name selection limit must be an integer")
        return value


class NamePreflightCapabilities(Model):
    format: Literal["tensorus-bundle-restore-name-preflight-v1"] = (
        "tensorus-bundle-restore-name-preflight-v1"
    )
    path: Literal["/v1/restore-bundles/inspect-progress"] = "/v1/restore-bundles/inspect-progress"
    envelope_field: Literal["include_name_preflight"] = "include_name_preflight"
    apply_field: Literal["reject_observed_name_conflicts"] = "reject_observed_name_conflicts"
    shared_inspection_slot: Literal[True] = True
    point_in_time_atomic: Literal[False] = False
    retry_guaranteed: Literal[False] = False
    authorizes_restore: Literal[False] = False

    @field_validator(
        "shared_inspection_slot",
        "point_in_time_atomic",
        "retry_guaranteed",
        "authorizes_restore",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Bundle name preflight flags must be booleans")
        return value


class BundleRestoreApplyCapabilities(Model):
    format: Literal["tensorus-bundle-restore-apply-v1"]
    path: Literal["/v1/restore-bundles/apply"]
    media_type: Literal["application/vnd.tensorus.bundle-restore-apply"]
    max_internal_request_bytes: StrictInt
    max_internal_total_request_bytes: StrictInt
    shared_inspection_slot: Literal[True]
    requires_original_review: Literal[True]
    requires_restore_confirmation: Literal[True]
    requires_claim_confirmation: Literal[True]
    requires_selected_validity_confirmation: Literal[True]
    point_in_time_atomic: Literal[False]
    retry_guaranteed: Literal[False]
    source_claims_authenticated: Literal[False]
    approval_transferred: Literal[False]
    hard_deadline: Literal[False]

    @field_validator(
        "shared_inspection_slot",
        "requires_original_review",
        "requires_restore_confirmation",
        "requires_claim_confirmation",
        "requires_selected_validity_confirmation",
        "point_in_time_atomic",
        "retry_guaranteed",
        "source_claims_authenticated",
        "approval_transferred",
        "hard_deadline",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Bundle restore capability flags must be booleans")
        return value

    @field_validator(
        "max_internal_request_bytes", "max_internal_total_request_bytes", mode="before"
    )
    @classmethod
    def exact_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Bundle restore request limits must be integers")
        return value

    @model_validator(mode="after")
    def fixed_limits(self) -> Self:
        if (
            self.max_internal_request_bytes != MAX_APPLY_REQUEST_BYTES
            or self.max_internal_total_request_bytes != MAX_APPLY_TOTAL_REQUEST_BYTES
        ):
            raise ValueError("Bundle restore request limits are inconsistent")
        return self


class BundleRestorePreparationCapabilities(Model):
    format: Literal["tensorus-bundle-restore-preparation-v1"]
    path: Literal["/v1/restore-bundles/prepare"]
    media_type: Literal["application/vnd.tensorus.bundle-restore-preparation"]
    result_format: Literal["tensorus-bundle-restore-preparation-result-v1"]
    max_header_bytes: StrictInt
    shared_inspection_slot: Literal[True]
    archive_inspection_supported: Literal[True]
    review_supported: Literal[True]
    payloads_verified: Literal[True]
    mutations_performed: Literal[False]
    source_claims_authenticated: Literal[False]
    review_authorizes_restore: Literal[False]
    physical_memory_reserved: Literal[False]
    hard_deadline: Literal[False]

    @field_validator(
        "shared_inspection_slot",
        "archive_inspection_supported",
        "review_supported",
        "payloads_verified",
        "mutations_performed",
        "source_claims_authenticated",
        "review_authorizes_restore",
        "physical_memory_reserved",
        "hard_deadline",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Bundle preparation capability flags must be booleans")
        return value

    @model_validator(mode="after")
    def fixed_limits(self) -> Self:
        if self.max_header_bytes != MAX_PREPARE_HEADER_BYTES:
            raise ValueError("Bundle preparation header limit is inconsistent")
        return self


def validate_progress(
    kind: Literal["product", "history"],
    progress: Any,
    review: dict[str, Any],
    *,
    progress_version: Literal[2, 3] = 2,
    payloads_only: bool = False,
) -> None:
    """Reject progress that contradicts its original review or its own policy evidence."""

    def binding(value: object) -> WriteContractBinding | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("Invalid write-contract evidence")
        return WriteContractBinding.model_validate(value, strict=True)

    def original_generation(policy: object) -> int:
        if not isinstance(policy, dict):
            raise ValueError("Missing original policy")
        generation = policy.get("generation")
        if type(generation) is not int or not 0 <= generation <= 2_147_483_647:
            raise ValueError("Invalid original policy generation")
        return generation

    def record_state(value: object, *, not_requested: bool = False) -> str:
        if not isinstance(value, dict) or set(value) != {"status", "record"}:
            raise ValueError("Invalid retained-record progress")
        status = value["status"]
        allowed = {"not_checked", "not_observed", "committed"}
        if not_requested:
            allowed.add("not_requested")
        if status not in allowed or (
            not isinstance(value["record"], dict)
            if status == "committed"
            else value["record"] is not None
        ):
            raise ValueError("Retained-record progress contradicts its status")
        return str(status)

    try:
        common = {
            "format",
            "review_hash",
            "current_review_hash",
            "bundle_hash",
            "source_claims_authenticated",
            "payloads",
            "validity_bindings",
            "complete",
            "status",
            "review_policy_current",
            "continuation",
            "next_action",
            "point_in_time_atomic",
            "retry_guaranteed",
            "approval_transferred",
            "local_release_created",
        }
        specific = (
            {"product", "source_import"}
            if kind == "product"
            else {"history_imports", "source_lineage_recreated", "source_plans_recreated"}
        )
        if payloads_only:
            common |= {
                "retained_records_checked",
                "completion_assessed",
                "heads_checked",
                "names_checked",
                "authorizes_restore",
            }
        expected_format = (
            (
                "tensorus-product-bundle-restore-payload-observation-v1"
                if kind == "product"
                else "tensorus-history-payload-restore-payload-observation-v1"
            )
            if payloads_only
            else (
                f"tensorus-product-bundle-restore-progress-v{progress_version}"
                if kind == "product"
                else f"tensorus-history-payload-restore-progress-v{progress_version}"
            )
        )
        if (
            not isinstance(progress, dict)
            or set(progress) != common | specific
            or progress["format"] != expected_format
            or progress["review_hash"] != review["review_hash"]
            or progress["bundle_hash"] != review["bundle_hash"]
            or not isinstance(progress["current_review_hash"], str)
            or len(progress["current_review_hash"]) != 64
            or any(c not in "0123456789abcdef" for c in progress["current_review_hash"])
            or any(
                progress[field] is not False
                for field in (
                    "source_claims_authenticated",
                    "point_in_time_atomic",
                    "retry_guaranteed",
                    "approval_transferred",
                    "local_release_created",
                )
            )
            or (
                progress["complete"] is not None
                if payloads_only
                else type(progress["complete"]) is not bool
            )
            or type(progress["review_policy_current"]) is not bool
            or (
                payloads_only
                and any(
                    progress[field] is not False
                    for field in (
                        "retained_records_checked",
                        "completion_assessed",
                        "heads_checked",
                        "names_checked",
                        "authorizes_restore",
                    )
                )
            )
        ):
            raise ValueError("Invalid bundle progress identity or flags")
        destinations = review["request"]["destinations"]
        payloads = progress["payloads"]
        if (
            not isinstance(destinations, dict)
            or not isinstance(payloads, dict)
            or set(payloads) != set(destinations)
            or not 1 <= len(payloads) <= (32 if kind == "product" else 128)
        ):
            raise ValueError("Bundle progress payload coverage differs from the review")
        if kind == "product":
            generations: dict[str, int] = {}
            for role in review["roles"]:
                source_id = role["source_version_id"]
                generation = original_generation(role["active_write_contract"])
                if generations.setdefault(source_id, generation) != generation:
                    raise ValueError("Original product policies disagree")
        else:
            datasets = {
                item["dataset"]: original_generation(item["active_write_contract"])
                for item in review["datasets"]
            }
            generations = {
                source_id: datasets[target["dataset"]] for source_id, target in destinations.items()
            }
        if set(generations) != set(payloads):
            raise ValueError("Original policy coverage is incomplete")

        payload_fields = {
            "status",
            "version",
            "payload_verified",
            "retry_guaranteed",
            "review_policy_current",
            "retained_policy_matches_review",
            "retained_policy_current",
            "current_write_contract",
            "continuation",
            "next_action",
        }
        blocked = copied = False
        all_current = True
        for source_id, evidence in payloads.items():
            if str(UUID(source_id)) != source_id or not isinstance(evidence, dict):
                raise ValueError("Invalid bundle payload identity")
            if set(evidence) != payload_fields:
                raise ValueError("Invalid bundle payload progress fields")
            version = evidence["version"]
            observed = evidence["status"] == "committed"
            if (
                evidence["status"] not in ("not_observed", "committed")
                or evidence["payload_verified"] is not observed
                or evidence["retry_guaranteed"] is not False
                or (not isinstance(version, dict) if observed else version is not None)
            ):
                raise ValueError("Payload progress contradicts its status")
            current = binding(evidence["current_write_contract"])
            current_generation = current.generation if current else 0
            old_generation = generations[source_id]
            retained = binding(version["write_contract"]) if observed else None
            retained_generation = retained.generation if retained else 0
            if observed and (
                version["tenant_id"] != review["tenant_id"]
                or version["dataset"] != destinations[source_id]["dataset"]
                or version["name"] != destinations[source_id]["name"]
                or str(UUID(version["id"])) != version["id"]
            ):
                raise ValueError("Observed payload belongs to another destination")
            policy_blocked = old_generation != current_generation or (
                observed and retained_generation != current_generation
            )
            if (
                old_generation > current_generation
                or (observed and retained_generation > current_generation)
                or evidence["review_policy_current"] is not (old_generation == current_generation)
                or evidence["retained_policy_matches_review"]
                is not (retained_generation == old_generation if observed else None)
                or evidence["retained_policy_current"]
                is not (retained_generation == current_generation if observed else None)
                or (observed and retained_generation == current_generation and retained != current)
                or evidence["continuation"]
                != ("blocked_policy_changed" if policy_blocked else "not_assessed")
                or evidence["next_action"]
                != (
                    "review_partial_and_select_new_destinations"
                    if policy_blocked
                    else "review_before_retry"
                )
            ):
                raise ValueError("Payload progress contradicts historical or current policy")
            blocked |= policy_blocked
            copied |= observed
            all_current &= old_generation == current_generation

        bindings = progress["validity_bindings"]
        if kind == "product":
            role_sources = {item["role"]: item["source_version_id"] for item in review["roles"]}
            binding_sources = {
                item["role"]: (item["source_version_id"], role_sources[item["validity_mask_role"]])
                for item in review["roles"]
                if item["validity_mask_role"] is not None
            }
        else:
            binding_sources = {
                item["source_binding_id"]: (item["data_version_id"], item["mask_version_id"])
                for item in review["selected_validity"]
            }
            if type(review["request"]["assert_selected_validity"]) is not bool:
                raise ValueError("Invalid selected-validity request")
        if not isinstance(bindings, dict) or set(bindings) != set(binding_sources):
            raise ValueError("Invalid validity progress")
        for key, value in bindings.items():
            if not isinstance(value, dict) or set(value) != {"status", "binding"}:
                raise ValueError("Invalid validity progress fields")
            allowed = {"not_checked", "not_observed", "committed"}
            if kind == "history":
                allowed.add("not_requested")
            if value["status"] not in allowed or (
                not isinstance(value["binding"], dict)
                if value["status"] == "committed"
                else value["binding"] is not None
            ):
                raise ValueError("Validity progress contradicts its status")
            data_id, mask_id = binding_sources[key]
            data, mask = payloads[data_id]["version"], payloads[mask_id]["version"]
            if kind == "history" and not review["request"]["assert_selected_validity"]:
                if value["status"] != "not_requested":
                    raise ValueError("Unrequested validity binding has progress")
            elif data is None or mask is None:
                if value["status"] != "not_checked":
                    raise ValueError("Validity was checked before both payloads were observed")
            elif value["status"] not in ("not_observed", "committed"):
                raise ValueError("Available validity pair was not checked")
            if value["status"] == "committed":
                if data is None or mask is None:
                    raise ValueError("Observed validity binding has unobserved payloads")
                from tensorus.product_bundle_restore import _checked_asserted_binding

                checked = _checked_asserted_binding(
                    value["binding"], data, mask, UUID(review["tenant_id"])
                )
                if canonical_json(checked) != canonical_json(value["binding"]):
                    raise ValueError("Observed validity binding is not exact")

        if kind == "product":
            product = record_state(progress["product"])
            imported = record_state(progress["source_import"])
            complete = product == imported == "committed"
            if (product != "committed" and imported != "not_checked") or (
                product == "committed" and imported == "not_checked"
            ):
                raise ValueError("Product and import progress disagree")
            if product == "committed":
                product_record = progress["product"]["record"]
                if (
                    product_record["tenant_id"] != review["tenant_id"]
                    or product_record["name"] != review["request"]["product_name"]
                    or type(product_record["revision"]) is not int
                    or product_record["revision"] != 1
                    or str(UUID(product_record["id"])) != product_record["id"]
                ):
                    raise ValueError("Retained product differs from the original review")
                if imported == "committed":
                    retained = progress["source_import"]["record"]
                    retained_request = retained["request"]
                    if (
                        retained["tenant_id"] != review["tenant_id"]
                        or retained["product_id"] != product_record["id"]
                        or retained["product_name"] != product_record["name"]
                        or type(retained["product_revision"]) is not int
                        or retained["product_revision"] != product_record["revision"]
                        or retained["name"] != review["request"]["import_name"]
                        or retained["source_hash"] != review["source_hash"]
                        or str(UUID(retained["id"])) != retained["id"]
                        or retained_request["name"] != retained["name"]
                        or retained_request["product_revision"] != retained["product_revision"]
                        or retained_request["descriptor_hash"] != product_record["descriptor_hash"]
                        or retained_request["source"]["content_hash"] != retained["source_hash"]
                    ):
                        raise ValueError("Retained product import differs from the original review")
            assertions_complete = all(item["status"] == "committed" for item in bindings.values())
            ready = assertions_complete and all(
                item["status"] == "committed" for item in payloads.values()
            )
            if payloads_only and (product != "not_checked" or imported != "not_checked"):
                raise ValueError("Payload observation must not claim product progress")
            if not payloads_only and (product == "not_checked") == ready:
                raise ValueError("Product progress contradicts payload and validity coverage")
        else:
            imports = progress["history_imports"]
            expected_imports = {item["release_id"] for item in review["request"]["releases"]}
            if not isinstance(imports, dict) or not imports or set(imports) != expected_imports:
                raise ValueError("Invalid selected-history progress")
            choices = {item["release_id"]: item for item in review["request"]["releases"]}
            target_dataset = review["request"]["dataset"]
            dataset_id = next(
                item["dataset_id"]
                for item in review["datasets"]
                if item["dataset"] == target_dataset
            )
            import_statuses = []
            seen_imports: set[str] = set()
            seen_snapshots: set[str] = set()
            for source_id, item in imports.items():
                status = record_state(item)
                import_statuses.append(status)
                if status != "committed":
                    continue
                retained = item["record"]
                retained_request = retained["request"]
                choice = choices[source_id]
                if (
                    retained["tenant_id"] != review["tenant_id"]
                    or retained["dataset_id"] != dataset_id
                    or retained["dataset"] != target_dataset
                    or retained["name"] != choice["name"]
                    or type(retained["contract_revision"]) is not int
                    or retained["contract_revision"] != choice["contract_revision"]
                    or str(UUID(retained["id"])) != retained["id"]
                    or str(UUID(retained["snapshot_id"])) != retained["snapshot_id"]
                    or retained["id"] in seen_imports
                    or retained["snapshot_id"] in seen_snapshots
                    or retained_request["name"] != choice["name"]
                    or retained_request["snapshot_name"] != choice["snapshot_name"]
                    or retained_request["contract_revision"] != choice["contract_revision"]
                    or retained_request["source"]["release"]["id"] != source_id
                    or retained_request["source"]["content_hash"] != retained["source_hash"]
                ):
                    raise ValueError("Retained history differs from the original review")
                seen_imports.add(retained["id"])
                seen_snapshots.add(retained["snapshot_id"])
            complete = all(status == "committed" for status in import_statuses)
            assertions_complete = all(
                item["status"] == "committed"
                or (
                    not review["request"]["assert_selected_validity"]
                    and item["status"] == "not_requested"
                )
                for item in bindings.values()
            )
            ready = assertions_complete and all(
                item["status"] == "committed" for item in payloads.values()
            )
            if payloads_only and any(status != "not_checked" for status in import_statuses):
                raise ValueError("Payload observation must not claim history progress")
            if not payloads_only and any(
                (status != "not_checked") != ready for status in import_statuses
            ):
                raise ValueError("History progress contradicts payload and validity coverage")
            if (
                progress["source_lineage_recreated"] is not False
                or progress["source_plans_recreated"] is not False
            ):
                raise ValueError("History progress claims uncreated source records")
        if complete and (
            not assertions_complete
            or any(item["status"] != "committed" for item in payloads.values())
        ):
            raise ValueError("Complete progress has unobserved payloads or assertions")
        expected_status = (
            "complete"
            if complete
            else "blocked_policy_changed"
            if blocked
            else "payloads_observed"
            if payloads_only and copied
            else "no_payloads_observed"
            if payloads_only
            else "partial"
            if copied
            else "not_started"
        )
        if (
            (not payloads_only and progress["complete"] is not complete)
            or progress["status"] != expected_status
            or progress["review_policy_current"] is not all_current
            or (progress["current_review_hash"] == progress["review_hash"]) is not all_current
            or progress["continuation"] != ("blocked_policy_changed" if blocked else "not_assessed")
            or progress["next_action"]
            != (
                "review_partial_and_select_new_destinations"
                if blocked
                else "inspect_complete_progress_before_retry"
                if payloads_only
                else "review_before_retry"
            )
        ):
            raise ValueError("Bundle progress summary contradicts its evidence")
    except (
        AttributeError,
        KeyError,
        StopIteration,
        TensorusError,
        TypeError,
        ValueError,
        ValidationError,
    ) as exc:
        raise ValueError("Invalid or contradictory bundle progress") from exc
