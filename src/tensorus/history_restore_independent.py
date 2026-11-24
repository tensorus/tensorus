"""Independently reviewed import of one prepared history release."""

import re
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from pydantic import ValidationError

from tensorus.arrays import canonical_hash, canonical_json
from tensorus.bundle_restore_names import (
    HistoryNames,
    HistoryNamesEntry,
    NameOccupancy,
    NameOccupancyRequest,
    observe_destination_names,
)
from tensorus.errors import Conflict, LimitExceeded, TensorusError
from tensorus.history_payload_bundle import HistoryPayloadManifest, _read_history_payload_bundle
from tensorus.history_payload_restore import (
    HistoryPayloadRestoreRequest,
    _check_snapshot,
    _checked_request,
    _history_destination,
    _release_requirements,
    _review,
)
from tensorus.history_payload_selection import (
    review_history_payload_selection,
    validate_history_payload_selection_review,
)
from tensorus.history_restore_selection import _matching_names, _release_id, _write_selected_import
from tensorus.product_bundle import (
    MAX_BUNDLE_VERSION_RESPONSE_BYTES,
    _capture_write_credential,
    _checked_reviewed_identity,
    _checked_write_identity,
    _inspection_review,
    _read_model,
    _require_write_credential,
)
from tensorus.release_portability import MAX_PORTABLE_BYTES, ImportedRelease, ReleaseHistoryImport

if TYPE_CHECKING:
    from tensorus.client import Client


SELECTION_FORMAT = "tensorus-history-restore-independent-selection-v1"
REVIEW_FORMAT = "tensorus-history-restore-independent-review-v1"


def _names_request(request: HistoryPayloadRestoreRequest, release_id: UUID) -> NameOccupancyRequest:
    choice = next(item for item in request.releases if item.release_id == release_id)
    return NameOccupancyRequest(
        kind="history",
        history=HistoryNames(
            dataset=request.dataset,
            releases=[
                HistoryNamesEntry(
                    release_id=release_id, name=choice.name, snapshot_name=choice.snapshot_name
                )
            ],
        ),
    )


def _selection(
    manifest: HistoryPayloadManifest,
    request: HistoryPayloadRestoreRequest,
    review: dict[str, Any],
    payload_review: dict[str, Any],
    release_id: UUID,
) -> tuple[dict[str, Any], NameOccupancyRequest]:
    release = next(item for item in manifest.source.releases if item.release.id == release_id)
    choice = next(item for item in request.releases if item.release_id == release_id)
    copied = {key: value["version"] for key, value in payload_review["payloads"].items()}
    wanted = _history_destination(release, choice, copied).import_request(
        release, manifest.source.manifest
    )
    encoded = wanted.model_dump(mode="json")
    if len(canonical_json(encoded).encode()) > MAX_PORTABLE_BYTES:
        raise LimitExceeded("The selected history import exceeds its byte limit")
    selection = {
        "format": SELECTION_FORMAT,
        "restore_review_hash": review["review_hash"],
        "bundle_hash": manifest.content_hash,
        "tenant_id": review["tenant_id"],
        "actor_key_id": review["actor_key_id"],
        "actor_role": review["actor_role"],
        "dataset": request.dataset,
        "release_id": str(release_id),
        "payload_selection_hash": payload_review["selection_hash"],
        "validity_assertions_required": payload_review["selection"]["validity_assertions_required"],
        "selected_validity_bindings": payload_review["validity_bindings"],
        "import_request": encoded,
    }
    return selection, _names_request(request, release_id)


def _observe_selected_import(
    client: "Client",
    request: HistoryPayloadRestoreRequest,
    selection: dict[str, Any],
    review: dict[str, Any],
    copied: dict[str, Any],
    *,
    credential: str,
) -> tuple[Literal["not_observed", "committed", "conflicting"], dict[str, Any] | None]:
    wanted = ReleaseHistoryImport.model_validate(selection["import_request"])
    dataset_id = next(
        item["dataset_id"] for item in review["datasets"] if item["dataset"] == request.dataset
    )
    try:
        imported = client._read_imported_release_by_name(
            request.dataset, wanted.name, _credential=credential
        )
    except TensorusError as exc:
        if exc.status != 404 or exc.code != "not_found":
            raise
        return "not_observed", None
    if imported["tenant_id"] != review["tenant_id"] or imported["dataset_id"] != dataset_id:
        raise TensorusError("Destination history identifies another workspace")
    stored = _read_model(
        client,
        f"/v1/release-history/imports/{imported['id']}",
        ImportedRelease,
        "Imported release",
        MAX_PORTABLE_BYTES + MAX_BUNDLE_VERSION_RESPONSE_BYTES,
        credential=credential,
    )
    if canonical_json(stored.model_dump(mode="json")) != canonical_json(imported):
        raise TensorusError("Destination returned contradictory retained history evidence")
    if canonical_json(imported["request"]) != canonical_json(selection["import_request"]):
        return "conflicting", imported
    try:
        _check_snapshot(
            client,
            UUID(imported["snapshot_id"]),
            wanted,
            review,
            copied,
            dataset_id,
            imported["created_by"],
            credential=credential,
        )
    except (UnicodeDecodeError, ValueError, RecursionError, KeyError) as exc:
        raise TensorusError("Destination returned invalid imported snapshot evidence") from exc
    return "committed", imported


def validate_independent_history_selection_review(
    result: object,
    manifest: HistoryPayloadManifest,
    request: HistoryPayloadRestoreRequest,
    review: dict[str, Any],
    release_id: UUID,
) -> None:
    """Validate the selected evidence without assuming unrelated imports were inspected."""
    try:
        if (
            not isinstance(result, dict)
            or set(result)
            != {
                "format",
                "selection",
                "selection_hash",
                "payload_review",
                "release_observation",
                "name_observation",
                "unselected_receipts_checked",
                "names_reserved",
                "point_in_time_atomic",
                "authorizes_restore",
            }
            or result["format"] != REVIEW_FORMAT
            or any(
                result[field] is not False
                for field in (
                    "unselected_receipts_checked",
                    "names_reserved",
                    "point_in_time_atomic",
                    "authorizes_restore",
                )
            )
        ):
            raise ValueError("Invalid independent history review")
        payload_review = result["payload_review"]
        validate_history_payload_selection_review(
            payload_review, manifest, request, review, release_id
        )
        if not payload_review["complete"] or payload_review["blocked_observed"]:
            raise ValueError("Selected payloads are not ready")
        selection, names_request = _selection(manifest, request, review, payload_review, release_id)
        release = next(item for item in manifest.source.releases if item.release.id == release_id)
        observed = result["release_observation"]
        if (
            not isinstance(observed, dict)
            or set(observed) != {"status", "requirements", "record", "snapshot_verified"}
            or observed["requirements"] != _release_requirements(release)
            or observed["status"] not in ("not_observed", "committed")
            or observed["snapshot_verified"] is not (observed["status"] == "committed")
        ):
            raise ValueError("Selected history observation is inconsistent")
        dataset_id = next(
            item["dataset_id"] for item in review["datasets"] if item["dataset"] == request.dataset
        )
        if observed["status"] == "not_observed":
            if observed["record"] is not None:
                raise ValueError("Unobserved import contains a retained record")
        else:
            raw = observed["record"]
            imported = ImportedRelease.model_validate_json(canonical_json(raw), strict=True)
            if (
                canonical_json(raw) != canonical_json(imported.model_dump(mode="json"))
                or str(imported.tenant_id) != review["tenant_id"]
                or str(imported.dataset_id) != dataset_id
                or imported.dataset != request.dataset
                or canonical_json(imported.request.model_dump(mode="json"))
                != canonical_json(selection["import_request"])
                or imported.name != imported.request.name
                or imported.request_hash != canonical_hash(selection["import_request"])
                or imported.source_hash != imported.request.source.content_hash
                or str(UUID(imported.created_by)) != imported.created_by
            ):
                raise ValueError("Retained import contradicts the selected request")
        names_raw = result["name_observation"]
        names = NameOccupancy.model_validate_json(canonical_json(names_raw), strict=True)
        if (
            canonical_json(names_raw) != canonical_json(names.model_dump(mode="json"))
            or names.request != names_request
            or str(names.tenant_id) != review["tenant_id"]
            or str(names.actor_key_id) != review["actor_key_id"]
            or names.actor_role != review["actor_role"]
            or str(names.dataset_id) != dataset_id
            or not _matching_names({"releases": {str(release_id): observed}}, names, release_id)
            or canonical_json(result["selection"]) != canonical_json(selection)
            or result["selection_hash"] != canonical_hash(selection)
        ):
            raise ValueError("Independent history review contradicts its evidence")
    except (
        AttributeError,
        KeyError,
        StopIteration,
        TensorusError,
        TypeError,
        ValueError,
        ValidationError,
        RecursionError,
    ) as exc:
        raise ValueError("Invalid or contradictory independent history review") from exc


def review_independent_history_selection(
    client: "Client",
    payload: bytes,
    request: HistoryPayloadRestoreRequest,
    *,
    expected_review_hash: str,
    release_id: str | UUID,
    reviewed: dict[str, Any] | None = None,
    _credential: str | None = None,
) -> dict[str, Any]:
    """Review one prepared release without inspecting unrelated destination receipts."""
    if not isinstance(payload, bytes):
        raise TensorusError("A history bundle archive is required")
    credential = _capture_write_credential(client) if _credential is None else _credential
    _require_write_credential(client, credential)
    request = _checked_request(request)
    selected_id = _release_id(release_id)
    if reviewed is not None:
        reviewed = _inspection_review(reviewed, expected_review_hash, reviewed, "datasets")
    payload_review = review_history_payload_selection(
        client,
        payload,
        request,
        expected_review_hash=expected_review_hash,
        release_id=selected_id,
        reviewed=reviewed,
        _credential=credential,
    )
    if not payload_review["complete"]:
        raise Conflict("Selected release payloads or validity bindings are not ready")
    manifest, arrays = _read_history_payload_bundle(client, payload)
    del arrays
    current = _review(client, manifest, request, _credential=credential)
    if current["review_hash"] != expected_review_hash:
        raise Conflict("History payload restore review changed; inspect it again")
    identity = _checked_reviewed_identity(client, credential, current)
    client._require_imported_history_name_lookup(_credential=credential)
    selection, names_request = _selection(manifest, request, current, payload_review, selected_id)
    release = next(item for item in manifest.source.releases if item.release.id == selected_id)
    copied = {key: value["version"] for key, value in payload_review["payloads"].items()}
    status, imported = _observe_selected_import(
        client, request, selection, current, copied, credential=credential
    )
    if status == "conflicting":
        raise Conflict("Selected release name contains a different retained import")
    observed = {
        "status": status,
        "requirements": _release_requirements(release),
        "record": imported,
        "snapshot_verified": status == "committed",
    }
    names_raw = observe_destination_names(client, names_request, _credential=credential)
    names = NameOccupancy.model_validate(names_raw)
    if not _matching_names({"releases": {str(selected_id): observed}}, names, selected_id):
        raise Conflict("Selected release or snapshot name is occupied or changed")
    if _review(client, manifest, request, _credential=credential) != current:
        raise Conflict("History payload restore review changed during selection")
    result = {
        "format": REVIEW_FORMAT,
        "selection": selection,
        "selection_hash": canonical_hash(selection),
        "payload_review": payload_review,
        "release_observation": observed,
        "name_observation": names_raw,
        "unselected_receipts_checked": False,
        "names_reserved": False,
        "point_in_time_atomic": False,
        "authorizes_restore": False,
    }
    try:
        validate_independent_history_selection_review(
            result, manifest, request, current, selected_id
        )
    except ValueError as exc:
        raise TensorusError(
            "Independent history review is inconsistent", code="invalid_response"
        ) from exc
    _checked_write_identity(client, credential, identity)
    return result


def apply_independent_history_selection(
    client: "Client",
    payload: bytes,
    request: HistoryPayloadRestoreRequest,
    *,
    expected_review_hash: str,
    release_id: str | UUID,
    expected_selection_hash: str,
    confirm_reviewed_selection: bool,
    confirm_unauthenticated_claims: bool,
    confirm_validity_assertions: bool = False,
    reviewed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply one independently reviewed import, retaining uncertainty on failure."""
    if confirm_reviewed_selection is not True:
        raise TensorusError("Explicit confirmation of the reviewed selection is required")
    if confirm_unauthenticated_claims is not True:
        raise TensorusError("Explicit confirmation of unauthenticated source claims is required")
    if type(confirm_validity_assertions) is not bool:
        raise TensorusError("Validity assertion confirmation must be a boolean")
    if (
        not isinstance(expected_selection_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_selection_hash) is None
    ):
        raise TensorusError("A valid selected history review hash is required")
    credential = _capture_write_credential(client)
    request = _checked_request(request)
    before = review_independent_history_selection(
        client,
        payload,
        request,
        expected_review_hash=expected_review_hash,
        release_id=release_id,
        reviewed=reviewed,
        _credential=credential,
    )
    if before["selection_hash"] != expected_selection_hash:
        raise Conflict("Selected history review changed; inspect it again")
    selection = before["selection"]
    identity = _checked_reviewed_identity(client, credential, selection)
    if selection["validity_assertions_required"] and not confirm_validity_assertions:
        raise TensorusError("Explicit confirmation of local validity assertions is required")
    if before["release_observation"]["status"] == "committed":
        return before
    raw = _write_selected_import(client, request, selection, credential)
    try:
        _checked_write_identity(client, credential, identity)
        after = review_independent_history_selection(
            client,
            payload,
            request,
            expected_review_hash=expected_review_hash,
            release_id=release_id,
            reviewed=reviewed,
            _credential=credential,
        )
    except TensorusError as exc:
        raise TensorusError(
            f"Selected import may have committed; {exc}",
            code=exc.code,
            status=exc.status,
            details=exc.details,
        ) from exc
    retained = after["release_observation"]
    if (
        after["selection_hash"] != expected_selection_hash
        or retained["status"] != "committed"
        or canonical_json(retained["record"]) != canonical_json(raw)
    ):
        raise TensorusError("Selected import may have committed; inspect it again")
    return after
