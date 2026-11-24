"""Read-only observation of one history import and its selected prerequisites."""

from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import ValidationError

from tensorus.arrays import canonical_hash, canonical_json
from tensorus.bundle_restore_names import NameOccupancy, observe_destination_names
from tensorus.errors import Conflict, TensorusError
from tensorus.history_payload_bundle import HistoryPayloadManifest, _read_history_payload_bundle
from tensorus.history_payload_restore import (
    HistoryPayloadRestoreRequest,
    _checked_request,
    _release_requirements,
    _review,
)
from tensorus.history_payload_selection import (
    review_history_payload_selection,
    validate_history_payload_selection_review,
)
from tensorus.history_restore_independent import (
    _names_request,
    _observe_selected_import,
    _selection,
)
from tensorus.history_restore_selection import _release_id
from tensorus.product_bundle import (
    _capture_write_credential,
    _checked_reviewed_identity,
    _checked_write_identity,
    _inspection_review,
    _require_write_credential,
)
from tensorus.release_portability import ImportedRelease

if TYPE_CHECKING:
    from tensorus.client import Client


FORMAT = "tensorus-history-restore-independent-observation-v1"


def _name_assessment(names: NameOccupancy, release_id: UUID, observed: dict[str, Any]) -> str:
    if names.history is None:
        raise ValueError("Selected history names are unavailable")
    occupant = names.history[release_id]
    status = observed["status"]
    if status == "not_checked":
        return (
            "occupied_unverified"
            if occupant.import_id is not None or occupant.snapshot_id is not None
            else "not_observed"
        )
    if status == "not_observed":
        if occupant.import_id is not None:
            raise ValueError("Selected import appeared during name observation")
        return "snapshot_occupied" if occupant.snapshot_id is not None else "not_observed"
    record = observed["record"]
    if (
        str(occupant.import_id) != record["id"]
        or str(occupant.import_snapshot_id) != record["snapshot_id"]
    ):
        raise ValueError("Selected import changed during name observation")
    if status == "committed":
        if str(occupant.snapshot_id) != record["snapshot_id"]:
            raise ValueError("Selected snapshot changed during name observation")
        return "retained"
    if status == "conflicting":
        return "conflicting_import"
    raise ValueError("Unknown selected history status")


def validate_independent_history_observation(
    result: object,
    manifest: HistoryPayloadManifest,
    request: HistoryPayloadRestoreRequest,
    review: dict[str, Any],
    release_id: UUID,
) -> None:
    """Validate the selected observations without treating them as write authority."""
    try:
        if (
            not isinstance(result, dict)
            or set(result)
            != {
                "format",
                "review_hash",
                "bundle_hash",
                "release_id",
                "payload_review",
                "expected_import_request",
                "release_observation",
                "name_observation",
                "name_assessment",
                "unselected_receipts_checked",
                "names_reserved",
                "point_in_time_atomic",
                "authorizes_restore",
                "source_claims_authenticated",
            }
            or result["format"] != FORMAT
            or result["review_hash"] != review["review_hash"]
            or result["bundle_hash"] != manifest.content_hash
            or result["release_id"] != str(release_id)
            or any(
                result[field] is not False
                for field in (
                    "unselected_receipts_checked",
                    "names_reserved",
                    "point_in_time_atomic",
                    "authorizes_restore",
                    "source_claims_authenticated",
                )
            )
        ):
            raise ValueError("Invalid independent history observation identity")
        payload_review = result["payload_review"]
        validate_history_payload_selection_review(
            payload_review, manifest, request, review, release_id
        )
        release = next(item for item in manifest.source.releases if item.release.id == release_id)
        choice = next(item for item in request.releases if item.release_id == release_id)
        observed = result["release_observation"]
        if (
            not isinstance(observed, dict)
            or set(observed) != {"status", "requirements", "record", "snapshot_verified"}
            or observed["requirements"] != _release_requirements(release)
            or observed["status"] not in ("not_checked", "not_observed", "committed", "conflicting")
            or observed["snapshot_verified"] is not (observed["status"] == "committed")
            or (observed["status"] == "not_checked") is payload_review["complete"]
        ):
            raise ValueError("Selected history observation is inconsistent")
        names_request = _names_request(request, release_id)
        if payload_review["complete"]:
            selection, _ = _selection(manifest, request, review, payload_review, release_id)
            expected = selection["import_request"]
        else:
            expected = None
        if canonical_json(result["expected_import_request"]) != canonical_json(expected):
            raise ValueError("Expected history request contradicts selected payloads")
        dataset_id = next(
            item["dataset_id"] for item in review["datasets"] if item["dataset"] == request.dataset
        )
        status = observed["status"]
        if status in ("not_checked", "not_observed"):
            if observed["record"] is not None:
                raise ValueError("Unobserved import contains a retained record")
        else:
            raw = observed["record"]
            imported = ImportedRelease.model_validate_json(canonical_json(raw), strict=True)
            encoded = imported.request.model_dump(mode="json")
            if (
                canonical_json(raw) != canonical_json(imported.model_dump(mode="json"))
                or str(imported.tenant_id) != review["tenant_id"]
                or str(imported.dataset_id) != dataset_id
                or imported.dataset != request.dataset
                or imported.name != choice.name
                or imported.request.name != imported.name
                or imported.request_hash != canonical_hash(encoded)
                or imported.source_hash != imported.request.source.content_hash
                or str(UUID(imported.created_by)) != imported.created_by
                or (status == "committed")
                is not (canonical_json(encoded) == canonical_json(expected))
            ):
                raise ValueError("Retained import contradicts its evidence")
        names_raw = result["name_observation"]
        names = NameOccupancy.model_validate_json(canonical_json(names_raw), strict=True)
        if (
            canonical_json(names_raw) != canonical_json(names.model_dump(mode="json"))
            or names.request != names_request
            or str(names.tenant_id) != review["tenant_id"]
            or str(names.actor_key_id) != review["actor_key_id"]
            or names.actor_role != review["actor_role"]
            or str(names.dataset_id) != dataset_id
            or result["name_assessment"] != _name_assessment(names, release_id, observed)
        ):
            raise ValueError("Selected name observation contradicts its evidence")
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
        raise ValueError("Invalid or contradictory independent history observation") from exc


def observe_independent_history_release(
    client: "Client",
    payload: bytes,
    request: HistoryPayloadRestoreRequest,
    *,
    expected_review_hash: str,
    release_id: str | UUID,
    reviewed: dict[str, Any] | None = None,
    _credential: str | None = None,
) -> dict[str, Any]:
    """Observe one selected import even when unrelated destination receipts are inconsistent."""
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
    manifest, arrays = _read_history_payload_bundle(client, payload)
    del arrays
    current = _review(client, manifest, request, _credential=credential)
    if current["review_hash"] != expected_review_hash:
        raise Conflict("History payload restore review changed; inspect it again")
    identity = _checked_reviewed_identity(client, credential, current)
    release = next(item for item in manifest.source.releases if item.release.id == selected_id)
    names_request = _names_request(request, selected_id)
    expected = None
    status, imported = "not_checked", None
    if payload_review["complete"]:
        client._require_imported_history_name_lookup(_credential=credential)
        selection, _ = _selection(manifest, request, current, payload_review, selected_id)
        expected = selection["import_request"]
        copied = {key: value["version"] for key, value in payload_review["payloads"].items()}
        status, imported = _observe_selected_import(
            client, request, selection, current, copied, credential=credential
        )
    observed = {
        "status": status,
        "requirements": _release_requirements(release),
        "record": imported,
        "snapshot_verified": status == "committed",
    }
    names_raw = observe_destination_names(client, names_request, _credential=credential)
    names = NameOccupancy.model_validate(names_raw)
    try:
        assessment = _name_assessment(names, selected_id, observed)
    except (KeyError, TypeError, ValueError) as exc:
        raise Conflict("Selected history names changed during observation") from exc
    if _review(client, manifest, request, _credential=credential) != current:
        raise Conflict("History payload restore review changed during observation")
    result = {
        "format": FORMAT,
        "review_hash": current["review_hash"],
        "bundle_hash": manifest.content_hash,
        "release_id": str(selected_id),
        "payload_review": payload_review,
        "expected_import_request": expected,
        "release_observation": observed,
        "name_observation": names_raw,
        "name_assessment": assessment,
        "unselected_receipts_checked": False,
        "names_reserved": False,
        "point_in_time_atomic": False,
        "authorizes_restore": False,
        "source_claims_authenticated": False,
    }
    try:
        validate_independent_history_observation(result, manifest, request, current, selected_id)
    except ValueError as exc:
        raise TensorusError(
            "Independent history observation is inconsistent", code="invalid_response"
        ) from exc
    _checked_write_identity(client, credential, identity)
    return result
