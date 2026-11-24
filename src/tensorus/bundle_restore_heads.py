"""Separately versioned, non-atomic destination-head observations for bundle restores."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote
from uuid import UUID

from pydantic import ValidationError

from tensorus import responses as dto
from tensorus.arrays import canonical_json, decode_unique_json
from tensorus.bundle_inspection_contract import validate_progress
from tensorus.errors import Conflict, TensorusError
from tensorus.history_payload_bundle import MAX_HISTORY_PAYLOAD_VERSIONS
from tensorus.product_bundle import (
    MAX_BUNDLE_VERSION_RESPONSE_BYTES,
    _checked_reviewed_identity,
    _checked_write_identity,
)

if TYPE_CHECKING:
    from tensorus.client import Client
    from tensorus.history_payload_bundle import HistoryPayloadManifest
    from tensorus.product_bundle import ProductBundleManifest

FORMAT = "tensorus-bundle-restore-head-preflight-v1"


@dataclass(frozen=True)
class HeadExpectation:
    source_id: str
    dataset: str
    dataset_id: str
    name: str
    expected_version: int


def product_head_expectations(
    manifest: "ProductBundleManifest", review: dict[str, Any]
) -> list[HeadExpectation]:
    datasets = {
        item["source_version_id"]: item["destination_dataset_id"] for item in review["roles"]
    }
    destinations = review["request"]["destinations"]
    return [
        HeadExpectation(
            str(entry.source_version_id),
            destinations[str(entry.source_version_id)]["dataset"],
            datasets[str(entry.source_version_id)],
            destinations[str(entry.source_version_id)]["name"],
            0,
        )
        for entry in manifest.payloads
    ]


def history_head_expectations(
    manifest: "HistoryPayloadManifest", review: dict[str, Any]
) -> list[HeadExpectation]:
    datasets = {item["dataset"]: item["dataset_id"] for item in review["datasets"]}
    destinations = review["request"]["destinations"]
    counts: dict[UUID, int] = {}
    targets: list[HeadExpectation] = []
    ordered = sorted(
        (entry.source for entry in manifest.payloads),
        key=lambda item: (str(item.tensor_id), item.version, str(item.id)),
    )
    for source in ordered:
        target = destinations[str(source.id)]
        expected = counts.get(source.tensor_id, 0)
        targets.append(
            HeadExpectation(
                str(source.id),
                target["dataset"],
                datasets[target["dataset"]],
                target["name"],
                expected,
            )
        )
        counts[source.tensor_id] = expected + 1
    return targets


def expectations_from_bundle(
    client: "Client", kind: Literal["product", "history"], payload: bytes, review: dict[str, Any]
) -> list[HeadExpectation]:
    if kind == "product":
        from tensorus.product_bundle import _read_bundle

        product, arrays = _read_bundle(client, payload)
        del arrays
        if product.content_hash != review["bundle_hash"]:
            raise ValueError("Bundle differs from the original review")
        return product_head_expectations(product, review)
    from tensorus.history_payload_bundle import _read_history_payload_bundle

    history, arrays = _read_history_payload_bundle(client, payload)
    del arrays
    if history.content_hash != review["bundle_hash"]:
        raise ValueError("Bundle differs from the original review")
    return history_head_expectations(history, review)


def _read_head(
    client: "Client", target: HeadExpectation, tenant_id: str, *, credential: str
) -> dict[str, Any] | None:
    path = (
        f"/v1/datasets/{quote(target.dataset, safe='')}"
        f"/tensors/{quote(target.name, safe='')}/versions"
    )
    try:
        encoded = client._bounded_report(
            "GET",
            path,
            MAX_BUNDLE_VERSION_RESPONSE_BYTES,
            "Destination tensor history",
            headers={"Authorization": credential},
            params={"limit": 1, "offset": 0},
        )
    except TensorusError as exc:
        if exc.status != 404 or exc.code != "not_found":
            raise
        return None
    try:
        raw = decode_unique_json(encoded)
        page = dto.Page[dto.TensorVersion].model_validate_json(encoded, strict=True)
        if (
            canonical_json(raw) != canonical_json(page.model_dump(mode="json"))
            or page.limit != 1
            or page.offset != 0
            or page.total < 1
            or len(page.items) != 1
        ):
            raise ValueError("Invalid destination history page")
        version = page.items[0]
        if (
            str(version.tenant_id) != tenant_id
            or str(version.dataset_id) != target.dataset_id
            or version.dataset != target.dataset
            or version.name != target.name
            or version.version < 1
            or version.version != version.head_version
        ):
            raise ValueError("Destination head contradicts its reviewed identity")
        return {
            "tensor_id": str(version.tensor_id),
            "version_id": str(version.id),
            "version": version.version,
            "deleted": version.deleted_at is not None,
        }
    except (UnicodeDecodeError, ValueError, RecursionError, ValidationError) as exc:
        raise TensorusError("Destination returned invalid tensor head evidence") from exc


def _assessment(expected_version: int, head: dict[str, Any] | None, receipt: dict[str, Any]) -> str:
    version = receipt["version"]
    if version is not None and (
        head is None
        or head["tensor_id"] != version["tensor_id"]
        or head["version"] < version["version"]
        or (head["version"] == version["version"] and head["version_id"] != version["id"])
    ):
        raise Conflict("Destination head and retained receipt changed during inspection")
    if head is not None and head["deleted"]:
        return "tombstoned"
    if version is not None:
        return "committed"
    if head is None:
        return "no_head_observed" if expected_version == 0 else "awaiting_prior_versions"
    if head["version"] > expected_version:
        return "head_advanced"
    if head["version"] < expected_version:
        return "awaiting_prior_versions"
    return "expected_head_observed"


def _selected_assessment(
    expected_version: int, head: dict[str, Any] | None, evidence: dict[str, Any]
) -> str:
    assessment = _assessment(expected_version, head, evidence)
    if evidence["version"] is not None and evidence["version"]["deleted_at"] is not None:
        return "receipt_tombstoned"
    return assessment


def inspect_destination_heads(
    client: "Client",
    *,
    kind: Literal["product", "history"],
    review: dict[str, Any],
    progress: dict[str, Any],
    targets: list[HeadExpectation],
    credential: str,
    recheck: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Observe heads separately from receipts; this does not reserve or authorize writes."""
    try:
        validate_progress(kind, progress, review, progress_version=3)
    except ValueError as exc:
        raise TensorusError("Restore progress is inconsistent") from exc
    identity = _checked_reviewed_identity(client, credential, review)

    heads: dict[tuple[str, str], dict[str, Any] | None] = {}
    observations: dict[str, dict[str, Any]] = {}
    for target in targets:
        key = (target.dataset, target.name)
        if key not in heads:
            heads[key] = _read_head(client, target, str(identity.id), credential=credential)
        head = heads[key]
        observations[target.source_id] = {
            "expected_version": target.expected_version,
            "head": head,
            "assessment": _assessment(
                target.expected_version, head, progress["payloads"][target.source_id]
            ),
        }
    if recheck()["review_hash"] != progress["current_review_hash"]:
        raise Conflict("Restore review changed during destination head inspection")

    report = {
        "format": FORMAT,
        "kind": kind,
        "review_hash": progress["review_hash"],
        "bundle_hash": progress["bundle_hash"],
        "progress": progress,
        "targets": observations,
        "conflicts_observed": any(
            item["assessment"] in ("head_advanced", "tombstoned") for item in observations.values()
        ),
        "point_in_time_atomic": False,
        "retry_guaranteed": False,
        "authorizes_restore": False,
    }
    try:
        validate_head_preflight(kind, report, review, targets=targets)
    except ValueError as exc:
        raise TensorusError("Destination head inspection is inconsistent") from exc
    _checked_write_identity(client, credential, identity)
    return report


def validate_head_preflight(
    kind: Literal["product", "history"],
    report: Any,
    review: dict[str, Any],
    *,
    targets: list[HeadExpectation] | None = None,
) -> None:
    """Check the wrapper without treating observations as authority or a retry guarantee."""
    try:
        if (
            not isinstance(report, dict)
            or set(report)
            != {
                "format",
                "kind",
                "review_hash",
                "bundle_hash",
                "progress",
                "targets",
                "conflicts_observed",
                "point_in_time_atomic",
                "retry_guaranteed",
                "authorizes_restore",
            }
            or report["format"] != FORMAT
            or report["kind"] != kind
            or report["review_hash"] != review["review_hash"]
            or report["bundle_hash"] != review["bundle_hash"]
            or type(report["conflicts_observed"]) is not bool
            or any(
                report[field] is not False
                for field in ("point_in_time_atomic", "retry_guaranteed", "authorizes_restore")
            )
        ):
            raise ValueError("Invalid destination head report identity")
        progress = report["progress"]
        validate_progress(kind, progress, review, progress_version=3)
        observations = report["targets"]
        destinations = review["request"]["destinations"]
        if not isinstance(observations, dict) or set(observations) != set(destinations):
            raise ValueError("Destination head coverage differs from the review")
        expected = {target.source_id: target.expected_version for target in targets or []}
        if targets is not None and (
            len(expected) != len(targets) or set(expected) != set(observations)
        ):
            raise ValueError("Destination head expectations are incomplete")
        grouped_heads: dict[tuple[str, str], dict[str, Any] | None] = {}
        grouped_versions: dict[tuple[str, str], set[int]] = {}
        conflicts = False
        for source_id, item in observations.items():
            if (
                str(UUID(source_id)) != source_id
                or not isinstance(item, dict)
                or set(item) != {"expected_version", "head", "assessment"}
                or type(item["expected_version"]) is not int
                or not 0 <= item["expected_version"] < MAX_HISTORY_PAYLOAD_VERSIONS
                or (targets is not None and item["expected_version"] != expected[source_id])
            ):
                raise ValueError("Invalid destination head expectation")
            head = item["head"]
            if head is not None and (
                not isinstance(head, dict)
                or set(head) != {"tensor_id", "version_id", "version", "deleted"}
                or str(UUID(head["tensor_id"])) != head["tensor_id"]
                or str(UUID(head["version_id"])) != head["version_id"]
                or type(head["version"]) is not int
                or head["version"] < 1
                or type(head["deleted"]) is not bool
            ):
                raise ValueError("Invalid destination head evidence")
            target = destinations[source_id]
            key = (target["dataset"], target["name"])
            if key in grouped_heads and grouped_heads[key] != head:
                raise ValueError("Destination head observations disagree")
            grouped_heads[key] = head
            versions = grouped_versions.setdefault(key, set())
            if item["expected_version"] in versions:
                raise ValueError("Destination head expectations repeat")
            versions.add(item["expected_version"])
            try:
                assessment = _assessment(
                    item["expected_version"], head, progress["payloads"][source_id]
                )
            except Conflict as exc:
                raise ValueError("Destination head contradicts a receipt") from exc
            if item["assessment"] != assessment:
                raise ValueError("Destination head assessment contradicts its evidence")
            conflicts |= assessment in ("head_advanced", "tombstoned")
        if any(versions != set(range(len(versions))) for versions in grouped_versions.values()):
            raise ValueError("Destination head expectations are not consecutive")
        if kind == "product" and any(len(versions) != 1 for versions in grouped_versions.values()):
            raise ValueError("Product destinations are not distinct")
        if report["conflicts_observed"] is not conflicts:
            raise ValueError("Destination head conflict summary is inconsistent")
    except (AttributeError, KeyError, TypeError, ValueError, RecursionError) as exc:
        raise ValueError("Invalid bundle destination head preflight") from exc


def require_no_observed_head_conflicts(report: dict[str, Any], review: dict[str, Any]) -> None:
    if not report["conflicts_observed"]:
        return
    conflicts = [
        {
            "source_version_id": source_id,
            "destination": review["request"]["destinations"][source_id],
            **item,
        }
        for source_id, item in report["targets"].items()
        if item["assessment"] in ("head_advanced", "tombstoned")
    ]
    raise TensorusError(
        "Destination version heads conflict with the reviewed restore; inspect before retrying",
        code="bundle_restore_head_conflict",
        status=409,
        details={
            "format": "tensorus-bundle-restore-head-conflicts-v1",
            "review_hash": report["review_hash"],
            "conflicts": conflicts,
            "point_in_time_atomic": False,
            "retry_guaranteed": False,
        },
    )
