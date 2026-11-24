"""Shared, opt-in diagnostics for non-atomic bundle restoration."""

from collections.abc import Callable
from typing import Any, Literal

import httpx

from tensorus.errors import TensorusError

FailureInspection = Literal["progress", "heads", "names", "occupancy", "payloads", "releases"]


def _failure_inspection(enabled: bool, value: str) -> FailureInspection:
    if type(enabled) is not bool:
        raise ValueError("inspect_on_failure must be a boolean")
    if type(value) is not str or value not in (
        "progress",
        "heads",
        "names",
        "occupancy",
        "payloads",
        "releases",
    ):
        raise ValueError(
            "failure_inspection must be progress, heads, names, occupancy, payloads or releases"
        )
    if not enabled and value != "progress":
        raise ValueError("failure_inspection requires inspect_on_failure")
    if value == "heads":
        return "heads"
    if value == "names":
        return "names"
    if value == "occupancy":
        return "occupancy"
    if value == "payloads":
        return "payloads"
    if value == "releases":
        return "releases"
    return "progress"


class BundleRestoreApplyFailure(TensorusError):
    def __init__(
        self,
        original: TensorusError,
        progress: dict[str, Any] | None,
        inspection_error: TensorusError | None,
        *,
        inspection_kind: FailureInspection = "progress",
        inspection: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            str(original), code=original.code, status=original.status, details=original.details
        )
        self.original = original
        self.progress = progress
        self.inspection_error = inspection_error
        self.inspection_kind = inspection_kind
        self.inspection = inspection

    def report(self) -> dict[str, Any]:
        error = {"code": self.code, "message": str(self), "status": self.status}
        result = {
            "inspection_error": (
                {
                    "code": self.inspection_error.code,
                    "message": str(self.inspection_error),
                    "status": self.inspection_error.status,
                }
                if self.inspection_error is not None
                else None
            ),
            "may_have_committed": True,
            "point_in_time_atomic": False,
            "retry_guaranteed": False,
        }
        if self.inspection_kind == "progress":
            return {
                "format": "tensorus-bundle-restore-apply-failure-v1",
                "apply_error": error,
                "progress": self.progress,
                **result,
            }
        return {
            "format": (
                "tensorus-bundle-restore-apply-failure-v5"
                if self.inspection_kind == "releases"
                else "tensorus-bundle-restore-apply-failure-v4"
                if self.inspection_kind == "payloads"
                else "tensorus-bundle-restore-apply-failure-v3"
                if self.inspection_kind == "occupancy"
                else "tensorus-bundle-restore-apply-failure-v2"
            ),
            "apply_error": error,
            "inspection_kind": self.inspection_kind,
            "inspection": self.inspection,
            **result,
        }


def _prepare_inspection_review(
    enabled: bool,
    reviewed: dict[str, Any] | None,
    expected_hash: str,
    entries_key: Literal["roles", "datasets"],
) -> dict[str, Any] | None:
    if type(enabled) is not bool:
        raise ValueError("inspect_on_failure must be a boolean")
    if reviewed is None:
        return None
    if not enabled:
        raise ValueError("reviewed requires inspect_on_failure")
    from tensorus.product_bundle import _inspection_review

    return _inspection_review(reviewed, expected_hash, reviewed, entries_key)


def _apply_with_inspection(
    apply: Callable[[], dict[str, Any]],
    inspect: Callable[[], dict[str, Any]],
    enabled: bool,
    *,
    inspection_kind: FailureInspection = "progress",
) -> dict[str, Any]:
    inspection_kind = _failure_inspection(enabled, inspection_kind)

    def perform(action: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        try:
            return action()
        except httpx.HTTPError as exc:
            raise TensorusError(
                "The destination service is unavailable", code="connection_error", status=503
            ) from exc
        except ValueError as exc:
            raise TensorusError("The destination URL or restore inputs are invalid") from exc

    try:
        return perform(apply)
    except TensorusError as original:
        if not enabled:
            raise
        try:
            observation = perform(inspect)
            if not isinstance(observation, dict):
                raise TensorusError(
                    "The destination returned invalid failure inspection evidence",
                    code="invalid_response",
                )
            progress: object = None
            if inspection_kind == "occupancy":
                from tensorus.arrays import canonical_json
                from tensorus.bundle_restore_names import NameOccupancy

                try:
                    checked = NameOccupancy.model_validate_json(
                        canonical_json(observation), strict=True
                    )
                    if canonical_json(observation) != canonical_json(
                        checked.model_dump(mode="json")
                    ):
                        raise ValueError("Name occupancy evidence is not exact")
                except (TypeError, ValueError, RecursionError, TensorusError) as exc:
                    raise TensorusError(
                        "The destination returned invalid failure inspection evidence",
                        code="invalid_response",
                    ) from exc
            elif inspection_kind == "payloads":
                if (
                    observation.get("format")
                    not in {
                        "tensorus-product-bundle-restore-payload-observation-v1",
                        "tensorus-history-payload-restore-payload-observation-v1",
                    }
                    or observation.get("complete", False) is not None
                    or any(
                        observation.get(field) is not False
                        for field in (
                            "retained_records_checked",
                            "completion_assessed",
                            "heads_checked",
                            "names_checked",
                            "authorizes_restore",
                        )
                    )
                ):
                    raise TensorusError(
                        "The destination returned invalid failure inspection evidence",
                        code="invalid_response",
                    )
            elif inspection_kind == "releases":
                if (
                    observation.get("format") != "tensorus-history-restore-release-observation-v1"
                    or not isinstance(observation.get("releases"), dict)
                    or any(
                        observation.get(field) is not False
                        for field in (
                            "heads_checked",
                            "names_reserved",
                            "point_in_time_atomic",
                            "retry_guaranteed",
                            "authorizes_restore",
                            "source_claims_authenticated",
                        )
                    )
                ):
                    raise TensorusError(
                        "The destination returned invalid failure inspection evidence",
                        code="invalid_response",
                    )
            elif inspection_kind == "progress":
                progress = observation
            elif inspection_kind == "heads":
                progress = observation.get("progress")
            else:
                head = observation.get("head_preflight")
                progress = head.get("progress") if isinstance(head, dict) else None
            if inspection_kind not in ("occupancy", "payloads", "releases") and not isinstance(
                progress, dict
            ):
                raise TensorusError(
                    "The destination returned invalid failure inspection evidence",
                    code="invalid_response",
                )
        except TensorusError as inspection_error:
            raise BundleRestoreApplyFailure(
                original, None, inspection_error, inspection_kind=inspection_kind
            ) from original
        raise BundleRestoreApplyFailure(
            original,
            progress if isinstance(progress, dict) else None,
            None,
            inspection_kind=inspection_kind,
            inspection=observation,
        ) from original
