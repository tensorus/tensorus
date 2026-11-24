"""Bounded, non-reserving observations of bundle restore destination names."""
from collections.abc import Callable
from typing import Any
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import StrictBool
from pydantic import StrictInt
from pydantic import ValidationError
from pydantic import field_validator
from pydantic import model_validator
from tensorus.arrays import canonical_json
from tensorus.arrays import decode_unique_json
from tensorus.bundle_restore_heads import HeadExpectation
from tensorus.bundle_restore_heads import validate_head_preflight
from tensorus.errors import Conflict
from tensorus.errors import Forbidden
from tensorus.errors import TensorusError
from tensorus.models import Model
from tensorus.models import Name
from tensorus.models import Role
from tensorus.product_bundle import _capture_write_credential
from tensorus.product_bundle import _checked_reviewed_identity
from tensorus.product_bundle import _checked_write_identity
from tensorus.product_bundle import _require_write_credential
from tensorus.release_portability import MAX_PORTABLE_HISTORY_RELEASES
PREFLIGHT_FORMAT = "tensorus-bundle-restore-name-preflight-v1"
PATH = "/v1/restore-bundles/name-occupancy"
MAX_RESPONSE_BYTES = 32 * 1024
class ProductNames(Model):
    name: Name
    import_name: Name
class HistoryNamesEntry(Model):
    release_id: UUID
    name: Name
    snapshot_name: Name
class HistoryNames(Model):
    dataset: Name
    releases: list[HistoryNamesEntry] = Field(
        min_length=1, max_length=MAX_PORTABLE_HISTORY_RELEASES
    )

    @model_validator(mode="after")
    def distinct_names(self) -> Self:
        for values in (
            [item.release_id for item in self.releases],
            [item.name for item in self.releases],
            [item.snapshot_name for item in self.releases],
        ):
            if len(set(values)) != len(values):
                raise ValueError("Restore release and snapshot names must be distinct")
        return self
class NameOccupancyRequest(Model):
    kind: Literal["product", "history"]
    product: ProductNames | None = None
    history: HistoryNames | None = None

    @model_validator(mode="after")
    def matching_kind(self) -> Self:
        if (self.kind == "product") != (self.product is not None) or (self.kind == "history") != (
            self.history is not None
        ):
            raise ValueError("Name selections must match the bundle kind")
        return self
class ProductOccupant(Model):
    id: UUID
    revision: StrictInt = Field(ge=1, le=2**31 - 1)
    retired: StrictBool
    import_id: UUID | None
class HistoryOccupant(Model):
    import_id: UUID | None
    import_snapshot_id: UUID | None
    snapshot_id: UUID | None

    @model_validator(mode="after")
    def import_pair(self) -> Self:
        if (self.import_id is None) != (self.import_snapshot_id is None):
            raise ValueError("Imported history identity is incomplete")
        return self
class NameOccupancy(Model):
    format: Literal["tensorus-bundle-restore-name-occupancy-v1"]
    tenant_id: UUID
    actor_key_id: UUID
    actor_role: Literal["writer", "admin"]
    request: NameOccupancyRequest
    dataset_id: UUID | None
    product: ProductOccupant | None
    history: dict[UUID, HistoryOccupant] | None
    point_in_time_atomic: Literal[False]
    names_reserved: Literal[False]

    @field_validator("point_in_time_atomic", "names_reserved", mode="before")
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Name observation flags must be booleans")
        return value

    @model_validator(mode="after")
    def coverage(self) -> Self:
        if self.request.kind == "product":
            if self.dataset_id is not None or self.history is not None:
                raise ValueError("Product name evidence is inconsistent")
        else:
            request = self.request.history
            if (
                request is None
                or self.dataset_id is None
                or self.product is not None
                or self.history is None
                or set(self.history) != {item.release_id for item in request.releases}
            ):
                raise ValueError("History name evidence is incomplete")
        return self
def request_from_review(
    kind: Literal["product", "history"], review: dict[str, Any]
) -> NameOccupancyRequest:
    request = review["request"]
    if kind == "product":
        return NameOccupancyRequest(
            kind=kind,
            product=ProductNames(name=request["product_name"], import_name=request["import_name"]),
        )
    return NameOccupancyRequest.model_validate(
        {
            "kind": kind,
            "history": {
                "dataset": request["dataset"],
                "releases": [
                    {
                        "release_id": item["release_id"],
                        "name": item["name"],
                        "snapshot_name": item["snapshot_name"],
                    }
                    for item in request["releases"]
                ],
            },
        }
    )
def observe_destination_names(
    client: "Client", request: NameOccupancyRequest, *, _credential: str | None = None
) -> dict[str, Any]:
    if not isinstance(request, NameOccupancyRequest):
        raise TypeError("A destination name occupancy request is required")
    request = NameOccupancyRequest.model_validate_json(
        canonical_json(request.model_dump(mode="json")), strict=True
    )
    credential = _capture_write_credential(client) if _credential is None else _credential
    _require_write_credential(client, credential)
    identity, capabilities = client._checked_capabilities(_credential=credential)
    _checked_write_identity(client, credential, identity)
    if identity.scope is not None or identity.role not in (Role.writer, Role.admin):
        raise Forbidden("Destination name observation requires an unchanged unscoped writer")
    if "bundle_restore_name_occupancy" not in capabilities:
        raise TensorusError(
            "The server does not advertise bundle destination name observation",
            code="unsupported_capability",
            status=409,
        )
    data = client._bounded_report(
        "POST",
        PATH,
        MAX_RESPONSE_BYTES,
        "Destination names",
        headers={"Authorization": credential},
        json=request.model_dump(mode="json"),
    )
    try:
        names = decode_unique_json(data)
        checked = NameOccupancy.model_validate_json(data, strict=True)
        if (
            canonical_json(names) != canonical_json(checked.model_dump(mode="json"))
            or checked.request != request
            or checked.tenant_id != identity.id
            or checked.actor_key_id != identity.key_id
            or checked.actor_role != identity.role.value
        ):
            raise ValueError("Name observation differs from the request or identity")
    except (
        TypeError,
        UnicodeError,
        ValueError,
        RecursionError,
        TensorusError,
    ) as exc:
        raise TensorusError("Destination returned invalid name evidence") from exc
    _checked_write_identity(client, credential, identity)
    return checked.model_dump(mode="json")
def inspect_reviewed_occupancy(
    client: "Client",
    payload: bytes,
    request: "ProductBundleRestoreRequest | HistoryPayloadRestoreRequest",
    *,
    kind: Literal["product", "history"],
    expected_review_hash: str,
    reviewed: dict[str, Any] | None,
) -> dict[str, Any]:
    from tensorus.history_payload_restore import HistoryPayloadRestoreRequest
    from tensorus.product_bundle import _inspection_review
    from tensorus.product_bundle_restore import ProductBundleRestoreRequest

    credential = _capture_write_credential(client)
    if kind == "product" and isinstance(request, ProductBundleRestoreRequest):
        request = request.model_copy(deep=True)
        snapshot = (
            _inspection_review(reviewed, expected_review_hash, reviewed, "roles")
            if reviewed is not None
            else None
        )
        current = client.review_product_bundle_restore(payload, request)
        original = _inspection_review(current, expected_review_hash, snapshot, "roles")
        observed = observe_destination_names(
            client, request_from_review(kind, original), _credential=credential
        )
        rechecked = client.review_product_bundle_restore(payload, request)
    elif kind == "history" and isinstance(request, HistoryPayloadRestoreRequest):
        request = request.model_copy(deep=True)
        snapshot = (
            _inspection_review(reviewed, expected_review_hash, reviewed, "datasets")
            if reviewed is not None
            else None
        )
        current = client.review_release_history_payload_restore(payload, request)
        original = _inspection_review(current, expected_review_hash, snapshot, "datasets")
        observed = observe_destination_names(
            client, request_from_review(kind, original), _credential=credential
        )
        rechecked = client.review_release_history_payload_restore(payload, request)
        destinations = [
            item["dataset_id"]
            for item in original["datasets"]
            if item["dataset"] == request.dataset
        ]
        if len(destinations) != 1 or observed["dataset_id"] != destinations[0]:
            raise TensorusError("Destination name observation identifies another dataset")
    else:
        raise TensorusError("The restore request does not match the bundle type")
    if any(
        observed[field] != original[field] for field in ("tenant_id", "actor_key_id", "actor_role")
    ):
        raise Forbidden("The authenticated identity changed during destination name observation")
    if rechecked != current:
        raise Conflict("Restore review changed during destination name observation")
    _checked_reviewed_identity(client, credential, original)
    return observed
def _assess(kind: str, names: dict[str, Any], progress: dict[str, Any]) -> dict[str, Any]:
    if kind == "product":
        occupant = names["product"]
        product = progress["product"]
        imported = progress["source_import"]
        if occupant is None:
            if product["status"] == "committed" or imported["status"] == "committed":
                raise ValueError("Retained product disappeared during name inspection")
            return {"product": "not_observed", "product_import": "not_observed"}
        if product["status"] == "committed":
            if (
                occupant["id"] != product["record"]["id"]
                or occupant["revision"] != product["record"]["revision"]
                or occupant["retired"]
            ):
                raise ValueError("Retained product changed during name inspection")
            product_status = "retained"
        else:
            product_status = "retired" if occupant["retired"] else "occupied_unverified"
        import_id = occupant["import_id"]
        if imported["status"] == "committed":
            if import_id != imported["record"]["id"]:
                raise ValueError("Retained product import changed during name inspection")
            import_status = "retained"
        else:
            import_status = "occupied_unverified" if import_id is not None else "not_observed"
        return {"product": product_status, "product_import": import_status}

    result: dict[str, str] = {}
    for release_id, occupant in names["history"].items():
        imported = progress["history_imports"][release_id]
        if imported["status"] == "committed":
            record = imported["record"]
            if (
                occupant["import_id"] != record["id"]
                or occupant["import_snapshot_id"] != record["snapshot_id"]
                or occupant["snapshot_id"] != record["snapshot_id"]
            ):
                raise ValueError("Retained history changed during name inspection")
            status = "retained"
        elif occupant["import_id"] is not None:
            status = "occupied_unverified"
        elif occupant["snapshot_id"] is not None:
            status = "snapshot_occupied"
        else:
            status = "not_observed"
        result[release_id] = status
    return {"history": result}
def validate_name_preflight(
    kind: Literal["product", "history"],
    report: Any,
    review: dict[str, Any],
    *,
    targets: list[HeadExpectation] | None = None,
) -> None:
    try:
        if (
            not isinstance(report, dict)
            or set(report)
            != {
                "format",
                "kind",
                "review_hash",
                "bundle_hash",
                "head_preflight",
                "names",
                "assessments",
                "blocked_observed",
                "point_in_time_atomic",
                "retry_guaranteed",
                "authorizes_restore",
            }
            or report["format"] != PREFLIGHT_FORMAT
            or report["kind"] != kind
            or report["review_hash"] != review["review_hash"]
            or report["bundle_hash"] != review["bundle_hash"]
            or type(report["blocked_observed"]) is not bool
            or any(
                report[field] is not False
                for field in ("point_in_time_atomic", "retry_guaranteed", "authorizes_restore")
            )
        ):
            raise ValueError("Invalid name preflight identity")
        head = report["head_preflight"]
        validate_head_preflight(kind, head, review, targets=targets)
        names = report["names"]
        checked = NameOccupancy.model_validate_json(canonical_json(names), strict=True)
        if (
            canonical_json(names) != canonical_json(checked.model_dump(mode="json"))
            or str(checked.tenant_id) != review["tenant_id"]
            or str(checked.actor_key_id) != review["actor_key_id"]
            or checked.actor_role != review["actor_role"]
            or checked.request != request_from_review(kind, review)
        ):
            raise ValueError("Name observation differs from the review")
        if kind == "history":
            dataset_id = next(
                item["dataset_id"]
                for item in review["datasets"]
                if item["dataset"] == review["request"]["dataset"]
            )
            if str(checked.dataset_id) != dataset_id:
                raise ValueError("Name observation references another dataset")
        assessments = _assess(kind, names, head["progress"])
        if assessments != report["assessments"]:
            raise ValueError("Name assessment contradicts its evidence")
        values = assessments.values() if kind == "product" else assessments["history"].values()
        blocked = head["conflicts_observed"] or any(
            value not in ("not_observed", "retained") for value in values
        )
        if report["blocked_observed"] is not blocked:
            raise ValueError("Name preflight summary is inconsistent")
    except (AttributeError, KeyError, TypeError, ValueError, RecursionError, StopIteration) as exc:
        raise ValueError("Invalid bundle destination name preflight") from exc
def inspect_destination_names(
    client: "Client",
    *,
    kind: Literal["product", "history"],
    review: dict[str, Any],
    head: dict[str, Any],
    credential: str,
    recheck: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    validate_head_preflight(kind, head, review)
    identity = _checked_reviewed_identity(client, credential, review)
    request = request_from_review(kind, review)
    names = observe_destination_names(client, request, _credential=credential)
    try:
        checked = NameOccupancy.model_validate_json(canonical_json(names), strict=True)
        if (
            canonical_json(names) != canonical_json(checked.model_dump(mode="json"))
            or checked.request != request
            or checked.tenant_id != identity.id
            or checked.actor_key_id != identity.key_id
            or checked.actor_role != identity.role.value
        ):
            raise ValueError("Name observation differs from the request or identity")
        assessments = _assess(kind, names, head["progress"])
        values = assessments.values() if kind == "product" else assessments["history"].values()
        blocked = head["conflicts_observed"] or any(
            value not in ("not_observed", "retained") for value in values
        )
        report = {
            "format": PREFLIGHT_FORMAT,
            "kind": kind,
            "review_hash": head["review_hash"],
            "bundle_hash": head["bundle_hash"],
            "head_preflight": head,
            "names": names,
            "assessments": assessments,
            "blocked_observed": blocked,
            "point_in_time_atomic": False,
            "retry_guaranteed": False,
            "authorizes_restore": False,
        }
        validate_name_preflight(kind, report, review)
    except (
        AttributeError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        RecursionError,
        ValidationError,
    ) as exc:
        raise TensorusError("Destination returned invalid name evidence") from exc
    if recheck()["review_hash"] != head["progress"]["current_review_hash"]:
        raise Conflict("Restore review changed during destination name inspection")
    _checked_write_identity(client, credential, identity)
    return report
def require_no_observed_name_conflicts(report: dict[str, Any]) -> None:
    if report["blocked_observed"]:
        raise TensorusError(
            "Destination names are occupied or cannot be verified; inspect before retrying",
            code="bundle_restore_name_blocked",
            status=409,
            details={
                "format": "tensorus-bundle-restore-name-blocks-v1",
                "review_hash": report["review_hash"],
                "assessments": report["assessments"],
                "point_in_time_atomic": False,
                "retry_guaranteed": False,
            },
        )
