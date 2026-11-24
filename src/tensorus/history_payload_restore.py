"""Reviewed, resumable reconstruction of selected release-history payloads."""

from typing import TYPE_CHECKING, Annotated, Any
from urllib.parse import quote
from uuid import UUID

from pydantic import Field, StrictBool, StrictInt, ValidationError, field_validator

from tensorus import responses as dto
from tensorus.arrays import canonical_hash, canonical_json, decode_unique_json
from tensorus.bundle_restore_heads import (
    history_head_expectations,
    inspect_destination_heads,
    require_no_observed_head_conflicts,
)
from tensorus.contracts import MAX_RELEASE_CONTRACT_OVERRIDES, ReleaseHash
from tensorus.errors import Conflict, Forbidden, LimitExceeded, TensorusError
from tensorus.history_payload_bundle import (
    MAX_HISTORY_PAYLOAD_VERSIONS,
    HistoryPayloadManifest,
    _check_history_destinations,
    _history_payload_key,
    _read_history_payload_bundle,
    copy_history_payload_bundle,
)
from tensorus.models import Model, Name, NpyWriteCondition, Role, TensorQuery, WriteContractBinding
from tensorus.product_bundle import (
    MAX_BUNDLE_VERSION_RESPONSE_BYTES,
    ProductBundleDestination,
    _capture_write_credential,
    _checked_reviewed_identity,
    _checked_write_identity,
    _destination_state,
    _inspect_copied_payload,
    _inspection_review,
    _read_model,
    _require_conditional_ingestion,
    _require_write_credential,
)
from tensorus.product_bundle_restore import _checked_asserted_binding, _find_asserted_binding
from tensorus.release_portability import (
    MAX_DESCRIPTOR_BYTES,
    MAX_PORTABLE_BYTES,
    MAX_PORTABLE_HISTORY_IMPORT_BYTES,
    MAX_PORTABLE_HISTORY_RELEASES,
    ImportedRelease,
    ImportedReleaseHistoryBundle,
    PortableRelease,
    ReleaseContractMapping,
    ReleaseHistoryBundleImport,
    ReleaseHistoryDestination,
    ReleaseHistoryImport,
)
from tensorus.validity import ValidityBinding, ValidityBindingCreate

if TYPE_CHECKING:
    from tensorus.client import Client


MAX_RESTORE_HISTORY_RESPONSE_BYTES = MAX_PORTABLE_HISTORY_IMPORT_BYTES + 1024**2
MAX_RESTORE_SNAPSHOT_QUERY_BYTES = MAX_DESCRIPTOR_BYTES + 2 * 1024**2


class HistoryRestoreRelease(Model):
    release_id: UUID
    name: Name
    snapshot_name: Name
    contract_revision: Annotated[StrictInt, Field(ge=1, le=2**31 - 1)]
    contract_mappings: list[ReleaseContractMapping] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_RELEASE_CONTRACT_OVERRIDES,
        exclude_if=lambda value: value is None,
    )


class HistoryPayloadRestoreRequest(Model):
    dataset: Name
    destinations: dict[UUID, ProductBundleDestination] = Field(
        min_length=1, max_length=MAX_HISTORY_PAYLOAD_VERSIONS
    )
    releases: list[HistoryRestoreRelease] = Field(
        min_length=1, max_length=MAX_PORTABLE_HISTORY_RELEASES
    )
    assert_selected_validity: StrictBool = False

    @field_validator("destinations", mode="before")
    @classmethod
    def unique_source_identities(cls, value: object) -> object:
        if isinstance(value, dict):
            try:
                identities = {UUID(str(item)) for item in value}
            except ValueError as exc:
                raise ValueError("Bundle destination keys must be version IDs") from exc
            if len(identities) != len(value):
                raise ValueError("Bundle destination version IDs must be distinct")
        return value


def _checked_request(request: HistoryPayloadRestoreRequest) -> HistoryPayloadRestoreRequest:
    try:
        return HistoryPayloadRestoreRequest.model_validate(request.model_dump(mode="json"))
    except ValidationError as exc:
        raise TensorusError("Invalid history payload restore request") from exc


def _history_import(
    manifest: HistoryPayloadManifest,
    request: HistoryPayloadRestoreRequest,
    copied: dict[str, Any] | None = None,
) -> ReleaseHistoryBundleImport:
    choices = {item.release_id: item for item in request.releases}
    if len(choices) != len(request.releases) or set(choices) != {
        release.release.id for release in manifest.source.releases
    }:
        raise TensorusError("Restore releases must cover each selected release exactly once")

    try:
        destinations = [
            _history_destination(release, choices[release.release.id], copied)
            for release in manifest.source.releases
        ]
        result = ReleaseHistoryBundleImport.model_validate(
            {
                "source": manifest.source.model_dump(mode="json"),
                "destinations": [item.model_dump(mode="json") for item in destinations],
            }
        )
        if any(
            len(canonical_json(item.model_dump(mode="json")).encode()) > MAX_PORTABLE_BYTES
            for item in result.import_requests()
        ):
            raise LimitExceeded("An individual release history import exceeds 2 MiB")
        return result
    except ValidationError as exc:
        raise TensorusError("Invalid selected release or contract mappings") from exc


def _history_destination(
    release: PortableRelease,
    choice: HistoryRestoreRelease,
    copied: dict[str, Any] | None = None,
) -> ReleaseHistoryDestination:
    def mappings(ids: list[UUID]) -> list[dict[str, str]]:
        return [
            {
                "source_version_id": str(source_id),
                "destination_version_id": (
                    copied[str(source_id)]["id"] if copied is not None else str(source_id)
                ),
            }
            for source_id in ids
        ]

    item = choice.model_dump(mode="json")
    item["members"] = mappings([member.version_id for member in release.release.report.members])
    if release.lineage is not None:
        item["lineage_members"] = mappings([node.id for node in release.lineage.versions])
    return ReleaseHistoryDestination.model_validate(item)


def _selected_bindings(manifest: HistoryPayloadManifest) -> list[ValidityBinding]:
    selected: dict[UUID, ValidityBinding] = {}
    for release in manifest.source.releases:
        if release.validity is not None:
            bindings = {binding.id: binding for binding in release.validity.bindings}
            for binding_id in release.validity.selected_binding_ids:
                selected[binding_id] = bindings[binding_id]
    return [selected[key] for key in sorted(selected, key=str)]


def _review(
    client: "Client",
    manifest: HistoryPayloadManifest,
    request: HistoryPayloadRestoreRequest,
    *,
    _credential: str | None = None,
) -> dict[str, Any]:
    destinations = request.model_dump(mode="json")["destinations"]
    _, mapping = _check_history_destinations(client, manifest, request.dataset, destinations)
    _history_import(manifest, request)
    selected = _selected_bindings(manifest)
    pairs = [(item.data_version_id, item.mask_version_id) for item in selected]
    if request.assert_selected_validity and len(set(pairs)) != len(pairs):
        raise TensorusError("Distinct selected bindings cannot share one local asserted pair")
    credential = _capture_write_credential(client) if _credential is None else _credential
    actor = _checked_write_identity(client, credential)
    if actor.scope is not None or actor.role not in (Role.writer, Role.admin):
        raise Forbidden("History restoration requires an active unscoped writer")
    datasets: dict[str, dto.Dataset] = {}
    states: dict[str, dict[str, Any]] = {}
    for name in sorted({target.dataset for target in mapping.values()}):
        datasets[name], states[name] = _destination_state(
            client, name, actor.id, credential=credential
        )
    if len({item.id for item in datasets.values()}) != len(datasets):
        raise TensorusError("Destination dataset identities are inconsistent")
    if canonical_json(datasets[request.dataset].spec.model_dump(mode="json")) != canonical_json(
        manifest.source.releases[0].write_spec.model_dump(mode="json")
    ):
        raise Conflict(
            "Destination dataset write specification does not match the selected history"
        )

    choices = {item.release_id: item for item in request.releases}
    contracts: list[dict[str, Any]] = []
    cached: dict[int, dto.ContractRevision] = {}
    for release in manifest.source.releases:
        choice = choices[release.release.id]
        sources = {item.revision: item for item in release.member_contracts or []}
        pairs_to_check = [(release.contract, choice.contract_revision)]
        pairs_to_check.extend(
            (sources[item.source_revision], item.destination_revision)
            for item in choice.contract_mappings or []
        )
        for source, revision in pairs_to_check:
            if revision not in cached:
                cached[revision] = _read_model(
                    client,
                    f"/v1/datasets/{quote(request.dataset, safe='')}/contracts/{revision}",
                    dto.ContractRevision,
                    "Contract revision",
                    MAX_PORTABLE_BYTES,
                    credential=credential,
                )
            local = cached[revision]
            if (
                local.dataset_id != datasets[request.dataset].id
                or local.revision != revision
                or canonical_hash(local.definition.model_dump(mode="json")) != local.definition_hash
            ):
                raise TensorusError("Destination returned inconsistent contract evidence")
            if local.definition_hash != source.definition_hash:
                raise Conflict("Destination contract does not match the selected source contract")
            contracts.append(
                {
                    "release_id": str(release.release.id),
                    "source_revision": source.revision,
                    "destination_revision": revision,
                    "definition_hash": local.definition_hash,
                }
            )
    _checked_write_identity(client, credential, actor)
    body = {
        "format": "tensorus-history-payload-restore-review-v2",
        "write_condition_format": "tensorus-conditional-npy-ingest-v1",
        "bundle_hash": manifest.content_hash,
        "source_manifest_hash": manifest.source.manifest.manifest_hash,
        "source_claims_authenticated": False,
        "tenant_id": str(actor.id),
        "actor_key_id": str(actor.key_id),
        "actor_role": actor.role.value,
        "request": request.model_dump(mode="json"),
        "datasets": [states[name] for name in sorted(states)],
        "contracts": contracts,
        "selected_validity": [
            {
                "source_binding_id": str(item.id),
                "source_descriptor_hash": item.descriptor_hash,
                "source_origin": item.origin.kind,
                "data_version_id": str(item.data_version_id),
                "mask_version_id": str(item.mask_version_id),
                "destination_origin": "asserted" if request.assert_selected_validity else None,
            }
            for item in selected
        ],
        "validity_assertions_required": bool(selected) and request.assert_selected_validity,
        "source_lineage_recreated": False,
        "source_plans_recreated": False,
        "source_validity_provenance_recreated": False,
        "approval_transferred": False,
        "local_release_created": False,
        "whole_restore_atomic": False,
    }
    return {**body, "review_hash": canonical_hash(body)}


def review_history_payload_restore(
    client: "Client", payload: bytes, request: HistoryPayloadRestoreRequest
) -> dict[str, Any]:
    """Verify selected payloads, local mappings and contracts without writing."""
    credential = _capture_write_credential(client)
    manifest, arrays = _read_history_payload_bundle(client, payload)
    del arrays
    _require_conditional_ingestion(client, credential=credential)
    return _review(client, manifest, _checked_request(request), _credential=credential)


def _checked_import(
    client: "Client",
    request: HistoryPayloadRestoreRequest,
    history: ReleaseHistoryBundleImport,
    review: dict[str, Any],
    copied: dict[str, Any],
    credential: str,
) -> dict[str, Any]:
    _require_write_credential(client, credential)
    data = client._bounded_report(
        "POST",
        f"/v1/datasets/{quote(request.dataset, safe='')}/release-history/bundle-imports",
        MAX_RESTORE_HISTORY_RESPONSE_BYTES,
        "Imported history",
        headers={"Authorization": credential},
        json=history.model_dump(mode="json"),
    )
    _require_write_credential(client, credential)
    try:
        raw = decode_unique_json(data)
        result = ImportedReleaseHistoryBundle.model_validate_json(data, strict=True)
        expected = history.import_requests()
        dataset_id = next(
            item["dataset_id"] for item in review["datasets"] if item["dataset"] == request.dataset
        )
        if (
            canonical_json(raw) != canonical_json(result.model_dump(mode="json"))
            or result.manifest_hash != history.source.manifest.manifest_hash
            or len(result.imports) != len(expected)
            or len({item.id for item in result.imports}) != len(expected)
            or len({item.snapshot_id for item in result.imports}) != len(expected)
        ):
            raise ValueError("Imported history coverage is inconsistent")
        for item, wanted in zip(result.imports, expected, strict=True):
            encoded = wanted.model_dump(mode="json")
            if (
                str(item.tenant_id) != review["tenant_id"]
                or str(item.dataset_id) != dataset_id
                or item.dataset != request.dataset
                or item.name != wanted.name
                or item.contract_revision != wanted.contract_revision
                or item.request_hash != canonical_hash(encoded)
                or item.source_hash != wanted.source.content_hash
                or canonical_json(item.request.model_dump(mode="json")) != canonical_json(encoded)
            ):
                raise ValueError("Imported history contradicts the reviewed request")
            if str(UUID(item.created_by)) != item.created_by:
                raise ValueError("Imported history creator identity is not canonical")
    except (UnicodeDecodeError, ValueError, RecursionError, TensorusError) as exc:
        raise TensorusError("Destination returned invalid imported history evidence") from exc
    for item, wanted in zip(result.imports, expected, strict=True):
        stored = _read_model(
            client,
            f"/v1/release-history/imports/{item.id}",
            ImportedRelease,
            "Imported release",
            MAX_PORTABLE_BYTES + MAX_BUNDLE_VERSION_RESPONSE_BYTES,
            credential=credential,
        )
        if canonical_json(stored.model_dump(mode="json")) != canonical_json(
            item.model_dump(mode="json")
        ):
            raise TensorusError("Destination returned contradictory retained history evidence")
        try:
            _check_snapshot(
                client,
                item.snapshot_id,
                wanted,
                review,
                copied,
                dataset_id,
                item.created_by,
                credential=credential,
            )
        except (UnicodeDecodeError, ValueError, RecursionError, KeyError) as exc:
            raise TensorusError("Destination returned invalid imported snapshot evidence") from exc
    return result.model_dump(mode="json")


def _check_snapshot(
    client: "Client",
    snapshot_id: UUID,
    wanted: ReleaseHistoryImport,
    review: dict[str, Any],
    copied: dict[str, Any],
    dataset_id: str,
    created_by: str,
    *,
    credential: str,
    allow_tombstoned: bool = False,
) -> None:
    snapshot = _read_model(
        client,
        f"/v1/snapshots/{snapshot_id}",
        dto.SnapshotDetail,
        "Snapshot",
        MAX_BUNDLE_VERSION_RESPONSE_BYTES,
        credential=credential,
    )
    expected = {
        str(item.destination_version_id): copied[str(item.source_version_id)]
        for item in wanted.members
    }
    if (
        snapshot.id != snapshot_id
        or str(snapshot.tenant_id) != review["tenant_id"]
        or str(snapshot.dataset_id) != dataset_id
        or snapshot.name != wanted.snapshot_name
        or snapshot.created_by != created_by
        or snapshot.tensor_count != len(expected)
    ):
        raise ValueError("Imported snapshot contradicts its reviewed membership")
    query = TensorQuery(
        dataset=review["request"]["dataset"],
        snapshot_id=snapshot_id,
        limit=MAX_HISTORY_PAYLOAD_VERSIONS,
    )
    encoded = client._bounded_report(
        "POST",
        "/v1/query",
        MAX_RESTORE_SNAPSHOT_QUERY_BYTES,
        "Snapshot query",
        headers={"Authorization": credential},
        json=query.model_dump(mode="json"),
    )
    raw = decode_unique_json(encoded)
    page = dto.Page[dto.TensorVersion].model_validate_json(encoded, strict=True)
    if (
        canonical_json(raw) != canonical_json(page.model_dump(mode="json"))
        or page.total != len(expected)
        or page.offset != 0
        or page.limit != MAX_HISTORY_PAYLOAD_VERSIONS
        or len(page.items) != len(expected)
        or {str(item.id) for item in page.items} != set(expected)
        or [(item.dataset, item.name) for item in page.items]
        != sorted({(item.dataset, item.name) for item in page.items})
    ):
        raise ValueError("Imported snapshot query has inconsistent membership or totals")
    for item in page.items:
        target = expected[str(item.id)]
        actual = item.model_dump(mode="json")
        stable_fields = (
            "id",
            "tenant_id",
            "dataset_id",
            "dataset",
            "tensor_id",
            "name",
            "version",
            "dtype",
            "shape",
            "nbytes",
            "checksum",
            "metadata",
            "statistics",
            "statistics_scope",
            "operation",
            "parents",
            "created_at",
            "created_by",
            "write_contract",
            "access",
        )
        if (
            str(item.tenant_id) != review["tenant_id"]
            or str(item.dataset_id) != dataset_id
            or item.dataset != review["request"]["dataset"]
            or item.operation is not None
            or item.parents
            or (item.deleted_at is not None and not allow_tombstoned)
            or any(
                canonical_json(actual.get(field)) != canonical_json(target.get(field))
                for field in stable_fields
            )
        ):
            raise ValueError("Imported snapshot member contradicts its verified payload")


def _release_requirements(release: PortableRelease) -> dict[str, list[str]]:
    versions = {str(item.version_id) for item in release.release.report.members}
    if release.lineage is not None:
        versions.update(str(item.id) for item in release.lineage.versions)
    bindings = (
        {str(item) for item in release.validity.selected_binding_ids}
        if release.validity is not None
        else set()
    )
    return {"payloads": sorted(versions), "validity_bindings": sorted(bindings)}


def validate_history_release_observation(
    observation: object,
    manifest: HistoryPayloadManifest,
    request: HistoryPayloadRestoreRequest,
    review: dict[str, Any],
) -> None:
    """Check a per-release observation against the original review and archive."""
    from tensorus.bundle_inspection_contract import validate_progress

    try:
        if (
            not isinstance(observation, dict)
            or set(observation)
            != {
                "format",
                "review_hash",
                "current_review_hash",
                "bundle_hash",
                "payload_observation",
                "releases",
                "complete",
                "conflicts_observed",
                "heads_checked",
                "names_reserved",
                "point_in_time_atomic",
                "retry_guaranteed",
                "authorizes_restore",
                "source_claims_authenticated",
            }
            or observation["format"] != "tensorus-history-restore-release-observation-v1"
            or observation["review_hash"] != review["review_hash"]
            or observation["bundle_hash"] != manifest.content_hash
            or observation["bundle_hash"] != review["bundle_hash"]
            or canonical_json(request.model_dump(mode="json")) != canonical_json(review["request"])
            or any(
                observation[field] is not False
                for field in (
                    "heads_checked",
                    "names_reserved",
                    "point_in_time_atomic",
                    "retry_guaranteed",
                    "authorizes_restore",
                    "source_claims_authenticated",
                )
            )
            or type(observation["complete"]) is not bool
            or type(observation["conflicts_observed"]) is not bool
        ):
            raise ValueError("Invalid history release observation identity")
        payloads = observation["payload_observation"]
        validate_progress("history", payloads, review, payloads_only=True)
        if observation["current_review_hash"] != payloads["current_review_hash"]:
            raise ValueError("Current policy evidence differs from the payload observation")
        releases = observation["releases"]
        choices = {str(item.release_id): item for item in request.releases}
        expected = {str(item.release.id): item for item in manifest.source.releases}
        if (
            not isinstance(releases, dict)
            or set(releases) != set(expected)
            or set(choices) != set(expected)
        ):
            raise ValueError("History release observation coverage is inconsistent")
        copied = {
            key: value["version"]
            for key, value in payloads["payloads"].items()
            if value["status"] == "committed"
        }
        dataset_id = next(
            item["dataset_id"] for item in review["datasets"] if item["dataset"] == request.dataset
        )
        seen_imports: set[str] = set()
        seen_snapshots: set[str] = set()
        statuses: list[str] = []
        for source_id, release in expected.items():
            item = releases[source_id]
            requirements = _release_requirements(release)
            if (
                not isinstance(item, dict)
                or set(item) != {"status", "requirements", "record", "snapshot_verified"}
                or item["requirements"] != requirements
                or type(item["snapshot_verified"]) is not bool
                or item["status"] not in ("not_checked", "not_observed", "committed", "conflicting")
            ):
                raise ValueError("Invalid history release observation")
            ready = all(identifier in copied for identifier in requirements["payloads"]) and (
                not request.assert_selected_validity
                or all(
                    payloads["validity_bindings"][identifier]["status"] == "committed"
                    for identifier in requirements["validity_bindings"]
                )
            )
            status = item["status"]
            statuses.append(status)
            if (status == "not_checked") is ready:
                raise ValueError("History release readiness is inconsistent")
            if status in ("not_checked", "not_observed"):
                if item["record"] is not None or item["snapshot_verified"] is not False:
                    raise ValueError("Unobserved history has retained evidence")
                continue
            retained = ImportedRelease.model_validate_json(
                canonical_json(item["record"]), strict=True
            )
            encoded = retained.model_dump(mode="json")
            if (
                canonical_json(item["record"]) != canonical_json(encoded)
                or str(retained.tenant_id) != review["tenant_id"]
                or str(retained.dataset_id) != dataset_id
                or retained.dataset != request.dataset
                or retained.name != choices[source_id].name
                or retained.request.name != retained.name
                or retained.request_hash != canonical_hash(retained.request.model_dump(mode="json"))
                or retained.source_hash != retained.request.source.content_hash
                or str(UUID(retained.created_by)) != retained.created_by
                or str(retained.id) in seen_imports
                or str(retained.snapshot_id) in seen_snapshots
            ):
                raise ValueError("Retained history observation has inconsistent identity")
            seen_imports.add(str(retained.id))
            seen_snapshots.add(str(retained.snapshot_id))
            wanted = _history_destination(release, choices[source_id], copied).import_request(
                release, manifest.source.manifest
            )
            matches = canonical_json(retained.request.model_dump(mode="json")) == canonical_json(
                wanted.model_dump(mode="json")
            )
            if (status == "committed") is not matches or item["snapshot_verified"] is not matches:
                raise ValueError("History release status contradicts its retained request")
        complete = (
            all(status == "committed" for status in statuses)
            and all(value["status"] == "committed" for value in payloads["payloads"].values())
            and all(
                value["status"] == "committed"
                or (not request.assert_selected_validity and value["status"] == "not_requested")
                for value in payloads["validity_bindings"].values()
            )
        )
        if observation["complete"] is not complete or observation["conflicts_observed"] is not any(
            status == "conflicting" for status in statuses
        ):
            raise ValueError("History release observation summary is inconsistent")
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
        raise ValueError("Invalid or contradictory history release observation") from exc


def _observe_history_releases(
    client: "Client",
    manifest: HistoryPayloadManifest,
    request: HistoryPayloadRestoreRequest,
    review: dict[str, Any],
    current_review: dict[str, Any],
    payload_observation: dict[str, Any],
    *,
    credential: str,
) -> dict[str, Any]:
    copied = {
        key: value["version"]
        for key, value in payload_observation["payloads"].items()
        if value["status"] == "committed"
    }
    choices = {item.release_id: item for item in request.releases}
    dataset_id = next(
        item["dataset_id"] for item in review["datasets"] if item["dataset"] == request.dataset
    )
    releases: dict[str, dict[str, Any]] = {}
    for source in manifest.source.releases:
        requirements = _release_requirements(source)
        ready = all(identifier in copied for identifier in requirements["payloads"]) and (
            not request.assert_selected_validity
            or all(
                payload_observation["validity_bindings"][identifier]["status"] == "committed"
                for identifier in requirements["validity_bindings"]
            )
        )
        status = "not_checked"
        imported = None
        if ready:
            wanted = _history_destination(
                source, choices[source.release.id], copied
            ).import_request(source, manifest.source.manifest)
            try:
                imported = client._read_imported_release_by_name(
                    request.dataset, wanted.name, _credential=credential
                )
            except TensorusError as exc:
                if exc.status != 404 or exc.code != "not_found":
                    raise
            if imported is None:
                status = "not_observed"
            else:
                if imported["tenant_id"] != review["tenant_id"] or imported["dataset_id"] != (
                    dataset_id
                ):
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
                    raise TensorusError(
                        "Destination returned contradictory retained history evidence"
                    )
                if canonical_json(imported["request"]) != canonical_json(
                    wanted.model_dump(mode="json")
                ):
                    status = "conflicting"
                else:
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
                        raise TensorusError(
                            "Destination returned invalid imported snapshot evidence"
                        ) from exc
                    status = "committed"
        releases[str(source.release.id)] = {
            "status": status,
            "requirements": requirements,
            "record": imported,
            "snapshot_verified": status == "committed",
        }
    if _review(client, manifest, request, _credential=credential) != current_review:
        raise Conflict("History payload restore review changed during release observation")
    complete = (
        all(item["status"] == "committed" for item in releases.values())
        and all(item["status"] == "committed" for item in payload_observation["payloads"].values())
        and all(
            item["status"] == "committed"
            or (not request.assert_selected_validity and item["status"] == "not_requested")
            for item in payload_observation["validity_bindings"].values()
        )
    )
    observation = {
        "format": "tensorus-history-restore-release-observation-v1",
        "review_hash": review["review_hash"],
        "current_review_hash": current_review["review_hash"],
        "bundle_hash": manifest.content_hash,
        "payload_observation": payload_observation,
        "releases": releases,
        "complete": complete,
        "conflicts_observed": any(item["status"] == "conflicting" for item in releases.values()),
        "heads_checked": False,
        "names_reserved": False,
        "point_in_time_atomic": False,
        "retry_guaranteed": False,
        "authorizes_restore": False,
        "source_claims_authenticated": False,
    }
    try:
        validate_history_release_observation(observation, manifest, request, review)
    except ValueError as exc:
        raise TensorusError(
            "History release observation is inconsistent", code="invalid_response"
        ) from exc
    _checked_reviewed_identity(client, credential, review)
    return observation


def inspect_history_payload_restore(
    client: "Client",
    payload: bytes,
    request: HistoryPayloadRestoreRequest,
    *,
    expected_review_hash: ReleaseHash,
    reviewed: dict[str, Any] | None = None,
    _include_head_preflight: bool = False,
    _include_name_preflight: bool = False,
    _payloads_only: bool = False,
    _release_observation: bool = False,
    _credential: str | None = None,
) -> dict[str, Any]:
    """Inspect receipt-backed payloads, local bindings and exact imported snapshot membership."""
    credential = _capture_write_credential(client) if _credential is None else _credential
    _require_write_credential(client, credential)
    _payloads_only = _payloads_only or _release_observation
    if _payloads_only and (_include_head_preflight or _include_name_preflight):
        raise TensorusError("Payload-only observation cannot include destination preflight")
    _include_head_preflight = _include_head_preflight or _include_name_preflight
    manifest, arrays = _read_history_payload_bundle(client, payload)
    del arrays
    request = _checked_request(request)
    _require_conditional_ingestion(client, credential=credential)
    client._require_conditional_npy_historical_receipt(_credential=credential)
    if not _payloads_only or _release_observation:
        client._require_imported_history_name_lookup(_credential=credential)
    current_review = _review(client, manifest, request, _credential=credential)
    review = _inspection_review(current_review, expected_review_hash, reviewed, "datasets")
    identity = _checked_reviewed_identity(client, credential, review)
    states = {item["dataset"]: item for item in review["datasets"]}
    current_states = {item["dataset"]: item for item in current_review["datasets"]}
    ordered = sorted(
        (entry.source for entry in manifest.payloads),
        key=lambda item: (str(item.tensor_id), item.version, str(item.id)),
    )
    copied: dict[str, Any] = {}
    payloads: dict[str, dict[str, Any]] = {}
    counts: dict[UUID, int] = {}
    source_tensors: dict[str, UUID] = {}
    destination_tensors: dict[UUID, str] = {}
    destination_versions: set[str] = set()
    for source in ordered:
        target = request.destinations[source.id]
        state = states[target.dataset]
        active = state["active_write_contract"]
        condition = NpyWriteCondition(
            dataset_id=state["dataset_id"], write_contract_generation=active["generation"]
        )
        contract = (
            WriteContractBinding.model_validate(
                {field: active[field] for field in ("generation", "revision", "definition_hash")}
            )
            if active["generation"]
            else None
        )
        current_active = current_states[target.dataset]["active_write_contract"]
        current_contract = (
            WriteContractBinding.model_validate(
                {
                    field: current_active[field]
                    for field in ("generation", "revision", "definition_hash")
                }
            )
            if current_active["generation"]
            else None
        )
        expected = counts.get(source.tensor_id, 0)
        evidence = _inspect_copied_payload(
            client,
            identity,
            source.model_dump(mode="json"),
            target,
            expected,
            condition,
            contract,
            current_contract,
            _history_payload_key(manifest.content_hash, source.id, target, expected),
            credential=credential,
            allow_tombstoned=_include_head_preflight,
        )
        payloads[str(source.id)] = evidence
        counts[source.tensor_id] = expected + 1
        version = evidence["version"]
        if version is not None:
            if (
                version["id"] in destination_versions
                or source_tensors.setdefault(version["tensor_id"], source.tensor_id)
                != source.tensor_id
                or destination_tensors.setdefault(source.tensor_id, version["tensor_id"])
                != version["tensor_id"]
            ):
                raise TensorusError(
                    "Destination receipt identities contradict the selected history"
                )
            copied[str(source.id)] = version
            destination_versions.add(version["id"])

    bindings: dict[str, dict[str, Any]] = {}
    binding_progress: dict[str, dict[str, Any]] = {}
    for selected in _selected_bindings(manifest):
        source_id = str(selected.id)
        if not request.assert_selected_validity:
            binding_progress[source_id] = {"status": "not_requested", "binding": None}
            continue
        data_id, mask_id = str(selected.data_version_id), str(selected.mask_version_id)
        if data_id not in copied or mask_id not in copied:
            binding_progress[source_id] = {"status": "not_checked", "binding": None}
            continue
        binding = _find_asserted_binding(
            client, copied[data_id], copied[mask_id], identity.id, credential=credential
        )
        binding_progress[source_id] = {
            "status": "committed" if binding is not None else "not_observed",
            "binding": binding,
        }
        if binding is not None:
            bindings[source_id] = binding

    import_progress: dict[str, dict[str, Any]] = {
        str(item.release_id): {"status": "not_checked", "record": None} for item in request.releases
    }
    assertions_complete = not request.assert_selected_validity or len(bindings) == len(
        binding_progress
    )
    if not _payloads_only and len(copied) == len(manifest.payloads) and assertions_complete:
        history = _history_import(manifest, request, copied)
        dataset_id = states[request.dataset]["dataset_id"]
        found_ids: set[str] = set()
        snapshot_ids: set[str] = set()
        for source_release, wanted in zip(
            manifest.source.releases, history.import_requests(), strict=True
        ):
            try:
                imported = client._read_imported_release_by_name(
                    request.dataset, wanted.name, _credential=credential
                )
            except TensorusError as exc:
                if exc.status != 404 or exc.code != "not_found":
                    raise
                imported = None
            if imported is not None:
                if (
                    imported["tenant_id"] != review["tenant_id"]
                    or imported["dataset_id"] != dataset_id
                    or canonical_json(imported["request"])
                    != canonical_json(wanted.model_dump(mode="json"))
                    or imported["id"] in found_ids
                    or imported["snapshot_id"] in snapshot_ids
                ):
                    raise Conflict("Destination history differs from the reviewed reconstruction")
                stored = _read_model(
                    client,
                    f"/v1/release-history/imports/{imported['id']}",
                    ImportedRelease,
                    "Imported release",
                    MAX_PORTABLE_BYTES + MAX_BUNDLE_VERSION_RESPONSE_BYTES,
                    credential=credential,
                )
                if canonical_json(stored.model_dump(mode="json")) != canonical_json(imported):
                    raise TensorusError(
                        "Destination returned contradictory retained history evidence"
                    )
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
                        allow_tombstoned=_include_head_preflight,
                    )
                except (UnicodeDecodeError, ValueError, RecursionError, KeyError) as exc:
                    raise TensorusError(
                        "Destination returned invalid imported snapshot evidence"
                    ) from exc
                found_ids.add(imported["id"])
                snapshot_ids.add(imported["snapshot_id"])
            import_progress[str(source_release.release.id)] = {
                "status": "committed" if imported is not None else "not_observed",
                "record": imported,
            }
    complete = all(item["status"] == "committed" for item in import_progress.values())
    if _review(client, manifest, request, _credential=credential) != current_review:
        raise Conflict("History payload restore review changed during inspection")
    blocked = any(item["continuation"] == "blocked_policy_changed" for item in payloads.values())
    progress = {
        "format": (
            "tensorus-history-payload-restore-payload-observation-v1"
            if _payloads_only
            else "tensorus-history-payload-restore-progress-v3"
            if _include_head_preflight
            else "tensorus-history-payload-restore-progress-v2"
        ),
        "review_hash": review["review_hash"],
        "current_review_hash": current_review["review_hash"],
        "bundle_hash": manifest.content_hash,
        "source_claims_authenticated": False,
        "payloads": payloads,
        "validity_bindings": binding_progress,
        "history_imports": import_progress,
        "complete": None if _payloads_only else complete,
        "status": (
            "complete"
            if complete
            else "blocked_policy_changed"
            if blocked
            else "payloads_observed"
            if _payloads_only and copied
            else "no_payloads_observed"
            if _payloads_only
            else "partial"
            if copied
            else "not_started"
        ),
        "review_policy_current": all(item["review_policy_current"] for item in payloads.values()),
        "continuation": "blocked_policy_changed" if blocked else "not_assessed",
        "next_action": (
            "review_partial_and_select_new_destinations"
            if blocked
            else "inspect_complete_progress_before_retry"
            if _payloads_only
            else "review_before_retry"
        ),
        "point_in_time_atomic": False,
        "retry_guaranteed": False,
        "source_lineage_recreated": False,
        "source_plans_recreated": False,
        "approval_transferred": False,
        "local_release_created": False,
    }
    if _payloads_only:
        progress.update(
            retained_records_checked=False,
            completion_assessed=False,
            heads_checked=False,
            names_checked=False,
            authorizes_restore=False,
        )
        from tensorus.bundle_inspection_contract import validate_progress

        try:
            validate_progress("history", progress, review, payloads_only=True)
        except ValueError as exc:
            raise TensorusError(
                "History payload observation is inconsistent", code="invalid_response"
            ) from exc
        if _release_observation:
            return _observe_history_releases(
                client, manifest, request, review, current_review, progress, credential=credential
            )
    if not _include_head_preflight:
        _checked_write_identity(client, credential, identity)
        return progress
    heads = inspect_destination_heads(
        client,
        kind="history",
        review=review,
        progress=progress,
        targets=history_head_expectations(manifest, review),
        credential=credential,
        recheck=lambda: _review(client, manifest, request, _credential=credential),
    )
    if not _include_name_preflight:
        return heads
    from tensorus.bundle_restore_names import inspect_destination_names

    return inspect_destination_names(
        client,
        kind="history",
        review=review,
        head=heads,
        credential=credential,
        recheck=lambda: _review(client, manifest, request, _credential=credential),
    )


def restore_history_payload_bundle(
    client: "Client",
    payload: bytes,
    request: HistoryPayloadRestoreRequest,
    *,
    expected_review_hash: ReleaseHash,
    confirm_unauthenticated_claims: bool,
    confirm_validity_assertions: bool = False,
    reject_observed_name_conflicts: bool = False,
) -> dict[str, Any]:
    """Copy, assert selected local masks and retain claims without transferring approval."""
    if type(reject_observed_name_conflicts) is not bool:
        raise TensorusError("Name-conflict preflight selection must be a boolean")
    credential = _capture_write_credential(client)
    manifest, arrays = _read_history_payload_bundle(client, payload)
    del arrays
    request = _checked_request(request)
    _require_conditional_ingestion(client, credential=credential)
    review = _review(client, manifest, request, _credential=credential)
    if review["review_hash"] != expected_review_hash:
        raise Conflict("History payload restore review changed; inspect it again")
    actor = _checked_reviewed_identity(client, credential, review)
    if confirm_unauthenticated_claims is not True:
        raise TensorusError("Explicit confirmation of unauthenticated source claims is required")
    if review["validity_assertions_required"] and confirm_validity_assertions is not True:
        raise TensorusError("Explicit confirmation of local validity assertions is required")
    contracts_capability = client.capabilities(_credential=credential).get("write_contracts", {})
    if (
        reject_observed_name_conflicts
        or "conditional_npy_historical_receipt" in contracts_capability
    ):
        preflight = inspect_history_payload_restore(
            client,
            payload,
            request,
            expected_review_hash=expected_review_hash,
            _include_head_preflight=True,
            _include_name_preflight=reject_observed_name_conflicts,
            _credential=credential,
        )
        names = preflight if reject_observed_name_conflicts else None
        if names is not None:
            preflight = names["head_preflight"]
        progress = preflight["progress"]
        if progress["continuation"] == "blocked_policy_changed":
            raise Conflict(
                "Retained partial uses a different write policy; select new destinations"
            )
        require_no_observed_head_conflicts(preflight, review)
        if names is not None:
            from tensorus.bundle_restore_names import require_no_observed_name_conflicts

            require_no_observed_name_conflicts(names)

    def recheck() -> None:
        _require_write_credential(client, credential)
        if _review(client, manifest, request, _credential=credential) != review:
            raise Conflict("History payload restore review changed during reconstruction")
        _require_write_credential(client, credential)

    states = {item["dataset"]: item for item in review["datasets"]}

    def guard(dataset: str) -> None:
        _require_write_credential(client, credential)
        identity = _checked_write_identity(client, credential, actor)
        _, state = _destination_state(client, dataset, identity.id, credential=credential)
        if state != states[dataset]:
            raise Conflict("History payload restore review changed during reconstruction")
        _require_write_credential(client, credential)

    contracts = {
        name: (
            WriteContractBinding.model_validate(
                {
                    field: state["active_write_contract"][field]
                    for field in ("generation", "revision", "definition_hash")
                }
            )
            if state["active_write_contract"]["generation"]
            else None
        )
        for name, state in states.items()
    }
    conditions = {
        name: NpyWriteCondition(
            dataset_id=state["dataset_id"],
            write_contract_generation=state["active_write_contract"]["generation"],
        )
        for name, state in states.items()
    }
    copied = copy_history_payload_bundle(
        client,
        payload,
        request.dataset,
        request.model_dump(mode="json")["destinations"],
        _guard=guard,
        _expected_contracts=contracts,
        _conditions=conditions,
        _credential=credential,
        _expected_identity=actor,
    )["destination_versions"]
    recheck()
    bindings: dict[str, dict[str, Any]] = {}
    if request.assert_selected_validity:
        for selected in _selected_bindings(manifest):
            data, mask = (
                copied[str(selected.data_version_id)],
                copied[str(selected.mask_version_id)],
            )
            guard(data["dataset"])
            guard(mask["dataset"])
            create = ValidityBindingCreate(data_version_id=data["id"], mask_version_id=mask["id"])
            _require_write_credential(client, credential)
            encoded = client._bounded_report(
                "POST",
                "/v1/validity-bindings",
                MAX_BUNDLE_VERSION_RESPONSE_BYTES,
                "Validity binding",
                headers={"Authorization": credential},
                json=create.model_dump(mode="json"),
            )
            _require_write_credential(client, credential)
            try:
                raw = decode_unique_json(encoded)
                checked = ValidityBinding.model_validate_json(encoded, strict=True)
                if canonical_json(raw) != canonical_json(checked.model_dump(mode="json")):
                    raise ValueError("Validity response is not exact")
                bindings[str(selected.id)] = _checked_asserted_binding(
                    raw, data, mask, UUID(review["tenant_id"])
                )
            except (UnicodeDecodeError, ValueError, RecursionError, TensorusError) as exc:
                raise TensorusError("Destination returned invalid validity evidence") from exc
            stored = _read_model(
                client,
                f"/v1/validity-bindings/{checked.id}",
                ValidityBinding,
                "Validity binding",
                MAX_BUNDLE_VERSION_RESPONSE_BYTES,
                credential=credential,
            )
            if canonical_json(stored.model_dump(mode="json")) != canonical_json(
                checked.model_dump(mode="json")
            ):
                raise TensorusError("Destination returned contradictory retained validity evidence")
            guard(data["dataset"])
            guard(mask["dataset"])
    recheck()
    history = _history_import(manifest, request, copied)
    imported = _checked_import(client, request, history, review, copied, credential)
    recheck()
    return {
        "format": "tensorus-history-payload-restore-v2",
        "write_condition_format": "tensorus-conditional-npy-ingest-v1",
        "review_hash": review["review_hash"],
        "bundle_hash": manifest.content_hash,
        "source_claims_authenticated": False,
        "destination_versions": copied,
        "validity_bindings": bindings,
        "history_import": imported,
        "source_lineage_recreated": False,
        "source_plans_recreated": False,
        "source_validity_provenance_recreated": False,
        "approval_transferred": False,
        "local_release_created": False,
        "whole_restore_atomic": False,
    }
