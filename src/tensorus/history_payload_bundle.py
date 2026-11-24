"""Payload-bearing packages for explicitly selected, unauthenticated release histories."""

import hashlib
import io
import tarfile
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self
from urllib.parse import quote
from uuid import UUID

from pydantic import Field, JsonValue, TypeAdapter, ValidationError, model_validator

from tensorus.arrays import (
    Array,
    canonical_hash,
    canonical_json,
    decode_unique_json,
    descriptor_nbytes,
)
from tensorus.contracts import ReleaseHash
from tensorus.errors import Conflict, Forbidden, LimitExceeded, TensorusError
from tensorus.models import DType, Model, Name, NpyWriteCondition, Role, WriteContractBinding
from tensorus.product_bundle import (
    MAX_BUNDLE_PAYLOAD_BYTES,
    MAX_BUNDLE_VERSION_RESPONSE_BYTES,
    ProductBundleDestination,
    _add_file,
    _capture_write_credential,
    _check_member,
    _checked_write_identity,
    _destination_dataset,
    _require_write_credential,
    _write_payload,
)
from tensorus.release_portability import (
    MAX_PORTABLE_HISTORY_BYTES,
    PortableHistoryExport,
    PortableReleaseHistory,
)
from tensorus.responses import Identity, TensorVersion
from tensorus.serialization import NPY_MAX_OVERHEAD_BYTES, dump_npy, load_npy

if TYPE_CHECKING:
    from tensorus.client import Client


HISTORY_PAYLOAD_BUNDLE_FORMAT: Literal["tensorus-release-history-payload-bundle-v1"] = (
    "tensorus-release-history-payload-bundle-v1"
)
MAX_HISTORY_PAYLOAD_VERSIONS = 128
MAX_HISTORY_VERSION_RESPONSE_BYTES = MAX_BUNDLE_VERSION_RESPONSE_BYTES
MAX_HISTORY_PAYLOAD_MANIFEST_BYTES = MAX_PORTABLE_HISTORY_BYTES + 2 * 1024**2
MAX_HISTORY_PAYLOAD_BUNDLE_BYTES = (
    MAX_HISTORY_PAYLOAD_MANIFEST_BYTES + MAX_BUNDLE_PAYLOAD_BYTES + 128 * 1024
)


class HistoryPayloadVersion(Model):
    id: UUID
    tenant_id: UUID
    tensor_id: UUID
    version: Annotated[int, Field(strict=True, ge=1)]
    dataset: Name
    dataset_id: UUID
    name: Name
    dtype: DType
    shape: list[Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]] = Field(max_length=16)
    nbytes: Annotated[int, Field(strict=True, ge=0)]
    checksum: ReleaseHash
    metadata: dict[str, JsonValue]
    operation: dict[str, JsonValue] | None
    parents: list[UUID]

    @model_validator(mode="after")
    def validate_descriptor(self) -> Self:
        try:
            descriptor_nbytes(self.model_dump(mode="json"))
        except TensorusError as exc:
            raise ValueError("Invalid history payload descriptor") from exc
        return self


class HistoryPayloadEntry(Model):
    source: HistoryPayloadVersion
    path: str
    file_bytes: Annotated[int, Field(strict=True, ge=1)]
    file_sha256: ReleaseHash


class HistoryPayloadManifest(Model):
    format: Literal["tensorus-release-history-payload-bundle-v1"] = HISTORY_PAYLOAD_BUNDLE_FORMAT
    coverage: Literal["selected_members_and_lineage_only"] = "selected_members_and_lineage_only"
    trust: Literal["unauthenticated_source_claims"] = "unauthenticated_source_claims"
    source: PortableReleaseHistory
    payloads: list[HistoryPayloadEntry] = Field(
        min_length=1, max_length=MAX_HISTORY_PAYLOAD_VERSIONS
    )
    content_hash: ReleaseHash

    @model_validator(mode="after")
    def validate_coverage(self) -> Self:
        entries = {item.source.id: item for item in self.payloads}
        ids = [item.source.id for item in self.payloads]
        if ids != sorted(entries, key=str) or len(entries) != len(ids):
            raise ValueError("History payloads must have unique, sorted source versions")
        expected: set[UUID] = set()
        groups: dict[UUID, tuple[UUID, UUID, str, str]] = {}
        numbered: set[tuple[UUID, int]] = set()
        for entry in self.payloads:
            version = entry.source
            group = (version.tenant_id, version.dataset_id, version.dataset, version.name)
            if (
                version.tenant_id != self.source.manifest.tenant_id
                or groups.setdefault(version.tensor_id, group) != group
                or (version.tensor_id, version.version) in numbered
                or entry.path != f"payload/{version.id}.npy"
                or not version.nbytes < entry.file_bytes <= version.nbytes + NPY_MAX_OVERHEAD_BYTES
            ):
                raise ValueError("History payload identity, path or byte length is inconsistent")
            numbered.add((version.tensor_id, version.version))
        if sum(item.file_bytes for item in self.payloads) > MAX_BUNDLE_PAYLOAD_BYTES:
            raise ValueError("History payloads exceed the byte limit")
        for release in self.source.releases:
            for member in release.release.report.members:
                expected.add(member.version_id)
                member_entry = entries.get(member.version_id)
                if member_entry is None:
                    raise ValueError("A selected release member has no payload")
                version = member_entry.source
                if (
                    version.dataset_id != release.release.dataset_id
                    or version.dataset != release.release.dataset
                    or version.name != member.name
                    or version.dtype != member.dtype
                    or version.shape != member.shape
                    or version.nbytes != member.nbytes
                    or version.checksum != member.checksum
                    or canonical_hash(version.metadata) != member.metadata_hash
                    or canonical_json(version.operation) != canonical_json(member.operation)
                    or [str(item) for item in version.parents] != member.parents
                ):
                    raise ValueError("History payload differs from its release member claim")
            if release.lineage is not None:
                for node in release.lineage.versions:
                    expected.add(node.id)
                    lineage_entry = entries.get(node.id)
                    if lineage_entry is None:
                        raise ValueError("A selected lineage version has no payload")
                    node_data = node.model_dump(mode="json")
                    version_data = lineage_entry.source.model_dump(mode="json")
                    if canonical_json({field: node_data[field] for field in version_data}) != (
                        canonical_json(version_data)
                    ):
                        raise ValueError("History payload differs from its lineage version claim")
        if set(entries) != expected:
            raise ValueError("History payloads must cover exactly the selected members and lineage")
        if (
            canonical_hash(self.model_dump(mode="json", exclude={"content_hash"}))
            != self.content_hash
        ):
            raise ValueError("History payload manifest hash does not match")
        return self


def _read_history(
    client: "Client", dataset: str, selection: PortableHistoryExport
) -> PortableReleaseHistory:
    data = client._bounded_report(
        "POST",
        f"/v1/datasets/{quote(dataset, safe='')}/release-history/export",
        MAX_PORTABLE_HISTORY_BYTES,
        "Release history",
        json=selection.model_dump(mode="json"),
    )
    try:
        raw = decode_unique_json(data)
        history = PortableReleaseHistory.model_validate(raw)
        if canonical_json(raw) != canonical_json(history.model_dump(mode="json")):
            raise ValueError("Release history is not in its exact supported representation")
        if history.manifest.dataset != dataset or len(history.releases) != len(selection.releases):
            raise ValueError("Release history does not match the request")
        for selected, release in zip(selection.releases, history.releases, strict=True):
            if (
                selected.release_id != release.release.id
                or selected.include_lineage != (release.lineage is not None)
                or selected.validity_binding_ids
                != (release.validity.selected_binding_ids if release.validity else None)
                or selected.plan_ids != (release.plans.selected_plan_ids if release.plans else None)
            ):
                raise ValueError("Release history omits or changes selected evidence")
        return history
    except (UnicodeDecodeError, ValueError, RecursionError, TensorusError) as exc:
        raise TensorusError("The server returned invalid release history evidence") from exc


def _history_ids(history: PortableReleaseHistory) -> list[UUID]:
    ids = {
        member.version_id
        for release in history.releases
        for member in release.release.report.members
    }
    ids.update(
        node.id
        for release in history.releases
        if release.lineage is not None
        for node in release.lineage.versions
    )
    if not ids or len(ids) > MAX_HISTORY_PAYLOAD_VERSIONS:
        raise LimitExceeded("Selected history exceeds the payload version-count limit")
    return sorted(ids, key=str)


def _source_version(client: "Client", version_id: UUID) -> HistoryPayloadVersion:
    data = client._bounded_report(
        "GET", f"/v1/versions/{version_id}", MAX_HISTORY_VERSION_RESPONSE_BYTES, "Version"
    )
    try:
        raw = decode_unique_json(data)
        if not isinstance(raw, dict):
            raise ValueError("Version descriptor is not an object")
        fields = HistoryPayloadVersion.model_fields
        version = HistoryPayloadVersion.model_validate({field: raw[field] for field in fields})
        if version.id != version_id:
            raise ValueError("Version identity differs from the request")
        return version
    except (UnicodeDecodeError, KeyError, ValueError, RecursionError, TensorusError) as exc:
        raise TensorusError("The server returned invalid history version evidence") from exc


def export_history_payload_bundle(
    client: "Client", dataset: str, selection: PortableHistoryExport
) -> bytes:
    """Fetch selected release members and lineage as verified canonical NPY payloads."""
    selection = PortableHistoryExport.model_validate(selection.model_dump(mode="json"))
    identity = client._checked_identity()
    history = _read_history(client, dataset, selection)
    if identity.id != history.manifest.tenant_id:
        raise TensorusError("Release history tenant differs from the authenticated identity")
    versions = [_source_version(client, version_id) for version_id in _history_ids(history)]
    if (
        sum(version.nbytes + NPY_MAX_OVERHEAD_BYTES for version in versions)
        > MAX_BUNDLE_PAYLOAD_BYTES
    ):
        raise LimitExceeded("Selected history exceeds the bundle payload byte budget")
    for version in versions:
        client._npy_headers(version.metadata, None)
        if version.nbytes > client.max_tensor_bytes:
            raise LimitExceeded("History member exceeds the SDK full-read byte limit")
    entries: list[dict[str, Any]] = []
    files: list[tuple[str, bytes]] = []
    for version in versions:
        data = dump_npy(client._get_verified_npy(version.model_dump(mode="json")))
        path = f"payload/{version.id}.npy"
        entries.append(
            {
                "source": version.model_dump(mode="json"),
                "path": path,
                "file_bytes": len(data),
                "file_sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        files.append((path, data))
    for version in versions:
        current = _source_version(client, version.id)
        if canonical_json(current.model_dump(mode="json")) != canonical_json(
            version.model_dump(mode="json")
        ):
            raise TensorusError("Source version changed during history payload export")
    current_history = _read_history(client, dataset, selection)
    if (
        canonical_json(current_history.model_dump(mode="json"))
        != canonical_json(history.model_dump(mode="json"))
        or client._checked_identity() != identity
    ):
        raise TensorusError("Source history or authenticated identity changed during export")
    body = {
        "format": HISTORY_PAYLOAD_BUNDLE_FORMAT,
        "coverage": "selected_members_and_lineage_only",
        "trust": "unauthenticated_source_claims",
        "source": history.model_dump(mode="json"),
        "payloads": entries,
    }
    try:
        manifest = HistoryPayloadManifest.model_validate(
            {**body, "content_hash": canonical_hash(body)}
        )
    except ValidationError as exc:
        raise TensorusError("Source history and payload descriptors are inconsistent") from exc
    encoded = canonical_json(manifest.model_dump(mode="json")).encode()
    if len(encoded) > MAX_HISTORY_PAYLOAD_MANIFEST_BYTES:
        raise LimitExceeded("History payload manifest exceeds the byte limit")
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        _add_file(archive, "manifest.json", encoded)
        for path, data in files:
            _add_file(archive, path, data)
    result = output.getvalue()
    if len(result) > MAX_HISTORY_PAYLOAD_BUNDLE_BYTES:
        raise LimitExceeded("History payload bundle exceeds the byte limit")
    return result


def _read_history_payload_bundle(
    client: "Client", payload: bytes
) -> tuple[HistoryPayloadManifest, dict[UUID, Array]]:
    if len(payload) > MAX_HISTORY_PAYLOAD_BUNDLE_BYTES:
        raise LimitExceeded("History payload bundle exceeds the byte limit")
    if not payload or len(payload) % tarfile.BLOCKSIZE:
        raise TensorusError("Invalid history payload bundle length")
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
            first = archive.next()
            if first is None or first.name != "manifest.json":
                raise TensorusError("History payload bundle must begin with its manifest")
            _check_member(first, MAX_HISTORY_PAYLOAD_MANIFEST_BYTES)
            stream = archive.extractfile(first)
            if stream is None:
                raise TensorusError("History payload manifest is unavailable")
            raw = decode_unique_json(stream.read())
            manifest = HistoryPayloadManifest.model_validate(raw)
            if canonical_json(raw) != canonical_json(manifest.model_dump(mode="json")):
                raise ValueError("History manifest is not in its exact supported representation")
            arrays: dict[UUID, Array] = {}
            for entry in manifest.payloads:
                member = archive.next()
                if member is None or member.name != entry.path or member.size != entry.file_bytes:
                    raise TensorusError("History payload coverage does not match its manifest")
                _check_member(member, MAX_BUNDLE_PAYLOAD_BYTES)
                stream = archive.extractfile(member)
                if stream is None:
                    raise TensorusError("History payload is unavailable")
                data = stream.read()
                if (
                    len(data) != entry.file_bytes
                    or hashlib.sha256(data).hexdigest() != entry.file_sha256
                ):
                    raise TensorusError("History payload file checksum does not match")
                descriptor = entry.source.model_dump(mode="json")
                if entry.source.nbytes > client.max_tensor_bytes:
                    raise LimitExceeded("History payload exceeds the SDK full-read byte limit")
                client._verify_npy_download(io.BytesIO(data), len(data), descriptor)
                arrays[entry.source.id] = load_npy(data, client.max_tensor_bytes)
            if archive.next() is not None:
                raise TensorusError("History payload bundle contains an unexpected member")
            trailer = payload[archive.offset :]
            if len(trailer) < 2 * tarfile.BLOCKSIZE or any(trailer):
                raise TensorusError("History payload bundle has invalid trailing data")
            return manifest, arrays
    except (
        tarfile.TarError,
        UnicodeDecodeError,
        ValidationError,
        ValueError,
        RecursionError,
    ) as exc:
        raise TensorusError("Invalid history payload bundle") from exc


def inspect_history_payload_bundle(client: "Client", payload: bytes) -> dict[str, Any]:
    """Verify all selected files and claims without writing destination records."""
    manifest, _ = _read_history_payload_bundle(client, payload)
    return manifest.model_dump(mode="json")


def _check_history_destinations(
    client: "Client",
    manifest: HistoryPayloadManifest,
    dataset: str,
    destinations: dict[str, dict[str, str]],
) -> tuple[str, dict[UUID, ProductBundleDestination]]:
    try:
        dataset = TypeAdapter(Name).validate_python(dataset)
        mapping = TypeAdapter(dict[UUID, ProductBundleDestination]).validate_python(destinations)
    except ValidationError as exc:
        raise TensorusError("Invalid history payload destination mapping") from exc
    if set(mapping) != {entry.source.id for entry in manifest.payloads} or len(mapping) != len(
        destinations
    ):
        raise TensorusError("Destination mapping must identify every source version exactly once")
    by_id = {entry.source.id: entry.source for entry in manifest.payloads}
    for release in manifest.source.releases:
        for member in release.release.report.members:
            target = mapping[member.version_id]
            if target.dataset != dataset or target.name != member.name:
                raise TensorusError(
                    "Release members require the destination dataset and source name"
                )
    groups: dict[UUID, tuple[str, str]] = {}
    targets: dict[tuple[str, str], UUID] = {}
    for source_id, target in mapping.items():
        version = by_id[source_id]
        destination = (target.dataset, target.name)
        if (
            groups.setdefault(version.tensor_id, destination) != destination
            or targets.setdefault(destination, version.tensor_id) != version.tensor_id
        ):
            raise TensorusError("Each source tensor requires one distinct destination tensor")
        client._npy_headers(version.metadata, None)
    return dataset, mapping


def _history_payload_key(
    bundle_hash: str,
    source_version_id: UUID,
    target: ProductBundleDestination,
    expected_version: int,
) -> str:
    return "history-payload-v1-" + canonical_hash(
        {
            "bundle": bundle_hash,
            "source_version_id": str(source_version_id),
            "dataset": target.dataset,
            "name": target.name,
            "expected_version": expected_version,
        }
    )


def copy_history_payload_bundle(
    client: "Client",
    payload: bytes,
    dataset: str,
    destinations: dict[str, dict[str, str]],
    *,
    _guard: Callable[[str], None] | None = None,
    _expected_contracts: dict[str, WriteContractBinding | None] | None = None,
    _conditions: dict[str, NpyWriteCondition] | None = None,
    _credential: str | None = None,
    _expected_identity: Identity | None = None,
    _source_ids: set[UUID] | None = None,
) -> dict[str, Any]:
    """Copy payloads into empty named tensors; source claims need a separate local import."""
    credential = _capture_write_credential(client) if _credential is None else _credential
    _require_write_credential(client, credential)
    manifest, arrays = _read_history_payload_bundle(client, payload)
    dataset, mapping = _check_history_destinations(client, manifest, dataset, destinations)
    by_id = {entry.source.id: entry.source for entry in manifest.payloads}
    ordered = sorted(
        by_id.values(), key=lambda item: (str(item.tensor_id), item.version, str(item.id))
    )
    selected = set(by_id) if _source_ids is None else set(_source_ids)
    if not selected or not selected <= set(by_id):
        raise TensorusError("Selected history payloads must identify source versions in the bundle")
    skipped_tensors: set[UUID] = set()
    for source in ordered:
        if source.id not in selected:
            skipped_tensors.add(source.tensor_id)
        elif source.tensor_id in skipped_tensors:
            raise TensorusError(
                "Selected history payloads must include each required tensor prefix"
            )
    identity = _checked_write_identity(client, credential, _expected_identity)
    if identity.scope is not None or identity.role not in (Role.writer, Role.admin):
        raise Forbidden("History payload copying requires an active unscoped writer")
    datasets = {
        name: _destination_dataset(client, name, identity.id, credential=credential)
        for name in sorted({target.dataset for target in mapping.values()})
    }
    if len({item.id for item in datasets.values()}) != len(datasets):
        raise TensorusError("Destination dataset identities are inconsistent")
    if _conditions is not None and (
        set(_conditions) != set(datasets)
        or any(_conditions[name].dataset_id != item.id for name, item in datasets.items())
    ):
        raise TensorusError("Destination dataset identities differ from the reviewed conditions")
    if canonical_json(datasets[dataset].spec.model_dump(mode="json")) != canonical_json(
        manifest.source.releases[0].write_spec.model_dump(mode="json")
    ):
        raise Conflict(
            "Destination dataset write specification does not match the selected history"
        )
    if client._checked_identity() != identity:
        raise Forbidden("The authenticated identity changed during history payload copying")
    copied: dict[str, Any] = {}
    copied_tensor_ids: dict[UUID, UUID] = {}
    source_tensor_ids: dict[UUID, UUID] = {}
    destination_ids: set[UUID] = set()
    counts: dict[UUID, int] = {}
    for source in ordered:
        expected_version = counts.get(source.tensor_id, 0)
        counts[source.tensor_id] = expected_version + 1
        if source.id not in selected:
            continue
        _require_write_credential(client, credential)
        if client._checked_identity() != identity:
            raise Forbidden("The authenticated identity changed during history payload copying")
        target = mapping[source.id]
        if _guard is not None:
            _guard(target.dataset)
        key = _history_payload_key(manifest.content_hash, source.id, target, expected_version)
        response = _write_payload(
            client,
            target,
            source.metadata,
            arrays[source.id],
            expected_version,
            key,
            _conditions[target.dataset] if _conditions is not None else None,
            credential=credential,
        )
        _require_write_credential(client, credential)
        try:
            if (
                not isinstance(response, dict)
                or type(response.get("version")) is not int
                or type(response.get("head_version")) is not int
                or type(response.get("nbytes")) is not int
                or not isinstance(response.get("shape"), list)
                or any(type(size) is not int for size in response["shape"])
            ):
                raise ValueError("Imported version has non-integer dimensions or sizes")
            result = TensorVersion.model_validate(response)
        except (ValidationError, ValueError) as exc:
            raise TensorusError("Destination returned an invalid imported version") from exc
        if (
            result.tenant_id != identity.id
            or result.dataset != target.dataset
            or result.dataset_id != datasets[target.dataset].id
            or result.name != target.name
            or result.version != expected_version + 1
            or result.head_version < result.version
            or result.deleted_at is not None
            or result.dtype != source.dtype
            or result.shape != source.shape
            or result.nbytes != source.nbytes
            or result.checksum != source.checksum
            or canonical_json(result.metadata) != canonical_json(source.metadata)
            or result.parents
            or result.operation is not None
            or copied_tensor_ids.setdefault(source.tensor_id, result.tensor_id) != result.tensor_id
            or source_tensor_ids.setdefault(result.tensor_id, source.tensor_id) != source.tensor_id
            or result.id in destination_ids
        ):
            raise TensorusError("Destination version contradicts the verified source payload")
        if _expected_contracts is not None and (
            target.dataset not in _expected_contracts
            or result.write_contract != _expected_contracts[target.dataset]
        ):
            raise Conflict("Imported version does not match the reviewed write contract")
        client._get_verified_npy(result.model_dump(mode="json"), _credential=credential)
        if client._checked_identity() != identity:
            raise Forbidden("The authenticated identity changed during history payload copying")
        _require_write_credential(client, credential)
        if _guard is not None:
            _guard(target.dataset)
        copied[str(source.id)] = result.model_dump(mode="json")
        destination_ids.add(result.id)
    output: dict[str, Any] = {
        "format": (
            "tensorus-release-history-payload-copy-v1"
            if _source_ids is None
            else "tensorus-selected-history-payload-copy-v1"
        ),
        "bundle_hash": manifest.content_hash,
        "source_claims_authenticated": False,
        "destination_versions": copied,
        "created_history_imports": False,
        "created_validity_bindings": False,
        "approval_transferred": False,
        "whole_copy_atomic": False,
        "local_review_required": True,
    }
    if _source_ids is not None:
        output["source_version_ids"] = [
            str(source.id) for source in ordered if source.id in selected
        ]
    return output
