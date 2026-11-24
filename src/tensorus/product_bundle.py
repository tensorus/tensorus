"""Bounded payload packages for explicitly reviewed, unauthenticated product claims."""

import copy
import hashlib
import io
import tarfile
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Self
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
from tensorus.models import (
    ConditionalNpyReceiptRequest,
    Model,
    Name,
    NpyWriteCondition,
    WriteContractBinding,
)
from tensorus.product_portability import PortableProductRelease
from tensorus.product_releases import MAX_PORTABLE_PRODUCT_BYTES
from tensorus.products import MAX_PRODUCT_MEMBERS
from tensorus.responses import Dataset, Identity, TensorVersion, WriteContractState
from tensorus.serialization import NPY_MAX_OVERHEAD_BYTES, dump_npy, load_npy

if TYPE_CHECKING:
    from tensorus.client import Client


PRODUCT_BUNDLE_FORMAT: Literal["tensorus-product-payload-bundle-v1"] = (
    "tensorus-product-payload-bundle-v1"
)
MAX_BUNDLE_PAYLOAD_BYTES = 16 * 1024**2
MAX_BUNDLE_MANIFEST_BYTES = MAX_PORTABLE_PRODUCT_BYTES + 64 * 1024
MAX_BUNDLE_BYTES = MAX_BUNDLE_MANIFEST_BYTES + MAX_BUNDLE_PAYLOAD_BYTES + 64 * 1024
MAX_BUNDLE_VERSION_RESPONSE_BYTES = 96 * 1024
MAX_ACTIVE_WRITE_CONTRACT_BYTES = 2 * 1024**2
MAX_RESTORE_REVIEW_BYTES = 8 * 1024**2


class ProductBundlePayload(Model):
    source_version_id: UUID
    path: str
    file_bytes: int = Field(strict=True, ge=1)
    file_sha256: ReleaseHash


class ProductBundleManifest(Model):
    format: Literal["tensorus-product-payload-bundle-v1"] = PRODUCT_BUNDLE_FORMAT
    coverage: Literal["selected_product_members_only"] = "selected_product_members_only"
    trust: Literal["unauthenticated_source_claims"] = "unauthenticated_source_claims"
    source: PortableProductRelease
    payloads: list[ProductBundlePayload] = Field(min_length=1, max_length=MAX_PRODUCT_MEMBERS)
    content_hash: ReleaseHash

    @model_validator(mode="after")
    def validate_coverage(self) -> Self:
        versions = {
            member.version.id: member.version for member in self.source.product.definition.members
        }
        ids = [entry.source_version_id for entry in self.payloads]
        if ids != sorted(versions, key=str) or len(set(ids)) != len(ids):
            raise ValueError("Bundle payloads must cover each selected version exactly once")
        for entry in self.payloads:
            version = versions[entry.source_version_id]
            size = descriptor_nbytes(version.model_dump(mode="json"))
            if (
                entry.path != f"payload/{entry.source_version_id}.npy"
                or not size < entry.file_bytes <= size + NPY_MAX_OVERHEAD_BYTES
            ):
                raise ValueError("Bundle payload path or byte length does not match its version")
        if sum(entry.file_bytes for entry in self.payloads) > MAX_BUNDLE_PAYLOAD_BYTES:
            raise ValueError("Bundle payloads exceed the byte limit")
        if (
            canonical_hash(self.model_dump(mode="json", exclude={"content_hash"}))
            != self.content_hash
        ):
            raise ValueError("Bundle manifest hash does not match")
        return self


class ProductBundleDestination(Model):
    dataset: Name
    name: Name


def _read_model[T: Model](
    client: "Client",
    path: str,
    model: type[T],
    label: str,
    limit: int,
    *,
    credential: str | None = None,
) -> T:
    data = client._bounded_report(
        "GET",
        path,
        limit,
        label,
        headers={"Authorization": credential} if credential is not None else None,
    )
    try:
        raw = decode_unique_json(data)
        result = model.model_validate_json(data, strict=True)
        if canonical_json(raw) != canonical_json(result.model_dump(mode="json")):
            raise ValueError("Evidence is not in its exact supported representation")
        return result
    except (UnicodeDecodeError, ValueError, RecursionError, TensorusError) as exc:
        raise TensorusError(f"Destination returned invalid {label.lower()} evidence") from exc


def _destination_dataset(
    client: "Client", name: str, tenant_id: UUID, *, credential: str | None = None
) -> Dataset:
    data = client._bounded_report(
        "GET",
        f"/v1/datasets/{quote(name, safe='')}",
        MAX_BUNDLE_VERSION_RESPONSE_BYTES,
        "Destination dataset",
        headers={"Authorization": credential} if credential is not None else None,
    )
    try:
        raw = decode_unique_json(data)
        dataset = Dataset.model_validate_json(data, strict=True)
        if (
            canonical_json(raw) != canonical_json(dataset.model_dump(mode="json"))
            or dataset.tenant_id != tenant_id
            or dataset.name != name
            or dataset.archived_at is not None
        ):
            raise ValueError("Destination dataset identity does not match")
        return dataset
    except (UnicodeDecodeError, ValueError, RecursionError, TensorusError) as exc:
        raise TensorusError("The server returned invalid destination dataset evidence") from exc


def _destination_state(
    client: "Client", name: str, tenant_id: UUID, *, credential: str | None = None
) -> tuple[Dataset, dict[str, Any]]:
    dataset = _destination_dataset(client, name, tenant_id, credential=credential)
    data = client._bounded_report(
        "GET",
        f"/v1/datasets/{quote(name, safe='')}/write-contract",
        MAX_ACTIVE_WRITE_CONTRACT_BYTES,
        "Write contract",
        headers={"Authorization": credential} if credential is not None else None,
    )
    try:
        raw = decode_unique_json(data)
        state = WriteContractState.model_validate_json(data, strict=True)
        if (
            canonical_json(raw) != canonical_json(state.model_dump(mode="json"))
            or state.dataset_id != dataset.id
            or state.generation < 0
        ):
            raise ValueError("Write contract belongs to another dataset")
        if state.generation == 0:
            if any(
                item is not None
                for item in (
                    state.revision,
                    state.definition_hash,
                    state.definition,
                    state.activated_at,
                    state.activated_by,
                )
            ):
                raise ValueError("Inactive write contract has activation evidence")
        else:
            WriteContractBinding.model_validate(
                {
                    "generation": state.generation,
                    "revision": state.revision,
                    "definition_hash": state.definition_hash,
                }
            )
            if (
                state.definition is None
                or canonical_hash(state.definition.model_dump(mode="json")) != state.definition_hash
                or not state.activated_at
                or not state.activated_by
            ):
                raise ValueError("Active write contract evidence is inconsistent")
    except (UnicodeDecodeError, ValidationError, ValueError, RecursionError, TensorusError) as exc:
        raise TensorusError("Destination returned inconsistent write-contract evidence") from exc
    return dataset, {
        "dataset": name,
        "dataset_id": str(dataset.id),
        "write_spec_hash": canonical_hash(dataset.spec.model_dump(mode="json")),
        "active_write_contract": {
            "generation": state.generation,
            "revision": state.revision,
            "definition_hash": state.definition_hash,
            "activated_at": state.activated_at,
            "activated_by": state.activated_by,
        },
    }


def _require_conditional_ingestion(client: "Client", *, credential: str | None = None) -> None:
    _, raw = (
        client._checked_capabilities()
        if credential is None
        else client._checked_capabilities(_credential=credential)
    )
    contracts = raw.get("write_contracts")
    if not isinstance(contracts, dict):
        raise TensorusError(
            "Destination returned invalid conditional ingestion capabilities",
            code="invalid_response",
        )
    if "conditional_npy_ingest" not in contracts:
        raise TensorusError(
            "Destination server does not support conditional NPY ingestion",
            code="unsupported_capability",
            status=409,
        )


def _capture_write_credential(client: "Client") -> str:
    credential = client.http.headers.get("Authorization")
    if credential is None:
        raise Forbidden("An authenticated credential is required for these writes")
    return credential


def _require_write_credential(client: "Client", credential: str) -> None:
    if client.http.headers.get("Authorization") != credential:
        raise Forbidden("The authenticated credential changed; earlier writes may have committed")


def _checked_write_identity(
    client: "Client", credential: str, expected: Identity | None = None
) -> Identity:
    _require_write_credential(client, credential)
    identity = client._checked_identity()
    pinned_identity = client._checked_identity_for_credential(credential)
    _require_write_credential(client, credential)
    if identity != pinned_identity or (expected is not None and pinned_identity != expected):
        raise Forbidden("The authenticated identity changed during the operation")
    return pinned_identity


def _checked_reviewed_identity(
    client: "Client", credential: str, review: dict[str, Any]
) -> Identity:
    identity = _checked_write_identity(client, credential)
    if (
        str(identity.id) != review["tenant_id"]
        or str(identity.key_id) != review["actor_key_id"]
        or identity.role.value != review["actor_role"]
        or identity.scope is not None
    ):
        raise Forbidden("The authenticated identity differs from the restore review")
    return identity


def _write_payload(
    client: "Client",
    target: ProductBundleDestination,
    metadata: dict[str, JsonValue],
    array: Array,
    expected_version: int,
    key: str,
    condition: NpyWriteCondition | None = None,
    *,
    credential: str,
) -> Any:
    path = f"/v1/datasets/{quote(target.dataset, safe='')}/tensors/{quote(target.name, safe='')}"
    params: dict[str, str | int] = {"expected_version": expected_version}
    if condition is None:
        path += "/npy"
    else:
        path += "/conditional-npy"
        params.update(
            expected_dataset_id=str(condition.dataset_id),
            expected_write_contract_generation=condition.write_contract_generation,
        )
    headers = client._npy_headers(metadata, key)
    headers["Authorization"] = credential
    data = client._bounded_report(
        "PUT",
        path,
        MAX_BUNDLE_VERSION_RESPONSE_BYTES,
        "Imported version",
        headers=headers,
        params=params,
        content=dump_npy(array),
    )
    try:
        return decode_unique_json(data)
    except (UnicodeDecodeError, ValueError, RecursionError, TensorusError) as exc:
        raise TensorusError("Destination returned an invalid imported version") from exc


def _product_payload_key(
    bundle_hash: str, source_version_id: UUID, target: ProductBundleDestination
) -> str:
    return "product-bundle-v1-" + canonical_hash(
        {
            "bundle": bundle_hash,
            "source_version_id": str(source_version_id),
            "dataset": target.dataset,
            "name": target.name,
        }
    )


def _inspection_review(
    current: dict[str, Any],
    expected_hash: str,
    reviewed: dict[str, Any] | None,
    entries_key: Literal["roles", "datasets"],
) -> dict[str, Any]:
    if reviewed is None:
        if current["review_hash"] != expected_hash:
            raise Conflict("Restore review changed; supply the original review for inspection")
        return current
    try:
        encoded = canonical_json(reviewed).encode()
        if len(encoded) > MAX_RESTORE_REVIEW_BYTES:
            raise ValueError("Restore review exceeds its byte limit")
        snapshot = decode_unique_json(encoded)
        if (
            snapshot.get("review_hash") != expected_hash
            or canonical_hash(
                {key: value for key, value in snapshot.items() if key != "review_hash"}
            )
            != expected_hash
        ):
            raise ValueError("Restore review hash does not match")
        normalized = copy.deepcopy(snapshot)
        old_entries, current_entries = normalized[entries_key], current[entries_key]
        if not isinstance(old_entries, list) or len(old_entries) != len(current_entries):
            raise ValueError("Restore review entries differ")
        seen: dict[str, str] = {}
        for old, now in zip(old_entries, current_entries, strict=True):
            old_policy, new_policy = old["active_write_contract"], now["active_write_contract"]
            if not isinstance(old_policy, dict) or set(old_policy) != set(new_policy):
                raise ValueError("Restore review policy fields differ")
            generation = old_policy["generation"]
            if type(generation) is not int or not 0 <= generation <= new_policy["generation"]:
                raise ValueError("Restore review policy generation is invalid")
            if generation:
                WriteContractBinding.model_validate(
                    {
                        field: old_policy[field]
                        for field in ("generation", "revision", "definition_hash")
                    }
                )
                if "activated_at" in old_policy and (
                    not isinstance(old_policy["activated_at"], str)
                    or not old_policy["activated_at"]
                    or not isinstance(old_policy["activated_by"], str)
                    or not old_policy["activated_by"]
                ):
                    raise ValueError("Restore review activation is invalid")
            elif any(value is not None for key, value in old_policy.items() if key != "generation"):
                raise ValueError("Inactive restore review policy has activation evidence")
            if generation == new_policy["generation"] and old_policy != new_policy:
                raise ValueError("Restore review policy identity contradicts current policy")
            dataset_id = now["destination_dataset_id" if entries_key == "roles" else "dataset_id"]
            policy_json = canonical_json(old_policy)
            if seen.setdefault(dataset_id, policy_json) != policy_json:
                raise ValueError("Restore review has contradictory dataset policies")
            old["active_write_contract"] = new_policy
        normalized["review_hash"] = current["review_hash"]
        if canonical_json(normalized) != canonical_json(current):
            raise ValueError("Restore review differs beyond its write policies")
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        RecursionError,
        ValidationError,
    ) as exc:
        raise Conflict(
            "Original restore review is invalid or differs beyond policy changes"
        ) from exc
    return snapshot


def _inspect_copied_payload(
    client: "Client",
    identity: Identity,
    source: dict[str, Any],
    target: ProductBundleDestination,
    expected_version: int,
    condition: NpyWriteCondition,
    contract: WriteContractBinding | None,
    current_contract: WriteContractBinding | None,
    key: str,
    *,
    credential: str,
    allow_tombstoned: bool = False,
) -> dict[str, Any]:
    request = ConditionalNpyReceiptRequest(
        checksum=source["checksum"],
        metadata=source["metadata"],
        expected_version=expected_version,
        condition=condition,
    )
    if allow_tombstoned:
        receipt = client._read_conditional_npy_historical_receipt(
            target.dataset,
            target.name,
            request,
            key,
            identity,
            allow_tombstoned=True,
            _credential=credential,
        )
    else:
        receipt = client._read_conditional_npy_historical_receipt(
            target.dataset, target.name, request, key, identity, _credential=credential
        )
    expected_current = current_contract.model_dump(mode="json") if current_contract else None
    if receipt["current_write_contract"] != expected_current:
        raise Conflict("Restore review changed during inspection")
    version = receipt["version"]
    if version is not None:
        expected_contract = contract.model_dump(mode="json") if contract is not None else None
        if (
            version["dtype"] != source["dtype"]
            or version["shape"] != source["shape"]
            or version["nbytes"] != source["nbytes"]
            or (
                receipt["retained_policy_matches_review"]
                and version["write_contract"] != expected_contract
            )
        ):
            raise TensorusError("Destination receipt contradicts the verified source payload")
        retained = _read_model(
            client,
            f"/v1/versions/{quote(version['id'], safe='')}",
            TensorVersion,
            "Retained version",
            MAX_BUNDLE_VERSION_RESPONSE_BYTES,
            credential=credential,
        ).model_dump(mode="json")
        if canonical_json({k: v for k, v in version.items() if k != "head_version"}) != (
            canonical_json({k: v for k, v in retained.items() if k != "head_version"})
        ):
            raise TensorusError("Destination returned contradictory retained version evidence")
        client._get_verified_npy(version, _credential=credential)
    return {
        "status": receipt["status"],
        "version": version,
        "payload_verified": version is not None,
        "retry_guaranteed": False,
        "review_policy_current": receipt["review_policy_current"],
        "retained_policy_matches_review": receipt["retained_policy_matches_review"],
        "retained_policy_current": receipt["retained_policy_current"],
        "current_write_contract": receipt["current_write_contract"],
        "continuation": receipt["continuation"],
        "next_action": receipt["next_action"],
    }


def _add_file(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o600
    info.mtime = 0
    archive.addfile(info, io.BytesIO(data))


def export_product_bundle(client: "Client", release_id: str) -> bytes:
    """Fetch every selected immutable member and package its verified canonical NPY."""
    identity = client._checked_identity()
    source = PortableProductRelease.model_validate(client.export_product_release(release_id))
    if source.product.tenant_id != identity.id:
        raise TensorusError("Product-release tenant differs from the authenticated identity")
    versions = {member.version.id: member.version for member in source.product.definition.members}
    if (
        sum(
            descriptor_nbytes(version.model_dump(mode="json")) + NPY_MAX_OVERHEAD_BYTES
            for version in versions.values()
        )
        > MAX_BUNDLE_PAYLOAD_BYTES
    ):
        raise LimitExceeded("Selected product members exceed the bundle payload budget")
    entries: list[dict[str, Any]] = []
    files: list[tuple[str, bytes]] = []
    for version_id in sorted(versions, key=str):
        descriptor = versions[version_id].model_dump(mode="json")
        array = client._get_verified_npy(descriptor)
        data = dump_npy(array)
        path = f"payload/{version_id}.npy"
        entries.append(
            {
                "source_version_id": str(version_id),
                "path": path,
                "file_bytes": len(data),
                "file_sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        files.append((path, data))
    if (
        canonical_json(client.export_product_release(release_id))
        != canonical_json(source.model_dump(mode="json"))
        or client._checked_identity() != identity
    ):
        raise TensorusError(
            "Product-release claims or authenticated identity changed during export"
        )
    body = {
        "format": PRODUCT_BUNDLE_FORMAT,
        "coverage": "selected_product_members_only",
        "trust": "unauthenticated_source_claims",
        "source": source.model_dump(mode="json"),
        "payloads": entries,
    }
    manifest = ProductBundleManifest.model_validate({**body, "content_hash": canonical_hash(body)})
    encoded = canonical_json(manifest.model_dump(mode="json")).encode()
    if len(encoded) > MAX_BUNDLE_MANIFEST_BYTES:
        raise LimitExceeded("Bundle manifest exceeds the byte limit")
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        _add_file(archive, "manifest.json", encoded)
        for path, data in files:
            _add_file(archive, path, data)
    result = output.getvalue()
    if len(result) > MAX_BUNDLE_BYTES:
        raise LimitExceeded("Product bundle exceeds the byte limit")
    return result


def _read_bundle(
    client: "Client", payload: bytes
) -> tuple[ProductBundleManifest, dict[UUID, Array]]:
    if len(payload) > MAX_BUNDLE_BYTES:
        raise LimitExceeded("Product bundle exceeds the byte limit")
    if not payload or len(payload) % tarfile.BLOCKSIZE:
        raise TensorusError("Invalid product bundle length")
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
            first = archive.next()
            if first is None or first.name != "manifest.json":
                raise TensorusError("Product bundle must begin with its manifest")
            _check_member(first, MAX_BUNDLE_MANIFEST_BYTES)
            stream = archive.extractfile(first)
            if stream is None:
                raise TensorusError("Product bundle manifest is unavailable")
            raw = decode_unique_json(stream.read())
            manifest = ProductBundleManifest.model_validate(raw)
            if canonical_json(raw) != canonical_json(manifest.model_dump(mode="json")):
                raise ValueError("Product manifest is not in its exact supported representation")
            versions = {
                member.version.id: member.version
                for member in manifest.source.product.definition.members
            }
            arrays: dict[UUID, Array] = {}
            for entry in manifest.payloads:
                member = archive.next()
                if member is None or member.name != entry.path or member.size != entry.file_bytes:
                    raise TensorusError(
                        "Product bundle payload coverage does not match its manifest"
                    )
                _check_member(member, MAX_BUNDLE_PAYLOAD_BYTES)
                stream = archive.extractfile(member)
                if stream is None:
                    raise TensorusError("Product bundle payload is unavailable")
                data = stream.read()
                if (
                    len(data) != entry.file_bytes
                    or hashlib.sha256(data).hexdigest() != entry.file_sha256
                ):
                    raise TensorusError("Product bundle payload file checksum does not match")
                descriptor = versions[entry.source_version_id].model_dump(mode="json")
                if descriptor_nbytes(descriptor) > client.max_tensor_bytes:
                    raise LimitExceeded("Bundle member exceeds the SDK full-read byte limit")
                client._verify_npy_download(io.BytesIO(data), len(data), descriptor)
                arrays[entry.source_version_id] = load_npy(data, client.max_tensor_bytes)
            if archive.next() is not None:
                raise TensorusError("Product bundle contains an unexpected member")
            trailer = payload[archive.offset :]
            if len(trailer) < 2 * tarfile.BLOCKSIZE or any(trailer):
                raise TensorusError("Product bundle has invalid trailing data")
            return manifest, arrays
    except (
        tarfile.TarError,
        UnicodeDecodeError,
        ValidationError,
        ValueError,
        RecursionError,
    ) as exc:
        raise TensorusError("Invalid product bundle") from exc


def _check_member(member: tarfile.TarInfo, byte_limit: int) -> None:
    if (
        member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE)
        or member.offset_data != member.offset + tarfile.BLOCKSIZE
        or member.pax_headers
        or member.size < 0
        or member.size > byte_limit
    ):
        raise TensorusError("Unsupported or oversized product bundle member")


def inspect_product_bundle(client: "Client", payload: bytes) -> dict[str, Any]:
    """Verify every file and return claims without creating destination records."""
    manifest, _ = _read_bundle(client, payload)
    return manifest.model_dump(mode="json")


def import_product_bundle(
    client: "Client",
    payload: bytes,
    destinations: dict[str, dict[str, str]],
    *,
    _conditions: dict[str, NpyWriteCondition] | None = None,
    _expected_contracts: dict[str, WriteContractBinding | None] | None = None,
    _credential: str | None = None,
    _expected_identity: Identity | None = None,
    _guard: Callable[[str], None] | None = None,
    _source_ids: set[UUID] | None = None,
) -> dict[str, Any]:
    """Copy verified payloads only; an interrupted import can be retried with the same mapping."""
    credential = _capture_write_credential(client) if _credential is None else _credential
    _require_write_credential(client, credential)
    manifest, arrays = _read_bundle(client, payload)
    try:
        mapping = TypeAdapter(dict[UUID, ProductBundleDestination]).validate_python(destinations)
    except ValidationError as exc:
        raise TensorusError("Invalid product bundle destination mapping") from exc
    if set(mapping) != set(arrays) or len(mapping) != len(destinations):
        raise TensorusError("Destination mapping must identify every source version exactly once")
    selected = set(mapping) if _source_ids is None else set(_source_ids)
    if not selected or not selected <= set(mapping):
        raise TensorusError("Selected product payloads must identify source versions in the bundle")
    if len({(target.dataset, target.name) for target in mapping.values()}) != len(mapping):
        raise TensorusError("Different source versions require distinct destination tensors")
    versions = {
        member.version.id: member.version for member in manifest.source.product.definition.members
    }
    for version in versions.values():
        client._npy_headers(version.metadata, None)
    identity = _checked_write_identity(client, credential, _expected_identity)
    datasets = {
        name: _destination_dataset(client, name, identity.id, credential=credential)
        for name in sorted({mapping[source_id].dataset for source_id in selected})
    }
    if len({item.id for item in datasets.values()}) != len(datasets):
        raise TensorusError("Destination dataset identities are inconsistent")
    if _conditions is not None and (
        set(_conditions) != set(datasets)
        or any(_conditions[name].dataset_id != dataset.id for name, dataset in datasets.items())
    ):
        raise TensorusError("Destination dataset identities differ from the reviewed conditions")
    if _expected_contracts is not None and set(_expected_contracts) != set(datasets):
        raise TensorusError("Destination write contracts differ from the reviewed conditions")
    if client._checked_identity() != identity:
        raise TensorusError("The authenticated identity changed during product payload copying")
    copied: dict[str, Any] = {}
    destination_ids: set[UUID] = set()
    destination_tensor_ids: set[UUID] = set()
    for entry in manifest.payloads:
        if entry.source_version_id not in selected:
            continue
        _require_write_credential(client, credential)
        if client._checked_identity() != identity:
            raise TensorusError("The authenticated identity changed during product payload copying")
        version_id = entry.source_version_id
        source = versions[version_id]
        target = mapping[version_id]
        if _guard is not None:
            _guard(target.dataset)
        key = _product_payload_key(manifest.content_hash, version_id, target)
        condition = _conditions[target.dataset] if _conditions is not None else None
        response = _write_payload(
            client,
            target,
            source.metadata,
            arrays[version_id],
            0,
            key,
            condition,
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
            or result.version != 1
            or result.head_version < result.version
            or result.deleted_at is not None
            or result.dtype != source.dtype
            or result.shape != source.shape
            or result.nbytes != source.nbytes
            or result.checksum != source.checksum
            or canonical_json(result.metadata) != canonical_json(source.metadata)
            or result.parents
            or result.operation is not None
            or result.id in destination_ids
            or result.tensor_id in destination_tensor_ids
            or (
                condition is not None
                and (result.write_contract.generation if result.write_contract is not None else 0)
                != condition.write_contract_generation
            )
            or (
                _expected_contracts is not None
                and result.write_contract != _expected_contracts[target.dataset]
            )
        ):
            raise TensorusError("Destination version contradicts the verified source payload")
        client._get_verified_npy(result.model_dump(mode="json"), _credential=credential)
        if client._checked_identity() != identity:
            raise TensorusError("The authenticated identity changed during product payload copying")
        _require_write_credential(client, credential)
        if _guard is not None:
            _guard(target.dataset)
        copied[str(version_id)] = result.model_dump(mode="json")
        destination_ids.add(result.id)
        destination_tensor_ids.add(result.tensor_id)
    return {
        "format": (
            "tensorus-product-payload-import-v1"
            if _source_ids is None
            else "tensorus-selected-product-payload-import-v1"
        ),
        "bundle_hash": manifest.content_hash,
        "source_claims_authenticated": False,
        "destination_versions": copied,
        "created_validity_bindings": False,
        "created_product_or_release": False,
        "whole_import_atomic": False,
        "local_review_required": True,
    }
