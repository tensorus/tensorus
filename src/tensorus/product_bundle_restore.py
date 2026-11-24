"""Explicit, resumable reconstruction of selected product roles from a payload bundle."""

from typing import TYPE_CHECKING, Any
from urllib.parse import quote
from uuid import UUID

from pydantic import Field, ValidationError, field_validator

from tensorus import responses as dto
from tensorus.arrays import canonical_hash, canonical_json
from tensorus.bundle_restore_heads import (
    inspect_destination_heads,
    product_head_expectations,
    require_no_observed_head_conflicts,
)
from tensorus.contracts import ReleaseHash
from tensorus.errors import Conflict, Forbidden, TensorusError
from tensorus.models import Model, Name, NpyWriteCondition, Role, WriteContractBinding
from tensorus.product_bundle import (
    MAX_BUNDLE_VERSION_RESPONSE_BYTES,
    MAX_PRODUCT_MEMBERS,
    ProductBundleDestination,
    ProductBundleManifest,
    _capture_write_credential,
    _checked_reviewed_identity,
    _checked_write_identity,
    _destination_state,
    _inspect_copied_payload,
    _inspection_review,
    _product_payload_key,
    _read_bundle,
    _read_model,
    _require_conditional_ingestion,
    _require_write_credential,
    import_product_bundle,
)
from tensorus.product_portability import (
    ImportedProductRelease,
    ImportedProductReleaseSummary,
    ProductReleaseImportRequest,
    _validate_source_product,
)
from tensorus.product_releases import (
    MAX_PORTABLE_PRODUCT_BYTES,
    MAX_PRODUCT_RELEASE_IMPORTS_PER_PRODUCT,
)
from tensorus.products import (
    MAX_PRODUCT_BYTES,
    DataProductRevision,
    DataProductWrite,
    ProductVersion,
    Revision,
)
from tensorus.release_portability import MAX_PORTABLE_BYTES
from tensorus.validity import (
    MAX_BINDINGS_PER_VERSION,
    ValidityBinding,
    ValidityBindingCreate,
    ValidityBindingList,
    _descriptor_hash,
)

if TYPE_CHECKING:
    from tensorus.client import Client


class ProductBundleRestoreRequest(Model):
    product_name: Name
    import_name: Name
    destinations: dict[UUID, ProductBundleDestination] = Field(
        min_length=1, max_length=MAX_PRODUCT_MEMBERS
    )
    contract_revisions: dict[Name, Revision] = Field(min_length=1, max_length=MAX_PRODUCT_MEMBERS)

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


def _checked_request(request: ProductBundleRestoreRequest) -> ProductBundleRestoreRequest:
    try:
        return ProductBundleRestoreRequest.model_validate(request.model_dump(mode="json"))
    except ValidationError as exc:
        raise TensorusError("Invalid product bundle restore request") from exc


def _review(
    client: "Client",
    manifest: ProductBundleManifest,
    request: ProductBundleRestoreRequest,
    *,
    _credential: str | None = None,
) -> dict[str, Any]:
    members = manifest.source.product.definition.members
    source_ids = {entry.source_version_id for entry in manifest.payloads}
    if set(request.destinations) != source_ids:
        raise TensorusError("Restore destinations must cover every source version exactly once")
    if len({(item.dataset, item.name) for item in request.destinations.values()}) != len(
        source_ids
    ):
        raise TensorusError("Different source versions require distinct destination tensors")
    if set(request.contract_revisions) != {member.role for member in members}:
        raise TensorusError("Restore contract revisions must cover every product role exactly once")

    credential = _capture_write_credential(client) if _credential is None else _credential
    try:
        actor = _checked_write_identity(client, credential)
        if actor.scope is not None or actor.role not in (Role.writer, Role.admin):
            raise Forbidden("Product bundle restoration requires an unscoped writer")
        datasets: dict[str, dto.Dataset] = {}
        active_contracts: dict[str, dict[str, Any]] = {}
        contracts: dict[tuple[str, int], dto.ContractRevision] = {}
        reports = {item.role: item for item in manifest.source.release.report.members}
        roles: list[dict[str, Any]] = []
        for member in members:
            target = request.destinations[member.version.id]
            if target.dataset not in datasets:
                dataset, reviewed_state = _destination_state(
                    client, target.dataset, actor.id, credential=credential
                )
                datasets[target.dataset] = dataset
                state = reviewed_state["active_write_contract"]
                active_contracts[target.dataset] = {
                    field: state[field] for field in ("generation", "revision", "definition_hash")
                }
            dataset = datasets[target.dataset]
            revision = request.contract_revisions[member.role]
            key = (target.dataset, revision)
            if key not in contracts:
                contract = _read_model(
                    client,
                    f"/v1/datasets/{quote(target.dataset, safe='')}/contracts/{revision}",
                    dto.ContractRevision,
                    "Contract revision",
                    MAX_PORTABLE_BYTES,
                    credential=credential,
                )
                if (
                    contract.dataset_id != dataset.id
                    or contract.revision != revision
                    or canonical_hash(contract.definition.model_dump(mode="json"))
                    != contract.definition_hash
                ):
                    raise TensorusError("Destination returned inconsistent contract evidence")
                contracts[key] = contract
            contract = contracts[key]
            write_spec_hash = canonical_hash(dataset.spec.model_dump(mode="json"))
            if (
                contract.definition_hash != member.contract.definition_hash
                or write_spec_hash != reports[member.role].write_spec_hash
            ):
                raise Conflict("Destination role write spec or contract does not match the source")
            roles.append(
                {
                    "role": member.role,
                    "source_version_id": str(member.version.id),
                    "destination_dataset": target.dataset,
                    "destination_dataset_id": str(dataset.id),
                    "destination_name": target.name,
                    "contract_revision": revision,
                    "contract_hash": contract.definition_hash,
                    "write_spec_hash": write_spec_hash,
                    "active_write_contract": active_contracts[target.dataset],
                    "validity_mask_role": member.validity.mask_role if member.validity else None,
                    "source_quality_attached": member.quality_release is not None,
                }
            )
    except ValidationError as exc:
        raise TensorusError("Destination returned invalid restore review evidence") from exc

    if len({dataset.id for dataset in datasets.values()}) != len(datasets):
        raise TensorusError("Destination dataset identities are inconsistent")
    _checked_write_identity(client, credential, actor)

    body = {
        "format": "tensorus-product-bundle-restore-review-v2",
        "write_condition_format": "tensorus-conditional-npy-ingest-v1",
        "bundle_hash": manifest.content_hash,
        "source_hash": manifest.source.content_hash,
        "source_claims_authenticated": False,
        "tenant_id": str(actor.id),
        "actor_key_id": str(actor.key_id),
        "actor_role": actor.role.value,
        "request": request.model_dump(mode="json"),
        "roles": roles,
        "relationships": (
            [
                item.model_dump(mode="json")
                for item in manifest.source.product.definition.relationships
            ]
            if manifest.source.product.definition.relationships is not None
            else None
        ),
        "validity_assertions_required": any(member.validity is not None for member in members),
        "source_quality_attached_locally": False,
        "source_lineage_recreated": False,
        "approval_transferred": False,
        "local_release_created": False,
        "whole_restore_atomic": False,
    }
    return {**body, "review_hash": canonical_hash(body)}


def review_product_bundle_restore(
    client: "Client", payload: bytes, request: ProductBundleRestoreRequest
) -> dict[str, Any]:
    """Verify all payloads and local mappings without creating destination records."""
    credential = _capture_write_credential(client)
    manifest, arrays = _read_bundle(client, payload)
    del arrays
    _require_conditional_ingestion(client, credential=credential)
    return _review(client, manifest, _checked_request(request), _credential=credential)


def _checked_asserted_binding(
    response: object, data: dict[str, Any], mask: dict[str, Any], tenant_id: UUID
) -> dict[str, Any]:
    try:
        if (
            not isinstance(response, dict)
            or not isinstance(response.get("shape"), list)
            or any(type(size) is not int for size in response["shape"])
        ):
            raise ValueError("Validity binding dimensions must be integers")
        binding = ValidityBinding.model_validate(response)
        if (
            binding.tenant_id != tenant_id
            or str(binding.data_version_id) != data["id"]
            or str(binding.mask_version_id) != mask["id"]
            or binding.data_checksum != data["checksum"]
            or binding.mask_checksum != mask["checksum"]
            or binding.data_dtype != data["dtype"]
            or binding.shape != data["shape"]
            or binding.shape != mask["shape"]
            or binding.origin.kind != "asserted"
            or binding.descriptor_hash
            != _descriptor_hash(
                data,
                mask,
                binding.origin.model_dump(mode="json"),
                str(binding.id),
                binding.created_at,
                binding.created_by,
            )
        ):
            raise ValueError("Validity binding differs from the selected local roles")
    except (ValidationError, ValueError, KeyError) as exc:
        raise TensorusError("Destination returned invalid validity evidence") from exc
    return binding.model_dump(mode="json")


def _find_asserted_binding(
    client: "Client",
    data: dict[str, Any],
    mask: dict[str, Any],
    tenant_id: UUID,
    *,
    credential: str,
) -> dict[str, Any] | None:
    listing = _read_model(
        client,
        f"/v1/versions/{quote(data['id'], safe='')}/validity-bindings",
        ValidityBindingList,
        "Validity binding list",
        MAX_BINDINGS_PER_VERSION * MAX_BUNDLE_VERSION_RESPONSE_BYTES,
        credential=credential,
    )
    if (
        len(listing.items) > MAX_BINDINGS_PER_VERSION
        or len({item.id for item in listing.items}) != len(listing.items)
        or any(
            item.tenant_id != tenant_id or str(item.data_version_id) != data["id"]
            for item in listing.items
        )
    ):
        raise TensorusError("Destination returned inconsistent validity binding membership")
    matches = [
        item
        for item in listing.items
        if str(item.mask_version_id) == mask["id"] and item.origin.kind == "asserted"
    ]
    if len(matches) > 1:
        raise TensorusError("Destination returned ambiguous asserted validity evidence")
    if not matches:
        return None
    binding = _checked_asserted_binding(matches[0].model_dump(mode="json"), data, mask, tenant_id)
    stored = _read_model(
        client,
        f"/v1/validity-bindings/{binding['id']}",
        ValidityBinding,
        "Validity binding",
        MAX_BUNDLE_VERSION_RESPONSE_BYTES,
        credential=credential,
    )
    if canonical_json(stored.model_dump(mode="json")) != canonical_json(binding):
        raise TensorusError("Destination returned contradictory retained validity evidence")
    return binding


def _product_write_request(
    manifest: ProductBundleManifest,
    request: ProductBundleRestoreRequest,
    copied: dict[str, Any],
    bindings: dict[str, dict[str, Any]],
) -> DataProductWrite:
    selections = []
    for member in manifest.source.product.definition.members:
        selection: dict[str, Any] = {
            "role": member.role,
            "version_id": copied[str(member.version.id)]["id"],
            "contract_revision": request.contract_revisions[member.role],
            "contract_hash": member.contract.definition_hash,
        }
        if member.validity is not None:
            binding = bindings[member.role]
            selection["validity"] = {
                "binding_id": binding["id"],
                "descriptor_hash": binding["descriptor_hash"],
                "mask_role": member.validity.mask_role,
            }
        selections.append(selection)
    definition = manifest.source.product.definition
    return DataProductWrite.model_validate(
        {
            "expected_revision": 0,
            "title": definition.title,
            "description": definition.description,
            "members": selections,
            "relationships": (
                [item.model_dump(mode="json") for item in definition.relationships]
                if definition.relationships is not None
                else None
            ),
        }
    )


def _check_product_members(
    product: dict[str, Any],
    manifest: ProductBundleManifest,
    copied: dict[str, Any],
    bindings: dict[str, dict[str, Any]],
) -> None:
    members = manifest.source.product.definition.members
    local_members = {item["role"]: item for item in product["definition"]["members"]}
    if set(local_members) != {member.role for member in members}:
        raise TensorusError("Destination product contradicts the reviewed roles")
    for member in members:
        local = local_members[member.role]
        data = copied[str(member.version.id)]
        expected = {field: data.get(field) for field in ProductVersion.model_fields}
        local_validity = local.get("validity")
        if (
            canonical_json(local["version"]) != canonical_json(expected)
            or (
                member.validity is not None
                and (
                    not isinstance(local_validity, dict)
                    or local_validity.get("binding") != bindings[member.role]
                )
            )
            or (member.validity is None and local_validity is not None)
        ):
            raise TensorusError("Destination product contradicts the copied versions or bindings")


def _retained_product(client: "Client", name: str, *, credential: str) -> dict[str, Any]:
    product = _read_model(
        client,
        f"/v1/data-products/{quote(name, safe='')}",
        DataProductRevision,
        "Data product",
        2 * MAX_PRODUCT_BYTES,
        credential=credential,
    )
    try:
        _validate_source_product(product)
        if product.name != name or str(UUID(product.created_by)) != product.created_by:
            raise ValueError("Product identity is inconsistent")
    except (ValueError, TensorusError) as exc:
        raise TensorusError("Destination returned invalid retained product evidence") from exc
    return product.model_dump(mode="json")


def _retained_import(
    client: "Client",
    import_id: str,
    product: dict[str, Any],
    *,
    credential: str,
) -> dict[str, Any]:
    retained = _read_model(
        client,
        f"/v1/product-release-imports/{quote(import_id, safe='')}",
        ImportedProductRelease,
        "Product release import",
        2 * MAX_PORTABLE_PRODUCT_BYTES,
        credential=credential,
    )
    if (
        str(retained.id) != import_id
        or str(retained.tenant_id) != product["tenant_id"]
        or str(retained.product_id) != product["id"]
        or retained.product_name != product["name"]
        or retained.product_revision != product["revision"]
        or retained.name != retained.request.name
        or retained.product_revision != retained.request.product_revision
        or retained.request.descriptor_hash != product["descriptor_hash"]
        or retained.request_hash != canonical_hash(retained.request.model_dump(mode="json"))
        or retained.source_hash != retained.request.source.content_hash
    ):
        raise TensorusError("Destination source import contradicts the reviewed product")
    try:
        if str(UUID(retained.created_by)) != retained.created_by:
            raise ValueError("Noncanonical creator")
    except ValueError as exc:
        raise TensorusError("Destination source import has an invalid creator") from exc
    return retained.model_dump(mode="json")


def _find_retained_import(
    client: "Client",
    product: dict[str, Any],
    request: ProductReleaseImportRequest,
    *,
    credential: str,
) -> dict[str, Any] | None:
    page = _read_model(
        client,
        f"/v1/data-products/{quote(product['name'], safe='')}/release-imports?limit=200&offset=0",
        dto.Page[ImportedProductReleaseSummary],
        "Product release import list",
        128 * 1024,
        credential=credential,
    )
    if (
        page.limit != 200
        or page.offset != 0
        or page.total != len(page.items)
        or page.total > MAX_PRODUCT_RELEASE_IMPORTS_PER_PRODUCT
        or len({item.id for item in page.items}) != page.total
        or len({item.name for item in page.items}) != page.total
        or any(
            str(item.tenant_id) != product["tenant_id"]
            or str(item.product_id) != product["id"]
            or item.product_name != product["name"]
            for item in page.items
        )
    ):
        raise TensorusError("Destination returned inconsistent product import membership")
    selected = next((item for item in page.items if item.name == request.name), None)
    if selected is None:
        return None
    retained = _retained_import(client, str(selected.id), product, credential=credential)
    if any(
        canonical_json(retained[field]) != canonical_json(value)
        for field, value in selected.model_dump(mode="json").items()
    ):
        raise TensorusError("Destination returned contradictory product import evidence")
    if canonical_json(retained["request"]) != canonical_json(request.model_dump(mode="json")):
        raise Conflict("Destination product import name contains different source evidence")
    return retained


def inspect_product_bundle_restore(
    client: "Client",
    payload: bytes,
    request: ProductBundleRestoreRequest,
    *,
    expected_review_hash: ReleaseHash,
    reviewed: dict[str, Any] | None = None,
    _include_head_preflight: bool = False,
    _include_name_preflight: bool = False,
    _payloads_only: bool = False,
    _credential: str | None = None,
) -> dict[str, Any]:
    """Inspect bounded, independently verified progress without creating local records."""
    credential = _capture_write_credential(client) if _credential is None else _credential
    _require_write_credential(client, credential)
    if _payloads_only and (_include_head_preflight or _include_name_preflight):
        raise TensorusError("Payload-only observation cannot include destination preflight")
    _include_head_preflight = _include_head_preflight or _include_name_preflight
    manifest, arrays = _read_bundle(client, payload)
    del arrays
    request = _checked_request(request)
    _require_conditional_ingestion(client, credential=credential)
    client._require_conditional_npy_historical_receipt(_credential=credential)
    current_review = _review(client, manifest, request, _credential=credential)
    review = _inspection_review(current_review, expected_review_hash, reviewed, "roles")
    identity = _checked_reviewed_identity(client, credential, review)
    members = manifest.source.product.definition.members
    by_role = {member.role: member for member in members}
    sources = {member.version.id: member.version for member in members}
    roles = {item["source_version_id"]: item for item in review["roles"]}
    current_roles = {item["source_version_id"]: item for item in current_review["roles"]}
    copied: dict[str, Any] = {}
    payloads: dict[str, dict[str, Any]] = {}
    version_ids: set[str] = set()
    tensor_ids: set[str] = set()
    for entry in manifest.payloads:
        source_id = str(entry.source_version_id)
        target = request.destinations[entry.source_version_id]
        role = roles[source_id]
        active = role["active_write_contract"]
        condition = NpyWriteCondition(
            dataset_id=role["destination_dataset_id"],
            write_contract_generation=active["generation"],
        )
        contract = WriteContractBinding.model_validate(active) if active["generation"] else None
        current_active = current_roles[source_id]["active_write_contract"]
        current_contract = (
            WriteContractBinding.model_validate(current_active)
            if current_active["generation"]
            else None
        )
        evidence = _inspect_copied_payload(
            client,
            identity,
            sources[entry.source_version_id].model_dump(mode="json"),
            target,
            0,
            condition,
            contract,
            current_contract,
            _product_payload_key(manifest.content_hash, entry.source_version_id, target),
            credential=credential,
            allow_tombstoned=_include_head_preflight,
        )
        payloads[source_id] = evidence
        version = evidence["version"]
        if version is not None:
            if version["id"] in version_ids or version["tensor_id"] in tensor_ids:
                raise TensorusError("Destination receipt identities are not distinct")
            copied[source_id] = version
            version_ids.add(version["id"])
            tensor_ids.add(version["tensor_id"])

    bindings: dict[str, dict[str, Any]] = {}
    binding_progress: dict[str, dict[str, Any]] = {}
    for member in members:
        if member.validity is None:
            continue
        data_id = str(member.version.id)
        mask_id = str(by_role[member.validity.mask_role].version.id)
        if data_id not in copied or mask_id not in copied:
            binding_progress[member.role] = {"status": "not_checked", "binding": None}
            continue
        binding = _find_asserted_binding(
            client, copied[data_id], copied[mask_id], identity.id, credential=credential
        )
        binding_progress[member.role] = {
            "status": "committed" if binding is not None else "not_observed",
            "binding": binding,
        }
        if binding is not None:
            bindings[member.role] = binding

    product_progress: dict[str, Any] = {"status": "not_checked", "record": None}
    import_progress: dict[str, Any] = {"status": "not_checked", "record": None}
    complete = False
    if (
        not _payloads_only
        and len(copied) == len(manifest.payloads)
        and len(bindings) == len(binding_progress)
    ):
        expected = _product_write_request(manifest, request, copied, bindings)
        try:
            product = _retained_product(client, request.product_name, credential=credential)
        except TensorusError as exc:
            if exc.status != 404 or exc.code != "not_found":
                raise
            product = None
        if product is None:
            product_progress["status"] = "not_observed"
        else:
            if (
                product["tenant_id"] != review["tenant_id"]
                or product["revision"] != 1
                or canonical_json(product["request"])
                != canonical_json(expected.model_dump(mode="json"))
            ):
                raise Conflict("Destination product differs from the reviewed reconstruction")
            _check_product_members(product, manifest, copied, bindings)
            product_progress = {"status": "committed", "record": product}
            wanted = ProductReleaseImportRequest(
                name=request.import_name,
                product_revision=product["revision"],
                descriptor_hash=product["descriptor_hash"],
                source=manifest.source,
            )
            retained = _find_retained_import(client, product, wanted, credential=credential)
            import_progress = {
                "status": "committed" if retained is not None else "not_observed",
                "record": retained,
            }
            complete = retained is not None
    if _review(client, manifest, request, _credential=credential) != current_review:
        raise Conflict("Product bundle restore review changed during inspection")
    blocked = any(item["continuation"] == "blocked_policy_changed" for item in payloads.values())
    progress = {
        "format": (
            "tensorus-product-bundle-restore-payload-observation-v1"
            if _payloads_only
            else "tensorus-product-bundle-restore-progress-v3"
            if _include_head_preflight
            else "tensorus-product-bundle-restore-progress-v2"
        ),
        "review_hash": review["review_hash"],
        "current_review_hash": current_review["review_hash"],
        "bundle_hash": manifest.content_hash,
        "source_claims_authenticated": False,
        "payloads": payloads,
        "validity_bindings": binding_progress,
        "product": product_progress,
        "source_import": import_progress,
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
            validate_progress("product", progress, review, payloads_only=True)
        except ValueError as exc:
            raise TensorusError(
                "Product payload observation is inconsistent", code="invalid_response"
            ) from exc
    if not _include_head_preflight:
        _checked_write_identity(client, credential, identity)
        return progress
    heads = inspect_destination_heads(
        client,
        kind="product",
        review=review,
        progress=progress,
        targets=product_head_expectations(manifest, review),
        credential=credential,
        recheck=lambda: _review(client, manifest, request, _credential=credential),
    )
    if not _include_name_preflight:
        return heads
    from tensorus.bundle_restore_names import inspect_destination_names

    return inspect_destination_names(
        client,
        kind="product",
        review=review,
        head=heads,
        credential=credential,
        recheck=lambda: _review(client, manifest, request, _credential=credential),
    )


def restore_product_bundle(
    client: "Client",
    payload: bytes,
    request: ProductBundleRestoreRequest,
    *,
    expected_review_hash: ReleaseHash,
    confirm_unauthenticated_claims: bool,
    confirm_validity_assertions: bool = False,
    reject_observed_name_conflicts: bool = False,
) -> dict[str, Any]:
    """Restore selected local roles and retain source claims without approval or trust."""
    if type(reject_observed_name_conflicts) is not bool:
        raise TensorusError("Name-conflict preflight selection must be a boolean")
    credential = _capture_write_credential(client)
    manifest, arrays = _read_bundle(client, payload)
    del arrays
    request = _checked_request(request)
    _require_conditional_ingestion(client, credential=credential)
    review = _review(client, manifest, request, _credential=credential)
    if review["review_hash"] != expected_review_hash:
        raise Conflict("Product bundle restore review changed; inspect it again")
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
        preflight = inspect_product_bundle_restore(
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
            raise Conflict("Product bundle restore review changed during reconstruction")
        _require_write_credential(client, credential)

    destinations = request.model_dump(mode="json")["destinations"]
    conditions = {
        item["destination_dataset"]: NpyWriteCondition(
            dataset_id=item["destination_dataset_id"],
            write_contract_generation=item["active_write_contract"]["generation"],
        )
        for item in review["roles"]
    }
    contracts = {
        item["destination_dataset"]: (
            WriteContractBinding.model_validate(item["active_write_contract"])
            if item["active_write_contract"]["generation"]
            else None
        )
        for item in review["roles"]
    }
    copied = import_product_bundle(
        client,
        payload,
        destinations,
        _conditions=conditions,
        _expected_contracts=contracts,
        _credential=credential,
        _expected_identity=actor,
    )["destination_versions"]
    recheck()
    members = manifest.source.product.definition.members
    by_role = {member.role: member for member in members}
    reviewed_roles = {item["role"]: item for item in review["roles"]}
    for member in members:
        data = copied[str(member.version.id)]
        if (
            data["tenant_id"] != review["tenant_id"]
            or data["dataset_id"] != reviewed_roles[member.role]["destination_dataset_id"]
        ):
            raise TensorusError("Destination version contradicts the reviewed tenant or dataset")
    bindings: dict[str, dict[str, Any]] = {}
    for member in members:
        data = copied[str(member.version.id)]
        if member.validity is not None:
            mask_role = member.validity.mask_role
            mask = copied[str(by_role[mask_role].version.id)]
            _require_write_credential(client, credential)
            response = client.create_validity_binding(
                ValidityBindingCreate(data_version_id=data["id"], mask_version_id=mask["id"]),
                _credential=credential,
            )
            _require_write_credential(client, credential)
            binding = _checked_asserted_binding(response, data, mask, UUID(review["tenant_id"]))
            stored = _read_model(
                client,
                f"/v1/validity-bindings/{binding['id']}",
                ValidityBinding,
                "Validity binding",
                MAX_BUNDLE_VERSION_RESPONSE_BYTES,
                credential=credential,
            )
            if canonical_json(stored.model_dump(mode="json")) != canonical_json(binding):
                raise TensorusError("Destination returned contradictory retained validity evidence")
            bindings[member.role] = binding
    recheck()
    product_request = _product_write_request(manifest, request, copied, bindings)
    reused_product = False
    try:
        _require_write_credential(client, credential)
        product = client.save_data_product(
            request.product_name, product_request, _credential=credential
        )
        _require_write_credential(client, credential)
    except TensorusError as exc:
        if exc.status != 409 or exc.code != "conflict":
            raise
        product = _retained_product(client, request.product_name, credential=credential)
        if product["revision"] != 1 or canonical_json(product["request"]) != canonical_json(
            product_request.model_dump(mode="json")
        ):
            raise exc
        reused_product = True
    stored_product = _retained_product(client, request.product_name, credential=credential)
    if (
        product["tenant_id"] != review["tenant_id"]
        or (not reused_product and product["created_by"] != review["actor_key_id"])
        or canonical_json(product) != canonical_json(stored_product)
    ):
        raise TensorusError("Destination product contradicts the reviewed tenant, actor or record")
    _check_product_members(product, manifest, copied, bindings)
    recheck()
    import_request = ProductReleaseImportRequest(
        name=request.import_name,
        product_revision=product["revision"],
        descriptor_hash=product["descriptor_hash"],
        source=manifest.source,
    )
    reused_import = False
    try:
        _require_write_credential(client, credential)
        retained = client.import_product_release(
            request.product_name, import_request, _credential=credential
        )
        _require_write_credential(client, credential)
    except TensorusError as exc:
        if exc.status != 409 or exc.code != "conflict":
            raise
        previous = _find_retained_import(client, product, import_request, credential=credential)
        if previous is None:
            raise exc
        retained = previous
        reused_import = True
    if (
        retained["tenant_id"] != review["tenant_id"]
        or retained["product_id"] != product["id"]
        or canonical_json(retained["request"])
        != canonical_json(import_request.model_dump(mode="json"))
        or (not reused_import and retained["created_by"] != review["actor_key_id"])
    ):
        raise TensorusError("Destination source import contradicts the reviewed product or actor")
    stored_import = _retained_import(client, retained["id"], product, credential=credential)
    if canonical_json(retained) != canonical_json(stored_import):
        raise TensorusError("Destination returned contradictory retained source import evidence")
    recheck()
    return {
        "format": "tensorus-product-bundle-restore-v2",
        "write_condition_format": "tensorus-conditional-npy-ingest-v1",
        "review_hash": review["review_hash"],
        "bundle_hash": manifest.content_hash,
        "source_claims_authenticated": False,
        "destination_versions": copied,
        "validity_bindings": bindings,
        "product": product,
        "source_import": retained,
        "source_quality_attached_locally": False,
        "source_lineage_recreated": False,
        "approval_transferred": False,
        "local_release_created": False,
        "whole_restore_atomic": False,
    }
