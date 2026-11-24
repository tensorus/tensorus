"""Transfer release evidence without transferring source authority."""
from dataclasses import dataclass
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import JsonValue
from pydantic import StrictBool
from pydantic import StrictInt
from pydantic import ValidationError
from pydantic import model_validator
from tensorus import responses as dto
from tensorus.arrays import canonical_json
from tensorus.arrays import descriptor_nbytes
from tensorus.contracts import MAX_RELEASE_CONTRACT_OVERRIDES
from tensorus.contracts import release_binding_selections_match
from tensorus.contracts import release_contract_overrides_match
from tensorus.contracts import release_quality_overrides_match
from tensorus.errors import StorageError
from tensorus.errors import TensorusError
from tensorus.history_summary_policy import MAX_SELECTED_HISTORY_BYTES
from tensorus.models import DType
from tensorus.models import Model
from tensorus.models import Name
from tensorus.models import Operation
from tensorus.models import PlanDefinition
from tensorus.models import Step
from tensorus.models import TensorSpec
from tensorus.models import WriteContractBinding
from tensorus.operations import TensorShape
from tensorus.operations import parameters
from tensorus.operations import reindex_fill
from tensorus.receipt import PlanReceipt
from tensorus.receipt import _parameters_match
from tensorus.receipt import _receipt_parameters
from tensorus.releases import _hash
from tensorus.semantics import derive_semantics
from tensorus.semantics import read_semantics
from tensorus.semantics import require_matching_axes
from tensorus.validity import BinaryTransformedValidityOrigin
from tensorus.validity import TransformedValidityOrigin
from tensorus.validity import ValidityBinding
from tensorus.validity import _descriptor_hash
from tensorus.validity import _has_selected_preprocessing
from tensorus.validity import _paired_binary_definition
from tensorus.validity import _paired_definition
from tensorus.validity import _pinned_v2_cast
from tensorus.validity import _pinned_v2_conversion
from tensorus.validity import _pinned_v2_positional
from tensorus.validity import _pinned_v2_reduction
from tensorus.validity import _pinned_v2_scalar
from tensorus.validity import _pinned_v2_scaling
from tensorus.validity import _selected_preprocessing
from tensorus.validity import _v2_alignment_plan_matches
from tensorus.validity import _v2_cast_definition
from tensorus.validity import _v2_cast_step_matches
from tensorus.validity import _v2_conversion_definition
from tensorus.validity import _v2_conversion_step_matches
from tensorus.validity import _v2_positional_definition
from tensorus.validity import _v2_positional_step_matches
from tensorus.validity import _v2_reduction_step_matches
from tensorus.validity import _v2_reindex_step_matches
from tensorus.validity import _v2_scalar_definition
from tensorus.validity import _v2_scalar_step_matches
from tensorus.validity import _v2_scaling_definition
from tensorus.validity import _v2_scaling_step_matches
from tensorus.validity import _v2_window_plan_matches
MAX_PORTABLE_BYTES = 2 * 1024**2
MAX_DESCRIPTOR_BYTES = 8 * 1024**2
MAX_PORTABLE_LINEAGE_VERSIONS = 256
MAX_PORTABLE_LINEAGE_EDGES = 1024
MAX_PORTABLE_LINEAGE_DEPTH = 24
MAX_PORTABLE_VALIDITY_SELECTIONS = 32
MAX_PORTABLE_VALIDITY_BINDINGS = 64
MAX_PORTABLE_PLAN_SELECTIONS = 16
MAX_PORTABLE_HISTORY_RELEASES = 8
MAX_PORTABLE_HISTORY_BYTES = 4 * 1024**2
MAX_PORTABLE_HISTORY_IMPORT_BYTES = MAX_SELECTED_HISTORY_BYTES
MAX_HISTORY_INDEX_PAGE = 8
MAX_HISTORY_INDEX_DIAGNOSTIC_OFFSET = 10_000
Hash = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
class PortableLineageVersion(Model):
    id: UUID
    tenant_id: UUID
    tensor_id: UUID
    version: Annotated[StrictInt, Field(ge=1)]
    dataset: Name
    dataset_id: UUID
    name: Name
    dtype: DType
    shape: list[Annotated[StrictInt, Field(ge=0, le=2**63 - 1)]] = Field(max_length=16)
    nbytes: Annotated[StrictInt, Field(ge=0)]
    checksum: Hash
    metadata: dict[str, JsonValue]
    statistics: dict[str, JsonValue]
    operation: dict[str, JsonValue] | None
    created_at: str
    created_by: str
    parents: list[UUID] = Field(max_length=MAX_PORTABLE_LINEAGE_EDGES)
    write_contract: WriteContractBinding | None
    access: dto.VersionAccess | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_descriptor(self) -> Self:
        try:
            descriptor_nbytes(self.model_dump(mode="json"))
        except StorageError as exc:
            raise ValueError("Portable lineage tensor descriptor is invalid") from exc
        if len(set(self.parents)) != len(self.parents):
            raise ValueError("Portable lineage contains duplicate parent edges")
        return self
class PortableLineage(Model):
    versions: list[PortableLineageVersion] = Field(
        min_length=1, max_length=MAX_PORTABLE_LINEAGE_VERSIONS
    )
class PortableValidity(Model):
    selected_binding_ids: list[UUID] = Field(
        min_length=1, max_length=MAX_PORTABLE_VALIDITY_SELECTIONS
    )
    bindings: list[ValidityBinding] = Field(min_length=1, max_length=MAX_PORTABLE_VALIDITY_BINDINGS)
class PortablePlanExecution(Model):
    plan: dict[str, JsonValue]
    receipt: PlanReceipt
class PortablePlans(Model):
    selected_plan_ids: list[UUID] = Field(min_length=1, max_length=MAX_PORTABLE_PLAN_SELECTIONS)
    executions: list[PortablePlanExecution] = Field(
        min_length=1, max_length=MAX_PORTABLE_PLAN_SELECTIONS
    )
class _PlanStep:
    definition: Step
    sources: tuple[UUID, ...]
    output: UUID
    checkpoint: dto.ProfileResult | dto.TransformResult
class _PlanEvidence:
    plan: dto.Plan
    steps: list[_PlanStep]
def _plan_evidence(raw: dict[str, JsonValue]) -> _PlanEvidence:
    try:
        plan = dto.Plan.model_validate(raw)
    except ValidationError as exc:
        raise ValueError("Portable plan record is invalid") from exc
    if (
        _hash(raw["definition"]) != plan.plan_hash
        or plan.status != "succeeded"
        or plan.writes != plan.definition.writes
        or set(plan.results) != {step.id for step in plan.definition.steps}
    ):
        raise ValueError("Portable plan definition, status or checkpoints do not match")
    steps = []
    for step in plan.definition.steps:
        checkpoint = plan.results[step.id]
        if (step.operation == Operation.profile) != isinstance(checkpoint, dto.ProfileResult):
            raise ValueError("Portable plan checkpoint has the wrong type")
        sources = tuple(
            plan.results[ref[1:]].version_id if ref.startswith("@") else UUID(ref)
            for ref in step.inputs
        )
        steps.append(_PlanStep(step, sources, checkpoint.version_id, checkpoint))
    return _PlanEvidence(plan, steps)
def _lineage_depths(
    nodes: dict[UUID, PortableLineageVersion], roots: list[UUID]
) -> dict[UUID, int]:
    visiting: set[UUID] = set()
    depths: dict[UUID, int] = {}

    def depth(version_id: UUID) -> int:
        if version_id in visiting:
            raise ValueError("Portable lineage contains a cycle")
        if version_id not in depths:
            visiting.add(version_id)
            parents = nodes[version_id].parents
            value = 1 + max((depth(parent) for parent in parents), default=-1)
            visiting.remove(version_id)
            depths[version_id] = value
        return depths[version_id]

    for root in roots:
        depth(root)
    return depths
def _direct_validity_links(
    nodes: dict[UUID, PortableLineageVersion],
    binding: ValidityBinding,
    parent: ValidityBinding,
) -> bool:
    return (
        parent.data_version_id in nodes[binding.data_version_id].parents
        and parent.mask_version_id in nodes[binding.mask_version_id].parents
    )
def _has_aligned_validity_intermediate(
    nodes: dict[UUID, PortableLineageVersion], binding: ValidityBinding
) -> bool:
    if not isinstance(binding.origin, BinaryTransformedValidityOrigin):
        return False
    for version_id in (binding.data_version_id, binding.mask_version_id):
        for parent_id in nodes[version_id].parents:
            operation = nodes[parent_id].operation
            if (
                operation is not None
                and operation.get("plan_id") == str(binding.origin.plan_id)
                and operation.get("step_id")
                in (
                    "aligned_data",
                    "aligned_mask",
                    "select_left_data",
                    "select_left_mask",
                    "select_right_data",
                    "select_right_mask",
                )
            ):
                return True
    return False
def _portable_binary_steps_match(
    nodes: dict[UUID, PortableLineageVersion],
    binding: ValidityBinding,
    parents: list[ValidityBinding],
    outputs: list[PortableLineageVersion],
    ids: list[str],
    inputs: list[list[str]],
    resolved: list[list[UUID]],
    policy: Literal["v1", "v2"],
) -> bool:
    origin = binding.origin
    if not isinstance(origin, BinaryTransformedValidityOrigin):
        return False
    data = nodes[binding.data_version_id]
    operations = [node.operation for node in outputs]
    if any(
        node.dataset_id != data.dataset_id
        or node.dataset != data.dataset
        or operation is None
        or operation.get("plan_id") != str(origin.plan_id)
        or operation.get("step_id") != step_id
        or set(node.parents) != set(refs)
        or operation.get("source_checksums") != [nodes[ref].checksum for ref in refs]
        for node, operation, step_id, refs in zip(outputs, operations, ids, resolved, strict=True)
    ):
        return False
    try:
        definition = PlanDefinition.model_validate(
            {
                "dataset": data.dataset,
                "title": "Portable aligned validity evidence",
                "semantic_policy": policy,
                "access_policy": "source_datasets_v1",
                "validity_sources": [item.model_dump(mode="json") for item in origin.sources],
                "steps": [
                    {
                        "id": step_id,
                        "operation": operation["name"],
                        "inputs": refs,
                        "output": node.name,
                        "parameters": operation["parameters"],
                    }
                    for node, operation, step_id, refs in zip(
                        outputs, operations, ids, inputs, strict=True
                    )
                    if operation is not None
                ],
            }
        )
        if not _paired_binary_definition(
            definition, [parent.model_dump(mode="json") for parent in parents]
        ):
            return False
        for step, output in zip(definition.steps, outputs, strict=True):
            if step.operation == Operation.reindex_coordinates:
                reindex_fill(output.dtype, parameters(step)["fill_value"])
        if policy == "v2" and not _v2_alignment_plan_matches(
            definition,
            [nodes[parent.data_version_id].model_dump(mode="json") for parent in parents],
        ):
            return False
        selected_preprocessing = _selected_preprocessing(
            definition,
            [parent.model_dump(mode="json") for parent in parents],
            [nodes[parent.data_version_id].model_dump(mode="json") for parent in parents],
        )
        if any(
            operation is None
            or not _parameters_match(
                step.operation, _receipt_parameters(step)[0], operation.get("parameters")
            )
            for step, operation in zip(definition.steps, operations, strict=True)
        ):
            return False
        for step, output, refs in zip(definition.steps, outputs, resolved, strict=True):
            input_nodes = [nodes[ref] for ref in refs]
            shapes = [TensorShape(tuple(node.shape), node.dtype) for node in input_nodes]
            semantics = [
                read_semantics(node.metadata, shape.shape, node.operation)
                for node, shape in zip(input_nodes, shapes, strict=True)
            ]
            derived = derive_semantics(
                step,
                shapes,
                semantics,
                TensorShape(tuple(output.shape), output.dtype),
                policy,
                selected_preprocessing=(selected_preprocessing if step.id == "values" else None),
            )
            actual = read_semantics(output.metadata, tuple(output.shape), output.operation)
            if policy == "v2":
                if (
                    derived is None
                    or actual is None
                    or canonical_json(derived.model_dump(mode="json"))
                    != canonical_json(actual.model_dump(mode="json"))
                ):
                    return False
            elif derived != actual:
                return False
        return True
    except (KeyError, ValueError, TensorusError):
        return False
def _aligned_binary_lineage_matches(
    nodes: dict[UUID, PortableLineageVersion],
    binding: ValidityBinding,
    parents: list[ValidityBinding],
) -> bool:
    origin = binding.origin
    if not isinstance(origin, BinaryTransformedValidityOrigin) or len(parents) != 2:
        return False
    data, mask = nodes[binding.data_version_id], nodes[binding.mask_version_id]

    def intermediate(child: PortableLineageVersion, step_id: str) -> PortableLineageVersion | None:
        matches = []
        for parent_id in child.parents:
            node = nodes[parent_id]
            operation = node.operation
            if (
                operation is not None
                and operation.get("plan_id") == str(origin.plan_id)
                and operation.get("step_id") == step_id
            ):
                matches.append(node)
        return matches[0] if len(matches) == 1 else None

    aligned_data = intermediate(data, "aligned_data")
    aligned_mask = intermediate(mask, "aligned_mask")
    if aligned_data is None or aligned_mask is None:
        return False
    try:
        parent_data = nodes[parents[0].data_version_id]
        parent_semantics = read_semantics(
            parent_data.metadata, tuple(parent_data.shape), parent_data.operation
        )
    except TensorusError:
        return False
    if parent_semantics is None:
        return False
    policy: Literal["v1", "v2"] = "v2" if parent_semantics.version == 2 else "v1"
    for source in (0, 1):
        anchor = 1 - source
        side = "left" if source == 0 else "right"
        transposed_data = intermediate(aligned_data, f"transpose_{side}_data")
        transposed_mask = intermediate(aligned_mask, f"transpose_{side}_mask")
        if (transposed_data is None) != (transposed_mask is None):
            continue
        ids: list[str] = []
        inputs: list[list[str]] = []
        resolved: list[list[UUID]] = []
        outputs: list[PortableLineageVersion] = []
        data_refs = [str(parent.data_version_id) for parent in parents]
        mask_refs = [str(parent.mask_version_id) for parent in parents]
        source_data_id = parents[source].data_version_id
        source_mask_id = parents[source].mask_version_id
        if transposed_data is not None and transposed_mask is not None:
            operation = transposed_data.operation or {}
            params = operation.get("parameters")
            axes = params.get("axes") if isinstance(params, dict) else None
            source_data = nodes[source_data_id]
            if policy != "v2" or not isinstance(axes, list):
                continue
            permutation = [
                axis for axis in axes if isinstance(axis, int) and not isinstance(axis, bool)
            ]
            if (
                len(permutation) != len(axes)
                or sorted(permutation) != list(range(len(source_data.shape)))
                or transposed_data.shape != [source_data.shape[axis] for axis in permutation]
                or transposed_mask.shape != transposed_data.shape
                or transposed_data.dtype != source_data.dtype
                or transposed_mask.dtype != DType.bool
            ):
                continue
            ids.extend((f"transpose_{side}_data", f"transpose_{side}_mask"))
            inputs.extend(([data_refs[source]], [mask_refs[source]]))
            resolved.extend(([source_data_id], [source_mask_id]))
            outputs.extend((transposed_data, transposed_mask))
            data_refs[source], mask_refs[source] = f"@{ids[0]}", f"@{ids[1]}"
            source_data_id, source_mask_id = transposed_data.id, transposed_mask.id
        ids.extend(("aligned_data", "aligned_mask", "validity", "values"))
        inputs.extend(
            (
                [data_refs[source], data_refs[anchor]],
                [mask_refs[source], data_refs[anchor]],
            )
        )
        resolved.extend(
            (
                [source_data_id, parents[anchor].data_version_id],
                [source_mask_id, parents[anchor].data_version_id],
            )
        )
        data_refs[source], mask_refs[source] = "@aligned_data", "@aligned_mask"
        inputs.extend((mask_refs, [*data_refs, "@validity"]))
        resolved.extend(
            (
                [
                    aligned_mask.id if index == source else parent.mask_version_id
                    for index, parent in enumerate(parents)
                ],
                [
                    *[
                        aligned_data.id if index == source else parent.data_version_id
                        for index, parent in enumerate(parents)
                    ],
                    mask.id,
                ],
            )
        )
        outputs.extend((aligned_data, aligned_mask, mask, data))
        if (
            aligned_data.shape != data.shape
            or aligned_mask.shape != data.shape
            or nodes[parents[anchor].data_version_id].shape != data.shape
            or aligned_data.dtype != nodes[parents[source].data_version_id].dtype
            or aligned_mask.dtype != DType.bool
            or data.dtype != DType.float64
        ):
            continue
        if _portable_binary_steps_match(
            nodes, binding, parents, outputs, ids, inputs, resolved, policy
        ):
            return True
    return False
def _indexed_binary_lineage_matches(
    nodes: dict[UUID, PortableLineageVersion],
    binding: ValidityBinding,
    parents: list[ValidityBinding],
) -> bool:
    origin = binding.origin
    if not isinstance(origin, BinaryTransformedValidityOrigin) or len(parents) != 2:
        return False
    data, mask = nodes[binding.data_version_id], nodes[binding.mask_version_id]

    def intermediate(child: PortableLineageVersion, step_id: str) -> PortableLineageVersion | None:
        matches = []
        for parent_id in child.parents:
            node = nodes[parent_id]
            operation = node.operation
            if (
                operation is not None
                and operation.get("plan_id") == str(origin.plan_id)
                and operation.get("step_id") == step_id
            ):
                matches.append(node)
        return matches[0] if len(matches) == 1 else None

    left_data = intermediate(data, "select_left_data")
    left_mask = intermediate(mask, "select_left_mask")
    right_data = intermediate(data, "select_right_data")
    right_mask = intermediate(mask, "select_right_mask")
    if left_data is None or left_mask is None or right_data is None or right_mask is None:
        return False
    selected = [left_data, left_mask, right_data, right_mask]
    if (
        any(node.shape != data.shape for node in selected)
        or left_data.dtype != nodes[parents[0].data_version_id].dtype
        or right_data.dtype != nodes[parents[1].data_version_id].dtype
        or left_mask.dtype != DType.bool
        or right_mask.dtype != DType.bool
        or data.dtype != DType.float64
    ):
        return False
    transposed_data = intermediate(right_data, "transpose_right_data")
    transposed_mask = intermediate(right_mask, "transpose_right_mask")
    if (transposed_data is None) != (transposed_mask is None):
        return False
    ids: list[str] = []
    inputs: list[list[str]] = []
    resolved: list[list[UUID]] = []
    outputs: list[PortableLineageVersion] = []
    right_data_ref, right_mask_ref = (
        str(parents[1].data_version_id),
        str(parents[1].mask_version_id),
    )
    right_data_id, right_mask_id = parents[1].data_version_id, parents[1].mask_version_id
    if transposed_data is not None and transposed_mask is not None:
        operation = transposed_data.operation or {}
        params = operation.get("parameters")
        axes = params.get("axes") if isinstance(params, dict) else None
        right_source = nodes[parents[1].data_version_id]
        if not isinstance(axes, list):
            return False
        permutation = [
            axis for axis in axes if isinstance(axis, int) and not isinstance(axis, bool)
        ]
        if (
            len(permutation) != len(axes)
            or sorted(permutation) != list(range(len(right_source.shape)))
            or transposed_data.shape != [right_source.shape[axis] for axis in permutation]
            or transposed_mask.shape != transposed_data.shape
            or transposed_data.dtype != right_source.dtype
            or transposed_mask.dtype != DType.bool
        ):
            return False
        ids.extend(("transpose_right_data", "transpose_right_mask"))
        inputs.extend(([right_data_ref], [right_mask_ref]))
        resolved.extend(([right_data_id], [right_mask_id]))
        outputs.extend((transposed_data, transposed_mask))
        right_data_ref, right_mask_ref = "@transpose_right_data", "@transpose_right_mask"
        right_data_id, right_mask_id = transposed_data.id, transposed_mask.id
    ids.extend(
        (
            "select_left_data",
            "select_left_mask",
            "select_right_data",
            "select_right_mask",
            "validity",
            "values",
        )
    )
    inputs.extend(
        (
            [str(parents[0].data_version_id)],
            [str(parents[0].mask_version_id)],
            [right_data_ref],
            [right_mask_ref],
            ["@select_left_mask", "@select_right_mask"],
            ["@select_left_data", "@select_right_data", "@validity"],
        )
    )
    resolved.extend(
        (
            [parents[0].data_version_id],
            [parents[0].mask_version_id],
            [right_data_id],
            [right_mask_id],
            [left_mask.id, right_mask.id],
            [left_data.id, right_data.id, mask.id],
        )
    )
    outputs.extend((*selected, mask, data))
    return _portable_binary_steps_match(
        nodes, binding, parents, outputs, ids, inputs, resolved, "v2"
    )
def _requires_positional_binding(definition: PlanDefinition) -> bool:
    return (
        _pinned_v2_positional(definition)
        or definition.coordinate_window is not None
        or (definition.validity_sources is not None and _v2_positional_definition(definition))
    )
class PortableRelease(Model):
    format: Literal[
        "tensorus-portable-release-v1",
        "tensorus-portable-release-v2",
        "tensorus-portable-release-v3",
        "tensorus-portable-release-v4",
        "tensorus-portable-release-v5",
        "tensorus-portable-release-v6",
        "tensorus-portable-release-v7",
        "tensorus-portable-release-v8",
        "tensorus-portable-release-v9",
    ] = "tensorus-portable-release-v1"
    feature_format: (
        Literal[
            "tensorus-portable-release-v1",
            "tensorus-portable-release-v2",
            "tensorus-portable-release-v3",
            "tensorus-portable-release-v4",
            "tensorus-portable-release-v5",
            "tensorus-portable-release-v6",
        ]
        | None
    ) = Field(default=None, exclude_if=lambda value: value is None)
    release: dto.Release
    snapshot: dto.Snapshot
    contract: dto.ContractRevision
    member_contracts: list[dto.ContractRevision] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_RELEASE_CONTRACT_OVERRIDES,
        exclude_if=lambda value: value is None,
    )
    write_spec: TensorSpec
    lineage: PortableLineage | None = Field(default=None, exclude_if=lambda value: value is None)
    validity: PortableValidity | None = Field(default=None, exclude_if=lambda value: value is None)
    plans: PortablePlans | None = Field(default=None, exclude_if=lambda value: value is None)
    content_hash: Hash

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        release, report, request = self.release, self.release.report, self.release.request
        if _hash(self.model_dump(mode="json", exclude={"content_hash"})) != self.content_hash:
            raise ValueError("Portable release content hash does not match")
        if (
            _hash(report.model_dump(mode="json")) != release.report_hash
            or _hash(self.contract.definition.model_dump(mode="json"))
            != self.contract.definition_hash
            or _hash(self.write_spec.model_dump(mode="json")) != report.write_spec_hash
        ):
            raise ValueError("Portable release report or contract hash does not match")
        if (
            report.tenant_id != release.tenant_id
            or report.dataset_id != release.dataset_id
            or report.dataset != release.dataset
            or report.release_name != release.name
            or report.proposed_by != release.created_by
            or report.snapshot_id != release.snapshot_id
            or report.contract_revision != release.contract_revision
            or request.snapshot_id != release.snapshot_id
            or request.contract_revision != release.contract_revision
            or request.name != release.name
            or request.quality != report.quality
            or not release_binding_selections_match(request, report)
            or not release_quality_overrides_match(request, report)
            or not release_contract_overrides_match(request, report)
            or self.snapshot.id != release.snapshot_id
            or self.snapshot.tenant_id != release.tenant_id
            or self.snapshot.dataset_id != release.dataset_id
            or self.contract.dataset_id != release.dataset_id
            or self.contract.revision != release.contract_revision
            or self.contract.definition_hash != report.contract_hash
        ):
            raise ValueError("Portable release references do not match")
        selected_contracts = {
            item.revision: item.definition_hash for item in request.contract_overrides or []
        }
        if self.format == "tensorus-portable-release-v9":
            contracts = self.member_contracts or []
            if (
                not selected_contracts
                or len(contracts) != len(selected_contracts)
                or len({item.revision for item in contracts}) != len(contracts)
                or {item.revision: item.definition_hash for item in contracts} != selected_contracts
                or any(
                    selected_contracts[item.revision] != item.definition_hash
                    for item in request.contract_overrides or []
                )
                or any(
                    item.dataset_id != release.dataset_id
                    or _hash(item.definition.model_dump(mode="json")) != item.definition_hash
                    or (item.revision == self.contract.revision and item != self.contract)
                    for item in contracts
                )
            ):
                raise ValueError("Portable member contract revisions do not match the selections")
        elif self.member_contracts is not None or selected_contracts:
            raise ValueError("Portable member contracts require v9")
        if (
            (release.status in ("proposed", "approved")) != report.passed
            or (
                release.status == "approved"
                and (
                    release.approved_at is None
                    or release.approved_by is None
                    or release.approved_by == release.created_by
                )
            )
            or (
                release.status != "approved"
                and (release.approved_at is not None or release.approved_by is not None)
            )
        ):
            raise ValueError("Portable release status does not match its evidence")
        if (
            not report.members
            or len(report.members) > 10_000
            or len({member.name for member in report.members}) != len(report.members)
            or len({member.version_id for member in report.members}) != len(report.members)
            or report.passed != all(member.passed for member in report.members)
        ):
            raise ValueError("Portable release membership or outcome is inconsistent")
        feature_format: str
        if self.format in (
            "tensorus-portable-release-v7",
            "tensorus-portable-release-v8",
            "tensorus-portable-release-v9",
        ):
            expected = (
                "tensorus-release-report-v2"
                if self.format == "tensorus-portable-release-v7"
                else "tensorus-release-report-v3"
                if self.format == "tensorus-portable-release-v8"
                else "tensorus-release-report-v4"
            )
            if self.feature_format is None or report.format != expected:
                version = self.format.rsplit("-", 1)[-1]
                report_version = expected.rsplit("-", 1)[-1]
                raise ValueError(
                    f"Portable {version} requires a {report_version} report "
                    "and an explicit feature format"
                )
            feature_format = self.feature_format
        else:
            if self.feature_format is not None or report.format != "tensorus-release-report-v1":
                raise ValueError("Legacy portable formats require a v1 release report")
            feature_format = self.format
        if feature_format == "tensorus-portable-release-v1":
            if self.lineage is not None or self.validity is not None:
                raise ValueError("Portable v1 release cannot contain lineage or validity")
            if self.plans is not None:
                raise ValueError("Portable v1 release cannot contain plans")
        elif feature_format == "tensorus-portable-release-v2":
            if self.lineage is None:
                raise ValueError("Portable v2 release requires complete lineage")
            if self.validity is not None:
                raise ValueError("Portable v2 release cannot contain validity")
            if self.plans is not None:
                raise ValueError("Portable v2 release cannot contain plans")
            self._validate_lineage()
        elif feature_format == "tensorus-portable-release-v3":
            if self.lineage is None or self.validity is None:
                raise ValueError("Portable v3 release requires complete lineage and validity")
            if self.plans is not None:
                raise ValueError("Portable v3 release cannot contain plans")
            self._validate_lineage()
            self._validate_validity(feature_format)
        elif feature_format == "tensorus-portable-release-v4":
            if self.lineage is None or self.plans is None:
                raise ValueError("Portable v4 release requires complete lineage and selected plans")
            self._validate_lineage()
            if self.validity is not None:
                self._validate_validity(feature_format)
            self._validate_plans()
        else:
            version = feature_format.rsplit("-", 1)[-1]
            if self.lineage is None or self.validity is None:
                raise ValueError(
                    f"Portable {version} release requires lineage and multi-source validity"
                )
            if not any(
                isinstance(binding.origin, BinaryTransformedValidityOrigin)
                for binding in self.validity.bindings
            ):
                raise ValueError(f"Portable {version} release requires a binary validity origin")
            self._validate_lineage()
            self._validate_validity(feature_format)
            if self.plans is not None:
                self._validate_plans()
        if (
            feature_format not in ("tensorus-portable-release-v5", "tensorus-portable-release-v6")
            and self.validity is not None
            and any(
                isinstance(binding.origin, BinaryTransformedValidityOrigin)
                for binding in self.validity.bindings
            )
        ):
            raise ValueError("Binary validity origins require portable release v5 or v6")
        if self.validity is not None:
            masks = {
                member.quality_mask.binding_id: (member, member.quality_mask)
                for member in report.members
                if member.quality_mask is not None
            }
            for binding in self.validity.bindings:
                claim = masks.get(binding.id)
                if claim is not None:
                    member, mask = claim
                    if (
                        binding.descriptor_hash != mask.descriptor_hash
                        or binding.data_version_id != member.version_id
                        or binding.data_checksum != member.checksum
                        or binding.mask_version_id != mask.version_id
                        or binding.mask_checksum != mask.checksum
                        or binding.origin.kind != mask.origin_kind
                    ):
                        raise ValueError("Portable validity does not match release mask evidence")
        return self

    def _plan_entries(self) -> list[tuple[PortablePlanExecution, _PlanEvidence]]:
        assert self.plans is not None
        entries = [(entry, _plan_evidence(entry.plan)) for entry in self.plans.executions]
        ids = [evidence.plan.id for _, evidence in entries]
        selected = self.plans.selected_plan_ids
        if (
            len(set(selected)) != len(selected)
            or len(set(ids)) != len(ids)
            or set(ids) != set(selected)
        ):
            raise ValueError("Portable plan selections and executions must be unique and match")
        return entries

    def _validate_lineage(self) -> None:
        assert self.lineage is not None
        nodes = {node.id: node for node in self.lineage.versions}
        if len(nodes) != len(self.lineage.versions):
            raise ValueError("Portable lineage contains duplicate versions")
        if sum(len(node.parents) for node in nodes.values()) > MAX_PORTABLE_LINEAGE_EDGES:
            raise ValueError("Portable lineage exceeds the edge limit")
        if any(node.tenant_id != self.release.tenant_id for node in nodes.values()):
            raise ValueError("Portable lineage versions must belong to the source tenant")
        if any(parent not in nodes for node in nodes.values() for parent in node.parents):
            raise ValueError("Portable lineage is not closed over parent versions")
        dataset_names: dict[UUID, str] = {}
        dataset_ids: dict[str, UUID] = {}
        tensor_names: dict[UUID, tuple[UUID, str]] = {}
        tensor_ids: dict[tuple[UUID, str], UUID] = {}
        version_keys: set[tuple[UUID, int]] = set()
        for node in nodes.values():
            tensor_name = (node.dataset_id, node.name)
            version_key = (node.tensor_id, node.version)
            if (
                dataset_names.setdefault(node.dataset_id, node.dataset) != node.dataset
                or dataset_ids.setdefault(node.dataset, node.dataset_id) != node.dataset_id
                or tensor_names.setdefault(node.tensor_id, tensor_name) != tensor_name
                or tensor_ids.setdefault(tensor_name, node.tensor_id) != node.tensor_id
                or version_key in version_keys
            ):
                raise ValueError("Portable lineage contains inconsistent source identities")
            version_keys.add(version_key)

        roots = {member.version_id: member for member in self.release.report.members}
        for version_id, member in roots.items():
            root = nodes.get(version_id)
            if (
                root is None
                or root.dataset_id != self.release.dataset_id
                or root.dataset != self.release.dataset
                or root.name != member.name
                or root.dtype != member.dtype
                or root.shape != member.shape
                or root.nbytes != member.nbytes
                or root.checksum != member.checksum
                or _hash(root.metadata) != member.metadata_hash
                or root.operation != member.operation
                or [str(parent) for parent in root.parents] != member.parents
            ):
                raise ValueError("Portable lineage does not match release membership")

        release_depths = _lineage_depths(nodes, list(roots))
        extra_roots: list[UUID] = []
        if self.validity is not None:
            bindings = {binding.id: binding for binding in self.validity.bindings}
            selected = self.validity.selected_binding_ids
            if len(bindings) != len(self.validity.bindings) or len(set(selected)) != len(selected):
                raise ValueError("Portable validity contains duplicate bindings or selections")
            if any(binding_id not in bindings for binding_id in selected):
                raise ValueError("Portable validity selection references a missing binding")
            if any(
                bindings[binding_id].data_version_id not in release_depths
                for binding_id in selected
            ):
                raise ValueError("Selected validity data must be in release lineage")
            extra_roots = [
                version_id
                for binding in bindings.values()
                for version_id in (binding.data_version_id, binding.mask_version_id)
            ]
            if any(version_id not in nodes for version_id in extra_roots):
                raise ValueError("Portable validity references a missing lineage version")
        base_depths = _lineage_depths(nodes, [*roots, *extra_roots])
        if self.plans is not None:
            for _, evidence in self._plan_entries():
                if evidence.plan.tenant_id != self.release.tenant_id:
                    raise ValueError("Portable plans must belong to the source tenant")
                if not any(
                    step.definition.operation != Operation.profile and step.output in base_depths
                    for step in evidence.steps
                ):
                    raise ValueError(
                        "Selected plan must produce a release or validity graph version"
                    )
                references = [
                    version_id
                    for step in evidence.steps
                    for version_id in (*step.sources, step.output)
                ]
                if any(version_id not in nodes for version_id in references):
                    raise ValueError("Portable plan references a missing lineage version")
                extra_roots.extend(references)
        depths = _lineage_depths(nodes, [*roots, *extra_roots])
        if any(depth > MAX_PORTABLE_LINEAGE_DEPTH for depth in depths.values()):
            raise ValueError("Portable lineage exceeds the depth limit")
        if set(depths) != set(nodes):
            raise ValueError("Portable lineage contains unreachable versions")

    def _validate_validity(self, feature_format: str) -> None:
        assert self.lineage is not None and self.validity is not None
        nodes = {node.id: node for node in self.lineage.versions}
        bindings = {binding.id: binding for binding in self.validity.bindings}
        for binding in bindings.values():
            data, mask = nodes[binding.data_version_id], nodes[binding.mask_version_id]
            if (
                binding.tenant_id != self.release.tenant_id
                or binding.data_checksum != data.checksum
                or binding.data_dtype != data.dtype
                or binding.mask_checksum != mask.checksum
                or binding.shape != data.shape
                or mask.shape != data.shape
                or mask.dtype != DType.bool
            ):
                raise ValueError("Portable validity does not match its version descriptors")
            try:
                data_semantics = read_semantics(data.metadata, tuple(data.shape), data.operation)
                mask_semantics = read_semantics(mask.metadata, tuple(mask.shape), mask.operation)
                if data_semantics is None or mask_semantics is None:
                    raise ValueError("Portable validity requires declared axes")
                require_matching_axes(data_semantics, mask_semantics, tuple(data.shape))
            except TensorusError as exc:
                raise ValueError("Portable validity axes are incompatible") from exc
            if (
                _descriptor_hash(
                    data.model_dump(mode="json"),
                    mask.model_dump(mode="json"),
                    binding.origin.model_dump(mode="json"),
                    str(binding.id),
                    binding.created_at,
                    binding.created_by,
                )
                != binding.descriptor_hash
            ):
                raise ValueError("Portable validity descriptor hash does not match")

        reached: set[UUID] = set()
        visiting: set[UUID] = set()
        aligned_found = False

        def visit(binding_id: UUID) -> None:
            nonlocal aligned_found
            if binding_id in visiting:
                raise ValueError("Portable validity contains a binding cycle")
            if binding_id in reached:
                return
            visiting.add(binding_id)
            binding = bindings[binding_id]
            origin = binding.origin
            if isinstance(origin, TransformedValidityOrigin):
                sources = [(origin.binding_id, origin.binding_hash)]
            elif isinstance(origin, BinaryTransformedValidityOrigin):
                sources = [(source.binding_id, source.descriptor_hash) for source in origin.sources]
            else:
                sources = []
            parents = []
            for parent_id, parent_hash in sources:
                parent = bindings.get(parent_id)
                if parent is None or parent.descriptor_hash != parent_hash:
                    raise ValueError("Portable validity parent provenance does not match")
                parents.append(parent)
            if _has_aligned_validity_intermediate(nodes, binding) or any(
                not _direct_validity_links(nodes, binding, parent) for parent in parents
            ):
                if feature_format != "tensorus-portable-release-v6" or not (
                    _aligned_binary_lineage_matches(nodes, binding, parents)
                    or _indexed_binary_lineage_matches(nodes, binding, parents)
                ):
                    raise ValueError("Portable validity parent provenance does not match")
                aligned_found = True
            for parent in parents:
                visit(parent.id)
            visiting.remove(binding_id)
            reached.add(binding_id)

        for selected in self.validity.selected_binding_ids:
            visit(selected)
        if reached != set(bindings):
            raise ValueError("Portable validity contains unreachable bindings")
        if feature_format == "tensorus-portable-release-v6" and not aligned_found:
            raise ValueError("Portable v6 release requires aligned binary validity provenance")

    def _validate_plans(self) -> None:
        assert self.lineage is not None and self.plans is not None
        nodes = {node.id: node for node in self.lineage.versions}
        entries = self._plan_entries()
        for entry, evidence in entries:
            plan, receipt = evidence.plan, entry.receipt
            if _has_selected_preprocessing(plan.definition) and (
                self.validity is None
                or not any(
                    isinstance(binding.origin, BinaryTransformedValidityOrigin)
                    and binding.origin.plan_id == plan.id
                    for binding in self.validity.bindings
                )
            ):
                raise ValueError(
                    "Selected preprocessing plan evidence requires its completed validity binding"
                )
            if _pinned_v2_reduction(plan.definition) and (
                self.validity is None
                or not any(
                    isinstance(binding.origin, TransformedValidityOrigin)
                    and binding.origin.plan_id == plan.id
                    for binding in self.validity.bindings
                )
            ):
                raise ValueError("Pinned reduction plan evidence requires its completed binding")
            if _requires_positional_binding(plan.definition) and (
                self.validity is None
                or not any(
                    isinstance(binding.origin, TransformedValidityOrigin)
                    and binding.origin.plan_id == plan.id
                    for binding in self.validity.bindings
                )
            ):
                raise ValueError("Pinned positional plan evidence requires its completed binding")
            if _pinned_v2_conversion(plan.definition) and (
                self.validity is None
                or not any(
                    isinstance(binding.origin, TransformedValidityOrigin)
                    and binding.origin.plan_id == plan.id
                    for binding in self.validity.bindings
                )
            ):
                raise ValueError("Pinned conversion plan evidence requires its completed binding")
            if _pinned_v2_scaling(plan.definition) and (
                self.validity is None
                or not any(
                    isinstance(binding.origin, TransformedValidityOrigin)
                    and binding.origin.plan_id == plan.id
                    for binding in self.validity.bindings
                )
            ):
                raise ValueError("Pinned scaling plan evidence requires its completed binding")
            if _pinned_v2_cast(plan.definition) and (
                self.validity is None
                or not any(
                    isinstance(binding.origin, TransformedValidityOrigin)
                    and binding.origin.plan_id == plan.id
                    for binding in self.validity.bindings
                )
            ):
                raise ValueError("Pinned cast plan evidence requires its completed binding")
            if _pinned_v2_scalar(plan.definition) and (
                self.validity is None
                or not any(
                    isinstance(binding.origin, TransformedValidityOrigin)
                    and binding.origin.plan_id == plan.id
                    for binding in self.validity.bindings
                )
            ):
                raise ValueError("Pinned scalar plan evidence requires its completed binding")
            if (
                receipt.plan_id != str(plan.id)
                or receipt.plan_hash != plan.plan_hash
                or receipt.status != "succeeded"
                or receipt.evidence_consistent is not True
                or receipt.goal_verified is not False
                or receipt.issue is not None
                or len(receipt.steps) != len(evidence.steps)
            ):
                raise ValueError("Portable plan receipt does not match the succeeded plan")
            for step, recorded in zip(evidence.steps, receipt.steps, strict=True):
                node = nodes[step.output]
                sources = [nodes[source_id] for source_id in step.sources]
                if (
                    recorded.step_id != step.definition.id
                    or recorded.operation != step.definition.operation
                    or recorded.version_id != str(step.output)
                    or recorded.source_version_ids != [str(source_id) for source_id in step.sources]
                    or recorded.checksum != node.checksum
                ):
                    raise ValueError("Portable receipt steps do not match plan and lineage")
                if step.definition.operation == Operation.profile:
                    checkpoint = step.checkpoint
                    if (
                        not isinstance(checkpoint, dto.ProfileResult)
                        or step.output != step.sources[0]
                        or checkpoint.shape != node.shape
                        or checkpoint.dtype != node.dtype
                        or checkpoint.statistics.model_dump(mode="json") != node.statistics
                        or recorded.content_integrity != "not_read"
                    ):
                        raise ValueError("Portable profile evidence does not match lineage")
                    continue
                try:
                    expected_parameters, _ = _receipt_parameters(step.definition)
                except (TensorusError, ValidationError) as exc:
                    raise ValueError("Portable plan parameters are invalid") from exc
                operation = node.operation or {}
                if (
                    not isinstance(step.checkpoint, dto.TransformResult)
                    or node.dataset != plan.definition.dataset
                    or node.name != step.definition.output
                    or set(node.parents) != set(step.sources)
                    or node.write_contract != plan.definition.write_contract
                    or operation.get("plan_id") != str(plan.id)
                    or operation.get("step_id") != step.definition.id
                    or operation.get("name") != step.definition.operation
                    or not _parameters_match(
                        step.definition.operation, expected_parameters, operation.get("parameters")
                    )
                    or operation.get("source_checksums") != [source.checksum for source in sources]
                    or step.checkpoint.nbytes != node.nbytes
                    or recorded.content_integrity != "sha256_verified"
                ):
                    raise ValueError("Portable output evidence does not match plan and lineage")
                if _v2_conversion_definition(plan.definition) and not _v2_conversion_step_matches(
                    step.definition,
                    [source.model_dump(mode="json") for source in sources],
                    node.model_dump(mode="json"),
                ):
                    raise ValueError("Portable conversion evidence does not match its semantics")
                if _v2_positional_definition(plan.definition) and not _v2_positional_step_matches(
                    step.definition,
                    [source.model_dump(mode="json") for source in sources],
                    node.model_dump(mode="json"),
                ):
                    raise ValueError("Portable positional evidence does not match its semantics")
                if _v2_scaling_definition(plan.definition) and not _v2_scaling_step_matches(
                    step.definition,
                    [source.model_dump(mode="json") for source in sources],
                    node.model_dump(mode="json"),
                ):
                    raise ValueError("Portable scaling evidence does not match its semantics")
                if _v2_cast_definition(plan.definition) and not _v2_cast_step_matches(
                    step.definition,
                    [source.model_dump(mode="json") for source in sources],
                    node.model_dump(mode="json"),
                ):
                    raise ValueError("Portable cast evidence does not match its semantics")
                if _v2_scalar_definition(plan.definition) and not _v2_scalar_step_matches(
                    step.definition,
                    [source.model_dump(mode="json") for source in sources],
                    node.model_dump(mode="json"),
                ):
                    raise ValueError("Portable scalar evidence does not match its semantics")

        if self.validity is not None:
            plans = {evidence.plan.id: evidence.plan for _, evidence in entries}
            evidence_by_plan = {evidence.plan.id: evidence for _, evidence in entries}
            bindings = {binding.id: binding for binding in self.validity.bindings}
            for binding in bindings.values():
                origin = binding.origin
                if (
                    not isinstance(
                        origin, (TransformedValidityOrigin, BinaryTransformedValidityOrigin)
                    )
                    or origin.plan_id not in plans
                ):
                    continue
                plan = plans[origin.plan_id]
                if isinstance(origin, TransformedValidityOrigin):
                    parent = bindings[origin.binding_id]
                    matches = _paired_definition(
                        plan.definition,
                        str(parent.data_version_id),
                        str(parent.mask_version_id),
                        str(parent.id),
                        parent.descriptor_hash,
                    )
                    if matches and plan.definition.coordinate_window is not None:
                        try:
                            matches = _v2_window_plan_matches(
                                plan.definition,
                                nodes[parent.data_version_id].model_dump(mode="json"),
                            )
                        except TensorusError as exc:
                            raise ValueError(
                                "Invalid portable coordinate window semantics"
                            ) from exc
                    if (
                        matches
                        and plan.definition.semantic_policy == "v2"
                        and plan.definition.steps[0].operation == Operation.reindex_coordinates
                    ):
                        matches = all(
                            _v2_reindex_step_matches(
                                step.definition,
                                [nodes[source].model_dump(mode="json") for source in step.sources],
                                nodes[step.output].model_dump(mode="json"),
                            )
                            for step in evidence_by_plan[origin.plan_id].steps
                        )
                    if matches and _pinned_v2_reduction(plan.definition):
                        matches = all(
                            _v2_reduction_step_matches(
                                step.definition,
                                [nodes[source].model_dump(mode="json") for source in step.sources],
                                nodes[step.output].model_dump(mode="json"),
                            )
                            for step in evidence_by_plan[origin.plan_id].steps
                        )
                    if matches and _v2_positional_definition(plan.definition):
                        matches = all(
                            _v2_positional_step_matches(
                                step.definition,
                                [nodes[source].model_dump(mode="json") for source in step.sources],
                                nodes[step.output].model_dump(mode="json"),
                            )
                            for step in evidence_by_plan[origin.plan_id].steps
                        )
                else:
                    parents = [
                        bindings[source.binding_id].model_dump(mode="json")
                        for source in origin.sources
                    ]
                    matches = _paired_binary_definition(plan.definition, parents)
                    if (
                        matches
                        and plan.definition.semantic_policy == "v2"
                        and len(plan.definition.steps) in (4, 6, 8)
                    ):
                        try:
                            matches = _v2_alignment_plan_matches(
                                plan.definition,
                                [
                                    nodes[bindings[source.binding_id].data_version_id].model_dump(
                                        mode="json"
                                    )
                                    for source in origin.sources
                                ],
                            )
                        except TensorusError as exc:
                            raise ValueError(
                                "Invalid portable coordinate intersection semantics"
                            ) from exc
                if (
                    plan.plan_hash != origin.plan_hash
                    or not matches
                    or plan.results["values"].version_id != binding.data_version_id
                    or plan.results["validity"].version_id != binding.mask_version_id
                ):
                    raise ValueError("Portable plan does not match transformed validity provenance")
class PortableHistorySelection(Model):
    release_id: UUID
    include_lineage: StrictBool = False
    validity_binding_ids: list[UUID] | None = Field(
        default=None, min_length=1, max_length=MAX_PORTABLE_VALIDITY_SELECTIONS
    )
    plan_ids: list[UUID] | None = Field(
        default=None, min_length=1, max_length=MAX_PORTABLE_PLAN_SELECTIONS
    )

    @model_validator(mode="after")
    def validate_selections(self) -> Self:
        if not self.include_lineage and (
            self.validity_binding_ids is not None or self.plan_ids is not None
        ):
            raise ValueError("Validity and plan selections require lineage")
        for selected in (self.validity_binding_ids, self.plan_ids):
            if selected is not None and len(selected) != len(set(selected)):
                raise ValueError("Validity and plan selections must be unique")
        return self
class PortableHistoryExport(Model):
    releases: list[PortableHistorySelection] = Field(
        min_length=1, max_length=MAX_PORTABLE_HISTORY_RELEASES
    )

    @model_validator(mode="after")
    def unique_releases(self) -> Self:
        ids = [item.release_id for item in self.releases]
        if len(set(ids)) != len(ids):
            raise ValueError("Release selections must be unique")
        return self
class PortableHistoryReference(Model):
    release_id: UUID
    content_hash: Hash
class PortableHistoryManifest(Model):
    format: Literal["tensorus-release-history-manifest-v1"]
    tenant_id: UUID
    dataset_id: UUID
    dataset: Name
    releases: list[PortableHistoryReference] = Field(
        min_length=1, max_length=MAX_PORTABLE_HISTORY_RELEASES
    )
    manifest_hash: Hash

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        ids = [item.release_id for item in self.releases]
        if len(set(ids)) != len(ids):
            raise ValueError("Release history manifest contains duplicate release IDs")
        if _hash(self.model_dump(mode="json", exclude={"manifest_hash"})) != self.manifest_hash:
            raise ValueError("Release history manifest hash does not match")
        return self
class PortableReleaseHistory(Model):
    format: Literal["tensorus-portable-release-history-v1"]
    manifest: PortableHistoryManifest
    releases: list[PortableRelease] = Field(min_length=1, max_length=MAX_PORTABLE_HISTORY_RELEASES)

    @model_validator(mode="after")
    def validate_history(self) -> Self:
        if len(canonical_json(self.model_dump(mode="json")).encode()) > MAX_PORTABLE_HISTORY_BYTES:
            raise ValueError("Portable release history exceeds 4 MiB")
        if len(self.releases) != len(self.manifest.releases):
            raise ValueError("Release history does not match its manifest")
        snapshots: dict[UUID, str] = {}
        contracts: dict[int, str] = {}
        members: dict[UUID, str] = {}
        member_descriptors: dict[UUID, str] = {}
        lineage: dict[UUID, str] = {}
        lineage_descriptors: dict[UUID, str] = {}
        bindings: dict[UUID, str] = {}
        plans: dict[UUID, str] = {}
        write_spec: str | None = None

        def remember(items: dict[Any, str], key: Any, value: object, label: str) -> None:
            encoded = canonical_json(value)
            if items.setdefault(key, encoded) != encoded:
                raise ValueError(f"Release history contains inconsistent {label}")

        for reference, portable in zip(self.manifest.releases, self.releases, strict=True):
            release = portable.release
            if (
                reference.release_id != release.id
                or reference.content_hash != portable.content_hash
                or release.tenant_id != self.manifest.tenant_id
                or release.dataset_id != self.manifest.dataset_id
                or release.dataset != self.manifest.dataset
            ):
                raise ValueError("Release history does not match its manifest")
            spec = canonical_json(portable.write_spec.model_dump(mode="json"))
            if write_spec is not None and spec != write_spec:
                raise ValueError("Release history contains inconsistent write specifications")
            write_spec = spec
            remember(
                contracts,
                portable.contract.revision,
                portable.contract.model_dump(mode="json"),
                "contract revisions",
            )
            for contract in portable.member_contracts or []:
                remember(
                    contracts,
                    contract.revision,
                    contract.model_dump(mode="json"),
                    "contract revisions",
                )
            remember(
                snapshots,
                portable.snapshot.id,
                {
                    "snapshot": portable.snapshot.model_dump(mode="json"),
                    "members": sorted(
                        (member.name, str(member.version_id)) for member in release.report.members
                    ),
                },
                "snapshot membership",
            )
            for member in release.report.members:
                remember(
                    members,
                    member.version_id,
                    member.model_dump(
                        mode="json",
                        exclude={
                            "contract_errors",
                            "quality",
                            "quality_mask",
                            "quality_policy",
                            "contract_override",
                            "passed",
                        },
                    ),
                    "version claims",
                )
                remember(
                    member_descriptors,
                    member.version_id,
                    member.model_dump(
                        mode="json",
                        include={
                            "name",
                            "dtype",
                            "shape",
                            "nbytes",
                            "checksum",
                            "metadata_hash",
                            "operation",
                            "parents",
                        },
                    ),
                    "version descriptors",
                )
            if portable.lineage is not None:
                for node in portable.lineage.versions:
                    remember(lineage, node.id, node.model_dump(mode="json"), "lineage versions")
                    remember(
                        lineage_descriptors,
                        node.id,
                        {
                            "name": node.name,
                            "dtype": node.dtype,
                            "shape": node.shape,
                            "nbytes": node.nbytes,
                            "checksum": node.checksum,
                            "metadata_hash": _hash(node.metadata),
                            "operation": node.operation,
                            "parents": [str(parent) for parent in node.parents],
                        },
                        "lineage descriptors",
                    )
            if portable.validity is not None:
                for binding in portable.validity.bindings:
                    remember(bindings, binding.id, binding.model_dump(mode="json"), "bindings")
            if portable.plans is not None:
                for execution in portable.plans.executions:
                    evidence = _plan_evidence(execution.plan)
                    remember(plans, evidence.plan.id, execution.model_dump(mode="json"), "plans")
        for version_id in member_descriptors.keys() & lineage_descriptors.keys():
            if member_descriptors[version_id] != lineage_descriptors[version_id]:
                raise ValueError("Release history member and lineage descriptors do not match")
        return self
class ReleaseVersionMapping(Model):
    source_version_id: UUID
    destination_version_id: UUID
class ReleaseContractMapping(Model):
    source_revision: Annotated[StrictInt, Field(ge=1, le=2**31 - 1)]
    destination_revision: Annotated[StrictInt, Field(ge=1, le=2**31 - 1)]
class ReleaseHistoryImport(Model):
    name: Name
    snapshot_name: Name
    contract_revision: Annotated[StrictInt, Field(ge=1, le=2**31 - 1)]
    source: PortableRelease
    members: list[ReleaseVersionMapping] = Field(min_length=1, max_length=10_000)
    lineage_members: list[ReleaseVersionMapping] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_PORTABLE_LINEAGE_VERSIONS,
        exclude_if=lambda value: value is None,
    )
    contract_mappings: list[ReleaseContractMapping] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_RELEASE_CONTRACT_OVERRIDES,
        exclude_if=lambda value: value is None,
    )
    history_manifest: PortableHistoryManifest | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_members(self) -> Self:
        contracts = {item.revision for item in self.source.member_contracts or []}
        mappings = self.contract_mappings or []
        contract_sources = [item.source_revision for item in mappings]
        contract_destinations = [item.destination_revision for item in mappings]
        if (
            set(contract_sources) != contracts
            or len(contract_sources) != len(set(contract_sources))
            or len(contract_destinations) != len(set(contract_destinations))
            or (not contracts and self.contract_mappings is not None)
            or any(
                (item.source_revision == self.source.contract.revision)
                != (item.destination_revision == self.contract_revision)
                for item in mappings
            )
        ):
            raise ValueError("Map each selected member contract to a distinct matching revision")
        sources = [item.source_version_id for item in self.members]
        destinations = [item.destination_version_id for item in self.members]
        if (
            len(set(sources)) != len(sources)
            or len(set(destinations)) != len(destinations)
            or set(sources) != {member.version_id for member in self.source.release.report.members}
        ):
            raise ValueError("Map each source member to one distinct destination version")
        if self.source.lineage is None:
            if self.lineage_members is not None:
                raise ValueError("Portable v1 release cannot include lineage mappings")
        else:
            if self.lineage_members is None:
                raise ValueError("Map every portable lineage version")
            lineage_sources = [item.source_version_id for item in self.lineage_members]
            lineage_destinations = [item.destination_version_id for item in self.lineage_members]
            lineage_mapping = dict(zip(lineage_sources, lineage_destinations, strict=True))
            if (
                len(set(lineage_sources)) != len(lineage_sources)
                or len(set(lineage_destinations)) != len(lineage_destinations)
                or set(lineage_sources) != {node.id for node in self.source.lineage.versions}
                or any(
                    lineage_mapping[item.source_version_id] != item.destination_version_id
                    for item in self.members
                )
            ):
                raise ValueError(
                    "Map every lineage version to a distinct destination, matching release members"
                )
        if self.history_manifest is not None:
            manifest = self.history_manifest
            release = self.source.release
            references = {item.release_id: item.content_hash for item in manifest.releases}
            if (
                manifest.tenant_id != release.tenant_id
                or manifest.dataset_id != release.dataset_id
                or manifest.dataset != release.dataset
                or references.get(release.id) != self.source.content_hash
            ):
                raise ValueError("Imported release does not match its history manifest")
        return self
class ReleaseHistoryDestination(Model):
    release_id: UUID
    name: Name
    snapshot_name: Name
    contract_revision: Annotated[StrictInt, Field(ge=1, le=2**31 - 1)]
    members: list[ReleaseVersionMapping] = Field(min_length=1, max_length=10_000)
    lineage_members: list[ReleaseVersionMapping] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_PORTABLE_LINEAGE_VERSIONS,
        exclude_if=lambda value: value is None,
    )
    contract_mappings: list[ReleaseContractMapping] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_RELEASE_CONTRACT_OVERRIDES,
        exclude_if=lambda value: value is None,
    )

    def import_request(
        self, release: PortableRelease, manifest: PortableHistoryManifest
    ) -> ReleaseHistoryImport:
        if self.release_id != release.release.id:
            raise ValueError("Destination does not match the selected release")
        return ReleaseHistoryImport.model_validate(
            {
                **self.model_dump(mode="json", exclude={"release_id"}),
                "source": release.model_dump(mode="json"),
                "history_manifest": manifest.model_dump(mode="json"),
            }
        )
class ReleaseHistoryBundleImport(Model):
    source: PortableReleaseHistory
    destinations: list[ReleaseHistoryDestination] = Field(
        min_length=1, max_length=MAX_PORTABLE_HISTORY_RELEASES
    )

    @model_validator(mode="after")
    def validate_destinations(self) -> Self:
        if len(canonical_json(self.model_dump(mode="json")).encode()) > (
            MAX_PORTABLE_HISTORY_IMPORT_BYTES
        ):
            raise ValueError("Release history bundle import exceeds 8 MiB")
        ids = [item.release_id for item in self.destinations]
        if len(set(ids)) != len(ids) or set(ids) != {
            release.release.id for release in self.source.releases
        }:
            raise ValueError("Map each selected release exactly once")
        for values in (
            [item.name for item in self.destinations],
            [item.snapshot_name for item in self.destinations],
        ):
            if len(set(values)) != len(values):
                raise ValueError("Destination history and snapshot names must be unique")
        source_to_destination: dict[UUID, UUID] = {}
        destination_to_source: dict[UUID, UUID] = {}
        contract_sources: dict[int, int] = {}
        contract_destinations: dict[int, int] = {}
        check_contracts = any(
            release.format == "tensorus-portable-release-v9" for release in self.source.releases
        )
        for request in self.import_requests():
            if check_contracts:
                contract_pairs = [
                    (request.source.contract.revision, request.contract_revision),
                    *(
                        (item.source_revision, item.destination_revision)
                        for item in request.contract_mappings or []
                    ),
                ]
                for source_revision, destination_revision in contract_pairs:
                    if (
                        contract_sources.setdefault(source_revision, destination_revision)
                        != destination_revision
                        or contract_destinations.setdefault(destination_revision, source_revision)
                        != source_revision
                    ):
                        raise ValueError(
                            "Contract mappings must be consistent across release history"
                        )
            for mapping in request.lineage_members or request.members:
                source, destination = mapping.source_version_id, mapping.destination_version_id
                if (
                    source_to_destination.setdefault(source, destination) != destination
                    or destination_to_source.setdefault(destination, source) != source
                ):
                    raise ValueError("Version mappings must be consistent across release history")
        return self

    def import_requests(self) -> list[ReleaseHistoryImport]:
        destinations = {item.release_id: item for item in self.destinations}
        return [
            destinations[release.release.id].import_request(release, self.source.manifest)
            for release in self.source.releases
        ]
class ImportedReleaseSummary(Model):
    id: UUID
    tenant_id: UUID
    dataset_id: UUID
    dataset: str
    name: str
    snapshot_id: UUID
    contract_revision: int
    request_hash: Hash
    source_hash: Hash
    created_at: str
    created_by: str
class ImportedRelease(ImportedReleaseSummary):
    request: ReleaseHistoryImport

    @model_validator(mode="after")
    def checked_contract_revision(self) -> Self:
        if self.contract_revision != self.request.contract_revision:
            raise ValueError("Imported history contract revision contradicts its request")
        return self
class ImportedReleaseHistoryBundle(Model):
    format: Literal["tensorus-release-history-bundle-import-v1"]
    manifest_hash: Hash
    imports: list[ImportedRelease] = Field(min_length=1, max_length=MAX_PORTABLE_HISTORY_RELEASES)
class ImportedHistoryInspectionRequest(Model):
    import_ids: list[UUID] = Field(min_length=1, max_length=MAX_PORTABLE_HISTORY_RELEASES)

    @model_validator(mode="after")
    def unique_imports(self) -> Self:
        if len(set(self.import_ids)) != len(self.import_ids):
            raise ValueError("Imported history IDs must be unique")
        return self
class ImportedHistoryIndexLookupRequest(Model):
    anchor_import_id: UUID
    after_id: UUID | None = None
    limit: Annotated[StrictInt, Field(ge=1, le=MAX_HISTORY_INDEX_PAGE)] = MAX_HISTORY_INDEX_PAGE
class ImportedHistoryScanRequest(ImportedHistoryIndexLookupRequest):
    pass
class ImportedHistoryIndexBackfillRequest(Model):
    after_id: UUID | None = None
    limit: Annotated[StrictInt, Field(ge=1, le=MAX_HISTORY_INDEX_PAGE)] = MAX_HISTORY_INDEX_PAGE
class ImportedHistoryIndexReviewRequest(ImportedHistoryIndexBackfillRequest):
    pass
class ImportedHistoryIndexDiagnosticRequest(Model):
    offset: Annotated[StrictInt, Field(ge=0, le=MAX_HISTORY_INDEX_DIAGNOSTIC_OFFSET)] = 0
    limit: Annotated[StrictInt, Field(ge=1, le=MAX_HISTORY_INDEX_PAGE)] = MAX_HISTORY_INDEX_PAGE
class ImportedHistoryIndexReconcileRequest(ImportedHistoryIndexReviewRequest):
    review_hash: Hash
