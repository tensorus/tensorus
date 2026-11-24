"""Verified terminal plan checkpoints, including failed and cancelled plans."""
from typing import Annotated
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import JsonValue
from pydantic import StrictInt
from pydantic import TypeAdapter
from pydantic import ValidationError
from pydantic import model_validator
from tensorus.arrays import canonical_hash
from tensorus.arrays import canonical_json
from tensorus.arrays import descriptor_nbytes
from tensorus.errors import StorageError
from tensorus.errors import TensorusError
from tensorus.models import DType
from tensorus.models import Model
from tensorus.models import Name
from tensorus.models import Operation
from tensorus.models import PlanDefinition
from tensorus.models import StepName
from tensorus.models import WriteContractBinding
from tensorus.observation_scope import OBSERVATION_SCOPE_V3_FORMAT
from tensorus.observation_scope import OBSERVATION_SCOPE_V4_FORMAT
from tensorus.observation_scope import V4_COORDINATE_OPERATIONS
from tensorus.observation_scope import V4_FITTED_OPERATIONS
from tensorus.observation_scope import validate_arithmetic_input_count
from tensorus.operations import ARITHMETIC
from tensorus.receipt import PlanReceipt
from tensorus.receipt import _parameters_match
from tensorus.receipt import _receipt_parameters
from tensorus.responses import ExecutionError
from tensorus.validity import Hash
from tensorus.validity import ValidityBinding
from tensorus.validity import _check_pair_descriptors
from tensorus.validity import _coordinate_step_matches
from tensorus.validity import _descriptor_hash
from tensorus.validity import _paired_binary_definition
from tensorus.validity import _paired_definition
from tensorus.validity import _selected_preprocessing
from tensorus.validity import _v2_alignment_plan_matches
from tensorus.validity import _v2_paired_step_matches
from tensorus.validity import _v2_window_plan_matches
PLAN_OUTCOME_FORMAT: Literal["tensorus-plan-outcome-v1"] = "tensorus-plan-outcome-v1"
SCOPED_PLAN_OUTCOME_FORMAT: Literal["tensorus-plan-outcome-v2"] = "tensorus-plan-outcome-v2"
COORDINATE_PLAN_OUTCOME_FORMAT: Literal["tensorus-plan-outcome-v3"] = "tensorus-plan-outcome-v3"
PLAN_OUTCOME_SNAPSHOT_FORMAT: Literal["tensorus-plan-outcome-catalog-snapshot-v1"] = (
    "tensorus-plan-outcome-catalog-snapshot-v1"
)
SCOPED_PLAN_OUTCOME_SNAPSHOT_FORMAT: Literal["tensorus-plan-outcome-catalog-snapshot-v2"] = (
    "tensorus-plan-outcome-catalog-snapshot-v2"
)
COORDINATE_PLAN_OUTCOME_SNAPSHOT_FORMAT: Literal["tensorus-plan-outcome-catalog-snapshot-v3"] = (
    "tensorus-plan-outcome-catalog-snapshot-v3"
)
PlanOutcomeVersion = Literal["v1", "v2", "v3"]
MAX_PLAN_OUTCOME_BYTES = 512 * 1024
_HASH = TypeAdapter(Hash)
def plan_outcome_format(version: PlanOutcomeVersion) -> str:
    if version == "v1":
        return PLAN_OUTCOME_FORMAT
    if version == "v2":
        return SCOPED_PLAN_OUTCOME_FORMAT
    if version == "v3":
        return COORDINATE_PLAN_OUTCOME_FORMAT
    raise ValueError("Plan outcome version must be v1, v2 or v3")
class PlanMaterializedOutput(Model):
    step_id: StepName
    version_id: UUID
    dataset: Name
    name: Name
    checksum: Hash
    dtype: DType
    shape: list[Annotated[StrictInt, Field(ge=0, le=2**63 - 1)]] = Field(max_length=16)
    nbytes: Annotated[StrictInt, Field(ge=0, le=2**63 - 1)]
    source_checksums: list[Hash] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def valid_descriptor(self) -> Self:
        try:
            descriptor_nbytes(self.model_dump(mode="json"))
        except StorageError as exc:
            raise ValueError("Materialized output descriptor is invalid") from exc
        return self
class OutcomeTensorDescriptor(Model):
    id: UUID
    tenant_id: UUID
    tensor_id: UUID
    dataset_id: UUID
    version: Annotated[StrictInt, Field(ge=1)]
    name: Name
    dataset: Name
    created_by: str
    created_at: str
    checksum: Hash
    shape: list[Annotated[StrictInt, Field(ge=0, le=2**63 - 1)]] = Field(max_length=16)
    dtype: DType
    nbytes: Annotated[StrictInt, Field(ge=0, le=2**63 - 1)]
    metadata: dict[str, JsonValue]
    operation: dict[str, JsonValue] | None
    statistics: dict[str, JsonValue]
    parents: list[UUID]
    write_contract: WriteContractBinding | None
    access: dict[str, JsonValue] | None

    @model_validator(mode="after")
    def valid_descriptor(self) -> Self:
        try:
            descriptor_nbytes(self.model_dump(mode="json"))
        except StorageError as exc:
            raise ValueError("Outcome tensor descriptor is invalid") from exc
        return self
class PlanOutcome(Model):
    format: Literal[
        "tensorus-plan-outcome-v1", "tensorus-plan-outcome-v2", "tensorus-plan-outcome-v3"
    ] = PLAN_OUTCOME_FORMAT
    tenant_id: UUID
    definition: dict[str, JsonValue]
    receipt: PlanReceipt
    created_by: str
    approved_by: str | None
    approved_at: str | None
    started_at: float | None
    finished_at: str
    attempts: Annotated[StrictInt, Field(ge=0)]
    cancel_requested: bool
    error: ExecutionError | None
    checkpoint_state: Literal["none", "partial", "all"]
    remaining_steps: list[StepName] = Field(max_length=16)
    materialized_outputs: list[PlanMaterializedOutput] = Field(max_length=16)
    checked_bytes: Annotated[StrictInt, Field(ge=0)]
    audit_checked: Literal[True] = True
    coverage: Literal["recorded_checkpoint_prefix_only"] = "recorded_checkpoint_prefix_only"
    uncheckpointed_objects_checked: Literal[False] = False
    independent_recomputation: Literal[False] = False
    version_descriptors: list[OutcomeTensorDescriptor] | None = Field(
        default=None, min_length=1, max_length=64, exclude_if=lambda value: value is None
    )
    source_bindings: list[ValidityBinding] | None = Field(
        default=None, max_length=2, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def consistent_outcome(self) -> Self:
        definition = PlanDefinition.model_validate(self.definition)
        receipt = self.receipt
        if self.format != COORDINATE_PLAN_OUTCOME_FORMAT and any(
            step.operation in V4_FITTED_OPERATIONS for step in definition.steps
        ):
            raise ValueError("Fitted preprocessing outcomes require v3 descriptor evidence")
        if (
            canonical_hash(self.definition) != receipt.plan_hash
            or not definition.matches_recorded_proposer(str(self.tenant_id), self.created_by)
            or receipt.observation_scope_format
            != (
                OBSERVATION_SCOPE_V4_FORMAT
                if self.format == COORDINATE_PLAN_OUTCOME_FORMAT
                else OBSERVATION_SCOPE_V3_FORMAT
                if self.format == SCOPED_PLAN_OUTCOME_FORMAT
                else None
            )
            or receipt.status not in ("succeeded", "failed", "cancelled")
            or receipt.evidence_consistent is not True
            or receipt.goal_verified is not False
            or receipt.issue is not None
            or len(receipt.steps) > len(definition.steps)
            or (self.approved_by is None) != (self.approved_at is None)
            or (receipt.status != "cancelled" and self.approved_by is None)
            or (receipt.status != "cancelled" and self.started_at is None)
            or (self.started_at is not None and self.approved_by is None)
            or (receipt.steps and self.approved_by is None)
            or (self.started_at is None) != (self.attempts == 0)
            or (receipt.status == "failed") != (self.error is not None)
            or (receipt.status == "cancelled" and not self.cancel_requested)
        ):
            raise ValueError("Terminal plan outcome identity or state is inconsistent")
        try:
            UUID(receipt.plan_id)
        except ValueError as exc:
            raise ValueError("Terminal plan outcome ID is invalid") from exc
        completed = definition.steps[: len(receipt.steps)]
        remaining = definition.steps[len(receipt.steps) :]
        expected_state = "none" if not completed else "partial" if remaining else "all"
        if (
            self.checkpoint_state != expected_state
            or self.remaining_steps != [step.id for step in remaining]
            or (receipt.status == "succeeded" and remaining)
        ):
            raise ValueError("Terminal plan checkpoint prefix is inconsistent")
        outputs = iter(self.materialized_outputs)
        prior: dict[str, str] = {}
        checked_checksums: dict[str, str] = {}
        output_bytes = 0
        for definition_step, receipt_step in zip(completed, receipt.steps, strict=True):
            try:
                UUID(receipt_step.version_id)
                _HASH.validate_python(receipt_step.checksum)
            except (ValueError, ValidationError) as exc:
                raise ValueError("Terminal plan receipt version evidence is invalid") from exc
            expected_sources = [
                prior[ref[1:]] if ref.startswith("@") else ref for ref in definition_step.inputs
            ]
            if (
                receipt_step.step_id != definition_step.id
                or receipt_step.operation != definition_step.operation
                or receipt_step.source_version_ids != expected_sources
            ):
                raise ValueError("Terminal plan receipt steps do not match the definition")
            if (
                self.format in (SCOPED_PLAN_OUTCOME_FORMAT, COORDINATE_PLAN_OUTCOME_FORMAT)
                and definition_step.operation in ARITHMETIC
            ):
                try:
                    normalized, _ = _receipt_parameters(definition_step)
                    validate_arithmetic_input_count(
                        definition_step.operation, len(expected_sources), normalized
                    )
                except (TensorusError, ValueError) as exc:
                    raise ValueError("Terminal plan arithmetic scope is inconsistent") from exc
            if definition_step.operation == Operation.profile:
                if (
                    receipt_step.version_id != expected_sources[0]
                    or receipt_step.content_integrity != "not_read"
                ):
                    raise ValueError("Profile evidence does not describe its recorded input")
            else:
                output = next(outputs, None)
                if (
                    output is None
                    or output.step_id != definition_step.id
                    or str(output.version_id) != receipt_step.version_id
                    or output.checksum != receipt_step.checksum
                    or output.dataset != definition.dataset
                    or output.name != definition_step.output
                    or len(output.source_checksums) != len(expected_sources)
                    or any(
                        source in checked_checksums and checked_checksums[source] != checksum
                        for source, checksum in zip(
                            expected_sources, output.source_checksums, strict=True
                        )
                    )
                    or receipt_step.content_integrity != "sha256_verified"
                ):
                    raise ValueError("Materialized output does not match its receipt step")
                output_bytes += output.nbytes
            if (
                receipt_step.version_id in checked_checksums
                and checked_checksums[receipt_step.version_id] != receipt_step.checksum
            ):
                raise ValueError("A repeated receipt version has different checksums")
            checked_checksums[receipt_step.version_id] = receipt_step.checksum
            prior[definition_step.id] = receipt_step.version_id
        if next(outputs, None) is not None or output_bytes != self.checked_bytes:
            raise ValueError("Terminal plan materialized-output coverage is inconsistent")
        if self.format == COORDINATE_PLAN_OUTCOME_FORMAT:
            self._check_coordinate_descriptors(definition)
        elif self.version_descriptors is not None or self.source_bindings is not None:
            raise ValueError("Legacy outcomes cannot contain v3 descriptor evidence")
        return self

    def _check_coordinate_descriptors(self, definition: PlanDefinition) -> None:
        if self.version_descriptors is None or self.source_bindings is None:
            raise ValueError("V3 outcomes require descriptors and explicit source bindings")
        descriptors = {
            str(item.id): item.model_dump(mode="json") for item in self.version_descriptors
        }
        ids = [str(item.id) for item in self.version_descriptors]
        expected = {
            ref for step in definition.steps for ref in step.inputs if not ref.startswith("@")
        }
        expected.update(step.version_id for step in self.receipt.steps)
        expected.update(ref for step in self.receipt.steps for ref in step.source_version_ids)
        if (
            len(ids) != len(descriptors)
            or ids != sorted(ids)
            or set(ids) != expected
            or any(item.tenant_id != self.tenant_id for item in self.version_descriptors)
        ):
            raise ValueError("V3 outcomes require the exact distinct source and output descriptors")
        outputs = {item.step_id: item for item in self.materialized_outputs}
        try:
            for step, receipt_step in zip(definition.steps, self.receipt.steps, strict=False):
                sources = [
                    descriptors[version_id] for version_id in receipt_step.source_version_ids
                ]
                result = descriptors[receipt_step.version_id]
                if result["checksum"] != receipt_step.checksum:
                    raise ValueError("Receipt checksum does not match the retained descriptor")
                if step.operation == Operation.profile:
                    continue
                output = outputs[step.id]
                operation = result["operation"]
                normalized, _ = _receipt_parameters(step)
                if (
                    not isinstance(operation, dict)
                    or result["dataset"] != output.dataset
                    or result["name"] != output.name
                    or result["dtype"] != output.dtype
                    or result["shape"] != output.shape
                    or result["nbytes"] != output.nbytes
                    or result["write_contract"]
                    != (
                        definition.write_contract.model_dump(mode="json")
                        if definition.write_contract is not None
                        else None
                    )
                    or result["parents"] != sorted(set(receipt_step.source_version_ids))
                    or operation.get("plan_id") != self.receipt.plan_id
                    or operation.get("step_id") != step.id
                    or operation.get("name") != step.operation
                    or not _parameters_match(
                        step.operation, normalized, operation.get("parameters")
                    )
                    or operation.get("source_checksums")
                    != [source["checksum"] for source in sources]
                    or output.source_checksums != [source["checksum"] for source in sources]
                ):
                    raise ValueError("V3 output descriptor does not match its plan and sources")
                if step.operation in V4_COORDINATE_OPERATIONS and (
                    canonical_json(operation.get("parameters")) != canonical_json(normalized)
                    or not _coordinate_step_matches(
                        step, sources, result, definition.semantic_policy
                    )
                ):
                    raise ValueError("Coordinate mapping, anchor or output descriptor is invalid")
            selected_sources = definition.validity_sources or []
            if len(selected_sources) != len(self.source_bindings):
                raise ValueError("V3 source bindings do not match the reviewed plan")
            bindings = [item.model_dump(mode="json") for item in self.source_bindings]
            for selected, binding in zip(selected_sources, self.source_bindings, strict=True):
                data = descriptors[str(binding.data_version_id)]
                mask = descriptors[str(binding.mask_version_id)]
                if (
                    binding.id != selected.binding_id
                    or binding.descriptor_hash != selected.descriptor_hash
                    or binding.tenant_id != self.tenant_id
                    or data["checksum"] != binding.data_checksum
                    or data["dtype"] != binding.data_dtype
                    or mask["checksum"] != binding.mask_checksum
                    or mask["dtype"] != DType.bool
                    or data["shape"] != binding.shape
                    or mask["shape"] != binding.shape
                    or _descriptor_hash(
                        data,
                        mask,
                        binding.origin.model_dump(mode="json"),
                        str(binding.id),
                        binding.created_at,
                        binding.created_by,
                    )
                    != binding.descriptor_hash
                ):
                    raise ValueError("V3 source binding does not match its exact descriptors")
                _check_pair_descriptors(data, mask)
            selected_preprocessing = None
            if len(bindings) == 1:
                binding_record = bindings[0]
                if not _paired_definition(
                    definition,
                    binding_record["data_version_id"],
                    binding_record["mask_version_id"],
                    binding_record["id"],
                    binding_record["descriptor_hash"],
                ) or not _v2_window_plan_matches(
                    definition, descriptors[binding_record["data_version_id"]]
                ):
                    raise ValueError("V3 source binding does not match the paired plan")
            elif len(bindings) == 2:
                data_sources = [descriptors[binding["data_version_id"]] for binding in bindings]
                if not _paired_binary_definition(definition, bindings) or not (
                    _v2_alignment_plan_matches(definition, data_sources)
                ):
                    raise ValueError("V3 source bindings do not match the paired plan")
                selected_preprocessing = _selected_preprocessing(definition, bindings, data_sources)
            if bindings and definition.semantic_policy == "v2":
                for step, receipt_step in zip(definition.steps, self.receipt.steps, strict=False):
                    sources = [
                        descriptors[version_id] for version_id in receipt_step.source_version_ids
                    ]
                    if not _v2_paired_step_matches(
                        definition,
                        step,
                        sources,
                        descriptors[receipt_step.version_id],
                        selected_preprocessing,
                    ):
                        raise ValueError("V3 paired output semantics do not match the plan")
        except (TensorusError, KeyError, TypeError, IndexError) as exc:
            raise ValueError("V3 outcome descriptor semantics or evidence is inconsistent") from exc
class PlanOutcomeEvidence(Model):
    outcome: PlanOutcome
    outcome_hash: Hash

    @model_validator(mode="after")
    def consistent_hash(self) -> Self:
        encoded = self.outcome.model_dump(mode="json")
        if len(
            canonical_json(encoded).encode()
        ) > MAX_PLAN_OUTCOME_BYTES or self.outcome_hash != canonical_hash(encoded):
            raise ValueError("Terminal plan outcome exceeds its limit or has a different hash")
        return self
class OutcomeVersionDescriptor(Model):
    version_id: UUID
    descriptor_hash: Hash
class PlanOutcomeSnapshot(Model):
    """A verified outcome and the catalog state needed to recheck its dependencies."""

    format: Literal[
        "tensorus-plan-outcome-catalog-snapshot-v1",
        "tensorus-plan-outcome-catalog-snapshot-v2",
        "tensorus-plan-outcome-catalog-snapshot-v3",
    ] = PLAN_OUTCOME_SNAPSHOT_FORMAT
    evidence: PlanOutcomeEvidence
    plan_record_hash: Hash
    versions: list[OutcomeVersionDescriptor] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def complete_version_set(self) -> Self:
        outcome = self.evidence.outcome
        expected_format = {
            PLAN_OUTCOME_FORMAT: PLAN_OUTCOME_SNAPSHOT_FORMAT,
            SCOPED_PLAN_OUTCOME_FORMAT: SCOPED_PLAN_OUTCOME_SNAPSHOT_FORMAT,
            COORDINATE_PLAN_OUTCOME_FORMAT: COORDINATE_PLAN_OUTCOME_SNAPSHOT_FORMAT,
        }[outcome.format]
        if self.format != expected_format:
            raise ValueError("Plan outcome snapshot format does not match its evidence")
        definition = PlanDefinition.model_validate(outcome.definition)
        expected = {
            UUID(ref) for step in definition.steps for ref in step.inputs if not ref.startswith("@")
        }
        expected.update(UUID(step.version_id) for step in outcome.receipt.steps)
        expected.update(
            UUID(ref) for step in outcome.receipt.steps for ref in step.source_version_ids
        )
        ids = [entry.version_id for entry in self.versions]
        if len(ids) != len(set(ids)) or set(ids) != expected or ids != sorted(ids, key=str):
            raise ValueError("Plan outcome snapshot must identify every distinct source and output")
        return self
