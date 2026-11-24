"""Read-only checks of durable plan evidence, not proof of user-goal success."""
import math
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Self
from pydantic import Field
from pydantic import model_validator
from tensorus.models import Model
from tensorus.models import Operation
from tensorus.models import Step
from tensorus.observation_scope import OBSERVATION_SCOPE_V2_FORMAT
from tensorus.observation_scope import OBSERVATION_SCOPE_V3_FORMAT
from tensorus.observation_scope import OBSERVATION_SCOPE_V4_FORMAT
from tensorus.observation_scope import ObservationScopeFormat
from tensorus.observation_scope import StepObservationScope
from tensorus.observation_scope import validate_step_observation_scope
from tensorus.observation_scope import validate_step_observation_scope_v2
from tensorus.observation_scope import validate_step_observation_scope_v3
from tensorus.observation_scope import validate_step_observation_scope_v4
from tensorus.operations import ARITHMETIC
from tensorus.operations import is_ambiguous_arithmetic_scalar
from tensorus.operations import parameters
class _HistoricalArithmeticParameters(Model):
    scalar: Annotated[float, Field(strict=True)] | None = None
def _receipt_parameters(step: Step) -> tuple[dict[str, Any], bool]:
    # This pre-guard normalization is only for comparing existing evidence, never execution.
    if step.operation in ARITHMETIC and is_ambiguous_arithmetic_scalar(
        step.parameters.get("scalar")
    ):
        return (
            _HistoricalArithmeticParameters.model_validate(step.parameters).model_dump(mode="json"),
            True,
        )
    return parameters(step), False
def _parameters_match(operation: Operation, expected: dict[str, Any], recorded: object) -> bool:
    if operation not in ARITHMETIC:
        return recorded == expected
    if not isinstance(recorded, dict) or set(recorded) != {"scalar"}:
        return False
    actual, scalar = recorded["scalar"], expected["scalar"]
    if scalar is None:
        return actual is None
    if type(actual) not in (int, float) or actual != scalar:
        return False
    return scalar != 0 or math.copysign(1.0, actual) == math.copysign(1.0, scalar)
class ReceiptStep(Model):
    step_id: str
    operation: Operation
    version_id: str
    source_version_ids: list[str]
    checksum: str
    content_integrity: Literal["sha256_verified", "not_read"]
    observation_scope: StepObservationScope | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def scope_matches_operation(self) -> Self:
        if self.observation_scope is not None:
            try:
                validate_step_observation_scope(
                    self.operation, self.observation_scope, len(self.source_version_ids)
                )
            except ValueError:
                try:
                    validate_step_observation_scope_v2(
                        self.operation, self.observation_scope, len(self.source_version_ids)
                    )
                except ValueError:
                    try:
                        validate_step_observation_scope_v3(
                            self.operation, self.observation_scope, len(self.source_version_ids)
                        )
                    except ValueError:
                        validate_step_observation_scope_v4(
                            self.operation, self.observation_scope, len(self.source_version_ids)
                        )
        return self
class PlanReceipt(Model):
    plan_id: str
    plan_hash: str
    status: str
    evidence_consistent: bool
    goal_verified: Literal[False]
    steps: list[ReceiptStep]
    limitations: list[str]
    issue: str | None = None
    observation_scope_format: ObservationScopeFormat | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def scope_matches_consistency(self) -> Self:
        scopes = [step.observation_scope is not None for step in self.steps]
        if self.observation_scope_format is None and any(scopes):
            raise ValueError("Observation scope requires its format marker")
        if self.observation_scope_format is not None and (
            (self.evidence_consistent and not all(scopes))
            or (not self.evidence_consistent and any(scopes))
        ):
            raise ValueError("Observation scope requires consistent complete step evidence")
        for step in self.steps:
            if step.observation_scope is None:
                continue
            validator = (
                validate_step_observation_scope_v4
                if self.observation_scope_format == OBSERVATION_SCOPE_V4_FORMAT
                else validate_step_observation_scope_v3
                if self.observation_scope_format == OBSERVATION_SCOPE_V3_FORMAT
                else validate_step_observation_scope_v2
                if self.observation_scope_format == OBSERVATION_SCOPE_V2_FORMAT
                else validate_step_observation_scope
            )
            validator(step.operation, step.observation_scope, len(step.source_version_ids))
        return self
