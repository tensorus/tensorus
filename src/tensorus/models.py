import re
from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Any, ClassVar, Literal, Self
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SecretStr,
    SerializerFunctionWrapHandler,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_serializer,
    model_validator,
)

Name = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}$")]
StepName = Annotated[str, Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,31}$")]
NPY_MAX_OVERHEAD_BYTES = 16_400
# Python, Rust and ECMAScript differ on whitespace and end anchors.
_PREPROCESSING_BOUNDARY_SPACE = (
    r"\x09-\x0d\x1c-\x20\x85\xa0\u1680\u2000-\u200a"
    r"\u2028\u2029\u202f\u205f\u3000\ufeff"
)
_PREPROCESSING_BOUNDARY_CHARACTER = rf"[^{_PREPROCESSING_BOUNDARY_SPACE}]"
# Pydantic rejects unpaired surrogates; mirror that rule in the public schema.
_PREPROCESSING_UNPAIRED_SURROGATE = (
    r"[\ud800-\udbff](?:[^\udc00-\udfff]|$)|(?:^|[^\ud800-\udbff])[\udc00-\udfff]"
)
PreprocessingIdentityText = Annotated[
    str,
    Field(
        strict=True,
        min_length=1,
        max_length=256,
        pattern=rf"^{_PREPROCESSING_BOUNDARY_CHARACTER}(?:[\s\S]*{_PREPROCESSING_BOUNDARY_CHARACTER})?$",
        json_schema_extra={
            "pattern": rf"^{_PREPROCESSING_BOUNDARY_CHARACTER}",
            "not": {"pattern": rf"[{_PREPROCESSING_BOUNDARY_SPACE}]$"},
            "allOf": [{"not": {"pattern": _PREPROCESSING_UNPAIRED_SURROGATE}}],
        },
    ),
]
ReindexFill = Annotated[StrictInt, Field(ge=-(2**53 - 1), le=2**53 - 1)] | StrictBool


def _printable_reason(value: str) -> str:
    if not value.isprintable():
        raise ValueError("Reason must contain only printable characters")
    return value


WithdrawalReason = Annotated[
    str,
    Field(
        min_length=1,
        max_length=512,
        pattern=r"\S",
        description="A nonblank, single-line reason containing only printable characters.",
    ),
    AfterValidator(_printable_reason),
]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Role(StrEnum):
    reader = "reader"
    writer = "writer"
    admin = "admin"


class PlanStatus(StrEnum):
    proposed = "proposed"
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class DType(StrEnum):
    bool = "bool"
    int8 = "int8"
    uint8 = "uint8"
    int16 = "int16"
    uint16 = "uint16"
    int32 = "int32"
    uint32 = "uint32"
    int64 = "int64"
    uint64 = "uint64"
    float16 = "float16"
    float32 = "float32"
    float64 = "float64"


AgentAction = Literal["read", "ingest", "propose", "cancel"]


def validate_agent_actions(actions: list[AgentAction]) -> None:
    if len(set(actions)) != len(actions) or "read" not in actions:
        raise ValueError("Agent actions must be unique and include read")
    if "cancel" in actions and "propose" not in actions:
        raise ValueError("Cancellation requires proposal permission")


class AgentScope(Model):
    name: str
    dataset: Name
    dataset_id: str
    actions: list[AgentAction]
    expires_at: float
    require_protected_outputs: StrictBool = Field(
        default=False, exclude_if=lambda value: value is False
    )
    require_full_scope_for_ingest: StrictBool = Field(
        default=False, exclude_if=lambda value: value is False
    )

    @model_validator(mode="after")
    def validate_ingest_constraint(self) -> Self:
        if self.require_full_scope_for_ingest and "ingest" not in self.actions:
            raise ValueError("Scope-protected ingestion requires ingest permission")
        return self


class AgentDatasetScope(Model):
    dataset: Name
    dataset_id: str
    actions: list[AgentAction] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def validate_actions(self) -> Self:
        validate_agent_actions(self.actions)
        return self


class MultiDatasetAgentScope(Model):
    name: str
    datasets: list[AgentDatasetScope] = Field(min_length=2, max_length=16)
    expires_at: float
    require_protected_outputs: StrictBool = Field(
        default=False, exclude_if=lambda value: value is False
    )
    require_full_scope_for_ingest: StrictBool = Field(
        default=False, exclude_if=lambda value: value is False
    )

    @model_validator(mode="after")
    def distinct_datasets(self) -> Self:
        if len({item.dataset_id for item in self.datasets}) != len(self.datasets) or len(
            {item.dataset for item in self.datasets}
        ) != len(self.datasets):
            raise ValueError("Agent datasets must be unique")
        if self.require_full_scope_for_ingest and not any(
            "ingest" in item.actions for item in self.datasets
        ):
            raise ValueError("Scope-protected ingestion requires ingest permission")
        return self


AgentScopeValue = AgentScope | MultiDatasetAgentScope


class Principal(Model):
    tenant_id: str
    key_id: str
    role: Role
    scope: AgentScopeValue | None = None

    def require(self, role: Role) -> None:
        from tensorus.errors import Forbidden

        levels = {Role.reader: 0, Role.writer: 1, Role.admin: 2}
        if levels[self.role] < levels[role]:
            raise Forbidden()

    def require_action(self, action: AgentAction) -> None:
        from tensorus.errors import Forbidden

        if self.scope is not None and not self.permitted_dataset_ids(action):
            raise Forbidden("The agent grant does not permit this action")

    def permitted_dataset_ids(self, action: AgentAction) -> list[str]:
        if isinstance(self.scope, AgentScope):
            return [self.scope.dataset_id] if action in self.scope.actions else []
        if isinstance(self.scope, MultiDatasetAgentScope):
            return [item.dataset_id for item in self.scope.datasets if action in item.actions]
        raise ValueError("An unscoped credential has no agent dataset list")

    def require_dataset_action(self, dataset_id: str, action: AgentAction) -> None:
        from tensorus.errors import Forbidden

        if self.scope is not None and dataset_id not in self.permitted_dataset_ids(action):
            raise Forbidden("The agent grant does not permit this action on the dataset")

    def require_unscoped(self) -> None:
        from tensorus.errors import Forbidden

        if self.scope is not None:
            raise Forbidden("This action requires an unscoped tenant credential")


class TensorSpec(Model):
    model_config = ConfigDict(
        json_schema_extra={
            "not": {
                "properties": {"dtype": {"type": "string"}, "dtypes": {"type": "array"}},
                "required": ["dtype", "dtypes"],
            }
        }
    )

    dtype: DType | None = None
    dtypes: (
        Annotated[
            list[DType],
            Field(min_length=1, max_length=len(DType), json_schema_extra={"uniqueItems": True}),
        ]
        | None
    ) = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Unique allowed dtypes; cannot be combined with a single required dtype.",
    )
    shape: list[Annotated[int, Field(ge=0)] | None] | None = Field(default=None, max_length=16)
    metadata_schema: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_dtype_constraints(self) -> Self:
        if self.dtype is not None and self.dtypes is not None:
            raise ValueError("Specify either dtype or dtypes, not both")
        if self.dtypes is not None and len(set(self.dtypes)) != len(self.dtypes):
            raise ValueError("Allowed dtypes must be unique")
        return self


class WriteContractBinding(Model):
    generation: Annotated[StrictInt, Field(ge=1, le=2_147_483_647)]
    revision: Annotated[StrictInt, Field(ge=1, le=2_147_483_647)]
    definition_hash: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]


class NpyWriteCondition(Model):
    dataset_id: UUID
    write_contract_generation: Annotated[StrictInt, Field(ge=0, le=2_147_483_647)]


class ConditionalNpyReceiptRequest(Model):
    checksum: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    expected_version: Annotated[StrictInt, Field(ge=0)]
    condition: NpyWriteCondition


class WriteContractActivation(Model):
    expected_generation: Annotated[StrictInt, Field(ge=0, le=2_147_483_646)]
    revision: Annotated[StrictInt, Field(ge=1, le=2_147_483_647)]
    definition_hash: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]


class TenantCreate(Model):
    name: Name
    title: str = Field(default="", max_length=256)


class DatasetCreate(TenantCreate):
    description: str = Field(default="", max_length=4096)
    spec: TensorSpec = Field(default_factory=TensorSpec)


class KeyCreate(Model):
    name: str = Field(min_length=1, max_length=128)
    role: Role = Role.reader
    expires_in_days: int | None = Field(default=None, ge=1, le=365)


class WorkspaceMemberInvite(Model):
    subject: Annotated[str, Field(strict=True, pattern=r"^[!-~]{1,255}$")]
    role: Role = Role.reader

    @field_validator("subject")
    @classmethod
    def validate_subject(cls, value: str) -> str:
        if not re.fullmatch(r"[!-~]{1,255}", value):
            raise ValueError("OIDC subject must be printable ASCII without whitespace")
        return value


class WorkspaceMemberUpdate(Model):
    expected_generation: Annotated[StrictInt, Field(ge=1)]
    role: Role
    status: Literal["invited", "active", "suspended", "removed"]


class IdentityExchange(Model):
    workspace: Name
    expected_membership_id: UUID | None = None


class WorkspaceLockRequest(Model):
    expected_generation: Annotated[StrictInt, Field(ge=1, le=2**63 - 3)]
    request_id: UUID
    reason: Annotated[str, Field(strict=True, min_length=1, max_length=512)]
    confirmation: Literal["LOCK"]

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("A trimmed, nonempty reason is required")
        return value


class AgentGrantRetryOptions(Model):
    client_api_key: SecretStr | None = Field(default=None, exclude_if=lambda value: value is None)

    @field_validator("client_api_key")
    @classmethod
    def validate_client_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        from tensorus.credentials import valid_generated_token

        if value is not None and not valid_generated_token(value.get_secret_value()):
            raise ValueError("The client-held credential must be a generated Tensorus key")
        return value


class AgentGrantCreate(AgentGrantRetryOptions):
    name: str = Field(strict=True, min_length=1, max_length=128)
    dataset: Name
    actions: list[AgentAction] = Field(min_length=1, max_length=4)
    expires_in_seconds: Annotated[StrictInt, Field(ge=1, le=86_400)]
    require_protected_outputs: StrictBool = Field(
        default=False, exclude_if=lambda value: value is False
    )
    require_full_scope_for_ingest: StrictBool = Field(
        default=False, exclude_if=lambda value: value is False
    )
    max_plan_proposals: Annotated[StrictInt, Field(ge=1, le=100_000)] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    max_estimated_execution_units: (
        Annotated[StrictInt, Field(ge=1, le=1_000_000_000_000)] | None
    ) = Field(default=None, exclude_if=lambda value: value is None)
    max_direct_ingest_bytes: Annotated[StrictInt, Field(ge=1, le=1_000_000_000_000)] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        if self.name != self.name.strip():
            raise ValueError("Agent name must not have leading or trailing whitespace")
        validate_agent_actions(self.actions)
        if self.require_full_scope_for_ingest and "ingest" not in self.actions:
            raise ValueError("Scope-protected ingestion requires ingest permission")
        if self.max_plan_proposals is not None and "propose" not in self.actions:
            raise ValueError("A proposal allowance requires propose permission")
        if self.max_estimated_execution_units is not None and "propose" not in self.actions:
            raise ValueError("An execution allowance requires propose permission")
        if self.max_direct_ingest_bytes is not None and "ingest" not in self.actions:
            raise ValueError("A direct-ingestion allowance requires ingest permission")
        return self


class AgentDatasetGrant(Model):
    dataset: Name
    actions: list[AgentAction] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def validate_actions(self) -> Self:
        validate_agent_actions(self.actions)
        return self


class MultiDatasetAgentGrantCreate(AgentGrantRetryOptions):
    name: str = Field(strict=True, min_length=1, max_length=128)
    datasets: list[AgentDatasetGrant] = Field(min_length=2, max_length=16)
    expires_in_seconds: Annotated[StrictInt, Field(ge=1, le=86_400)]
    require_protected_outputs: StrictBool = Field(
        default=False, exclude_if=lambda value: value is False
    )
    require_full_scope_for_ingest: StrictBool = Field(
        default=False, exclude_if=lambda value: value is False
    )
    max_plan_proposals: Annotated[StrictInt, Field(ge=1, le=100_000)] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    max_estimated_execution_units: (
        Annotated[StrictInt, Field(ge=1, le=1_000_000_000_000)] | None
    ) = Field(default=None, exclude_if=lambda value: value is None)
    max_direct_ingest_bytes: Annotated[StrictInt, Field(ge=1, le=1_000_000_000_000)] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        if self.name != self.name.strip():
            raise ValueError("Agent name must not have leading or trailing whitespace")
        if len({item.dataset for item in self.datasets}) != len(self.datasets):
            raise ValueError("Agent datasets must be unique")
        if self.require_full_scope_for_ingest and not any(
            "ingest" in item.actions for item in self.datasets
        ):
            raise ValueError("Scope-protected ingestion requires ingest permission")
        if self.max_plan_proposals is not None and not any(
            "propose" in item.actions for item in self.datasets
        ):
            raise ValueError("A proposal allowance requires propose permission")
        if self.max_estimated_execution_units is not None and not any(
            "propose" in item.actions for item in self.datasets
        ):
            raise ValueError("An execution allowance requires propose permission")
        if self.max_direct_ingest_bytes is not None and not any(
            "ingest" in item.actions for item in self.datasets
        ):
            raise ValueError("A direct-ingestion allowance requires ingest permission")
        return self


class TensorWrite(Model):
    data: JsonValue
    dtype: DType = DType.float32
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    expected_version: int | None = Field(default=None, ge=0)


class UploadCreate(Model):
    dataset: Name
    name: Name
    size: Annotated[StrictInt, Field(ge=1)]
    checksum: Annotated[
        str, Field(strict=True, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    ]
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    expected_version: Annotated[StrictInt, Field(ge=0)]
    expires_in_seconds: Annotated[StrictInt, Field(ge=1, le=86_400)] = 3600


def _check_coordinate_float(value: int | float) -> int | float:
    # Large binary64 floats cannot identify adjacent integer coordinates reliably.
    if isinstance(value, float) and abs(value) >= 2**53:
        raise ValueError("Large coordinate numbers must be provided as JSON integers")
    return value


ReferenceText = Annotated[
    str,
    Field(
        strict=True,
        min_length=1,
        max_length=256,
        pattern=r"^[^\s\x00-\x1f\x7f](?:[^\x00-\x1f\x7f]*[^\s\x00-\x1f\x7f])?$",
    ),
]
_EXCLUSIVE_REFERENCES_SCHEMA: dict[str, Any] = {
    "not": {
        "required": ["time_reference", "coordinate_system"],
        "properties": {
            "time_reference": {"type": "object"},
            "coordinate_system": {"type": "object"},
        },
    }
}


def _validate_reference_text(value: str) -> str:
    if value != value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("Reference identifiers must not have surrounding whitespace or controls")
    return value


class ReferenceIdentity(Model):
    authority: ReferenceText
    identifier: ReferenceText
    revision: ReferenceText

    @field_validator("authority", "identifier", "revision")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _validate_reference_text(value)


class TimeReference(Model):
    calendar: ReferenceIdentity
    origin: ReferenceText
    time_scale: ReferenceIdentity
    time_zone: ReferenceIdentity

    @field_validator("origin")
    @classmethod
    def validate_origin(cls, value: str) -> str:
        return _validate_reference_text(value)


class CoordinateSystemReference(Model):
    system: ReferenceIdentity
    axis: ReferenceText
    axis_index: Annotated[StrictInt, Field(ge=0, le=15)]

    @field_validator("axis")
    @classmethod
    def validate_axis(cls, value: str) -> str:
        return _validate_reference_text(value)


class CoordinateRange(Model):
    model_config = ConfigDict(json_schema_extra=_EXCLUSIVE_REFERENCES_SCHEMA)
    min: StrictInt | StrictFloat
    max: StrictInt | StrictFloat
    unit: Annotated[str, Field(strict=True, min_length=1, max_length=128)] | None
    time_reference: TimeReference | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    coordinate_system: CoordinateSystemReference | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @field_validator("min", "max")
    @classmethod
    def validate_bound(cls, value: int | float) -> int | float:
        return _check_coordinate_float(value)

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        if self.time_reference is not None and self.coordinate_system is not None:
            raise ValueError("A coordinate range cannot specify both reference kinds")
        return self


class AxisDescriptor(Model):
    label: Name
    unit: Annotated[str, Field(strict=True, min_length=1, max_length=128)] | None = None

    @field_validator("unit")
    @classmethod
    def validate_unit(cls, value: str | None) -> str | None:
        if value is not None and value != value.strip():
            raise ValueError("Unit must not have leading or trailing whitespace")
        return value


class CoordinateSpan(CoordinateRange):
    unit: Annotated[str, Field(strict=True, min_length=1, max_length=128)]
    relation: Literal["covers", "overlaps", "within"] = "covers"

    @field_validator("unit")
    @classmethod
    def validate_unit(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("Coordinate span requires an exact unit")
        return value

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        if self.min > self.max:
            raise ValueError("Coordinate span minimum must not exceed its maximum")
        return self


MAX_COORDINATE_QUERY_POINTS = 64
COORDINATE_POINT_QUERY_FORMAT = "tensorus-coordinate-points-query-v1"
TYPED_COORDINATE_POINT_QUERY_FORMAT = "tensorus-typed-coordinate-points-query-v1"
MAX_COORDINATE_QUERY_COUNT = 8192
COORDINATE_COUNT_QUERY_FORMAT = "tensorus-coordinate-count-query-v1"


class CoordinateCount(CoordinateRange):
    kind: Literal["numeric_v1"] = "numeric_v1"
    unit: Annotated[str, Field(strict=True, min_length=1, max_length=128)]
    min_count: Annotated[StrictInt, Field(ge=0, le=MAX_COORDINATE_QUERY_COUNT)] = 1
    max_count: Annotated[StrictInt, Field(ge=0, le=MAX_COORDINATE_QUERY_COUNT)] | None = None

    @field_validator("unit")
    @classmethod
    def validate_unit(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("Coordinate count requires an exact unit")
        return value

    @model_validator(mode="after")
    def validate_count(self) -> Self:
        if self.min > self.max:
            raise ValueError("Coordinate count minimum must not exceed its maximum")
        if self.max_count is not None and self.min_count > self.max_count:
            raise ValueError("Coordinate count minimum must not exceed its maximum count")
        return self


class CoordinatePoints(Model):
    model_config = ConfigDict(json_schema_extra=_EXCLUSIVE_REFERENCES_SCHEMA)
    values: list[StrictInt | StrictFloat] = Field(
        min_length=1,
        max_length=MAX_COORDINATE_QUERY_POINTS,
        json_schema_extra={"uniqueItems": True},
    )
    unit: Annotated[str, Field(strict=True, min_length=1, max_length=128)]
    relation: Literal["all", "any"] = "all"
    comparison: Literal["typed_v2"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    time_reference: TimeReference | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    coordinate_system: CoordinateSystemReference | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @field_validator("values")
    @classmethod
    def validate_values(cls, values: list[int | float]) -> list[int | float]:
        values = [_check_coordinate_float(value) for value in values]
        if len(set(values)) != len(values):
            raise ValueError("Coordinate points must be distinct")
        return values

    @field_validator("unit")
    @classmethod
    def validate_unit(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("Coordinate points require an exact unit")
        return value

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        if self.time_reference is not None and self.coordinate_system is not None:
            raise ValueError("Coordinate points cannot specify both reference kinds")
        return self


class SemanticAxisFilter(AxisDescriptor):
    model_config = ConfigDict(
        json_schema_extra={
            **_EXCLUSIVE_REFERENCES_SCHEMA,
            "allOf": [
                {
                    "if": {
                        "required": ["coordinates"],
                        "properties": {"coordinates": {"const": "absent"}},
                    },
                    "then": {
                        "properties": {
                            "coordinate_span": {"type": "null"},
                            "coordinate_points": {"type": "null"},
                            "coordinate_count": {"type": "null"},
                        }
                    },
                }
            ],
        }
    )
    coordinates: Literal["present", "absent"] | None = None
    coordinate_span: CoordinateSpan | None = None
    coordinate_points: CoordinatePoints | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    coordinate_count: CoordinateCount | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    time_reference: TimeReference | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    coordinate_system: CoordinateSystemReference | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_coordinates(self) -> Self:
        if self.time_reference is not None and self.coordinate_system is not None:
            raise ValueError("An axis filter cannot specify both reference kinds")
        for requirement in (self.coordinate_span, self.coordinate_points, self.coordinate_count):
            if requirement is None:
                continue
            if self.coordinates == "absent":
                raise ValueError(
                    "Coordinate value filters cannot be combined with absent coordinates"
                )
            if self.unit is not None and self.unit != requirement.unit:
                raise ValueError("Axis and coordinate filter units must match exactly")
            for name in ("time_reference", "coordinate_system"):
                axis_reference = getattr(self, name)
                filter_reference = getattr(requirement, name)
                if (
                    axis_reference is not None
                    and filter_reference is not None
                    and axis_reference != filter_reference
                ):
                    raise ValueError("Axis and coordinate filter references must match exactly")
            if (self.time_reference is not None and requirement.coordinate_system is not None) or (
                self.coordinate_system is not None and requirement.time_reference is not None
            ):
                raise ValueError("Axis and coordinate filter reference kinds must match")
        if self.coordinate_span is not None and self.coordinate_points is not None:
            if self.coordinate_span.unit != self.coordinate_points.unit:
                raise ValueError("Coordinate span and point units must match exactly")
            for name in ("time_reference", "coordinate_system"):
                axis_reference = getattr(self, name)
                if (axis_reference or getattr(self.coordinate_span, name)) != (
                    axis_reference or getattr(self.coordinate_points, name)
                ):
                    raise ValueError("Coordinate span and point references must match exactly")
        if self.coordinate_count is not None:
            for other in (self.coordinate_span, self.coordinate_points):
                if other is None:
                    continue
                if other.unit != self.coordinate_count.unit:
                    raise ValueError("Coordinate count and other filter units must match exactly")
                for name in ("time_reference", "coordinate_system"):
                    axis_reference = getattr(self, name)
                    if (axis_reference or getattr(other, name)) != (
                        axis_reference or getattr(self.coordinate_count, name)
                    ):
                        raise ValueError(
                            "Coordinate count and other filter references must match exactly"
                        )
        return self


class SharedAxisPredicate(Model):
    kind: Literal[
        "exact_shared_axis_v1",
        "exact_cross_snapshot_shared_axis_v1",
        "overlap_shared_axis_v1",
        "overlap_cross_snapshot_shared_axis_v1",
    ] = "exact_shared_axis_v1"
    version_id: UUID
    axes: list[Name] = Field(min_length=1, max_length=16)
    anchor_snapshot_id: UUID | None = Field(default=None, exclude_if=lambda value: value is None)

    @field_validator("axes")
    @classmethod
    def unique_axes(cls, axes: list[str]) -> list[str]:
        if len(set(axes)) != len(axes):
            raise ValueError("Shared-axis query labels must be distinct")
        return axes

    @model_validator(mode="after")
    def anchor_snapshot(self) -> Self:
        cross = self.kind in (
            "exact_cross_snapshot_shared_axis_v1",
            "overlap_cross_snapshot_shared_axis_v1",
        )
        if cross != (self.anchor_snapshot_id is not None):
            raise ValueError(
                "A cross-snapshot shared-axis query requires exactly one anchor snapshot"
            )
        return self


class SemanticFilter(Model):
    axes: list[SemanticAxisFilter] = Field(default_factory=list, max_length=16)
    value_unit: Annotated[str, Field(strict=True, min_length=1, max_length=128)] | None = None
    model_id: Annotated[str, Field(strict=True, min_length=1, max_length=256)] | None = None
    model_version: Annotated[str, Field(strict=True, min_length=1, max_length=256)] | None = None
    preprocessing_id: PreprocessingIdentityText | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    preprocessing_version: PreprocessingIdentityText | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    shared_axis: SharedAxisPredicate | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @field_validator(
        "value_unit", "model_id", "model_version", "preprocessing_id", "preprocessing_version"
    )
    @classmethod
    def validate_text(cls, value: str | None) -> str | None:
        if value is not None and value != value.strip():
            raise ValueError("Semantic filters must not have leading or trailing whitespace")
        return value

    @model_validator(mode="after")
    def validate_filter(self) -> Self:
        if (
            not self.axes
            and self.shared_axis is None
            and all(
                value is None
                for value in (
                    self.value_unit,
                    self.model_id,
                    self.model_version,
                    self.preprocessing_id,
                    self.preprocessing_version,
                )
            )
        ):
            raise ValueError("Select at least one semantic filter")
        if len({axis.label for axis in self.axes}) != len(self.axes):
            raise ValueError("Semantic axis filters must have unique labels")
        return self


class TensorQuery(Model):
    dataset: Name | None = None
    text: str | None = Field(default=None, max_length=256)
    dtype: DType | None = None
    shape: list[Annotated[int, Field(ge=0)]] | None = Field(default=None, max_length=16)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    min_norm: float | None = Field(default=None, ge=0)
    max_norm: float | None = Field(default=None, ge=0)
    snapshot_id: UUID | None = None
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0, le=1_000_000)
    semantics: SemanticFilter | None = None
    indexed_execution: Literal["streamed_exact_v2"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    # A return annotation replaces the typed serialization schema with a generic dict.
    @model_serializer(mode="wrap")
    def serialize_semantics(self, handler: SerializerFunctionWrapHandler):  # type: ignore[no-untyped-def]
        result: dict[str, Any] = handler(self)
        if result.get("semantics") is None:
            result.pop("semantics", None)
        return result

    @model_validator(mode="after")
    def norm_range(self) -> Self:
        if (
            self.min_norm is not None
            and self.max_norm is not None
            and self.min_norm > self.max_norm
        ):
            raise ValueError("min_norm cannot exceed max_norm")
        if (
            self.semantics is not None
            and self.semantics.shared_axis is not None
            and self.snapshot_id is None
        ):
            raise ValueError("Shared-axis discovery requires a pinned snapshot")
        return self


class IndexedSnapshotQuery(Model):
    query: TensorQuery
    cursor: str | None = Field(default=None, max_length=2048)

    @model_validator(mode="after")
    def stable_query(self) -> Self:
        if self.query.snapshot_id is None or self.query.offset != 0:
            raise ValueError("Snapshot cursor queries require a snapshot and zero base offset")
        if self.query.text is not None:
            raise ValueError("Snapshot cursor queries do not support mutable dataset text")
        return self


class SliceRequest(Model):
    selection: list[list[StrictInt | None]] = Field(default_factory=list, max_length=16)

    @field_validator("selection")
    @classmethod
    def validate_selection(cls, value: list[list[int | None]]) -> list[list[int | None]]:
        for item in value:
            if len(item) not in (2, 3):
                raise ValueError("Each slice must be [start, stop] or [start, stop, step]")
            if len(item) == 3 and item[2] is not None and item[2] <= 0:
                raise ValueError("Only positive slice steps are supported")
        return value


class AxisSliceRequest(Model):
    selection: dict[Name, list[StrictInt | None]] = Field(max_length=16)

    @field_validator("selection")
    @classmethod
    def validate_selection(cls, value: dict[str, list[int | None]]) -> dict[str, list[int | None]]:
        SliceRequest.validate_selection(list(value.values()))
        return value


class CoordinateValues(Model):
    values: list[StrictInt | StrictFloat] = Field(max_length=8192)
    unit: Annotated[str, Field(strict=True, min_length=1, max_length=128)] | None

    @field_validator("values")
    @classmethod
    def validate_values(cls, values: list[int | float]) -> list[int | float]:
        return [_check_coordinate_float(value) for value in values]


class CoordinateSliceRequest(Model):
    selection: dict[Name, CoordinateRange] = Field(min_length=1, max_length=16)


class SimilarityQuery(Model):
    version_id: UUID
    dataset: Name | None = None
    metric: Literal["cosine", "l2"] = "cosine"
    k: int = Field(default=10, ge=1, le=100)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class SimilarityEngine(Model):
    algorithm: Literal["scaled-longdouble-v1", "difference-scaled-longdouble-v2"]
    numpy_version: str
    longdouble_bits: Annotated[StrictInt, Field(ge=1)]
    longdouble_mantissa_bits: Annotated[StrictInt, Field(ge=1)]
    block_elements: Annotated[StrictInt, Field(ge=1)]

    @model_validator(mode="after")
    def consistent_precision(self) -> Self:
        if self.longdouble_mantissa_bits > self.longdouble_bits:
            raise ValueError("Similarity mantissa precision exceeds its storage width")
        return self

    def require_metric(self, metric: Literal["cosine", "l2"]) -> None:
        expected = (
            "scaled-longdouble-v1" if metric == "cosine" else "difference-scaled-longdouble-v2"
        )
        if self.algorithm != expected:
            raise ValueError("Similarity engine does not match the requested metric")


class SnapshotCreate(Model):
    name: Name


class SnapshotRetirementReceipt(Model):
    format: Literal["tensorus-snapshot-retirement-v1"]
    tenant_id: UUID
    snapshot_id: UUID
    dataset_id: UUID
    dataset: Name
    name: Name
    tensor_count: Annotated[StrictInt, Field(ge=0, le=10_000)]
    retired_at: Annotated[str, Field(min_length=1, max_length=40)]
    retired_by: Annotated[str, Field(min_length=1, max_length=36)]
    tensor_data_deleted: Literal[False]


class Operation(StrEnum):
    profile = "profile"
    normalize = "normalize"
    standardize = "standardize"
    masked_normalize_with_placeholders = "masked_normalize_with_placeholders"
    masked_standardize_with_placeholders = "masked_standardize_with_placeholders"
    fit_selected_preprocessing = "fit_selected_preprocessing"
    apply_fitted_preprocessing = "apply_fitted_preprocessing"
    add = "add"
    masked_add_with_placeholders = "masked_add_with_placeholders"
    subtract = "subtract"
    masked_subtract_with_placeholders = "masked_subtract_with_placeholders"
    multiply = "multiply"
    masked_multiply_with_placeholders = "masked_multiply_with_placeholders"
    divide = "divide"
    masked_divide_with_placeholders = "masked_divide_with_placeholders"
    masked_scalar_arithmetic_with_placeholders = "masked_scalar_arithmetic_with_placeholders"
    matmul = "matmul"
    batched_matmul = "batched_matmul"
    transpose = "transpose"
    slice = "slice"
    chunked_slice = "chunked_slice"
    select_indices = "select_indices"
    reindex_coordinates = "reindex_coordinates"
    reindex_validity = "reindex_validity"
    coordinate_coverage = "coordinate_coverage"
    logical_and = "logical_and"
    reshape = "reshape"
    cast = "cast"
    masked_cast_with_placeholders = "masked_cast_with_placeholders"
    convert_value_unit = "convert_value_unit"
    masked_convert_value_unit = "masked_convert_value_unit"
    convert_axis_unit = "convert_axis_unit"
    convert_temperature_unit = "convert_temperature_unit"
    masked_convert_temperature_unit = "masked_convert_temperature_unit"
    chunked_count_nonzero = "chunked_count_nonzero"
    chunked_masked_count_nonzero = "chunked_masked_count_nonzero"
    chunked_count_nonzero_by_axis = "chunked_count_nonzero_by_axis"
    chunked_masked_count_nonzero_by_axis = "chunked_masked_count_nonzero_by_axis"
    chunked_masked_count_nonzero_with_placeholders = (
        "chunked_masked_count_nonzero_with_placeholders"
    )
    chunked_sum_by_axis = "chunked_sum_by_axis"
    chunked_masked_sum_with_placeholders = "chunked_masked_sum_with_placeholders"
    chunked_mean_by_axis = "chunked_mean_by_axis"
    chunked_masked_mean_with_placeholders = "chunked_masked_mean_with_placeholders"
    chunked_variance_by_axis = "chunked_variance_by_axis"
    chunked_std_by_axis = "chunked_std_by_axis"
    chunked_masked_variance_with_placeholders = "chunked_masked_variance_with_placeholders"
    chunked_masked_std_with_placeholders = "chunked_masked_std_with_placeholders"
    chunked_min_by_axis = "chunked_min_by_axis"
    chunked_max_by_axis = "chunked_max_by_axis"
    chunked_masked_min_with_placeholders = "chunked_masked_min_with_placeholders"
    chunked_masked_max_with_placeholders = "chunked_masked_max_with_placeholders"
    chunked_reduce_validity_any = "chunked_reduce_validity_any"
    sum = "sum"
    mean = "mean"
    masked_sum = "masked_sum"
    masked_mean = "masked_mean"
    masked_sum_with_placeholders = "masked_sum_with_placeholders"
    masked_mean_with_placeholders = "masked_mean_with_placeholders"
    masked_min_with_placeholders = "masked_min_with_placeholders"
    masked_max_with_placeholders = "masked_max_with_placeholders"
    masked_variance_with_placeholders = "masked_variance_with_placeholders"
    masked_std_with_placeholders = "masked_std_with_placeholders"
    reduce_validity_any = "reduce_validity_any"
    variance = "variance"
    std = "std"
    min = "min"
    max = "max"


MASKED_BINARY_OPERATIONS = {
    Operation.masked_add_with_placeholders: Operation.add,
    Operation.masked_subtract_with_placeholders: Operation.subtract,
    Operation.masked_multiply_with_placeholders: Operation.multiply,
    Operation.masked_divide_with_placeholders: Operation.divide,
}
PINNED_SCALING_VALUE_OPERATIONS = frozenset(
    {
        Operation.masked_normalize_with_placeholders,
        Operation.masked_standardize_with_placeholders,
    }
)
PINNED_CAST_VALUE_OPERATIONS = frozenset({Operation.masked_cast_with_placeholders})
PINNED_SCALAR_VALUE_OPERATIONS = frozenset({Operation.masked_scalar_arithmetic_with_placeholders})
PINNED_FITTED_OPERATIONS = frozenset(
    {Operation.fit_selected_preprocessing, Operation.apply_fitted_preprocessing}
)
PINNED_SELECTED_VALUE_OPERATIONS = (
    PINNED_SCALING_VALUE_OPERATIONS
    | PINNED_CAST_VALUE_OPERATIONS
    | PINNED_SCALAR_VALUE_OPERATIONS
    | PINNED_FITTED_OPERATIONS
)


class Step(Model):
    named_inputs: ClassVar[bool] = False
    id: StepName
    operation: Operation
    inputs: list[str] = Field(min_length=1, max_length=3)
    output: Name | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict, max_length=8)

    @model_validator(mode="after")
    def validate_step(self) -> Self:
        if self.operation == Operation.apply_fitted_preprocessing:
            if len(self.inputs) != 3:
                raise ValueError("Fitted preprocessing requires values, state and one mask")
        elif self.operation in MASKED_BINARY_OPERATIONS:
            if len(self.inputs) != 3:
                kind = (
                    "addition"
                    if self.operation == Operation.masked_add_with_placeholders
                    else "arithmetic"
                )
                raise ValueError(f"Masked {kind} requires two values and one mask")
        elif len(self.inputs) > 2:
            raise ValueError("This operation accepts at most two tensor inputs")
        for ref in self.inputs:
            if self.named_inputs and ref.startswith("tensor:"):
                if not re.fullmatch(r"tensor:[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}", ref):
                    raise ValueError("Named inputs use tensor:<name> in the workflow dataset")
            elif ref.startswith("@"):
                if not re.fullmatch(r"@[a-zA-Z][a-zA-Z0-9_-]{0,31}", ref):
                    raise ValueError("Invalid step reference")
            else:
                UUID(ref)
        if self.operation == Operation.profile and self.output is not None:
            raise ValueError("Profile is read-only and cannot name an output")
        if self.operation != Operation.profile and self.output is None:
            raise ValueError("A transforming step must name a new output tensor")
        return self


def validate_step_graph(steps: Sequence[Step]) -> None:
    seen: dict[str, Step] = {}
    outputs: set[str] = set()
    for step in steps:
        if step.id in seen:
            raise ValueError("Step IDs must be unique")
        for ref in step.inputs:
            if ref.startswith("@"):
                prior = seen.get(ref[1:])
                if prior is None:
                    raise ValueError("Step references must point to an earlier step")
                if prior.operation == Operation.profile:
                    raise ValueError("Profile results cannot be tensor inputs")
        if step.output:
            if step.output in outputs:
                raise ValueError("Output tensor names must be unique within a plan")
            outputs.add(step.output)
        seen[step.id] = step


class SemanticPolicyModel(Model):
    semantic_policy: Literal["v1", "v2"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    access_policy: Literal["source_datasets_v1"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class PlanValiditySource(Model):
    binding_id: UUID
    descriptor_hash: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]


class PlanProposerAuthority(Model):
    format: Literal["tensorus-plan-proposer-authority-v1"] = "tensorus-plan-proposer-authority-v1"
    tenant_id: UUID
    key_id: UUID
    role: Role


class PlanDefinition(SemanticPolicyModel):
    model_config = ConfigDict(
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "required": ["paired_validity_policy"],
                        "properties": {
                            "paired_validity_policy": {
                                "enum": [
                                    "pinned_positional_v1",
                                    "pinned_conversion_v1",
                                    "pinned_scaling_v1",
                                    "pinned_cast_v1",
                                    "pinned_scalar_v1",
                                    "pinned_fitted_fit_v1",
                                    "pinned_fitted_apply_v1",
                                ]
                            }
                        },
                    },
                    "then": {
                        "required": ["semantic_policy", "access_policy", "validity_sources"],
                        "properties": {
                            "semantic_policy": {"const": "v2"},
                            "access_policy": {"const": "source_datasets_v1"},
                            "validity_sources": {"type": "array", "minItems": 1, "maxItems": 1},
                            "coordinate_window": {"type": "null"},
                        },
                    },
                },
                {
                    "if": {
                        "properties": {
                            "steps": {
                                "contains": {
                                    "properties": {
                                        "operation": {
                                            "enum": [
                                                "masked_normalize_with_placeholders",
                                                "masked_standardize_with_placeholders",
                                            ]
                                        }
                                    },
                                    "required": ["operation"],
                                }
                            }
                        },
                        "required": ["steps"],
                    },
                    "then": {
                        "required": ["paired_validity_policy"],
                        "properties": {"paired_validity_policy": {"const": "pinned_scaling_v1"}},
                    },
                },
                {
                    "if": {
                        "properties": {
                            "steps": {
                                "contains": {
                                    "properties": {
                                        "operation": {"const": "masked_cast_with_placeholders"}
                                    },
                                    "required": ["operation"],
                                }
                            }
                        },
                        "required": ["steps"],
                    },
                    "then": {
                        "required": ["paired_validity_policy"],
                        "properties": {"paired_validity_policy": {"const": "pinned_cast_v1"}},
                    },
                },
                {
                    "if": {
                        "properties": {
                            "steps": {
                                "contains": {
                                    "properties": {
                                        "operation": {
                                            "const": "masked_scalar_arithmetic_with_placeholders"
                                        }
                                    },
                                    "required": ["operation"],
                                }
                            }
                        },
                        "required": ["steps"],
                    },
                    "then": {
                        "required": ["paired_validity_policy"],
                        "properties": {"paired_validity_policy": {"const": "pinned_scalar_v1"}},
                    },
                },
                {
                    "if": {
                        "required": ["steps"],
                        "properties": {
                            "steps": {
                                "contains": {
                                    "required": ["operation"],
                                    "properties": {
                                        "operation": {"const": "fit_selected_preprocessing"}
                                    },
                                }
                            }
                        },
                    },
                    "then": {
                        "required": ["paired_validity_policy"],
                        "properties": {"paired_validity_policy": {"const": "pinned_fitted_fit_v1"}},
                    },
                },
                {
                    "if": {
                        "required": ["steps"],
                        "properties": {
                            "steps": {
                                "contains": {
                                    "required": ["operation"],
                                    "properties": {
                                        "operation": {"const": "apply_fitted_preprocessing"}
                                    },
                                }
                            }
                        },
                    },
                    "then": {
                        "required": ["paired_validity_policy"],
                        "properties": {
                            "paired_validity_policy": {"const": "pinned_fitted_apply_v1"}
                        },
                    },
                },
            ]
        }
    )
    dataset: Name
    title: str = Field(min_length=1, max_length=256)
    steps: list[Step] = Field(min_length=1, max_length=16)
    proposer_authority: PlanProposerAuthority | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "Optionally bind the proposing workspace, catalog credential and role into the "
            "plan hash and require that credential to remain active."
        ),
    )
    write_contract: WriteContractBinding | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    validity_sources: list[PlanValiditySource] | None = Field(
        default=None, min_length=1, max_length=2, exclude_if=lambda value: value is None
    )
    paired_validity_policy: (
        Literal[
            "pinned_positional_v1",
            "pinned_conversion_v1",
            "pinned_scaling_v1",
            "pinned_cast_v1",
            "pinned_scalar_v1",
            "pinned_fitted_fit_v1",
            "pinned_fitted_apply_v1",
        ]
        | None
    ) = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "Explicitly pin and check a v2 paired positional, conversion, scaling, cast, "
            "selected scalar or fitted-preprocessing plan."
        ),
    )
    coordinate_window: CoordinateSliceRequest | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        validate_step_graph(self.steps)
        if self.validity_sources is not None:
            if (
                self.semantic_policy not in ("v1", "v2")
                or self.access_policy != "source_datasets_v1"
            ):
                raise ValueError(
                    "Reviewed validity sources require v1 or v2 semantics and source access"
                )
            if (
                len(self.validity_sources) == 2
                and self.validity_sources[0].binding_id == self.validity_sources[1].binding_id
            ):
                raise ValueError("Reviewed validity sources must be distinct")
        if self.coordinate_window is not None and (
            self.semantic_policy != "v2"
            or self.validity_sources is None
            or len(self.validity_sources) != 1
        ):
            raise ValueError("A coordinate window requires one reviewed v2 validity source")
        if self.paired_validity_policy is not None and (
            self.semantic_policy != "v2"
            or self.access_policy != "source_datasets_v1"
            or self.validity_sources is None
            or len(self.validity_sources) != 1
            or self.coordinate_window is not None
        ):
            raise ValueError("Pinned paired validity requires one v2 source and source access")
        if any(step.operation in PINNED_SCALING_VALUE_OPERATIONS for step in self.steps) and (
            self.paired_validity_policy != "pinned_scaling_v1"
        ):
            raise ValueError("Selected scaling requires a pinned v2 paired-validity plan")
        if any(step.operation in PINNED_CAST_VALUE_OPERATIONS for step in self.steps) and (
            self.paired_validity_policy != "pinned_cast_v1"
        ):
            raise ValueError("Selected casting requires a pinned v2 paired-validity plan")
        if any(step.operation in PINNED_SCALAR_VALUE_OPERATIONS for step in self.steps) and (
            self.paired_validity_policy != "pinned_scalar_v1"
        ):
            raise ValueError("Selected scalar arithmetic requires a pinned v2 paired-validity plan")
        for operation, policy in (
            (Operation.fit_selected_preprocessing, "pinned_fitted_fit_v1"),
            (Operation.apply_fitted_preprocessing, "pinned_fitted_apply_v1"),
        ):
            if any(step.operation == operation for step in self.steps) and (
                self.paired_validity_policy != policy
            ):
                raise ValueError("Fitted preprocessing requires its pinned v2 plan policy")
        return self

    def matches_recorded_proposer(self, tenant_id: str, created_by: str) -> bool:
        authority = self.proposer_authority
        return authority is None or (
            str(authority.tenant_id) == tenant_id and str(authority.key_id) == created_by
        )

    @property
    def writes(self) -> bool:
        return any(step.operation != Operation.profile for step in self.steps)


AxisCoordinateMode = Literal[
    "exact",
    "inner",
    "inner_indexed",
    "left_anchored",
    "right_anchored",
    "left_filled",
    "right_filled",
]


class AxisAlignmentRequest(Model):
    dataset: Name
    left_version_id: UUID
    right_version_id: UUID
    operation: Literal["add", "subtract", "multiply", "divide"]
    output: Name
    coordinate_mode: AxisCoordinateMode = "exact"
    aligned_output: Name | None = None
    left_slice_output: Name | None = None
    right_slice_output: Name | None = None
    left_reindex_output: Name | None = None
    right_reindex_output: Name | None = None
    coverage_output: Name | None = None
    fill_value: ReindexFill | None = None
    access_policy: Literal["source_datasets_v1"] | None = None
    title: Annotated[str, Field(strict=True, min_length=1, max_length=256)] = (
        "Align and combine tensors"
    )

    @model_validator(mode="after")
    def distinct_outputs(self) -> Self:
        names = [
            name
            for name in (
                self.output,
                self.aligned_output,
                self.left_slice_output,
                self.right_slice_output,
                self.left_reindex_output,
                self.right_reindex_output,
                self.coverage_output,
            )
            if name is not None
        ]
        if len(set(names)) != len(names):
            raise ValueError("Intermediate and final output names must be distinct")
        filled = self.coordinate_mode in ("left_filled", "right_filled")
        if filled != (self.fill_value is not None):
            raise ValueError("An explicit fill value is required only for filled alignment")
        if not filled and (
            self.left_reindex_output is not None
            or self.right_reindex_output is not None
            or self.coverage_output is not None
        ):
            raise ValueError("Reindex and coverage outputs are only valid for filled alignment")
        if filled and (self.left_slice_output is not None or self.right_slice_output is not None):
            raise ValueError("Slice outputs are not used by filled alignment")
        return self

    @model_serializer(mode="wrap")
    def serialize_fill(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        result = handler(self)
        for field in (
            "left_reindex_output",
            "right_reindex_output",
            "coverage_output",
            "fill_value",
            "access_policy",
        ):
            if result.get(field) is None:
                result.pop(field, None)
        return result


class WorkflowStep(Step):
    named_inputs: ClassVar[bool] = True


class WorkflowDefinition(SemanticPolicyModel):
    model_config = ConfigDict(
        json_schema_extra={
            "allOf": [
                {
                    "not": {
                        "required": ["steps"],
                        "properties": {
                            "steps": {
                                "contains": {
                                    "required": ["operation"],
                                    "properties": {
                                        "operation": {
                                            "enum": [
                                                str(operation)
                                                for operation in sorted(
                                                    PINNED_SELECTED_VALUE_OPERATIONS
                                                )
                                            ]
                                        }
                                    },
                                }
                            }
                        },
                    }
                }
            ]
        }
    )
    dataset: Name
    title: str = Field(min_length=1, max_length=256)
    description: str = Field(default="", max_length=4096)
    steps: list[WorkflowStep] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        validate_step_graph(self.steps)
        if any(step.operation in PINNED_SELECTED_VALUE_OPERATIONS for step in self.steps):
            raise ValueError("Selected operations require a reviewed binding-pinned plan")
        return self


class WorkflowWrite(Model):
    definition: WorkflowDefinition
    expected_version: int = Field(ge=0)


class WorkflowRun(Model):
    expected_version: int = Field(ge=1)
    suffix: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,31}$")
    expected_plan_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    proposer_authority: PlanProposerAuthority | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class WorkflowOrigin(Model):
    id: UUID
    name: Name
    version: int = Field(ge=1)


class PlanQuery(Model):
    status: PlanStatus | None = None
    dataset: Name | None = None
    text: str | None = Field(default=None, max_length=256)
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0, le=1_000_000)


class AuditQuery(Model):
    action: str | None = Field(default=None, max_length=64)
    resource_id: str | None = Field(default=None, max_length=128)
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0, le=1_000_000)


class GoalRequest(Model):
    dataset: Name
    goal: str = Field(min_length=1, max_length=2000)
    planner: Literal["builtin", "llm"] = "builtin"
    allow_external_catalog: StrictBool = False
    expected_model_destination_sha256: (
        Annotated[str, Field(strict=True, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")]
        | None
    ) = Field(default=None, exclude_if=lambda value: value is None)
    expected_model_transport_sha256: (
        Annotated[str, Field(strict=True, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")]
        | None
    ) = Field(default=None, exclude_if=lambda value: value is None)
    semantic_policy: Literal["v1", "v2"] | None = None
    access_policy: Literal["source_datasets_v1"] | None = None
    proposer_authority: PlanProposerAuthority | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_serializer(mode="wrap")
    def serialize_access_policy(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        result = handler(self)
        if self.access_policy is None:
            result.pop("access_policy", None)
        return result

    @model_validator(mode="after")
    def destination_requires_external_planner(self) -> Self:
        if (
            self.expected_model_destination_sha256 is not None
            or self.expected_model_transport_sha256 is not None
        ) and self.planner != "llm":
            raise ValueError("Model disclosure pins require the external planner")
        if (
            self.expected_model_transport_sha256 is not None
            and self.expected_model_destination_sha256 is None
        ):
            raise ValueError("A model transport pin requires a destination pin")
        return self


class Approval(Model):
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class DelegatedExecution(Model):
    request_id: UUID
    definition: PlanDefinition
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    authorize: Literal[True]

    @field_validator("authorize", mode="before")
    @classmethod
    def require_explicit_authorization(cls, value: object) -> object:
        if value is not True:
            raise ValueError("Explicit authorization must be true")
        return value


class JsonPage(Model):
    items: list[dict[str, JsonValue]]
    total: int
    limit: int
    offset: int
