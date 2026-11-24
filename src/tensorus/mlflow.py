"""Native MLflow Dataset (`mlflow.log_input`), evaluation metrics, and fail-closed lineage."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self
from uuid import UUID

from pydantic import Field, StrictBool, StrictInt, ValidationError, model_validator

from tensorus.arrays import (
    Array,
    array_checksum,
    canonical_hash,
    canonical_json,
    decode_unique_json,
    descriptor_nbytes,
)
from tensorus.contracts import ReleaseHash
from tensorus.errors import Conflict, LimitExceeded, StorageError, TensorusError
from tensorus.models import DType, Model, Name, TensorQuery
from tensorus.product_releases import ProductRelease
from tensorus.reference_evaluations import ReferenceEvaluation
from tensorus.responses import Release, SnapshotDetail, TensorVersion
from tensorus.semantics import ModelIdentity, PreprocessingIdentity, read_semantics
from tensorus.validity import FittedPreprocessingState, ValidityBinding, _descriptor_hash

if TYPE_CHECKING:
    from tensorus.client import Client

LINEAGE_FORMAT: Literal["tensorus-mlflow-lineage-v1"] = "tensorus-mlflow-lineage-v1"
MAX_LINEAGE_MANIFEST_BYTES = 512 * 1024
DEFAULT_ARTIFACT_DIR = "tensorus"
_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_MANIFEST_KINDS = Literal[
    "tensor_version",
    "snapshot",
    "validity_binding",
    "fitted_preprocessing_state",
    "quality_release",
    "product_release",
    "reference_evaluation",
]


def _require_mlflow(tracking_client: Any = None) -> tuple[Any, Any]:
    """Resolve an explicit duck-typed tracking client or lazily import `mlflow`."""
    if tracking_client is not None:
        return None, tracking_client
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] == "mlflow":
            raise ModuleNotFoundError(
                "MLflow interoperability requires the optional tensorus[mlflow] extra "
                '(pip install "tensorus[mlflow]" or pip install mlflow-skinny)'
            ) from exc
        raise
    return mlflow, MlflowClient()


def _resolve_run_id(mlflow_mod: Any, run_id: str | None) -> str:
    if run_id is not None:
        cleaned = run_id.strip()
        if not cleaned or not cleaned.isprintable() or "/" in cleaned or "\\" in cleaned:
            raise TensorusError("run_id must be a non-empty printable MLflow run identifier")
        return cleaned
    if mlflow_mod is not None:
        active = mlflow_mod.active_run()
        if active is not None:
            return str(active.info.run_id)
    raise TensorusError(
        "No active MLflow run found; pass run_id=... explicitly or call inside mlflow.start_run()"
    )


def _validate_label(label: str) -> str:
    if not isinstance(label, str) or not _LABEL_PATTERN.fullmatch(label):
        raise TensorusError(
            "label must start with an alphanumeric character and contain 1..64 "
            "ASCII letters, digits, '.', '_', or '-'"
        )
    return label


def _validate_artifact_dir(artifact_dir: str) -> str:
    cleaned = artifact_dir.strip("/")
    if (
        not cleaned
        or ".." in cleaned.split("/")
        or "\\" in cleaned
        or any(not _LABEL_PATTERN.fullmatch(part) for part in cleaned.split("/"))
    ):
        raise TensorusError("artifact_dir must be a safe relative artifact directory path")
    return cleaned


def _truncate_param(value: str, limit: int = 500) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


class LoggedVersionPin(Model):
    version_id: UUID
    tenant_id: UUID
    dataset_id: UUID
    dataset: str
    tensor_id: UUID
    name: str
    version: StrictInt = Field(ge=1)
    dtype: DType
    shape: list[Annotated[StrictInt, Field(ge=0)]] = Field(max_length=16)
    nbytes: Annotated[StrictInt, Field(ge=0)]
    checksum: ReleaseHash
    value_unit: str | None = None
    model: ModelIdentity | None = None
    preprocessing: PreprocessingIdentity | None = None


class LoggedSnapshotMemberPin(Model):
    name: str
    version_id: UUID
    dtype: DType
    shape: list[Annotated[StrictInt, Field(ge=0)]] = Field(max_length=16)
    nbytes: Annotated[StrictInt, Field(ge=0)]
    checksum: ReleaseHash


class LoggedSnapshotPin(Model):
    snapshot_id: UUID
    tenant_id: UUID
    dataset_id: UUID
    name: str
    tensor_count: StrictInt = Field(ge=0)
    created_at: str
    membership_Complete: StrictBool = Field(default=False, alias="membership_complete")
    members: list[LoggedSnapshotMemberPin] = Field(default_factory=list, max_length=256)

    model_config = Model.model_config | {"populate_by_name": True}


class LoggedValidityPin(Model):
    binding_id: UUID
    tenant_id: UUID
    descriptor_hash: ReleaseHash
    data_version_id: UUID
    data_checksum: ReleaseHash
    data_dtype: DType
    mask_version_id: UUID
    mask_checksum: ReleaseHash
    shape: list[Annotated[StrictInt, Field(ge=0)]] = Field(max_length=16)


class LoggedFittedStatePin(Model):
    state_version_id: UUID
    state_checksum: ReleaseHash
    state_shape: list[Annotated[StrictInt, Field(ge=0)]] = Field(max_length=16)
    algorithm: Literal["l2_normalize", "standardize"]
    algorithm_version: Annotated[StrictInt, Field(ge=1, le=2)]
    fit_plan_id: UUID
    fit_plan_hash: ReleaseHash
    training_binding_id: UUID
    training_binding_hash: ReleaseHash


class LoggedQualityMemberPin(Model):
    name: Name
    version_id: UUID
    dtype: DType
    shape: list[Annotated[StrictInt, Field(ge=0)]] = Field(max_length=16)
    nbytes: Annotated[StrictInt, Field(ge=0)]
    checksum: ReleaseHash


class LoggedQualityReleasePin(Model):
    release_id: UUID
    tenant_id: UUID
    dataset_id: UUID
    dataset: Name
    release_name: Name
    snapshot_id: UUID
    contract_revision: StrictInt = Field(ge=1)
    contract_hash: ReleaseHash
    status: Literal["failed", "proposed", "approved"]
    report_hash: ReleaseHash
    members: list[LoggedQualityMemberPin] = Field(max_length=256)


class LoggedProductRolePin(Model):
    role: Name
    version_id: UUID
    checksum: ReleaseHash
    nbytes: Annotated[StrictInt, Field(ge=0)]
    contract_hash: ReleaseHash


class LoggedProductReleasePin(Model):
    release_id: UUID
    tenant_id: UUID
    product_id: UUID
    product_name: Name
    product_revision: Annotated[StrictInt, Field(ge=1)]
    release_name: Name
    status: Literal["proposed", "approved"]
    report_hash: ReleaseHash
    descriptor_hash: ReleaseHash
    members: list[LoggedProductRolePin] = Field(min_length=1, max_length=32)


class LoggedReferenceEvaluationPin(Model):
    evaluation_id: UUID
    tenant_id: UUID
    name: Name
    status: Literal["proposed", "approved"]
    passed: StrictBool
    report_hash: ReleaseHash
    output_version_id: UUID
    output_checksum: ReleaseHash
    reference_version_id: UUID
    reference_checksum: ReleaseHash
    output_binding_id: UUID
    output_binding_hash: ReleaseHash
    reference_binding_id: UUID
    reference_binding_hash: ReleaseHash
    included_count: StrictInt = Field(ge=0)
    excluded_count: StrictInt = Field(ge=0)
    tolerance_violations: StrictInt | None = None
    mae_value: float | None = None


class MlflowLineageManifest(Model):
    format: Literal["tensorus-mlflow-lineage-v1"] = LINEAGE_FORMAT
    kind: _MANIFEST_KINDS
    label: str
    manifest_sha256: ReleaseHash
    tensor_version: LoggedVersionPin | None = None
    snapshot: LoggedSnapshotPin | None = None
    validity_binding: LoggedValidityPin | None = None
    fitted_state: LoggedFittedStatePin | None = None
    quality_release: LoggedQualityReleasePin | None = None
    product_release: LoggedProductReleasePin | None = None
    reference_evaluation: LoggedReferenceEvaluationPin | None = None

    @model_validator(mode="after")
    def check_payload_and_hash(self) -> Self:
        _validate_label(self.label)
        present = {
            "tensor_version": self.tensor_version is not None,
            "snapshot": self.snapshot is not None,
            "validity_binding": self.validity_binding is not None,
            "fitted_preprocessing_state": self.fitted_state is not None,
            "quality_release": self.quality_release is not None,
            "product_release": self.product_release is not None,
            "reference_evaluation": self.reference_evaluation is not None,
        }
        if sum(present.values()) != 1 or not present[self.kind]:
            raise ValueError("Lineage manifest kind must match exactly one populated pin payload")
        body = self._body_without_hash()
        if canonical_hash(body) != self.manifest_sha256:
            raise ValueError("Lineage manifest_sha256 does not match canonical payload")
        return self

    def _body_without_hash(self) -> dict[str, Any]:
        dumped = self.model_dump(mode="json", by_alias=True)
        dumped.pop("manifest_sha256", None)
        return dumped


@dataclass(frozen=True)
class LoggedLineageReceipt:
    run_id: str
    label: str
    kind: str
    artifact_path: str
    manifest_sha256: str
    manifest: dict[str, Any]
    params: dict[str, str]
    tags: dict[str, str]
    metrics: dict[str, float]


@dataclass(frozen=True)
class VerifiedLoggedTensor:
    values: Array
    mask: Array | None
    version: dict[str, Any]
    binding: dict[str, Any] | None
    manifest: dict[str, Any]


@dataclass(frozen=True)
class _DatasetInputSpec:
    name: str
    digest: str
    source: dict[str, Any]
    schema: dict[str, Any]
    profile: dict[str, Any]
    context: str


def _build_manifest(
    kind: _MANIFEST_KINDS, label: str, field_name: str, payload: Model
) -> MlflowLineageManifest:
    raw_payload = payload.model_dump(mode="json", by_alias=True)
    body: dict[str, Any] = {
        "format": LINEAGE_FORMAT,
        "kind": kind,
        "label": label,
        "tensor_version": None,
        "snapshot": None,
        "validity_binding": None,
        "fitted_state": None,
        "quality_release": None,
        "product_release": None,
        "reference_evaluation": None,
        field_name: raw_payload,
    }
    digest = canonical_hash(body)
    return MlflowLineageManifest.model_validate({**body, "manifest_sha256": digest})


def as_mlflow_dataset(
    resource: Mapping[str, Any],
    *,
    name: str | None = None,
) -> Any:
    """Wrap a Tensorus version, validity binding, snapshot, or release as a native MLflow `Dataset`.

    Works directly with standard `mlflow.log_input(as_mlflow_dataset(...), context="training")`
    so Tensorus tensors and releases populate MLflow's first-class Datasets UI tab.
    """
    _require_mlflow(None)
    from mlflow.data.dataset import Dataset
    from mlflow.data.dataset_source import DatasetSource

    spec = _infer_dataset_input_spec(resource, custom_name=name, context="training")

    class _TensorusDatasetSource(DatasetSource):
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        @staticmethod
        def _get_source_type() -> str:
            return "tensorus"

        def load(self, **kwargs: Any) -> dict[str, Any]:
            return dict(self._payload)

        @staticmethod
        def _can_resolve(raw_source: Any) -> bool:
            return isinstance(raw_source, str) and raw_source.startswith("tensorus://")

        @classmethod
        def _resolve(cls, raw_source: Any) -> _TensorusDatasetSource:
            return cls({"uri": str(raw_source)})

        def to_dict(self) -> dict[str, Any]:
            return dict(self._payload)

        @classmethod
        def from_dict(cls, source_dict: dict[Any, Any]) -> _TensorusDatasetSource:
            return cls(dict(source_dict))

    class _TensorusDataset(Dataset):
        def __init__(self, ds_spec: _DatasetInputSpec) -> None:
            self._schema_dict = ds_spec.schema
            self._profile_dict = ds_spec.profile
            super().__init__(
                source=_TensorusDatasetSource(ds_spec.source),
                name=ds_spec.name,
                digest=ds_spec.digest,
            )

        def _compute_digest(self) -> str:
            return "00000000"

        @property
        def profile(self) -> dict[str, Any]:
            return dict(self._profile_dict)

        @property
        def schema(self) -> dict[str, Any]:
            return dict(self._schema_dict)

        def to_dict(self) -> dict[str, str]:
            base = super().to_dict()
            base["schema"] = canonical_json(self._schema_dict)
            base["profile"] = canonical_json(self._profile_dict)
            return base

    return _TensorusDataset(spec)


def _infer_dataset_input_spec(
    resource: Mapping[str, Any],
    *,
    custom_name: str | None = None,
    context: str = "training",
) -> _DatasetInputSpec:
    raw = dict(resource)
    # 1. ProductRelease
    if "product_name" in raw and "report" in raw and "report_hash" in raw:
        pr = ProductRelease.model_validate(raw)
        colspecs = [
            {
                "type": "tensor",
                "name": m.role,
                "tensor-spec": {
                    "dtype": str(m.write_spec.dtype or "any"),
                    "shape": list(m.write_spec.shape or [-1]),
                },
            }
            for m in pr.report.members
        ]
        return _DatasetInputSpec(
            name=custom_name or f"{pr.product_name}@r{pr.product_revision}/{pr.name}",
            digest=pr.report_hash[:10],
            source={
                "uri": f"tensorus://product-releases/{pr.id}",
                "kind": "product_release",
                "release_id": str(pr.id),
                "product_name": pr.product_name,
                "product_revision": pr.product_revision,
                "report_hash": pr.report_hash,
            },
            schema={"mlflow_colspec": colspecs},
            profile={
                "status": pr.status,
                "roles": [m.role for m in pr.report.members],
                "total_nbytes": sum(m.nbytes for m in pr.report.members),
            },
            context=context,
        )
    # 2. Quality Release (`Release`)
    if "snapshot_id" in raw and "contract_revision" in raw and "report_hash" in raw:
        qr = Release.model_validate(raw)
        colspecs = [
            {
                "type": "tensor",
                "name": m.name,
                "tensor-spec": {"dtype": m.dtype.value, "shape": m.shape},
            }
            for m in qr.report.members
        ]
        return _DatasetInputSpec(
            name=custom_name or f"{qr.dataset}/{qr.name}",
            digest=qr.report_hash[:10],
            source={
                "uri": f"tensorus://releases/{qr.id}",
                "kind": "quality_release",
                "release_id": str(qr.id),
                "snapshot_id": str(qr.snapshot_id),
                "report_hash": qr.report_hash,
            },
            schema={"mlflow_colspec": colspecs},
            profile={
                "status": qr.status,
                "member_count": len(qr.report.members),
                "total_nbytes": sum(m.nbytes for m in qr.report.members),
            },
            context=context,
        )
    # 3. ValidityBinding
    if "descriptor_hash" in raw and "data_version_id" in raw and "mask_version_id" in raw:
        vb = ValidityBinding.model_validate(raw)
        return _DatasetInputSpec(
            name=custom_name or f"validity/{str(vb.id)[:8]}",
            digest=vb.descriptor_hash[:10],
            source={
                "uri": f"tensorus://validity-bindings/{vb.id}",
                "kind": "validity_binding",
                "binding_id": str(vb.id),
                "descriptor_hash": vb.descriptor_hash,
                "data_version_id": str(vb.data_version_id),
                "mask_version_id": str(vb.mask_version_id),
            },
            schema={
                "mlflow_colspec": [
                    {
                        "type": "tensor",
                        "name": "data",
                        "tensor-spec": {"dtype": vb.data_dtype.value, "shape": vb.shape},
                    },
                    {
                        "type": "tensor",
                        "name": "mask",
                        "tensor-spec": {"dtype": "bool", "shape": vb.shape},
                    },
                ]
            },
            profile={"shape": vb.shape, "data_dtype": vb.data_dtype.value},
            context=context,
        )
    # 4. Snapshot / SnapshotDetail
    if "dataset_id" in raw and "created_at" in raw and "checksum" not in raw:
        raw_snap = {**raw, "tensor_count": raw.get("tensor_count", 0)}
        sn = SnapshotDetail.model_validate(raw_snap)
        snap_digest = canonical_hash({"id": str(sn.id), "created_at": sn.created_at})[:10]
        return _DatasetInputSpec(
            name=custom_name or f"snapshot/{sn.name}",
            digest=snap_digest,
            source={
                "uri": f"tensorus://snapshots/{sn.id}",
                "kind": "snapshot",
                "snapshot_id": str(sn.id),
                "name": sn.name,
            },
            schema={"mlflow_colspec": []},
            profile={"tensor_count": sn.tensor_count, "created_at": sn.created_at},
            context=context,
        )
    # 5. TensorVersion
    if "checksum" in raw and "dtype" in raw and "shape" in raw:
        tv = TensorVersion.model_validate(raw)
        return _DatasetInputSpec(
            name=custom_name or f"{tv.dataset}/{tv.name}@v{tv.version}",
            digest=tv.checksum[:10],
            source={
                "uri": f"tensorus://datasets/{tv.dataset}/tensors/{tv.name}/versions/{tv.id}",
                "kind": "tensor_version",
                "version_id": str(tv.id),
                "checksum": tv.checksum,
            },
            schema={
                "mlflow_colspec": [
                    {
                        "type": "tensor",
                        "name": tv.name,
                        "tensor-spec": {"dtype": tv.dtype.value, "shape": tv.shape},
                    }
                ]
            },
            profile={"nbytes": tv.nbytes, "shape": tv.shape, "dtype": tv.dtype.value},
            context=context,
        )
    raise TensorusError(
        "Unsupported resource for as_mlflow_dataset; pass a TensorVersion, ValidityBinding, "
        "Snapshot, Quality Release, or ProductRelease mapping"
    )


def _write_to_mlflow(
    *,
    run_id: str | None,
    tracking_client: Any,
    artifact_dir: str,
    manifest: MlflowLineageManifest,
    params: dict[str, str],
    tags: dict[str, str],
    metrics: dict[str, float] | None = None,
    dataset_input: _DatasetInputSpec | None = None,
) -> LoggedLineageReceipt:
    mlflow_mod, client_obj = _require_mlflow(tracking_client)
    resolved_run = _resolve_run_id(mlflow_mod, run_id)
    clean_dir = _validate_artifact_dir(artifact_dir)
    artifact_path = f"{clean_dir}/{manifest.label}.json"
    manifest_dict = manifest.model_dump(mode="json", by_alias=True)
    encoded = canonical_json(manifest_dict).encode("utf-8")
    if len(encoded) > MAX_LINEAGE_MANIFEST_BYTES:
        raise LimitExceeded("Tensorus MLflow lineage manifest exceeds the 512 KiB limit")
    clean_metrics = dict(sorted((metrics or {}).items()))

    if hasattr(client_obj, "log_batch"):
        try:
            import time

            from mlflow.entities import Metric, Param, RunTag

            now_ms = int(time.time() * 1000)
            metric_entities = [Metric(k, float(v), now_ms, 0) for k, v in clean_metrics.items()]
            param_entities = [Param(k, _truncate_param(v)) for k, v in sorted(params.items())]
            tag_entities = [RunTag(k, v) for k, v in sorted(tags.items())]
            client_obj.log_batch(
                resolved_run,
                metrics=metric_entities,
                params=param_entities,
                tags=tag_entities,
            )
        except ImportError:
            _fallback_log_primitives(client_obj, resolved_run, params, tags, clean_metrics)
    else:
        _fallback_log_primitives(client_obj, resolved_run, params, tags, clean_metrics)

    # Register native MLflow DatasetInput (`Inputs` column in MLflow UI) when supported
    if dataset_input is not None and hasattr(client_obj, "log_inputs"):
        try:
            from mlflow.entities import Dataset as DatasetEntity
            from mlflow.entities import DatasetInput, InputTag

            ds_ent = DatasetEntity(
                name=dataset_input.name,
                digest=dataset_input.digest,
                source_type="tensorus",
                source=canonical_json(dataset_input.source),
                schema=canonical_json(dataset_input.schema),
                profile=canonical_json(dataset_input.profile),
            )
            in_tags = [
                InputTag(key="mlflow.data.context", value=dataset_input.context),
                InputTag(key="tensorus.label", value=manifest.label),
                InputTag(key="tensorus.manifest_sha256", value=manifest.manifest_sha256),
            ]
            client_obj.log_inputs(
                resolved_run, datasets=[DatasetInput(dataset=ds_ent, tags=in_tags)]
            )
        except ImportError:
            pass

    if hasattr(client_obj, "log_dict"):
        client_obj.log_dict(resolved_run, manifest_dict, artifact_path)
    elif hasattr(client_obj, "log_artifact"):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / f"{manifest.label}.json"
            target.write_bytes(encoded)
            client_obj.log_artifact(resolved_run, str(target), artifact_path=clean_dir)
    else:
        raise TensorusError("Tracking client must provide log_dict or log_artifact")

    return LoggedLineageReceipt(
        run_id=resolved_run,
        label=manifest.label,
        kind=manifest.kind,
        artifact_path=artifact_path,
        manifest_sha256=manifest.manifest_sha256,
        manifest=manifest_dict,
        params=dict(sorted(params.items())),
        tags=dict(sorted(tags.items())),
        metrics=clean_metrics,
    )


def _fallback_log_primitives(
    client_obj: Any,
    run_id: str,
    params: dict[str, str],
    tags: dict[str, str],
    metrics: dict[str, float],
) -> None:
    for pk, pv in sorted(params.items()):
        client_obj.log_param(run_id, pk, _truncate_param(pv))
    for tk, tv in sorted(tags.items()):
        client_obj.set_tag(run_id, tk, tv)
    if metrics and hasattr(client_obj, "log_metric"):
        for mk, mv in sorted(metrics.items()):
            client_obj.log_metric(run_id, mk, float(mv))


def read_logged_manifest(
    run_id: str | None = None,
    label: str = "input",
    *,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
    tracking_client: Any = None,
) -> dict[str, Any]:
    """Download and cryptographically validate a logged Tensorus lineage manifest from MLflow."""
    clean_label = _validate_label(label)
    clean_dir = _validate_artifact_dir(artifact_dir)
    mlflow_mod, client_obj = _require_mlflow(tracking_client)
    resolved_run = _resolve_run_id(mlflow_mod, run_id)
    rel_path = f"{clean_dir}/{clean_label}.json"

    raw_bytes: bytes
    if hasattr(client_obj, "download_artifacts"):
        with TemporaryDirectory() as tmp:
            local_path = Path(client_obj.download_artifacts(resolved_run, rel_path, tmp))
            if not local_path.is_file():
                raise StorageError(f"MLflow artifact {rel_path!r} was not found")
            if local_path.stat().st_size > MAX_LINEAGE_MANIFEST_BYTES:
                raise LimitExceeded("Downloaded MLflow lineage manifest exceeds the 512 KiB limit")
            raw_bytes = local_path.read_bytes()
    elif mlflow_mod is not None:
        loaded = mlflow_mod.artifacts.load_dict(f"runs:/{resolved_run}/{rel_path}")
        raw_bytes = json.dumps(loaded, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(raw_bytes) > MAX_LINEAGE_MANIFEST_BYTES:
            raise LimitExceeded("Downloaded MLflow lineage manifest exceeds the 512 KiB limit")
    else:
        raise TensorusError("Tracking client must provide download_artifacts")

    try:
        decoded = decode_unique_json(raw_bytes)
        validated = MlflowLineageManifest.model_validate(decoded)
    except (UnicodeDecodeError, ValueError, ValidationError, TensorusError) as exc:
        raise StorageError(
            "Logged Tensorus MLflow lineage manifest is invalid or tampered"
        ) from exc
    if validated.label != clean_label:
        raise Conflict("Logged MLflow manifest label does not match the requested label")
    return validated.model_dump(mode="json", by_alias=True)


def list_logged_manifests(
    run_id: str | None = None,
    *,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
    tracking_client: Any = None,
) -> list[dict[str, Any]]:
    """Discover and cryptographically validate all Tensorus lineage manifests in an MLflow run."""
    clean_dir = _validate_artifact_dir(artifact_dir)
    mlflow_mod, client_obj = _require_mlflow(tracking_client)
    resolved_run = _resolve_run_id(mlflow_mod, run_id)
    if not hasattr(client_obj, "list_artifacts"):
        raise TensorusError("Tracking client must provide list_artifacts")
    entries = client_obj.list_artifacts(resolved_run, path=clean_dir)
    labels: list[str] = []
    for entry in entries:
        is_dir = getattr(entry, "is_dir", False)
        path_str = str(getattr(entry, "path", ""))
        if not is_dir and path_str.endswith(".json"):
            stem = Path(path_str).stem
            if _LABEL_PATTERN.fullmatch(stem):
                labels.append(stem)
    return [
        read_logged_manifest(
            resolved_run, lbl, artifact_dir=clean_dir, tracking_client=tracking_client
        )
        for lbl in sorted(set(labels))
    ]


def _coerce_version(
    client: Client | None, version: str | UUID | Mapping[str, Any]
) -> TensorVersion:
    if isinstance(version, (str, UUID)):
        if client is None:
            raise TensorusError("Provide an authenticated Client when passing a version UUID")
        record = client.version(str(UUID(str(version))))
    elif isinstance(version, Mapping):
        record = dict(version)
    else:
        raise TensorusError("version must be a version UUID or a TensorVersion mapping")
    try:
        parsed = TensorVersion.model_validate(record)
    except ValidationError as exc:
        raise StorageError("Tensor version descriptor is invalid") from exc
    if parsed.deleted_at is not None:
        raise Conflict("Cannot log or verify a retired/deleted tensor version")
    return parsed


def log_tensor_version(
    version: str | UUID | Mapping[str, Any],
    *,
    client: Client | None = None,
    label: str = "input",
    context: str | None = "training",
    run_id: str | None = None,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
    tracking_client: Any = None,
) -> LoggedLineageReceipt:
    """Record an immutable TensorVersion SHA-256 identity and native DatasetInput in MLflow."""
    clean_label = _validate_label(label)
    parsed = _coerce_version(client, version)
    dumped = parsed.model_dump(mode="json")
    semantics = read_semantics(dumped["metadata"], tuple(dumped["shape"]), dumped["operation"])
    pin = LoggedVersionPin(
        version_id=parsed.id,
        tenant_id=parsed.tenant_id,
        dataset_id=parsed.dataset_id,
        dataset=parsed.dataset,
        tensor_id=parsed.tensor_id,
        name=parsed.name,
        version=parsed.version,
        dtype=parsed.dtype,
        shape=parsed.shape,
        nbytes=parsed.nbytes,
        checksum=parsed.checksum,
        value_unit=semantics.value_unit if semantics is not None else None,
        model=semantics.model if semantics is not None else None,
        preprocessing=semantics.preprocessing if semantics is not None else None,
    )
    manifest = _build_manifest("tensor_version", clean_label, "tensor_version", pin)
    prefix = f"tensorus.{clean_label}"
    params = {
        f"{prefix}.version_id": str(parsed.id),
        f"{prefix}.checksum": parsed.checksum,
        f"{prefix}.dataset": parsed.dataset,
        f"{prefix}.tensor": parsed.name,
        f"{prefix}.dtype": parsed.dtype.value,
        f"{prefix}.shape": json.dumps(parsed.shape, separators=(",", ":")),
    }
    tags = {
        "tensorus.lineage.format": LINEAGE_FORMAT,
        f"{prefix}.kind": "tensor_version",
        f"{prefix}.manifest_sha256": manifest.manifest_sha256,
    }
    ds_spec = _infer_dataset_input_spec(dumped, context=context) if context is not None else None
    return _write_to_mlflow(
        run_id=run_id,
        tracking_client=tracking_client,
        artifact_dir=artifact_dir,
        manifest=manifest,
        params=params,
        tags=tags,
        dataset_input=ds_spec,
    )


def log_snapshot(
    snapshot: str | UUID | Mapping[str, Any],
    *,
    client: Client | None = None,
    pin_members: bool = True,
    label: str = "dataset_snapshot",
    context: str | None = "training",
    run_id: str | None = None,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
    tracking_client: Any = None,
) -> LoggedLineageReceipt:
    """Record an immutable dataset Snapshot (and member SHA-256s when `client` is given)."""
    clean_label = _validate_label(label)
    if isinstance(snapshot, (str, UUID)):
        if client is None:
            raise TensorusError("Provide an authenticated Client when passing a snapshot UUID")
        raw = client.get_snapshot(str(UUID(str(snapshot))))
    elif isinstance(snapshot, Mapping):
        raw = dict(snapshot)
        if "tensor_count" not in raw and client is not None and "id" in raw:
            raw = client.get_snapshot(str(UUID(str(raw["id"]))))
        elif "tensor_count" not in raw:
            raw = {**raw, "tensor_count": 0}
    else:
        raise TensorusError("snapshot must be a snapshot UUID or SnapshotDetail mapping")
    try:
        parsed = SnapshotDetail.model_validate(raw)
    except ValidationError as exc:
        raise StorageError("Snapshot descriptor is invalid") from exc

    member_pins: list[LoggedSnapshotMemberPin] = []
    membership_complete = False
    if client is not None and pin_members and parsed.tensor_count <= 200:
        queried = client.query(TensorQuery(snapshot_id=parsed.id, limit=200, offset=0))
        items = queried.get("items", [])
        if len(items) == parsed.tensor_count:
            member_pins = [
                LoggedSnapshotMemberPin(
                    name=str(item["name"]),
                    version_id=UUID(str(item["id"])),
                    dtype=DType(str(item["dtype"])),
                    shape=list(item["shape"]),
                    nbytes=int(item["nbytes"]),
                    checksum=str(item["checksum"]),
                )
                for item in sorted(items, key=lambda x: str(x["name"]))
            ]
            membership_complete = True

    pin = LoggedSnapshotPin(
        snapshot_id=parsed.id,
        tenant_id=parsed.tenant_id,
        dataset_id=parsed.dataset_id,
        name=parsed.name,
        tensor_count=parsed.tensor_count,
        created_at=parsed.created_at,
        membership_complete=membership_complete,
        members=member_pins,
    )
    manifest = _build_manifest("snapshot", clean_label, "snapshot", pin)
    prefix = f"tensorus.{clean_label}"
    params = {
        f"{prefix}.snapshot_id": str(parsed.id),
        f"{prefix}.name": parsed.name,
        f"{prefix}.tensor_count": str(parsed.tensor_count),
    }
    tags = {
        "tensorus.lineage.format": LINEAGE_FORMAT,
        f"{prefix}.kind": "snapshot",
        f"{prefix}.manifest_sha256": manifest.manifest_sha256,
    }
    ds_spec = (
        _infer_dataset_input_spec(parsed.model_dump(mode="json"), context=context)
        if context is not None
        else None
    )
    return _write_to_mlflow(
        run_id=run_id,
        tracking_client=tracking_client,
        artifact_dir=artifact_dir,
        manifest=manifest,
        params=params,
        tags=tags,
        dataset_input=ds_spec,
    )


def log_validity_binding(
    binding: str | UUID | Mapping[str, Any],
    *,
    client: Client | None = None,
    label: str = "validity",
    context: str | None = "training",
    run_id: str | None = None,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
    tracking_client: Any = None,
) -> LoggedLineageReceipt:
    """Record a pinned ValidityBinding (data + boolean mask hashes) in an MLflow run."""
    clean_label = _validate_label(label)
    if isinstance(binding, (str, UUID)):
        if client is None:
            raise TensorusError("Provide an authenticated Client when passing a binding UUID")
        raw = client.validity_binding(str(UUID(str(binding))))
    elif isinstance(binding, Mapping):
        raw = dict(binding)
    else:
        raise TensorusError("binding must be a binding UUID or ValidityBinding mapping")
    try:
        parsed = ValidityBinding.model_validate(raw)
    except ValidationError as exc:
        raise StorageError("Validity binding descriptor is invalid") from exc
    pin = LoggedValidityPin(
        binding_id=parsed.id,
        tenant_id=parsed.tenant_id,
        descriptor_hash=parsed.descriptor_hash,
        data_version_id=parsed.data_version_id,
        data_checksum=parsed.data_checksum,
        data_dtype=parsed.data_dtype,
        mask_version_id=parsed.mask_version_id,
        mask_checksum=parsed.mask_checksum,
        shape=parsed.shape,
    )
    manifest = _build_manifest("validity_binding", clean_label, "validity_binding", pin)
    prefix = f"tensorus.{clean_label}"
    params = {
        f"{prefix}.binding_id": str(parsed.id),
        f"{prefix}.descriptor_hash": parsed.descriptor_hash,
        f"{prefix}.data_version_id": str(parsed.data_version_id),
        f"{prefix}.mask_version_id": str(parsed.mask_version_id),
    }
    tags = {
        "tensorus.lineage.format": LINEAGE_FORMAT,
        f"{prefix}.kind": "validity_binding",
        f"{prefix}.manifest_sha256": manifest.manifest_sha256,
    }
    ds_spec = (
        _infer_dataset_input_spec(parsed.model_dump(mode="json"), context=context)
        if context is not None
        else None
    )
    return _write_to_mlflow(
        run_id=run_id,
        tracking_client=tracking_client,
        artifact_dir=artifact_dir,
        manifest=manifest,
        params=params,
        tags=tags,
        dataset_input=ds_spec,
    )


def log_fitted_state(
    state: str | UUID | Mapping[str, Any],
    *,
    client: Client | None = None,
    label: str = "scaler",
    run_id: str | None = None,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
    tracking_client: Any = None,
) -> LoggedLineageReceipt:
    """Record a leak-free FittedPreprocessingState (`state_checksum` + binding) in MLflow."""
    clean_label = _validate_label(label)
    if isinstance(state, (str, UUID)):
        if client is None:
            raise TensorusError("Provide an authenticated Client when passing a state version UUID")
        raw = client.fitted_preprocessing_state(str(UUID(str(state))))
    elif isinstance(state, Mapping):
        raw = dict(state)
    else:
        raise TensorusError(
            "state must be a state version UUID or FittedPreprocessingState mapping"
        )
    try:
        parsed = FittedPreprocessingState.model_validate(raw)
    except ValidationError as exc:
        raise StorageError("Fitted preprocessing state descriptor is invalid") from exc
    pin = LoggedFittedStatePin(
        state_version_id=parsed.state_version_id,
        state_checksum=parsed.state_checksum,
        state_shape=parsed.state_shape,
        algorithm=parsed.algorithm,
        algorithm_version=parsed.algorithm_version,
        fit_plan_id=parsed.fit_plan_id,
        fit_plan_hash=parsed.fit_plan_hash,
        training_binding_id=parsed.training_binding.id,
        training_binding_hash=parsed.training_binding.descriptor_hash,
    )
    manifest = _build_manifest("fitted_preprocessing_state", clean_label, "fitted_state", pin)
    prefix = f"tensorus.{clean_label}"
    params = {
        f"{prefix}.state_version_id": str(parsed.state_version_id),
        f"{prefix}.state_checksum": parsed.state_checksum,
        f"{prefix}.algorithm": f"{parsed.algorithm}@v{parsed.algorithm_version}",
        f"{prefix}.fit_plan_hash": parsed.fit_plan_hash,
        f"{prefix}.training_binding_hash": parsed.training_binding.descriptor_hash,
    }
    tags = {
        "tensorus.lineage.format": LINEAGE_FORMAT,
        f"{prefix}.kind": "fitted_preprocessing_state",
        f"{prefix}.manifest_sha256": manifest.manifest_sha256,
    }
    return _write_to_mlflow(
        run_id=run_id,
        tracking_client=tracking_client,
        artifact_dir=artifact_dir,
        manifest=manifest,
        params=params,
        tags=tags,
    )


def log_quality_release(
    release: str | UUID | Mapping[str, Any],
    *,
    client: Client | None = None,
    require_approved: bool = True,
    label: str = "quality_release",
    context: str | None = "training",
    run_id: str | None = None,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
    tracking_client: Any = None,
) -> LoggedLineageReceipt:
    """Record a single-dataset Quality Release (`report_hash` + member SHA-256s) in MLflow."""
    clean_label = _validate_label(label)
    if isinstance(release, (str, UUID)):
        if client is None:
            raise TensorusError("Provide an authenticated Client when passing a release UUID")
        raw = client.release(str(UUID(str(release))))
    elif isinstance(release, Mapping):
        raw = dict(release)
    else:
        raise TensorusError("release must be a release UUID or Quality Release mapping")
    try:
        parsed = Release.model_validate(raw)
    except ValidationError as exc:
        raise StorageError("Quality release descriptor is invalid") from exc
    if require_approved and parsed.status != "approved":
        raise Conflict(
            "Quality release must be approved before logging (or pass require_approved=False)"
        )
    if len(parsed.report.members) > 256:
        raise LimitExceeded("Quality release exceeds the 256-member MLflow manifest limit")
    members = [
        LoggedQualityMemberPin(
            name=m.name,
            version_id=m.version_id,
            dtype=m.dtype,
            shape=m.shape,
            nbytes=m.nbytes,
            checksum=m.checksum,
        )
        for m in parsed.report.members
    ]
    pin = LoggedQualityReleasePin(
        release_id=parsed.id,
        tenant_id=parsed.tenant_id,
        dataset_id=parsed.dataset_id,
        dataset=parsed.report.dataset,
        release_name=parsed.report.release_name,
        snapshot_id=parsed.snapshot_id,
        contract_revision=parsed.contract_revision,
        contract_hash=parsed.report.contract_hash,
        status=parsed.status,
        report_hash=parsed.report_hash,
        members=members,
    )
    manifest = _build_manifest("quality_release", clean_label, "quality_release", pin)
    prefix = f"tensorus.{clean_label}"
    params = {
        f"{prefix}.release_id": str(parsed.id),
        f"{prefix}.dataset": parsed.dataset,
        f"{prefix}.snapshot_id": str(parsed.snapshot_id),
        f"{prefix}.status": parsed.status,
        f"{prefix}.report_hash": parsed.report_hash,
    }
    tags = {
        "tensorus.lineage.format": LINEAGE_FORMAT,
        f"{prefix}.kind": "quality_release",
        f"{prefix}.manifest_sha256": manifest.manifest_sha256,
    }
    ds_spec = (
        _infer_dataset_input_spec(parsed.model_dump(mode="json"), context=context)
        if context is not None
        else None
    )
    return _write_to_mlflow(
        run_id=run_id,
        tracking_client=tracking_client,
        artifact_dir=artifact_dir,
        manifest=manifest,
        params=params,
        tags=tags,
        dataset_input=ds_spec,
    )


def log_product_release(
    release: str | UUID | Mapping[str, Any],
    *,
    client: Client | None = None,
    require_approved: bool = True,
    label: str = "product_release",
    context: str | None = "training",
    run_id: str | None = None,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
    tracking_client: Any = None,
) -> LoggedLineageReceipt:
    """Record a multi-role ProductRelease (`report_hash` + per-role SHA-256 pins) in MLflow."""
    clean_label = _validate_label(label)
    if isinstance(release, (str, UUID)):
        if client is None:
            raise TensorusError(
                "Provide an authenticated Client when passing a product release UUID"
            )
        raw = client.product_release(str(UUID(str(release))))
    elif isinstance(release, Mapping):
        raw = dict(release)
    else:
        raise TensorusError("release must be a release UUID or ProductRelease mapping")
    try:
        parsed = ProductRelease.model_validate(raw)
    except ValidationError as exc:
        raise StorageError("Product release descriptor is invalid") from exc
    if require_approved and parsed.status != "approved":
        raise Conflict(
            "Product release must be approved before logging (or pass require_approved=False)"
        )
    members = [
        LoggedProductRolePin(
            role=m.role,
            version_id=m.version_id,
            checksum=m.checksum,
            nbytes=m.nbytes,
            contract_hash=m.contract_hash,
        )
        for m in parsed.report.members
    ]
    pin = LoggedProductReleasePin(
        release_id=parsed.id,
        tenant_id=parsed.tenant_id,
        product_id=parsed.product_id,
        product_name=parsed.product_name,
        product_revision=parsed.product_revision,
        release_name=parsed.name,
        status=parsed.status,
        report_hash=parsed.report_hash,
        descriptor_hash=parsed.request.descriptor_hash,
        members=members,
    )
    manifest = _build_manifest("product_release", clean_label, "product_release", pin)
    prefix = f"tensorus.{clean_label}"
    params = {
        f"{prefix}.release_id": str(parsed.id),
        f"{prefix}.product": f"{parsed.product_name}@r{parsed.product_revision}",
        f"{prefix}.status": parsed.status,
        f"{prefix}.report_hash": parsed.report_hash,
        f"{prefix}.roles": ",".join(m.role for m in members),
    }
    tags = {
        "tensorus.lineage.format": LINEAGE_FORMAT,
        f"{prefix}.kind": "product_release",
        f"{prefix}.manifest_sha256": manifest.manifest_sha256,
    }
    ds_spec = (
        _infer_dataset_input_spec(parsed.model_dump(mode="json"), context=context)
        if context is not None
        else None
    )
    return _write_to_mlflow(
        run_id=run_id,
        tracking_client=tracking_client,
        artifact_dir=artifact_dir,
        manifest=manifest,
        params=params,
        tags=tags,
        dataset_input=ds_spec,
    )


def log_reference_evaluation(
    evaluation: str | UUID | Mapping[str, Any],
    *,
    client: Client | None = None,
    require_approved: bool = True,
    label: str = "eval_gate",
    run_id: str | None = None,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
    tracking_client: Any = None,
) -> LoggedLineageReceipt:
    """Record a pairwise ReferenceEvaluation in MLflow, including scalar evaluation metrics."""
    clean_label = _validate_label(label)
    if isinstance(evaluation, (str, UUID)):
        if client is None:
            raise TensorusError(
                "Provide an authenticated Client when passing a reference evaluation UUID"
            )
        raw = client.reference_evaluation(str(UUID(str(evaluation))))
    elif isinstance(evaluation, Mapping):
        raw = dict(evaluation)
    else:
        raise TensorusError("evaluation must be an evaluation UUID or ReferenceEvaluation mapping")
    try:
        parsed = ReferenceEvaluation.model_validate(raw)
    except ValidationError as exc:
        raise StorageError("Reference evaluation descriptor is invalid") from exc
    if require_approved and parsed.status != "approved":
        raise Conflict(
            "Reference evaluation must be approved before logging (or pass require_approved=False)"
        )
    ref_ev = parsed.report.result.reference
    inter = ref_ev.validity_intersection if ref_ev is not None else None
    if ref_ev is None or inter is None:
        raise StorageError("Reference evaluation is missing pairwise validity evidence")

    tol_violations: int | None = None
    if ref_ev.tolerance is not None and ref_ev.tolerance.violations is not None:
        tol_violations = int(ref_ev.tolerance.violations)

    mae_val: float | None = None
    if (
        ref_ev.mean_absolute_error is not None
        and ref_ev.mean_absolute_error.numerator is not None
        and ref_ev.mean_absolute_error.denominator is not None
    ):
        mae_val = float(
            Decimal(ref_ev.mean_absolute_error.numerator)
            / Decimal(ref_ev.mean_absolute_error.denominator)
        )

    pin = LoggedReferenceEvaluationPin(
        evaluation_id=parsed.id,
        tenant_id=parsed.tenant_id,
        name=parsed.name,
        status=parsed.status,
        passed=parsed.report.passed,
        report_hash=parsed.report_hash,
        output_version_id=parsed.report.result.version_id,
        output_checksum=parsed.report.result.checksum,
        reference_version_id=ref_ev.version_id,
        reference_checksum=ref_ev.checksum,
        output_binding_id=inter.output_mask.validity_binding.id,
        output_binding_hash=inter.output_mask.validity_binding.descriptor_hash,
        reference_binding_id=inter.reference_mask.validity_binding.id,
        reference_binding_hash=inter.reference_mask.validity_binding.descriptor_hash,
        included_count=inter.included_count,
        excluded_count=inter.excluded_count,
        tolerance_violations=tol_violations,
        mae_value=mae_val,
    )
    manifest = _build_manifest("reference_evaluation", clean_label, "reference_evaluation", pin)
    prefix = f"tensorus.{clean_label}"
    params = {
        f"{prefix}.evaluation_id": str(parsed.id),
        f"{prefix}.name": parsed.name,
        f"{prefix}.status": parsed.status,
        f"{prefix}.report_hash": parsed.report_hash,
        f"{prefix}.output_version_id": str(parsed.report.result.version_id),
        f"{prefix}.reference_version_id": str(ref_ev.version_id),
    }
    tags = {
        "tensorus.lineage.format": LINEAGE_FORMAT,
        f"{prefix}.kind": "reference_evaluation",
        f"{prefix}.manifest_sha256": manifest.manifest_sha256,
    }
    metrics: dict[str, float] = {
        f"{prefix}.passed": 1.0 if parsed.report.passed else 0.0,
        f"{prefix}.included_count": float(inter.included_count),
        f"{prefix}.excluded_count": float(inter.excluded_count),
    }
    if tol_violations is not None:
        metrics[f"{prefix}.tolerance_violations"] = float(tol_violations)
    if mae_val is not None:
        metrics[f"{prefix}.mae"] = mae_val

    return _write_to_mlflow(
        run_id=run_id,
        tracking_client=tracking_client,
        artifact_dir=artifact_dir,
        manifest=manifest,
        params=params,
        tags=tags,
        metrics=metrics,
    )


def _download_verified_array(
    client: Client, expected_id: UUID, expected_checksum: str
) -> tuple[Array, dict[str, Any]]:
    if type(client.max_tensor_bytes) is not int or client.max_tensor_bytes <= 0:
        raise ValueError("client.max_tensor_bytes must be a positive integer")
    record = client.version(str(expected_id))
    try:
        parsed = TensorVersion.model_validate(record)
    except ValidationError as exc:
        raise StorageError("Server returned an invalid tensor version descriptor") from exc
    if parsed.id != expected_id:
        raise StorageError("Server returned a different tensor version UUID")
    if parsed.deleted_at is not None:
        raise Conflict("Logged tensor version has been retired or deleted on the server")
    if parsed.checksum != expected_checksum:
        raise Conflict("Server tensor checksum does not match the MLflow-pinned SHA-256 digest")
    dumped = parsed.model_dump(mode="json")
    if descriptor_nbytes(dumped) > client.max_tensor_bytes:
        raise LimitExceeded("Logged tensor exceeds client.max_tensor_bytes")
    values = client._get_verified_npy(dumped)
    if (
        list(values.shape) != dumped["shape"]
        or values.dtype.name != dumped["dtype"]
        or array_checksum(values) != expected_checksum
    ):
        raise StorageError("Downloaded tensor payload failed end-to-end SHA-256 verification")
    return values, dumped


def load_logged_tensor(
    client: Client,
    run_id: str | None = None,
    label: str = "input",
    *,
    role: str | None = None,
    tensor: str | None = None,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
    tracking_client: Any = None,
) -> VerifiedLoggedTensor:
    """Reload and SHA-256-verify a logged tensor, validity pair, snapshot, or release role."""
    manifest = read_logged_manifest(
        run_id, label, artifact_dir=artifact_dir, tracking_client=tracking_client
    )
    kind = manifest["kind"]
    if kind == "tensor_version":
        if role is not None or tensor is not None:
            raise TensorusError(
                "role=... / tensor=... are not valid for a single tensor_version manifest"
            )
        pin = LoggedVersionPin.model_validate(manifest["tensor_version"])
        values, record = _download_verified_array(client, pin.version_id, pin.checksum)
        if (
            UUID(record["tenant_id"]) != pin.tenant_id
            or UUID(record["dataset_id"]) != pin.dataset_id
            or record["dtype"] != pin.dtype.value
            or record["shape"] != pin.shape
        ):
            raise Conflict("Server tensor metadata does not match the MLflow-pinned descriptor")
        return VerifiedLoggedTensor(
            values=values, mask=None, version=record, binding=None, manifest=manifest
        )

    if kind == "validity_binding":
        if role is not None or tensor is not None:
            raise TensorusError(
                "role=... / tensor=... are not valid for a validity_binding manifest"
            )
        vpin = LoggedValidityPin.model_validate(manifest["validity_binding"])
        live_raw = client.validity_binding(str(vpin.binding_id))
        try:
            live_binding = ValidityBinding.model_validate(live_raw)
        except ValidationError as exc:
            raise StorageError("Server returned an invalid validity binding") from exc
        if (
            live_binding.id != vpin.binding_id
            or live_binding.descriptor_hash != vpin.descriptor_hash
            or live_binding.data_version_id != vpin.data_version_id
            or live_binding.data_checksum != vpin.data_checksum
            or live_binding.mask_version_id != vpin.mask_version_id
            or live_binding.mask_checksum != vpin.mask_checksum
        ):
            raise Conflict(
                "Server validity binding does not match the MLflow-pinned descriptor_hash"
            )
        values, data_rec = _download_verified_array(
            client, vpin.data_version_id, vpin.data_checksum
        )
        mask, mask_rec = _download_verified_array(client, vpin.mask_version_id, vpin.mask_checksum)
        if (
            _descriptor_hash(
                data_rec,
                mask_rec,
                live_binding.origin.model_dump(mode="json"),
                str(live_binding.id),
                live_binding.created_at,
                live_binding.created_by,
            )
            != vpin.descriptor_hash
        ):
            raise StorageError("Recomputed validity descriptor_hash did not match the MLflow pin")
        return VerifiedLoggedTensor(
            values=values,
            mask=mask,
            version=data_rec,
            binding=live_binding.model_dump(mode="json"),
            manifest=manifest,
        )

    if kind == "snapshot":
        target_name = tensor or role
        if not target_name:
            raise TensorusError("Specify tensor=... when loading a member from a logged snapshot")
        verify_logged_snapshot(
            client, run_id, label, artifact_dir=artifact_dir, tracking_client=tracking_client
        )
        spin = LoggedSnapshotPin.model_validate(manifest["snapshot"])
        if spin.membership_Complete:
            matched_sm = next((m for m in spin.members if m.name == target_name), None)
            if matched_sm is None:
                raise TensorusError(
                    f"Tensor {target_name!r} is not in logged snapshot {spin.name!r}"
                )
            values, record = _download_verified_array(
                client, matched_sm.version_id, matched_sm.checksum
            )
            return VerifiedLoggedTensor(
                values=values, mask=None, version=record, binding=None, manifest=manifest
            )
        queried = client.query(
            TensorQuery(snapshot_id=spin.snapshot_id, text=target_name, limit=200)
        )
        exact = next((i for i in queried.get("items", []) if i["name"] == target_name), None)
        if exact is None:
            raise TensorusError(f"Tensor {target_name!r} is not in snapshot {spin.name!r}")
        values, record = _download_verified_array(
            client, UUID(str(exact["id"])), str(exact["checksum"])
        )
        return VerifiedLoggedTensor(
            values=values, mask=None, version=record, binding=None, manifest=manifest
        )

    if kind == "quality_release":
        target_name = tensor or role
        if not target_name:
            raise TensorusError(
                "Specify tensor=... when loading a member from a logged quality_release"
            )
        verify_logged_quality_release(
            client, run_id, label, artifact_dir=artifact_dir, tracking_client=tracking_client
        )
        qpin = LoggedQualityReleasePin.model_validate(manifest["quality_release"])
        matched_qm = next((m for m in qpin.members if m.name == target_name), None)
        if matched_qm is None:
            raise TensorusError(
                f"Tensor {target_name!r} is not in logged quality release {qpin.release_name!r}"
            )
        values, record = _download_verified_array(
            client, matched_qm.version_id, matched_qm.checksum
        )
        return VerifiedLoggedTensor(
            values=values, mask=None, version=record, binding=None, manifest=manifest
        )

    if kind == "product_release":
        target_role = role or tensor
        if not target_role:
            raise TensorusError(
                "Specify role=... when loading a tensor from a logged product_release"
            )
        verify_logged_product_release(
            client, run_id, label, artifact_dir=artifact_dir, tracking_client=tracking_client
        )
        rpin = LoggedProductReleasePin.model_validate(manifest["product_release"])
        matched = next((m for m in rpin.members if m.role == target_role), None)
        if matched is None:
            raise TensorusError(
                f"Role {target_role!r} is not in logged product release {rpin.product_name!r}"
            )
        values, record = _download_verified_array(client, matched.version_id, matched.checksum)
        return VerifiedLoggedTensor(
            values=values, mask=None, version=record, binding=None, manifest=manifest
        )

    raise TensorusError(f"Cannot load tensor array directly from manifest kind {kind!r}")


def verify_logged_snapshot(
    client: Client,
    run_id: str | None = None,
    label: str = "dataset_snapshot",
    *,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
    tracking_client: Any = None,
) -> dict[str, Any]:
    """Verify that a live dataset Snapshot and its pinned members match the MLflow record."""
    manifest = read_logged_manifest(
        run_id, label, artifact_dir=artifact_dir, tracking_client=tracking_client
    )
    if manifest["kind"] != "snapshot":
        raise TensorusError(f"Expected snapshot manifest, got {manifest['kind']!r}")
    pin = LoggedSnapshotPin.model_validate(manifest["snapshot"])
    live_raw = client.get_snapshot(str(pin.snapshot_id))
    try:
        live = SnapshotDetail.model_validate(live_raw)
    except ValidationError as exc:
        raise StorageError("Server returned an invalid snapshot descriptor") from exc
    if (
        live.id != pin.snapshot_id
        or live.tenant_id != pin.tenant_id
        or live.dataset_id != pin.dataset_id
        or live.name != pin.name
        or live.tensor_count != pin.tensor_count
        or live.created_at != pin.created_at
    ):
        raise Conflict("Live snapshot descriptor does not match the MLflow-pinned snapshot")
    if pin.membership_Complete:
        queried = client.query(TensorQuery(snapshot_id=pin.snapshot_id, limit=200, offset=0))
        items = queried.get("items", [])
        live_members = [
            LoggedSnapshotMemberPin(
                name=str(item["name"]),
                version_id=UUID(str(item["id"])),
                dtype=DType(str(item["dtype"])),
                shape=list(item["shape"]),
                nbytes=int(item["nbytes"]),
                checksum=str(item["checksum"]),
            )
            for item in sorted(items, key=lambda x: str(x["name"]))
        ]
        if live_members != pin.members:
            raise Conflict("Live snapshot membership does not match the MLflow-pinned SHA-256 list")
    return live.model_dump(mode="json")


def verify_logged_fitted_state(
    client: Client,
    run_id: str | None = None,
    label: str = "scaler",
    *,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
    tracking_client: Any = None,
) -> dict[str, Any]:
    """Reconcile a logged FittedPreprocessingState against live server state and SHA-256 payload."""
    manifest = read_logged_manifest(
        run_id, label, artifact_dir=artifact_dir, tracking_client=tracking_client
    )
    if manifest["kind"] != "fitted_preprocessing_state":
        raise TensorusError(
            f"Expected fitted_preprocessing_state manifest, got {manifest['kind']!r}"
        )
    pin = LoggedFittedStatePin.model_validate(manifest["fitted_state"])
    live_raw = client.fitted_preprocessing_state(str(pin.state_version_id))
    try:
        live = FittedPreprocessingState.model_validate(live_raw)
    except ValidationError as exc:
        raise StorageError("Server returned an invalid fitted preprocessing state") from exc
    if (
        live.state_version_id != pin.state_version_id
        or live.state_checksum != pin.state_checksum
        or live.state_shape != pin.state_shape
        or live.algorithm != pin.algorithm
        or live.algorithm_version != pin.algorithm_version
        or live.fit_plan_id != pin.fit_plan_id
        or live.fit_plan_hash != pin.fit_plan_hash
        or live.training_binding.id != pin.training_binding_id
        or live.training_binding.descriptor_hash != pin.training_binding_hash
    ):
        raise Conflict("Live fitted preprocessing state does not match the MLflow-pinned hashes")
    _download_verified_array(client, pin.state_version_id, pin.state_checksum)
    return live.model_dump(mode="json")


def verify_logged_quality_release(
    client: Client,
    run_id: str | None = None,
    label: str = "quality_release",
    *,
    require_approved: bool = True,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
    tracking_client: Any = None,
) -> dict[str, Any]:
    """Verify a live Quality Release against member UUIDs, SHA-256s, and report_hash in MLflow."""
    manifest = read_logged_manifest(
        run_id, label, artifact_dir=artifact_dir, tracking_client=tracking_client
    )
    if manifest["kind"] != "quality_release":
        raise TensorusError(f"Expected quality_release manifest, got {manifest['kind']!r}")
    pin = LoggedQualityReleasePin.model_validate(manifest["quality_release"])
    live_raw = client.release(str(pin.release_id))
    try:
        live = Release.model_validate(live_raw)
    except ValidationError as exc:
        raise StorageError("Server returned an invalid quality release") from exc
    if require_approved and live.status != "approved":
        raise Conflict("Live quality release is not in approved status")
    live_members = [
        LoggedQualityMemberPin(
            name=m.name,
            version_id=m.version_id,
            dtype=m.dtype,
            shape=m.shape,
            nbytes=m.nbytes,
            checksum=m.checksum,
        )
        for m in live.report.members
    ]
    if (
        live.id != pin.release_id
        or live.tenant_id != pin.tenant_id
        or live.dataset_id != pin.dataset_id
        or live.snapshot_id != pin.snapshot_id
        or live.contract_revision != pin.contract_revision
        or live.report.contract_hash != pin.contract_hash
        or live.report_hash != pin.report_hash
        or live_members != pin.members
    ):
        raise Conflict(
            "Live quality release does not match the MLflow-pinned report_hash or members"
        )
    return live.model_dump(mode="json")


def verify_logged_product_release(
    client: Client,
    run_id: str | None = None,
    label: str = "product_release",
    *,
    require_approved: bool = True,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
    tracking_client: Any = None,
) -> dict[str, Any]:
    """Verify a live ProductRelease against every role UUID, SHA-256, and report_hash in MLflow."""
    manifest = read_logged_manifest(
        run_id, label, artifact_dir=artifact_dir, tracking_client=tracking_client
    )
    if manifest["kind"] != "product_release":
        raise TensorusError(f"Expected product_release manifest, got {manifest['kind']!r}")
    pin = LoggedProductReleasePin.model_validate(manifest["product_release"])
    live_raw = client.product_release(str(pin.release_id))
    try:
        live = ProductRelease.model_validate(live_raw)
    except ValidationError as exc:
        raise StorageError("Server returned an invalid product release") from exc
    if require_approved and live.status != "approved":
        raise Conflict("Live product release is not in approved status")
    live_roles = [
        LoggedProductRolePin(
            role=m.role,
            version_id=m.version_id,
            checksum=m.checksum,
            nbytes=m.nbytes,
            contract_hash=m.contract_hash,
        )
        for m in live.report.members
    ]
    if (
        live.id != pin.release_id
        or live.tenant_id != pin.tenant_id
        or live.product_id != pin.product_id
        or live.product_name != pin.product_name
        or live.product_revision != pin.product_revision
        or live.report_hash != pin.report_hash
        or live.request.descriptor_hash != pin.descriptor_hash
        or live_roles != pin.members
    ):
        raise Conflict(
            "Live product release does not match the MLflow-pinned report_hash or members"
        )
    return live.model_dump(mode="json")


def verify_logged_reference_evaluation(
    client: Client,
    run_id: str | None = None,
    label: str = "eval_gate",
    *,
    require_approved: bool = True,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
    tracking_client: Any = None,
) -> dict[str, Any]:
    """Verify that a live ReferenceEvaluation still exists (not retired/withdrawn) and matches."""
    manifest = read_logged_manifest(
        run_id, label, artifact_dir=artifact_dir, tracking_client=tracking_client
    )
    if manifest["kind"] != "reference_evaluation":
        raise TensorusError(f"Expected reference_evaluation manifest, got {manifest['kind']!r}")
    pin = LoggedReferenceEvaluationPin.model_validate(manifest["reference_evaluation"])
    live_raw = client.reference_evaluation(str(pin.evaluation_id))
    try:
        live = ReferenceEvaluation.model_validate(live_raw)
    except ValidationError as exc:
        raise StorageError("Server returned an invalid reference evaluation") from exc
    if require_approved and live.status != "approved":
        raise Conflict("Live reference evaluation is not in approved status")
    ref_ev = live.report.result.reference
    if (
        ref_ev is None
        or live.id != pin.evaluation_id
        or live.tenant_id != pin.tenant_id
        or live.report_hash != pin.report_hash
        or live.report.passed != pin.passed
        or live.report.result.version_id != pin.output_version_id
        or live.report.result.checksum != pin.output_checksum
        or ref_ev.version_id != pin.reference_version_id
        or ref_ev.checksum != pin.reference_checksum
    ):
        raise Conflict(
            "Live reference evaluation does not match the MLflow-pinned report_hash or versions"
        )
    return live.model_dump(mode="json")
