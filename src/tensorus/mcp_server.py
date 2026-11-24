import base64
import binascii
import hashlib
import io
import logging
import math
import os
import sys
from typing import Annotated, Any, BinaryIO, Literal
from uuid import UUID

from pydantic import Field, JsonValue, StrictBool, StrictInt

from tensorus.arrays import canonical_json
from tensorus.client import Client
from tensorus.contracts import ReleaseHash
from tensorus.errors import Forbidden, LimitExceeded, TensorusError
from tensorus.evaluation import (
    EvaluationRequest,
    TensorEvaluationRequest,
    ValidityStatisticsRequest,
)
from tensorus.history_payload_restore import HistoryPayloadRestoreRequest
from tensorus.models import (
    AxisAlignmentRequest,
    CoordinateRange,
    DType,
    GoalRequest,
    Name,
    NpyWriteCondition,
    PlanDefinition,
    PlanProposerAuthority,
    PlanStatus,
    Role,
    SemanticFilter,
    TensorQuery,
    UploadCreate,
    WorkflowRun,
)
from tensorus.observation_scope import ObservationScopeVersion
from tensorus.operations import ImportedFittedStateSelection
from tensorus.plan_outcomes import PlanOutcomeVersion
from tensorus.product_bundle_restore import ProductBundleRestoreRequest
from tensorus.product_similarity_retirements import ProductSimilarityRetirementReviewRequest
from tensorus.product_source_evaluations import ProductSourceEvaluationReviewRequest
from tensorus.recovery_verification import RecoveryVerificationRequest
from tensorus.reference_evaluations import ReferenceDispositionRequest
from tensorus.release_portability import (
    MAX_PORTABLE_PLAN_SELECTIONS,
    MAX_PORTABLE_VALIDITY_SELECTIONS,
    ImportedHistoryIndexDiagnosticRequest,
    ImportedHistoryIndexLookupRequest,
    ImportedHistoryIndexReviewRequest,
    ImportedHistoryInspectionRequest,
    ImportedHistoryScanRequest,
    PortableHistoryExport,
)
from tensorus.retirement_archive import RetirementArchiveSelection
from tensorus.serialization import dump_npy
from tensorus.validity import (
    FittedPreprocessingApply,
    FittedPreprocessingFit,
    ValidityBindingCreate,
    ValidityCoordinateWindow,
    ValidityTransform,
    ValidityTransformCompletion,
)
from tensorus.validity_similarity import ValiditySnapshotSimilarityRequest

MAX_MCP_NPY_RANGE_BYTES = 1024**2
DEFAULT_MCP_TENSOR_BYTES = 64 * 1024**2
MAX_MCP_TENSOR_BYTES = 2 * 1024**3
MAX_UPLOAD_CHUNK_BYTES = 8 * 1024**2
MAX_UPLOAD_BASE64_CHARS = 4 * ((MAX_UPLOAD_CHUNK_BYTES + 2) // 3)
MAX_MCP_STDIN_MESSAGE_BYTES = 12 * 1024**2
MAX_MCP_PRODUCT_BUNDLE_BYTES = 4 * 1024**2
MAX_MCP_PRODUCT_BUNDLE_BASE64_CHARS = 4 * ((MAX_MCP_PRODUCT_BUNDLE_BYTES + 2) // 3)
MAX_MCP_HISTORY_PAYLOAD_BYTES = 4 * 1024**2
MAX_MCP_HISTORY_PAYLOAD_BASE64_CHARS = 4 * ((MAX_MCP_HISTORY_PAYLOAD_BYTES + 2) // 3)
logger = logging.getLogger("tensorus.mcp")


class _StdioFrameError(RuntimeError):
    pass


class _BoundedStdioInput(io.StringIO):
    def __init__(self, stream: BinaryIO, limit: int) -> None:
        super().__init__()
        self._stream = stream
        self._limit = limit

    def readable(self) -> bool:
        return True

    # Mypy's _IOBase stub also declares a bytes-returning readline for text streams.
    def readline(self, size: int | None = -1) -> str:  # type: ignore[override]
        if size not in (-1, None):
            raise io.UnsupportedOperation("Sized stdio reads are not supported")
        frame = self._stream.readline(self._limit + 1)
        if not frame:
            return ""
        terminated = frame.endswith(b"\n")
        if len(frame) - int(terminated) > self._limit:
            logger.error("MCP stdio input frame exceeds %d bytes", self._limit)
            raise _StdioFrameError("MCP stdio input frame exceeds the configured limit")
        if not terminated:
            logger.error("MCP stdio input ended before the newline delimiter")
            raise _StdioFrameError("MCP stdio input frame is not newline-terminated")
        try:
            return frame.decode("utf-8")
        except UnicodeDecodeError:
            logger.error("MCP stdio input frame is not valid UTF-8")
            raise _StdioFrameError("MCP stdio input frame is not valid UTF-8") from None


def _validate_metadata(metadata: Any) -> None:
    nodes = 0
    characters = 0

    def inspect(value: Any, depth: int) -> None:
        nonlocal nodes, characters
        nodes += 1
        if nodes > 2048 or depth > 8:
            raise ValueError("Metadata exceeds node or depth budget")
        if isinstance(value, str):
            characters += len(value)
            if characters > 8192:
                raise ValueError("Metadata exceeds size budget")
        elif isinstance(value, dict):
            if len(value) > 2048:
                raise ValueError("Metadata exceeds node budget")
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValueError("Metadata keys must be strings")
                inspect(key, depth + 1)
                inspect(child, depth + 1)
        elif isinstance(value, list):
            if len(value) > 2048:
                raise ValueError("Metadata exceeds node budget")
            for child in value:
                inspect(child, depth + 1)
        elif (
            value is None
            or type(value) in (bool, int)
            or (type(value) is float and math.isfinite(value))
        ):
            return
        else:
            raise ValueError("Metadata must contain finite JSON values")

    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a JSON object")
    inspect(metadata, 0)
    if len(canonical_json(metadata).encode("utf-8")) > 8192:
        raise ValueError("Metadata exceeds 8192 bytes")


def _validate_ingestion(data: Any, dtype: str, metadata: Any) -> None:
    """Check bounded JSON before Client.put can allocate a NumPy array."""
    count = 0

    def shape(value: Any, depth: int) -> tuple[int, ...]:
        nonlocal count
        if isinstance(value, list):
            if depth >= 8 or not value or len(value) > 4096:
                raise ValueError("Tensor requires nonempty arrays, rank <= 8 and <= 4096 elements")
            first = shape(value[0], depth + 1)
            for item in value[1:]:
                if shape(item, depth + 1) != first:
                    raise ValueError("Tensor values must be rectangular")
            return (len(value), *first)
        count += 1
        if count > 4096:
            raise ValueError("Tensor exceeds 4096 elements")
        if type(value) not in (int, float):
            raise ValueError("Tensor values must be JSON numbers, not booleans or strings")
        if dtype == "int64":
            if type(value) is not int or not -(2**63) <= value < 2**63:
                raise ValueError("int64 values must be integral and within int64 range")
        else:
            try:
                number = float(value)
            except OverflowError as exc:
                raise ValueError("Tensor number is out of range") from exc
            limit = 3.4028234663852886e38 if dtype == "float32" else float("inf")
            if not math.isfinite(number) or abs(number) > limit:
                raise ValueError("Tensor number is not finite or is out of range")
        return ()

    if not isinstance(data, list):
        raise ValueError("Tensor data must be a nonempty JSON array")
    shape(data, 0)
    _validate_metadata(metadata)


def _validate_ingestion_key(idempotency_key: str) -> None:
    if not (
        1 <= len(idempotency_key) <= 128
        and idempotency_key.isascii()
        and idempotency_key.isprintable()
    ):
        raise ValueError("idempotency_key must be 1..128 printable ASCII characters")


def _decode_upload_chunk(data_base64: str) -> bytes:
    if not data_base64 or len(data_base64) > MAX_UPLOAD_BASE64_CHARS:
        raise ValueError("Encoded upload chunk exceeds the bridge limit or is empty")
    try:
        data = base64.b64decode(data_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Upload chunk must be valid standard base64") from exc
    if len(data) > MAX_UPLOAD_CHUNK_BYTES:
        raise ValueError("Decoded upload chunk exceeds the bridge limit")
    if base64.b64encode(data).decode("ascii") != data_base64:
        raise ValueError("Upload chunk must use canonical padded base64")
    return data


def _decode_bounded_archive(data_base64: str, limit: int, label: str) -> bytes:
    if not data_base64 or len(data_base64) > 4 * ((limit + 2) // 3):
        raise ValueError(f"Encoded {label} exceeds the bridge limit or is empty")
    try:
        data = base64.b64decode(data_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{label.capitalize()} must be valid standard base64") from exc
    if len(data) > limit:
        raise ValueError(f"Decoded {label} exceeds the bridge limit")
    if base64.b64encode(data).decode("ascii") != data_base64:
        raise ValueError(f"{label.capitalize()} must use canonical padded base64")
    return data


def _decode_product_bundle(data_base64: str) -> bytes:
    return _decode_bounded_archive(data_base64, MAX_MCP_PRODUCT_BUNDLE_BYTES, "product bundle")


def _decode_history_payload_bundle(data_base64: str) -> bytes:
    return _decode_bounded_archive(
        data_base64, MAX_MCP_HISTORY_PAYLOAD_BYTES, "history payload bundle"
    )


def _mcp_tensor_byte_limit() -> int:
    value = os.environ.get("TENSORUS_MCP_MAX_TENSOR_BYTES")
    if value is None:
        return DEFAULT_MCP_TENSOR_BYTES
    if not value.isascii() or not value.isdecimal() or len(value) > 10:
        raise RuntimeError("TENSORUS_MCP_MAX_TENSOR_BYTES must be an integer from 1 to 2147483648")
    limit = int(value)
    if not 1 <= limit <= MAX_MCP_TENSOR_BYTES:
        raise RuntimeError("TENSORUS_MCP_MAX_TENSOR_BYTES must be an integer from 1 to 2147483648")
    return limit


def create_server(
    client: Client,
    *,
    enable_proposals: bool = False,
    enable_ingestion: bool = False,
    enable_cancellation: bool = False,
    enable_npy_download: bool = False,
    enable_resumable_ingestion: bool = False,
    enable_release_history_import: bool = False,
    enable_history_index_backfill: bool = False,
    enable_history_index_reconciliation: bool = False,
    enable_snapshot_retirement: bool = False,
    enable_retirement_archive_export: bool = False,
    enable_product_bundle_export: bool = False,
    enable_history_payload_export: bool = False,
) -> Any:
    unsupported_admin_mutations = [
        name
        for enabled, name in (
            (enable_release_history_import, "release-history import"),
            (enable_history_index_backfill, "history-index backfill"),
            (enable_history_index_reconciliation, "history-index reconciliation"),
            (enable_snapshot_retirement, "snapshot retirement"),
        )
        if enabled
    ]
    if unsupported_admin_mutations:
        raise RuntimeError(
            "MCP administrative mutations are unavailable "
            f"({', '.join(unsupported_admin_mutations)}); remove legacy flags and use "
            "HTTP or SDK operator paths"
        )

    try:
        import anyio
        from mcp.server.fastmcp import FastMCP
        from mcp.server.stdio import stdio_server
        from mcp.types import ToolAnnotations
    except ModuleNotFoundError as exc:
        if exc.name == "mcp":
            raise RuntimeError("MCP integration requires: uv sync --extra mcp") from exc
        raise

    class BoundedFastMCP(FastMCP):
        async def run_stdio_async(self) -> None:
            stdin = anyio.wrap_file(
                _BoundedStdioInput(sys.stdin.buffer, MAX_MCP_STDIN_MESSAGE_BYTES)
            )
            try:
                # FastMCP does not expose a custom stdin on its public stdio runner.
                async with stdio_server(stdin=stdin) as (read_stream, write_stream):
                    await self._mcp_server.run(
                        read_stream,
                        write_stream,
                        self._mcp_server.create_initialization_options(),
                    )
            except* _StdioFrameError:
                raise _StdioFrameError("MCP stdio input rejected; details were logged") from None

    server = BoundedFastMCP("Tensorus")
    read_only = ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    )
    proposal = ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False
    )
    mutation = ToolAnnotations(
        readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False
    )

    def require_scoped_mutator() -> None:
        identity, _ = client._checked_capabilities()
        if identity.scope is None or identity.role != Role.reader:
            raise Forbidden("MCP agent mutations require a scoped agent credential")

    @server.tool(annotations=read_only)
    def get_service_context() -> dict[str, Any]:
        """Discover the authenticated workspace, role, operations, plan schema and limits.

        Capabilities describe the HTTP service, not additional MCP tools or permission to
        execute. LLM configuration does not imply consent to disclose observations.
        """
        identity, capabilities = client._checked_capabilities()
        return {
            "identity": identity.model_dump(mode="json"),
            "capabilities": capabilities,
            "mcp_governance_boundary": {
                "plan_approval_exposed": False,
                "acl_mutation_exposed": False,
                "physical_gc_exposed": False,
                "host_filesystem_exposed": False,
                "regional_and_settlement_tools": [
                    "get_workspace_region",
                    "get_service_status",
                    "get_plan_usage_settlement",
                ],
            },
        }

    @server.tool(annotations=read_only)
    def get_workspace_region() -> dict[str, Any]:
        """Inspect workspace home-cell binding and whether mutations are authorized here."""
        return client.workspace_region()

    @server.tool(annotations=read_only)
    def get_service_status() -> dict[str, Any]:
        """Inspect public cell identity, read-only state, and operator incident runbook."""
        return client.service_status()

    @server.tool(annotations=read_only)
    def get_service_disclosures() -> dict[str, Any]:
        """Read public operator declarations, not policy enforcement or legal approval."""
        identity = client._checked_identity()
        disclosures = client.read_service_disclosures()
        if client._checked_identity() != identity:
            raise Forbidden("The credential changed during disclosure inspection")
        return disclosures

    @server.tool(annotations=read_only)
    def get_agent_proposal_budget() -> dict[str, Any]:
        """Inspect this scoped agent's proposal allowance, not execution usage or charges."""
        return client.agent_proposal_budget()

    @server.tool(annotations=read_only)
    def get_service_actor_identity() -> dict[str, Any]:
        """Inspect only this credential's provider-backed service actor association."""
        return client.service_actor_identity()

    @server.tool(annotations=read_only)
    def get_service_actor_recipient() -> dict[str, Any]:
        """Inspect only this credential's explicitly bound service actor recipient."""
        return client.service_actor_recipient()

    @server.tool(annotations=read_only)
    def get_agent_execution_budget() -> dict[str, Any]:
        """Inspect estimated worker-attempt allowance; it is not measured usage or a charge."""
        return client.agent_execution_budget()

    @server.tool(annotations=read_only)
    def get_agent_ingest_budget() -> dict[str, Any]:
        """Inspect committed direct-ingestion bytes, not attempted work, storage or charges."""
        return client.agent_ingest_budget()

    @server.tool(annotations=read_only)
    def get_agent_upload_staging() -> dict[str, Any]:
        """Inspect this scoped agent's declared open-upload reservations, not disk usage."""
        return client.agent_upload_staging()

    @server.tool(annotations=read_only)
    def get_operation_contract(operation: str) -> dict[str, Any]:
        """Inspect one advertised operation, its input roles, validity and reviewed pair routes."""
        matrix = client.operation_contracts()
        contract = matrix["operations"].get(operation)
        if contract is None:
            raise TensorusError("Operation is not present in the advertised contract matrix")
        guided = matrix["guided_validity"]["requests"]
        return {
            "format": matrix["format"],
            "operation": operation,
            "contract": contract,
            "source_identity": matrix["source_identity"],
            "semantics_policy": matrix["semantics_policy"],
            "publication": matrix["publication"],
            "limits": matrix["limits"],
            "guided_validity_requests": {
                name: pair
                for name, pair in guided.items()
                if name == operation or operation in pair.values()
            },
            "guided_validity_policy": matrix["guided_validity"]["policy"],
        }

    @server.tool(annotations=read_only)
    def get_bridge_limits() -> dict[str, Any]:
        """Report the bridge's stdio input frame limit, separate from service limits."""
        return {
            "stdio": {
                "max_inbound_message_bytes": MAX_MCP_STDIN_MESSAGE_BYTES,
                "newline_counts_toward_limit": False,
                "on_invalid_frame": "close_session",
            },
            "proposals": {
                "format": "tensorus-mcp-scoped-proposals-v1",
                "enabled": enable_proposals,
                "scoped_agent_required": True,
                "approval_in_bridge": False,
            },
            "agent_mutations": {
                "format": "tensorus-mcp-scoped-agent-mutations-v1",
                "ingestion_enabled": enable_ingestion,
                "resumable_ingestion_enabled": enable_resumable_ingestion,
                "cancellation_enabled": enable_cancellation,
                "scoped_agent_required": True,
            },
            "administrative_mutations": {
                "format": "tensorus-mcp-administrative-mutations-v1",
                "supported": False,
                "legacy_opt_ins_rejected": True,
            },
            "product_bundle": {
                "format": "tensorus-mcp-product-bundle-v1",
                "export_and_inspection_enabled": enable_product_bundle_export,
                "max_archive_bytes": MAX_MCP_PRODUCT_BUNDLE_BYTES,
                "import_or_restore_supported": False,
                "source_claims_authenticated": False,
            },
            "release_history_payload_bundle": {
                "format": "tensorus-mcp-release-history-payload-bundle-v1",
                "export_and_inspection_enabled": enable_history_payload_export,
                "max_archive_bytes": MAX_MCP_HISTORY_PAYLOAD_BYTES,
                "copy_or_restore_supported": False,
                "source_claims_authenticated": False,
            },
            "bundle_restore_progress": {
                "format": "tensorus-mcp-bundle-restore-progress-v2",
                "product_enabled": enable_product_bundle_export,
                "history_enabled": enable_history_payload_export,
                "requires_unscoped_writer": True,
                "review_actor_must_match": True,
                "stale_review_requires_original": True,
                "policy_change_blocks_continuation": True,
                "read_only": True,
                "point_in_time_atomic": False,
                "retry_guaranteed": False,
            },
            "bundle_restore_head_preflight": {
                "format": "tensorus-bundle-restore-head-preflight-v1",
                "product_enabled": enable_product_bundle_export,
                "history_enabled": enable_history_payload_export,
                "requires_unscoped_writer": True,
                "read_only": True,
                "point_in_time_atomic": False,
                "retry_guaranteed": False,
                "authorizes_restore": False,
            },
            "bundle_restore_name_preflight": {
                "format": "tensorus-bundle-restore-name-preflight-v1",
                "product_enabled": enable_product_bundle_export,
                "history_enabled": enable_history_payload_export,
                "requires_unscoped_writer": True,
                "read_only": True,
                "point_in_time_atomic": False,
                "retry_guaranteed": False,
                "authorizes_restore": False,
            },
            "bundle_restore_payload_observation": {
                "format": "tensorus-mcp-bundle-restore-payload-observation-v1",
                "product_enabled": enable_product_bundle_export,
                "history_enabled": enable_history_payload_export,
                "requires_unscoped_writer": True,
                "retained_records_checked": False,
                "completion_assessed": False,
                "read_only": True,
                "point_in_time_atomic": False,
                "authorizes_restore": False,
            },
            "history_release_observation": {
                "format": "tensorus-history-restore-release-observation-v1",
                "history_enabled": enable_history_payload_export,
                "requires_unscoped_writer": True,
                "independent_release_checks": True,
                "conflicting_imports_reported": True,
                "read_only": True,
                "point_in_time_atomic": False,
                "authorizes_restore": False,
            },
            "history_restore_selection_review": {
                "format": "tensorus-history-restore-selection-review-v1",
                "history_enabled": enable_history_payload_export,
                "requires_unscoped_writer": True,
                "read_only": True,
                "names_reserved": False,
                "point_in_time_atomic": False,
                "authorizes_restore": False,
            },
            "history_payload_selection_review": {
                "format": "tensorus-history-payload-selection-review-v1",
                "history_enabled": enable_history_payload_export,
                "requires_unscoped_writer": True,
                "read_only": True,
                "point_in_time_atomic": False,
                "authorizes_restore": False,
            },
            "product_payload_selection_review": {
                "format": "tensorus-product-payload-selection-review-v1",
                "product_enabled": enable_product_bundle_export,
                "requires_unscoped_writer": True,
                "read_only": True,
                "unselected_receipts_checked": False,
                "point_in_time_atomic": False,
                "authorizes_restore": False,
            },
            "history_independent_selection_review": {
                "format": "tensorus-history-restore-independent-review-v1",
                "history_enabled": enable_history_payload_export,
                "requires_unscoped_writer": True,
                "read_only": True,
                "unselected_receipts_checked": False,
                "names_reserved": False,
                "point_in_time_atomic": False,
                "authorizes_restore": False,
            },
            "history_independent_observation": {
                "format": "tensorus-history-restore-independent-observation-v1",
                "history_enabled": enable_history_payload_export,
                "requires_unscoped_writer": True,
                "read_only": True,
                "unselected_receipts_checked": False,
                "names_reserved": False,
                "point_in_time_atomic": False,
                "authorizes_restore": False,
            },
        }

    @server.tool(annotations=read_only)
    def list_datasets(
        limit: int = 50,
        offset: int = 0,
        text: str | None = None,
    ) -> dict[str, Any]:
        """List datasets visible to this tenant credential, with explicit pagination."""
        return client.datasets(limit=limit, offset=offset, text=text)

    @server.tool(annotations=read_only)
    def list_dataset_options(
        limit: int = 25,
        offset: int = 0,
        text: str | None = None,
    ) -> dict[str, Any]:
        """List one compact authorized dataset page without full schemas or write authority."""
        return client.dataset_options(limit=limit, offset=offset, text=text)

    @server.tool(annotations=read_only)
    def list_data_products(
        dataset: str | None = None, limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        """Discover products only when every pinned member and dependency is readable."""
        return client.data_products(dataset=dataset, limit=limit, offset=offset)

    @server.tool(annotations=read_only)
    def get_data_product(name: str, revision: int | None = None) -> dict[str, Any]:
        """Read pinned roles and evidence; this does not verify payloads or apply validity."""
        return client.data_product(name, revision=revision)

    @server.tool(annotations=read_only)
    def list_data_product_revisions(name: str, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List only revisions whose complete pinned dependencies are readable."""
        return client.data_product_revisions(name, limit=limit, offset=offset)

    @server.tool(annotations=read_only)
    def list_product_releases(name: str, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List reviewed or proposed product releases with all dependencies authorized."""
        return client.product_releases(name, limit=limit, offset=offset)

    @server.tool(annotations=read_only)
    def get_product_release(release_id: UUID) -> dict[str, Any]:
        """Inspect selected quality, reference and plan evidence without rereading payloads."""
        return client.product_release(str(release_id))

    @server.tool(annotations=read_only)
    def list_product_similarity_evaluations(
        name: str, limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        """List retained evaluations only when all product and snapshot evidence is readable."""
        return client.product_similarity_evaluations(name, limit=limit, offset=offset)

    @server.tool(annotations=read_only)
    def get_product_similarity_evaluation(evaluation_id: UUID) -> dict[str, Any]:
        """Read an exact retained report; approval is not proof of goal success or truth."""
        return client.product_similarity_evaluation(str(evaluation_id))

    @server.tool(annotations=read_only)
    def list_reference_evaluations(limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List retained comparisons only when all four selected inputs are readable."""
        return client.reference_evaluations(limit=limit, offset=offset)

    @server.tool(annotations=read_only)
    def get_reference_evaluation(evaluation_id: UUID) -> dict[str, Any]:
        """Read a retained pairwise report; approval is not proof of truth or goal success."""
        return client.reference_evaluation(str(evaluation_id))

    @server.tool(annotations=read_only)
    def list_reference_evaluation_archives(limit: int = 10, offset: int = 0) -> dict[str, Any]:
        """List retained withdrawal and retirement receipts with an unscoped credential."""
        return client.reference_evaluation_archives(limit=limit, offset=offset)

    @server.tool(annotations=read_only)
    def get_reference_evaluation_archive(evaluation_id: UUID) -> dict[str, Any]:
        """Read archived comparison evidence; its hashes are not signatures."""
        return client.reference_evaluation_archive(str(evaluation_id))

    @server.tool(annotations=read_only)
    def review_reference_retirement(
        evaluation_id: UUID, report_hash: str, reason: str
    ) -> dict[str, Any]:
        """Review retirement without mutation; only unscoped tenant readers may access it."""
        return client.review_reference_retirement(
            str(evaluation_id), ReferenceDispositionRequest(report_hash=report_hash, reason=reason)
        )

    @server.tool(annotations=read_only)
    def list_product_similarity_withdrawals(
        name: str, limit: int = 10, offset: int = 0
    ) -> dict[str, Any]:
        """List retained withdrawal receipts; an unscoped tenant credential is required."""
        return client.product_similarity_withdrawals(name, limit=limit, offset=offset)

    @server.tool(annotations=read_only)
    def get_product_similarity_withdrawal(evaluation_id: UUID) -> dict[str, Any]:
        """Read exact archived proposal evidence; hashes are not signatures."""
        return client.product_similarity_withdrawal(str(evaluation_id))

    @server.tool(annotations=read_only)
    def review_product_similarity_retirement(
        evaluation_id: UUID, report_hash: str, reason: str
    ) -> dict[str, Any]:
        """Review a retirement without changing state; only unscoped readers may access it."""
        return client.review_product_similarity_retirement(
            str(evaluation_id),
            ProductSimilarityRetirementReviewRequest(report_hash=report_hash, reason=reason),
        )

    @server.tool(annotations=read_only)
    def list_product_similarity_retirements(
        name: str, limit: int = 10, offset: int = 0
    ) -> dict[str, Any]:
        """List approved-evaluation retirement receipts for unscoped tenant readers."""
        return client.product_similarity_retirements(name, limit=limit, offset=offset)

    @server.tool(annotations=read_only)
    def get_product_similarity_retirement(evaluation_id: UUID) -> dict[str, Any]:
        """Read retained approved evidence and its retirement receipt; hashes are not signatures."""
        return client.product_similarity_retirement(str(evaluation_id))

    @server.tool(annotations=read_only)
    def review_product_source_evaluation(
        name: str, request: ProductSourceEvaluationReviewRequest
    ) -> dict[str, Any]:
        """Review a source-claim association without mutation or source authentication."""
        return client.review_product_source_evaluation(name, request)

    @server.tool(annotations=read_only)
    def list_product_source_evaluations(
        name: str, limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        """List readable reviewed source-to-local evaluation bindings."""
        return client.product_source_evaluations(name, limit=limit, offset=offset)

    @server.tool(annotations=read_only)
    def get_product_source_evaluation(binding_id: UUID) -> dict[str, Any]:
        """Read a binding without claiming source authenticity or quality equivalence."""
        return client.product_source_evaluation(str(binding_id))

    @server.tool(annotations=read_only)
    def export_product_release(release_id: UUID) -> dict[str, Any]:
        """Export approved, unauthenticated source claims including attached quality reports."""
        return client.export_product_release(str(release_id))

    if enable_product_bundle_export:

        @server.tool(annotations=read_only)
        def export_product_bundle(release_id: UUID) -> dict[str, Any]:
            """Export selected verified NPY payloads and unsigned claims; no approval transfers."""
            try:
                payload = client.export_product_bundle(str(release_id))
            except TensorusError as exc:
                if exc.code != "storage_error":
                    raise
                raise TensorusError(
                    "Source storage could not be verified; retry or inspect recovery evidence",
                    code="storage_error",
                    status=503,
                ) from None
            if len(payload) > MAX_MCP_PRODUCT_BUNDLE_BYTES:
                raise LimitExceeded("Product bundle exceeds the MCP archive byte limit")
            return {
                "format": "tensorus-mcp-product-bundle-v1",
                "release_id": str(release_id),
                "archive_bytes": len(payload),
                "archive_sha256": hashlib.sha256(payload).hexdigest(),
                "archive_base64": base64.b64encode(payload).decode("ascii"),
                "source_claims_authenticated": False,
                "approval_transferred": False,
            }

        @server.tool(annotations=read_only)
        def inspect_product_bundle(
            archive_base64: Annotated[
                str,
                Field(strict=True, min_length=4, max_length=MAX_MCP_PRODUCT_BUNDLE_BASE64_CHARS),
            ],
        ) -> dict[str, Any]:
            """Verify every selected payload in an unsigned archive without writing locally."""
            payload = _decode_product_bundle(archive_base64)
            return {
                "format": "tensorus-mcp-product-bundle-inspection-v1",
                "archive_bytes": len(payload),
                "archive_sha256": hashlib.sha256(payload).hexdigest(),
                "manifest": client.inspect_product_bundle(payload),
            }

        @server.tool(annotations=read_only)
        def review_product_bundle_restore(
            archive_base64: Annotated[
                str,
                Field(strict=True, min_length=4, max_length=MAX_MCP_PRODUCT_BUNDLE_BASE64_CHARS),
            ],
            request: ProductBundleRestoreRequest,
        ) -> dict[str, Any]:
            """Review one unsigned bundle for the current unscoped writer without writing."""
            return client.review_product_bundle_restore(
                _decode_product_bundle(archive_base64), request
            )

        @server.tool(annotations=read_only)
        def inspect_product_bundle_restore_progress(
            archive_base64: Annotated[
                str,
                Field(strict=True, min_length=4, max_length=MAX_MCP_PRODUCT_BUNDLE_BASE64_CHARS),
            ],
            request: ProductBundleRestoreRequest,
            expected_review_hash: ReleaseHash,
            reviewed: dict[str, Any] | None = None,
            include_head_preflight: StrictBool = False,
            include_name_preflight: StrictBool = False,
            payloads_only: StrictBool = False,
            product_payload_selection_role: Name | None = None,
        ) -> dict[str, Any]:
            """Inspect exact receipt-backed progress; this is neither approval nor a restore."""
            if include_head_preflight and include_name_preflight:
                raise TensorusError("Select one bundle preflight format")
            if payloads_only and (include_head_preflight or include_name_preflight):
                raise TensorusError("Select one bundle observation format")
            if product_payload_selection_role is not None:
                if payloads_only or include_head_preflight or include_name_preflight:
                    raise TensorusError("Select one bundle observation format")
                if reviewed is None or reviewed.get("review_hash") != expected_review_hash:
                    raise TensorusError("The complete original restore review is required")
                return client.review_product_payload_selection(
                    _decode_product_bundle(archive_base64),
                    request,
                    reviewed=reviewed,
                    role=product_payload_selection_role,
                )
            inspect = (
                client.inspect_product_bundle_restore_payloads
                if payloads_only
                else client.inspect_product_bundle_restore_names
                if include_name_preflight
                else client.inspect_product_bundle_restore_heads
                if include_head_preflight
                else client.inspect_product_bundle_restore
            )
            return inspect(
                _decode_product_bundle(archive_base64),
                request,
                expected_review_hash=expected_review_hash,
                reviewed=reviewed,
            )

    if enable_history_payload_export:

        @server.tool(annotations=read_only)
        def export_release_history_payload_bundle(
            dataset: str, request: PortableHistoryExport
        ) -> dict[str, Any]:
            """Export selected verified payloads and unsigned history claims without approvals."""
            try:
                payload = client.export_release_history_payload_bundle(dataset, request)
            except TensorusError as exc:
                if exc.code != "storage_error":
                    raise
                raise TensorusError(
                    "Source storage could not be verified; retry or inspect recovery evidence",
                    code="storage_error",
                    status=503,
                ) from None
            if len(payload) > MAX_MCP_HISTORY_PAYLOAD_BYTES:
                raise LimitExceeded("History payload bundle exceeds the MCP archive byte limit")
            return {
                "format": "tensorus-mcp-release-history-payload-bundle-v1",
                "dataset": dataset,
                "archive_bytes": len(payload),
                "archive_sha256": hashlib.sha256(payload).hexdigest(),
                "archive_base64": base64.b64encode(payload).decode("ascii"),
                "source_claims_authenticated": False,
                "approval_transferred": False,
            }

        @server.tool(annotations=read_only)
        def inspect_release_history_payload_bundle(
            archive_base64: Annotated[
                str,
                Field(strict=True, min_length=4, max_length=MAX_MCP_HISTORY_PAYLOAD_BASE64_CHARS),
            ],
        ) -> dict[str, Any]:
            """Verify selected unsigned history claims and every included NPY without writing."""
            payload = _decode_history_payload_bundle(archive_base64)
            return {
                "format": "tensorus-mcp-release-history-payload-inspection-v1",
                "archive_bytes": len(payload),
                "archive_sha256": hashlib.sha256(payload).hexdigest(),
                "manifest": client.inspect_release_history_payload_bundle(payload),
            }

        @server.tool(annotations=read_only)
        def review_release_history_payload_restore(
            archive_base64: Annotated[
                str,
                Field(strict=True, min_length=4, max_length=MAX_MCP_HISTORY_PAYLOAD_BASE64_CHARS),
            ],
            request: HistoryPayloadRestoreRequest,
        ) -> dict[str, Any]:
            """Review selected unsigned history for the current unscoped writer without writing."""
            return client.review_release_history_payload_restore(
                _decode_history_payload_bundle(archive_base64), request
            )

        @server.tool(annotations=read_only)
        def inspect_release_history_payload_restore_progress(
            archive_base64: Annotated[
                str,
                Field(strict=True, min_length=4, max_length=MAX_MCP_HISTORY_PAYLOAD_BASE64_CHARS),
            ],
            request: HistoryPayloadRestoreRequest,
            expected_review_hash: ReleaseHash,
            reviewed: dict[str, Any] | None = None,
            include_head_preflight: StrictBool = False,
            include_name_preflight: StrictBool = False,
            payloads_only: StrictBool = False,
            history_release_observation: StrictBool = False,
            history_selection_release_id: UUID | None = None,
            history_payload_selection_release_id: UUID | None = None,
            history_independent_selection_release_id: UUID | None = None,
            history_independent_observation_release_id: UUID | None = None,
        ) -> dict[str, Any]:
            """Inspect selected receipt-backed history without approving or restoring it."""
            if include_head_preflight and include_name_preflight:
                raise TensorusError("Select one bundle preflight format")
            if payloads_only and (include_head_preflight or include_name_preflight):
                raise TensorusError("Select one bundle observation format")
            if history_release_observation and (
                payloads_only or include_head_preflight or include_name_preflight
            ):
                raise TensorusError("Select one bundle observation format")
            if (
                sum(
                    item is not None
                    for item in (
                        history_selection_release_id,
                        history_payload_selection_release_id,
                        history_independent_selection_release_id,
                        history_independent_observation_release_id,
                    )
                )
                > 1
            ):
                raise TensorusError("Select one history continuation")
            if any(
                item is not None
                for item in (
                    history_selection_release_id,
                    history_payload_selection_release_id,
                    history_independent_selection_release_id,
                    history_independent_observation_release_id,
                )
            ):
                if (
                    payloads_only
                    or include_head_preflight
                    or include_name_preflight
                    or history_release_observation
                ):
                    raise TensorusError("Select one bundle observation format")
                review_selection = (
                    client.observe_independent_history_release
                    if history_independent_observation_release_id is not None
                    else client.review_history_payload_selection
                    if history_payload_selection_release_id is not None
                    else client.review_independent_history_selection
                    if history_independent_selection_release_id is not None
                    else client.review_history_restore_selection
                )
                selected_id = (
                    history_independent_observation_release_id
                    or history_payload_selection_release_id
                    or history_independent_selection_release_id
                    or history_selection_release_id
                )
                assert selected_id is not None
                return review_selection(
                    _decode_history_payload_bundle(archive_base64),
                    request,
                    expected_review_hash=expected_review_hash,
                    release_id=selected_id,
                    reviewed=reviewed,
                )
            inspect = (
                client.inspect_release_history_restore_releases
                if history_release_observation
                else client.inspect_release_history_payload_restore_payloads
                if payloads_only
                else client.inspect_release_history_payload_restore_names
                if include_name_preflight
                else client.inspect_release_history_payload_restore_heads
                if include_head_preflight
                else client.inspect_release_history_payload_restore
            )
            return inspect(
                _decode_history_payload_bundle(archive_base64),
                request,
                expected_review_hash=expected_review_hash,
                reviewed=reviewed,
            )

    @server.tool(annotations=read_only)
    def list_product_release_imports(name: str, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List visible retained source claims; imports are not local approvals."""
        return client.product_release_imports(name, limit=limit, offset=offset)

    @server.tool(annotations=read_only)
    def get_product_release_import(import_id: UUID) -> dict[str, Any]:
        """Inspect source claims retained against the authorized destination product."""
        return client.product_release_import(str(import_id))

    @server.tool(annotations=read_only)
    def query_tensors(
        dataset: str | None = None,
        dtype: DType | None = None,
        shape: list[int] | None = None,
        text: str | None = None,
        limit: int = 50,
        offset: int = 0,
        snapshot_id: UUID | None = None,
        semantics: SemanticFilter | None = None,
        indexed: bool = False,
        indexed_execution: Literal["streamed_exact_v2"] | None = None,
        snapshot_cursor: bool = False,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Find current tensor heads, or snapshot members when snapshot_id is supplied.

        Uses typed catalog filters and pagination; does not execute SQL or arbitrary code.
        Coordinate spans and points over v2 axes require their exact declared reference.
        Points may opt into typed_v2 matching of exact numeric types on v2 axes.
        Coordinate counts use inclusive numeric intervals on declared v2 axes, not validity.
        Shared-axis overlap requires exact typed coordinate matches on every selected axis;
        it does not apply validity or perform an alignment.
        Indexed mode requires a semantic property and reconciles the bounded catalog candidates.
        streamed_exact_v2 opts into bounded batched evaluation of more property matches.
        Snapshot cursor mode requires indexed=True, a pinned snapshot, and offset=0.
        """
        if indexed_execution is not None and not indexed:
            raise TensorusError("Streamed indexed execution requires indexed discovery")
        query = TensorQuery(
            dataset=dataset,
            dtype=dtype,
            shape=shape,
            text=text,
            limit=limit,
            offset=offset,
            snapshot_id=snapshot_id,
            semantics=semantics,
            indexed_execution=indexed_execution,
        )
        if snapshot_cursor:
            if not indexed:
                raise TensorusError("Snapshot cursor mode requires indexed discovery")
            return client.query_indexed_snapshot_page(query, cursor=cursor)
        if cursor is not None:
            raise TensorusError("A cursor requires snapshot cursor mode")
        return client.query_indexed(query) if indexed else client.query(query)

    @server.tool(annotations=read_only)
    def profile_tensor(version_id: str) -> dict[str, Any]:
        """Read a descriptor and raw stored statistics; no validity binding is applied."""
        return client.version(version_id)

    @server.tool(annotations=read_only)
    def get_validity_statistics(
        version_id: UUID, request: ValidityStatisticsRequest
    ) -> dict[str, Any]:
        """Compare raw statistics with positions selected by an explicit pinned binding.

        Both sources are independently authorized and fully checksum-verified.
        The binding is an assertion, not proof of observation truth.
        """
        return client.validity_statistics(str(version_id), request)

    @server.tool(annotations=read_only)
    def list_validity_bindings(data_version_id: UUID) -> dict[str, Any]:
        """List visible explicit assertions pairing this data version with boolean masks."""
        return client.validity_bindings(str(data_version_id))

    @server.tool(annotations=read_only)
    def get_validity_binding(binding_id: UUID) -> dict[str, Any]:
        """Inspect one immutable validity assertion; it does not change ordinary reads."""
        return client.validity_binding(str(binding_id))

    @server.tool(annotations=read_only)
    def explain_validity_transform(binding_id: UUID, request: ValidityTransform) -> dict[str, Any]:
        """Prepare a reviewed paired transform; this neither approves nor executes it.

        V2 semantics support advertised positional transforms, single-binding reindexing against
        an explicit typed anchor, masked reductions, exact two-binding arithmetic, reviewed common
        typed-coordinate intersections with optional explicit right-axis permutation, anchored
        alignment with false gaps and optional explicit non-anchor permutation, and the advertised
        bounded unit conversions. Permuted filled alignment materializes both data and mask
        transposes, even for an identity permutation.
        V2 axis conversion rejects coordinate-system-referenced axes; a time reference requires
        fixed time units.
        Arithmetic, v2 reindexing and v2 conversions require source protection. Paired reindexing
        pins the exact source binding; raw fills remain visible and gaps are false in the mask.
        When advertised, nearest_integer_v1 uses signed 64-bit coordinate offsets on changed axes,
        chooses the lower coordinate on ties and copies the source mask bit or false for a gap.
        The SDK checks the bounded reviewed mapping against source and anchor descriptors; the
        server's capability and descriptors remain unauthenticated claims.
        For filled paired arithmetic, selected_preprocessing_policy="bound_selected_v1" explicitly
        compares matching declared source identities only at positions included by both bound
        masks. It does not label intermediates containing fills or outputs as globally preprocessed.
        For an advertised v2 paired reduction or positional transform, expected_binding_hash
        optionally pins the source binding and requires source protection. Reductions do not give
        the result a preprocessing identity; positional transforms retain selected raw values and
        mask bits without proving observation truth.
        Selected global scaling, exact numeric casting and scalar_arithmetic require the exact
        source-binding hash and source protection. Casting rejects any included value that would
        change, including negative zero in an integer target. Scalar arithmetic uses approximate
        float64, with dimensionless addition/subtraction and a dimensionless multiplier/divisor.
        Excluded positions become zero placeholders; the unchanged mask commits separately.
        Completion is an explicit separate mutation after the exact approved plan succeeds.
        """
        return client.explain_validity_transform(str(binding_id), request)

    @server.tool(annotations=read_only)
    def explain_fitted_preprocessing_fit(
        binding_id: UUID, request: FittedPreprocessingFit
    ) -> dict[str, Any]:
        """Review a pinned selected fit; proposal, approval, and execution are separate."""
        return client.explain_fitted_preprocessing_fit(str(binding_id), request)

    @server.tool(annotations=read_only)
    def get_fitted_preprocessing_state(state_version_id: UUID) -> dict[str, Any]:
        """Inspect a verified local fitted-state version and its pinned training provenance."""
        return client.fitted_preprocessing_state(str(state_version_id))

    @server.tool(annotations=read_only)
    def review_imported_fitted_preprocessing_state(
        state_version_id: UUID, selection: ImportedFittedStateSelection
    ) -> dict[str, Any]:
        """Review an unsigned imported state and separate local product approval; no fit is run."""
        return client.review_imported_fitted_preprocessing_state(str(state_version_id), selection)

    @server.tool(annotations=read_only)
    def explain_fitted_preprocessing_apply(
        binding_id: UUID, request: FittedPreprocessingApply
    ) -> dict[str, Any]:
        """Review reuse of exact fitted state; data and mask need separate completion."""
        return client.explain_fitted_preprocessing_apply(str(binding_id), request)

    @server.tool(annotations=read_only)
    def explain_validity_coordinate_window(
        binding_id: UUID, request: ValidityCoordinateWindow
    ) -> dict[str, Any]:
        """Review a v2 data/mask window without reading payloads or creating work.

        Inclusive numeric bounds require matching declared units and references. The plan
        pins the selected binding, original bounds and resolved positions. Approval, execution
        and completion remain separate; the mask is an assertion, not proof. Optional
        chunked_verified execution bounds input memory but scans and checksum-verifies
        both entire sources; it is not partial I/O.
        """
        return client.explain_validity_coordinate_window(str(binding_id), request)

    if enable_npy_download:

        @server.tool(annotations=read_only)
        def read_npy_chunk(
            version_id: UUID,
            offset: Annotated[StrictInt, Field(ge=0, lt=2**63)] = 0,
            length: Annotated[StrictInt, Field(ge=1, le=MAX_MCP_NPY_RANGE_BYTES)] = 262_144,
            expected_etag: Annotated[
                str | None, Field(min_length=66, max_length=66, pattern=r'^"[0-9a-f]{64}"$')
            ] = None,
            expected_total_bytes: Annotated[StrictInt | None, Field(gt=0, lt=2**63)] = None,
        ) -> dict[str, Any]:
            """Read one NPY range; supply its first ETag and total size for later ranges."""
            result = client.read_npy_range(
                str(version_id),
                offset=offset,
                length=length,
                expected_etag=expected_etag,
                expected_total_bytes=expected_total_bytes,
            )
            next_offset = result.offset + len(result.data)
            return {
                "version_id": str(version_id),
                "offset": result.offset,
                "next_offset": next_offset,
                "total_bytes": result.total_bytes,
                "etag": result.etag,
                "data_base64": base64.b64encode(result.data).decode("ascii"),
                "chunk_sha256": hashlib.sha256(result.data).hexdigest(),
                "complete": next_offset == result.total_bytes,
            }

        @server.tool(annotations=read_only)
        def read_npy_tile(
            version_id: UUID,
            selection: Annotated[list[list[StrictInt | None]], Field(max_length=16)],
        ) -> dict[str, Any]:
            """Read a bounded positional NPY tile; full-object verification is not guaranteed."""
            result = client.read_npy_tile(str(version_id), selection)
            data = dump_npy(result)
            return {
                "version_id": str(version_id),
                "selection": selection,
                "shape": list(result.shape),
                "dtype": result.dtype.name,
                "data_base64": base64.b64encode(data).decode("ascii"),
                "tile_sha256": hashlib.sha256(data).hexdigest(),
            }

    @server.tool(annotations=read_only)
    def slice_tensor(version_id: str, selection: list[list[StrictInt | None]]) -> dict[str, Any]:
        """Read an element-budgeted tensor slice. Each axis uses [start, stop, positive_step]."""
        return client.preview(version_id, selection)

    @server.tool(annotations=read_only)
    def slice_tensor_axes(
        version_id: str, selection: dict[str, list[StrictInt | None]]
    ) -> dict[str, Any]:
        """Slice by version axis labels using positional [start, stop, positive_step].

        Labels are caller declarations, not proof of meaning. Derived versions require
        supported, propagated versioned semantics; legacy derived labels are not trusted.
        """
        return client.preview_axes(version_id, selection)

    @server.tool(annotations=read_only)
    def slice_tensor_coordinates(
        version_id: str, selection: dict[str, CoordinateRange]
    ) -> dict[str, Any]:
        """Slice inclusive numeric ranges using declared axis coordinates and exact units.

        Coordinates and references are caller declarations, not proof of meaning. A referenced
        axis requires its exact declared reference; no conversion is performed. Derived versions
        require supported, propagated versioned semantics; legacy derived coordinates are not
        trusted.
        """
        return client.preview_coordinates(
            version_id, {axis: bounds.model_dump(mode="json") for axis, bounds in selection.items()}
        )

    @server.tool(annotations=read_only)
    def search_similar_tensors(
        version_id: str,
        metric: Literal["cosine", "l2"] = "cosine",
        k: int = 10,
        dataset: str | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> dict[str, Any]:
        """Exhaustive same-shape similarity against visible current tensors, excluding the query.

        Cosine ranks higher first; l2 ranks lower first. Server candidate and byte budgets apply;
        narrow by dataset or metadata when exceeded. Scores are approximate; this is not ANN search.
        """
        return client.search(version_id, metric=metric, k=k, dataset=dataset, metadata=metadata)

    @server.tool(annotations=read_only)
    def search_snapshot_similar_tensors(
        version_id: UUID,
        snapshot_id: UUID,
        metric: Literal["cosine", "l2"] = "cosine",
        k: int = 10,
        metadata: dict[str, JsonValue] | None = None,
        expected_membership_hash: str | None = None,
        expected_candidate_hash: str | None = None,
    ) -> dict[str, Any]:
        """Compare raw values against a pinned snapshot with complete authorized membership.

        Selected payloads are checksum verified. Masks are not applied, hashes do not
        authenticate the report, and floating-point scores may differ across environments.
        The report format and scoring engine identify the raw numerical algorithm.
        """
        return client.search_snapshot(
            str(version_id),
            str(snapshot_id),
            metric=metric,
            k=k,
            metadata=metadata,
            expected_membership_hash=expected_membership_hash,
            expected_candidate_hash=expected_candidate_hash,
        )

    @server.tool(annotations=read_only)
    def search_snapshot_validity_similar_tensors(
        request: ValiditySnapshotSimilarityRequest,
    ) -> dict[str, Any]:
        """Compare explicitly selected true/true masks against every eligible pinned candidate.

        All snapshot members, source requirements and selected masks must be readable.
        Counts and thresholds disclose pairwise coverage; assertions do not prove truth.
        The report is unretained, unapproved and unauthenticated.
        """
        return client.search_snapshot_validity(request)

    @server.tool(annotations=read_only)
    def list_tensor_versions(
        dataset: str, name: str, limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        """List immutable versions of a named tenant tensor with pagination."""
        return client.versions(dataset, name, limit=limit, offset=offset)

    @server.tool(annotations=read_only)
    def get_lineage(version_id: str) -> dict[str, Any]:
        """Read ancestor lineage, bounded by the service's default depth and node limit.

        Check the returned truncated flag; this does not list descendants.
        """
        return client.lineage(version_id)

    @server.tool(annotations=read_only)
    def get_descendants(version_id: str, max_depth: int = 12) -> dict[str, Any]:
        """Read bounded, authorized downstream version links; not predicted impact."""
        return client.descendants(version_id, max_depth=max_depth)

    @server.tool(annotations=read_only)
    def list_snapshots(dataset: str, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List a dataset's immutable snapshots with pagination; does not create one."""
        return client.snapshots(dataset, limit=limit, offset=offset)

    @server.tool(annotations=read_only)
    def list_snapshot_cursor_page(
        dataset: str,
        limit: Annotated[StrictInt, Field(ge=1, le=200)] = 50,
        after: UUID | None = None,
        before: UUID | None = None,
    ) -> dict[str, Any]:
        """Page snapshot descriptors by a retained visible anchor, not a pinned catalog snapshot."""
        return client.snapshot_cursor_page(
            dataset,
            limit=limit,
            after=str(after) if after is not None else None,
            before=str(before) if before is not None else None,
        )

    @server.tool(annotations=read_only)
    def get_snapshot(snapshot_id: str) -> dict[str, Any]:
        """Read snapshot metadata; use query_tensors with snapshot_id for its members."""
        return client.get_snapshot(snapshot_id)

    @server.tool(annotations=read_only)
    def preview_snapshot_retirement(snapshot_id: str) -> dict[str, Any]:
        """Inspect retained-reference blockers; requires an active unscoped reader or writer.

        A preview does not reserve the snapshot. Tensor versions and bytes are never erased.
        """
        return client.preview_snapshot_retirement(snapshot_id)

    @server.tool(annotations=read_only)
    def verify_recovery_objects(request: RecoveryVerificationRequest) -> dict[str, Any]:
        """Check selected object checksums without repair; unscoped tenant admin only.

        This is not a paired backup, source authentication or a hard time or memory limit.
        Inspect the returned status; incomplete checks do not prove corruption.
        """
        return client.verify_recovery_objects(request)

    @server.tool(annotations=read_only)
    def get_tenant_staging_reservations() -> dict[str, Any]:
        """Inspect tenant-local catalog reservations; active unscoped tenant admin only.

        This does not inspect live locks, free disk or other tenants and does not reconcile.
        """
        return client.tenant_staging_reservations()

    @server.tool(annotations=read_only)
    def get_tenant_staging_work_quota() -> dict[str, Any]:
        """Inspect tenant-local pooled work tokens; unpooled work and free capacity are excluded."""
        return client.tenant_staging_work_quota()

    if enable_retirement_archive_export:

        @server.tool(annotations=read_only)
        def export_snapshot_retirement_archive(
            request: RetirementArchiveSelection,
        ) -> dict[str, Any]:
            """Export selected receipts; requires an active unscoped administrator.

            The archive is an unauthenticated copy; audit records and quota remain unchanged.
            """
            return client.export_snapshot_retirement_archive(request)

    @server.tool(annotations=read_only)
    def diff_snapshots(left: str, right: str) -> dict[str, Any]:
        """Compare two snapshots of the same dataset by added, removed and changed tensor names."""
        return client.snapshot_diff(left, right)

    @server.tool(annotations=read_only)
    def list_contract_revisions(dataset: str, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """Inspect retained contracts; revisions become write overlays only after activation."""
        return client.contracts(dataset, limit=limit, offset=offset)

    @server.tool(annotations=read_only)
    def get_contract_revision(dataset: str, revision: int) -> dict[str, Any]:
        """Read one exact contract and its hash. Declared semantics are not proof of meaning."""
        return client.contract(dataset, revision)

    @server.tool(annotations=read_only)
    def get_write_contract(dataset: str) -> dict[str, Any]:
        """Inspect the current additive overlay; the original dataset spec also applies."""
        return client.write_contract(dataset)

    @server.tool(annotations=read_only)
    def list_write_contract_activations(
        dataset: str, limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        """Read retained activation history; this does not authorize or change a contract."""
        return client.write_contract_history(dataset, limit=limit, offset=offset)

    @server.tool(annotations=read_only)
    def list_quality_releases(dataset: str, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List retained snapshot quality reports and approval status."""
        return client.releases(dataset, limit=limit, offset=offset)

    @server.tool(annotations=read_only)
    def get_quality_release(release_id: str) -> dict[str, Any]:
        """Read membership, pinned member contracts and quality; does not prove goal success."""
        return client.release(release_id)

    @server.tool(annotations=read_only)
    def export_release_evidence(
        release_id: str,
        include_lineage: StrictBool = False,
        validity_binding_ids: Annotated[
            list[str], Field(min_length=1, max_length=MAX_PORTABLE_VALIDITY_SELECTIONS)
        ]
        | None = None,
        plan_ids: Annotated[list[str], Field(min_length=1, max_length=MAX_PORTABLE_PLAN_SELECTIONS)]
        | None = None,
    ) -> dict[str, Any]:
        """Read source contracts, validity and plan claims without transferring authority."""
        return client.export_release(
            release_id,
            include_lineage=include_lineage,
            validity_binding_ids=validity_binding_ids,
            plan_ids=plan_ids,
        )

    @server.tool(annotations=read_only)
    def export_release_history(dataset: str, request: PortableHistoryExport) -> dict[str, Any]:
        """Export selected claims, not complete or authenticated history."""
        return client.export_release_history(dataset, request)

    @server.tool(annotations=read_only)
    def list_imported_releases(dataset: str, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List source release histories retained in a readable destination dataset."""
        return client.imported_releases(dataset, limit=limit, offset=offset)

    @server.tool(annotations=read_only)
    def get_imported_release(import_id: str) -> dict[str, Any]:
        """Read mapped source claims and destination snapshot; source approval is not local."""
        return client.imported_release(import_id)

    @server.tool(annotations=read_only)
    def inspect_imported_history(
        dataset: str, request: ImportedHistoryInspectionRequest
    ) -> dict[str, Any]:
        """Inspect selected imports and mapping conflicts, not complete or authenticated history."""
        return client.inspect_imported_history(dataset, request)

    @server.tool(annotations=read_only)
    def lookup_imported_history_index(
        dataset: str, request: ImportedHistoryIndexLookupRequest
    ) -> dict[str, Any]:
        """Find indexed manifest matches with an unscoped reader; history may be incomplete."""
        return client.lookup_imported_history_index(dataset, request)

    @server.tool(annotations=read_only)
    def scan_imported_history(dataset: str, request: ImportedHistoryScanRequest) -> dict[str, Any]:
        """Find matches only after authorizing every import in one bounded page."""
        return client.scan_imported_history(dataset, request)

    @server.tool(annotations=read_only)
    def scan_readable_imported_history(
        dataset: str, request: ImportedHistoryScanRequest
    ) -> dict[str, Any]:
        """Find readable matches and report how many inaccessible candidates were omitted."""
        return client.scan_readable_imported_history(dataset, request)

    @server.tool(annotations=read_only)
    def review_imported_history_index(
        dataset: str, request: ImportedHistoryIndexReviewRequest
    ) -> dict[str, Any]:
        """Review one index page; requires an unscoped administrator."""
        return client.review_imported_history_index(dataset, request)

    @server.tool(annotations=read_only)
    def diagnose_imported_history_index(
        dataset: str, request: ImportedHistoryIndexDiagnosticRequest
    ) -> dict[str, Any]:
        """Classify one index page with foreign ownership redacted; unscoped admin only."""
        return client.diagnose_imported_history_index(dataset, request)

    if enable_proposals:

        @server.tool(annotations=proposal)
        def propose_plan(
            definition: dict[str, Any], idempotency_key: str | None = None
        ) -> dict[str, Any]:
            """Propose with a scoped agent; reuse the same key after an uncertain result."""
            require_scoped_mutator()
            return client.propose(
                PlanDefinition.model_validate(definition), idempotency_key=idempotency_key
            )

        @server.tool(annotations=proposal)
        def propose_goal(
            dataset: str,
            goal: str,
            semantic_policy: Literal["v1", "v2"] | None = None,
            access_policy: Literal["source_datasets_v1"] | None = None,
            proposer_authority: PlanProposerAuthority | None = None,
        ) -> dict[str, Any]:
            """Propose a built-in recipe with a scoped agent; never approve or execute it."""
            require_scoped_mutator()
            return client.propose_goal(
                GoalRequest(
                    dataset=dataset,
                    goal=goal,
                    semantic_policy=semantic_policy,
                    access_policy=access_policy,
                    proposer_authority=proposer_authority,
                )
            )

    @server.tool(annotations=read_only)
    def get_plan(plan_id: str) -> dict[str, Any]:
        """Inspect a proposed or running plan and its real durable step results."""
        return client.plan(plan_id)

    @server.tool(annotations=read_only)
    def get_plan_output_reservations(plan_id: str) -> dict[str, Any]:
        """Inspect this plan's staging tokens, not live filesystem allocation or owner locks."""
        return client.plan_output_reservations(plan_id)

    @server.tool(annotations=read_only)
    def get_plan_worker_attempt_evidence(plan_id: str, expected_plan_hash: str) -> dict[str, Any]:
        """Inspect recorded local worker limits; missing records and claims are not attestation."""
        return client.plan_worker_attempt_evidence(plan_id, expected_plan_hash)

    @server.tool(annotations=read_only)
    def get_plan_worker_claim_accounting(plan_id: str, expected_plan_hash: str) -> dict[str, Any]:
        """Inspect permitted plan claim outcomes, not measured resource use or charges."""
        return client.plan_worker_claim_accounting(plan_id, expected_plan_hash)

    @server.tool(annotations=read_only)
    def get_plan_estimated_attempts(plan_id: str, expected_plan_hash: str) -> dict[str, Any]:
        """Inspect permitted recorded attempt estimates, not measured usage or charges."""
        return client.plan_estimated_attempts(plan_id, expected_plan_hash)

    @server.tool(annotations=read_only)
    def get_plan_usage_observations(plan_id: str, expected_plan_hash: str) -> dict[str, Any]:
        """Inspect permitted logical outputs and elapsed wall time, not charges."""
        return client.plan_usage_observations(plan_id, expected_plan_hash)

    @server.tool(annotations=read_only)
    def get_ingest_usage_observation(version_id: str, expected_checksum: str) -> dict[str, Any]:
        """Inspect permitted committed logical bytes; missing evidence is unknown, not zero."""
        return client.ingest_usage_observation(version_id, expected_checksum)

    @server.tool(annotations=read_only)
    def get_plan_process_usage_observations(
        plan_id: str, expected_plan_hash: str
    ) -> dict[str, Any]:
        """Inspect partial permitted bounded-worker CPU and RSS evidence, not charges."""
        return client.plan_process_usage_observations(plan_id, expected_plan_hash)

    @server.tool(annotations=read_only)
    def get_plan_usage_handoff(plan_id: str, expected_plan_hash: str) -> dict[str, Any]:
        """Copy permitted plan usage evidence; never submit it or create a charge."""
        return client.plan_usage_handoff(plan_id, expected_plan_hash)

    @server.tool(annotations=read_only)
    def get_plan_usage_settlement(plan_id: str, expected_plan_hash: str) -> dict[str, Any]:
        """Inspect permitted durable plan usage settlement status; never creates a charge."""
        return client.plan_usage_settlement(plan_id, expected_plan_hash)

    @server.tool(annotations=read_only)
    def get_plan_receipt(
        plan_id: str,
        include_observation_scope: bool = False,
        observation_scope_version: ObservationScopeVersion = "v1",
    ) -> dict[str, Any]:
        """Check output integrity and recorded provenance; this does not verify goal success.

        May fully read plan outputs within the service's configured byte limits. Opt in to
        classify direct inputs and include a checked terminal checkpoint prefix; this does not
        infer a validity binding or independently recompute the operation. Set
        observation_scope_version to v2 to classify direct raw-arithmetic inputs, or v3 for
        positional, matrix and boolean-conjunction inputs, or v4 for coordinate mapping,
        descriptor-only inputs and coverage. Scope does not infer a binding.
        """
        return client.receipt(
            plan_id,
            include_observation_scope=include_observation_scope,
            observation_scope_version=observation_scope_version,
        )

    @server.tool(annotations=read_only)
    def get_plan_outcome(
        plan_id: str, outcome_version: PlanOutcomeVersion = "v1"
    ) -> dict[str, Any]:
        """Check terminal plan checkpoints and output checksums, including partial commits.

        Set outcome_version to v2 for v3 direct-input scope or v3 for v4 coordinate scope,
        source descriptors and explicitly pinned bindings. This may fully read committed outputs.
        Scope does not infer a binding or prove observation truth; this does not recompute the
        operation or prove a goal.
        """
        return client.plan_outcome(plan_id, outcome_version=outcome_version)

    @server.tool(annotations=read_only)
    def evaluate_plan_outputs(plan_id: str, criteria: EvaluationRequest) -> dict[str, Any]:
        """Check explicit criteria, optionally against a fully read immutable reference.

        Separate optional boolean masks or pinned bindings select positions for quality and
        reference checks. A reference can instead select the intersection of explicitly pinned
        output and reference bindings. Minimum fractions are optional; contracts use all values.
        Float tolerances use explicit decimal strings and the reference-relative formula.
        Byte and element limits apply; a mask or match does not prove validity or the goal.
        """
        return client.evaluate(plan_id, criteria)

    @server.tool(annotations=read_only)
    def evaluate_tensor_version(
        version_id: str, criteria: TensorEvaluationRequest
    ) -> dict[str, Any]:
        """Evaluate selected criteria on an immutable, checksum-pinned tensor version.

        Explicit masks or pinned validity bindings and references are independently authorized
        and fully read. An optional reference validity intersection requires separate bindings
        for output and reference. Assertions do not prove observation truth.
        This does not verify execution, observation truth, reference correctness or the goal.
        """
        return client.evaluate_tensor(version_id, criteria)

    @server.tool(annotations=read_only)
    def list_plans(
        limit: int = 50,
        offset: int = 0,
        status: PlanStatus | None = None,
        dataset: str | None = None,
        text: str | None = None,
    ) -> dict[str, Any]:
        """List tenant plans and their statuses with pagination and optional filters."""
        return client.plans(limit=limit, offset=offset, status=status, dataset=dataset, text=text)

    @server.tool(annotations=read_only)
    def explain_plan(definition: dict[str, Any]) -> dict[str, Any]:
        """Validate a typed plan and inspect shapes/cost without persisting or executing it."""
        return client.explain(PlanDefinition.model_validate(definition))

    @server.tool(annotations=read_only)
    def explain_axis_alignment(request: AxisAlignmentRequest) -> dict[str, Any]:
        """Preview raw-value alignment, without applying validity bindings or creating a plan.

        For v2 semantics, matching typed coordinates and references and recorded-source access
        are required. Check server capabilities before relying on v2 guided alignment.
        """
        return client.explain_alignment(request)

    @server.tool(annotations=read_only)
    def list_workflows(
        dataset: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Discover tenant-scoped, versioned reusable workflows."""
        return client.workflows(dataset=dataset, limit=limit, offset=offset)

    @server.tool(annotations=read_only)
    def get_workflow(name: str) -> dict[str, Any]:
        """Read a saved workflow's exact current revision and named input contract."""
        return client.workflow(name)

    @server.tool(annotations=read_only)
    def explain_workflow(
        name: str,
        expected_version: int,
        suffix: str,
        proposer_authority: PlanProposerAuthority | None = None,
    ) -> dict[str, Any]:
        """Resolve current tensor heads and inspect a run without creating or approving it."""
        return client.explain_workflow(
            name,
            WorkflowRun(
                expected_version=expected_version,
                suffix=suffix,
                proposer_authority=proposer_authority,
            ),
        )

    if enable_proposals:

        @server.tool(annotations=proposal)
        def propose_workflow(
            name: str,
            expected_version: int,
            suffix: str,
            expected_plan_hash: str | None = None,
            proposer_authority: PlanProposerAuthority | None = None,
        ) -> dict[str, Any]:
            """Propose a saved workflow with a scoped agent; never approve or execute it."""
            require_scoped_mutator()
            return client.run_workflow(
                name,
                WorkflowRun(
                    expected_version=expected_version,
                    suffix=suffix,
                    expected_plan_hash=expected_plan_hash,
                    proposer_authority=proposer_authority,
                ),
            )

    if enable_cancellation:

        @server.tool(annotations=mutation)
        def cancel_plan(plan_id: str) -> dict[str, Any]:
            """Request cooperative cancellation; already committed outputs remain."""
            require_scoped_mutator()
            return client.cancel(plan_id)

    if enable_ingestion:

        @server.tool(annotations=mutation)
        def create_validity_binding(request: ValidityBindingCreate) -> dict[str, Any]:
            """Assert that true entries of an aligned mask include observed data."""
            require_scoped_mutator()
            return client.create_validity_binding(request)

        @server.tool(annotations=mutation)
        def complete_validity_transform(
            binding_id: UUID, request: ValidityTransformCompletion
        ) -> dict[str, Any]:
            """Record a paired validity result only after its approved plan succeeds."""
            require_scoped_mutator()
            return client.complete_validity_transform(str(binding_id), request)

        @server.tool(annotations=mutation)
        def ingest_tensor(
            dataset: str,
            name: str,
            data: Any,
            dtype: Literal["float32", "float64", "int64"],
            expected_version: StrictInt,
            idempotency_key: str,
            metadata: dict[str, JsonValue] | None = None,
            write_condition: NpyWriteCondition | None = None,
        ) -> dict[str, Any]:
            """Store bounded numeric JSON, optionally conditioning its dataset and write policy."""
            if expected_version < 0:
                raise ValueError("expected_version must be nonnegative")
            _validate_ingestion_key(idempotency_key)
            attributes = metadata if metadata is not None else {}
            _validate_ingestion(data, dtype, attributes)
            require_scoped_mutator()
            return client.put(
                dataset,
                name,
                data,
                dtype=dtype,
                metadata=attributes,
                expected_version=expected_version,
                idempotency_key=idempotency_key,
                write_condition=write_condition,
            )

    if enable_resumable_ingestion:

        @server.tool(annotations=mutation)
        def create_npy_upload(request: UploadCreate, idempotency_key: str) -> dict[str, Any]:
            """Reserve a credential-bound resumable NPY upload; this consumes staging quota."""
            _validate_ingestion_key(idempotency_key)
            _validate_metadata(request.metadata)
            require_scoped_mutator()
            return client.create_upload(
                request.dataset,
                request.name,
                size=request.size,
                checksum=request.checksum,
                expected_version=request.expected_version,
                idempotency_key=idempotency_key,
                metadata=request.metadata,
                expires_in_seconds=request.expires_in_seconds,
            )

        @server.tool(annotations=read_only)
        def get_npy_upload(upload_id: UUID) -> dict[str, Any]:
            """Inspect this credential's durable upload offset and status."""
            return client.upload_status(str(upload_id))

        @server.tool(annotations=read_only)
        def get_npy_upload_attempt_observations(
            upload_id: UUID,
            limit: Annotated[StrictInt, Field(ge=1, le=50)] = 50,
            after: UUID | None = None,
        ) -> dict[str, Any]:
            """Inspect this credential's recorded attempts, not complete failures or charges."""
            return client.upload_attempt_observations(
                str(upload_id), limit=limit, after=str(after) if after is not None else None
            )

        @server.tool(annotations=read_only)
        def get_npy_upload_preclaim_rejection(upload_id: UUID, attempt_id: UUID) -> dict[str, Any]:
            """Inspect one creator-owned keyed rejection, not complete usage or failure history."""
            return client.upload_preclaim_rejection(str(upload_id), str(attempt_id))

        @server.tool(annotations=mutation)
        def put_npy_upload_chunk(
            upload_id: UUID,
            offset: Annotated[StrictInt, Field(ge=0)],
            data_base64: Annotated[
                str, Field(strict=True, min_length=4, max_length=MAX_UPLOAD_BASE64_CHARS)
            ],
        ) -> dict[str, Any]:
            """Stage one exact-size NPY chunk as canonical base64; identical retries are safe."""
            data = _decode_upload_chunk(data_base64)
            require_scoped_mutator()
            return client.put_upload_chunk(str(upload_id), offset, data)

        @server.tool(annotations=mutation)
        def finalize_npy_upload(upload_id: UUID, attempt_id: UUID | None = None) -> dict[str, Any]:
            """Finalize; an optional caller-held ID identifies selected pre-claim evidence."""
            require_scoped_mutator()
            return client.finalize_upload(
                str(upload_id), attempt_id=str(attempt_id) if attempt_id is not None else None
            )

        @server.tool(annotations=mutation)
        def cancel_npy_upload(upload_id: UUID) -> dict[str, Any]:
            """Cancel an unfinished upload and remove its staged chunks."""
            require_scoped_mutator()
            return client.cancel_upload(str(upload_id))

    return server


def run() -> None:
    url, key = os.environ.get("TENSORUS_URL"), os.environ.get("TENSORUS_API_KEY")
    if not url or not key:
        raise RuntimeError("Set TENSORUS_URL and TENSORUS_API_KEY for the authenticated MCP bridge")
    with Client(url, key, max_tensor_bytes=_mcp_tensor_byte_limit()) as client:
        create_server(
            client,
            enable_proposals=os.environ.get("TENSORUS_MCP_ENABLE_PROPOSALS") == "1",
            enable_ingestion=os.environ.get("TENSORUS_MCP_ENABLE_INGESTION") == "1",
            enable_cancellation=os.environ.get("TENSORUS_MCP_ENABLE_CANCELLATION") == "1",
            enable_npy_download=os.environ.get("TENSORUS_MCP_ENABLE_NPY_DOWNLOAD") == "1",
            enable_resumable_ingestion=(
                os.environ.get("TENSORUS_MCP_ENABLE_RESUMABLE_INGESTION") == "1"
            ),
            enable_release_history_import=(
                os.environ.get("TENSORUS_MCP_ENABLE_RELEASE_HISTORY_IMPORT") == "1"
            ),
            enable_history_index_backfill=(
                os.environ.get("TENSORUS_MCP_ENABLE_HISTORY_INDEX_BACKFILL") == "1"
            ),
            enable_history_index_reconciliation=(
                os.environ.get("TENSORUS_MCP_ENABLE_HISTORY_INDEX_RECONCILIATION") == "1"
            ),
            enable_snapshot_retirement=(
                os.environ.get("TENSORUS_MCP_ENABLE_SNAPSHOT_RETIREMENT") == "1"
            ),
            enable_retirement_archive_export=(
                os.environ.get("TENSORUS_MCP_ENABLE_RETIREMENT_ARCHIVE_EXPORT") == "1"
            ),
            enable_product_bundle_export=(
                os.environ.get("TENSORUS_MCP_ENABLE_PRODUCT_BUNDLE_EXPORT") == "1"
            ),
            enable_history_payload_export=(
                os.environ.get("TENSORUS_MCP_ENABLE_HISTORY_PAYLOAD_EXPORT") == "1"
            ),
        ).run(transport="stdio")
