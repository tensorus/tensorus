import hashlib
import io
import json
import math
import os
import re
import secrets
import stat
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal, Self
from urllib.parse import quote
from uuid import UUID, uuid4

import httpx
import numpy as np
from pydantic import BaseModel, JsonValue, SecretStr, TypeAdapter, ValidationError

from tensorus.advisory_pricing import (
    MAX_RESPONSE_BYTES as MAX_ADVISORY_PRICING_RESPONSE_BYTES,
)
from tensorus.advisory_pricing import (
    AdvisoryPricingCapabilities,
    AdvisoryPricingStatus,
    PlanPriceScenario,
)
from tensorus.agent_grant_retry import (
    MAX_RESPONSE_BYTES as MAX_AGENT_GRANT_RETRY_RESPONSE_BYTES,
)
from tensorus.agent_grant_retry import (
    AgentGrantRetryCapabilities,
    grant_id,
    require_grant_lifetime,
)
from tensorus.agent_upload_staging import (
    MAX_RESPONSE_BYTES as MAX_AGENT_UPLOAD_STAGING_RESPONSE_BYTES,
)
from tensorus.agent_upload_staging import (
    AgentUploadStagingAdminCapabilities,
    AgentUploadStagingAdminStatus,
    AgentUploadStagingCapabilities,
    AgentUploadStagingStatus,
)
from tensorus.arrays import (
    MAX_NPY_TILE_BYTES,
    Array,
    Record,
    array_checksum,
    canonical_hash,
    canonical_json,
    decode_unique_json,
    descriptor_nbytes,
    ingestion_request_hash,
    positional_selection,
    validated_array,
    version_integrity_descriptor,
)
from tensorus.authenticated_read_fence import AuthenticatedReadFenceCapabilities
from tensorus.browser_identity import (
    MAX_LOGIN_RESPONSE_BYTES,
    BrowserLoginComplete,
    BrowserLoginConfiguration,
)
from tensorus.bundle_inspection_contract import (
    BundleHeadPreflightCapabilities,
    BundleInspectionCapabilities,
    BundlePayloadObservationCapabilities,
    BundleRestoreApplyCapabilities,
    BundleRestorePreparationCapabilities,
    HistoryIndependentObservationCapabilities,
    HistoryIndependentSelectionCapabilities,
    HistoryPayloadSelectionCapabilities,
    HistoryReleaseObservationCapabilities,
    HistoryRestoreSelectionCapabilities,
    NameOccupancyCapabilities,
    NamePreflightCapabilities,
    ProductPayloadSelectionCapabilities,
)
from tensorus.bundle_restore import FailureInspection
from tensorus.bundle_restore_names import NameOccupancyRequest
from tensorus.contracts import ContractDefinition, ContractWrite, ReleaseRequest
from tensorus.control_request_limits import (
    LOCAL_USAGE_FIXTURE_DELIVERY_PATH,
    LOCAL_USAGE_FIXTURE_INSPECTION_PATH,
    LOCAL_USAGE_FIXTURE_REVIEW_PATH,
    MAX_CONTROL_REQUEST_BYTES,
    MAX_PLAN_REVIEW_REQUEST_BYTES,
    WORKSPACE_LOCK_EXTENDED_PATH,
    WORKSPACE_LOCK_RECEIPT_PATH,
)
from tensorus.credentials import new_token, valid_generated_token, validate_endpoint
from tensorus.dataset_options import (
    MAX_OFFSET as MAX_DATASET_OPTIONS_OFFSET,
)
from tensorus.dataset_options import (
    MAX_PAGE_SIZE as MAX_DATASET_OPTIONS_PAGE_SIZE,
)
from tensorus.dataset_options import (
    MAX_RESPONSE_BYTES as MAX_DATASET_OPTIONS_RESPONSE_BYTES,
)
from tensorus.dataset_options import DatasetOptionsCapabilities, DatasetOptionsPage
from tensorus.dataset_read_fence import DatasetReadFenceCapabilities
from tensorus.discovery_index import (
    INDEX_FORMAT,
    INDEXED_PROPERTIES,
    SNAPSHOT_CURSOR_FORMAT,
    STREAMED_EVALUATION_FORMAT,
    SnapshotCursor,
    decode_snapshot_cursor,
    encode_snapshot_cursor,
)
from tensorus.errors import Conflict, Forbidden, LimitExceeded, TensorusError
from tensorus.evaluation import (
    EvaluationRequest,
    TensorEvaluationRequest,
    ValidityStatisticsRequest,
)
from tensorus.history_payload_restore import HistoryPayloadRestoreRequest
from tensorus.history_read_admission import (
    HistoryListAdmissionCapabilities,
    HistoryReadAdmissionCapabilities,
)
from tensorus.history_summary_policy import ScopedHistorySummaryPolicyCapabilities
from tensorus.http_body_admission import HttpBodyAdmissionCapabilities
from tensorus.identity_admission import IdentityProviderAdmissionCapabilities
from tensorus.ingest_usage import FORMAT as INGEST_USAGE_FORMAT
from tensorus.ingest_usage import MAX_RESPONSE_BYTES as MAX_INGEST_USAGE_RESPONSE_BYTES
from tensorus.ingest_usage import IngestUsageCapabilities, IngestUsageObservation
from tensorus.investigation import InvestigationCapabilities
from tensorus.key_inventory import MAX_PAGE_SIZE as MAX_CREDENTIAL_PAGE_SIZE
from tensorus.key_inventory import MAX_RESPONSE_BYTES as MAX_CREDENTIAL_PAGE_RESPONSE_BYTES
from tensorus.key_inventory import (
    CredentialInventoryAdmissionCapabilities,
    CredentialInventoryAdmissionV2Capabilities,
    CredentialPageCapabilities,
    CredentialPageV2Capabilities,
)
from tensorus.key_rotation import MAX_RESPONSE_BYTES as MAX_KEY_ROTATION_RESPONSE_BYTES
from tensorus.key_rotation import (
    KeyRotationCapabilities,
    KeyRotationReceipt,
    KeyRotationRequest,
    replacement_key_id,
)
from tensorus.local_usage_delivery import CAPABILITY_FORMAT as LOCAL_USAGE_FIXTURE_FORMAT
from tensorus.local_usage_delivery import INSPECTION_FORMAT as LOCAL_USAGE_INSPECTION_FORMAT
from tensorus.local_usage_delivery import (
    MAX_INSPECTION_RESPONSE_BYTES as MAX_LOCAL_USAGE_INSPECTION_RESPONSE_BYTES,
)
from tensorus.local_usage_delivery import (
    MAX_RESPONSE_BYTES as MAX_LOCAL_USAGE_FIXTURE_RESPONSE_BYTES,
)
from tensorus.local_usage_delivery import (
    FixtureCapabilities,
    FixtureDeliveryRequest,
    FixtureFileInspection,
    FixtureFileInspectionCapabilities,
    FixtureFileInspectionRequest,
    FixtureNamespaceGuardCapabilities,
    FixtureReceipt,
    FixtureReceiptReadCapabilities,
    FixtureRecovery,
    FixtureReview,
    FixtureReviewRequest,
    review_matches_request,
)
from tensorus.local_usage_intent import CAPABILITY_FORMAT as LOCAL_USAGE_INTENT_FORMAT
from tensorus.local_usage_intent import FixtureIntent, FixtureIntentCapabilities
from tensorus.model_admission import (
    ModelProviderAdmissionCapabilities,
    ModelProviderTenantAdmissionCapabilities,
)
from tensorus.model_destination import DirectTransportCapabilities, ModelDestinationCapabilities
from tensorus.model_provider_http import ModelProviderTimeoutCapabilities
from tensorus.models import (
    COORDINATE_COUNT_QUERY_FORMAT,
    COORDINATE_POINT_QUERY_FORMAT,
    MAX_COORDINATE_QUERY_COUNT,
    MAX_COORDINATE_QUERY_POINTS,
    TYPED_COORDINATE_POINT_QUERY_FORMAT,
    AgentAction,
    AgentDatasetGrant,
    AgentGrantCreate,
    AgentScope,
    AuditQuery,
    AxisAlignmentRequest,
    ConditionalNpyReceiptRequest,
    DatasetCreate,
    GoalRequest,
    IndexedSnapshotQuery,
    KeyCreate,
    Model,
    MultiDatasetAgentGrantCreate,
    MultiDatasetAgentScope,
    Name,
    NpyWriteCondition,
    PlanDefinition,
    PlanProposerAuthority,
    PlanQuery,
    PlanStatus,
    Role,
    SimilarityQuery,
    SliceRequest,
    TensorQuery,
    UploadCreate,
    WorkflowDefinition,
    WorkflowRun,
    WorkflowWrite,
    WorkspaceLockRequest,
    WorkspaceMemberInvite,
    WorkspaceMemberUpdate,
    WriteContractActivation,
    WriteContractBinding,
)
from tensorus.observation_scope import ObservationScopeVersion, observation_scope_format
from tensorus.operation_contracts import OperationContractMatrix
from tensorus.operations import (
    ImportedFittedStateSelection,
    NearestIntegerPolicy,
    ReindexCoordinatesParameters,
    TensorShape,
)
from tensorus.organization_detachment import (
    MAX_RESPONSE_BYTES as MAX_ORGANIZATION_DETACHMENT_RESPONSE_BYTES,
)
from tensorus.organization_detachment import (
    DetachmentCapabilities,
    DetachmentReceipt,
    DetachmentRequest,
    DetachmentStatus,
)
from tensorus.organization_detachment_receipt import (
    DetachmentReceiptCapabilities,
    RecoveredDetachmentReceipt,
)
from tensorus.organization_detachment_receipt import (
    validate_receipt_secret as validate_detachment_receipt_secret,
)
from tensorus.organizations import (
    MAX_RESPONSE_BYTES as MAX_ORGANIZATION_RESPONSE_BYTES,
)
from tensorus.organizations import (
    OrganizationAttach,
    OrganizationCapabilities,
    OrganizationCreate,
    OrganizationDirectory,
    OrganizationInvite,
    OrganizationMember,
    OrganizationMemberPage,
    OrganizationMemberUpdate,
    OrganizationSessionFenceCapabilities,
    OrganizationSessionFenceV2Capabilities,
    OrganizationView,
    OrganizationWorkspace,
    OrganizationWorkspaceList,
    OrgRole,
    OrgStatus,
)
from tensorus.plan_estimated_attempts import FORMAT as PLAN_ESTIMATED_ATTEMPTS_FORMAT
from tensorus.plan_estimated_attempts import (
    MAX_RESPONSE_BYTES as MAX_PLAN_ESTIMATED_ATTEMPTS_RESPONSE_BYTES,
)
from tensorus.plan_estimated_attempts import (
    PlanEstimatedAttempts,
    PlanEstimatedAttemptsCapabilities,
)
from tensorus.plan_outcomes import (
    MAX_PLAN_OUTCOME_BYTES,
    PlanOutcomeEvidence,
    PlanOutcomeVersion,
    plan_outcome_format,
)
from tensorus.plan_process_usage import FORMAT as PLAN_PROCESS_USAGE_FORMAT
from tensorus.plan_process_usage import MAX_RESPONSE_BYTES as MAX_PLAN_PROCESS_USAGE_RESPONSE_BYTES
from tensorus.plan_process_usage import PlanProcessUsage, PlanProcessUsageCapabilities
from tensorus.plan_proposer_authority import PlanProposerAuthorityCapabilities, matches_plan_schema
from tensorus.plan_usage import FORMAT as PLAN_USAGE_FORMAT
from tensorus.plan_usage import MAX_RESPONSE_BYTES as MAX_PLAN_USAGE_RESPONSE_BYTES
from tensorus.plan_usage import PlanUsageCapabilities, PlanUsageObservations
from tensorus.plan_usage_adjusted_handoff import FORMAT as ADJUSTED_USAGE_HANDOFF_FORMAT
from tensorus.plan_usage_adjusted_handoff import (
    MAX_RESPONSE_BYTES as MAX_ADJUSTED_USAGE_HANDOFF_RESPONSE_BYTES,
)
from tensorus.plan_usage_adjusted_handoff import (
    AdjustedHandoffCapabilities,
    AdjustedHandoffDocument,
    AdjustedUsageHandoff,
)
from tensorus.plan_usage_adjustments import FORMAT as PLAN_USAGE_ADJUSTMENTS_FORMAT
from tensorus.plan_usage_adjustments import (
    MAX_RESPONSE_BYTES as MAX_PLAN_USAGE_ADJUSTMENTS_RESPONSE_BYTES,
)
from tensorus.plan_usage_adjustments import (
    AdjustmentCapabilities,
    AdjustmentReceipt,
    AdjustmentReport,
    AdjustmentRequest,
)
from tensorus.plan_usage_comparison import (
    compare_references,
    read_reference,
    reference_from_document,
)
from tensorus.plan_usage_handoff import FORMAT as PLAN_USAGE_HANDOFF_FORMAT
from tensorus.plan_usage_handoff import MAX_RESPONSE_BYTES as MAX_PLAN_USAGE_HANDOFF_RESPONSE_BYTES
from tensorus.plan_usage_handoff import (
    HandoffDocument,
    PlanUsageHandoff,
    PlanUsageHandoffCapabilities,
)
from tensorus.product_bundle_restore import ProductBundleRestoreRequest
from tensorus.product_descriptor_admission import ProductDescriptorAdmissionCapabilities
from tensorus.product_portability import (
    ImportedProductRelease,
    PortableProductRelease,
    ProductReleaseImportRequest,
    _validate_source_product,
)
from tensorus.product_release_retirement import (
    MAX_PRODUCT_RELEASE_RETIREMENT_PAGE,
    MAX_PRODUCT_RELEASE_RETIREMENT_RECEIPT_BYTES,
    MAX_RETIRED_PRODUCT_RELEASE_BYTES,
    ProductReleaseRetirement,
    ProductReleaseRetirementRequest,
    ProductReleaseRetirementReviewRequest,
    ProductReleaseRetirementReviewResult,
    ProductReleaseRetirementSummary,
)
from tensorus.product_releases import (
    MAX_PORTABLE_PRODUCT_BYTES,
    ProductRelease,
    ProductReleaseRequest,
)
from tensorus.product_retirement import (
    MAX_PRODUCT_RETIREMENT_BYTES,
    MAX_PRODUCT_RETIREMENT_PAGE,
    MAX_PRODUCT_RETIREMENT_RECEIPT_BYTES,
    ProductRetirement,
    ProductRetirementRequest,
    ProductRetirementReviewRequest,
    ProductRetirementReviewResult,
    ProductRetirementSummary,
)
from tensorus.product_revision_holds import (
    ProductRevisionHold,
    ProductRevisionHoldCreate,
    ProductRevisionHoldHistory,
)
from tensorus.product_revision_retirement import (
    MAX_REVISION_RETIREMENT_BYTES,
    MAX_REVISION_RETIREMENT_PAGE,
    MAX_REVISION_RETIREMENT_RECEIPT_BYTES,
    ProductRevisionRetirement,
    ProductRevisionRetirementRequest,
    ProductRevisionRetirementReviewRequest,
    ProductRevisionRetirementReviewResult,
    ProductRevisionRetirementSummary,
)
from tensorus.product_similarity_evaluations import (
    ProductSimilarityEvaluation,
    ProductSimilarityEvaluationRequest,
    ProductSimilarityWithdrawal,
    ProductSimilarityWithdrawalRequest,
    ProductSimilarityWithdrawalSummary,
)
from tensorus.product_similarity_retirements import (
    ProductSimilarityRetirement,
    ProductSimilarityRetirementRequest,
    ProductSimilarityRetirementReviewRequest,
    ProductSimilarityRetirementReviewResult,
    ProductSimilarityRetirementSummary,
)
from tensorus.product_source_evaluations import (
    ProductSourceEvaluationBinding,
    ProductSourceEvaluationBindingRequest,
    ProductSourceEvaluationBindingSummary,
    ProductSourceEvaluationReviewRequest,
    ProductSourceEvaluationReviewResult,
)
from tensorus.products import MAX_PRODUCT_BYTES, DataProductRevision, DataProductWrite
from tensorus.quality_release_retirement import (
    MAX_QUALITY_RELEASE_RETIREMENT_BYTES,
    MAX_QUALITY_RELEASE_RETIREMENT_PAGE,
    MAX_QUALITY_RELEASE_RETIREMENT_RECEIPT_BYTES,
    MAX_QUALITY_RELEASE_RETIREMENTS_PER_TENANT,
    QualityReleaseRetirement,
    QualityReleaseRetirementRequest,
    QualityReleaseRetirementReviewRequest,
    QualityReleaseRetirementReviewResult,
    QualityReleaseRetirementSummary,
)
from tensorus.recovery_inventory import (
    MAX_INVENTORY_BATCH_RESPONSE_BYTES,
    MAX_INVENTORY_RESPONSE_BYTES,
    RecoveryInventory,
    RecoveryInventoryBatchReport,
    RecoveryInventoryBatchRequest,
    RecoveryInventoryCapabilities,
    RecoveryPartitionBatchReport,
    RecoveryPartitionBatchRequest,
    RecoveryPartitionCapabilities,
    RecoveryPartitionInventory,
    review_recovery_inventory_progress,
    validate_batch_inventory,
)
from tensorus.recovery_verification import (
    MAX_REMOTE_RECOVERY_RESPONSE_BYTES,
    RecoveryVerificationCapabilities,
    RecoveryVerificationReport,
    RecoveryVerificationRequest,
)
from tensorus.reference_evaluations import (
    ReferenceArchiveSummary,
    ReferenceDispositionRequest,
    ReferenceEvaluation,
    ReferenceEvaluationArchive,
    ReferenceEvaluationRequest,
    ReferenceEvaluationSummary,
    ReferenceRetirementRequest,
    ReferenceRetirementReviewResult,
)
from tensorus.release_portability import (
    MAX_PORTABLE_BYTES,
    MAX_PORTABLE_PLAN_SELECTIONS,
    MAX_PORTABLE_VALIDITY_SELECTIONS,
    ImportedHistoryIndexBackfillRequest,
    ImportedHistoryIndexDiagnosticRequest,
    ImportedHistoryIndexLookupRequest,
    ImportedHistoryIndexReconcileRequest,
    ImportedHistoryIndexReviewRequest,
    ImportedHistoryInspectionRequest,
    ImportedHistoryScanRequest,
    ImportedRelease,
    PortableHistoryExport,
    ReleaseHistoryBundleImport,
    ReleaseHistoryImport,
)
from tensorus.responses import (
    MAX_CAPABILITIES_RESPONSE_BYTES,
    AgentExecutionBudgetCapabilities,
    AgentExecutionBudgetStatus,
    AgentIngestBudget,
    AgentIngestBudgetCapabilities,
    AgentIngestBudgetStatus,
    AgentProposalBudgetCapabilities,
    AgentProposalBudgetStatus,
    ChunkedSupervisionCapabilities,
    ConditionalNpyHistoricalReceipt,
    ConditionalNpyHistoricalReceiptCapabilities,
    ConditionalNpyIngestCapabilities,
    ConditionalNpyReceipt,
    ConditionalNpyReceiptCapabilities,
    ControlRequestBodyCapabilities,
    CredentialPage,
    EagerIngestStagingCapabilities,
    FittedPreprocessingApplyExplanation,
    FittedPreprocessingFitExplanation,
    HttpPeerRateLimitCapabilities,
    HttpRateLimitCapabilities,
    Identity,
    IdentitySession,
    ImportedHistoryNameLookupCapabilities,
    IndexedSnapshotPage,
    IndexedTensorPage,
    IndexWatermarkChecksCapabilities,
    JsonRequestBodyCapabilities,
    KeyIssued,
    LocalEagerSupervisionCapabilities,
    NearestCoordinateCapabilities,
    NpyDownloadAdmissionCapabilities,
    Page,
    PlanStatusResponseFenceCapabilities,
    PortableExportResponseFenceCapabilities,
    ProductReadResponseFenceCapabilities,
    ProductReadResponseFenceV2Capabilities,
    ProductRetirementCapabilities,
    ProductRevisionHoldCapabilities,
    ProductRevisionRetirementCapabilities,
    QualityReleaseRetirementCapabilities,
    RemoteRequestBodyStagingCapabilities,
    RequestBodyStagingCapabilities,
    RequestBodyStagingDeadlineCapabilities,
    ScientificReadAdmissionCapabilities,
    ScientificReadAdmissionV2Capabilities,
    ScientificReadAdmissionV3Capabilities,
    ScientificReadAdmissionV4Capabilities,
    ScientificReadAdmissionV5Capabilities,
    ScientificReadAdmissionV6Capabilities,
    ScientificReadAdmissionV7Capabilities,
    ScientificReadAdmissionV8Capabilities,
    ScientificReadAdmissionV9Capabilities,
    ScientificReadAdmissionV10Capabilities,
    ScientificReadAdmissionV11Capabilities,
    ScopedMetadataDiscoveryCapabilities,
    SemanticIndexRebuild,
    SemanticIndexStatus,
    SimilarityAdmissionCapabilities,
    SnapshotCursorCapabilities,
    SnapshotCursorPage,
    SnapshotDetail,
    SnapshotRetirementReview,
    StreamedIndexEvaluationCapabilities,
    TenantFairSchedulingCapabilities,
    TensorVersion,
    Upload,
    UploadInputReservationCapabilities,
    ValidityTransformExplanation,
    WorkerOutputReservationStatus,
    WorkspaceLifecycle,
    WorkspaceLifecycleCapabilities,
    WorkspaceLifecycleExtendedCapabilities,
    WorkspaceLockdownExtended,
    WorkspaceLockdownReceipt,
    WorkspaceMemberListResponseFenceCapabilities,
    WorkspaceSessionRevocation,
    WorkspaceSessionRevocationCapabilities,
)
from tensorus.retention import (
    MAX_HOLDS_PER_RESOURCE,
    RetentionHold,
    RetentionHoldCreate,
    RetentionHoldHistory,
    RetentionHoldRelease,
    RetentionResource,
)
from tensorus.retirement_archive import RetirementArchiveSelection
from tensorus.semantics import (
    MAX_SHARED_AXIS_QUERY_COORDINATE_VALUES,
    OVERLAP_CROSS_SNAPSHOT_QUERY_FORMAT,
    OVERLAP_SHARED_AXIS_QUERY_FORMAT,
    SHARED_AXIS_QUERY_FORMAT,
    SharedAxisQueryMatcher,
    derive_semantics,
    matches_semantics,
    matches_shared_axis,
    read_semantics,
    referenced_nearest_coordinate_alignment,
)
from tensorus.serialization import (
    NPY_MAX_OVERHEAD_BYTES,
    dump_npy,
    load_npy,
    read_npy_header,
)
from tensorus.service_actor_grants import (
    MAX_CAPACITY_RESPONSE_BYTES,
    MAX_REVIEW_RESPONSE_BYTES,
    RecipientAgentGrantCreate,
    RecipientBindingReview,
    RecipientBindingReviewCapabilities,
    RecipientCapabilities,
    RecipientCapacityCapabilities,
    RecipientCapacityStatus,
    RecipientMultiDatasetAgentGrantCreate,
    RecipientReview,
    RecipientStatus,
)
from tensorus.service_actors import (
    MAX_PAGE_SIZE as MAX_SERVICE_ACTOR_PAGE_SIZE,
)
from tensorus.service_actors import (
    MAX_RESPONSE_BYTES as MAX_SERVICE_ACTOR_RESPONSE_BYTES,
)
from tensorus.service_actors import (
    MAX_SELF_RESPONSE_BYTES as MAX_SERVICE_ACTOR_SELF_RESPONSE_BYTES,
)
from tensorus.service_actors import (
    ServiceActorCapabilities,
    ServiceActorIdentity,
    ServiceActorInvitation,
    ServiceActorInvitationReceipt,
    ServiceActorMember,
    ServiceActorPage,
)
from tensorus.service_disclosures import (
    FIELDS as SERVICE_DISCLOSURE_FIELDS,
)
from tensorus.service_disclosures import (
    MAX_RESPONSE_BYTES as MAX_SERVICE_DISCLOSURE_RESPONSE_BYTES,
)
from tensorus.service_disclosures import (
    DisclosureEvent,
    DisclosureUpdate,
    DisclosureUpdateReceipt,
    DisclosureView,
)
from tensorus.snapshot_read_fence import SnapshotReadFenceCapabilities
from tensorus.snapshot_similarity import SnapshotSimilarityRequest
from tensorus.staging_status import (
    MAX_RESPONSE_BYTES as MAX_TENANT_STAGING_RESPONSE_BYTES,
)
from tensorus.staging_status import (
    TenantInputQuotaCapabilities,
    TenantInputQuotaStatus,
    TenantStagingCapabilities,
    TenantStagingReadAdmissionCapabilities,
    TenantStagingStatus,
    TenantWorkQuotaCapabilities,
    TenantWorkQuotaStatus,
)
from tensorus.tenant_admission import (
    MAX_RESPONSE_BYTES as MAX_TENANT_HTTP_ADMISSION_RESPONSE_BYTES,
)
from tensorus.tenant_admission import TenantHttpAdmission, TenantHttpAdmissionCapabilities
from tensorus.upload_chunk_admission import UploadChunkAdmissionCapabilities
from tensorus.upload_observations import FORMAT as UPLOAD_ATTEMPT_OBSERVATIONS_FORMAT
from tensorus.upload_observations import (
    MAX_PAGE_SIZE as MAX_UPLOAD_ATTEMPT_PAGE_SIZE,
)
from tensorus.upload_observations import (
    MAX_RESPONSE_BYTES as MAX_UPLOAD_ATTEMPT_RESPONSE_BYTES,
)
from tensorus.upload_observations import UploadAttemptCapabilities, UploadAttemptObservations
from tensorus.upload_preclaim import FORMAT as UPLOAD_PRECLAIM_FORMAT
from tensorus.upload_preclaim import HEADER as UPLOAD_PRECLAIM_HEADER
from tensorus.upload_preclaim import MAX_RESPONSE_BYTES as MAX_UPLOAD_PRECLAIM_RESPONSE_BYTES
from tensorus.upload_preclaim import PreclaimCapabilities, PreclaimRejection, parse_attempt_id
from tensorus.validity import (
    FittedPreprocessingApply,
    FittedPreprocessingFit,
    FittedPreprocessingState,
    ReviewedImportedFittedState,
    ValidityBinding,
    ValidityBindingCreate,
    ValidityBindingList,
    ValidityCoordinateWindow,
    ValidityTransform,
    ValidityTransformCompletion,
    _check_pair_descriptors,
    _descriptor_hash,
    _paired_definition,
)
from tensorus.validity_similarity import (
    ValiditySnapshotSimilarityReport,
    ValiditySnapshotSimilarityRequest,
)
from tensorus.worker_attempts import (
    MAX_RESPONSE_BYTES as MAX_WORKER_ATTEMPT_RESPONSE_BYTES,
)
from tensorus.worker_attempts import (
    STATUS_FORMAT as WORKER_ATTEMPT_STATUS_FORMAT,
)
from tensorus.worker_attempts import WorkerAttemptEvidence, WorkerAttemptEvidenceCapabilities
from tensorus.worker_usage import (
    FORMAT as CLAIM_ACCOUNTING_FORMAT,
)
from tensorus.worker_usage import (
    MAX_RESPONSE_BYTES as MAX_WORKER_CLAIM_RESPONSE_BYTES,
)
from tensorus.worker_usage import (
    WorkerCancellationRecoveryCapabilities,
    WorkerCancellationRecoveryV2Capabilities,
    WorkerClaimAccounting,
    WorkerClaimCapabilities,
)
from tensorus.workflow_inventory_admission import WorkflowInventoryAdmissionCapabilities
from tensorus.workspace_capacity import (
    MAX_RESPONSE_BYTES as MAX_WORKSPACE_CAPACITY_RESPONSE_BYTES,
)
from tensorus.workspace_capacity import (
    PlanCapacityReview,
    WorkspaceCapacity,
    WorkspaceCapacityCapabilities,
)
from tensorus.workspace_departure import (
    MAX_RESPONSE_BYTES as MAX_WORKSPACE_DEPARTURE_RESPONSE_BYTES,
)
from tensorus.workspace_departure import (
    DepartureCapabilities,
    DepartureReceipt,
    DepartureReceiptRecoveryCapabilities,
    DepartureRequest,
    RecoveredDepartureReceipt,
    validate_receipt_secret,
)
from tensorus.workspace_directory import (
    MAX_PAGE_SIZE as MAX_WORKSPACE_DIRECTORY_PAGE_SIZE,
)
from tensorus.workspace_directory import (
    MAX_RESPONSE_BYTES as MAX_WORKSPACE_DIRECTORY_RESPONSE_BYTES,
)
from tensorus.workspace_directory import WorkspaceDirectoryCapabilities, WorkspaceDirectoryPage
from tensorus.workspace_identity_sessions import (
    MAX_RESPONSE_BYTES as MAX_IDENTITY_SESSIONS_RESPONSE_BYTES,
)
from tensorus.workspace_identity_sessions import (
    IdentitySessionsCapabilities,
    IdentitySessionsStatus,
    IdentitySessionsTransition,
)
from tensorus.workspace_invitations import (
    MAX_RESPONSE_BYTES as MAX_WORKSPACE_INVITATION_RESPONSE_BYTES,
)
from tensorus.workspace_invitations import (
    InvitationClaim,
    InvitationIssue,
    InvitationOperatorStatus,
    InvitationReceipt,
    InvitationRevocationReceipt,
    InvitationRevoke,
    InvitationView,
    new_invitation_token,
    valid_invitation_token,
)
from tensorus.workspace_lockdown_receipt import (
    MAX_RESPONSE_BYTES as MAX_WORKSPACE_LOCKDOWN_RECEIPT_RESPONSE_BYTES,
)
from tensorus.workspace_lockdown_receipt import (
    LockdownReceiptCapabilities,
)
from tensorus.workspace_lockdown_receipt import (
    validate_receipt_secret as validate_lockdown_receipt_secret,
)
from tensorus.workspace_member_invitation import (
    MAX_RESPONSE_BYTES as MAX_WORKSPACE_MEMBER_INVITATION_RESPONSE_BYTES,
)
from tensorus.workspace_member_invitation import (
    MemberInvitation,
    MemberInvitationCapabilities,
    MemberInvitationReceipt,
)
from tensorus.workspace_member_transition import (
    MAX_RESPONSE_BYTES as MAX_WORKSPACE_MEMBER_TRANSITION_RESPONSE_BYTES,
)
from tensorus.workspace_member_transition import (
    MemberTransition,
    MemberTransitionCapabilities,
    MemberTransitionReceipt,
)
from tensorus.workspace_overview import (
    WorkspaceOverviewAdmissionCapabilities,
    WorkspaceOverviewResponseFenceCapabilities,
)
from tensorus.workspace_provisioning import (
    MAX_RESPONSE_BYTES as MAX_WORKSPACE_PROVISION_RESPONSE_BYTES,
)
from tensorus.workspace_provisioning import (
    WorkspaceProvisionReceipt,
    WorkspaceProvisionRequest,
    initial_key_id,
)
from tensorus.workspace_usage_comparison import (
    compare_references_for_workspace,
    parse_namespace,
    reference_from_bundle,
)
from tensorus.workspace_usage_comparison import (
    read_reference as read_workspace_usage_reference,
)
from tensorus.workspace_usage_comparison_v2 import (
    MAX_REFERENCE_BYTES as MAX_WORKSPACE_USAGE_REFERENCE_V2_BYTES,
)
from tensorus.workspace_usage_comparison_v2 import (
    compare_references as compare_mixed_workspace_usage_references,
)
from tensorus.workspace_usage_comparison_v2 import (
    read_reference as read_mixed_workspace_usage_reference,
)
from tensorus.workspace_usage_comparison_v2 import (
    reference_from_bundle as mixed_workspace_usage_reference,
)
from tensorus.workspace_usage_handoff import FORMAT as WORKSPACE_USAGE_HANDOFF_FORMAT
from tensorus.workspace_usage_handoff import (
    MAX_RESPONSE_BYTES as MAX_WORKSPACE_USAGE_HANDOFF_RESPONSE_BYTES,
)
from tensorus.workspace_usage_handoff import (
    BundleDocument,
    BundleRequest,
    WorkspaceUsageHandoff,
    WorkspaceUsageHandoffCapabilities,
)
from tensorus.workspace_usage_handoff_v2 import FORMAT as WORKSPACE_USAGE_HANDOFF_V2_FORMAT
from tensorus.workspace_usage_handoff_v2 import (
    MAX_RESPONSE_BYTES as MAX_WORKSPACE_USAGE_HANDOFF_V2_RESPONSE_BYTES,
)
from tensorus.workspace_usage_handoff_v2 import (
    MixedBundleDocument,
    MixedBundleRequest,
    MixedWorkspaceUsageHandoff,
    MixedWorkspaceUsageHandoffCapabilities,
)

MAX_NPY_RANGE_BYTES = 8 * 1024**2


@dataclass(frozen=True)
class NpyRange:
    offset: int
    total_bytes: int
    etag: str
    data: bytes


class Client:
    """Synchronous HTTP SDK. Mutation retries must reuse an explicit idempotency key."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 60,
        max_tensor_bytes: int = 64 * 1024**2,
        max_resumable_upload_bytes: int | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        base_url = validate_endpoint(base_url)
        if timeout <= 0 or max_tensor_bytes <= 0:
            raise ValueError("timeout and max_tensor_bytes must be positive")
        if max_resumable_upload_bytes is not None and (
            type(max_resumable_upload_bytes) is not int or max_resumable_upload_bytes <= 0
        ):
            raise ValueError("max_resumable_upload_bytes must be a positive integer")
        self.max_tensor_bytes = max_tensor_bytes
        self.max_resumable_upload_bytes = (
            max_tensor_bytes if max_resumable_upload_bytes is None else max_resumable_upload_bytes
        )
        self.http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout, connect=min(timeout, 10)),
            transport=transport,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.http.close()

    @staticmethod
    def _check_response(response: httpx.Response, payload: bytes | None = None) -> None:
        if not response.is_success:
            details: Record = {"request_id": response.headers.get("X-Request-ID")}
            if "Retry-After" in response.headers:
                details["retry_after"] = response.headers["Retry-After"]
            try:
                error = json.loads(payload if payload is not None else response.content)["error"]
                message, code = error["message"], error["code"]
                if error.get("details") is not None:
                    details["validation"] = error["details"]
            except (ValueError, KeyError, TypeError):
                message, code = f"HTTP {response.status_code}", "http_error"
            raise TensorusError(
                message,
                code=code,
                status=response.status_code,
                details=details,
            )

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self.http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise TensorusError(
                f"Tensorus connection failed: {type(exc).__name__}",
                code="connection_error",
                status=503,
            ) from exc
        self._check_response(response)
        return response

    def _checked_identity(self) -> Identity:
        return self._checked_identity_for_credential(None)

    def _checked_identity_for_credential(self, credential: str | None) -> Identity:
        encoded = self._bounded_report(
            "GET",
            "/v1/me",
            MAX_TENANT_STAGING_RESPONSE_BYTES,
            "Identity",
            headers={"Authorization": credential} if credential is not None else None,
        )
        try:
            raw = decode_unique_json(encoded)
            if not isinstance(raw, dict) or "scope" not in raw:
                raise ValueError("Identity scope evidence is missing")
            return Identity.model_validate_json(encoded, strict=True)
        except (ValidationError, TensorusError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "The server returned invalid identity evidence", code="invalid_response"
            ) from exc

    def _bounded_report(
        self,
        method: str,
        path: str,
        limit: int,
        label: str,
        *,
        headers: dict[str, str] | None = None,
        limit_exceeded: bool = False,
        **kwargs: Any,
    ) -> bytes:
        request_headers = {**(headers or {}), "Accept-Encoding": "identity"}
        try:
            with self.http.stream(method, path, headers=request_headers, **kwargs) as response:
                encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
                if encoding not in ("", "identity"):
                    raise TensorusError(
                        f"{label} response has unsupported content encoding",
                        code="invalid_response",
                    )
                payload = bytearray()
                for block in response.iter_bytes(chunk_size=8192):
                    if len(payload) + len(block) > limit:
                        if limit_exceeded:
                            raise LimitExceeded(f"{label} response exceeds the SDK byte limit")
                        raise TensorusError(
                            f"{label} response exceeds its byte limit",
                            code="invalid_response",
                        )
                    payload.extend(block)
                encoded = bytes(payload)
                self._check_response(response, encoded)
                return encoded
        except httpx.HTTPError as exc:
            raise TensorusError(
                f"Tensorus connection failed: {type(exc).__name__}",
                code="connection_error",
                status=503,
            ) from exc

    def investigate(
        self,
        dataset: str,
        goal: str,
        *,
        allow_external_observations: bool = False,
        expected_model_destination_sha256: str | None = None,
        expected_model_transport_sha256: str | None = None,
    ) -> Record:
        """Investigate with an external model; the result never grants execution authority."""
        if expected_model_destination_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", expected_model_destination_sha256
        ):
            raise ValueError("A valid model destination hash is required")
        if expected_model_transport_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", expected_model_transport_sha256
        ):
            raise ValueError("A valid model transport hash is required")
        if (
            expected_model_transport_sha256 is not None
            and expected_model_destination_sha256 is None
        ):
            raise ValueError("A model transport pin requires a destination pin")
        return self._request(
            "POST",
            "/v1/agent/investigate",
            json={
                "dataset": dataset,
                "goal": goal,
                "allow_external_observations": allow_external_observations,
                **(
                    {"expected_model_destination_sha256": expected_model_destination_sha256}
                    if expected_model_destination_sha256 is not None
                    else {}
                ),
                **(
                    {"expected_model_transport_sha256": expected_model_transport_sha256}
                    if expected_model_transport_sha256 is not None
                    else {}
                ),
            },
        ).json()

    def me(self) -> Record:
        return self._request("GET", "/v1/me").json()

    def model_destination(self) -> Record:
        """Review the configured endpoint and model before explicitly pinning disclosure."""
        identity, capabilities = self._checked_capabilities()
        if identity.scope is not None:
            raise Forbidden("Scoped agents cannot authorize external model disclosure")
        marker = capabilities.get("model_destination")
        if not isinstance(marker, dict):
            raise TensorusError(
                "The server does not advertise a required model destination pin",
                code="unsupported_feature",
            )
        return marker

    def model_investigation(self) -> Record:
        """Review the model investigation disclosure and its limits."""
        identity, capabilities = self._checked_capabilities()
        if identity.scope is not None:
            raise Forbidden("Scoped agents cannot authorize external model disclosure")
        marker = capabilities.get("model_investigation")
        if not isinstance(marker, dict):
            raise TensorusError(
                "The server does not advertise model investigation",
                code="unsupported_feature",
            )
        return marker

    def model_direct_transport(self) -> Record:
        """Review the direct model transport policy before explicitly pinning disclosure."""
        identity, capabilities = self._checked_capabilities()
        if identity.scope is not None:
            raise Forbidden("Scoped agents cannot authorize external model disclosure")
        marker = capabilities.get("model_direct_transport")
        if not isinstance(marker, dict):
            raise TensorusError(
                "The server does not advertise direct model transport",
                code="unsupported_feature",
            )
        return marker

    def agent_proposal_budget(self) -> Record:
        """Inspect this scoped agent's committed-proposal allowance."""
        identity, capabilities = self._checked_capabilities()
        if identity.scope is None:
            raise Forbidden("A scoped agent credential is required")
        if "agent_proposal_budget" not in capabilities:
            raise TensorusError(
                "The server does not advertise agent proposal allowances",
                code="unsupported_feature",
            )
        encoded = self._bounded_report(
            "GET", "/v1/agent/proposal-budget", 4096, "Agent proposal allowance"
        )
        try:
            raw = decode_unique_json(encoded)
            status = AgentProposalBudgetStatus.model_validate_json(encoded, strict=True)
            result = status.model_dump(mode="json")
            if not isinstance(raw, dict) or canonical_json(raw) != canonical_json(result):
                raise ValueError("Agent proposal allowance is not exact")
        except (ValidationError, TensorusError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "The server returned invalid agent proposal allowance evidence",
                code="invalid_response",
            ) from exc
        if self._checked_identity() != identity:
            raise Forbidden("The authenticated identity changed during allowance inspection")
        return result

    def agent_execution_budget(self) -> Record:
        """Inspect this scoped agent's estimated worker-attempt allowance."""
        identity, capabilities = self._checked_capabilities()
        if identity.scope is None:
            raise Forbidden("A scoped agent credential is required")
        if "agent_execution_budget" not in capabilities:
            raise TensorusError(
                "The server does not advertise agent execution allowances",
                code="unsupported_feature",
            )
        encoded = self._bounded_report(
            "GET", "/v1/agent/execution-budget", 4096, "Agent execution allowance"
        )
        try:
            raw = decode_unique_json(encoded)
            status = AgentExecutionBudgetStatus.model_validate_json(encoded, strict=True)
            result = status.model_dump(mode="json")
            if not isinstance(raw, dict) or canonical_json(raw) != canonical_json(result):
                raise ValueError("Agent execution allowance is not exact")
        except (ValidationError, TensorusError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "The server returned invalid agent execution allowance evidence",
                code="invalid_response",
            ) from exc
        if self._checked_identity() != identity:
            raise Forbidden("The authenticated identity changed during allowance inspection")
        return result

    def agent_ingest_budget(self) -> Record:
        """Inspect this scoped agent's committed direct-ingestion byte allowance."""
        identity, capabilities = self._checked_capabilities()
        if identity.scope is None:
            raise Forbidden("A scoped agent credential is required")
        if "agent_ingest_budget" not in capabilities:
            raise TensorusError(
                "The server does not advertise agent ingestion allowances",
                code="unsupported_feature",
            )
        encoded = self._bounded_report(
            "GET", "/v1/agent/ingest-budget", 4096, "Agent ingestion allowance"
        )
        try:
            raw = decode_unique_json(encoded)
            status = AgentIngestBudgetStatus.model_validate_json(encoded, strict=True)
            result = status.model_dump(mode="json")
            if not isinstance(raw, dict) or canonical_json(raw) != canonical_json(result):
                raise ValueError("Agent ingestion allowance is not exact")
        except (ValidationError, TensorusError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "The server returned invalid agent ingestion allowance evidence",
                code="invalid_response",
            ) from exc
        if self._checked_identity() != identity:
            raise Forbidden("The authenticated identity changed during allowance inspection")
        return result

    def agent_upload_staging(self) -> Record:
        """Inspect this scoped agent's declared resumable-upload reservations."""
        identity, capabilities = self._checked_capabilities()
        if identity.scope is None:
            raise Forbidden("A scoped agent credential is required")
        marker = capabilities.get("agent_upload_staging")
        if marker is None:
            raise TensorusError(
                "The server does not advertise agent upload staging limits",
                code="unsupported_feature",
            )
        policy = AgentUploadStagingCapabilities.model_validate(marker, strict=True)
        encoded = self._bounded_report(
            "GET",
            "/v1/agent/upload-staging",
            MAX_AGENT_UPLOAD_STAGING_RESPONSE_BYTES,
            "Agent upload staging",
        )
        try:
            raw = decode_unique_json(encoded)
            status = AgentUploadStagingStatus.model_validate_json(encoded, strict=True)
            result = status.model_dump(mode="json")
            if (
                not isinstance(raw, dict)
                or canonical_json(raw) != canonical_json(result)
                or status.tenant_id != identity.id
                or status.key_id != identity.key_id
                or status.max_staged_bytes != policy.max_staged_bytes
                or status.max_open_sessions != policy.max_open_sessions
            ):
                raise ValueError("Agent upload staging status is not exact")
        except (ValidationError, TensorusError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "The server returned invalid agent upload staging evidence",
                code="invalid_response",
            ) from exc
        if self._checked_identity() != identity:
            raise Forbidden("The authenticated identity changed during staging inspection")
        return result

    def agent_grant_upload_staging(self, key_id: str) -> Record:
        """Review one agent grant's recorded resumable-upload reservations as an administrator."""
        target = UUID(key_id)
        identity, capabilities = self._checked_capabilities()
        if identity.scope is not None or identity.role != Role.admin:
            raise Forbidden("An unscoped workspace administrator is required")
        if "agent_upload_staging_admin" not in capabilities:
            raise TensorusError(
                "The server does not advertise administrator upload staging review",
                code="unsupported_feature",
            )
        policy = AgentUploadStagingCapabilities.model_validate(
            capabilities["agent_upload_staging"], strict=True
        )
        encoded = self._bounded_report(
            "GET",
            f"/v1/agent/grants/{target}/upload-staging",
            MAX_AGENT_UPLOAD_STAGING_RESPONSE_BYTES,
            "Agent upload staging review",
        )
        try:
            raw = decode_unique_json(encoded)
            report = AgentUploadStagingAdminStatus.model_validate_json(encoded, strict=True)
            result = report.model_dump(mode="json")
            status = report.status
            if (
                not isinstance(raw, dict)
                or canonical_json(raw) != canonical_json(result)
                or status.tenant_id != identity.id
                or status.key_id != target
                or status.max_staged_bytes != policy.max_staged_bytes
                or status.max_open_sessions != policy.max_open_sessions
            ):
                raise ValueError("Agent upload staging review is not exact")
        except (ValidationError, TensorusError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "The server returned invalid agent upload staging review evidence",
                code="invalid_response",
            ) from exc
        if self._checked_identity() != identity:
            raise Forbidden("The administrator changed during staging review")
        return result

    def stats(self) -> Record:
        return self._request("GET", "/v1/stats").json()

    def insights(self, *, days: int = 7) -> Record:
        return self._request("GET", "/v1/insights", params={"days": days}).json()

    def _capacity_identity(self) -> Identity:
        identity, capabilities = self._checked_capabilities()
        if identity.role != Role.admin or identity.scope is not None:
            raise Forbidden("Workspace capacity requires an active unscoped administrator")
        if "workspace_capacity" not in capabilities:
            raise TensorusError(
                "The server does not advertise workspace capacity review",
                code="unsupported_feature",
            )
        return identity

    def workspace_capacity(self) -> Record:
        """Inspect tenant-local catalog quotas and claim counts, not a physical bill."""
        identity = self._capacity_identity()
        encoded = self._bounded_report(
            "GET", "/v1/workspace/capacity", MAX_WORKSPACE_CAPACITY_RESPONSE_BYTES, "Capacity"
        )
        result = self._checked_exact_response(encoded, WorkspaceCapacity, "workspace capacity")
        if result.tenant_id != identity.id:
            raise TensorusError("The server returned capacity for a different workspace")
        if self._checked_identity() != identity:
            raise Forbidden("The workspace identity changed during capacity inspection")
        return result.model_dump(mode="json")

    def plan_capacity_review(self, plan_id: str, expected_plan_hash: str) -> Record:
        """Review a proposed plan's predicted retained outputs without reserving quota."""
        requested_id = UUID(plan_id)
        if not re.fullmatch(r"[0-9a-f]{64}", expected_plan_hash):
            raise ValueError("A lowercase SHA-256 plan hash is required")
        identity = self._capacity_identity()
        encoded = self._bounded_report(
            "GET",
            f"/v1/plans/{quote(plan_id, safe='')}/capacity-review",
            MAX_WORKSPACE_CAPACITY_RESPONSE_BYTES,
            "Plan capacity review",
            params={"expected_plan_hash": expected_plan_hash},
        )
        result = self._checked_exact_response(encoded, PlanCapacityReview, "plan capacity review")
        if (
            result.tenant_id != identity.id
            or result.plan_id != requested_id
            or result.plan_hash != expected_plan_hash
        ):
            raise TensorusError("The server returned capacity for a different plan")
        if self._checked_identity() != identity:
            raise Forbidden("The workspace identity changed during capacity review")
        return result.model_dump(mode="json")

    def _pricing_identity(self) -> Identity:
        identity, capabilities = self._checked_capabilities()
        if identity.role != Role.admin or identity.scope is not None:
            raise Forbidden("Advisory pricing requires an active unscoped administrator")
        if "advisory_pricing" not in capabilities:
            raise TensorusError(
                "The server does not advertise advisory pricing",
                code="unsupported_feature",
            )
        return identity

    def workspace_pricing(self) -> Record:
        """Inspect the shared operator-configured rates; no provider or charge is involved."""
        identity = self._pricing_identity()
        encoded = self._bounded_report(
            "GET", "/v1/workspace/pricing", MAX_ADVISORY_PRICING_RESPONSE_BYTES, "Advisory pricing"
        )
        result = self._checked_exact_response(encoded, AdvisoryPricingStatus, "advisory pricing")
        if result.tenant_id != identity.id:
            raise TensorusError("The server returned pricing for a different workspace")
        if self._checked_identity() != identity:
            raise Forbidden("The workspace identity changed during pricing inspection")
        return result.model_dump(mode="json")

    def plan_price_scenario(self, plan_id: str, expected_plan_hash: str) -> Record:
        """Estimate hypothetical output prices without measuring use or creating charges."""
        requested_id = UUID(plan_id)
        if not re.fullmatch(r"[0-9a-f]{64}", expected_plan_hash):
            raise ValueError("A lowercase SHA-256 plan hash is required")
        identity = self._pricing_identity()
        encoded = self._bounded_report(
            "GET",
            f"/v1/plans/{quote(plan_id, safe='')}/price-scenario",
            MAX_ADVISORY_PRICING_RESPONSE_BYTES,
            "Plan price scenario",
            params={"expected_plan_hash": expected_plan_hash},
        )
        result = self._checked_exact_response(encoded, PlanPriceScenario, "plan price scenario")
        if (
            result.tenant_id != identity.id
            or result.plan_id != requested_id
            or result.plan_hash != expected_plan_hash
        ):
            raise TensorusError("The server returned pricing for a different plan")
        if self._checked_identity() != identity:
            raise Forbidden("The workspace identity changed during pricing inspection")
        return result.model_dump(mode="json")

    def capabilities(self, *, _credential: str | None = None) -> Record:
        encoded = self._bounded_report(
            "GET",
            "/v1/capabilities",
            MAX_CAPABILITIES_RESPONSE_BYTES,
            "Capabilities",
            headers={"Authorization": _credential} if _credential is not None else None,
        )
        try:
            record = decode_unique_json(encoded)
            if not isinstance(record, dict):
                raise ValueError("Capabilities must be an object")
            if "portable_export_response_fence" in record:
                marker = record["portable_export_response_fence"]
                portable_fence = PortableExportResponseFenceCapabilities.model_validate(
                    marker, strict=True
                )
                if canonical_json(marker) != canonical_json(portable_fence.model_dump(mode="json")):
                    raise ValueError("Portable export response fence capabilities are not exact")
            if "plan_status_response_fence" in record:
                marker = record["plan_status_response_fence"]
                plan_fence = PlanStatusResponseFenceCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(plan_fence.model_dump(mode="json")):
                    raise ValueError("Plan status response fence capabilities are not exact")
            if "workspace_member_list_response_fence" in record:
                marker = record["workspace_member_list_response_fence"]
                member_fence = WorkspaceMemberListResponseFenceCapabilities.model_validate(
                    marker, strict=True
                )
                if canonical_json(marker) != canonical_json(member_fence.model_dump(mode="json")):
                    raise ValueError("Membership list response fence capabilities are not exact")
            product_fence_formats = (
                "product_read_response_fence",
                "product_read_response_fence_v2",
            )
            if sum(name in record for name in product_fence_formats) > 1:
                raise ValueError("Product read response fence formats are contradictory")
            for name, fence_model in (
                ("product_read_response_fence", ProductReadResponseFenceCapabilities),
                ("product_read_response_fence_v2", ProductReadResponseFenceV2Capabilities),
            ):
                if name not in record:
                    continue
                marker = record[name]
                product_fence = fence_model.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(product_fence.model_dump(mode="json")):
                    raise ValueError("Product read response fence capabilities are not exact")
            if "scoped_metadata_discovery" in record:
                marker = record["scoped_metadata_discovery"]
                discovery = ScopedMetadataDiscoveryCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(discovery.model_dump(mode="json")):
                    raise ValueError("Scoped metadata discovery capabilities are not exact")
            if "plan_proposer_authority" in record:
                marker = record["plan_proposer_authority"]
                authority = PlanProposerAuthorityCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(
                    authority.model_dump(mode="json")
                ) or not matches_plan_schema(record.get("plan_schema")):
                    raise ValueError("Plan proposer authority capabilities are not exact")
            if "model_destination" in record:
                marker = record["model_destination"]
                destination = ModelDestinationCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(destination.model_dump(mode="json")):
                    raise ValueError("Model destination capabilities are not exact")
            if "model_investigation" in record:
                marker = record["model_investigation"]
                investigation = InvestigationCapabilities.model_validate(marker, strict=True)
                if (
                    canonical_json(marker) != canonical_json(investigation.model_dump(mode="json"))
                    or record.get("llm_planning") is not True
                    or record.get("scope") is not None
                ):
                    raise ValueError("Model investigation capabilities are inconsistent")
            if "model_direct_transport" in record:
                marker = record["model_direct_transport"]
                transport = DirectTransportCapabilities.model_validate(marker, strict=True)
                destination_marker = record.get("model_destination")
                if (
                    canonical_json(marker) != canonical_json(transport.model_dump(mode="json"))
                    or not isinstance(destination_marker, dict)
                    or record.get("llm_planning") is not True
                    or transport.policy.destination_sha256
                    != destination_marker.get("destination_sha256")
                ):
                    raise ValueError("Model direct transport capabilities are inconsistent")
            schema = record.get("plan_schema")
            plan_limit = (
                schema.get("x-tensorus-max-review-request-bytes")
                if isinstance(schema, dict)
                else None
            )
            if "workspace_member_invitation" in record:
                marker = record["workspace_member_invitation"]
                invitation = MemberInvitationCapabilities.model_validate(marker, strict=True)
                if (
                    canonical_json(marker) != canonical_json(invitation.model_dump(mode="json"))
                    or type(plan_limit) is not int
                    or invitation.max_request_bytes != min(MAX_CONTROL_REQUEST_BYTES, plan_limit)
                ):
                    raise ValueError("Workspace invitation capabilities are inconsistent")
            if "workspace_member_transition" in record:
                marker = record["workspace_member_transition"]
                transition = MemberTransitionCapabilities.model_validate(marker, strict=True)
                if (
                    canonical_json(marker) != canonical_json(transition.model_dump(mode="json"))
                    or type(plan_limit) is not int
                    or transition.max_request_bytes != min(MAX_CONTROL_REQUEST_BYTES, plan_limit)
                ):
                    raise ValueError("Workspace transition capabilities are inconsistent")
            if "request_body_staging" in record and "request_body_staging_v2" in record:
                raise ValueError("Request staging capability formats are contradictory")
            staging_marker = record.get(
                "request_body_staging", record.get("request_body_staging_v2")
            )
            for name, staging_model in (
                ("request_body_staging", RequestBodyStagingCapabilities),
                ("request_body_staging_v2", RemoteRequestBodyStagingCapabilities),
            ):
                if name in record:
                    marker = record[name]
                    checked_staging = staging_model.model_validate(marker, strict=True)
                    if canonical_json(marker) != canonical_json(
                        checked_staging.model_dump(mode="json")
                    ):
                        raise ValueError("Request staging capabilities are not exact")
            if "json_request_body_limit" in record:
                marker = record["json_request_body_limit"]
                json_limit = JsonRequestBodyCapabilities.model_validate(marker, strict=True)
                if (
                    canonical_json(marker) != canonical_json(json_limit.model_dump(mode="json"))
                    or type(plan_limit) is not int
                    or plan_limit != min(MAX_PLAN_REVIEW_REQUEST_BYTES, json_limit.max_body_bytes)
                ):
                    raise ValueError("JSON request body capabilities are inconsistent")
                staging = staging_marker
                if isinstance(staging, dict) and (
                    staging.get("max_content_bytes_per_request") != json_limit.global_body_bytes
                ):
                    raise ValueError("JSON and request staging limits are inconsistent")
            if "http_body_admission" in record:
                marker = record["http_body_admission"]
                body_admission = HttpBodyAdmissionCapabilities.model_validate(marker, strict=True)
                staging = staging_marker
                if (
                    canonical_json(marker) != canonical_json(body_admission.model_dump(mode="json"))
                    or not isinstance(staging, dict)
                    or staging.get("max_content_bytes_per_request") != body_admission.max_body_bytes
                ):
                    raise ValueError("HTTP body admission capabilities are inconsistent")
            if "control_request_body_limit" in record:
                marker = record["control_request_body_limit"]
                control = ControlRequestBodyCapabilities.model_validate(marker, strict=True)
                if (
                    canonical_json(marker) != canonical_json(control.model_dump(mode="json"))
                    or type(plan_limit) is not int
                    or control.max_body_bytes != min(MAX_CONTROL_REQUEST_BYTES, plan_limit)
                ):
                    raise ValueError("Control request body capabilities are inconsistent")
            if "bundle_restore_inspection" in record:
                from tensorus.bundle_inspection_contract import MAX_BODY_BYTES
                from tensorus.history_payload_bundle import MAX_HISTORY_PAYLOAD_BUNDLE_BYTES
                from tensorus.product_bundle import MAX_BUNDLE_BYTES

                marker = record["bundle_restore_inspection"]
                inspection = BundleInspectionCapabilities.model_validate(marker, strict=True)
                staging = staging_marker
                if (
                    not isinstance(staging, dict)
                    or type(staging.get("concurrent_slots")) is not int
                    or staging["concurrent_slots"] < 2
                    or type(staging.get("max_content_bytes_per_request")) is not int
                    or inspection.max_body_bytes
                    != min(MAX_BODY_BYTES, staging["max_content_bytes_per_request"])
                    or inspection.max_product_bundle_bytes != MAX_BUNDLE_BYTES
                    or inspection.max_history_bundle_bytes != MAX_HISTORY_PAYLOAD_BUNDLE_BYTES
                    or canonical_json(marker) != canonical_json(inspection.model_dump(mode="json"))
                ):
                    raise ValueError("Bundle inspection capabilities are inconsistent")
            if "bundle_restore_apply" in record:
                marker = record["bundle_restore_apply"]
                apply = BundleRestoreApplyCapabilities.model_validate(marker, strict=True)
                if "bundle_restore_inspection" not in record or canonical_json(
                    marker
                ) != canonical_json(apply.model_dump(mode="json")):
                    raise ValueError("Bundle restore apply capabilities are inconsistent")
            if "bundle_restore_head_preflight" in record:
                marker = record["bundle_restore_head_preflight"]
                preflight = BundleHeadPreflightCapabilities.model_validate(marker, strict=True)
                if (
                    "bundle_restore_inspection" not in record
                    or preflight.server_apply_checks_heads != ("bundle_restore_apply" in record)
                    or canonical_json(marker) != canonical_json(preflight.model_dump(mode="json"))
                ):
                    raise ValueError("Bundle head preflight capabilities are inconsistent")
            if "bundle_restore_payload_observation" in record:
                marker = record["bundle_restore_payload_observation"]
                observation = BundlePayloadObservationCapabilities.model_validate(
                    marker, strict=True
                )
                if "bundle_restore_inspection" not in record or canonical_json(
                    marker
                ) != canonical_json(observation.model_dump(mode="json")):
                    raise ValueError("Bundle payload observation capabilities are inconsistent")
            if "history_release_observation" in record:
                marker = record["history_release_observation"]
                release_observation = HistoryReleaseObservationCapabilities.model_validate(
                    marker, strict=True
                )
                if (
                    "bundle_restore_inspection" not in record
                    or "bundle_restore_payload_observation" not in record
                    or canonical_json(marker)
                    != canonical_json(release_observation.model_dump(mode="json"))
                ):
                    raise ValueError("History release observation capabilities are inconsistent")
            if "history_restore_selection" in record:
                marker = record["history_restore_selection"]
                selection = HistoryRestoreSelectionCapabilities.model_validate(marker, strict=True)
                if (
                    "bundle_restore_inspection" not in record
                    or "bundle_restore_name_occupancy" not in record
                    or selection.server_apply_enabled != ("bundle_restore_apply" in record)
                    or canonical_json(marker) != canonical_json(selection.model_dump(mode="json"))
                ):
                    raise ValueError("Selected history capabilities are inconsistent")
            if "history_independent_selection" in record:
                marker = record["history_independent_selection"]
                independent_selection = HistoryIndependentSelectionCapabilities.model_validate(
                    marker, strict=True
                )
                if (
                    "bundle_restore_inspection" not in record
                    or "bundle_restore_name_occupancy" not in record
                    or "history_payload_selection" not in record
                    or independent_selection.server_apply_enabled
                    != ("bundle_restore_apply" in record)
                    or canonical_json(marker)
                    != canonical_json(independent_selection.model_dump(mode="json"))
                ):
                    raise ValueError("Independent history capabilities are inconsistent")
            if "history_independent_observation" in record:
                marker = record["history_independent_observation"]
                independent_observation = HistoryIndependentObservationCapabilities.model_validate(
                    marker, strict=True
                )
                if (
                    "bundle_restore_inspection" not in record
                    or "bundle_restore_name_occupancy" not in record
                    or "history_payload_selection" not in record
                    or canonical_json(marker)
                    != canonical_json(independent_observation.model_dump(mode="json"))
                ):
                    raise ValueError(
                        "Independent history observation capabilities are inconsistent"
                    )
            if "history_payload_selection" in record:
                marker = record["history_payload_selection"]
                payload_selection = HistoryPayloadSelectionCapabilities.model_validate(
                    marker, strict=True
                )
                if (
                    "bundle_restore_inspection" not in record
                    or payload_selection.server_apply_enabled != ("bundle_restore_apply" in record)
                    or canonical_json(marker)
                    != canonical_json(payload_selection.model_dump(mode="json"))
                ):
                    raise ValueError("Selected history payload capabilities are inconsistent")
            if "product_payload_selection" in record:
                marker = record["product_payload_selection"]
                product_selection = ProductPayloadSelectionCapabilities.model_validate(
                    marker, strict=True
                )
                if (
                    "bundle_restore_inspection" not in record
                    or product_selection.server_apply_enabled != ("bundle_restore_apply" in record)
                    or canonical_json(marker)
                    != canonical_json(product_selection.model_dump(mode="json"))
                ):
                    raise ValueError("Selected product payload capabilities are inconsistent")
            if "bundle_restore_name_occupancy" in record:
                marker = record["bundle_restore_name_occupancy"]
                names = NameOccupancyCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(names.model_dump(mode="json")):
                    raise ValueError("Bundle name observation capabilities are inconsistent")
            if "bundle_restore_name_preflight" in record:
                marker = record["bundle_restore_name_preflight"]
                names_preflight = NamePreflightCapabilities.model_validate(marker, strict=True)
                if (
                    "bundle_restore_name_occupancy" not in record
                    or "bundle_restore_head_preflight" not in record
                    or canonical_json(marker)
                    != canonical_json(names_preflight.model_dump(mode="json"))
                ):
                    raise ValueError("Bundle name preflight capabilities are inconsistent")
            if "bundle_restore_preparation" in record:
                marker = record["bundle_restore_preparation"]
                preparation = BundleRestorePreparationCapabilities.model_validate(
                    marker, strict=True
                )
                if "bundle_restore_inspection" not in record or canonical_json(
                    marker
                ) != canonical_json(preparation.model_dump(mode="json")):
                    raise ValueError("Bundle preparation capabilities are inconsistent")
            if "recovery_verification" in record:
                RecoveryVerificationCapabilities.model_validate(
                    record["recovery_verification"], strict=True
                )
            if "recovery_inventory" in record:
                RecoveryInventoryCapabilities.model_validate(
                    record["recovery_inventory"], strict=True
                )
            if "recovery_inventory_partition" in record:
                marker = record["recovery_inventory_partition"]
                partition = RecoveryPartitionCapabilities.model_validate(marker, strict=True)
                if "recovery_inventory" not in record or canonical_json(marker) != canonical_json(
                    partition.model_dump(mode="json")
                ):
                    raise ValueError("Recovery partition capabilities are inconsistent")
            if "tenant_staging_reservations" in record:
                TenantStagingCapabilities.model_validate(
                    record["tenant_staging_reservations"], strict=True
                )
            if "tenant_staging_work_quota" in record:
                TenantWorkQuotaCapabilities.model_validate(
                    record["tenant_staging_work_quota"], strict=True
                )
            if "tenant_staging_input_quota" in record:
                marker = record["tenant_staging_input_quota"]
                input_quota = TenantInputQuotaCapabilities.model_validate(marker, strict=True)
                if (
                    record.get("role") != Role.admin
                    or record.get("scope") is not None
                    or "tenant_staging_reservations" not in record
                    or canonical_json(marker) != canonical_json(input_quota.model_dump(mode="json"))
                ):
                    raise ValueError("Staging input quota capabilities are inconsistent")
            if "worker_attempt_evidence" in record:
                WorkerAttemptEvidenceCapabilities.model_validate(
                    record["worker_attempt_evidence"], strict=True
                )
            if "worker_claim_accounting" in record:
                marker = record["worker_claim_accounting"]
                claims = WorkerClaimCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(claims.model_dump(mode="json")):
                    raise ValueError("Worker claim capabilities are not exact")
            if "worker_cancellation_recovery" in record:
                marker = record["worker_cancellation_recovery"]
                recovery = WorkerCancellationRecoveryCapabilities.model_validate(
                    marker, strict=True
                )
                if "worker_claim_accounting" not in record or canonical_json(
                    marker
                ) != canonical_json(recovery.model_dump(mode="json")):
                    raise ValueError("Worker cancellation recovery capabilities are not exact")
            if "worker_cancellation_recovery_v2" in record:
                marker = record["worker_cancellation_recovery_v2"]
                recovery_v2 = WorkerCancellationRecoveryV2Capabilities.model_validate(
                    marker, strict=True
                )
                if "worker_cancellation_recovery" not in record or canonical_json(
                    marker
                ) != canonical_json(recovery_v2.model_dump(mode="json")):
                    raise ValueError("Worker cancellation recovery v2 capabilities are not exact")
            if "plan_estimated_attempts" in record:
                marker = record["plan_estimated_attempts"]
                estimate = PlanEstimatedAttemptsCapabilities.model_validate(marker, strict=True)
                if (
                    "worker_claim_accounting" not in record
                    or (
                        estimate.separate_scientific_read_admission_enabled
                        and not any(
                            name in record
                            for name in (
                                "scientific_read_admission",
                                *(
                                    f"scientific_read_admission_v{version}"
                                    for version in range(2, 12)
                                ),
                            )
                        )
                    )
                    or canonical_json(marker) != canonical_json(estimate.model_dump(mode="json"))
                ):
                    raise ValueError("Estimated-attempt capabilities are not exact")
            if "plan_usage_observations" in record:
                marker = record["plan_usage_observations"]
                usage = PlanUsageCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(usage.model_dump(mode="json")):
                    raise ValueError("Plan usage capabilities are not exact")
            if "ingest_usage_observations" in record:
                marker = record["ingest_usage_observations"]
                ingest_usage = IngestUsageCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(ingest_usage.model_dump(mode="json")):
                    raise ValueError("Ingestion observation capabilities are not exact")
            if "upload_attempt_observations" in record:
                marker = record["upload_attempt_observations"]
                upload_usage = UploadAttemptCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(upload_usage.model_dump(mode="json")):
                    raise ValueError("Upload observation capabilities are not exact")
            if "upload_preclaim_rejections" in record:
                marker = record["upload_preclaim_rejections"]
                preclaim = PreclaimCapabilities.model_validate(marker, strict=True)
                attempt_marker = record.get("upload_attempt_observations")
                if (
                    not isinstance(attempt_marker, dict)
                    or attempt_marker.get("shared_scientific_read_admission_enabled")
                    != preclaim.shared_scientific_read_admission_enabled
                    or canonical_json(marker) != canonical_json(preclaim.model_dump(mode="json"))
                ):
                    raise ValueError("Upload preclaim capabilities are not exact")
            if "plan_process_usage_observations" in record:
                marker = record["plan_process_usage_observations"]
                process_usage = PlanProcessUsageCapabilities.model_validate(marker, strict=True)
                if "plan_usage_observations" not in record or canonical_json(
                    marker
                ) != canonical_json(process_usage.model_dump(mode="json")):
                    raise ValueError("Plan process usage capabilities are not exact")
            if "plan_usage_handoff" in record:
                marker = record["plan_usage_handoff"]
                handoff = PlanUsageHandoffCapabilities.model_validate(marker, strict=True)
                if (
                    "plan_usage_observations" not in record
                    or "plan_process_usage_observations" not in record
                    or canonical_json(marker) != canonical_json(handoff.model_dump(mode="json"))
                ):
                    raise ValueError("Plan usage handoff capabilities are not exact")
            if "plan_usage_adjustments" in record:
                marker = record["plan_usage_adjustments"]
                adjustments = AdjustmentCapabilities.model_validate(marker, strict=True)
                if "plan_usage_observations" not in record or canonical_json(
                    marker
                ) != canonical_json(adjustments.model_dump(mode="json")):
                    raise ValueError("Plan usage adjustment capabilities are not exact")
            if "plan_adjusted_usage_handoff" in record:
                marker = record["plan_adjusted_usage_handoff"]
                adjusted = AdjustedHandoffCapabilities.model_validate(marker, strict=True)
                if (
                    "plan_usage_handoff" not in record
                    or "plan_usage_adjustments" not in record
                    or canonical_json(marker) != canonical_json(adjusted.model_dump(mode="json"))
                ):
                    raise ValueError("Adjusted usage handoff capabilities are not exact")
            if "workspace_usage_handoff" in record:
                marker = record["workspace_usage_handoff"]
                workspace_handoff = WorkspaceUsageHandoffCapabilities.model_validate(
                    marker, strict=True
                )
                if (
                    "plan_adjusted_usage_handoff" not in record
                    or type(plan_limit) is not int
                    or workspace_handoff.max_request_bytes
                    != min(MAX_CONTROL_REQUEST_BYTES, plan_limit)
                    or workspace_handoff.separate_scientific_read_admission_enabled
                    != ("scientific_read_admission_v11" in record)
                    or canonical_json(marker)
                    != canonical_json(workspace_handoff.model_dump(mode="json"))
                ):
                    raise ValueError("Workspace usage handoff capabilities are not exact")
            if "workspace_usage_handoff_v2" in record:
                marker = record["workspace_usage_handoff_v2"]
                mixed_handoff = MixedWorkspaceUsageHandoffCapabilities.model_validate(
                    marker, strict=True
                )
                if (
                    "workspace_usage_handoff" not in record
                    or "ingest_usage_observations" not in record
                    or type(plan_limit) is not int
                    or mixed_handoff.max_request_bytes != min(MAX_CONTROL_REQUEST_BYTES, plan_limit)
                    or mixed_handoff.separate_scientific_read_admission_enabled
                    != ("scientific_read_admission_v11" in record)
                    or canonical_json(marker)
                    != canonical_json(mixed_handoff.model_dump(mode="json"))
                ):
                    raise ValueError("Mixed usage handoff capabilities are not exact")
            if "local_usage_fixture" in record:
                marker = record["local_usage_fixture"]
                fixture = FixtureCapabilities.model_validate(marker, strict=True)
                if (
                    "workspace_usage_handoff_v2" not in record
                    or type(plan_limit) is not int
                    or fixture.max_request_bytes != min(MAX_CONTROL_REQUEST_BYTES, plan_limit)
                    or fixture.separate_scientific_read_admission_enabled
                    != ("scientific_read_admission_v11" in record)
                    or canonical_json(marker) != canonical_json(fixture.model_dump(mode="json"))
                ):
                    raise ValueError("Local usage fixture capabilities are not exact")
            if "local_usage_fixture_inspection" in record:
                marker = record["local_usage_fixture_inspection"]
                fixture_inspection = FixtureFileInspectionCapabilities.model_validate(
                    marker, strict=True
                )
                if (
                    "local_usage_fixture" not in record
                    or type(plan_limit) is not int
                    or fixture_inspection.max_request_bytes
                    != min(MAX_CONTROL_REQUEST_BYTES, plan_limit)
                    or fixture_inspection.separate_scientific_read_admission_enabled
                    != ("scientific_read_admission_v11" in record)
                    or canonical_json(marker)
                    != canonical_json(fixture_inspection.model_dump(mode="json"))
                ):
                    raise ValueError("Local usage fixture inspection capabilities are not exact")
            if "local_usage_fixture_intents" in record:
                marker = record["local_usage_fixture_intents"]
                fixture_intents = FixtureIntentCapabilities.model_validate(marker, strict=True)
                if "local_usage_fixture" not in record or canonical_json(marker) != canonical_json(
                    fixture_intents.model_dump(mode="json")
                ):
                    raise ValueError("Local usage fixture intent capabilities are not exact")
            if "local_usage_fixture_namespace_guard" in record:
                marker = record["local_usage_fixture_namespace_guard"]
                namespace_guard = FixtureNamespaceGuardCapabilities.model_validate(
                    marker, strict=True
                )
                if (
                    "local_usage_fixture" not in record
                    or "local_usage_fixture_intents" not in record
                    or canonical_json(marker)
                    != canonical_json(namespace_guard.model_dump(mode="json"))
                ):
                    raise ValueError(
                        "Local usage fixture namespace guard capabilities are not exact"
                    )
            if "local_usage_fixture_receipt_reads" in record:
                marker = record["local_usage_fixture_receipt_reads"]
                receipt_reads = FixtureReceiptReadCapabilities.model_validate(marker, strict=True)
                if (
                    "local_usage_fixture" not in record
                    or receipt_reads.separate_scientific_read_admission_enabled
                    != ("scientific_read_admission_v11" in record)
                    or canonical_json(marker)
                    != canonical_json(receipt_reads.model_dump(mode="json"))
                ):
                    raise ValueError("Local usage fixture receipt read capabilities are not exact")
            if "agent_proposal_budget" in record:
                marker = record["agent_proposal_budget"]
                budget = AgentProposalBudgetCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(budget.model_dump(mode="json")):
                    raise ValueError("Agent proposal allowance capabilities are not exact")
            if "agent_execution_budget" in record:
                marker = record["agent_execution_budget"]
                execution_capability = AgentExecutionBudgetCapabilities.model_validate(
                    marker, strict=True
                )
                if canonical_json(marker) != canonical_json(
                    execution_capability.model_dump(mode="json")
                ):
                    raise ValueError("Agent execution allowance capabilities are not exact")
            if "agent_ingest_budget" in record:
                marker = record["agent_ingest_budget"]
                ingestion_capability = AgentIngestBudgetCapabilities.model_validate(
                    marker, strict=True
                )
                if canonical_json(marker) != canonical_json(
                    ingestion_capability.model_dump(mode="json")
                ):
                    raise ValueError("Agent ingestion allowance capabilities are not exact")
            if "agent_upload_staging" in record:
                marker = record["agent_upload_staging"]
                upload_policy = AgentUploadStagingCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(upload_policy.model_dump(mode="json")):
                    raise ValueError("Agent upload staging capabilities are not exact")
            if "agent_upload_staging_admin" in record:
                marker = record["agent_upload_staging_admin"]
                admin_policy = AgentUploadStagingAdminCapabilities.model_validate(
                    marker, strict=True
                )
                if "agent_upload_staging" not in record or canonical_json(marker) != canonical_json(
                    admin_policy.model_dump(mode="json")
                ):
                    raise ValueError("Agent staging review capabilities are inconsistent")
            if "agent_grant_retry" in record:
                marker = record["agent_grant_retry"]
                grant_retry = AgentGrantRetryCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(grant_retry.model_dump(mode="json")):
                    raise ValueError("Agent grant retry capabilities are not exact")
            if "workspace_lifecycle" in record:
                marker = record["workspace_lifecycle"]
                lifecycle = WorkspaceLifecycleCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(lifecycle.model_dump(mode="json")):
                    raise ValueError("Workspace lifecycle capabilities are not exact")
            if "workspace_lifecycle_extended" in record:
                marker = record["workspace_lifecycle_extended"]
                extended = WorkspaceLifecycleExtendedCapabilities.model_validate(
                    marker, strict=True
                )
                if (
                    "workspace_lifecycle" not in record
                    or canonical_json(marker) != canonical_json(extended.model_dump(mode="json"))
                    or type(plan_limit) is not int
                    or extended.max_request_bytes != min(MAX_CONTROL_REQUEST_BYTES, plan_limit)
                ):
                    raise ValueError("Extended workspace lockdown capabilities are not exact")
            if "workspace_lockdown_receipt_v1" in record:
                marker = record["workspace_lockdown_receipt_v1"]
                lockdown_receipt = LockdownReceiptCapabilities.model_validate(marker, strict=True)
                if (
                    "workspace_lifecycle" not in record
                    or "workspace_lifecycle_extended" not in record
                    or canonical_json(marker)
                    != canonical_json(lockdown_receipt.model_dump(mode="json"))
                    or type(plan_limit) is not int
                    or lockdown_receipt.max_request_bytes
                    != min(MAX_CONTROL_REQUEST_BYTES, plan_limit)
                ):
                    raise ValueError("Lockdown receipt capabilities are inconsistent")
            if "workspace_directory" in record:
                marker = record["workspace_directory"]
                directory = WorkspaceDirectoryCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(directory.model_dump(mode="json")):
                    raise ValueError("Workspace directory capabilities are not exact")
            if "credential_pages" in record and "credential_pages_v2" in record:
                raise ValueError("Credential page capabilities are contradictory")
            if "credential_pages" in record:
                marker = record["credential_pages"]
                pages = CredentialPageCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(pages.model_dump(mode="json")):
                    raise ValueError("Credential page capabilities are not exact")
            if "credential_pages_v2" in record:
                marker = record["credential_pages_v2"]
                pages_v2 = CredentialPageV2Capabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(pages_v2.model_dump(mode="json")):
                    raise ValueError("Credential page capabilities are not exact")
            if "service_actor_recipient_review" in record:
                marker = record["service_actor_recipient_review"]
                review = RecipientBindingReviewCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(review.model_dump(mode="json")):
                    raise ValueError("Recipient binding review capabilities are not exact")
            if "workspace_service_actors" in record:
                marker = record["workspace_service_actors"]
                service_actors = ServiceActorCapabilities.model_validate(marker, strict=True)
                if (
                    canonical_json(marker) != canonical_json(service_actors.model_dump(mode="json"))
                    or type(plan_limit) is not int
                    or service_actors.max_request_bytes
                    != min(MAX_CONTROL_REQUEST_BYTES, plan_limit)
                ):
                    raise ValueError("Service actor capabilities are not exact")
            if "service_actor_recipient_grants" in record:
                marker = record["service_actor_recipient_grants"]
                recipient_grants = RecipientCapabilities.model_validate(marker, strict=True)
                if (
                    "workspace_service_actors" not in record
                    or "agent_grant_retry" not in record
                    or canonical_json(marker)
                    != canonical_json(recipient_grants.model_dump(mode="json"))
                    or type(plan_limit) is not int
                    or recipient_grants.max_request_bytes
                    != min(MAX_CONTROL_REQUEST_BYTES, plan_limit)
                ):
                    raise ValueError("Service actor recipient capabilities are not exact")
            if "service_actor_recipient_capacity" in record:
                marker = record["service_actor_recipient_capacity"]
                recipient_capacity = RecipientCapacityCapabilities.model_validate(
                    marker, strict=True
                )
                if "service_actor_recipient_grants" not in record or canonical_json(
                    marker
                ) != canonical_json(recipient_capacity.model_dump(mode="json")):
                    raise ValueError("Service actor recipient capacity is not exact")
            if "identity_provider_admission" in record:
                marker = record["identity_provider_admission"]
                provider_admission = IdentityProviderAdmissionCapabilities.model_validate(
                    marker, strict=True
                )
                if canonical_json(marker) != canonical_json(
                    provider_admission.model_dump(mode="json")
                ):
                    raise ValueError("Identity provider admission capabilities are not exact")
            if "model_provider_admission" in record:
                marker = record["model_provider_admission"]
                model_admission = ModelProviderAdmissionCapabilities.model_validate(
                    marker, strict=True
                )
                if canonical_json(marker) != canonical_json(
                    model_admission.model_dump(mode="json")
                ):
                    raise ValueError("Model provider admission capabilities are not exact")
            if "model_provider_tenant_admission" in record:
                marker = record["model_provider_tenant_admission"]
                tenant_admission = ModelProviderTenantAdmissionCapabilities.model_validate(
                    marker, strict=True
                )
                shared = record.get("model_provider_admission")
                if (
                    canonical_json(marker)
                    != canonical_json(tenant_admission.model_dump(mode="json"))
                    or not isinstance(shared, dict)
                    or tenant_admission.shared_slots != shared.get("slots")
                    or tenant_admission.global_request_slots != shared.get("global_request_slots")
                ):
                    raise ValueError(
                        "Model provider workspace admission capabilities are not exact"
                    )
            if "model_provider_timeout" in record:
                marker = record["model_provider_timeout"]
                timeout = ModelProviderTimeoutCapabilities.model_validate(marker, strict=True)
                if (
                    canonical_json(marker) != canonical_json(timeout.model_dump(mode="json"))
                    or record.get("llm_planning") is not True
                ):
                    raise ValueError("Model provider timeout capabilities are not exact")
            if "organization_governance" in record:
                marker = record["organization_governance"]
                organization = OrganizationCapabilities.model_validate(marker, strict=True)
                if (
                    canonical_json(marker) != canonical_json(organization.model_dump(mode="json"))
                    or type(plan_limit) is not int
                    or organization.max_request_bytes != min(MAX_CONTROL_REQUEST_BYTES, plan_limit)
                    or organization.workspace_detach_supported
                    != ("organization_workspace_detachment" in record)
                ):
                    raise ValueError("Organization capabilities are not exact")
            if "organization_workspace_detachment" in record:
                marker = record["organization_workspace_detachment"]
                detachment = DetachmentCapabilities.model_validate(marker, strict=True)
                governance = record.get("organization_governance")
                if (
                    not isinstance(governance, dict)
                    or governance.get("format") != "tensorus-organization-governance-v2"
                    or governance.get("workspace_detach_supported") is not True
                    or canonical_json(marker) != canonical_json(detachment.model_dump(mode="json"))
                    or type(plan_limit) is not int
                    or detachment.max_request_bytes != min(MAX_CONTROL_REQUEST_BYTES, plan_limit)
                ):
                    raise ValueError("Organization detachment capabilities are not exact")
            if "organization_detachment_receipt_v1" in record:
                marker = record["organization_detachment_receipt_v1"]
                detachment_receipt = DetachmentReceiptCapabilities.model_validate(
                    marker, strict=True
                )
                if (
                    "organization_workspace_detachment" not in record
                    or canonical_json(marker)
                    != canonical_json(detachment_receipt.model_dump(mode="json"))
                    or type(plan_limit) is not int
                    or detachment_receipt.max_request_bytes
                    != min(MAX_CONTROL_REQUEST_BYTES, plan_limit)
                ):
                    raise ValueError("Organization detachment receipt capabilities are not exact")
            if "organization_session_fence" in record:
                marker = record["organization_session_fence"]
                fence = (
                    OrganizationSessionFenceV2Capabilities.model_validate(marker, strict=True)
                    if isinstance(marker, dict)
                    and marker.get("format") == "tensorus-organization-session-fence-v2"
                    else OrganizationSessionFenceCapabilities.model_validate(marker, strict=True)
                )
                if "organization_governance" not in record or canonical_json(
                    marker
                ) != canonical_json(fence.model_dump(mode="json")):
                    raise ValueError("Organization session fence capabilities are not exact")
            if "workspace_session_revocation" in record:
                marker = record["workspace_session_revocation"]
                revocation = WorkspaceSessionRevocationCapabilities.model_validate(
                    marker, strict=True
                )
                if (
                    canonical_json(marker) != canonical_json(revocation.model_dump(mode="json"))
                    or type(plan_limit) is not int
                    or revocation.max_request_bytes != min(MAX_CONTROL_REQUEST_BYTES, plan_limit)
                ):
                    raise ValueError("Session revocation capabilities are not exact")
            if "workspace_departure" in record:
                marker = record["workspace_departure"]
                departure = DepartureCapabilities.model_validate(marker, strict=True)
                if (
                    canonical_json(marker) != canonical_json(departure.model_dump(mode="json"))
                    or type(plan_limit) is not int
                    or departure.max_request_bytes != min(MAX_CONTROL_REQUEST_BYTES, plan_limit)
                ):
                    raise ValueError("Workspace departure capabilities are not exact")
            if "workspace_departure_receipt_v2" in record:
                marker = record["workspace_departure_receipt_v2"]
                departure_recovery = DepartureReceiptRecoveryCapabilities.model_validate(
                    marker, strict=True
                )
                if (
                    canonical_json(marker)
                    != canonical_json(departure_recovery.model_dump(mode="json"))
                    or "workspace_departure" not in record
                    or type(plan_limit) is not int
                    or departure_recovery.max_request_bytes
                    != min(MAX_CONTROL_REQUEST_BYTES, plan_limit)
                ):
                    raise ValueError("Departure receipt capabilities are not exact")
            if "workspace_identity_sessions" in record:
                marker = record["workspace_identity_sessions"]
                sessions = IdentitySessionsCapabilities.model_validate(marker, strict=True)
                if (
                    canonical_json(marker) != canonical_json(sessions.model_dump(mode="json"))
                    or type(plan_limit) is not int
                    or sessions.max_request_bytes != min(MAX_CONTROL_REQUEST_BYTES, plan_limit)
                ):
                    raise ValueError("Identity-session capabilities are not exact")
            if "key_rotation" in record:
                marker = record["key_rotation"]
                rotation = KeyRotationCapabilities.model_validate(marker, strict=True)
                if (
                    canonical_json(marker) != canonical_json(rotation.model_dump(mode="json"))
                    or type(plan_limit) is not int
                    or rotation.max_request_bytes != min(MAX_CONTROL_REQUEST_BYTES, plan_limit)
                ):
                    raise ValueError("Key rotation capabilities are not exact")
            if "tenant_http_admission" in record:
                marker = record["tenant_http_admission"]
                admission = TenantHttpAdmissionCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(admission.model_dump(mode="json")):
                    raise ValueError("Tenant HTTP admission capabilities are not exact")
            if "workspace_capacity" in record:
                marker = record["workspace_capacity"]
                capacity = WorkspaceCapacityCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(capacity.model_dump(mode="json")):
                    raise ValueError("Workspace capacity capabilities are not exact")
            if "advisory_pricing" in record:
                marker = record["advisory_pricing"]
                pricing = AdvisoryPricingCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(pricing.model_dump(mode="json")):
                    raise ValueError("Advisory pricing capabilities are not exact")
            if "product_revision_retirement" in record:
                marker = record["product_revision_retirement"]
                retirement = ProductRevisionRetirementCapabilities.model_validate(
                    marker, strict=True
                )
                if canonical_json(marker) != canonical_json(retirement.model_dump(mode="json")):
                    raise ValueError("Product-revision retirement capabilities are not exact")
            if "product_retirement" in record:
                marker = record["product_retirement"]
                product_retirement = ProductRetirementCapabilities.model_validate(
                    marker, strict=True
                )
                if (
                    canonical_json(marker)
                    != canonical_json(product_retirement.model_dump(mode="json"))
                    or type(plan_limit) is not int
                    or product_retirement.max_request_bytes
                    != min(MAX_CONTROL_REQUEST_BYTES, plan_limit)
                ):
                    raise ValueError("Product retirement capabilities are not exact")
            if "quality_release_retirement" in record:
                marker = record["quality_release_retirement"]
                quality_retirement_capability = QualityReleaseRetirementCapabilities.model_validate(
                    marker, strict=True
                )
                if (
                    canonical_json(marker)
                    != canonical_json(quality_retirement_capability.model_dump(mode="json"))
                    or type(plan_limit) is not int
                    or quality_retirement_capability.max_request_bytes
                    != min(MAX_CONTROL_REQUEST_BYTES, plan_limit)
                ):
                    raise ValueError("Quality-release retirement capabilities are not exact")
            if "product_revision_holds" in record:
                marker = record["product_revision_holds"]
                revision_holds = ProductRevisionHoldCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(revision_holds.model_dump(mode="json")):
                    raise ValueError("Product-revision hold capabilities are not exact")
            if "tenant_fair_scheduling" in record:
                marker = record["tenant_fair_scheduling"]
                scheduling = TenantFairSchedulingCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(scheduling.model_dump(mode="json")):
                    raise ValueError("Tenant scheduling capabilities are not exact")
            if "http_rate_limit" in record:
                marker = record["http_rate_limit"]
                rate = HttpRateLimitCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(rate.model_dump(mode="json")):
                    raise ValueError("HTTP rate-limit capabilities are not exact")
            if "http_peer_rate_limit" in record:
                marker = record["http_peer_rate_limit"]
                peer_rate = HttpPeerRateLimitCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(peer_rate.model_dump(mode="json")):
                    raise ValueError("HTTP peer rate-limit capabilities are not exact")
            if "request_body_staging_deadline" in record:
                marker = record["request_body_staging_deadline"]
                deadline = RequestBodyStagingDeadlineCapabilities.model_validate(
                    marker, strict=True
                )
                if staging_marker is None or canonical_json(marker) != canonical_json(
                    deadline.model_dump(mode="json")
                ):
                    raise ValueError("Request staging deadline capabilities are inconsistent")
            if "npy_download_admission" in record:
                marker = record["npy_download_admission"]
                download = NpyDownloadAdmissionCapabilities.model_validate(marker, strict=True)
                if staging_marker is None or canonical_json(marker) != canonical_json(
                    download.model_dump(mode="json")
                ):
                    raise ValueError("NPY download admission capabilities are inconsistent")
            if "similarity_admission" in record:
                marker = record["similarity_admission"]
                similarity = SimilarityAdmissionCapabilities.model_validate(marker, strict=True)
                if (
                    not isinstance(staging_marker, dict)
                    or similarity.slots > staging_marker["concurrent_slots"]
                    or canonical_json(marker) != canonical_json(similarity.model_dump(mode="json"))
                ):
                    raise ValueError("Similarity admission capabilities are inconsistent")
            scientific_formats = (
                "scientific_read_admission",
                "scientific_read_admission_v2",
                "scientific_read_admission_v3",
                "scientific_read_admission_v4",
                "scientific_read_admission_v5",
                "scientific_read_admission_v6",
                "scientific_read_admission_v7",
                "scientific_read_admission_v8",
                "scientific_read_admission_v9",
                "scientific_read_admission_v10",
                "scientific_read_admission_v11",
            )
            if sum(name in record for name in scientific_formats) > 1:
                raise ValueError("Scientific read admission formats are contradictory")
            if "authenticated_get_caller_fence" in record:
                marker = record["authenticated_get_caller_fence"]
                read_fence = AuthenticatedReadFenceCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(read_fence.model_dump(mode="json")):
                    raise ValueError("Authenticated GET response fence is inconsistent")
            if "dataset_read_response_fence" in record:
                marker = record["dataset_read_response_fence"]
                dataset_fence = DatasetReadFenceCapabilities.model_validate(marker, strict=True)
                if sum(name in record for name in scientific_formats) != 1 or canonical_json(
                    marker
                ) != canonical_json(dataset_fence.model_dump(mode="json")):
                    raise ValueError("Dataset read response fence is inconsistent")
            if "dataset_options" in record:
                marker = record["dataset_options"]
                options = DatasetOptionsCapabilities.model_validate(marker, strict=True)
                expected_rechecks = "dataset_read_response_fence" in record
                expected_admission = expected_rechecks or any(
                    name in record
                    for name in ("scientific_read_admission_v10", "scientific_read_admission_v11")
                )
                if (
                    canonical_json(marker) != canonical_json(options.model_dump(mode="json"))
                    or options.response_rechecks_enabled != expected_rechecks
                    or options.shared_admission_enabled != expected_admission
                ):
                    raise ValueError("Dataset option capabilities are inconsistent")
            if "snapshot_read_response_fence" in record:
                marker = record["snapshot_read_response_fence"]
                snapshot_fence = SnapshotReadFenceCapabilities.model_validate(marker, strict=True)
                if sum(name in record for name in scientific_formats) != 1 or canonical_json(
                    marker
                ) != canonical_json(snapshot_fence.model_dump(mode="json")):
                    raise ValueError("Snapshot read response fence is inconsistent")
            if "snapshot_cursor_list" in record:
                marker = record["snapshot_cursor_list"]
                cursor_list = SnapshotCursorCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(cursor_list.model_dump(mode="json")):
                    raise ValueError("Snapshot cursor capability is inconsistent")
            if "workspace_overview_response_fence" in record:
                marker = record["workspace_overview_response_fence"]
                overview_fence = WorkspaceOverviewResponseFenceCapabilities.model_validate(
                    marker, strict=True
                )
                if canonical_json(marker) != canonical_json(overview_fence.model_dump(mode="json")):
                    raise ValueError("Workspace overview response fence is inconsistent")
            if "workspace_overview_admission" in record:
                marker = record["workspace_overview_admission"]
                overview_admission = WorkspaceOverviewAdmissionCapabilities.model_validate(
                    marker, strict=True
                )
                if (
                    "workspace_overview_response_fence" not in record
                    or sum(name in record for name in scientific_formats) != 1
                    or canonical_json(marker)
                    != canonical_json(overview_admission.model_dump(mode="json"))
                ):
                    raise ValueError("Workspace overview admission capabilities are inconsistent")
            if "tenant_staging_read_admission" in record:
                marker = record["tenant_staging_read_admission"]
                staging_admission = TenantStagingReadAdmissionCapabilities.model_validate(
                    marker, strict=True
                )
                if (
                    record.get("role") != Role.admin
                    or record.get("scope") is not None
                    or "tenant_staging_reservations" not in record
                    or "tenant_staging_work_quota" not in record
                    or "tenant_staging_input_quota" not in record
                    or sum(name in record for name in scientific_formats) != 1
                    or canonical_json(marker)
                    != canonical_json(staging_admission.model_dump(mode="json"))
                ):
                    raise ValueError("Tenant staging admission capabilities are inconsistent")
            if "upload_chunk_admission" in record:
                marker = record["upload_chunk_admission"]
                chunk_admission = UploadChunkAdmissionCapabilities.model_validate(
                    marker, strict=True
                )
                if (
                    staging_marker is None
                    or sum(name in record for name in scientific_formats) != 1
                    or canonical_json(marker)
                    != canonical_json(chunk_admission.model_dump(mode="json"))
                ):
                    raise ValueError("Upload chunk admission capabilities are inconsistent")
            if "product_descriptor_admission" in record:
                marker = record["product_descriptor_admission"]
                descriptors = ProductDescriptorAdmissionCapabilities.model_validate(
                    marker, strict=True
                )
                if (
                    sum(name in record for name in scientific_formats) != 1
                    or sum(name in record for name in product_fence_formats) != 1
                    or canonical_json(marker) != canonical_json(descriptors.model_dump(mode="json"))
                    or (
                        any(name in record for name in scientific_formats[7:])
                        != ("product_read_response_fence_v2" in record)
                    )
                ):
                    raise ValueError("Product descriptor admission capabilities are inconsistent")
            elif "product_read_response_fence_v2" in record:
                raise ValueError("Product read response fence admission is inconsistent")
            if (
                "credential_inventory_admission" in record
                and "credential_inventory_admission_v2" in record
            ):
                raise ValueError("Credential inventory admission formats are contradictory")
            if "credential_inventory_admission" in record:
                marker = record["credential_inventory_admission"]
                inventory = CredentialInventoryAdmissionCapabilities.model_validate(
                    marker, strict=True
                )
                if (
                    "credential_pages" not in record
                    or sum(name in record for name in scientific_formats) != 1
                    or canonical_json(marker) != canonical_json(inventory.model_dump(mode="json"))
                ):
                    raise ValueError("Credential inventory admission capabilities are inconsistent")
            if "credential_inventory_admission_v2" in record:
                marker = record["credential_inventory_admission_v2"]
                inventory_v2 = CredentialInventoryAdmissionV2Capabilities.model_validate(
                    marker, strict=True
                )
                if (
                    "credential_pages_v2" not in record
                    or sum(name in record for name in scientific_formats) != 1
                    or canonical_json(marker)
                    != canonical_json(inventory_v2.model_dump(mode="json"))
                ):
                    raise ValueError("Credential inventory admission capabilities are inconsistent")
            if "workflow_inventory_admission" in record:
                marker = record["workflow_inventory_admission"]
                workflows = WorkflowInventoryAdmissionCapabilities.model_validate(
                    marker, strict=True
                )
                if sum(name in record for name in scientific_formats) != 1 or canonical_json(
                    marker
                ) != canonical_json(workflows.model_dump(mode="json")):
                    raise ValueError("Workflow inventory admission capabilities are inconsistent")
            if "history_read_admission" in record:
                marker = record["history_read_admission"]
                history = HistoryReadAdmissionCapabilities.model_validate(marker, strict=True)
                if sum(name in record for name in scientific_formats) != 1 or canonical_json(
                    marker
                ) != canonical_json(history.model_dump(mode="json")):
                    raise ValueError("History read admission capabilities are inconsistent")
            if "history_list_admission" in record:
                marker = record["history_list_admission"]
                history_list = HistoryListAdmissionCapabilities.model_validate(marker, strict=True)
                if (
                    "history_read_admission" not in record
                    or sum(name in record for name in scientific_formats) != 1
                    or canonical_json(marker)
                    != canonical_json(history_list.model_dump(mode="json"))
                ):
                    raise ValueError("History list admission capabilities are inconsistent")
            if "scoped_history_summary_policy" in record:
                marker = record["scoped_history_summary_policy"]
                summaries = ScopedHistorySummaryPolicyCapabilities.model_validate(
                    marker, strict=True
                )
                if (
                    "history_read_admission" not in record
                    or sum(name in record for name in scientific_formats) != 1
                    or canonical_json(marker) != canonical_json(summaries.model_dump(mode="json"))
                ):
                    raise ValueError("Scoped history summary policy is inconsistent")
            for name, model in (
                ("scientific_read_admission", ScientificReadAdmissionCapabilities),
                ("scientific_read_admission_v2", ScientificReadAdmissionV2Capabilities),
                ("scientific_read_admission_v3", ScientificReadAdmissionV3Capabilities),
                ("scientific_read_admission_v4", ScientificReadAdmissionV4Capabilities),
                ("scientific_read_admission_v5", ScientificReadAdmissionV5Capabilities),
                ("scientific_read_admission_v6", ScientificReadAdmissionV6Capabilities),
                ("scientific_read_admission_v7", ScientificReadAdmissionV7Capabilities),
                ("scientific_read_admission_v8", ScientificReadAdmissionV8Capabilities),
                ("scientific_read_admission_v9", ScientificReadAdmissionV9Capabilities),
                ("scientific_read_admission_v10", ScientificReadAdmissionV10Capabilities),
                ("scientific_read_admission_v11", ScientificReadAdmissionV11Capabilities),
            ):
                if name not in record:
                    continue
                marker = record[name]
                scientific = model.model_validate(marker, strict=True)
                if (
                    not isinstance(staging_marker, dict)
                    or scientific.slots > staging_marker["concurrent_slots"]
                    or canonical_json(marker) != canonical_json(scientific.model_dump(mode="json"))
                ):
                    raise ValueError("Scientific read admission capabilities are inconsistent")
            index = record.get("semantic_discovery_index")
            if isinstance(index, dict) and "watermark_checks" in index:
                marker = index["watermark_checks"]
                watermark_checks = IndexWatermarkChecksCapabilities.model_validate(
                    marker, strict=True
                )
                if canonical_json(marker) != canonical_json(
                    watermark_checks.model_dump(mode="json")
                ):
                    raise ValueError("Index watermark capabilities are not exact")
            semantics = record.get("tensor_semantics")
            reference = (
                semantics.get("reference_semantics") if isinstance(semantics, dict) else None
            )
            if isinstance(reference, dict) and "paired_validity_nearest" in reference:
                marker = reference["paired_validity_nearest"]
                nearest = NearestCoordinateCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(nearest.model_dump(mode="json")):
                    raise ValueError("Nearest-coordinate capabilities are not exact")
                schema = record.get("plan_schema")
                limit = (
                    schema.get("x-tensorus-max-review-request-bytes")
                    if isinstance(schema, dict)
                    else None
                )
                if type(limit) is not int or not 0 < limit <= nearest.max_request_bytes:
                    raise ValueError("Plan review request limit is inconsistent")
            contracts = record.get("write_contracts")
            if "write_contracts" in record and not isinstance(contracts, dict):
                raise ValueError("Write-contract capabilities must be an object")
            if isinstance(contracts, dict) and "conditional_npy_ingest" in contracts:
                condition = contracts["conditional_npy_ingest"]
                parsed = ConditionalNpyIngestCapabilities.model_validate(condition, strict=True)
                if canonical_json(condition) != canonical_json(parsed.model_dump(mode="json")):
                    raise ValueError("Conditional ingestion capabilities are not exact")
            if isinstance(contracts, dict) and "conditional_npy_receipt" in contracts:
                marker = contracts["conditional_npy_receipt"]
                receipt = ConditionalNpyReceiptCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(receipt.model_dump(mode="json")):
                    raise ValueError("Conditional ingestion receipt capabilities are not exact")
            if isinstance(contracts, dict) and "conditional_npy_historical_receipt" in contracts:
                marker = contracts["conditional_npy_historical_receipt"]
                historical = ConditionalNpyHistoricalReceiptCapabilities.model_validate(
                    marker, strict=True
                )
                if canonical_json(marker) != canonical_json(historical.model_dump(mode="json")):
                    raise ValueError("Historical receipt capabilities are not exact")
            releases = record.get("release_contracts")
            if isinstance(releases, dict) and "portable_history_name_lookup" in releases:
                marker = releases["portable_history_name_lookup"]
                lookup = ImportedHistoryNameLookupCapabilities.model_validate(marker, strict=True)
                if canonical_json(marker) != canonical_json(lookup.model_dump(mode="json")):
                    raise ValueError("Imported history name lookup capabilities are not exact")
            supervision = record.get("execution_supervision")
            if isinstance(supervision, dict) and "eager_local_reads" in supervision:
                local = LocalEagerSupervisionCapabilities.model_validate(
                    supervision["eager_local_reads"], strict=True
                )
                if (
                    supervision.get("format") != "tensorus-supervised-eager-v1"
                    or supervision.get("supported_platform") != "linux"
                    or supervision.get("scope") != "eager_operations_except_profile_and_chunked"
                    or type(supervision.get("enabled")) is not bool
                    or (local.enabled and not supervision["enabled"])
                    or type(supervision.get("child_address_space_bytes")) is not int
                    or type(supervision.get("input_bytes")) is not int
                    or type(supervision.get("output_bytes")) is not int
                    or supervision["child_address_space_bytes"] != local.child_address_space_bytes
                    or supervision["input_bytes"] != local.input_bytes
                    or supervision["output_bytes"] != local.output_bytes
                    or supervision.get("parent_process_memory_bounded") is not False
                    or supervision.get("io_and_publication_hard_stopped") is not False
                    or supervision.get("filesystem_network_isolation") is not False
                ):
                    raise ValueError("Local eager supervision capabilities disagree")
            if isinstance(supervision, dict) and "chunked_local" in supervision:
                chunked = ChunkedSupervisionCapabilities.model_validate(
                    supervision["chunked_local"], strict=True
                )
                if (
                    supervision.get("format") != "tensorus-supervised-eager-v1"
                    or supervision.get("supported_platform") != "linux"
                    or supervision.get("scope") != "eager_operations_except_profile_and_chunked"
                    or type(supervision.get("enabled")) is not bool
                    or type(supervision.get("child_address_space_bytes")) is not int
                    or type(supervision.get("output_bytes")) is not int
                    or supervision["child_address_space_bytes"] != chunked.child_address_space_bytes
                    or supervision["output_bytes"] != chunked.output_bytes
                    or supervision.get("parent_process_memory_bounded") is not False
                    or supervision.get("io_and_publication_hard_stopped") is not False
                    or supervision.get("filesystem_network_isolation") is not False
                ):
                    raise ValueError("Chunked supervision capabilities disagree")
            transfers = record.get("npy_downloads")
            if isinstance(transfers, dict) and "finalizer_ownership" in transfers:
                finalizer = transfers["finalizer_ownership"]
                if not isinstance(finalizer, dict):
                    raise ValueError("Upload finalizer capabilities must be an object")
                if "input_reservation" in finalizer:
                    advertised = finalizer["input_reservation"]
                    if not isinstance(advertised, dict) or (
                        "catalog_bound_scratch" in advertised
                        and not isinstance(advertised["catalog_bound_scratch"], dict)
                    ):
                        raise ValueError("Invalid catalog-bound scratch capability")
                    scratch = advertised.get("catalog_bound_scratch")
                    if isinstance(scratch, dict) and (
                        "preallocated_chunks" in scratch
                        and not isinstance(scratch["preallocated_chunks"], dict)
                    ):
                        raise ValueError("Invalid preallocated chunk capability")
                    reservation = UploadInputReservationCapabilities.model_validate(
                        advertised, strict=True
                    )
                    if (
                        finalizer.get("format") != "tensorus-upload-finalizer-v1"
                        or finalizer.get("physical_capacity_reserved") is not False
                        or (
                            reservation.configured
                            and finalizer.get("input_file_preallocation") is not True
                        )
                    ):
                        raise ValueError("Upload input reservation capabilities disagree")
            if "eager_ingest_staging" in record:
                eager = EagerIngestStagingCapabilities.model_validate(
                    record["eager_ingest_staging"], strict=True
                )
                finalizer = (
                    transfers.get("finalizer_ownership") if isinstance(transfers, dict) else None
                )
                upload = finalizer.get("input_reservation") if isinstance(finalizer, dict) else None
                scratch = upload.get("catalog_bound_scratch") if isinstance(upload, dict) else None
                chunks = scratch.get("preallocated_chunks") if isinstance(scratch, dict) else None
                if (
                    not isinstance(transfers, dict)
                    or type(transfers.get("streaming_upload_work_bytes")) is not int
                    or transfers["streaming_upload_work_bytes"] != eager.work_token_bytes
                    or not isinstance(upload, dict)
                    or upload.get("work_capacity_bytes") != eager.work_pool_capacity_bytes
                    or (
                        isinstance(chunks, dict)
                        and (
                            chunks.get("max_chunks") != eager.max_chunks
                            or chunks.get("max_filesystem_fragment_bytes")
                            != eager.max_filesystem_fragment_bytes
                            or chunks.get("slot_rule") != eager.slot_rule
                        )
                    )
                    or (
                        eager.configured
                        and (
                            upload.get("configured") is not True
                            or not isinstance(chunks, dict)
                            or chunks.get("enabled") is not True
                        )
                    )
                ):
                    raise ValueError("Eager ingestion staging capabilities disagree")
            return record
        except (ValidationError, TensorusError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "The server returned invalid capability evidence", code="invalid_response"
            ) from exc

    def _checked_capabilities(self, *, _credential: str | None = None) -> tuple[Identity, Record]:
        identity = (
            self._checked_identity()
            if _credential is None
            else self._checked_identity_for_credential(_credential)
        )
        capabilities = (
            self.capabilities()
            if _credential is None
            else self.capabilities(_credential=_credential)
        )
        try:
            scope = identity.scope.model_dump(mode="json") if identity.scope is not None else None
            if (
                not isinstance(capabilities, dict)
                or capabilities.get("role") != identity.role.value
                or "scope" not in capabilities
                or canonical_json(capabilities["scope"]) != canonical_json(scope)
            ):
                raise ValueError("Capabilities contradict the authenticated identity")
        except (ValueError, TypeError, RecursionError, TensorusError) as exc:
            raise TensorusError(
                "The server returned capabilities inconsistent with the active identity",
                code="invalid_response",
            ) from exc
        final_identity = (
            self._checked_identity()
            if _credential is None
            else self._checked_identity_for_credential(_credential)
        )
        if final_identity != identity:
            raise Forbidden("The authenticated identity changed during capability review")
        return identity, capabilities

    def verify_recovery_objects(self, request: RecoveryVerificationRequest) -> Record:
        """Check selected tenant objects without repair, mutation or source authentication."""
        request = RecoveryVerificationRequest.model_validate(request.model_dump(mode="json"))
        capabilities = self.capabilities()
        if "recovery_verification" not in capabilities:
            raise TensorusError(
                "The server does not advertise selected recovery verification",
                code="unsupported_feature",
            )

        before = self._checked_identity()
        if (
            before.role != Role.admin
            or before.scope is not None
            or capabilities.get("role") != before.role
            or capabilities.get("scope") is not None
        ):
            raise TensorusError(
                "Recovery verification requires an active unscoped tenant administrator",
                code="forbidden",
                status=403,
            )
        encoded = self._bounded_report(
            "POST",
            "/v1/maintenance/verify-objects",
            MAX_REMOTE_RECOVERY_RESPONSE_BYTES,
            "Recovery verification",
            json=request.model_dump(mode="json"),
        )
        try:
            raw = decode_unique_json(encoded)
            report = RecoveryVerificationReport.model_validate_json(encoded, strict=True)
            if (
                not isinstance(raw, dict)
                or {item.version_id for item in report.versions} != set(request.version_ids)
                or any(item.tenant_id != before.id for item in report.versions)
                or report.max_logical_bytes != request.max_logical_bytes
                or report.max_decoded_work_bytes != request.max_decoded_work_bytes
                or (
                    request.expected_selection_hash is not None
                    and report.selection_hash != request.expected_selection_hash
                )
            ):
                raise ValueError("Recovery verification response does not match the request")
        except (ValidationError, TensorusError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "The server returned inconsistent recovery verification evidence",
                code="invalid_response",
            ) from exc
        after = self._checked_identity()
        if (
            after.id != before.id
            or after.key_id != before.key_id
            or after.role != before.role
            or after.scope != before.scope
        ):
            raise TensorusError(
                "The workspace identity changed during recovery verification",
                code="invalid_response",
            )
        return raw

    def recovery_inventory(self, expected_inventory_hash: str | None = None) -> Record:
        """Capture a bounded tenant reference inventory without checking object payloads."""
        return self._recovery_inventory(expected_inventory_hash, None)

    def recovery_inventory_partition(
        self, partition_prefix: str, expected_inventory_hash: str | None = None
    ) -> Record:
        """Capture a UUID-prefix subset, not a snapshot shared with other partitions."""
        if (
            not isinstance(partition_prefix, str)
            or re.fullmatch(r"[0-9a-f]{1,8}", partition_prefix) is None
        ):
            raise TensorusError(
                "Recovery partition prefix must contain 1 to 8 lowercase hex digits"
            )
        return self._recovery_inventory(expected_inventory_hash, partition_prefix)

    def _recovery_inventory(
        self, expected_inventory_hash: str | None, partition_prefix: str | None
    ) -> Record:
        if expected_inventory_hash is not None and (
            not isinstance(expected_inventory_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_inventory_hash) is None
        ):
            raise TensorusError("Expected recovery inventory hash must be lowercase SHA-256")
        before, capabilities = self._checked_capabilities()
        capability = (
            "recovery_inventory_partition" if partition_prefix is not None else "recovery_inventory"
        )
        if capability not in capabilities:
            raise TensorusError(
                "The server does not advertise tenant recovery inventory",
                code="unsupported_feature",
            )
        if before.role != Role.admin or before.scope is not None:
            raise Forbidden("Recovery inventory requires an active unscoped tenant administrator")
        params: dict[str, str] = {}
        if expected_inventory_hash is not None:
            params["expected_inventory_hash"] = expected_inventory_hash
        if partition_prefix is not None:
            params["partition_prefix"] = partition_prefix
        encoded = self._bounded_report(
            "GET",
            (
                "/v1/maintenance/recovery-inventory/partition"
                if partition_prefix is not None
                else "/v1/maintenance/recovery-inventory"
            ),
            MAX_INVENTORY_RESPONSE_BYTES,
            "Recovery inventory",
            params=params or None,
        )
        try:
            raw = decode_unique_json(encoded)
            report = (
                RecoveryPartitionInventory.model_validate_json(encoded, strict=True)
                if partition_prefix is not None
                else RecoveryInventory.model_validate_json(encoded, strict=True)
            )
            if (
                not isinstance(raw, dict)
                or canonical_json(raw) != canonical_json(report.model_dump(mode="json"))
                or report.tenant_id != before.id
                or (
                    isinstance(report, RecoveryPartitionInventory)
                    and report.partition_prefix != partition_prefix
                )
                or (
                    expected_inventory_hash is not None
                    and report.inventory_hash != expected_inventory_hash
                )
            ):
                raise ValueError("Recovery inventory does not match the request")
        except (ValidationError, TensorusError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "The server returned inconsistent recovery inventory evidence",
                code="invalid_response",
            ) from exc
        if self._checked_identity() != before:
            raise TensorusError(
                "The workspace identity changed during recovery inventory review",
                code="invalid_response",
            )
        return raw

    def verify_recovery_inventory_batch(
        self, inventory: Record, request: RecoveryInventoryBatchRequest
    ) -> Record:
        """Verify one slice against a retained inventory, without certifying earlier slices."""
        return self._verify_recovery_inventory_batch(inventory, request, partitioned=False)

    def verify_recovery_inventory_partition_batch(
        self, inventory: Record, request: RecoveryPartitionBatchRequest
    ) -> Record:
        """Verify one retained UUID-prefix slice, without certifying other partitions."""
        return self._verify_recovery_inventory_batch(inventory, request, partitioned=True)

    def _verify_recovery_inventory_batch(
        self, inventory: Record, request: RecoveryInventoryBatchRequest, *, partitioned: bool
    ) -> Record:
        try:
            encoded_inventory = canonical_json(inventory)
            if len(encoded_inventory.encode()) > MAX_INVENTORY_RESPONSE_BYTES:
                raise LimitExceeded("Recovery inventory evidence exceeds its byte limit")
            reviewed = (
                RecoveryPartitionInventory.model_validate_json(encoded_inventory, strict=True)
                if partitioned
                else RecoveryInventory.model_validate_json(encoded_inventory, strict=True)
            )
            if canonical_json(reviewed.model_dump(mode="json")) != encoded_inventory:
                raise ValueError("Recovery inventory evidence is not canonical")
            checked_request = (
                RecoveryPartitionBatchRequest.model_validate(request.model_dump(mode="json"))
                if partitioned
                else RecoveryInventoryBatchRequest.model_validate(request.model_dump(mode="json"))
            )
            if (
                checked_request.expected_inventory_hash != reviewed.inventory_hash
                or checked_request.start_index >= reviewed.version_count
                or (
                    isinstance(reviewed, RecoveryPartitionInventory)
                    and (
                        not isinstance(checked_request, RecoveryPartitionBatchRequest)
                        or checked_request.partition_prefix != reviewed.partition_prefix
                    )
                )
            ):
                raise ValueError("Recovery batch request does not match the retained inventory")
        except (ValidationError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError("Invalid retained recovery inventory or batch request") from exc
        before, capabilities = self._checked_capabilities()
        capability = "recovery_inventory_partition" if partitioned else "recovery_inventory"
        if capability not in capabilities:
            raise TensorusError(
                "The server does not advertise tenant recovery inventory",
                code="unsupported_feature",
            )
        if before.role != Role.admin or before.scope is not None or before.id != reviewed.tenant_id:
            raise Forbidden("Recovery inventory requires the same unscoped tenant administrator")
        encoded = self._bounded_report(
            "POST",
            (
                "/v1/maintenance/recovery-inventory/verify-partition-batch"
                if partitioned
                else "/v1/maintenance/recovery-inventory/verify-batch"
            ),
            MAX_INVENTORY_BATCH_RESPONSE_BYTES,
            "Recovery inventory batch",
            json=checked_request.model_dump(mode="json"),
        )
        try:
            raw = decode_unique_json(encoded)
            batch = (
                RecoveryPartitionBatchReport.model_validate_json(encoded, strict=True)
                if partitioned
                else RecoveryInventoryBatchReport.model_validate_json(encoded, strict=True)
            )
            validate_batch_inventory(reviewed, batch)
            if (
                not isinstance(raw, dict)
                or canonical_json(raw) != canonical_json(batch.model_dump(mode="json"))
                or batch.start_index != checked_request.start_index
                or batch.end_index
                != min(checked_request.start_index + checked_request.limit, reviewed.version_count)
                or batch.verification.max_logical_bytes != checked_request.max_logical_bytes
                or batch.verification.max_decoded_work_bytes
                != checked_request.max_decoded_work_bytes
            ):
                raise ValueError("Recovery batch does not match the request")
        except (ValidationError, TensorusError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "The server returned inconsistent recovery batch evidence",
                code="invalid_response",
            ) from exc
        if self._checked_identity() != before:
            raise TensorusError(
                "The workspace identity changed during recovery batch verification",
                code="invalid_response",
            )
        return raw

    @staticmethod
    def review_recovery_inventory_progress(inventory: Record, receipts: list[Record]) -> Record:
        """Review retained contiguous receipts; this does not authenticate or recheck them."""
        try:
            return review_recovery_inventory_progress(inventory, receipts)
        except (ValidationError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError("Invalid recovery inventory progress evidence") from exc

    def tenant_staging_reservations(self) -> Record:
        """Read tenant-local catalog reservation totals, not live capacity or locks."""
        capabilities = self.capabilities()
        if "tenant_staging_reservations" not in capabilities:
            raise TensorusError(
                "The server does not advertise tenant staging reservation status",
                code="unsupported_feature",
            )
        before = self._checked_identity()
        if (
            before.role != Role.admin
            or before.scope is not None
            or capabilities.get("role") != before.role
            or capabilities.get("scope") is not None
        ):
            raise TensorusError(
                "Staging reservation status requires an active unscoped tenant administrator",
                code="forbidden",
                status=403,
            )
        encoded = self._bounded_report(
            "GET",
            "/v1/maintenance/staging-reservations",
            MAX_TENANT_STAGING_RESPONSE_BYTES,
            "Staging reservation",
        )
        try:
            raw = decode_unique_json(encoded)
            report = TenantStagingStatus.model_validate_json(encoded, strict=True)
            if not isinstance(raw, dict) or report.tenant_id != before.id:
                raise ValueError("Staging reservation response is for a different tenant")
        except (ValidationError, TensorusError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "The server returned inconsistent staging reservation evidence",
                code="invalid_response",
            ) from exc
        after = self._checked_identity()
        if (
            after.id != before.id
            or after.key_id != before.key_id
            or after.role != before.role
            or after.scope != before.scope
        ):
            raise TensorusError(
                "The workspace identity changed during staging reservation inspection",
                code="invalid_response",
            )
        return raw

    def tenant_staging_work_quota(self) -> Record:
        """Read this tenant's pooled work-token quota, not live free capacity."""
        return self._tenant_staging_quota("work")

    def tenant_staging_input_quota(self) -> Record:
        """Read this tenant's pooled upload-input quota, not live free capacity."""
        return self._tenant_staging_quota("input")

    def _tenant_staging_quota(self, kind: Literal["work", "input"]) -> Record:
        capabilities = self.capabilities()
        if f"tenant_staging_{kind}_quota" not in capabilities:
            raise TensorusError(
                f"The server does not advertise tenant staging {kind} quota status",
                code="unsupported_feature",
            )
        before = self._checked_identity()
        if (
            before.role != Role.admin
            or before.scope is not None
            or capabilities.get("role") != before.role
            or capabilities.get("scope") is not None
        ):
            raise TensorusError(
                f"Staging {kind} quota status requires an active unscoped tenant administrator",
                code="forbidden",
                status=403,
            )
        encoded = self._bounded_report(
            "GET",
            f"/v1/maintenance/staging-{kind}-quota",
            MAX_TENANT_STAGING_RESPONSE_BYTES,
            f"Staging {kind} quota",
        )
        try:
            raw = decode_unique_json(encoded)
            report = (
                TenantWorkQuotaStatus.model_validate_json(encoded, strict=True)
                if kind == "work"
                else TenantInputQuotaStatus.model_validate_json(encoded, strict=True)
            )
            if not isinstance(raw, dict) or report.tenant_id != before.id:
                raise ValueError(f"Staging {kind} quota response is for a different tenant")
        except (ValidationError, TensorusError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                f"The server returned inconsistent staging {kind} quota evidence",
                code="invalid_response",
            ) from exc
        after = self._checked_identity()
        if (
            after.id != before.id
            or after.key_id != before.key_id
            or after.role != before.role
            or after.scope != before.scope
        ):
            raise TensorusError(
                f"The workspace identity changed during staging {kind} quota inspection",
                code="invalid_response",
            )
        return raw

    def operation_contracts(self) -> Record:
        """Return the advertised v1 operation matrix, rejecting absent or unknown formats."""
        capabilities = self.capabilities()
        plan_schema = capabilities.get("plan_schema")
        matrix = (
            plan_schema.get("x-tensorus-operation-contract-matrix")
            if isinstance(plan_schema, dict)
            else None
        )
        try:
            result = OperationContractMatrix.model_validate(matrix, strict=True)
        except ValidationError as exc:
            raise TensorusError(
                "The server operation-contract matrix is missing or unsupported"
            ) from exc
        advertised = capabilities.get("operations")
        if (
            not isinstance(advertised, list)
            or not all(isinstance(item, str) for item in advertised)
            or len(set(advertised)) != len(advertised)
            or set(advertised) != set(result.operations)
        ):
            raise TensorusError("The server operation list and contract matrix disagree")
        semantics = capabilities.get("tensor_semantics")
        reference = semantics.get("reference_semantics") if isinstance(semantics, dict) else None
        v2_operations = reference.get("plan_operations") if isinstance(reference, dict) else None
        if (
            not isinstance(v2_operations, list)
            or not all(isinstance(item, str) for item in v2_operations)
            or len(set(v2_operations)) != len(v2_operations)
            or not set(v2_operations).issubset(result.operations)
            or any(
                item.supports_semantic_policy_v2 != (name in v2_operations)
                for name, item in result.operations.items()
            )
        ):
            raise TensorusError("The server v2 operation advertisements disagree")
        return result.model_dump(mode="json")

    def keys(self) -> list[Record]:
        return self._request("GET", "/v1/keys").json()

    def keys_page(self, *, limit: int = 100, after: str | UUID | None = None) -> Record:
        """Review one bounded page of the workspace's retained credentials."""
        if type(limit) is not int or not 1 <= limit <= MAX_CREDENTIAL_PAGE_SIZE:
            raise ValueError("Credential page size is outside the supported range")
        cursor = UUID(str(after)) if after is not None else None
        credential = self.http.headers.get("Authorization")
        identity, capabilities = self._checked_capabilities()
        if (
            credential is None
            or self.http.headers.get("Authorization") != credential
            or identity.role != Role.admin
            or identity.scope is not None
        ):
            raise Forbidden("Credential review requires an unchanged unscoped administrator")
        if "credential_pages" not in capabilities and "credential_pages_v2" not in capabilities:
            raise TensorusError(
                "The server does not advertise credential pages", code="unsupported_feature"
            )
        params: dict[str, str | int] = {"limit": limit}
        if cursor is not None:
            params["after"] = str(cursor)
        encoded = self._bounded_report(
            "GET",
            "/v1/keys/pages",
            MAX_CREDENTIAL_PAGE_RESPONSE_BYTES,
            "Credential page",
            headers={"Authorization": credential},
            params=params,
        )
        try:
            raw = decode_unique_json(encoded)
            page = CredentialPage.model_validate_json(encoded, strict=True)
            result = page.model_dump(mode="json")
            if (
                not isinstance(raw, dict)
                or canonical_json(raw) != canonical_json(result)
                or page.tenant_id != identity.id
                or page.limit != limit
                or page.after != cursor
            ):
                raise ValueError("Credential page evidence is inconsistent")
        except (ValidationError, TensorusError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "The server returned invalid credential page evidence", code="invalid_response"
            ) from exc
        if (
            self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise Forbidden("The authenticated identity changed during credential review")
        return result

    @staticmethod
    def new_initial_administrator_key() -> str:
        """Generate a key that the operator must securely retain before provisioning."""
        return new_token()

    @staticmethod
    def new_workspace_invitation_token() -> str:
        """Generate a single-workspace invitation token for secure out-of-band delivery."""
        return new_invitation_token()

    @classmethod
    def service_disclosures(
        cls,
        base_url: str,
        *,
        timeout: float = 60,
        transport: httpx.BaseTransport | None = None,
    ) -> Record:
        """Read public operator declarations; they do not enforce policy or prove approval."""
        with cls(base_url, "", timeout=timeout, transport=transport) as client:
            del client.http.headers["Authorization"]
            return client.read_service_disclosures()

    def read_service_disclosures(self) -> Record:
        """Read public disclosures from this client's service, without granting authority."""
        encoded = self._bounded_report(
            "GET",
            "/v1/service-disclosures",
            MAX_SERVICE_DISCLOSURE_RESPONSE_BYTES,
            "Service disclosures",
        )
        result = self._checked_exact_response(encoded, DisclosureView, "service disclosures")
        return result.model_dump(mode="json")

    @classmethod
    def service_disclosure_event(
        cls,
        base_url: str,
        bootstrap_token: str,
        generation: int,
        *,
        timeout: float = 60,
        transport: httpx.BaseTransport | None = None,
    ) -> Record:
        """Read one retained operator disclosure event, including its reason."""
        if type(generation) is not int or not 1 <= generation <= 1000:
            raise ValueError("Invalid service disclosure generation")
        with cls(base_url, bootstrap_token, timeout=timeout, transport=transport) as client:
            encoded = client._bounded_report(
                "GET",
                f"/v1/operator/service-disclosures/{generation}",
                MAX_SERVICE_DISCLOSURE_RESPONSE_BYTES,
                "Service disclosure event",
            )
            result = client._checked_exact_response(
                encoded, DisclosureEvent, "service disclosure event"
            )
            if result.generation != generation:
                raise TensorusError("The disclosure event identifies a different generation")
            return result.model_dump(mode="json")

    @classmethod
    def publish_service_disclosures(
        cls,
        base_url: str,
        bootstrap_token: str,
        *,
        expected_generation: int,
        request_id: str | UUID,
        reason: str,
        usage: str | None,
        residency: str | None,
        retention: str | None,
        privacy: str | None,
        support: str | None,
        timeout: float = 60,
        transport: httpx.BaseTransport | None = None,
    ) -> Record:
        """Publish a complete declaration set or retry the exact retained request."""
        try:
            body = DisclosureUpdate(
                expected_generation=expected_generation,
                request_id=UUID(str(request_id)),
                reason=reason,
                usage=usage,
                residency=residency,
                retention=retention,
                privacy=privacy,
                support=support,
                confirmation="PUBLISH",
            )
        except (ValidationError, TypeError, ValueError):
            raise ValueError("Invalid service disclosure request") from None
        with cls(base_url, bootstrap_token, timeout=timeout, transport=transport) as client:
            try:
                encoded = client._bounded_report(
                    "PUT",
                    "/v1/operator/service-disclosures",
                    MAX_SERVICE_DISCLOSURE_RESPONSE_BYTES,
                    "Service disclosure update",
                    json=body.model_dump(mode="json"),
                )
                receipt = client._checked_exact_response(
                    encoded, DisclosureUpdateReceipt, "service disclosure update", mutation=True
                )
                if (
                    receipt.event.request_id != body.request_id
                    or receipt.event.expected_generation != body.expected_generation
                    or receipt.event.reason != body.reason
                    or any(
                        getattr(receipt.event, field) != getattr(body, field)
                        for field in SERVICE_DISCLOSURE_FIELDS
                    )
                ):
                    raise ValueError("Service disclosure update does not match the request")
                return receipt.model_dump(mode="json")
            except (TensorusError, ValueError) as exc:
                if (
                    isinstance(exc, TensorusError)
                    and exc.status < 500
                    and exc.code != "invalid_response"
                ):
                    raise
                raise TensorusError(
                    "Service disclosure update may have committed; retry the identical request",
                    code="service_disclosure_outcome_unknown",
                    status=502,
                ) from exc

    @classmethod
    def issue_workspace_invitation(
        cls,
        base_url: str,
        bootstrap_token: str,
        *,
        request_id: str | UUID,
        invitation_token: str,
        name: str,
        title: str,
        reason: str,
        expires_at: int,
        max_initial_expires_at: int,
        timeout: float = 60,
        transport: httpx.BaseTransport | None = None,
    ) -> Record:
        """Issue or exactly retry an invitation; retain its token and inputs beforehand."""
        try:
            body = InvitationIssue(
                request_id=UUID(str(request_id)),
                invitation_token=SecretStr(invitation_token),
                name=name,
                title=title,
                reason=reason,
                expires_at=expires_at,
                max_initial_expires_at=max_initial_expires_at,
            )
        except (ValidationError, TypeError, ValueError):
            raise ValueError("Invalid workspace invitation request") from None
        with cls(base_url, bootstrap_token, timeout=timeout, transport=transport) as client:
            try:
                encoded = client._bounded_report(
                    "POST",
                    "/v1/operator/workspace-invitations",
                    MAX_WORKSPACE_INVITATION_RESPONSE_BYTES,
                    "Workspace invitation issuance",
                    json={
                        **body.model_dump(mode="json", exclude={"invitation_token"}),
                        "invitation_token": invitation_token,
                    },
                )
                result = client._checked_exact_response(
                    encoded, InvitationReceipt, "workspace invitation", mutation=True
                )
                if (
                    result.id != body.request_id
                    or result.name != body.name
                    or result.title != body.title
                    or result.expires_at != body.expires_at
                    or result.max_initial_expires_at != body.max_initial_expires_at
                ):
                    raise ValueError("Invitation receipt does not match the request")
                return result.model_dump(mode="json")
            except (TensorusError, ValueError) as exc:
                if (
                    isinstance(exc, TensorusError)
                    and exc.status < 500
                    and exc.code != "invalid_response"
                ):
                    raise
                raise TensorusError(
                    "Invitation issuance may have committed; retain the token and retry the "
                    "identical request",
                    code="workspace_invitation_outcome_unknown",
                    status=502,
                ) from exc

    @classmethod
    def inspect_workspace_invitation(
        cls,
        base_url: str,
        invitation_token: str,
        *,
        timeout: float = 60,
        transport: httpx.BaseTransport | None = None,
    ) -> Record:
        """Inspect the single workspace and expiry authorized by an invitation."""
        if not isinstance(invitation_token, str) or not valid_invitation_token(invitation_token):
            raise ValueError("Invalid workspace invitation token")
        with cls(base_url, invitation_token, timeout=timeout, transport=transport) as client:
            encoded = client._bounded_report(
                "GET",
                "/v1/identity/workspace-invitation",
                MAX_WORKSPACE_INVITATION_RESPONSE_BYTES,
                "Workspace invitation",
            )
            result = client._checked_exact_response(encoded, InvitationView, "workspace invitation")
            return result.model_dump(mode="json")

    @classmethod
    def claim_workspace_invitation(
        cls,
        base_url: str,
        invitation_token: str,
        *,
        invitation_id: str | UUID,
        name: str,
        title: str,
        initial_api_key: str,
        expires_at: int,
        timeout: float = 60,
        transport: httpx.BaseTransport | None = None,
    ) -> Record:
        """Claim or exactly retry; verify the new key and retain all inputs until confirmed."""
        if not isinstance(invitation_token, str) or not valid_invitation_token(invitation_token):
            raise ValueError("Invalid workspace invitation token")
        try:
            body = InvitationClaim(
                invitation_id=UUID(str(invitation_id)),
                name=name,
                title=title,
                initial_api_key=SecretStr(initial_api_key),
                expires_at=expires_at,
            )
        except (ValidationError, TypeError, ValueError):
            raise ValueError("Invalid workspace invitation claim") from None
        with cls(base_url, invitation_token, timeout=timeout, transport=transport) as client:
            mutation_confirmed = False
            try:
                encoded = client._bounded_report(
                    "POST",
                    "/v1/identity/workspace-invitation/claim",
                    MAX_WORKSPACE_PROVISION_RESPONSE_BYTES,
                    "Workspace invitation claim",
                    json={
                        **body.model_dump(mode="json", exclude={"initial_api_key"}),
                        "initial_api_key": initial_api_key,
                    },
                )
                mutation_confirmed = True
                receipt = client._checked_exact_response(
                    encoded, WorkspaceProvisionReceipt, "workspace claim", mutation=True
                )
                if (
                    receipt.name != body.name
                    or receipt.title != body.title
                    or receipt.expires_at != body.expires_at
                    or receipt.key_id != initial_key_id(receipt.id)
                ):
                    raise ValueError("Workspace claim receipt does not match the request")
                identity_bytes = client._bounded_report(
                    "GET",
                    "/v1/me",
                    MAX_TENANT_STAGING_RESPONSE_BYTES,
                    "Initial workspace identity",
                    headers={"Authorization": f"Bearer {initial_api_key}"},
                )
                identity = client._checked_exact_response(
                    identity_bytes, Identity, "initial workspace identity", mutation=True
                )
                if (
                    identity.id != receipt.id
                    or identity.key_id != receipt.key_id
                    or identity.name != receipt.name
                    or identity.title != receipt.title
                    or identity.created_at != receipt.created_at
                    or identity.role != Role.admin
                    or identity.scope is not None
                    or identity.workspace_session is not None
                ):
                    raise ValueError("Initial workspace identity does not match the claim")
                return receipt.model_dump(mode="json")
            except (TensorusError, ValueError) as exc:
                if (
                    isinstance(exc, TensorusError)
                    and exc.status < 500
                    and exc.code != "invalid_response"
                    and not mutation_confirmed
                ):
                    raise
                raise TensorusError(
                    "Workspace claim may have committed; retain the invitation, initial key "
                    "and exact inputs for retry or operator inspection",
                    code="workspace_claim_outcome_unknown",
                    status=502,
                ) from exc

    @classmethod
    def workspace_invitation_status(
        cls,
        base_url: str,
        bootstrap_token: str,
        invitation_id: str | UUID,
        *,
        timeout: float = 60,
        transport: httpx.BaseTransport | None = None,
    ) -> Record:
        """Read retained operator evidence without exposing the invitation token."""
        target = UUID(str(invitation_id))
        with cls(base_url, bootstrap_token, timeout=timeout, transport=transport) as client:
            encoded = client._bounded_report(
                "GET",
                f"/v1/operator/workspace-invitations/{target}",
                MAX_WORKSPACE_INVITATION_RESPONSE_BYTES,
                "Workspace invitation status",
            )
            result = client._checked_exact_response(
                encoded, InvitationOperatorStatus, "workspace invitation status"
            )
            if result.id != target:
                raise TensorusError("The invitation status identifies a different invitation")
            return result.model_dump(mode="json")

    @classmethod
    def revoke_workspace_invitation(
        cls,
        base_url: str,
        bootstrap_token: str,
        invitation_id: str | UUID,
        *,
        request_id: str | UUID,
        reason: str,
        timeout: float = 60,
        transport: httpx.BaseTransport | None = None,
    ) -> Record:
        """Revoke or exactly retry; a previously issued workspace key is not revoked."""
        target = UUID(str(invitation_id))
        body = InvitationRevoke(
            request_id=UUID(str(request_id)), reason=reason, confirmation="REVOKE"
        )
        with cls(base_url, bootstrap_token, timeout=timeout, transport=transport) as client:
            try:
                encoded = client._bounded_report(
                    "POST",
                    f"/v1/operator/workspace-invitations/{target}/revoke",
                    MAX_WORKSPACE_INVITATION_RESPONSE_BYTES,
                    "Workspace invitation revocation",
                    json=body.model_dump(mode="json"),
                )
                result = client._checked_exact_response(
                    encoded, InvitationRevocationReceipt, "invitation revocation", mutation=True
                )
                if (
                    result.invitation_id != target
                    or result.request_id != body.request_id
                    or result.reason != body.reason
                ):
                    raise ValueError("Invitation revocation does not match the request")
                return result.model_dump(mode="json")
            except (TensorusError, ValueError) as exc:
                if (
                    isinstance(exc, TensorusError)
                    and exc.status < 500
                    and exc.code != "invalid_response"
                ):
                    raise
                raise TensorusError(
                    "Invitation revocation may have committed; retry the identical request",
                    code="workspace_invitation_revocation_outcome_unknown",
                    status=502,
                ) from exc

    @staticmethod
    def new_rotation_key() -> str:
        """Generate a replacement key to retain securely before preparing rotation."""
        return new_token()

    def prepare_key_rotation(
        self, source_key_id: str, replacement_api_key: str, expires_at: int, *, name: str
    ) -> Record:
        """Prepare and verify a replacement; retain the same inputs for an uncertain retry."""
        if not isinstance(replacement_api_key, str) or not valid_generated_token(
            replacement_api_key
        ):
            raise ValueError("The replacement credential must be a generated Tensorus key")
        source_id = UUID(source_key_id)
        request = KeyRotationRequest(
            name=name,
            replacement_api_key=SecretStr(replacement_api_key),
            expires_at=expires_at,
        )
        credential = self.http.headers.get("Authorization")
        identity, capabilities = self._checked_capabilities()
        if (
            credential is None
            or self.http.headers.get("Authorization") != credential
            or identity.role != Role.admin
            or identity.scope is not None
            or identity.workspace_session is not None
        ):
            raise Forbidden("Key rotation requires an unchanged ordinary administrator key")
        if "key_rotation" not in capabilities:
            raise TensorusError(
                "The server does not advertise retry-safe key rotation",
                code="unsupported_feature",
            )
        try:
            encoded = self._bounded_report(
                "POST",
                f"/v1/keys/{source_id}/rotation/prepare",
                MAX_KEY_ROTATION_RESPONSE_BYTES,
                "Key rotation preparation",
                headers={"Authorization": credential},
                json={
                    "name": request.name,
                    "replacement_api_key": replacement_api_key,
                    "expires_at": request.expires_at,
                },
            )
        except TensorusError as exc:
            if (
                exc.status in (400, 401, 403, 404, 409, 413, 422, 429)
                and exc.code != "invalid_response"
            ):
                raise
            raise TensorusError(
                "Key rotation may have committed; retain the replacement and retry identical "
                "inputs",
                code="key_rotation_outcome_unknown",
                status=exc.status if exc.status >= 500 else 502,
            ) from exc
        try:
            raw = decode_unique_json(encoded)
            receipt = KeyRotationReceipt.model_validate_json(encoded, strict=True)
            if (
                not isinstance(raw, dict)
                or canonical_json(raw) != canonical_json(receipt.model_dump(mode="json"))
                or receipt.tenant_id != identity.id
                or receipt.source_key_id != source_id
                or receipt.replacement_key_id != replacement_key_id(source_id, replacement_api_key)
                or receipt.name != request.name
                or receipt.expires_at != request.expires_at
            ):
                raise ValueError("Key rotation receipt is inconsistent")
            verification = self._bounded_report(
                "GET",
                "/v1/me",
                MAX_TENANT_STAGING_RESPONSE_BYTES,
                "Replacement identity",
                headers={"Authorization": f"Bearer {replacement_api_key}"},
            )
            checked = decode_unique_json(verification)
            replacement = Identity.model_validate_json(verification, strict=True)
            if (
                not isinstance(checked, dict)
                or "scope" not in checked
                or replacement.id != identity.id
                or replacement.key_id != receipt.replacement_key_id
                or replacement.role != receipt.role
                or replacement.scope is not None
                or replacement.workspace_session is not None
                or self.http.headers.get("Authorization") != credential
                or self._checked_identity() != identity
            ):
                raise ValueError("Replacement identity is inconsistent")
        except (ValidationError, TensorusError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "Key rotation could not be verified and may have committed; retain the "
                "replacement and retry identical inputs",
                code="key_rotation_outcome_unknown",
                status=502,
            ) from exc
        return receipt.model_dump(mode="json")

    @classmethod
    def provision_workspace(
        cls,
        base_url: str,
        bootstrap_token: str,
        name: str,
        initial_api_key: str,
        expires_at: int,
        *,
        title: str = "",
        timeout: float = 60,
        transport: httpx.BaseTransport | None = None,
    ) -> Record:
        """Provision or exactly retry a workspace using the same client-held credential."""
        if not isinstance(initial_api_key, str) or not valid_generated_token(initial_api_key):
            raise ValueError("The initial credential must be a generated Tensorus key")
        request = WorkspaceProvisionRequest(
            name=name,
            title=title,
            initial_api_key=SecretStr(initial_api_key),
            expires_at=expires_at,
        )
        with cls(base_url, bootstrap_token, timeout=timeout, transport=transport) as client:
            try:
                encoded = client._bounded_report(
                    "POST",
                    "/v1/tenants/provision",
                    MAX_WORKSPACE_PROVISION_RESPONSE_BYTES,
                    "Workspace provisioning",
                    json={
                        "name": request.name,
                        "title": request.title,
                        "initial_api_key": initial_api_key,
                        "expires_at": request.expires_at,
                    },
                )
                raw = decode_unique_json(encoded)
                receipt = WorkspaceProvisionReceipt.model_validate_json(encoded, strict=True)
                if (
                    not isinstance(raw, dict)
                    or canonical_json(raw) != canonical_json(receipt.model_dump(mode="json"))
                    or receipt.name != request.name
                    or receipt.title != request.title
                    or receipt.expires_at != request.expires_at
                    or receipt.key_id != initial_key_id(receipt.id)
                ):
                    raise ValueError("Workspace provisioning receipt is inconsistent")
                return receipt.model_dump(mode="json")
            except TensorusError as exc:
                if exc.status < 500 and exc.code != "invalid_response":
                    raise
                raise TensorusError(
                    "Workspace provisioning may have committed; retry with the same name, "
                    "title, initial key and expiry",
                    code="provisioning_outcome_unknown",
                    status=exc.status if exc.status >= 500 else 502,
                ) from exc
            except (ValidationError, ValueError, TypeError, RecursionError) as exc:
                raise TensorusError(
                    "Workspace provisioning returned invalid evidence and may have committed; "
                    "retry with the same name, title, initial key and expiry",
                    code="provisioning_outcome_unknown",
                    status=502,
                ) from exc

    @classmethod
    def exchange_identity(
        cls,
        base_url: str,
        workspace: str,
        assertion: str,
        *,
        expected_membership_id: str | UUID | None = None,
        timeout: float = 60,
        transport: httpx.BaseTransport | None = None,
    ) -> Record:
        """Exchange one verified, single-use OIDC assertion for a workspace session."""
        body = {"workspace": workspace}
        if expected_membership_id is not None:
            body["expected_membership_id"] = str(UUID(str(expected_membership_id)))
        with cls(base_url, assertion, timeout=timeout, transport=transport) as client:
            result = client._request("POST", "/v1/identity/exchange", json=body).json()
            if expected_membership_id is not None and (
                not isinstance(result, dict)
                or result.get("membership_id") != body["expected_membership_id"]
            ):
                raise TensorusError(
                    "The identity session did not match the selected membership and may have "
                    "been issued; inspect before starting a new login",
                    code="identity_exchange_outcome_unknown",
                    status=502,
                )
            return result

    @classmethod
    def browser_login_configuration(
        cls,
        base_url: str,
        *,
        timeout: float = 60,
        transport: httpx.BaseTransport | None = None,
    ) -> Record:
        """Read the fixed public-client login configuration without a workspace key."""
        with cls(base_url, "", timeout=timeout, transport=transport) as client:
            del client.http.headers["Authorization"]
            encoded = client._bounded_report(
                "GET", "/v1/identity/browser-login", MAX_LOGIN_RESPONSE_BYTES, "Browser login"
            )
            result = client._checked_exact_response(
                encoded, BrowserLoginConfiguration, "browser login configuration"
            )
            return result.model_dump(mode="json")

    @classmethod
    def complete_browser_login(
        cls,
        base_url: str,
        *,
        workspace: str,
        code: str,
        code_verifier: str,
        nonce: str,
        policy_hash: str,
        expected_membership_id: str | UUID | None = None,
        timeout: float = 60,
        transport: httpx.BaseTransport | None = None,
    ) -> Record:
        """Complete a caller-validated browser transaction; a lost response may issue a session."""
        try:
            body = BrowserLoginComplete(
                workspace=workspace,
                code=code,
                code_verifier=code_verifier,
                nonce=nonce,
                policy_hash=policy_hash,
                expected_membership_id=(
                    UUID(str(expected_membership_id))
                    if expected_membership_id is not None
                    else None
                ),
            )
        except ValidationError:
            raise ValueError("Invalid browser login request") from None
        with cls(base_url, "", timeout=timeout, transport=transport) as client:
            del client.http.headers["Authorization"]
            try:
                encoded = client._bounded_report(
                    "POST",
                    "/v1/identity/browser-login/complete",
                    MAX_LOGIN_RESPONSE_BYTES,
                    "Browser login",
                    json=body.model_dump(mode="json"),
                )
                result = client._checked_exact_response(
                    encoded, IdentitySession, "browser login session", mutation=True
                )
                if not valid_generated_token(result.api_key) or (
                    body.expected_membership_id is not None
                    and result.membership_id != body.expected_membership_id
                ):
                    raise ValueError("Invalid workspace session credential")
                return result.model_dump(mode="json")
            except (TensorusError, ValueError) as exc:
                if (
                    isinstance(exc, TensorusError)
                    and exc.status < 500
                    and exc.code != "invalid_response"
                ):
                    raise
                raise TensorusError(
                    "Browser login may have issued a session; start a new login",
                    code="browser_login_outcome_unknown",
                    status=502,
                ) from exc

    def workspace_members(self, *, limit: int = 100, offset: int = 0) -> Record:
        credential = self.http.headers.get("Authorization")
        if credential is None:
            raise Forbidden("An authenticated administrator is required")
        identity = self._checked_identity_for_credential(credential)
        if identity.role != Role.admin or identity.scope is not None:
            raise Forbidden("Workspace membership requires an unscoped administrator")
        result = self._request(
            "GET",
            "/v1/workspace/members",
            headers={"Authorization": credential},
            params={"limit": limit, "offset": offset},
        ).json()
        if (
            self.http.headers.get("Authorization") != credential
            or self._checked_identity_for_credential(credential) != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise Forbidden("The authenticated administrator changed during membership inspection")
        return result

    def _service_actor_context(self, *, admin: bool) -> tuple[Identity, str]:
        credential = self.http.headers.get("Authorization")
        identity, capabilities = self._checked_capabilities()
        if "workspace_service_actors" not in capabilities:
            raise TensorusError(
                "The server does not advertise service actors", code="unsupported_feature"
            )
        if admin and (identity.role != Role.admin or identity.scope is not None):
            raise Forbidden("Service actor administration requires an unscoped administrator")
        if credential is None or self.http.headers.get("Authorization") != credential:
            raise Forbidden("The authenticated credential changed during service actor review")
        return identity, credential

    def workspace_service_actors(self, *, limit: int = 100, offset: int = 0) -> Record:
        """List provider-backed service memberships without transferring authority."""
        if type(limit) is not int or not 1 <= limit <= MAX_SERVICE_ACTOR_PAGE_SIZE:
            raise ValueError("Service actor page size is outside its supported range")
        if type(offset) is not int or not 0 <= offset <= 10_000:
            raise ValueError("Service actor page offset is outside its supported range")
        identity, credential = self._service_actor_context(admin=True)
        encoded = self._bounded_report(
            "GET",
            "/v1/workspace/service-actors",
            MAX_SERVICE_ACTOR_RESPONSE_BYTES,
            "Service actor page",
            params={"limit": limit, "offset": offset},
        )
        page = self._checked_exact_response(encoded, ServiceActorPage, "service actor page")
        if page.tenant_id != identity.id or page.limit != limit or page.offset != offset:
            raise TensorusError("The server returned a different service actor page")
        if (
            self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise Forbidden("The authenticated identity changed during service actor review")
        return page.model_dump(mode="json")

    def invite_service_actor(
        self,
        subject: str,
        *,
        display_name: str,
        expected_issuer: str,
        expected_policy_hash: str,
        request_id: str | UUID,
        role: Role = Role.reader,
    ) -> Record:
        """Invite a new provider subject with an exact, retryable membership identity."""
        request = ServiceActorInvitation(
            request_id=UUID(str(request_id)),
            subject=subject,
            expected_issuer=expected_issuer,
            expected_policy_hash=expected_policy_hash,
            role=role,
            display_name=display_name,
            confirmation="INVITE",
        )
        identity, credential = self._service_actor_context(admin=True)
        try:
            encoded = self._bounded_report(
                "POST",
                "/v1/workspace/service-actors",
                MAX_SERVICE_ACTOR_RESPONSE_BYTES,
                "Service actor invitation",
                json=request.model_dump(mode="json"),
            )
        except TensorusError as exc:
            if exc.status < 500 and exc.code != "invalid_response":
                raise
            raise TensorusError(
                "The service actor invitation may have committed; retry the exact request "
                "with the same active issuing credential and provider",
                code="service_actor_invitation_outcome_unknown",
                status=502,
            ) from exc
        try:
            receipt = self._checked_exact_response(
                encoded, ServiceActorInvitationReceipt, "service actor invitation", mutation=True
            )
            member = receipt.actor.member
            if (
                receipt.request_id != request.request_id
                or member.tenant_id != identity.id
                or member.subject != request.subject
                or member.issuer != request.expected_issuer
                or receipt.actor.invited_policy_hash != request.expected_policy_hash
                or member.invited_by != str(identity.key_id)
                or receipt.actor.display_name != request.display_name
                or receipt.actor.invited_role != request.role
                or self.http.headers.get("Authorization") != credential
                or self._checked_identity() != identity
                or self.http.headers.get("Authorization") != credential
            ):
                raise ValueError("Service actor invitation evidence does not match the request")
            return receipt.model_dump(mode="json")
        except (TensorusError, ValueError, TypeError) as exc:
            raise TensorusError(
                "The service actor invitation may have committed; retry the exact request "
                "with the same active issuing credential and provider",
                code="service_actor_invitation_outcome_unknown",
                status=502,
            ) from exc

    def service_actor_identity(self) -> Record:
        """Inspect only this session's or delegated grant's service actor association."""
        identity, credential = self._service_actor_context(admin=False)
        encoded = self._bounded_report(
            "GET",
            "/v1/identity/service-actor",
            MAX_SERVICE_ACTOR_SELF_RESPONSE_BYTES,
            "Service actor identity",
        )
        result = self._checked_exact_response(
            encoded, ServiceActorIdentity, "service actor identity"
        )
        expected_source = (
            "delegated_grant"
            if identity.scope is not None
            else "workspace_session"
            if identity.workspace_session is not None
            else "other"
        )
        if (
            result.tenant_id != identity.id
            or result.key_id != identity.key_id
            or result.source != expected_source
            or (
                identity.workspace_session is not None
                and result.actor is not None
                and result.actor.membership_id != identity.workspace_session.membership_id
            )
        ):
            raise TensorusError("The server returned a different service actor identity")
        if (
            self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise Forbidden("The authenticated identity changed during service actor review")
        return result.model_dump(mode="json")

    def service_actor_recipient(self) -> Record:
        """Inspect this credential's recipient binding, without administering identities."""
        credential = self.http.headers.get("Authorization")
        identity, capabilities = self._checked_capabilities()
        if "service_actor_recipient_grants" not in capabilities:
            raise TensorusError(
                "The server does not advertise service actor recipient grants",
                code="unsupported_feature",
            )
        if credential is None or self.http.headers.get("Authorization") != credential:
            raise Forbidden("The authenticated credential changed during recipient review")
        encoded = self._bounded_report(
            "GET",
            "/v1/identity/service-actor-recipient",
            4_096,
            "Service actor recipient",
        )
        result = self._checked_exact_response(encoded, RecipientStatus, "service actor recipient")
        if (
            result.tenant_id != identity.id
            or result.key_id != identity.key_id
            or (identity.scope is None and result.recipient is not None)
        ):
            raise TensorusError("The server returned a different service actor recipient")
        if (
            self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise Forbidden("The authenticated identity changed during recipient review")
        return result.model_dump(mode="json")

    def service_actor_recipient_review(self, key_id: str | UUID) -> Record:
        """Review retained recipient-binding evidence, not current grant authorization."""
        requested = UUID(str(key_id))
        credential = self.http.headers.get("Authorization")
        identity, capabilities = self._checked_capabilities()
        if "service_actor_recipient_review" not in capabilities:
            raise TensorusError(
                "The server does not advertise recipient binding review", code="unsupported_feature"
            )
        if (
            credential is None
            or self.http.headers.get("Authorization") != credential
            or identity.role != Role.admin
            or identity.scope is not None
        ):
            raise Forbidden("Recipient review requires an unchanged unscoped administrator")
        encoded = self._bounded_report(
            "GET",
            f"/v1/keys/{requested}/service-actor-recipient",
            MAX_REVIEW_RESPONSE_BYTES,
            "Recipient binding review",
            headers={"Authorization": credential},
        )
        result = self._checked_exact_response(
            encoded, RecipientBindingReview, "recipient binding review"
        )
        if result.tenant_id != identity.id or result.key_id != requested:
            raise TensorusError(
                "The server returned a different recipient binding review", code="invalid_response"
            )
        if (
            self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise Forbidden("The authenticated identity changed during recipient review")
        return result.model_dump(mode="json")

    def service_actor_recipient_capacity(self) -> Record:
        """Inspect the configured recipient-grant guard without reserving a slot."""
        credential = self.http.headers.get("Authorization")
        identity, capabilities = self._checked_capabilities()
        marker = capabilities.get("service_actor_recipient_capacity")
        if marker is None:
            raise TensorusError(
                "The server does not advertise recipient grant capacity",
                code="unsupported_feature",
            )
        if (
            credential is None
            or self.http.headers.get("Authorization") != credential
            or identity.role != Role.admin
            or identity.scope is not None
        ):
            raise Forbidden("Recipient capacity requires an unchanged unscoped administrator")
        encoded = self._bounded_report(
            "GET",
            "/v1/workspace/service-actor-grant-capacity",
            MAX_CAPACITY_RESPONSE_BYTES,
            "Recipient grant capacity",
            headers={"Authorization": credential},
        )
        result = self._checked_exact_response(
            encoded, RecipientCapacityStatus, "recipient grant capacity"
        )
        if result.tenant_id != identity.id or result.unrevoked_limit != marker["unrevoked_limit"]:
            raise TensorusError(
                "The server returned a different recipient grant capacity", code="invalid_response"
            )
        if (
            self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise Forbidden("The authenticated identity changed during recipient capacity review")
        return result.model_dump(mode="json")

    def identity_workspaces(self, *, limit: int = 100, after: str | UUID | None = None) -> Record:
        """Discover this membership identity's available workspaces without transferring access."""
        if type(limit) is not int or not 1 <= limit <= MAX_WORKSPACE_DIRECTORY_PAGE_SIZE:
            raise ValueError("Workspace directory limit is outside the supported range")
        cursor = UUID(str(after)) if after is not None else None
        credential = self.http.headers.get("Authorization")
        identity, capabilities = self._checked_capabilities()
        if identity.scope is not None or identity.workspace_session is None:
            raise Forbidden("A workspace membership session is required")
        if "workspace_directory" not in capabilities:
            raise TensorusError(
                "The server does not advertise workspace discovery", code="unsupported_feature"
            )
        if credential is None or self.http.headers.get("Authorization") != credential:
            raise Forbidden("The authenticated credential changed during workspace discovery")
        params: dict[str, str | int] = {"limit": limit}
        if cursor is not None:
            params["after"] = str(cursor)
        encoded = self._bounded_report(
            "GET",
            "/v1/identity/workspaces",
            MAX_WORKSPACE_DIRECTORY_RESPONSE_BYTES,
            "Workspace directory",
            params=params,
        )
        try:
            raw = decode_unique_json(encoded)
            page = WorkspaceDirectoryPage.model_validate_json(encoded, strict=True)
            result = page.model_dump(mode="json")
            if (
                not isinstance(raw, dict)
                or canonical_json(raw) != canonical_json(result)
                or page.current_tenant_id != identity.id
                or page.current_membership_id != identity.workspace_session.membership_id
                or page.limit != limit
                or page.after != cursor
                or any(
                    item.tenant_id == identity.id and item.role != identity.role
                    for item in page.items
                )
            ):
                raise ValueError("Workspace directory evidence is inconsistent")
        except (ValidationError, TensorusError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "The server returned invalid workspace directory evidence", code="invalid_response"
            ) from exc
        if (
            self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise Forbidden("The authenticated identity changed during workspace discovery")
        return result

    def _organization_context(
        self, *, require_detachment: bool = False, require_detachment_receipt: bool = False
    ) -> tuple[Identity, str]:
        credential = self.http.headers.get("Authorization")
        identity, capabilities = self._checked_capabilities()
        if (
            credential is None
            or self.http.headers.get("Authorization") != credential
            or identity.scope is not None
            or identity.workspace_session is None
        ):
            raise Forbidden("An unchanged workspace membership session is required")
        if "organization_governance" not in capabilities:
            raise TensorusError(
                "The server does not advertise organization governance", code="unsupported_feature"
            )
        if "organization_session_fence" not in capabilities:
            raise TensorusError(
                "The server does not advertise organization session fencing",
                code="unsupported_feature",
            )
        if require_detachment and "organization_workspace_detachment" not in capabilities:
            raise TensorusError(
                "The server does not advertise organization workspace detachment",
                code="unsupported_feature",
            )
        if require_detachment_receipt and "organization_detachment_receipt_v1" not in capabilities:
            raise TensorusError(
                "The server does not advertise organization detachment receipt recovery",
                code="unsupported_feature",
            )
        return identity, credential

    def _organization_recheck(self, identity: Identity, credential: str) -> None:
        if (
            self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise Forbidden("The authenticated identity changed during organization review")

    def _organization_mutation[T: BaseModel](
        self,
        method: str,
        path: str,
        request: BaseModel,
        model: type[T],
        credential: str,
    ) -> T:
        try:
            encoded = self._bounded_report(
                method,
                path,
                MAX_ORGANIZATION_RESPONSE_BYTES,
                "Organization operation",
                headers={"Authorization": credential},
                json=request.model_dump(mode="json"),
            )
            return self._checked_exact_response(
                encoded, model, "organization operation", mutation=True
            )
        except TensorusError as exc:
            if exc.status < 500 and exc.code != "invalid_response":
                raise
            raise TensorusError(
                "The organization operation may have committed; retry the identical request ID",
                code="organization_outcome_unknown",
                status=exc.status if exc.status >= 500 else 502,
            ) from exc

    def _organization_mutation_check(
        self, valid: bool, identity: Identity, credential: str
    ) -> None:
        try:
            if not valid:
                raise ValueError("Organization response does not match the request")
            self._organization_recheck(identity, credential)
        except (TensorusError, ValueError) as exc:
            raise TensorusError(
                "The organization response could not be verified and may have committed; "
                "retry the identical request ID",
                code="organization_outcome_unknown",
                status=502,
            ) from exc

    def identity_organizations(self) -> Record:
        """Discover organizations for this verified user without transferring workspace access."""
        identity, credential = self._organization_context()
        encoded = self._bounded_report(
            "GET",
            "/v1/identity/organizations",
            MAX_ORGANIZATION_RESPONSE_BYTES,
            "Organization directory",
            headers={"Authorization": credential},
        )
        result = self._checked_exact_response(
            encoded, OrganizationDirectory, "organization directory"
        )
        if (
            result.current_tenant_id != identity.id
            or identity.workspace_session is None
            or result.current_membership_id != identity.workspace_session.membership_id
        ):
            raise TensorusError(
                "The organization directory is inconsistent", code="invalid_response"
            )
        self._organization_recheck(identity, credential)
        return result.model_dump(mode="json")

    def create_organization(
        self, *, request_id: str | UUID, name: str, title: str = "", confirmation: Literal["CREATE"]
    ) -> Record:
        """Create an organization from the current workspace and its invited or active members."""
        request = OrganizationCreate(
            request_id=UUID(str(request_id)), name=name, title=title, confirmation=confirmation
        )
        identity, credential = self._organization_context()
        if identity.role != Role.admin:
            raise Forbidden("An unscoped workspace administrator session is required")
        result = self._organization_mutation(
            "POST", "/v1/organizations", request, OrganizationView, credential
        )
        self._organization_mutation_check(
            result.id == request.request_id
            and result.name == request.name
            and result.title == request.title
            and result.current_workspace_attached
            and result.status == "active"
            and result.role == "owner",
            identity,
            credential,
        )
        return result.model_dump(mode="json")

    def attach_organization_workspace(
        self,
        organization_id: str | UUID,
        *,
        request_id: str | UUID,
        confirmation: Literal["ATTACH"],
    ) -> Record:
        """Attach this workspace when its existing members have organization memberships."""
        organization = UUID(str(organization_id))
        request = OrganizationAttach(request_id=UUID(str(request_id)), confirmation=confirmation)
        identity, credential = self._organization_context()
        if identity.role != Role.admin:
            raise Forbidden("An unscoped workspace administrator session is required")
        result = self._organization_mutation(
            "POST",
            f"/v1/organizations/{organization}/workspaces/attach-current",
            request,
            OrganizationWorkspace,
            credential,
        )
        self._organization_mutation_check(
            result.organization_id == organization
            and result.tenant_id == identity.id
            and result.request_id == request.request_id,
            identity,
            credential,
        )
        return result.model_dump(mode="json")

    def workspace_organization_detachment(self) -> Record:
        """Inspect the current workspace association and any retained detachment receipt."""
        identity, credential = self._organization_context(require_detachment=True)
        if identity.role != Role.admin:
            raise Forbidden("An unscoped workspace administrator session is required")
        encoded = self._bounded_report(
            "GET",
            "/v1/workspace/organization-detachment",
            MAX_ORGANIZATION_DETACHMENT_RESPONSE_BYTES,
            "Organization detachment status",
            headers={"Authorization": credential},
        )
        result = self._checked_exact_response(
            encoded, DetachmentStatus, "organization detachment status"
        )
        if result.tenant_id != identity.id:
            raise TensorusError(
                "The organization detachment status is inconsistent", code="invalid_response"
            )
        self._organization_recheck(identity, credential)
        return result.model_dump(mode="json")

    def detach_organization_workspace(
        self,
        organization_id: str | UUID,
        *,
        request_id: str | UUID,
        attachment_request_id: str | UUID,
        reason: str,
        confirmation: Literal["DETACH"],
    ) -> Record:
        """Detach once, retaining association history and revoking this workspace session."""
        organization = UUID(str(organization_id))
        request = DetachmentRequest(
            request_id=UUID(str(request_id)),
            attachment_request_id=UUID(str(attachment_request_id)),
            reason=reason,
            confirmation=confirmation,
        )
        identity, credential = self._organization_context(require_detachment=True)
        if identity.role != Role.admin:
            raise Forbidden("An unscoped workspace administrator session is required")
        try:
            encoded = self._bounded_report(
                "POST",
                f"/v1/organizations/{organization}/workspaces/detach-current",
                MAX_ORGANIZATION_DETACHMENT_RESPONSE_BYTES,
                "Organization detachment",
                headers={"Authorization": credential},
                json=request.model_dump(mode="json"),
            )
            result = self._checked_exact_response(
                encoded, DetachmentReceipt, "organization detachment", mutation=True
            )
            if (
                result.tenant_id != identity.id
                or result.organization_id != organization
                or result.attachment_request_id != request.attachment_request_id
                or result.request_id != request.request_id
                or result.reason != request.reason
                or self.http.headers.get("Authorization") != credential
            ):
                raise ValueError("Detachment receipt does not match the request")
            return result.model_dump(mode="json")
        except (TensorusError, ValueError) as exc:
            raise TensorusError(
                "Detachment may have committed; discard this session, authenticate again and "
                "inspect the status or retry the identical request",
                code="organization_detachment_outcome_unknown",
                status=(
                    exc.status
                    if isinstance(exc, TensorusError) and exc.code != "invalid_response"
                    else 502
                ),
            ) from exc
        finally:
            if self.http.headers.get("Authorization") == credential:
                self.http.headers.pop("Authorization", None)

    @staticmethod
    def new_organization_detachment_receipt_key() -> tuple[str, str]:
        """Create a client-held receipt ID and secret; protect them until lookup finishes."""
        return str(uuid4()), "tor_" + secrets.token_urlsafe(32)

    @staticmethod
    def _check_organization_detachment_receipt(
        result: RecoveredDetachmentReceipt,
        *,
        request: DetachmentRequest,
        receipt_id: UUID,
        organization_id: UUID,
        tenant_id: UUID,
        actor_session_id: UUID,
        actor_membership_id: UUID,
    ) -> None:
        receipt = result.receipt
        if (
            result.receipt_id != receipt_id
            or result.actor_session_id != actor_session_id
            or result.actor_membership_id != actor_membership_id
            or receipt.tenant_id != tenant_id
            or receipt.organization_id != organization_id
            or receipt.attachment_request_id != request.attachment_request_id
            or receipt.request_id != request.request_id
            or receipt.reason != request.reason
        ):
            raise ValueError("Detachment receipt does not match the reviewed request")

    def recover_organization_detachment_receipt(
        self,
        *,
        receipt_id: str | UUID,
        receipt_secret: str,
        request: DetachmentRequest,
        organization_id: str | UUID,
        tenant_id: str | UUID,
        actor_session_id: str | UUID,
        actor_membership_id: str | UUID,
    ) -> Record:
        """Read a matching receipt without authenticating the revoked workspace session."""
        checked_id = UUID(str(receipt_id))
        checked_request = DetachmentRequest.model_validate(request.model_dump(mode="json"))
        organization = UUID(str(organization_id))
        tenant = UUID(str(tenant_id))
        session = UUID(str(actor_session_id))
        membership = UUID(str(actor_membership_id))
        validate_detachment_receipt_secret(receipt_secret)
        try:
            encoded = self._bounded_report(
                "GET",
                f"/v1/identity/organization-detachment-receipts/{checked_id}",
                MAX_ORGANIZATION_DETACHMENT_RESPONSE_BYTES,
                "Organization detachment receipt",
                headers={"Authorization": f"Bearer {receipt_secret}"},
                follow_redirects=False,
            )
            result = self._checked_exact_response(
                encoded, RecoveredDetachmentReceipt, "organization detachment receipt"
            )
        except TensorusError as exc:
            raise TensorusError(
                "The organization detachment receipt could not be verified",
                code=(
                    "invalid_response"
                    if exc.code == "invalid_response"
                    else "detachment_receipt_unavailable"
                ),
                status=exc.status,
            ) from None
        try:
            self._check_organization_detachment_receipt(
                result,
                request=checked_request,
                receipt_id=checked_id,
                organization_id=organization,
                tenant_id=tenant,
                actor_session_id=session,
                actor_membership_id=membership,
            )
        except ValueError as exc:
            raise TensorusError(
                "The organization detachment receipt identity is inconsistent",
                code="invalid_response",
            ) from exc
        return result.model_dump(mode="json")

    def detach_organization_workspace_with_receipt(
        self,
        organization_id: str | UUID,
        *,
        request_id: str | UUID,
        attachment_request_id: str | UUID,
        reason: str,
        confirmation: Literal["DETACH"],
        receipt_recovery: tuple[str | UUID, str] | None = None,
    ) -> Record:
        """Detach once and inspect the same receipt if the response is uncertain."""
        organization = UUID(str(organization_id))
        request = DetachmentRequest(
            request_id=UUID(str(request_id)),
            attachment_request_id=UUID(str(attachment_request_id)),
            reason=reason,
            confirmation=confirmation,
        )
        identity, credential = self._organization_context(
            require_detachment=True, require_detachment_receipt=True
        )
        if identity.role != Role.admin or identity.workspace_session is None:
            raise Forbidden("An unscoped workspace administrator session is required")
        recovery = receipt_recovery or self.new_organization_detachment_receipt_key()
        receipt_id = UUID(str(recovery[0]))
        validate_detachment_receipt_secret(recovery[1])
        if not receipt_id.int or not request.request_id.int or receipt_id == request.request_id:
            raise ValueError("Distinct non-nil detachment and receipt IDs are required")
        try:
            encoded = self._bounded_report(
                "POST",
                f"/v1/organizations/{organization}/workspaces/detach-current-with-receipt",
                MAX_ORGANIZATION_DETACHMENT_RESPONSE_BYTES,
                "Organization detachment",
                headers={
                    "Authorization": credential,
                    "X-Tensorus-Organization-Detachment-Receipt-Key": recovery[1],
                },
                json={**request.model_dump(mode="json"), "receipt_id": str(receipt_id)},
                follow_redirects=False,
            )
            result = self._checked_exact_response(
                encoded, RecoveredDetachmentReceipt, "organization detachment", mutation=True
            )
            self._check_organization_detachment_receipt(
                result,
                request=request,
                receipt_id=receipt_id,
                organization_id=organization,
                tenant_id=identity.id,
                actor_session_id=identity.key_id,
                actor_membership_id=identity.workspace_session.membership_id,
            )
            if self.http.headers.get("Authorization") != credential:
                raise ValueError("The detachment credential changed")
            return result.model_dump(mode="json")
        except (TensorusError, ValidationError, ValueError, TypeError, RecursionError) as exc:
            lookup_note = "the authenticated credential changed"
            if self.http.headers.get("Authorization") == credential:
                try:
                    recovered = self.recover_organization_detachment_receipt(
                        receipt_id=receipt_id,
                        receipt_secret=recovery[1],
                        request=request,
                        organization_id=organization,
                        tenant_id=identity.id,
                        actor_session_id=identity.key_id,
                        actor_membership_id=identity.workspace_session.membership_id,
                    )
                    if self.http.headers.get("Authorization") != credential:
                        raise Forbidden("The detachment credential changed")
                    return recovered
                except (
                    TensorusError,
                    ValidationError,
                    ValueError,
                    TypeError,
                    RecursionError,
                ) as lookup_error:
                    lookup_note = (
                        f"HTTP {lookup_error.status}"
                        if isinstance(lookup_error, TensorusError)
                        and type(lookup_error.status) is int
                        and 100 <= lookup_error.status <= 599
                        else "invalid evidence"
                    )
            raise TensorusError(
                "Detachment may have committed; receipt lookup did not confirm it "
                f"({lookup_note}). A missing receipt does not prove that detachment failed; "
                "authenticate again or ask an administrator to inspect the request",
                code="organization_detachment_outcome_unknown",
                status=exc.status if isinstance(exc, TensorusError) else 502,
                details={
                    "request_id": str(request.request_id),
                    "receipt_id": str(receipt_id),
                    "organization_id": str(organization),
                    "tenant_id": str(identity.id),
                    "actor_session_id": str(identity.key_id),
                    "actor_membership_id": str(identity.workspace_session.membership_id),
                },
            ) from None
        finally:
            if self.http.headers.get("Authorization") == credential:
                self.http.headers.pop("Authorization", None)

    def organization_members(
        self, organization_id: str | UUID, *, limit: int = 25, offset: int = 0
    ) -> Record:
        organization = UUID(str(organization_id))
        if type(limit) is not int or not 1 <= limit <= 25:
            raise ValueError("Organization member limit must be between 1 and 25")
        if type(offset) is not int or not 0 <= offset <= 10_000:
            raise ValueError("Organization member offset is outside the supported range")
        identity, credential = self._organization_context()
        encoded = self._bounded_report(
            "GET",
            f"/v1/organizations/{organization}/members",
            MAX_ORGANIZATION_RESPONSE_BYTES,
            "Organization members",
            headers={"Authorization": credential},
            params={"limit": limit, "offset": offset},
        )
        result = self._checked_exact_response(
            encoded, OrganizationMemberPage, "organization members"
        )
        if (
            result.organization_id != organization
            or result.limit != limit
            or result.offset != offset
        ):
            raise TensorusError("Organization member page is inconsistent", code="invalid_response")
        self._organization_recheck(identity, credential)
        return result.model_dump(mode="json")

    def organization_workspaces(self, organization_id: str | UUID) -> Record:
        organization = UUID(str(organization_id))
        identity, credential = self._organization_context()
        encoded = self._bounded_report(
            "GET",
            f"/v1/organizations/{organization}/workspaces",
            MAX_ORGANIZATION_RESPONSE_BYTES,
            "Organization workspaces",
            headers={"Authorization": credential},
        )
        result = self._checked_exact_response(
            encoded, OrganizationWorkspaceList, "organization workspaces"
        )
        if result.organization_id != organization:
            raise TensorusError("Organization workspaces are inconsistent", code="invalid_response")
        self._organization_recheck(identity, credential)
        return result.model_dump(mode="json")

    def invite_organization_member(
        self,
        organization_id: str | UUID,
        *,
        request_id: str | UUID,
        subject: str,
        role: OrgRole,
    ) -> Record:
        organization = UUID(str(organization_id))
        request = OrganizationInvite(request_id=UUID(str(request_id)), subject=subject, role=role)
        identity, credential = self._organization_context()
        result = self._organization_mutation(
            "POST",
            f"/v1/organizations/{organization}/members",
            request,
            OrganizationMember,
            credential,
        )
        self._organization_mutation_check(
            result.id == request.request_id
            and result.organization_id == organization
            and result.subject == request.subject
            and result.role == request.role
            and result.generation == 1
            and result.status in ("invited", "active"),
            identity,
            credential,
        )
        return result.model_dump(mode="json")

    def update_organization_member(
        self,
        organization_id: str | UUID,
        member_id: str | UUID,
        *,
        request_id: str | UUID,
        expected_generation: int,
        role: OrgRole,
        status: OrgStatus,
        reason: str,
        confirmation: Literal["UPDATE"],
    ) -> Record:
        organization, member = UUID(str(organization_id)), UUID(str(member_id))
        request = OrganizationMemberUpdate(
            request_id=UUID(str(request_id)),
            expected_generation=expected_generation,
            role=role,
            status=status,
            reason=reason,
            confirmation=confirmation,
        )
        identity, credential = self._organization_context()
        result = self._organization_mutation(
            "PUT",
            f"/v1/organizations/{organization}/members/{member}",
            request,
            OrganizationMember,
            credential,
        )
        self._organization_mutation_check(
            result.id == member
            and result.organization_id == organization
            and result.generation == request.expected_generation + 1
            and result.role == request.role
            and result.status == request.status,
            identity,
            credential,
        )
        return result.model_dump(mode="json")

    def workspace_identity_sessions(self) -> Record:
        """Inspect the current, auditable workspace identity-session state."""
        credential = self.http.headers.get("Authorization")
        identity, capabilities = self._checked_capabilities()
        if (
            credential is None
            or self.http.headers.get("Authorization") != credential
            or identity.role != Role.admin
            or identity.scope is not None
        ):
            raise Forbidden("Identity-session review requires an unchanged unscoped administrator")
        if "workspace_identity_sessions" not in capabilities:
            raise TensorusError(
                "The server does not advertise identity-session coordination",
                code="unsupported_feature",
            )
        encoded = self._bounded_report(
            "GET",
            "/v1/workspace/identity/sessions",
            MAX_IDENTITY_SESSIONS_RESPONSE_BYTES,
            "Identity-session status",
            headers={"Authorization": credential},
        )
        result = self._checked_exact_response(encoded, IdentitySessionsStatus, "identity-session")
        if (
            result.tenant_id != identity.id
            or self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
        ):
            raise Forbidden("The workspace identity changed during session review")
        return result.model_dump(mode="json")

    def transition_workspace_identity_sessions(
        self,
        action: Literal["pause", "resume"],
        *,
        expected_generation: int,
        request_id: str | UUID,
        reason: str,
        confirmation: Literal["PAUSE", "RESUME"],
    ) -> Record:
        """Pause or resume session exchange; retain identical inputs for an uncertain retry."""
        if action not in ("pause", "resume"):
            raise ValueError("The identity-session action must be pause or resume")
        request = IdentitySessionsTransition(
            expected_generation=expected_generation,
            request_id=UUID(str(request_id)),
            reason=reason,
            confirmation=confirmation,
        )
        if request.confirmation != ("PAUSE" if action == "pause" else "RESUME"):
            raise ValueError("Identity-session confirmation does not match the action")
        credential = self.http.headers.get("Authorization")
        identity, capabilities = self._checked_capabilities()
        if (
            credential is None
            or self.http.headers.get("Authorization") != credential
            or identity.role != Role.admin
            or identity.scope is not None
            or identity.workspace_session is not None
        ):
            raise Forbidden("An unchanged ordinary administrator key is required")
        if "workspace_identity_sessions" not in capabilities:
            raise TensorusError(
                "The server does not advertise identity-session coordination",
                code="unsupported_feature",
            )
        try:
            encoded = self._bounded_report(
                "POST",
                f"/v1/workspace/identity/sessions/{action}",
                MAX_IDENTITY_SESSIONS_RESPONSE_BYTES,
                "Identity-session transition",
                headers={"Authorization": credential},
                json=request.model_dump(mode="json"),
            )
        except TensorusError as exc:
            if exc.status < 500 and exc.code != "invalid_response":
                raise
            raise TensorusError(
                "The identity-session transition may have committed; retry identical inputs "
                "while this remains the current transition",
                code="identity_sessions_outcome_unknown",
                status=exc.status if exc.status >= 500 else 502,
            ) from exc
        try:
            result = self._checked_exact_response(
                encoded, IdentitySessionsStatus, "identity-session transition", mutation=True
            )
            event = result.last_event
            if (
                result.tenant_id != identity.id
                or result.state != ("paused" if action == "pause" else "active")
                or result.generation != request.expected_generation + 1
                or event is None
                or event.id != request.request_id
                or event.expected_generation != request.expected_generation
                or event.reason != request.reason
                or self.http.headers.get("Authorization") != credential
                or self._checked_identity() != identity
            ):
                raise ValueError("Identity-session transition evidence is inconsistent")
            return result.model_dump(mode="json")
        except (TensorusError, ValidationError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "The identity-session transition could not be verified and may have committed; "
                "retry identical inputs while this remains the current transition",
                code="identity_sessions_outcome_unknown",
                status=502,
            ) from exc

    def revoke_workspace_session(self) -> Record:
        """Revoke this membership session and its issued grants, then discard the SDK credential."""
        credential = self.http.headers.get("Authorization")
        identity, capabilities = self._checked_capabilities()
        if credential is None or self.http.headers.get("Authorization") != credential:
            raise Forbidden("The authenticated credential changed during session review")
        if identity.scope is not None or identity.workspace_session is None:
            raise Forbidden("A workspace membership session is required")
        if "workspace_session_revocation" not in capabilities:
            raise TensorusError(
                "The server does not advertise workspace session revocation",
                code="unsupported_feature",
            )
        try:
            encoded = self._bounded_report(
                "POST",
                "/v1/identity/session/revoke",
                4096,
                "Workspace session revocation",
                headers={"Authorization": credential},
            )
            raw = decode_unique_json(encoded)
            receipt = WorkspaceSessionRevocation.model_validate_json(encoded, strict=True)
            if (
                not isinstance(raw, dict)
                or canonical_json(raw) != canonical_json(receipt.model_dump(mode="json"))
                or receipt.tenant_id != identity.id
                or receipt.key_id != identity.key_id
                or receipt.membership_id != identity.workspace_session.membership_id
            ):
                raise ValueError("Session revocation receipt is inconsistent")
            return receipt.model_dump(mode="json")
        except TensorusError as exc:
            raise TensorusError(
                "Session revocation may have committed; discard this credential and ask an "
                f"administrator to inspect its audit event. {exc}",
                code="revocation_outcome_unknown",
                status=exc.status,
                details=exc.details,
            ) from exc
        except (ValidationError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "The session revocation response is invalid and the request may have committed; "
                "discard this credential and ask an administrator to inspect its audit event",
                code="revocation_outcome_unknown",
                status=502,
            ) from exc
        finally:
            if self.http.headers.get("Authorization") == credential:
                self.http.headers.pop("Authorization", None)

    @staticmethod
    def new_workspace_departure_receipt_key() -> tuple[str, str]:
        """Create a client-held receipt ID and secret; protect both until lookup finishes."""
        return str(uuid4()), "tdr_" + secrets.token_urlsafe(32)

    def recover_workspace_departure_receipt(
        self,
        *,
        receipt_id: str | UUID,
        receipt_secret: str,
        request: DepartureRequest,
        tenant_id: str | UUID,
        role: Role,
    ) -> Record:
        """Read a matching retained receipt without authenticating the revoked session."""
        checked_id = UUID(str(receipt_id))
        validate_receipt_secret(receipt_secret)
        try:
            encoded = self._bounded_report(
                "GET",
                f"/v1/identity/workspace-departure-receipts/{checked_id}",
                MAX_WORKSPACE_DEPARTURE_RESPONSE_BYTES,
                "Workspace departure receipt",
                headers={"Authorization": f"Bearer {receipt_secret}"},
                follow_redirects=False,
            )
            result = self._checked_exact_response(
                encoded, RecoveredDepartureReceipt, "workspace departure receipt"
            )
        except TensorusError as exc:
            raise TensorusError(
                "The workspace departure receipt could not be verified",
                code=(
                    "invalid_response"
                    if exc.code == "invalid_response"
                    else "departure_receipt_unavailable"
                ),
                status=exc.status,
            ) from None
        try:
            self._check_workspace_departure_receipt(
                result.receipt, request, UUID(str(tenant_id)), role
            )
            if result.receipt_id != checked_id:
                raise ValueError("The departure receipt identity is inconsistent")
        except ValueError as exc:
            raise TensorusError(
                "The departure receipt identity is inconsistent", code="invalid_response"
            ) from exc
        return result.model_dump(mode="json")

    @staticmethod
    def _check_workspace_departure_receipt(
        receipt: DepartureReceipt, request: DepartureRequest, tenant_id: UUID, role: Role
    ) -> None:
        if (
            receipt.tenant_id != tenant_id
            or receipt.membership_id != request.membership_id
            or receipt.session_id != request.session_id
            or receipt.request_id != request.request_id
            or receipt.role != role
            or receipt.reason != request.reason
        ):
            raise ValueError("Workspace departure receipt does not match the request")

    def leave_workspace(
        self,
        *,
        request_id: str | UUID,
        reason: str,
        confirmation: Literal["LEAVE"],
        receipt_recovery: tuple[str | UUID, str] | None = None,
    ) -> Record:
        """Remove this workspace membership and discard its session after submission."""
        credential = self.http.headers.get("Authorization")
        identity, capabilities = self._checked_capabilities()
        if credential is None or self.http.headers.get("Authorization") != credential:
            raise Forbidden("The authenticated credential changed during workspace review")
        if identity.scope is not None or identity.workspace_session is None:
            raise Forbidden("A workspace membership session is required")
        if "workspace_departure" not in capabilities:
            raise TensorusError(
                "The server does not advertise workspace departure", code="unsupported_feature"
            )
        request = DepartureRequest(
            request_id=UUID(str(request_id)),
            membership_id=identity.workspace_session.membership_id,
            session_id=identity.key_id,
            reason=reason,
            confirmation=confirmation,
        )
        supports_recovery = "workspace_departure_receipt_v2" in capabilities
        if receipt_recovery is not None and not supports_recovery:
            raise TensorusError(
                "The server does not advertise departure receipt recovery",
                code="unsupported_feature",
            )
        recovery = (
            receipt_recovery or self.new_workspace_departure_receipt_key()
            if supports_recovery
            else None
        )
        if recovery is not None:
            recovery_id = UUID(str(recovery[0]))
            validate_receipt_secret(recovery[1])
            if recovery_id == request.request_id:
                raise ValueError("Departure and receipt IDs must differ")
        try:
            if recovery is None:
                encoded = self._bounded_report(
                    "POST",
                    "/v1/workspace/membership/leave",
                    MAX_WORKSPACE_DEPARTURE_RESPONSE_BYTES,
                    "Workspace departure",
                    headers={"Authorization": credential},
                    json=request.model_dump(mode="json"),
                )
                result = self._checked_exact_response(
                    encoded, DepartureReceipt, "workspace departure", mutation=True
                )
            else:
                encoded = self._bounded_report(
                    "POST",
                    "/v1/workspace/membership/leave-with-receipt",
                    MAX_WORKSPACE_DEPARTURE_RESPONSE_BYTES,
                    "Workspace departure",
                    headers={
                        "Authorization": credential,
                        "X-Tensorus-Departure-Receipt-Key": recovery[1],
                    },
                    json={**request.model_dump(mode="json"), "receipt_id": str(recovery_id)},
                    follow_redirects=False,
                )
                evidence = self._checked_exact_response(
                    encoded, RecoveredDepartureReceipt, "workspace departure", mutation=True
                )
                if evidence.receipt_id != recovery_id:
                    raise ValueError("Workspace departure receipt ID changed")
                result = evidence.receipt
            self._check_workspace_departure_receipt(result, request, identity.id, identity.role)
            if self.http.headers.get("Authorization") != credential:
                raise ValueError("The workspace departure credential changed")
            return result.model_dump(mode="json")
        except (TensorusError, ValidationError, ValueError, TypeError, RecursionError) as exc:
            lookup_error: str | None = None
            if recovery is not None and self.http.headers.get("Authorization") == credential:
                try:
                    recovered = self.recover_workspace_departure_receipt(
                        receipt_id=recovery_id,
                        receipt_secret=recovery[1],
                        request=request,
                        tenant_id=identity.id,
                        role=identity.role,
                    )
                    if self.http.headers.get("Authorization") != credential:
                        raise Forbidden("The workspace departure credential changed")
                    return recovered["receipt"]
                except (
                    TensorusError,
                    ValidationError,
                    ValueError,
                    TypeError,
                    RecursionError,
                ) as error:
                    lookup_error = (
                        f"HTTP {error.status}"
                        if isinstance(error, TensorusError)
                        and type(error.status) is int
                        and 100 <= error.status <= 599
                        else "invalid evidence"
                    )
            lookup_note = (
                f". Receipt lookup did not confirm the result ({lookup_error})"
                if lookup_error
                else ""
            )
            raise TensorusError(
                "Workspace departure may have committed; discard this session and ask an "
                "administrator to inspect the membership and request audit" + lookup_note,
                code="workspace_departure_outcome_unknown",
                status=exc.status if isinstance(exc, TensorusError) else 502,
                details={
                    "request_id": str(request.request_id),
                    "membership_id": str(request.membership_id),
                    "session_id": str(request.session_id),
                },
            ) from (exc if recovery is None else None)
        finally:
            if self.http.headers.get("Authorization") == credential:
                self.http.headers.pop("Authorization", None)

    def tenant_http_admission(self) -> Record:
        identity, capabilities = self._checked_capabilities()
        if identity.role != Role.admin or identity.scope is not None:
            raise Forbidden("Tenant HTTP admission requires an unscoped administrator")
        if "tenant_http_admission" not in capabilities:
            raise TensorusError(
                "The server does not advertise tenant HTTP admission", code="unsupported_feature"
            )
        encoded = self._bounded_report(
            "GET",
            "/v1/tenant/http-admission",
            MAX_TENANT_HTTP_ADMISSION_RESPONSE_BYTES,
            "Tenant HTTP admission",
        )
        try:
            raw = decode_unique_json(encoded)
            status = TenantHttpAdmission.model_validate_json(encoded, strict=True)
            result = status.model_dump(mode="json")
            if not isinstance(raw, dict) or canonical_json(raw) != canonical_json(result):
                raise ValueError("Tenant HTTP admission evidence is not exact")
        except (ValidationError, TensorusError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "The server returned invalid tenant HTTP admission evidence",
                code="invalid_response",
            ) from exc
        if status.tenant_id != identity.id or self._checked_identity() != identity:
            raise Forbidden("The authenticated identity changed during admission inspection")
        return result

    @staticmethod
    def _workspace_lifecycle_response(encoded: bytes) -> WorkspaceLifecycle:
        try:
            raw = decode_unique_json(encoded)
            lifecycle = WorkspaceLifecycle.model_validate_json(encoded, strict=True)
            if not isinstance(raw, dict) or canonical_json(raw) != canonical_json(
                lifecycle.model_dump(mode="json")
            ):
                raise ValueError("Workspace lifecycle evidence is not exact")
            return lifecycle
        except (ValidationError, TensorusError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "The server returned invalid workspace lifecycle evidence; "
                "a submitted lockdown may have committed",
                code="invalid_response",
            ) from exc

    def workspace_lifecycle(self) -> Record:
        identity = self._checked_identity()
        if identity.role != Role.admin or identity.scope is not None:
            raise Forbidden("Workspace lifecycle requires an unscoped administrator")
        encoded = self._bounded_report(
            "GET", "/v1/workspace/lifecycle", 16 * 1024, "Workspace lifecycle"
        )
        lifecycle = self._workspace_lifecycle_response(encoded)
        if lifecycle.tenant_id != identity.id or self._checked_identity() != identity:
            raise Forbidden("The authenticated identity changed during lifecycle inspection")
        return lifecycle.model_dump(mode="json")

    def lock_workspace(
        self,
        *,
        expected_generation: int,
        request_id: str | UUID,
        reason: str,
        confirmation: Literal["LOCK"],
    ) -> Record:
        request = WorkspaceLockRequest(
            expected_generation=expected_generation,
            request_id=UUID(str(request_id)),
            reason=reason,
            confirmation=confirmation,
        )
        identity = self._checked_identity()
        if identity.role != Role.admin or identity.scope is not None:
            raise Forbidden("Workspace lockdown requires an unscoped administrator")
        encoded = self._bounded_report(
            "POST",
            "/v1/workspace/lifecycle/lock",
            16 * 1024,
            "Workspace lockdown",
            json=request.model_dump(mode="json"),
        )
        lifecycle = self._workspace_lifecycle_response(encoded)
        event = lifecycle.last_event
        if (
            lifecycle.tenant_id != identity.id
            or lifecycle.state != "locked"
            or lifecycle.generation != request.expected_generation + 1
            or event is None
            or event.id != request.request_id
            or event.actor_id != str(identity.key_id)
            or event.reason != request.reason
        ):
            raise TensorusError(
                "The lockdown response is inconsistent and the request may have committed; "
                "ask an operator to inspect its request ID",
                code="invalid_response",
            )
        return lifecycle.model_dump(mode="json")

    def lock_workspace_extended(
        self,
        *,
        expected_generation: int,
        request_id: str | UUID,
        reason: str,
        confirmation: Literal["LOCK"],
    ) -> Record:
        """Lock a larger retained credential history with a tenant-scoped catalog update."""
        request = WorkspaceLockRequest(
            expected_generation=expected_generation,
            request_id=UUID(str(request_id)),
            reason=reason,
            confirmation=confirmation,
        )
        credential = self.http.headers.get("Authorization")
        identity, capabilities = self._checked_capabilities()
        if (
            credential is None
            or self.http.headers.get("Authorization") != credential
            or identity.role != Role.admin
            or identity.scope is not None
        ):
            raise Forbidden("Extended lockdown requires an unchanged unscoped administrator")
        if "workspace_lifecycle_extended" not in capabilities:
            raise TensorusError(
                "The server does not advertise extended workspace lockdown",
                code="unsupported_feature",
            )
        try:
            encoded = self._bounded_report(
                "POST",
                WORKSPACE_LOCK_EXTENDED_PATH,
                16 * 1024,
                "Extended workspace lockdown",
                headers={"Authorization": credential},
                json=request.model_dump(mode="json"),
            )
        except TensorusError as exc:
            if exc.status < 500 and exc.code != "invalid_response":
                raise
            raise TensorusError(
                "Extended lockdown may have committed; ask an operator to inspect the request ID",
                code="workspace_lockdown_outcome_unknown",
                status=exc.status if exc.status >= 500 else 502,
            ) from exc
        try:
            result = self._checked_exact_response(
                encoded, WorkspaceLockdownExtended, "extended lockdown", mutation=True
            )
            lifecycle = result.lifecycle
            event = lifecycle.last_event
            if (
                lifecycle.tenant_id != identity.id
                or lifecycle.generation != request.expected_generation + 1
                or event is None
                or event.id != request.request_id
                or event.expected_generation != request.expected_generation
                or event.actor_id != str(identity.key_id)
                or event.reason != request.reason
                or self.http.headers.get("Authorization") != credential
            ):
                raise ValueError("Extended lockdown evidence is inconsistent")
            return result.model_dump(mode="json")
        except (TensorusError, ValidationError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "Extended lockdown could not be verified and may have committed; "
                "ask an operator to inspect the request ID",
                code="workspace_lockdown_outcome_unknown",
                status=502,
            ) from exc

    @staticmethod
    def new_workspace_lockdown_receipt_key() -> tuple[str, str]:
        """Create a client-held receipt ID and secret; protect them until lookup finishes."""
        return str(uuid4()), "tlr_" + secrets.token_urlsafe(32)

    @staticmethod
    def _check_workspace_lockdown_receipt(
        result: WorkspaceLockdownReceipt,
        *,
        request: WorkspaceLockRequest,
        receipt_id: UUID,
        mode: Literal["standard", "extended"],
        tenant_id: UUID,
        actor_key_id: UUID,
    ) -> None:
        lifecycle = result.lifecycle
        event = lifecycle.last_event
        if (
            result.receipt_id != receipt_id
            or result.mode != mode
            or lifecycle.tenant_id != tenant_id
            or lifecycle.state != "locked"
            or lifecycle.generation != request.expected_generation + 1
            or event is None
            or event.id != request.request_id
            or event.expected_generation != request.expected_generation
            or event.actor_id != str(actor_key_id)
            or event.reason != request.reason
        ):
            raise ValueError("Lockdown receipt does not match the reviewed request")

    def recover_workspace_lockdown_receipt(
        self,
        *,
        receipt_id: str | UUID,
        receipt_secret: str,
        request: WorkspaceLockRequest,
        mode: Literal["standard", "extended"],
        tenant_id: str | UUID,
        actor_key_id: str | UUID,
    ) -> Record:
        """Read a matching lockdown receipt without authenticating the revoked credential."""
        checked_id = UUID(str(receipt_id))
        checked_tenant_id = UUID(str(tenant_id))
        checked_actor_id = UUID(str(actor_key_id))
        checked_request = WorkspaceLockRequest.model_validate(request.model_dump(mode="json"))
        validate_lockdown_receipt_secret(receipt_secret)
        if mode not in ("standard", "extended"):
            raise ValueError("A supported lockdown mode is required")
        try:
            encoded = self._bounded_report(
                "GET",
                f"/v1/identity/workspace-lockdown-receipts/{checked_id}",
                MAX_WORKSPACE_LOCKDOWN_RECEIPT_RESPONSE_BYTES,
                "Workspace lockdown receipt",
                headers={"Authorization": f"Bearer {receipt_secret}"},
                follow_redirects=False,
            )
            result = self._checked_exact_response(
                encoded, WorkspaceLockdownReceipt, "workspace lockdown receipt"
            )
        except TensorusError as exc:
            raise TensorusError(
                "The workspace lockdown receipt could not be verified",
                code=(
                    "invalid_response"
                    if exc.code == "invalid_response"
                    else "lockdown_receipt_unavailable"
                ),
                status=exc.status,
            ) from None
        try:
            self._check_workspace_lockdown_receipt(
                result,
                request=checked_request,
                receipt_id=checked_id,
                mode=mode,
                tenant_id=checked_tenant_id,
                actor_key_id=checked_actor_id,
            )
        except ValueError as exc:
            raise TensorusError(
                "The workspace lockdown receipt identity is inconsistent",
                code="invalid_response",
            ) from exc
        return result.model_dump(mode="json")

    def lock_workspace_with_receipt(
        self,
        *,
        expected_generation: int,
        request_id: str | UUID,
        reason: str,
        confirmation: Literal["LOCK"],
        mode: Literal["standard", "extended"],
        receipt_recovery: tuple[str | UUID, str] | None = None,
    ) -> Record:
        """Lock once and inspect the same receipt if the response is uncertain."""
        request = WorkspaceLockRequest(
            expected_generation=expected_generation,
            request_id=UUID(str(request_id)),
            reason=reason,
            confirmation=confirmation,
        )
        if mode not in ("standard", "extended"):
            raise ValueError("A supported lockdown mode is required")
        credential = self.http.headers.get("Authorization")
        identity, capabilities = self._checked_capabilities()
        if (
            credential is None
            or self.http.headers.get("Authorization") != credential
            or identity.role != Role.admin
            or identity.scope is not None
        ):
            raise Forbidden("Lockdown requires an unchanged unscoped administrator")
        if "workspace_lockdown_receipt_v1" not in capabilities:
            raise TensorusError(
                "The server does not advertise lockdown receipt recovery",
                code="unsupported_feature",
            )
        recovery = receipt_recovery or self.new_workspace_lockdown_receipt_key()
        receipt_id = UUID(str(recovery[0]))
        validate_lockdown_receipt_secret(recovery[1])
        if not receipt_id.int or not request.request_id.int or receipt_id == request.request_id:
            raise ValueError("Distinct non-nil lockdown and receipt IDs are required")
        try:
            encoded = self._bounded_report(
                "POST",
                WORKSPACE_LOCK_RECEIPT_PATH,
                MAX_WORKSPACE_LOCKDOWN_RECEIPT_RESPONSE_BYTES,
                "Workspace lockdown",
                headers={
                    "Authorization": credential,
                    "X-Tensorus-Lockdown-Receipt-Key": recovery[1],
                },
                json={
                    **request.model_dump(mode="json"),
                    "receipt_id": str(receipt_id),
                    "mode": mode,
                },
                follow_redirects=False,
            )
            result = self._checked_exact_response(
                encoded, WorkspaceLockdownReceipt, "workspace lockdown", mutation=True
            )
            self._check_workspace_lockdown_receipt(
                result,
                request=request,
                receipt_id=receipt_id,
                mode=mode,
                tenant_id=identity.id,
                actor_key_id=identity.key_id,
            )
            if self.http.headers.get("Authorization") != credential:
                raise ValueError("The lockdown credential changed")
            return result.model_dump(mode="json")
        except (TensorusError, ValidationError, ValueError, TypeError, RecursionError) as exc:
            lookup_note = "the authenticated credential changed"
            if self.http.headers.get("Authorization") == credential:
                try:
                    recovered = self.recover_workspace_lockdown_receipt(
                        receipt_id=receipt_id,
                        receipt_secret=recovery[1],
                        request=request,
                        mode=mode,
                        tenant_id=identity.id,
                        actor_key_id=identity.key_id,
                    )
                    if self.http.headers.get("Authorization") != credential:
                        raise Forbidden("The lockdown credential changed")
                    return recovered
                except (
                    TensorusError,
                    ValidationError,
                    ValueError,
                    TypeError,
                    RecursionError,
                ) as lookup_error:
                    lookup_note = (
                        f"HTTP {lookup_error.status}"
                        if isinstance(lookup_error, TensorusError)
                        and type(lookup_error.status) is int
                        and 100 <= lookup_error.status <= 599
                        else "invalid evidence"
                    )
            raise TensorusError(
                f"Workspace lockdown may have committed; receipt lookup did not confirm it "
                f"({lookup_note}). "
                "A missing receipt does not prove that the lockdown failed; "
                "ask an operator to inspect the request",
                code="workspace_lockdown_outcome_unknown",
                status=exc.status if isinstance(exc, TensorusError) else 502,
                details={
                    "request_id": str(request.request_id),
                    "receipt_id": str(receipt_id),
                    "tenant_id": str(identity.id),
                    "actor_key_id": str(identity.key_id),
                    "mode": mode,
                },
            ) from None
        finally:
            if self.http.headers.get("Authorization") == credential:
                self.http.headers.pop("Authorization", None)

    def invite_workspace_member(self, subject: str, *, role: Role = Role.reader) -> Record:
        request = WorkspaceMemberInvite(subject=subject, role=role)
        return self._request(
            "POST", "/v1/workspace/members", json=request.model_dump(mode="json")
        ).json()

    def _workspace_member_invitation_context(
        self,
    ) -> tuple[Identity, str, MemberInvitationCapabilities]:
        credential = self.http.headers.get("Authorization")
        if credential is None:
            raise Forbidden("An authenticated administrator is required")
        identity = self._checked_identity_for_credential(credential)
        if identity.role != Role.admin or identity.scope is not None:
            raise Forbidden("Workspace invitations require an unscoped administrator")
        capabilities = self.capabilities(_credential=credential)
        marker = capabilities.get("workspace_member_invitation")
        if marker is None:
            raise TensorusError(
                "The server does not advertise reviewed workspace invitations",
                code="unsupported_feature",
            )
        invitation = MemberInvitationCapabilities.model_validate(marker, strict=True)
        if (
            self.http.headers.get("Authorization") != credential
            or self._checked_identity_for_credential(credential) != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise Forbidden("The authenticated administrator changed during invitation review")
        return identity, credential, invitation

    def workspace_member_invitation_policy(self) -> Record:
        """Review the configured provider policy before inviting a workspace identity."""
        _, _, marker = self._workspace_member_invitation_context()
        return marker.model_dump(mode="json")

    def invite_workspace_member_reviewed(
        self,
        subject: str,
        *,
        request_id: str | UUID,
        expected_issuer: str,
        expected_policy_hash: str,
        role: Role = Role.reader,
    ) -> Record:
        """Invite a new member with actor-bound, provider-pinned exact retries."""
        request = MemberInvitation(
            request_id=UUID(str(request_id)),
            subject=subject,
            role=role,
            expected_issuer=expected_issuer,
            expected_policy_hash=expected_policy_hash,
            confirmation="INVITE",
        )
        identity, credential, marker = self._workspace_member_invitation_context()
        if (
            request.expected_issuer != marker.configured_issuer
            or request.expected_policy_hash != marker.identity_policy_hash
        ):
            raise Conflict("The identity provider policy changed; review the invitation again")
        try:
            encoded = self._bounded_report(
                "POST",
                marker.path,
                MAX_WORKSPACE_MEMBER_INVITATION_RESPONSE_BYTES,
                "Workspace invitation",
                headers={"Authorization": credential},
                json=request.model_dump(mode="json"),
            )
        except TensorusError as exc:
            if exc.status < 500 and exc.code not in (
                "invalid_response",
                "workspace_member_invitation_outcome_unknown",
            ):
                raise
            raise TensorusError(
                "The invitation may have committed; retry the exact request with the same "
                "active administrator and provider policy",
                code="workspace_member_invitation_outcome_unknown",
                status=502,
                details={"request_id": str(request.request_id)},
            ) from exc
        try:
            receipt = self._checked_exact_response(
                encoded, MemberInvitationReceipt, "workspace invitation", mutation=True
            )
            member = receipt.member
            if (
                receipt.request_id != request.request_id
                or receipt.invited_role != request.role
                or receipt.invited_policy_hash != request.expected_policy_hash
                or member.tenant_id != identity.id
                or member.subject != request.subject
                or member.issuer != request.expected_issuer
                or member.invited_by != str(identity.key_id)
                or self.http.headers.get("Authorization") != credential
                or self._checked_identity_for_credential(credential) != identity
                or self.http.headers.get("Authorization") != credential
            ):
                raise ValueError("Workspace invitation receipt does not match the reviewed request")
            return receipt.model_dump(mode="json")
        except (TensorusError, ValueError, TypeError) as exc:
            raise TensorusError(
                "The invitation may have committed; retry the exact request with the same "
                "active administrator and provider policy",
                code="workspace_member_invitation_outcome_unknown",
                status=502,
                details={"request_id": str(request.request_id)},
            ) from exc

    def update_workspace_member(
        self,
        member_id: str,
        *,
        expected_generation: int,
        role: Role,
        status: Literal["invited", "active", "suspended", "removed"],
    ) -> Record:
        request = WorkspaceMemberUpdate(
            expected_generation=expected_generation, role=role, status=status
        )
        return self._request(
            "PUT",
            f"/v1/workspace/members/{quote(member_id, safe='')}",
            json=request.model_dump(mode="json"),
        ).json()

    def transition_workspace_member_reviewed(
        self,
        member: Record,
        *,
        request_id: str | UUID,
        role: Role,
        status: Literal["invited", "active", "suspended", "removed"],
    ) -> Record:
        """Change a reviewed membership with actor-bound, non-reapplying exact retries."""
        try:
            reviewed = ServiceActorMember.model_validate(member)
            if canonical_json(member) != canonical_json(reviewed.model_dump(mode="json")):
                raise ValueError("Membership evidence is not exact")
        except (ValidationError, TensorusError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError("Refresh the workspace membership before changing it") from exc
        request = MemberTransition(
            request_id=UUID(str(request_id)),
            member_id=reviewed.id,
            expected_issuer=reviewed.issuer,
            expected_subject=reviewed.subject,
            expected_generation=reviewed.generation,
            expected_role=reviewed.role,
            expected_status=reviewed.status,
            role=role,
            status=status,
            confirmation="CHANGE",
        )
        credential = self.http.headers.get("Authorization")
        if credential is None:
            raise Forbidden("An authenticated administrator is required")
        identity = self._checked_identity_for_credential(credential)
        if (
            identity.role != Role.admin
            or identity.scope is not None
            or identity.id != reviewed.tenant_id
        ):
            raise Forbidden("Workspace transitions require the reviewed workspace administrator")
        capabilities = self.capabilities(_credential=credential)
        marker = capabilities.get("workspace_member_transition")
        if marker is None:
            raise TensorusError(
                "The server does not advertise reviewed workspace transitions",
                code="unsupported_feature",
            )
        transition = MemberTransitionCapabilities.model_validate(marker, strict=True)
        if (
            self.http.headers.get("Authorization") != credential
            or self._checked_identity_for_credential(credential) != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise Forbidden("The authenticated administrator changed during transition review")
        try:
            encoded = self._bounded_report(
                "POST",
                transition.path,
                MAX_WORKSPACE_MEMBER_TRANSITION_RESPONSE_BYTES,
                "Workspace transition",
                headers={"Authorization": credential},
                json=request.model_dump(mode="json"),
            )
        except TensorusError as exc:
            if exc.status < 500 and exc.code not in (
                "invalid_response",
                "workspace_member_transition_outcome_unknown",
            ):
                raise
            raise TensorusError(
                "The membership change may have committed; retry the exact request with the "
                "same active administrator or inspect it with another administrator",
                code="workspace_member_transition_outcome_unknown",
                status=502,
                details={"request_id": str(request.request_id)},
            ) from exc
        try:
            receipt = self._checked_exact_response(
                encoded, MemberTransitionReceipt, "workspace transition", mutation=True
            )
            current = receipt.member
            if (
                receipt.request_id != request.request_id
                or receipt.previous_role != request.expected_role
                or receipt.previous_status != request.expected_status
                or receipt.requested_role != request.role
                or receipt.requested_status != request.status
                or receipt.committed_generation != request.expected_generation + 1
                or current.id != reviewed.id
                or current.tenant_id != reviewed.tenant_id
                or current.user_id != reviewed.user_id
                or current.issuer != reviewed.issuer
                or current.subject != reviewed.subject
                or current.invited_by != reviewed.invited_by
                or current.created_at != reviewed.created_at
                or self.http.headers.get("Authorization") != credential
                or self._checked_identity_for_credential(credential) != identity
                or self.http.headers.get("Authorization") != credential
            ):
                raise ValueError("Workspace transition receipt contradicts the reviewed request")
            return receipt.model_dump(mode="json")
        except (TensorusError, ValueError, TypeError) as exc:
            raise TensorusError(
                "The membership change may have committed; retry the exact request with the "
                "same active administrator or inspect it with another administrator",
                code="workspace_member_transition_outcome_unknown",
                status=502,
                details={"request_id": str(request.request_id)},
            ) from exc

    def create_key(
        self,
        name: str,
        *,
        role: Role = Role.reader,
        expires_in_days: int | None = None,
    ) -> Record:
        request = KeyCreate(name=name, role=role, expires_in_days=expires_in_days)
        return self._request("POST", "/v1/keys", json=request.model_dump(mode="json")).json()

    def revoke_key(self, key_id: str) -> None:
        self._request("DELETE", f"/v1/keys/{quote(key_id, safe='')}")

    @staticmethod
    def new_agent_grant_key() -> str:
        """Generate a client-held agent key to retain securely before issuance."""
        return new_token()

    def create_agent_grant(
        self,
        name: str,
        dataset: str,
        *,
        actions: list[AgentAction],
        expires_in_seconds: int,
        require_protected_outputs: bool = False,
        require_full_scope_for_ingest: bool = False,
        max_plan_proposals: int | None = None,
        max_estimated_execution_units: int | None = None,
        max_direct_ingest_bytes: int | None = None,
        client_api_key: str | None = None,
    ) -> Record:
        if client_api_key is not None and (
            not isinstance(client_api_key, str) or not valid_generated_token(client_api_key)
        ):
            raise ValueError("The client-held credential must be a generated Tensorus key")
        request = AgentGrantCreate(
            name=name,
            dataset=dataset,
            actions=actions,
            expires_in_seconds=expires_in_seconds,
            require_protected_outputs=require_protected_outputs,
            require_full_scope_for_ingest=require_full_scope_for_ingest,
            max_plan_proposals=max_plan_proposals,
            max_estimated_execution_units=max_estimated_execution_units,
            max_direct_ingest_bytes=max_direct_ingest_bytes,
            client_api_key=SecretStr(client_api_key) if client_api_key is not None else None,
        )
        if client_api_key is not None:
            return self._retryable_agent_grant("/v1/agent/grants", request, client_api_key)
        if max_direct_ingest_bytes is not None:
            self._require_ingest_budget_capability()
        response = self._request("POST", "/v1/agent/grants", json=request.model_dump(mode="json"))
        return self._issued_agent_grant(response, max_direct_ingest_bytes)

    def create_multi_dataset_agent_grant(
        self,
        name: str,
        datasets: list[AgentDatasetGrant],
        *,
        expires_in_seconds: int,
        require_protected_outputs: bool = False,
        require_full_scope_for_ingest: bool = False,
        max_plan_proposals: int | None = None,
        max_estimated_execution_units: int | None = None,
        max_direct_ingest_bytes: int | None = None,
        client_api_key: str | None = None,
    ) -> Record:
        if client_api_key is not None and (
            not isinstance(client_api_key, str) or not valid_generated_token(client_api_key)
        ):
            raise ValueError("The client-held credential must be a generated Tensorus key")
        request = MultiDatasetAgentGrantCreate(
            name=name,
            datasets=datasets,
            expires_in_seconds=expires_in_seconds,
            require_protected_outputs=require_protected_outputs,
            require_full_scope_for_ingest=require_full_scope_for_ingest,
            max_plan_proposals=max_plan_proposals,
            max_estimated_execution_units=max_estimated_execution_units,
            max_direct_ingest_bytes=max_direct_ingest_bytes,
            client_api_key=SecretStr(client_api_key) if client_api_key is not None else None,
        )
        if client_api_key is not None:
            return self._retryable_agent_grant(
                "/v1/agent/multi-dataset-grants", request, client_api_key
            )
        if max_direct_ingest_bytes is not None:
            self._require_ingest_budget_capability()
        response = self._request(
            "POST", "/v1/agent/multi-dataset-grants", json=request.model_dump(mode="json")
        )
        return self._issued_agent_grant(response, max_direct_ingest_bytes)

    def create_service_actor_grant(
        self,
        membership_id: str | UUID,
        name: str,
        dataset: str,
        *,
        expected_actor_id: str | UUID,
        expected_generation: int,
        expected_policy_hash: str,
        client_api_key: str,
        actions: list[AgentAction],
        expires_in_seconds: int,
        require_protected_outputs: bool = False,
        require_full_scope_for_ingest: bool = False,
        max_plan_proposals: int | None = None,
        max_estimated_execution_units: int | None = None,
        max_direct_ingest_bytes: int | None = None,
    ) -> Record:
        """Issue a client-held scoped key to a reviewed, active service membership."""
        review = RecipientReview(
            membership_id=UUID(str(membership_id)),
            expected_actor_id=UUID(str(expected_actor_id)),
            expected_generation=expected_generation,
            expected_policy_hash=expected_policy_hash,
        )
        request = RecipientAgentGrantCreate(
            name=name,
            dataset=dataset,
            actions=actions,
            expires_in_seconds=expires_in_seconds,
            require_protected_outputs=require_protected_outputs,
            require_full_scope_for_ingest=require_full_scope_for_ingest,
            max_plan_proposals=max_plan_proposals,
            max_estimated_execution_units=max_estimated_execution_units,
            max_direct_ingest_bytes=max_direct_ingest_bytes,
            client_api_key=SecretStr(client_api_key),
            expected_actor_id=review.expected_actor_id,
            expected_generation=review.expected_generation,
            expected_policy_hash=review.expected_policy_hash,
            confirmation="GRANT",
        )
        return self._retryable_agent_grant(
            f"/v1/workspace/service-actors/{review.membership_id}/grants",
            request,
            client_api_key,
            recipient=review,
        )

    def create_multi_dataset_service_actor_grant(
        self,
        membership_id: str | UUID,
        name: str,
        datasets: list[AgentDatasetGrant],
        *,
        expected_actor_id: str | UUID,
        expected_generation: int,
        expected_policy_hash: str,
        client_api_key: str,
        expires_in_seconds: int,
        require_protected_outputs: bool = False,
        require_full_scope_for_ingest: bool = False,
        max_plan_proposals: int | None = None,
        max_estimated_execution_units: int | None = None,
        max_direct_ingest_bytes: int | None = None,
    ) -> Record:
        """Issue a client-held multi-dataset key to a reviewed service membership."""
        review = RecipientReview(
            membership_id=UUID(str(membership_id)),
            expected_actor_id=UUID(str(expected_actor_id)),
            expected_generation=expected_generation,
            expected_policy_hash=expected_policy_hash,
        )
        request = RecipientMultiDatasetAgentGrantCreate(
            name=name,
            datasets=datasets,
            expires_in_seconds=expires_in_seconds,
            require_protected_outputs=require_protected_outputs,
            require_full_scope_for_ingest=require_full_scope_for_ingest,
            max_plan_proposals=max_plan_proposals,
            max_estimated_execution_units=max_estimated_execution_units,
            max_direct_ingest_bytes=max_direct_ingest_bytes,
            client_api_key=SecretStr(client_api_key),
            expected_actor_id=review.expected_actor_id,
            expected_generation=review.expected_generation,
            expected_policy_hash=review.expected_policy_hash,
            confirmation="GRANT",
        )
        return self._retryable_agent_grant(
            f"/v1/workspace/service-actors/{review.membership_id}/multi-dataset-grants",
            request,
            client_api_key,
            recipient=review,
        )

    def _retryable_agent_grant(
        self,
        path: str,
        request: AgentGrantCreate | MultiDatasetAgentGrantCreate,
        token: str,
        *,
        recipient: RecipientReview | None = None,
    ) -> Record:
        credential = self.http.headers.get("Authorization")
        identity, capabilities = self._checked_capabilities()
        if (
            credential is None
            or self.http.headers.get("Authorization") != credential
            or identity.role != Role.admin
            or identity.scope is not None
        ):
            raise Forbidden("Agent issuance requires an unchanged unscoped administrator")
        if "agent_grant_retry" not in capabilities:
            raise TensorusError(
                "The server does not advertise retry-safe agent issuance",
                code="unsupported_feature",
            )
        if recipient is not None and "service_actor_recipient_grants" not in capabilities:
            raise TensorusError(
                "The server does not advertise service actor recipient grants",
                code="unsupported_feature",
            )
        if (
            request.max_direct_ingest_bytes is not None
            and "agent_ingest_budget" not in capabilities
        ):
            raise TensorusError(
                "The server does not advertise agent ingestion allowances",
                code="unsupported_feature",
            )
        payload = request.model_dump(mode="json", exclude={"client_api_key"})
        payload["client_api_key"] = token
        try:
            encoded = self._bounded_report(
                "POST",
                path,
                MAX_AGENT_GRANT_RETRY_RESPONSE_BYTES,
                "Agent grant issuance",
                headers={"Authorization": credential},
                json=payload,
            )
        except TensorusError as exc:
            if exc.status < 500 and exc.code != "invalid_response":
                raise
            raise TensorusError(
                "Agent issuance may have committed; retain the client key and retry "
                "the identical request with the same issuer",
                code="agent_grant_outcome_unknown",
                status=exc.status if exc.status >= 500 else 502,
            ) from exc
        try:
            raw = decode_unique_json(encoded)
            issued = KeyIssued.model_validate_json(encoded, strict=True)
            scope = issued.scope
            if (
                not isinstance(raw, dict)
                or canonical_json(raw) != canonical_json(issued.model_dump(mode="json"))
                or issued.id != UUID(grant_id(str(identity.id), str(identity.key_id), token))
                or issued.tenant_id != identity.id
                or issued.name != request.name
                or issued.role != Role.reader
                or issued.api_key != token
                or issued.revoked_at is not None
                or issued.expires_at is None
                or scope is None
                or scope.name != request.name
                or scope.expires_at != issued.expires_at
                or scope.require_protected_outputs != request.require_protected_outputs
                or scope.require_full_scope_for_ingest != request.require_full_scope_for_ingest
            ):
                raise ValueError("Agent grant response is inconsistent")
            require_grant_lifetime(issued.created_at, issued.expires_at, request.expires_in_seconds)
            if isinstance(request, AgentGrantCreate):
                if (
                    not isinstance(scope, AgentScope)
                    or scope.dataset != request.dataset
                    or scope.actions != request.actions
                ):
                    raise ValueError("Agent grant scope differs from the request")
            elif not isinstance(scope, MultiDatasetAgentScope) or [
                (item.dataset, item.actions) for item in scope.datasets
            ] != [
                (item.dataset, item.actions)
                for item in sorted(request.datasets, key=lambda x: x.dataset)
            ]:
                raise ValueError("Agent grant scope differs from the request")
            for budget, field, limit in (
                (issued.proposal_budget, "limit", request.max_plan_proposals),
                (issued.execution_budget, "limit_units", request.max_estimated_execution_units),
                (issued.ingest_budget, "limit_bytes", request.max_direct_ingest_bytes),
            ):
                if (budget is None) != (limit is None) or (
                    budget is not None and getattr(budget, field) != limit
                ):
                    raise ValueError("Agent grant allowance differs from the request")
            verification = self._bounded_report(
                "GET",
                "/v1/me",
                MAX_TENANT_STAGING_RESPONSE_BYTES,
                "Agent identity",
                headers={"Authorization": f"Bearer {token}"},
            )
            checked = decode_unique_json(verification)
            agent = Identity.model_validate_json(verification, strict=True)
            if (
                not isinstance(checked, dict)
                or "scope" not in checked
                or agent.id != identity.id
                or agent.key_id != issued.id
                or agent.role != Role.reader
                or agent.scope != scope
                or agent.workspace_session is not None
                or self.http.headers.get("Authorization") != credential
                or self._checked_identity() != identity
            ):
                raise ValueError("Agent identity is inconsistent")
            if recipient is not None:
                recipient_encoded = self._bounded_report(
                    "GET",
                    "/v1/identity/service-actor-recipient",
                    4_096,
                    "Service actor recipient",
                    headers={"Authorization": f"Bearer {token}"},
                )
                status = self._checked_exact_response(
                    recipient_encoded, RecipientStatus, "service actor recipient"
                )
                if (
                    status.tenant_id != identity.id
                    or status.key_id != issued.id
                    or status.recipient is None
                    or status.recipient.actor_id != recipient.expected_actor_id
                    or status.recipient.membership_id != recipient.membership_id
                    or status.recipient.membership_generation != recipient.expected_generation
                    or status.recipient.identity_policy_hash != recipient.expected_policy_hash
                    or self.http.headers.get("Authorization") != credential
                    or self._checked_identity() != identity
                ):
                    raise ValueError("Agent recipient differs from the reviewed membership")
        except (ValidationError, TensorusError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "Agent issuance could not be verified and may have committed; retain the "
                "client key and retry the identical request with the same issuer",
                code="agent_grant_outcome_unknown",
                status=502,
            ) from exc
        return issued.model_dump(mode="json")

    def _require_ingest_budget_capability(self) -> None:
        _, capabilities = self._checked_capabilities()
        if "agent_ingest_budget" not in capabilities:
            raise TensorusError(
                "The server does not advertise agent ingestion allowances",
                code="unsupported_feature",
            )

    @staticmethod
    def _issued_agent_grant(response: httpx.Response, limit: int | None) -> Record:
        if limit is None:
            return response.json()
        try:
            issued = decode_unique_json(response.content)
            if not isinstance(issued, dict):
                raise ValueError("The issued grant is not an object")
        except (ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "A grant may have been issued despite the invalid response; "
                "inspect workspace keys and revoke it if necessary",
                code="invalid_response",
            ) from exc
        Client._check_issued_ingest_budget(issued, limit)
        return issued

    @staticmethod
    def _check_issued_ingest_budget(issued: Record, limit: int) -> None:
        try:
            raw = issued["ingest_budget"]
            budget = AgentIngestBudget.model_validate(raw, strict=True)
            if (
                canonical_json(raw) != canonical_json(budget.model_dump(mode="json"))
                or budget.limit_bytes != limit
                or budget.committed_bytes != 0
                or budget.committed_versions != 0
            ):
                raise ValueError("The issued ingestion allowance is inconsistent")
        except (
            KeyError,
            ValidationError,
            TensorusError,
            ValueError,
            TypeError,
            RecursionError,
        ) as exc:
            raise TensorusError(
                "The issued grant has invalid ingestion allowance evidence; "
                "inspect workspace keys and revoke it if necessary",
                code="invalid_response",
            ) from exc

    def datasets(self, *, limit: int = 100, offset: int = 0, text: str | None = None) -> Record:
        params: Record = {"limit": limit, "offset": offset}
        if text is not None:
            params["text"] = text
        return self._request("GET", "/v1/datasets", params=params).json()

    def dataset_options(
        self, *, limit: int = 25, offset: int = 0, text: str | None = None
    ) -> Record:
        """Read one compact authorized page; choices do not grant access to later actions."""
        if (
            type(limit) is not int
            or not 1 <= limit <= MAX_DATASET_OPTIONS_PAGE_SIZE
            or type(offset) is not int
            or not 0 <= offset <= MAX_DATASET_OPTIONS_OFFSET
            or (text is not None and (not isinstance(text, str) or len(text) > 256))
        ):
            raise ValueError("Dataset option pagination or search is invalid")
        credential = self.http.headers.get("Authorization")
        if credential is None:
            raise Forbidden("Dataset options require an authenticated credential")
        identity, capabilities = self._checked_capabilities(_credential=credential)
        if "dataset_options" not in capabilities:
            raise TensorusError(
                "The server does not advertise compact dataset options",
                code="unsupported_feature",
            )
        if self.http.headers.get("Authorization") != credential:
            raise Forbidden("The credential changed during dataset option discovery")
        params: Record = {"limit": limit, "offset": offset}
        if text is not None:
            params["text"] = text
        encoded = self._bounded_report(
            "GET",
            "/v1/dataset-options",
            MAX_DATASET_OPTIONS_RESPONSE_BYTES,
            "Dataset option page",
            headers={"Authorization": credential},
            params=params,
        )
        page = self._checked_exact_response(encoded, DatasetOptionsPage, "dataset option page")
        if page.tenant_id != identity.id or page.limit != limit or page.offset != offset:
            raise TensorusError("The server returned a different dataset option page")
        if identity.scope is not None:
            scopes = (
                [identity.scope]
                if isinstance(identity.scope, AgentScope)
                else identity.scope.datasets
            )
            allowed = {item.dataset_id: item.dataset for item in scopes if "read" in item.actions}
            if any(allowed.get(str(item.id)) != item.name for item in page.items):
                raise TensorusError("The server returned dataset options outside the agent scope")
        final_identity = self._checked_identity_for_credential(credential)
        if self.http.headers.get("Authorization") != credential or final_identity != identity:
            raise Forbidden("The authenticated identity changed during dataset option discovery")
        return page.model_dump(mode="json")

    def create_dataset(self, name: str, **kwargs: Any) -> Record:
        request = DatasetCreate(name=name, **kwargs)
        return self._request("POST", "/v1/datasets", json=request.model_dump(mode="json")).json()

    def dataset(self, name: str) -> Record:
        return self._request("GET", f"/v1/datasets/{quote(name, safe='')}").json()

    def archive_dataset(self, name: str) -> None:
        self._request("DELETE", f"/v1/datasets/{quote(name, safe='')}")

    @staticmethod
    def _npy_headers(
        metadata: dict[str, Any] | None, idempotency_key: str | None
    ) -> dict[str, str]:
        encoded = canonical_json(metadata or {})
        if len(encoded.encode()) > 8192:
            raise TensorusError("Binary SDK uploads support at most 8 KiB of metadata")
        headers = {"Content-Type": "application/x-npy", "X-Tensor-Metadata": encoded}
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        return headers

    def _require_conditional_npy_receipt(self, *, _credential: str | None = None) -> None:
        contracts = (
            self.capabilities()
            if _credential is None
            else self.capabilities(_credential=_credential)
        ).get("write_contracts")
        if not isinstance(contracts, dict) or "conditional_npy_receipt" not in contracts:
            raise TensorusError(
                "Destination server does not support conditional ingestion receipts",
                code="unsupported_capability",
                status=409,
            )

    def _require_conditional_npy_historical_receipt(
        self, *, _credential: str | None = None
    ) -> None:
        contracts = (
            self.capabilities()
            if _credential is None
            else self.capabilities(_credential=_credential)
        ).get("write_contracts")
        if not isinstance(contracts, dict) or "conditional_npy_historical_receipt" not in contracts:
            raise TensorusError(
                "Destination server does not support historical conditional ingestion receipts",
                code="unsupported_capability",
                status=409,
            )

    def _read_conditional_npy_receipt(
        self,
        dataset: str,
        name: str,
        request: ConditionalNpyReceiptRequest,
        idempotency_key: str,
        identity: Identity,
        *,
        _credential: str | None = None,
    ) -> Record:
        return self._read_conditional_npy_evidence(
            dataset,
            name,
            request,
            idempotency_key,
            identity,
            historical=False,
            _credential=_credential,
        )

    def _read_conditional_npy_historical_receipt(
        self,
        dataset: str,
        name: str,
        request: ConditionalNpyReceiptRequest,
        idempotency_key: str,
        identity: Identity,
        *,
        allow_tombstoned: bool = False,
        _credential: str | None = None,
    ) -> Record:
        return self._read_conditional_npy_evidence(
            dataset,
            name,
            request,
            idempotency_key,
            identity,
            historical=True,
            allow_tombstoned=allow_tombstoned,
            _credential=_credential,
        )

    def _read_conditional_npy_evidence(
        self,
        dataset: str,
        name: str,
        request: ConditionalNpyReceiptRequest,
        idempotency_key: str,
        identity: Identity,
        *,
        historical: bool,
        allow_tombstoned: bool = False,
        _credential: str | None = None,
    ) -> Record:
        path = (
            f"/v1/datasets/{quote(dataset, safe='')}/tensors/{quote(name, safe='')}"
            "/conditional-npy/receipt"
        )
        if historical:
            path += "/history"
        headers = {"Idempotency-Key": idempotency_key}
        if _credential is not None:
            headers["Authorization"] = _credential
        data = self._bounded_report(
            "POST",
            path,
            128 * 1024,
            "Conditional ingestion receipt",
            headers=headers,
            json=request.model_dump(mode="json"),
        )
        try:
            raw = decode_unique_json(data)
            receipt = (
                ConditionalNpyHistoricalReceipt.model_validate_json(data, strict=True)
                if historical
                else ConditionalNpyReceipt.model_validate_json(data, strict=True)
            )
            version = receipt.version
            if (
                canonical_json(raw) != canonical_json(receipt.model_dump(mode="json"))
                or receipt.tenant_id != identity.id
                or receipt.actor_key_id != identity.key_id
                or receipt.condition != request.condition
                or receipt.request_hash
                != ingestion_request_hash(
                    dataset, name, request.checksum, request.metadata, request.expected_version
                )
                or (
                    version is not None
                    and (
                        version.tenant_id != identity.id
                        or version.dataset_id != request.condition.dataset_id
                        or version.dataset != dataset
                        or version.name != name
                        or version.version != request.expected_version + 1
                        or version.head_version < version.version
                        or (version.deleted_at is not None and not allow_tombstoned)
                        or version.checksum != request.checksum
                        or canonical_json(version.metadata) != canonical_json(request.metadata)
                        or version.parents
                        or version.operation is not None
                        or (
                            not historical
                            and (
                                version.write_contract.generation
                                if version.write_contract is not None
                                else 0
                            )
                            != request.condition.write_contract_generation
                        )
                    )
                )
            ):
                raise ValueError("Conditional ingestion receipt contradicts the request")
        except (UnicodeDecodeError, ValueError, RecursionError, TensorusError) as exc:
            raise TensorusError(
                "Destination returned invalid conditional ingestion receipt evidence",
                code="invalid_response",
            ) from exc
        return receipt.model_dump(mode="json")

    def conditional_npy_historical_receipt(
        self,
        dataset: str,
        name: str,
        request: ConditionalNpyReceiptRequest,
        *,
        idempotency_key: str,
    ) -> Record:
        """Observe retained policy and current policy without authorizing another write."""
        from tensorus.product_bundle import _capture_write_credential, _checked_write_identity

        try:
            checked = ConditionalNpyReceiptRequest.model_validate_json(
                request.model_dump_json(), strict=True
            )
        except ValidationError as exc:
            raise TensorusError("Invalid conditional ingestion receipt request") from exc
        credential = _capture_write_credential(self)
        self._require_conditional_npy_historical_receipt(_credential=credential)
        identity = _checked_write_identity(self, credential)
        if identity.scope is not None or identity.role not in (Role.writer, Role.admin):
            raise Forbidden("Conditional ingestion receipts require an active unscoped writer")
        receipt = self._read_conditional_npy_historical_receipt(
            dataset, name, checked, idempotency_key, identity, _credential=credential
        )
        _checked_write_identity(self, credential, identity)
        return receipt

    def conditional_npy_receipt(
        self,
        dataset: str,
        name: str,
        request: ConditionalNpyReceiptRequest,
        *,
        idempotency_key: str,
    ) -> Record:
        """Inspect a matching retry record without writing or asserting payload integrity."""
        from tensorus.product_bundle import _capture_write_credential, _checked_write_identity

        try:
            checked = ConditionalNpyReceiptRequest.model_validate_json(
                request.model_dump_json(), strict=True
            )
        except ValidationError as exc:
            raise TensorusError("Invalid conditional ingestion receipt request") from exc
        credential = _capture_write_credential(self)
        self._require_conditional_npy_receipt(_credential=credential)
        identity = _checked_write_identity(self, credential)
        if identity.scope is not None or identity.role not in (Role.writer, Role.admin):
            raise Forbidden("Conditional ingestion receipts require an active unscoped writer")
        receipt = self._read_conditional_npy_receipt(
            dataset, name, checked, idempotency_key, identity, _credential=credential
        )
        _checked_write_identity(self, credential, identity)
        return receipt

    def put(
        self,
        dataset: str,
        name: str,
        data: object,
        *,
        dtype: str | None = None,
        metadata: dict[str, Any] | None = None,
        expected_version: int | None = None,
        idempotency_key: str | None = None,
        write_condition: NpyWriteCondition | None = None,
    ) -> Record:
        array = validated_array(data, self.max_tensor_bytes, dtype)
        if write_condition is not None:
            if (
                type(expected_version) is not int
                or expected_version < 0
                or not isinstance(idempotency_key, str)
                or not idempotency_key
            ):
                raise TensorusError(
                    "Conditional NPY ingestion requires an expected version and idempotency key"
                )
            from tensorus.product_bundle import (
                ProductBundleDestination,
                _capture_write_credential,
                _checked_write_identity,
                _destination_state,
                _require_write_credential,
                _write_payload,
            )

            try:
                condition = NpyWriteCondition.model_validate_json(write_condition.model_dump_json())
                target = ProductBundleDestination(dataset=dataset, name=name)
            except ValidationError as exc:
                raise TensorusError("Invalid conditional NPY ingestion request") from exc
            credential = _capture_write_credential(self)
            identity = _checked_write_identity(self, credential)
            _, state = _destination_state(self, dataset, identity.id, credential=credential)
            active = state["active_write_contract"]
            if (
                state["dataset_id"] != str(condition.dataset_id)
                or active["generation"] != condition.write_contract_generation
            ):
                raise Conflict("The dataset or write policy differs from the requested condition")
            binding = (
                WriteContractBinding.model_validate(
                    {
                        field: active[field]
                        for field in ("generation", "revision", "definition_hash")
                    }
                )
                if active["generation"]
                else None
            )
            _require_write_credential(self, credential)
            raw = _write_payload(
                self,
                target,
                metadata or {},
                array,
                expected_version,
                idempotency_key,
                condition,
                credential=credential,
            )
            _require_write_credential(self, credential)
            try:
                result = TensorVersion.model_validate_json(canonical_json(raw), strict=True)
                if (
                    canonical_json(raw) != canonical_json(result.model_dump(mode="json"))
                    or result.tenant_id != identity.id
                    or result.dataset_id != condition.dataset_id
                    or result.dataset != dataset
                    or result.name != name
                    or result.version != expected_version + 1
                    or result.head_version < result.version
                    or result.dtype.value != array.dtype.name
                    or result.shape != list(array.shape)
                    or result.nbytes != array.nbytes
                    or result.checksum != array_checksum(array)
                    or canonical_json(result.metadata) != canonical_json(metadata or {})
                    or result.deleted_at is not None
                    or result.operation is not None
                    or result.parents
                    or result.write_contract != binding
                ):
                    raise ValueError("Conditional ingestion evidence differs from the request")
            except (ValidationError, ValueError, RecursionError, TensorusError) as exc:
                raise TensorusError(
                    "The server returned invalid conditional ingestion evidence",
                    code="invalid_response",
                ) from exc
            if self._checked_identity() != identity:
                raise TensorusError("The authenticated identity changed during ingestion")
            if _destination_state(self, dataset, identity.id, credential=credential)[1] != state:
                raise Conflict("The dataset or write policy changed during conditional ingestion")
            _require_write_credential(self, credential)
            return result.model_dump(mode="json")
        headers = self._npy_headers(metadata, idempotency_key)
        params = {} if expected_version is None else {"expected_version": expected_version}
        path = f"/v1/datasets/{quote(dataset, safe='')}/tensors/{quote(name, safe='')}/npy"
        return self._request(
            "PUT", path, content=dump_npy(array), headers=headers, params=params
        ).json()

    def put_npy_file(
        self,
        dataset: str,
        name: str,
        source: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
        expected_version: int | None = None,
        idempotency_key: str | None = None,
    ) -> Record:
        """Upload a stable local NPY file without constructing an in-memory request body."""
        headers = self._npy_headers(metadata, idempotency_key)
        params = {} if expected_version is None else {"expected_version": expected_version}
        path = f"/v1/datasets/{quote(dataset, safe='')}/tensors/{quote(name, safe='')}/npy"
        with Path(source).open("rb") as stream:
            if os.fstat(stream.fileno()).st_size > self.max_tensor_bytes + NPY_MAX_OVERHEAD_BYTES:
                raise LimitExceeded("NPY file exceeds the SDK tensor-byte limit")
            return self._request("PUT", path, content=stream, headers=headers, params=params).json()

    def create_upload(
        self,
        dataset: str,
        name: str,
        *,
        size: int,
        checksum: str,
        expected_version: int,
        idempotency_key: str,
        metadata: dict[str, Any] | None = None,
        expires_in_seconds: int = 3600,
        _credential: str | None = None,
    ) -> Record:
        request = UploadCreate(
            dataset=dataset,
            name=name,
            size=size,
            checksum=checksum,
            expected_version=expected_version,
            metadata=metadata or {},
            expires_in_seconds=expires_in_seconds,
        )
        return self._request(
            "POST",
            "/v1/uploads",
            json=request.model_dump(mode="json"),
            headers={
                "Idempotency-Key": idempotency_key,
                **({"Authorization": _credential} if _credential is not None else {}),
            },
        ).json()

    def upload_status(self, upload_id: str) -> Record:
        response = self._request("GET", f"/v1/uploads/{quote(upload_id, safe='')}")
        try:
            record = decode_unique_json(response.content)
            checked = Upload.model_validate(record)
            if checked.id != UUID(upload_id):
                raise ValueError("Upload identity does not match the request")
            return record
        except (ValidationError, TensorusError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "The server returned invalid upload status evidence", code="invalid_response"
            ) from exc

    def upload_attempt_observations(
        self, upload_id: str, *, limit: int = 50, after: str | None = None
    ) -> Record:
        """Inspect this credential's recorded upload attempts, not usage or charges."""
        try:
            requested_id = UUID(upload_id)
            cursor = UUID(after) if after is not None else None
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("A valid upload ID and optional cursor are required") from exc
        if type(limit) is not int or not 1 <= limit <= MAX_UPLOAD_ATTEMPT_PAGE_SIZE:
            raise ValueError("Upload observation page size is outside the supported range")
        credential = self.http.headers.get("Authorization")
        if credential is None:
            raise Forbidden("Upload observations require an authenticated credential")
        identity, capabilities = self._checked_capabilities(_credential=credential)
        if self.http.headers.get("Authorization") != credential:
            raise Forbidden("The credential changed during upload observation")
        marker = capabilities.get("upload_attempt_observations")
        if (
            not isinstance(marker, dict)
            or marker.get("format") != UPLOAD_ATTEMPT_OBSERVATIONS_FORMAT
        ):
            raise TensorusError(
                "The server does not advertise upload attempt observations",
                code="unsupported_feature",
            )
        encoded = self._bounded_report(
            "GET",
            f"/v1/uploads/{quote(str(requested_id), safe='')}/attempt-observations",
            MAX_UPLOAD_ATTEMPT_RESPONSE_BYTES,
            "Upload observation",
            headers={"Authorization": credential},
            params={"limit": limit, **({"after": str(cursor)} if cursor is not None else {})},
        )
        result = self._checked_exact_response(
            encoded, UploadAttemptObservations, "upload observation"
        )
        if (
            result.tenant_id != identity.id
            or result.upload_id != requested_id
            or result.created_by != identity.key_id
            or result.limit != limit
            or result.after != cursor
        ):
            raise TensorusError(
                "The upload observation does not match the requested identity or page",
                code="invalid_response",
            )
        self._require_upload_observation_scope(identity, result.dataset_id, result.dataset)
        if (
            self.http.headers.get("Authorization") != credential
            or self._checked_identity_for_credential(credential) != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise Forbidden("The credential changed during upload observation")
        return result.model_dump(mode="json")

    @staticmethod
    def _require_upload_observation_scope(
        identity: Identity, dataset_id: UUID, dataset: str
    ) -> None:
        if identity.scope is not None:
            scopes = (
                [identity.scope]
                if isinstance(identity.scope, AgentScope)
                else identity.scope.datasets
            )
            if not any(
                item.dataset_id == str(dataset_id)
                and item.dataset == dataset
                and "ingest" in item.actions
                for item in scopes
            ):
                raise TensorusError(
                    "The upload observation is outside the credential's ingest scope",
                    code="invalid_response",
                )
        elif identity.role not in {Role.writer, Role.admin}:
            raise Forbidden("Upload observations require writer access")

    def upload_preclaim_rejection(self, upload_id: str, attempt_id: str) -> Record:
        """Inspect the first retained keyed pre-claim rejection, not a complete failure history."""
        try:
            requested_id = UUID(upload_id)
            requested_attempt = parse_attempt_id(attempt_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("A valid upload ID and canonical attempt ID are required") from exc
        credential = self.http.headers.get("Authorization")
        if credential is None:
            raise Forbidden("Upload observations require an authenticated credential")
        identity, capabilities = self._checked_capabilities(_credential=credential)
        if self.http.headers.get("Authorization") != credential:
            raise Forbidden("The credential changed during upload observation")
        marker = capabilities.get("upload_preclaim_rejections")
        if not isinstance(marker, dict) or marker.get("format") != UPLOAD_PRECLAIM_FORMAT:
            raise TensorusError(
                "The server does not advertise upload preclaim rejection observations",
                code="unsupported_feature",
            )
        encoded = self._bounded_report(
            "GET",
            f"/v1/uploads/{quote(str(requested_id), safe='')}"
            f"/preclaim-rejections/{quote(str(requested_attempt), safe='')}",
            MAX_UPLOAD_PRECLAIM_RESPONSE_BYTES,
            "Upload preclaim observation",
            headers={"Authorization": credential},
        )
        result = self._checked_exact_response(
            encoded, PreclaimRejection, "upload preclaim observation"
        )
        if (
            result.tenant_id != identity.id
            or result.upload_id != requested_id
            or result.created_by != identity.key_id
            or result.attempt_id != requested_attempt
        ):
            raise TensorusError(
                "The upload preclaim observation does not match the requested identity",
                code="invalid_response",
            )
        self._require_upload_observation_scope(identity, result.dataset_id, result.dataset)
        if (
            self.http.headers.get("Authorization") != credential
            or self._checked_identity_for_credential(credential) != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise Forbidden("The credential changed during upload observation")
        return result.model_dump(mode="json")

    def put_upload_chunk(
        self, upload_id: str, offset: int, data: bytes, *, _credential: str | None = None
    ) -> Record:
        return self._request(
            "PUT",
            f"/v1/uploads/{quote(upload_id, safe='')}/chunks/{offset}",
            content=data,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Upload-Checksum": hashlib.sha256(data).hexdigest(),
                **({"Authorization": _credential} if _credential is not None else {}),
            },
        ).json()

    def _check_upload_credential(self, credential: str) -> None:
        if self.http.headers.get("Authorization") != credential:
            raise Forbidden("The credential changed; the upload may have staged or committed")

    def _require_upload_preclaim_capability(self, credential: str) -> None:
        _, capabilities = self._checked_capabilities(_credential=credential)
        marker = capabilities.get("upload_preclaim_rejections")
        if not isinstance(marker, dict) or marker.get("format") != UPLOAD_PRECLAIM_FORMAT:
            raise TensorusError(
                "The server does not advertise upload preclaim rejection observations",
                code="unsupported_feature",
            )
        if self.http.headers.get("Authorization") != credential:
            raise Forbidden("The credential changed before upload finalization")

    def finalize_upload(
        self,
        upload_id: str,
        *,
        attempt_id: str | None = None,
        _credential: str | None = None,
    ) -> Record:
        """Finalize an upload; an optional attempt ID identifies selected pre-claim evidence."""
        path = f"/v1/uploads/{quote(upload_id, safe='')}/finalize"
        if attempt_id is None:
            if _credential is None:
                return self._request("POST", path).json()
            self._check_upload_credential(_credential)
            result = self._request("POST", path, headers={"Authorization": _credential}).json()
            self._check_upload_credential(_credential)
            return result
        try:
            requested_attempt = parse_attempt_id(attempt_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("A non-nil lowercase canonical attempt ID is required") from exc
        credential = self.http.headers.get("Authorization") if _credential is None else _credential
        if credential is None:
            raise Forbidden("Upload observations require an authenticated credential")
        self._check_upload_credential(credential)
        self._require_upload_preclaim_capability(credential)
        result = self._request(
            "POST",
            path,
            headers={"Authorization": credential, UPLOAD_PRECLAIM_HEADER: str(requested_attempt)},
        ).json()
        self._check_upload_credential(credential)
        return result

    def cancel_upload(self, upload_id: str) -> Record:
        return self._request("DELETE", f"/v1/uploads/{quote(upload_id, safe='')}").json()

    def resume_npy_upload(
        self,
        dataset: str,
        name: str,
        source: str | Path,
        *,
        expected_version: int,
        idempotency_key: str,
        metadata: dict[str, Any] | None = None,
        expires_in_seconds: int = 3600,
        finalization_attempt_id: str | None = None,
    ) -> Record:
        """Resume an explicitly identified upload of an unchanged local NPY file.

        Larger C-order files also require the server's streaming-upload opt-in.
        """
        if finalization_attempt_id is not None:
            try:
                parse_attempt_id(finalization_attempt_id)
            except (TypeError, ValueError, AttributeError) as exc:
                raise ValueError("A non-nil lowercase canonical attempt ID is required") from exc
        credential = self.http.headers.get("Authorization")
        if credential is None:
            raise Forbidden("Resumable upload requires an authenticated credential")
        if finalization_attempt_id is not None:
            self._require_upload_preclaim_capability(credential)

        def finish(upload_id: str) -> Record:
            return self.finalize_upload(
                upload_id, attempt_id=finalization_attempt_id, _credential=credential
            )

        with Path(source).open("rb") as stream:
            file_info = os.fstat(stream.fileno())
            if not stat.S_ISREG(file_info.st_mode):
                raise TensorusError("Resumable upload requires a regular NPY file")
            size = file_info.st_size
            if size > self.max_resumable_upload_bytes + NPY_MAX_OVERHEAD_BYTES:
                raise LimitExceeded("NPY file exceeds the SDK resumable-upload byte limit")
            read_npy_header(stream, size, self.max_resumable_upload_bytes)
            stream.seek(0)
            digest = hashlib.sha256()
            while chunk := stream.read(1024**2):
                digest.update(chunk)
            self._check_upload_credential(credential)
            session = self.create_upload(
                dataset,
                name,
                size=size,
                checksum=digest.hexdigest(),
                expected_version=expected_version,
                idempotency_key=idempotency_key,
                metadata=metadata,
                expires_in_seconds=expires_in_seconds,
                _credential=credential,
            )
            self._check_upload_credential(credential)
            upload_id = session["id"]
            if session["status"] == "completed":
                return finish(upload_id)
            if session["status"] != "open":
                raise Conflict("The upload was cancelled or expired")
            offset, chunk_size = session["offset"], session["chunk_size"]
            if (
                not isinstance(offset, int)
                or isinstance(offset, bool)
                or not isinstance(chunk_size, int)
                or isinstance(chunk_size, bool)
                or not 1 <= chunk_size <= 8 * 1024**2
                or not 0 <= offset <= size
                or (offset != size and offset % chunk_size != 0)
            ):
                raise TensorusError("The server returned an invalid upload checkpoint")
            while offset < size:
                stream.seek(offset)
                chunk = stream.read(min(chunk_size, size - offset))
                if len(chunk) != min(chunk_size, size - offset):
                    raise TensorusError("The NPY source changed or was truncated during upload")
                self._check_upload_credential(credential)
                checkpoint = self.put_upload_chunk(
                    upload_id,
                    offset,
                    chunk,
                    _credential=credential,
                )
                self._check_upload_credential(credential)
                next_offset = checkpoint["offset"]
                if (
                    not isinstance(next_offset, int)
                    or isinstance(next_offset, bool)
                    or not offset + len(chunk) <= next_offset <= size
                    or (next_offset != size and next_offset % chunk_size != 0)
                ):
                    raise TensorusError("The server returned an invalid upload checkpoint")
                offset = next_offset
            return finish(upload_id)

    def head(self, dataset: str, name: str) -> Record:
        path = f"/v1/datasets/{quote(dataset, safe='')}/tensors/{quote(name, safe='')}"
        return self._request("GET", path).json()

    def versions(self, dataset: str, name: str, *, limit: int = 50, offset: int = 0) -> Record:
        path = f"/v1/datasets/{quote(dataset, safe='')}/tensors/{quote(name, safe='')}/versions"
        return self._request("GET", path, params={"limit": limit, "offset": offset}).json()

    def delete_tensor(self, dataset: str, name: str, *, expected_version: int) -> None:
        path = f"/v1/datasets/{quote(dataset, safe='')}/tensors/{quote(name, safe='')}"
        self._request("DELETE", path, params={"expected_version": expected_version})

    def version(self, version_id: str) -> Record:
        """Read a version descriptor with raw statistics; no validity assertion is applied."""
        return self._request("GET", f"/v1/versions/{quote(version_id, safe='')}").json()

    def ingest_usage_observation(self, version_id: str, expected_checksum: str) -> Record:
        """Inspect recorded committed logical bytes, never attempted work or charges."""
        try:
            requested_id = UUID(version_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("A valid tensor version ID is required") from exc
        if not isinstance(expected_checksum, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_checksum
        ):
            raise ValueError("A valid expected tensor checksum is required")
        identity, capabilities = self._checked_capabilities()
        marker = capabilities.get("ingest_usage_observations")
        if not isinstance(marker, dict) or marker.get("format") != INGEST_USAGE_FORMAT:
            raise TensorusError(
                "The server does not advertise ingestion observations", code="unsupported_feature"
            )
        encoded = self._bounded_report(
            "GET",
            f"/v1/versions/{quote(str(requested_id), safe='')}/ingest-observation",
            MAX_INGEST_USAGE_RESPONSE_BYTES,
            "Ingestion observation",
            params={"expected_checksum": expected_checksum},
        )
        result = self._checked_exact_response(
            encoded, IngestUsageObservation, "ingestion observation"
        )
        if (
            result.tenant_id != identity.id
            or result.version_id != requested_id
            or result.checksum != expected_checksum
            or (
                identity.scope is not None
                and result.commit is not None
                and result.commit.committed_by != str(identity.key_id)
            )
        ):
            raise TensorusError("The ingestion observation is for a different identity or version")
        if self._checked_identity() != identity:
            raise Forbidden("The workspace identity changed during ingestion observation")
        return result.model_dump(mode="json")

    def evaluate_tensor(self, version_id: str, criteria: TensorEvaluationRequest) -> Record:
        """Check selected criteria, including optional pairwise validity, not goal truth."""
        return self._request(
            "POST",
            f"/v1/versions/{quote(version_id, safe='')}/evaluate",
            json=criteria.model_dump(mode="json"),
        ).json()

    def prepare_reference_evaluation(self, request: ReferenceEvaluationRequest) -> Record:
        """Retain a pairwise comparison; retry the identical request with the same writer."""
        response = self._request(
            "POST", "/v1/reference-evaluations", json=request.model_dump(mode="json")
        ).json()
        return ReferenceEvaluation.model_validate(response).model_dump(mode="json")

    def reference_evaluations(self, *, limit: int = 50, offset: int = 0) -> Record:
        response = self._request(
            "GET", "/v1/reference-evaluations", params={"limit": limit, "offset": offset}
        ).json()
        return Page[ReferenceEvaluationSummary].model_validate(response).model_dump(mode="json")

    def reference_evaluation(self, evaluation_id: str) -> Record:
        response = self._request(
            "GET", f"/v1/reference-evaluations/{quote(evaluation_id, safe='')}"
        ).json()
        return ReferenceEvaluation.model_validate(response).model_dump(mode="json")

    def approve_reference_evaluation(self, evaluation_id: str, report_hash: str) -> Record:
        """Reproduce and approve the report with a different active unscoped writer."""
        response = self._request(
            "POST",
            f"/v1/reference-evaluations/{quote(evaluation_id, safe='')}/approve",
            json={"report_hash": report_hash},
        ).json()
        return ReferenceEvaluation.model_validate(response).model_dump(mode="json")

    def withdraw_reference_evaluation(
        self, evaluation_id: str, request: ReferenceDispositionRequest
    ) -> Record:
        response = self._request(
            "POST",
            f"/v1/reference-evaluations/{quote(evaluation_id, safe='')}/withdraw",
            json=request.model_dump(mode="json"),
        ).json()
        return ReferenceEvaluationArchive.model_validate(response).model_dump(mode="json")

    def review_reference_retirement(
        self, evaluation_id: str, request: ReferenceDispositionRequest
    ) -> Record:
        response = self._request(
            "POST",
            f"/v1/reference-evaluations/{quote(evaluation_id, safe='')}/retirement-review",
            json=request.model_dump(mode="json"),
        ).json()
        return ReferenceRetirementReviewResult.model_validate(response).model_dump(mode="json")

    def retire_reference_evaluation(
        self, evaluation_id: str, request: ReferenceRetirementRequest
    ) -> Record:
        response = self._request(
            "POST",
            f"/v1/reference-evaluations/{quote(evaluation_id, safe='')}/retire",
            json=request.model_dump(mode="json"),
        ).json()
        return ReferenceEvaluationArchive.model_validate(response).model_dump(mode="json")

    def reference_evaluation_archive(self, evaluation_id: str) -> Record:
        response = self._request(
            "GET", f"/v1/reference-evaluation-archives/{quote(evaluation_id, safe='')}"
        ).json()
        return ReferenceEvaluationArchive.model_validate(response).model_dump(mode="json")

    def reference_evaluation_archives(self, *, limit: int = 10, offset: int = 0) -> Record:
        response = self._request(
            "GET", "/v1/reference-evaluation-archives", params={"limit": limit, "offset": offset}
        ).json()
        return Page[ReferenceArchiveSummary].model_validate(response).model_dump(mode="json")

    def validity_statistics(self, version_id: str, request: ValidityStatisticsRequest) -> Record:
        """Compare raw values with positions selected by one pinned validity assertion."""
        return self._request(
            "POST",
            f"/v1/versions/{quote(version_id, safe='')}/validity-statistics",
            json=request.model_dump(mode="json"),
        ).json()

    def validity_bindings(self, data_version_id: str) -> Record:
        path = f"/v1/versions/{quote(data_version_id, safe='')}/validity-bindings"
        return self._request("GET", path).json()

    def validity_binding(self, binding_id: str) -> Record:
        return self._request("GET", f"/v1/validity-bindings/{quote(binding_id, safe='')}").json()

    def create_validity_binding(
        self, request: ValidityBindingCreate, *, _credential: str | None = None
    ) -> Record:
        return self._request(
            "POST",
            "/v1/validity-bindings",
            headers={"Authorization": _credential} if _credential is not None else None,
            json=request.model_dump(mode="json"),
        ).json()

    def explain_validity_transform(self, binding_id: str, request: ValidityTransform) -> Record:
        if request.parameters.get("nearest") is not None:
            return self._review_nearest_transform(binding_id, request)
        return self._request(
            "POST",
            f"/v1/validity-bindings/{quote(binding_id, safe='')}/explain-transform",
            json=request.model_dump(mode="json"),
        ).json()

    def _review_nearest_transform(self, binding_id: str, request: ValidityTransform) -> Record:
        from tensorus.product_bundle import MAX_BUNDLE_VERSION_RESPONSE_BYTES, _read_model

        try:
            policy = NearestIntegerPolicy.model_validate(request.parameters["nearest"], strict=True)
            if (
                request.operation != "reindex_coordinates"
                or request.anchor_version_id is None
                or request.access_policy != "source_datasets_v1"
            ):
                raise ValueError("Nearest mapping requires an anchor and source protection")
        except (KeyError, ValueError, TypeError) as exc:
            raise TensorusError("Invalid nearest-coordinate review request") from exc

        identity, capabilities = self._checked_capabilities()
        semantics = capabilities.get("tensor_semantics")
        reference = semantics.get("reference_semantics") if isinstance(semantics, dict) else None
        if not isinstance(reference, dict) or "paired_validity_nearest" not in reference:
            raise TensorusError(
                "Server does not advertise nearest-coordinate paired review",
                code="unsupported_capability",
            )
        data = self._bounded_report(
            "POST",
            f"/v1/validity-bindings/{quote(binding_id, safe='')}/explain-transform",
            2 * MAX_PLAN_OUTCOME_BYTES,
            "Nearest-coordinate review",
            limit_exceeded=True,
            json=request.model_dump(mode="json"),
        )
        try:
            raw = decode_unique_json(data)
            review = ValidityTransformExplanation.model_validate_json(data, strict=True)
            result = review.model_dump(mode="json")
            if canonical_json(raw) != canonical_json(result):
                raise ValueError("Review has a noncanonical representation")
            binding = review.binding
            explanation = review.explanation
            definition = explanation.definition
            if (
                binding.id != UUID(binding_id)
                or binding.tenant_id != identity.id
                or definition.dataset != request.dataset
                or definition.semantic_policy != "v2"
                or definition.access_policy != request.access_policy
                or not _paired_definition(
                    definition,
                    str(binding.data_version_id),
                    str(binding.mask_version_id),
                    str(binding.id),
                    binding.descriptor_hash,
                )
                or len(definition.steps) != 2
                or definition.steps[0].output != request.data_output
                or definition.steps[1].output != request.mask_output
                or definition.steps[0].inputs[1:] != [str(request.anchor_version_id)]
                or explanation.observation_scope_v4_format != "tensorus-step-observation-scope-v4"
                or explanation.plan_hash != canonical_hash(definition.model_dump(mode="json"))
            ):
                raise ValueError("Review does not match the requested binding and plan")
        except (KeyError, IndexError, TypeError, ValueError, RecursionError, TensorusError) as exc:
            raise TensorusError(
                "The server returned inconsistent nearest-coordinate review evidence",
                code="invalid_response",
            ) from exc
        stored = _read_model(
            self,
            f"/v1/validity-bindings/{quote(binding_id, safe='')}",
            ValidityBinding,
            "Source binding",
            MAX_BUNDLE_VERSION_RESPONSE_BYTES,
        )
        versions = [
            _read_model(
                self,
                f"/v1/versions/{quote(str(version_id), safe='')}",
                TensorVersion,
                "Source version",
                MAX_BUNDLE_VERSION_RESPONSE_BYTES,
            ).model_dump(mode="json")
            for version_id in (
                binding.data_version_id,
                binding.mask_version_id,
                request.anchor_version_id,
            )
        ]
        try:
            if stored != binding:
                raise ValueError("Source binding evidence changed during review")
            source, mask, anchor = versions
            ids = [
                str(binding.data_version_id),
                str(binding.mask_version_id),
                str(request.anchor_version_id),
            ]
            if any(
                version["id"] != expected
                or version["tenant_id"] != str(identity.id)
                or descriptor_nbytes(version) != version["nbytes"]
                for version, expected in zip(versions, ids, strict=True)
            ):
                raise ValueError("Source version identity or size is inconsistent")
            if (
                binding.data_checksum != source["checksum"]
                or binding.mask_checksum != mask["checksum"]
                or binding.data_dtype != source["dtype"]
                or binding.shape != source["shape"]
                or binding.descriptor_hash
                != _descriptor_hash(
                    source,
                    mask,
                    binding.origin.model_dump(mode="json"),
                    str(binding.id),
                    binding.created_at,
                    binding.created_by,
                )
            ):
                raise ValueError("Source binding contradicts its versions")
            _check_pair_descriptors(source, mask)
            declared = [
                read_semantics(version["metadata"], tuple(version["shape"]), version["operation"])
                for version in versions
            ]
            if any(item is None for item in declared):
                raise ValueError("Source or anchor semantics are missing")
            source_semantics, mask_semantics, anchor_semantics = declared
            assert source_semantics is not None
            assert mask_semantics is not None
            assert anchor_semantics is not None
            permutation, mappings = referenced_nearest_coordinate_alignment(
                source_semantics,
                anchor_semantics,
                tuple(source["shape"]),
                tuple(anchor["shape"]),
                policy.max_distance,
            )
            if permutation != list(range(len(permutation))):
                raise ValueError("Nearest anchor axes are not in source order")
            expected_axes = (
                request.parameters["axes"]
                if "axes" in request.parameters
                else [
                    item.model_dump(mode="json") if item is not None else None for item in mappings
                ]
            )
            values, bits = definition.steps
            expected_values = ReindexCoordinatesParameters.model_validate(
                {
                    **request.parameters,
                    "axes": expected_axes,
                },
                strict=True,
            ).model_dump(mode="json")
            expected_mask = {"axes": expected_axes, "nearest": policy.model_dump(mode="json")}
            if (
                canonical_json(values.parameters) != canonical_json(expected_values)
                or canonical_json(bits.parameters) != canonical_json(expected_mask)
                or len(explanation.steps) != 2
            ):
                raise ValueError("Reviewed mapping or policy differs from the request")
            for step, described, first, first_semantics in (
                (values, explanation.steps[0], source, source_semantics),
                (bits, explanation.steps[1], mask, mask_semantics),
            ):
                output = described.output
                output_shape = tuple(anchor["shape"])
                expected_size = math.prod(output_shape) * np.dtype(first["dtype"]).itemsize
                predicted = TensorShape(output_shape, first["dtype"])
                derived = derive_semantics(
                    step,
                    [
                        TensorShape(tuple(first["shape"]), first["dtype"]),
                        TensorShape(tuple(anchor["shape"]), anchor["dtype"]),
                    ],
                    [first_semantics, anchor_semantics],
                    predicted,
                    "v2",
                )
                required = (
                    {
                        (item.dataset, str(item.dataset_id))
                        for item in output.access.required_datasets
                    }
                    if output is not None and output.access is not None
                    else set()
                )
                direct = {
                    (version["dataset"], version["dataset_id"]) for version in (first, anchor)
                }
                inherited = {
                    (item["dataset"], item["dataset_id"])
                    for version in (first, anchor)
                    for item in (version.get("access") or {}).get("required_datasets", [])
                }
                if (
                    output is None
                    or output.shape != list(output_shape)
                    or output.dtype != first["dtype"]
                    or output.nbytes != expected_size
                    or output.semantics != derived
                    or output.access is None
                    or output.access.policy != "source_datasets_v1"
                    or len(required) != len(output.access.required_datasets)
                    or not (direct | inherited).issubset(required)
                    or request.dataset not in {name for name, _ in required}
                    or described.output_name != step.output
                    or len(described.inputs) != 2
                    or any(
                        item.reference != version["id"]
                        or item.name != version["name"]
                        or item.dataset != version["dataset"]
                        or item.shape != version["shape"]
                        or item.dtype != version["dtype"]
                        or item.nbytes != version["nbytes"]
                        or item.semantics != semantic
                        or (
                            item.access.model_dump(mode="json") if item.access is not None else None
                        )
                        != version.get("access")
                        for item, version, semantic in zip(
                            described.inputs,
                            (first, anchor),
                            (first_semantics, anchor_semantics),
                            strict=True,
                        )
                    )
                ):
                    raise ValueError("Predicted nearest output or inputs are inconsistent")
            if (
                explanation.output_bytes
                != sum(step.output.nbytes for step in explanation.steps if step.output is not None)
                or explanation.estimated_compute
                != sum(step.estimated_compute for step in explanation.steps)
                or any(step.estimated_compute < 0 for step in explanation.steps)
                or explanation.output_bytes < 0
                or explanation.estimated_compute < 0
                or explanation.output_bytes > explanation.output_byte_limit
                or explanation.estimated_compute > explanation.compute_limit
            ):
                raise ValueError("Nearest review totals or limits are inconsistent")
        except (KeyError, IndexError, TypeError, ValueError, RecursionError, TensorusError) as exc:
            raise TensorusError(
                "The server returned inconsistent nearest-coordinate review evidence",
                code="invalid_response",
            ) from exc
        if self._checked_identity() != identity:
            raise Forbidden("The authenticated identity changed during nearest-coordinate review")
        return result

    def explain_fitted_preprocessing_fit(
        self, binding_id: str, request: FittedPreprocessingFit
    ) -> Record:
        data = self._bounded_report(
            "POST",
            f"/v1/validity-bindings/{quote(binding_id, safe='')}/explain-fit-preprocessing",
            2 * MAX_PLAN_OUTCOME_BYTES,
            "Fitted preprocessing",
            limit_exceeded=True,
            json=request.model_dump(mode="json"),
        )
        result = self._fitted_response(data, FittedPreprocessingFitExplanation)
        definition = result["explanation"]["definition"]
        if (
            not self._fitted_id_matches(binding_id, result["binding"]["id"])
            or result["binding"]["descriptor_hash"] != request.expected_binding_hash
            or definition["dataset"] != request.dataset
            or definition["access_policy"] != request.access_policy
            or definition["steps"][0]["output"] != request.state_output
            or definition["steps"][0]["parameters"] != request.parameters.model_dump(mode="json")
        ):
            raise TensorusError("The server returned a different fitted fit request")
        return result

    @staticmethod
    def _fitted_response(
        data: bytes, schema: type[Model], *, imported_state: bool = False
    ) -> Record:
        if len(data) > 2 * MAX_PLAN_OUTCOME_BYTES:
            raise LimitExceeded("Fitted preprocessing response exceeds the SDK byte limit")
        try:
            raw = decode_unique_json(data)
            if imported_state:
                state = (
                    raw
                    if schema is ReviewedImportedFittedState
                    else raw.get("state")
                    if isinstance(raw, dict)
                    else None
                )
                if (
                    not isinstance(state, dict)
                    or not set(ReviewedImportedFittedState.model_fields).issubset(state)
                    or state["source_claims_authenticated"] is not False
                    or state["independent_recomputation"] is not False
                    or state["inverse_supported"] is not False
                ):
                    raise ValueError("Imported fitted-state evidence is incomplete")
            return schema.model_validate(raw).model_dump(mode="json")
        except (UnicodeDecodeError, ValueError, RecursionError) as exc:
            raise TensorusError(
                "The server returned invalid fitted preprocessing evidence"
            ) from exc

    @staticmethod
    def _fitted_id_matches(requested: str, returned: str) -> bool:
        try:
            return UUID(requested) == UUID(returned)
        except ValueError:
            return False

    def fitted_preprocessing_state(self, state_version_id: str) -> Record:
        data = self._bounded_report(
            "GET",
            f"/v1/fitted-preprocessing/{quote(state_version_id, safe='')}",
            2 * MAX_PLAN_OUTCOME_BYTES,
            "Fitted preprocessing",
            limit_exceeded=True,
        )
        result = self._fitted_response(data, FittedPreprocessingState)
        if not self._fitted_id_matches(state_version_id, result["state_version_id"]):
            raise TensorusError("The server returned a different fitted state")
        return result

    def _require_imported_fitted_capability(self) -> Identity:
        identity, capabilities = self._checked_capabilities()
        try:
            marker = capabilities["tensor_semantics"]["descriptor_schema"][
                "x-tensorus-observation-validity"
            ]["fitted_preprocessing"]["imported_state_review"]
            selection_schema = marker.get("selection_schema") if isinstance(marker, dict) else None
            state_schema = marker.get("state_schema") if isinstance(marker, dict) else None
            selection_properties = (
                selection_schema.get("properties") if isinstance(selection_schema, dict) else None
            )
            state_properties = (
                state_schema.get("properties") if isinstance(state_schema, dict) else None
            )
            state_format = (
                state_properties.get("format") if isinstance(state_properties, dict) else None
            )
            expected_selection = ImportedFittedStateSelection.model_json_schema()["properties"]
            selection_shape_matches = isinstance(selection_properties, dict) and all(
                isinstance(selection_properties.get(name), dict)
                and all(
                    selection_properties[name].get(key) == value
                    for key, value in schema.items()
                    if key in {"type", "format", "pattern"}
                )
                for name, schema in expected_selection.items()
            )
            state_shape_matches = isinstance(state_properties, dict) and set(
                ReviewedImportedFittedState.model_fields
            ).issubset(state_properties)
            trust_flags_match = isinstance(state_properties, dict) and all(
                isinstance(state_properties.get(name), dict)
                and state_properties[name].get("const") is False
                for name in (
                    "source_claims_authenticated",
                    "independent_recomputation",
                    "inverse_supported",
                )
            )
            if (
                not isinstance(marker, dict)
                or marker.get("format") != "tensorus-reviewed-imported-fitted-state-v1"
                or marker.get("source_claims_authenticated") is not False
                or marker.get("independent_recomputation") is not False
                or marker.get("distinct_plan_approval") is not True
                or marker.get("local_product_release_required") is not True
                or marker.get("all_product_members_required") is not True
                or marker.get("local_bindings") != "asserted_only"
                or marker.get("local_release_selections") != "no_references_or_outcomes"
                or not isinstance(selection_schema, dict)
                or selection_schema.get("type") != "object"
                or not selection_shape_matches
                or set(selection_schema.get("required", []))
                != {
                    "import_id",
                    "source_hash",
                    "state_role",
                    "local_release_id",
                    "local_release_report_hash",
                }
                or not isinstance(state_schema, dict)
                or not state_shape_matches
                or not trust_flags_match
                or not isinstance(state_format, dict)
                or state_format.get("const") != "tensorus-reviewed-imported-fitted-state-v1"
            ):
                raise ValueError("Imported fitted-state capability is inconsistent")
        except (KeyError, TypeError, ValueError) as exc:
            raise TensorusError(
                "Server does not advertise reviewed imported fitted-state use",
                code="unsupported_capability",
            ) from exc
        return identity

    def _verify_imported_fitted_state(
        self,
        state_version_id: str,
        selection: ImportedFittedStateSelection,
        result: Record,
        identity: Identity,
    ) -> None:
        from tensorus.imported_fitted import derive_reviewed_imported_state
        from tensorus.product_bundle import MAX_BUNDLE_VERSION_RESPONSE_BYTES, _read_model

        imported_data = self._bounded_report(
            "GET",
            f"/v1/product-release-imports/{quote(str(selection.import_id), safe='')}",
            2 * MAX_PORTABLE_PRODUCT_BYTES,
            "Product import",
            limit_exceeded=True,
        )
        imported = self._imported_product_response(imported_data)
        release_data = self._bounded_report(
            "GET",
            f"/v1/product-releases/{quote(str(selection.local_release_id), safe='')}",
            2 * MAX_PORTABLE_PRODUCT_BYTES,
            "Product release",
            limit_exceeded=True,
        )
        release = self._product_release_response(release_data)
        product_data = self._bounded_report(
            "GET",
            f"/v1/data-products/{quote(imported.product_name, safe='')}",
            2 * MAX_PRODUCT_BYTES,
            "Data product",
            limit_exceeded=True,
            params={"revision": imported.product_revision},
        )
        product = self._product_response(product_data)
        version = _read_model(
            self,
            f"/v1/versions/{quote(state_version_id, safe='')}",
            TensorVersion,
            "Fitted state version",
            MAX_BUNDLE_VERSION_RESPONSE_BYTES,
        )
        try:
            if (
                imported.id != selection.import_id
                or release.id != selection.local_release_id
                or product.name != imported.product_name
                or product.revision != imported.product_revision
                or version.id != UUID(state_version_id)
                or any(
                    item != identity.id
                    for item in (
                        imported.tenant_id,
                        release.tenant_id,
                        product.tenant_id,
                        version.tenant_id,
                    )
                )
            ):
                raise ValueError("Imported fitted-state dependencies have different identities")
            member = next(
                item for item in product.definition.members if item.role == selection.state_role
            )
            if canonical_json(version_integrity_descriptor(version.model_dump(mode="json"))) != (
                canonical_json(version_integrity_descriptor(member.version.model_dump(mode="json")))
            ):
                raise ValueError("Imported state descriptor does not match the reviewed role")
            expected = derive_reviewed_imported_state(
                imported.model_dump(mode="json"),
                release.model_dump(mode="json"),
                product,
                state_version_id,
                version.checksum,
                selection,
            ).model_dump(mode="json")
            if canonical_json(expected) != canonical_json(result):
                raise ValueError("Imported fitted-state evidence disagrees with its dependencies")
        except (
            ValidationError,
            TensorusError,
            ValueError,
            KeyError,
            TypeError,
            StopIteration,
        ) as exc:
            raise TensorusError(
                "The server returned contradictory imported fitted-state evidence"
            ) from exc
        if self._checked_identity() != identity:
            raise Forbidden("The authenticated identity changed during imported state review")

    def review_imported_fitted_preprocessing_state(
        self, state_version_id: str, selection: ImportedFittedStateSelection
    ) -> Record:
        identity = self._require_imported_fitted_capability()
        data = self._bounded_report(
            "POST",
            f"/v1/fitted-preprocessing/{quote(state_version_id, safe='')}/review-imported",
            2 * MAX_PLAN_OUTCOME_BYTES,
            "Fitted preprocessing",
            limit_exceeded=True,
            json=selection.model_dump(mode="json"),
        )
        result = self._fitted_response(data, ReviewedImportedFittedState, imported_state=True)
        if not self._fitted_id_matches(state_version_id, result["state_version_id"]) or result[
            "selection"
        ] != selection.model_dump(mode="json"):
            raise TensorusError("The server returned a different imported fitted-state selection")
        self._verify_imported_fitted_state(state_version_id, selection, result, identity)
        return result

    def explain_fitted_preprocessing_apply(
        self, binding_id: str, request: FittedPreprocessingApply
    ) -> Record:
        identity = (
            self._require_imported_fitted_capability()
            if request.imported_state is not None
            else None
        )
        data = self._bounded_report(
            "POST",
            f"/v1/validity-bindings/{quote(binding_id, safe='')}/explain-apply-preprocessing",
            2 * MAX_PLAN_OUTCOME_BYTES,
            "Fitted preprocessing",
            limit_exceeded=True,
            json=request.model_dump(mode="json"),
        )
        result = self._fitted_response(
            data,
            FittedPreprocessingApplyExplanation,
            imported_state=request.imported_state is not None,
        )
        definition = result["explanation"]["definition"]
        if (
            not self._fitted_id_matches(binding_id, result["binding"]["id"])
            or result["binding"]["descriptor_hash"] != request.expected_binding_hash
            or result["state"]["state_version_id"] != str(request.state_version_id)
            or result["state"]["state_checksum"] != request.expected_state_checksum
            or definition["dataset"] != request.dataset
            or definition["access_policy"] != request.access_policy
            or definition["steps"][0]["output"] != request.data_output
            or definition["steps"][1]["output"] != request.mask_output
            or (
                request.imported_state is not None
                and result["state"].get("selection")
                != request.imported_state.model_dump(mode="json")
            )
            or (
                request.imported_state is None
                and result["state"].get("format") == "tensorus-reviewed-imported-fitted-state-v1"
            )
        ):
            raise TensorusError("The server returned a different fitted transform request")
        if request.imported_state is not None:
            assert identity is not None
            self._verify_imported_fitted_state(
                str(request.state_version_id), request.imported_state, result["state"], identity
            )
        return result

    def explain_validity_coordinate_window(
        self, binding_id: str, request: ValidityCoordinateWindow
    ) -> Record:
        """Review a source-pinned v2 coordinate window without reading tensor payloads.

        ``chunked_verified`` execution bounds input memory but scans and checksums both
        complete sources. It does not change the separate approval and completion steps.
        """
        return self._request(
            "POST",
            f"/v1/validity-bindings/{quote(binding_id, safe='')}/explain-coordinate-window",
            json=request.model_dump(mode="json"),
        ).json()

    def complete_validity_transform(
        self, binding_id: str, request: ValidityTransformCompletion
    ) -> Record:
        path = f"/v1/validity-bindings/{quote(binding_id, safe='')}/complete-transform"
        body = request.model_dump(mode="json")
        try:
            return self._request("POST", path, json=body).json()
        except TensorusError as exc:
            if exc.code != "connection_error":
                raise
            plan = self.plan(str(request.plan_id))
            if (
                not isinstance(plan, dict)
                or plan.get("plan_hash") != request.expected_plan_hash
                or plan.get("status") != "succeeded"
            ):
                raise TensorusError(
                    f"Completion response was lost for plan {request.plan_id}; "
                    "inspect its exact hash and status before retrying",
                    code="connection_error",
                    status=503,
                ) from exc
            try:
                definition = PlanDefinition.model_validate(plan["definition"])
                anchor_id = str(UUID(binding_id))
                if (
                    plan["id"] != str(request.plan_id)
                    or canonical_hash(plan["definition"]) != request.expected_plan_hash
                ):
                    raise ValueError("Plan definition does not match the requested identity")
                sources = definition.validity_sources
                if sources is None:
                    source = ValidityBinding.model_validate(self.validity_binding(anchor_id))
                    if str(source.id) != anchor_id or not _paired_definition(
                        definition, str(source.data_version_id), str(source.mask_version_id)
                    ):
                        raise ValueError("Legacy plan does not match its source binding")
                    source_hash = source.descriptor_hash
                else:
                    if str(sources[0].binding_id) != anchor_id:
                        raise ValueError("Plan is anchored to a different source binding")
                    source_hash = sources[0].descriptor_hash
                expected_origin: Record = {
                    "kind": "transformed",
                    "binding_id": anchor_id,
                    "binding_hash": source_hash,
                    "plan_id": str(request.plan_id),
                    "plan_hash": request.expected_plan_hash,
                }
                if sources is not None and len(sources) == 2:
                    expected_origin = {
                        "kind": "binary_transformed",
                        "sources": [source.model_dump(mode="json") for source in sources],
                        "plan_id": str(request.plan_id),
                        "plan_hash": request.expected_plan_hash,
                    }
            except (KeyError, IndexError, TypeError, ValueError, TensorusError) as error:
                raise TensorusError(
                    "Completed plan source provenance is invalid",
                    code="storage_error",
                    status=503,
                ) from error
            results = plan.get("results")
            if not isinstance(results, dict):
                raise TensorusError(
                    "The succeeded plan has malformed results", code="storage_error", status=503
                ) from exc
            data_result, mask_result = results.get("values"), results.get("validity")
            values = data_result.get("version_id") if isinstance(data_result, dict) else None
            mask = mask_result.get("version_id") if isinstance(mask_result, dict) else None
            if not isinstance(values, str) or not isinstance(mask, str):
                raise TensorusError(
                    "The succeeded plan does not expose a complete data/mask pair",
                    code="storage_error",
                    status=503,
                ) from exc
            matches = []
            aliases = []
            try:
                listing = ValidityBindingList.model_validate(self.validity_bindings(values))
            except (TypeError, ValueError) as error:
                raise TensorusError(
                    "Completed binding list is malformed", code="storage_error", status=503
                ) from error
            for item in listing.items:
                binding = item.model_dump(mode="json")
                origin = binding["origin"]
                # Legacy unary plans may be completed for different bindings of the same pair.
                legacy_alias = (
                    sources is None
                    and origin.get("kind") == "transformed"
                    and origin.get("binding_id") != anchor_id
                    and origin.get("plan_id") == str(request.plan_id)
                    and binding["mask_version_id"] == mask
                )
                if origin == expected_origin and binding["mask_version_id"] == mask:
                    matches.append(binding)
                elif legacy_alias:
                    try:
                        source_alias = ValidityBinding.model_validate(
                            self.validity_binding(origin["binding_id"])
                        )
                    except (ValueError, TensorusError) as error:
                        raise TensorusError(
                            "Legacy binding source provenance is invalid",
                            code="storage_error",
                            status=503,
                        ) from error
                    if (
                        str(source_alias.id) != origin["binding_id"]
                        or source_alias.tenant_id != item.tenant_id
                        or source_alias.descriptor_hash != origin["binding_hash"]
                        or origin["plan_hash"] != request.expected_plan_hash
                        or not _paired_definition(
                            definition,
                            str(source_alias.data_version_id),
                            str(source_alias.mask_version_id),
                        )
                    ):
                        raise TensorusError(
                            "Legacy binding provenance contradicts the reviewed plan",
                            code="storage_error",
                            status=503,
                        ) from exc
                    aliases.append(binding)
                elif (
                    origin.get("kind") == "asserted"
                    and binding["data_version_id"] == values
                    and binding["mask_version_id"] == mask
                ):
                    aliases.append(binding)
                elif (
                    binding["data_version_id"] == values and binding["mask_version_id"] == mask
                ) or (
                    origin.get("kind") != "asserted"
                    and (
                        origin.get("plan_id") == str(request.plan_id)
                        or origin.get("binding_id") == anchor_id
                        or any(
                            source["binding_id"] == anchor_id
                            for source in origin.get("sources", [])
                        )
                    )
                ):
                    raise TensorusError(
                        "Completed binding provenance contradicts the reviewed plan",
                        code="storage_error",
                        status=503,
                    ) from exc
            if len(matches) > 1:
                raise TensorusError(
                    "Multiple completed bindings match the same plan and pair",
                    code="storage_error",
                    status=503,
                ) from exc

            for alias in aliases:
                try:
                    stored = ValidityBinding.model_validate(self.validity_binding(alias["id"]))
                except (ValueError, TensorusError) as error:
                    raise TensorusError(
                        "Existing binding alias provenance is invalid",
                        code="storage_error",
                        status=503,
                    ) from error
                if stored.model_dump(mode="json") != alias:
                    raise TensorusError(
                        "Existing binding alias contradicts its recorded provenance",
                        code="storage_error",
                        status=503,
                    ) from exc

            def validate_descriptors(bindings: list[Record]) -> None:
                data, mask_descriptor = self.version(values), self.version(mask)
                for binding in bindings:
                    try:
                        if (
                            binding["data_version_id"] != values
                            or binding["mask_version_id"] != mask
                            or data["id"] != values
                            or mask_descriptor["id"] != mask
                            or binding["tenant_id"] != data["tenant_id"]
                            or binding["tenant_id"] != mask_descriptor["tenant_id"]
                            or binding["data_checksum"] != data["checksum"]
                            or binding["mask_checksum"] != mask_descriptor["checksum"]
                            or binding["data_dtype"] != data["dtype"]
                            or mask_descriptor["dtype"] != "bool"
                            or binding["shape"] != data["shape"]
                            or binding["shape"] != mask_descriptor["shape"]
                            or binding["descriptor_hash"]
                            != _descriptor_hash(
                                data,
                                mask_descriptor,
                                binding["origin"],
                                binding["id"],
                                binding["created_at"],
                                binding["created_by"],
                            )
                        ):
                            raise ValueError("Binding does not match its output descriptors")
                    except (KeyError, TypeError, ValueError, TensorusError) as error:
                        raise TensorusError(
                            "Completed binding descriptors are inconsistent",
                            code="storage_error",
                            status=503,
                        ) from error

            if matches or aliases:
                validate_descriptors([*matches, *aliases])
            if matches:
                return matches[0]
            response = self._request("POST", path, json=body)
            try:
                retried = ValidityBinding.model_validate(response.json()).model_dump(mode="json")
            except ValueError as error:
                raise TensorusError(
                    "Retried completion returned a malformed binding",
                    code="storage_error",
                    status=503,
                ) from error
            if retried["origin"] != expected_origin:
                raise TensorusError(
                    "Retried completion provenance contradicts the reviewed plan",
                    code="storage_error",
                    status=503,
                ) from exc
            validate_descriptors([retried])
            return retried

    def get(self, version_id: str) -> Array:
        response = self._request("GET", f"/v1/versions/{quote(version_id, safe='')}/npy")
        return load_npy(response.content, self.max_tensor_bytes)

    def _get_verified_npy(self, descriptor: Record, *, _credential: str | None = None) -> Array:
        nbytes = descriptor_nbytes(descriptor)
        if nbytes > self.max_tensor_bytes:
            raise LimitExceeded("Tensor exceeds the SDK full-read byte limit")
        path = f"/v1/versions/{quote(descriptor['id'], safe='')}/npy"
        data = bytearray()
        limit = nbytes + NPY_MAX_OVERHEAD_BYTES
        headers = {"Accept-Encoding": "identity"}
        if _credential is not None:
            headers["Authorization"] = _credential
        try:
            with self.http.stream("GET", path, headers=headers) as response:
                if response.headers.get("content-encoding", "identity").lower() != "identity":
                    raise TensorusError("Verified NPY reads require identity encoding")
                if not response.is_success:
                    error = bytearray()
                    for chunk in response.iter_bytes(chunk_size=8192):
                        error.extend(chunk)
                        if len(error) > 65_536:
                            break
                    self._check_response(response, bytes(error))
                if response.status_code != 200:
                    raise TensorusError("Expected a complete NPY response")
                if response.headers.get("content-type", "").split(";")[0] != "application/x-npy":
                    raise TensorusError("Expected an application/x-npy response")
                for chunk in response.iter_bytes(chunk_size=64 * 1024):
                    if len(chunk) > limit - len(data):
                        raise LimitExceeded("NPY response exceeds its descriptor byte limit")
                    data.extend(chunk)
        except httpx.HTTPError as exc:
            raise TensorusError(
                f"Tensorus connection failed: {type(exc).__name__}",
                code="connection_error",
                status=503,
            ) from exc
        payload = bytes(data)
        self._verify_npy_download(io.BytesIO(payload), len(payload), descriptor)
        return load_npy(payload, self.max_tensor_bytes)

    def _verify_npy_download(self, stream: IO[bytes], size: int, descriptor: Record) -> None:
        stream.seek(0)
        shape, fortran_order, dtype = read_npy_header(stream, size, self.max_tensor_bytes)
        if (
            list(shape) != descriptor["shape"]
            or dtype.name != descriptor["dtype"]
            or fortran_order
            or (dtype.byteorder == ">" or (dtype.byteorder == "=" and sys.byteorder != "little"))
        ):
            raise TensorusError("Downloaded NPY does not match the canonical descriptor")
        digest = hashlib.sha256()
        digest.update(canonical_json({"dtype": dtype.name, "shape": list(shape)}).encode())
        digest.update(b"\0")
        while chunk := stream.read(1024**2):
            digest.update(chunk)
        if digest.hexdigest() != descriptor["checksum"]:
            raise TensorusError("Downloaded NPY checksum does not match the version")

    @staticmethod
    def _publish_npy_download(temporary: Path, target: Path, overwrite: bool) -> None:
        if overwrite:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
            temporary.unlink()

    @staticmethod
    def _partial_npy_headers(response: httpx.Response) -> tuple[int, int, int, str]:
        if not response.is_success:
            error = bytearray()
            for chunk in response.iter_bytes(chunk_size=8192):
                error.extend(chunk)
                if len(error) > 65_536:
                    break
            Client._check_response(response, bytes(error))
        if response.status_code != 206:
            raise TensorusError("Server did not return the requested NPY byte range")
        if response.headers.get("content-type", "").split(";")[0] != "application/x-npy":
            raise TensorusError("Expected an application/x-npy response")
        if response.headers.get("content-encoding", "identity").lower() != "identity":
            raise TensorusError("Range downloads require an identity-encoded NPY response")
        content_range = response.headers.get("content-range", "")
        match = re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)", content_range)
        etag = response.headers.get("etag", "")
        length = response.headers.get("content-length", "")
        if (
            len(content_range) > 128
            or match is None
            or re.fullmatch(r'"[0-9a-f]{64}"', etag) is None
            or len(length) > 20
            or not length.isascii()
            or not length.isdigit()
        ):
            raise TensorusError("Invalid NPY byte-range response headers")
        start, end, total = (int(value) for value in match.groups())
        if start > end or end >= total or int(length) != end - start + 1:
            raise TensorusError("Invalid NPY byte-range response bounds")
        return start, end, total, etag

    def read_npy_range(
        self,
        version_id: str,
        *,
        offset: int,
        length: int,
        expected_etag: str | None = None,
        expected_total_bytes: int | None = None,
    ) -> NpyRange:
        """Read one bounded NPY range, binding resumed requests to a strong validator."""
        if type(offset) is not int or not 0 <= offset < 2**63:
            raise ValueError("NPY range offset must be a nonnegative signed 64-bit integer")
        if type(length) is not int or not 1 <= length <= MAX_NPY_RANGE_BYTES:
            raise ValueError(f"NPY range length must be between 1 and {MAX_NPY_RANGE_BYTES}")
        if offset + length > 2**63:
            raise ValueError("NPY range end exceeds the signed 64-bit limit")
        if expected_etag is not None and (
            not isinstance(expected_etag, str)
            or re.fullmatch(r'"[0-9a-f]{64}"', expected_etag) is None
        ):
            raise ValueError("Expected NPY validator must be a strong Tensorus ETag")
        if expected_total_bytes is not None and (
            type(expected_total_bytes) is not int or not 1 <= expected_total_bytes < 2**63
        ):
            raise ValueError("Expected NPY size must be a positive signed 64-bit integer")
        if offset and (expected_etag is None or expected_total_bytes is None):
            raise ValueError("The first range's ETag and total size are required when resuming")
        if expected_total_bytes is not None and offset >= expected_total_bytes:
            raise ValueError("NPY range offset must be below the expected total size")
        descriptor = self.version(version_id)
        nbytes = descriptor_nbytes(descriptor)
        if nbytes > self.max_tensor_bytes:
            raise LimitExceeded("Tensor exceeds the SDK tensor-byte limit")
        path = f"/v1/versions/{quote(version_id, safe='')}/npy"
        headers = {
            "Range": f"bytes={offset}-{offset + length - 1}",
            "Accept-Encoding": "identity",
        }
        if expected_etag is not None:
            headers["If-Range"] = expected_etag
        try:
            with self.http.stream("GET", path, headers=headers) as response:
                start, end, total, etag = self._partial_npy_headers(response)
                if total > self.max_tensor_bytes + NPY_MAX_OVERHEAD_BYTES:
                    raise LimitExceeded("NPY representation exceeds the SDK tensor-byte limit")
                if start != offset or end + 1 != min(offset + length, total):
                    raise TensorusError("The NPY range response has unexpected bounds")
                if expected_etag is not None and etag != expected_etag:
                    raise TensorusError("The NPY representation changed during resumption")
                if expected_total_bytes is not None and total != expected_total_bytes:
                    raise TensorusError("The NPY representation size changed during resumption")
                if not 0 < total - nbytes <= NPY_MAX_OVERHEAD_BYTES:
                    raise TensorusError("The NPY representation size disagrees with its descriptor")
                expected_length = end - start + 1
                data = bytearray()
                for block in response.iter_bytes(chunk_size=64 * 1024):
                    data.extend(block)
                    if len(data) > expected_length:
                        raise TensorusError("The NPY range response exceeded its length")
                if len(data) != expected_length:
                    raise TensorusError("The NPY range response was truncated")
                return NpyRange(offset=start, total_bytes=total, etag=etag, data=bytes(data))
        except httpx.HTTPError as exc:
            raise TensorusError(
                f"Tensorus connection failed: {type(exc).__name__}",
                code="connection_error",
                status=503,
            ) from exc

    def download_npy_file(
        self, version_id: str, destination: str | Path, *, overwrite: bool = False
    ) -> Path:
        """Verify an incremental download before atomically publishing the destination."""
        descriptor = self.version(version_id)
        if descriptor["nbytes"] > self.max_tensor_bytes:
            raise LimitExceeded("Tensor exceeds the SDK full-read byte limit")
        target = Path(destination)
        if not overwrite and target.exists():
            raise FileExistsError(target)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{target.name}.tensorus-", dir=target.parent, mode="w+b", delete=False
            ) as stream:
                temporary = Path(stream.name)
                try:
                    with self.http.stream(
                        "GET", f"/v1/versions/{quote(version_id, safe='')}/npy"
                    ) as response:
                        if not response.is_success:
                            error = bytearray()
                            for chunk in response.iter_bytes(chunk_size=8192):
                                error.extend(chunk)
                                if len(error) > 65_536:
                                    break
                            self._check_response(response, bytes(error))
                        if response.headers.get("content-type", "").split(";")[0] != (
                            "application/x-npy"
                        ):
                            raise TensorusError("Expected an application/x-npy response")
                        size = 0
                        for chunk in response.iter_bytes(chunk_size=64 * 1024):
                            size += len(chunk)
                            if size > self.max_tensor_bytes + NPY_MAX_OVERHEAD_BYTES:
                                raise LimitExceeded("NPY download exceeds the SDK byte limit")
                            stream.write(chunk)
                except httpx.HTTPError as exc:
                    raise TensorusError(
                        f"Tensorus connection failed: {type(exc).__name__}",
                        code="connection_error",
                        status=503,
                    ) from exc
                stream.flush()
                self._verify_npy_download(stream.file, size, descriptor)
                os.fsync(stream.fileno())
            self._publish_npy_download(temporary, target, overwrite)
            return target
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def resume_npy_file(
        self,
        version_id: str,
        destination: str | Path,
        checkpoint: IO[bytes],
        *,
        overwrite: bool = False,
    ) -> Path:
        """Resume from a caller-owned regular file and publish only verified bytes."""
        descriptor = self.version(version_id)
        if descriptor["nbytes"] > self.max_tensor_bytes:
            raise LimitExceeded("Tensor exceeds the SDK full-read byte limit")
        if not checkpoint.seekable() or not checkpoint.readable() or not checkpoint.writable():
            raise TensorusError("The checkpoint must be a seekable readable and writable file")
        try:
            checkpoint_stat = os.fstat(checkpoint.fileno())
        except (AttributeError, OSError) as exc:
            raise TensorusError("The checkpoint must be an open regular file") from exc
        if not stat.S_ISREG(checkpoint_stat.st_mode):
            raise TensorusError("The checkpoint must be an open regular file")
        target = Path(destination)
        if target.exists():
            if os.path.samestat(checkpoint_stat, target.stat()):
                raise TensorusError("Checkpoint and destination must be different files")
            if not overwrite:
                raise FileExistsError(target)
        checkpoint.flush()
        checkpoint.seek(0, os.SEEK_END)
        offset = checkpoint.tell()
        if offset > self.max_tensor_bytes + NPY_MAX_OVERHEAD_BYTES:
            raise LimitExceeded("NPY checkpoint exceeds the SDK byte limit")
        path = f"/v1/versions/{quote(version_id, safe='')}/npy"
        prefix_length = max(1, min(offset, NPY_MAX_OVERHEAD_BYTES))
        try:
            with self.http.stream(
                "GET",
                path,
                headers={"Range": f"bytes=0-{prefix_length - 1}", "Accept-Encoding": "identity"},
            ) as response:
                start, end, total, etag = self._partial_npy_headers(response)
                expected_prefix = min(prefix_length, total)
                if start != 0 or end + 1 != expected_prefix:
                    raise TensorusError("The NPY prefix response has unexpected bounds")
                if total > self.max_tensor_bytes + NPY_MAX_OVERHEAD_BYTES:
                    raise LimitExceeded("NPY download exceeds the SDK byte limit")
                if offset > total:
                    raise TensorusError("NPY checkpoint is longer than the representation")
                prefix = bytearray()
                for chunk in response.iter_bytes(chunk_size=8192):
                    prefix.extend(chunk)
                    if len(prefix) > expected_prefix:
                        raise TensorusError("The NPY prefix response exceeds its declared length")
                if len(prefix) != expected_prefix:
                    raise TensorusError("The NPY prefix response was truncated")
            if offset:
                checkpoint.seek(0)
                if checkpoint.read(expected_prefix) != prefix:
                    raise TensorusError("NPY checkpoint prefix does not match the current version")
            if offset < total:
                with self.http.stream(
                    "GET",
                    path,
                    headers={
                        "Range": f"bytes={offset}-",
                        "If-Range": etag,
                        "Accept-Encoding": "identity",
                    },
                ) as response:
                    received_start, received_end, received_total, received_etag = (
                        self._partial_npy_headers(response)
                    )
                    if (received_start, received_end, received_total, received_etag) != (
                        offset,
                        total - 1,
                        total,
                        etag,
                    ):
                        raise TensorusError("The NPY range response changed during resumption")
                    checkpoint.seek(0, os.SEEK_END)
                    if checkpoint.tell() != offset:
                        raise TensorusError("NPY checkpoint changed during resumption")
                    received = 0
                    try:
                        for chunk in response.iter_bytes(chunk_size=64 * 1024):
                            received += len(chunk)
                            if received > total - offset:
                                raise TensorusError("The NPY range response exceeded its length")
                            if checkpoint.write(chunk) != len(chunk):
                                raise TensorusError("Could not write the complete NPY checkpoint")
                    finally:
                        checkpoint.flush()
                    if received != total - offset:
                        raise TensorusError("The NPY range response was truncated")
        except httpx.HTTPError as exc:
            raise TensorusError(
                f"Tensorus connection failed: {type(exc).__name__}",
                code="connection_error",
                status=503,
            ) from exc

        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{target.name}.tensorus-", dir=target.parent, mode="w+b", delete=False
            ) as stream:
                temporary = Path(stream.name)
                checkpoint.seek(0)
                size = 0
                while chunk := checkpoint.read(1024**2):
                    size += len(chunk)
                    if size > total:
                        raise TensorusError("NPY checkpoint changed before publication")
                    stream.write(chunk)
                if size != total:
                    raise TensorusError("NPY checkpoint changed before publication")
                stream.flush()
                self._verify_npy_download(stream.file, size, descriptor)
                os.fsync(stream.fileno())
            self._publish_npy_download(temporary, target, overwrite)
            return target
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def preview(self, version_id: str, selection: list[list[int | None]] | None = None) -> Record:
        path = f"/v1/versions/{quote(version_id, safe='')}"
        if selection is None:
            return self._request("GET", f"{path}/preview").json()
        return self._request("POST", f"{path}/slice", json={"selection": selection}).json()

    def read_npy_tile(self, version_id: str, selection: list[list[int | None]]) -> Array:
        """Read a bounded positional slice without requiring a full-source checksum."""
        request = SliceRequest(selection=selection)
        descriptor = self.version(version_id)
        descriptor_nbytes(descriptor)
        _, selected_shape = positional_selection(descriptor["shape"], request.selection)
        expected_bytes = math.prod(selected_shape) * np.dtype(descriptor["dtype"]).itemsize
        maximum = min(MAX_NPY_TILE_BYTES, self.max_tensor_bytes)
        if expected_bytes > maximum:
            raise LimitExceeded(f"NPY tile exceeds the {maximum}-byte SDK output limit")
        path = f"/v1/versions/{quote(version_id, safe='')}/slice/npy"
        try:
            with self.http.stream(
                "POST",
                path,
                json=request.model_dump(mode="json"),
                headers={"Accept-Encoding": "identity"},
            ) as response:
                if not response.is_success:
                    error = bytearray()
                    for block in response.iter_bytes(chunk_size=8192):
                        error.extend(block)
                        if len(error) > 65_536:
                            break
                    self._check_response(response, bytes(error))
                length = response.headers.get("content-length", "")
                digest = response.headers.get("x-tensorus-tile-sha256", "")
                if (
                    response.status_code != 200
                    or response.headers.get("content-type", "").split(";", 1)[0].lower()
                    != "application/x-npy"
                    or response.headers.get("content-encoding", "identity").lower() != "identity"
                    or response.headers.get("x-tensorus-version") != descriptor["id"]
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                    or len(length) > 20
                    or not length.isascii()
                    or not length.isdigit()
                    or not 0 < int(length) - expected_bytes <= NPY_MAX_OVERHEAD_BYTES
                ):
                    raise TensorusError("Invalid NPY tile response headers")
                expected_length = int(length)
                data = bytearray()
                for block in response.iter_bytes(chunk_size=64 * 1024):
                    data.extend(block)
                    if len(data) > expected_length:
                        raise TensorusError("The NPY tile response exceeded its length")
                if len(data) != expected_length:
                    raise TensorusError("The NPY tile response was truncated")
                if hashlib.sha256(data).hexdigest() != digest:
                    raise TensorusError("The NPY tile response digest does not match")
                result = load_npy(bytes(data), maximum)
                if result.shape != selected_shape or result.dtype.name != descriptor["dtype"]:
                    raise TensorusError("The NPY tile does not match its requested descriptor")
                return result
        except httpx.HTTPError as exc:
            raise TensorusError(
                f"Tensorus connection failed: {type(exc).__name__}",
                code="connection_error",
                status=503,
            ) from exc

    def preview_axes(self, version_id: str, selection: dict[str, list[int | None]]) -> Record:
        path = f"/v1/versions/{quote(version_id, safe='')}/axis-slice"
        return self._request("POST", path, json={"selection": selection}).json()

    def preview_coordinates(self, version_id: str, selection: dict[str, dict[str, Any]]) -> Record:
        """Select inclusive numeric coordinate ranges from a version's declared coordinates."""
        path = f"/v1/versions/{quote(version_id, safe='')}/coordinate-slice"
        return self._request("POST", path, json={"selection": selection}).json()

    def _require_coordinate_points(
        self, request: TensorQuery, capabilities: Record | None = None
    ) -> bool:
        points_requested = request.semantics is not None and any(
            axis.coordinate_points is not None for axis in request.semantics.axes
        )
        if points_requested:
            capabilities = capabilities if capabilities is not None else self.capabilities()
            semantics = (
                capabilities.get("tensor_semantics") if isinstance(capabilities, dict) else None
            )
            schema = semantics.get("descriptor_schema") if isinstance(semantics, dict) else None
            marker = (
                schema.get("x-tensorus-coordinate-points-query")
                if isinstance(schema, dict)
                else None
            )
            if (
                marker
                != {
                    "format": COORDINATE_POINT_QUERY_FORMAT,
                    "max_points": MAX_COORDINATE_QUERY_POINTS,
                    "relations": ["all", "any"],
                }
                or type(marker["max_points"]) is not int
            ):
                raise TensorusError(
                    "Server does not advertise the required coordinate-points query contract",
                    code="unsupported_capability",
                )
            typed_requested = request.semantics is not None and any(
                axis.coordinate_points is not None
                and axis.coordinate_points.comparison == "typed_v2"
                for axis in request.semantics.axes
            )
            typed_marker = (
                schema.get("x-tensorus-typed-coordinate-points-query")
                if isinstance(schema, dict)
                else None
            )
            if typed_requested and typed_marker != {
                "format": TYPED_COORDINATE_POINT_QUERY_FORMAT,
                "comparison": "typed_v2",
                "semantics": "v2",
            }:
                raise TensorusError(
                    "Server does not advertise the required typed coordinate-points query contract",
                    code="unsupported_capability",
                )
        return points_requested

    def _require_coordinate_counts(
        self, request: TensorQuery, capabilities: Record | None = None
    ) -> bool:
        requested = request.semantics is not None and any(
            axis.coordinate_count is not None for axis in request.semantics.axes
        )
        if requested:
            capabilities = capabilities if capabilities is not None else self.capabilities()
            semantics = (
                capabilities.get("tensor_semantics") if isinstance(capabilities, dict) else None
            )
            schema = semantics.get("descriptor_schema") if isinstance(semantics, dict) else None
            marker = (
                schema.get("x-tensorus-coordinate-count-query")
                if isinstance(schema, dict)
                else None
            )
            if (
                marker
                != {
                    "format": COORDINATE_COUNT_QUERY_FORMAT,
                    "kind": "numeric_v1",
                    "semantics": "v2",
                    "interval": "closed",
                    "max_coordinates": MAX_COORDINATE_QUERY_COUNT,
                }
                or type(marker["max_coordinates"]) is not int
            ):
                raise TensorusError(
                    "Server does not advertise the required coordinate-count query contract",
                    code="unsupported_capability",
                )
        return requested

    def _shared_axis_anchor(
        self, request: TensorQuery, capabilities: Record | None = None
    ) -> tuple[SharedAxisQueryMatcher, str] | None:
        selected = request.semantics.shared_axis if request.semantics is not None else None
        if selected is None:
            return None
        capabilities = capabilities if capabilities is not None else self.capabilities()
        index = capabilities.get("semantic_discovery_index")
        cross = selected.anchor_snapshot_id is not None
        overlap = selected.kind in (
            "overlap_shared_axis_v1",
            "overlap_cross_snapshot_shared_axis_v1",
        )
        marker_name = {
            (False, False): "shared_axis",
            (False, True): "cross_snapshot_shared_axis",
            (True, False): "overlap_shared_axis",
            (True, True): "cross_snapshot_overlap_shared_axis",
        }[(overlap, cross)]
        marker = index.get(marker_name) if isinstance(index, dict) else None
        expected = (
            {
                "format": (
                    OVERLAP_CROSS_SNAPSHOT_QUERY_FORMAT
                    if overlap
                    else "tensorus-exact-cross-snapshot-shared-axis-query-v1"
                ),
                "kind": selected.kind,
                "max_axes": 16,
                "coordinate_values": MAX_SHARED_AXIS_QUERY_COORDINATE_VALUES,
                "candidate_and_anchor_snapshots_required": True,
                "same_tenant_required": True,
            }
            if cross
            else {
                "format": OVERLAP_SHARED_AXIS_QUERY_FORMAT if overlap else SHARED_AXIS_QUERY_FORMAT,
                "kind": selected.kind,
                "max_axes": 16,
                "coordinate_values": MAX_SHARED_AXIS_QUERY_COORDINATE_VALUES,
                "snapshot_member_required": True,
            }
        )
        if marker != expected:
            raise TensorusError(
                "Server does not advertise the required shared-axis query contract",
                code="unsupported_capability",
            )
        response = self._request("GET", f"/v1/versions/{selected.version_id}")
        snapshot_response = (
            self._request("GET", f"/v1/snapshots/{selected.anchor_snapshot_id}")
            if selected.anchor_snapshot_id is not None
            else None
        )
        try:
            raw = decode_unique_json(response.content)
            version = TensorVersion.model_validate_json(response.content, strict=True)
            if not isinstance(raw, dict) or version.id != selected.version_id:
                raise ValueError("Shared-axis anchor identity differs from the request")
            descriptor = version.model_dump(mode="json")
            descriptor_nbytes(descriptor)
            snapshot_descriptor = None
            if selected.anchor_snapshot_id is not None and snapshot_response is not None:
                snapshot_raw = decode_unique_json(snapshot_response.content)
                snapshot = SnapshotDetail.model_validate_json(
                    snapshot_response.content, strict=True
                )
                if (
                    not isinstance(snapshot_raw, dict)
                    or snapshot.id != selected.anchor_snapshot_id
                    or snapshot.tenant_id != version.tenant_id
                    or snapshot.dataset_id != version.dataset_id
                ):
                    raise ValueError("Shared-axis anchor snapshot identity is inconsistent")
                snapshot_descriptor = snapshot.model_dump(mode="json")
            semantics = read_semantics(version.metadata, tuple(version.shape), version.operation)
            if semantics is None:
                raise ValueError("Shared-axis anchor has no declared semantics")
            matcher = SharedAxisQueryMatcher(
                semantics,
                tuple(version.shape),
                tuple(selected.axes),
                anchor_scope=(str(version.tenant_id), str(version.dataset_id)),
                across_datasets=cross,
                overlap=overlap,
            )
            immutable_fields = (
                "id",
                "tenant_id",
                "tensor_id",
                "version",
                "dataset_id",
                "dataset",
                "name",
                "dtype",
                "shape",
                "nbytes",
                "checksum",
                "metadata",
                "operation",
            )
            fingerprint = canonical_hash(
                {
                    "version": {key: descriptor[key] for key in immutable_fields},
                    "anchor_snapshot": snapshot_descriptor,
                }
            )
            return matcher, fingerprint
        except (ValidationError, TensorusError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "Server returned an invalid shared-axis anchor", code="invalid_response"
            ) from exc

    def _confirm_shared_axis_anchor(
        self, request: TensorQuery, capabilities: Record | None, fingerprint: str | None
    ) -> None:
        if fingerprint is None:
            return
        current = self._shared_axis_anchor(request, capabilities)
        if current is None or current[1] != fingerprint:
            raise TensorusError(
                "Shared-axis anchor changed during discovery", code="invalid_response"
            )

    def query(self, query: TensorQuery | None = None, **kwargs: Any) -> Record:
        request = query or TensorQuery(**kwargs)
        if request.indexed_execution is not None:
            raise TensorusError("Indexed execution requires an indexed discovery endpoint")
        coordinate_capabilities = (
            self.capabilities()
            if request.semantics is not None
            and any(
                axis.coordinate_points is not None or axis.coordinate_count is not None
                for axis in request.semantics.axes
            )
            else None
        )
        points_requested = self._require_coordinate_points(request, coordinate_capabilities)
        counts_requested = self._require_coordinate_counts(request, coordinate_capabilities)
        anchor = self._shared_axis_anchor(request)
        matcher = anchor[0] if anchor is not None else None
        response = self._request("POST", "/v1/query", json=request.model_dump(mode="json"))
        if not points_requested and not counts_requested and matcher is None:
            return response.json()
        try:
            raw = decode_unique_json(response.content)
            page = Page[TensorVersion].model_validate_json(response.content, strict=True)
            if (
                not isinstance(raw, dict)
                or page.total < 0
                or page.limit != request.limit
                or page.offset != request.offset
                or len(page.items) != min(request.limit, max(page.total - request.offset, 0))
                or len({item.id for item in page.items}) != len(page.items)
                or [(item.dataset, item.name) for item in page.items]
                != sorted({(item.dataset, item.name) for item in page.items})
            ):
                raise ValueError("Inconsistent query pagination")
            for item in page.items:
                descriptor = item.model_dump(mode="json")
                descriptor["attributes"] = descriptor["metadata"]
                if (
                    (request.dataset is not None and item.dataset != request.dataset)
                    or (request.dtype is not None and item.dtype != request.dtype)
                    or (request.shape is not None and item.shape != request.shape)
                    or request.semantics is None
                    or not matches_semantics(descriptor, request.semantics)
                    or not matches_shared_axis(descriptor, matcher)
                    or (
                        request.min_norm is not None
                        and (
                            item.statistics.norm is None or item.statistics.norm < request.min_norm
                        )
                    )
                    or (
                        request.max_norm is not None
                        and (
                            item.statistics.norm is None or item.statistics.norm > request.max_norm
                        )
                    )
                    or not self._matches_metadata(item.metadata, request.metadata)
                ):
                    raise ValueError("Query item does not match the requested filters")
        except (ValidationError, TensorusError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "Server returned inconsistent semantic query results",
                code="invalid_response",
            ) from exc
        self._confirm_shared_axis_anchor(request, None, anchor[1] if anchor else None)
        return raw

    def _semantic_index_capability(self) -> Record:
        capabilities = self.capabilities()
        marker = capabilities.get("semantic_discovery_index")
        if (
            not isinstance(marker, dict)
            or marker.get("format") != INDEX_FORMAT
            or marker.get("properties") != list(INDEXED_PROPERTIES)
            or marker.get("query_reconciles_source") is not True
            or marker.get("automatic_legacy_backfill") is not False
            or type(marker.get("reconciliation_candidates")) is not int
            or marker["reconciliation_candidates"] < 1
            or type(marker.get("exact_match_candidates")) is not int
            or marker["exact_match_candidates"] < 1
            or type(marker.get("rebuild_versions")) is not int
            or marker["rebuild_versions"] < 1
            or type(marker.get("rebuild_descriptor_bytes")) is not int
            or marker["rebuild_descriptor_bytes"] < 1
        ):
            raise TensorusError(
                "Server does not advertise the required semantic index contract",
                code="unsupported_capability",
            )
        return capabilities

    @staticmethod
    def _require_streamed_index_capability(
        request: TensorQuery, capabilities: Record
    ) -> StreamedIndexEvaluationCapabilities | None:
        if request.indexed_execution is None:
            return None
        marker = capabilities["semantic_discovery_index"]
        try:
            streamed = StreamedIndexEvaluationCapabilities.model_validate(
                marker.get("streamed_evaluation"), strict=True
            )
            if (
                streamed.format != STREAMED_EVALUATION_FORMAT
                or streamed.execution != request.indexed_execution
                or streamed.max_candidates > marker["reconciliation_candidates"]
                or streamed.fetch_batch > streamed.max_candidates
            ):
                raise ValueError("Streamed evaluation capabilities disagree")
        except (ValidationError, ValueError, TypeError) as exc:
            raise TensorusError(
                "Server does not advertise bounded streamed indexed evaluation",
                code="unsupported_capability",
            ) from exc
        return streamed

    def query_indexed(self, query: TensorQuery | None = None, **kwargs: Any) -> Record:
        """Query a reconciled property index; completeness is checked by the server."""
        request = query or TensorQuery(**kwargs)
        capabilities = self._semantic_index_capability()
        self._require_streamed_index_capability(request, capabilities)
        self._require_coordinate_points(request, capabilities)
        self._require_coordinate_counts(request, capabilities)
        anchor = self._shared_axis_anchor(request, capabilities)
        response = self._request(
            "POST", "/v1/discovery/semantic-index/query", json=request.model_dump(mode="json")
        )
        try:
            raw = decode_unique_json(response.content)
            page = IndexedTensorPage.model_validate_json(response.content, strict=True)
            if not isinstance(raw, dict):
                raise ValueError("Indexed response must be an object")
            self._check_indexed_page(page, request, capabilities, anchor[0] if anchor else None)
        except (ValidationError, TensorusError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "Server returned inconsistent indexed discovery results", code="invalid_response"
            ) from exc
        self._confirm_shared_axis_anchor(request, capabilities, anchor[1] if anchor else None)
        return raw

    @staticmethod
    def _matches_metadata(metadata: dict[str, JsonValue], filters: dict[str, JsonValue]) -> bool:
        for key, expected in filters.items():
            actual = metadata.get(key)
            if (
                key not in metadata
                or (
                    isinstance(expected, bool)
                    and (not isinstance(actual, bool) or actual != expected)
                )
                or (
                    not isinstance(expected, bool)
                    and (
                        isinstance(actual, bool)
                        or actual != expected
                        or (
                            isinstance(expected, (int, float))
                            and not isinstance(actual, (int, float))
                        )
                    )
                )
            ):
                return False
        return True

    @staticmethod
    def _check_indexed_page(
        page: IndexedTensorPage,
        request: TensorQuery,
        capabilities: Record,
        matcher: SharedAxisQueryMatcher | None,
    ) -> None:
        evidence = page.index
        index_capability = capabilities["semantic_discovery_index"]
        limit = index_capability["reconciliation_candidates"]
        streamed = evidence.streamed_evaluation
        if request.indexed_execution is None:
            if streamed is not None:
                raise ValueError("Unexpected streamed indexed evaluation evidence")
            exact_limit = index_capability["exact_match_candidates"]
        else:
            advertised = Client._require_streamed_index_capability(request, capabilities)
            if (
                advertised is None
                or streamed is None
                or streamed.format != STREAMED_EVALUATION_FORMAT
                or not page.total <= streamed.candidates_evaluated <= evidence.reconciled_candidates
                or streamed.candidates_evaluated > advertised.max_candidates
                or streamed.descriptor_bytes > advertised.max_descriptor_bytes
                or (streamed.descriptor_bytes == 0) != (streamed.candidates_evaluated == 0)
            ):
                raise ValueError("Inconsistent streamed indexed evaluation evidence")
            exact_limit = advertised.max_candidates
        if (
            page.total < 0
            or page.limit != request.limit
            or page.offset != request.offset
            or len(page.items) != min(request.limit, max(page.total - request.offset, 0))
            or len({item.id for item in page.items}) != len(page.items)
            or [(item.dataset, item.name) for item in page.items]
            != sorted({(item.dataset, item.name) for item in page.items})
            or evidence.query_hash != canonical_hash(request.model_dump(mode="json"))
            or evidence.snapshot_id != request.snapshot_id
            or not page.total <= evidence.reconciled_candidates <= limit
            or page.total > exact_limit
        ):
            raise ValueError("Inconsistent indexed discovery evidence")
        selected = request.semantics.shared_axis if request.semantics is not None else None
        shared = evidence.shared_axis
        overlap = selected is not None and selected.kind in (
            "overlap_shared_axis_v1",
            "overlap_cross_snapshot_shared_axis_v1",
        )
        if selected is None:
            if shared is not None or matcher is not None:
                raise ValueError("Unexpected shared-axis evidence")
        elif (
            shared is None
            or matcher is None
            or shared.format
            != (OVERLAP_SHARED_AXIS_QUERY_FORMAT if overlap else SHARED_AXIS_QUERY_FORMAT)
            or shared.version_id != selected.version_id
            or shared.axes != selected.axes
            or not page.total <= shared.candidates_evaluated <= evidence.reconciled_candidates
            or shared.candidates_evaluated > exact_limit
            or (
                streamed is not None and shared.candidates_evaluated > streamed.candidates_evaluated
            )
            or shared.coordinate_values > MAX_SHARED_AXIS_QUERY_COORDINATE_VALUES
            or shared.kind
            != (selected.kind if overlap or selected.anchor_snapshot_id is not None else None)
            or shared.anchor_snapshot_id != selected.anchor_snapshot_id
        ):
            raise ValueError("Inconsistent shared-axis query evidence")
        for item in page.items:
            descriptor = item.model_dump(mode="json")
            descriptor["attributes"] = descriptor["metadata"]
            if (
                (request.dataset is not None and item.dataset != request.dataset)
                or (request.dtype is not None and item.dtype != request.dtype)
                or (request.shape is not None and item.shape != request.shape)
                or request.semantics is None
                or not matches_semantics(descriptor, request.semantics)
                or not matches_shared_axis(descriptor, matcher)
                or (
                    request.min_norm is not None
                    and (item.statistics.norm is None or item.statistics.norm < request.min_norm)
                )
                or (
                    request.max_norm is not None
                    and (item.statistics.norm is None or item.statistics.norm > request.max_norm)
                )
                or not Client._matches_metadata(item.metadata, request.metadata)
            ):
                raise ValueError("Indexed result does not match the requested filters")
        if (
            matcher is not None
            and shared is not None
            and matcher.compared_coordinate_values > shared.coordinate_values
        ):
            raise ValueError("Shared-axis work evidence is inconsistent")

    def query_indexed_snapshot_page(
        self, query: TensorQuery, *, cursor: str | None = None
    ) -> Record:
        """Traverse a pinned snapshot; cursors are not authentication tokens."""
        request = IndexedSnapshotQuery(query=query, cursor=cursor)
        snapshot_id = query.snapshot_id
        if snapshot_id is None:
            raise TensorusError("Snapshot cursor queries require a snapshot")
        capabilities = self._semantic_index_capability()
        self._require_streamed_index_capability(query, capabilities)
        if (
            capabilities["semantic_discovery_index"].get("snapshot_cursor_format")
            != SNAPSHOT_CURSOR_FORMAT
        ):
            raise TensorusError(
                "Server does not advertise indexed snapshot cursors",
                code="unsupported_capability",
            )
        self._require_coordinate_points(query, capabilities)
        self._require_coordinate_counts(query, capabilities)
        anchor = self._shared_axis_anchor(query, capabilities)
        previous = decode_snapshot_cursor(cursor) if cursor is not None else None
        query_hash = canonical_hash(query.model_dump(mode="json"))
        if previous is not None and (
            previous.query_hash != query_hash or previous.snapshot_id != query.snapshot_id
        ):
            raise TensorusError("Snapshot cursor does not match the query")
        offset = previous.offset if previous is not None else 0
        response = self._request(
            "POST",
            "/v1/discovery/semantic-index/snapshot-page",
            json=request.model_dump(mode="json"),
        )
        try:
            raw = decode_unique_json(response.content)
            page = IndexedSnapshotPage.model_validate_json(response.content, strict=True)
            if not isinstance(raw, dict) or page.query_hash != query_hash:
                raise ValueError("Snapshot page query identity differs")
            self._check_indexed_page(
                page,
                query.model_copy(update={"offset": offset}),
                capabilities,
                anchor[0] if anchor else None,
            )
            if previous is not None and page.total != previous.total:
                raise ValueError("Snapshot page total differs from its cursor")
            next_offset = offset + len(page.items)
            if next_offset < page.total:
                last = page.items[-1]
                expected = encode_snapshot_cursor(
                    SnapshotCursor(
                        query_hash=query_hash,
                        snapshot_id=snapshot_id,
                        offset=next_offset,
                        total=page.total,
                        last_version_id=last.id,
                        last_dataset=last.dataset,
                        last_name=last.name,
                    )
                )
                if page.next_cursor != expected:
                    raise ValueError("Snapshot continuation cursor is inconsistent")
            elif page.next_cursor is not None:
                raise ValueError("Completed snapshot page returned a cursor")
        except (ValidationError, TensorusError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "Server returned inconsistent indexed snapshot results", code="invalid_response"
            ) from exc
        self._confirm_shared_axis_anchor(query, capabilities, anchor[1] if anchor else None)
        return raw

    def semantic_index_status(self) -> Record:
        self._semantic_index_capability()
        response = self._request("GET", "/v1/discovery/semantic-index")
        try:
            raw = decode_unique_json(response.content)
            status = SemanticIndexStatus.model_validate_json(response.content, strict=True)
            if (
                not isinstance(raw, dict)
                or status.versions < 0
                or status.index_entries < 0
                or (status.generation is None) != (status.watermark is None)
                or status.counts_match
                != (status.generation is not None and status.versions == status.index_entries)
            ):
                raise ValueError("Inconsistent semantic index status")
        except (ValidationError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "Server returned inconsistent semantic index status", code="invalid_response"
            ) from exc
        return raw

    def rebuild_semantic_index(self) -> Record:
        capabilities = self._semantic_index_capability()
        response = self._request("POST", "/v1/discovery/semantic-index/rebuild")
        try:
            raw = decode_unique_json(response.content)
            result = SemanticIndexRebuild.model_validate_json(response.content, strict=True)
            limits = capabilities["semantic_discovery_index"]
            if (
                not isinstance(raw, dict)
                or result.generation < 1
                or result.watermark < 1
                or not 0 <= result.versions <= limits["rebuild_versions"]
                or not 0 <= result.descriptor_bytes <= limits["rebuild_descriptor_bytes"]
            ):
                raise ValueError("Inconsistent semantic index rebuild result")
        except (ValidationError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "Server returned inconsistent semantic index rebuild evidence",
                code="invalid_response",
            ) from exc
        return raw

    def search(
        self,
        version_id: str,
        *,
        metric: Literal["cosine", "l2"] = "cosine",
        k: int = 10,
        dataset: str | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> Record:
        """Search raw current heads and return the versioned numerical scoring engine."""
        request = SimilarityQuery(
            version_id=UUID(version_id),
            metric=metric,
            k=k,
            dataset=dataset,
            metadata=metadata or {},
        )
        return self._request("POST", "/v1/search", json=request.model_dump(mode="json")).json()

    def search_snapshot(
        self,
        version_id: str,
        snapshot_id: str,
        *,
        metric: Literal["cosine", "l2"] = "cosine",
        k: int = 10,
        metadata: dict[str, JsonValue] | None = None,
        expected_membership_hash: str | None = None,
        expected_candidate_hash: str | None = None,
    ) -> Record:
        """Return unretained raw snapshot evidence, including its versioned scoring engine."""
        request = SnapshotSimilarityRequest(
            snapshot_id=UUID(snapshot_id),
            version_id=UUID(version_id),
            metric=metric,
            k=k,
            metadata={} if metadata is None else metadata,
            expected_membership_hash=expected_membership_hash,
            expected_candidate_hash=expected_candidate_hash,
        )
        return self._request(
            "POST", "/v1/search/snapshot", json=request.model_dump(mode="json")
        ).json()

    def search_snapshot_validity(self, request: ValiditySnapshotSimilarityRequest) -> Record:
        """Compare explicitly selected pairwise validity intersections with unretained evidence."""
        response = self._request(
            "POST", "/v1/search/snapshot/validity", json=request.model_dump(mode="json")
        ).json()
        return ValiditySnapshotSimilarityReport.model_validate(response).model_dump(mode="json")

    def lineage(self, version_id: str) -> Record:
        return self._request("GET", f"/v1/versions/{quote(version_id, safe='')}/lineage").json()

    def descendants(self, version_id: str, *, max_depth: int = 12) -> Record:
        """Find bounded recorded downstream versions, not predicted future effects."""
        return self._request(
            "GET",
            f"/v1/versions/{quote(version_id, safe='')}/descendants",
            params={"max_depth": max_depth},
        ).json()

    def snapshot(self, dataset: str, name: str) -> Record:
        return self._request(
            "POST", f"/v1/datasets/{quote(dataset, safe='')}/snapshots", json={"name": name}
        ).json()

    def snapshots(self, dataset: str, *, limit: int = 100, offset: int = 0) -> Record:
        return self._request(
            "GET",
            f"/v1/datasets/{quote(dataset, safe='')}/snapshots",
            params={"limit": limit, "offset": offset},
        ).json()

    def snapshot_cursor_page(
        self,
        dataset: str,
        *,
        limit: int = 100,
        after: str | None = None,
        before: str | None = None,
    ) -> Record:
        if (
            type(limit) is not int
            or not 1 <= limit <= 200
            or (after is not None and before is not None)
        ):
            raise ValueError("Snapshot cursor pagination requires limit 1..200 and one direction")
        for cursor in (after, before):
            if cursor is not None:
                try:
                    if type(cursor) is not str or str(UUID(cursor)) != cursor:
                        raise ValueError
                except (TypeError, ValueError, AttributeError) as exc:
                    raise ValueError("Snapshot page cursor must be a canonical UUID") from exc
        if "snapshot_cursor_list" not in self.capabilities():
            raise TensorusError(
                "The server does not advertise snapshot cursor pages",
                code="unsupported_capability",
            )
        params: dict[str, str | int] = {"limit": limit, "mode": "cursor"}
        if after is not None:
            params["after"] = after
        if before is not None:
            params["before"] = before
        encoded = self._bounded_report(
            "GET",
            f"/v1/datasets/{quote(dataset, safe='')}/snapshots",
            262_144,
            "Snapshot cursor page",
            params=params,
        )
        page = self._checked_exact_response(encoded, SnapshotCursorPage, "snapshot cursor page")
        if (
            page.limit != limit
            or (str(page.after) if page.after is not None else None) != after
            or (str(page.before) if page.before is not None else None) != before
        ):
            raise TensorusError(
                "The server returned a different snapshot cursor request", code="invalid_response"
            )
        return page.model_dump(mode="json")

    def get_snapshot(self, snapshot_id: str) -> Record:
        return self._request("GET", f"/v1/snapshots/{quote(snapshot_id, safe='')}").json()

    @staticmethod
    def _checked_exact_response[T: BaseModel](
        encoded: bytes, model: type[T], label: str, *, mutation: bool = False
    ) -> T:
        try:
            raw = decode_unique_json(encoded)
            result = model.model_validate(raw)
            if canonical_json(raw) != canonical_json(result.model_dump(mode="json")):
                raise ValueError("Response evidence is not exact")
            return result
        except (ValidationError, TensorusError, ValueError, UnicodeError, RecursionError) as exc:
            suffix = (
                "; the operation may have committed, so inspect before retrying" if mutation else ""
            )
            raise TensorusError(
                f"The server returned invalid {label}{suffix}", code="invalid_response"
            ) from exc

    def _lifecycle_post(self, path: str, body: Record, limit: int, label: str) -> bytes:
        try:
            return self._bounded_report("POST", path, limit, label, json=body)
        except TensorusError as exc:
            raise TensorusError(
                f"{exc}; {label} may have committed, so inspect before retrying",
                code=exc.code,
                status=exc.status,
                details=exc.details,
            ) from exc

    def preview_snapshot_retirement(self, snapshot_id: str) -> Record:
        """Inspect retained-reference blockers; a preview does not reserve the snapshot."""
        return self._request(
            "GET", f"/v1/snapshots/{quote(snapshot_id, safe='')}/retirement"
        ).json()

    def review_snapshot_retirement(self, snapshot_id: str) -> Record:
        """Review retained references and active holds without reserving the snapshot."""
        encoded = self._bounded_report(
            "GET",
            f"/v1/snapshots/{quote(snapshot_id, safe='')}/retirement-review",
            8192,
            "Snapshot retirement review",
        )
        result = self._checked_exact_response(
            encoded, SnapshotRetirementReview, "snapshot retirement review"
        )
        if str(result.preview.snapshot_id) != str(UUID(snapshot_id)):
            raise TensorusError("The server returned a different snapshot retirement review")
        return result.model_dump(mode="json")

    def retire_snapshot(self, snapshot_id: str) -> Record:
        """Retire unreferenced membership; retry the same ID after an uncertain response."""
        return self._request("DELETE", f"/v1/snapshots/{quote(snapshot_id, safe='')}").json()

    def export_snapshot_retirement_archive(self, request: RetirementArchiveSelection) -> Record:
        """Export selected audit receipts without pruning them or freeing receipt quota."""
        return self._request(
            "POST",
            "/v1/snapshot-retirement-archives/export",
            json=request.model_dump(mode="json"),
        ).json()

    def retention_holds(self, resource_type: RetentionResource, resource_id: str) -> Record:
        """Inspect the retained hold history for a snapshot or product release."""
        encoded = self._bounded_report(
            "GET",
            "/v1/retention-holds",
            MAX_HOLDS_PER_RESOURCE * 16_384 + 4096,
            "Retention hold history",
            params={"resource_type": resource_type, "resource_id": resource_id},
        )
        result = self._checked_exact_response(
            encoded, RetentionHoldHistory, "retention hold history"
        )
        if result.resource_type != resource_type or str(result.resource_id) != str(
            UUID(resource_id)
        ):
            raise TensorusError("The server returned hold history for a different resource")
        return result.model_dump(mode="json")

    def create_retention_hold(self, request: RetentionHoldCreate) -> Record:
        """Place an explicit hold; reuse the request ID for an exact retry."""
        encoded = self._lifecycle_post(
            "/v1/retention-holds", request.model_dump(mode="json"), 16_384, "Retention hold"
        )
        result = self._checked_exact_response(
            encoded, RetentionHold, "retention hold", mutation=True
        )
        if (
            result.id != request.request_id
            or result.resource_type != request.resource_type
            or result.resource_id != request.resource_id
            or result.reason != request.reason
        ):
            raise TensorusError(
                "The server returned a different retention hold; it may have committed, "
                "so inspect before retrying",
                code="invalid_response",
            )
        return result.model_dump(mode="json")

    def release_retention_hold(self, hold_id: str, request: RetentionHoldRelease) -> Record:
        """Release one hold with an auditable reason and exact retry identity."""
        encoded = self._lifecycle_post(
            f"/v1/retention-holds/{quote(hold_id, safe='')}/release",
            request.model_dump(mode="json"),
            16_384,
            "Retention hold release",
        )
        result = self._checked_exact_response(
            encoded, RetentionHold, "retention hold release", mutation=True
        )
        if (
            str(result.id) != str(UUID(hold_id))
            or result.state != "released"
            or result.release_request_id != request.request_id
            or result.release_reason != request.reason
        ):
            raise TensorusError(
                "The server returned a different retention hold release; it may have committed, "
                "so inspect before retrying",
                code="invalid_response",
            )
        return result.model_dump(mode="json")

    def snapshot_diff(self, left: str, right: str) -> Record:
        return self._request(
            "GET",
            f"/v1/snapshots/{quote(left, safe='')}/diff/{quote(right, safe='')}",
        ).json()

    def contracts(self, dataset: str, *, limit: int = 50, offset: int = 0) -> Record:
        return self._request(
            "GET",
            f"/v1/datasets/{quote(dataset, safe='')}/contracts",
            params={"limit": limit, "offset": offset},
        ).json()

    def contract(self, dataset: str, revision: int) -> Record:
        return self._request(
            "GET", f"/v1/datasets/{quote(dataset, safe='')}/contracts/{revision}"
        ).json()

    def append_contract(
        self, dataset: str, definition: ContractDefinition, *, expected_revision: int
    ) -> Record:
        request = ContractWrite(definition=definition, expected_revision=expected_revision)
        return self._request(
            "POST",
            f"/v1/datasets/{quote(dataset, safe='')}/contracts",
            json=request.model_dump(mode="json"),
        ).json()

    def write_contract(self, dataset: str) -> Record:
        """Inspect the active additive write overlay, or generation zero if none."""
        return self._request("GET", f"/v1/datasets/{quote(dataset, safe='')}/write-contract").json()

    def write_contract_history(self, dataset: str, *, limit: int = 50, offset: int = 0) -> Record:
        return self._request(
            "GET",
            f"/v1/datasets/{quote(dataset, safe='')}/write-contract/activations",
            params={"limit": limit, "offset": offset},
        ).json()

    def activate_write_contract(self, dataset: str, request: WriteContractActivation) -> Record:
        """Activate a revision; reuse the exact request after an uncertain result."""
        return self._request(
            "POST",
            f"/v1/datasets/{quote(dataset, safe='')}/write-contract/activations",
            json=request.model_dump(mode="json"),
        ).json()

    def prepare_release(self, dataset: str, request: ReleaseRequest) -> Record:
        """Evaluate a pinned snapshot; repeat the identical named request after a timeout."""
        return self._request(
            "POST",
            f"/v1/datasets/{quote(dataset, safe='')}/releases",
            json=request.model_dump(mode="json"),
        ).json()

    def releases(self, dataset: str, *, limit: int = 50, offset: int = 0) -> Record:
        return self._request(
            "GET",
            f"/v1/datasets/{quote(dataset, safe='')}/releases",
            params={"limit": limit, "offset": offset},
        ).json()

    def release(self, release_id: str) -> Record:
        return self._request("GET", f"/v1/releases/{quote(release_id, safe='')}").json()

    def export_release(
        self,
        release_id: str,
        *,
        include_lineage: bool = False,
        validity_binding_ids: list[str] | None = None,
        plan_ids: list[str] | None = None,
    ) -> Record:
        """Return source claims, optionally including lineage, validity and plan evidence."""
        if type(include_lineage) is not bool:
            raise ValueError("include_lineage must be a boolean")
        params = [("include_lineage", "true")] if include_lineage else []
        if validity_binding_ids is not None:
            if (
                not include_lineage
                or not isinstance(validity_binding_ids, list)
                or not 1 <= len(validity_binding_ids) <= MAX_PORTABLE_VALIDITY_SELECTIONS
                or any(not isinstance(binding_id, str) for binding_id in validity_binding_ids)
            ):
                raise ValueError("Validity selections require lineage and 1..32 binding IDs")
            params.extend(
                ("validity_binding_id", binding_id) for binding_id in validity_binding_ids
            )
        if plan_ids is not None:
            if (
                not include_lineage
                or not isinstance(plan_ids, list)
                or not 1 <= len(plan_ids) <= MAX_PORTABLE_PLAN_SELECTIONS
                or any(not isinstance(plan_id, str) for plan_id in plan_ids)
            ):
                raise ValueError("Plan selections require lineage and 1..16 plan IDs")
            params.extend(("plan_id", plan_id) for plan_id in plan_ids)
        return self._request(
            "GET", f"/v1/releases/{quote(release_id, safe='')}/portable", params=params or None
        ).json()

    def import_release_history(self, dataset: str, request: ReleaseHistoryImport) -> Record:
        """Pin a snapshot and mapped contract claims; retry the identical request after timeout."""
        return self._request(
            "POST",
            f"/v1/datasets/{quote(dataset, safe='')}/release-history/imports",
            json=request.model_dump(mode="json"),
        ).json()

    def export_release_history(self, dataset: str, request: PortableHistoryExport) -> Record:
        """Export explicitly selected releases as bounded, unauthenticated source claims."""
        return self._request(
            "POST",
            f"/v1/datasets/{quote(dataset, safe='')}/release-history/export",
            json=request.model_dump(mode="json"),
        ).json()

    def export_release_history_payload_bundle(
        self, dataset: str, request: PortableHistoryExport
    ) -> bytes:
        """Export selected release and lineage claims with verified canonical NPY payloads."""
        from tensorus.history_payload_bundle import export_history_payload_bundle

        return export_history_payload_bundle(self, dataset, request)

    def inspect_release_history_payload_bundle(self, payload: bytes) -> Record:
        """Verify selected history claims and payloads without writing destination records."""
        from tensorus.history_payload_bundle import inspect_history_payload_bundle

        return inspect_history_payload_bundle(self, payload)

    def copy_release_history_payload_bundle(
        self, payload: bytes, dataset: str, destinations: dict[str, dict[str, str]]
    ) -> Record:
        """Copy selected payloads; local history import and review remain separate."""
        from tensorus.history_payload_bundle import copy_history_payload_bundle

        return copy_history_payload_bundle(self, payload, dataset, destinations)

    def review_release_history_payload_restore(
        self, payload: bytes, request: HistoryPayloadRestoreRequest
    ) -> Record:
        """Review selected local versions, contracts and optional asserted masks without writing."""
        from tensorus.history_payload_restore import review_history_payload_restore

        return review_history_payload_restore(self, payload, request)

    def review_product_payload_selection(
        self,
        payload: bytes,
        request: ProductBundleRestoreRequest,
        *,
        reviewed: dict[str, Any],
        role: str,
    ) -> Record:
        """Review one product role and its explicitly selected validity mask."""
        from tensorus.product_payload_selection import inspect_product_payload_selection

        return inspect_product_payload_selection(
            self, payload, request, reviewed=reviewed, role=role
        )

    def prepare_product_payload_selection(
        self,
        payload: bytes,
        request: ProductBundleRestoreRequest,
        *,
        expected_review_hash: str,
        role: str,
        expected_selection_hash: str,
        confirm_reviewed_selection: bool,
        confirm_unauthenticated_claims: bool,
        confirm_validity_assertions: bool = False,
    ) -> Record:
        """Prepare the reviewed role only; failed responses may follow partial writes."""
        from tensorus.product_payload_selection import prepare_product_payload_selection

        return prepare_product_payload_selection(
            self,
            payload,
            request,
            expected_review_hash=expected_review_hash,
            role=role,
            expected_selection_hash=expected_selection_hash,
            confirm_reviewed_selection=confirm_reviewed_selection,
            confirm_unauthenticated_claims=confirm_unauthenticated_claims,
            confirm_validity_assertions=confirm_validity_assertions,
        )

    def inspect_release_history_payload_restore(
        self,
        payload: bytes,
        request: HistoryPayloadRestoreRequest,
        *,
        expected_review_hash: str,
        reviewed: dict[str, Any] | None = None,
    ) -> Record:
        """Inspect receipt-backed payloads and exact local history membership without writing."""
        from tensorus.history_payload_restore import inspect_history_payload_restore

        return inspect_history_payload_restore(
            self, payload, request, expected_review_hash=expected_review_hash, reviewed=reviewed
        )

    def inspect_release_history_payload_restore_payloads(
        self,
        payload: bytes,
        request: HistoryPayloadRestoreRequest,
        *,
        expected_review_hash: str,
        reviewed: dict[str, Any] | None = None,
    ) -> Record:
        """Observe verified receipts and validity without assessing history or completion."""
        from tensorus.history_payload_restore import inspect_history_payload_restore
        from tensorus.product_bundle import _inspection_review

        request = request.model_copy(deep=True)
        if reviewed is not None:
            reviewed = _inspection_review(reviewed, expected_review_hash, reviewed, "datasets")
        return inspect_history_payload_restore(
            self,
            payload,
            request,
            expected_review_hash=expected_review_hash,
            reviewed=reviewed,
            _payloads_only=True,
        )

    def inspect_release_history_restore_releases(
        self,
        payload: bytes,
        request: HistoryPayloadRestoreRequest,
        *,
        expected_review_hash: str,
        reviewed: dict[str, Any] | None = None,
    ) -> Record:
        """Observe independently verifiable history imports and conflicting retained requests."""
        from tensorus.history_payload_restore import inspect_history_payload_restore
        from tensorus.product_bundle import _inspection_review

        request = request.model_copy(deep=True)
        if reviewed is not None:
            reviewed = _inspection_review(reviewed, expected_review_hash, reviewed, "datasets")
        return inspect_history_payload_restore(
            self,
            payload,
            request,
            expected_review_hash=expected_review_hash,
            reviewed=reviewed,
            _release_observation=True,
        )

    def review_history_restore_selection(
        self,
        payload: bytes,
        request: HistoryPayloadRestoreRequest,
        *,
        expected_review_hash: str,
        release_id: str | UUID,
        reviewed: dict[str, Any] | None = None,
    ) -> Record:
        """Review one already-copied history release without reserving or writing."""
        from tensorus.history_restore_selection import review_history_restore_selection

        return review_history_restore_selection(
            self,
            payload,
            request,
            expected_review_hash=expected_review_hash,
            release_id=release_id,
            reviewed=reviewed,
        )

    def review_history_payload_selection(
        self,
        payload: bytes,
        request: HistoryPayloadRestoreRequest,
        *,
        expected_review_hash: str,
        release_id: str | UUID,
        reviewed: dict[str, Any] | None = None,
    ) -> Record:
        """Review one history release's payload prefixes and requested validity."""
        from tensorus.history_payload_selection import review_history_payload_selection

        return review_history_payload_selection(
            self,
            payload,
            request,
            expected_review_hash=expected_review_hash,
            release_id=release_id,
            reviewed=reviewed,
        )

    def review_independent_history_selection(
        self,
        payload: bytes,
        request: HistoryPayloadRestoreRequest,
        *,
        expected_review_hash: str,
        release_id: str | UUID,
        reviewed: dict[str, Any] | None = None,
    ) -> Record:
        """Review one prepared history import without inspecting unrelated receipts."""
        from tensorus.history_restore_independent import review_independent_history_selection

        return review_independent_history_selection(
            self,
            payload,
            request,
            expected_review_hash=expected_review_hash,
            release_id=release_id,
            reviewed=reviewed,
        )

    def observe_independent_history_release(
        self,
        payload: bytes,
        request: HistoryPayloadRestoreRequest,
        *,
        expected_review_hash: str,
        release_id: str | UUID,
        reviewed: dict[str, Any] | None = None,
    ) -> Record:
        """Observe one selected history import and its names without unrelated receipt reads."""
        from tensorus.history_restore_independent_observation import (
            observe_independent_history_release,
        )

        return observe_independent_history_release(
            self,
            payload,
            request,
            expected_review_hash=expected_review_hash,
            release_id=release_id,
            reviewed=reviewed,
        )

    def apply_independent_history_selection(
        self,
        payload: bytes,
        request: HistoryPayloadRestoreRequest,
        *,
        expected_review_hash: str,
        release_id: str | UUID,
        expected_selection_hash: str,
        confirm_reviewed_selection: bool,
        confirm_unauthenticated_claims: bool,
        confirm_validity_assertions: bool = False,
        reviewed: dict[str, Any] | None = None,
    ) -> Record:
        """Import one independently reviewed history selection."""
        from tensorus.history_restore_independent import apply_independent_history_selection

        return apply_independent_history_selection(
            self,
            payload,
            request,
            expected_review_hash=expected_review_hash,
            release_id=release_id,
            expected_selection_hash=expected_selection_hash,
            confirm_reviewed_selection=confirm_reviewed_selection,
            confirm_unauthenticated_claims=confirm_unauthenticated_claims,
            confirm_validity_assertions=confirm_validity_assertions,
            reviewed=reviewed,
        )

    def prepare_history_payload_selection(
        self,
        payload: bytes,
        request: HistoryPayloadRestoreRequest,
        *,
        expected_review_hash: str,
        release_id: str | UUID,
        expected_selection_hash: str,
        confirm_reviewed_selection: bool,
        confirm_unauthenticated_claims: bool,
        confirm_validity_assertions: bool = False,
        reviewed: dict[str, Any] | None = None,
    ) -> Record:
        """Prepare selected payloads; a failure can leave committed partials."""
        from tensorus.history_payload_selection import prepare_history_payload_selection

        return prepare_history_payload_selection(
            self,
            payload,
            request,
            expected_review_hash=expected_review_hash,
            release_id=release_id,
            expected_selection_hash=expected_selection_hash,
            confirm_reviewed_selection=confirm_reviewed_selection,
            confirm_unauthenticated_claims=confirm_unauthenticated_claims,
            confirm_validity_assertions=confirm_validity_assertions,
            reviewed=reviewed,
        )

    def apply_history_restore_selection(
        self,
        payload: bytes,
        request: HistoryPayloadRestoreRequest,
        *,
        expected_review_hash: str,
        release_id: str | UUID,
        expected_selection_hash: str,
        confirm_reviewed_selection: bool,
        confirm_unauthenticated_claims: bool,
        confirm_validity_assertions: bool = False,
        reviewed: dict[str, Any] | None = None,
    ) -> Record:
        """Import one exact selection; a failure can leave the import committed."""
        from tensorus.history_restore_selection import apply_history_restore_selection

        return apply_history_restore_selection(
            self,
            payload,
            request,
            expected_review_hash=expected_review_hash,
            release_id=release_id,
            expected_selection_hash=expected_selection_hash,
            confirm_reviewed_selection=confirm_reviewed_selection,
            confirm_unauthenticated_claims=confirm_unauthenticated_claims,
            confirm_validity_assertions=confirm_validity_assertions,
            reviewed=reviewed,
        )

    def inspect_release_history_payload_restore_heads(
        self,
        payload: bytes,
        request: HistoryPayloadRestoreRequest,
        *,
        expected_review_hash: str,
        reviewed: dict[str, Any] | None = None,
    ) -> Record:
        """Observe receipt-backed history progress and destination heads without writing."""
        from tensorus.history_payload_restore import inspect_history_payload_restore

        return inspect_history_payload_restore(
            self,
            payload,
            request,
            expected_review_hash=expected_review_hash,
            reviewed=reviewed,
            _include_head_preflight=True,
        )

    def inspect_release_history_payload_restore_names(
        self,
        payload: bytes,
        request: HistoryPayloadRestoreRequest,
        *,
        expected_review_hash: str,
        reviewed: dict[str, Any] | None = None,
    ) -> Record:
        """Observe destination names, receipts and heads without reserving or writing."""
        from tensorus.history_payload_restore import inspect_history_payload_restore

        return inspect_history_payload_restore(
            self,
            payload,
            request,
            expected_review_hash=expected_review_hash,
            reviewed=reviewed,
            _include_name_preflight=True,
        )

    def inspect_release_history_payload_restore_occupancy(
        self,
        payload: bytes,
        request: HistoryPayloadRestoreRequest,
        *,
        expected_review_hash: str,
        reviewed: dict[str, Any] | None = None,
    ) -> Record:
        """Check reviewed names without inspecting receipt-backed restore progress."""
        from tensorus.bundle_restore_names import inspect_reviewed_occupancy

        return inspect_reviewed_occupancy(
            self,
            payload,
            request,
            kind="history",
            expected_review_hash=expected_review_hash,
            reviewed=reviewed,
        )

    def restore_release_history_payload_bundle(
        self,
        payload: bytes,
        request: HistoryPayloadRestoreRequest,
        *,
        expected_review_hash: str,
        confirm_unauthenticated_claims: bool,
        confirm_validity_assertions: bool = False,
        reject_observed_name_conflicts: bool = False,
        inspect_on_failure: bool = False,
        reviewed: dict[str, Any] | None = None,
        failure_inspection: FailureInspection = "progress",
    ) -> Record:
        """Restore without approval or trust; optionally inspect a failed apply."""
        from tensorus.bundle_restore import (
            _apply_with_inspection,
            _failure_inspection,
            _prepare_inspection_review,
        )
        from tensorus.history_payload_restore import restore_history_payload_bundle

        selected_inspection = _failure_inspection(inspect_on_failure, failure_inspection)
        inspection_review = _prepare_inspection_review(
            inspect_on_failure, reviewed, expected_review_hash, "datasets"
        )
        if inspect_on_failure:
            request = request.model_copy(deep=True)

        def apply() -> Record:
            return restore_history_payload_bundle(
                self,
                payload,
                request,
                expected_review_hash=expected_review_hash,
                confirm_unauthenticated_claims=confirm_unauthenticated_claims,
                confirm_validity_assertions=confirm_validity_assertions,
                reject_observed_name_conflicts=reject_observed_name_conflicts,
            )

        if not inspect_on_failure:
            return apply()

        inspect = {
            "progress": self.inspect_release_history_payload_restore,
            "heads": self.inspect_release_history_payload_restore_heads,
            "names": self.inspect_release_history_payload_restore_names,
            "occupancy": self.inspect_release_history_payload_restore_occupancy,
            "payloads": self.inspect_release_history_payload_restore_payloads,
            "releases": self.inspect_release_history_restore_releases,
        }[selected_inspection]
        return _apply_with_inspection(
            apply,
            lambda: inspect(
                payload,
                request,
                expected_review_hash=expected_review_hash,
                reviewed=inspection_review,
            ),
            enabled=True,
            inspection_kind=selected_inspection,
        )

    def inspect_imported_history(
        self, dataset: str, request: ImportedHistoryInspectionRequest
    ) -> Record:
        """Inspect selected imported claims without inferring an atomic or complete history."""
        return self._request(
            "POST",
            f"/v1/datasets/{quote(dataset, safe='')}/release-history/inspect",
            json=request.model_dump(mode="json"),
        ).json()

    def lookup_imported_history_index(
        self, dataset: str, request: ImportedHistoryIndexLookupRequest
    ) -> Record:
        """Find indexed matches for a retained manifest, without claiming complete history."""
        return self._request(
            "POST",
            f"/v1/datasets/{quote(dataset, safe='')}/release-history/index/lookup",
            json=request.model_dump(mode="json"),
        ).json()

    def scan_imported_history(self, dataset: str, request: ImportedHistoryScanRequest) -> Record:
        """Find matching manifests in one fully authorized page of retained imports."""
        return self._request(
            "POST",
            f"/v1/datasets/{quote(dataset, safe='')}/release-history/scan",
            json=request.model_dump(mode="json"),
        ).json()

    def scan_readable_imported_history(
        self, dataset: str, request: ImportedHistoryScanRequest
    ) -> Record:
        """Find readable matches while explicitly omitting inaccessible page candidates."""
        return self._request(
            "POST",
            f"/v1/datasets/{quote(dataset, safe='')}/release-history/scan-readable",
            json=request.model_dump(mode="json"),
        ).json()

    def backfill_imported_history_index(
        self, dataset: str, request: ImportedHistoryIndexBackfillRequest
    ) -> Record:
        """Verify and index one bounded page of retained imports."""
        return self._request(
            "POST",
            f"/v1/datasets/{quote(dataset, safe='')}/release-history/index/backfill",
            json=request.model_dump(mode="json"),
        ).json()

    def review_imported_history_index(
        self, dataset: str, request: ImportedHistoryIndexReviewRequest
    ) -> Record:
        """Review one page of intended index replacements as an unscoped administrator."""
        return self._request(
            "POST",
            f"/v1/datasets/{quote(dataset, safe='')}/release-history/index/review",
            json=request.model_dump(mode="json"),
        ).json()

    def diagnose_imported_history_index(
        self, dataset: str, request: ImportedHistoryIndexDiagnosticRequest
    ) -> Record:
        """Classify one index page, redacting foreign or inconsistent ownership."""
        return self._request(
            "POST",
            f"/v1/datasets/{quote(dataset, safe='')}/release-history/index/diagnose",
            json=request.model_dump(mode="json"),
        ).json()

    def reconcile_imported_history_index(
        self, dataset: str, request: ImportedHistoryIndexReconcileRequest
    ) -> Record:
        """Apply reviewed replacements only while page membership and statuses still match."""
        return self._request(
            "POST",
            f"/v1/datasets/{quote(dataset, safe='')}/release-history/index/reconcile",
            json=request.model_dump(mode="json"),
        ).json()

    def import_release_history_bundle(
        self, dataset: str, request: ReleaseHistoryBundleImport
    ) -> Record:
        """Atomically reconstruct selected snapshots; repeat the identical request after timeout."""
        return self._request(
            "POST",
            f"/v1/datasets/{quote(dataset, safe='')}/release-history/bundle-imports",
            json=request.model_dump(mode="json"),
        ).json()

    def imported_releases(self, dataset: str, *, limit: int = 50, offset: int = 0) -> Record:
        return self._request(
            "GET",
            f"/v1/datasets/{quote(dataset, safe='')}/release-history/imports",
            params={"limit": limit, "offset": offset},
        ).json()

    def _require_imported_history_name_lookup(self, *, _credential: str | None = None) -> None:
        releases = (
            self.capabilities()
            if _credential is None
            else self.capabilities(_credential=_credential)
        ).get("release_contracts")
        if not isinstance(releases, dict) or "portable_history_name_lookup" not in releases:
            raise TensorusError(
                "Destination server does not support imported history name lookup",
                code="unsupported_capability",
                status=409,
            )

    def _read_imported_release_by_name(
        self, dataset: str, name: str, *, _credential: str | None = None
    ) -> Record:
        data = self._bounded_report(
            "GET",
            f"/v1/datasets/{quote(dataset, safe='')}/release-history/imports/by-name/"
            f"{quote(name, safe='')}",
            MAX_PORTABLE_BYTES + 96 * 1024,
            "Imported release",
            headers={"Authorization": _credential} if _credential is not None else None,
        )
        try:
            raw = decode_unique_json(data)
            result = ImportedRelease.model_validate_json(data, strict=True)
            if (
                canonical_json(raw) != canonical_json(result.model_dump(mode="json"))
                or result.dataset != dataset
                or result.name != name
                or result.request.name != name
                or result.request_hash != canonical_hash(result.request.model_dump(mode="json"))
                or result.source_hash != result.request.source.content_hash
                or str(UUID(result.created_by)) != result.created_by
            ):
                raise ValueError("Imported history contradicts the requested identity")
        except (UnicodeDecodeError, ValueError, RecursionError, TensorusError) as exc:
            raise TensorusError(
                "Destination returned invalid imported history evidence", code="invalid_response"
            ) from exc
        return result.model_dump(mode="json")

    def imported_release_by_name(self, dataset: str, name: str) -> Record:
        """Find one fully authorized imported history record by its unique dataset name."""
        from tensorus.product_bundle import _capture_write_credential, _checked_write_identity

        credential = _capture_write_credential(self)
        self._require_imported_history_name_lookup(_credential=credential)
        identity = _checked_write_identity(self, credential)
        result = self._read_imported_release_by_name(dataset, name, _credential=credential)
        _checked_write_identity(self, credential, identity)
        if result["tenant_id"] != str(identity.id):
            raise Forbidden("The authenticated identity changed during imported history lookup")
        return result

    def imported_release(self, import_id: str) -> Record:
        return self._request(
            "GET", f"/v1/release-history/imports/{quote(import_id, safe='')}"
        ).json()

    def approve_release(self, release_id: str, report_hash: str) -> Record:
        """Authorize a passing report with a different unscoped writer credential."""
        return self._request(
            "POST",
            f"/v1/releases/{quote(release_id, safe='')}/approve",
            json={"report_hash": report_hash},
        ).json()

    def plan_proposer_authority(self) -> PlanProposerAuthority:
        """Bind this catalog credential to an opt-in plan proposer policy."""
        identity, capabilities = self._checked_capabilities()
        if "plan_proposer_authority" not in capabilities:
            raise TensorusError(
                "The server does not advertise plan proposer authority",
                code="unsupported_feature",
            )
        return PlanProposerAuthority(
            tenant_id=identity.id, key_id=identity.key_id, role=identity.role
        )

    def propose(self, definition: PlanDefinition, *, idempotency_key: str | None = None) -> Record:
        """Propose a plan; reuse the key and definition after an uncertain result."""
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key is not None else {}
        return self._request(
            "POST", "/v1/plans", json=definition.model_dump(mode="json"), headers=headers
        ).json()

    def delegate(self, definition: PlanDefinition, plan_hash: str, *, request_id: str) -> Record:
        """Authorize exactly the reviewed plan; reuse request_id for retries, not new runs."""
        return self._request(
            "POST",
            "/v1/agent/delegate",
            json={
                "definition": definition.model_dump(mode="json"),
                "plan_hash": plan_hash,
                "request_id": request_id,
                "authorize": True,
            },
        ).json()

    def explain(self, definition: PlanDefinition) -> Record:
        return self._request(
            "POST", "/v1/plans/explain", json=definition.model_dump(mode="json")
        ).json()

    def explain_alignment(self, request: AxisAlignmentRequest) -> Record:
        """Preview raw-value alignment; v2 requires recorded-source access and exact references."""
        return self._request("POST", "/v1/agent/align", json=request.model_dump(mode="json")).json()

    def data_products(
        self, *, dataset: str | None = None, limit: int = 50, offset: int = 0
    ) -> Record:
        params: Record = {"limit": limit, "offset": offset}
        if dataset is not None:
            params["dataset"] = dataset
        return self._request("GET", "/v1/data-products", params=params).json()

    def data_product(self, name: str, *, revision: int | None = None) -> Record:
        params = {"revision": revision} if revision is not None else {}
        response = self._request("GET", f"/v1/data-products/{quote(name, safe='')}", params=params)
        result = self._product_response(response.content)
        if result.name != name or (revision is not None and result.revision != revision):
            raise TensorusError("The server returned a different data-product revision")
        return result.model_dump(mode="json")

    def data_product_revisions(self, name: str, *, limit: int = 50, offset: int = 0) -> Record:
        return self._request(
            "GET",
            f"/v1/data-products/{quote(name, safe='')}/revisions",
            params={"limit": limit, "offset": offset},
        ).json()

    def _revision_retirement_identity(self) -> Identity:
        identity, capabilities = self._checked_capabilities()
        if identity.role != Role.admin or identity.scope is not None:
            raise Forbidden("Product-revision retirement requires an unscoped administrator")
        if "product_revision_retirement" not in capabilities:
            raise TensorusError(
                "The server does not advertise product-revision retirement",
                code="unsupported_feature",
            )
        return identity

    def _product_retirement_identity(self) -> Identity:
        identity, capabilities = self._checked_capabilities()
        if identity.role != Role.admin or identity.scope is not None:
            raise Forbidden("Product retirement requires an unscoped administrator")
        if "product_retirement" not in capabilities:
            raise TensorusError(
                "The server does not advertise complete product retirement",
                code="unsupported_feature",
            )
        return identity

    def _quality_release_retirement_identity(self) -> Identity:
        identity, capabilities = self._checked_capabilities()
        if identity.role != Role.admin or identity.scope is not None:
            raise Forbidden("Quality-release retirement requires an unscoped administrator")
        if "quality_release_retirement" not in capabilities:
            raise TensorusError(
                "The server does not advertise quality-release retirement",
                code="unsupported_feature",
            )
        return identity

    def review_quality_release_retirement(
        self, release_id: str, request: QualityReleaseRetirementReviewRequest
    ) -> Record:
        """Review the exact release, retained dependencies and snapshot holds."""
        identity = self._quality_release_retirement_identity()
        requested_id = UUID(release_id)
        encoded = self._bounded_report(
            "POST",
            f"/v1/releases/{quote(str(requested_id), safe='')}/retirement-review",
            MAX_QUALITY_RELEASE_RETIREMENT_RECEIPT_BYTES,
            "Quality-release retirement review",
            json=request.model_dump(mode="json"),
        )
        result = self._checked_exact_response(
            encoded, QualityReleaseRetirementReviewResult, "quality-release retirement review"
        )
        review = result.review
        if (
            review.tenant_id != identity.id
            or review.release_id != requested_id
            or review.report_hash != request.report_hash
            or review.reason != request.reason
            or review.reviewed_by != str(identity.key_id)
        ):
            raise TensorusError("The server returned a different quality-release review")
        if self._checked_identity() != identity:
            raise Forbidden("The workspace identity changed during retirement review")
        return result.model_dump(mode="json")

    def retire_quality_release(
        self, release_id: str, request: QualityReleaseRetirementRequest
    ) -> Record:
        """Retire the reviewed release; retry identical inputs after an uncertain result."""
        identity = self._quality_release_retirement_identity()
        requested_id = UUID(release_id)
        encoded = self._lifecycle_post(
            f"/v1/releases/{quote(str(requested_id), safe='')}/retire",
            request.model_dump(mode="json"),
            MAX_QUALITY_RELEASE_RETIREMENT_BYTES,
            "Quality-release retirement",
        )
        result = self._checked_exact_response(
            encoded, QualityReleaseRetirement, "quality-release retirement", mutation=True
        )
        review = result.receipt.review
        if (
            review.tenant_id != identity.id
            or review.release_id != requested_id
            or review.report_hash != request.report_hash
            or review.reason != request.reason
            or result.receipt.review_hash != request.review_hash
            or result.receipt.retired_by != str(identity.key_id)
        ):
            raise TensorusError(
                "The server returned a different quality-release retirement; it may have "
                "committed, so inspect before retrying",
                code="invalid_response",
            )
        try:
            if self._checked_identity() != identity:
                raise Forbidden("The workspace identity changed")
        except TensorusError as exc:
            raise TensorusError(
                f"{exc}; retirement may have committed, so inspect before retrying",
                code=exc.code,
                status=exc.status,
            ) from exc
        return result.model_dump(mode="json")

    def quality_release_retirement(self, release_id: str) -> Record:
        """Inspect the retained source evidence and retirement receipt."""
        identity = self._quality_release_retirement_identity()
        requested_id = UUID(release_id)
        encoded = self._bounded_report(
            "GET",
            f"/v1/quality-release-retirements/{quote(str(requested_id), safe='')}",
            MAX_QUALITY_RELEASE_RETIREMENT_BYTES,
            "Quality-release retirement",
        )
        result = self._checked_exact_response(
            encoded, QualityReleaseRetirement, "quality-release retirement"
        )
        if (
            result.source.release.tenant_id != identity.id
            or result.source.release.id != requested_id
        ):
            raise TensorusError("The server returned a different quality-release retirement")
        if self._checked_identity() != identity:
            raise Forbidden("The workspace identity changed during retirement inspection")
        return result.model_dump(mode="json")

    def quality_release_retirements(self, *, limit: int = 10, offset: int = 0) -> Record:
        """List retained quality-release retirement receipts in this workspace."""
        identity = self._quality_release_retirement_identity()
        encoded = self._bounded_report(
            "GET",
            "/v1/quality-release-retirements",
            MAX_QUALITY_RELEASE_RETIREMENT_PAGE * MAX_QUALITY_RELEASE_RETIREMENT_RECEIPT_BYTES
            + 8192,
            "Quality-release retirement page",
            params={"limit": limit, "offset": offset},
        )
        page = self._checked_exact_response(
            encoded, Page[QualityReleaseRetirementSummary], "quality-release retirement page"
        )
        if (
            page.limit != limit
            or page.offset != offset
            or not 0 <= page.total <= MAX_QUALITY_RELEASE_RETIREMENTS_PER_TENANT
            or len(page.items) != min(limit, max(page.total - offset, 0))
            or any(item.receipt.review.tenant_id != identity.id for item in page.items)
            or len({item.receipt.review.release_id for item in page.items}) != len(page.items)
        ):
            raise TensorusError("The server returned a different quality-release retirement page")
        if self._checked_identity() != identity:
            raise Forbidden("The workspace identity changed during retirement inspection")
        return page.model_dump(mode="json")

    @staticmethod
    def _product_retirement_path(name: str) -> str:
        return f"/v1/data-products/{quote(name, safe='')}/retirement"

    def review_product_retirement(
        self, name: str, request: ProductRetirementReviewRequest
    ) -> Record:
        """Review the exact product head and retained blockers without changing anything."""
        identity = self._product_retirement_identity()
        encoded = self._bounded_report(
            "POST",
            f"{self._product_retirement_path(name)}-review",
            MAX_PRODUCT_RETIREMENT_RECEIPT_BYTES,
            "Product retirement review",
            json=request.model_dump(mode="json"),
        )
        result = self._checked_exact_response(
            encoded, ProductRetirementReviewResult, "product retirement review"
        )
        review = result.review
        if (
            review.tenant_id != identity.id
            or review.product_name != name
            or review.descriptor_hash != request.descriptor_hash
            or review.reason != request.reason
            or review.reviewed_by != str(identity.key_id)
        ):
            raise TensorusError("The server returned a different product retirement review")
        if self._checked_identity() != identity:
            raise Forbidden("The workspace identity changed during retirement review")
        return result.model_dump(mode="json")

    def retire_product(self, name: str, request: ProductRetirementRequest) -> Record:
        """Retire the reviewed product; retry identical inputs after an uncertain result."""
        identity = self._product_retirement_identity()
        encoded = self._lifecycle_post(
            self._product_retirement_path(name),
            request.model_dump(mode="json"),
            MAX_PRODUCT_RETIREMENT_BYTES,
            "Product retirement",
        )
        result = self._checked_exact_response(
            encoded, ProductRetirement, "product retirement", mutation=True
        )
        review = result.receipt.review
        if (
            review.tenant_id != identity.id
            or review.product_name != name
            or review.descriptor_hash != request.descriptor_hash
            or review.reason != request.reason
            or result.receipt.review_hash != request.review_hash
            or result.receipt.retired_by != str(identity.key_id)
        ):
            raise TensorusError(
                "The server returned a different product retirement; it may have committed, "
                "so inspect before retrying",
                code="invalid_response",
            )
        try:
            if self._checked_identity() != identity:
                raise Forbidden("The workspace identity changed")
        except TensorusError as exc:
            raise TensorusError(
                f"{exc}; retirement may have committed, so inspect before retrying",
                code=exc.code,
                status=exc.status,
            ) from exc
        return result.model_dump(mode="json")

    def product_retirement(self, name: str) -> Record:
        """Inspect the retained product descriptor and retirement receipt."""
        identity = self._product_retirement_identity()
        encoded = self._bounded_report(
            "GET",
            self._product_retirement_path(name),
            MAX_PRODUCT_RETIREMENT_BYTES,
            "Product retirement",
        )
        result = self._checked_exact_response(encoded, ProductRetirement, "product retirement")
        if result.product.tenant_id != identity.id or result.product.name != name:
            raise TensorusError("The server returned a different product retirement")
        if self._checked_identity() != identity:
            raise Forbidden("The workspace identity changed during retirement inspection")
        return result.model_dump(mode="json")

    def product_retirements(self, *, limit: int = 10, offset: int = 0) -> Record:
        """List retained product retirement receipts in the current workspace."""
        identity = self._product_retirement_identity()
        encoded = self._bounded_report(
            "GET",
            "/v1/data-product-retirements",
            MAX_PRODUCT_RETIREMENT_PAGE * MAX_PRODUCT_RETIREMENT_RECEIPT_BYTES + 8192,
            "Product retirement page",
            params={"limit": limit, "offset": offset},
        )
        page = self._checked_exact_response(
            encoded, Page[ProductRetirementSummary], "product retirement page"
        )
        if (
            page.limit != limit
            or page.offset != offset
            or not 0 <= page.total <= 128
            or len(page.items) > limit
            or len(page.items) > page.total
            or any(item.receipt.review.tenant_id != identity.id for item in page.items)
            or len({item.receipt.review.product_id for item in page.items}) != len(page.items)
            or len({item.receipt.review.product_name for item in page.items}) != len(page.items)
        ):
            raise TensorusError("The server returned a different product retirement page")
        if self._checked_identity() != identity:
            raise Forbidden("The workspace identity changed during retirement inspection")
        return page.model_dump(mode="json")

    @staticmethod
    def _revision_retirement_path(name: str, revision: int) -> str:
        if type(revision) is not int or not 1 <= revision <= 2**31 - 1:
            raise ValueError("A valid product revision number is required")
        return f"/v1/data-products/{quote(name, safe='')}/revisions/{revision}/retirement"

    def _revision_holds_identity(self) -> Identity:
        identity, capabilities = self._checked_capabilities()
        if identity.role != Role.admin or identity.scope is not None:
            raise Forbidden("Product-revision holds require an unscoped administrator")
        if "product_revision_holds" not in capabilities:
            raise TensorusError(
                "The server does not advertise product-revision holds",
                code="unsupported_feature",
            )
        return identity

    @staticmethod
    def _revision_holds_path(name: str, revision: int) -> str:
        if type(revision) is not int or not 1 <= revision <= 2**31 - 1:
            raise ValueError("A valid product revision number is required")
        return f"/v1/data-products/{quote(name, safe='')}/revisions/{revision}/holds"

    def product_revision_holds(self, name: str, revision: int) -> Record:
        """Inspect the retained, audited holds for one exact product revision."""
        identity = self._revision_holds_identity()
        encoded = self._bounded_report(
            "GET",
            self._revision_holds_path(name, revision),
            MAX_HOLDS_PER_RESOURCE * 16_384 + 4096,
            "Product-revision hold history",
        )
        result = self._checked_exact_response(
            encoded, ProductRevisionHoldHistory, "product-revision hold history"
        )
        if (
            result.tenant_id != identity.id
            or result.product_name != name
            or result.revision != revision
        ):
            raise TensorusError("The server returned a different product-revision hold history")
        if self._checked_identity() != identity:
            raise Forbidden("The workspace identity changed during hold inspection")
        return result.model_dump(mode="json")

    def create_product_revision_hold(
        self, name: str, revision: int, request: ProductRevisionHoldCreate
    ) -> Record:
        """Place a revision hold; retry the identical request ID and reason after uncertainty."""
        identity = self._revision_holds_identity()
        history = self.product_revision_holds(name, revision)
        if self._checked_identity() != identity:
            raise Forbidden("The workspace identity changed before placing the hold")
        encoded = self._lifecycle_post(
            self._revision_holds_path(name, revision),
            request.model_dump(mode="json"),
            16_384,
            "Product-revision hold",
        )
        result = self._checked_exact_response(
            encoded, ProductRevisionHold, "product-revision hold", mutation=True
        )
        if (
            result.tenant_id != identity.id
            or str(result.product_id) != history["product_id"]
            or result.revision != revision
            or result.id != request.request_id
            or result.reason != request.reason
            or result.created_by != str(identity.key_id)
        ):
            raise TensorusError(
                "The server returned a different product-revision hold; it may have committed, "
                "so inspect before retrying",
                code="invalid_response",
            )
        try:
            if self._checked_identity() != identity:
                raise Forbidden("The workspace identity changed")
        except TensorusError as exc:
            raise TensorusError(
                f"{exc}; the hold may have committed, so inspect before retrying",
                code=exc.code,
                status=exc.status,
            ) from exc
        return result.model_dump(mode="json")

    def release_product_revision_hold(
        self, name: str, revision: int, hold_id: str, request: RetentionHoldRelease
    ) -> Record:
        """Release one revision hold with an auditable reason and exact retry identity."""
        identity = self._revision_holds_identity()
        history = self.product_revision_holds(name, revision)
        if self._checked_identity() != identity:
            raise Forbidden("The workspace identity changed before releasing the hold")
        encoded = self._lifecycle_post(
            f"{self._revision_holds_path(name, revision)}/{quote(hold_id, safe='')}/release",
            request.model_dump(mode="json"),
            16_384,
            "Product-revision hold release",
        )
        result = self._checked_exact_response(
            encoded, ProductRevisionHold, "product-revision hold release", mutation=True
        )
        if (
            result.tenant_id != identity.id
            or str(result.product_id) != history["product_id"]
            or result.revision != revision
            or str(result.id) != str(UUID(hold_id))
            or result.state != "released"
            or result.release_request_id != request.request_id
            or result.release_reason != request.reason
            or result.released_by != str(identity.key_id)
        ):
            raise TensorusError(
                "The server returned a different product-revision hold release; it may have "
                "committed, so inspect before retrying",
                code="invalid_response",
            )
        try:
            if self._checked_identity() != identity:
                raise Forbidden("The workspace identity changed")
        except TensorusError as exc:
            raise TensorusError(
                f"{exc}; release may have committed, so inspect before retrying",
                code=exc.code,
                status=exc.status,
            ) from exc
        return result.model_dump(mode="json")

    def review_product_revision_retirement(
        self, name: str, revision: int, request: ProductRevisionRetirementReviewRequest
    ) -> Record:
        """Review an exact descriptor and current blockers without changing it."""
        identity = self._revision_retirement_identity()
        encoded = self._bounded_report(
            "POST",
            f"{self._revision_retirement_path(name, revision)}-review",
            MAX_REVISION_RETIREMENT_RECEIPT_BYTES,
            "Product-revision retirement review",
            json=request.model_dump(mode="json"),
        )
        result = self._checked_exact_response(
            encoded, ProductRevisionRetirementReviewResult, "product-revision retirement review"
        )
        review = result.review
        if (
            review.tenant_id != identity.id
            or review.product_name != name
            or review.product_revision != revision
            or review.descriptor_hash != request.descriptor_hash
            or review.reason != request.reason
            or review.reviewed_by != str(identity.key_id)
        ):
            raise TensorusError("The server returned a different product-revision review")
        if self._checked_identity() != identity:
            raise Forbidden("The workspace identity changed during retirement review")
        return result.model_dump(mode="json")

    def retire_product_revision(
        self, name: str, revision: int, request: ProductRevisionRetirementRequest
    ) -> Record:
        """Tombstone an exact reviewed historical revision; retry the identical request."""
        identity = self._revision_retirement_identity()
        path = self._revision_retirement_path(name, revision)
        encoded = self._lifecycle_post(
            path,
            request.model_dump(mode="json"),
            MAX_REVISION_RETIREMENT_BYTES,
            "Revision retirement",
        )
        result = self._checked_exact_response(
            encoded, ProductRevisionRetirement, "product-revision retirement", mutation=True
        )
        review = result.receipt.review
        if (
            review.tenant_id != identity.id
            or review.product_name != name
            or review.product_revision != revision
            or review.descriptor_hash != request.descriptor_hash
            or review.reason != request.reason
            or result.receipt.review_hash != request.review_hash
            or result.receipt.retired_by != str(identity.key_id)
        ):
            raise TensorusError(
                "The server returned a different revision retirement; it may have committed, "
                "so inspect before retrying",
                code="invalid_response",
            )
        try:
            if self._checked_identity() != identity:
                raise Forbidden("The workspace identity changed")
        except TensorusError as exc:
            raise TensorusError(
                f"{exc}; retirement may have committed, so inspect before retrying",
                code=exc.code,
                status=exc.status,
            ) from exc
        return result.model_dump(mode="json")

    def product_revision_retirement(self, name: str, revision: int) -> Record:
        identity = self._revision_retirement_identity()
        encoded = self._bounded_report(
            "GET",
            self._revision_retirement_path(name, revision),
            MAX_REVISION_RETIREMENT_BYTES,
            "Product-revision retirement",
        )
        result = self._checked_exact_response(
            encoded, ProductRevisionRetirement, "product-revision retirement"
        )
        if (
            result.product.tenant_id != identity.id
            or result.product.name != name
            or result.product.revision != revision
        ):
            raise TensorusError("The server returned a different revision retirement")
        if self._checked_identity() != identity:
            raise Forbidden("The workspace identity changed during retirement inspection")
        return result.model_dump(mode="json")

    def product_revision_retirements(
        self, name: str, *, limit: int = 10, offset: int = 0
    ) -> Record:
        identity = self._revision_retirement_identity()
        encoded = self._bounded_report(
            "GET",
            f"/v1/data-products/{quote(name, safe='')}/retirements",
            MAX_REVISION_RETIREMENT_PAGE * MAX_REVISION_RETIREMENT_RECEIPT_BYTES + 8192,
            "Product-revision retirement page",
            params={"limit": limit, "offset": offset},
        )
        page = self._checked_exact_response(
            encoded, Page[ProductRevisionRetirementSummary], "product-revision retirement page"
        )
        if (
            page.limit != limit
            or page.offset != offset
            or not 0 <= page.total <= 64
            or len(page.items) > limit
            or len(page.items) > page.total
            or any(
                item.receipt.review.tenant_id != identity.id
                or item.receipt.review.product_name != name
                for item in page.items
            )
            or len({item.receipt.review.product_revision for item in page.items}) != len(page.items)
        ):
            raise TensorusError("The server returned a different revision retirement page")
        if self._checked_identity() != identity:
            raise Forbidden("The workspace identity changed during retirement inspection")
        return page.model_dump(mode="json")

    def save_data_product(
        self, name: str, request: DataProductWrite, *, _credential: str | None = None
    ) -> Record:
        """Save explicit roles; retry the identical expected revision after uncertainty."""
        response = self._request(
            "PUT",
            f"/v1/data-products/{quote(name, safe='')}",
            headers={"Authorization": _credential} if _credential is not None else None,
            json=request.model_dump(mode="json"),
        )
        result = self._product_response(response.content)
        if (
            result.name != name
            or result.revision != request.expected_revision + 1
            or result.request != request
        ):
            raise TensorusError("The server returned a different data-product request")
        return result.model_dump(mode="json")

    @staticmethod
    def _product_response(data: bytes) -> DataProductRevision:
        if len(data) > 2 * MAX_PRODUCT_BYTES:
            raise LimitExceeded("Data-product response exceeds the SDK byte limit")
        try:
            result = DataProductRevision.model_validate(decode_unique_json(data))
            _validate_source_product(result)
            return result
        except (UnicodeDecodeError, ValueError, RecursionError, TensorusError) as exc:
            raise TensorusError("The server returned invalid data-product evidence") from exc

    def prepare_product_release(self, name: str, request: ProductReleaseRequest) -> Record:
        """Verify role payloads and selected reference or plan claims; retry the same request."""
        response = self._request(
            "POST",
            f"/v1/data-products/{quote(name, safe='')}/releases",
            json=request.model_dump(mode="json"),
        )
        result = self._product_release_response(response.content)
        if result.product_name != name or result.request != request:
            raise TensorusError("The server returned a different product-release request")
        return result.model_dump(mode="json")

    def product_releases(self, name: str, *, limit: int = 50, offset: int = 0) -> Record:
        return self._request(
            "GET",
            f"/v1/data-products/{quote(name, safe='')}/releases",
            params={"limit": limit, "offset": offset},
        ).json()

    def product_release(self, release_id: str) -> Record:
        response = self._request("GET", f"/v1/product-releases/{quote(release_id, safe='')}")
        result = self._product_release_response(response.content)
        if str(result.id) != str(UUID(release_id)):
            raise TensorusError("The server returned a different product release")
        return result.model_dump(mode="json")

    def approve_product_release(self, release_id: str, report_hash: str) -> Record:
        """Approve the exact report; selected reference evaluations must still be active."""
        response = self._request(
            "POST",
            f"/v1/product-releases/{quote(release_id, safe='')}/approve",
            json={"report_hash": report_hash},
        )
        result = self._product_release_response(response.content)
        if (
            str(result.id) != str(UUID(release_id))
            or result.report_hash != report_hash
            or result.status != "approved"
        ):
            raise TensorusError("The server returned a different product-release approval")
        return result.model_dump(mode="json")

    def review_product_release_retirement(
        self, release_id: str, request: ProductReleaseRetirementReviewRequest
    ) -> Record:
        """Review retained references, holds and the exact release before retirement."""
        encoded = self._bounded_report(
            "POST",
            f"/v1/product-releases/{quote(release_id, safe='')}/retirement-review",
            MAX_PRODUCT_RELEASE_RETIREMENT_RECEIPT_BYTES,
            "Product-release retirement review",
            json=request.model_dump(mode="json"),
        )
        result = self._checked_exact_response(
            encoded, ProductReleaseRetirementReviewResult, "product-release retirement review"
        )
        if (
            str(result.review.release_id) != str(UUID(release_id))
            or result.review.report_hash != request.report_hash
            or result.review.reason != request.reason
        ):
            raise TensorusError("The server returned a different product-release review")
        return result.model_dump(mode="json")

    def retire_product_release(
        self, release_id: str, request: ProductReleaseRetirementRequest
    ) -> Record:
        """Retire the exact reviewed release; retry the identical request after uncertainty."""
        encoded = self._lifecycle_post(
            f"/v1/product-releases/{quote(release_id, safe='')}/retire",
            request.model_dump(mode="json"),
            MAX_RETIRED_PRODUCT_RELEASE_BYTES + MAX_PRODUCT_RELEASE_RETIREMENT_RECEIPT_BYTES + 4096,
            "Product-release retirement",
        )
        result = self._checked_exact_response(
            encoded, ProductReleaseRetirement, "product-release retirement", mutation=True
        )
        if (
            str(result.release.id) != str(UUID(release_id))
            or result.receipt.review_hash != request.review_hash
            or result.receipt.review.report_hash != request.report_hash
            or result.receipt.review.reason != request.reason
        ):
            raise TensorusError(
                "The server returned a different product-release retirement; it may have "
                "committed, so inspect before retrying",
                code="invalid_response",
            )
        return result.model_dump(mode="json")

    def product_release_retirement(self, release_id: str) -> Record:
        encoded = self._bounded_report(
            "GET",
            f"/v1/product-release-retirements/{quote(release_id, safe='')}",
            MAX_RETIRED_PRODUCT_RELEASE_BYTES + MAX_PRODUCT_RELEASE_RETIREMENT_RECEIPT_BYTES + 4096,
            "Product-release retirement",
        )
        result = self._checked_exact_response(
            encoded, ProductReleaseRetirement, "product-release retirement"
        )
        if str(result.release.id) != str(UUID(release_id)):
            raise TensorusError("The server returned a different product-release retirement")
        return result.model_dump(mode="json")

    def product_release_retirements(self, *, limit: int = 10, offset: int = 0) -> Record:
        encoded = self._bounded_report(
            "GET",
            "/v1/product-release-retirements",
            MAX_PRODUCT_RELEASE_RETIREMENT_PAGE * MAX_PRODUCT_RELEASE_RETIREMENT_RECEIPT_BYTES
            + 8192,
            "Product-release retirement page",
            params={"limit": limit, "offset": offset},
        )
        page = self._checked_exact_response(
            encoded, Page[ProductReleaseRetirementSummary], "product-release retirement page"
        )
        if (
            page.limit != limit
            or page.offset != offset
            or page.total < 0
            or len(page.items) > limit
            or len(page.items) > page.total
            or len({item.receipt.review.release_id for item in page.items}) != len(page.items)
        ):
            raise TensorusError("The server returned a different product-release retirement page")
        return page.model_dump(mode="json")

    @staticmethod
    def _product_release_response(data: bytes) -> ProductRelease:
        if len(data) > 2 * MAX_PORTABLE_PRODUCT_BYTES:
            raise LimitExceeded("Product-release response exceeds the SDK byte limit")
        try:
            result = ProductRelease.model_validate(decode_unique_json(data))
            report, request = result.report, result.request
            approved = result.approved_at is not None and result.approved_by is not None
            if (
                canonical_hash(report.model_dump(mode="json")) != result.report_hash
                or report.tenant_id != result.tenant_id
                or report.product_id != result.product_id
                or report.product_name != result.product_name
                or report.product_revision != result.product_revision
                or report.descriptor_hash != request.descriptor_hash
                or report.release_name != result.name
                or report.proposed_by != result.created_by
                or report.quality_coverage != request.quality_coverage
                or request.name != result.name
                or request.product_revision != result.product_revision
                or (
                    result.status == "approved"
                    and (not approved or result.approved_by == result.created_by)
                )
                or (
                    result.status == "proposed"
                    and (result.approved_at is not None or result.approved_by is not None)
                )
            ):
                raise ValueError("Product-release report does not match its envelope")
            return result
        except (UnicodeDecodeError, ValueError, RecursionError, TensorusError) as exc:
            raise TensorusError("The server returned invalid product-release evidence") from exc

    def prepare_product_similarity_evaluation(
        self, name: str, request: ProductSimilarityEvaluationRequest
    ) -> Record:
        """Retain a selected comparison; retry the identical request with the same writer."""
        response = self._request(
            "POST",
            f"/v1/data-products/{quote(name, safe='')}/similarity-evaluations",
            json=request.model_dump(mode="json"),
        ).json()
        return ProductSimilarityEvaluation.model_validate(response).model_dump(mode="json")

    def product_similarity_evaluations(
        self, name: str, *, limit: int = 50, offset: int = 0
    ) -> Record:
        return self._request(
            "GET",
            f"/v1/data-products/{quote(name, safe='')}/similarity-evaluations",
            params={"limit": limit, "offset": offset},
        ).json()

    def product_similarity_evaluation(self, evaluation_id: str) -> Record:
        response = self._request(
            "GET", f"/v1/product-similarity-evaluations/{quote(evaluation_id, safe='')}"
        ).json()
        return ProductSimilarityEvaluation.model_validate(response).model_dump(mode="json")

    def approve_product_similarity_evaluation(self, evaluation_id: str, report_hash: str) -> Record:
        """Reproduce and approve the exact report with a different unscoped writer."""
        response = self._request(
            "POST",
            f"/v1/product-similarity-evaluations/{quote(evaluation_id, safe='')}/approve",
            json={"report_hash": report_hash},
        ).json()
        return ProductSimilarityEvaluation.model_validate(response).model_dump(mode="json")

    def withdraw_product_similarity_evaluation(
        self, evaluation_id: str, request: ProductSimilarityWithdrawalRequest
    ) -> Record:
        """Withdraw a proposed evaluation, retaining its exact evidence and audit receipt."""
        response = self._request(
            "POST",
            f"/v1/product-similarity-evaluations/{quote(evaluation_id, safe='')}/withdraw",
            json=request.model_dump(mode="json"),
        ).json()
        return ProductSimilarityWithdrawal.model_validate(response).model_dump(mode="json")

    def product_similarity_withdrawal(self, evaluation_id: str) -> Record:
        """Read retained withdrawal evidence with an unscoped tenant credential."""
        response = self._request(
            "GET",
            f"/v1/product-similarity-evaluation-withdrawals/{quote(evaluation_id, safe='')}",
        ).json()
        return ProductSimilarityWithdrawal.model_validate(response).model_dump(mode="json")

    def product_similarity_withdrawals(
        self, name: str, *, limit: int = 10, offset: int = 0
    ) -> Record:
        response = self._request(
            "GET",
            f"/v1/data-products/{quote(name, safe='')}/similarity-evaluation-withdrawals",
            params={"limit": limit, "offset": offset},
        ).json()
        return (
            Page[ProductSimilarityWithdrawalSummary]
            .model_validate(response)
            .model_dump(mode="json")
        )

    def review_product_similarity_retirement(
        self, evaluation_id: str, request: ProductSimilarityRetirementReviewRequest
    ) -> Record:
        """Review an approved evaluation's exact evidence and retirement reason without mutation."""
        response = self._request(
            "POST",
            f"/v1/product-similarity-evaluations/{quote(evaluation_id, safe='')}/retirement-review",
            json=request.model_dump(mode="json"),
        ).json()
        return ProductSimilarityRetirementReviewResult.model_validate(response).model_dump(
            mode="json"
        )

    def retire_product_similarity_evaluation(
        self, evaluation_id: str, request: ProductSimilarityRetirementRequest
    ) -> Record:
        """Retire with a different administrator; retry the exact request with the same actor."""
        response = self._request(
            "POST",
            f"/v1/product-similarity-evaluations/{quote(evaluation_id, safe='')}/retire",
            json=request.model_dump(mode="json"),
        ).json()
        return ProductSimilarityRetirement.model_validate(response).model_dump(mode="json")

    def product_similarity_retirement(self, evaluation_id: str) -> Record:
        """Read approved evidence and its retirement receipt with an unscoped tenant credential."""
        response = self._request(
            "GET",
            f"/v1/product-similarity-evaluation-retirements/{quote(evaluation_id, safe='')}",
        ).json()
        return ProductSimilarityRetirement.model_validate(response).model_dump(mode="json")

    def product_similarity_retirements(
        self, name: str, *, limit: int = 10, offset: int = 0
    ) -> Record:
        response = self._request(
            "GET",
            f"/v1/data-products/{quote(name, safe='')}/similarity-evaluation-retirements",
            params={"limit": limit, "offset": offset},
        ).json()
        return (
            Page[ProductSimilarityRetirementSummary]
            .model_validate(response)
            .model_dump(mode="json")
        )

    def review_product_source_evaluation(
        self, name: str, request: ProductSourceEvaluationReviewRequest
    ) -> Record:
        """Review a proposed association without retaining or authenticating source claims."""
        response = self._request(
            "POST",
            f"/v1/data-products/{quote(name, safe='')}/source-evaluation-bindings/review",
            json=request.model_dump(mode="json"),
        ).json()
        return ProductSourceEvaluationReviewResult.model_validate(response).model_dump(mode="json")

    def bind_product_source_evaluation(
        self, name: str, request: ProductSourceEvaluationBindingRequest
    ) -> Record:
        """Reproduce and bind the reviewed evidence with a writer other than the importer."""
        response = self._request(
            "POST",
            f"/v1/data-products/{quote(name, safe='')}/source-evaluation-bindings",
            json=request.model_dump(mode="json"),
        ).json()
        return ProductSourceEvaluationBinding.model_validate(response).model_dump(mode="json")

    def product_source_evaluation(self, binding_id: str) -> Record:
        response = self._request(
            "GET", f"/v1/product-source-evaluation-bindings/{quote(binding_id, safe='')}"
        ).json()
        return ProductSourceEvaluationBinding.model_validate(response).model_dump(mode="json")

    def product_source_evaluations(self, name: str, *, limit: int = 50, offset: int = 0) -> Record:
        response = self._request(
            "GET",
            f"/v1/data-products/{quote(name, safe='')}/source-evaluation-bindings",
            params={"limit": limit, "offset": offset},
        ).json()
        return (
            Page[ProductSourceEvaluationBindingSummary]
            .model_validate(response)
            .model_dump(mode="json")
        )

    def export_product_release(self, release_id: str) -> Record:
        """Return bounded source claims; their hashes do not authenticate their origin."""
        data = self._bounded_report(
            "GET",
            f"/v1/product-releases/{quote(release_id, safe='')}/portable",
            MAX_PORTABLE_PRODUCT_BYTES,
            "Portable product",
            limit_exceeded=True,
        )
        try:
            raw = decode_unique_json(data)
            result = PortableProductRelease.model_validate(raw)
            normalized = result.model_dump(mode="json")
            if canonical_json(raw) != canonical_json(normalized):
                raise ValueError("Portable product evidence has a non-exact representation")
            if str(result.release.id) != str(UUID(release_id)):
                raise ValueError("Portable release identity differs from the request")
            return normalized
        except (UnicodeDecodeError, ValueError, RecursionError) as exc:
            raise TensorusError("The server returned invalid portable product evidence") from exc

    def export_product_bundle(self, release_id: str) -> bytes:
        """Export selected verified NPY payloads and explicitly unauthenticated claims."""
        from tensorus.product_bundle import export_product_bundle

        return export_product_bundle(self, release_id)

    def inspect_product_bundle(self, payload: bytes) -> Record:
        """Verify bundle coverage and payloads without writing destination records."""
        from tensorus.product_bundle import inspect_product_bundle

        return inspect_product_bundle(self, payload)

    def import_product_bundle(
        self, payload: bytes, destinations: dict[str, dict[str, str]]
    ) -> Record:
        """Copy verified payloads without transferring approval, validity or authority."""
        from tensorus.product_bundle import import_product_bundle

        return import_product_bundle(self, payload, destinations)

    def review_product_bundle_restore(
        self, payload: bytes, request: ProductBundleRestoreRequest
    ) -> Record:
        """Review selected local roles, contracts and asserted validity without writing."""
        from tensorus.product_bundle_restore import review_product_bundle_restore

        return review_product_bundle_restore(self, payload, request)

    def inspect_product_bundle_restore(
        self,
        payload: bytes,
        request: ProductBundleRestoreRequest,
        *,
        expected_review_hash: str,
        reviewed: dict[str, Any] | None = None,
    ) -> Record:
        """Inspect receipt-backed restore progress without mutating the destination."""
        from tensorus.product_bundle_restore import inspect_product_bundle_restore

        return inspect_product_bundle_restore(
            self, payload, request, expected_review_hash=expected_review_hash, reviewed=reviewed
        )

    def inspect_product_bundle_restore_payloads(
        self,
        payload: bytes,
        request: ProductBundleRestoreRequest,
        *,
        expected_review_hash: str,
        reviewed: dict[str, Any] | None = None,
    ) -> Record:
        """Observe verified receipts and validity without assessing products or completion."""
        from tensorus.product_bundle import _inspection_review
        from tensorus.product_bundle_restore import inspect_product_bundle_restore

        request = request.model_copy(deep=True)
        if reviewed is not None:
            reviewed = _inspection_review(reviewed, expected_review_hash, reviewed, "roles")
        return inspect_product_bundle_restore(
            self,
            payload,
            request,
            expected_review_hash=expected_review_hash,
            reviewed=reviewed,
            _payloads_only=True,
        )

    def inspect_product_bundle_restore_heads(
        self,
        payload: bytes,
        request: ProductBundleRestoreRequest,
        *,
        expected_review_hash: str,
        reviewed: dict[str, Any] | None = None,
    ) -> Record:
        """Observe receipt-backed product progress and destination heads without writing."""
        from tensorus.product_bundle_restore import inspect_product_bundle_restore

        return inspect_product_bundle_restore(
            self,
            payload,
            request,
            expected_review_hash=expected_review_hash,
            reviewed=reviewed,
            _include_head_preflight=True,
        )

    def inspect_product_bundle_restore_names(
        self,
        payload: bytes,
        request: ProductBundleRestoreRequest,
        *,
        expected_review_hash: str,
        reviewed: dict[str, Any] | None = None,
    ) -> Record:
        """Observe destination names, receipts and heads without reserving or writing."""
        from tensorus.product_bundle_restore import inspect_product_bundle_restore

        return inspect_product_bundle_restore(
            self,
            payload,
            request,
            expected_review_hash=expected_review_hash,
            reviewed=reviewed,
            _include_name_preflight=True,
        )

    def inspect_product_bundle_restore_occupancy(
        self,
        payload: bytes,
        request: ProductBundleRestoreRequest,
        *,
        expected_review_hash: str,
        reviewed: dict[str, Any] | None = None,
    ) -> Record:
        """Check reviewed names without inspecting receipt-backed restore progress."""
        from tensorus.bundle_restore_names import inspect_reviewed_occupancy

        return inspect_reviewed_occupancy(
            self,
            payload,
            request,
            kind="product",
            expected_review_hash=expected_review_hash,
            reviewed=reviewed,
        )

    def observe_bundle_restore_names(self, request: NameOccupancyRequest) -> Record:
        """Observe selected names without inspecting progress, reserving names or writing."""
        from tensorus.bundle_restore_names import observe_destination_names

        return observe_destination_names(self, request)

    def inspect_bundle_archive_on_server(
        self, payload: bytes, *, kind: Literal["product", "history"]
    ) -> Record:
        """Verify bundle claims through the opted-in destination without writing."""
        return self._prepare_bundle_restore_on_server(payload, kind=kind, request=None)

    def review_bundle_restore_on_server(
        self,
        payload: bytes,
        *,
        kind: Literal["product", "history"],
        request: ProductBundleRestoreRequest | HistoryPayloadRestoreRequest,
    ) -> Record:
        """Review a bundle mapping without approving or executing its restore."""
        if request is None:
            raise TensorusError("An explicit bundle restore mapping is required")
        return self._prepare_bundle_restore_on_server(payload, kind=kind, request=request)

    def _prepare_bundle_restore_on_server(
        self,
        payload: bytes,
        *,
        kind: Literal["product", "history"],
        request: ProductBundleRestoreRequest | HistoryPayloadRestoreRequest | None,
    ) -> Record:
        import struct

        from tensorus.bundle_inspection_contract import (
            PREPARE_MAGIC,
            PREPARE_MEDIA_TYPE,
            PREPARE_PATH,
            PREPARE_RESULT_FORMAT,
        )

        if kind not in ("product", "history") or not isinstance(payload, bytes):
            raise TensorusError("A product or history bundle is required")
        if request is not None and not (
            (kind == "product" and isinstance(request, ProductBundleRestoreRequest))
            or (kind == "history" and isinstance(request, HistoryPayloadRestoreRequest))
        ):
            raise TensorusError("The restore request does not match the bundle type")
        credential = self.http.headers.get("Authorization")
        identity, capabilities = self._checked_capabilities()
        if (
            credential is None
            or self.http.headers.get("Authorization") != credential
            or identity.scope is not None
            or identity.role not in (Role.writer, Role.admin)
        ):
            raise Forbidden("Bundle preparation requires an unscoped writer")
        inspection = capabilities.get("bundle_restore_inspection")
        preparation = capabilities.get("bundle_restore_preparation")
        if not isinstance(inspection, dict) or not isinstance(preparation, dict):
            raise TensorusError(
                "The server does not advertise bundle restore preparation",
                code="unsupported_capability",
                status=409,
            )
        header = canonical_json(
            {
                "kind": kind,
                "request": request.model_dump(mode="json") if request is not None else None,
            }
        ).encode()
        limit = inspection[
            "max_product_bundle_bytes" if kind == "product" else "max_history_bundle_bytes"
        ]
        if (
            len(header) > preparation["max_header_bytes"]
            or len(payload) > limit
            or len(PREPARE_MAGIC) + 4 + len(header) + len(payload) > inspection["max_body_bytes"]
        ):
            raise LimitExceeded("Bundle preparation input exceeds its byte limit")
        body = PREPARE_MAGIC + struct.pack(">I", len(header)) + header + payload
        raw = self._bounded_report(
            "POST",
            PREPARE_PATH,
            inspection["max_response_bytes"],
            "Bundle preparation",
            headers={"Authorization": credential, "Content-Type": PREPARE_MEDIA_TYPE},
            content=body,
        )
        try:
            response = decode_unique_json(raw)
            if (
                not isinstance(response, dict)
                or set(response) != {"format", "kind", "mode", "result"}
                or response["format"] != PREPARE_RESULT_FORMAT
                or response["kind"] != kind
                or response["mode"] != ("manifest" if request is None else "review")
                or not isinstance(response["result"], dict)
            ):
                raise ValueError("Bundle preparation response is inconsistent")
            result = response["result"]
        except (UnicodeDecodeError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                "The server returned invalid bundle preparation evidence", code="invalid_response"
            ) from exc
        if self.http.headers.get("Authorization") != credential:
            raise Forbidden("The authenticated identity changed during bundle preparation")
        if request is None:
            expected = (
                self.inspect_product_bundle(payload)
                if kind == "product"
                else self.inspect_release_history_payload_bundle(payload)
            )
        elif isinstance(request, ProductBundleRestoreRequest):
            expected = self.review_product_bundle_restore(payload, request)
        else:
            expected = self.review_release_history_payload_restore(payload, request)
        try:
            if canonical_json(result) != canonical_json(expected):
                raise ValueError("Bundle preparation differs from independently checked evidence")
        except (TypeError, ValueError, RecursionError, TensorusError) as exc:
            raise TensorusError(
                "The server returned contradictory bundle preparation evidence",
                code="invalid_response",
            ) from exc
        if (
            self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
        ):
            raise Forbidden("The authenticated identity changed during bundle preparation")
        return result

    def inspect_bundle_restore_on_server(
        self,
        payload: bytes,
        *,
        kind: Literal["product", "history"],
        reviewed: dict[str, Any],
    ) -> Record:
        """Ask the opted-in destination to perform its checked, read-only inspection."""
        return self._bundle_restore_on_server(payload, kind=kind, reviewed=reviewed)

    def inspect_bundle_restore_payloads_on_server(
        self,
        payload: bytes,
        *,
        kind: Literal["product", "history"],
        reviewed: dict[str, Any],
    ) -> Record:
        """Observe verified payload receipts without assessing retained records or completion."""
        return self._bundle_restore_on_server(
            payload, kind=kind, reviewed=reviewed, payloads_only=True
        )

    def inspect_history_restore_releases_on_server(
        self, payload: bytes, *, reviewed: dict[str, Any]
    ) -> Record:
        """Observe independently verifiable history imports on the opted-in destination."""
        return self._bundle_restore_on_server(
            payload, kind="history", reviewed=reviewed, history_release_observation=True
        )

    def review_history_restore_selection_on_server(
        self, payload: bytes, *, reviewed: dict[str, Any], release_id: str | UUID
    ) -> Record:
        """Review one already-copied history release on the opted-in destination."""
        return self._bundle_restore_on_server(
            payload,
            kind="history",
            reviewed=reviewed,
            history_selection_release_id=release_id,
        )

    def apply_history_restore_selection_on_server(
        self,
        payload: bytes,
        *,
        reviewed: dict[str, Any],
        release_id: str | UUID,
        expected_selection_hash: str,
        confirm_reviewed_selection: bool,
        confirm_unauthenticated_claims: bool,
        confirm_validity_assertions: bool = False,
    ) -> Record:
        """Apply one confirmed selection; an interrupted response may follow a commit."""
        if confirm_reviewed_selection is not True:
            raise TensorusError("Explicit confirmation of the reviewed selection is required")
        if confirm_unauthenticated_claims is not True:
            raise TensorusError(
                "Explicit confirmation of unauthenticated source claims is required"
            )
        if type(confirm_validity_assertions) is not bool:
            raise TensorusError("Validity assertion confirmation must be a boolean")
        return self._bundle_restore_on_server(
            payload,
            kind="history",
            reviewed=reviewed,
            apply=True,
            confirm_validity_assertions=confirm_validity_assertions,
            history_selection_release_id=release_id,
            expected_selection_hash=expected_selection_hash,
        )

    def review_history_payload_selection_on_server(
        self, payload: bytes, *, reviewed: dict[str, Any], release_id: str | UUID
    ) -> Record:
        """Review one history release's payload preparation on the opted-in destination."""
        return self._bundle_restore_on_server(
            payload,
            kind="history",
            reviewed=reviewed,
            history_payload_selection_release_id=release_id,
        )

    def review_product_payload_selection_on_server(
        self, payload: bytes, *, reviewed: dict[str, Any], role: str
    ) -> Record:
        """Review one product role without inspecting unrelated destination receipts."""
        return self._bundle_restore_on_server(
            payload, kind="product", reviewed=reviewed, product_payload_selection_role=role
        )

    def prepare_product_payload_selection_on_server(
        self,
        payload: bytes,
        *,
        reviewed: dict[str, Any],
        role: str,
        expected_selection_hash: str,
        confirm_reviewed_selection: bool,
        confirm_unauthenticated_claims: bool,
        confirm_validity_assertions: bool = False,
    ) -> Record:
        """Prepare a confirmed role; a lost response may follow partial writes."""
        if confirm_reviewed_selection is not True:
            raise TensorusError(
                "Explicit confirmation of the reviewed payload selection is required"
            )
        if confirm_unauthenticated_claims is not True:
            raise TensorusError(
                "Explicit confirmation of unauthenticated source claims is required"
            )
        if type(confirm_validity_assertions) is not bool:
            raise TensorusError("Validity assertion confirmation must be a boolean")
        return self._bundle_restore_on_server(
            payload,
            kind="product",
            reviewed=reviewed,
            apply=True,
            confirm_validity_assertions=confirm_validity_assertions,
            product_payload_selection_role=role,
            expected_selection_hash=expected_selection_hash,
        )

    def review_independent_history_selection_on_server(
        self, payload: bytes, *, reviewed: dict[str, Any], release_id: str | UUID
    ) -> Record:
        """Review one prepared history import without unrelated receipt reads."""
        return self._bundle_restore_on_server(
            payload,
            kind="history",
            reviewed=reviewed,
            history_independent_selection_release_id=release_id,
        )

    def observe_independent_history_release_on_server(
        self, payload: bytes, *, reviewed: dict[str, Any], release_id: str | UUID
    ) -> Record:
        """Observe one history conflict without inspecting unrelated destination receipts."""
        return self._bundle_restore_on_server(
            payload,
            kind="history",
            reviewed=reviewed,
            history_independent_observation_release_id=release_id,
        )

    def apply_independent_history_selection_on_server(
        self,
        payload: bytes,
        *,
        reviewed: dict[str, Any],
        release_id: str | UUID,
        expected_selection_hash: str,
        confirm_reviewed_selection: bool,
        confirm_unauthenticated_claims: bool,
        confirm_validity_assertions: bool = False,
    ) -> Record:
        """Import one reviewed release; a lost response may follow a commit."""
        if confirm_reviewed_selection is not True:
            raise TensorusError("Explicit confirmation of the reviewed selection is required")
        if confirm_unauthenticated_claims is not True:
            raise TensorusError(
                "Explicit confirmation of unauthenticated source claims is required"
            )
        if type(confirm_validity_assertions) is not bool:
            raise TensorusError("Validity assertion confirmation must be a boolean")
        return self._bundle_restore_on_server(
            payload,
            kind="history",
            reviewed=reviewed,
            apply=True,
            confirm_validity_assertions=confirm_validity_assertions,
            history_independent_selection_release_id=release_id,
            expected_selection_hash=expected_selection_hash,
        )

    def prepare_history_payload_selection_on_server(
        self,
        payload: bytes,
        *,
        reviewed: dict[str, Any],
        release_id: str | UUID,
        expected_selection_hash: str,
        confirm_reviewed_selection: bool,
        confirm_unauthenticated_claims: bool,
        confirm_validity_assertions: bool = False,
    ) -> Record:
        """Prepare selected payloads; a lost response may follow partial writes."""
        if confirm_reviewed_selection is not True:
            raise TensorusError(
                "Explicit confirmation of the reviewed payload selection is required"
            )
        if confirm_unauthenticated_claims is not True:
            raise TensorusError(
                "Explicit confirmation of unauthenticated source claims is required"
            )
        if type(confirm_validity_assertions) is not bool:
            raise TensorusError("Validity assertion confirmation must be a boolean")
        return self._bundle_restore_on_server(
            payload,
            kind="history",
            reviewed=reviewed,
            apply=True,
            confirm_validity_assertions=confirm_validity_assertions,
            history_payload_selection_release_id=release_id,
            expected_selection_hash=expected_selection_hash,
        )

    def inspect_bundle_restore_heads_on_server(
        self,
        payload: bytes,
        *,
        kind: Literal["product", "history"],
        reviewed: dict[str, Any],
    ) -> Record:
        """Ask the opted-in destination for separate receipt and head observations."""
        return self._bundle_restore_on_server(
            payload, kind=kind, reviewed=reviewed, include_head_preflight=True
        )

    def inspect_bundle_restore_names_on_server(
        self,
        payload: bytes,
        *,
        kind: Literal["product", "history"],
        reviewed: dict[str, Any],
    ) -> Record:
        """Ask the opted-in destination for non-reserving name and head observations."""
        return self._bundle_restore_on_server(
            payload, kind=kind, reviewed=reviewed, include_name_preflight=True
        )

    def apply_bundle_restore_on_server(
        self,
        payload: bytes,
        *,
        kind: Literal["product", "history"],
        reviewed: dict[str, Any],
        confirm_reviewed_restore: bool,
        confirm_unauthenticated_claims: bool,
        confirm_validity_assertions: bool = False,
        reject_observed_name_conflicts: bool = False,
        inspect_on_failure: bool = False,
        failure_inspection: FailureInspection = "progress",
    ) -> Record:
        """Apply a review; optionally inspect a failed attempt without retrying its writes."""
        from tensorus.bundle_restore import _failure_inspection

        selected_inspection = _failure_inspection(inspect_on_failure, failure_inspection)
        if selected_inspection == "releases" and kind != "history":
            raise ValueError("Release observation is available only for history bundles")
        if confirm_reviewed_restore is not True:
            raise TensorusError("Explicit confirmation of the reviewed restore is required")
        if confirm_unauthenticated_claims is not True:
            raise TensorusError(
                "Explicit confirmation of unauthenticated source claims is required"
            )
        if type(confirm_validity_assertions) is not bool:
            raise TensorusError("Validity assertion confirmation must be a boolean")
        if type(reject_observed_name_conflicts) is not bool:
            raise ValueError("reject_observed_name_conflicts must be a boolean")
        if not inspect_on_failure:
            return self._bundle_restore_on_server(
                payload,
                kind=kind,
                reviewed=reviewed,
                apply=True,
                confirm_validity_assertions=confirm_validity_assertions,
                reject_observed_name_conflicts=reject_observed_name_conflicts,
            )

        from tensorus.bundle_restore import _apply_with_inspection, _prepare_inspection_review

        if kind not in ("product", "history") or not isinstance(reviewed, dict):
            raise TensorusError("An original product or history restore review is required")
        review_hash = reviewed.get("review_hash")
        if not isinstance(review_hash, str):
            raise TensorusError("Original restore review is invalid")
        snapshot = _prepare_inspection_review(
            True, reviewed, review_hash, "roles" if kind == "product" else "datasets"
        )
        if snapshot is None:
            raise TensorusError("Original restore review is required")

        def inspect_failure() -> Record:
            if selected_inspection == "occupancy":
                if kind == "product":
                    request = ProductBundleRestoreRequest.model_validate(snapshot.get("request"))
                    return self.inspect_product_bundle_restore_occupancy(
                        payload, request, expected_review_hash=review_hash, reviewed=snapshot
                    )
                history = HistoryPayloadRestoreRequest.model_validate(snapshot.get("request"))
                return self.inspect_release_history_payload_restore_occupancy(
                    payload, history, expected_review_hash=review_hash, reviewed=snapshot
                )
            return self._bundle_restore_on_server(
                payload,
                kind=kind,
                reviewed=snapshot,
                include_head_preflight=selected_inspection == "heads",
                include_name_preflight=selected_inspection == "names",
                payloads_only=selected_inspection == "payloads",
                history_release_observation=selected_inspection == "releases",
            )

        return _apply_with_inspection(
            lambda: self._bundle_restore_on_server(
                payload,
                kind=kind,
                reviewed=snapshot,
                apply=True,
                confirm_validity_assertions=confirm_validity_assertions,
                reject_observed_name_conflicts=reject_observed_name_conflicts,
            ),
            inspect_failure,
            enabled=True,
            inspection_kind=selected_inspection,
        )

    def _bundle_restore_on_server(
        self,
        payload: bytes,
        *,
        kind: Literal["product", "history"],
        reviewed: dict[str, Any],
        apply: bool = False,
        confirm_validity_assertions: bool = False,
        include_head_preflight: bool = False,
        include_name_preflight: bool = False,
        payloads_only: bool = False,
        history_release_observation: bool = False,
        reject_observed_name_conflicts: bool = False,
        history_selection_release_id: str | UUID | None = None,
        history_payload_selection_release_id: str | UUID | None = None,
        history_independent_selection_release_id: str | UUID | None = None,
        history_independent_observation_release_id: str | UUID | None = None,
        product_payload_selection_role: str | None = None,
        expected_selection_hash: str | None = None,
    ) -> Record:
        import struct

        from tensorus.bundle_inspection_contract import (
            APPLY_MAGIC,
            APPLY_MEDIA_TYPE,
            APPLY_PATH,
            MAGIC,
            MEDIA_TYPE,
            PATH,
            validate_progress,
        )
        from tensorus.product_bundle import MAX_RESTORE_REVIEW_BYTES

        if kind not in ("product", "history") or not isinstance(payload, bytes):
            raise TensorusError("A product or history bundle is required")
        if (
            (include_name_preflight and include_head_preflight)
            or (apply and (include_name_preflight or include_head_preflight))
            or (not apply and reject_observed_name_conflicts)
        ):
            raise TensorusError("Invalid bundle name preflight selection")
        if payloads_only and (include_name_preflight or include_head_preflight or apply):
            raise TensorusError("Invalid bundle observation selection")
        if history_release_observation and (
            kind != "history"
            or payloads_only
            or include_name_preflight
            or include_head_preflight
            or apply
        ):
            raise TensorusError("Invalid history release observation selection")
        if (
            sum(
                item is not None
                for item in (
                    history_selection_release_id,
                    history_payload_selection_release_id,
                    history_independent_selection_release_id,
                    history_independent_observation_release_id,
                    product_payload_selection_role,
                )
            )
            > 1
        ):
            raise TensorusError(
                "Select one bundle continuation"
                if product_payload_selection_role is not None
                else "Select one history continuation"
            )
        if product_payload_selection_role is not None:
            try:
                TypeAdapter(Name).validate_python(product_payload_selection_role, strict=True)
            except ValidationError as exc:
                raise TensorusError("Invalid selected product role") from exc
            if (
                kind != "product"
                or payloads_only
                or history_release_observation
                or include_name_preflight
                or include_head_preflight
                or reject_observed_name_conflicts
            ):
                raise TensorusError("Invalid selected product continuation")
        selection_id = None
        payload_selection_id = None
        independent_selection_id = None
        independent_observation_id = None
        if any(
            item is not None
            for item in (
                history_selection_release_id,
                history_payload_selection_release_id,
                history_independent_selection_release_id,
                history_independent_observation_release_id,
            )
        ):
            from tensorus.history_restore_selection import _release_id

            if history_selection_release_id is not None:
                selection_id = _release_id(history_selection_release_id)
            if history_payload_selection_release_id is not None:
                payload_selection_id = _release_id(history_payload_selection_release_id)
            if history_independent_selection_release_id is not None:
                independent_selection_id = _release_id(history_independent_selection_release_id)
            if history_independent_observation_release_id is not None:
                independent_observation_id = _release_id(history_independent_observation_release_id)
            if (
                kind != "history"
                or payloads_only
                or history_release_observation
                or include_name_preflight
                or include_head_preflight
                or reject_observed_name_conflicts
            ):
                raise TensorusError("Invalid selected history continuation")
        if independent_observation_id is not None and apply:
            raise TensorusError("Independent history observation is read-only")
        selected_id = (
            selection_id
            or payload_selection_id
            or independent_selection_id
            or product_payload_selection_role
        )
        if apply and selected_id is not None:
            if (
                not isinstance(expected_selection_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected_selection_hash) is None
            ):
                raise TensorusError(
                    "A valid selected product review hash is required"
                    if product_payload_selection_role is not None
                    else "A valid selected history review hash is required"
                )
        elif expected_selection_hash is not None:
            raise TensorusError("A selection hash is only supported for selected history apply")
        try:
            encoded_review = canonical_json(reviewed).encode()
            if len(encoded_review) > MAX_RESTORE_REVIEW_BYTES:
                raise LimitExceeded("Original restore review exceeds its byte limit")
            snapshot = decode_unique_json(encoded_review)
            if not isinstance(snapshot, dict):
                raise ValueError("Restore review must be an object")
            review_hash = snapshot.get("review_hash")
            expected_review_format = (
                "tensorus-product-bundle-restore-review-v2"
                if kind == "product"
                else "tensorus-history-payload-restore-review-v2"
            )
            if (
                not isinstance(review_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", review_hash) is None
                or canonical_hash({k: v for k, v in snapshot.items() if k != "review_hash"})
                != review_hash
                or snapshot.get("format") != expected_review_format
            ):
                raise ValueError("Original restore review is invalid")
        except (ValueError, TypeError, RecursionError) as exc:
            raise TensorusError("Original restore review is invalid") from exc
        credential = self.http.headers.get("Authorization")
        identity, capabilities = self._checked_capabilities()
        if (
            credential is None
            or self.http.headers.get("Authorization") != credential
            or identity.scope is not None
            or identity.role not in (Role.writer, Role.admin)
        ):
            raise Forbidden("Bundle inspection requires an unscoped writer")
        if (
            snapshot.get("tenant_id") != str(identity.id)
            or snapshot.get("actor_key_id") != str(identity.key_id)
            or snapshot.get("actor_role") != identity.role.value
        ):
            raise Forbidden("The original review belongs to a different authenticated identity")
        marker = capabilities.get("bundle_restore_inspection")
        if not isinstance(marker, dict):
            raise TensorusError(
                "The server does not advertise bundle restore inspection",
                code="unsupported_capability",
                status=409,
            )
        if apply and "bundle_restore_apply" not in capabilities:
            raise TensorusError(
                "The server does not advertise bundle restore apply",
                code="unsupported_capability",
                status=409,
            )
        if include_head_preflight and "bundle_restore_head_preflight" not in capabilities:
            raise TensorusError(
                "The server does not advertise bundle destination head preflight",
                code="unsupported_capability",
                status=409,
            )
        if (
            include_name_preflight or reject_observed_name_conflicts
        ) and "bundle_restore_name_preflight" not in capabilities:
            raise TensorusError(
                "The server does not advertise bundle destination name preflight",
                code="unsupported_capability",
                status=409,
            )
        if payloads_only and "bundle_restore_payload_observation" not in capabilities:
            raise TensorusError(
                "The server does not advertise bundle payload observation",
                code="unsupported_capability",
                status=409,
            )
        if history_release_observation and "history_release_observation" not in capabilities:
            raise TensorusError(
                "The server does not advertise history release observation",
                code="unsupported_capability",
                status=409,
            )
        if selection_id is not None and "history_restore_selection" not in capabilities:
            raise TensorusError(
                "The server does not advertise selected history continuation",
                code="unsupported_capability",
                status=409,
            )
        if payload_selection_id is not None and "history_payload_selection" not in capabilities:
            raise TensorusError(
                "The server does not advertise selected history payload preparation",
                code="unsupported_capability",
                status=409,
            )
        if (
            product_payload_selection_role is not None
            and "product_payload_selection" not in capabilities
        ):
            raise TensorusError(
                "The server does not advertise selected product payload preparation",
                code="unsupported_capability",
                status=409,
            )
        if (
            independent_selection_id is not None
            and "history_independent_selection" not in capabilities
        ):
            raise TensorusError(
                "The server does not advertise independent history continuation",
                code="unsupported_capability",
                status=409,
            )
        if (
            independent_observation_id is not None
            and "history_independent_observation" not in capabilities
        ):
            raise TensorusError(
                "The server does not advertise independent history observation",
                code="unsupported_capability",
                status=409,
            )
        if (
            apply
            and selected_id is None
            and snapshot.get("validity_assertions_required") is True
            and not confirm_validity_assertions
        ):
            raise TensorusError("Explicit confirmation of local validity assertions is required")
        limit = marker[
            "max_product_bundle_bytes" if kind == "product" else "max_history_bundle_bytes"
        ]
        fields: dict[str, Any] = {
            "kind": kind,
            "expected_review_hash": review_hash,
            "reviewed": snapshot,
        }
        if apply:
            fields.update(
                confirm_reviewed_restore=True,
                confirm_unauthenticated_claims=True,
                confirm_validity_assertions=confirm_validity_assertions,
            )
        if include_head_preflight:
            fields["include_head_preflight"] = True
        if include_name_preflight:
            fields["include_name_preflight"] = True
        if payloads_only:
            fields["payloads_only"] = True
        if history_release_observation:
            fields["history_release_observation"] = True
        if selection_id is not None:
            fields["history_selection_release_id"] = str(selection_id)
        if payload_selection_id is not None:
            fields["history_payload_selection_release_id"] = str(payload_selection_id)
        if product_payload_selection_role is not None:
            fields["product_payload_selection_role"] = product_payload_selection_role
        if independent_selection_id is not None:
            fields["history_independent_selection_release_id"] = str(independent_selection_id)
        if independent_observation_id is not None:
            fields["history_independent_observation_release_id"] = str(independent_observation_id)
        if selected_id is not None and apply:
            fields["expected_selection_hash"] = expected_selection_hash
        if reject_observed_name_conflicts:
            fields["reject_observed_name_conflicts"] = True
        header = canonical_json(fields).encode()
        if len(payload) > limit or len(header) > marker["max_header_bytes"]:
            raise LimitExceeded("Bundle inspection input exceeds its byte limit")
        body = (APPLY_MAGIC if apply else MAGIC) + struct.pack(">I", len(header)) + header + payload
        if len(body) > marker["max_body_bytes"]:
            raise LimitExceeded("Bundle inspection request exceeds its byte limit")
        uncertain = (
            "Selected product preparation may have committed; "
            if apply and product_payload_selection_role is not None
            else "Selected import may have committed; "
            if apply and independent_selection_id is not None
            else ""
        )
        try:
            raw = self._bounded_report(
                "POST",
                APPLY_PATH if apply else PATH,
                marker["max_response_bytes"],
                "Bundle restore progress",
                headers={
                    "Authorization": credential,
                    "Content-Type": APPLY_MEDIA_TYPE if apply else MEDIA_TYPE,
                },
                content=body,
            )
        except TensorusError as exc:
            if uncertain:
                raise TensorusError(
                    f"{uncertain}{exc}",
                    code=exc.code,
                    status=exc.status,
                    details=exc.details,
                ) from exc
            raise
        try:
            result = decode_unique_json(raw)
            if product_payload_selection_role is not None:
                from tensorus.product_bundle import _read_bundle
                from tensorus.product_payload_selection import (
                    validate_product_payload_selection_review,
                )

                product_manifest, arrays = _read_bundle(self, payload)
                del arrays
                product_request = ProductBundleRestoreRequest.model_validate(
                    snapshot.get("request")
                )
                validate_product_payload_selection_review(
                    result,
                    product_manifest,
                    product_request,
                    snapshot,
                    product_payload_selection_role,
                )
                if apply and (
                    result["selection_hash"] != expected_selection_hash
                    or result["complete"] is not True
                ):
                    raise ValueError("Selected product preparation did not report completion")
                progress = result
            elif independent_observation_id is not None:
                from tensorus.history_payload_bundle import _read_history_payload_bundle
                from tensorus.history_restore_independent_observation import (
                    validate_independent_history_observation,
                )

                manifest, arrays = _read_history_payload_bundle(self, payload)
                del arrays
                request = HistoryPayloadRestoreRequest.model_validate(snapshot.get("request"))
                validate_independent_history_observation(
                    result, manifest, request, snapshot, independent_observation_id
                )
                progress = result["payload_review"]
            elif payload_selection_id is not None:
                from tensorus.history_payload_bundle import _read_history_payload_bundle
                from tensorus.history_payload_selection import (
                    validate_history_payload_selection_review,
                )

                manifest, arrays = _read_history_payload_bundle(self, payload)
                del arrays
                request = HistoryPayloadRestoreRequest.model_validate(snapshot.get("request"))
                validate_history_payload_selection_review(
                    result, manifest, request, snapshot, payload_selection_id
                )
                if apply and (
                    result["selection_hash"] != expected_selection_hash
                    or result["complete"] is not True
                ):
                    raise ValueError("Selected payload preparation did not report completion")
                progress = result
            elif independent_selection_id is not None:
                from tensorus.history_payload_bundle import _read_history_payload_bundle
                from tensorus.history_restore_independent import (
                    validate_independent_history_selection_review,
                )

                manifest, arrays = _read_history_payload_bundle(self, payload)
                del arrays
                request = HistoryPayloadRestoreRequest.model_validate(snapshot.get("request"))
                validate_independent_history_selection_review(
                    result, manifest, request, snapshot, independent_selection_id
                )
                if apply and (
                    result["selection_hash"] != expected_selection_hash
                    or result["release_observation"]["status"] != "committed"
                ):
                    raise ValueError("Selected history import did not report a matching commit")
                progress = result["payload_review"]
            elif selection_id is not None:
                from tensorus.history_payload_bundle import _read_history_payload_bundle
                from tensorus.history_restore_selection import validate_history_selection_review

                manifest, arrays = _read_history_payload_bundle(self, payload)
                del arrays
                request = HistoryPayloadRestoreRequest.model_validate(snapshot.get("request"))
                validate_history_selection_review(result, manifest, request, snapshot, selection_id)
                if apply and (
                    result["selection_hash"] != expected_selection_hash
                    or result["release_observation"]["releases"][str(selection_id)]["status"]
                    != "committed"
                ):
                    raise ValueError("Selected history import did not report a matching commit")
                progress = result["release_observation"]["payload_observation"]
            elif history_release_observation:
                from tensorus.history_payload_bundle import _read_history_payload_bundle
                from tensorus.history_payload_restore import validate_history_release_observation

                manifest, arrays = _read_history_payload_bundle(self, payload)
                del arrays
                request = HistoryPayloadRestoreRequest.model_validate(snapshot.get("request"))
                validate_history_release_observation(result, manifest, request, snapshot)
                progress = result["payload_observation"]
            elif include_head_preflight or include_name_preflight:
                from tensorus.bundle_restore_heads import (
                    expectations_from_bundle,
                    validate_head_preflight,
                )

                targets = expectations_from_bundle(self, kind, payload, snapshot)
                if include_name_preflight:
                    from tensorus.bundle_restore_names import validate_name_preflight

                    validate_name_preflight(kind, result, snapshot, targets=targets)
                    progress = result["head_preflight"]["progress"]
                else:
                    validate_head_preflight(kind, result, snapshot, targets=targets)
                    progress = result["progress"]
            else:
                progress = result
                validate_progress(kind, progress, snapshot, payloads_only=payloads_only)
            if apply and selected_id is None and progress["complete"] is not True:
                raise ValueError("Bundle restore did not report complete observed progress")
            for item in progress["payloads"].values():
                version = item["version"]
                if version is not None:
                    checked = TensorVersion.model_validate_json(
                        canonical_json(version), strict=True
                    )
                    if canonical_json(version) != canonical_json(checked.model_dump(mode="json")):
                        raise ValueError("Observed tensor descriptor is not exact")
        except (UnicodeDecodeError, ValidationError, ValueError, TypeError, RecursionError) as exc:
            raise TensorusError(
                (uncertain) + "The server returned invalid bundle progress evidence",
                code="invalid_response",
            ) from exc
        if (
            self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
        ):
            raise Forbidden(
                (uncertain) + "The authenticated identity changed during bundle inspection"
            )
        return result

    def restore_product_bundle(
        self,
        payload: bytes,
        request: ProductBundleRestoreRequest,
        *,
        expected_review_hash: str,
        confirm_unauthenticated_claims: bool,
        confirm_validity_assertions: bool = False,
        reject_observed_name_conflicts: bool = False,
        inspect_on_failure: bool = False,
        reviewed: dict[str, Any] | None = None,
        failure_inspection: FailureInspection = "progress",
    ) -> Record:
        """Restore without approval or trust; optionally inspect a failed apply."""
        from tensorus.bundle_restore import (
            _apply_with_inspection,
            _failure_inspection,
            _prepare_inspection_review,
        )
        from tensorus.product_bundle_restore import restore_product_bundle

        selected_inspection = _failure_inspection(inspect_on_failure, failure_inspection)
        if selected_inspection == "releases":
            raise ValueError("Release observation is available only for history bundles")
        inspection_review = _prepare_inspection_review(
            inspect_on_failure, reviewed, expected_review_hash, "roles"
        )
        if inspect_on_failure:
            request = request.model_copy(deep=True)

        def apply() -> Record:
            return restore_product_bundle(
                self,
                payload,
                request,
                expected_review_hash=expected_review_hash,
                confirm_unauthenticated_claims=confirm_unauthenticated_claims,
                confirm_validity_assertions=confirm_validity_assertions,
                reject_observed_name_conflicts=reject_observed_name_conflicts,
            )

        if not inspect_on_failure:
            return apply()

        inspect = {
            "progress": self.inspect_product_bundle_restore,
            "heads": self.inspect_product_bundle_restore_heads,
            "names": self.inspect_product_bundle_restore_names,
            "occupancy": self.inspect_product_bundle_restore_occupancy,
            "payloads": self.inspect_product_bundle_restore_payloads,
        }[selected_inspection]
        return _apply_with_inspection(
            apply,
            lambda: inspect(
                payload,
                request,
                expected_review_hash=expected_review_hash,
                reviewed=inspection_review,
            ),
            enabled=True,
            inspection_kind=selected_inspection,
        )

    def import_product_release(
        self, name: str, request: ProductReleaseImportRequest, *, _credential: str | None = None
    ) -> Record:
        """Retain mapped source claims without creating a local release approval."""
        response = self._request(
            "POST",
            f"/v1/data-products/{quote(name, safe='')}/release-imports",
            headers={"Authorization": _credential} if _credential is not None else None,
            json=request.model_dump(mode="json"),
        )
        result = self._imported_product_response(response.content)
        if result.product_name != name or result.request != request:
            raise TensorusError("The server returned a different product-import request")
        return result.model_dump(mode="json")

    def product_release_imports(self, name: str, *, limit: int = 50, offset: int = 0) -> Record:
        return self._request(
            "GET",
            f"/v1/data-products/{quote(name, safe='')}/release-imports",
            params={"limit": limit, "offset": offset},
        ).json()

    def product_release_import(self, import_id: str) -> Record:
        response = self._request("GET", f"/v1/product-release-imports/{quote(import_id, safe='')}")
        result = self._imported_product_response(response.content)
        if str(result.id) != str(UUID(import_id)):
            raise TensorusError("The server returned a different product-release import")
        return result.model_dump(mode="json")

    @staticmethod
    def _imported_product_response(data: bytes) -> ImportedProductRelease:
        if len(data) > 2 * MAX_PORTABLE_PRODUCT_BYTES:
            raise LimitExceeded("Product-import response exceeds the SDK byte limit")
        try:
            result = ImportedProductRelease.model_validate(decode_unique_json(data))
            request = result.request
            if (
                canonical_hash(request.model_dump(mode="json")) != result.request_hash
                or request.source.content_hash != result.source_hash
                or request.name != result.name
                or request.product_revision != result.product_revision
            ):
                raise ValueError("Product-import request does not match its envelope")
            return result
        except (UnicodeDecodeError, ValueError, RecursionError, TensorusError) as exc:
            raise TensorusError("The server returned invalid product-import evidence") from exc

    def workflows(
        self,
        *,
        dataset: str | None = None,
        include_archived: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> Record:
        params: Record = {"limit": limit, "offset": offset, "include_archived": include_archived}
        if dataset is not None:
            params["dataset"] = dataset
        return self._request("GET", "/v1/workflows", params=params).json()

    def workflow(self, name: str) -> Record:
        return self._request("GET", f"/v1/workflows/{quote(name, safe='')}").json()

    def save_workflow(
        self,
        name: str,
        definition: WorkflowDefinition,
        *,
        expected_version: int,
    ) -> Record:
        request = WorkflowWrite(definition=definition, expected_version=expected_version)
        return self._request(
            "PUT", f"/v1/workflows/{quote(name, safe='')}", json=request.model_dump(mode="json")
        ).json()

    def workflow_versions(self, name: str, *, limit: int = 50, offset: int = 0) -> Record:
        return self._request(
            "GET",
            f"/v1/workflows/{quote(name, safe='')}/versions",
            params={"limit": limit, "offset": offset},
        ).json()

    def explain_workflow(self, name: str, request: WorkflowRun) -> Record:
        return self._request(
            "POST",
            f"/v1/workflows/{quote(name, safe='')}/explain",
            json=request.model_dump(mode="json"),
        ).json()

    def run_workflow(self, name: str, request: WorkflowRun) -> Record:
        """Create a proposed plan only; approval remains a separate operation."""
        return self._request(
            "POST",
            f"/v1/workflows/{quote(name, safe='')}/plans",
            json=request.model_dump(mode="json"),
        ).json()

    def archive_workflow(self, name: str, *, expected_version: int) -> None:
        self._request(
            "DELETE",
            f"/v1/workflows/{quote(name, safe='')}",
            params={"expected_version": expected_version},
        )

    def propose_goal(self, request: GoalRequest) -> Record:
        return self._request(
            "POST", "/v1/agent/propose", json=request.model_dump(mode="json")
        ).json()

    def approve(self, plan_id: str, plan_hash: str) -> Record:
        return self._request(
            "POST", f"/v1/plans/{quote(plan_id, safe='')}/approve", json={"plan_hash": plan_hash}
        ).json()

    def plan(self, plan_id: str) -> Record:
        return self._request("GET", f"/v1/plans/{quote(plan_id, safe='')}").json()

    def plan_output_reservations(self, plan_id: str) -> Record:
        """Read checked plan-local staging reservations, not live disk or lock state."""
        response = self._request("GET", f"/v1/plans/{quote(plan_id, safe='')}/output-reservations")
        if len(response.content) > 64 * 1024:
            raise TensorusError("The server returned oversized output reservation evidence")
        try:
            result = decode_unique_json(response.content)
            checked = WorkerOutputReservationStatus.model_validate(result)
            matches_plan = checked.plan_id == UUID(plan_id)
        except (ValidationError, TypeError, ValueError, UnicodeDecodeError, RecursionError) as exc:
            raise TensorusError("The server returned invalid output reservation evidence") from exc
        if not matches_plan:
            raise TensorusError("The output reservation evidence is for a different plan")
        return result

    def plan_worker_attempt_evidence(self, plan_id: str, expected_plan_hash: str) -> Record:
        """Read catalog-reported local attempt limits, not independent attestation."""
        try:
            requested_id = UUID(plan_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("A valid plan ID is required") from exc
        if not isinstance(expected_plan_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_plan_hash
        ):
            raise ValueError("A valid expected plan hash is required")
        advertised = self.capabilities().get("worker_attempt_evidence")
        if not isinstance(advertised, dict) or advertised.get("format") != (
            WORKER_ATTEMPT_STATUS_FORMAT
        ):
            raise TensorusError("The server does not advertise worker attempt evidence")
        encoded = self._bounded_report(
            "GET",
            f"/v1/plans/{quote(plan_id, safe='')}/worker-attempts",
            MAX_WORKER_ATTEMPT_RESPONSE_BYTES,
            "Worker attempt evidence",
            params={"expected_plan_hash": expected_plan_hash},
        )
        try:
            result = decode_unique_json(encoded)
            checked = WorkerAttemptEvidence.model_validate(result)
        except (ValidationError, TypeError, ValueError, UnicodeDecodeError, RecursionError) as exc:
            raise TensorusError("The server returned invalid worker attempt evidence") from exc
        if checked.plan_id != requested_id or checked.plan_hash != expected_plan_hash:
            raise TensorusError("The worker attempt evidence is for a different plan or hash")
        return result

    def plan_worker_claim_accounting(self, plan_id: str, expected_plan_hash: str) -> Record:
        """Read plan claim outcomes; no resource use or customer charge is measured."""
        try:
            requested_id = UUID(plan_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("A valid plan ID is required") from exc
        if not isinstance(expected_plan_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_plan_hash
        ):
            raise ValueError("A valid expected plan hash is required")
        advertised = self.capabilities().get("worker_claim_accounting")
        if not isinstance(advertised, dict) or advertised.get("format") != CLAIM_ACCOUNTING_FORMAT:
            raise TensorusError("The server does not advertise worker claim accounting")
        encoded = self._bounded_report(
            "GET",
            f"/v1/plans/{quote(plan_id, safe='')}/worker-claims",
            MAX_WORKER_CLAIM_RESPONSE_BYTES,
            "Worker claim accounting",
            params={"expected_plan_hash": expected_plan_hash},
        )
        try:
            result = decode_unique_json(encoded)
            checked = WorkerClaimAccounting.model_validate_json(encoded, strict=True)
        except (ValidationError, TypeError, ValueError, UnicodeDecodeError, RecursionError) as exc:
            raise TensorusError("The server returned invalid worker claim accounting") from exc
        if checked.plan_id != requested_id or checked.plan_hash != expected_plan_hash:
            raise TensorusError("The worker claim accounting is for a different plan or hash")
        return result

    def plan_estimated_attempts(self, plan_id: str, expected_plan_hash: str) -> Record:
        """Inspect recorded allowance estimates, not measured usage or customer charges."""
        try:
            requested_id = UUID(plan_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("A valid plan ID is required") from exc
        if not isinstance(expected_plan_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_plan_hash
        ):
            raise ValueError("A valid expected plan hash is required")
        credential = self.http.headers.get("Authorization")
        identity, capabilities = self._checked_capabilities()
        marker = capabilities.get("plan_estimated_attempts")
        if not isinstance(marker, dict) or marker.get("format") != PLAN_ESTIMATED_ATTEMPTS_FORMAT:
            raise TensorusError("The server does not advertise plan estimated attempts")
        if credential is None or self.http.headers.get("Authorization") != credential:
            raise Forbidden("The authenticated credential changed during estimate inspection")
        encoded = self._bounded_report(
            "GET",
            f"/v1/plans/{quote(plan_id, safe='')}/estimated-attempts",
            MAX_PLAN_ESTIMATED_ATTEMPTS_RESPONSE_BYTES,
            "Plan estimated attempts",
            params={"expected_plan_hash": expected_plan_hash},
        )
        try:
            result = decode_unique_json(encoded)
            checked = PlanEstimatedAttempts.model_validate_json(encoded, strict=True)
            if not isinstance(result, dict) or canonical_json(result) != canonical_json(
                checked.model_dump(mode="json")
            ):
                raise ValueError("Estimated-attempt evidence is not exact")
        except (
            ValidationError,
            TensorusError,
            TypeError,
            ValueError,
            UnicodeError,
            RecursionError,
        ) as exc:
            raise TensorusError(
                "The server returned invalid plan estimated-attempt evidence",
                code="invalid_response",
            ) from exc
        if (
            checked.tenant_id != identity.id
            or checked.plan_id != requested_id
            or checked.plan_hash != expected_plan_hash
        ):
            raise TensorusError(
                "The estimate is for a different identity or plan", code="invalid_response"
            )
        if (
            self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise Forbidden("The authenticated identity changed during estimate inspection")
        return result

    def plan_usage_observations(self, plan_id: str, expected_plan_hash: str) -> Record:
        """Inspect recorded logical outputs and wall time, not physical usage or charges."""
        try:
            requested_id = UUID(plan_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("A valid plan ID is required") from exc
        if not isinstance(expected_plan_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_plan_hash
        ):
            raise ValueError("A valid expected plan hash is required")
        identity, capabilities = self._checked_capabilities()
        advertised = capabilities.get("plan_usage_observations")
        if not isinstance(advertised, dict) or advertised.get("format") != PLAN_USAGE_FORMAT:
            raise TensorusError("The server does not advertise plan usage observations")
        encoded = self._bounded_report(
            "GET",
            f"/v1/plans/{quote(plan_id, safe='')}/usage-observations",
            MAX_PLAN_USAGE_RESPONSE_BYTES,
            "Plan usage observations",
            params={"expected_plan_hash": expected_plan_hash},
        )
        try:
            result = decode_unique_json(encoded)
            checked = PlanUsageObservations.model_validate_json(encoded, strict=True)
            if canonical_json(result) != canonical_json(checked.model_dump(mode="json")):
                raise ValueError("Plan usage observations are not exact")
        except (ValidationError, TypeError, ValueError, UnicodeDecodeError, RecursionError) as exc:
            raise TensorusError("The server returned invalid plan usage observations") from exc
        if (
            checked.tenant_id != identity.id
            or checked.plan_id != requested_id
            or checked.plan_hash != expected_plan_hash
            or self._checked_identity() != identity
        ):
            raise TensorusError("The plan usage observations are for a different identity or plan")
        return result

    def plan_process_usage_observations(self, plan_id: str, expected_plan_hash: str) -> Record:
        """Inspect partial isolated-worker CPU and RSS observations, not charges."""
        try:
            requested_id = UUID(plan_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("A valid plan ID is required") from exc
        if not isinstance(expected_plan_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_plan_hash
        ):
            raise ValueError("A valid expected plan hash is required")
        identity, capabilities = self._checked_capabilities()
        advertised = capabilities.get("plan_process_usage_observations")
        if (
            not isinstance(advertised, dict)
            or advertised.get("format") != PLAN_PROCESS_USAGE_FORMAT
        ):
            raise TensorusError("The server does not advertise plan process usage observations")
        encoded = self._bounded_report(
            "GET",
            f"/v1/plans/{quote(plan_id, safe='')}/process-usage-observations",
            MAX_PLAN_PROCESS_USAGE_RESPONSE_BYTES,
            "Plan process usage observations",
            params={"expected_plan_hash": expected_plan_hash},
        )
        try:
            result = decode_unique_json(encoded)
            checked = PlanProcessUsage.model_validate_json(encoded, strict=True)
            if canonical_json(result) != canonical_json(checked.model_dump(mode="json")):
                raise ValueError("Plan process usage observations are not exact")
        except (ValidationError, TypeError, ValueError, UnicodeDecodeError, RecursionError) as exc:
            raise TensorusError(
                "The server returned invalid plan process usage observations"
            ) from exc
        if (
            checked.tenant_id != identity.id
            or checked.plan_id != requested_id
            or checked.plan_hash != expected_plan_hash
            or self._checked_identity() != identity
        ):
            raise TensorusError(
                "The process usage observations are for a different identity or plan"
            )
        return result

    @staticmethod
    def check_plan_usage_handoff(data: bytes) -> Record:
        """Check a copied handoff's consistency offline, not its source authenticity."""
        if not isinstance(data, bytes):
            raise TypeError("Usage handoff evidence must be bytes")
        if len(data) > MAX_PLAN_USAGE_HANDOFF_RESPONSE_BYTES:
            raise LimitExceeded("Plan usage handoff exceeds its byte limit")
        try:
            result = decode_unique_json(data)
            checked = PlanUsageHandoff.model_validate_json(data, strict=True)
            if not isinstance(result, dict) or canonical_json(result) != canonical_json(
                checked.model_dump(mode="json")
            ):
                raise ValueError("Plan usage handoff is not exact")
            return result
        except (
            ValidationError,
            TensorusError,
            TypeError,
            ValueError,
            UnicodeError,
            RecursionError,
        ) as exc:
            raise TensorusError(
                "The plan usage handoff evidence is invalid", code="invalid_handoff"
            ) from exc

    def plan_usage_handoff(self, plan_id: str, expected_plan_hash: str) -> Record:
        """Copy permitted plan usage evidence; this does not submit or create charges."""
        try:
            requested_id = UUID(plan_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("A valid plan ID is required") from exc
        if not isinstance(expected_plan_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_plan_hash
        ):
            raise ValueError("A valid expected plan hash is required")
        credential = self.http.headers.get("Authorization")
        identity, capabilities = self._checked_capabilities()
        advertised = capabilities.get("plan_usage_handoff")
        if (
            not isinstance(advertised, dict)
            or advertised.get("format") != PLAN_USAGE_HANDOFF_FORMAT
        ):
            raise TensorusError("The server does not advertise plan usage handoffs")
        if credential is None or self.http.headers.get("Authorization") != credential:
            raise Forbidden("The authenticated credential changed during usage handoff")
        encoded = self._bounded_report(
            "GET",
            f"/v1/plans/{quote(plan_id, safe='')}/usage-handoff",
            MAX_PLAN_USAGE_HANDOFF_RESPONSE_BYTES,
            "Plan usage handoff",
            params={"expected_plan_hash": expected_plan_hash},
        )
        try:
            result = self.check_plan_usage_handoff(encoded)
            document = HandoffDocument.model_validate_json(result["document_json"], strict=True)
        except (TensorusError, ValidationError, TypeError, ValueError, UnicodeError) as exc:
            raise TensorusError(
                "The server returned invalid plan usage handoff evidence", code="invalid_response"
            ) from exc
        if (
            document.usage.tenant_id != identity.id
            or document.usage.plan_id != requested_id
            or document.usage.plan_hash != expected_plan_hash
        ):
            raise TensorusError(
                "The plan usage handoff is for a different identity or plan",
                code="invalid_response",
            )
        if (
            self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise Forbidden("The authenticated identity changed during usage handoff")
        return result

    def _plan_usage_adjustment_context(self) -> tuple[Identity, str]:
        credential = self.http.headers.get("Authorization")
        identity, capabilities = self._checked_capabilities()
        if (
            credential is None
            or self.http.headers.get("Authorization") != credential
            or identity.role != Role.admin
            or identity.scope is not None
        ):
            raise Forbidden("Usage assertions require an unchanged unscoped administrator")
        marker = capabilities.get("plan_usage_adjustments")
        if not isinstance(marker, dict) or marker.get("format") != PLAN_USAGE_ADJUSTMENTS_FORMAT:
            raise TensorusError(
                "The server does not advertise usage assertions", code="unsupported_feature"
            )
        return identity, credential

    @staticmethod
    def check_adjusted_usage_handoff(data: bytes) -> Record:
        """Check a copied handoff offline; hashes do not authenticate its source."""
        if not isinstance(data, bytes):
            raise TypeError("Adjusted usage handoff evidence must be bytes")
        if len(data) > MAX_ADJUSTED_USAGE_HANDOFF_RESPONSE_BYTES:
            raise LimitExceeded("Adjusted usage handoff exceeds its byte limit")
        try:
            result = decode_unique_json(data)
            checked = AdjustedUsageHandoff.model_validate_json(data, strict=True)
            if not isinstance(result, dict) or canonical_json(result) != canonical_json(
                checked.model_dump(mode="json")
            ):
                raise ValueError("Adjusted usage handoff is not exact")
            return result
        except (
            ValidationError,
            TensorusError,
            TypeError,
            ValueError,
            UnicodeError,
            RecursionError,
        ) as exc:
            raise TensorusError(
                "The adjusted usage handoff evidence is invalid", code="invalid_handoff"
            ) from exc

    def adjusted_usage_handoff(self, plan_id: str, expected_plan_hash: str) -> Record:
        """Copy original observations and unverified assertions without creating charges."""
        requested_id = UUID(plan_id)
        if not isinstance(expected_plan_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_plan_hash
        ):
            raise ValueError("A valid expected plan hash is required")
        credential = self.http.headers.get("Authorization")
        identity, capabilities = self._checked_capabilities()
        if (
            credential is None
            or self.http.headers.get("Authorization") != credential
            or identity.role != Role.admin
            or identity.scope is not None
        ):
            raise Forbidden("Adjusted usage handoffs require an unchanged unscoped administrator")
        marker = capabilities.get("plan_adjusted_usage_handoff")
        if not isinstance(marker, dict) or marker.get("format") != ADJUSTED_USAGE_HANDOFF_FORMAT:
            raise TensorusError(
                "The server does not advertise adjusted usage handoffs",
                code="unsupported_feature",
            )
        if self.http.headers.get("Authorization") != credential:
            raise Forbidden("The authenticated credential changed during usage handoff")
        encoded = self._bounded_report(
            "GET",
            f"/v1/plans/{requested_id}/adjusted-usage-handoff",
            MAX_ADJUSTED_USAGE_HANDOFF_RESPONSE_BYTES,
            "Adjusted usage handoff",
            headers={"Authorization": credential},
            params={"expected_plan_hash": expected_plan_hash},
        )
        try:
            result = self.check_adjusted_usage_handoff(encoded)
            document = AdjustedHandoffDocument.model_validate_json(
                result["document_json"], strict=True
            )
        except (TensorusError, ValidationError, TypeError, ValueError, UnicodeError) as exc:
            raise TensorusError(
                "The server returned invalid adjusted usage handoff evidence",
                code="invalid_response",
            ) from exc
        if (
            document.original.usage.tenant_id != identity.id
            or document.original.usage.plan_id != requested_id
            or document.original.usage.plan_hash != expected_plan_hash
            or self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise TensorusError("Adjusted usage handoff identity changed", code="invalid_response")
        return result

    @staticmethod
    def check_workspace_usage_handoff(data: bytes) -> Record:
        """Check a copied bundle offline; hashes do not authenticate its source."""
        if not isinstance(data, bytes):
            raise TypeError("Workspace usage handoff evidence must be bytes")
        if len(data) > MAX_WORKSPACE_USAGE_HANDOFF_RESPONSE_BYTES:
            raise LimitExceeded("Workspace usage handoff exceeds its byte limit")
        try:
            result = decode_unique_json(data)
            checked = WorkspaceUsageHandoff.model_validate_json(data, strict=True)
            if not isinstance(result, dict) or canonical_json(result) != canonical_json(
                checked.model_dump(mode="json")
            ):
                raise ValueError("Workspace usage handoff is not exact")
            return result
        except (
            ValidationError,
            TensorusError,
            TypeError,
            ValueError,
            UnicodeError,
            RecursionError,
        ) as exc:
            raise TensorusError(
                "The workspace usage handoff evidence is invalid", code="invalid_handoff"
            ) from exc

    def workspace_usage_handoff(self, plans: list[dict[str, str]]) -> Record:
        """Copy selected terminal plan evidence without submitting or creating charges."""
        selection = BundleRequest.model_validate({"plans": plans})
        credential = self.http.headers.get("Authorization")
        identity, capabilities = self._checked_capabilities()
        if (
            credential is None
            or self.http.headers.get("Authorization") != credential
            or identity.role != Role.admin
            or identity.scope is not None
        ):
            raise Forbidden("Usage bundles require an unchanged unscoped administrator")
        marker = capabilities.get("workspace_usage_handoff")
        if not isinstance(marker, dict) or marker.get("format") != WORKSPACE_USAGE_HANDOFF_FORMAT:
            raise TensorusError(
                "The server does not advertise workspace usage handoffs",
                code="unsupported_feature",
            )
        if self.http.headers.get("Authorization") != credential:
            raise Forbidden("The authenticated credential changed during usage handoff")
        encoded = self._bounded_report(
            "POST",
            "/v1/workspace/usage-handoff-bundle",
            MAX_WORKSPACE_USAGE_HANDOFF_RESPONSE_BYTES,
            "Workspace usage handoff",
            headers={"Authorization": credential},
            json=selection.model_dump(mode="json"),
        )
        try:
            result = self.check_workspace_usage_handoff(encoded)
            document = BundleDocument.model_validate_json(result["document_json"], strict=True)
        except (TensorusError, ValidationError, TypeError, ValueError, UnicodeError) as exc:
            raise TensorusError(
                "The server returned invalid workspace usage handoff evidence",
                code="invalid_response",
            ) from exc
        if (
            document.tenant_id != identity.id
            or [(item.plan_id, item.plan_hash) for item in document.items]
            != [(item.plan_id, item.expected_plan_hash) for item in selection.plans]
            or self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise TensorusError("Usage bundle identity changed", code="invalid_response")
        return result

    @staticmethod
    def check_workspace_usage_handoff_v2(data: bytes) -> Record:
        """Check selected plan and ingestion evidence offline, without authenticating it."""
        if not isinstance(data, bytes):
            raise TypeError("Workspace usage handoff evidence must be bytes")
        if len(data) > MAX_WORKSPACE_USAGE_HANDOFF_V2_RESPONSE_BYTES:
            raise LimitExceeded("Workspace usage handoff exceeds its byte limit")
        try:
            result = decode_unique_json(data)
            checked = MixedWorkspaceUsageHandoff.model_validate_json(data, strict=True)
            if not isinstance(result, dict) or canonical_json(result) != canonical_json(
                checked.model_dump(mode="json")
            ):
                raise ValueError("Workspace usage handoff is not exact")
            return result
        except (
            ValidationError,
            TensorusError,
            TypeError,
            ValueError,
            UnicodeError,
            RecursionError,
        ) as exc:
            raise TensorusError(
                "The workspace usage handoff evidence is invalid", code="invalid_handoff"
            ) from exc

    def workspace_usage_handoff_v2(
        self,
        *,
        plans: list[dict[str, str]] | None = None,
        ingestions: list[dict[str, str]] | None = None,
    ) -> Record:
        """Copy selected plan and ingestion evidence without provider submission."""
        selection = MixedBundleRequest.model_validate(
            {
                "plans": plans if plans is not None else [],
                "ingestions": ingestions if ingestions is not None else [],
            }
        )
        credential = self.http.headers.get("Authorization")
        identity, capabilities = self._checked_capabilities()
        if (
            credential is None
            or self.http.headers.get("Authorization") != credential
            or identity.role != Role.admin
            or identity.scope is not None
        ):
            raise Forbidden("Usage bundles require an unchanged unscoped administrator")
        marker = capabilities.get("workspace_usage_handoff_v2")
        if (
            not isinstance(marker, dict)
            or marker.get("format") != WORKSPACE_USAGE_HANDOFF_V2_FORMAT
        ):
            raise TensorusError(
                "The server does not advertise mixed workspace usage handoffs",
                code="unsupported_feature",
            )
        if self.http.headers.get("Authorization") != credential:
            raise Forbidden("The authenticated credential changed during usage handoff")
        encoded = self._bounded_report(
            "POST",
            "/v1/workspace/usage-handoff-bundle-v2",
            MAX_WORKSPACE_USAGE_HANDOFF_V2_RESPONSE_BYTES,
            "Workspace usage handoff",
            headers={"Authorization": credential},
            json=selection.model_dump(mode="json"),
        )
        try:
            result = self.check_workspace_usage_handoff_v2(encoded)
            document = MixedBundleDocument.model_validate_json(result["document_json"], strict=True)
            plan_items = (
                BundleDocument.model_validate_json(document.plans.document_json, strict=True).items
                if document.plans is not None
                else []
            )
        except (TensorusError, ValidationError, TypeError, ValueError, UnicodeError) as exc:
            raise TensorusError(
                "The server returned invalid workspace usage handoff evidence",
                code="invalid_response",
            ) from exc
        if (
            document.tenant_id != identity.id
            or [(item.plan_id, item.plan_hash) for item in plan_items]
            != [(item.plan_id, item.expected_plan_hash) for item in selection.plans]
            or [
                (item.observation.version_id, item.observation.checksum)
                for item in document.ingestions
            ]
            != [(item.version_id, item.expected_checksum) for item in selection.ingestions]
            or self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise TensorusError("Usage bundle identity changed", code="invalid_response")
        return result

    def _local_usage_fixture_context(
        self, *, inspect_files: bool = False, inspect_intents: bool = False
    ) -> tuple[Identity, str]:
        credential = self.http.headers.get("Authorization")
        identity, capabilities = self._checked_capabilities()
        if (
            credential is None
            or self.http.headers.get("Authorization") != credential
            or identity.role != Role.admin
            or identity.scope is not None
        ):
            raise Forbidden("Local fixture delivery requires an unchanged unscoped administrator")
        marker = capabilities.get("local_usage_fixture")
        if not isinstance(marker, dict) or marker.get("format") != LOCAL_USAGE_FIXTURE_FORMAT:
            raise TensorusError(
                "The server does not advertise local usage fixture delivery",
                code="unsupported_feature",
            )
        inspection = capabilities.get("local_usage_fixture_inspection")
        if inspect_files and (
            not isinstance(inspection, dict)
            or inspection.get("format") != LOCAL_USAGE_INSPECTION_FORMAT
        ):
            raise TensorusError(
                "The server does not advertise local fixture file inspection",
                code="unsupported_feature",
            )
        intents = capabilities.get("local_usage_fixture_intents")
        if inspect_intents and (
            not isinstance(intents, dict) or intents.get("format") != LOCAL_USAGE_INTENT_FORMAT
        ):
            raise TensorusError(
                "The server does not advertise local fixture intent inspection",
                code="unsupported_feature",
            )
        return identity, credential

    def review_local_usage_fixture(
        self,
        source_namespace: str,
        *,
        plans: list[dict[str, str]] | None = None,
        ingestions: list[dict[str, str]] | None = None,
    ) -> Record:
        """Review selected observations for an opt-in, non-billing local fixture."""
        request = FixtureReviewRequest(
            source_namespace=parse_namespace(source_namespace),
            selection=MixedBundleRequest.model_validate(
                {"plans": plans if plans is not None else [], "ingestions": ingestions or []}
            ),
        )
        identity, credential = self._local_usage_fixture_context()
        encoded = self._bounded_report(
            "POST",
            LOCAL_USAGE_FIXTURE_REVIEW_PATH,
            MAX_LOCAL_USAGE_FIXTURE_RESPONSE_BYTES,
            "Local usage fixture review",
            headers={"Authorization": credential},
            json=request.model_dump(mode="json"),
        )
        result = self._checked_exact_response(encoded, FixtureReview, "local usage fixture review")
        if (
            not review_matches_request(result.review, request, identity.id)
            or self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise TensorusError(
                "Local usage fixture review identity changed", code="invalid_response"
            )
        return result.model_dump(mode="json")

    def deliver_local_usage_fixture(self, request: FixtureDeliveryRequest) -> Record:
        """Confirm an exact review; retry uncertain outcomes with identical inputs and ID."""
        return self._deliver_local_usage_fixture(request)

    def _deliver_local_usage_fixture(
        self,
        request: FixtureDeliveryRequest,
        *,
        required_identity: Identity | None = None,
        required_credential: str | None = None,
    ) -> Record:
        request = FixtureDeliveryRequest.model_validate_json(
            canonical_json(request.model_dump(mode="json")), strict=True
        )
        identity, credential = self._local_usage_fixture_context()
        if (required_identity is not None and identity != required_identity) or (
            required_credential is not None and credential != required_credential
        ):
            raise Forbidden("The authenticated identity changed before fixture delivery")
        try:
            encoded = self._bounded_report(
                "POST",
                LOCAL_USAGE_FIXTURE_DELIVERY_PATH,
                MAX_LOCAL_USAGE_FIXTURE_RESPONSE_BYTES,
                "Local usage fixture delivery",
                headers={"Authorization": credential},
                json=request.model_dump(mode="json"),
            )
        except TensorusError as exc:
            if exc.status < 500 and exc.code != "invalid_response":
                raise
            raise TensorusError(
                "Local fixture delivery may be partial; inspect or retry the identical request",
                code="local_usage_delivery_outcome_unknown",
                status=exc.status if exc.status >= 500 else 502,
            ) from exc
        try:
            result = self._checked_exact_response(
                encoded, FixtureReceipt, "local usage fixture receipt", mutation=True
            )
            if (
                result.request != request
                or result.tenant_id != identity.id
                or result.actor_id != str(identity.key_id)
                or self.http.headers.get("Authorization") != credential
                or self._checked_identity() != identity
                or self.http.headers.get("Authorization") != credential
            ):
                raise ValueError("Local usage fixture receipt identity changed")
            return result.model_dump(mode="json")
        except (TensorusError, ValidationError, TypeError, ValueError, RecursionError) as exc:
            raise TensorusError(
                "Local fixture delivery could not be verified; inspect or retry the "
                "identical request",
                code="local_usage_delivery_outcome_unknown",
                status=502,
            ) from exc

    def prepare_local_usage_fixture_recovery(
        self, request: FixtureDeliveryRequest, reviewed: dict[str, Any]
    ) -> Record:
        """Create a checked local copy before delivery; it grants no authority."""
        request = FixtureDeliveryRequest.model_validate_json(
            canonical_json(request.model_dump(mode="json")), strict=True
        )
        review = FixtureReview.model_validate_json(canonical_json(reviewed), strict=True)
        if canonical_json(reviewed) != canonical_json(review.model_dump(mode="json")):
            raise ValueError("The fixture review is not exact")
        identity, credential = self._local_usage_fixture_context()
        recovery = FixtureRecovery(
            tenant_id=identity.id,
            actor_id=identity.key_id,
            request=request,
            review=review.review,
        )
        if (
            review.review_sha256 != request.expected_review_sha256
            or self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise Forbidden("The identity or review changed before preparing recovery")
        result = recovery.model_dump(mode="json")
        if len(canonical_json(result).encode("utf-8")) > MAX_LOCAL_USAGE_FIXTURE_RESPONSE_BYTES:
            raise LimitExceeded("Local usage fixture recovery exceeds its byte limit")
        return result

    @staticmethod
    def check_local_usage_fixture_recovery(data: bytes) -> Record:
        """Check a saved recovery copy offline, without authenticating it."""
        if not isinstance(data, bytes):
            raise TypeError("Local usage fixture recovery must be bytes")
        if len(data) > MAX_LOCAL_USAGE_FIXTURE_RESPONSE_BYTES:
            raise LimitExceeded("Local usage fixture recovery exceeds its byte limit")
        try:
            raw = decode_unique_json(data)
            result = FixtureRecovery.model_validate_json(data, strict=True)
            if canonical_json(raw) != canonical_json(result.model_dump(mode="json")):
                raise ValueError("Recovery evidence is not exact")
        except (
            TensorusError,
            ValidationError,
            TypeError,
            ValueError,
            UnicodeError,
            RecursionError,
        ) as exc:
            raise TensorusError(
                "Local usage fixture recovery is invalid", code="invalid_recovery"
            ) from exc
        return result.model_dump(mode="json")

    def inspect_local_usage_fixture_recovery(self, data: bytes) -> Record:
        """Compare a retained receipt with a saved recovery copy; do not retry writes."""
        recovery = FixtureRecovery.model_validate(self.check_local_usage_fixture_recovery(data))
        identity, credential = self._local_usage_fixture_context()
        if recovery.tenant_id != identity.id:
            raise Forbidden("The recovery copy belongs to a different workspace")
        receipt = self.local_usage_fixture_receipt(str(recovery.request.request_id))
        checked = FixtureReceipt.model_validate(receipt)
        if (
            checked.tenant_id != recovery.tenant_id
            or checked.actor_id != str(recovery.actor_id)
            or checked.request != recovery.request
            or checked.review != recovery.review
            or self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise TensorusError(
                "The retained receipt does not match the recovery copy", code="invalid_response"
            )
        return receipt

    def inspect_local_usage_fixture_recovery_intent(self, data: bytes) -> Record:
        """Compare a retained intent with a saved recovery copy without retrying delivery."""
        recovery = FixtureRecovery.model_validate(self.check_local_usage_fixture_recovery(data))
        identity, credential = self._local_usage_fixture_context(inspect_intents=True)
        if recovery.tenant_id != identity.id:
            raise Forbidden("The recovery copy belongs to a different workspace")
        intent = self.local_usage_fixture_intent(str(recovery.request.request_id))
        checked = FixtureIntent.model_validate(intent)
        if (
            checked.tenant_id != recovery.tenant_id
            or checked.actor_id != str(recovery.actor_id)
            or checked.request != recovery.request
            or checked.review != recovery.review
            or self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise TensorusError(
                "The retained intent does not match the recovery copy", code="invalid_response"
            )
        return intent

    def retry_local_usage_fixture_recovery(self, data: bytes) -> Record:
        """Explicitly retry the exact request with the credential claimed by the saved copy."""
        recovery = FixtureRecovery.model_validate(self.check_local_usage_fixture_recovery(data))
        identity, credential = self._local_usage_fixture_context()
        if recovery.tenant_id != identity.id or recovery.actor_id != identity.key_id:
            raise Forbidden("Fixture recovery requires the original workspace credential")
        receipt = self._deliver_local_usage_fixture(
            recovery.request, required_identity=identity, required_credential=credential
        )
        checked = FixtureReceipt.model_validate(receipt)
        if checked.request != recovery.request or checked.review != recovery.review:
            raise TensorusError(
                "Local fixture delivery could not be verified; inspect the saved request",
                code="local_usage_delivery_outcome_unknown",
                status=502,
            )
        return receipt

    def local_usage_fixture_receipt(self, request_id: str) -> Record:
        """Inspect a retained receipt; a missing receipt does not rule out partial files."""
        requested_id = UUID(request_id)
        identity, credential = self._local_usage_fixture_context()
        encoded = self._bounded_report(
            "GET",
            f"/v1/workspace/local-usage-fixture/receipts/{requested_id}",
            MAX_LOCAL_USAGE_FIXTURE_RESPONSE_BYTES,
            "Local usage fixture receipt",
            headers={"Authorization": credential},
        )
        result = self._checked_exact_response(
            encoded, FixtureReceipt, "local usage fixture receipt"
        )
        if (
            result.request.request_id != requested_id
            or result.tenant_id != identity.id
            or self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise TensorusError(
                "Local usage fixture receipt identity changed", code="invalid_response"
            )
        return result.model_dump(mode="json")

    def local_usage_fixture_intent(self, request_id: str) -> Record:
        """Inspect retained intent, not file attribution or completed delivery."""
        requested_id = UUID(request_id)
        identity, credential = self._local_usage_fixture_context(inspect_intents=True)
        encoded = self._bounded_report(
            "GET",
            f"/v1/workspace/local-usage-fixture/intents/{requested_id}",
            MAX_LOCAL_USAGE_FIXTURE_RESPONSE_BYTES,
            "Local usage fixture intent",
            headers={"Authorization": credential},
        )
        result = self._checked_exact_response(encoded, FixtureIntent, "local usage fixture intent")
        if (
            result.request.request_id != requested_id
            or result.tenant_id != identity.id
            or self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise TensorusError(
                "Local usage fixture intent identity changed", code="invalid_response"
            )
        return result.model_dump(mode="json")

    def inspect_local_usage_fixture_files(
        self, source_namespace: str, events: list[dict[str, str]]
    ) -> Record:
        """Inspect selected retained file claims without attributing or modifying delivery."""
        request = FixtureFileInspectionRequest.model_validate(
            {"source_namespace": parse_namespace(source_namespace), "events": events}
        )
        identity, credential = self._local_usage_fixture_context(inspect_files=True)
        encoded = self._bounded_report(
            "POST",
            LOCAL_USAGE_FIXTURE_INSPECTION_PATH,
            MAX_LOCAL_USAGE_INSPECTION_RESPONSE_BYTES,
            "Local fixture file observation",
            headers={"Authorization": credential},
            json=request.model_dump(mode="json"),
        )
        result = self._checked_exact_response(
            encoded, FixtureFileInspection, "local fixture file observation"
        )
        if (
            result.tenant_id != identity.id
            or result.request != request
            or self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise TensorusError(
                "Local fixture file observation identity changed", code="invalid_response"
            )
        return result.model_dump(mode="json")

    @staticmethod
    def workspace_usage_reference_v2(data: bytes, source_namespace: str) -> Record:
        """Derive a selected mixed-usage reference without authenticating its source."""
        namespace = parse_namespace(source_namespace)
        handoff = Client.check_workspace_usage_handoff_v2(data)
        bundle = MixedBundleDocument.model_validate_json(handoff["document_json"], strict=True)
        result = mixed_workspace_usage_reference(bundle, namespace).model_dump(mode="json")
        if len(canonical_json(result).encode("utf-8")) > MAX_WORKSPACE_USAGE_REFERENCE_V2_BYTES:
            raise LimitExceeded("Workspace usage reference exceeds its byte limit")
        return result

    @staticmethod
    def check_workspace_usage_reference_v2(data: bytes) -> Record:
        """Check structure only; retained claims and namespace are unauthenticated."""
        return read_mixed_workspace_usage_reference(data).model_dump(mode="json")

    @staticmethod
    def compare_workspace_usage_references_v2(previous: bytes, current: bytes) -> Record:
        """Compare two unauthenticated mixed-usage references."""
        return compare_mixed_workspace_usage_references(
            read_mixed_workspace_usage_reference(previous),
            read_mixed_workspace_usage_reference(current),
        )

    def compare_workspace_usage_reference_v2(
        self, previous: bytes, source_namespace: str
    ) -> Record:
        """Compare a retained selection with fresh administrator evidence."""
        namespace = parse_namespace(source_namespace)
        prior = read_mixed_workspace_usage_reference(previous)
        if prior.source_namespace != namespace:
            raise Conflict("Workspace usage reference identifies a different source namespace")
        credential = self.http.headers.get("Authorization")
        identity = self._checked_identity()
        if (
            credential is None
            or self.http.headers.get("Authorization") != credential
            or identity.role != Role.admin
            or identity.scope is not None
        ):
            raise Forbidden("Usage comparison requires an unchanged unscoped administrator")
        if prior.tenant_id != identity.id:
            raise Conflict("Workspace usage reference identifies a different tenant")
        plans = (
            [
                {
                    "plan_id": str(item.reference.plan_id),
                    "expected_plan_hash": item.reference.plan_hash,
                }
                for item in prior.plans.plans
            ]
            if prior.plans is not None
            else []
        )
        ingestions = [
            {"version_id": str(item.version_id), "expected_checksum": item.checksum}
            for item in prior.ingestions
        ]
        handoff = self.workspace_usage_handoff_v2(plans=plans, ingestions=ingestions)
        bundle = MixedBundleDocument.model_validate_json(handoff["document_json"], strict=True)
        current = mixed_workspace_usage_reference(bundle, namespace)
        result = compare_mixed_workspace_usage_references(prior, current)
        if (
            self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise Forbidden("The authenticated identity changed during usage comparison")
        return result

    @staticmethod
    def workspace_usage_reference(data: bytes, source_namespace: str) -> Record:
        """Derive a selected-plan reference from an unauthenticated handoff copy."""
        namespace = parse_namespace(source_namespace)
        handoff = Client.check_workspace_usage_handoff(data)
        bundle = BundleDocument.model_validate_json(handoff["document_json"], strict=True)
        return reference_from_bundle(bundle, namespace).model_dump(mode="json")

    @staticmethod
    def check_workspace_usage_reference(data: bytes) -> Record:
        """Check structure only; the namespace and retained claims are not authenticated."""
        return read_workspace_usage_reference(data).model_dump(mode="json")

    @staticmethod
    def compare_workspace_usage_references(previous: bytes, current: bytes) -> Record:
        """Compare two unauthenticated references for the same selected plans."""
        return compare_references_for_workspace(
            read_workspace_usage_reference(previous), read_workspace_usage_reference(current)
        )

    def compare_workspace_usage_reference(self, previous: bytes, source_namespace: str) -> Record:
        """Compare a retained selection with fresh administrator handoffs."""
        namespace = parse_namespace(source_namespace)
        prior = read_workspace_usage_reference(previous)
        if prior.source_namespace != namespace:
            raise Conflict("Workspace usage reference identifies a different source namespace")
        credential = self.http.headers.get("Authorization")
        identity = self._checked_identity()
        if (
            credential is None
            or self.http.headers.get("Authorization") != credential
            or identity.role != Role.admin
            or identity.scope is not None
        ):
            raise Forbidden("Usage comparison requires an unchanged unscoped administrator")
        if prior.tenant_id != identity.id:
            raise Conflict("Workspace usage reference identifies a different tenant")
        selections = [
            {
                "plan_id": str(item.reference.plan_id),
                "expected_plan_hash": item.reference.plan_hash,
            }
            for item in prior.plans
        ]
        handoff = self.workspace_usage_handoff(selections)
        bundle = BundleDocument.model_validate_json(handoff["document_json"], strict=True)
        current = reference_from_bundle(bundle, namespace)
        result = compare_references_for_workspace(prior, current)
        if (
            self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise Forbidden("The authenticated identity changed during usage comparison")
        return result

    @staticmethod
    def adjusted_usage_reference(data: bytes) -> Record:
        """Derive a compact reference from a checked, unauthenticated handoff copy."""
        handoff = Client.check_adjusted_usage_handoff(data)
        document = AdjustedHandoffDocument.model_validate_json(
            handoff["document_json"], strict=True
        )
        return reference_from_document(document, handoff["document_sha256"]).model_dump(mode="json")

    @staticmethod
    def check_adjusted_usage_reference(data: bytes) -> Record:
        """Check the structure of a reference, not its authenticity or completeness."""
        return read_reference(data).model_dump(mode="json")

    @staticmethod
    def compare_adjusted_usage_references(previous: bytes, current: bytes) -> Record:
        """Compare two caller-supplied references without authenticating either source."""
        return compare_references(read_reference(previous), read_reference(current))

    def compare_adjusted_usage_reference(
        self, plan_id: str, expected_plan_hash: str, previous: bytes
    ) -> Record:
        """Compare a retained reference with a fresh administrator handoff."""
        requested_id = UUID(plan_id)
        if not isinstance(expected_plan_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_plan_hash
        ):
            raise ValueError("A valid expected plan hash is required")
        prior = read_reference(previous)
        if prior.plan_id != requested_id or prior.plan_hash != expected_plan_hash:
            raise Conflict("Usage evidence reference identifies a different plan")
        credential = self.http.headers.get("Authorization")
        identity = self._checked_identity()
        if (
            credential is None
            or self.http.headers.get("Authorization") != credential
            or identity.role != Role.admin
            or identity.scope is not None
        ):
            raise Forbidden("Usage comparison requires an unchanged unscoped administrator")
        if prior.tenant_id != identity.id:
            raise Conflict("Usage evidence reference identifies a different tenant")
        handoff = self.adjusted_usage_handoff(plan_id, expected_plan_hash)
        document = AdjustedHandoffDocument.model_validate_json(
            handoff["document_json"], strict=True
        )
        current = reference_from_document(document, handoff["document_sha256"])
        result = compare_references(prior, current)
        if (
            self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise Forbidden("The authenticated identity changed during usage comparison")
        return result

    def plan_usage_adjustments(self, plan_id: str, expected_plan_hash: str) -> Record:
        """Review administrator assertions alongside unchanged plan usage observations."""
        requested_id = UUID(plan_id)
        if not isinstance(expected_plan_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_plan_hash
        ):
            raise ValueError("A valid expected plan hash is required")
        identity, credential = self._plan_usage_adjustment_context()
        encoded = self._bounded_report(
            "GET",
            f"/v1/plans/{requested_id}/usage-adjustments",
            MAX_PLAN_USAGE_ADJUSTMENTS_RESPONSE_BYTES,
            "Plan usage assertions",
            headers={"Authorization": credential},
            params={"expected_plan_hash": expected_plan_hash},
        )
        result = self._checked_exact_response(encoded, AdjustmentReport, "plan usage assertions")
        if (
            result.usage.tenant_id != identity.id
            or result.usage.plan_id != requested_id
            or result.usage.plan_hash != expected_plan_hash
            or self.http.headers.get("Authorization") != credential
            or self._checked_identity() != identity
            or self.http.headers.get("Authorization") != credential
        ):
            raise TensorusError("Plan usage assertion identity changed", code="invalid_response")
        return result.model_dump(mode="json")

    def assert_plan_usage_adjustment(self, plan_id: str, request: AdjustmentRequest) -> Record:
        """Record an unverified assertion, without changing measurements or creating charges."""
        requested_id = UUID(plan_id)
        request = AdjustmentRequest.model_validate_json(
            canonical_json(request.model_dump(mode="json")), strict=True
        )
        identity, credential = self._plan_usage_adjustment_context()
        try:
            encoded = self._bounded_report(
                "POST",
                f"/v1/plans/{requested_id}/usage-adjustments",
                MAX_PLAN_USAGE_ADJUSTMENTS_RESPONSE_BYTES,
                "Plan usage assertion",
                headers={"Authorization": credential},
                json=request.model_dump(mode="json"),
            )
        except TensorusError as exc:
            if exc.status < 500 and exc.code != "invalid_response":
                raise
            raise TensorusError(
                "Usage assertion may have committed; retry the identical request ID and inputs",
                code="usage_assertion_outcome_unknown",
                status=exc.status if exc.status >= 500 else 502,
            ) from exc
        try:
            result = self._checked_exact_response(
                encoded, AdjustmentReceipt, "plan usage assertion", mutation=True
            )
            if (
                result.event.request != request
                or result.event.actor_id != str(identity.key_id)
                or result.current.usage.tenant_id != identity.id
                or result.current.usage.plan_id != requested_id
                or result.current.usage.plan_hash != request.expected_plan_hash
                or self.http.headers.get("Authorization") != credential
                or self._checked_identity() != identity
                or self.http.headers.get("Authorization") != credential
            ):
                raise ValueError("Plan usage assertion receipt is inconsistent")
            return result.model_dump(mode="json")
        except (TensorusError, ValidationError, TypeError, ValueError, RecursionError) as exc:
            raise TensorusError(
                "Usage assertion could not be verified and may have committed; retry the "
                "identical request ID and inputs",
                code="usage_assertion_outcome_unknown",
                status=502,
            ) from exc

    def receipt(
        self,
        plan_id: str,
        *,
        include_observation_scope: bool = False,
        observation_scope_version: ObservationScopeVersion = "v1",
    ) -> Record:
        """Check plan evidence, optionally classifying each checked direct input scope."""
        scope_format = observation_scope_format(
            include_observation_scope, observation_scope_version
        )
        result = self._request(
            "GET",
            f"/v1/plans/{quote(plan_id, safe='')}/receipt",
            params={"observation_scope": observation_scope_version} if scope_format else None,
        ).json()
        if scope_format is not None:
            from tensorus.receipt import PlanReceipt

            try:
                checked = PlanReceipt.model_validate(result)
            except ValidationError as exc:
                raise TensorusError(
                    "The server returned invalid observation-scope evidence"
                ) from exc
            if checked.observation_scope_format != scope_format:
                raise TensorusError("The server does not provide the requested observation scope")
            try:
                matches_plan = UUID(checked.plan_id) == UUID(plan_id)
            except (TypeError, ValueError, AttributeError) as exc:
                raise TensorusError("The observation-scope receipt has an invalid plan ID") from exc
            if not matches_plan:
                raise TensorusError("The observation-scope receipt is for a different plan")
        return result

    def plan_outcome(self, plan_id: str, *, outcome_version: PlanOutcomeVersion = "v1") -> Record:
        """Check terminal checkpoints; v2/v3 retain v3/v4 input scope, respectively."""
        requested_format = plan_outcome_format(outcome_version)
        response = self._request(
            "GET",
            f"/v1/plans/{quote(plan_id, safe='')}/outcome",
            params={"outcome_version": outcome_version} if outcome_version != "v1" else None,
        )
        if outcome_version == "v3":
            if len(response.content) > 2 * MAX_PLAN_OUTCOME_BYTES + 1024:
                raise TensorusError("The server returned oversized scoped plan outcome evidence")
            try:
                result = decode_unique_json(response.content)
            except (UnicodeDecodeError, ValueError, RecursionError) as exc:
                raise TensorusError("The server returned invalid scoped plan outcome JSON") from exc
        else:
            result = response.json()
        if outcome_version in ("v2", "v3"):
            try:
                checked = PlanOutcomeEvidence.model_validate(result)
                matches_plan = UUID(checked.outcome.receipt.plan_id) == UUID(plan_id)
            except (ValidationError, TypeError, ValueError, AttributeError) as exc:
                raise TensorusError(
                    "The server returned invalid scoped plan outcome evidence"
                ) from exc
            if checked.outcome.format != requested_format or not matches_plan:
                raise TensorusError("The server did not provide the requested scoped plan outcome")
        return result

    def evaluate(self, plan_id: str, criteria: EvaluationRequest) -> Record:
        """Check explicit criteria, optionally using pinned validity assertions, not goal truth."""
        return self._request(
            "POST",
            f"/v1/plans/{quote(plan_id, safe='')}/evaluate",
            json=criteria.model_dump(mode="json"),
        ).json()

    def plans(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: PlanStatus | None = None,
        dataset: str | None = None,
        text: str | None = None,
    ) -> Record:
        query = PlanQuery(limit=limit, offset=offset, status=status, dataset=dataset, text=text)
        return self._request(
            "GET", "/v1/plans", params=query.model_dump(mode="json", exclude_none=True)
        ).json()

    def audit(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        action: str | None = None,
        resource_id: str | None = None,
    ) -> Record:
        query = AuditQuery(limit=limit, offset=offset, action=action, resource_id=resource_id)
        return self._request(
            "GET", "/v1/audit", params=query.model_dump(mode="json", exclude_none=True)
        ).json()

    def cancel(self, plan_id: str) -> Record:
        return self._request("POST", f"/v1/plans/{quote(plan_id, safe='')}/cancel").json()

    def wait(self, plan_id: str, *, timeout: float = 120, poll: float = 0.5) -> Record:
        if timeout <= 0 or poll <= 0:
            raise ValueError("timeout and poll must be positive")
        deadline = time.monotonic() + timeout
        while True:
            plan = self.plan(plan_id)
            if plan["status"] in {"succeeded", "failed", "cancelled"}:
                return plan
            if time.monotonic() >= deadline:
                raise TimeoutError("Plan is still pending; query its ID to resume waiting")
            time.sleep(min(poll, max(0, deadline - time.monotonic())))

    # --- Commercial Lifecycle, Governance, Settlement, and Regional SDK Methods ---

    def version_lifecycle(self, version_id: str) -> Record:
        return self._request("GET", f"/v1/versions/{quote(version_id, safe='')}/lifecycle").json()

    def retire_version(self, version_id: str, *, expected_checksum: str, reason: str) -> Record:
        return self._request(
            "POST",
            f"/v1/versions/{quote(version_id, safe='')}/retire",
            json={"expected_checksum": expected_checksum, "reason": reason},
        ).json()

    def review_gc(
        self,
        *,
        version_ids: list[str] | None = None,
        max_versions: int = 128,
        backup_copies_may_exist: bool = False,
    ) -> Record:
        body: Record = {
            "max_versions": max_versions,
            "backup_copies_may_exist": backup_copies_may_exist,
        }
        if version_ids is not None:
            body["version_ids"] = version_ids
        return self._request("POST", "/v1/lifecycle/gc/review", json=body).json()

    def execute_gc(
        self,
        *,
        request_id: str,
        review_hash: str,
        version_ids: list[str],
        reason: str,
        backup_copies_may_exist: bool = False,
    ) -> Record:
        return self._request(
            "POST",
            "/v1/lifecycle/gc/execute",
            json={
                "request_id": request_id,
                "review_hash": review_hash,
                "version_ids": version_ids,
                "backup_copies_may_exist": backup_copies_may_exist,
                "reason": reason,
                "confirmation": "COLLECT_UNREACHABLE",
            },
        ).json()

    def gc_runs(self, *, limit: int = 50) -> Record:
        return self._request("GET", "/v1/lifecycle/gc/runs", params={"limit": limit}).json()

    def gc_run(self, run_id: str) -> Record:
        return self._request("GET", f"/v1/lifecycle/gc/runs/{quote(run_id, safe='')}").json()

    def dataset_acl(self, dataset: str) -> Record:
        return self._request("GET", f"/v1/datasets/{quote(dataset, safe='')}/acl").json()

    def set_dataset_acl(
        self,
        dataset: str,
        *,
        mode: Literal["workspace_default", "restricted_acl_v1"],
        expected_generation: int = 0,
        entries: list[dict[str, Any]] | None = None,
        reason: str = "Dataset resource ACL update",
    ) -> Record:
        return self._request(
            "PUT",
            f"/v1/datasets/{quote(dataset, safe='')}/acl",
            json={
                "mode": mode,
                "expected_generation": expected_generation,
                "entries": entries or [],
                "reason": reason,
            },
        ).json()

    def key_ownership(self, key_id: str) -> Record:
        return self._request("GET", f"/v1/keys/{quote(key_id, safe='')}/ownership").json()

    def set_key_ownership(
        self,
        key_id: str,
        *,
        owner_kind: Literal["member", "service_actor", "key"],
        owner_id: str,
        sponsor_membership_id: str | None = None,
        lifecycle_policy: Literal[
            "cascade_on_owner_offboard", "independent"
        ] = "cascade_on_owner_offboard",
    ) -> Record:
        return self._request(
            "PUT",
            f"/v1/keys/{quote(key_id, safe='')}/ownership",
            json={
                "owner_kind": owner_kind,
                "owner_id": owner_id,
                "sponsor_membership_id": sponsor_membership_id,
                "lifecycle_policy": lifecycle_policy,
            },
        ).json()

    def complete_member_offboard(
        self,
        *,
        request_id: str,
        member_id: str,
        reason: str,
        remove_acl_entries: bool = True,
    ) -> Record:
        return self._request(
            "POST",
            "/v1/workspace/members/complete-offboard",
            json={
                "request_id": request_id,
                "member_id": member_id,
                "reason": reason,
                "remove_acl_entries": remove_acl_entries,
                "confirmation": "OFFBOARD_MEMBER",
            },
        ).json()

    def organization_lifecycle(self, organization_id: str) -> Record:
        return self._request(
            "GET", f"/v1/organizations/{quote(organization_id, safe='')}/lifecycle"
        ).json()

    def close_organization(
        self,
        organization_id: str,
        *,
        request_id: str,
        expected_generation: int = 1,
        reason: str,
    ) -> Record:
        return self._request(
            "POST",
            f"/v1/organizations/{quote(organization_id, safe='')}/close",
            json={
                "request_id": request_id,
                "expected_generation": expected_generation,
                "reason": reason,
                "confirmation": "CLOSE_ORGANIZATION",
            },
        ).json()

    def reconnect_organization_workspace(
        self, organization_id: str, *, request_id: str, reason: str
    ) -> Record:
        return self._request(
            "POST",
            f"/v1/organizations/{quote(organization_id, safe='')}/workspaces/reconnect-current",
            json={
                "request_id": request_id,
                "reason": reason,
                "confirmation": "RECONNECT_WORKSPACE",
            },
        ).json()

    def onboard_organization_workspace(
        self,
        organization_id: str,
        *,
        request_id: str,
        name: str,
        title: str = "",
        home_cell_id: str | None = None,
        residency_tag: str = "unspecified",
    ) -> Record:
        return self._request(
            "POST",
            f"/v1/organizations/{quote(organization_id, safe='')}/workspaces/onboard",
            json={
                "request_id": request_id,
                "name": name,
                "title": title,
                "home_cell_id": home_cell_id,
                "residency_tag": residency_tag,
                "confirmation": "ONBOARD_WORKSPACE",
            },
        ).json()

    def egress_policy(self) -> Record:
        return self._request("GET", "/v1/workspace/egress-policy").json()

    def set_egress_policy(
        self,
        *,
        mode: Literal["default_unpinned", "enforce_allowlist_v1"],
        expected_generation: int = 0,
        reason: str = "Workspace outbound egress policy update",
    ) -> Record:
        return self._request(
            "PUT",
            "/v1/workspace/egress-policy",
            json={
                "mode": mode,
                "expected_generation": expected_generation,
                "reason": reason,
            },
        ).json()

    def egress_destinations(self) -> Record:
        return self._request("GET", "/v1/workspace/egress-destinations").json()

    def register_egress_destination(
        self,
        *,
        name: str,
        kind: Literal["model_provider", "billing_adapter", "export_target"],
        endpoint_url: str,
        allowed_grant_ids: list[str] | None = None,
        enabled: bool = True,
    ) -> Record:
        return self._request(
            "POST",
            "/v1/workspace/egress-destinations",
            json={
                "name": name,
                "kind": kind,
                "endpoint_url": endpoint_url,
                "allowed_grant_ids": allowed_grant_ids or [],
                "enabled": enabled,
            },
        ).json()

    def update_egress_destination(
        self,
        destination_id: str,
        *,
        endpoint_url: str | None = None,
        allowed_grant_ids: list[str] | None = None,
        enabled: bool | None = None,
        expected_destination_hash: str | None = None,
    ) -> Record:
        body: Record = {}
        if endpoint_url is not None:
            body["endpoint_url"] = endpoint_url
        if allowed_grant_ids is not None:
            body["allowed_grant_ids"] = allowed_grant_ids
        if enabled is not None:
            body["enabled"] = enabled
        if expected_destination_hash is not None:
            body["expected_destination_hash"] = expected_destination_hash
        return self._request(
            "PUT",
            f"/v1/workspace/egress-destinations/{quote(destination_id, safe='')}",
            json=body,
        ).json()

    def delete_egress_destination(
        self,
        destination_id: str,
        *,
        expected_destination_hash: str | None = None,
    ) -> Record:
        params: Record = {}
        if expected_destination_hash is not None:
            params["expected_destination_hash"] = expected_destination_hash
        return self._request(
            "DELETE",
            f"/v1/workspace/egress-destinations/{quote(destination_id, safe='')}",
            params=params,
        ).json()

    def plan_usage_settlement(self, plan_id: str, expected_plan_hash: str | None = None) -> Record:
        result: Record = self._request(
            "GET", f"/v1/plans/{quote(plan_id, safe='')}/settlement"
        ).json()
        if expected_plan_hash is not None and result.get("subject_hash") != expected_plan_hash:
            raise TensorusError(
                "Plan usage settlement subject_hash does not match expected_plan_hash",
                code="plan_hash_mismatch",
            )
        return result

    def settle_plan_usage(self, plan_id: str) -> Record:
        return self._request("POST", f"/v1/plans/{quote(plan_id, safe='')}/settlement").json()

    def settle_upload_usage(self, upload_id: str) -> Record:
        return self._request("POST", f"/v1/uploads/{quote(upload_id, safe='')}/settlement").json()

    def usage_settlements(self, *, limit: int = 50) -> Record:
        return self._request(
            "GET", "/v1/workspace/usage/settlements", params={"limit": limit}
        ).json()

    def usage_outbox(self, *, delivery_state: str | None = None, limit: int = 50) -> Record:
        params: Record = {"limit": limit}
        if delivery_state is not None:
            params["delivery_state"] = delivery_state
        return self._request("GET", "/v1/workspace/usage/outbox", params=params).json()

    def submit_usage_outbox(
        self, *, max_events: int = 25, settlement_id: str | None = None
    ) -> Record:
        body: Record = {"max_events": max_events}
        if settlement_id is not None:
            body["settlement_id"] = settlement_id
        return self._request("POST", "/v1/workspace/usage/outbox/submit", json=body).json()

    def reconcile_usage_outbox(self, *, max_events: int = 100) -> Record:
        return self._request(
            "POST", "/v1/workspace/usage/outbox/reconcile", params={"max_events": max_events}
        ).json()

    def service_status(self) -> Record:
        return self._request("GET", "/v1/service-status").json()

    def workspace_region(self) -> Record:
        return self._request("GET", "/v1/workspace/region").json()

    def _operator_headers(self, bootstrap_token: str | None) -> dict[str, str]:
        return {"Authorization": f"Bearer {bootstrap_token}"} if bootstrap_token else {}

    def operator_fence_status(self, *, bootstrap_token: str | None = None) -> Record:
        return self._request(
            "GET",
            "/v1/operator/recovery/fence",
            headers=self._operator_headers(bootstrap_token),
        ).json()

    def operator_paired_recovery_points(
        self, *, limit: int = 25, bootstrap_token: str | None = None
    ) -> Record:
        return self._request(
            "GET",
            "/v1/operator/recovery/paired-points",
            params={"limit": limit},
            headers=self._operator_headers(bootstrap_token),
        ).json()

    def operator_capture_paired_recovery_point(
        self,
        *,
        require_quiesced: bool = True,
        verify_object_checksums: bool = True,
        reason: str = "Operator paired catalog and object recovery point",
        bootstrap_token: str | None = None,
    ) -> Record:
        return self._request(
            "POST",
            "/v1/operator/recovery/paired-points",
            json={
                "require_quiesced": require_quiesced,
                "verify_object_checksums": verify_object_checksums,
                "reason": reason,
            },
            headers=self._operator_headers(bootstrap_token),
        ).json()

    def operator_verify_paired_recovery_point(
        self,
        *,
        expected_paired_digest: str | None = None,
        bootstrap_token: str | None = None,
    ) -> Record:
        params: Record = {}
        if expected_paired_digest is not None:
            params["expected_paired_digest"] = expected_paired_digest
        return self._request(
            "POST",
            "/v1/operator/recovery/paired-points/verify",
            params=params,
            headers=self._operator_headers(bootstrap_token),
        ).json()

    def operator_reconcile_restored_catalog(
        self,
        *,
        expected_paired_digest: str | None = None,
        allow_unpaired_emergency_restore: bool = False,
        active_primary_cell: str | None = None,
        revoke_active_sessions: bool = True,
        revoke_key_ids: list[str] | None = None,
        revoke_member_ids: list[str] | None = None,
        rotate_ordinary_keys_cutover: bool = False,
        reason: str = "Operator post-restore catalog reconciliation",
        bootstrap_token: str | None = None,
    ) -> Record:
        return self._request(
            "POST",
            "/v1/operator/recovery/reconcile-restore",
            json={
                "expected_paired_digest": expected_paired_digest,
                "allow_unpaired_emergency_restore": allow_unpaired_emergency_restore,
                "active_primary_cell": active_primary_cell,
                "revoke_active_sessions": revoke_active_sessions,
                "revoke_key_ids": revoke_key_ids or [],
                "revoke_member_ids": revoke_member_ids or [],
                "rotate_ordinary_keys_cutover": rotate_ordinary_keys_cutover,
                "reason": reason,
                "confirmation": "RECONCILE_RESTORED_CATALOG",
            },
            headers=self._operator_headers(bootstrap_token),
        ).json()

    def operator_envelope_diagnostics(self, *, bootstrap_token: str | None = None) -> Record:
        return self._request(
            "GET",
            "/v1/operator/envelope-diagnostics",
            headers=self._operator_headers(bootstrap_token),
        ).json()

    def operator_bind_workspace_region(
        self,
        *,
        tenant: str,
        home_cell_id: str,
        residency_tag: str = "regional-primary",
        reason: str = "Operator regional workspace binding",
        bootstrap_token: str | None = None,
    ) -> Record:
        return self._request(
            "POST",
            "/v1/operator/regions/bind-workspace",
            json={
                "tenant": tenant,
                "home_cell_id": home_cell_id,
                "residency_tag": residency_tag,
                "reason": reason,
            },
            headers=self._operator_headers(bootstrap_token),
        ).json()

    def operator_promote_region(
        self,
        *,
        new_primary_cell: str,
        reason: str,
        previous_primary_cell: str | None = None,
        expected_paired_digest: str | None = None,
        migrate_bound_workspaces: bool = True,
        bootstrap_token: str | None = None,
    ) -> Record:
        return self._request(
            "POST",
            "/v1/operator/regions/promote",
            json={
                "new_primary_cell": new_primary_cell,
                "previous_primary_cell": previous_primary_cell,
                "expected_paired_digest": expected_paired_digest,
                "migrate_bound_workspaces": migrate_bound_workspaces,
                "reason": reason,
                "confirmation": "PROMOTE_PRIMARY_CELL",
            },
            headers=self._operator_headers(bootstrap_token),
        ).json()

    def operator_secret_lifecycle(self, *, bootstrap_token: str | None = None) -> Record:
        return self._request(
            "GET",
            "/v1/operator/secrets/lifecycle",
            headers=self._operator_headers(bootstrap_token),
        ).json()

    def operator_update_incident_runbook(
        self,
        *,
        incident_state: Literal["normal", "degraded", "maintenance", "incident"] = "normal",
        headline: str = "",
        runbook_reference: str = "",
        support_contact: str = "",
        reason: str = "Operator runbook status update",
        bootstrap_token: str | None = None,
    ) -> Record:
        return self._request(
            "PUT",
            "/v1/operator/incident-runbook",
            json={
                "incident_state": incident_state,
                "headline": headline,
                "runbook_reference": runbook_reference,
                "support_contact": support_contact,
                "reason": reason,
            },
            headers=self._operator_headers(bootstrap_token),
        ).json()

    def commercial_plans(self) -> Record:
        return self._request("GET", "/v1/commercial/plans").json()

    def submit_waitlist(
        self,
        *,
        email: str,
        full_name: str,
        use_case: str,
        organization_name: str = "",
        desired_tier: Literal["starter", "pro", "scale", "enterprise"] = "starter",
    ) -> Record:
        return self._request(
            "POST",
            "/v1/waitlist",
            json={
                "email": email,
                "full_name": full_name,
                "organization_name": organization_name,
                "desired_tier": desired_tier,
                "use_case": use_case,
            },
        ).json()

    def workspace_subscription(self) -> Record:
        return self._request("GET", "/v1/workspace/subscription").json()

    def operator_overview(self, *, limit: int = 200, bootstrap_token: str | None = None) -> Record:
        return self._request(
            "GET",
            "/v1/operator/overview",
            params={"limit": limit},
            headers=self._operator_headers(bootstrap_token),
        ).json()

    def operator_waitlist(
        self,
        *,
        status: Literal["pending", "approved", "declined"] | None = None,
        limit: int = 100,
        bootstrap_token: str | None = None,
    ) -> Record:
        params: dict[str, Any] = {"limit": limit}
        if status is not None:
            params["status"] = status
        return self._request(
            "GET",
            "/v1/operator/waitlist",
            params=params,
            headers=self._operator_headers(bootstrap_token),
        ).json()

    def operator_approve_waitlist(
        self,
        entry_id: str,
        *,
        workspace_name: str,
        workspace_title: str = "",
        tier: Literal["starter", "pro", "scale", "enterprise"] = "starter",
        invitation_valid_hours: int = 168,
        review_note: str = "",
        bootstrap_token: str | None = None,
    ) -> Record:
        return self._request(
            "POST",
            f"/v1/operator/waitlist/{quote(entry_id, safe='')}/approve",
            json={
                "workspace_name": workspace_name,
                "workspace_title": workspace_title,
                "tier": tier,
                "invitation_valid_hours": invitation_valid_hours,
                "review_note": review_note,
            },
            headers=self._operator_headers(bootstrap_token),
        ).json()

    def operator_decline_waitlist(
        self, entry_id: str, *, review_note: str, bootstrap_token: str | None = None
    ) -> Record:
        return self._request(
            "POST",
            f"/v1/operator/waitlist/{quote(entry_id, safe='')}/decline",
            json={"review_note": review_note},
            headers=self._operator_headers(bootstrap_token),
        ).json()

    def operator_set_tenant_subscription(
        self,
        tenant_id: str,
        *,
        tier: Literal["starter", "pro", "scale", "enterprise"],
        reason: str,
        stripe_customer_id: str | None = None,
        stripe_subscription_id: str | None = None,
        bootstrap_token: str | None = None,
    ) -> Record:
        return self._request(
            "PUT",
            f"/v1/operator/tenants/{quote(tenant_id, safe='')}/subscription",
            json={
                "tier": tier,
                "reason": reason,
                "stripe_customer_id": stripe_customer_id,
                "stripe_subscription_id": stripe_subscription_id,
            },
            headers=self._operator_headers(bootstrap_token),
        ).json()
