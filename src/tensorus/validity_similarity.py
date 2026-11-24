"""Explicit, validity-selected comparison against pinned snapshot members."""
import math
from decimal import Decimal
from fractions import Fraction
from typing import Annotated
from typing import Literal
from typing import Self
from uuid import UUID
import numpy as np
from pydantic import Field
from pydantic import StrictInt
from pydantic import field_validator
from pydantic import model_validator
from tensorus.arrays import MAX_SNAPSHOT_TENSORS
from tensorus.arrays import canonical_hash
from tensorus.contracts import DecimalBound
from tensorus.contracts import ReleaseHash
from tensorus.evaluation import QualityMaskCriterion
from tensorus.models import Model
from tensorus.snapshot_similarity import SnapshotSimilarityMember
from tensorus.snapshot_similarity import SnapshotSimilarityRequest
from tensorus.snapshot_similarity import SnapshotSimilaritySnapshot
from tensorus.snapshot_similarity import SnapshotSimilarityVersion
from tensorus.validity import ValidityBinding
VALIDITY_SIMILARITY_FORMAT: Literal["tensorus-validity-snapshot-similarity-v1"] = (
    "tensorus-validity-snapshot-similarity-v1"
)
MAX_VALIDITY_SIMILARITY_CANDIDATES = 256
MAX_VALIDITY_SIMILARITY_ELEMENTS = 1_000_000
class ValiditySelection(Model):
    binding_id: UUID
    descriptor_hash: ReleaseHash
class CandidateValiditySelection(ValiditySelection):
    version_id: UUID
class ValiditySnapshotSimilarityRequest(SnapshotSimilarityRequest):
    reference_validity: ValiditySelection
    candidate_validity: list[CandidateValiditySelection] = Field(
        max_length=MAX_VALIDITY_SIMILARITY_CANDIDATES
    )
    minimum_common_count: Annotated[StrictInt, Field(ge=1, le=MAX_VALIDITY_SIMILARITY_ELEMENTS)] = 1
    minimum_common_fraction: DecimalBound | None = None
    expected_validity_hash: ReleaseHash | None = None

    @field_validator("minimum_common_fraction")
    @classmethod
    def validate_minimum_fraction(cls, value: str | None) -> str | None:
        return QualityMaskCriterion.validate_minimum(value)

    @model_validator(mode="after")
    def distinct_candidates(self) -> Self:
        if len({item.version_id for item in self.candidate_validity}) != len(
            self.candidate_validity
        ):
            raise ValueError("Candidate validity selections must name distinct versions")
        if len({item.binding_id for item in self.candidate_validity}) != len(
            self.candidate_validity
        ):
            raise ValueError("Candidate validity selections must use distinct bindings")
        return self
class ValiditySelectionEvidence(Model):
    binding: ValidityBinding
    mask: SnapshotSimilarityVersion
class CandidateValidityEvidence(Model):
    version_id: UUID
    selection: ValiditySelectionEvidence
class ValidityComparison(Model):
    version_id: UUID
    candidate_included_count: Annotated[StrictInt, Field(ge=0)]
    common_count: Annotated[StrictInt, Field(ge=0)]
    status: Literal[
        "scored", "no_common_observations", "below_minimum_count", "below_minimum_fraction"
    ]
    score: float | None

    @model_validator(mode="after")
    def score_matches_status(self) -> Self:
        if (self.status == "scored") != (self.score is not None):
            raise ValueError("Only scored comparisons may contain a score")
        return self
class ValiditySimilarityEngine(Model):
    algorithm: Literal["intersection-scaled-longdouble-v1"] = "intersection-scaled-longdouble-v1"
    numpy_version: str
    longdouble_bits: int
    longdouble_mantissa_bits: int
    block_elements: int
class ValiditySnapshotSimilarityReport(Model):
    format: Literal["tensorus-validity-snapshot-similarity-v1"] = VALIDITY_SIMILARITY_FORMAT
    request: ValiditySnapshotSimilarityRequest
    snapshot: SnapshotSimilaritySnapshot
    snapshot_members: list[SnapshotSimilarityMember] = Field(max_length=MAX_SNAPSHOT_TENSORS)
    membership_hash: ReleaseHash
    reference: SnapshotSimilarityVersion
    candidate_manifest: list[SnapshotSimilarityVersion] = Field(
        max_length=MAX_VALIDITY_SIMILARITY_CANDIDATES
    )
    candidate_hash: ReleaseHash
    reference_validity: ValiditySelectionEvidence
    candidate_validity: list[CandidateValidityEvidence] = Field(
        max_length=MAX_VALIDITY_SIMILARITY_CANDIDATES
    )
    validity_hash: ReleaseHash
    shape: list[Annotated[StrictInt, Field(ge=0)]] = Field(max_length=16)
    total_positions: Annotated[StrictInt, Field(ge=0, le=MAX_VALIDITY_SIMILARITY_ELEMENTS)]
    reference_included_count: Annotated[StrictInt, Field(ge=0)]
    candidates: Annotated[StrictInt, Field(ge=0, le=MAX_VALIDITY_SIMILARITY_CANDIDATES)]
    scored_candidates: Annotated[StrictInt, Field(ge=0, le=MAX_VALIDITY_SIMILARITY_CANDIDATES)]
    checked_bytes: Annotated[StrictInt, Field(ge=0)]
    value_scope: Literal["pairwise_validity_intersection"] = "pairwise_validity_intersection"
    score_unit: str | None
    engine: ValiditySimilarityEngine
    comparisons: list[ValidityComparison] = Field(max_length=MAX_VALIDITY_SIMILARITY_CANDIDATES)
    items: list[ValidityComparison] = Field(max_length=100)
    warnings: list[str]
    content_hash: ReleaseHash

    @model_validator(mode="after")
    def consistent_evidence(self) -> Self:
        members = [item.model_dump(mode="json") for item in self.snapshot_members]
        candidates = [item.model_dump(mode="json") for item in self.candidate_manifest]
        selections = [item.model_dump(mode="json") for item in self.candidate_validity]
        validity = {
            "reference": self.reference_validity.model_dump(mode="json"),
            "candidates": selections,
        }
        if (
            self.membership_hash != canonical_hash(members)
            or self.candidate_hash != canonical_hash(candidates)
            or self.validity_hash != canonical_hash(validity)
            or (
                self.request.expected_membership_hash is not None
                and self.request.expected_membership_hash != self.membership_hash
            )
            or (
                self.request.expected_candidate_hash is not None
                and self.request.expected_candidate_hash != self.candidate_hash
            )
            or (
                self.request.expected_validity_hash is not None
                and self.request.expected_validity_hash != self.validity_hash
            )
            or self.total_positions != math.prod(self.shape)
            or self.reference_included_count > self.total_positions
            or self.candidates != len(self.candidate_manifest)
            or len(self.candidate_validity) != self.candidates
            or len(self.comparisons) != self.candidates
        ):
            raise ValueError("Validity similarity evidence is inconsistent")
        expected_selections = {item.version_id: item for item in self.request.candidate_validity}
        candidate_ids = [item.id for item in self.candidate_manifest]
        member_ids = {item.version_id for item in self.snapshot_members}
        members_by_name = {item.name: item.version_id for item in self.snapshot_members}
        if (
            self.snapshot.id != self.request.snapshot_id
            or self.reference.id != self.request.version_id
            or len(members_by_name) != len(self.snapshot_members)
            or len(member_ids) != len(self.snapshot_members)
            or len(set(candidate_ids)) != self.candidates
            or set(candidate_ids) != set(expected_selections)
            or not set(candidate_ids).issubset(member_ids)
            or self.reference.id in candidate_ids
            or any(
                item.dataset_id != self.snapshot.dataset_id
                or item.dataset != self.snapshot.dataset
                or members_by_name.get(item.name) != item.id
                for item in self.candidate_manifest
            )
        ):
            raise ValueError("Validity similarity candidate membership is inconsistent")
        reference = self.reference_validity
        if (
            reference.binding.id != self.request.reference_validity.binding_id
            or reference.binding.descriptor_hash != self.request.reference_validity.descriptor_hash
            or reference.binding.data_version_id != self.reference.id
            or reference.binding.data_checksum != self.reference.checksum
            or reference.binding.shape != self.shape
            or reference.binding.mask_version_id != reference.mask.id
            or reference.binding.mask_checksum != reference.mask.checksum
            or self.reference.nbytes
            != self.total_positions * np.dtype(reference.binding.data_dtype).itemsize
            or reference.mask.nbytes != self.total_positions
        ):
            raise ValueError("Reference validity evidence is inconsistent")
        checked_bytes = self.reference.nbytes + reference.mask.nbytes
        scored: list[ValidityComparison] = []
        for candidate, selected, comparison in zip(
            self.candidate_manifest, self.candidate_validity, self.comparisons, strict=True
        ):
            binding, mask = selected.selection.binding, selected.selection.mask
            requested = expected_selections.get(candidate.id)
            if (
                requested is None
                or selected.version_id != candidate.id
                or comparison.version_id != candidate.id
                or binding.id != requested.binding_id
                or binding.descriptor_hash != requested.descriptor_hash
                or binding.data_version_id != candidate.id
                or binding.data_checksum != candidate.checksum
                or binding.shape != self.shape
                or binding.mask_version_id != mask.id
                or binding.mask_checksum != mask.checksum
                or candidate.nbytes != self.total_positions * np.dtype(binding.data_dtype).itemsize
                or mask.nbytes != self.total_positions
                or comparison.common_count > self.reference_included_count
                or comparison.common_count > comparison.candidate_included_count
                or comparison.common_count
                < self.reference_included_count
                + comparison.candidate_included_count
                - self.total_positions
                or comparison.candidate_included_count > self.total_positions
            ):
                raise ValueError("Candidate validity evidence is inconsistent")
            if comparison.status != _comparison_status(
                self.request, comparison.common_count, self.total_positions
            ):
                raise ValueError("Candidate overlap status is inconsistent")
            if comparison.score is not None and (
                (self.request.metric == "cosine" and not -1 <= comparison.score <= 1)
                or (self.request.metric == "l2" and comparison.score < 0)
            ):
                raise ValueError("Candidate score is outside the metric range")
            checked_bytes += candidate.nbytes + mask.nbytes
            if comparison.score is not None:
                scored.append(comparison)
        if len(expected_selections) != self.candidates or checked_bytes != self.checked_bytes:
            raise ValueError("Validity similarity selection or byte count is inconsistent")
        ranked = sorted(scored, key=lambda item: _score_order(item, self.request.metric))
        if (
            self.scored_candidates != len(scored)
            or self.items != ranked[: self.request.k]
            or (self.request.metric == "cosine" and self.score_unit != "1")
        ):
            raise ValueError("Validity similarity ranking is inconsistent")
        if self.content_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"content_hash"})
        ):
            raise ValueError("Validity similarity content hash is inconsistent")
        return self
def _comparison_status(
    request: ValiditySnapshotSimilarityRequest, count: int, total: int
) -> Literal["scored", "no_common_observations", "below_minimum_count", "below_minimum_fraction"]:
    if not count:
        return "no_common_observations"
    if count < request.minimum_common_count:
        return "below_minimum_count"
    if request.minimum_common_fraction is not None and Fraction(count, total) < Fraction(
        Decimal(request.minimum_common_fraction)
    ):
        return "below_minimum_fraction"
    return "scored"
def _score_order(item: ValidityComparison, metric: Literal["cosine", "l2"]) -> tuple[float, str]:
    if item.score is None:
        raise ValueError("An unscored comparison cannot be ranked")
    return (-item.score if metric == "cosine" else item.score, str(item.version_id))
