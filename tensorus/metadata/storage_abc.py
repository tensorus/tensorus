import abc
from typing import List, Optional, Dict, Any
from uuid import UUID

from .schemas import (
    TensorDescriptor, SemanticMetadata,
    LineageMetadata, ComputationalMetadata, QualityMetadata,
    RelationalMetadata, UsageMetadata
)

__all__ = ["MetadataStorage"] # Added __all__

class MetadataStorage(abc.ABC):
    """
    Abstract Base Class for metadata storage.
    Defines the interface for all storage backend implementations.
    """

    # --- TensorDescriptor Methods ---
    @abc.abstractmethod
    def add_tensor_descriptor(self, descriptor: TensorDescriptor) -> None:
        pass

    @abc.abstractmethod
    def get_tensor_descriptor(self, tensor_id: UUID) -> Optional[TensorDescriptor]:
        pass

    @abc.abstractmethod
    def update_tensor_descriptor(self, tensor_id: UUID, **kwargs) -> Optional[TensorDescriptor]:
        pass

    @abc.abstractmethod
    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        pass

    @abc.abstractmethod
    def delete_tensor_descriptor(self, tensor_id: UUID) -> bool:
        pass

    # --- SemanticMetadata Methods ---
    @abc.abstractmethod
    def add_semantic_metadata(self, metadata: SemanticMetadata) -> None:
        pass

    @abc.abstractmethod
    def get_semantic_metadata(self, tensor_id: UUID) -> List[SemanticMetadata]:
        pass

    @abc.abstractmethod
    def get_semantic_metadata_by_name(self, tensor_id: UUID, name: str) -> Optional[SemanticMetadata]:
        pass

    @abc.abstractmethod
    def update_semantic_metadata(self, tensor_id: UUID, name: str, new_description: Optional[str] = None, new_name: Optional[str] = None) -> Optional[SemanticMetadata]:
        pass

    @abc.abstractmethod
    def delete_semantic_metadata(self, tensor_id: UUID, name: str) -> bool:
        pass

    # --- LineageMetadata Methods ---
    @abc.abstractmethod
    def add_lineage_metadata(self, metadata: LineageMetadata) -> None:
        pass
    @abc.abstractmethod
    def get_lineage_metadata(self, tensor_id: UUID) -> Optional[LineageMetadata]:
        pass
    @abc.abstractmethod
    def update_lineage_metadata(self, tensor_id: UUID, **kwargs) -> Optional[LineageMetadata]:
        pass
    @abc.abstractmethod
    def delete_lineage_metadata(self, tensor_id: UUID) -> bool:
        pass

    # --- ComputationalMetadata Methods ---
    @abc.abstractmethod
    def add_computational_metadata(self, metadata: ComputationalMetadata) -> None:
        pass
    @abc.abstractmethod
    def get_computational_metadata(self, tensor_id: UUID) -> Optional[ComputationalMetadata]:
        pass
    @abc.abstractmethod
    def update_computational_metadata(self, tensor_id: UUID, **kwargs) -> Optional[ComputationalMetadata]:
        pass
    @abc.abstractmethod
    def delete_computational_metadata(self, tensor_id: UUID) -> bool:
        pass

    # --- QualityMetadata Methods ---
    @abc.abstractmethod
    def add_quality_metadata(self, metadata: QualityMetadata) -> None:
        pass
    @abc.abstractmethod
    def get_quality_metadata(self, tensor_id: UUID) -> Optional[QualityMetadata]:
        pass
    @abc.abstractmethod
    def update_quality_metadata(self, tensor_id: UUID, **kwargs) -> Optional[QualityMetadata]:
        pass
    @abc.abstractmethod
    def delete_quality_metadata(self, tensor_id: UUID) -> bool:
        pass

    # --- RelationalMetadata Methods ---
    @abc.abstractmethod
    def add_relational_metadata(self, metadata: RelationalMetadata) -> None:
        pass
    @abc.abstractmethod
    def get_relational_metadata(self, tensor_id: UUID) -> Optional[RelationalMetadata]:
        pass
    @abc.abstractmethod
    def update_relational_metadata(self, tensor_id: UUID, **kwargs) -> Optional[RelationalMetadata]:
        pass
    @abc.abstractmethod
    def delete_relational_metadata(self, tensor_id: UUID) -> bool:
        pass

    # --- UsageMetadata Methods ---
    @abc.abstractmethod
    def add_usage_metadata(self, metadata: UsageMetadata) -> None:
        pass
    @abc.abstractmethod
    def get_usage_metadata(self, tensor_id: UUID) -> Optional[UsageMetadata]:
        pass
    @abc.abstractmethod
    def update_usage_metadata(self, tensor_id: UUID, **kwargs) -> Optional[UsageMetadata]:
        pass
    @abc.abstractmethod
    def delete_usage_metadata(self, tensor_id: UUID) -> bool:
        pass

    # --- Versioning and Lineage Specific Methods ---
    @abc.abstractmethod
    def get_parent_tensor_ids(self, tensor_id: UUID) -> List[UUID]:
        pass

    @abc.abstractmethod
    def get_child_tensor_ids(self, tensor_id: UUID) -> List[UUID]:
        pass

    # --- Search and Aggregation Methods ---
    @abc.abstractmethod
    def search_tensor_descriptors(self, text_query: str, fields: List[str]) -> List[TensorDescriptor]:
        pass

    @abc.abstractmethod
    def aggregate_tensor_descriptors(self, group_by_field: str, agg_function: str, agg_field: Optional[str] = None) -> Dict[Any, Any]:
        pass

    # --- Utility Methods ---
    @abc.abstractmethod
    def clear_all_data(self) -> None:
        """Clears all data from the storage. Primarily for testing."""
        pass

    # --- Export/Import Methods ---
    @abc.abstractmethod
    def get_export_data(self, tensor_ids: Optional[List[UUID]] = None) -> 'TensorusExportData': # Forward reference for TensorusExportData
        pass

    @abc.abstractmethod
    def import_data(self, data: 'TensorusExportData', conflict_strategy: str = "skip") -> Dict[str, int]:
        pass

    # --- Health and Metrics Methods ---
    @abc.abstractmethod
    def check_health(self) -> tuple[bool, str]: # Returns (is_healthy, backend_type_str)
        pass

    @abc.abstractmethod
    def get_tensor_descriptors_count(self) -> int:
        pass

    @abc.abstractmethod
    def get_extended_metadata_count(self, metadata_model_name: str) -> int:
        pass

    # --- Analytics Methods ---
    @abc.abstractmethod
    def get_co_occurring_tags(self, min_co_occurrence: int = 2, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        pass

    @abc.abstractmethod
    def get_stale_tensors(self, threshold_days: int, limit: int = 100) -> List[TensorDescriptor]:
        pass

    @abc.abstractmethod
    def get_complex_tensors(self, min_parent_count: Optional[int] = None, min_transformation_steps: Optional[int] = None, limit: int = 100) -> List[TensorDescriptor]:
        pass
