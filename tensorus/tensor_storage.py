"""
TensorStorage: A flexible tensor storage system with in-memory and persistent storage capabilities.
"""

import torch
import logging
import os
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetNotFoundError(Exception):
    """Raised when a dataset is not found."""
    pass


class TensorNotFoundError(Exception):
    """Raised when a tensor is not found."""
    pass


class SchemaValidationError(Exception):
    """Raised when tensor or metadata does not match the dataset schema."""
    pass


class TensorStorage:
    """
    A storage system for PyTorch tensors with both in-memory and persistent storage options.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the TensorStorage.
        
        Args:
            storage_path (Optional[str]): Path for persistent storage. If None, only in-memory storage is used.
        """
        self.datasets: Dict[str, Dict[str, Any]] = {}
        self.storage_path = None
        
        if storage_path is not None:
            try:
                path = Path(storage_path)
                if path.is_file():
                    logger.warning(f"Storage path '{storage_path}' is a file. Falling back to in-memory storage.")
                else:
                    path.mkdir(parents=True, exist_ok=True)
                    self.storage_path = str(path.resolve())
                    logger.info(f"Persistent storage enabled at: {self.storage_path}")
                    self._load_persistent_datasets()
            except Exception as e:
                logger.error(f"Failed to initialize persistent storage at '{storage_path}': {e}. Falling back to in-memory storage.")
    
    def _load_persistent_datasets(self) -> None:
        """Load all datasets from persistent storage."""
        if self.storage_path is None:
            return
            
        try:
            path = Path(self.storage_path)
            for file_path in path.glob("*.pt"):
                try:
                    data = torch.load(file_path)
                    dataset_name = file_path.stem
                    self.datasets[dataset_name] = data
                    logger.debug(f"Loaded dataset '{dataset_name}' from {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to load dataset from {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading persistent datasets: {e}")
    
    def _save_dataset(self, name: str) -> None:
        """Save a dataset to persistent storage if enabled."""
        if self.storage_path is None or name not in self.datasets:
            return
            
        try:
            file_path = Path(self.storage_path) / f"{name}.pt"
            torch.save(self.datasets[name], file_path)
            logger.debug(f"Saved dataset '{name}' to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save dataset '{name}': {e}")
    
    def _delete_dataset_file(self, name: str) -> None:
        """Delete a dataset file from persistent storage if it exists."""
        if self.storage_path is None:
            return
            
        try:
            file_path = Path(self.storage_path) / f"{name}.pt"
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Deleted dataset file '{name}.pt'")
        except Exception as e:
            logger.error(f"Failed to delete dataset file '{name}.pt': {e}")
    
    def create_dataset(self, name: str, schema: Optional[Dict[str, Any]] = None) -> None:
        """
        Create a new dataset.
        
        Args:
            name (str): The name of the dataset.
            schema (Optional[Dict]): Optional schema for the dataset.
            
        Raises:
            ValueError: If a dataset with the same name already exists.
        """
        if name in self.datasets:
            raise ValueError(f"Dataset '{name}' already exists.")
            
        self.datasets[name] = {
            "tensors": [],
            "metadata": [],
            "schema": schema
        }
        
        # Save to persistent storage if enabled
        self._save_dataset(name)
        
        logger.info(f"Created dataset '{name}'")
    
    def list_datasets(self) -> List[str]:
        """
        List all dataset names.
        
        Returns:
            List[str]: A list of dataset names.
        """
        return list(self.datasets.keys())
    
    def _validate_tensor(self, dataset_name: str, tensor: torch.Tensor, metadata: Dict[str, Any]) -> None:
        """
        Validate tensor and metadata against dataset schema if one exists.
        
        Args:
            dataset_name (str): Name of the dataset.
            tensor (torch.Tensor): The tensor to validate.
            metadata (Dict[str, Any]): The metadata to validate.
            
        Raises:
            SchemaValidationError: If validation fails.
        """
        if dataset_name not in self.datasets:
            return
            
        schema = self.datasets[dataset_name].get("schema")
        if schema is None:
            return
            
        # Validate tensor shape
        expected_shape = schema.get("shape")
        if expected_shape is not None:
            if list(tensor.shape) != expected_shape:
                raise SchemaValidationError(
                    f"Tensor shape {list(tensor.shape)} does not match expected shape {expected_shape}"
                )
                
        # Validate tensor dtype
        expected_dtype = schema.get("dtype")
        if expected_dtype is not None:
            if str(tensor.dtype) != expected_dtype:
                raise SchemaValidationError(
                    f"Tensor dtype {tensor.dtype} does not match expected dtype {expected_dtype}"
                )
                
        # Validate metadata fields
        expected_metadata = schema.get("metadata", {})
        for field, expected_type in expected_metadata.items():
            if field not in metadata:
                raise SchemaValidationError(f"Missing required metadata field: {field}")
                
            # Type checking (simplified)
            if expected_type == "str" and not isinstance(metadata[field], str):
                raise SchemaValidationError(f"Metadata field {field} should be str, got {type(metadata[field])}")
            elif expected_type == "int" and not isinstance(metadata[field], int):
                raise SchemaValidationError(f"Metadata field {field} should be int, got {type(metadata[field])}")
    
    def insert(self, dataset_name: str, tensor: torch.Tensor, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Insert a tensor into a dataset.
        
        Args:
            dataset_name (str): The name of the dataset.
            tensor (torch.Tensor): The tensor to insert.
            metadata (Optional[Dict[str, Any]]): Optional metadata for the tensor.
            
        Returns:
            str: The record ID of the inserted tensor.
            
        Raises:
            DatasetNotFoundError: If the dataset does not exist.
            TypeError: If tensor is not a torch.Tensor.
            SchemaValidationError: If tensor or metadata doesn't match schema.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Tensor must be a torch.Tensor")
            
        if dataset_name not in self.datasets:
            raise DatasetNotFoundError(f"Dataset '{dataset_name}' not found")
            
        # Validate against schema if one exists
        if metadata is None:
            metadata = {}
            
        self._validate_tensor(dataset_name, tensor, metadata)
        
        # Generate record ID and timestamp
        record_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        # Add auto-generated fields to metadata
        final_metadata = metadata.copy()
        final_metadata["record_id"] = record_id
        final_metadata["timestamp_utc"] = timestamp
        final_metadata["shape"] = list(tensor.shape)
        final_metadata["dtype"] = str(tensor.dtype)
        
        # Store tensor and metadata
        self.datasets[dataset_name]["tensors"].append(tensor)
        self.datasets[dataset_name]["metadata"].append(final_metadata)
        
        # Save to persistent storage if enabled
        self._save_dataset(dataset_name)
        
        logger.debug(f"Inserted tensor with record_id '{record_id}' into dataset '{dataset_name}'")
        return record_id
    
    def get_dataset(self, name: str) -> List[torch.Tensor]:
        """
        Retrieves all tensors from a specified dataset.
        
        Args:
            name (str): The name of the dataset to retrieve.
            
        Returns:
            List[torch.Tensor]: A list of all tensors in the dataset.
            
        Raises:
            DatasetNotFoundError: If the dataset 'name' is not found.
        """
        if name not in self.datasets:
            logger.error(f"Dataset '{name}' not found for retrieval.")
            raise DatasetNotFoundError(f"Dataset '{name}' not found.")
            
        logger.debug(f"Retrieving all {len(self.datasets[name]['tensors'])} tensors from dataset '{name}'.")
        return self.datasets[name]["tensors"]
    
    def get_dataset_with_metadata(self, name: str) -> List[Dict[str, Any]]:
        """
        Retrieves all tensors and their metadata from a specified dataset.
        
        Args:
            name (str): The name of the dataset to retrieve.
            
        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing tensors and their metadata.
            
        Raises:
            DatasetNotFoundError: If the dataset 'name' is not found.
        """
        if name not in self.datasets:
            logger.error(f"Dataset '{name}' not found for retrieval with metadata.")
            raise DatasetNotFoundError(f"Dataset '{name}' not found.")
            
        logger.debug(f"Retrieving all {len(self.datasets[name]['tensors'])} tensors and metadata from dataset '{name}'.")
        
        results = []
        for tensor, meta in zip(self.datasets[name]["tensors"], self.datasets[name]["metadata"]):
            results.append({"tensor": tensor, "metadata": meta})
        return results
    
    def get_tensor_by_id(self, dataset_name: str, record_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific tensor by its record ID.
        
        Args:
            dataset_name (str): The name of the dataset.
            record_id (str): The record ID of the tensor.
            
        Returns:
            Dict[str, Any]: A dictionary containing the tensor and its metadata.
            
        Raises:
            DatasetNotFoundError: If the dataset does not exist.
            TensorNotFoundError: If no tensor with the given record_id exists.
        """
        if dataset_name not in self.datasets:
            raise DatasetNotFoundError(f"Dataset '{dataset_name}' not found")
            
        for tensor, metadata in zip(self.datasets[dataset_name]["tensors"], self.datasets[dataset_name]["metadata"]):
            if metadata.get("record_id") == record_id:
                return {"tensor": tensor, "metadata": metadata}
                
        raise TensorNotFoundError(f"Tensor with record_id '{record_id}' not found in dataset '{dataset_name}'")
    
    def query(self, name: str, query_fn: Callable[[torch.Tensor, Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
        """
        Queries a dataset using a function that filters tensors based on the
        tensor data itself and/or its metadata.
        
        Args:
            name: The name of the dataset to query.
            query_fn (Callable[[torch.Tensor, Dict[str, Any]], bool]):
                A function that takes a tensor and its metadata and returns True if
                the tensor should be included in the result.
                
        Returns:
            List[Dict[str, Any]]: A list of matching tensor-metadata pairs.
            
        Raises:
            DatasetNotFoundError: If the dataset 'name' is not found.
            TypeError: If query_fn is not callable.
        """
        if name not in self.datasets:
            raise DatasetNotFoundError(f"Dataset '{name}' not found.")
            
        if not callable(query_fn):
            raise TypeError("query_fn must be callable")
            
        results = []
        for tensor, metadata in zip(self.datasets[name]["tensors"], self.datasets[name]["metadata"]):
            if query_fn(tensor, metadata):
                results.append({"tensor": tensor, "metadata": metadata})
                
        return results
    
    def update_tensor_metadata(self, dataset_name: str, record_id: str, new_metadata: Dict[str, Any]) -> bool:
        """
        Updates the metadata for a specific tensor, identified by its record_id.
        
        Args:
            dataset_name (str): The name of the dataset.
            record_id (str): The record ID of the tensor to update.
            new_metadata (Dict[str, Any]): The new metadata to set.
            
        Returns:
            bool: True if the metadata was updated, False otherwise.
            
        Raises:
            DatasetNotFoundError: If the dataset does not exist.
            TensorNotFoundError: If no tensor with the given record_id exists.
        """
        if dataset_name not in self.datasets:
            raise DatasetNotFoundError(f"Dataset '{dataset_name}' not found")
            
        # Prevent changing the record_id
        if "record_id" in new_metadata:
            logger.warning("Cannot change record_id. Ignoring this field in new_metadata.")
            new_metadata = {k: v for k, v in new_metadata.items() if k != "record_id"}
            
        for i, metadata in enumerate(self.datasets[dataset_name]["metadata"]):
            if metadata.get("record_id") == record_id:
                # Update metadata (keeping auto-generated fields)
                updated_metadata = metadata.copy()
                updated_metadata.update(new_metadata)
                self.datasets[dataset_name]["metadata"][i] = updated_metadata
                
                # Save to persistent storage if enabled
                self._save_dataset(dataset_name)
                
                logger.debug(f"Updated metadata for tensor with record_id '{record_id}' in dataset '{dataset_name}'")
                return True
                
        raise TensorNotFoundError(f"Tensor with record_id '{record_id}' not found in dataset '{dataset_name}'")
    
    def delete_tensor(self, dataset_name: str, record_id: str) -> bool:
        """
        Deletes a tensor identified by its record_id from a dataset.
        
        Args:
            dataset_name (str): The name of the dataset.
            record_id (str): The record ID of the tensor to delete.
            
        Returns:
            bool: True if the tensor was deleted, False otherwise.
            
        Raises:
            DatasetNotFoundError: If the dataset does not exist.
            TensorNotFoundError: If no tensor with the given record_id exists.
        """
        if dataset_name not in self.datasets:
            raise DatasetNotFoundError(f"Dataset '{dataset_name}' not found")
            
        tensors = self.datasets[dataset_name]["tensors"]
        metadata_list = self.datasets[dataset_name]["metadata"]
        
        for i, metadata in enumerate(metadata_list):
            if metadata.get("record_id") == record_id:
                # Remove tensor and metadata
                tensors.pop(i)
                metadata_list.pop(i)
                
                # Save to persistent storage if enabled
                self._save_dataset(dataset_name)
                
                logger.debug(f"Deleted tensor with record_id '{record_id}' from dataset '{dataset_name}'")
                return True
                
        raise TensorNotFoundError(f"Tensor with record_id '{record_id}' not found in dataset '{dataset_name}'")
    
    def delete_dataset(self, name: str) -> bool:
        """
        Deletes an entire dataset. Use with caution!
        
        Args:
            name (str): The name of the dataset to delete.
            
        Returns:
            bool: True if the dataset was deleted (from memory and disk if applicable).
            
        Raises:
            DatasetNotFoundError: If the dataset `name` is not found.
        """
        if name in self.datasets:
            # Remove from memory
            del self.datasets[name]
            
            # Remove from disk if persistent storage is enabled
            self._delete_dataset_file(name)
            
            logger.info(f"Deleted dataset '{name}'")
            return True
        else:
            raise DatasetNotFoundError(f"Dataset '{name}' not found.")
    
    def sample_dataset(self, name: str, k: int) -> List[Dict[str, Any]]:
        """
        Randomly samples k tensors from a dataset.
        
        Args:
            name (str): The name of the dataset.
            k (int): The number of tensors to sample.
            
        Returns:
            List[Dict[str, Any]]: A list of k randomly sampled tensor-metadata pairs.
            If k is greater than the dataset size, returns all items.
            
        Raises:
            DatasetNotFoundError: If the dataset 'name' is not found.
        """
        import random
        
        if name not in self.datasets:
            raise DatasetNotFoundError(f"Dataset '{name}' not found.")
            
        # Get all tensor-metadata pairs
        all_data = self.get_dataset_with_metadata(name)
        
        # Sample k items
        if k >= len(all_data):
            return all_data
        else:
            return random.sample(all_data, k)
    
    def get_records_paginated(self, name: str, offset: int = 0, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves a paginated list of tensors and their metadata.
        
        Args:
            name (str): The name of the dataset.
            offset (int): The starting index for the page.
            limit (int): The maximum number of records to return.
            
        Returns:
            List[Dict[str, Any]]: A list of tensor-metadata pairs for the requested page.
            
        Raises:
            DatasetNotFoundError: If the dataset 'name' is not found.
        """
        if name not in self.datasets:
            raise DatasetNotFoundError(f"Dataset '{name}' not found.")
            
        # Get all tensor-metadata pairs
        all_data = self.get_dataset_with_metadata(name)
        
        # Apply pagination
        return all_data[offset:offset + limit]
    
    def count(self, name: str) -> int:
        """
        Returns the number of tensors in a dataset.
        
        Args:
            name (str): The name of the dataset.
            
        Returns:
            int: The number of tensors in the dataset.
            
        Raises:
            DatasetNotFoundError: If the dataset 'name' is not found.
        """
        if name not in self.datasets:
            raise DatasetNotFoundError(f"Dataset '{name}' not found.")
            
        return len(self.datasets[name]["tensors"])
