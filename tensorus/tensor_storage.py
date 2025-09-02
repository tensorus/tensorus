if __package__ in (None, ""):
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "tensorus"

import torch
from typing import List, Dict, Callable, Optional, Any, Tuple
import logging
import time
import uuid
import random # Added for sampling
import os
import threading
from pathlib import Path
from io import BytesIO
from contextlib import contextmanager

try:
    from .compression import TensorCompression, CompressionConfig, get_compression_preset
except ImportError:
    TensorCompression = None
    CompressionConfig = None
    get_compression_preset = None

try:
    from .indexing import IndexManager, QueryBuilder, HashIndex, RangeIndex, TensorPropertyIndex
except ImportError:
    IndexManager = None
    QueryBuilder = None
    HashIndex = None
    RangeIndex = None
    TensorPropertyIndex = None

try:
    import boto3  # Optional dependency for S3 support
    from botocore.exceptions import ClientError
except Exception:  # pragma: no cover - boto3 is optional at runtime
    boto3 = None
    ClientError = Exception

# Public exceptions for dataset/tensor lookups
class DatasetNotFoundError(ValueError):
    """Raised when a requested dataset is not found."""

class TensorNotFoundError(ValueError):
    """Raised when a requested tensor record cannot be found."""

class SchemaValidationError(ValueError):
    """Raised when inserted data does not comply with the dataset schema."""

# Simple mapping from string names to Python types for schema validation
_TYPE_MAP = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
}

class TransactionError(Exception):
    """Raised when a transaction cannot be completed."""
    pass

class TensorStorage:
    """
    Manages datasets stored as collections of tensors in memory with transactional support.
    Optionally, can persist datasets to disk if a storage_path is provided.
    """

    def __init__(self, 
                 storage_path: Optional[str] = "tensor_data",
                 compression_config: Optional[CompressionConfig] = None,
                 compression_preset: Optional[str] = None):
        """
        Initializes the TensorStorage.

        If a `storage_path` is provided, datasets will be persisted to disk in that
        directory. Each dataset is stored as a separate '.pt' file. If the path
        is None, or if directory creation fails (e.g., due to permissions),
        the storage will operate in in-memory mode only.

        Args:
            storage_path (Optional[str]): Path to a directory for storing datasets.
                                         If None, storage is in-memory only.
                                         Defaults to "tensor_data".
            compression_config (Optional[CompressionConfig]): Compression configuration.
            compression_preset (Optional[str]): Named compression preset (overrides config).
        """
        self.datasets: Dict[str, Dict[str, List[Any]]] = {}
        self.storage_path: Optional[Path] = None
        self._use_s3: bool = False
        self._s3_bucket: Optional[str] = None
        self._s3_prefix: str = ""
        self._s3_client = None
        
        # Compression support
        self._compression_enabled = TensorCompression is not None
        if self._compression_enabled:
            if compression_preset:
                self.compression_config = get_compression_preset(compression_preset)
            elif compression_config:
                self.compression_config = compression_config
            else:
                self.compression_config = CompressionConfig()  # No compression by default
            self.tensor_compression = self.compression_config.create_tensor_compression()
            logging.info(f"TensorStorage compression enabled: {self.compression_config.compression}/{self.compression_config.quantization}")
        else:
            self.compression_config = None
            self.tensor_compression = None
            logging.warning("TensorStorage compression module not available")
        
        # Indexing support
        self._indexing_enabled = IndexManager is not None
        self._dataset_indexes: Dict[str, IndexManager] = {}
        if self._indexing_enabled:
            logging.info("TensorStorage indexing enabled")
        else:
            logging.warning("TensorStorage indexing module not available")
        
        # Transactional support
        self._transaction_lock = threading.RLock()
        self._active_transactions: Dict[str, Dict[str, Any]] = {}  # transaction_id -> transaction_state
        self._dataset_locks: Dict[str, threading.RLock] = {}  # dataset_name -> lock

        if storage_path:
            # Support S3 URI form: s3://bucket/prefix (prefix optional)
            if isinstance(storage_path, str) and storage_path.lower().startswith("s3://"):
                self._configure_s3_backend(storage_path)
                if self._use_s3:
                    logging.info(
                        f"TensorStorage initialized with S3 backend: bucket='{self._s3_bucket}', prefix='{self._s3_prefix}'"
                    )
                    self._load_all_datasets_from_disk()  # Loads from S3 in this mode
                else:
                    logging.info("TensorStorage initialized (In-Memory only mode).")
            else:
                self.storage_path = Path(storage_path)
                try:
                    self.storage_path.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist
                    logging.info(f"TensorStorage initialized with persistence directory: {self.storage_path}")
                    self._load_all_datasets_from_disk() # Attempt to load existing datasets from disk
                except OSError as e: # Handle errors like permission issues
                    logging.error(f"Error creating storage directory {self.storage_path}: {e}. Falling back to in-memory only mode.")
                    self.storage_path = None # Fallback to in-memory if directory creation fails
        else:
            logging.info("TensorStorage initialized (In-Memory only mode).")
            
        # Initialize dataset locks and indexes for any existing datasets
        for dataset_name in self.datasets.keys():
            if dataset_name not in self._dataset_locks:
                self._dataset_locks[dataset_name] = threading.RLock()
            # Initialize indexes for existing datasets
            if self._indexing_enabled and dataset_name not in self._dataset_indexes:
                self._dataset_indexes[dataset_name] = IndexManager(dataset_name)
                self._rebuild_indexes_for_dataset(dataset_name)
                
    def _decompress_tensor_if_needed(self, tensor_data: Any, metadata: Dict[str, Any]) -> torch.Tensor:
        """Helper method to decompress tensor data if it's compressed."""
        if not metadata.get("compressed", False):
            # Not compressed, return as-is
            if isinstance(tensor_data, torch.Tensor):
                return tensor_data
            elif isinstance(tensor_data, (list, tuple)):
                return torch.tensor(tensor_data)
            else:
                # Handle other data types
                return tensor_data
            
        if not self._compression_enabled or not self.tensor_compression:
            logging.error("Cannot decompress tensor: compression module not available")
            raise RuntimeError("Cannot decompress tensor: compression module not available")
            
        try:
            compression_metadata = metadata.get("compression_metadata", {})
            decompressed_tensor = self.tensor_compression.decompress_tensor(tensor_data, compression_metadata)
            return decompressed_tensor
        except Exception as e:
            logging.error(f"Failed to decompress tensor {metadata.get('record_id', 'unknown')}: {e}")
            raise RuntimeError(f"Failed to decompress tensor: {e}")

    def _rebuild_indexes_for_dataset(self, dataset_name: str) -> None:
        """Rebuild all indexes for a dataset from existing data."""
        if not self._indexing_enabled:
            return
        
        index_manager = self._dataset_indexes.get(dataset_name)
        if not index_manager:
            return
        
        try:
            # Get all records for this dataset
            dataset = self.datasets.get(dataset_name, {})
            
            # Initialize the basic indexes if they don't exist
            if not index_manager.get_index("record_id"):
                from tensorus.indexing import HashIndex
                index_manager.add_index("record_id", HashIndex("record_id"))
            if not index_manager.get_index("timestamp"):
                from tensorus.indexing import RangeIndex
                index_manager.add_index("timestamp", RangeIndex("timestamp"))
            
            # Rebuild indexes from existing data
            for record_id, record_data in dataset.items():
                if isinstance(record_data, dict):
                    metadata = record_data.get("metadata", {})
                    tensor_data = record_data.get("tensor")
                    
                    # Update record_id index
                    record_id_index = index_manager.get_index("record_id")
                    if record_id_index:
                        record_id_index.insert(record_id, record_id)
                    
                    # Update timestamp index if available
                    if "timestamp" in metadata:
                        timestamp_index = index_manager.get_index("timestamp")
                        if timestamp_index:
                            timestamp_index.insert(record_id, metadata["timestamp"])
                    
                    # Update metadata indexes
                    for key, value in metadata.items():
                        if key not in ["tensor", "compressed", "compression_metadata"]:
                            # Create or update index for this metadata key
                            if not index_manager.get_index(f"metadata_{key}"):
                                if isinstance(value, (int, float)):
                                    from tensorus.indexing import RangeIndex
                                    index_manager.add_index(f"metadata_{key}", RangeIndex(f"metadata_{key}"))
                                else:
                                    from tensorus.indexing import HashIndex
                                    index_manager.add_index(f"metadata_{key}", HashIndex(f"metadata_{key}"))
                            
                            meta_index = index_manager.get_index(f"metadata_{key}")
                            if meta_index:
                                meta_index.insert(record_id, value)
                    
                    # Update tensor property indexes if tensor is available
                    if tensor_data is not None:
                        try:
                            # Decompress if needed
                            tensor = self._decompress_tensor_if_needed(tensor_data, metadata)
                            
                            # Add tensor property indexes
                            if not index_manager.get_index("tensor_shape"):
                                from tensorus.indexing import TensorPropertyIndex
                                shape_extractor = lambda t: tuple(t.shape)
                                index_manager.add_index("tensor_shape", TensorPropertyIndex("tensor_shape", shape_extractor))
                            if not index_manager.get_index("tensor_dtype"):
                                from tensorus.indexing import HashIndex
                                index_manager.add_index("tensor_dtype", HashIndex("tensor_dtype"))
                            
                            # Update tensor property indexes
                            shape_index = index_manager.get_index("tensor_shape")
                            if shape_index:
                                shape_index.insert(record_id, None, tensor)
                            
                            dtype_index = index_manager.get_index("tensor_dtype")
                            if dtype_index:
                                dtype_index.insert(record_id, str(tensor.dtype))
                                
                        except Exception as e:
                            logging.warning(f"Failed to index tensor properties for {record_id}: {e}")
                            
            logging.info(f"Successfully rebuilt indexes for dataset '{dataset_name}'")
            
        except Exception as e:
            logging.error(f"Failed to rebuild indexes for dataset '{dataset_name}': {e}")

    def _get_or_create_index_manager(self, dataset_name: str):
        """Get or create an index manager for the specified dataset."""
        if not self._indexing_enabled:
            return None
        
        if dataset_name not in self._dataset_indexes:
            from tensorus.indexing import IndexManager
            self._dataset_indexes[dataset_name] = IndexManager(dataset_name)
        
        return self._dataset_indexes[dataset_name]

    # --- S3 Helpers ---
    def _configure_s3_backend(self, uri: str) -> None:
        """Parse S3 URI and initialize client if boto3 is available."""
        if boto3 is None:
            logging.error("boto3 is not installed. Cannot use S3 backend. Falling back to in-memory mode.")
            self._use_s3 = False
            return
        # Strip scheme and split bucket/prefix
        s3_path = uri[5:]  # remove 's3://'
        parts = s3_path.split('/', 1)
        bucket = parts[0].strip()
        prefix = parts[1].strip('/') + '/' if len(parts) > 1 and parts[1] else ""
        if not bucket:
            logging.error("Invalid S3 URI: bucket missing. Falling back to in-memory mode.")
            self._use_s3 = False
            return
        try:
            self._s3_client = boto3.client("s3")
            # Optionally verify access by a lightweight call
            # self._s3_client.list_buckets()
            self._s3_bucket = bucket
            self._s3_prefix = prefix
            self._use_s3 = True
        except Exception as e:  # pragma: no cover - environment dependent
            logging.error(f"Failed to initialize S3 client: {e}. Falling back to in-memory mode.")
            self._use_s3 = False

    def _get_dataset_lock(self, dataset_name: str) -> threading.RLock:
        """Get or create a lock for a dataset."""
        if dataset_name not in self._dataset_locks:
            with self._transaction_lock:
                if dataset_name not in self._dataset_locks:
                    self._dataset_locks[dataset_name] = threading.RLock()
        return self._dataset_locks[dataset_name]
        
    @contextmanager
    def transaction(self, dataset_names: List[str], transaction_id: Optional[str] = None):
        """Context manager for atomic multi-tensor operations.
        
        Args:
            dataset_names: List of dataset names involved in the transaction
            transaction_id: Optional custom transaction ID
            
        Yields:
            str: Transaction ID for tracking
            
        Raises:
            TransactionError: If transaction cannot be completed
        """
        if transaction_id is None:
            transaction_id = str(uuid.uuid4())
            
        # Sort dataset names to prevent deadlocks
        sorted_datasets = sorted(set(dataset_names))
        locks = [self._get_dataset_lock(name) for name in sorted_datasets]
        
        # Acquire locks in sorted order
        acquired_locks = []
        try:
            for lock in locks:
                lock.acquire()
                acquired_locks.append(lock)
                
            # Initialize transaction state
            with self._transaction_lock:
                self._active_transactions[transaction_id] = {
                    'datasets': sorted_datasets,
                    'operations': [],
                    'rollback_data': {},
                    'committed': False
                }
                
            # Store rollback data
            for dataset_name in sorted_datasets:
                if dataset_name in self.datasets:
                    self._active_transactions[transaction_id]['rollback_data'][dataset_name] = {
                        'tensors': [t.clone() for t in self.datasets[dataset_name]['tensors']],
                        'metadata': [m.copy() for m in self.datasets[dataset_name]['metadata']],
                        'schema': self.datasets[dataset_name].get('schema')
                    }
                    
            yield transaction_id
            
            # Commit transaction
            self._commit_transaction(transaction_id)
            
        except Exception as e:
            # Rollback transaction
            self._rollback_transaction(transaction_id)
            raise TransactionError(f"Transaction {transaction_id} failed: {str(e)}")
        finally:
            # Release locks in reverse order
            for lock in reversed(acquired_locks):
                lock.release()
            # Clean up transaction state
            with self._transaction_lock:
                self._active_transactions.pop(transaction_id, None)
                
    def _commit_transaction(self, transaction_id: str) -> None:
        """Commit a transaction by persisting all changes."""
        transaction = self._active_transactions.get(transaction_id)
        if not transaction:
            return
            
        # Save all modified datasets
        for dataset_name in transaction['datasets']:
            if dataset_name in self.datasets:
                self._save_dataset(dataset_name)
                
        transaction['committed'] = True
        logging.debug(f"Transaction {transaction_id} committed successfully")
        
    def _rollback_transaction(self, transaction_id: str) -> None:
        """Rollback a transaction by restoring original data."""
        transaction = self._active_transactions.get(transaction_id)
        if not transaction or transaction.get('committed'):
            return
            
        # Restore original data
        for dataset_name, rollback_data in transaction['rollback_data'].items():
            if dataset_name in self.datasets:
                self.datasets[dataset_name] = rollback_data
                
        logging.debug(f"Transaction {transaction_id} rolled back")
        
    def _s3_key_for_dataset(self, dataset_name: str) -> str:
        return f"{self._s3_prefix}{dataset_name}.pt"

    def _save_dataset(self, dataset_name: str) -> None:
        """
        Internal helper to save a single dataset to a .pt file.

        This method is called by public methods that modify dataset content
        (e.g., insert, delete_tensor, create_dataset) if persistence is enabled.
        Tensors are cloned and moved to CPU before saving to ensure data integrity
        and broad compatibility.

        Args:
            dataset_name (str): The name of the dataset to save.
        """
        if (not self.storage_path and not self._use_s3) or dataset_name not in self.datasets:
            # Do nothing if persistence is not enabled or dataset doesn't exist (e.g., during deletion)
            return
        # Prepare data for saving with compression support
        dataset = self.datasets[dataset_name]
        
        # Handle tensors based on compression state
        tensors_data = []
        for i, tensor_data in enumerate(dataset["tensors"]):
            metadata = dataset["metadata"][i]
            if metadata.get("compressed", False):
                # Already compressed data (bytes)
                tensors_data.append(tensor_data)
            else:
                # Raw tensor data
                if isinstance(tensor_data, torch.Tensor):
                    tensors_data.append(tensor_data.clone().cpu())
                else:
                    tensors_data.append(tensor_data)
            
        data_to_save = {
            "tensors": tensors_data,
            "metadata": dataset["metadata"],
            "schema": dataset.get("schema"),
            "compression_config": self.compression_config.to_dict() if self.compression_config else None
        }

        try:
            if self._use_s3:
                key = self._s3_key_for_dataset(dataset_name)
                buffer = BytesIO()
                torch.save(data_to_save, buffer)
                buffer.seek(0)
                self._s3_client.put_object(Bucket=self._s3_bucket, Key=key, Body=buffer.getvalue())
                logging.info(f"Dataset '{dataset_name}' saved successfully to s3://{self._s3_bucket}/{key}")
            else:
                file_path = self.storage_path / f"{dataset_name}.pt"  # type: ignore[union-attr]
                torch.save(data_to_save, file_path)
                logging.info(f"Dataset '{dataset_name}' saved successfully to {file_path}")
        except Exception as e: # Catch a broad range of exceptions during file I/O or serialization
            if self._use_s3:
                logging.error(f"Error saving dataset '{dataset_name}' to S3: {e}")
            else:
                logging.error(f"Error saving dataset '{dataset_name}' to {file_path}: {e}")

    def _load_all_datasets_from_disk(self) -> None:
        """
        Internal helper to load all datasets from .pt files in the storage_path directory.

        This method is called during initialization if persistence is enabled.
        It scans the `storage_path` for files ending with '.pt', attempts to load
        them, and populates the in-memory `self.datasets` dictionary.
        """
        if not self.storage_path and not self._use_s3:
            return
        if self._use_s3:
            logging.info(f"Scanning for existing datasets in s3://{self._s3_bucket}/{self._s3_prefix}...")
            try:
                paginator = self._s3_client.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=self._s3_bucket, Prefix=self._s3_prefix):
                    for obj in page.get("Contents", []) or []:
                        key: str = obj.get("Key", "")
                        if key.endswith(".pt"):
                            dataset_name = Path(key).stem
                            try:
                                resp = self._s3_client.get_object(Bucket=self._s3_bucket, Key=key)
                                body = resp["Body"].read()
                                buffer = BytesIO(body)
                                loaded_data = torch.load(buffer, map_location="cpu")
                                if isinstance(loaded_data, dict) and "tensors" in loaded_data and "metadata" in loaded_data:
                                    self.datasets[dataset_name] = {
                                        "tensors": loaded_data["tensors"],
                                        "metadata": loaded_data["metadata"],
                                        "schema": loaded_data.get("schema"),
                                        "compression_config": loaded_data.get("compression_config")
                                    }
                                    logging.info(f"Dataset '{dataset_name}' loaded successfully from s3://{self._s3_bucket}/{key}")
                                else:
                                    logging.warning(f"Object s3://{self._s3_bucket}/{key} is not a valid dataset file. Skipping.")
                            except Exception as e:  # pragma: no cover - environment dependent
                                logging.error(f"Error loading dataset from s3://{self._s3_bucket}/{key}: {e}")
            except Exception as e:  # pragma: no cover - environment dependent
                logging.error(f"Failed to list S3 objects: {e}")
        else:
            logging.info(f"Scanning for existing datasets in {self.storage_path}...")
            for file_path in self.storage_path.glob("*.pt"): # type: ignore[union-attr]
                dataset_name = file_path.stem
                try:
                    loaded_data = torch.load(file_path)
                    if isinstance(loaded_data, dict) and "tensors" in loaded_data and "metadata" in loaded_data:
                        self.datasets[dataset_name] = {
                            "tensors": loaded_data["tensors"],
                            "metadata": loaded_data["metadata"],
                            "schema": loaded_data.get("schema"),
                            "compression_config": loaded_data.get("compression_config")
                        }
                        logging.info(f"Dataset '{dataset_name}' loaded successfully from {file_path}")
                    else:
                        logging.warning(f"File {file_path} does not appear to be a valid dataset file (missing keys or wrong format). Skipping.")
                except Exception as e:
                    logging.error(f"Error loading dataset from {file_path}: {e}. The file might be corrupted or not a PyTorch file.")

    def list_datasets(self) -> List[str]:
        """
        Lists the names of all datasets currently stored.

        Returns:
            List[str]: A list of dataset names.
        """
        dataset_names = list(self.datasets.keys())
        logging.info(f"Available datasets: {dataset_names}")
        return dataset_names
 
    def dataset_exists(self, name: str) -> bool:
        """Check whether a dataset exists either in memory or on disk."""
        if name in self.datasets:
            return True
        if self._use_s3:
            try:
                self._s3_client.head_object(Bucket=self._s3_bucket, Key=self._s3_key_for_dataset(name))
                return True
            except ClientError:
                return False
        if self.storage_path:
            file_path = self.storage_path / f"{name}.pt"  # type: ignore[union-attr]
            return file_path.exists()
        return False

    def count(self, dataset_name: str) -> int:
        """Return the number of records stored in a dataset without loading tensors.

        This method relies only on metadata length for in-memory datasets and
        falls back to reading metadata from disk if the dataset is not loaded.

        Args:
            dataset_name: The name of the dataset to count.

        Returns:
            int: Number of records in the dataset.

        Raises:
            DatasetNotFoundError: If the dataset is not found.
        """
        if dataset_name in self.datasets:
            return len(self.datasets[dataset_name]["metadata"])

        if self._use_s3:
            try:
                key = self._s3_key_for_dataset(dataset_name)
                resp = self._s3_client.get_object(Bucket=self._s3_bucket, Key=key)
                body = resp["Body"].read()
                buffer = BytesIO(body)
                data = torch.load(buffer, map_location="cpu")
                if isinstance(data, dict) and "metadata" in data:
                    return len(data["metadata"])
            except Exception as e:
                logging.error(f"Error loading dataset '{dataset_name}' from S3 for count: {e}")
            logging.error(f"Dataset '{dataset_name}' not found for count on S3.")
            raise DatasetNotFoundError(f"Dataset '{dataset_name}' not found.")

        if self.storage_path:
            file_path = self.storage_path / f"{dataset_name}.pt"  # type: ignore[union-attr]
            if file_path.exists():
                try:
                    data = torch.load(file_path, map_location="cpu")
                    if isinstance(data, dict) and "metadata" in data:
                        return len(data["metadata"])
                except Exception as e:
                    logging.error(
                        f"Error loading dataset '{dataset_name}' for count: {e}"
                    )
            logging.error(f"Dataset '{dataset_name}' not found for count on disk.")
            raise DatasetNotFoundError(f"Dataset '{dataset_name}' not found.")

        logging.error(f"Dataset '{dataset_name}' not found for count.")
        raise DatasetNotFoundError(f"Dataset '{dataset_name}' not found.")

    def create_dataset(self, name: str, schema: Optional[Dict[str, Any]] = None) -> None:
        """
        Creates a new, empty dataset. Optionally associates a schema used for
        validating future insertions.

        Args:
            name (str): The unique name for the new dataset.
            schema (Optional[Dict[str, Any]]): Optional schema definition with
                keys ``shape``, ``dtype`` and ``metadata``.

        Raises:
            ValueError: If a dataset with the same name already exists.
        """
        with self._get_dataset_lock(name):
            if name in self.datasets:
                logging.warning(f"Attempted to create dataset '{name}' which already exists.")
                raise ValueError(f"Dataset '{name}' already exists.")

            self.datasets[name] = {"tensors": [], "metadata": [], "schema": schema}
            
            # Initialize indexes for new dataset
            if self._indexing_enabled:
                self._get_or_create_index_manager(name)
            
            logging.info(f"Dataset '{name}' created successfully.")
            self._save_dataset(name) # Save after creation
 
    def insert(self, name: str, tensor: torch.Tensor, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Inserts a tensor into a specified dataset.

        The method generates a unique `record_id` and a `version` number for the
        tensor. User-provided `metadata` is copied and can include custom fields.
        If `record_id` is present in the user-provided `metadata`, it will be
        ignored and replaced by a system-generated one (a warning will be logged).
        Users can specify `timestamp_utc`, `shape`, and `dtype`; otherwise, these
        are generated automatically.

        Args:
            name (str): The name of the dataset to insert into.
            tensor (torch.Tensor): The PyTorch tensor to insert.
            metadata (Optional[Dict[str, Any]]): Optional dictionary containing
                                                metadata about the tensor.

        Returns:
            str: A unique ID assigned to the inserted tensor record. This ID is
                 system-generated.

        Raises:
            DatasetNotFoundError: If the dataset `name` is not found.
            TypeError: If the provided `tensor` object is not a PyTorch tensor.
        """
        if name not in self.datasets:
            logging.error(f"Dataset '{name}' not found for insertion.")
            raise DatasetNotFoundError(f"Dataset '{name}' not found.")

        if not isinstance(tensor, torch.Tensor):
            logging.error(f"Attempted to insert non-tensor data into dataset '{name}'.")
            raise TypeError("Data to be inserted must be a torch.Tensor.")

        # Ensure metadata consistency if not provided
        if metadata is None:
            metadata = {}
        else:
            # Make a copy to avoid modifying the caller's dictionary
            metadata = metadata.copy()

        # Validate against dataset schema if present
        schema = self.datasets[name].get("schema")
        if schema:
            if "shape" in schema and tuple(tensor.shape) != tuple(schema["shape"]):
                raise SchemaValidationError(
                    f"Tensor shape {tuple(tensor.shape)} does not match schema shape {schema['shape']} for dataset '{name}'."
                )
            if "dtype" in schema and str(tensor.dtype) != schema["dtype"]:
                raise SchemaValidationError(
                    f"Tensor dtype {tensor.dtype} does not match schema dtype {schema['dtype']} for dataset '{name}'."
                )
            if "metadata" in schema:
                for field, type_name in schema["metadata"].items():
                    if field not in metadata:
                        raise SchemaValidationError(
                            f"Metadata missing required field '{field}' for dataset '{name}'."
                        )
                    expected_type = _TYPE_MAP.get(type_name)
                    if expected_type and not isinstance(metadata[field], expected_type):
                        raise SchemaValidationError(
                            f"Metadata field '{field}' expected type {type_name}, got {type(metadata[field]).__name__}."
                        )

        # Generate essential metadata fields
        system_record_id = str(uuid.uuid4())
        current_version = len(self.datasets[name]["tensors"]) + 1

        # Use provided record_id if given; otherwise fall back to system generated
        user_record_id = metadata.get("record_id")
        if user_record_id is None:
            metadata_record_id = system_record_id
        else:
            metadata_record_id = user_record_id
        
        # Initialize final_metadata with the (potentially modified) user's metadata.
        # This ensures custom fields are preserved.
        final_metadata = metadata # `metadata` is already a copy or a new dict.

        # Set or overwrite essential fields in final_metadata.
        final_metadata["record_id"] = metadata_record_id
        
        # For other standard fields, use user's value if provided, otherwise generate default.
        final_metadata["timestamp_utc"] = metadata.get("timestamp_utc", time.time())
        final_metadata["shape"] = metadata.get("shape", tuple(tensor.shape))
        final_metadata["dtype"] = metadata.get("dtype", str(tensor.dtype))
        final_metadata["version"] = current_version # Version is system-controlled.

        # --- Compression and Storage ---
        # Apply compression if enabled and algorithms are not 'none'
        should_compress = (
            self._compression_enabled and 
            self.tensor_compression and
            (self.compression_config.compression != "none" or self.compression_config.quantization != "none")
        )
        
        if should_compress:
            try:
                compressed_bytes, compression_metadata = self.tensor_compression.compress_tensor(tensor.clone())
                # Store compressed data instead of raw tensor
                tensor_to_store = compressed_bytes
                # Add compression metadata to tensor metadata
                final_metadata.update({
                    "compressed": True,
                    "compression_metadata": compression_metadata
                })
                logging.debug(f"Tensor compressed: {compression_metadata['original_size']} -> {compression_metadata['compressed_size']} bytes")
            except Exception as e:
                logging.warning(f"Compression failed for tensor {metadata_record_id}: {e}. Storing uncompressed.")
                tensor_to_store = tensor.clone()
                final_metadata["compressed"] = False
        else:
            # Store raw tensor (backward compatibility)
            tensor_to_store = tensor.clone()
            final_metadata["compressed"] = False

        self.datasets[name]["tensors"].append(tensor_to_store)
        self.datasets[name]["metadata"].append(final_metadata)
        
        # Update indexes
        if self._indexing_enabled:
            index_manager = self._get_or_create_index_manager(name)
            if index_manager:
                try:
                    index_manager.insert_record(final_metadata['record_id'], final_metadata, tensor)
                except Exception as e:
                    logging.warning(f"Failed to update indexes for record '{final_metadata['record_id']}': {e}")
        
        # Log the record_id actually stored (either user provided or generated)
        logging.debug(
            f"Tensor with shape {tuple(tensor.shape)} inserted into dataset '{name}'. Record ID: {final_metadata['record_id']}"
        )
        self._save_dataset(name) # Persist dataset changes if storage_path is configured
        return final_metadata['record_id'] # Return the actual record_id used for storage


    def get_dataset(self, name: str) -> List[torch.Tensor]:
        """
        Retrieves all tensors from a specified dataset.

        Args:
            name (str): The name of the dataset to retrieve.

        Returns:
            List[torch.Tensor]: A list of all tensors in the dataset.

        Raises:
            DatasetNotFoundError: If the dataset `name` is not found.
        """
        if name not in self.datasets:
            logging.error(f"Dataset '{name}' not found for retrieval.")
            raise DatasetNotFoundError(f"Dataset '{name}' not found.")

        logging.debug(f"Retrieving all {len(self.datasets[name]['tensors'])} tensors from dataset '{name}'.")
        
        # Decompress tensors if needed
        tensors = []
        for i, tensor_data in enumerate(self.datasets[name]["tensors"]):
            metadata = self.datasets[name]["metadata"][i]
            decompressed_tensor = self._decompress_tensor_if_needed(tensor_data, metadata)
            tensors.append(decompressed_tensor)
        
        return tensors

    def get_dataset_with_metadata(self, name: str) -> List[Dict[str, Any]]:
        """
        Retrieves all tensors and their metadata from a specified dataset.

        Args:
            name (str): The name of the dataset to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                 contains a 'tensor' (torch.Tensor) and its
                                 associated 'metadata' (Dict[str, Any]).

        Raises:
            DatasetNotFoundError: If the dataset `name` is not found.
        """
        if name not in self.datasets:
            logging.error(f"Dataset '{name}' not found for retrieval with metadata.")
            raise DatasetNotFoundError(f"Dataset '{name}' not found.")

        logging.debug(f"Retrieving all {len(self.datasets[name]['tensors'])} tensors and metadata from dataset '{name}'.")

        results = []
        for tensor_data, meta in zip(self.datasets[name]["tensors"], self.datasets[name]["metadata"]):
            decompressed_tensor = self._decompress_tensor_if_needed(tensor_data, meta)
            results.append({"tensor": decompressed_tensor, "metadata": meta})
        return results


    def query(self, name: str, query_fn: Callable[[torch.Tensor, Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
        """
        Queries a dataset using a function that filters tensors based on the
        tensor data itself and/or its metadata.

        Args:
            name: The name of the dataset to query.
            query_fn (Callable[[torch.Tensor, Dict[str, Any]], bool]):
                      A callable that takes a tensor and its metadata dictionary
                      as input and returns True if the tensor should be included
                      in the result, False otherwise.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing a 'tensor'
                                 and its 'metadata' for records that satisfy the
                                 query function.

        Raises:
            DatasetNotFoundError: If the dataset `name` is not found.
            TypeError: If `query_fn` is not a callable function.
        """
        if name not in self.datasets:
            logging.error(f"Dataset '{name}' not found for querying.")
            raise DatasetNotFoundError(f"Dataset '{name}' not found.")

        if not callable(query_fn):
             logging.error(f"Provided query_fn is not callable for dataset '{name}'.")
             raise TypeError("query_fn must be a callable function.")

        logging.debug(f"Querying dataset '{name}' with custom function.")
        results = []
        # --- Optimized Querying with Decompression ---
        # In a real system, metadata indexing would speed this up significantly.
        # Query might operate directly on chunks or specific metadata fields first.
        # ----------------------------------------
        for tensor_data, meta in zip(self.datasets[name]["tensors"], self.datasets[name]["metadata"]):
            try:
                decompressed_tensor = self._decompress_tensor_if_needed(tensor_data, meta)
                if query_fn(decompressed_tensor, meta):
                    results.append({"tensor": decompressed_tensor, "metadata": meta})
            except Exception as e:
                logging.warning(f"Error executing query_fn on tensor {meta.get('record_id', 'N/A')} in dataset '{name}': {e}")
                # Optionally re-raise or continue based on desired strictness
                continue

        logging.info(f"Query on dataset '{name}' returned {len(results)} results.")
        return results # Returns List of dictionaries, each containing 'tensor' and 'metadata'


    def get_tensor_by_id(self, name: str, record_id: str) -> Dict[str, Any]:
        """
        Retrieves a specific tensor and its metadata by its unique record ID.

        Args:
            name (str): The name of the dataset.
            record_id (str): The unique ID of the record to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing the 'tensor' and 'metadata'.

        Raises:
            DatasetNotFoundError: If the dataset `name` is not found.
            TensorNotFoundError: If the `record_id` is not found in the dataset.
        """
        if name not in self.datasets:
            logging.error(f"Dataset '{name}' not found for get_tensor_by_id.")
            raise DatasetNotFoundError(f"Dataset '{name}' not found.")

        # Use index for O(1) lookup if available
        if self._indexing_enabled:
            index_manager = self._get_or_create_index_manager(name)
            if index_manager:
                record_id_index = index_manager.get_index("record_id")
                if record_id_index:
                    matching_ids = record_id_index.lookup(record_id)
                    if matching_ids:
                        # Find the record in the dataset
                        for i, meta in enumerate(self.datasets[name]["metadata"]):
                            if meta.get("record_id") == record_id:
                                tensor_data = self.datasets[name]["tensors"][i]
                                decompressed_tensor = self._decompress_tensor_if_needed(tensor_data, meta)
                                logging.debug(f"Tensor with record_id '{record_id}' found in dataset '{name}' via index.")
                                return {"tensor": decompressed_tensor, "metadata": meta}
        
        # Fallback to linear search (for backward compatibility or when indexing is disabled)
        for tensor_data, meta in zip(self.datasets[name]["tensors"], self.datasets[name]["metadata"]):
             if meta.get("record_id") == record_id:
                 logging.debug(f"Tensor with record_id '{record_id}' found in dataset '{name}' via linear search.")
                 decompressed_tensor = self._decompress_tensor_if_needed(tensor_data, meta)
                 return {"tensor": decompressed_tensor, "metadata": meta}

        logging.warning(f"Tensor with record_id '{record_id}' not found in dataset '{name}'.")
        raise TensorNotFoundError(f"Tensor {record_id} not found in dataset {name}.")

    # --- ADDED METHOD (from Step 3) ---
    def sample_dataset(self, name: str, n_samples: int) -> List[Dict[str, Any]]:
        """
        Retrieves a random sample of records (tensor and metadata) from a dataset.

        Args:
            name (str): The name of the dataset to sample from.
            n_samples (int): The number of samples to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing 'tensor'
                                 and 'metadata' for the sampled records. Returns
                                 fewer than `n_samples` if the dataset size is
                                 smaller than `n_samples`. Returns an empty list
                                 if `n_samples` is non-positive.

        Raises:
            DatasetNotFoundError: If the dataset `name` is not found.
        """
        if name not in self.datasets:
            logging.error(f"Dataset '{name}' not found for sampling.")
            raise DatasetNotFoundError(f"Dataset '{name}' not found.")

        dataset_size = len(self.datasets[name]["tensors"])
        if n_samples <= 0:
             return []
        if n_samples >= dataset_size:
            logging.warning(f"Requested {n_samples} samples from dataset '{name}' which only has {dataset_size} items. Returning all items shuffled.")
            # Return all items shuffled if n_samples >= dataset_size
            indices = list(range(dataset_size))
            random.shuffle(indices)
        else:
            indices = random.sample(range(dataset_size), n_samples)

        logging.debug(f"Sampling {len(indices)} records from dataset '{name}'.")

        # In-memory sampling is easy. For persistent storage, this would
        # likely involve optimized queries or index lookups.
        sampled_records = []
        for i in indices:
            tensor_data = self.datasets[name]["tensors"][i]
            metadata = self.datasets[name]["metadata"][i]
            decompressed_tensor = self._decompress_tensor_if_needed(tensor_data, metadata)
            sampled_records.append({
                "tensor": decompressed_tensor,
                "metadata": metadata
            })

        return sampled_records

    def get_records_paginated(self, name: str, offset: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve a slice of records from a dataset.

        Args:
            name: Dataset name.
            offset: Starting index of the slice.
            limit: Maximum number of records to return.

        Returns:
            List of dictionaries each containing ``tensor`` and ``metadata``.

        Raises:
            DatasetNotFoundError: If the dataset is not found.
        """
        if name not in self.datasets:
            logging.error(f"Dataset '{name}' not found for pagination.")
            raise DatasetNotFoundError(f"Dataset '{name}' not found.")

        tensors = self.datasets[name]["tensors"]
        metadata = self.datasets[name]["metadata"]
        end = offset + limit if limit is not None else None
        sliced = list(zip(tensors, metadata))[offset:end]
        return [{"tensor": self._decompress_tensor_if_needed(tensor_data, meta), "metadata": meta} for tensor_data, meta in sliced]

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
            # If persistence is enabled, attempt to delete the dataset file
            if self._use_s3:
                key = self._s3_key_for_dataset(name)
                try:
                    self._s3_client.delete_object(Bucket=self._s3_bucket, Key=key)
                    logging.info(f"Dataset object s3://{self._s3_bucket}/{key} deleted successfully.")
                except Exception as e:  # pragma: no cover - environment dependent
                    logging.error(f"Error deleting dataset object s3://{self._s3_bucket}/{key}: {e}")
            elif self.storage_path:
                file_path = self.storage_path / f"{name}.pt"  # type: ignore[union-attr]
                try:
                    if file_path.exists():
                        file_path.unlink() # Delete the .pt file
                        logging.info(f"Dataset file {file_path} deleted successfully from disk.")
                    else:
                        # This case might occur if a file was manually deleted or never saved
                        logging.warning(f"Dataset file {file_path} not found on disk for deletion.")
                except OSError as e: # Catch potential file system errors
                    logging.error(f"Error deleting dataset file {file_path} from disk: {e}. "
                                  "The dataset will still be removed from memory.")
                    # Proceed to delete from memory even if file deletion fails, but log the error.

            # Delete from in-memory storage
            del self.datasets[name]
            logging.warning(f"Dataset '{name}' has been permanently deleted from memory.")
            return True
        else:
            logging.warning(f"Attempted to delete non-existent dataset '{name}'.")
            raise DatasetNotFoundError(f"Dataset '{name}' not found.")

    def update_tensor_metadata(self, dataset_name: str, record_id: str, new_metadata: Dict[str, Any]) -> bool:
        """
        Replace the metadata for a specific tensor in a dataset.

        The provided ``new_metadata`` dictionary completely overwrites the
        existing metadata for the record, except that the original
        ``record_id`` is always preserved.  If ``record_id`` is present in
        ``new_metadata`` it is ignored and a warning is logged.

        Args:
            dataset_name (str): The name of the dataset containing the tensor.
            record_id (str): The unique ID of the tensor record to update.
            new_metadata (Dict[str, Any]): The new metadata dictionary to store
                for the tensor. ``record_id`` inside this dictionary (if any) is
                ignored.

        Returns:
            bool: True if the metadata was updated successfully.

        Raises:
            DatasetNotFoundError: If the dataset `dataset_name` is not found.
            TensorNotFoundError: If `record_id` is not found in the dataset.
        """
        if dataset_name not in self.datasets:
            logging.warning(
                f"Dataset '{dataset_name}' not found for metadata update of record '{record_id}'."
            )
            raise DatasetNotFoundError(f"Dataset '{dataset_name}' not found.")

        dataset = self.datasets[dataset_name]
        found_record = False
        for i, meta_item in enumerate(dataset["metadata"]):
            if meta_item.get("record_id") == record_id:
                # Prevent changing the record_id
                if "record_id" in new_metadata:
                    logging.warning(
                        f"Attempt to change 'record_id' for record '{record_id}' in dataset '{dataset_name}' "
                        f"during metadata update was ignored."
                    )
                    metadata_replacement = new_metadata.copy()
                    del metadata_replacement["record_id"]
                else:
                    metadata_replacement = new_metadata.copy()

                # Overwrite the metadata dictionary while keeping the original record_id
                dataset["metadata"][i] = {"record_id": record_id, **metadata_replacement}
                logging.info(
                    f"Metadata replaced for record '{record_id}' in dataset '{dataset_name}'."
                )
                self._save_dataset(dataset_name)
                found_record = True
                break
        
        if not found_record:
            logging.warning(
                f"Record '{record_id}' not found in dataset '{dataset_name}' for metadata update."
            )
            raise TensorNotFoundError(
                f"Tensor '{record_id}' not found in dataset '{dataset_name}'."
            )
        return True

    def delete_tensor(self, dataset_name: str, record_id: str) -> bool:
        """
        Deletes a specific tensor and its metadata from a dataset by its record ID.

        Args:
            dataset_name (str): The name of the dataset from which to delete the tensor.
            record_id (str): The unique ID of the tensor record to delete.

        Returns:
            bool: True if the tensor was deleted successfully.

        Raises:
            DatasetNotFoundError: If the dataset `dataset_name` is not found.
            TensorNotFoundError: If `record_id` is not found in the dataset.
        """
        if dataset_name not in self.datasets:
            logging.warning(
                f"Dataset '{dataset_name}' not found for deletion of record '{record_id}'."
            )
            raise DatasetNotFoundError(f"Dataset '{dataset_name}' not found.")

        dataset = self.datasets[dataset_name]
        for i, meta in enumerate(dataset["metadata"]):
            if meta.get("record_id") == record_id:
                # Get tensor data for index cleanup
                tensor_data = dataset["tensors"][i]
                
                # Update indexes before deletion
                if self._indexing_enabled:
                    index_manager = self._get_or_create_index_manager(dataset_name)
                    if index_manager:
                        try:
                            # Decompress tensor if needed for index cleanup
                            tensor = self._decompress_tensor_if_needed(tensor_data, meta)
                            index_manager.delete_record(record_id, meta, tensor)
                        except Exception as e:
                            logging.warning(f"Failed to update indexes during deletion of record '{record_id}': {e}")
                
                del dataset["tensors"][i]
                del dataset["metadata"][i]
                logging.info(f"Tensor record '{record_id}' deleted from dataset '{dataset_name}'.")
                self._save_dataset(dataset_name) # Save after tensor deletion
                return True

        logging.warning(f"Record '{record_id}' not found in dataset '{dataset_name}' for deletion.")
        raise TensorNotFoundError(f"Tensor {record_id} not found in dataset {dataset_name}.")
    def batch_insert(self, dataset_name: str, tensors_and_metadata: List[Tuple[torch.Tensor, Optional[Dict[str, Any]]]]) -> List[str]:
        """Insert multiple tensors atomically within a single transaction.
        
        Args:
            dataset_name: Name of the target dataset
            tensors_and_metadata: List of (tensor, metadata) tuples
            
        Returns:
            List of record IDs for inserted tensors
            
        Raises:
            TransactionError: If the batch insertion fails
        """
        with self.transaction([dataset_name]):
            record_ids = []
            for tensor, metadata in tensors_and_metadata:
                record_id = self.insert(dataset_name, tensor, metadata)
                record_ids.append(record_id)
            return record_ids
            
    def batch_update_metadata(self, dataset_name: str, updates: List[Tuple[str, Dict[str, Any]]]) -> bool:
        """Update metadata for multiple tensors atomically.
        
        Args:
            dataset_name: Name of the target dataset
            updates: List of (record_id, new_metadata) tuples
            
        Returns:
            bool: True if all updates succeeded
            
        Raises:
            TransactionError: If any update fails
        """
        with self.transaction([dataset_name]):
            for record_id, new_metadata in updates:
                self.update_tensor_metadata(dataset_name, record_id, new_metadata)
            return True
            
    def batch_delete_tensors(self, dataset_name: str, record_ids: List[str]) -> bool:
        """Delete multiple tensors atomically.
        
        Args:
            dataset_name: Name of the target dataset
            record_ids: List of record IDs to delete
            
        Returns:
            bool: True if all deletions succeeded
            
        Raises:
            TransactionError: If any deletion fails
        """
        with self.transaction([dataset_name]):
            for record_id in record_ids:
                self.delete_tensor(dataset_name, record_id)
            return True
            
    # === Compression Management Methods ===
    
    def set_compression_config(self, config: 'CompressionConfig') -> None:
        """Update compression configuration.
        
        Args:
            config: New compression configuration
        """
        if not self._compression_enabled:
            raise RuntimeError("Compression module not available")
            
        self.compression_config = config
        self.tensor_compression = config.create_tensor_compression()
        logging.info(f"Compression config updated: {config.compression}/{config.quantization}")
    
    def set_compression_preset(self, preset: str) -> None:
        """Set compression using a preset.
        
        Args:
            preset: Named preset (none, fast, balanced, maximum, fp16_only, int8_only)
        """
        if not self._compression_enabled:
            raise RuntimeError("Compression module not available")
            
        config = get_compression_preset(preset)
        self.set_compression_config(config)
    
    def get_compression_stats(self, dataset_name: str) -> Dict[str, Any]:
        """Get compression statistics for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with compression statistics
        """
        if dataset_name not in self.datasets:
            raise DatasetNotFoundError(f"Dataset '{dataset_name}' not found.")
            
        stats = {
            "total_tensors": len(self.datasets[dataset_name]["tensors"]),
            "compressed_tensors": 0,
            "total_original_size": 0,
            "total_compressed_size": 0,
            "compression_algorithms": set(),
            "quantization_algorithms": set()
        }
        
        for meta in self.datasets[dataset_name]["metadata"]:
            if meta.get("compressed", False):
                stats["compressed_tensors"] += 1
                comp_meta = meta.get("compression_metadata", {})
                stats["total_original_size"] += comp_meta.get("original_size", 0)
                stats["total_compressed_size"] += comp_meta.get("compressed_size", 0)
                stats["compression_algorithms"].add(comp_meta.get("compression", "unknown"))
                stats["quantization_algorithms"].add(comp_meta.get("quantization", "unknown"))
        
        stats["compression_algorithms"] = list(stats["compression_algorithms"])
        stats["quantization_algorithms"] = list(stats["quantization_algorithms"])
        
        if stats["total_original_size"] > 0:
            stats["average_compression_ratio"] = stats["total_original_size"] / stats["total_compressed_size"]
        else:
            stats["average_compression_ratio"] = 1.0
            
        return stats
    
    # === Indexing and Advanced Query Methods ===
    
    def create_index(self, dataset_name: str, index_name: str, field_name: str, 
                    index_type: str = "hash") -> bool:
        """Create a custom index on a metadata field.
        
        Args:
            dataset_name: Name of the dataset
            index_name: Name for the new index
            field_name: Metadata field to index
            index_type: Type of index ("hash" or "range")
            
        Returns:
            bool: True if index was created successfully
        """
        if not self._indexing_enabled:
            logging.warning("Indexing not available")
            return False
            
        if dataset_name not in self.datasets:
            raise DatasetNotFoundError(f"Dataset '{dataset_name}' not found.")
            
        index_manager = self._get_or_create_index_manager(dataset_name)
        if not index_manager:
            return False
            
        try:
            if index_type == "hash":
                from .indexing import HashIndex
                index = HashIndex(index_name)
            elif index_type == "range":
                from .indexing import RangeIndex  
                index = RangeIndex(index_name)
            else:
                raise ValueError(f"Unknown index type: {index_type}")
                
            index_manager.add_index(index_name, index)
            
            # Populate the new index with existing data
            dataset = self.datasets[dataset_name]
            for i, metadata in enumerate(dataset["metadata"]):
                if field_name in metadata:
                    tensor_data = dataset["tensors"][i]
                    tensor = self._decompress_tensor_if_needed(tensor_data, metadata)
                    index.insert(metadata.get("record_id", str(i)), metadata[field_name], tensor)
            
            logging.info(f"Created {index_type} index '{index_name}' on field '{field_name}' for dataset '{dataset_name}'")
            return True
            
        except Exception as e:
            logging.error(f"Failed to create index '{index_name}': {e}")
            return False
    
    def query_by_metadata(self, dataset_name: str, field: str, value: Any) -> List[Dict[str, Any]]:
        """Query records by exact metadata field value using indexes.
        
        Args:
            dataset_name: Name of the dataset
            field: Metadata field name
            value: Value to search for
            
        Returns:
            List of matching records with tensors and metadata
        """
        if dataset_name not in self.datasets:
            raise DatasetNotFoundError(f"Dataset '{dataset_name}' not found.")
        
        if not self._indexing_enabled:
            # Fallback to linear search
            return self.query(dataset_name, lambda t, m: m.get(field) == value)
        
        index_manager = self._get_or_create_index_manager(dataset_name)
        if not index_manager:
            return self.query(dataset_name, lambda t, m: m.get(field) == value)
            
        # Try to use index
        index = index_manager.get_index(field)
        if index:
            record_ids = index.lookup(value)
            results = []
            for record_id in record_ids:
                try:
                    result = self.get_tensor_by_id(dataset_name, record_id)
                    results.append(result)
                except TensorNotFoundError:
                    continue
            return results
        else:
            # Fallback to linear search
            return self.query(dataset_name, lambda t, m: m.get(field) == value)
    
    def query_by_tensor_shape(self, dataset_name: str, shape: Tuple[int, ...]) -> List[Dict[str, Any]]:
        """Query records by exact tensor shape using spatial index.
        
        Args:
            dataset_name: Name of the dataset
            shape: Tensor shape tuple to search for
            
        Returns:
            List of matching records with tensors and metadata
        """
        if dataset_name not in self.datasets:
            raise DatasetNotFoundError(f"Dataset '{dataset_name}' not found.")
        
        if not self._indexing_enabled:
            return self.query(dataset_name, lambda t, m: tuple(t.shape) == shape)
        
        index_manager = self._get_or_create_index_manager(dataset_name)
        if not index_manager:
            return self.query(dataset_name, lambda t, m: tuple(t.shape) == shape)
        
        # Use spatial index
        spatial_index = index_manager.get_index("spatial")
        if spatial_index and hasattr(spatial_index, 'lookup_by_shape'):
            record_ids = spatial_index.lookup_by_shape(shape)
            results = []
            for record_id in record_ids:
                try:
                    result = self.get_tensor_by_id(dataset_name, record_id)
                    results.append(result)
                except TensorNotFoundError:
                    continue
            return results
        else:
            return self.query(dataset_name, lambda t, m: tuple(t.shape) == shape)
    
    def get_index_stats(self, dataset_name: str) -> Dict[str, Any]:
        """Get statistics about dataset indexes.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with index statistics
        """
        if not self._indexing_enabled:
            return {"indexing_enabled": False}
            
        if dataset_name not in self.datasets:
            raise DatasetNotFoundError(f"Dataset '{dataset_name}' not found.")
            
        index_manager = self._get_or_create_index_manager(dataset_name)
        if not index_manager:
            return {"indexing_enabled": False}
            
        stats = index_manager.get_index_stats()
        stats["indexing_enabled"] = True
        stats["dataset_name"] = dataset_name
        return stats
    
    def rebuild_indexes(self, dataset_name: str) -> bool:
        """Rebuild all indexes for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            bool: True if indexes were rebuilt successfully
        """
        if not self._indexing_enabled:
            logging.warning("Indexing not available")
            return False
            
        if dataset_name not in self.datasets:
            raise DatasetNotFoundError(f"Dataset '{dataset_name}' not found.")
            
        try:
            self._rebuild_indexes_for_dataset(dataset_name)
            return True
        except Exception as e:
            logging.error(f"Failed to rebuild indexes for dataset '{dataset_name}': {e}")
            return False

# Public API exports
__all__ = ["TensorStorage", "DatasetNotFoundError", "TensorNotFoundError", "SchemaValidationError", "TransactionError"]

# Example Usage (can be run directly if needed)
if __name__ == "__main__":
    # --- Test with persistence ---
    print("--- Testing with Persistence ---")
    persistent_storage_path = "my_tensor_data"
    # Clean up previous test runs if any
    if Path(persistent_storage_path).exists():
        import shutil
        shutil.rmtree(persistent_storage_path)
        print(f"Cleaned up old '{persistent_storage_path}' directory.")

    storage1 = TensorStorage(storage_path=persistent_storage_path)
    storage1.create_dataset("persistent_images")
    img_tensor1 = torch.rand(3, 32, 32)
    img_id1 = storage1.insert("persistent_images", img_tensor1, metadata={"source": "cam1", "label": "truck"})
    print(f"Inserted image {img_id1} into 'persistent_images' in storage1.")

    # Verify data in storage1
    print("Datasets in storage1:", storage1.list_datasets())
    retrieved1 = storage1.get_tensor_by_id("persistent_images", img_id1)
    if retrieved1:
        print(f"Retrieved from storage1: {retrieved1['metadata']}")

    # Create a new storage instance pointing to the same path
    print("\nCreating storage2 pointing to the same path...")
    storage2 = TensorStorage(storage_path=persistent_storage_path)
    print("Datasets in storage2 (loaded from disk):", storage2.list_datasets())
    retrieved2 = storage2.get_tensor_by_id("persistent_images", img_id1)
    if retrieved2:
        print(f"Retrieved from storage2: {retrieved2['metadata']}")
        assert torch.equal(retrieved1['tensor'], retrieved2['tensor']), "Tensors are not equal after loading!"
        assert retrieved1['metadata'] == retrieved2['metadata'], "Metadata is not equal after loading!"
        print("Data consistency verified between storage1 and storage2 for 'persistent_images'.")

    # Test deletion with persistence
    storage2.delete_tensor("persistent_images", img_id1)
    print(f"Deleted tensor {img_id1} from 'persistent_images' in storage2.")
    assert storage2.get_tensor_by_id("persistent_images", img_id1) is None, "Tensor not deleted from storage2"

    # Create a third storage instance to check if deletion persisted
    print("\nCreating storage3 pointing to the same path...")
    storage3 = TensorStorage(storage_path=persistent_storage_path)
    print("Datasets in storage3 (should reflect deletion):", storage3.list_datasets())
    assert storage3.get_tensor_by_id("persistent_images", img_id1) is None, "Tensor deletion did not persist!"
    print("Tensor deletion persistence verified.")

    storage3.delete_dataset("persistent_images")
    print("Deleted dataset 'persistent_images' from storage3.")
    assert "persistent_images" not in storage3.list_datasets(), "Dataset not deleted from storage3"
    assert not (Path(persistent_storage_path) / "persistent_images.pt").exists(), "Dataset file not deleted."
    print("Dataset file deletion verified.")


    # --- Test in-memory (original behavior) ---
    print("\n\n--- Testing In-Memory Storage ---")
    storage_in_memory = TensorStorage(storage_path=None) # Explicitly in-memory

    # Create datasets
    storage_in_memory.create_dataset("images")
    storage_in_memory.create_dataset("sensor_readings")

    # Insert tensors
    img_tensor = torch.rand(3, 64, 64) # Example image tensor (Channels, H, W)
    sensor_tensor1_mem = torch.tensor([10.5, 11.2, 10.9])
    sensor_tensor2_mem = torch.tensor([11.1, 11.5, 11.3])
    sensor_tensor3_mem = torch.tensor([9.8, 10.1, 9.9])

    img_id_mem = storage_in_memory.insert("images", img_tensor, metadata={"source": "camera_A", "label": "cat"})
    sensor_id1_mem = storage_in_memory.insert("sensor_readings", sensor_tensor1_mem, metadata={"sensor_id": "XYZ", "location": "lab1"})
    sensor_id2_mem = storage_in_memory.insert("sensor_readings", sensor_tensor2_mem, metadata={"sensor_id": "XYZ", "location": "lab1"})
    sensor_id3_mem = storage_in_memory.insert("sensor_readings", sensor_tensor3_mem, metadata={"sensor_id": "ABC", "location": "lab2"})

    print(f"Inserted image with ID: {img_id_mem}")
    print(f"Inserted sensor reading 1 with ID: {sensor_id1_mem}")

    # Retrieve a dataset
    all_sensor_tensors_meta_mem = storage_in_memory.get_dataset_with_metadata("sensor_readings")
    print(f"\nRetrieved {len(all_sensor_tensors_meta_mem)} sensor records from in-memory storage:")
    for item in all_sensor_tensors_meta_mem:
        print(f"  Metadata: {item['metadata']}, Tensor shape: {item['tensor'].shape}")

    # Query a dataset
    print("\nQuerying in-memory sensor readings with first value > 11.0:")
    query_result_mem = storage_in_memory.query(
        "sensor_readings",
        lambda tensor, meta: tensor[0].item() > 11.0
    )
    for item in query_result_mem:
        print(f"  Metadata: {item['metadata']}, Tensor: {item['tensor']}")

    # Test list_datasets
    print("\nListing all datasets in-memory:")
    all_datasets_mem = storage_in_memory.list_datasets()
    print(f"  Available datasets: {all_datasets_mem}")

    # Test update_tensor_metadata
    print(f"\nUpdating metadata for sensor {sensor_id1_mem} in 'sensor_readings' (in-memory):")
    update_success_mem = storage_in_memory.update_tensor_metadata(
        "sensor_readings",
        sensor_id1_mem,
        {"location": "lab1_updated_mem", "status": "calibrated_mem"}
    )
    if update_success_mem:
        updated_item_mem = storage_in_memory.get_tensor_by_id("sensor_readings", sensor_id1_mem)
        print(f"  Update successful. New metadata: {updated_item_mem['metadata']}")

    # Test delete_tensor
    print(f"\nDeleting sensor {sensor_id2_mem} from 'sensor_readings' (in-memory):")
    delete_success_mem = storage_in_memory.delete_tensor("sensor_readings", sensor_id2_mem)
    print(f"  Deletion successful: {delete_success_mem}")
    assert storage_in_memory.get_tensor_by_id("sensor_readings", sensor_id2_mem) is None

    # Delete a dataset
    storage_in_memory.delete_dataset("images")
    try:
        storage_in_memory.get_dataset("images")
    except ValueError as e:
        print(f"\nSuccessfully deleted 'images' dataset (in-memory): {e}")

    print("\n--- All tests completed ---")
    # Clean up the persistent storage path after tests
    if Path(persistent_storage_path).exists():
        import shutil
        shutil.rmtree(persistent_storage_path)
        print(f"Cleaned up test directory '{persistent_storage_path}' after all tests.")

