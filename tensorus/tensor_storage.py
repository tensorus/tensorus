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

    def __init__(self, storage_path: Optional[str] = "tensor_data"):
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
        """
        self.datasets: Dict[str, Dict[str, List[Any]]] = {}
        self.storage_path: Optional[Path] = None
        self._use_s3: bool = False
        self._s3_bucket: Optional[str] = None
        self._s3_prefix: str = ""
        self._s3_client = None
        
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
            
        # Initialize dataset locks for any existing datasets
        for dataset_name in self.datasets.keys():
            if dataset_name not in self._dataset_locks:
                self._dataset_locks[dataset_name] = threading.RLock()

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
        # Prepare data for saving
        data_to_save = {
            "tensors": [t.clone().cpu() for t in self.datasets[dataset_name]["tensors"]],
            "metadata": self.datasets[dataset_name]["metadata"],
            "schema": self.datasets[dataset_name].get("schema")
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
                                        "schema": loaded_data.get("schema")
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
                            "schema": loaded_data.get("schema")
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
            str: The unique record ID of the inserted tensor.

        Raises:
            DatasetNotFoundError: If the dataset is not found.
            SchemaValidationError: If the tensor or metadata doesn't match the dataset schema.
        """
        # Initialize metadata if None
        if metadata is None:
            metadata = {}
        
        # Validate tensor type
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor).__name__}")
        
        # Schema validation
        if name in self.datasets and "schema" in self.datasets[name]:
            schema = self.datasets[name]["schema"]
            if schema and "shape" in schema and tuple(tensor.shape) != tuple(schema["shape"]):
                raise SchemaValidationError(
                    f"Tensor shape {tuple(tensor.shape)} does not match schema shape {schema['shape']} for dataset '{name}'."
                )
            if schema and "dtype" in schema and str(tensor.dtype) != schema["dtype"]:
                raise SchemaValidationError(
                    f"Tensor dtype {tensor.dtype} does not match schema dtype {schema['dtype']} for dataset '{name}'."
                )
            if schema and "metadata" in schema:
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

        # Check if dataset exists
        if name not in self.datasets:
            raise DatasetNotFoundError(f"Dataset '{name}' not found.")
        
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

        # --- Placeholder for Chunking Logic ---
        # In a real-world scenario, large tensors might be split into chunks here.
        # Each chunk would be stored, and metadata would track these chunks.
        # For this example, the entire tensor is stored directly.
        # ------------------------------------

        self.datasets[name]["tensors"].append(tensor.clone()) # Store a copy of the tensor to prevent external modifications
        self.datasets[name]["metadata"].append(final_metadata)
        
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
        # --- Placeholder for Reassembling Chunks ---
        # If data was chunked, it would be reassembled here before returning.
        # -----------------------------------------
        return self.datasets[name]["tensors"]

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
        for tensor, meta in zip(self.datasets[name]["tensors"], self.datasets[name]["metadata"]):
            results.append({"tensor": tensor, "metadata": meta})
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
        # --- Placeholder for Optimized Querying ---
        # In a real system, metadata indexing would speed this up significantly.
        # Query might operate directly on chunks or specific metadata fields first.
        # ----------------------------------------
        for tensor, meta in zip(self.datasets[name]["tensors"], self.datasets[name]["metadata"]):
            try:
                if query_fn(tensor, meta):
                    results.append({"tensor": tensor, "metadata": meta})
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

        # This is inefficient for large datasets; requires an index in a real system.
        for tensor, meta in zip(self.datasets[name]["tensors"], self.datasets[name]["metadata"]):
             if meta.get("record_id") == record_id:
                 logging.debug(f"Tensor with record_id '{record_id}' found in dataset '{name}'.")
                 return {"tensor": tensor, "metadata": meta}

        logging.warning(f"Tensor with record_id '{record_id}' not found in dataset '{name}'.")
        raise TensorNotFoundError(f"Tensor '{record_id}' not found in dataset '{name}'.")

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
            sampled_records.append({
                "tensor": self.datasets[name]["tensors"][i],
                "metadata": self.datasets[name]["metadata"][i]
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
        return [{"tensor": t, "metadata": m} for t, m in sliced]

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
                del dataset["tensors"][i]
                del dataset["metadata"][i]
                logging.info(f"Tensor record '{record_id}' deleted from dataset '{dataset_name}'.")
                self._save_dataset(dataset_name) # Save after tensor deletion
                return True

            logging.warning(f"Record '{record_id}' not found in dataset '{dataset_name}' for deletion.")
            raise TensorNotFoundError(f"Tensor '{record_id}' not found in dataset '{dataset_name}'.")
            
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
        """
        Update metadata for multiple tensors atomically.
        
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

# Public API exports
__all__ = ["TensorStorage", "DatasetNotFoundError", "TensorNotFoundError"]

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

