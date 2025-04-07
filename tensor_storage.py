
import torch
from typing import List, Dict, Callable, Optional, Any
import logging
import time
import uuid
import random # Added for sampling

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TensorStorage:
    """
    Manages datasets stored as collections of tensors in memory.
    """

    def __init__(self):
        """Initializes the TensorStorage with an empty dictionary for datasets."""
        # In-memory storage. Replace with persistent storage solution for production.
        # Structure: { dataset_name: { "tensors": List[Tensor], "metadata": List[Dict] } }
        self.datasets: Dict[str, Dict[str, List[Any]]] = {}
        logging.info("TensorStorage initialized (In-Memory).")

    def create_dataset(self, name: str) -> None:
        """
        Creates a new, empty dataset.

        Args:
            name: The unique name for the new dataset.

        Raises:
            ValueError: If a dataset with the same name already exists.
        """
        if name in self.datasets:
            logging.warning(f"Attempted to create dataset '{name}' which already exists.")
            raise ValueError(f"Dataset '{name}' already exists.")

        self.datasets[name] = {"tensors": [], "metadata": []}
        logging.info(f"Dataset '{name}' created successfully.")

    def insert(self, name: str, tensor: torch.Tensor, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Inserts a tensor into a specified dataset.

        Args:
            name: The name of the dataset to insert into.
            tensor: The PyTorch tensor to insert.
            metadata: Optional dictionary containing metadata about the tensor
                      (e.g., source, timestamp, custom tags).

        Returns:
            str: A unique ID assigned to the inserted tensor record.

        Raises:
            ValueError: If the dataset does not exist.
            TypeError: If the provided object is not a PyTorch tensor.
        """
        if name not in self.datasets:
            logging.error(f"Dataset '{name}' not found for insertion.")
            raise ValueError(f"Dataset '{name}' does not exist. Create it first.")

        if not isinstance(tensor, torch.Tensor):
            logging.error(f"Attempted to insert non-tensor data into dataset '{name}'.")
            raise TypeError("Data to be inserted must be a torch.Tensor.")

        # Ensure metadata consistency if not provided
        if metadata is None:
            metadata = {} # Start with empty dict if none provided

        # Basic metadata generation
        record_id = str(uuid.uuid4())
        default_metadata = {
            "record_id": record_id,
            "timestamp_utc": time.time(),
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            # Placeholder for versioning - simple sequence for now
            "version": len(self.datasets[name]["tensors"]) + 1,
        }
        # Update default_metadata with provided metadata, overwriting reserved keys if necessary
        # Check for reserved keys before updating
        for key in default_metadata:
            if key in metadata and key != 'record_id': # Allow users to specify record_id if really needed, though risky
                logging.warning(f"Provided metadata key '{key}' might conflict with generated defaults.")

        # Merge: user-provided metadata takes precedence for non-essential fields
        # but essential fields from default_metadata are always included.
        final_metadata = {**metadata, **default_metadata} # Default values overwrite if keys conflict (like record_id)
        final_metadata.update(metadata) # Ensure user metadata takes priority after defaults are set

        # --- Placeholder for Chunking Logic ---
        # In a real implementation, large tensors would be chunked here.
        # Each chunk would be stored separately with associated metadata.
        # For now, we store the whole tensor.
        # ------------------------------------

        self.datasets[name]["tensors"].append(tensor.clone()) # Store a copy
        self.datasets[name]["metadata"].append(final_metadata)
        logging.debug(f"Tensor with shape {tuple(tensor.shape)} inserted into dataset '{name}'. Record ID: {record_id}")
        return record_id # Return the generated ID


    def get_dataset(self, name: str) -> List[torch.Tensor]:
        """
        Retrieves all tensors from a specified dataset.

        Args:
            name: The name of the dataset to retrieve.

        Returns:
            A list of all tensors in the dataset.

        Raises:
            ValueError: If the dataset does not exist.
        """
        if name not in self.datasets:
            logging.error(f"Dataset '{name}' not found for retrieval.")
            raise ValueError(f"Dataset '{name}' does not exist.")

        logging.debug(f"Retrieving all {len(self.datasets[name]['tensors'])} tensors from dataset '{name}'.")
        # --- Placeholder for Reassembling Chunks ---
        # If data was chunked, it would be reassembled here before returning.
        # -----------------------------------------
        return self.datasets[name]["tensors"]

    def get_dataset_with_metadata(self, name: str) -> List[Dict[str, Any]]:
        """
        Retrieves all tensors and their metadata from a specified dataset.

        Args:
            name: The name of the dataset to retrieve.

        Returns:
            A list of dictionaries, each containing a 'tensor' and its 'metadata'.

        Raises:
            ValueError: If the dataset does not exist.
        """
        if name not in self.datasets:
            logging.error(f"Dataset '{name}' not found for retrieval with metadata.")
            raise ValueError(f"Dataset '{name}' does not exist.")

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
            query_fn: A callable that takes a tensor and its metadata dictionary
                      as input and returns True if the tensor should be included
                      in the result, False otherwise.

        Returns:
            A list of dictionaries, each containing a 'tensor' and its 'metadata'
            that satisfy the query function.

        Raises:
            ValueError: If the dataset does not exist.
            TypeError: If query_fn is not callable.
        """
        if name not in self.datasets:
            logging.error(f"Dataset '{name}' not found for querying.")
            raise ValueError(f"Dataset '{name}' does not exist.")

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


    def get_tensor_by_id(self, name: str, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a specific tensor and its metadata by its unique record ID.

        Args:
            name: The name of the dataset.
            record_id: The unique ID of the record to retrieve.

        Returns:
            A dictionary containing the 'tensor' and 'metadata', or None if not found.

        Raises:
            ValueError: If the dataset does not exist.
        """
        if name not in self.datasets:
            logging.error(f"Dataset '{name}' not found for get_tensor_by_id.")
            raise ValueError(f"Dataset '{name}' does not exist.")

        # This is inefficient for large datasets; requires an index in a real system.
        for tensor, meta in zip(self.datasets[name]["tensors"], self.datasets[name]["metadata"]):
             if meta.get("record_id") == record_id:
                 logging.debug(f"Tensor with record_id '{record_id}' found in dataset '{name}'.")
                 return {"tensor": tensor, "metadata": meta}

        logging.warning(f"Tensor with record_id '{record_id}' not found in dataset '{name}'.")
        return None

    # --- ADDED METHOD (from Step 3) ---
    def sample_dataset(self, name: str, n_samples: int) -> List[Dict[str, Any]]:
        """
        Retrieves a random sample of records (tensor and metadata) from a dataset.

        Args:
            name: The name of the dataset to sample from.
            n_samples: The number of samples to retrieve.

        Returns:
            A list of dictionaries, each containing 'tensor' and 'metadata' for
            the sampled records. Returns fewer than n_samples if the dataset is smaller.

        Raises:
            ValueError: If the dataset does not exist.
        """
        if name not in self.datasets:
            logging.error(f"Dataset '{name}' not found for sampling.")
            raise ValueError(f"Dataset '{name}' does not exist.")

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

    def delete_dataset(self, name: str) -> bool:
        """
        Deletes an entire dataset. Use with caution!

        Args:
            name: The name of the dataset to delete.

        Returns:
            True if the dataset was deleted, False if it didn't exist.
        """
        if name in self.datasets:
            del self.datasets[name]
            logging.warning(f"Dataset '{name}' has been permanently deleted.")
            return True
        else:
            logging.warning(f"Attempted to delete non-existent dataset '{name}'.")
            return False

# Example Usage (can be run directly if needed)
if __name__ == "__main__":
    storage = TensorStorage()

    # Create datasets
    storage.create_dataset("images")
    storage.create_dataset("sensor_readings")

    # Insert tensors
    img_tensor = torch.rand(3, 64, 64) # Example image tensor (Channels, H, W)
    sensor_tensor1 = torch.tensor([10.5, 11.2, 10.9])
    sensor_tensor2 = torch.tensor([11.1, 11.5, 11.3])
    sensor_tensor3 = torch.tensor([9.8, 10.1, 9.9])

    img_id = storage.insert("images", img_tensor, metadata={"source": "camera_A", "label": "cat"})
    sensor_id1 = storage.insert("sensor_readings", sensor_tensor1, metadata={"sensor_id": "XYZ", "location": "lab1"})
    sensor_id2 = storage.insert("sensor_readings", sensor_tensor2, metadata={"sensor_id": "XYZ", "location": "lab1"})
    sensor_id3 = storage.insert("sensor_readings", sensor_tensor3, metadata={"sensor_id": "ABC", "location": "lab2"})


    print(f"Inserted image with ID: {img_id}")
    print(f"Inserted sensor reading 1 with ID: {sensor_id1}")
    print(f"Inserted sensor reading 2 with ID: {sensor_id2}")
    print(f"Inserted sensor reading 3 with ID: {sensor_id3}")


    # Retrieve a dataset
    all_sensor_tensors_meta = storage.get_dataset_with_metadata("sensor_readings")
    print(f"\nRetrieved {len(all_sensor_tensors_meta)} sensor records:")
    for item in all_sensor_tensors_meta:
        print(f"  Metadata: {item['metadata']}, Tensor shape: {item['tensor'].shape}")

    # Query a dataset
    print("\nQuerying sensor readings with first value > 11.0:")
    query_result = storage.query(
        "sensor_readings",
        lambda tensor, meta: tensor[0].item() > 11.0
    )
    for item in query_result:
        print(f"  Metadata: {item['metadata']}, Tensor: {item['tensor']}")

    print("\nQuerying sensor readings from sensor 'XYZ':")
    query_result_meta = storage.query(
        "sensor_readings",
        lambda tensor, meta: meta.get("sensor_id") == "XYZ"
    )
    for item in query_result_meta:
        print(f"  Metadata: {item['metadata']}, Tensor: {item['tensor']}")


    # Retrieve by ID
    print(f"\nRetrieving sensor reading with ID {sensor_id1}:")
    retrieved_item = storage.get_tensor_by_id("sensor_readings", sensor_id1)
    if retrieved_item:
        print(f"  Metadata: {retrieved_item['metadata']}, Tensor: {retrieved_item['tensor']}")

    # Sample the dataset
    print(f"\nSampling 2 records from sensor_readings:")
    sampled_items = storage.sample_dataset("sensor_readings", 2)
    print(f" Got {len(sampled_items)} samples.")
    for i, item in enumerate(sampled_items):
         print(f"  Sample {i+1} - Record ID: {item['metadata'].get('record_id')}, Tensor shape: {item['tensor'].shape}")

    print(f"\nSampling 5 records (more than available):")
    sampled_items_all = storage.sample_dataset("sensor_readings", 5)
    print(f" Got {len(sampled_items_all)} samples.")
    for i, item in enumerate(sampled_items_all):
         print(f"  Sample {i+1} - Record ID: {item['metadata'].get('record_id')}") # Showing IDs to see shuffle

    # Delete a dataset
    storage.delete_dataset("images")
    try:
        storage.get_dataset("images")
    except ValueError as e:
        print(f"\nSuccessfully deleted 'images' dataset: {e}")

