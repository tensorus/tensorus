import h5py
import numpy as np
import uuid
import os
import json
from typing import Dict, Tuple, Any, Optional, List

class TensorStorage:
    """
    Storage layer for tensor data using HDF5.
    Provides functionality to save, load, and manage tensor data with metadata.
    """
    
    def __init__(self, filename: str = "tensor_db.h5", create_dir: bool = True):
        """
        Initialize the tensor storage with a specified HDF5 file.
        
        Args:
            filename: Path to the HDF5 file
            create_dir: Whether to create parent directories if they don't exist
        """
        self.filename = filename
        
        # Create parent directories if needed
        if create_dir:
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
        
        # Create the file if it doesn't exist
        if not os.path.exists(filename):
            with h5py.File(filename, "w") as _:
                pass
    
    def save_tensor(self, tensor: np.ndarray, metadata: Dict = None) -> str:
        """
        Save a tensor to storage with optional metadata.
        
        Args:
            tensor: The tensor data to store
            metadata: Optional metadata associated with the tensor
            
        Returns:
            tensor_id: Unique identifier for the stored tensor
        """
        if metadata is None:
            metadata = {}
            
        # Add creation timestamp to metadata
        metadata.update({
            "created_at": np.datetime64('now').astype(str),
            "shape": tensor.shape,
            "dtype": str(tensor.dtype)
        })
        
        tensor_id = str(uuid.uuid4())
        
        with h5py.File(self.filename, "a") as f:
            grp = f.create_group(tensor_id)
            grp.create_dataset("data", data=tensor, compression="gzip")
            grp.attrs["metadata"] = json.dumps(metadata)
            
        return tensor_id

    def load_tensor(self, tensor_id: str) -> Tuple[np.ndarray, Dict]:
        """
        Load a tensor and its metadata from storage.
        
        Args:
            tensor_id: The unique identifier of the tensor
            
        Returns:
            data: The tensor data
            metadata: The tensor metadata
            
        Raises:
            KeyError: If the tensor_id does not exist
        """
        with h5py.File(self.filename, "r") as f:
            if tensor_id not in f:
                raise KeyError(f"Tensor with ID {tensor_id} not found")
                
            data = np.array(f[tensor_id]["data"])
            metadata = json.loads(f[tensor_id].attrs["metadata"])
            
        return data, metadata

    def update_tensor(self, tensor_id: str, tensor: np.ndarray, 
                      metadata: Optional[Dict] = None) -> bool:
        """
        Update an existing tensor and/or its metadata.
        
        Args:
            tensor_id: The unique identifier of the tensor
            tensor: The new tensor data
            metadata: New metadata to update (or None to keep existing)
            
        Returns:
            success: True if update was successful
            
        Raises:
            KeyError: If the tensor_id does not exist
        """
        try:
            with h5py.File(self.filename, "a") as f:
                if tensor_id not in f:
                    raise KeyError(f"Tensor with ID {tensor_id} not found")
                
                # Delete existing data
                del f[tensor_id]["data"]
                
                # Create new dataset
                f[tensor_id].create_dataset("data", data=tensor, compression="gzip")
                
                # Update metadata if provided
                if metadata is not None:
                    existing_metadata = json.loads(f[tensor_id].attrs["metadata"])
                    existing_metadata.update(metadata)
                    existing_metadata["updated_at"] = np.datetime64('now').astype(str)
                    f[tensor_id].attrs["metadata"] = json.dumps(existing_metadata)
                    
            return True
        except Exception as e:
            print(f"Error updating tensor: {e}")
            return False

    def delete_tensor(self, tensor_id: str) -> bool:
        """
        Delete a tensor from storage.
        
        Args:
            tensor_id: The unique identifier of the tensor
            
        Returns:
            success: True if deletion was successful
        """
        try:
            with h5py.File(self.filename, "a") as f:
                del f[tensor_id]
            return True
        except Exception as e:
            print(f"Error deleting tensor: {e}")
            return False
            
    def list_tensors(self) -> List[Dict[str, Any]]:
        """
        List all tensors in the storage with their metadata.
        
        Returns:
            tensors: List of dictionaries containing tensor IDs and metadata
        """
        tensors = []
        
        with h5py.File(self.filename, "r") as f:
            for tensor_id in f.keys():
                metadata = json.loads(f[tensor_id].attrs["metadata"])
                tensors.append({
                    "id": tensor_id,
                    "metadata": metadata
                })
                
        return tensors 