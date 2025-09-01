"""
Index Persistence and Maintenance System for Tensorus

This module provides:
- Index persistence to disk for recovery after restarts
- Automatic index maintenance for CRUD operations
- Index optimization and defragmentation
- Integration with existing metadata storage
- Backup and recovery mechanisms
"""

import os
import time
import pickle
import json
import threading
import shutil
from typing import Dict, List, Set, Optional, Any, Tuple
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor

from .index_manager import IndexManager, IndexManagerConfig
from .indexing import BaseIndex, IndexMetadata
from .storage_abc import MetadataStorage


class IndexPersistenceManager:
    """
    Manages persistence of indexes to disk for recovery and backup.

    Provides:
    - Index serialization and deserialization
    - Incremental updates and snapshots
    - Recovery from disk on startup
    - Backup and restore operations
    """

    def __init__(self, storage_path: str = "./tensorus_indexes"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.backup_path = self.storage_path / "backups"
        self.backup_path.mkdir(exist_ok=True)
        self._lock = threading.RLock()

        # Index metadata cache
        self.index_metadata: Dict[str, IndexMetadata] = {}

    def save_index(self, index: BaseIndex, force_full: bool = False) -> bool:
        """
        Save an index to disk.

        Args:
            index: The index to save
            force_full: Force full save instead of incremental

        Returns:
            True if save was successful
        """
        with self._lock:
            try:
                index_path = self.storage_path / f"{index.name}.idx"

                # Prepare index data for serialization
                index_data = {
                    "metadata": asdict(index.metadata),
                    "type": index.index_type.value,
                    "structure": index.structure.value,
                    "properties": index.indexed_properties,
                    "data": self._serialize_index_data(index),
                    "timestamp": time.time()
                }

                # Save to temporary file first
                temp_path = index_path.with_suffix('.tmp')
                with open(temp_path, 'wb') as f:
                    pickle.dump(index_data, f)

                # Atomic move to final location
                temp_path.replace(index_path)

                # Update metadata cache
                self.index_metadata[index.name] = index.metadata

                return True

            except Exception as e:
                print(f"Failed to save index {index.name}: {e}")
                return False

    def load_index(self, index_name: str, index_manager: IndexManager) -> Optional[BaseIndex]:
        """
        Load an index from disk.

        Args:
            index_name: Name of the index to load
            index_manager: IndexManager instance to create the index

        Returns:
            Loaded index or None if loading failed
        """
        with self._lock:
            try:
                index_path = self.storage_path / f"{index_name}.idx"

                if not index_path.exists():
                    return None

                with open(index_path, 'rb') as f:
                    index_data = pickle.load(f)

                # Recreate the index
                metadata = index_data["metadata"]
                index_type = index_data["type"]
                structure = index_data["structure"]
                properties = index_data["properties"]

                # Create index using IndexManager
                if index_manager.create_index(index_name, index_type, properties, structure):
                    index = index_manager.indexes[index_name]

                    # Restore index data
                    self._deserialize_index_data(index, index_data["data"])

                    # Restore metadata
                    index.metadata = IndexMetadata(**metadata)

                    return index

            except Exception as e:
                print(f"Failed to load index {index_name}: {e}")

            return None

    def _serialize_index_data(self, index: BaseIndex) -> Dict[str, Any]:
        """Serialize index-specific data."""
        if hasattr(index, 'index'):  # HashIndex, CompositeIndex
            return {
                "index": dict(index.index),
                "reverse_index": dict(index.reverse_index)
            }
        elif hasattr(index, 'reverse_index') and hasattr(index, 'root'):  # BTreeIndex
            return {
                "reverse_index": dict(index.reverse_index),
                # Note: Full B-tree serialization would be more complex
            }
        elif hasattr(index, 'shape_index'):  # SpatialIndex
            return {
                "shape_index": {str(k): list(v) for k, v in index.shape_index.items()},
                "reverse_index": {k: str(v) for k, v in index.reverse_index.items()}
            }
        else:
            return {}

    def _deserialize_index_data(self, index: BaseIndex, data: Dict[str, Any]) -> None:
        """Deserialize and restore index-specific data."""
        if hasattr(index, 'index') and 'index' in data:
            index.index.update(data['index'])
            index.reverse_index.update(data['reverse_index'])
        elif hasattr(index, 'shape_index') and 'shape_index' in data:
            # Restore spatial index
            for shape_str, tensor_ids in data['shape_index'].items():
                shape = tuple(map(int, shape_str.strip('()').split(',')))
                index.shape_index[shape] = set(tensor_ids)
            for tensor_id, shape_str in data['reverse_index'].items():
                shape = tuple(map(int, shape_str.strip('()').split(',')))
                index.reverse_index[tensor_id] = shape

    def save_all_indexes(self, index_manager: IndexManager) -> Dict[str, bool]:
        """Save all indexes to disk."""
        results = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.save_index, index): name
                for name, index in index_manager.indexes.items()
            }

            for future in futures:
                index_name = futures[future]
                try:
                    results[index_name] = future.result()
                except Exception as e:
                    print(f"Failed to save index {index_name}: {e}")
                    results[index_name] = False

        return results

    def load_all_indexes(self, index_manager: IndexManager) -> Dict[str, bool]:
        """Load all saved indexes from disk."""
        results = {}

        if not self.storage_path.exists():
            return results

        for index_file in self.storage_path.glob("*.idx"):
            index_name = index_file.stem
            try:
                index = self.load_index(index_name, index_manager)
                results[index_name] = index is not None
            except Exception as e:
                print(f"Failed to load index {index_name}: {e}")
                results[index_name] = False

        return results

    def create_backup(self, backup_name: Optional[str] = None) -> str:
        """Create a backup of all indexes."""
        if backup_name is None:
            backup_name = f"backup_{int(time.time())}"

        backup_dir = self.backup_path / backup_name
        backup_dir.mkdir(exist_ok=True)

        try:
            # Copy all index files
            for index_file in self.storage_path.glob("*.idx"):
                shutil.copy2(index_file, backup_dir / index_file.name)

            # Save backup metadata
            metadata = {
                "backup_name": backup_name,
                "timestamp": time.time(),
                "index_files": [f.name for f in self.storage_path.glob("*.idx")]
            }

            with open(backup_dir / "backup_info.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            return str(backup_dir)

        except Exception as e:
            print(f"Failed to create backup: {e}")
            # Clean up failed backup
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            raise

    def restore_backup(self, backup_name: str, index_manager: IndexManager) -> bool:
        """Restore indexes from a backup."""
        backup_dir = self.backup_path / backup_name

        if not backup_dir.exists():
            raise FileNotFoundError(f"Backup {backup_name} not found")

        try:
            # Clear existing indexes
            index_manager.indexes.clear()
            index_manager.query_cache.clear()

            # Copy backup files to main storage
            for backup_file in backup_dir.glob("*.idx"):
                shutil.copy2(backup_file, self.storage_path / backup_file.name)

            # Load all indexes
            self.load_all_indexes(index_manager)

            return True

        except Exception as e:
            print(f"Failed to restore backup {backup_name}: {e}")
            return False

    def cleanup_old_backups(self, keep_last_n: int = 5) -> int:
        """Clean up old backups, keeping only the most recent N."""
        backups = sorted(
            [d for d in self.backup_path.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        if len(backups) <= keep_last_n:
            return 0

        removed_count = 0
        for backup in backups[keep_last_n:]:
            try:
                shutil.rmtree(backup)
                removed_count += 1
            except Exception as e:
                print(f"Failed to remove backup {backup.name}: {e}")

        return removed_count

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for indexes."""
        total_size = 0
        index_files = []

        for index_file in self.storage_path.glob("*.idx"):
            size = index_file.stat().st_size
            total_size += size
            index_files.append({
                "name": index_file.stem,
                "size_bytes": size,
                "modified": index_file.stat().st_mtime
            })

        backup_count = len([d for d in self.backup_path.iterdir() if d.is_dir()])

        return {
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "index_count": len(index_files),
            "backup_count": backup_count,
            "index_files": index_files
        }


class IndexMaintenanceManager:
    """
    Manages automatic maintenance of indexes.

    Provides:
    - Automatic index updates on CRUD operations
    - Index optimization and defragmentation
    - Performance monitoring and alerting
    - Integration with metadata storage operations
    """

    def __init__(self, index_manager: IndexManager,
                 persistence_manager: IndexPersistenceManager,
                 maintenance_interval: int = 3600):  # 1 hour
        self.index_manager = index_manager
        self.persistence_manager = persistence_manager
        self.maintenance_interval = maintenance_interval
        self.last_maintenance = time.time()
        self._maintenance_thread = None
        self._running = False
        self._lock = threading.RLock()

    def start_maintenance(self) -> None:
        """Start the background maintenance thread."""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._maintenance_thread = threading.Thread(
                target=self._maintenance_loop,
                daemon=True
            )
            self._maintenance_thread.start()

    def stop_maintenance(self) -> None:
        """Stop the background maintenance thread."""
        with self._lock:
            self._running = False
            if self._maintenance_thread:
                self._maintenance_thread.join(timeout=5)

    def _maintenance_loop(self) -> None:
        """Main maintenance loop."""
        while self._running:
            try:
                time.sleep(self.maintenance_interval)
                self.perform_maintenance()
            except Exception as e:
                print(f"Maintenance error: {e}")

    def perform_maintenance(self) -> Dict[str, Any]:
        """Perform comprehensive index maintenance."""
        with self._lock:
            results = {
                "timestamp": time.time(),
                "operations": {}
            }

            # Save indexes to disk
            save_results = self.persistence_manager.save_all_indexes(self.index_manager)
            results["operations"]["index_saving"] = save_results

            # Optimize indexes
            optimization_results = self._optimize_indexes()
            results["operations"]["optimization"] = optimization_results

            # Clean up old backups
            cleanup_count = self.persistence_manager.cleanup_old_backups()
            results["operations"]["backup_cleanup"] = {"removed": cleanup_count}

            # Clear old cache entries
            self.index_manager.clear_cache()
            results["operations"]["cache_cleanup"] = {"cache_cleared": True}

            self.last_maintenance = time.time()

            return results

    def _optimize_indexes(self) -> Dict[str, Any]:
        """Optimize index performance."""
        results = {}

        for name, index in self.index_manager.indexes.items():
            try:
                # Get current statistics
                stats_before = index.get_statistics()

                # Perform index-specific optimizations
                if hasattr(index, 'index') and isinstance(index.index, dict):
                    # Rebuild hash index if efficiency is low
                    efficiency = stats_before.get("index_efficiency", 1.0)
                    if efficiency < 0.5:
                        # Rebuild the index
                        old_index = index.index.copy()
                        index.index.clear()
                        # Rebuild from reverse index
                        for tensor_id, key in index.reverse_index.items():
                            index.index[key].add(tensor_id)

                results[name] = {
                    "optimized": True,
                    "stats_before": stats_before
                }

            except Exception as e:
                results[name] = {
                    "optimized": False,
                    "error": str(e)
                }

        return results

    def on_tensor_created(self, tensor_id: str, tensor_data: Dict[str, Any]) -> None:
        """Handle tensor creation - update all relevant indexes."""
        try:
            self.index_manager.add_tensor(tensor_id, tensor_data)
        except Exception as e:
            print(f"Failed to index new tensor {tensor_id}: {e}")

    def on_tensor_updated(self, tensor_id: str, old_data: Dict[str, Any],
                         new_data: Dict[str, Any]) -> None:
        """Handle tensor update - update all relevant indexes."""
        try:
            self.index_manager.update_tensor(tensor_id, old_data, new_data)
        except Exception as e:
            print(f"Failed to update index for tensor {tensor_id}: {e}")

    def on_tensor_deleted(self, tensor_id: str) -> None:
        """Handle tensor deletion - remove from all indexes."""
        try:
            self.index_manager.remove_tensor(tensor_id)
        except Exception as e:
            print(f"Failed to remove tensor {tensor_id} from indexes: {e}")

    def get_maintenance_status(self) -> Dict[str, Any]:
        """Get current maintenance status."""
        return {
            "maintenance_running": self._running,
            "last_maintenance": self.last_maintenance,
            "next_maintenance": self.last_maintenance + self.maintenance_interval,
            "time_until_next": max(0, self.last_maintenance + self.maintenance_interval - time.time())
        }


class IndexedMetadataStorage:
    """
    Enhanced metadata storage with integrated indexing.

    Wraps existing MetadataStorage implementations and adds
    comprehensive indexing capabilities.
    """

    def __init__(self, base_storage: MetadataStorage,
                 index_storage_path: str = "./tensorus_indexes"):
        self.base_storage = base_storage

        # Initialize indexing components
        self.index_manager = IndexManager()
        self.persistence_manager = IndexPersistenceManager(index_storage_path)
        self.maintenance_manager = IndexMaintenanceManager(
            self.index_manager, self.persistence_manager
        )

        # Connect components
        self.index_manager.set_metadata_storage(base_storage)

        # Load existing indexes
        self._load_existing_indexes()

        # Start maintenance
        self.maintenance_manager.start_maintenance()

    def _load_existing_indexes(self) -> None:
        """Load existing indexes from disk."""
        try:
            load_results = self.persistence_manager.load_all_indexes(self.index_manager)
            successful_loads = sum(1 for success in load_results.values() if success)

            if successful_loads > 0:
                print(f"✅ Loaded {successful_loads} indexes from disk")
            else:
                print("ℹ️  No existing indexes found, starting fresh")

        except Exception as e:
            print(f"⚠️  Failed to load existing indexes: {e}")

    def add_tensor_descriptor(self, descriptor: Any) -> None:
        """Add tensor descriptor with indexing."""
        # Add to base storage first
        self.base_storage.add_tensor_descriptor(descriptor)

        # Add to indexes
        tensor_data = {
            "tensor_id": str(descriptor.tensor_id),
            "shape": list(descriptor.shape),
            "data_type": descriptor.data_type.value,
            "owner": descriptor.owner,
            "tags": descriptor.tags,
            "byte_size": descriptor.byte_size,
            "creation_timestamp": descriptor.creation_timestamp.timestamp(),
            "dimensionality": descriptor.dimensionality
        }

        self.maintenance_manager.on_tensor_created(str(descriptor.tensor_id), tensor_data)

    def update_tensor_descriptor(self, tensor_id: Any, **updates) -> Any:
        """Update tensor descriptor with index updates."""
        # Get old data for index updates
        old_descriptor = self.base_storage.get_tensor_descriptor(tensor_id)
        if not old_descriptor:
            raise ValueError(f"Tensor {tensor_id} not found")

        old_data = {
            "tensor_id": str(tensor_id),
            "shape": list(old_descriptor.shape),
            "data_type": old_descriptor.data_type.value,
            "owner": old_descriptor.owner,
            "tags": old_descriptor.tags,
            "byte_size": old_descriptor.byte_size,
            "creation_timestamp": old_descriptor.creation_timestamp.timestamp(),
            "dimensionality": old_descriptor.dimensionality
        }

        # Update in base storage
        updated_descriptor = self.base_storage.update_tensor_descriptor(tensor_id, **updates)

        # Update indexes
        new_data = old_data.copy()
        for key, value in updates.items():
            if hasattr(updated_descriptor, key):
                attr_value = getattr(updated_descriptor, key)
                if key == "data_type" and hasattr(attr_value, "value"):
                    new_data[key] = attr_value.value
                elif key == "creation_timestamp" and hasattr(attr_value, "timestamp"):
                    new_data[key] = attr_value.timestamp()
                else:
                    new_data[key] = attr_value

        self.maintenance_manager.on_tensor_updated(str(tensor_id), old_data, new_data)

        return updated_descriptor

    def delete_tensor_descriptor(self, tensor_id: Any) -> None:
        """Delete tensor descriptor and update indexes."""
        # Delete from base storage
        self.base_storage.delete_tensor_descriptor(tensor_id)

        # Remove from indexes
        self.maintenance_manager.on_tensor_deleted(str(tensor_id))

    def query_tensors(self, conditions: Dict[str, Any],
                     limit: Optional[int] = None,
                     sort_by: Optional[str] = None) -> List[Any]:
        """
        Query tensors using optimized indexes.

        Falls back to base storage for complex queries not supported by indexes.
        """
        # Try indexed query first
        tensor_ids = self.index_manager.query_tensors(conditions, limit, sort_by)

        if tensor_ids:
            # Fetch full tensor descriptors
            results = []
            for tensor_id in tensor_ids:
                try:
                    descriptor = self.base_storage.get_tensor_descriptor(tensor_id)
                    if descriptor:
                        results.append(descriptor)
                except Exception:
                    continue
            return results
        else:
            # Fallback to base storage query
            return self.base_storage.list_tensor_descriptors(**conditions)[:limit] if limit else self.base_storage.list_tensor_descriptors(**conditions)

    def get_indexing_stats(self) -> Dict[str, Any]:
        """Get comprehensive indexing statistics."""
        return {
            "index_manager": self.index_manager.get_statistics(),
            "persistence": self.persistence_manager.get_storage_stats(),
            "maintenance": self.maintenance_manager.get_maintenance_status()
        }

    def create_backup(self, backup_name: Optional[str] = None) -> str:
        """Create a backup of indexes."""
        return self.persistence_manager.create_backup(backup_name)

    def restore_backup(self, backup_name: str) -> bool:
        """Restore indexes from backup."""
        return self.persistence_manager.restore_backup(backup_name, self.index_manager)

    def optimize_indexes(self) -> Dict[str, Any]:
        """Optimize all indexes."""
        return self.maintenance_manager.perform_maintenance()

    def __getattr__(self, name):
        """Delegate other methods to base storage."""
        return getattr(self.base_storage, name)

    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'maintenance_manager'):
            self.maintenance_manager.stop_maintenance()
