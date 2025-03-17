import numpy as np
import logging
import os
import json
import time
import uuid
import threading
import asyncio
import aiohttp
import socket
from typing import Dict, List, Any, Tuple, Optional, Union, Set, Callable
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NodeInfo:
    """Information about a node in the distributed system."""
    
    def __init__(self, node_id: str, address: str, port: int, role: str = "worker"):
        """
        Initialize node information.
        
        Args:
            node_id: Unique identifier for the node
            address: IP address or hostname
            port: Port number
            role: Node role ("worker", "coordinator", or "hybrid")
        """
        self.node_id = node_id
        self.address = address
        self.port = port
        self.role = role
        self.status = "online"
        self.last_heartbeat = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "node_id": self.node_id,
            "address": self.address,
            "port": self.port,
            "role": self.role,
            "status": self.status,
            "last_heartbeat": self.last_heartbeat
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeInfo':
        """Create from dictionary representation."""
        node = cls(
            node_id=data["node_id"],
            address=data["address"],
            port=data["port"],
            role=data["role"]
        )
        node.status = data["status"]
        node.last_heartbeat = data["last_heartbeat"]
        return node
    
    def get_url(self, endpoint: str = "") -> str:
        """Get the URL for this node."""
        return f"http://{self.address}:{self.port}/{endpoint.lstrip('/')}"

class PartitionStrategy:
    """Strategy for partitioning tensors across nodes."""
    
    @staticmethod
    def by_dimension(tensor: np.ndarray, dimension: int, num_partitions: int) -> List[np.ndarray]:
        """
        Partition a tensor along a specific dimension.
        
        Args:
            tensor: The tensor to partition
            dimension: The dimension to partition along
            num_partitions: Number of partitions to create
            
        Returns:
            List of tensor partitions
        """
        if dimension >= tensor.ndim:
            raise ValueError(f"Dimension {dimension} out of range for tensor with {tensor.ndim} dimensions")
        
        # Calculate partition size
        dim_size = tensor.shape[dimension]
        partition_size = max(1, dim_size // num_partitions)
        
        # Create slices
        partitions = []
        for i in range(0, dim_size, partition_size):
            # Create a slice for each partition
            slices = [slice(None)] * tensor.ndim
            slices[dimension] = slice(i, min(i + partition_size, dim_size))
            partitions.append(tensor[tuple(slices)])
        
        return partitions
    
    @staticmethod
    def by_chunks(tensor: np.ndarray, chunk_sizes: List[int]) -> List[np.ndarray]:
        """
        Partition a tensor into chunks of specified sizes.
        
        Args:
            tensor: The tensor to partition
            chunk_sizes: List of chunk sizes for each dimension
            
        Returns:
            List of tensor chunks
        """
        if len(chunk_sizes) != tensor.ndim:
            raise ValueError(f"Expected {tensor.ndim} chunk sizes, got {len(chunk_sizes)}")
        
        # Calculate number of chunks per dimension
        chunks_per_dim = [max(1, shape // size) for shape, size in zip(tensor.shape, chunk_sizes)]
        total_chunks = np.prod(chunks_per_dim)
        
        # Create chunks
        chunks = []
        for i in range(int(total_chunks)):
            # Convert linear index to multi-dimensional indices
            indices = []
            temp = i
            for dim_chunks in chunks_per_dim:
                indices.append(temp % dim_chunks)
                temp //= dim_chunks
            
            # Create slices for this chunk
            slices = []
            for idx, size, shape in zip(indices, chunk_sizes, tensor.shape):
                start = idx * size
                end = min(start + size, shape)
                slices.append(slice(start, end))
            
            chunks.append(tensor[tuple(slices)])
        
        return chunks
    
    @staticmethod
    def by_hash(tensor_id: str, num_partitions: int) -> int:
        """
        Determine the partition for a tensor based on its ID.
        
        Args:
            tensor_id: The tensor ID
            num_partitions: Number of available partitions
            
        Returns:
            Partition index
        """
        # Simple hash-based partitioning
        return hash(tensor_id) % num_partitions

class TensorDistributedNode:
    """
    Node in the distributed tensor system.
    Manages local tensor storage and processing, and communicates with other nodes.
    """
    
    def __init__(self, 
                 database_ref,
                 node_id: Optional[str] = None,
                 address: Optional[str] = None,
                 port: int = 5050,
                 coordinator_url: Optional[str] = None,
                 role: str = "worker"):
        """
        Initialize a distributed node.
        
        Args:
            database_ref: Reference to the local TensorDatabase
            node_id: Unique identifier for this node (defaults to a UUID)
            address: IP address or hostname (defaults to local hostname)
            port: Port to run the node's API server
            coordinator_url: URL of the coordinator node (if this is a worker)
            role: Node role ("worker", "coordinator", or "hybrid")
        """
        self.database = database_ref
        self.node_id = node_id or str(uuid.uuid4())
        self.address = address or socket.gethostbyname(socket.gethostname())
        self.port = port
        self.role = role
        self.coordinator_url = coordinator_url
        
        # Node registry (only used by coordinator or hybrid nodes)
        self.nodes: Dict[str, NodeInfo] = {}
        
        # Track tensor locations (which node has which tensor)
        self.tensor_locations: Dict[str, Set[str]] = {}
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Node status
        self.is_running = False
        self.heartbeat_interval = 10  # seconds
        
        # Replication factor for redundancy
        self.replication_factor = 2
        
        logger.info(f"Initialized distributed node {self.node_id} ({role}) at {self.address}:{self.port}")
    
    def start(self):
        """Start the node's services."""
        self.is_running = True
        
        # Start API server
        threading.Thread(target=self._start_api_server, daemon=True).start()
        
        # Register with coordinator if we're a worker
        if self.coordinator_url and self.role in ["worker", "hybrid"]:
            threading.Thread(target=self._register_with_coordinator, daemon=True).start()
        
        # Start heartbeat if we're a worker
        if self.coordinator_url and self.role in ["worker", "hybrid"]:
            threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        
        # Start node monitoring if we're a coordinator
        if self.role in ["coordinator", "hybrid"]:
            threading.Thread(target=self._monitor_nodes, daemon=True).start()
        
        logger.info(f"Node {self.node_id} started")
    
    def stop(self):
        """Stop the node's services."""
        self.is_running = False
        logger.info(f"Node {self.node_id} stopped")
    
    def _start_api_server(self):
        """Start the API server for inter-node communication."""
        # This would be implemented with a proper web server (Flask, FastAPI, etc.)
        # For simplicity, we're not implementing the full server here
        logger.info(f"API server started at {self.address}:{self.port}")
    
    def _register_with_coordinator(self):
        """Register this node with the coordinator."""
        try:
            # Again, this would make a real HTTP request in a full implementation
            logger.info(f"Registered with coordinator at {self.coordinator_url}")
        except Exception as e:
            logger.error(f"Failed to register with coordinator: {e}")
    
    def _heartbeat_loop(self):
        """Send periodic heartbeats to the coordinator."""
        while self.is_running:
            try:
                # Send heartbeat
                logger.debug(f"Sending heartbeat to coordinator")
                # This would make a real HTTP request in a full implementation
                
                # Wait for next heartbeat
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Heartbeat failed: {e}")
                time.sleep(5)  # Shorter interval for retry
    
    def _monitor_nodes(self):
        """Monitor node health (coordinator function)."""
        while self.is_running:
            # Check for expired heartbeats
            current_time = time.time()
            for node_id, node in list(self.nodes.items()):
                if current_time - node.last_heartbeat > self.heartbeat_interval * 3:
                    if node.status == "online":
                        logger.warning(f"Node {node_id} appears to be offline")
                        node.status = "offline"
                        
                        # Initiate tensor recovery for this node
                        threading.Thread(
                            target=self._recover_tensors_from_node,
                            args=(node_id,),
                            daemon=True
                        ).start()
            
            # Sleep before next check
            time.sleep(self.heartbeat_interval)
    
    def _recover_tensors_from_node(self, node_id: str):
        """
        Recover tensors from a failed node by redistributing from replicas.
        
        Args:
            node_id: ID of the failed node
        """
        logger.info(f"Initiating tensor recovery for node {node_id}")
        
        # Find tensors on the failed node
        affected_tensors = []
        for tensor_id, nodes in self.tensor_locations.items():
            if node_id in nodes:
                affected_tensors.append(tensor_id)
                # Remove failed node from locations
                nodes.remove(node_id)
        
        # Redistribute affected tensors
        for tensor_id in affected_tensors:
            self._ensure_replication(tensor_id)
        
        logger.info(f"Completed tensor recovery for {len(affected_tensors)} tensors from node {node_id}")
    
    def _ensure_replication(self, tensor_id: str):
        """
        Ensure a tensor is replicated to the desired number of nodes.
        
        Args:
            tensor_id: ID of the tensor to replicate
        """
        # Skip if the tensor doesn't exist in our registry
        if tensor_id not in self.tensor_locations:
            return
        
        # Get current locations
        current_locations = self.tensor_locations[tensor_id]
        
        # Check how many more replicas we need
        needed_replicas = max(0, self.replication_factor - len(current_locations))
        if needed_replicas == 0:
            return
        
        # Find available online nodes that don't have this tensor
        available_nodes = [
            node_id for node_id, node in self.nodes.items()
            if node.status == "online" and node_id not in current_locations
        ]
        
        # Pick random nodes for replication
        import random
        selected_nodes = random.sample(available_nodes, min(needed_replicas, len(available_nodes)))
        
        # Replicate to selected nodes
        for target_node_id in selected_nodes:
            self._replicate_tensor(tensor_id, target_node_id)
            current_locations.add(target_node_id)
    
    def _replicate_tensor(self, tensor_id: str, target_node_id: str):
        """
        Replicate a tensor to a target node.
        
        Args:
            tensor_id: ID of the tensor to replicate
            target_node_id: ID of the target node
        """
        # Find a source node that has the tensor
        source_nodes = [
            node_id for node_id in self.tensor_locations.get(tensor_id, set())
            if self.nodes.get(node_id, NodeInfo("", "", 0)).status == "online"
        ]
        
        if not source_nodes:
            logger.error(f"Cannot replicate tensor {tensor_id}: no available source nodes")
            return
        
        source_node_id = source_nodes[0]
        source_node = self.nodes[source_node_id]
        target_node = self.nodes[target_node_id]
        
        # In a real implementation, this would initiate a transfer between nodes
        logger.info(f"Replicating tensor {tensor_id} from {source_node_id} to {target_node_id}")
    
    def register_node(self, node_info: Dict[str, Any]) -> bool:
        """
        Register a node in the distributed system (coordinator function).
        
        Args:
            node_info: Information about the node to register
            
        Returns:
            True if registration was successful
        """
        if self.role not in ["coordinator", "hybrid"]:
            logger.error("Only coordinator nodes can register other nodes")
            return False
        
        # Create NodeInfo object
        node = NodeInfo(
            node_id=node_info["node_id"],
            address=node_info["address"],
            port=node_info["port"],
            role=node_info["role"]
        )
        
        # Add to registry
        self.nodes[node.node_id] = node
        logger.info(f"Registered node {node.node_id} ({node.role}) at {node.address}:{node.port}")
        
        return True
    
    def update_heartbeat(self, node_id: str) -> bool:
        """
        Update the heartbeat timestamp for a node (coordinator function).
        
        Args:
            node_id: ID of the node sending the heartbeat
            
        Returns:
            True if the heartbeat was accepted
        """
        if self.role not in ["coordinator", "hybrid"]:
            logger.error("Only coordinator nodes can update heartbeats")
            return False
        
        if node_id not in self.nodes:
            logger.warning(f"Received heartbeat from unknown node {node_id}")
            return False
        
        # Update heartbeat timestamp
        self.nodes[node_id].last_heartbeat = time.time()
        
        # Update status if the node was previously offline
        if self.nodes[node_id].status == "offline":
            self.nodes[node_id].status = "online"
            logger.info(f"Node {node_id} is back online")
        
        return True
    
    def save_distributed(self, tensor: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a tensor in the distributed system.
        
        Args:
            tensor: Tensor data to save
            metadata: Optional metadata
            
        Returns:
            Unique ID for the stored tensor
        """
        # Generate a unique ID for this tensor
        tensor_id = str(uuid.uuid4())
        
        if metadata is None:
            metadata = {}
        
        # Add distributed metadata
        metadata.update({
            "distributed": True,
            "origin_node": self.node_id,
            "created_at": time.time()
        })
        
        # Case 1: Small tensor, store locally and replicate
        if np.prod(tensor.shape) < 1000000:  # Arbitrary threshold
            # Save in local database
            self.database.save(tensor, metadata)
            
            # Track locations
            if tensor_id not in self.tensor_locations:
                self.tensor_locations[tensor_id] = set()
            self.tensor_locations[tensor_id].add(self.node_id)
            
            # Replicate if we are a coordinator
            if self.role in ["coordinator", "hybrid"]:
                self._ensure_replication(tensor_id)
            
            # If we're a worker, inform coordinator
            elif self.coordinator_url:
                # Would make a real HTTP request in a full implementation
                pass
            
            return tensor_id
        
        # Case 2: Large tensor, partition and distribute
        else:
            # Determine partition strategy based on tensor shape
            if max(tensor.shape) > 10 * min(tensor.shape):
                # Very uneven dimensions, partition along largest dimension
                largest_dim = np.argmax(tensor.shape)
                num_partitions = min(len(self.nodes) if self.nodes else 1, 10)  # Cap at 10 partitions
                partitions = PartitionStrategy.by_dimension(tensor, largest_dim, num_partitions)
            else:
                # Relatively even dimensions, use chunk-based partitioning
                chunk_sizes = [max(1, dim // 2) for dim in tensor.shape]  # Simple heuristic
                partitions = PartitionStrategy.by_chunks(tensor, chunk_sizes)
            
            # Add partitioning info to metadata
            metadata.update({
                "partitioned": True,
                "partition_count": len(partitions),
                "partition_strategy": "dimension" if 'largest_dim' in locals() else "chunks"
            })
            
            # Save partition information in the coordinator
            if self.role in ["coordinator", "hybrid"]:
                # In a real implementation, we would store partition metadata
                pass
            
            # Distribute partitions to available nodes
            partition_locations = {}
            
            # If we're a coordinator, distribute to workers
            if self.role in ["coordinator", "hybrid"] and self.nodes:
                # Get online worker nodes
                workers = [
                    node_id for node_id, node in self.nodes.items()
                    if node.status == "online" and node.role in ["worker", "hybrid"]
                ]
                
                if workers:
                    for i, partition in enumerate(partitions):
                        # Select target node (round-robin for simplicity)
                        target_node_id = workers[i % len(workers)]
                        partition_id = f"{tensor_id}_part_{i}"
                        
                        # In a real implementation, we would send the partition to the target node
                        # Here we'll just track the locations
                        partition_locations[i] = target_node_id
                        
                        if tensor_id not in self.tensor_locations:
                            self.tensor_locations[tensor_id] = set()
                        self.tensor_locations[tensor_id].add(target_node_id)
            
            # If we're a worker or no workers available, store locally
            else:
                for i, partition in enumerate(partitions):
                    partition_id = f"{tensor_id}_part_{i}"
                    part_metadata = {**metadata, "partition_index": i, "tensor_id": tensor_id}
                    
                    # Save in local database
                    self.database.save(partition, part_metadata)
                    
                    partition_locations[i] = self.node_id
                    
                    if tensor_id not in self.tensor_locations:
                        self.tensor_locations[tensor_id] = set()
                    self.tensor_locations[tensor_id].add(self.node_id)
            
            # Save partition mapping in metadata
            metadata["partition_locations"] = partition_locations
            
            # In a real implementation, we would store this metadata in a central location
            
            return tensor_id
    
    def get_distributed(self, tensor_id: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get a tensor from the distributed system.
        
        Args:
            tensor_id: ID of the tensor to retrieve
            
        Returns:
            Tensor data and metadata
            
        Raises:
            KeyError: If the tensor is not found
        """
        # Check if tensor exists locally
        try:
            tensor, metadata = self.database.get(tensor_id)
            return tensor, metadata
        except KeyError:
            # Not found locally
            pass
        
        # If we're a coordinator, check if we know where the tensor is
        if self.role in ["coordinator", "hybrid"] and tensor_id in self.tensor_locations:
            # Get nodes that have this tensor
            nodes_with_tensor = [
                node_id for node_id in self.tensor_locations[tensor_id]
                if self.nodes.get(node_id, NodeInfo("", "", 0)).status == "online"
            ]
            
            if not nodes_with_tensor:
                raise KeyError(f"Tensor {tensor_id} exists but no online nodes have it")
            
            # Check if the tensor is partitioned
            # In a real implementation, we would check the metadata
            # For simplicity, we'll assume it's not partitioned
            
            # Get from the first available node
            source_node = self.nodes[nodes_with_tensor[0]]
            
            # In a real implementation, we would send a request to the source node
            # Here we'll just raise an error
            raise KeyError(f"Tensor {tensor_id} is on node {source_node.node_id} but remote retrieval is not implemented")
        
        # If we're a worker, ask the coordinator
        elif self.coordinator_url:
            # In a real implementation, we would send a request to the coordinator
            # Here we'll just raise an error
            raise KeyError(f"Tensor {tensor_id} not found locally and remote retrieval is not implemented")
        
        # Not found
        raise KeyError(f"Tensor {tensor_id} not found")
    
    def process_distributed(self, 
                           operation: str, 
                           tensor_ids: List[str], 
                           **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process tensors in the distributed system.
        
        Args:
            operation: Name of the operation to perform
            tensor_ids: IDs of the tensors to process
            **kwargs: Additional arguments for the operation
            
        Returns:
            Result tensor and metadata
            
        Raises:
            ValueError: If an error occurs during processing
        """
        # For simplicity, we'll just process locally
        # In a real implementation, we would optimize based on tensor locations
        
        try:
            # Load tensors
            tensors = []
            for tensor_id in tensor_ids:
                tensor, _ = self.get_distributed(tensor_id)
                tensors.append(tensor)
            
            # Perform operation
            result = self.database.process(operation, tensors, **kwargs)
            
            # Create metadata
            metadata = {
                "distributed": True,
                "origin_node": self.node_id,
                "created_at": time.time(),
                "operation": operation,
                "source_tensors": tensor_ids,
                "kwargs": str(kwargs)
            }
            
            return result, metadata
        except Exception as e:
            raise ValueError(f"Error in distributed processing: {e}")
    
    def search_distributed(self, 
                          query_tensor: np.ndarray, 
                          k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar tensors in the distributed system.
        
        Args:
            query_tensor: Query tensor
            k: Number of results to return
            
        Returns:
            List of search results
        """
        # Local search first
        try:
            local_results = self.database.search_similar(query_tensor, k)
        except Exception as e:
            logger.error(f"Local search error: {e}")
            local_results = []
        
        # If we're a coordinator, also search on worker nodes
        if self.role in ["coordinator", "hybrid"] and self.nodes:
            # Get online worker nodes
            workers = [
                node_id for node_id, node in self.nodes.items()
                if node.status == "online" and node_id != self.node_id
            ]
            
            # In a real implementation, we would send search requests to workers
            # Here we'll just return the local results
            
            return local_results
        
        # If we're a worker, also ask the coordinator to search other nodes
        elif self.coordinator_url:
            # In a real implementation, we would send a search request to the coordinator
            # Here we'll just return the local results
            
            return local_results
        
        # Return local results
        return local_results

class FederatedLearning:
    """
    Implements federated learning capabilities for distributed tensor processing.
    """
    
    def __init__(self, node: TensorDistributedNode):
        """
        Initialize federated learning.
        
        Args:
            node: Reference to the distributed node
        """
        self.node = node
        self.models = {}  # Store model weights
        self.training_configs = {}  # Store training configurations
        self.aggregation_threads = {}  # Track ongoing aggregation tasks
        
        logger.info("Federated learning initialized")
    
    def create_federated_model(self, 
                              model_id: str, 
                              initial_weights: Dict[str, np.ndarray],
                              config: Dict[str, Any]) -> bool:
        """
        Create a new federated model.
        
        Args:
            model_id: Unique identifier for the model
            initial_weights: Initial weights for the model
            config: Training configuration
            
        Returns:
            True if model was created successfully
        """
        if model_id in self.models:
            logger.warning(f"Model {model_id} already exists")
            return False
        
        # Store model weights
        self.models[model_id] = {
            "weights": initial_weights,
            "version": 1,
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        # Store training configuration
        self.training_configs[model_id] = config
        
        # Save to local database for persistence
        weights_tensor = np.array([1])  # Placeholder
        weights_metadata = {
            "model_id": model_id,
            "type": "federated_model_weights",
            "version": 1,
            "weights": initial_weights  # In a real implementation, we'd serialize properly
        }
        
        self.node.database.save(weights_tensor, weights_metadata)
        
        logger.info(f"Created federated model {model_id}")
        return True
    
    def get_model_weights(self, model_id: str) -> Tuple[Dict[str, np.ndarray], int]:
        """
        Get the weights for a federated model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Model weights and version
            
        Raises:
            KeyError: If the model is not found
        """
        if model_id not in self.models:
            raise KeyError(f"Model {model_id} not found")
        
        return self.models[model_id]["weights"], self.models[model_id]["version"]
    
    def submit_model_update(self, 
                           model_id: str, 
                           updated_weights: Dict[str, np.ndarray],
                           training_stats: Dict[str, Any],
                           base_version: int) -> bool:
        """
        Submit updated weights for a federated model.
        
        Args:
            model_id: ID of the model
            updated_weights: Updated weights
            training_stats: Statistics from the training process
            base_version: Version of the weights that were updated
            
        Returns:
            True if update was accepted
        """
        if model_id not in self.models:
            logger.warning(f"Model {model_id} not found")
            return False
        
        # For a coordinator node, store the update for aggregation
        if self.node.role in ["coordinator", "hybrid"]:
            # In a real implementation, we would store updates and aggregate them
            # Here we'll just apply the update directly
            
            current_version = self.models[model_id]["version"]
            
            # Check if the update is based on the current version
            if base_version != current_version:
                logger.warning(f"Update for model {model_id} is based on outdated version {base_version} (current: {current_version})")
                return False
            
            # Apply the update
            self.models[model_id]["weights"] = updated_weights
            self.models[model_id]["version"] += 1
            self.models[model_id]["updated_at"] = time.time()
            
            logger.info(f"Updated federated model {model_id} to version {self.models[model_id]['version']}")
            return True
        
        # For a worker node, send the update to the coordinator
        elif self.node.coordinator_url:
            # In a real implementation, we would send the update to the coordinator
            # Here we'll just return failure
            logger.warning("Worker nodes cannot directly update models")
            return False
        
        return False
    
    def aggregate_updates(self, 
                         model_id: str, 
                         updates: List[Tuple[Dict[str, np.ndarray], Dict[str, Any]]],
                         strategy: str = "fedavg") -> Dict[str, np.ndarray]:
        """
        Aggregate model updates from multiple nodes.
        
        Args:
            model_id: ID of the model
            updates: List of (weights, stats) tuples
            strategy: Aggregation strategy
            
        Returns:
            Aggregated weights
        """
        if not updates:
            raise ValueError("No updates provided for aggregation")
        
        # Extract weights and stats
        weights_list = [u[0] for u in updates]
        stats_list = [u[1] for u in updates]
        
        # Perform aggregation based on strategy
        if strategy == "fedavg":
            return self._fedavg_aggregation(weights_list, stats_list)
        elif strategy == "weighted":
            return self._weighted_aggregation(weights_list, stats_list)
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")
    
    def _fedavg_aggregation(self, 
                           weights_list: List[Dict[str, np.ndarray]], 
                           stats_list: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Perform FedAvg aggregation.
        
        Args:
            weights_list: List of model weights
            stats_list: List of training statistics
            
        Returns:
            Aggregated weights
        """
        # Simple average of all weights
        aggregated = {}
        
        # Get keys from first set of weights
        keys = weights_list[0].keys()
        
        for key in keys:
            # Stack weights along a new axis
            stacked = np.stack([w[key] for w in weights_list], axis=0)
            # Take mean along the first axis
            aggregated[key] = np.mean(stacked, axis=0)
        
        return aggregated
    
    def _weighted_aggregation(self, 
                             weights_list: List[Dict[str, np.ndarray]], 
                             stats_list: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Perform weighted aggregation based on training stats.
        
        Args:
            weights_list: List of model weights
            stats_list: List of training statistics
            
        Returns:
            Aggregated weights
        """
        # Extract sample counts from stats
        sample_counts = [stats.get("num_samples", 1) for stats in stats_list]
        total_samples = sum(sample_counts)
        
        # Compute weights based on sample counts
        weight_factors = [count / total_samples for count in sample_counts]
        
        # Weighted average of all weights
        aggregated = {}
        
        # Get keys from first set of weights
        keys = weights_list[0].keys()
        
        for key in keys:
            # Initialize with zeros of the right shape
            shape = weights_list[0][key].shape
            aggregated[key] = np.zeros(shape)
            
            # Add weighted contributions
            for w, factor in zip(weights_list, weight_factors):
                aggregated[key] += w[key] * factor
        
        return aggregated
    
    def start_federated_training(self, 
                               model_id: str, 
                               nodes: List[str] = None,
                               rounds: int = 1) -> bool:
        """
        Start federated training for a model.
        
        Args:
            model_id: ID of the model
            nodes: List of node IDs to include (None for all available)
            rounds: Number of training rounds
            
        Returns:
            True if training was started successfully
        """
        if model_id not in self.models:
            logger.warning(f"Model {model_id} not found")
            return False
        
        if self.node.role not in ["coordinator", "hybrid"]:
            logger.warning("Only coordinator nodes can start federated training")
            return False
        
        # Check if training is already in progress
        if model_id in self.aggregation_threads and self.aggregation_threads[model_id].is_alive():
            logger.warning(f"Federated training for model {model_id} is already in progress")
            return False
        
        # Start training thread
        thread = threading.Thread(
            target=self._federated_training_loop,
            args=(model_id, nodes, rounds),
            daemon=True
        )
        self.aggregation_threads[model_id] = thread
        thread.start()
        
        logger.info(f"Started federated training for model {model_id}, {rounds} rounds")
        return True
    
    def _federated_training_loop(self, model_id: str, nodes: List[str] = None, rounds: int = 1):
        """
        Execute federated training rounds.
        
        Args:
            model_id: ID of the model
            nodes: List of node IDs to include
            rounds: Number of training rounds
        """
        try:
            # Get available nodes
            if nodes is None:
                nodes = [
                    node_id for node_id, node in self.node.nodes.items()
                    if node.status == "online"
                ]
            
            if not nodes:
                logger.warning(f"No nodes available for federated training of model {model_id}")
                return
            
            # Execute training rounds
            for round_num in range(1, rounds + 1):
                logger.info(f"Starting round {round_num}/{rounds} for model {model_id}")
                
                # Get current model weights
                weights, version = self.get_model_weights(model_id)
                
                # Distribute to nodes
                # In a real implementation, we would send the weights to the nodes
                # and have them perform local training
                
                # Simulate receiving updates from nodes
                # In a real implementation, nodes would send back their updates
                updates = []
                for _ in range(min(3, len(nodes))):  # Simulate 3 nodes responding
                    # Create a slightly modified version of the weights
                    node_weights = {k: v * (1 + np.random.normal(0, 0.01)) for k, v in weights.items()}
                    node_stats = {"num_samples": np.random.randint(100, 1000)}
                    updates.append((node_weights, node_stats))
                
                # Aggregate updates
                if updates:
                    aggregated = self.aggregate_updates(model_id, updates, "fedavg")
                    
                    # Apply aggregated update
                    self.models[model_id]["weights"] = aggregated
                    self.models[model_id]["version"] += 1
                    self.models[model_id]["updated_at"] = time.time()
                    
                    logger.info(f"Completed round {round_num}/{rounds} for model {model_id}, new version: {self.models[model_id]['version']}")
                else:
                    logger.warning(f"No updates received for round {round_num}/{rounds} of model {model_id}")
                
                # Sleep between rounds
                time.sleep(1)
            
            logger.info(f"Completed all {rounds} rounds of federated training for model {model_id}")
            
        except Exception as e:
            logger.error(f"Error in federated training loop for model {model_id}: {e}")
    
    def get_training_status(self, model_id: str) -> Dict[str, Any]:
        """
        Get the status of federated training for a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Training status information
        """
        if model_id not in self.models:
            raise KeyError(f"Model {model_id} not found")
        
        status = {
            "model_id": model_id,
            "version": self.models[model_id]["version"],
            "in_progress": False,
            "last_updated": self.models[model_id]["updated_at"]
        }
        
        # Check if training is in progress
        if model_id in self.aggregation_threads:
            thread = self.aggregation_threads[model_id]
            status["in_progress"] = thread.is_alive()
        
        return status 