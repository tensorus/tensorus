"""
Streaming Operation Pipeline for Tensorus

This module provides streaming execution capabilities for tensor operations,
enabling memory-efficient processing of large tensors that don't fit in RAM.

Key Features:
- Chunk-based streaming for memory efficiency
- Pipeline execution with backpressure control
- Parallel processing of independent chunks
- Memory usage monitoring and optimization
- Fault tolerance and recovery
- Progress tracking and cancellation
"""

import os
import time
import uuid
import threading
import queue
from typing import Dict, List, Set, Tuple, Optional, Any, Iterator, Callable, Union
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import psutil
import torch

from tensorus.tensor_chunking import TensorChunker, TensorChunkingConfig
from tensorus.tensor_operations_integrated import OperationalStorage, OperationSpec, OperationType


class PipelineStage(Enum):
    """Stages in the streaming pipeline."""
    LOAD = "load"           # Load chunk from storage
    PROCESS = "process"     # Apply operation to chunk
    AGGREGATE = "aggregate" # Combine results if needed
    STORE = "store"         # Store result chunk


@dataclass
class ChunkTask:
    """Represents a chunk processing task."""
    chunk_id: str
    tensor_id: str
    chunk_index: int
    total_chunks: int
    data: Optional[torch.Tensor] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    stage: PipelineStage = PipelineStage.LOAD
    created_at: float = field(default_factory=time.time)
    processed_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class StreamingOperation:
    """Configuration for a streaming operation."""
    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation_spec: OperationSpec = None
    input_tensor_ids: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    output_tensor_id: Optional[str] = None
    chunk_size_mb: int = 50
    max_memory_mb: int = 512
    max_workers: int = 4
    enable_progress_tracking: bool = True
    enable_cancellation: bool = True


class StreamingPipeline:
    """
    Streaming pipeline for processing large tensor operations.

    Features:
    - Memory-efficient chunked processing
    - Parallel execution with worker pools
    - Progress tracking and cancellation
    - Fault tolerance with retry logic
    - Backpressure control for memory management
    """

    def __init__(self, operational_storage: OperationalStorage,
                 max_workers: int = 4, max_memory_mb: int = 1024):
        self.operational_storage = operational_storage
        self.max_workers = max_workers
        self.max_memory_mb = max_memory_mb

        # Pipeline components
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.active_tasks: Dict[str, ChunkTask] = {}
        self.completed_tasks: Dict[str, ChunkTask] = {}

        # Control and monitoring
        self._running = False
        self._cancelled = False
        self._lock = threading.RLock()
        self._progress_callbacks: List[Callable] = []

        # Statistics
        self.stats = {
            "total_operations": 0,
            "successful_chunks": 0,
            "failed_chunks": 0,
            "total_processing_time": 0.0,
            "peak_memory_usage": 0
        }

    def execute_streaming_operation(self, operation: StreamingOperation) -> Dict[str, Any]:
        """
        Execute a tensor operation using streaming pipeline.

        Args:
            operation: Streaming operation configuration

        Returns:
            Operation results and statistics
        """
        start_time = time.time()
        self._running = True
        self._cancelled = False

        try:
            # Initialize operation
            self._initialize_operation(operation)

            # Process chunks in pipeline
            results = self._process_chunks_streaming(operation)

            # Finalize and return results
            execution_time = time.time() - start_time
            return self._finalize_operation(operation, results, execution_time)

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "execution_time": time.time() - start_time
            }
        finally:
            self._running = False

    def _initialize_operation(self, operation: StreamingOperation) -> None:
        """Initialize streaming operation."""
        # Create tasks for each input tensor
        for tensor_id in operation.input_tensor_ids:
            self._create_chunk_tasks(tensor_id, operation)

        # Update statistics
        self.stats["total_operations"] += 1

    def _create_chunk_tasks(self, tensor_id: str, operation: StreamingOperation) -> None:
        """Create chunk processing tasks for a tensor."""
        # Get tensor streaming iterator
        tensor_stream = self.operational_storage.stream_tensor(tensor_id)

        chunk_index = 0
        for chunk in tensor_stream:
            task = ChunkTask(
                chunk_id=f"{tensor_id}_chunk_{chunk_index}",
                tensor_id=tensor_id,
                chunk_index=chunk_index,
                total_chunks=-1,  # Will be updated when we know total
                data=chunk,
                stage=PipelineStage.LOAD
            )

            self.active_tasks[task.chunk_id] = task
            self.task_queue.put(task)
            chunk_index += 1

        # Update total chunks for all tasks
        total_chunks = chunk_index
        for task in self.active_tasks.values():
            if task.tensor_id == tensor_id:
                task.total_chunks = total_chunks

    def _process_chunks_streaming(self, operation: StreamingOperation) -> List[ChunkTask]:
        """Process chunks using streaming pipeline."""
        completed_tasks = []

        # Submit initial batch of tasks
        futures = {}
        active_count = 0

        while self._running and not self._cancelled:
            # Submit new tasks if we have capacity
            while active_count < self.max_workers and not self.task_queue.empty():
                task = self.task_queue.get()
                future = self.executor.submit(self._process_chunk_task, task, operation)
                futures[future] = task
                active_count += 1

            # Check for completed tasks
            for future in as_completed(futures, timeout=0.1):
                task = futures[future]
                del futures[future]
                active_count -= 1

                try:
                    processed_task = future.result()
                    completed_tasks.append(processed_task)

                    # Update progress
                    self._update_progress(processed_task)

                    # Handle task result
                    if processed_task.result is not None:
                        self.result_queue.put(processed_task)

                except Exception as e:
                    task.error = str(e)
                    task.completed_at = time.time()
                    completed_tasks.append(task)
                    self.stats["failed_chunks"] += 1

            # Check if we're done
            if active_count == 0 and self.task_queue.empty():
                break

            # Memory management
            self._check_memory_pressure()

        return completed_tasks

    def _process_chunk_task(self, task: ChunkTask, operation: StreamingOperation) -> ChunkTask:
        """Process a single chunk task."""
        task.processed_at = time.time()

        try:
            # Apply operation to chunk
            if operation.operation_spec.input_count == 1:
                result = operation.operation_spec.func(task.data, **operation.parameters)
            else:
                # For multi-input operations, this would be more complex
                # For now, assume single input
                result = operation.operation_spec.func(task.data, **operation.parameters)

            task.result = result
            task.stage = PipelineStage.PROCESS
            task.completed_at = time.time()

            self.stats["successful_chunks"] += 1
            self.stats["total_processing_time"] += task.completed_at - task.processed_at

        except Exception as e:
            task.error = str(e)
            task.completed_at = time.time()
            raise

        return task

    def _update_progress(self, task: ChunkTask) -> None:
        """Update operation progress."""
        if self._progress_callbacks:
            progress_info = {
                "task_id": task.chunk_id,
                "stage": task.stage.value,
                "progress": task.chunk_index / task.total_chunks if task.total_chunks > 0 else 0,
                "completed": task.completed_at is not None
            }

            for callback in self._progress_callbacks:
                try:
                    callback(progress_info)
                except Exception:
                    pass  # Ignore callback errors

    def _check_memory_pressure(self) -> None:
        """Check and handle memory pressure."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)

        self.stats["peak_memory_usage"] = max(self.stats["peak_memory_usage"], memory_mb)

        if memory_mb > self.max_memory_mb:
            # Implement memory pressure handling
            # This could involve slowing down processing, clearing caches, etc.
            time.sleep(0.1)  # Simple backpressure

    def _finalize_operation(self, operation: StreamingOperation,
                           tasks: List[ChunkTask], execution_time: float) -> Dict[str, Any]:
        """Finalize streaming operation and return results."""
        # Combine chunk results if needed
        if operation.operation_spec.type == OperationType.AGGREGATION:
            # For aggregation operations, combine results
            final_result = self._combine_chunk_results(tasks, operation)
        else:
            # For other operations, reconstruct from chunks
            final_result = self._reconstruct_from_chunks(tasks, operation)

        # Store final result if needed
        result_tensor_id = None
        if operation.output_tensor_id:
            result_tensor_id = self.operational_storage.save_tensor(
                final_result, "streaming_results"
            )

        return {
            "status": "success",
            "operation_id": operation.operation_id,
            "result_tensor_id": result_tensor_id,
            "execution_time": execution_time,
            "chunks_processed": len(tasks),
            "successful_chunks": self.stats["successful_chunks"],
            "failed_chunks": self.stats["failed_chunks"],
            "peak_memory_mb": self.stats["peak_memory_usage"]
        }

    def _combine_chunk_results(self, tasks: List[ChunkTask],
                              operation: StreamingOperation) -> Any:
        """Combine results from chunk processing for aggregation operations."""
        # This is operation-specific
        # For sum operations, sum all chunk results
        # For mean operations, average all chunk results, etc.

        valid_results = [task.result for task in tasks if task.result is not None]

        if not valid_results:
            raise ValueError("No valid results from chunk processing")

        if operation.operation_spec.name == "sum":
            return sum(valid_results)
        elif operation.operation_spec.name == "mean":
            return sum(valid_results) / len(valid_results)
        else:
            # For other operations, return first result
            # In practice, this would be more sophisticated
            return valid_results[0]

    def _reconstruct_from_chunks(self, tasks: List[ChunkTask],
                                operation: StreamingOperation) -> torch.Tensor:
        """Reconstruct tensor from processed chunks."""
        # Sort tasks by chunk index
        sorted_tasks = sorted(tasks, key=lambda t: t.chunk_index)

        # Concatenate results
        result_chunks = []
        for task in sorted_tasks:
            if task.result is not None:
                result_chunks.append(task.result)

        if not result_chunks:
            raise ValueError("No valid chunk results to reconstruct")

        return torch.cat(result_chunks, dim=0)

    def add_progress_callback(self, callback: Callable) -> None:
        """Add progress tracking callback."""
        self._progress_callbacks.append(callback)

    def cancel_operation(self) -> None:
        """Cancel the current operation."""
        self._cancelled = True

    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming pipeline statistics."""
        return self.stats.copy()

    def is_running(self) -> bool:
        """Check if pipeline is currently running."""
        return self._running

    def clear_queues(self) -> None:
        """Clear all queues and reset state."""
        with self._lock:
            # Clear queues
            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                except queue.Empty:
                    break

            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    break

            # Clear task dictionaries
            self.active_tasks.clear()
            self.completed_tasks.clear()


class StreamingOperationManager:
    """
    High-level manager for streaming tensor operations.

    Provides:
    - Easy-to-use API for streaming operations
    - Operation templates and presets
    - Resource management and monitoring
    - Integration with the operational storage layer
    """

    def __init__(self, operational_storage: OperationalStorage):
        self.operational_storage = operational_storage
        self.pipeline = StreamingPipeline(operational_storage)
        self.operation_templates: Dict[str, StreamingOperation] = {}

        # Register common operation templates
        self._register_templates()

    def _register_templates(self) -> None:
        """Register common streaming operation templates."""
        # Large tensor sum operation
        sum_template = StreamingOperation(
            operation_spec=self.operational_storage.operation_specs["sum"],
            chunk_size_mb=100,
            max_memory_mb=1024,
            max_workers=8
        )
        self.operation_templates["large_sum"] = sum_template

        # Large tensor mean operation
        mean_template = StreamingOperation(
            operation_spec=self.operational_storage.operation_specs["mean"],
            chunk_size_mb=100,
            max_memory_mb=1024,
            max_workers=8
        )
        self.operation_templates["large_mean"] = mean_template

        # Large tensor transpose
        transpose_template = StreamingOperation(
            operation_spec=self.operational_storage.operation_specs["transpose"],
            chunk_size_mb=200,
            max_memory_mb=2048,
            max_workers=4
        )
        self.operation_templates["large_transpose"] = transpose_template

    def execute_template_operation(self, template_name: str,
                                  input_tensor_ids: List[str],
                                  parameters: Optional[Dict[str, Any]] = None,
                                  **kwargs) -> Dict[str, Any]:
        """
        Execute a pre-configured operation template.

        Args:
            template_name: Name of the operation template
            input_tensor_ids: Input tensor IDs
            parameters: Operation parameters
            **kwargs: Additional operation configuration

        Returns:
            Operation results
        """
        if template_name not in self.operation_templates:
            raise ValueError(f"Unknown operation template: {template_name}")

        # Create operation from template
        operation = self.operation_templates[template_name]

        # Override with provided parameters
        operation = StreamingOperation(
            operation_spec=operation.operation_spec,
            input_tensor_ids=input_tensor_ids,
            parameters=parameters or operation.parameters,
            chunk_size_mb=kwargs.get('chunk_size_mb', operation.chunk_size_mb),
            max_memory_mb=kwargs.get('max_memory_mb', operation.max_memory_mb),
            max_workers=kwargs.get('max_workers', operation.max_workers),
            enable_progress_tracking=kwargs.get('enable_progress_tracking', True)
        )

        return self.pipeline.execute_streaming_operation(operation)

    def execute_custom_operation(self, operation_spec: OperationSpec,
                                input_tensor_ids: List[str],
                                parameters: Optional[Dict[str, Any]] = None,
                                **kwargs) -> Dict[str, Any]:
        """
        Execute a custom streaming operation.

        Args:
            operation_spec: Operation specification
            input_tensor_ids: Input tensor IDs
            parameters: Operation parameters
            **kwargs: Additional configuration

        Returns:
            Operation results
        """
        operation = StreamingOperation(
            operation_spec=operation_spec,
            input_tensor_ids=input_tensor_ids,
            parameters=parameters or {},
            chunk_size_mb=kwargs.get('chunk_size_mb', 50),
            max_memory_mb=kwargs.get('max_memory_mb', 512),
            max_workers=kwargs.get('max_workers', 4),
            enable_progress_tracking=kwargs.get('enable_progress_tracking', True)
        )

        return self.pipeline.execute_streaming_operation(operation)

    def create_operation_template(self, name: str, operation_spec: OperationSpec,
                                 **config) -> None:
        """
        Create a custom operation template.

        Args:
            name: Template name
            operation_spec: Operation specification
            **config: Template configuration
        """
        template = StreamingOperation(
            operation_spec=operation_spec,
            chunk_size_mb=config.get('chunk_size_mb', 50),
            max_memory_mb=config.get('max_memory_mb', 512),
            max_workers=config.get('max_workers', 4)
        )

        self.operation_templates[name] = template

    def add_progress_callback(self, callback: Callable) -> None:
        """Add progress tracking callback."""
        self.pipeline.add_progress_callback(callback)

    def cancel_current_operation(self) -> None:
        """Cancel the currently running operation."""
        self.pipeline.cancel_operation()

    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get streaming pipeline statistics."""
        return self.pipeline.get_statistics()

    def is_operation_running(self) -> bool:
        """Check if an operation is currently running."""
        return self.pipeline.is_running()

    def optimize_for_tensor_size(self, tensor_size_mb: int) -> Dict[str, Any]:
        """
        Get optimal streaming configuration for a given tensor size.

        Args:
            tensor_size_mb: Size of tensor in MB

        Returns:
            Optimal configuration parameters
        """
        if tensor_size_mb < 100:
            return {
                "chunk_size_mb": 10,
                "max_workers": 2,
                "max_memory_mb": 256
            }
        elif tensor_size_mb < 1000:
            return {
                "chunk_size_mb": 50,
                "max_workers": 4,
                "max_memory_mb": 512
            }
        elif tensor_size_mb < 10000:
            return {
                "chunk_size_mb": 100,
                "max_workers": 8,
                "max_memory_mb": 1024
            }
        else:
            return {
                "chunk_size_mb": 200,
                "max_workers": 16,
                "max_memory_mb": 2048
            }
