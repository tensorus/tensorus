import time
import logging
import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from functools import wraps
import tracemalloc
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorBenchmark:
    """
    Benchmarking tools for measuring and optimizing Tensorus performance.
    """
    
    def __init__(self, database_ref=None, output_dir: str = "benchmarks"):
        """
        Initialize the benchmarking tools.
        
        Args:
            database_ref: Reference to the TensorDatabase instance
            output_dir: Directory to save benchmark results
        """
        self.database = database_ref
        self.output_dir = output_dir
        self.results = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("TensorBenchmark initialized")
    
    def set_database(self, database_ref):
        """
        Set the database reference.
        
        Args:
            database_ref: Reference to the TensorDatabase instance
        """
        self.database = database_ref
    
    def time_function(self, 
                      func: Callable, 
                      *args, 
                      iterations: int = 10, 
                      warmup: int = 1,
                      **kwargs) -> Dict[str, Any]:
        """
        Measure execution time of a function.
        
        Args:
            func: Function to benchmark
            *args: Arguments to pass to the function
            iterations: Number of iterations to run
            warmup: Number of warmup iterations
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Dictionary with timing results
        """
        # Run warmup iterations
        for _ in range(warmup):
            _ = func(*args, **kwargs)
        
        # Measure actual iterations
        times = []
        for _ in range(iterations):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate statistics
        times_array = np.array(times)
        stats = {
            "mean": float(np.mean(times_array)),
            "min": float(np.min(times_array)),
            "max": float(np.max(times_array)),
            "median": float(np.median(times_array)),
            "std": float(np.std(times_array)),
            "iterations": iterations,
            "warmup": warmup
        }
        
        return {
            "function": func.__name__,
            "times": times,
            "stats": stats,
            "result": result
        }
    
    def measure_memory(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Measure memory usage of a function.
        
        Args:
            func: Function to benchmark
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Dictionary with memory usage results
        """
        # Force garbage collection
        gc.collect()
        
        # Start memory tracing
        tracemalloc.start()
        
        # Run function
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            "function": func.__name__,
            "current_memory": current / (1024 * 1024),  # MB
            "peak_memory": peak / (1024 * 1024),  # MB
            "execution_time": execution_time,
            "result": result
        }
    
    def benchmark_indexing(self, 
                          dimensions: List[int] = None, 
                          index_types: List[str] = None,
                          metrics: List[str] = None,
                          num_tensors: int = 1000,
                          k: int = 10) -> Dict[str, Any]:
        """
        Benchmark indexing performance for different configurations.
        
        Args:
            dimensions: List of tensor dimensions to test
            index_types: List of index types to test
            metrics: List of distance metrics to test
            num_tensors: Number of tensors to index
            k: Number of nearest neighbors to search for
            
        Returns:
            Dictionary with benchmark results
        """
        if self.database is None:
            raise ValueError("Database reference not set")
        
        # Default test configurations
        if dimensions is None:
            dimensions = [64, 128, 256, 512, 1024]
        
        if index_types is None:
            index_types = ["flat", "ivf", "hnsw"]
        
        if metrics is None:
            metrics = ["l2", "ip"]
        
        results = {
            "index_creation": [],
            "tensor_addition": [],
            "search": []
        }
        
        for dimension in dimensions:
            for index_type in index_types:
                for metric in metrics:
                    logger.info(f"Benchmarking: dimension={dimension}, index_type={index_type}, metric={metric}")
                    
                    # Create temporary test data
                    tensors = [np.random.rand(dimension).astype('float32') for _ in range(num_tensors)]
                    query = np.random.rand(dimension).astype('float32')
                    
                    # Benchmark index creation
                    from tensor_indexer import TensorIndexer
                    
                    def create_index():
                        return TensorIndexer(dimension=dimension, index_type=index_type, metric=metric)
                    
                    creation_result = self.time_function(create_index, iterations=5)
                    indexer = creation_result["result"]
                    
                    # Benchmark tensor addition
                    def add_tensors():
                        for i, tensor in enumerate(tensors):
                            indexer.add_tensor(tensor, f"test_{i}")
                    
                    addition_result = self.time_function(add_tensors, iterations=3)
                    
                    # Benchmark search
                    def search_tensors():
                        return indexer.search_tensor(query, k=k)
                    
                    search_result = self.time_function(search_tensors, iterations=10)
                    
                    # Save results
                    config = {
                        "dimension": dimension,
                        "index_type": index_type,
                        "metric": metric,
                        "num_tensors": num_tensors,
                        "k": k
                    }
                    
                    results["index_creation"].append({
                        **config,
                        "mean_time": creation_result["stats"]["mean"],
                        "std_time": creation_result["stats"]["std"]
                    })
                    
                    results["tensor_addition"].append({
                        **config,
                        "mean_time": addition_result["stats"]["mean"],
                        "std_time": addition_result["stats"]["std"],
                        "tensors_per_second": num_tensors / addition_result["stats"]["mean"]
                    })
                    
                    results["search"].append({
                        **config,
                        "mean_time": search_result["stats"]["mean"],
                        "std_time": search_result["stats"]["std"],
                        "qps": 1.0 / search_result["stats"]["mean"]  # Queries per second
                    })
        
        # Save to class instance
        self.results["indexing"] = results
        
        # Save to file
        self._save_results("indexing_benchmark", results)
        
        # Generate plots
        self._plot_indexing_results(results)
        
        return results
    
    def benchmark_storage(self, 
                        tensor_sizes: List[Tuple[int, ...]] = None,
                        compression_levels: List[int] = None,
                        batch_sizes: List[int] = None,
                        num_tensors: int = 100) -> Dict[str, Any]:
        """
        Benchmark storage performance for different configurations.
        
        Args:
            tensor_sizes: List of tensor shapes to test
            compression_levels: List of compression levels to test
            batch_sizes: List of batch sizes to test
            num_tensors: Total number of tensors to store
            
        Returns:
            Dictionary with benchmark results
        """
        if self.database is None:
            raise ValueError("Database reference not set")
        
        # Default test configurations
        if tensor_sizes is None:
            tensor_sizes = [(10, 10), (50, 50), (100, 100), (10, 10, 10)]
        
        if compression_levels is None:
            compression_levels = [0, 1, 3, 5, 9]  # 0 = no compression, 9 = max compression
        
        if batch_sizes is None:
            batch_sizes = [1, 10, 25, 50, 100]
        
        results = {
            "save": [],
            "load": [],
            "batch_save": [],
            "compression": []
        }
        
        for tensor_size in tensor_sizes:
            # Create test tensors of this size
            test_tensors = [np.random.rand(*tensor_size) for _ in range(num_tensors)]
            
            # Benchmark single tensor saves
            def save_tensors(batch_size=1):
                ids = []
                for i in range(0, num_tensors, batch_size):
                    if batch_size == 1:
                        # Single tensor save
                        tensor_id = self.database.save(test_tensors[i], {"test": True})
                        ids.append(tensor_id)
                    else:
                        # Batch save
                        batch = test_tensors[i:i+batch_size]
                        if len(batch) < batch_size:
                            # Skip incomplete batches for consistent timing
                            continue
                        batch_ids = self.database.batch_save(batch, [{"test": True} for _ in batch])
                        ids.extend(batch_ids)
                return ids
            
            # Benchmark different batch sizes
            for batch_size in batch_sizes:
                if batch_size == 1:
                    # Single tensor saves
                    save_result = self.time_function(save_tensors, batch_size=1, iterations=3)
                    ids = save_result["result"]
                    
                    # Benchmark loading
                    def load_tensors():
                        loaded = []
                        for tensor_id in ids[:100]:  # Limit to prevent too long tests
                            tensor, metadata = self.database.get(tensor_id)
                            loaded.append(tensor)
                        return loaded
                    
                    load_result = self.time_function(load_tensors, iterations=3)
                    
                    # Record results
                    size_str = "x".join(str(s) for s in tensor_size)
                    results["save"].append({
                        "tensor_size": size_str,
                        "shape": tensor_size,
                        "elements": np.prod(tensor_size),
                        "batch_size": 1,
                        "mean_time": save_result["stats"]["mean"],
                        "std_time": save_result["stats"]["std"],
                        "tensors_per_second": num_tensors / save_result["stats"]["mean"]
                    })
                    
                    results["load"].append({
                        "tensor_size": size_str,
                        "shape": tensor_size,
                        "elements": np.prod(tensor_size),
                        "mean_time": load_result["stats"]["mean"] / min(100, len(ids)),  # Per tensor
                        "std_time": load_result["stats"]["std"],
                        "tensors_per_second": min(100, len(ids)) / load_result["stats"]["mean"]
                    })
                else:
                    # Batch saves
                    batch_result = self.time_function(save_tensors, batch_size=batch_size, iterations=3)
                    
                    # Record results
                    size_str = "x".join(str(s) for s in tensor_size)
                    results["batch_save"].append({
                        "tensor_size": size_str,
                        "shape": tensor_size,
                        "elements": np.prod(tensor_size),
                        "batch_size": batch_size,
                        "mean_time": batch_result["stats"]["mean"],
                        "std_time": batch_result["stats"]["std"],
                        "tensors_per_second": num_tensors / batch_result["stats"]["mean"]
                    })
            
            # Benchmark compression levels
            import h5py
            for compression in compression_levels:
                temp_file = os.path.join(self.output_dir, f"temp_comp_{compression}.h5")
                
                def save_compressed():
                    with h5py.File(temp_file, "w") as f:
                        for i, tensor in enumerate(test_tensors[:25]):  # Limit to 25 for faster tests
                            f.create_dataset(f"tensor_{i}", data=tensor, compression="gzip", compression_opts=compression)
                
                compression_result = self.time_function(save_compressed, iterations=3)
                
                # Get file size
                file_size = os.path.getsize(temp_file) / (1024 * 1024)  # MB
                
                # Record results
                size_str = "x".join(str(s) for s in tensor_size)
                results["compression"].append({
                    "tensor_size": size_str,
                    "shape": tensor_size,
                    "elements": np.prod(tensor_size),
                    "compression_level": compression,
                    "mean_time": compression_result["stats"]["mean"],
                    "std_time": compression_result["stats"]["std"],
                    "file_size_mb": file_size,
                    "file_size_per_tensor_mb": file_size / 25
                })
                
                # Clean up
                os.remove(temp_file)
        
        # Save to class instance
        self.results["storage"] = results
        
        # Save to file
        self._save_results("storage_benchmark", results)
        
        # Generate plots
        self._plot_storage_results(results)
        
        return results
    
    def benchmark_processor(self, 
                          tensor_sizes: List[Tuple[int, ...]] = None,
                          operations: List[str] = None,
                          use_gpu: bool = False) -> Dict[str, Any]:
        """
        Benchmark processor performance for different operations.
        
        Args:
            tensor_sizes: List of tensor shapes to test
            operations: List of operations to test
            use_gpu: Whether to use GPU for processing
            
        Returns:
            Dictionary with benchmark results
        """
        from tensor_processor import TensorProcessor
        
        # Default test configurations
        if tensor_sizes is None:
            tensor_sizes = [(10, 10), (50, 50), (100, 100), (10, 10, 10), (50, 50, 10)]
        
        if operations is None:
            operations = ["add", "subtract", "multiply", "matmul", "reshape", "transpose"]
        
        results = []
        
        # Initialize processor
        processor = TensorProcessor(use_gpu=use_gpu)
        
        for tensor_size in tensor_sizes:
            logger.info(f"Benchmarking processor operations for tensor size: {tensor_size}")
            
            # Create test tensors
            tensor1 = np.random.rand(*tensor_size)
            tensor2 = np.random.rand(*tensor_size)
            
            for operation in operations:
                # Skip operations that require special handling
                if operation == "matmul" and len(tensor_size) > 2:
                    # matmul needs 2D tensors
                    continue
                
                logger.info(f"  Testing operation: {operation}")
                
                # Get the operation function
                op_func = getattr(processor, operation, None)
                if op_func is None:
                    logger.warning(f"Operation {operation} not available in processor")
                    continue
                
                # Prepare arguments based on operation
                if operation in ["add", "subtract", "multiply"]:
                    # Binary operations
                    def test_func():
                        return op_func(tensor1, tensor2)
                elif operation == "matmul":
                    # Matrix multiplication needs compatible shapes
                    if len(tensor_size) == 2:
                        # For 2D tensors, transpose the second tensor
                        tensor2_t = tensor2.T
                        def test_func():
                            return op_func(tensor1, tensor2_t)
                    else:
                        continue
                elif operation == "reshape":
                    # Reshape to a different shape with same number of elements
                    new_shape = (np.prod(tensor_size) // 2, 2)
                    def test_func():
                        return processor.reshape(tensor1, new_shape)
                elif operation == "transpose":
                    # Simple transpose
                    def test_func():
                        return processor.transpose(tensor1)
                else:
                    # Unary operations or special cases
                    def test_func():
                        return op_func(tensor1)
                
                # Run benchmark
                benchmark_result = self.time_function(test_func, iterations=20, warmup=5)
                
                # Record results
                size_str = "x".join(str(s) for s in tensor_size)
                results.append({
                    "tensor_size": size_str,
                    "shape": tensor_size,
                    "elements": np.prod(tensor_size),
                    "operation": operation,
                    "use_gpu": use_gpu,
                    "mean_time": benchmark_result["stats"]["mean"],
                    "std_time": benchmark_result["stats"]["std"],
                    "min_time": benchmark_result["stats"]["min"],
                    "max_time": benchmark_result["stats"]["max"],
                    "operations_per_second": 1.0 / benchmark_result["stats"]["mean"]
                })
        
        # Save to class instance
        self.results["processor"] = results
        
        # Save to file
        self._save_results("processor_benchmark", results)
        
        # Generate plots
        self._plot_processor_results(results)
        
        return results
    
    def benchmark_end_to_end(self, 
                          scenarios: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Benchmark end-to-end performance for different workloads.
        
        Args:
            scenarios: List of test scenarios
            
        Returns:
            Dictionary with benchmark results
        """
        if self.database is None:
            raise ValueError("Database reference not set")
        
        # Default test scenarios
        if scenarios is None:
            scenarios = [
                {
                    "name": "Small tensors, read-heavy",
                    "tensor_shape": (10, 10),
                    "num_tensors": 1000,
                    "reads_per_write": 10,
                    "search_fraction": 0.2,
                    "operations": ["add", "reshape"]
                },
                {
                    "name": "Large tensors, write-heavy",
                    "tensor_shape": (100, 100),
                    "num_tensors": 100,
                    "reads_per_write": 2,
                    "search_fraction": 0.1,
                    "operations": ["matmul", "transpose"]
                },
                {
                    "name": "Mixed workload",
                    "tensor_shape": (50, 50),
                    "num_tensors": 500,
                    "reads_per_write": 5,
                    "search_fraction": 0.5,
                    "operations": ["add", "subtract", "multiply", "transpose"]
                }
            ]
        
        results = []
        
        for scenario in scenarios:
            logger.info(f"Running end-to-end benchmark: {scenario['name']}")
            
            # Create test tensors
            tensor_shape = scenario["tensor_shape"]
            num_tensors = scenario["num_tensors"]
            tensors = [np.random.rand(*tensor_shape) for _ in range(num_tensors)]
            
            # Save all tensors first
            tensor_ids = []
            save_start = time.time()
            
            for tensor in tensors:
                tensor_id = self.database.save(tensor, {"benchmark": True})
                tensor_ids.append(tensor_id)
            
            save_time = time.time() - save_start
            
            # Run mixed workload
            ops_count = {
                "reads": 0,
                "writes": 0,
                "searches": 0,
                "operations": {}
            }
            
            workload_start = time.time()
            
            # Number of operations to perform
            total_ops = num_tensors * 5  # 5 operations per tensor
            
            import random
            for _ in range(total_ops):
                # Decide which operation to perform
                # - Reads
                if random.random() < (scenario["reads_per_write"] / (scenario["reads_per_write"] + 1)):
                    tensor_id = random.choice(tensor_ids)
                    tensor, _ = self.database.get(tensor_id)
                    ops_count["reads"] += 1
                # - Writes
                else:
                    tensor = np.random.rand(*tensor_shape)
                    tensor_id = self.database.save(tensor, {"benchmark": True})
                    tensor_ids.append(tensor_id)
                    ops_count["writes"] += 1
                
                # Searches (based on search fraction)
                if random.random() < scenario["search_fraction"]:
                    query = np.random.rand(*tensor_shape)
                    self.database.search_similar(query, k=5)
                    ops_count["searches"] += 1
                
                # Tensor operations
                if random.random() < 0.3:  # 30% chance to perform an operation
                    operation = random.choice(scenario["operations"])
                    if operation not in ops_count["operations"]:
                        ops_count["operations"][operation] = 0
                    
                    # Get one or two tensors for the operation
                    tensor1, _ = self.database.get(random.choice(tensor_ids))
                    
                    if operation in ["add", "subtract", "multiply", "matmul"]:
                        tensor2, _ = self.database.get(random.choice(tensor_ids))
                        self.database.process(operation, [tensor1, tensor2])
                    else:
                        self.database.process(operation, [tensor1])
                    
                    ops_count["operations"][operation] += 1
            
            workload_time = time.time() - workload_start
            
            # Record results
            results.append({
                "scenario": scenario["name"],
                "tensor_shape": tensor_shape,
                "num_tensors": num_tensors,
                "save_time": save_time,
                "workload_time": workload_time,
                "operations": ops_count,
                "ops_per_second": total_ops / workload_time
            })
        
        # Save to class instance
        self.results["end_to_end"] = results
        
        # Save to file
        self._save_results("end_to_end_benchmark", results)
        
        return results
    
    def _save_results(self, name: str, results: Dict[str, Any]):
        """Save benchmark results to file."""
        try:
            filename = os.path.join(self.output_dir, f"{name}_{int(time.time())}.json")
            with open(filename, 'w') as f:
                # Convert complex types to JSON-serializable formats
                json_results = self._convert_to_serializable(results)
                json.dump(json_results, f, indent=2)
            
            logger.info(f"Saved benchmark results to {filename}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _convert_to_serializable(self, obj):
        """Convert numpy and other complex types to JSON-serializable formats."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(i) for i in obj]
        elif isinstance(obj, tuple):
            return [self._convert_to_serializable(i) for i in obj]
        else:
            return obj
    
    def _plot_indexing_results(self, results: Dict[str, Any]):
        """Generate plots for indexing benchmark results."""
        try:
            # Create DataFrame from results
            creation_df = pd.DataFrame(results["index_creation"])
            addition_df = pd.DataFrame(results["tensor_addition"])
            search_df = pd.DataFrame(results["search"])
            
            # Plot index creation time vs dimension
            plt.figure(figsize=(12, 8))
            for index_type in creation_df["index_type"].unique():
                df_subset = creation_df[creation_df["index_type"] == index_type]
                plt.plot(df_subset["dimension"], df_subset["mean_time"], 
                         marker='o', label=f"{index_type}")
            
            plt.title("Index Creation Time vs Dimension")
            plt.xlabel("Dimension")
            plt.ylabel("Time (seconds)")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, "index_creation_time.png"))
            
            # Plot search QPS vs dimension for different index types
            plt.figure(figsize=(12, 8))
            for index_type in search_df["index_type"].unique():
                df_subset = search_df[search_df["index_type"] == index_type]
                plt.plot(df_subset["dimension"], df_subset["qps"], 
                         marker='o', label=f"{index_type}")
            
            plt.title("Search Performance (QPS) vs Dimension")
            plt.xlabel("Dimension")
            plt.ylabel("Queries Per Second")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, "search_qps.png"))
            
            # Plot tensors per second vs dimension for different index types
            plt.figure(figsize=(12, 8))
            for index_type in addition_df["index_type"].unique():
                df_subset = addition_df[addition_df["index_type"] == index_type]
                plt.plot(df_subset["dimension"], df_subset["tensors_per_second"], 
                         marker='o', label=f"{index_type}")
            
            plt.title("Tensor Addition Rate vs Dimension")
            plt.xlabel("Dimension")
            plt.ylabel("Tensors Per Second")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, "addition_rate.png"))
            
            logger.info("Generated indexing benchmark plots")
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
    
    def _plot_storage_results(self, results: Dict[str, Any]):
        """Generate plots for storage benchmark results."""
        try:
            # Create DataFrames from results
            save_df = pd.DataFrame(results["save"])
            load_df = pd.DataFrame(results["load"])
            batch_df = pd.DataFrame(results["batch_save"])
            compression_df = pd.DataFrame(results["compression"])
            
            # Plot save tensors per second vs tensor elements
            plt.figure(figsize=(12, 8))
            plt.plot(save_df["elements"], save_df["tensors_per_second"], marker='o')
            plt.title("Save Performance vs Tensor Size")
            plt.xlabel("Tensor Elements")
            plt.ylabel("Tensors Per Second")
            plt.grid(True)
            plt.xscale('log')
            plt.savefig(os.path.join(self.output_dir, "save_performance.png"))
            
            # Plot load tensors per second vs tensor elements
            plt.figure(figsize=(12, 8))
            plt.plot(load_df["elements"], load_df["tensors_per_second"], marker='o')
            plt.title("Load Performance vs Tensor Size")
            plt.xlabel("Tensor Elements")
            plt.ylabel("Tensors Per Second")
            plt.grid(True)
            plt.xscale('log')
            plt.savefig(os.path.join(self.output_dir, "load_performance.png"))
            
            # Plot batch save performance for different batch sizes
            plt.figure(figsize=(12, 8))
            for batch_size in batch_df["batch_size"].unique():
                df_subset = batch_df[batch_df["batch_size"] == batch_size]
                plt.plot(df_subset["elements"], df_subset["tensors_per_second"], 
                         marker='o', label=f"Batch size {batch_size}")
            
            plt.title("Batch Save Performance vs Tensor Size")
            plt.xlabel("Tensor Elements")
            plt.ylabel("Tensors Per Second")
            plt.legend()
            plt.grid(True)
            plt.xscale('log')
            plt.savefig(os.path.join(self.output_dir, "batch_save_performance.png"))
            
            # Plot compression ratio vs level for different tensor sizes
            plt.figure(figsize=(12, 8))
            for tensor_size in compression_df["tensor_size"].unique():
                df_subset = compression_df[compression_df["tensor_size"] == tensor_size]
                # Calculate ratio relative to no compression (level 0)
                no_comp = df_subset[df_subset["compression_level"] == 0]["file_size_per_tensor_mb"].values[0]
                ratios = [no_comp / row["file_size_per_tensor_mb"] for _, row in df_subset.iterrows()]
                plt.plot(df_subset["compression_level"], ratios, 
                         marker='o', label=f"Size {tensor_size}")
            
            plt.title("Compression Ratio vs Level")
            plt.xlabel("Compression Level")
            plt.ylabel("Compression Ratio")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, "compression_ratio.png"))
            
            logger.info("Generated storage benchmark plots")
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
    
    def _plot_processor_results(self, results: List[Dict[str, Any]]):
        """Generate plots for processor benchmark results."""
        try:
            # Create DataFrame from results
            df = pd.DataFrame(results)
            
            # Plot operations per second vs tensor elements for different operations
            plt.figure(figsize=(12, 8))
            for operation in df["operation"].unique():
                df_subset = df[df["operation"] == operation]
                plt.plot(df_subset["elements"], df_subset["operations_per_second"], 
                         marker='o', label=operation)
            
            plt.title("Operation Performance vs Tensor Size")
            plt.xlabel("Tensor Elements")
            plt.ylabel("Operations Per Second")
            plt.legend()
            plt.grid(True)
            plt.xscale('log')
            plt.yscale('log')
            plt.savefig(os.path.join(self.output_dir, "operation_performance.png"))
            
            # Plot mean execution time heatmap for operations and tensor sizes
            plt.figure(figsize=(14, 10))
            pivot = df.pivot_table(
                index="tensor_size", 
                columns="operation", 
                values="mean_time", 
                aggfunc='mean'
            )
            
            # Sort by number of elements
            size_order = df.set_index("tensor_size")["elements"].to_dict()
            pivot = pivot.iloc[sorted(pivot.index, key=lambda x: size_order[x])]
            
            import seaborn as sns
            ax = sns.heatmap(pivot, annot=True, fmt=".2e", cmap="viridis")
            plt.title("Mean Execution Time by Operation and Tensor Size")
            plt.ylabel("Tensor Size")
            plt.xlabel("Operation")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "operation_time_heatmap.png"))
            
            logger.info("Generated processor benchmark plots")
        except Exception as e:
            logger.error(f"Error generating plots: {e}")

def benchmark(func=None, *, iterations=10, warmup=1):
    """
    Decorator for benchmarking functions.
    
    Usage:
        @benchmark
        def function_to_benchmark(arg1, arg2):
            # Function body
            
        @benchmark(iterations=100, warmup=10)
        def another_function(arg1, arg2):
            # Function body
    """
    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            # Run warmup iterations
            for _ in range(warmup):
                _ = func(*args, **kwargs)
            
            # Measure actual iterations
            times = []
            for _ in range(iterations):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Calculate statistics
            times_array = np.array(times)
            stats = {
                "mean": float(np.mean(times_array)),
                "min": float(np.min(times_array)),
                "max": float(np.max(times_array)),
                "median": float(np.median(times_array)),
                "std": float(np.std(times_array)),
                "iterations": iterations,
                "warmup": warmup
            }
            
            logger.info(f"Benchmark for {func.__name__}: mean={stats['mean']:.6f}s, "
                        f"min={stats['min']:.6f}s, max={stats['max']:.6f}s")
            
            return result
        
        return _wrapper
    
    # Handle both @benchmark and @benchmark() syntax
    if func is None:
        return _decorator
    else:
        return _decorator(func) 