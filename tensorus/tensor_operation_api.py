"""
Tensor Operation Query Language and API for Tensorus

This module provides a powerful query language and API for performing
complex operations on stored tensors, combining the indexing system
with operational capabilities.

Key Features:
- SQL-like query language for tensor operations
- Declarative operation specifications
- Integration with indexing for optimized execution
- Batch operations on multiple tensors
- Result aggregation and analysis
- Query optimization and planning
"""

import os
import time
import re
from typing import Dict, List, Set, Tuple, Optional, Any, Iterator, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import ast
import operator

from tensorus.tensor_operations_integrated import OperationalStorage, OperationalTensor
from tensorus.tensor_streaming_pipeline import StreamingOperationManager
from tensorus.metadata.index_manager import IndexManager


class QueryOperation(Enum):
    """Operations supported in tensor queries."""
    SELECT = "select"       # Select tensors based on conditions
    COMPUTE = "compute"     # Apply operations to selected tensors
    AGGREGATE = "aggregate" # Aggregate results
    TRANSFORM = "transform" # Transform tensor structures
    ANALYZE = "analyze"     # Analyze tensor properties


@dataclass
class TensorQuery:
    """Represents a tensor query with operations."""
    query_id: str = field(default_factory=lambda: str(id(time.time())))
    operation: QueryOperation = QueryOperation.SELECT
    conditions: Dict[str, Any] = field(default_factory=dict)
    operations: List[Dict[str, Any]] = field(default_factory=list)
    aggregation: Optional[Dict[str, Any]] = None
    output_format: str = "tensor"  # "tensor", "dataframe", "statistics"
    limit: Optional[int] = None
    sort_by: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class QueryResult:
    """Result of a tensor query execution."""
    query_id: str
    status: str
    result_tensors: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    error_message: Optional[str] = None


class TensorQueryParser:
    """
    Parser for tensor query language.

    Supports SQL-like syntax for tensor operations:
    - SELECT tensors WHERE conditions COMPUTE operation
    - SELECT tensors WHERE conditions AGGREGATE function
    - COMPUTE operation ON tensors
    """

    def __init__(self):
        self.operation_patterns = {
            'select': re.compile(r'SELECT\s+(.+?)\s+WHERE\s+(.+?)\s+COMPUTE\s+(.+)', re.IGNORECASE),
            'select_aggregate': re.compile(r'SELECT\s+(.+?)\s+WHERE\s+(.+?)\s+AGGREGATE\s+(.+)', re.IGNORECASE),
            'compute': re.compile(r'COMPUTE\s+(.+?)\s+ON\s+(.+)', re.IGNORECASE),
            'analyze': re.compile(r'ANALYZE\s+(.+?)\s+ON\s+(.+)', re.IGNORECASE)
        }

    def parse_query(self, query_string: str) -> TensorQuery:
        """
        Parse a query string into a TensorQuery object.

        Args:
            query_string: Query string in tensor query language

        Returns:
            Parsed TensorQuery object
        """
        query_string = query_string.strip()

        # Try different query patterns
        for query_type, pattern in self.operation_patterns.items():
            match = pattern.match(query_string)
            if match:
                return self._parse_matched_query(query_type, match)

        # If no pattern matches, try to parse as simple operation
        return self._parse_simple_operation(query_string)

    def _parse_matched_query(self, query_type: str, match) -> TensorQuery:
        """Parse a matched query pattern."""
        if query_type == 'select':
            target, conditions_str, operation_str = match.groups()
            return TensorQuery(
                operation=QueryOperation.SELECT,
                conditions=self._parse_conditions(conditions_str),
                operations=[self._parse_operation(operation_str)]
            )

        elif query_type == 'select_aggregate':
            target, conditions_str, aggregate_str = match.groups()
            return TensorQuery(
                operation=QueryOperation.AGGREGATE,
                conditions=self._parse_conditions(conditions_str),
                aggregation=self._parse_aggregation(aggregate_str)
            )

        elif query_type == 'compute':
            operation_str, target_str = match.groups()
            return TensorQuery(
                operation=QueryOperation.COMPUTE,
                operations=[self._parse_operation(operation_str)],
                conditions=self._parse_target(target_str)
            )

        elif query_type == 'analyze':
            analysis_str, target_str = match.groups()
            return TensorQuery(
                operation=QueryOperation.ANALYZE,
                operations=[self._parse_analysis(analysis_str)],
                conditions=self._parse_target(target_str)
            )

        return TensorQuery()

    def _parse_simple_operation(self, query_string: str) -> TensorQuery:
        """Parse a simple operation query."""
        # Handle simple cases like "sum(tensor_1)" or "tensor_1 + tensor_2"
        return TensorQuery(
            operation=QueryOperation.COMPUTE,
            operations=[{"type": "expression", "expression": query_string}]
        )

    def _parse_conditions(self, conditions_str: str) -> Dict[str, Any]:
        """Parse WHERE conditions."""
        conditions = {}

        # Split by AND
        condition_parts = conditions_str.split('AND')
        for part in condition_parts:
            part = part.strip()
            if '=' in part:
                key, value = part.split('=', 1)
                conditions[key.strip()] = self._parse_value(value.strip())
            elif '>' in part:
                key, value = part.split('>', 1)
                conditions[key.strip()] = {"$gt": self._parse_value(value.strip())}
            elif '<' in part:
                key, value = part.split('<', 1)
                conditions[key.strip()] = {"$lt": self._parse_value(value.strip())}
            elif '>=' in part:
                key, value = part.split('>=', 1)
                conditions[key.strip()] = {"$gte": self._parse_value(value.strip())}
            elif '<=' in part:
                key, value = part.split('<=', 1)
                conditions[key.strip()] = {"$lte": self._parse_value(value.strip())}

        return conditions

    def _parse_operation(self, operation_str: str) -> Dict[str, Any]:
        """Parse operation specification."""
        operation_str = operation_str.strip()

        # Handle function calls like "sum()", "mean(dim=0)"
        if '(' in operation_str and ')' in operation_str:
            func_name = operation_str.split('(')[0].strip()
            params_str = operation_str.split('(')[1].split(')')[0]

            params = {}
            if params_str:
                param_pairs = params_str.split(',')
                for pair in param_pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        params[key.strip()] = self._parse_value(value.strip())

            return {
                "type": "function",
                "name": func_name,
                "parameters": params
            }

        # Handle binary operations
        for op in ['+', '-', '*', '/', '@']:
            if op in operation_str:
                left, right = operation_str.split(op, 1)
                return {
                    "type": "binary",
                    "operator": op,
                    "left": left.strip(),
                    "right": right.strip()
                }

        # Default to expression
        return {
            "type": "expression",
            "expression": operation_str
        }

    def _parse_aggregation(self, aggregate_str: str) -> Dict[str, Any]:
        """Parse aggregation specification."""
        return {
            "function": aggregate_str.strip(),
            "parameters": {}
        }

    def _parse_analysis(self, analysis_str: str) -> Dict[str, Any]:
        """Parse analysis specification."""
        return {
            "type": "analysis",
            "analysis": analysis_str.strip()
        }

    def _parse_target(self, target_str: str) -> Dict[str, Any]:
        """Parse query target."""
        return {"target": target_str.strip()}

    def _parse_value(self, value_str: str) -> Any:
        """Parse a value from string."""
        value_str = value_str.strip()

        # Handle quoted strings
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]

        # Handle numbers
        try:
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass

        # Handle booleans
        if value_str.lower() == 'true':
            return True
        elif value_str.lower() == 'false':
            return False

        # Return as string
        return value_str


class TensorQueryExecutor:
    """
    Executor for tensor queries.

    Translates parsed queries into executable operations and manages
    their execution using the operational storage layer.
    """

    def __init__(self, operational_storage: OperationalStorage,
                 streaming_manager: Optional[StreamingOperationManager] = None):
        self.operational_storage = operational_storage
        self.streaming_manager = streaming_manager
        self.query_history: List[TensorQuery] = []
        self.result_cache: Dict[str, QueryResult] = {}

    def execute_query(self, query: Union[str, TensorQuery],
                     async_execution: bool = False) -> Union[QueryResult, str]:
        """
        Execute a tensor query.

        Args:
            query: Query string or TensorQuery object
            async_execution: Execute asynchronously

        Returns:
            QueryResult or operation ID (for async)
        """
        print(f"[DEBUG] execute_query called with query: {query}")
        start_time = time.time()

        # Parse query if it's a string
        if isinstance(query, str):
            print(f"[DEBUG] Parsing query string: {query}")
            parser = TensorQueryParser()
            query = parser.parse_query(query)
            print(f"[DEBUG] Parsed query: {query}")

        # Check cache
        cache_key = self._get_query_cache_key(query)
        if cache_key in self.result_cache:
            cached_result = self.result_cache[cache_key]
            # Check if cache is still valid (simplified)
            return cached_result

        try:
            # Execute based on operation type
            if query.operation == QueryOperation.SELECT:
                result = self._execute_select_query(query)
            elif query.operation == QueryOperation.COMPUTE:
                result = self._execute_compute_query(query)
            elif query.operation == QueryOperation.AGGREGATE:
                result = self._execute_aggregate_query(query)
            elif query.operation == QueryOperation.ANALYZE:
                result = self._execute_analyze_query(query)
            else:
                result = QueryResult(
                    query_id=query.query_id,
                    status="error",
                    error_message=f"Unsupported operation: {query.operation.value}"
                )

            result.execution_time = time.time() - start_time

            # Cache result
            self.result_cache[cache_key] = result
            self.query_history.append(query)

            return result

        except Exception as e:
            return QueryResult(
                query_id=query.query_id,
                status="error",
                error_message=str(e),
                execution_time=time.time() - start_time
            )

    def _execute_select_query(self, query: TensorQuery) -> QueryResult:
        """Execute a SELECT query."""
        # Use indexing to find tensors
        tensor_ids = self.operational_storage.index_manager.query_tensors(
            query.conditions, limit=query.limit, sort_by=query.sort_by
        )

        # Apply operations if specified
        if query.operations:
            result_ids = []
            for tensor_id in tensor_ids:
                for operation in query.operations:
                    try:
                        result = self.operational_storage.execute_operation(
                            operation["name"], [tensor_id],
                            operation.get("parameters", {}), "query_results"
                        )
                        if hasattr(result, 'result_tensor_ids'):
                            result_ids.extend(result.result_tensor_ids)
                        elif hasattr(result, 'status') and result.status == "success":
                            result_ids.extend(result.result_tensor_ids)
                    except Exception as e:
                        print(f"Failed to apply operation to {tensor_id}: {e}")
            tensor_ids = result_ids

        return QueryResult(
            query_id=query.query_id,
            status="success",
            result_tensors=tensor_ids,
            statistics={"selected_count": len(tensor_ids)}
        )

    def _execute_compute_query(self, query: TensorQuery) -> QueryResult:
        """Execute a COMPUTE query."""
        result_ids = []

        for operation in query.operations:
            if operation["type"] == "function":
                # Find target tensors
                target_conditions = query.conditions
                tensor_ids = self.operational_storage.index_manager.query_tensors(target_conditions)

                # Apply operation
                result = self.operational_storage.execute_operation(
                    operation["name"], tensor_ids,
                    operation.get("parameters", {}), "query_results"
                )

                if hasattr(result, 'result_tensor_ids'):
                    result_ids.extend(result.result_tensor_ids)

            elif operation["type"] == "binary":
                # Handle binary operations
                left_tensors = self._resolve_tensor_expression(operation["left"])
                right_tensors = self._resolve_tensor_expression(operation["right"])

                # Apply operation to corresponding tensors
                for left_id, right_id in zip(left_tensors, right_tensors):
                    result = self.operational_storage.execute_operation(
                        self._get_binary_operation_name(operation["operator"]),
                        [left_id, right_id], {}, "query_results"
                    )
                    if hasattr(result, 'result_tensor_ids'):
                        result_ids.extend(result.result_tensor_ids)

        return QueryResult(
            query_id=query.query_id,
            status="success",
            result_tensors=result_ids,
            statistics={"computed_count": len(result_ids)}
        )

    def _execute_aggregate_query(self, query: TensorQuery) -> QueryResult:
        """Execute an AGGREGATE query."""
        # Find target tensors
        tensor_ids = self.operational_storage.index_manager.query_tensors(query.conditions)

        if not tensor_ids:
            return QueryResult(
                query_id=query.query_id,
                status="success",
                statistics={"aggregated_count": 0, "result": None}
            )

        # Apply aggregation using streaming if available
        if self.streaming_manager and len(tensor_ids) > 1:
            aggregate_result = self.streaming_manager.execute_template_operation(
                f"large_{query.aggregation['function']}", tensor_ids
            )
            result_value = aggregate_result.get("result_tensor_id")
        else:
            # Simple aggregation for small datasets
            tensors = [self.operational_storage.load_tensor(tid) for tid in tensor_ids]

            if query.aggregation['function'] == 'sum':
                result_value = sum(tensors)
            elif query.aggregation['function'] == 'mean':
                result_value = sum(tensors) / len(tensors)
            else:
                result_value = tensors[0]  # Default

            # Store result
            result_id = self.operational_storage.save_tensor(result_value, "query_results")
            result_value = result_id

        return QueryResult(
            query_id=query.query_id,
            status="success",
            result_tensors=[result_value] if isinstance(result_value, str) else [],
            statistics={
                "aggregated_count": len(tensor_ids),
                "aggregation_function": query.aggregation['function'],
                "result": result_value
            }
        )

    def _execute_analyze_query(self, query: TensorQuery) -> QueryResult:
        """Execute an ANALYZE query."""
        # Find target tensors
        tensor_ids = self.operational_storage.index_manager.query_tensors(query.conditions)

        statistics = {}
        for tensor_id in tensor_ids:
            tensor = self.operational_storage.load_tensor(tensor_id)
            tensor_stats = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
                "mean": float(tensor.mean()) if tensor.numel() > 0 else 0,
                "std": float(tensor.std()) if tensor.numel() > 0 else 0,
                "min": float(tensor.min()) if tensor.numel() > 0 else 0,
                "max": float(tensor.max()) if tensor.numel() > 0 else 0
            }
            statistics[tensor_id] = tensor_stats

        return QueryResult(
            query_id=query.query_id,
            status="success",
            statistics={
                "analyzed_count": len(tensor_ids),
                "tensor_statistics": statistics
            }
        )

    def _resolve_tensor_expression(self, expression: str) -> List[str]:
        """Resolve a tensor expression to tensor IDs."""
        # Simple resolution - in practice this would be more sophisticated
        if expression.startswith("tensor_"):
            return [expression]
        else:
            # Assume it's a condition-based selection
            conditions = {"tensor_id": expression}
            return self.operational_storage.index_manager.query_tensors(conditions)

    def _get_binary_operation_name(self, operator: str) -> str:
        """Get operation name for binary operator."""
        op_map = {
            "+": "add",
            "-": "subtract",
            "*": "multiply",
            "/": "divide",
            "@": "matmul"
        }
        return op_map.get(operator, "add")

    def _get_query_cache_key(self, query: TensorQuery) -> str:
        """Generate cache key for query."""
        import hashlib
        query_str = f"{query.operation.value}_{str(sorted(query.conditions.items()))}_{str(query.operations)}"
        return hashlib.md5(query_str.encode()).hexdigest()

    def get_query_history(self) -> List[TensorQuery]:
        """Get query execution history."""
        return self.query_history.copy()

    def clear_cache(self) -> None:
        """Clear query result cache."""
        self.result_cache.clear()


class TensorOperationAPI:
    """
    High-level API for tensor operations on stored data.

    Provides:
    - Simple programmatic interface for tensor operations
    - Integration with query language
    - Batch operation support
    - Result management and retrieval
    """

    def __init__(self, operational_storage: OperationalStorage,
                 streaming_manager: Optional[StreamingOperationManager] = None):
        self.operational_storage = operational_storage
        self.streaming_manager = streaming_manager
        self.query_executor = TensorQueryExecutor(operational_storage, streaming_manager)

    def _format_conditions(self, conditions: Dict[str, Any]) -> str:
        """Format conditions into a query string.
        
        Args:
            conditions: Dictionary of conditions {field: value}
            
        Returns:
            Formatted conditions string
        """
        if not conditions:
            return "1=1"  # Always true condition
            
        conditions_list = []
        for field, value in conditions.items():
            if isinstance(value, str):
                # Handle string equality
                conditions_list.append(f"{field} = '{value}'")
            else:
                # Handle other types directly
                conditions_list.append(f"{field} = {value}")
                
        return " AND ".join(conditions_list)
        
    def select_tensors(self, conditions: Dict[str, Any],
                      operations: Optional[List[str]] = None,
                      limit: Optional[int] = None) -> List[OperationalTensor]:
        """
        Select tensors based on conditions and optionally apply operations.

        Args:
            conditions: Selection conditions
            operations: List of operations to apply
            limit: Maximum number of results

        Returns:
            List of OperationalTensor objects
        """
        print(f"[DEBUG] select_tensors called with conditions: {conditions}")
        # Build query
        query_str = f"SELECT tensors WHERE {self._format_conditions(conditions)}"
        print(f"[DEBUG] Generated query string: {query_str}")

        if operations:
            operation_str = " COMPUTE " + " AND ".join(operations)
            query_str += operation_str

        # Execute query
        result = self.query_executor.execute_query(query_str)

        if result.status == "success":
            return [self.operational_storage.get_tensor(tid) for tid in result.result_tensors]
        else:
            raise RuntimeError(f"Query failed: {result.error_message}")

    def compute_operation(self, operation: str, tensor_ids: List[str],
                         parameters: Optional[Dict[str, Any]] = None) -> OperationalTensor:
        """
        Apply an operation to specified tensors.

        Args:
            operation: Operation name
            tensor_ids: Target tensor IDs
            parameters: Operation parameters

        Returns:
            Result tensor
        """
        query_str = f"COMPUTE {operation} ON {','.join(tensor_ids)}"
        result = self.query_executor.execute_query(query_str)

        if result.status == "success" and result.result_tensors:
            return self.operational_storage.get_tensor(result.result_tensors[0])
        else:
            raise RuntimeError(f"Operation failed: {result.error_message}")

    def aggregate_tensors(self, conditions: Dict[str, Any],
                         aggregation: str) -> Any:
        """
        Aggregate tensors matching conditions.

        Args:
            conditions: Selection conditions
            aggregation: Aggregation function

        Returns:
            Aggregation result
        """
        query_str = f"SELECT tensors WHERE {self._format_conditions(conditions)} AGGREGATE {aggregation}"
        result = self.query_executor.execute_query(query_str)

        if result.status == "success":
            return result.statistics.get("result")
        else:
            raise RuntimeError(f"Aggregation failed: {result.error_message}")

    def analyze_tensors(self, conditions: Dict[str, Any],
                       analysis: str) -> Dict[str, Any]:
        """
        Analyze tensors matching conditions.

        Args:
            conditions: Selection conditions
            analysis: Analysis type

        Returns:
            Analysis results
        """
        query_str = f"ANALYZE {analysis} ON tensors WHERE {self._format_conditions(conditions)}"
        result = self.query_executor.execute_query(query_str)

        if result.status == "success":
            return result.statistics
        else:
            raise RuntimeError(f"Analysis failed: {result.error_message}")

    def batch_operations(self, operations: List[Dict[str, Any]]) -> List[OperationalTensor]:
        """
        Execute multiple operations in batch.

        Args:
            operations: List of operation specifications

        Returns:
            List of result tensors
        """
        results = []

        for op_spec in operations:
            try:
                if op_spec["type"] == "compute":
                    result = self.compute_operation(
                        op_spec["operation"],
                        op_spec["tensor_ids"],
                        op_spec.get("parameters")
                    )
                elif op_spec["type"] == "aggregate":
                    result = self.aggregate_tensors(
                        op_spec["conditions"],
                        op_spec["aggregation"]
                    )
                    # Convert to OperationalTensor if it's a tensor ID
                    if isinstance(result, str):
                        result = self.operational_storage.get_tensor(result)
                else:
                    continue

                results.append(result)

            except Exception as e:
                print(f"Batch operation failed: {e}")
                continue

        return results

    def _format_conditions(self, conditions: Dict[str, Any]) -> str:
        """Format conditions for query string."""
        condition_parts = []

        for key, value in conditions.items():
            if isinstance(value, dict):
                for op, val in value.items():
                    if op == "$gt":
                        condition_parts.append(f"{key} > {val}")
                    elif op == "$lt":
                        condition_parts.append(f"{key} < {val}")
                    elif op == "$gte":
                        condition_parts.append(f"{key} >= {val}")
                    elif op == "$lte":
                        condition_parts.append(f"{key} <= {val}")
            else:
                condition_parts.append(f"{key} = {value}")

        return " AND ".join(condition_parts)

    def get_operation_history(self) -> List[TensorQuery]:
        """Get operation execution history."""
        return self.query_executor.get_query_history()

    def clear_caches(self) -> None:
        """Clear all caches."""
        self.query_executor.clear_cache()
        self.operational_storage.cache.clear()
