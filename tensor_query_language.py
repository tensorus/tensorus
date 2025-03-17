import re
import ast
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
import uuid
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TQLSyntaxError(Exception):
    """Exception raised for syntax errors in TQL queries."""
    pass

class TQLRuntimeError(Exception):
    """Exception raised for runtime errors during TQL execution."""
    pass

class TQLParser:
    """
    Parser for Tensor Query Language (TQL).
    Translates TQL statements into executable operations.
    """
    
    def __init__(self, database_ref=None):
        """
        Initialize the TQL parser.
        
        Args:
            database_ref: Reference to the TensorDatabase instance
        """
        self.database = database_ref
        
        # Define TQL grammar patterns
        self.patterns = {
            "select": r"SELECT\s+(.*?)\s+FROM\s+(.*?)(?:\s+WHERE\s+(.*?))?(?:\s+LIMIT\s+(\d+))?$",
            "create": r"CREATE\s+TENSOR\s+(.*?)\s+AS\s+(.*?)$",
            "update": r"UPDATE\s+(.*?)\s+SET\s+(.*?)(?:\s+WHERE\s+(.*?))?$",
            "delete": r"DELETE\s+FROM\s+(.*?)(?:\s+WHERE\s+(.*?))?$",
            "transform": r"TRANSFORM\s+(.*?)\s+USING\s+(.*?)(?:\s+WITH\s+(.*?))?$",
            "search": r"SEARCH\s+(.*?)\s+SIMILAR\s+TO\s+(.*?)(?:\s+LIMIT\s+(\d+))?$",
            "decompose": r"DECOMPOSE\s+(.*?)\s+USING\s+(.*?)(?:\s+WITH\s+(.*?))?$"
        }
        
        # Define operation handlers
        self.operations = {
            "select": self._handle_select,
            "create": self._handle_create,
            "update": self._handle_update,
            "delete": self._handle_delete,
            "transform": self._handle_transform,
            "search": self._handle_search,
            "decompose": self._handle_decompose
        }
        
        # Register available tensor operations
        self.tensor_operations = {
            "reshape": self._op_reshape,
            "transpose": self._op_transpose,
            "add": self._op_add,
            "subtract": self._op_subtract,
            "multiply": self._op_multiply,
            "matmul": self._op_matmul,
            "slice": self._op_slice,
            "concat": self._op_concat,
            "stack": self._op_stack,
            "norm": self._op_norm,
            "mean": self._op_mean,
            "max": self._op_max,
            "min": self._op_min,
            "sum": self._op_sum
        }
        
        # Register tensor decomposition methods
        self.decomposition_methods = {
            "cp": "decompose_cp",
            "tucker": "decompose_tucker",
            "svd": "tensor_svd"
        }
        
        logger.info("TQL Parser initialized")
    
    def set_database(self, database_ref):
        """
        Set the database reference.
        
        Args:
            database_ref: Reference to the TensorDatabase instance
        """
        self.database = database_ref
        logger.info("TQL Parser database reference updated")
    
    def parse(self, query: str) -> Dict[str, Any]:
        """
        Parse a TQL query into an executable form.
        
        Args:
            query: TQL query string
            
        Returns:
            Dictionary with parsed operation details
            
        Raises:
            TQLSyntaxError: If the query syntax is invalid
        """
        query = query.strip()
        
        # Check which operation type this is
        operation_type = None
        for op_type, pattern in self.patterns.items():
            if re.match(pattern, query, re.IGNORECASE):
                operation_type = op_type
                break
        
        if operation_type is None:
            raise TQLSyntaxError(f"Invalid TQL query: {query}")
        
        # Parse the operation
        match = re.match(self.patterns[operation_type], query, re.IGNORECASE)
        groups = match.groups()
        
        # Build operation details based on type
        operation_details = {
            "type": operation_type,
            "raw_query": query,
            "timestamp": uuid.uuid4(),  # For unique identification
        }
        
        if operation_type == "select":
            operation_details.update({
                "fields": self._parse_fields(groups[0]),
                "tensors": self._parse_tensors(groups[1]),
                "conditions": self._parse_conditions(groups[2]) if groups[2] else None,
                "limit": int(groups[3]) if groups[3] else None
            })
        elif operation_type == "create":
            operation_details.update({
                "tensor_name": groups[0].strip(),
                "expression": groups[1].strip()
            })
        elif operation_type == "update":
            operation_details.update({
                "tensor_name": groups[0].strip(),
                "updates": self._parse_updates(groups[1]),
                "conditions": self._parse_conditions(groups[2]) if groups[2] else None
            })
        elif operation_type == "delete":
            operation_details.update({
                "tensor_name": groups[0].strip(),
                "conditions": self._parse_conditions(groups[1]) if groups[1] else None
            })
        elif operation_type == "transform":
            operation_details.update({
                "tensor_name": groups[0].strip(),
                "operation": groups[1].strip(),
                "parameters": self._parse_parameters(groups[2]) if groups[2] else {}
            })
        elif operation_type == "search":
            operation_details.update({
                "tensor_name": groups[0].strip(),
                "query_tensor": groups[1].strip(),
                "limit": int(groups[2]) if groups[2] else 5
            })
        elif operation_type == "decompose":
            operation_details.update({
                "tensor_name": groups[0].strip(),
                "method": groups[1].strip().lower(),
                "parameters": self._parse_parameters(groups[2]) if groups[2] else {}
            })
        
        return operation_details
    
    def execute(self, query: str) -> Dict[str, Any]:
        """
        Execute a TQL query.
        
        Args:
            query: TQL query string
            
        Returns:
            Dictionary with execution results
            
        Raises:
            TQLSyntaxError: If the query syntax is invalid
            TQLRuntimeError: If an error occurs during execution
        """
        if self.database is None:
            raise TQLRuntimeError("No database connection available")
        
        # Parse the query
        operation_details = self.parse(query)
        
        # Execute the appropriate handler
        operation_type = operation_details["type"]
        if operation_type in self.operations:
            return self.operations[operation_type](operation_details)
        else:
            raise TQLRuntimeError(f"Unsupported operation type: {operation_type}")
    
    def _parse_fields(self, fields_str: str) -> List[str]:
        """Parse SELECT fields."""
        if fields_str.strip() == "*":
            return ["*"]
        return [f.strip() for f in fields_str.split(",")]
    
    def _parse_tensors(self, tensors_str: str) -> List[str]:
        """Parse tensor names from FROM clause."""
        return [t.strip() for t in tensors_str.split(",")]
    
    def _parse_conditions(self, conditions_str: str) -> Dict[str, Any]:
        """Parse WHERE conditions."""
        if not conditions_str:
            return {}
        
        # This is a simplified parser for basic conditions
        # In a real implementation, this would be more sophisticated
        conditions = {}
        
        # Check for OR conditions (not fully implemented)
        if " OR " in conditions_str:
            parts = conditions_str.split(" OR ")
            conditions["type"] = "OR"
            conditions["conditions"] = [self._parse_conditions(part) for part in parts]
        # Check for AND conditions
        elif " AND " in conditions_str:
            parts = conditions_str.split(" AND ")
            conditions["type"] = "AND"
            conditions["conditions"] = [self._parse_conditions(part) for part in parts]
        # Check for basic comparisons
        else:
            for op in ["==", "!=", ">=", "<=", ">", "<", "LIKE", "IN"]:
                if op in conditions_str:
                    left, right = conditions_str.split(op, 1)
                    conditions["type"] = "comparison"
                    conditions["left"] = left.strip()
                    conditions["operator"] = op
                    conditions["right"] = right.strip()
                    break
        
        return conditions
    
    def _parse_updates(self, updates_str: str) -> Dict[str, str]:
        """Parse SET updates."""
        updates = {}
        for update in updates_str.split(","):
            key, value = update.split("=", 1)
            updates[key.strip()] = value.strip()
        return updates
    
    def _parse_parameters(self, params_str: str) -> Dict[str, Any]:
        """Parse parameters for operations."""
        if not params_str:
            return {}
        
        parameters = {}
        
        # Split by commas, but respect parentheses for nested structures
        parts = []
        current_part = ""
        paren_level = 0
        
        for char in params_str:
            if char == "," and paren_level == 0:
                parts.append(current_part)
                current_part = ""
            else:
                if char == "(":
                    paren_level += 1
                elif char == ")":
                    paren_level -= 1
                current_part += char
        
        if current_part:
            parts.append(current_part)
        
        # Process each parameter
        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                parameters[key.strip()] = self._parse_value(value.strip())
        
        return parameters
    
    def _parse_value(self, value_str: str) -> Any:
        """Parse a value string into the appropriate Python type."""
        # Try to interpret as a Python literal (number, list, etc.)
        try:
            return ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            # If not a literal, handle special cases
            if value_str.startswith("tensor(") and value_str.endswith(")"):
                # Reference to another tensor
                tensor_name = value_str[7:-1].strip()
                return {"type": "tensor_ref", "name": tensor_name}
            elif value_str.startswith("[") and value_str.endswith("]"):
                # Try to parse as a list of values
                try:
                    items = value_str[1:-1].split(",")
                    return [self._parse_value(item.strip()) for item in items]
                except Exception:
                    pass
            
            # Default to treating as a string
            return value_str
    
    def _handle_select(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Handle SELECT operation."""
        result = {
            "operation": "select",
            "success": True,
            "data": []
        }
        
        try:
            # Get tensor data
            tensors = {}
            for tensor_name in operation["tensors"]:
                # Check if this is a special function
                if "(" in tensor_name and ")" in tensor_name:
                    func_match = re.match(r"(\w+)\((.*)\)", tensor_name)
                    if func_match:
                        func_name = func_match.group(1).lower()
                        args = [arg.strip() for arg in func_match.group(2).split(",")]
                        
                        if func_name == "all":
                            # Special function to get all tensors
                            all_tensors = self.database.list_tensors()
                            for t in all_tensors:
                                tensor_id = t["id"]
                                tensor_data, metadata = self.database.get(tensor_id)
                                tensors[tensor_id] = {
                                    "data": tensor_data,
                                    "metadata": metadata
                                }
                        else:
                            # Unsupported function
                            raise TQLRuntimeError(f"Unsupported function: {func_name}")
                else:
                    # Regular tensor lookup by ID
                    try:
                        tensor_data, metadata = self.database.get(tensor_name)
                        tensors[tensor_name] = {
                            "data": tensor_data,
                            "metadata": metadata
                        }
                    except KeyError:
                        raise TQLRuntimeError(f"Tensor not found: {tensor_name}")
            
            # Apply conditions if specified
            if operation["conditions"]:
                filtered_tensors = {}
                for tensor_id, tensor_info in tensors.items():
                    if self._evaluate_conditions(operation["conditions"], tensor_info):
                        filtered_tensors[tensor_id] = tensor_info
                tensors = filtered_tensors
            
            # Select fields
            for tensor_id, tensor_info in tensors.items():
                entry = {"id": tensor_id}
                
                if "*" in operation["fields"]:
                    # Include all fields
                    entry["data"] = tensor_info["data"]
                    entry["metadata"] = tensor_info["metadata"]
                else:
                    # Include only specified fields
                    for field in operation["fields"]:
                        if field == "data":
                            entry["data"] = tensor_info["data"]
                        elif field == "shape":
                            entry["shape"] = tensor_info["data"].shape
                        elif field == "metadata":
                            entry["metadata"] = tensor_info["metadata"]
                        elif field in tensor_info["metadata"]:
                            if "metadata" not in entry:
                                entry["metadata"] = {}
                            entry["metadata"][field] = tensor_info["metadata"][field]
                
                result["data"].append(entry)
            
            # Apply limit if specified
            if operation["limit"] is not None and operation["limit"] < len(result["data"]):
                result["data"] = result["data"][:operation["limit"]]
                
        except Exception as e:
            logger.error(f"Error executing SELECT: {e}")
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    def _handle_create(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Handle CREATE TENSOR operation."""
        result = {
            "operation": "create",
            "success": True
        }
        
        try:
            tensor_name = operation["tensor_name"]
            expression = operation["expression"]
            
            # Evaluate the expression to create a tensor
            tensor_data = self._evaluate_expression(expression)
            
            # Save the tensor
            tensor_id = self.database.save(
                tensor_data, 
                {"name": tensor_name, "created_from": "tql"}
            )
            
            result["tensor_id"] = tensor_id
            
        except Exception as e:
            logger.error(f"Error executing CREATE: {e}")
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    def _handle_update(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Handle UPDATE operation."""
        result = {
            "operation": "update",
            "success": True
        }
        
        try:
            tensor_name = operation["tensor_name"]
            updates = operation["updates"]
            
            # Get the tensor
            try:
                tensor_data, metadata = self.database.get(tensor_name)
            except KeyError:
                raise TQLRuntimeError(f"Tensor not found: {tensor_name}")
            
            # Check conditions if specified
            if operation["conditions"] and not self._evaluate_conditions(
                operation["conditions"], 
                {"data": tensor_data, "metadata": metadata}
            ):
                result["success"] = False
                result["error"] = "Conditions not satisfied"
                return result
            
            # Apply updates
            new_tensor = None
            new_metadata = dict(metadata)
            
            for key, value in updates.items():
                if key == "data":
                    # Update the tensor data
                    new_tensor = self._evaluate_expression(value)
                else:
                    # Update metadata
                    new_metadata[key] = self._parse_value(value)
            
            # Save the updated tensor
            success = self.database.update(tensor_name, new_tensor, new_metadata)
            result["success"] = success
            
            if not success:
                result["error"] = "Failed to update tensor"
            
        except Exception as e:
            logger.error(f"Error executing UPDATE: {e}")
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    def _handle_delete(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Handle DELETE operation."""
        result = {
            "operation": "delete",
            "success": True
        }
        
        try:
            tensor_name = operation["tensor_name"]
            
            # Get the tensor for condition evaluation
            if operation["conditions"]:
                try:
                    tensor_data, metadata = self.database.get(tensor_name)
                    # Check conditions
                    if not self._evaluate_conditions(
                        operation["conditions"], 
                        {"data": tensor_data, "metadata": metadata}
                    ):
                        result["success"] = False
                        result["error"] = "Conditions not satisfied"
                        return result
                except KeyError:
                    raise TQLRuntimeError(f"Tensor not found: {tensor_name}")
            
            # Delete the tensor
            success = self.database.delete(tensor_name)
            result["success"] = success
            
            if not success:
                result["error"] = "Failed to delete tensor"
            
        except Exception as e:
            logger.error(f"Error executing DELETE: {e}")
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    def _handle_transform(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Handle TRANSFORM operation."""
        result = {
            "operation": "transform",
            "success": True
        }
        
        try:
            tensor_name = operation["tensor_name"]
            operation_name = operation["operation"]
            parameters = operation["parameters"]
            
            # Get the tensor
            try:
                tensor_data, metadata = self.database.get(tensor_name)
            except KeyError:
                raise TQLRuntimeError(f"Tensor not found: {tensor_name}")
            
            # Check if the operation exists
            if operation_name not in self.tensor_operations:
                raise TQLRuntimeError(f"Unsupported tensor operation: {operation_name}")
            
            # Apply the transformation
            transform_func = self.tensor_operations[operation_name]
            transformed_tensor = transform_func(tensor_data, parameters)
            
            # Create a new tensor with the result
            new_metadata = dict(metadata)
            new_metadata["transformed_from"] = tensor_name
            new_metadata["operation"] = operation_name
            
            tensor_id = self.database.save(
                transformed_tensor, 
                new_metadata
            )
            
            result["tensor_id"] = tensor_id
            
        except Exception as e:
            logger.error(f"Error executing TRANSFORM: {e}")
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    def _handle_search(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Handle SEARCH operation."""
        result = {
            "operation": "search",
            "success": True,
            "results": []
        }
        
        try:
            tensor_name = operation["tensor_name"]
            query_tensor_expr = operation["query_tensor"]
            limit = operation["limit"]
            
            # Get the query tensor
            if query_tensor_expr.startswith("tensor(") and query_tensor_expr.endswith(")"):
                # Reference to another tensor
                query_tensor_name = query_tensor_expr[7:-1].strip()
                try:
                    query_tensor, _ = self.database.get(query_tensor_name)
                except KeyError:
                    raise TQLRuntimeError(f"Query tensor not found: {query_tensor_name}")
            else:
                # Evaluate as an expression
                query_tensor = self._evaluate_expression(query_tensor_expr)
            
            # Search for similar tensors
            search_results = self.database.search_similar(query_tensor, limit)
            
            # Format the results
            for item in search_results:
                result["results"].append({
                    "tensor_id": item["tensor_id"],
                    "distance": item["distance"],
                    "metadata": item["metadata"]
                })
            
        except Exception as e:
            logger.error(f"Error executing SEARCH: {e}")
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    def _handle_decompose(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Handle DECOMPOSE operation."""
        result = {
            "operation": "decompose",
            "success": True
        }
        
        try:
            tensor_name = operation["tensor_name"]
            method = operation["method"]
            parameters = operation["parameters"]
            
            # Get the tensor
            try:
                tensor_data, metadata = self.database.get(tensor_name)
            except KeyError:
                raise TQLRuntimeError(f"Tensor not found: {tensor_name}")
            
            # Check if the decomposition method is supported
            if method not in self.decomposition_methods:
                raise TQLRuntimeError(f"Unsupported decomposition method: {method}")
            
            # Apply the decomposition
            processor_method = self.decomposition_methods[method]
            decomposition = self.database.process(processor_method, [tensor_data], **parameters)
            
            # Save the results
            result_metadata = {
                "original_tensor": tensor_name,
                "decomposition_method": method,
                "parameters": parameters
            }
            
            if method == "cp":
                # Save factors and weights
                weights_id = self.database.save(
                    decomposition["weights"],
                    {**result_metadata, "component": "weights"}
                )
                
                factor_ids = []
                for i, factor in enumerate(decomposition["factors"]):
                    factor_id = self.database.save(
                        factor,
                        {**result_metadata, "component": f"factor_{i}"}
                    )
                    factor_ids.append(factor_id)
                
                result["weights_id"] = weights_id
                result["factor_ids"] = factor_ids
                
            elif method == "tucker":
                # Save core tensor and factors
                core_id = self.database.save(
                    decomposition["core"],
                    {**result_metadata, "component": "core"}
                )
                
                factor_ids = []
                for i, factor in enumerate(decomposition["factors"]):
                    factor_id = self.database.save(
                        factor,
                        {**result_metadata, "component": f"factor_{i}"}
                    )
                    factor_ids.append(factor_id)
                
                result["core_id"] = core_id
                result["factor_ids"] = factor_ids
                
            else:  # SVD
                u_id = self.database.save(
                    decomposition[0],
                    {**result_metadata, "component": "U"}
                )
                
                s_id = self.database.save(
                    decomposition[1],
                    {**result_metadata, "component": "S"}
                )
                
                v_id = self.database.save(
                    decomposition[2],
                    {**result_metadata, "component": "V"}
                )
                
                result["u_id"] = u_id
                result["s_id"] = s_id
                result["v_id"] = v_id
            
        except Exception as e:
            logger.error(f"Error executing DECOMPOSE: {e}")
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    def _evaluate_expression(self, expression: str) -> np.ndarray:
        """
        Evaluate a tensor expression.
        
        Args:
            expression: TQL expression string
            
        Returns:
            Tensor data as a numpy array
        """
        # Check if this is a numpy array constructor
        if expression.startswith("np.") or expression.startswith("numpy."):
            # Use a restricted eval to create numpy arrays
            allowed_names = {
                "np": np,
                "numpy": np,
                "pi": np.pi,
                "e": np.e
            }
            try:
                return eval(expression, {"__builtins__": {}}, allowed_names)
            except Exception as e:
                raise TQLRuntimeError(f"Error evaluating numpy expression: {e}")
        
        # Check for tensor references
        if expression.startswith("tensor(") and expression.endswith(")"):
            tensor_name = expression[7:-1].strip()
            try:
                tensor_data, _ = self.database.get(tensor_name)
                return tensor_data
            except KeyError:
                raise TQLRuntimeError(f"Referenced tensor not found: {tensor_name}")
        
        # Check for function calls
        func_match = re.match(r"(\w+)\((.*)\)", expression)
        if func_match:
            func_name = func_match.group(1).lower()
            args_str = func_match.group(2)
            
            # Parse arguments
            args = []
            current_arg = ""
            paren_level = 0
            
            for char in args_str:
                if char == "," and paren_level == 0:
                    args.append(current_arg.strip())
                    current_arg = ""
                else:
                    if char == "(":
                        paren_level += 1
                    elif char == ")":
                        paren_level -= 1
                    current_arg += char
            
            if current_arg:
                args.append(current_arg.strip())
            
            # Evaluate arguments
            evaluated_args = [self._evaluate_expression(arg) for arg in args]
            
            # Apply the function
            if func_name in self.tensor_operations:
                return self.tensor_operations[func_name](evaluated_args[0], {})
            else:
                raise TQLRuntimeError(f"Unsupported function: {func_name}")
        
        # Handle lists
        if expression.startswith("[") and expression.endswith("]"):
            try:
                # Parse as a Python list
                return np.array(ast.literal_eval(expression))
            except Exception as e:
                raise TQLRuntimeError(f"Error parsing list: {e}")
        
        # Fall back to trying to parse as a literal
        try:
            return np.array(ast.literal_eval(expression))
        except Exception:
            raise TQLRuntimeError(f"Cannot evaluate expression: {expression}")
    
    def _evaluate_conditions(self, conditions: Dict[str, Any], tensor_info: Dict[str, Any]) -> bool:
        """
        Evaluate conditions on a tensor.
        
        Args:
            conditions: Condition dictionary
            tensor_info: Dictionary with tensor data and metadata
            
        Returns:
            True if conditions are satisfied, False otherwise
        """
        condition_type = conditions.get("type")
        
        if condition_type == "AND":
            return all(self._evaluate_conditions(c, tensor_info) for c in conditions["conditions"])
        elif condition_type == "OR":
            return any(self._evaluate_conditions(c, tensor_info) for c in conditions["conditions"])
        elif condition_type == "comparison":
            left = conditions["left"]
            op = conditions["operator"]
            right = conditions["right"]
            
            # Get left value
            if left == "data":
                left_value = tensor_info["data"]
            elif left in tensor_info["metadata"]:
                left_value = tensor_info["metadata"][left]
            else:
                try:
                    left_value = ast.literal_eval(left)
                except:
                    left_value = left
            
            # Get right value
            try:
                right_value = ast.literal_eval(right)
            except:
                if right == "data":
                    right_value = tensor_info["data"]
                elif right in tensor_info["metadata"]:
                    right_value = tensor_info["metadata"][right]
                else:
                    right_value = right
            
            # Apply operator
            if op == "==":
                return left_value == right_value
            elif op == "!=":
                return left_value != right_value
            elif op == ">":
                return left_value > right_value
            elif op == "<":
                return left_value < right_value
            elif op == ">=":
                return left_value >= right_value
            elif op == "<=":
                return left_value <= right_value
            elif op == "LIKE":
                # Simple string pattern matching
                return isinstance(left_value, str) and isinstance(right_value, str) and right_value in left_value
            elif op == "IN":
                # Check if left is in right
                if isinstance(right_value, list):
                    return left_value in right_value
                elif isinstance(right_value, np.ndarray):
                    return left_value in right_value
                return False
            else:
                raise TQLRuntimeError(f"Unsupported operator: {op}")
        
        return False
    
    # Tensor operations
    def _op_reshape(self, tensor: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Reshape a tensor."""
        new_shape = params.get("shape", None)
        if new_shape is None:
            raise TQLRuntimeError("Reshape operation requires 'shape' parameter")
        return self.database.process("reshape", [tensor], new_shape=new_shape)
    
    def _op_transpose(self, tensor: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Transpose a tensor."""
        dims = params.get("dims", None)
        return self.database.process("transpose", [tensor], dims=dims)
    
    def _op_add(self, tensor: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Add two tensors."""
        other = params.get("other", None)
        if other is None:
            raise TQLRuntimeError("Add operation requires 'other' parameter")
        if isinstance(other, dict) and other.get("type") == "tensor_ref":
            other_tensor, _ = self.database.get(other["name"])
            return self.database.process("add", [tensor, other_tensor])
        return self.database.process("add", [tensor, other])
    
    def _op_subtract(self, tensor: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Subtract a tensor from another."""
        other = params.get("other", None)
        if other is None:
            raise TQLRuntimeError("Subtract operation requires 'other' parameter")
        if isinstance(other, dict) and other.get("type") == "tensor_ref":
            other_tensor, _ = self.database.get(other["name"])
            return self.database.process("subtract", [tensor, other_tensor])
        return self.database.process("subtract", [tensor, other])
    
    def _op_multiply(self, tensor: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Element-wise multiply two tensors."""
        other = params.get("other", None)
        if other is None:
            raise TQLRuntimeError("Multiply operation requires 'other' parameter")
        if isinstance(other, dict) and other.get("type") == "tensor_ref":
            other_tensor, _ = self.database.get(other["name"])
            return self.database.process("multiply", [tensor, other_tensor])
        return self.database.process("multiply", [tensor, other])
    
    def _op_matmul(self, tensor: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Matrix multiply two tensors."""
        other = params.get("other", None)
        if other is None:
            raise TQLRuntimeError("Matmul operation requires 'other' parameter")
        if isinstance(other, dict) and other.get("type") == "tensor_ref":
            other_tensor, _ = self.database.get(other["name"])
            return self.database.process("matmul", [tensor, other_tensor])
        return self.database.process("matmul", [tensor, other])
    
    def _op_slice(self, tensor: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Slice a tensor."""
        start = params.get("start", None)
        end = params.get("end", None)
        
        if start is None or end is None:
            raise TQLRuntimeError("Slice operation requires 'start' and 'end' parameters")
        
        # Create slices
        slices = []
        for s, e in zip(start, end):
            slices.append(slice(s, e))
        
        return tensor[tuple(slices)]
    
    def _op_concat(self, tensor: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Concatenate tensors."""
        other = params.get("other", None)
        axis = params.get("axis", 0)
        
        if other is None:
            raise TQLRuntimeError("Concat operation requires 'other' parameter")
        
        if isinstance(other, dict) and other.get("type") == "tensor_ref":
            other_tensor, _ = self.database.get(other["name"])
            return np.concatenate([tensor, other_tensor], axis=axis)
        
        return np.concatenate([tensor, other], axis=axis)
    
    def _op_stack(self, tensor: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Stack tensors."""
        other = params.get("other", None)
        axis = params.get("axis", 0)
        
        if other is None:
            raise TQLRuntimeError("Stack operation requires 'other' parameter")
        
        if isinstance(other, dict) and other.get("type") == "tensor_ref":
            other_tensor, _ = self.database.get(other["name"])
            return np.stack([tensor, other_tensor], axis=axis)
        
        return np.stack([tensor, other], axis=axis)
    
    def _op_norm(self, tensor: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute the norm of a tensor."""
        order = params.get("order", 2)
        return np.linalg.norm(tensor, ord=order)
    
    def _op_mean(self, tensor: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute the mean of a tensor."""
        axis = params.get("axis", None)
        return np.mean(tensor, axis=axis)
    
    def _op_max(self, tensor: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute the maximum of a tensor."""
        axis = params.get("axis", None)
        return np.max(tensor, axis=axis)
    
    def _op_min(self, tensor: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute the minimum of a tensor."""
        axis = params.get("axis", None)
        return np.min(tensor, axis=axis)
    
    def _op_sum(self, tensor: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute the sum of a tensor."""
        axis = params.get("axis", None)
        return np.sum(tensor, axis=axis) 