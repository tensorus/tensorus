# nql_agent.py
"""
Implements the NQL (Natural Query Language) Query Agent for Tensorus.

This agent provides a basic natural language interface to query datasets
stored in TensorStorage. It uses regular expressions to parse simple queries
and translates them into calls to TensorStorage.query or other methods.

Limitations (without LLM):
- Understands only a very limited set of predefined sentence structures.
- Limited support for complex conditions (AND/OR not implemented).
- Limited support for data types in conditions (primarily numbers and exact strings).
- No support for aggregations (mean, sum, etc.) beyond simple counts.
- Error handling for parsing ambiguity is basic.

Future Enhancements:
- Integrate a local or remote LLM for robust NLU.
- Support for complex queries (multiple conditions, joins).
- Support for aggregations and projections (selecting specific fields).
- More sophisticated error handling and user feedback.
- Context awareness and conversation history.
"""

import re
import logging
import torch
from typing import List, Dict, Any, Optional, Callable, Tuple

from .tensor_storage import TensorStorage # Import our storage module

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NQLAgent:
    """Parses simple natural language queries and executes them against TensorStorage."""

    def __init__(self, tensor_storage: TensorStorage):
        """
        Initializes the NQL Agent.

        Args:
            tensor_storage: An instance of the TensorStorage class.
        """
        if not isinstance(tensor_storage, TensorStorage):
            raise TypeError("tensor_storage must be an instance of TensorStorage")
        self.tensor_storage = tensor_storage

        # --- Compile Regex Patterns for Query Parsing ---
        # Pattern to match variations of "get/show/find [all] X from dataset Y"
        self.pattern_get_all = re.compile(
            r"^(?:get|show|find)\s+(?:all\s+)?(?:data|tensors?|records?|entries|experiences?)\s+from\s+(?:dataset\s+)?([\w_.-]+)$",
            re.IGNORECASE
        )

        # Pattern for basic metadata filtering: "... from Y where meta_key op value"
        # Captures: dataset_name, key, operator, value
        # Value can be quoted or unquoted number/simple string
        self.pattern_filter_meta = re.compile(
            r"^(?:get|show|find)\s+.*\s+from\s+([\w_.-]+)\s+where\s+([\w_.-]+)\s*([<>=!]+)\s*'?([\w\s\d_.-]+?)'?$",
             re.IGNORECASE
        )
        # Simpler pattern allowing 'is'/'equals'/'eq' for '='
        self.pattern_filter_meta_alt = re.compile(
            r"^(?:get|show|find)\s+.*\s+from\s+([\w_.-]+)\s+where\s+([\w_.-]+)\s+(?:is|equals|eq)\s+'?([\w\s\d_.-]+?)'?$",
             re.IGNORECASE
        )


        # Pattern for filtering based on tensor value at a specific index: "... from Y where tensor_value[index] op value"
        # Captures: dataset_name, index, operator, value
        self.pattern_filter_tensor = re.compile(
            r"^(?:get|show|find)\s+.*\s+from\s+([\w_.-]+)\s+where\s+(?:tensor|value)\s*(?:\[(\d+)\])?\s*([<>=!]+)\s*([\d.-]+)$",
            re.IGNORECASE
        )
        # Note: tensor[index] assumes a 1D tensor or accessing the element at flat index `index`.
        # More complex tensor indexing (e.g., tensor[0, 1]) is not supported by this simple regex.

        # Pattern for counting records: "count [records/...] in Y"
        self.pattern_count = re.compile(
            r"^count\s+(?:records?|entries|experiences?)\s+(?:in|from)\s+(?:dataset\s+)?([\w_.-]+)$",
             re.IGNORECASE
        )

        logger.info("NQLAgent initialized with basic regex patterns.")


    def _parse_operator_and_value(self, op_str: str, val_str: str) -> Tuple[Callable, Any]:
        """Attempts to parse operator string and convert value string to number if possible."""
        val_str = val_str.strip()
        op_map = {
            '=': lambda a, b: a == b,
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            '<': lambda a, b: a < b,
            '<=': lambda a, b: a <= b,
            '>': lambda a, b: a > b,
            '>=': lambda a, b: a >= b,
        }

        op_func = op_map.get(op_str)
        if op_func is None:
            raise ValueError(f"Unsupported operator: {op_str}")

        # Try converting value to float or int
        try:
            value = float(val_str)
            if value.is_integer():
                value = int(value)
        except ValueError:
            # Keep as string if conversion fails
            value = val_str

        return op_func, value


    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Processes a natural language query string.

        Args:
            query: The natural language query.

        Returns:
            A dictionary containing:
                'success': bool, indicating if the query was processed.
                'message': str, status message or error description.
                'count': Optional[int], number of results found.
                'results': Optional[List[Dict]], the list of matching records
                           (each a dict with 'tensor' and 'metadata').
        """
        query = query.strip()
        logger.info(f"Processing NQL query: '{query}'")

        # Try matching patterns in order of specificity/complexity

        # --- 1. Count Pattern ---
        match = self.pattern_count.match(query)
        if match:
            dataset_name = match.group(1)
            logger.debug(f"Matched COUNT pattern for dataset '{dataset_name}'")
            try:
                # Inefficient: gets all data just to count.
                # TODO: Add count method to TensorStorage
                results = self.tensor_storage.get_dataset_with_metadata(dataset_name)
                count = len(results)
                return {
                    "success": True,
                    "message": f"Found {count} records in dataset '{dataset_name}'.",
                    "count": count,
                    "results": None # Or optionally return all results if needed?
                }
            except ValueError as e:
                logger.error(f"Error during COUNT query: {e}")
                return {"success": False, "message": str(e), "count": None, "results": None}
            except Exception as e:
                 logger.error(f"Unexpected error during COUNT query: {e}", exc_info=True)
                 return {"success": False, "message": f"An unexpected error occurred: {e}", "count": None, "results": None}


        # --- 2. Get All Pattern ---
        match = self.pattern_get_all.match(query)
        if match:
            dataset_name = match.group(1)
            logger.debug(f"Matched GET ALL pattern for dataset '{dataset_name}'")
            try:
                results = self.tensor_storage.get_dataset_with_metadata(dataset_name)
                count = len(results)
                return {
                    "success": True,
                    "message": f"Retrieved {count} records from dataset '{dataset_name}'.",
                    "count": count,
                    "results": results
                }
            except ValueError as e:
                logger.error(f"Error during GET ALL query: {e}")
                return {"success": False, "message": str(e), "count": None, "results": None}
            except Exception as e:
                 logger.error(f"Unexpected error during GET ALL query: {e}", exc_info=True)
                 return {"success": False, "message": f"An unexpected error occurred: {e}", "count": None, "results": None}


        # --- 3. Filter Metadata Pattern ---
        match_meta = self.pattern_filter_meta.match(query)
        if not match_meta: # Try alternative pattern if first fails
             match_meta = self.pattern_filter_meta_alt.match(query)
             if match_meta:
                  # Extract groups and manually set operator to '='
                  dataset_name = match_meta.group(1)
                  key = match_meta.group(2)
                  op_str = '=' # Implicitly '=' for 'is/equals/eq'
                  val_str = match_meta.group(3)
                  logger.debug(f"Matched FILTER META ALT pattern: dataset='{dataset_name}', key='{key}', op='{op_str}', value='{val_str}'")
             else:
                   match_meta = None # Reset if alt pattern didn't match either
        else:
            # Standard extraction
            dataset_name = match_meta.group(1)
            key = match_meta.group(2)
            op_str = match_meta.group(3)
            val_str = match_meta.group(4)
            logger.debug(f"Matched FILTER META pattern: dataset='{dataset_name}', key='{key}', op='{op_str}', value='{val_str}'")


        if match_meta:
             try:
                 op_func, filter_value = self._parse_operator_and_value(op_str, val_str)

                 # Construct the query function dynamically
                 def query_fn_meta(tensor: torch.Tensor, metadata: Dict[str, Any]) -> bool:
                     actual_value = metadata.get(key)
                     if actual_value is None:
                         return False # Key doesn't exist in this record's metadata

                     # Attempt type coercion if filter value is numeric but actual is not
                     # (Basic attempt, might need more robust type handling)
                     try:
                          if isinstance(filter_value, (int, float)) and not isinstance(actual_value, (int, float)):
                              actual_value = type(filter_value)(actual_value) # Try to cast actual to filter type
                          elif isinstance(filter_value, str) and not isinstance(actual_value, str):
                              actual_value = str(actual_value) # Cast actual to string if filter is string
                     except (ValueError, TypeError):
                          return False # Cannot compare types

                     try:
                         return op_func(actual_value, filter_value)
                     except TypeError:
                          # Mismatched types that couldn't be coerced
                          return False
                     except Exception as e_inner:
                         logger.warning(f"Error during query_fn execution for key '{key}': {e_inner}")
                         return False

                 results = self.tensor_storage.query(dataset_name, query_fn_meta)
                 count = len(results)
                 return {
                     "success": True,
                     "message": f"Query executed successfully. Found {count} matching records.",
                     "count": count,
                     "results": results
                 }

             except ValueError as e: # Catches parse_operator_and_value errors or dataset not found
                 logger.error(f"Error processing FILTER META query: {e}")
                 return {"success": False, "message": str(e), "count": None, "results": None}
             except Exception as e:
                 logger.error(f"Unexpected error during FILTER META query: {e}", exc_info=True)
                 return {"success": False, "message": f"An unexpected error occurred: {e}", "count": None, "results": None}


        # --- 4. Filter Tensor Pattern ---
        match = self.pattern_filter_tensor.match(query)
        if match:
            dataset_name = match.group(1)
            index_str = match.group(2) # Might be None if accessing whole tensor value (ambiguous)
            op_str = match.group(3)
            val_str = match.group(4)
            logger.debug(f"Matched FILTER TENSOR pattern: dataset='{dataset_name}', index='{index_str}', op='{op_str}', value='{val_str}'")

            try:
                op_func, filter_value = self._parse_operator_and_value(op_str, val_str)
                if not isinstance(filter_value, (int, float)):
                    raise ValueError(f"Tensor value filtering currently only supports numeric comparisons. Got value: {filter_value}")

                tensor_index: Optional[int] = None
                if index_str is not None:
                    tensor_index = int(index_str)

                # Construct the query function dynamically
                def query_fn_tensor(tensor: torch.Tensor, metadata: Dict[str, Any]) -> bool:
                    try:
                        if tensor_index is None:
                           # Ambiguous case: compare filter value against the whole tensor?
                           # Requires defining how comparison works (e.g., any element, all elements?)
                           # For now, let's assume it compares against the first element if tensor is 1D+ and index omitted
                           if tensor.numel() > 0:
                                actual_value = tensor.view(-1)[0].item() # Get first element's value
                           else:
                                return False # Empty tensor cannot satisfy condition
                        else:
                             # Access element at specified index (flattened tensor)
                             if tensor_index >= tensor.numel():
                                 return False # Index out of bounds
                             actual_value = tensor.view(-1)[tensor_index].item()

                        # Value comparison (assuming numeric)
                        return op_func(actual_value, filter_value)

                    except IndexError:
                        return False # Index out of bounds
                    except Exception as e_inner:
                        logger.warning(f"Error during query_fn_tensor execution: {e_inner}")
                        return False

                results = self.tensor_storage.query(dataset_name, query_fn_tensor)
                count = len(results)
                return {
                    "success": True,
                    "message": f"Query executed successfully. Found {count} matching records.",
                    "count": count,
                    "results": results
                }

            except ValueError as e: # Catches parse errors, type errors, dataset not found etc.
                logger.error(f"Error processing FILTER TENSOR query: {e}")
                return {"success": False, "message": str(e), "count": None, "results": None}
            except Exception as e:
                logger.error(f"Unexpected error during FILTER TENSOR query: {e}", exc_info=True)
                return {"success": False, "message": f"An unexpected error occurred: {e}", "count": None, "results": None}


        # --- No Match Found ---
        logger.warning(f"Query did not match any known patterns: '{query}'")
        return {
            "success": False,
            "message": "Sorry, I couldn't understand that query. Try simple commands like 'get all data from my_dataset' or 'find records from my_dataset where key = value'.",
            "count": None,
            "results": None
        }


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Starting NQL Agent Example ---")

    # 1. Setup TensorStorage and add some data
    storage = TensorStorage()
    storage.create_dataset("sensor_data")
    storage.create_dataset("rl_experiences_test")

    storage.insert("sensor_data", torch.tensor([10.5, 25.2]), metadata={"sensor_id": "A001", "location": "floor1", "status":"active"})
    storage.insert("sensor_data", torch.tensor([12.1, 26.8]), metadata={"sensor_id": "A002", "location": "floor1", "status":"active"})
    storage.insert("sensor_data", torch.tensor([-5.0, 24.1]), metadata={"sensor_id": "B001", "location": "floor2", "status":"inactive"})

    # Add dummy experience data (using the structure from RLAgent)
    storage.insert("rl_experiences_test", torch.tensor([1.0]), metadata={"state_id": "s1", "action": 0, "reward": -1.5, "next_state_id": "s2", "done": 0})
    storage.insert("rl_experiences_test", torch.tensor([1.0]), metadata={"state_id": "s2", "action": 1, "reward": 5.2, "next_state_id": "s3", "done": 0})
    storage.insert("rl_experiences_test", torch.tensor([1.0]), metadata={"state_id": "s3", "action": 0, "reward": -8.0, "next_state_id": None, "done": 1})


    # 2. Create the NQL Agent
    nql_agent = NQLAgent(storage)

    # 3. Define test queries
    queries = [
        "get all data from sensor_data",
        "show all records from rl_experiences_test",
        "count records in sensor_data",
        "find tensors from sensor_data where sensor_id = 'A001'", # Metadata string eq
        "find data from sensor_data where location is 'floor1'", # Metadata string eq (alt syntax)
        "get records from sensor_data where status != 'active'", # Metadata string neq
        "find experiences from rl_experiences_test where reward > 0", # Metadata numeric gt
        "get experiences from rl_experiences_test where reward < -5", # Metadata numeric lt
        "find entries from rl_experiences_test where done == 1", # Metadata numeric eq (bool as int)
        "get records from sensor_data where value[0] > 11", # Tensor value gt at index 0
        "find tensors from sensor_data where tensor[1] < 25", # Tensor value lt at index 1
        "show data from sensor_data where value = -5.0", # Tensor value eq (omitting index -> first element)
        "get everything from non_existent_dataset", # Test non-existent dataset
        "find data from sensor_data where invalid_key = 10", # Test non-existent key
        "give me the average sensor reading", # Test unsupported query
        "select * from sensor_data" # Test SQL-like query (unsupported)
    ]

    # 4. Process queries and print results
    print("\n--- Processing Queries ---")
    for q in queries:
        print(f"\n> Query: \"{q}\"")
        response = nql_agent.process_query(q)
        print(f"< Success: {response['success']}")
        print(f"< Message: {response['message']}")
        if response['success'] and response['results'] is not None:
            print(f"< Count: {response['count']}")
            # Print limited results for brevity
            limit = 3
            for i, item in enumerate(response['results']):
                if i >= limit:
                    print(f"  ... (omitting {len(response['results']) - limit} more results)")
                    break
                # Simplify tensor printing for readability
                tensor_str = f"Tensor(shape={item['tensor'].shape}, dtype={item['tensor'].dtype})"
                print(f"  - Result {i+1}: Metadata={item['metadata']}, Tensor={tensor_str}")
        elif response['success'] and response['count'] is not None:
             print(f"< Count: {response['count']}") # For count queries


    print("\n--- NQL Agent Example Finished ---")