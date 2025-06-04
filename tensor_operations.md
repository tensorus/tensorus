# Tensor Operations

This document outlines the specifications for a set of tensor operations to be implemented.

## Operations

### 1. Tensor Creation
- **`create_tensor(shape, dtype)`**: Creates a new tensor with the given shape and data type.
  - `shape`: A list or tuple representing the dimensions of the tensor.
  - `dtype`: The data type of the tensor elements (e.g., 'float32', 'int64').
- **`zeros_tensor(shape, dtype)`**: Creates a new tensor filled with zeros.
- **`ones_tensor(shape, dtype)`**: Creates a new tensor filled with ones.
- **`random_tensor(shape, dtype, min_val, max_val)`**: Creates a new tensor with random values within the specified range.

### 2. Tensor Manipulations
- **`reshape_tensor(tensor, new_shape)`**: Reshapes the tensor to the new shape.
- **`transpose_tensor(tensor, axes)`**: Transposes the tensor along the specified axes.
- **`slice_tensor(tensor, start_indices, end_indices)`**: Extracts a slice from the tensor.

### 3. Element-wise Operations
- **`add_tensors(tensor1, tensor2)`**: Adds two tensors element-wise.
- **`subtract_tensors(tensor1, tensor2)`**: Subtracts the second tensor from the first, element-wise.
- **`multiply_tensors(tensor1, tensor2)`**: Multiplies two tensors element-wise.
- **`divide_tensors(tensor1, tensor2)`**: Divides the first tensor by the second, element-wise.

### 4. Reduction Operations
- **`sum_tensor(tensor, axis=None)`**: Computes the sum of tensor elements along the specified axis.
- **`mean_tensor(tensor, axis=None)`**: Computes the mean of tensor elements along the specified axis.
- **`max_tensor(tensor, axis=None)`**: Finds the maximum value in the tensor along the specified axis.
- **`min_tensor(tensor, axis=None)`**: Finds the minimum value in the tensor along the specified axis.

### 5. Matrix Operations
- **`matmul_tensors(tensor1, tensor2)`**: Performs matrix multiplication of two tensors.
- **`dot_product(tensor1, tensor2)`**: Computes the dot product of two tensors.

## Error Handling
- All functions should perform appropriate error checking (e.g., shape compatibility, valid data types).
- Raise descriptive exceptions for invalid operations or inputs.

## Documentation
- Each function must include a docstring explaining its purpose, arguments, and return value.
- Provide examples of usage for each function.

## Testing
- Implement unit tests for all tensor operations.
- Test cases should cover various scenarios, including edge cases and invalid inputs.
