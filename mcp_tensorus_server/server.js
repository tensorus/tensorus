const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const { ListToolsRequestSchema, CallToolRequestSchema } = require('@modelcontextprotocol/sdk/types.js');
const axios = require('axios');

// Allow the Python API base URL to be overridden via an environment variable
// while falling back to the default localhost value used throughout the docs.
const PYTHON_API_BASE_URL = process.env.PYTHON_API_BASE_URL || 'http://127.0.0.1:8000';

const UNARY_OPS = ["log", "reshape", "transpose", "permute", "sum", "mean", "min", "max"];
const BINARY_OPS = ["add", "subtract", "multiply", "divide", "power", "matmul", "dot"];
const LIST_OPS = ["concatenate", "stack"];

const toolDefinitions = [
  // --- Storage Management Tools ---
  {
    name: 'tensorus_list_datasets',
    description: 'Lists all available datasets in the Tensorus system.',
    inputSchema: { type: 'object', properties: {}, required: [] },
  },
  {
    name: 'tensorus_create_dataset',
    description: 'Creates a new dataset in the Tensorus system.',
    inputSchema: {
      type: 'object',
      properties: {
        dataset_name: { type: 'string', description: 'The name for the new dataset.' },
      },
      required: ['dataset_name'],
    },
  },
  {
    name: 'tensorus_delete_dataset',
    description: 'Deletes an entire dataset and all its tensor records.',
    inputSchema: {
      type: 'object',
      properties: {
        dataset_name: { type: 'string', description: 'The name of the dataset to delete.' },
      },
      required: ['dataset_name'],
    },
  },
  {
    name: 'tensorus_get_tensor_details',
    description: 'Retrieves details (shape, dtype, data, metadata) of a specific tensor.',
    inputSchema: {
      type: 'object',
      properties: {
        dataset_name: { type: 'string', description: 'The name of the dataset.' },
        record_id: { type: 'string', description: 'The record ID of the tensor.' },
      },
      required: ['dataset_name', 'record_id'],
    },
  },
  {
    name: 'tensorus_ingest_tensor',
    description: 'Ingests a new tensor into a specified dataset.',
    inputSchema: {
      type: 'object',
      properties: {
        dataset_name: { type: 'string', description: 'Target dataset name.' },
        tensor_shape: { type: 'array', items: { type: 'integer' }, description: 'Shape of the tensor (e.g., [2, 3]).' },
        tensor_dtype: { type: 'string', description: "Data type (e.g., 'float32', 'int64')." },
        tensor_data: { type: ['array', 'number', 'integer'], description: 'Tensor data (nested list or scalar).' },
        metadata: { type: 'object', description: 'Optional key-value metadata.', properties: {}, additionalProperties: true },
      },
      required: ['dataset_name', 'tensor_shape', 'tensor_dtype', 'tensor_data'],
    },
  },
  {
    name: 'tensorus_delete_tensor',
    description: 'Deletes a specific tensor record from a dataset.',
    inputSchema: {
      type: 'object',
      properties: {
        dataset_name: { type: 'string', description: 'The name of the dataset.' },
        record_id: { type: 'string', description: 'The record ID of the tensor to delete.' },
      },
      required: ['dataset_name', 'record_id'],
    },
  },
  {
    name: 'tensorus_update_tensor_metadata',
    description: 'Updates the metadata for a specific tensor record.',
    inputSchema: {
      type: 'object',
      properties: {
        dataset_name: { type: 'string', description: 'The name of the dataset.' },
        record_id: { type: 'string', description: 'The record ID of the tensor to update.' },
        new_metadata: { type: 'object', description: 'New metadata to replace the existing metadata.', properties: {}, additionalProperties: true },
      },
      required: ['dataset_name', 'record_id', 'new_metadata'],
    },
  },
  // --- Tensor Operations ---
  {
    name: 'tensorus_apply_unary_operation',
    description: `Applies a unary tensor operation. Supported operations: ${UNARY_OPS.join(', ')}.`,
    inputSchema: {
      type: 'object',
      properties: {
        operation: { type: 'string', enum: UNARY_OPS, description: "The unary operation to perform." },
        input_dataset_name: { type: 'string', description: 'Dataset name of the input tensor.' },
        input_record_id: { type: 'string', description: 'Record ID of the input tensor.' },
        params: {
          type: 'object',
          description: 'Parameters for the operation (e.g., for reshape: {"new_shape": [1,2,3]}; for sum/mean/min/max: {"dim": 0, "keepdim": false}).',
          properties: { // Define specific optional params for validation/documentation
            new_shape: { type: 'array', items: { type: 'integer' } },
            dims: { type: 'array', items: { type: 'integer' } },
            dim0: { type: 'integer' },
            dim1: { type: 'integer' },
            dim: { type: ['integer', 'array'], items: { type: 'integer' } },
            keepdim: { type: 'boolean' }
          },
          additionalProperties: true // Allow other params if the user knows what they're doing
        },
        output_dataset_name: { type: 'string', description: 'Optional: Name for the output dataset. Defaults to "tensor_ops_results".' },
        output_metadata: { type: 'object', description: 'Optional: Metadata for the output tensor.', properties: {}, additionalProperties: true },
      },
      required: ['operation', 'input_dataset_name', 'input_record_id'],
    },
  },
  {
    name: 'tensorus_apply_binary_operation',
    description: `Applies a binary tensor operation. Supported operations: ${BINARY_OPS.join(', ')}.`,
    inputSchema: {
      type: 'object',
      properties: {
        operation: { type: 'string', enum: BINARY_OPS, description: "The binary operation to perform." },
        input1_dataset_name: { type: 'string', description: 'Dataset name of the first input tensor (or base_tensor for power).' },
        input1_record_id: { type: 'string', description: 'Record ID of the first input tensor (or base_tensor for power).' },
        input2_type: { type: 'string', enum: ['tensor', 'scalar'], description: "Type of the second input ('tensor' or 'scalar')." },
        input2_dataset_name: { type: 'string', description: 'Dataset name of the second input tensor (if type is tensor).' },
        input2_record_id: { type: 'string', description: 'Record ID of the second input tensor (if type is tensor).' },
        input2_scalar_value: { type: ['number', 'integer'], description: 'Scalar value for the second input (if type is scalar).' },
        output_dataset_name: { type: 'string', description: 'Optional: Name for the output dataset. Defaults to "tensor_ops_results".' },
        output_metadata: { type: 'object', description: 'Optional: Metadata for the output tensor.', properties: {}, additionalProperties: true },
      },
      required: ['operation', 'input1_dataset_name', 'input1_record_id', 'input2_type'],
      allOf: [
        {
          if: { properties: { input2_type: { const: 'tensor' } } },
          then: { required: ['input2_dataset_name', 'input2_record_id'] }
        },
        {
          if: { properties: { input2_type: { const: 'scalar' } } },
          then: { required: ['input2_scalar_value'] }
        }
      ]
    },
  },
  {
    name: 'tensorus_apply_list_operation',
    description: `Applies an operation to a list of tensors. Supported operations: ${LIST_OPS.join(', ')}.`,
    inputSchema: {
      type: 'object',
      properties: {
        operation: { type: 'string', enum: LIST_OPS, description: 'The list operation to perform (concatenate or stack).' },
        input_tensors: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              dataset_name: { type: 'string' },
              record_id: { type: 'string' },
            },
            required: ['dataset_name', 'record_id'],
          },
          minItems: 1,
          description: 'List of input tensor references.'
        },
        params: {
          type: 'object',
          properties: {
            dim: { type: 'integer', description: 'Dimension for concatenation/stacking.' }
          },
          required: ['dim'],
          description: 'Parameters for the list operation (e.g., {"dim": 0}).'
        },
        output_dataset_name: { type: 'string', description: 'Optional: Name for the output dataset.' },
        output_metadata: { type: 'object', description: 'Optional: Metadata for the output tensor.', properties: {}, additionalProperties: true },
      },
      required: ['operation', 'input_tensors', 'params'],
    }
  },
  {
    name: 'tensorus_apply_einsum',
    description: 'Applies Einstein summation to the input tensors.',
    inputSchema: {
      type: 'object',
      properties: {
        equation: { type: 'string', description: "Einstein summation equation (e.g., 'ij,jk->ik')." },
        input_tensors: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              dataset_name: { type: 'string' },
              record_id: { type: 'string' },
            },
            required: ['dataset_name', 'record_id'],
          },
          minItems: 1,
          description: 'List of input tensor references for Einsum.'
        },
        output_dataset_name: { type: 'string', description: 'Optional: Name for the output dataset.' },
        output_metadata: { type: 'object', description: 'Optional: Metadata for the output tensor.', properties: {}, additionalProperties: true },
      },
      required: ['equation', 'input_tensors'],
    }
  }
];

const mcpServerInstance = new Server(
  { name: 'tensorus-mcp-server', version: '0.1.1' },
  { capabilities: { tools: {} } }
);

mcpServerInstance.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools: toolDefinitions };
});

// Extracted function for easier testing
async function handleToolCall(name, args) {
  let response;
  let requestBody;
  let apiUrl;

  console.log(`MCP Server: Processing tools/call for ${name} with args:`, JSON.stringify(args, null, 2));

  try {
    switch (name) {
      case 'tensorus_list_datasets':
        response = await axios.get(`${PYTHON_API_BASE_URL}/datasets`);
        return { content: [{ type: 'text', text: JSON.stringify(response.data.data || []) }] };

      case 'tensorus_create_dataset':
        if (!args.dataset_name) throw new Error('Missing required argument: dataset_name');
        response = await axios.post(`${PYTHON_API_BASE_URL}/datasets/create`, { name: args.dataset_name });
        return { content: [{ type: 'text', text: response.data.message }] };
      
      case 'tensorus_delete_dataset':
        if (!args.dataset_name) throw new Error('Missing required argument: dataset_name');
        response = await axios.delete(`${PYTHON_API_BASE_URL}/datasets/${args.dataset_name}`);
        return { content: [{ type: 'text', text: response.data.message }] };

      case 'tensorus_get_tensor_details':
        if (!args.dataset_name || !args.record_id) throw new Error('Missing required arguments: dataset_name or record_id');
        response = await axios.get(`${PYTHON_API_BASE_URL}/datasets/${args.dataset_name}/tensors/${args.record_id}`);
        return { content: [{ type: 'text', text: JSON.stringify(response.data) }] };

      case 'tensorus_ingest_tensor':
        if (!args.dataset_name || !args.tensor_shape || !args.tensor_dtype || args.tensor_data === undefined) {
            throw new Error('Missing required arguments for ingest_tensor.');
        }
        requestBody = {
            shape: args.tensor_shape,
            dtype: args.tensor_dtype,
            data: args.tensor_data,
            metadata: args.metadata || null
        };
        response = await axios.post(`${PYTHON_API_BASE_URL}/datasets/${args.dataset_name}/ingest`, requestBody);
        return { content: [{ type: 'text', text: JSON.stringify(response.data) }] };

      case 'tensorus_delete_tensor':
        if (!args.dataset_name || !args.record_id) throw new Error('Missing required arguments: dataset_name or record_id');
        response = await axios.delete(`${PYTHON_API_BASE_URL}/datasets/${args.dataset_name}/tensors/${args.record_id}`);
        return { content: [{ type: 'text', text: response.data.message }] };
        
      case 'tensorus_update_tensor_metadata':
        if (!args.dataset_name || !args.record_id || !args.new_metadata) throw new Error('Missing required arguments for update_tensor_metadata.');
        response = await axios.put(`${PYTHON_API_BASE_URL}/datasets/${args.dataset_name}/tensors/${args.record_id}/metadata`, { new_metadata: args.new_metadata });
        return { content: [{ type: 'text', text: response.data.message }] };

      case 'tensorus_apply_unary_operation':
        if (!UNARY_OPS.includes(args.operation)) {
          throw new Error(`Unsupported unary operation: ${args.operation}. Supported: ${UNARY_OPS.join(', ')}`);
        }
        if (!args.input_dataset_name || !args.input_record_id) {
            throw new Error('Missing required arguments for unary operation: input_dataset_name or input_record_id');
        }
        
        requestBody = {
          input_tensor: { dataset_name: args.input_dataset_name, record_id: args.input_record_id },
          output_dataset_name: args.output_dataset_name || null,
          output_metadata: args.output_metadata || null,
        };
        
        if (args.params) {
            if (args.operation === 'reshape' && args.params.new_shape) {
                requestBody.params = { new_shape: args.params.new_shape };
            } else if (args.operation === 'permute' && args.params.dims) {
                requestBody.params = { dims: args.params.dims };
            } else if (args.operation === 'transpose' && args.params.dim0 !== undefined && args.params.dim1 !== undefined) {
                requestBody.params = { dim0: args.params.dim0, dim1: args.params.dim1 };
            } else if (['sum', 'mean', 'min', 'max'].includes(args.operation)) {
                if (args.params.dim !== undefined || args.params.keepdim !== undefined) {
                     requestBody.params = {
                         dim: args.params.dim !== undefined ? args.params.dim : null,
                         keepdim: args.params.keepdim !== undefined ? args.params.keepdim : false,
                     };
                } else if (args.operation === 'sum' || args.operation === 'mean') {
                     requestBody.params = { dim: null, keepdim: false };
                }
            }
        } else if (args.operation === 'sum' || args.operation === 'mean') {
            requestBody.params = { dim: null, keepdim: false };
        }

        apiUrl = `${PYTHON_API_BASE_URL}/ops/${args.operation}`;
        console.log(`MCP Server: Calling ${apiUrl} with body:`, JSON.stringify(requestBody, null, 2));
        response = await axios.post(apiUrl, requestBody);
        return { content: [{ type: 'text', text: JSON.stringify(response.data) }] };

      case 'tensorus_apply_binary_operation':
        if (!BINARY_OPS.includes(args.operation)) {
          throw new Error(`Unsupported binary operation: ${args.operation}. Supported: ${BINARY_OPS.join(', ')}`);
        }
        if (!args.input1_dataset_name || !args.input1_record_id || !args.input2_type) {
            throw new Error('Missing required arguments: input1_dataset_name, input1_record_id, or input2_type');
        }

        let input2_val_bin;
        if (args.input2_type === 'tensor') {
          if (!args.input2_dataset_name || !args.input2_record_id) {
            throw new Error('Missing required arguments for tensor input2: input2_dataset_name or input2_record_id');
          }
          input2_val_bin = { tensor_ref: { dataset_name: args.input2_dataset_name, record_id: args.input2_record_id } };
        } else if (args.input2_type === 'scalar') {
          if (args.input2_scalar_value === undefined || args.input2_scalar_value === null) {
            throw new Error('Missing required argument for scalar input2: input2_scalar_value');
          }
          input2_val_bin = { scalar_value: args.input2_scalar_value };
        } else {
            throw new Error(`Invalid input2_type: ${args.input2_type}. Must be 'tensor' or 'scalar'.`);
        }
        
        requestBody = {
            output_dataset_name: args.output_dataset_name || null,
            output_metadata: args.output_metadata || null,
        };

        if (args.operation === 'power') {
            requestBody.base_tensor = { dataset_name: args.input1_dataset_name, record_id: args.input1_record_id };
            requestBody.exponent = input2_val_bin;
        } else {
            requestBody.input1 = { dataset_name: args.input1_dataset_name, record_id: args.input1_record_id };
            requestBody.input2 = input2_val_bin;
        }
        
        if ((args.operation === 'matmul' || args.operation === 'dot') && args.input2_type !== 'tensor') {
            throw new Error(`Operation ${args.operation} requires input2 to be a tensor.`);
        }

        apiUrl = `${PYTHON_API_BASE_URL}/ops/${args.operation}`;
        console.log(`MCP Server: Calling ${apiUrl} with body:`, JSON.stringify(requestBody, null, 2));
        response = await axios.post(apiUrl, requestBody);
        return { content: [{ type: 'text', text: JSON.stringify(response.data) }] };

      case 'tensorus_apply_list_operation':
        if (!LIST_OPS.includes(args.operation)) {
            throw new Error(`Unsupported list operation: ${args.operation}. Supported: ${LIST_OPS.join(', ')}`);
        }
        if (!args.input_tensors || !Array.isArray(args.input_tensors) || args.input_tensors.length === 0) {
            throw new Error('input_tensors must be a non-empty array.');
        }
        if (!args.params || args.params.dim === undefined) {
            throw new Error('Missing required "params.dim" for list operation.');
        }
        requestBody = {
            input_tensors: args.input_tensors,
            params: { dim: args.params.dim },
            output_dataset_name: args.output_dataset_name || null,
            output_metadata: args.output_metadata || null,
        };
        apiUrl = `${PYTHON_API_BASE_URL}/ops/${args.operation}`;
        console.log(`MCP Server: Calling ${apiUrl} with body:`, JSON.stringify(requestBody, null, 2));
        response = await axios.post(apiUrl, requestBody);
        return { content: [{ type: 'text', text: JSON.stringify(response.data) }] };
        
      case 'tensorus_apply_einsum':
        if (!args.equation || !args.input_tensors || !Array.isArray(args.input_tensors) || args.input_tensors.length === 0) {
            throw new Error('Missing required arguments for einsum: equation or input_tensors (must be non-empty array).');
        }
        requestBody = {
            input_tensors: args.input_tensors,
            params: { equation: args.equation },
            output_dataset_name: args.output_dataset_name || null,
            output_metadata: args.output_metadata || null,
        };
        apiUrl = `${PYTHON_API_BASE_URL}/ops/einsum`;
        console.log(`MCP Server: Calling ${apiUrl} with body:`, JSON.stringify(requestBody, null, 2));
        response = await axios.post(apiUrl, requestBody);
        return { content: [{ type: 'text', text: JSON.stringify(response.data) }] };

      default:
        console.error('MCP Server: Unknown tool called - ' + name);
        throw new Error('Unknown tool: ' + name);
    }
  } catch (error) {
    let detailedErrorMessage = error.message;
    if (error.isAxiosError || (typeof axios.isAxiosError === 'function' && axios.isAxiosError(error))) {
      console.error(`MCP Server: Axios error calling tool ${name}:`, error.message);
      if (error.response) {
        console.error('MCP Server: Python API Response Error Data:', JSON.stringify(error.response.data, null, 2));
        console.error('MCP Server: Python API Response Status:', error.response.status);
        const apiErrorDetail = error.response.data?.detail || JSON.stringify(error.response.data);
        detailedErrorMessage = `Python API Error (${error.response.status}): ${apiErrorDetail}`;
      } else if (error.request) {
        console.error('MCP Server: Python API No Response (Network Error):', error.request);
        detailedErrorMessage = 'Python API did not respond. Network error or API is down.';
      }
    } else {
      console.error(`MCP Server: Non-Axios error calling tool ${name}:`, error);
    }
    const mcpError = new Error(detailedErrorMessage);
    mcpError.name = error.name || 'ToolExecutionError';
    throw mcpError;
  }
}

mcpServerInstance.setRequestHandler(CallToolRequestSchema, async (request) => {
    return handleToolCall(request.params.name, request.params.arguments);
});

// Only start server if not in test environment
if (process.env.NODE_ENV !== 'test') {
    const transport = new StdioServerTransport();
    mcpServerInstance.connect(transport).then(() => {
        console.log('Tensorus MCP Server connected via stdio and ready.');
    }).catch(err => {
        console.error('Tensorus MCP Server: Error during connection or startup:', err);
        process.exit(1);
    });

    process.on('uncaughtException', (err, origin) => {
      console.error(`Tensorus MCP Server: Uncaught Exception at: ${origin}, error:`, err);
    });
    process.on('unhandledRejection', (reason, promise) => {
      console.error('Tensorus MCP Server: Unhandled Rejection at:', promise, 'reason:', reason);
    });
    console.log('Tensorus MCP Server: Script execution started.');
}

module.exports = {
    server: mcpServerInstance, // Export the server instance
    toolDefinitions,           // Export tool definitions for testing
    handleToolCall,            // Export the handler logic for direct testing
    PYTHON_API_BASE_URL,       // Export for test configuration if needed
    UNARY_OPS,
    BINARY_OPS,
    LIST_OPS
};
