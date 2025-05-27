// mcp_tensorus_server/server.test.js
const axios = require('axios');
// Ensure the path is correct and server.js exports these
const { mcpServerInstance, toolDefinitions, handleToolCall, PYTHON_API_BASE_URL, UNARY_OPS, BINARY_OPS, LIST_OPS } = require('./server');

jest.mock('axios');

describe('Tensorus MCP Server', () => {
    describe('tools/list handler (direct test of toolDefinitions)', () => {
        it('should return the correct list of tool definitions', () => {
            expect(toolDefinitions).toBeInstanceOf(Array);
            expect(toolDefinitions.length).toBeGreaterThan(0);

            const listDatasetsTool = toolDefinitions.find(tool => tool.name === 'tensorus_list_datasets');
            expect(listDatasetsTool).toBeDefined();
            expect(listDatasetsTool.description).toBe('Lists all available datasets in the Tensorus system.');
            expect(listDatasetsTool.inputSchema).toEqual({ type: 'object', properties: {}, required: [] });
            
            const createDatasetTool = toolDefinitions.find(tool => tool.name === 'tensorus_create_dataset');
            expect(createDatasetTool).toBeDefined();

            const getTensorTool = toolDefinitions.find(tool => tool.name === 'tensorus_get_tensor_details');
            expect(getTensorTool).toBeDefined();

            const unaryOpTool = toolDefinitions.find(tool => tool.name === 'tensorus_apply_unary_operation');
            expect(unaryOpTool).toBeDefined();
            expect(unaryOpTool.inputSchema.properties.operation.enum).toEqual(UNARY_OPS);
            
            const binaryOpTool = toolDefinitions.find(tool => tool.name === 'tensorus_apply_binary_operation');
            expect(binaryOpTool).toBeDefined();
            expect(binaryOpTool.inputSchema.properties.operation.enum).toEqual(BINARY_OPS);

            const listOpTool = toolDefinitions.find(tool => tool.name === 'tensorus_apply_list_operation');
            expect(listOpTool).toBeDefined();
            expect(listOpTool.inputSchema.properties.operation.enum).toEqual(LIST_OPS);

            const einsumTool = toolDefinitions.find(tool => tool.name === 'tensorus_apply_einsum');
            expect(einsumTool).toBeDefined();
        });
    });

    describe('tools/call handler (via exported handleToolCall function)', () => {
        afterEach(() => {
            jest.clearAllMocks();
        });

        // --- tensorus_list_datasets ---
        describe('tensorus_list_datasets', () => {
            it('should list datasets successfully', async () => {
                const mockPyApiResponse = { data: { success: true, message: "Datasets listed", data: ["ds1", "ds2"] } };
                axios.get.mockResolvedValue(mockPyApiResponse);

                const response = await handleToolCall('tensorus_list_datasets', {});
                
                expect(axios.get).toHaveBeenCalledWith(`${PYTHON_API_BASE_URL}/datasets`);
                expect(response).toEqual({ content: [{ type: 'text', text: JSON.stringify(["ds1", "ds2"]) }] });
            });

            it('should handle Python API error for list_datasets', async () => {
                const errorMessage = 'Python API error for list_datasets';
                axios.get.mockRejectedValue({ isAxiosError: true, response: { data: { detail: errorMessage }, status: 500 } });
                
                await expect(handleToolCall('tensorus_list_datasets', {}))
                    .rejects
                    .toThrow(`Python API Error (500): ${errorMessage}`);
            });
        });

        // --- tensorus_create_dataset ---
        describe('tensorus_create_dataset', () => {
            const datasetName = 'new_test_ds';
            it('should create a dataset successfully', async () => {
                const mockPyApiResponse = { data: { success: true, message: `Dataset '${datasetName}' created successfully.` } };
                axios.post.mockResolvedValue(mockPyApiResponse);

                const response = await handleToolCall('tensorus_create_dataset', { dataset_name: datasetName });

                expect(axios.post).toHaveBeenCalledWith(`${PYTHON_API_BASE_URL}/datasets/create`, { name: datasetName });
                expect(response).toEqual({ content: [{ type: 'text', text: mockPyApiResponse.data.message }] });
            });
            
            it('should handle missing dataset_name for create_dataset', async () => {
                 await expect(handleToolCall('tensorus_create_dataset', {}))
                    .rejects
                    .toThrow('Missing required argument: dataset_name');
            });

            it('should handle Python API error for create_dataset', async () => {
                const errorMessage = 'Python API error for create_dataset';
                 axios.post.mockRejectedValue({ isAxiosError: true, response: { data: { detail: errorMessage }, status: 409 } }); // e.g. conflict
                
                await expect(handleToolCall('tensorus_create_dataset', { dataset_name: datasetName }))
                    .rejects
                    .toThrow(`Python API Error (409): ${errorMessage}`);
            });
        });
        
        // --- tensorus_get_tensor_details ---
        describe('tensorus_get_tensor_details', () => {
            const dsName = 'ds1';
            const recId = 'rec123';
            it('should get tensor details successfully', async () => {
                const mockPyApiResponse = { data: { record_id: recId, shape: [2,2], dtype: 'float32', data: [[1,2],[3,4]], metadata: {} } };
                axios.get.mockResolvedValue(mockPyApiResponse);

                const response = await handleToolCall('tensorus_get_tensor_details', { dataset_name: dsName, record_id: recId });

                expect(axios.get).toHaveBeenCalledWith(`${PYTHON_API_BASE_URL}/datasets/${dsName}/tensors/${recId}`);
                expect(response).toEqual({ content: [{ type: 'text', text: JSON.stringify(mockPyApiResponse.data) }] });
            });

             it('should handle Python API error for get_tensor_details (e.g., 404)', async () => {
                const errorMessage = 'Tensor not found';
                 axios.get.mockRejectedValue({ isAxiosError: true, response: { data: { detail: errorMessage }, status: 404 } });
                
                await expect(handleToolCall('tensorus_get_tensor_details', { dataset_name: dsName, record_id: recId }))
                    .rejects
                    .toThrow(`Python API Error (404): ${errorMessage}`);
            });
        });

        // --- tensorus_apply_unary_operation ('log') ---
        describe("tensorus_apply_unary_operation ('log')", () => {
            const opArgs = { operation: 'log', input_dataset_name: 'd1', input_record_id: 't1' };
            it("should apply 'log' operation successfully", async () => {
                const mockPyApiResponse = { data: { success: true, message: "Log op done", output_record_id: "res_log_123" } };
                axios.post.mockResolvedValue(mockPyApiResponse);

                const response = await handleToolCall('tensorus_apply_unary_operation', opArgs);

                expect(axios.post).toHaveBeenCalledWith(`${PYTHON_API_BASE_URL}/ops/log`, {
                    input_tensor: { dataset_name: opArgs.input_dataset_name, record_id: opArgs.input_record_id },
                    output_dataset_name: null,
                    output_metadata: null,
                });
                expect(response).toEqual({ content: [{ type: 'text', text: JSON.stringify(mockPyApiResponse.data) }] });
            });

            it('should handle Python API error for unary "log"', async () => {
                const errorMessage = 'Error during log operation';
                axios.post.mockRejectedValue({ isAxiosError: true, response: { data: { detail: errorMessage }, status: 400 } });
                await expect(handleToolCall('tensorus_apply_unary_operation', opArgs))
                    .rejects
                    .toThrow(`Python API Error (400): ${errorMessage}`);
            });
        });
        
        // --- tensorus_apply_unary_operation ('reshape') ---
        describe("tensorus_apply_unary_operation ('reshape')", () => {
            const opArgs = { 
                operation: 'reshape', 
                input_dataset_name: 'd1', 
                input_record_id: 't1',
                params: { new_shape: [2,3] }
            };
            it("should apply 'reshape' operation successfully", async () => {
                const mockPyApiResponse = { data: { success: true, message: "Reshape op done", output_record_id: "res_reshape_456" } };
                axios.post.mockResolvedValue(mockPyApiResponse);

                const response = await handleToolCall('tensorus_apply_unary_operation', opArgs);
                
                expect(axios.post).toHaveBeenCalledWith(`${PYTHON_API_BASE_URL}/ops/reshape`, {
                    input_tensor: { dataset_name: opArgs.input_dataset_name, record_id: opArgs.input_record_id },
                    params: { new_shape: opArgs.params.new_shape },
                    output_dataset_name: null,
                    output_metadata: null,
                });
                expect(response).toEqual({ content: [{ type: 'text', text: JSON.stringify(mockPyApiResponse.data) }] });
            });
             it('should handle Python API error for unary "reshape"', async () => {
                const errorMessage = 'Invalid shape for reshape';
                axios.post.mockRejectedValue({ isAxiosError: true, response: { data: { detail: errorMessage }, status: 400 } });
                await expect(handleToolCall('tensorus_apply_unary_operation', opArgs))
                    .rejects
                    .toThrow(`Python API Error (400): ${errorMessage}`);
            });
        });

        // --- tensorus_apply_binary_operation ('add' tensor+scalar) ---
        describe("tensorus_apply_binary_operation ('add' tensor+scalar)", () => {
            const opArgs = {
                operation: 'add',
                input1_dataset_name: 'd1', input1_record_id: 't1a',
                input2_type: 'scalar', input2_scalar_value: 10
            };
            it("should apply 'add' (tensor+scalar) successfully", async () => {
                const mockPyApiResponse = { data: { success: true, message: "Add op done" } };
                axios.post.mockResolvedValue(mockPyApiResponse);
                
                const response = await handleToolCall('tensorus_apply_binary_operation', opArgs);

                expect(axios.post).toHaveBeenCalledWith(`${PYTHON_API_BASE_URL}/ops/add`, {
                    input1: { dataset_name: opArgs.input1_dataset_name, record_id: opArgs.input1_record_id },
                    input2: { scalar_value: opArgs.input2_scalar_value },
                    output_dataset_name: null,
                    output_metadata: null,
                });
                expect(response).toEqual({ content: [{ type: 'text', text: JSON.stringify(mockPyApiResponse.data) }] });
            });
            it('should handle Python API error for binary "add" (tensor+scalar)', async () => {
                const errorMessage = 'Error during add operation';
                axios.post.mockRejectedValue({ isAxiosError: true, response: { data: { detail: errorMessage }, status: 400 } });
                await expect(handleToolCall('tensorus_apply_binary_operation', opArgs))
                    .rejects
                    .toThrow(`Python API Error (400): ${errorMessage}`);
            });
        });

        // --- tensorus_apply_binary_operation ('add' tensor+tensor) ---
        describe("tensorus_apply_binary_operation ('add' tensor+tensor)", () => {
            const opArgs = {
                operation: 'add',
                input1_dataset_name: 'd1', input1_record_id: 't1a',
                input2_type: 'tensor', input2_dataset_name: 'd2', input2_record_id: 't2b'
            };
            it("should apply 'add' (tensor+tensor) successfully", async () => {
                const mockPyApiResponse = { data: { success: true, message: "Add op done" } };
                axios.post.mockResolvedValue(mockPyApiResponse);

                const response = await handleToolCall('tensorus_apply_binary_operation', opArgs);

                expect(axios.post).toHaveBeenCalledWith(`${PYTHON_API_BASE_URL}/ops/add`, {
                    input1: { dataset_name: opArgs.input1_dataset_name, record_id: opArgs.input1_record_id },
                    input2: { tensor_ref: { dataset_name: opArgs.input2_dataset_name, record_id: opArgs.input2_record_id } },
                    output_dataset_name: null,
                    output_metadata: null,
                });
                 expect(response).toEqual({ content: [{ type: 'text', text: JSON.stringify(mockPyApiResponse.data) }] });
            });
        });
        
        // --- tensorus_apply_binary_operation ('matmul') ---
        describe("tensorus_apply_binary_operation ('matmul')", () => {
            const opArgs = {
                operation: 'matmul',
                input1_dataset_name: 'd1', input1_record_id: 't1mat',
                input2_type: 'tensor', input2_dataset_name: 'd2', input2_record_id: 't2mat'
            };
            it("should apply 'matmul' (tensor+tensor) successfully", async () => {
                const mockPyApiResponse = { data: { success: true, message: "Matmul op done" } };
                axios.post.mockResolvedValue(mockPyApiResponse);

                const response = await handleToolCall('tensorus_apply_binary_operation', opArgs);
                
                expect(axios.post).toHaveBeenCalledWith(`${PYTHON_API_BASE_URL}/ops/matmul`, {
                    input1: { dataset_name: opArgs.input1_dataset_name, record_id: opArgs.input1_record_id },
                    input2: { tensor_ref: { dataset_name: opArgs.input2_dataset_name, record_id: opArgs.input2_record_id } },
                    output_dataset_name: null,
                    output_metadata: null,
                });
                expect(response).toEqual({ content: [{ type: 'text', text: JSON.stringify(mockPyApiResponse.data) }] });
            });

            it("should fail 'matmul' if input2 is scalar (MCP server validation)", async () => {
                 const scalarOpArgs = {
                    operation: 'matmul',
                    input1_dataset_name: 'd1', input1_record_id: 't1mat',
                    input2_type: 'scalar', input2_scalar_value: 5
                };
                await expect(handleToolCall('tensorus_apply_binary_operation', scalarOpArgs))
                    .rejects
                    .toThrow("Operation matmul requires input2 to be a tensor.");
            });

            it('should handle Python API error for binary "matmul"', async () => {
                const errorMessage = 'Shape mismatch for matmul';
                axios.post.mockRejectedValue({ isAxiosError: true, response: { data: { detail: errorMessage }, status: 400 } });
                await expect(handleToolCall('tensorus_apply_binary_operation', opArgs))
                    .rejects
                    .toThrow(`Python API Error (400): ${errorMessage}`);
            });
        });

        // --- tensorus_apply_list_operation ('concatenate') ---
        describe("tensorus_apply_list_operation ('concatenate')", () => {
            const opArgs = {
                operation: 'concatenate',
                input_tensors: [
                    { dataset_name: 'd1', record_id: 'tc1' },
                    { dataset_name: 'd1', record_id: 'tc2' }
                ],
                params: { dim: 0 }
            };
            it("should apply 'concatenate' successfully", async () => {
                const mockPyApiResponse = { data: { success: true, message: "Concatenate op done" } };
                axios.post.mockResolvedValue(mockPyApiResponse);

                const response = await handleToolCall('tensorus_apply_list_operation', opArgs);

                expect(axios.post).toHaveBeenCalledWith(`${PYTHON_API_BASE_URL}/ops/concatenate`, {
                    input_tensors: opArgs.input_tensors,
                    params: opArgs.params,
                    output_dataset_name: null,
                    output_metadata: null,
                });
                expect(response).toEqual({ content: [{ type: 'text', text: JSON.stringify(mockPyApiResponse.data) }] });
            });

            it('should handle Python API error for list "concatenate"', async () => {
                const errorMessage = 'Dimension mismatch for concatenate';
                axios.post.mockRejectedValue({ isAxiosError: true, response: { data: { detail: errorMessage }, status: 400 } });
                await expect(handleToolCall('tensorus_apply_list_operation', opArgs))
                    .rejects
                    .toThrow(`Python API Error (400): ${errorMessage}`);
            });
        });
    });
});
