const { spawn } = require('child_process');
const { Client, StdioClientTransport } = require('@modelcontextprotocol/sdk/client');
const path = require('path');
const fs = require('fs');

jest.setTimeout(60000); // allow plenty of time for servers to start

const PYTHON_API_PORT = 8000; // Ensure this matches api.py
const PYTHON_API_BASE_URL = `http://127.0.0.1:${PYTHON_API_PORT}`;
const MCP_SERVER_SCRIPT = path.join(__dirname, '../mcp_tensorus_server/server.js');
const PYTHON_API_SCRIPT = path.join(__dirname, '../api.py');
const PYTHON_VENV_ACTIVATOR = path.join(__dirname, '../.venv/bin/activate'); // Path to venv activator

let pythonApiProcess;
let mcpServerProcess;
let mcpClient;

// Utility to poll the FastAPI server until it responds
function waitForApiReady(timeoutMs = 30000, intervalMs = 500) {
    return new Promise((resolve, reject) => {
        const start = Date.now();
        const check = () => {
            fetch(`${PYTHON_API_BASE_URL}/health`).then(res => {
                if (res.ok) {
                    console.log('Python API server is reachable.');
                    resolve();
                } else {
                    next();
                }
            }).catch(() => next());
        };
        const next = () => {
            if (Date.now() - start > timeoutMs) {
                reject(new Error('Timed out waiting for Python API to become available'));
            } else {
                setTimeout(check, intervalMs);
            }
        };
        check();
    });
}

// Function to activate venv and run python script, then wait until server responds
function startPythonApiWithVenv() {
    console.log('Attempting to start Python API server with virtual environment...');
    // Check if venv activate script exists
    if (!fs.existsSync(PYTHON_VENV_ACTIVATOR)) {
        console.warn(`Python virtual environment activator not found at ${PYTHON_VENV_ACTIVATOR}.`);
        console.warn("Attempting to run 'python api.py' directly. Ensure Python dependencies are globally available or api.py is executable and has a shebang.");
        pythonApiProcess = spawn('python', [PYTHON_API_SCRIPT], { stdio: ['ignore', 'pipe', 'pipe'] });
    } else {
        // Using bash to source venv then run python. This is OS-dependent (Linux/macOS).
        // For Windows, the command would be different (e.g., `cmd /c ".venv\\Scripts\\activate && python api.py"`)
        const command = `. ${PYTHON_VENV_ACTIVATOR} && python ${PYTHON_API_SCRIPT}`;
        pythonApiProcess = spawn('bash', ['-c', command], { stdio: ['ignore', 'pipe', 'pipe'] });
    }

    pythonApiProcess.stdout.on('data', (data) => {
        console.log(`Python API STDOUT: ${data.toString()}`);
    });

    pythonApiProcess.stderr.on('data', (data) => {
        console.error(`Python API STDERR: ${data.toString()}`);
    });

    pythonApiProcess.on('error', (err) => {
        console.error('Failed to start Python API process:', err);
    });

    pythonApiProcess.on('close', (code) => {
        if (code !== 0 && code !== null) {
            console.warn(`Python API process closed with code ${code}`);
        }
    });

    return waitForApiReady();
}


function startMcpServer() {
    return new Promise((resolve, reject) => {
        console.log('Starting Node.js MCP server...');
        mcpServerProcess = spawn('node', [MCP_SERVER_SCRIPT], { stdio: ['pipe', 'pipe', 'pipe'] });

        mcpServerProcess.stdout.on('data', (data) => {
            const output = data.toString();
            console.log(`MCP Server STDOUT: ${output}`);
            if (output.includes('Tensorus MCP Server connected via stdio and ready')) {
                console.log('MCP server started and connected.');
                resolve();
            }
        });
        mcpServerProcess.stderr.on('data', (data) => {
            console.error(`MCP Server STDERR: ${data.toString()}`);
        });
        mcpServerProcess.on('error', (err) => {
            console.error('Failed to start MCP server process:', err);
            reject(err);
        });
         mcpServerProcess.on('close', (code) => {
            if (code !== 0 && code !== null) {
                console.warn(`MCP Server process closed with code ${code}`);
            }
        });
    });
}

// Helper to delay execution
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

beforeAll(async () => {
    try {
        await startPythonApiWithVenv();
        // Short pause after the health check to ensure the API is fully ready
        console.log('Waiting briefly for Python API to settle...');
        await delay(1000);

        await startMcpServer();
        
        console.log('Initializing MCP Client...');
        const transport = new StdioClientTransport(mcpServerProcess.stdin, mcpServerProcess.stdout);
        mcpClient = new Client(transport);
        await mcpClient.connect();
        console.log('MCP Client connected.');

    } catch (error) {
        console.error("Error during beforeAll setup:", error);
        // Attempt to kill processes if they started
        if (pythonApiProcess && !pythonApiProcess.killed) {
            pythonApiProcess.kill();
        }
        if (mcpServerProcess && !mcpServerProcess.killed) {
            mcpServerProcess.kill();
        }
        throw error; // Fail the test suite
    }
});

afterAll(async () => {
    console.log('Cleaning up after tests...');
    if (mcpClient) {
        console.log('Disconnecting MCP client...');
        await mcpClient.disconnect();
    }
    if (mcpServerProcess && !mcpServerProcess.killed) {
        console.log('Stopping MCP server...');
        mcpServerProcess.kill('SIGTERM'); // or 'SIGKILL'
        await new Promise(resolve => mcpServerProcess.on('close', resolve));
        console.log('MCP server stopped.');
    }
    if (pythonApiProcess && !pythonApiProcess.killed) {
        console.log('Stopping Python API server...');
        pythonApiProcess.kill('SIGTERM'); // or 'SIGKILL'
        await new Promise(resolve => pythonApiProcess.on('close', resolve));
        console.log('Python API server stopped.');
    }
    console.log('Cleanup finished.');
});

// Helper function to parse JSON content from MCP response
function parseMCPResponse(response) {
    if (response && response.content && response.content[0] && response.content[0].type === 'text') {
        try {
            return JSON.parse(response.content[0].text);
        } catch (e) {
            console.error("Failed to parse MCP response content:", response.content[0].text, e);
            throw new Error("Failed to parse MCP response content: " + response.content[0].text);
        }
    }
    throw new Error("Invalid MCP response format");
}


describe('Tensorus MCP Integration Tests', () => {
    const testDataset1 = "test_dataset_integration";
    const testDatasetIngest = "test_ingest_retrieve";
    const testDatasetUnary = "test_unary_op";
    const testDatasetBinary = "test_binary_op";

    // Cleanup helper for datasets
    async function cleanupDataset(datasetName) {
        try {
            console.log(`Attempting to cleanup dataset: ${datasetName}`);
            await mcpClient.request('tools/call', { name: 'tensorus_delete_dataset', arguments: { dataset_name: datasetName } });
            console.log(`Successfully requested deletion of dataset: ${datasetName}`);
        } catch (error) {
            // Log error but don't fail test if cleanup fails, as it might have been deleted by the test itself
            console.warn(`Could not cleanup dataset ${datasetName}:`, error.message || error);
        }
    }
    
    // Scenario 1: Basic Dataset Operations
    describe('Scenario 1: Basic Dataset Operations', () => {
        afterAll(async () => await cleanupDataset(testDataset1));

        it('should create, list, and delete a dataset', async () => {
            // 1. Create dataset
            let response = await mcpClient.request('tools/call', { name: 'tensorus_create_dataset', arguments: { dataset_name: testDataset1 } });
            let parsed = parseMCPResponse(response);
            expect(parsed).toContain(testDataset1); // Message should contain dataset name

            // 2. List datasets and verify
            response = await mcpClient.request('tools/call', { name: 'tensorus_list_datasets', arguments: {} });
            parsed = parseMCPResponse(response);
            expect(parsed).toBeInstanceOf(Array);
            expect(parsed).toContain(testDataset1);

            // 3. Delete dataset (moved to afterAll)
            // response = await mcpClient.request('tools/call', { name: 'tensorus_delete_dataset', arguments: { dataset_name: testDataset1 } });
            // parsed = parseMCPResponse(response);
            // expect(parsed).toContain("deleted successfully");

            // response = await mcpClient.request('tools/call', { name: 'tensorus_list_datasets', arguments: {} });
            // parsed = parseMCPResponse(response);
            // expect(parsed).not.toContain(testDataset1);
        });
    });

    // Scenario 2: Tensor Ingestion and Retrieval
    describe('Scenario 2: Tensor Ingestion and Retrieval', () => {
        afterAll(async () => await cleanupDataset(testDatasetIngest));

        it('should ingest and retrieve a tensor', async () => {
            await mcpClient.request('tools/call', { name: 'tensorus_create_dataset', arguments: { dataset_name: testDatasetIngest } });

            const tensorData = {
                dataset_name: testDatasetIngest,
                tensor_shape: [2, 2],
                tensor_dtype: 'float32',
                tensor_data: [[1.0, 2.0], [3.0, 4.0]],
                metadata: { test_source: 'scenario2' }
            };
            let response = await mcpClient.request('tools/call', { name: 'tensorus_ingest_tensor', arguments: tensorData });
            let parsedIngest = parseMCPResponse(response);
            expect(parsedIngest.success).toBe(true);
            const recordId = parsedIngest.data.record_id;
            expect(recordId).toBeDefined();

            response = await mcpClient.request('tools/call', { 
                name: 'tensorus_get_tensor_details', 
                arguments: { dataset_name: testDatasetIngest, record_id: recordId } 
            });
            let parsedDetails = parseMCPResponse(response);
            expect(parsedDetails.record_id).toEqual(recordId);
            expect(parsedDetails.shape).toEqual(tensorData.tensor_shape);
            expect(parsedDetails.dtype).toEqual(tensorData.tensor_dtype);
            expect(parsedDetails.data).toEqual(tensorData.tensor_data);
            expect(parsedDetails.metadata.test_source).toEqual('scenario2');
        });
    });

    // Scenario 3: Unary Operation ('log')
    describe("Scenario 3: Unary Operation ('log')", () => {
        afterAll(async () => await cleanupDataset(testDatasetUnary));

        it('should apply log operation and verify result', async () => {
            await mcpClient.request('tools/call', { name: 'tensorus_create_dataset', arguments: { dataset_name: testDatasetUnary } });
            
            const inputTensor = {
                dataset_name: testDatasetUnary,
                tensor_shape: [2, 2],
                tensor_dtype: 'float32',
                tensor_data: [[1.0, Math.E], [10.0, 100.0]], // Math.E is approx 2.718
            };
            let ingestResponse = await mcpClient.request('tools/call', { name: 'tensorus_ingest_tensor', arguments: inputTensor });
            const recordIdA = parseMCPResponse(ingestResponse).data.record_id;

            let unaryOpResponse = await mcpClient.request('tools/call', {
                name: 'tensorus_apply_unary_operation',
                arguments: {
                    operation: 'log',
                    input_dataset_name: testDatasetUnary,
                    input_record_id: recordIdA
                }
            });
            const recordIdRes = parseMCPResponse(unaryOpResponse).output_record_id;
            expect(recordIdRes).toBeDefined();

            let detailsResponse = await mcpClient.request('tools/call', {
                name: 'tensorus_get_tensor_details',
                arguments: { dataset_name: "tensor_ops_results", record_id: recordIdRes }
            });
            let resultTensor = parseMCPResponse(detailsResponse);
            
            expect(resultTensor.shape).toEqual(inputTensor.tensor_shape);
            expect(resultTensor.data[0][0]).toBeCloseTo(Math.log(1.0));
            expect(resultTensor.data[0][1]).toBeCloseTo(Math.log(Math.E));
            expect(resultTensor.data[1][0]).toBeCloseTo(Math.log(10.0));
            expect(resultTensor.data[1][1]).toBeCloseTo(Math.log(100.0));
        });
    });
    
    // Scenario 4: Binary Operation ('add')
    describe("Scenario 4: Binary Operation ('add')", () => {
        afterAll(async () => await cleanupDataset(testDatasetBinary));

        it('should add two tensors and verify result', async () => {
            await mcpClient.request('tools/call', { name: 'tensorus_create_dataset', arguments: { dataset_name: testDatasetBinary } });

            const tensorXData = { dataset_name: testDatasetBinary, tensor_shape: [2,2], tensor_dtype: 'int32', tensor_data: [[1,1],[1,1]] };
            const tensorYData = { dataset_name: testDatasetBinary, tensor_shape: [2,2], tensor_dtype: 'int32', tensor_data: [[2,2],[2,2]] };

            let ingestX = await mcpClient.request('tools/call', { name: 'tensorus_ingest_tensor', arguments: tensorXData });
            const recordIdX = parseMCPResponse(ingestX).data.record_id;
            let ingestY = await mcpClient.request('tools/call', { name: 'tensorus_ingest_tensor', arguments: tensorYData });
            const recordIdY = parseMCPResponse(ingestY).data.record_id;

            let addOpResponse = await mcpClient.request('tools/call', {
                name: 'tensorus_apply_binary_operation',
                arguments: {
                    operation: 'add',
                    input1_dataset_name: testDatasetBinary,
                    input1_record_id: recordIdX,
                    input2_type: 'tensor',
                    input2_dataset_name: testDatasetBinary,
                    input2_record_id: recordIdY
                }
            });
            const recordIdSum = parseMCPResponse(addOpResponse).output_record_id;
            expect(recordIdSum).toBeDefined();

            let detailsResponse = await mcpClient.request('tools/call', {
                name: 'tensorus_get_tensor_details',
                arguments: { dataset_name: "tensor_ops_results", record_id: recordIdSum }
            });
            let resultTensor = parseMCPResponse(detailsResponse);

            expect(resultTensor.shape).toEqual([2,2]);
            expect(resultTensor.data).toEqual([[3,3],[3,3]]);
        });
    });

    // Scenario 5: Error Handling
    describe('Scenario 5: Error Handling', () => {
        it('should handle operation on non-existent tensor', async () => {
            try {
                await mcpClient.request('tools/call', {
                    name: 'tensorus_apply_unary_operation',
                    arguments: {
                        operation: 'log',
                        input_dataset_name: "non_existent_ds_for_error_test", // Or a real ds with fake tensor_id
                        input_record_id: "fake_record_id_for_error_test"
                    }
                });
                fail("Operation should have failed but succeeded."); // Jest's fail function
            } catch (error) {
                // Check if the error message indicates a 404 or similar from the Python API via MCP
                expect(error.message).toBeDefined();
                // Example: "Python API Error (404): Dataset 'non_existent_ds_for_error_test' not found."
                // Or "Python API Error (404): Input tensor 'fake_record_id_for_error_test' not found in dataset '...'"
                console.log("Caught expected error:", error.message);
                expect(error.message).toMatch(/Python API Error \(404\)/i);
                expect(error.message).toMatch(/not found/i);
            }
        });
    });
});
