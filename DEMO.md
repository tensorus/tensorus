# Tensorus Demo Script: Showcasing Agentic Tensor Management

## Introduction

This demo showcases the key capabilities of Tensorus, an agentic tensor database/data lake. We'll walk through data ingestion, storage, querying, tensor operations, and how to interact with the system via its UI and APIs.

## Prerequisites

*   Tensorus backend API running (`uvicorn tensorus.api:app --reload --host 127.0.0.1 --port 7860`)
*   Tensorus Streamlit UI running (`streamlit run app.py`)
*   Tensorus MCP Server running (`python -m tensorus.mcp_server`)
*   A terminal for API calls (e.g., using `curl`) or a tool like Postman.
*   A web browser.
*   API key configured for protected endpoints:
    * Generate (if needed): `python generate_api_key.py`
    * Use Bearer header (preferred): `Authorization: Bearer <api_key>`
    * Legacy also supported: `X-API-KEY: <api_key>`

## Vector & Embedding API Quickstart

These endpoints are mounted under `prefix="/api/v1/vector"` in `tensorus/api.py`, and require auth via `Authorization: Bearer <api_key>` by default (see `tensorus/api/security.py`).

Set your API key in PowerShell (Windows):

```powershell
$env:TENSORUS_API_KEY = "tsr_...your_key..."
```

### 1) Embed text(s) and store vectors

```powershell
curl -s -X POST "http://127.0.0.1:7860/api/v1/vector/embed" \
 -H "Authorization: Bearer $env:TENSORUS_API_KEY" \
 -H "Content-Type: application/json" \
 -d "{ \"texts\": [\"hello world\", \"tensorus vector db\"], \"dataset_name\": \"demo_vectors\" }"
```

Response includes `record_ids`, `embeddings_count`, and `model_info`.

### 2) Similarity search

```powershell
curl -s -X POST "http://127.0.0.1:7860/api/v1/vector/search" \
 -H "Authorization: Bearer $env:TENSORUS_API_KEY" \
 -H "Content-Type: application/json" \
 -d "{ \"query\": \"hello world\", \"dataset_name\": \"demo_vectors\", \"k\": 5 }"
```

Optional fields: `model_name`, `provider`, `namespace`, `tenant_id`, `similarity_threshold`, `include_vectors`.

### 3) Hybrid search (semantic + computational)

```powershell
curl -s -X POST "http://127.0.0.1:7860/api/v1/vector/hybrid-search" \
 -H "Authorization: Bearer $env:TENSORUS_API_KEY" \
 -H "Content-Type: application/json" \
 -d "{
  \"text_query\": \"hello\",
  \"dataset_name\": \"demo_vectors\",
  \"tensor_operations\": [
    { \"operation_name\": \"norm\", \"parameters\": { \"p\": 2 } }
  ],
  \"similarity_weight\": 0.7,
  \"computation_weight\": 0.3,
  \"k\": 5
}"
```

Note: `similarity_weight + computation_weight` must equal `1.0`.

### 4) Build vector index

```powershell
curl -s -X POST "http://127.0.0.1:7860/api/v1/vector/index/build" \
 -H "Authorization: Bearer $env:TENSORUS_API_KEY" \
 -H "Content-Type: application/json" \
 -d "{ \"dataset_name\": \"demo_vectors\", \"index_type\": \"partitioned\", \"metric\": \"cosine\", \"num_partitions\": 8 }"
```

### 5) List available embedding models

```powershell
curl -s -X GET "http://127.0.0.1:7860/api/v1/vector/models" \
 -H "Authorization: Bearer $env:TENSORUS_API_KEY"
```

### 6) Dataset embedding stats

```powershell
curl -s -X GET "http://127.0.0.1:7860/api/v1/vector/stats/demo_vectors" \
 -H "Authorization: Bearer $env:TENSORUS_API_KEY"
```

### 7) System performance metrics

```powershell
curl -s -X GET "http://127.0.0.1:7860/api/v1/vector/metrics" \
 -H "Authorization: Bearer $env:TENSORUS_API_KEY"
```

### 8) Delete vectors by IDs

Use repeated `vector_ids` query params:

```powershell
curl -s -X DELETE "http://127.0.0.1:7860/api/v1/vector/vectors/demo_vectors?vector_ids=<id1>&vector_ids=<id2>" \
 -H "Authorization: Bearer $env:TENSORUS_API_KEY"
```

## Demo Scenarios

### Scenario 1: Automated Image Data Ingestion and Exploration

**Goal:** Demonstrate how new image data is automatically ingested, stored, and can be explored.

1.  **Prepare for Ingestion:**
    *   Open the Streamlit UI in your browser (usually `http://localhost:8501`).
    *   Navigate to the "Agents" or "Control Panel" page.
    *   Ensure the "Data Ingestion Agent" is targeting the dataset `ingested_data_api` and monitoring the directory `temp_ingestion_source_api`. (You might need to configure this in `tensorus/api.py` if not dynamically configurable via UI).
    *   Start the "Data Ingestion Agent" if it's not already running. View its logs in the UI to see it polling.
    *   Verify the `temp_ingestion_source_api` directory (relative to where the API is run) is initially empty or clean it out.

2.  **Simulate New Data Arrival:**
    *   Download a sample image (e.g., a picture of a cat, `cat.jpg`) into the `temp_ingestion_source_api` directory on your local filesystem.
    *   **Attractive Element:** Use a visually distinct and interesting image.

3.  **Observe Ingestion:**
    *   In the Streamlit UI, watch the Ingestion Agent's logs. You should see messages indicating it detected `cat.jpg`, processed it, and ingested it into `ingested_data_api`.

4.  **Explore Ingested Data (UI):**
    *   Navigate to the "Data Explorer" page in the Streamlit UI.
    *   Select the `ingested_data_api` dataset from the dropdown.
    *   You should see the newly ingested `cat.jpg` (or its metadata) listed.
    *   Click to view its details (metadata, tensor shape, perhaps a preview if the UI supports it).
    *   **Attractive Element:** The UI showing the image tensor's information clearly, perhaps even a thumbnail if the UI is advanced.

5.  **Verify via API (Optional):**
    *   Use `curl` or Postman to fetch the tensor details. First, list datasets to find `ingested_data_api`, then fetch its records, identify the `record_id` for `cat.jpg` from its metadata.
        ```bash
        # Example: List records in the dataset to find the ID (paged)
        curl "http://127.0.0.1:7860/datasets/ingested_data_api/records?offset=0&limit=100"
        # Then use the ID:
        # curl http://127.0.0.1:7860/datasets/ingested_data_api/tensors/{record_id_of_cat_jpg}
        ```
    *   Confirm the API returns the tensor data and metadata.

### Scenario 2: Natural Language Querying for Specific Data

**Goal:** Show how NQL can be used to find specific tensors.

1.  **Ensure Data Exists:**
    *   Make sure the `cat.jpg` from Scenario 1 is present in `ingested_data_api`.
    *   Optionally, add another image, e.g., `dog.png`, to `temp_ingestion_source_api` and let it be ingested. Give it distinctive metadata if possible, e.g., by modifying the `ingestion_agent.py` to add `{"animal_type": "dog"}` or by using the API to update metadata post-ingestion.

2.  **Use NQL Chatbot (UI):**
    *   Navigate to the "NQL Chatbot" or "Query Hub" page in the Streamlit UI.
    *   Enter a query. Given the basic NQL, a query targeting the filename is most reliable:
        `show all data from ingested_data_api where source_file contains "cat.jpg"`
    *   If you added custom metadata like `{"animal_type": "cat"}` for the cat image, you could try:
        `find records from ingested_data_api where animal_type = 'cat'`
    *   **Attractive Element:** The chatbot interface and the directness of the natural language query yielding correct results.

3.  **Observe Results:**
    *   The UI should display the tensor(s) matching your query.

### Scenario 3: Performing Tensor Operations via API

**Goal:** Demonstrate applying a tensor operation to a stored tensor.

1.  **Identify a Target Tensor:**
    *   From Scenario 1 or 2, obtain the `record_id` of the `cat.jpg` tensor within the `ingested_data_api` dataset. Let's assume its `record_id` is `xyz123`.

2.  **Perform a Tensor Operation (e.g., Transpose API):**
    *   Image tensors are often (C, H, W) or (H, W, C). Let's assume it's (C, H, W) and we want to transpose H and W, which would be `dim0=1, dim1=2`.
    *   Use `curl` or Postman:
        ```bash
        curl -X POST -H "Content-Type: application/json" -d '{
          "input_tensor": {
            "dataset_name": "ingested_data_api",
            "record_id": "xyz123" 
          },
          "params": {
            "dim0": 1, 
            "dim1": 2  
          },
          "output_dataset_name": "ops_results",
          "output_metadata": {"original_id": "xyz123", "operation": "transpose_height_width_demo"}
        }' http://127.0.0.1:7860/ops/transpose
        ```
    *   **Attractive Element:** Showing the API call and the structured JSON response indicating success and the new transposed tensor's ID and details.

3.  **Verify Result:**
    *   The API response will contain the `record_id` and details of the new (transposed) tensor in the `ops_results` dataset.
    *   Note the new shape in the response. If the original was (3, 128, 128), the new shape should be (3, 128, 128) after transposing height and width (assuming they were the same). If they were different, e.g. (3, 128, 200), the new shape would be (3, 200, 128).
    *   Optionally, use the "Data Explorer" in the UI or another API call to fetch and inspect this new tensor.

### Scenario 4: Interacting with the MCP Server (Conceptual)

**Goal:** Explain how an external AI agent could leverage Tensorus via MCP.

1.  **Show MCP Server Running:**
    *   Point to the terminal where `python -m tensorus.mcp_server` is running and show its log output (e.g., "Tensorus MCP Server connected via stdio and ready.").

2.  **Explain Available Tools (Conceptual):**
    *   Briefly show the tool definitions in `tensorus/mcp_server.py` or refer to the README's "Available Tools" under "MCP Server Details".
    *   Highlight a few tools like `tensorus_list_datasets`, `tensorus_ingest_tensor`, and `tensorus_apply_binary_operation`.

3.  **Conceptual Client Interaction (Show code snippet from README):**
    *   Show the example client-side JavaScript snippet from the `README.md`:
        ```javascript
        // Conceptual MCP client-side JavaScript
        // Assuming 'client' is an initialized MCP client connected to the Tensorus MCP Server

        async function example() {
          // List available tools
          const { tools } = await client.request({ method: 'tools/list' }, {});
          console.log("Available Tensorus Tools:", tools.map(t => t.name));

          // Create a new dataset
          const createResponse = await client.request({ method: 'tools/call' }, {
            name: 'tensorus_create_dataset',
            arguments: { dataset_name: 'my_mcp_dataset_demo' }
          });
          console.log(JSON.parse(createResponse.content[0].text).message); // MCP server often returns JSON string in text
        }
        ```
    *   **Attractive Element:** Emphasize that this allows *other* AI agents or LLMs to programmatically use Tensorus as a modular component in a larger intelligent system, promoting interoperability.

### Scenario 5: Financial Time Series Forecasting with ARIMA

**Goal:** Demonstrate end-to-end time series forecasting using generated financial data, Tensorus for storage, an ARIMA model for prediction, and visualization within a dedicated UI page.

**Prerequisites Specific to this Demo:**
*   Ensure `statsmodels` is installed. If you used the standard setup, install it via the optional models extras:
    ```bash
    pip install -e .[models]
    ```
    or install the `tensorus-models` package which includes it.

**Steps:**

1.  **Navigate to the Demo Page:**
    *   Open the Streamlit UI (e.g., `http://localhost:8501`).
    *   From the top navigation bar (or sidebar if the UI structure varies), find and click on the "Financial Forecast Demo" page (it might be titled "📈 Financial Forecast Demo" or similar).

2.  **Generate and Ingest Data:**
    *   On the "Financial Forecast Demo" page, locate the section "1. Data Generation & Ingestion."
    *   Click the button labeled **"Generate & Ingest Sample Financial Data"**.
    *   Wait for the spinner to complete. You should see:
        *   A success message indicating data ingestion into a dataset like `financial_raw_data` in Tensorus.
        *   A sample DataFrame (head) of the generated data (Date, Close, Volume).
        *   A Plotly chart displaying the historical 'Close' prices that were just generated and ingested.
    *   **Attractive Element:** Observe the immediate visualization of the generated time series.

3.  **Configure and Run ARIMA Prediction:**
    *   Go to the "3. ARIMA Model Prediction" section on the page.
    *   You can adjust the ARIMA order (p, d, q) and the number of future predictions if you wish. Default values (e.g., p=5, d=1, q=0, predictions=30) are provided.
    *   Click the button labeled **"Run ARIMA Prediction"**.
    *   Wait for the spinner to complete. This step involves:
        *   Loading the historical data from Tensorus.
        *   Training/fitting the ARIMA model.
        *   Generating future predictions.
        *   Storing these predictions back into Tensorus (e.g., into `financial_predictions` dataset).

4.  **View Prediction Results:**
    *   Once the prediction is complete, scroll to the "4. Prediction Results" section.
    *   You should see:
        *   A Plotly chart displaying the original historical data with the ARIMA predictions plotted alongside/extending from it.
        *   A table or list showing the actual predicted values for future dates.
    *   **Attractive Element:** The clear visual comparison of the forecast against the historical data, showcasing the model's predictive attempt. The interactivity of Plotly charts (zoom, pan) enhances this.

5.  **Interpretation (What this demonstrates):**
    *   **Data Flow:** Generation -> Tensorus Storage (Raw Data) -> Retrieval for Modeling -> Prediction -> Tensorus Storage (Predictions) -> UI Visualization.
    *   **Ease of Use:** A user-friendly interface to perform a complex task like time series forecasting.
    *   **Modularity:** Integration of data generation, storage (Tensorus), modeling (statsmodels), and UI (Streamlit) components.
    *   **Revised UI:** Notice the potentially improved layout and charting capabilities on this dedicated demo page.

### Scenario 6: Dashboard Overview

**Goal:** Show the main dashboard providing a system overview.

1.  **Navigate to Dashboard:**
    *   In the Streamlit UI, go to the main "Nexus Dashboard" page (usually the default page when you run `streamlit run app.py`).
2.  **Review Metrics:**
    *   Point out the key metrics displayed:
        *   Total Tensors / Active Datasets (these might be placeholders or simple counts).
        *   Agents Online / Status (showing the Ingestion Agent as 'running' if you started it).
        *   API Status (should be 'Connected').
        *   Simulated metrics like data ingestion rate, query latency, RL rewards, AutoML progress. Explain these are illustrative.
    *   **Attractive Element:** A visually appealing dashboard. If any metrics are updating (even if simulated based on time), it adds to the dynamic feel.

3.  **Activity Feed (if populated):**
    *   Show the "Recent Agent Activity" feed. If the Ingestion Agent is running, it might populate this feed, or it might be placeholder data.
    *   Explain that in a fully operational system, this feed would show real-time updates from all active agents.

## Conclusion

This demo has provided a glimpse into Tensorus's capabilities, including automated data handling by agents, flexible data storage, natural language querying, powerful tensor operations, and a user-friendly interface. The MCP server further extends its reach, allowing programmatic interaction from other AI systems, paving the way for more complex, collaborative AI workflows.
