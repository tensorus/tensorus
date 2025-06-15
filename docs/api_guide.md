# API Guide

The Tensorus Metadata System provides a comprehensive RESTful API for managing and interacting with tensor metadata.

## Interactive API Documentation (Swagger UI)

The API is self-documenting using OpenAPI. Once the application is running (e.g., locally at `http://localhost:7860`), you can access the interactive Swagger UI at:

*   **`/docs`**: [http://localhost:7860/docs](http://localhost:7860/docs)

This interface allows you to explore all available endpoints, view their request and response schemas, and even try out API calls directly from your browser.

## Alternative API Documentation (ReDoc)

An alternative ReDoc interface is also available at:

*   **`/redoc`**: [http://localhost:7860/redoc](http://localhost:7860/redoc)

## Main API Categories

The API is organized into several categories based on functionality:

*   **Tensor Descriptors:** Core operations for creating, reading, updating, deleting, and listing tensor descriptors.
*   **Semantic Metadata (Per Tensor):** Managing human-readable names, descriptions, etc., associated with specific tensors, nested under `/tensor_descriptors/{tensor_id}/semantic/`.
*   **Extended Metadata (Per Tensor):** CRUD operations for detailed metadata types, nested under `/tensor_descriptors/{tensor_id}/`:
    *   Lineage Metadata (`/lineage`)
    *   Computational Metadata (`/computational`)
    *   Quality Metadata (`/quality`)
    *   Relational Metadata (`/relational`)
    *   Usage Metadata (`/usage`)
*   **Versioning & Lineage:** Endpoints for creating tensor versions and managing lineage relationships at a higher level.
*   **Search & Aggregation:** Advanced querying, text-based search across metadata, and metadata aggregation.
*   **Import/Export:** Endpoints for exporting and importing tensor metadata in JSON format.
*   **Management:** Health checks and system metrics.
*   **Authentication:** Write operations (POST, PUT, PATCH, DELETE) are protected by API keys. See [Installation and Configuration](./installation.md) for details on setting API keys. The API key should be passed in the HTTP header specified by `TENSORUS_API_KEY_HEADER_NAME` (default: `X-API-KEY`).

Please refer to the interactive `/docs` for detailed information on each endpoint, including request parameters, request bodies, and response structures.
