"""
Tensorus API Documentation

This module provides comprehensive API documentation for the Tensorus platform.
The documentation is automatically generated using FastAPI's OpenAPI/Swagger UI.
"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from typing import Dict, Any, List, Optional
from enum import Enum

# Import your main FastAPI app
from tensorus.api import app

# Custom OpenAPI schema with detailed documentation
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Tensorus API",
        version="1.0.0",
        description="""
        # Tensorus API Documentation
        
        Welcome to the Tensorus API! This API provides comprehensive tensor operations,
        vector database functionality, and metadata management.
        
        ## Authentication
        - Use the `X-API-KEY` header for API key authentication
        - Contact support to obtain an API key
        
        ## Rate Limiting
        - 1000 requests per minute per API key
        - 10000 requests per day per IP address
        
        ## Error Handling
        - 400: Bad Request - Invalid input
        - 401: Unauthorized - Missing or invalid API key
        - 403: Forbidden - Insufficient permissions
        - 404: Not Found - Resource not found
        - 422: Validation Error - Invalid request data
        - 429: Too Many Requests - Rate limit exceeded
        - 500: Internal Server Error - Something went wrong
        
        ## Common Headers
        - `X-API-KEY`: Your API key
        - `Content-Type: application/json` for JSON requests
        """,
        routes=app.routes,
    )
    
    # Add detailed parameter descriptions and examples
    components = openapi_schema.get("components", {})
    components["schemas"].update({
        "Tensor": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "format": "uuid", "description": "Unique identifier for the tensor"},
                "name": {"type": "string", "description": "Human-readable name for the tensor"},
                "data": {"type": "array", "items": {},"description": "Nested array containing tensor data"},
                "shape": {"type": "array", "items": {"type": "integer"}, "description": "Dimensions of the tensor"},
                "dtype": {"type": "string", "enum": ["float32", "float64", "int32", "int64", "bool"], "description": "Data type of tensor elements"},
                "metadata": {"type": "object", "description": "Additional metadata as key-value pairs"},
                "created_at": {"type": "string", "format": "date-time", "description": "Timestamp of creation"},
                "updated_at": {"type": "string", "format": "date-time", "description": "Timestamp of last update"}
            },
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "sample_matrix",
                "data": [[1, 2], [3, 4]],
                "shape": [2, 2],
                "dtype": "int32",
                "metadata": {"source": "synthetic", "tags": ["test", "example"]},
                "created_at": "2025-09-03T12:00:00Z",
                "updated_at": "2025-09-03T12:00:00Z"
            }
        },
        "VectorIndex": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Unique name of the index"},
                "dimension": {"type": "integer", "minimum": 1, "description": "Dimensionality of vectors in the index"},
                "metric": {"type": "string", "enum": ["cosine", "euclidean", "dot"], "description": "Distance metric for similarity search"},
                "description": {"type": "string", "description": "Human-readable description"},
                "vector_count": {"type": "integer", "minimum": 0, "description": "Number of vectors in the index"},
                "created_at": {"type": "string", "format": "date-time"}
            }
        },
        "Error": {
            "type": "object",
            "properties": {
                "error": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Machine-readable error code"},
                        "message": {"type": "string", "description": "Human-readable error message"},
                        "details": {"type": "object", "description": "Additional error details"}
                    }
                }
            },
            "example": {
                "error": {
                    "code": "invalid_input",
                    "message": "Invalid input data",
                    "details": {"field": "dimension", "issue": "must be a positive integer"}
                }
            }
        }
    })
    
    openapi_schema["components"] = components
    
    # Add detailed documentation for all endpoints
    paths = openapi_schema["paths"]
    
    # Tensor Endpoints
    if "/tensors" in paths:
        paths["/tensors"]["post"].update({
            "summary": "Create a new tensor",
            "description": "Create a new tensor with the provided data and metadata.",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "$ref": "#/components/schemas/Tensor",
                            "required": ["data"]
                        },
                        "examples": {
                            "simple_matrix": {
                                "summary": "Simple 2x2 matrix",
                                "value": {
                                    "name": "identity_matrix",
                                    "data": [[1, 0], [0, 1]],
                                    "dtype": "float32",
                                    "metadata": {"type": "identity"}
                                }
                            }
                        }
                    }
                }
            },
            "responses": {
                "201": {
                    "description": "Tensor created successfully",
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Tensor"}}}
                },
                "400": {"$ref": "#/components/responses/BadRequest"},
                "401": {"$ref": "#/components/responses/Unauthorized"}
            }
        })
    
    if "/tensors/{tensor_id}" in paths:
        paths["/tensors/{tensor_id}"]["get"].update({
            "summary": "Get tensor by ID",
            "description": "Retrieve detailed information about a specific tensor by its unique identifier.",
            "parameters": [{
                "name": "tensor_id",
                "in": "path",
                "required": True,
                "schema": {"type": "string", "format": "uuid"},
                "description": "UUID of the tensor to retrieve",
                "example": "550e8400-e29b-41d4-a716-446655440000"
            }],
            "responses": {
                "200": {
                    "description": "Tensor details",
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Tensor"}}}
                },
                "404": {"$ref": "#/components/responses/NotFound"}
            }
        })
    
    # Vector Index Endpoints
    if "/vector/indices" in paths:
        paths["/vector/indices"]["post"].update({
            "summary": "Create a new vector index",
            "description": "Create a new vector index for similarity search.",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["name", "dimension"],
                            "properties": {
                                "name": {"type": "string", "description": "Unique name for the index"},
                                "dimension": {"type": "integer", "minimum": 1, "description": "Dimensionality of vectors"},
                                "metric": {"type": "string", "enum": ["cosine", "euclidean", "dot"], "default": "cosine"},
                                "description": {"type": "string"}
                            }
                        },
                        "example": {
                            "name": "image_embeddings",
                            "dimension": 512,
                            "metric": "cosine",
                            "description": "Index for image feature vectors"
                        }
                    }
                }
            },
            "responses": {
                "201": {
                    "description": "Index created successfully",
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/VectorIndex"}}}
                },
                "400": {"$ref": "#/components/responses/BadRequest"}
            }
        })
    
    # Add common responses
    components = openapi_schema.setdefault("components", {})
    components.setdefault("responses", {
        "BadRequest": {
            "description": "Invalid request data",
            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}
        },
        "Unauthorized": {
            "description": "Missing or invalid authentication",
            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}
        },
        "NotFound": {
            "description": "Requested resource not found",
            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}
        },
        "RateLimitExceeded": {
            "description": "Rate limit exceeded",
            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}},
            "headers": {
                "X-RateLimit-Limit": {"schema": {"type": "integer"}, "description": "Request limit per minute"},
                "X-RateLimit-Remaining": {"schema": {"type": "integer"}, "description": "Remaining requests in current window"},
                "X-RateLimit-Reset": {"schema": {"type": "integer"}, "description": "Unix timestamp when limits reset"}
            }
        }
    })
    
    # Add more endpoint documentation as needed...
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Apply the custom OpenAPI schema
app.openapi = custom_openapi

# Custom Swagger UI endpoint with improved styling
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Tensorus API - Swagger UI",
        swagger_js_url="https://unpkg.com/swagger-ui-dist@4.5.0/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@4.5.0/swagger-ui.css",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
    )
