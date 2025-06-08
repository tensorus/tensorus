from fastapi import FastAPI

from .endpoints import (
    router_tensor_descriptor,
    router_semantic_metadata,
    router_search_aggregate,
    router_version_lineage,
    router_extended_metadata # Import the new router for extended metadata CRUD
)

app = FastAPI(
    title="Tensorus API",
    version="0.1.0",
    description="API for managing Tensor Descriptors and Semantic Metadata.",
    contact={
        "name": "Tensorus Development Team",
        "url": "http://example.com/contact", # Replace with actual contact/repo URL
        "email": "dev@example.com", # Replace with actual email
    },
    license_info={
        "name": "Apache 2.0", # Or your chosen license
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html", # Replace with actual license URL
    },
)

# Include the routers
app.include_router(router_tensor_descriptor)
app.include_router(router_semantic_metadata)
app.include_router(router_search_aggregate)
app.include_router(router_version_lineage)
app.include_router(router_extended_metadata) # Register the new router

@app.get("/", tags=["Root"], summary="Root Endpoint", description="Returns a welcome message for the Tensorus API.")
async def read_root():
    return {"message": "Welcome to the Tensorus API"}

# To run this application (for development):
# uvicorn tensorus.api.main:app --reload --port 8000
#
# You would typically have a `__main__.py` or a run script for this.
# For now, this structure allows importing `app` elsewhere if needed.

# Example of how to clear storage for testing (not a production endpoint)
# from tensorus.metadata.storage import storage_instance
# @app.post("/debug/clear_storage", tags=["Debug"], include_in_schema=False)
# async def debug_clear_storage():
#     storage_instance.clear_all_data()
#     return {"message": "All in-memory data cleared."}
