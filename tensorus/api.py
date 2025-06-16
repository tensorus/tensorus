import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from fastapi import Request, Response

from tensorus.api import context
from tensorus.api.endpoints import (
    router_tensor_descriptor,
    router_semantic_metadata,
    router_search_aggregate,
    router_version_lineage,
    router_extended_metadata,
    router_io,
    router_management,
    router_analytics,
)
from tensorus.api.routers.datasets import router as datasets_router
from tensorus.api.routers.query import router as query_router
from tensorus.api.routers.agents import router as agents_router
from tensorus.api.routers.tensor_ops import router as tensor_ops_router

logger = logging.getLogger(__name__)


class APIError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        logger.info("Request: %s %s", request.method, request.url)
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error("Request failed: %s", exc)
            if not isinstance(exc, APIError):
                exc = APIError(status_code=500, detail="Internal server error")
            response = JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        return response


def add_security_headers(request: Request, call_next):
    async def _inner():
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response
    return _inner()


app = FastAPI(title="Tensorus API", version="0.2.1")
app.add_middleware(LoggingMiddleware)
app.middleware("http")(add_security_headers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors(), "body": exc.body})


# Re-export globals for compatibility with tests
tensor_storage_instance = context.tensor_storage_instance
nql_agent_instance = context.nql_agent_instance
agent_registry = context.agent_registry
live_agents = context.live_agents
_get_or_create_ingestion_agent = context._get_or_create_ingestion_agent
get_tensor_storage = context.get_tensor_storage
get_nql_agent = context.get_nql_agent

# Include routers
app.include_router(router_tensor_descriptor)
app.include_router(router_semantic_metadata)
app.include_router(router_search_aggregate)
app.include_router(router_version_lineage)
app.include_router(router_extended_metadata)
app.include_router(router_io)
app.include_router(router_management)
app.include_router(router_analytics)
app.include_router(datasets_router)
app.include_router(query_router)
app.include_router(agents_router)
app.include_router(tensor_ops_router)


@app.get("/", include_in_schema=False)
async def read_root():
    return {"message": "Welcome to the Tensorus API! Visit /docs for documentation."}
