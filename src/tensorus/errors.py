class TensorusError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_request",
        status: int = 422,
        details: object | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.status = status
        self.details = details


class NotFound(TensorusError):
    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, code="not_found", status=404)


class Conflict(TensorusError):
    def __init__(self, message: str, *, code: str = "conflict", details: object | None = None):
        super().__init__(message, code=code, status=409, details=details)


class Forbidden(TensorusError):
    def __init__(
        self,
        message: str = "This credential does not permit the requested action",
        *,
        code: str = "forbidden",
        details: object | None = None,
    ):
        super().__init__(message, code=code, status=403, details=details)


class ScopedDataAccessDenied(Forbidden):
    """A scoped reader lacks access to a recorded data dependency."""


class Unauthenticated(TensorusError):
    def __init__(self, message: str = "A valid, unexpired Bearer API key is required"):
        super().__init__(message, code="unauthenticated", status=401)


class LimitExceeded(TensorusError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "limit_exceeded",
        status: int = 413,
        details: object | None = None,
    ):
        super().__init__(message, code=code, status=status, details=details)


class RateLimited(TensorusError):
    def __init__(self, retry_after: int):
        super().__init__(
            "Tenant request rate limit reached; retry later",
            code="tenant_rate_limited",
            status=429,
            details={"retry_after_seconds": retry_after},
        )
        self.retry_after = retry_after


class StorageError(TensorusError):
    def __init__(self, message: str):
        super().__init__(message, code="storage_error", status=503)


class LeaseLost(TensorusError):
    def __init__(self) -> None:
        super().__init__("Worker lease is no longer valid", code="lease_lost", status=409)
