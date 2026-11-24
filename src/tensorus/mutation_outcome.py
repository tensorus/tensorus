"""Preserve uncertain outcomes when a post-commit verification fails."""

from collections.abc import Callable

from tensorus.errors import StorageError, TensorusError

try:
    from sqlalchemy.exc import SQLAlchemyError as _BackendDatabaseError
except ImportError:  # pragma: no cover - standalone public SDK without SQLAlchemy
    _BackendDatabaseError = StorageError  # type: ignore[assignment,misc]


def verify_after_mutation[T](operation: Callable[[], T], label: str) -> T:
    try:
        return operation()
    except (TensorusError, _BackendDatabaseError) as exc:
        message = (
            f"{label} may have committed; inspect it with an active credential before retrying"
        )
        if isinstance(exc, TensorusError) and not isinstance(exc, StorageError):
            exc.args = (f"{exc}; {message}",)
            raise
        raise StorageError(message) from exc
