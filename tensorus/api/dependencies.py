from tensorus.metadata import storage_instance as globally_configured_storage_instance
from tensorus.metadata.storage_abc import MetadataStorage

# Note: The `storage_instance` imported here is already configured (InMemory or Postgres)
# based on the logic in `tensorus/metadata/__init__.py` which reads `tensorus.config.settings`.

def get_storage_instance() -> MetadataStorage:
    """
    FastAPI dependency to get the currently configured metadata storage instance.
    """
    return globally_configured_storage_instance

# If we needed to re-initialize or pass settings directly to the storage for each request
# (e.g. for request-scoped sessions or dynamic configuration per request, which is not the case here),
# this function would be more complex. For now, it just returns the global instance.
#
# Example of re-initializing if storage_instance could change or needs request context:
# from tensorus.metadata import get_configured_storage_instance, ConfigurationError
# from tensorus.config import settings
#
# def get_storage_instance_dynamic() -> MetadataStorage:
#     try:
#         # This would re-evaluate settings and re-create the instance per request if needed,
#         # or could access request-specific config.
#         # For our current setup, the global instance is fine.
#         return get_configured_storage_instance()
#     except ConfigurationError as e:
#         # This would ideally be caught by a global exception handler in FastAPI
#         # to return a 500 error if configuration is bad during a request.
#         # However, configuration should typically be validated at startup.
#         raise RuntimeError(f"Storage configuration error: {e}")

# The current `storage_instance` is initialized once when `tensorus.metadata` is first imported.
# This is generally fine for many applications unless the configuration needs to change without restarting.
# The FastAPI `Depends` system will call `get_storage_instance` for each request that uses it,
# but this function will always return the same globally initialized `storage_instance`.
