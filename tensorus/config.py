from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pydantic import field_validator

class Settings(BaseSettings):
    # Configuration for environment variable prefix, .env file, etc.
    # For Pydantic-Settings v2, environment variables are loaded by default.
    # To specify a prefix for env vars (e.g. TENSORUS_STORAGE_BACKEND):
    # model_config = SettingsConfigDict(env_prefix='TENSORUS_') # Pydantic v2
    # For Pydantic v1, it was `Config.env_prefix`.

    STORAGE_BACKEND: str = "in_memory"

    POSTGRES_HOST: Optional[str] = None
    POSTGRES_PORT: Optional[int] = 5432 # Default PostgreSQL port
    POSTGRES_USER: Optional[str] = None
    POSTGRES_PASSWORD: Optional[str] = None
    POSTGRES_DB: Optional[str] = None
    POSTGRES_DSN: Optional[str] = None # Alternative to individual params

    # Example of how to load from a .env file if needed (not strictly required by subtask)
    # model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')


# Global instance of the settings
# The environment variables will be loaded when this instance is created.
# e.g. TENSORUS_STORAGE_BACKEND=postgres will override the default.
# Note: For pydantic-settings, env var names are case-insensitive by default for matching.
# If env_prefix is set, it would be TENSORUS_STORAGE_BACKEND. Without it, it's just STORAGE_BACKEND.
# Let's assume no prefix for now, so environment variables should be STORAGE_BACKEND, POSTGRES_HOST etc.
# OR, more commonly, one would use the env_prefix.
# For this exercise, I will assume the user will set environment variables like:
# export STORAGE_BACKEND="postgres"
# export POSTGRES_USER="myuser"
# ...etc.
# Or, if using an .env file:
# STORAGE_BACKEND="postgres"
# POSTGRES_USER="myuser"
# ...
#
# For Pydantic V1 BaseSettings, it would be:
# class Settings(BaseSettings):
#     STORAGE_BACKEND: str = "in_memory"
#     # ... other fields
#     class Config:
#         env_prefix = "TENSORUS_" # e.g. TENSORUS_STORAGE_BACKEND
#         # case_sensitive = False # for Pydantic V1
#
# Given the project uses pydantic 1.10, I will use the V1 style for env_prefix.

class SettingsV1(BaseSettings):
    STORAGE_BACKEND: str = "in_memory"
    POSTGRES_HOST: Optional[str] = None
    POSTGRES_PORT: Optional[int] = 5432
    POSTGRES_USER: Optional[str] = None
    POSTGRES_PASSWORD: Optional[str] = None
    POSTGRES_DB: Optional[str] = None
    POSTGRES_DSN: Optional[str] = None

    # API Security
    VALID_API_KEYS: list[str] = [] # Comma-separated string in env, e.g., "key1,key2,key3"
    API_KEY_HEADER_NAME: str = "X-API-KEY"
    AUDIT_LOG_PATH: str = "tensorus_audit.log"




    model_config = SettingsConfigDict(env_prefix="TENSORUS_", case_sensitive=False)

    @field_validator("VALID_API_KEYS", mode="before")
    def split_valid_api_keys(cls, v):
        if isinstance(v, str):
            return [key.strip() for key in v.split(',') if key.strip()]
        return v

    # JWT Authentication (Conceptual Settings)
    AUTH_JWT_ENABLED: bool = False
    AUTH_JWT_ISSUER: Optional[str] = None
    AUTH_JWT_AUDIENCE: Optional[str] = None
    AUTH_JWT_ALGORITHM: str = "RS256"
    AUTH_JWT_JWKS_URI: Optional[str] = None
    AUTH_DEV_MODE_ALLOW_DUMMY_JWT: bool = False


# Use SettingsV1 for Pydantic v1.x compatibility
settings = SettingsV1()

# Manual parsing for VALID_API_KEYS if it's a comma-separated string from env
# This is a common workaround for Pydantic v1 BaseSettings if the env var is not a JSON list.
import os
raw_keys = os.getenv("TENSORUS_VALID_API_KEYS")
if raw_keys:
    settings.VALID_API_KEYS = [key.strip() for key in raw_keys.split(',')]
elif not settings.VALID_API_KEYS: # Ensure it's an empty list if env var is not set and default_factory wasn't used
    settings.VALID_API_KEYS = []
