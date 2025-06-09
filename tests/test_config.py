import pytest
import os
from typing import List

from tensorus.config import SettingsV1, settings as global_settings
from tensorus.metadata import get_configured_storage_instance, InMemoryStorage, PostgresMetadataStorage, ConfigurationError
from tensorus.metadata.storage_abc import MetadataStorage

# --- Test SettingsV1 Loading ---

def test_settings_default_values():
    # Test without any env vars set (relies on defaults in SettingsV1)
    # This test is tricky because settings are loaded globally at import time.
    # We need to ensure a clean environment or test specific fields not typically overridden by CI.
    # For now, let's assume a default state or test fields that are less likely to be in CI env.
    s = SettingsV1() # Creates a new instance, re-evaluating defaults if env vars are not set for these specific fields
    assert s.STORAGE_BACKEND == "in_memory" # Default
    assert s.API_KEY_HEADER_NAME == "X-API-KEY" # Default
    assert s.VALID_API_KEYS == [] # Default (before manual parsing from env)
    assert s.POSTGRES_PORT == 5432 # Default

def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("TENSORUS_STORAGE_BACKEND", "postgres_test")
    monkeypatch.setenv("TENSORUS_API_KEY_HEADER_NAME", "X-CUSTOM-API-KEY")
    monkeypatch.setenv("TENSORUS_VALID_API_KEYS", '["key_env1", "key_env2"]')
    monkeypatch.setenv("TENSORUS_POSTGRES_HOST", "testhost")
    monkeypatch.setenv("TENSORUS_POSTGRES_PORT", "1234")

    # Create a new SettingsV1 instance to load these monkeypatched env vars
    # Pydantic v1 BaseSettings loads from env at initialization.
    s = SettingsV1()

    assert s.STORAGE_BACKEND == "postgres_test"
    assert s.API_KEY_HEADER_NAME == "X-CUSTOM-API-KEY"
    assert s.VALID_API_KEYS == ["key_env1", "key_env2"]
    assert s.POSTGRES_HOST == "testhost"
    assert s.POSTGRES_PORT == 1234

def test_valid_api_keys_parsing_empty_env(monkeypatch):
    # Ensure if TENSORUS_VALID_API_KEYS is not set, it defaults to empty list
    monkeypatch.delenv("TENSORUS_VALID_API_KEYS", raising=False)

    # global_settings is already loaded. We need to simulate its re-initialization or check its state.
    # The manual parsing block in config.py affects the global `settings` instance.
    # To test this properly, we might need to re-run that logic or use a new settings instance.

    # Create a new instance to test its initialization behavior
    s = SettingsV1()

    assert s.VALID_API_KEYS == []

def test_valid_api_keys_json_list_in_env(monkeypatch):
    monkeypatch.setenv("TENSORUS_VALID_API_KEYS", '["json_key1", "json_key2"]')
    s = SettingsV1()
    assert s.VALID_API_KEYS == ["json_key1", "json_key2"]


# --- Test Dynamic Storage Instantiation ---

@pytest.fixture(autouse=True)
def reset_global_settings_after_test(monkeypatch):
    """ Ensures each test involving settings gets a fresh state relative to env vars,
        and attempts to reset global settings to avoid test interference.
        This is complex due to Python's import-time evaluation.
    """
    original_env = os.environ.copy()
    # Store original values of global_settings that might be changed by manual parsing in config.py
    original_valid_keys = global_settings.VALID_API_KEYS

    yield

    os.environ.clear()
    os.environ.update(original_env)
    # Attempt to restore global_settings to a semblance of its original state
    # This is hard because `settings = SettingsV1()` in config.py runs only once.
    # A better way would be a settings fixture that provides a fresh SettingsV1 instance.
    global_settings.VALID_API_KEYS = original_valid_keys
    # Update the global settings list to reflect the restored environment
    raw_keys = os.getenv("TENSORUS_VALID_API_KEYS")
    if raw_keys:
        global_settings.VALID_API_KEYS = [key.strip() for key in raw_keys.split(',')]
    else:
        global_settings.VALID_API_KEYS = []


def test_get_in_memory_storage_by_default(monkeypatch):
    monkeypatch.delenv("TENSORUS_STORAGE_BACKEND", raising=False)
    # Temporarily modify global_settings for this test's scope
    monkeypatch.setattr(global_settings, 'STORAGE_BACKEND', "in_memory")

    storage = get_configured_storage_instance()
    assert isinstance(storage, InMemoryStorage)

def test_get_in_memory_storage_explicitly(monkeypatch):
    monkeypatch.setenv("TENSORUS_STORAGE_BACKEND", "in_memory")
    # Update global_settings to reflect the monkeypatched env var for this test
    monkeypatch.setattr(global_settings, 'STORAGE_BACKEND', "in_memory")

    storage = get_configured_storage_instance()
    assert isinstance(storage, InMemoryStorage)

def test_get_postgres_storage_with_dsn(monkeypatch):
    dsn = "postgresql://testuser:testpass@testhost:5432/testdb"
    monkeypatch.setenv("TENSORUS_STORAGE_BACKEND", "postgres")
    monkeypatch.setenv("TENSORUS_POSTGRES_DSN", dsn)
    # Update global_settings
    monkeypatch.setattr(global_settings, 'STORAGE_BACKEND', "postgres")
    monkeypatch.setattr(global_settings, 'POSTGRES_DSN', dsn)
    monkeypatch.setattr(global_settings, 'POSTGRES_HOST', None) # Ensure DSN is chosen

    # Mock psycopg2.pool.SimpleConnectionPool to avoid actual connection
    class MockPool:
        def __init__(self, minconn, maxconn, dsn):
            assert dsn == "postgresql://testuser:testpass@testhost:5432/testdb"
        def getconn(self): pass
        def putconn(self, conn): pass
        def closeall(self): pass

    monkeypatch.setattr("psycopg2.pool.SimpleConnectionPool", MockPool)

    storage = get_configured_storage_instance()
    assert isinstance(storage, PostgresMetadataStorage)
    assert storage.dsn == dsn

def test_get_postgres_storage_with_params(monkeypatch):
    monkeypatch.setenv("TENSORUS_STORAGE_BACKEND", "postgres")
    monkeypatch.setenv("TENSORUS_POSTGRES_HOST", "localhost")
    monkeypatch.setenv("TENSORUS_POSTGRES_USER", "tensorus_user")
    monkeypatch.setenv("TENSORUS_POSTGRES_DB", "tensorus_db")
    monkeypatch.setenv("TENSORUS_POSTGRES_PASSWORD", "securepassword") # Optional but good to test
    monkeypatch.setenv("TENSORUS_POSTGRES_PORT", "5433")

    # Update global_settings
    monkeypatch.setattr(global_settings, 'STORAGE_BACKEND', "postgres")
    monkeypatch.setattr(global_settings, 'POSTGRES_HOST', "localhost")
    monkeypatch.setattr(global_settings, 'POSTGRES_USER', "tensorus_user")
    monkeypatch.setattr(global_settings, 'POSTGRES_DB', "tensorus_db")
    monkeypatch.setattr(global_settings, 'POSTGRES_PASSWORD', "securepassword")
    monkeypatch.setattr(global_settings, 'POSTGRES_PORT', 5433)
    monkeypatch.setattr(global_settings, 'POSTGRES_DSN', None) # Ensure params are chosen


    class MockPoolKwargs:
        def __init__(self, minconn, maxconn, host, port, user, password, database):
            assert host == "localhost"
            assert port == 5433
            assert user == "tensorus_user"
            assert password == "securepassword"
            assert database == "tensorus_db"
        def getconn(self): pass
        def putconn(self, conn): pass
        def closeall(self): pass

    monkeypatch.setattr("psycopg2.pool.SimpleConnectionPool", MockPoolKwargs)

    storage = get_configured_storage_instance()
    assert isinstance(storage, PostgresMetadataStorage)

def test_postgres_storage_missing_config_dsn_and_params(monkeypatch):
    monkeypatch.setenv("TENSORUS_STORAGE_BACKEND", "postgres")
    # Ensure all relevant PG env vars are unset or None in global_settings
    monkeypatch.delenv("TENSORUS_POSTGRES_DSN", raising=False)
    monkeypatch.delenv("TENSORUS_POSTGRES_HOST", raising=False)
    monkeypatch.delenv("TENSORUS_POSTGRES_USER", raising=False)
    monkeypatch.delenv("TENSORUS_POSTGRES_DB", raising=False)

    monkeypatch.setattr(global_settings, 'STORAGE_BACKEND', "postgres")
    monkeypatch.setattr(global_settings, 'POSTGRES_DSN', None)
    monkeypatch.setattr(global_settings, 'POSTGRES_HOST', None)
    monkeypatch.setattr(global_settings, 'POSTGRES_USER', None)
    monkeypatch.setattr(global_settings, 'POSTGRES_DB', None)

    with pytest.raises(ConfigurationError, match="PostgreSQL backend selected, but required connection details"):
        get_configured_storage_instance()

def test_postgres_storage_missing_some_params(monkeypatch):
    monkeypatch.setenv("TENSORUS_STORAGE_BACKEND", "postgres")
    monkeypatch.setenv("TENSORUS_POSTGRES_HOST", "localhost") # User and DB missing

    monkeypatch.setattr(global_settings, 'STORAGE_BACKEND', "postgres")
    monkeypatch.setattr(global_settings, 'POSTGRES_HOST', "localhost")
    monkeypatch.setattr(global_settings, 'POSTGRES_USER', None)
    monkeypatch.setattr(global_settings, 'POSTGRES_DB', None)
    monkeypatch.setattr(global_settings, 'POSTGRES_DSN', None)

    with pytest.raises(ConfigurationError, match="PostgreSQL backend selected, but required connection details"):
        get_configured_storage_instance()

def test_unsupported_storage_backend(monkeypatch):
    monkeypatch.setenv("TENSORUS_STORAGE_BACKEND", "mongodb")
    monkeypatch.setattr(global_settings, 'STORAGE_BACKEND', "mongodb")

    with pytest.raises(ConfigurationError, match="Unsupported storage backend: mongodb"):
        get_configured_storage_instance()

# Note: psycopg2 is not actually imported or used if mocked correctly.
# These tests verify the configuration logic that *would* pass params to psycopg2.
