from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Union, List
from pydantic import field_validator

class Settings(BaseSettings):
    STORAGE_BACKEND: str = "in_memory"
    POSTGRES_HOST: Optional[str] = None
    POSTGRES_PORT: Optional[int] = 5432
    POSTGRES_USER: Optional[str] = None
    POSTGRES_PASSWORD: Optional[str] = None
    POSTGRES_DB: Optional[str] = None
    POSTGRES_DSN: Optional[str] = None

    VALID_API_KEYS: Union[List[str], str] = []
    API_KEY_HEADER_NAME: str = "X-API-KEY"
    AUDIT_LOG_PATH: str = "tensorus_audit.log"

    AUTH_JWT_ENABLED: bool = False
    AUTH_JWT_ISSUER: Optional[str] = None
    AUTH_JWT_AUDIENCE: Optional[str] = None
    AUTH_JWT_ALGORITHM: str = "RS256"
    AUTH_JWT_JWKS_URI: Optional[str] = None
    AUTH_DEV_MODE_ALLOW_DUMMY_JWT: bool = False

    model_config = SettingsConfigDict(env_prefix="TENSORUS_", case_sensitive=False)

    @field_validator("VALID_API_KEYS", mode="before")
    def split_valid_api_keys(cls, v):
        if isinstance(v, str):
            try:
                import json
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except Exception:
                pass
            return [key.strip() for key in v.split(',') if key.strip()]
        return v

settings = Settings()
