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

    # Enhanced API Authentication (Bearer Token Style)
    AUTH_ENABLED: bool = True
    API_KEYS: str = ""  # Comma-separated API keys
    VALID_API_KEYS: Union[List[str], str] = []  # Backward compatibility
    API_KEY_HEADER_NAME: str = "Authorization"  # Standard Bearer token header
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
    
    @field_validator("API_KEYS", mode="before")
    def validate_api_keys(cls, v):
        """Validate and format API keys from environment"""
        if not v:
            return ""
        
        # Import here to avoid circular imports
        try:
            from tensorus.auth.key_generator import TensorusAPIKey
            
            # Split and validate each key
            keys = [key.strip() for key in v.split(',') if key.strip()]
            valid_keys = []
            
            for key in keys:
                if TensorusAPIKey.validate_format(key):
                    valid_keys.append(key)
                else:
                    # Log warning but don't fail - allows gradual migration
                    import logging
                    logging.warning(f"Invalid API key format detected: {TensorusAPIKey.mask_key(key) if key else 'empty'}")
                    valid_keys.append(key)  # Keep for backward compatibility
            
            return ','.join(valid_keys)
        except ImportError:
            # Fallback if auth module not available yet
            return v
    
    @property
    def valid_api_keys(self) -> List[str]:
        """Get list of valid API keys from both new and legacy sources"""
        keys = []
        
        # Primary source: new API_KEYS field
        if self.API_KEYS:
            keys.extend([key.strip() for key in self.API_KEYS.split(',') if key.strip()])
        
        # Fallback: legacy VALID_API_KEYS field
        if self.VALID_API_KEYS:
            if isinstance(self.VALID_API_KEYS, list):
                keys.extend(self.VALID_API_KEYS)
            elif isinstance(self.VALID_API_KEYS, str):
                keys.extend([key.strip() for key in self.VALID_API_KEYS.split(',') if key.strip()])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keys = []
        for key in keys:
            if key not in seen:
                seen.add(key)
                unique_keys.append(key)
        
        return unique_keys

settings = Settings()
