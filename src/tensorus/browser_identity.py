"""Bounded public-client OIDC code exchange for the browser console."""
import re
from typing import Annotated
from typing import Literal
from uuid import UUID
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationInfo
from pydantic import field_validator
from tensorus.models import Model
from tensorus.models import Name
FORMAT: Literal["tensorus-browser-oidc-login-v1"] = "tensorus-browser-oidc-login-v1"
MAX_LOGIN_RESPONSE_BYTES = 16 * 1024
MAX_CODE_LENGTH = 4096
class BrowserLoginConfiguration(Model):
    format: Literal["tensorus-browser-oidc-login-v1"] = FORMAT
    issuer: str
    client_id: str
    authorization_endpoint: str
    redirect_uri: str
    response_mode: Literal["form_post"] = "form_post"
    policy_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
class BrowserLoginComplete(Model):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, hide_input_in_errors=True)

    workspace: Name
    code: Annotated[str, Field(strict=True, min_length=1, max_length=MAX_CODE_LENGTH, repr=False)]
    code_verifier: Annotated[str, Field(strict=True, min_length=43, max_length=128, repr=False)]
    nonce: Annotated[str, Field(strict=True, min_length=43, max_length=43, repr=False)]
    policy_hash: Annotated[str, Field(strict=True, min_length=64, max_length=64)]
    expected_membership_id: UUID | None = None

    @field_validator("code", "code_verifier", "nonce", "policy_hash")
    @classmethod
    def validate_oauth_values(cls, value: str, info: ValidationInfo) -> str:
        patterns = {
            "code": r"[!-~]{1,4096}",
            "code_verifier": r"[A-Za-z0-9._~-]{43,128}",
            "nonce": r"[A-Za-z0-9_-]{43}",
            "policy_hash": r"[0-9a-f]{64}",
        }
        if info.field_name is None or not re.fullmatch(patterns[info.field_name], value):
            raise ValueError("Invalid browser login value")
        return value
