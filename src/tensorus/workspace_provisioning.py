"""Retry-safe bootstrap provisioning using a client-held initial credential."""
from datetime import datetime
from datetime import timedelta
from typing import Annotated
from typing import Literal
from uuid import UUID
from uuid import uuid5
from pydantic import Field
from pydantic import SecretStr
from pydantic import StrictInt
from pydantic import field_validator
from tensorus.credentials import valid_generated_token
from tensorus.models import Model
from tensorus.models import TenantCreate
MAX_RESPONSE_BYTES = 4096
def initial_key_id(tenant_id: UUID) -> UUID:
    return uuid5(tenant_id, "tensorus-workspace-initial-administrator-v1")
class WorkspaceProvisionRequest(TenantCreate):
    initial_api_key: SecretStr
    expires_at: Annotated[StrictInt, Field(ge=1, le=2**53 - 1)]

    @field_validator("initial_api_key")
    @classmethod
    def generated_credential(cls, value: SecretStr) -> SecretStr:
        if not valid_generated_token(value.get_secret_value()):
            raise ValueError("The initial credential must be a generated Tensorus key")
        return value
class WorkspaceProvisionReceipt(Model):
    format: Literal["tensorus-workspace-provisioning-v1"]
    id: UUID
    name: str
    title: str
    created_at: str
    key_id: UUID
    expires_at: Annotated[StrictInt, Field(ge=1, le=2**53 - 1)]

    @field_validator("created_at")
    @classmethod
    def canonical_timestamp(cls, value: str) -> str:
        timestamp = datetime.fromisoformat(value)
        if (
            timestamp.utcoffset() != timedelta(0)
            or timestamp.isoformat(timespec="microseconds") != value
        ):
            raise ValueError("The workspace creation timestamp must be canonical UTC")
        return value
