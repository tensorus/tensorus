"""Optional post-handler credential checks for authenticated HTTP GETs."""
from typing import Literal
from pydantic import field_validator
from tensorus.models import Model
class AuthenticatedReadFenceCapabilities(Model):
    format: Literal["tensorus-authenticated-get-caller-fence-v1"] = (
        "tensorus-authenticated-get-caller-fence-v1"
    )
    method: Literal["GET"] = "GET"
    coverage: Literal["successful_routes_using_tenant_principal_dependency"] = (
        "successful_routes_using_tenant_principal_dependency"
    )
    nonstreaming_response_serialized_before_check: Literal[True] = True
    caller_rechecked_before_response_start: Literal[True] = True
    cancelled_checks_hold_request_admission: Literal[True] = True
    catalog_recheck_deadline_enforced: Literal[False] = False
    stream_payloads_rechecked_by_fence: Literal[False] = False
    resource_authority_recomputed: Literal[False] = False
    other_authentication_modes_covered: Literal[False] = False
    non_get_mutations_covered: Literal[False] = False
    get_side_effects_rolled_back: Literal[False] = False
    response_delivery_atomic: Literal[False] = False

    @field_validator(
        "nonstreaming_response_serialized_before_check",
        "caller_rechecked_before_response_start",
        "cancelled_checks_hold_request_admission",
        "catalog_recheck_deadline_enforced",
        "stream_payloads_rechecked_by_fence",
        "resource_authority_recomputed",
        "other_authentication_modes_covered",
        "non_get_mutations_covered",
        "get_side_effects_rolled_back",
        "response_delivery_atomic",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Authenticated GET response fence flags must be booleans")
        return value
