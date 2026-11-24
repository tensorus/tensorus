"""Advertised scope of opt-in, hash-pinned plan proposer checks."""

from typing import Literal

from tensorus.models import Model

FORMAT: Literal["tensorus-plan-proposer-authority-v1"] = "tensorus-plan-proposer-authority-v1"


class PlanProposerAuthorityCapabilities(Model):
    format: Literal["tensorus-plan-proposer-authority-v1"] = FORMAT
    policy_field: Literal["proposer_authority"] = "proposer_authority"
    tenant_key_role_pinned: Literal[True] = True
    approval_rechecked: Literal[True] = True
    worker_checkpoints_rechecked: Literal[True] = True
    publication_rechecked: Literal[True] = True
    structured_goal_workflow_supported: Literal[True] = True
    guided_request_option: Literal[False] = False
    automatically_selected: Literal[False] = False
    legacy_ordinary_origin_independently_rechecked: Literal[False] = False
    embedded_supported: Literal[False] = False
    fleet_version_verified: Literal[False] = False
    copied_data_revocable: Literal[False] = False
    catalog_authenticated: Literal[False] = False


def matches_plan_schema(schema: object) -> bool:
    if not isinstance(schema, dict):
        return False
    fields = schema.get("properties")
    definitions = schema.get("$defs")
    if not isinstance(fields, dict) or not isinstance(definitions, dict):
        return False
    field = fields.get("proposer_authority")
    authority = definitions.get("PlanProposerAuthority")
    role = definitions.get("Role")
    if not isinstance(field, dict) or not isinstance(authority, dict) or not isinstance(role, dict):
        return False
    properties = authority.get("properties")
    if not isinstance(properties, dict) or set(properties) != {
        "format",
        "tenant_id",
        "key_id",
        "role",
    }:
        return False
    return (
        field.get("anyOf") == [{"$ref": "#/$defs/PlanProposerAuthority"}, {"type": "null"}]
        and "default" in field
        and field["default"] is None
        and authority.get("type") == "object"
        and authority.get("additionalProperties") is False
        and authority.get("required") == ["tenant_id", "key_id", "role"]
        and isinstance(properties["format"], dict)
        and properties["format"].get("const") == FORMAT
        and all(
            isinstance(properties[name], dict)
            and properties[name].get("type") == "string"
            and properties[name].get("format") == "uuid"
            for name in ("tenant_id", "key_id")
        )
        and properties["role"] == {"$ref": "#/$defs/Role"}
        and role.get("enum") == ["reader", "writer", "admin"]
    )
