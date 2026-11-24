"""Application-level body limits for selected bounded control requests."""

import re
from typing import Literal

MAX_CONTROL_REQUEST_BYTES = 16 * 1024
MAX_PLAN_REVIEW_REQUEST_BYTES = 1024 * 1024
SESSION_REVOCATION_PATH = "/v1/identity/session/revoke"
WORKSPACE_PROVISIONING_PATH = "/v1/tenants/provision"
BROWSER_IDENTITY_CALLBACK_PATH = "/v1/identity/browser-callback"
BROWSER_IDENTITY_COMPLETE_PATH = "/v1/identity/browser-login/complete"
WORKSPACE_INVITATION_ISSUE_PATH = "/v1/operator/workspace-invitations"
WORKSPACE_INVITATION_CLAIM_PATH = "/v1/identity/workspace-invitation/claim"
SERVICE_DISCLOSURE_UPDATE_PATH = "/v1/operator/service-disclosures"
WORKSPACE_LOCK_EXTENDED_PATH = "/v1/workspace/lifecycle/lock-extended"
WORKSPACE_LOCK_RECEIPT_PATH = "/v1/workspace/lifecycle/lock-with-receipt"
ORGANIZATION_CREATE_PATH = "/v1/organizations"
SERVICE_ACTOR_INVITE_PATH = "/v1/workspace/service-actors"
WORKSPACE_MEMBER_INVITATION_PATH: Literal["/v1/workspace/member-invitations"] = (
    "/v1/workspace/member-invitations"
)
WORKSPACE_MEMBER_TRANSITION_PATH: Literal["/v1/workspace/member-transitions"] = (
    "/v1/workspace/member-transitions"
)
WORKSPACE_USAGE_HANDOFF_PATH = "/v1/workspace/usage-handoff-bundle"
WORKSPACE_USAGE_HANDOFF_V2_PATH = "/v1/workspace/usage-handoff-bundle-v2"
LOCAL_USAGE_FIXTURE_REVIEW_PATH = "/v1/workspace/local-usage-fixture/review"
LOCAL_USAGE_FIXTURE_DELIVERY_PATH = "/v1/workspace/local-usage-fixture/deliver"
LOCAL_USAGE_FIXTURE_INSPECTION_PATH = "/v1/workspace/local-usage-fixture/inspect-files"
CONTROL_REQUEST_ROUTES: dict[str, tuple[str, ...]] = {
    "POST": (
        "/v1/tenants",
        "/v1/identity/exchange",
        "/v1/keys",
        "/v1/workspace/members",
        "/v1/workspace/membership/leave",
        "/v1/workspace/membership/leave-with-receipt",
        "/v1/workspace/lifecycle/lock",
        "/v1/agent/grants",
        "/v1/agent/multi-dataset-grants",
        "/v1/retention-holds",
        "/v1/retention-holds/{hold_id}/release",
        "/v1/product-releases/{release_id}/approve",
        "/v1/product-releases/{release_id}/retirement-review",
        "/v1/product-releases/{release_id}/retire",
        "/v1/plans/{plan_id}/approve",
        "/v1/plans/{plan_id}/cancel",
        "/v1/plans/{plan_id}/usage-adjustments",
        "/v1/restore-bundles/name-occupancy",
        "/v1/waitlist",
        "/v1/operator/waitlist/{entry_id}/approve",
        "/v1/operator/waitlist/{entry_id}/decline",
    ),
    "PUT": (
        "/v1/workspace/members/{member_id}",
        "/v1/operator/tenants/{tenant_id}/subscription",
    ),
}


def _route_pattern(routes: tuple[str, ...]) -> re.Pattern[str]:
    def expression(route: str) -> str:
        return "/".join(
            "[^/]+" if segment.startswith("{") and segment.endswith("}") else re.escape(segment)
            for segment in route.split("/")
        )

    return re.compile(r"^(?:" + "|".join(expression(route) for route in routes) + r")/?$")


CONTROL_REQUEST_PATTERNS = {
    method: _route_pattern(routes) for method, routes in CONTROL_REQUEST_ROUTES.items()
}
KEY_ROTATION_PATTERN = re.compile(r"^/v1/keys/[^/]+/rotation/prepare/?$")
IDENTITY_SESSIONS_PATTERN = re.compile(r"^/v1/workspace/identity/sessions/(?:pause|resume)/?$")
WORKSPACE_INVITATION_REVOCATION_PATTERN = re.compile(
    r"^/v1/operator/workspace-invitations/[^/]+/revoke/?$"
)
ORGANIZATION_POST_PATTERN = re.compile(
    r"^/v1/organizations/[^/]+/"
    r"(?:members|workspaces/(?:attach-current|detach-current(?:-with-receipt)?))/?$"
)
ORGANIZATION_MEMBER_PATTERN = re.compile(r"^/v1/organizations/[^/]+/members/[^/]+/?$")
SERVICE_ACTOR_GRANT_PATTERN = re.compile(
    r"^/v1/workspace/service-actors/[^/]+/(?:grants|multi-dataset-grants)/?$"
)
PRODUCT_RETIREMENT_PATTERN = re.compile(r"^/v1/data-products/[^/]+/retirement(?:-review)?/?$")
QUALITY_RELEASE_RETIREMENT_PATTERN = re.compile(
    r"^/v1/releases/[^/]+/(?:retirement-review|retire)/?$"
)


def is_control_request(method: str, path: str) -> bool:
    if method == "PUT" and path.rstrip("/") == SERVICE_DISCLOSURE_UPDATE_PATH:
        return True
    if method == "POST" and path.rstrip("/") in (
        SESSION_REVOCATION_PATH,
        WORKSPACE_PROVISIONING_PATH,
        BROWSER_IDENTITY_CALLBACK_PATH,
        BROWSER_IDENTITY_COMPLETE_PATH,
        WORKSPACE_INVITATION_ISSUE_PATH,
        WORKSPACE_INVITATION_CLAIM_PATH,
        WORKSPACE_LOCK_EXTENDED_PATH,
        WORKSPACE_LOCK_RECEIPT_PATH,
        ORGANIZATION_CREATE_PATH,
        SERVICE_ACTOR_INVITE_PATH,
        WORKSPACE_MEMBER_INVITATION_PATH,
        WORKSPACE_MEMBER_TRANSITION_PATH,
        WORKSPACE_USAGE_HANDOFF_PATH,
        WORKSPACE_USAGE_HANDOFF_V2_PATH,
        LOCAL_USAGE_FIXTURE_REVIEW_PATH,
        LOCAL_USAGE_FIXTURE_DELIVERY_PATH,
        LOCAL_USAGE_FIXTURE_INSPECTION_PATH,
    ):
        return True
    if method == "POST" and ORGANIZATION_POST_PATTERN.fullmatch(path):
        return True
    if method == "POST" and SERVICE_ACTOR_GRANT_PATTERN.fullmatch(path):
        return True
    if method == "PUT" and ORGANIZATION_MEMBER_PATTERN.fullmatch(path):
        return True
    if method == "POST" and KEY_ROTATION_PATTERN.fullmatch(path):
        return True
    if method == "POST" and IDENTITY_SESSIONS_PATTERN.fullmatch(path):
        return True
    if method == "POST" and WORKSPACE_INVITATION_REVOCATION_PATTERN.fullmatch(path):
        return True
    if method == "POST" and PRODUCT_RETIREMENT_PATTERN.fullmatch(path):
        return True
    if method == "POST" and QUALITY_RELEASE_RETIREMENT_PATTERN.fullmatch(path):
        return True
    pattern = CONTROL_REQUEST_PATTERNS.get(method)
    return pattern is not None and pattern.fullmatch(path) is not None
