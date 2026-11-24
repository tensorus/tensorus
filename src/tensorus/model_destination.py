"""Explicit, request-bound destinations for external model disclosure."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import Field, ValidationError, field_validator, model_validator

from tensorus.arrays import canonical_hash
from tensorus.contracts import ReleaseHash
from tensorus.errors import Conflict, Forbidden, TensorusError
from tensorus.models import Model

if TYPE_CHECKING:
    from tensorus.config import Settings

DESTINATION_FORMAT: Literal["tensorus-model-destination-v1"] = "tensorus-model-destination-v1"
CAPABILITY_FORMAT: Literal["tensorus-model-destination-capabilities-v1"] = (
    "tensorus-model-destination-capabilities-v1"
)
TRANSPORT_FORMAT: Literal["tensorus-model-direct-transport-v1"] = (
    "tensorus-model-direct-transport-v1"
)
TRANSPORT_CAPABILITY_FORMAT: Literal["tensorus-model-direct-transport-capabilities-v1"] = (
    "tensorus-model-direct-transport-capabilities-v1"
)


class ModelDestination(Model):
    format: Literal["tensorus-model-destination-v1"] = DESTINATION_FORMAT
    endpoint: Annotated[str, Field(strict=True, min_length=1, max_length=4096)]
    model: Annotated[str, Field(strict=True, min_length=1, max_length=1024)]

    @field_validator("endpoint")
    @classmethod
    def valid_endpoint(cls, value: str) -> str:
        from tensorus.credentials import validate_endpoint

        if (
            not value.isprintable()
            or validate_endpoint(value) != value
            or not value.endswith("/chat/completions")
        ):
            raise ValueError("Model destination must be an exact Chat Completions endpoint")
        return value

    @field_validator("model")
    @classmethod
    def printable_model(cls, value: str) -> str:
        if not value.isprintable() or value != value.strip():
            raise ValueError("Model destination name must be printable and trimmed")
        return value


class ModelDestinationCapabilities(Model):
    format: Literal["tensorus-model-destination-capabilities-v1"] = CAPABILITY_FORMAT
    destination: ModelDestination
    destination_sha256: ReleaseHash
    pin_required: Literal[True] = True
    scoped_agents_permitted: Literal[False] = False
    provider_identity_verified: Literal[False] = False

    @field_validator(
        "pin_required", "scoped_agents_permitted", "provider_identity_verified", mode="before"
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Model destination flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_destination(self) -> Self:
        if canonical_hash(self.destination.model_dump(mode="json")) != self.destination_sha256:
            raise ValueError("Model destination hash is inconsistent")
        return self


class DirectTransportPolicy(Model):
    format: Literal["tensorus-model-direct-transport-v1"] = TRANSPORT_FORMAT
    destination_sha256: ReleaseHash
    covered_calls: Literal["goal_planning_and_investigation"] = "goal_planning_and_investigation"
    environment_proxies_used: Literal[False] = False
    environment_ca_overrides_used: Literal[False] = False
    redirects_followed: Literal[False] = False
    https_certificate_verification_enabled: Literal[True] = True
    dns_and_transparent_routing_pinned: Literal[False] = False
    provider_identity_verified: Literal[False] = False
    network_egress_isolated: Literal[False] = False
    sent_bytes_revocable: Literal[False] = False
    other_outbound_requests_covered: Literal[False] = False
    fleet_configuration_verified: Literal[False] = False

    @field_validator(
        "environment_proxies_used",
        "environment_ca_overrides_used",
        "redirects_followed",
        "https_certificate_verification_enabled",
        "dns_and_transparent_routing_pinned",
        "provider_identity_verified",
        "network_egress_isolated",
        "sent_bytes_revocable",
        "other_outbound_requests_covered",
        "fleet_configuration_verified",
        mode="before",
    )
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Model transport flags must be booleans")
        return value


class DirectTransportCapabilities(Model):
    format: Literal["tensorus-model-direct-transport-capabilities-v1"] = TRANSPORT_CAPABILITY_FORMAT
    policy: DirectTransportPolicy
    policy_sha256: ReleaseHash
    policy_pin_required: Literal[True] = True
    scoped_agents_permitted: Literal[False] = False

    @field_validator("policy_pin_required", "scoped_agents_permitted", mode="before")
    @classmethod
    def exact_flags(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("Model transport capability flags must be booleans")
        return value

    @model_validator(mode="after")
    def consistent_policy(self) -> Self:
        if canonical_hash(self.policy.model_dump(mode="json")) != self.policy_sha256:
            raise ValueError("Model transport policy hash is inconsistent")
        return self


@dataclass(frozen=True)
class DestinationPin:
    endpoint: str
    model: str
    sha256: str
    transport_sha256: str | None = None


def configured_destination(settings: "Settings") -> ModelDestination:
    if not settings.llm_base_url or not settings.llm_model:
        raise TensorusError(
            "LLM planning is not configured", code="planner_unavailable", status=503
        )
    try:
        return ModelDestination(
            endpoint=f"{settings.llm_base_url.rstrip('/')}/chat/completions",
            model=settings.llm_model,
        )
    except (ValidationError, TypeError, ValueError) as exc:
        raise TensorusError(
            "The configured model destination is invalid", code="planner_unavailable", status=503
        ) from exc


def capabilities(settings: "Settings") -> ModelDestinationCapabilities:
    destination = configured_destination(settings)
    return ModelDestinationCapabilities(
        destination=destination,
        destination_sha256=canonical_hash(destination.model_dump(mode="json")),
    )


def direct_transport_capabilities(settings: "Settings") -> DirectTransportCapabilities:
    if not settings.llm_direct_transport or not settings.llm_require_destination_pin:
        raise Conflict("Direct model transport is not enabled with destination pinning")
    destination = configured_destination(settings)
    policy = DirectTransportPolicy(
        destination_sha256=canonical_hash(destination.model_dump(mode="json"))
    )
    return DirectTransportCapabilities(
        policy=policy,
        policy_sha256=canonical_hash(policy.model_dump(mode="json")),
    )


def prepare(
    settings: "Settings",
    expected_sha256: str | None,
    expected_transport_sha256: str | None = None,
) -> DestinationPin | None:
    if settings.llm_direct_transport and not settings.llm_require_destination_pin:
        raise Conflict("Direct model transport requires destination pinning")
    if expected_transport_sha256 is not None and not settings.llm_direct_transport:
        raise Conflict("The reviewed direct model transport is no longer enabled")
    if settings.llm_direct_transport and expected_transport_sha256 is None:
        raise Forbidden("Review and pin the direct model transport before disclosure")
    if expected_sha256 is None:
        if settings.llm_require_destination_pin:
            raise Forbidden("Review and pin the configured model destination before disclosure")
        return None
    destination = configured_destination(settings)
    digest = canonical_hash(destination.model_dump(mode="json"))
    if digest != expected_sha256:
        raise Conflict("The configured model destination changed; review it before disclosure")
    transport_sha256 = None
    if settings.llm_direct_transport:
        transport = direct_transport_capabilities(settings)
        if transport.policy.destination_sha256 != digest or transport.policy_sha256 != (
            expected_transport_sha256
        ):
            raise Conflict("The direct model transport changed; review it before disclosure")
        transport_sha256 = transport.policy_sha256
    return DestinationPin(destination.endpoint, destination.model, digest, transport_sha256)


def recheck(settings: "Settings", pin: DestinationPin | None) -> None:
    if settings.llm_direct_transport and not settings.llm_require_destination_pin:
        raise Conflict("Direct model transport requires destination pinning")
    if pin is None:
        if settings.llm_require_destination_pin or settings.llm_direct_transport:
            raise Forbidden("Review and pin the configured model destination before disclosure")
        return
    destination = configured_destination(settings)
    if (
        destination.endpoint != pin.endpoint
        or destination.model != pin.model
        or canonical_hash(destination.model_dump(mode="json")) != pin.sha256
    ):
        raise Conflict("The configured model destination changed during disclosure")
    if settings.llm_direct_transport:
        transport = direct_transport_capabilities(settings)
        if pin.transport_sha256 is None or transport.policy_sha256 != pin.transport_sha256:
            raise Conflict("The direct model transport changed during disclosure")
    elif pin.transport_sha256 is not None:
        raise Conflict("The reviewed direct model transport changed during disclosure")


def uses_environment(pin: DestinationPin | None) -> bool:
    return pin is None or pin.transport_sha256 is None
