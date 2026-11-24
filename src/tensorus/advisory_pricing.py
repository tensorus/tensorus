"""Operator-configured pricing scenarios, without metering or provider charges."""
from typing import Annotated
from typing import Literal
from typing import Self
from uuid import UUID
from pydantic import Field
from pydantic import StrictInt
from pydantic import model_validator
from tensorus.catalog_numbers import MAX_ADVISORY_PRICING_RATE
from tensorus.catalog_numbers import advisory_pricing_values_valid
from tensorus.contracts import ReleaseHash
from tensorus.models import Model
FORMAT: Literal["tensorus-advisory-pricing-v1"] = "tensorus-advisory-pricing-v1"
QUOTE_FORMAT: Literal["tensorus-plan-price-scenario-v1"] = "tensorus-plan-price-scenario-v1"
MAX_RESPONSE_BYTES: Literal[8192] = 8192
MAX_EVENTS = 1000
GIB: Literal[1073741824] = 1073741824
MILLION: Literal[1000000] = 1000000
Nonnegative = Annotated[StrictInt, Field(ge=0)]
Rate = Annotated[StrictInt, Field(ge=0, le=MAX_ADVISORY_PRICING_RATE)]
Currency = Annotated[str, Field(strict=True, pattern=r"^[A-Z]{3}$")]
class AdvisoryPricePolicy(Model):
    generation: Annotated[StrictInt, Field(ge=1, le=MAX_EVENTS)]
    currency: Currency | None
    storage_microunits_per_gib_month: Rate | None
    compute_microunits_per_million_units: Rate | None
    configured_at: Annotated[str, Field(strict=True, min_length=1, max_length=40)]

    @model_validator(mode="after")
    def complete_rates(self) -> Self:
        if not advisory_pricing_values_valid(
            self.currency,
            self.storage_microunits_per_gib_month,
            self.compute_microunits_per_million_units,
        ):
            raise ValueError("Advisory pricing values are incomplete")
        return self
class AdvisoryPricingStatus(Model):
    format: Literal["tensorus-advisory-pricing-v1"] = FORMAT
    tenant_id: UUID
    policy: AdvisoryPricePolicy
    price_configuration_shared: Literal[True] = True
    currency_independently_verified: Literal[False] = False
    provider_connected: Literal[False] = False
    charges_created: Literal[False] = False
    physical_usage_measured: Literal[False] = False
class PlanPriceScenario(Model):
    format: Literal["tensorus-plan-price-scenario-v1"] = QUOTE_FORMAT
    tenant_id: UUID
    plan_id: UUID
    plan_hash: ReleaseHash
    observed_at: str
    policy: AdvisoryPricePolicy
    predicted_output_versions: Annotated[StrictInt, Field(ge=0, le=16)]
    predicted_output_bytes: Nonnegative
    estimated_compute: Nonnegative
    storage_microunits_numerator: Nonnegative
    storage_denominator_bytes: Literal[1073741824] = GIB
    compute_microunits_numerator: Nonnegative
    compute_denominator_units: Literal[1000000] = MILLION
    storage_horizon_days: Literal[30] = 30
    price_and_plan_snapshot_atomic: Literal[False] = False
    quota_reserved: Literal[False] = False
    approval_granted: Literal[False] = False
    physical_usage_measured: Literal[False] = False
    provider_connected: Literal[False] = False
    charges_created: Literal[False] = False

    @model_validator(mode="after")
    def exact_scenario(self) -> Self:
        storage = self.policy.storage_microunits_per_gib_month
        compute = self.policy.compute_microunits_per_million_units
        if (
            self.policy.currency is None
            or storage is None
            or compute is None
            or (self.predicted_output_versions == 0 and self.predicted_output_bytes != 0)
            or self.storage_microunits_numerator != self.predicted_output_bytes * storage
            or self.compute_microunits_numerator != self.estimated_compute * compute
        ):
            raise ValueError("Plan price scenario arithmetic is inconsistent")
        return self
class AdvisoryPricingCapabilities(Model):
    format: Literal["tensorus-advisory-pricing-v1"] = FORMAT
    scenario_format: Literal["tensorus-plan-price-scenario-v1"] = QUOTE_FORMAT
    authority: Literal["active_unscoped_tenant_admin"] = "active_unscoped_tenant_admin"
    max_response_bytes: Literal[8192] = MAX_RESPONSE_BYTES
    catalog_policy: Literal[True] = True
    operator_configuration_only: Literal[True] = True
    provider_connected: Literal[False] = False
    charges_created: Literal[False] = False
    physical_usage_measured: Literal[False] = False
