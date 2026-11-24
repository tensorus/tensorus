"""Exact integer aggregates and policy bounds shared across catalog and SDK models."""

from decimal import Decimal

MAX_ADVISORY_PRICING_RATE = 1_000_000_000_000
MAX_SERVICE_DISCLOSURE_EVENTS = 1000


def advisory_pricing_values_valid(currency: object, storage: object, compute: object) -> bool:
    return (currency is None and storage is None and compute is None) or (
        isinstance(currency, str)
        and len(currency) == 3
        and all("A" <= character <= "Z" for character in currency)
        and type(storage) is int
        and 0 <= storage <= MAX_ADVISORY_PRICING_RATE
        and type(compute) is int
        and 0 <= compute <= MAX_ADVISORY_PRICING_RATE
    )


def tenant_http_limits_valid(rate: object, burst: object) -> bool:
    return (rate is None and burst is None) or (
        type(rate) is int
        and type(burst) is int
        and 1 <= rate <= 100_000
        and 1 <= burst <= 1_000_000
    )


def nonnegative_integer(value: object, *, maximum: int | None = None) -> int:
    if type(value) is int:
        if value >= 0 and (maximum is None or value <= maximum):
            return value
    elif (
        isinstance(value, Decimal)
        and value.is_finite()
        and value >= 0
        and (maximum is None or value <= maximum)
        and value == value.to_integral_value()
    ):
        return int(value)
    raise ValueError("Catalog aggregate is not an exact nonnegative integer")
