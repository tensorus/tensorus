"""Canonical client-held secrets for short-lived receipt observations."""
import base64
import re
from tensorus.errors import TensorusError
def validate_receipt_secret(secret: str, prefix: str, label: str) -> str:
    message = f"A canonical {label} receipt secret is required"
    if (
        not isinstance(secret, str)
        or re.fullmatch(rf"{re.escape(prefix)}[A-Za-z0-9_-]{{43}}", secret) is None
    ):
        raise TensorusError(message)
    encoded = secret[len(prefix) :]
    try:
        decoded = base64.urlsafe_b64decode(encoded + "=")
    except ValueError as exc:
        raise TensorusError(message) from exc
    if len(decoded) != 32 or base64.urlsafe_b64encode(decoded).decode().rstrip("=") != encoded:
        raise TensorusError(message)
    return secret
