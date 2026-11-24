"""Shared generation and validation of opaque Tensorus credentials and endpoints."""

import hashlib
import re
import secrets
from datetime import UTC, datetime
from urllib.parse import urlsplit
from uuid import uuid4


def uid() -> str:
    return str(uuid4())


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def new_token() -> str:
    return f"tns_{secrets.token_urlsafe(32)}"


def valid_generated_token(token: str) -> bool:
    return re.fullmatch(r"tns_[A-Za-z0-9_-]{42}[AEIMQUYcgkosw048]", token) is not None


def validate_endpoint(value: str) -> str:
    if any(
        character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ValueError("Endpoint URLs must not contain whitespace or control characters")
    try:
        parsed = urlsplit(value)
        host, port = parsed.hostname, parsed.port
    except ValueError as exc:
        raise ValueError("Invalid endpoint URL") from exc
    if port == 0:
        raise ValueError("Endpoint ports must be in 1..65535")
    if not host or parsed.username is not None or parsed.password is not None:
        raise ValueError("Endpoint URLs require a hostname and must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("Endpoint URLs must not contain query strings or fragments")
    if parsed.scheme == "https" or (
        parsed.scheme == "http" and host in {"localhost", "127.0.0.1", "::1"} and port is not None
    ):
        return value.rstrip("/")
    raise ValueError("Use HTTPS, or an explicit loopback development port")
