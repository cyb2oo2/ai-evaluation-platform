from __future__ import annotations

import ipaddress
import math
import socket
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

NetworkAccess = Literal["disabled", "loopback", "allowlist"]


class HttpPolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class HttpTargetPolicy:
    network_access: NetworkAccess = "disabled"
    allowed_origins: tuple[str, ...] = ()
    max_timeout_seconds: float = 60.0
    max_request_bytes: int = 256_000
    max_response_bytes: int = 1_000_000
    max_cost_usd: float = 1.0
    input_cost_per_million_tokens: float | None = None
    output_cost_per_million_tokens: float | None = None

    def validate(self) -> None:
        if self.network_access not in {"disabled", "loopback", "allowlist"}:
            raise HttpPolicyError("network_access must be disabled, loopback, or allowlist")
        if self.network_access == "allowlist" and not self.allowed_origins:
            raise HttpPolicyError("allowlist network access requires at least one allowed origin")
        _finite_number(
            self.max_timeout_seconds,
            label="max_timeout_seconds",
            minimum=0,
            exclusive_minimum=True,
        )
        _positive_integer(self.max_request_bytes, label="max_request_bytes")
        _positive_integer(self.max_response_bytes, label="max_response_bytes")
        _finite_number(self.max_cost_usd, label="max_cost_usd", minimum=0)
        for label, value in (
            ("input_cost_per_million_tokens", self.input_cost_per_million_tokens),
            ("output_cost_per_million_tokens", self.output_cost_per_million_tokens),
        ):
            if value is not None:
                _finite_number(value, label=label, minimum=0)

    def authorize_endpoint(self, endpoint: str) -> None:
        self.validate()
        parsed = urlsplit(endpoint)
        if self.network_access == "disabled":
            raise HttpPolicyError("HTTP target network access is disabled")
        if parsed.scheme not in {"http", "https"}:
            raise HttpPolicyError("HTTP target endpoint must use http or https")
        if parsed.username or parsed.password:
            raise HttpPolicyError("HTTP target endpoint must not contain user information")
        if parsed.fragment:
            raise HttpPolicyError("HTTP target endpoint must not contain a fragment")
        if parsed.query:
            raise HttpPolicyError("HTTP target endpoint must not contain a query")
        if parsed.path.rstrip("/") != "/v1/chat/completions":
            raise HttpPolicyError("HTTP target endpoint path must be /v1/chat/completions")
        host = parsed.hostname
        if not host:
            raise HttpPolicyError("HTTP target endpoint must contain a hostname")

        addresses = _resolve_addresses(host, parsed.port or _default_port(parsed.scheme))
        if self.network_access == "loopback":
            if not addresses or not all(address.is_loopback for address in addresses):
                raise HttpPolicyError("loopback policy only permits loopback addresses")
            return

        if parsed.scheme != "https":
            raise HttpPolicyError("allowlist policy requires https")
        origin = _origin(parsed.scheme, host, parsed.port)
        allowed = {_normalized_origin(value) for value in self.allowed_origins}
        if origin not in allowed:
            raise HttpPolicyError(f"HTTP target origin '{origin}' is not allowed")
        if not addresses or not all(address.is_global for address in addresses):
            raise HttpPolicyError(
                "allowlisted HTTP targets must resolve only to public addresses"
            )

    def effective_timeout(self, requested_seconds: float) -> float:
        self.validate()
        _finite_number(
            requested_seconds,
            label="requested timeout",
            minimum=0,
            exclusive_minimum=True,
        )
        return min(requested_seconds, self.max_timeout_seconds)

    def effective_response_limit(self, requested_bytes: int) -> int:
        self.validate()
        _positive_integer(requested_bytes, label="requested response byte limit")
        return min(requested_bytes, self.max_response_bytes)

    def effective_cost_limit(self, requested_usd: float) -> float:
        self.validate()
        _finite_number(requested_usd, label="requested cost limit", minimum=0)
        return min(requested_usd, self.max_cost_usd)

    def estimate_cost(self, *, prompt_tokens: int, completion_tokens: int) -> float:
        self.validate()
        if (
            self.input_cost_per_million_tokens is None
            or self.output_cost_per_million_tokens is None
        ):
            raise HttpPolicyError(
                "cost enforcement requires input and output token prices"
            )
        _non_negative_integer(prompt_tokens, label="prompt_tokens")
        _non_negative_integer(completion_tokens, label="completion_tokens")
        cost = (
            prompt_tokens * self.input_cost_per_million_tokens
            + completion_tokens * self.output_cost_per_million_tokens
        ) / 1_000_000
        return round(cost, 10)


def _resolve_addresses(
    host: str, port: int
) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    try:
        records = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise HttpPolicyError(f"HTTP target hostname resolution failed: {exc}") from exc
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for record in records:
        raw_address = record[4][0]
        address = ipaddress.ip_address(raw_address)
        if address not in addresses:
            addresses.append(address)
    return tuple(addresses)


def _default_port(scheme: str) -> int:
    return 443 if scheme == "https" else 80


def _origin(scheme: str, host: str, port: int | None) -> str:
    default_port = _default_port(scheme)
    port_suffix = "" if port in {None, default_port} else f":{port}"
    return f"{scheme.lower()}://{host.lower().rstrip('.')}{port_suffix}"


def _normalized_origin(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise HttpPolicyError(f"allowed origin must not contain a path or query: {value}")
    if parsed.scheme != "https" or not parsed.hostname:
        raise HttpPolicyError(f"allowed origin must be an https origin: {value}")
    return _origin(parsed.scheme, parsed.hostname, parsed.port)


def _finite_number(
    value: object,
    *,
    label: str,
    minimum: float,
    exclusive_minimum: bool = False,
) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < minimum
        or (exclusive_minimum and value == minimum)
    ):
        comparator = "greater than" if exclusive_minimum else "at least"
        raise HttpPolicyError(f"{label} must be a finite number {comparator} {minimum:g}")


def _positive_integer(value: object, *, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise HttpPolicyError(f"{label} must be a positive integer")


def _non_negative_integer(value: object, *, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise HttpPolicyError(f"{label} must be a non-negative integer")
