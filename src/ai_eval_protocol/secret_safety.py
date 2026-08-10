from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_eval_protocol.evaluators import SecretResolutionError, SecretResolver


def resolved_secret_values(
    *,
    scenario: dict[str, Any],
    request: dict[str, Any],
    secret_resolver: SecretResolver,
) -> list[str]:
    secret_values: list[str] = []
    references = [request["target"].get("secret_ref", "")]
    metadata = request["target"].get("metadata", {})
    if isinstance(metadata, dict):
        protected_context_refs = metadata.get("protected_context_refs", [])
        if isinstance(protected_context_refs, list):
            references.extend(protected_context_refs)
    references.extend(
        assertion.get("parameters", {}).get("canary_ref", "")
        for assertion in scenario["assertions"]
    )
    for reference in references:
        if not isinstance(reference, str) or not reference:
            continue
        try:
            value = secret_resolver.resolve(reference)
        except SecretResolutionError:
            continue
        if value and value not in secret_values:
            secret_values.append(value)
    return sorted(secret_values, key=len, reverse=True)


def redact_value(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, str):
        redacted = value
        for secret in sorted(set(secrets), key=len, reverse=True):
            redacted = redacted.replace(secret, "[REDACTED_SECRET]")
        return redacted
    if isinstance(value, list):
        return [redact_value(item, secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item, secrets) for item in value)
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            redacted_key = redact_value(key, secrets)
            if redacted_key in redacted:
                raise ValueError("secret redaction produced a duplicate object key")
            redacted[redacted_key] = redact_value(item, secrets)
        return redacted
    return value


def assert_no_secrets(output_dir: Path, secrets: list[str]) -> None:
    for path in output_dir.rglob("*.json"):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"bundle secret scan could not parse: {path.relative_to(output_dir)}"
            ) from exc
        if _contains_secret(document, secrets):
            raise ValueError(
                f"secret scan failed before bundle publication: {path.relative_to(output_dir)}"
            )


def _contains_secret(value: Any, secrets: list[str]) -> bool:
    if isinstance(value, str):
        return any(secret in value for secret in secrets)
    if isinstance(value, (list, tuple)):
        return any(_contains_secret(item, secrets) for item in value)
    if isinstance(value, dict):
        return any(
            _contains_secret(key, secrets) or _contains_secret(item, secrets)
            for key, item in value.items()
        )
    return False
