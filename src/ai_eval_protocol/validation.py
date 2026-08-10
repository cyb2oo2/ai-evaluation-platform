from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

from ai_eval_protocol.models import PROTOCOL_VERSION

_DOCUMENT_TYPES = {"scenario", "trial_request", "trial_trace", "evaluation_summary"}
_EVALUATOR_KINDS = {"deterministic", "probabilistic", "environment"}
_ASSERTION_STATUSES = {"pass", "fail", "error", "not_applicable"}
_TRIAL_STATUSES = {"completed", "error", "cancelled", "budget_exceeded"}
_EVENT_TYPES = {
    "message",
    "retrieval",
    "tool_call",
    "tool_result",
    "environment",
    "evaluator",
    "error",
}

_SCENARIO_FIELDS = {
    "protocol_version",
    "document_type",
    "scenario_id",
    "version",
    "title",
    "description",
    "risks",
    "target_requirements",
    "fixtures",
    "stimulus",
    "assertions",
    "controls",
}
_TRIAL_REQUEST_FIELDS = {
    "protocol_version",
    "document_type",
    "request_id",
    "version",
    "trial_id",
    "target",
    "scenario_ref",
    "trial_index",
    "seed",
    "limits",
}
_TRIAL_TRACE_FIELDS = {
    "protocol_version",
    "document_type",
    "trial_id",
    "version",
    "request_ref",
    "scenario_ref",
    "target",
    "started_at",
    "completed_at",
    "status",
    "events",
    "artifacts",
    "assertion_results",
}
_SUMMARY_FIELDS = {
    "protocol_version",
    "document_type",
    "evaluation_id",
    "scenario_ref",
    "target",
    "trial_refs",
    "metrics",
    "generated_at",
}


class ProtocolValidationError(ValueError):
    def __init__(self, errors: Iterable[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


def validate_document(document: dict[str, Any]) -> None:
    errors: list[str] = []
    _required_string(document, "protocol_version", "$", errors)
    _required_string(document, "document_type", "$", errors)

    if document.get("protocol_version") != PROTOCOL_VERSION:
        errors.append(f"$.protocol_version must be '{PROTOCOL_VERSION}'")

    document_type = document.get("document_type")
    if document_type not in _DOCUMENT_TYPES:
        errors.append(f"$.document_type must be one of {sorted(_DOCUMENT_TYPES)}")
    elif document_type == "scenario":
        _validate_scenario(document, errors)
    elif document_type == "trial_request":
        _validate_trial_request(document, errors)
    elif document_type == "trial_trace":
        _validate_trial_trace(document, errors)
    elif document_type == "evaluation_summary":
        _validate_evaluation_summary(document, errors)

    _reject_inline_credentials(document, "$", errors)

    if errors:
        raise ProtocolValidationError(errors)


def _validate_scenario(document: dict[str, Any], errors: list[str]) -> None:
    _reject_unknown_fields(document, _SCENARIO_FIELDS, "$", errors)
    for key in ("scenario_id", "version", "title", "description"):
        _required_string(document, key, "$", errors)

    risks = _required_list(document, "risks", "$", errors)
    if not risks:
        errors.append("$.risks must contain at least one risk")
    for index, risk in enumerate(risks):
        path = f"$.risks[{index}]"
        if not isinstance(risk, dict):
            errors.append(f"{path} must be an object")
            continue
        _reject_unknown_fields(risk, {"taxonomy", "category", "risk_id"}, path, errors)
        for key in ("taxonomy", "category", "risk_id"):
            _required_string(risk, key, path, errors)

    target_requirements = document.get("target_requirements", [])
    if not _is_string_list(target_requirements):
        errors.append("$.target_requirements must be an array of strings")

    stimulus = _required_list(document, "stimulus", "$", errors)
    if not stimulus:
        errors.append("$.stimulus must contain at least one message")
    for index, message in enumerate(stimulus):
        path = f"$.stimulus[{index}]"
        if not isinstance(message, dict):
            errors.append(f"{path} must be an object")
            continue
        _reject_unknown_fields(message, {"role", "content", "metadata"}, path, errors)
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            errors.append(f"{path}.role has an unsupported value")
        _required_string(message, "content", path, errors)
        _optional_object(message, "metadata", path, errors)

    fixtures = _required_list(document, "fixtures", "$", errors)
    fixture_ids: set[str] = set()
    for index, fixture in enumerate(fixtures):
        path = f"$.fixtures[{index}]"
        if not isinstance(fixture, dict):
            errors.append(f"{path} must be an object")
            continue
        _reject_unknown_fields(
            fixture,
            {"fixture_id", "kind", "trust", "content", "metadata"},
            path,
            errors,
        )
        fixture_id = _required_string(fixture, "fixture_id", path, errors)
        if fixture_id in fixture_ids:
            errors.append(f"{path}.fixture_id must be unique")
        fixture_ids.add(fixture_id)
        _required_string(fixture, "kind", path, errors)
        if fixture.get("trust") not in {"trusted", "untrusted"}:
            errors.append(f"{path}.trust must be trusted or untrusted")
        _required_string(fixture, "content", path, errors)
        _optional_object(fixture, "metadata", path, errors)

    assertions = _required_list(document, "assertions", "$", errors)
    if not assertions:
        errors.append("$.assertions must contain at least one assertion")
    assertion_ids: set[str] = set()
    for index, assertion in enumerate(assertions):
        path = f"$.assertions[{index}]"
        if not isinstance(assertion, dict):
            errors.append(f"{path} must be an object")
            continue
        _reject_unknown_fields(
            assertion,
            {"assertion_id", "objective", "evaluator", "parameters"},
            path,
            errors,
        )
        assertion_id = _required_string(assertion, "assertion_id", path, errors)
        if assertion_id in assertion_ids:
            errors.append(f"{path}.assertion_id must be unique")
        assertion_ids.add(assertion_id)
        _required_string(assertion, "objective", path, errors)
        _validate_evaluator(assertion.get("evaluator"), f"{path}.evaluator", errors)
        parameters = assertion.get("parameters", {})
        if not isinstance(parameters, dict):
            errors.append(f"{path}.parameters must be an object")

    controls = document.get("controls")
    if not isinstance(controls, dict):
        errors.append("$.controls must be an object")
    else:
        _reject_unknown_fields(
            controls,
            {"utility_assertion_id", "paired_scenario_ref"},
            "$.controls",
            errors,
        )
        utility_assertion_id = _required_string(
            controls,
            "utility_assertion_id",
            "$.controls",
            errors,
        )
        if utility_assertion_id and utility_assertion_id not in assertion_ids:
            errors.append("$.controls.utility_assertion_id must reference an assertion")
        if utility_assertion_id and not any(
            assertion_id != utility_assertion_id for assertion_id in assertion_ids
        ):
            errors.append("$.assertions must contain at least one security assertion")
        if "paired_scenario_ref" in controls:
            _validate_document_ref(
                controls.get("paired_scenario_ref"),
                "$.controls.paired_scenario_ref",
                errors,
            )


def _validate_trial_request(document: dict[str, Any], errors: list[str]) -> None:
    _reject_unknown_fields(document, _TRIAL_REQUEST_FIELDS, "$", errors)
    _required_string(document, "request_id", "$", errors)
    _required_string(document, "version", "$", errors)
    _required_string(document, "trial_id", "$", errors)
    _validate_target(document.get("target"), "$.target", errors)
    _validate_document_ref(document.get("scenario_ref"), "$.scenario_ref", errors)
    _required_int(document, "trial_index", "$", errors, minimum=0)
    seed = document.get("seed")
    if seed is not None and not isinstance(seed, int):
        errors.append("$.seed must be an integer or null")
    limits = document.get("limits")
    if not isinstance(limits, dict):
        errors.append("$.limits must be an object")
    else:
        _reject_unknown_fields(
            limits,
            {
                "timeout_seconds",
                "max_tokens",
                "max_tool_calls",
                "max_response_bytes",
                "max_cost_usd",
            },
            "$.limits",
            errors,
        )
        for key in ("timeout_seconds", "max_cost_usd"):
            value = limits.get(key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                errors.append(f"$.limits.{key} must be a non-negative number")
        timeout = limits.get("timeout_seconds")
        if isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and timeout <= 0:
            errors.append("$.limits.timeout_seconds must be greater than zero")
        for key in ("max_tokens", "max_tool_calls", "max_response_bytes"):
            minimum = 0 if key == "max_tool_calls" else 1
            _required_int(limits, key, "$.limits", errors, minimum=minimum)


def _validate_trial_trace(document: dict[str, Any], errors: list[str]) -> None:
    _reject_unknown_fields(document, _TRIAL_TRACE_FIELDS, "$", errors)
    _required_string(document, "trial_id", "$", errors)
    _required_string(document, "version", "$", errors)
    _validate_document_ref(document.get("request_ref"), "$.request_ref", errors)
    _validate_document_ref(document.get("scenario_ref"), "$.scenario_ref", errors)
    _validate_target(document.get("target"), "$.target", errors)
    for key in ("started_at", "completed_at"):
        _required_string(document, key, "$", errors)
    if document.get("status") not in _TRIAL_STATUSES:
        errors.append(f"$.status must be one of {sorted(_TRIAL_STATUSES)}")

    events = _required_list(document, "events", "$", errors)
    event_ids: set[str] = set()
    sequences: list[int] = []
    for index, event in enumerate(events):
        path = f"$.events[{index}]"
        if not isinstance(event, dict):
            errors.append(f"{path} must be an object")
            continue
        _reject_unknown_fields(
            event,
            {"event_id", "sequence", "event_type", "timestamp", "payload"},
            path,
            errors,
        )
        event_id = _required_string(event, "event_id", path, errors)
        if event_id in event_ids:
            errors.append(f"{path}.event_id must be unique")
        event_ids.add(event_id)
        sequence = _required_int(event, "sequence", path, errors, minimum=0)
        if isinstance(sequence, int):
            sequences.append(sequence)
        if event.get("event_type") not in _EVENT_TYPES:
            errors.append(f"{path}.event_type must be one of {sorted(_EVENT_TYPES)}")
        _required_string(event, "timestamp", path, errors)
        if not isinstance(event.get("payload"), dict):
            errors.append(f"{path}.payload must be an object")
    if sequences != list(range(len(sequences))):
        errors.append("$.events sequence values must be contiguous and start at 0")

    artifacts = _required_list(document, "artifacts", "$", errors)
    artifact_ids: set[str] = set()
    for index, artifact in enumerate(artifacts):
        path = f"$.artifacts[{index}]"
        if not isinstance(artifact, dict):
            errors.append(f"{path} must be an object")
            continue
        _reject_unknown_fields(
            artifact,
            {"artifact_id", "media_type", "sha256", "path"},
            path,
            errors,
        )
        artifact_id = _required_string(artifact, "artifact_id", path, errors)
        if artifact_id in artifact_ids:
            errors.append(f"{path}.artifact_id must be unique")
        artifact_ids.add(artifact_id)
        _required_string(artifact, "media_type", path, errors)
        digest = _required_string(artifact, "sha256", path, errors)
        if digest and not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"{path}.sha256 must be 64 lowercase hex characters")
        _required_string(artifact, "path", path, errors)

    results = _required_list(document, "assertion_results", "$", errors)
    result_ids: set[str] = set()
    for index, result in enumerate(results):
        path = f"$.assertion_results[{index}]"
        if not isinstance(result, dict):
            errors.append(f"{path} must be an object")
            continue
        _reject_unknown_fields(
            result,
            {"assertion_id", "status", "evaluator", "evidence", "score", "explanation"},
            path,
            errors,
        )
        assertion_id = _required_string(result, "assertion_id", path, errors)
        if assertion_id in result_ids:
            errors.append(f"{path}.assertion_id must be unique")
        result_ids.add(assertion_id)
        if result.get("status") not in _ASSERTION_STATUSES:
            errors.append(f"{path}.status must be one of {sorted(_ASSERTION_STATUSES)}")
        _validate_evaluator(result.get("evaluator"), f"{path}.evaluator", errors)
        evidence = _required_list(result, "evidence", path, errors)
        if result.get("status") in {"pass", "fail"} and not evidence:
            errors.append(f"{path}.evidence is required for pass or fail")
        for evidence_index, reference in enumerate(evidence):
            reference_path = f"{path}.evidence[{evidence_index}]"
            if not isinstance(reference, dict):
                errors.append(f"{reference_path} must be an object")
                continue
            _reject_unknown_fields(
                reference,
                {"ref_type", "ref_id", "selector"},
                reference_path,
                errors,
            )
            ref_type = reference.get("ref_type")
            ref_id = _required_string(reference, "ref_id", reference_path, errors)
            if ref_type == "event" and ref_id not in event_ids:
                errors.append(f"{reference_path}.ref_id does not reference an event")
            elif ref_type == "artifact" and ref_id not in artifact_ids:
                errors.append(f"{reference_path}.ref_id does not reference an artifact")
            elif ref_type not in {"event", "artifact"}:
                errors.append(f"{reference_path}.ref_type must be event or artifact")
        score = result.get("score")
        if score is not None and (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not 0 <= score <= 1
        ):
            errors.append(f"{path}.score must be between 0 and 1")


def _validate_evaluation_summary(document: dict[str, Any], errors: list[str]) -> None:
    _reject_unknown_fields(document, _SUMMARY_FIELDS, "$", errors)
    _required_string(document, "evaluation_id", "$", errors)
    _validate_document_ref(document.get("scenario_ref"), "$.scenario_ref", errors)
    _validate_target(document.get("target"), "$.target", errors)
    trial_refs = _required_list(document, "trial_refs", "$", errors)
    for index, reference in enumerate(trial_refs):
        _validate_document_ref(reference, f"$.trial_refs[{index}]", errors)
    metrics = document.get("metrics")
    if not isinstance(metrics, dict):
        errors.append("$.metrics must be an object")
    else:
        for key, value in metrics.items():
            if value is not None and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                errors.append(f"$.metrics.{key} must be a number or null")
    _required_string(document, "generated_at", "$", errors)


def _validate_target(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object")
        return
    _reject_unknown_fields(
        value,
        {
            "target_id",
            "target_type",
            "version",
            "capabilities",
            "endpoint_ref",
            "secret_ref",
            "metadata",
        },
        path,
        errors,
    )
    for key in ("target_id", "target_type", "version"):
        _required_string(value, key, path, errors)
    endpoint_ref = value.get("endpoint_ref", "")
    secret_ref = value.get("secret_ref", "")
    if endpoint_ref and not isinstance(endpoint_ref, str):
        errors.append(f"{path}.endpoint_ref must be a string")
    if secret_ref and not isinstance(secret_ref, str):
        errors.append(f"{path}.secret_ref must be a string")
    capabilities = value.get("capabilities", [])
    if not _is_string_list(capabilities):
        errors.append(f"{path}.capabilities must be an array of strings")
    metadata = value.get("metadata", {})
    if not isinstance(metadata, dict):
        errors.append(f"{path}.metadata must be an object")


def _validate_evaluator(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object")
        return
    _reject_unknown_fields(
        value,
        {"evaluator_id", "kind", "version", "implementation_ref", "model_ref"},
        path,
        errors,
    )
    for key in ("evaluator_id", "version", "implementation_ref"):
        _required_string(value, key, path, errors)
    kind = value.get("kind")
    if kind not in _EVALUATOR_KINDS:
        errors.append(f"{path}.kind must be one of {sorted(_EVALUATOR_KINDS)}")
    model_ref = value.get("model_ref", "")
    if kind == "probabilistic" and not isinstance(model_ref, str):
        errors.append(f"{path}.model_ref must be a string")
    if kind == "probabilistic" and not model_ref:
        errors.append(f"{path}.model_ref is required for probabilistic evaluators")
    if kind != "probabilistic" and model_ref:
        errors.append(f"{path}.model_ref is only valid for probabilistic evaluators")


def _validate_document_ref(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object")
        return
    _reject_unknown_fields(value, {"document_id", "version", "fingerprint"}, path, errors)
    for key in ("document_id", "version"):
        _required_string(value, key, path, errors)
    fingerprint = value.get("fingerprint", "")
    if fingerprint and (
        not isinstance(fingerprint, str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", fingerprint)
    ):
        errors.append(f"{path}.fingerprint must be sha256: followed by 64 lowercase hex characters")


def _reject_unknown_fields(
    value: dict[str, Any],
    allowed: set[str],
    path: str,
    errors: list[str],
) -> None:
    for key in sorted(set(value) - allowed):
        errors.append(f"{path}.{key} is an unsupported field")


def _optional_object(
    value: dict[str, Any],
    key: str,
    path: str,
    errors: list[str],
) -> None:
    if key in value and not isinstance(value[key], dict):
        errors.append(f"{path}.{key} must be an object")


def _reject_inline_credentials(value: Any, path: str, errors: list[str]) -> None:
    credential_keys = {
        "api_key",
        "apikey",
        "access_token",
        "auth_token",
        "authorization",
        "bearer",
        "client_secret",
        "credential",
        "credentials",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "token",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
            if normalized in credential_keys:
                errors.append(f"{path}.{key} must use a secret reference, never inline credentials")
            _reject_inline_credentials(item, f"{path}.{key}", errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_inline_credentials(item, f"{path}[{index}]", errors)


def _required_string(
    value: dict[str, Any],
    key: str,
    path: str,
    errors: list[str],
) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        errors.append(f"{path}.{key} must be a non-empty string")
        return ""
    return result


def _required_list(
    value: dict[str, Any],
    key: str,
    path: str,
    errors: list[str],
) -> list[Any]:
    result = value.get(key)
    if not isinstance(result, list):
        errors.append(f"{path}.{key} must be an array")
        return []
    return result


def _required_int(
    value: dict[str, Any],
    key: str,
    path: str,
    errors: list[str],
    *,
    minimum: int,
) -> int | None:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool) or result < minimum:
        errors.append(f"{path}.{key} must be an integer >= {minimum}")
        return None
    return result


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)
