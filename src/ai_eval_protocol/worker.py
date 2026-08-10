from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_eval_protocol.artifact_io import OutputTransaction, OutputTransactionError
from ai_eval_protocol.codec import fingerprint, load_document
from ai_eval_protocol.evaluators import (
    EnvironmentSecretResolver,
    EvaluatorError,
    EvaluatorRegistry,
    SecretResolver,
)
from ai_eval_protocol.models import PROTOCOL_VERSION
from ai_eval_protocol.secret_safety import (
    assert_no_secrets,
    redact_value,
    resolved_secret_values,
)
from ai_eval_protocol.targets import TargetAdapter, TargetAdapterError, TargetRegistry
from ai_eval_protocol.validation import ProtocolValidationError, validate_document


class TrialWorkerError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrialRunResult:
    output_dir: Path
    trace_path: Path
    summary_path: Path
    manifest_path: Path
    trace: dict[str, Any]
    summary: dict[str, Any]


class TrialWorker:
    def __init__(
        self,
        *,
        target_registry: TargetRegistry | None = None,
        evaluator_registry: EvaluatorRegistry | None = None,
        secret_resolver: SecretResolver | None = None,
    ) -> None:
        self.target_registry = target_registry or TargetRegistry()
        self.evaluator_registry = evaluator_registry or EvaluatorRegistry()
        self.secret_resolver = secret_resolver or EnvironmentSecretResolver()

    def run(
        self,
        *,
        scenario_path: Path,
        request_path: Path,
        output_dir: Path,
        force: bool = False,
    ) -> TrialRunResult:
        scenario = _load_and_validate(scenario_path, expected_type="scenario")
        request = _load_and_validate(request_path, expected_type="trial_request")
        _validate_binding(scenario, request)

        try:
            adapter = self.target_registry.resolve(request["target"])
            for assertion in scenario["assertions"]:
                self.evaluator_registry.resolve(assertion["evaluator"])
        except (TargetAdapterError, EvaluatorError) as exc:
            raise TrialWorkerError(str(exc)) from exc

        final_dir = output_dir.resolve()
        try:
            with OutputTransaction(final_dir, force=force) as stage_dir:
                trace, summary = self._write_trial_bundle(
                    scenario=scenario,
                    request=request,
                    adapter=adapter,
                    output_dir=stage_dir,
                )
        except OutputTransactionError as exc:
            raise TrialWorkerError(str(exc)) from exc
        return TrialRunResult(
            output_dir=final_dir,
            trace_path=final_dir / "trial_trace.json",
            summary_path=final_dir / "evaluation_summary.json",
            manifest_path=final_dir / "bundle_manifest.json",
            trace=trace,
            summary=summary,
        )

    def _write_trial_bundle(
        self,
        *,
        scenario: dict[str, Any],
        request: dict[str, Any],
        adapter: TargetAdapter,
        output_dir: Path,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        inputs_dir = output_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        scenario_copy = inputs_dir / "scenario.json"
        request_copy = inputs_dir / "request.json"
        _write_json(scenario_copy, scenario)
        _write_json(request_copy, request)

        started_at = _utc_now()
        try:
            execution = adapter.execute(
                scenario=scenario,
                request=request,
                secret_resolver=self.secret_resolver,
            )
        except TargetAdapterError as exc:
            raise TrialWorkerError(str(exc)) from exc
        assertion_results = self.evaluator_registry.evaluate_all(
            assertions=scenario["assertions"],
            events=execution.events,
            secret_resolver=self.secret_resolver,
        )
        secret_values = resolved_secret_values(
            scenario=scenario,
            request=request,
            secret_resolver=self.secret_resolver,
        )
        try:
            serialized_events = tuple(
                redact_value(event, secret_values) for event in execution.events
            )
            serialized_results = tuple(
                redact_value(result, secret_values) for result in assertion_results
            )
        except ValueError as exc:
            raise TrialWorkerError(str(exc)) from exc
        completed_at = _utc_now()

        trace = _build_trace(
            scenario=scenario,
            request=request,
            events=serialized_events,
            assertion_results=serialized_results,
            started_at=started_at,
            completed_at=completed_at,
            status=execution.status,
        )
        _validate_generated(trace, kind="trial trace")
        trace_path = output_dir / "trial_trace.json"
        _write_json(trace_path, trace)

        summary = _build_summary(
            scenario=scenario,
            request=request,
            trace=trace,
            generated_at=_utc_now(),
            execution_metrics=execution.metrics,
        )
        _validate_generated(summary, kind="evaluation summary")
        summary_path = output_dir / "evaluation_summary.json"
        _write_json(summary_path, summary)

        manifest = {
            "bundle_version": "0.2",
            "artifacts": [
                _artifact_entry(output_dir, scenario_copy, scenario),
                _artifact_entry(output_dir, request_copy, request),
                _artifact_entry(output_dir, trace_path, trace),
                _artifact_entry(output_dir, summary_path, summary),
            ],
        }
        manifest_path = output_dir / "bundle_manifest.json"
        _write_json(manifest_path, manifest)
        try:
            assert_no_secrets(output_dir, secret_values)
        except ValueError as exc:
            raise TrialWorkerError(str(exc)) from exc
        return trace, summary


def _load_and_validate(path: Path, *, expected_type: str) -> dict[str, Any]:
    try:
        document = load_document(path)
        validate_document(document)
    except (OSError, ValueError, ProtocolValidationError) as exc:
        raise TrialWorkerError(f"invalid {expected_type} document '{path}': {exc}") from exc
    if document.get("document_type") != expected_type:
        raise TrialWorkerError(
            f"expected {expected_type} document at '{path}', got "
            f"{document.get('document_type')!r}"
        )
    return document


def _validate_binding(scenario: dict[str, Any], request: dict[str, Any]) -> None:
    scenario_ref = request["scenario_ref"]
    if scenario_ref["document_id"] != scenario["scenario_id"]:
        raise TrialWorkerError("request scenario_ref.document_id does not match the scenario")
    if scenario_ref["version"] != scenario["version"]:
        raise TrialWorkerError("request scenario_ref.version does not match the scenario")
    declared_fingerprint = scenario_ref.get("fingerprint")
    if declared_fingerprint and declared_fingerprint != fingerprint(scenario):
        raise TrialWorkerError("request scenario_ref.fingerprint does not match the scenario")
    required = set(scenario.get("target_requirements", []))
    capabilities = set(request["target"].get("capabilities", []))
    missing = sorted(required - capabilities)
    if missing:
        raise TrialWorkerError(
            f"target is missing scenario-required capabilities: {', '.join(missing)}"
        )


def _build_trace(
    *,
    scenario: dict[str, Any],
    request: dict[str, Any],
    events: tuple[dict[str, Any], ...],
    assertion_results: tuple[dict[str, Any], ...],
    started_at: str,
    completed_at: str,
    status: str,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "document_type": "trial_trace",
        "trial_id": request["trial_id"],
        "version": "1.0.0",
        "request_ref": {
            "document_id": request["request_id"],
            "version": request["version"],
            "fingerprint": fingerprint(request),
        },
        "scenario_ref": {
            "document_id": scenario["scenario_id"],
            "version": scenario["version"],
            "fingerprint": fingerprint(scenario),
        },
        "target": request["target"],
        "started_at": started_at,
        "completed_at": completed_at,
        "status": status,
        "events": list(events),
        "artifacts": [],
        "assertion_results": list(assertion_results),
    }


def _build_summary(
    *,
    scenario: dict[str, Any],
    request: dict[str, Any],
    trace: dict[str, Any],
    generated_at: str,
    execution_metrics: dict[str, int | float | None],
) -> dict[str, Any]:
    results = trace["assertion_results"]
    utility_id = scenario["controls"]["utility_assertion_id"]
    security = [result for result in results if result["assertion_id"] != utility_id]
    utility = [result for result in results if result["assertion_id"] == utility_id]
    failed_security = sum(result["status"] == "fail" for result in security)
    security_inconclusive = not security or any(
        result["status"] not in {"pass", "fail"} for result in security
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "document_type": "evaluation_summary",
        "evaluation_id": f"evaluation-{request['trial_id']}",
        "scenario_ref": {
            "document_id": scenario["scenario_id"],
            "version": scenario["version"],
            "fingerprint": fingerprint(scenario),
        },
        "target": request["target"],
        "trial_refs": [
            {
                "document_id": trace["trial_id"],
                "version": trace["version"],
                "fingerprint": fingerprint(trace),
            }
        ],
        "metrics": {
            "trial_count": 1,
            "assertion_count": len(results),
            "assertion_error_count": sum(result["status"] == "error" for result in results),
            "assertion_pass_rate": _pass_rate(results),
            "security_assertion_pass_rate": _pass_rate(security),
            "utility_assertion_pass_rate": _pass_rate(utility),
            "attack_success_rate": (
                None
                if security_inconclusive or trace["status"] != "completed"
                else 1.0
                if failed_security
                else 0.0
            ),
            **execution_metrics,
        },
        "generated_at": generated_at,
    }


def _pass_rate(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    passed = sum(result["status"] == "pass" for result in results)
    return round(passed / len(results), 4)


def _validate_generated(document: dict[str, Any], *, kind: str) -> None:
    try:
        validate_document(document)
    except ProtocolValidationError as exc:
        raise TrialWorkerError(f"generated {kind} failed protocol validation: {exc}") from exc


def _artifact_entry(
    output_dir: Path,
    path: Path,
    document: dict[str, Any],
) -> dict[str, str]:
    return {
        "path": path.relative_to(output_dir).as_posix(),
        "document_type": str(document["document_type"]),
        "fingerprint": fingerprint(document),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
