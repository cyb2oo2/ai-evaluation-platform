from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_eval_protocol.artifact_io import OutputTransaction, OutputTransactionError
from ai_eval_protocol.codec import load_document
from ai_eval_protocol.secret_safety import assert_no_secrets, resolved_secret_values
from ai_eval_protocol.worker import TrialRunResult, TrialWorker, TrialWorkerError


@dataclass(frozen=True)
class SuiteRunResult:
    output_dir: Path
    summary_path: Path
    manifest_path: Path
    summary: dict[str, Any]
    trials: tuple[TrialRunResult, ...]


class SuiteWorker:
    def __init__(self, *, trial_worker: TrialWorker) -> None:
        self.trial_worker = trial_worker

    def run(
        self,
        *,
        suite_path: Path,
        output_dir: Path,
        force: bool = False,
    ) -> SuiteRunResult:
        suite = _load_suite(suite_path)
        final_dir = output_dir.resolve()
        try:
            with OutputTransaction(final_dir, force=force) as stage_dir:
                summary, trials = self._write_suite_bundle(
                    suite=suite,
                    suite_path=suite_path,
                    output_dir=stage_dir,
                )
        except OutputTransactionError as exc:
            raise TrialWorkerError(str(exc)) from exc
        rebased_trials = tuple(
            _rebase_trial(trial, old_root=stage_dir, new_root=final_dir)
            for trial in trials
        )
        return SuiteRunResult(
            output_dir=final_dir,
            summary_path=final_dir / "suite_summary.json",
            manifest_path=final_dir / "bundle_manifest.json",
            summary=summary,
            trials=rebased_trials,
        )

    def _write_suite_bundle(
        self,
        *,
        suite: dict[str, Any],
        suite_path: Path,
        output_dir: Path,
    ) -> tuple[dict[str, Any], tuple[TrialRunResult, ...]]:
        trials: list[TrialRunResult] = []
        case_results: list[dict[str, Any]] = []
        secret_values: list[str] = []
        suite_root = suite_path.parent
        for case in suite["cases"]:
            case_id = case["case_id"]
            scenario_path = (suite_root / case["scenario"]).resolve()
            request_path = (suite_root / case["request"]).resolve()
            for input_path in (scenario_path, request_path):
                if not input_path.is_relative_to(suite_root.resolve()):
                    raise TrialWorkerError(
                        f"suite case '{case_id}' input escapes the suite directory: {input_path}"
                    )
            scenario = load_document(scenario_path)
            request = load_document(request_path)
            for value in resolved_secret_values(
                scenario=scenario,
                request=request,
                secret_resolver=self.trial_worker.secret_resolver,
            ):
                if value not in secret_values:
                    secret_values.append(value)
            trial = self.trial_worker.run(
                scenario_path=scenario_path,
                request_path=request_path,
                output_dir=output_dir / "trials" / case_id,
            )
            trials.append(trial)
            metrics = trial.summary["metrics"]
            target = trial.summary["target"]
            target_metadata = target.get("metadata", {})
            case_results.append(
                {
                    "case_id": case_id,
                    "risk": case["risk"],
                    "target_id": target["target_id"],
                    "target_version": target["version"],
                    "model": (
                        target_metadata.get("model", "")
                        if isinstance(target_metadata, dict)
                        else ""
                    ),
                    "status": trial.trace["status"],
                    "attack_success_rate": metrics["attack_success_rate"],
                    "security_assertion_pass_rate": metrics[
                        "security_assertion_pass_rate"
                    ],
                    "utility_assertion_pass_rate": metrics[
                        "utility_assertion_pass_rate"
                    ],
                    "trial_bundle": f"trials/{case_id}/bundle_manifest.json",
                    "trial_fingerprint": trial.summary["trial_refs"][0]["fingerprint"],
                }
            )

        model_results = _model_results(case_results)
        summary = {
            "bundle_version": "0.2",
            "artifact_type": "evaluation_suite_summary",
            "suite_id": suite["suite_id"],
            "suite_version": suite["version"],
            "title": suite["title"],
            "generated_at": _utc_now(),
            "case_results": case_results,
            "model_results": model_results,
            "metrics": _aggregate_case_metrics(case_results),
        }
        inputs_dir = output_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        suite_copy = inputs_dir / "suite.json"
        _write_json(suite_copy, suite)
        summary_path = output_dir / "suite_summary.json"
        _write_json(summary_path, summary)
        manifest = {
            "bundle_version": "0.2",
            "artifact_type": "evaluation_suite_bundle",
            "suite_id": suite["suite_id"],
            "artifacts": [
                _file_entry(output_dir, suite_copy),
                _file_entry(output_dir, summary_path),
                *(
                    _file_entry(output_dir, trial.manifest_path)
                    for trial in trials
                ),
            ],
        }
        manifest_path = output_dir / "bundle_manifest.json"
        _write_json(manifest_path, manifest)
        try:
            assert_no_secrets(output_dir, secret_values)
        except ValueError as exc:
            raise TrialWorkerError(str(exc)) from exc
        return summary, tuple(trials)


def _load_suite(path: Path) -> dict[str, Any]:
    try:
        suite = load_document(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise TrialWorkerError(f"invalid suite document '{path}': {exc}") from exc
    errors: list[str] = []
    for key in ("suite_id", "version", "title"):
        if not isinstance(suite.get(key), str) or not suite[key].strip():
            errors.append(f"$.{key} must be a non-empty string")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append("$.cases must be a non-empty array")
        cases = []
    seen: set[str] = set()
    for index, case in enumerate(cases):
        path_prefix = f"$.cases[{index}]"
        if not isinstance(case, dict):
            errors.append(f"{path_prefix} must be an object")
            continue
        for key in ("case_id", "risk", "scenario", "request"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                errors.append(f"{path_prefix}.{key} must be a non-empty string")
        case_id = case.get("case_id")
        if isinstance(case_id, str):
            if case_id in seen:
                errors.append(f"{path_prefix}.case_id must be unique")
            seen.add(case_id)
            if Path(case_id).name != case_id or case_id in {".", ".."}:
                errors.append(f"{path_prefix}.case_id must be a safe path segment")
        for key in ("scenario", "request"):
            value = case.get(key)
            if isinstance(value, str) and Path(value).is_absolute():
                errors.append(f"{path_prefix}.{key} must be relative to the suite")
    if errors:
        raise TrialWorkerError("invalid suite document: " + "; ".join(errors))
    return suite


def _aggregate_case_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    unavailable = any(
        item["attack_success_rate"] is None or item["status"] != "completed"
        for item in case_results
    )
    successful_attacks = sum(
        item["attack_success_rate"] == 1.0 for item in case_results
    )
    return {
        "case_count": len(case_results),
        "completed_case_count": sum(
            item["status"] == "completed" for item in case_results
        ),
        "attack_success_count": None if unavailable else successful_attacks,
        "attack_success_rate": (
            None if unavailable else round(successful_attacks / len(case_results), 4)
        ),
        "security_assertion_pass_rate": round(
            sum(item["security_assertion_pass_rate"] for item in case_results)
            / len(case_results),
            4,
        ),
        "utility_assertion_pass_rate": round(
            sum(item["utility_assertion_pass_rate"] for item in case_results)
            / len(case_results),
            4,
        ),
    }


def _model_results(case_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = dict.fromkeys(
        (item["model"], item["target_id"], item["target_version"])
        for item in case_results
    )
    results: list[dict[str, Any]] = []
    for model, target_id, target_version in keys:
        model_cases = [
            item
            for item in case_results
            if (
                item["model"],
                item["target_id"],
                item["target_version"],
            )
            == (model, target_id, target_version)
        ]
        results.append(
            {
                "model": model,
                "target_id": target_id,
                "target_version": target_version,
                "metrics": _aggregate_case_metrics(model_cases),
            }
        )
    return results


def _rebase_trial(
    trial: TrialRunResult,
    *,
    old_root: Path,
    new_root: Path,
) -> TrialRunResult:
    relative = trial.output_dir.relative_to(old_root)
    output_dir = new_root / relative
    return replace(
        trial,
        output_dir=output_dir,
        trace_path=output_dir / "trial_trace.json",
        summary_path=output_dir / "evaluation_summary.json",
        manifest_path=output_dir / "bundle_manifest.json",
    )


def _file_entry(output_dir: Path, path: Path) -> dict[str, str]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(output_dir).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
