from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_eval_protocol.codec import fingerprint, load_document
from ai_eval_protocol.suite import _aggregate_case_metrics, _model_results
from ai_eval_protocol.validation import ProtocolValidationError, validate_document

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_TRIAL_ARTIFACT_PATHS = {
    "inputs/scenario.json",
    "inputs/request.json",
    "trial_trace.json",
    "evaluation_summary.json",
}


class BundleVerificationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("; ".join(errors))


@dataclass(frozen=True)
class BundleVerificationResult:
    root: Path
    bundle_type: str
    artifact_count: int
    child_bundle_count: int


def verify_bundle(root: Path) -> BundleVerificationResult:
    bundle_root = root.resolve()
    errors: list[str] = []
    if not bundle_root.is_dir():
        raise BundleVerificationError([f"bundle path is not a directory: {bundle_root}"])
    _check_transaction_residue(bundle_root, errors)
    manifest = _load_json(bundle_root / "bundle_manifest.json", errors)
    if manifest is None:
        raise BundleVerificationError(errors)
    if manifest.get("bundle_version") != "0.2":
        errors.append("bundle_manifest.json: bundle_version must be '0.2'")
    if manifest.get("artifact_type") == "evaluation_suite_bundle":
        result = _verify_suite_bundle(bundle_root, manifest, errors)
    else:
        result = _verify_trial_bundle(bundle_root, manifest, errors)
    if errors:
        raise BundleVerificationError(errors)
    return result


def _verify_trial_bundle(
    root: Path,
    manifest: dict[str, Any],
    errors: list[str],
) -> BundleVerificationResult:
    _reject_unknown_fields(
        manifest,
        {"bundle_version", "artifacts"},
        "bundle_manifest.json",
        errors,
    )
    artifacts = _artifact_entries(manifest, errors)
    paths = {entry.get("path") for entry in artifacts if isinstance(entry.get("path"), str)}
    if paths != _TRIAL_ARTIFACT_PATHS:
        errors.append(
            "bundle_manifest.json: trial artifact paths must be exactly "
            + ", ".join(sorted(_TRIAL_ARTIFACT_PATHS))
        )
    documents: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(artifacts):
        label = f"bundle_manifest.json.artifacts[{index}]"
        _reject_unknown_fields(
            entry,
            {"path", "document_type", "fingerprint", "sha256"},
            label,
            errors,
        )
        path = _verify_artifact(root, entry, label, errors)
        if path is None:
            continue
        document = _load_json(path, errors, display_path=entry["path"])
        if document is None:
            continue
        documents[entry["path"]] = document
        try:
            validate_document(document)
        except ProtocolValidationError as exc:
            errors.extend(f"{entry['path']}: {error}" for error in exc.errors)
        if entry.get("document_type") != document.get("document_type"):
            errors.append(f"{label}.document_type does not match the artifact")
        if entry.get("fingerprint") != fingerprint(document):
            errors.append(f"{label}.fingerprint does not match the artifact")
    if documents.keys() >= _TRIAL_ARTIFACT_PATHS:
        _verify_trial_references(documents, errors)
    return BundleVerificationResult(
        root=root,
        bundle_type="trial",
        artifact_count=len(artifacts),
        child_bundle_count=0,
    )


def _verify_suite_bundle(
    root: Path,
    manifest: dict[str, Any],
    errors: list[str],
) -> BundleVerificationResult:
    _reject_unknown_fields(
        manifest,
        {"bundle_version", "artifact_type", "suite_id", "artifacts"},
        "bundle_manifest.json",
        errors,
    )
    artifacts = _artifact_entries(manifest, errors)
    verified_paths: set[str] = set()
    child_manifests: list[str] = []
    for index, entry in enumerate(artifacts):
        label = f"bundle_manifest.json.artifacts[{index}]"
        _reject_unknown_fields(entry, {"path", "sha256"}, label, errors)
        path = _verify_artifact(root, entry, label, errors)
        if path is None:
            continue
        artifact_path = entry["path"]
        verified_paths.add(artifact_path)
        if artifact_path.startswith("trials/") and artifact_path.endswith(
            "/bundle_manifest.json"
        ):
            child_manifests.append(artifact_path)
    for required in ("inputs/suite.json", "suite_summary.json"):
        if required not in verified_paths:
            errors.append(f"bundle_manifest.json: missing required artifact '{required}'")
    suite = _load_json(root / "inputs" / "suite.json", errors, display_path="inputs/suite.json")
    summary = _load_json(root / "suite_summary.json", errors, display_path="suite_summary.json")
    child_results: dict[str, BundleVerificationResult] = {}
    for child_manifest in child_manifests:
        child_root = (root / child_manifest).parent
        try:
            child_results[child_manifest] = verify_bundle(child_root)
        except BundleVerificationError as exc:
            errors.extend(f"{child_manifest}: {error}" for error in exc.errors)
    if suite is not None and summary is not None:
        suite_cases = suite.get("cases")
        if not isinstance(suite_cases, list):
            suite_cases = []
        expected_paths = {
            "inputs/suite.json",
            "suite_summary.json",
            *(
                f"trials/{case.get('case_id')}/bundle_manifest.json"
                for case in suite_cases
                if isinstance(case, dict) and isinstance(case.get("case_id"), str)
            ),
        }
        if verified_paths != expected_paths:
            errors.append(
                "bundle_manifest.json: suite artifact paths do not exactly match suite cases"
            )
        _verify_suite_references(
            root=root,
            manifest=manifest,
            suite=suite,
            summary=summary,
            child_manifests=set(child_manifests),
            errors=errors,
        )
    return BundleVerificationResult(
        root=root,
        bundle_type="suite",
        artifact_count=len(artifacts)
        + sum(result.artifact_count for result in child_results.values()),
        child_bundle_count=len(child_results),
    )


def _verify_trial_references(
    documents: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    scenario = documents["inputs/scenario.json"]
    request = documents["inputs/request.json"]
    trace = documents["trial_trace.json"]
    summary = documents["evaluation_summary.json"]
    expected_refs = (
        (
            request.get("scenario_ref"),
            scenario,
            scenario.get("scenario_id"),
            "request.scenario_ref",
        ),
        (
            trace.get("scenario_ref"),
            scenario,
            scenario.get("scenario_id"),
            "trace.scenario_ref",
        ),
        (trace.get("request_ref"), request, request.get("request_id"), "trace.request_ref"),
        (
            summary.get("scenario_ref"),
            scenario,
            scenario.get("scenario_id"),
            "summary.scenario_ref",
        ),
    )
    for reference, document, document_id, label in expected_refs:
        _check_reference(reference, document, document_id, label, errors)
    trial_refs = summary.get("trial_refs", [])
    if len(trial_refs) == 1:
        _check_reference(
            trial_refs[0],
            trace,
            trace.get("trial_id"),
            "summary.trial_refs[0]",
            errors,
        )
    else:
        errors.append("evaluation_summary.json: trial_refs must contain exactly one trial")
    if request.get("trial_id") != trace.get("trial_id"):
        errors.append("trial_trace.json: trial_id does not match inputs/request.json")
    if not (request.get("target") == trace.get("target") == summary.get("target")):
        errors.append("trial target identity differs across request, trace, and summary")


def _verify_suite_references(
    *,
    root: Path,
    manifest: dict[str, Any],
    suite: dict[str, Any],
    summary: dict[str, Any],
    child_manifests: set[str],
    errors: list[str],
) -> None:
    if manifest.get("suite_id") != suite.get("suite_id"):
        errors.append("bundle_manifest.json: suite_id does not match inputs/suite.json")
    if summary.get("suite_id") != suite.get("suite_id"):
        errors.append("suite_summary.json: suite_id does not match inputs/suite.json")
    if summary.get("suite_version") != suite.get("version"):
        errors.append("suite_summary.json: suite_version does not match inputs/suite.json")
    cases = suite.get("cases")
    results = summary.get("case_results")
    if not isinstance(cases, list) or not isinstance(results, list):
        errors.append("suite inputs and summary must contain case arrays")
        return
    case_ids = [case.get("case_id") for case in cases if isinstance(case, dict)]
    result_ids = [result.get("case_id") for result in results if isinstance(result, dict)]
    if case_ids != result_ids:
        errors.append("suite_summary.json: case order or identities do not match inputs/suite.json")
    valid_results: list[dict[str, Any]] = []
    required_result_fields = {
        "case_id",
        "model",
        "target_id",
        "target_version",
        "status",
        "attack_success_rate",
        "security_assertion_pass_rate",
        "utility_assertion_pass_rate",
        "trial_bundle",
        "trial_fingerprint",
    }
    for result in results:
        if not isinstance(result, dict):
            errors.append("suite_summary.json: every case result must be an object")
            continue
        missing = sorted(required_result_fields - result.keys())
        if missing:
            errors.append(
                "suite_summary.json: case result is missing fields: " + ", ".join(missing)
            )
            continue
        valid_results.append(result)
        trial_bundle = result.get("trial_bundle")
        expected_trial_bundle = f"trials/{result.get('case_id')}/bundle_manifest.json"
        if trial_bundle != expected_trial_bundle:
            errors.append(
                f"suite_summary.json: case '{result.get('case_id')}' trial path differs"
            )
        if trial_bundle not in child_manifests:
            errors.append(
                "suite_summary.json: case "
                f"'{result.get('case_id')}' references an unmanifested trial"
            )
            continue
        child_summary = _load_json(
            (root / str(trial_bundle)).parent / "evaluation_summary.json",
            errors,
            display_path=f"{trial_bundle}/../evaluation_summary.json",
        )
        if child_summary is None:
            continue
        child_trace = _load_json(
            (root / str(trial_bundle)).parent / "trial_trace.json",
            errors,
            display_path=f"{trial_bundle}/../trial_trace.json",
        )
        trial_refs = child_summary.get("trial_refs", [])
        child_fingerprint = (
            trial_refs[0].get("fingerprint")
            if len(trial_refs) == 1 and isinstance(trial_refs[0], dict)
            else None
        )
        if result.get("trial_fingerprint") != child_fingerprint:
            errors.append(
                f"suite_summary.json: case '{result.get('case_id')}' trial fingerprint differs"
            )
        _verify_case_result(result, child_summary, child_trace, errors)
    if len(valid_results) == len(results) and valid_results:
        try:
            expected_metrics = _aggregate_case_metrics(valid_results)
            expected_models = _model_results(valid_results)
        except (KeyError, TypeError, ZeroDivisionError):
            errors.append("suite_summary.json: case result values are invalid")
        else:
            if summary.get("metrics") != expected_metrics:
                errors.append("suite_summary.json: aggregate metrics do not match case results")
            if summary.get("model_results") != expected_models:
                errors.append("suite_summary.json: model_results do not match case results")


def _verify_case_result(
    result: dict[str, Any],
    child_summary: dict[str, Any],
    child_trace: dict[str, Any] | None,
    errors: list[str],
) -> None:
    case_id = result.get("case_id")
    target = child_summary.get("target")
    metrics = child_summary.get("metrics")
    if not isinstance(target, dict) or not isinstance(metrics, dict) or child_trace is None:
        errors.append(f"suite_summary.json: case '{case_id}' child artifacts are incomplete")
        return
    metadata = target.get("metadata")
    model = metadata.get("model", "") if isinstance(metadata, dict) else ""
    expected = {
        "target_id": target.get("target_id"),
        "target_version": target.get("version"),
        "model": model,
        "status": child_trace.get("status"),
        "attack_success_rate": metrics.get("attack_success_rate"),
        "security_assertion_pass_rate": metrics.get("security_assertion_pass_rate"),
        "utility_assertion_pass_rate": metrics.get("utility_assertion_pass_rate"),
    }
    for field, value in expected.items():
        if result.get(field) != value:
            errors.append(
                f"suite_summary.json: case '{case_id}' {field} differs from child trial"
            )


def _artifact_entries(manifest: dict[str, Any], errors: list[str]) -> list[dict[str, Any]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("bundle_manifest.json: artifacts must be a non-empty array")
        return []
    if not all(isinstance(entry, dict) for entry in artifacts):
        errors.append("bundle_manifest.json: every artifact entry must be an object")
        return [entry for entry in artifacts if isinstance(entry, dict)]
    paths = [entry.get("path") for entry in artifacts]
    string_paths = [path for path in paths if isinstance(path, str)]
    if len(string_paths) != len(set(string_paths)):
        errors.append("bundle_manifest.json: artifact paths must be unique")
    return artifacts


def _verify_artifact(
    root: Path,
    entry: dict[str, Any],
    label: str,
    errors: list[str],
) -> Path | None:
    raw_path = entry.get("path")
    digest = entry.get("sha256")
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
        errors.append(f"{label}.path must be a non-empty portable relative path")
        return None
    path = (root / raw_path).resolve()
    if not path.is_relative_to(root):
        errors.append(f"{label}.path escapes the bundle root")
        return None
    if not path.is_file():
        errors.append(f"{label}.path does not exist: {raw_path}")
        return None
    if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
        errors.append(f"{label}.sha256 must be 64 lowercase hex characters")
        return path
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        errors.append(f"{label}.sha256 does not match '{raw_path}'")
    return path


def _check_reference(
    reference: Any,
    document: dict[str, Any],
    document_id: Any,
    label: str,
    errors: list[str],
) -> None:
    if not isinstance(reference, dict):
        errors.append(f"{label} must be an object")
        return
    if reference.get("document_id") != document_id:
        errors.append(f"{label}.document_id does not match its artifact")
    if reference.get("version") != document.get("version"):
        errors.append(f"{label}.version does not match its artifact")
    declared_fingerprint = reference.get("fingerprint")
    if declared_fingerprint is not None and declared_fingerprint != fingerprint(document):
        errors.append(f"{label}.fingerprint does not match its artifact")


def _load_json(
    path: Path,
    errors: list[str],
    *,
    display_path: str | None = None,
) -> dict[str, Any] | None:
    try:
        return load_document(path)
    except (OSError, ValueError) as exc:
        errors.append(f"{display_path or path.name}: cannot load JSON: {exc}")
        return None


def _reject_unknown_fields(
    value: dict[str, Any],
    allowed: set[str],
    label: str,
    errors: list[str],
) -> None:
    for key in sorted(set(value) - allowed):
        errors.append(f"{label}.{key} is an unsupported field")


def _check_transaction_residue(root: Path, errors: list[str]) -> None:
    prefixes = (
        f".{root.name}.staging-",
        f".{root.name}.backup-",
        f".{root.name}.failed-",
    )
    residue = sorted(
        path.name
        for path in root.parent.iterdir()
        if any(path.name.startswith(prefix) for prefix in prefixes)
    )
    if residue:
        errors.append("transaction residue exists beside bundle: " + ", ".join(residue))
