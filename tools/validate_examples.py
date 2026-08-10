from __future__ import annotations

import sys
from pathlib import Path

from ai_eval_protocol.codec import fingerprint, load_document
from ai_eval_protocol.suite import _load_suite
from ai_eval_protocol.validation import ProtocolValidationError, validate_document
from ai_eval_protocol.worker import TrialWorkerError

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def main() -> None:
    errors: list[str] = []
    protocol_count = 0
    suite_count = 0
    for path in sorted(EXAMPLES.rglob("*.json")):
        relative = path.relative_to(ROOT).as_posix()
        try:
            document = load_document(path)
            if "protocol_version" in document:
                validate_document(document)
                protocol_count += 1
            elif "suite_id" in document:
                suite = _load_suite(path)
                _validate_suite_links(path, suite)
                suite_count += 1
            else:
                errors.append(f"{relative}: unrecognized example document")
        except (OSError, ValueError, ProtocolValidationError, TrialWorkerError) as exc:
            errors.append(f"{relative}: {exc}")
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Validated {protocol_count} protocol document(s) and {suite_count} suite(s).")


def _validate_suite_links(path: Path, suite: dict[str, object]) -> None:
    root = path.parent.resolve()
    for case in suite["cases"]:  # type: ignore[index]
        case_id = case["case_id"]
        scenario_path = (root / case["scenario"]).resolve()
        request_path = (root / case["request"]).resolve()
        if not scenario_path.is_relative_to(root) or not request_path.is_relative_to(root):
            raise TrialWorkerError(f"suite case '{case_id}' input escapes the suite directory")
        scenario = load_document(scenario_path)
        request = load_document(request_path)
        validate_document(scenario)
        validate_document(request)
        if scenario.get("document_type") != "scenario":
            raise TrialWorkerError(f"suite case '{case_id}' scenario is not a scenario document")
        if request.get("document_type") != "trial_request":
            raise TrialWorkerError(f"suite case '{case_id}' request is not a trial request")
        reference = request["scenario_ref"]
        if reference["document_id"] != scenario["scenario_id"]:
            raise TrialWorkerError(f"suite case '{case_id}' scenario ID does not match request")
        if reference["version"] != scenario["version"]:
            raise TrialWorkerError(
                f"suite case '{case_id}' scenario version does not match request"
            )
        if reference.get("fingerprint") not in (None, fingerprint(scenario)):
            raise TrialWorkerError(
                f"suite case '{case_id}' scenario fingerprint does not match request"
            )


if __name__ == "__main__":
    main()
