from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ai_eval_protocol.codec import fingerprint, load_document
from ai_eval_protocol.validation import ProtocolValidationError, validate_document

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "examples" / "scenarios" / "rag-indirect-prompt-injection.json"
TRIAL = ROOT / "examples" / "trials" / "rag-indirect-prompt-injection.blocked.json"
REQUEST = ROOT / "examples" / "requests" / "rag-indirect-prompt-injection.request.json"
HTTP_REQUEST = (
    ROOT / "examples" / "requests" / "rag-indirect-prompt-injection.http-request.json"
)
SUMMARY = ROOT / "examples" / "summaries" / "rag-indirect-prompt-injection.summary.json"


class ProtocolValidationTests(unittest.TestCase):
    def test_example_scenario_is_valid(self) -> None:
        validate_document(load_document(SCENARIO))

    def test_example_trial_is_valid(self) -> None:
        validate_document(load_document(TRIAL))

    def test_example_request_is_valid(self) -> None:
        validate_document(load_document(REQUEST))

    def test_example_http_request_is_valid(self) -> None:
        validate_document(load_document(HTTP_REQUEST))

    def test_example_summary_is_valid(self) -> None:
        validate_document(load_document(SUMMARY))

    def test_fingerprint_is_independent_of_key_order_and_spacing(self) -> None:
        document = load_document(SCENARIO)
        reordered = json.loads(json.dumps(document, indent=7, sort_keys=False))
        self.assertEqual(fingerprint(document), fingerprint(reordered))

    def test_unknown_protocol_version_fails_closed(self) -> None:
        document = load_document(SCENARIO)
        document["protocol_version"] = "0.2"

        with self.assertRaisesRegex(ProtocolValidationError, "must be '0.1'"):
            validate_document(document)

    def test_inline_target_credential_is_rejected(self) -> None:
        document = load_document(TRIAL)
        document["target"]["api_key"] = "must-not-appear"

        with self.assertRaisesRegex(ProtocolValidationError, "never inline credentials"):
            validate_document(document)

    def test_nested_inline_target_credential_is_rejected(self) -> None:
        document = load_document(HTTP_REQUEST)
        document["target"]["metadata"]["auth"] = {"api_key": "must-not-appear"}

        with self.assertRaisesRegex(ProtocolValidationError, "never inline credentials"):
            validate_document(document)

    def test_inline_credential_is_rejected_outside_target(self) -> None:
        document = load_document(SCENARIO)
        document["assertions"][0]["parameters"]["client_secret"] = {
            "value": "must-not-appear"
        }

        with self.assertRaisesRegex(ProtocolValidationError, "never inline credentials"):
            validate_document(document)

    def test_target_metadata_must_be_an_object(self) -> None:
        document = load_document(HTTP_REQUEST)
        document["target"]["metadata"] = []

        with self.assertRaisesRegex(ProtocolValidationError, "metadata must be an object"):
            validate_document(document)

    def test_document_reference_requires_full_sha256(self) -> None:
        document = load_document(HTTP_REQUEST)
        document["scenario_ref"]["fingerprint"] = "sha256:x"

        with self.assertRaisesRegex(ProtocolValidationError, "64 lowercase hex"):
            validate_document(document)

    def test_unknown_protocol_field_is_rejected(self) -> None:
        document = load_document(SCENARIO)
        document["unexpected"] = True

        with self.assertRaisesRegex(ProtocolValidationError, "unsupported field"):
            validate_document(document)

    def test_zero_timeout_is_rejected(self) -> None:
        document = load_document(HTTP_REQUEST)
        document["limits"]["timeout_seconds"] = 0

        with self.assertRaisesRegex(ProtocolValidationError, "greater than zero"):
            validate_document(document)

    def test_loader_rejects_non_finite_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "request.json"
            path.write_text('{"value": NaN}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_document(path)

    def test_loader_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "request.json"
            path.write_text('{"protocol_version":"0.1","protocol_version":"0.1"}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_document(path)

    def test_pass_without_evidence_is_rejected(self) -> None:
        document = load_document(TRIAL)
        document["assertion_results"][0]["evidence"] = []

        with self.assertRaisesRegex(ProtocolValidationError, "required for pass or fail"):
            validate_document(document)

    def test_dangling_evidence_reference_is_rejected(self) -> None:
        document = load_document(TRIAL)
        document["assertion_results"][0]["evidence"][0]["ref_id"] = "missing-event"

        with self.assertRaisesRegex(ProtocolValidationError, "does not reference an event"):
            validate_document(document)

    def test_event_sequence_must_be_contiguous(self) -> None:
        document = load_document(TRIAL)
        document["events"][2]["sequence"] = 7

        with self.assertRaisesRegex(ProtocolValidationError, "must be contiguous"):
            validate_document(document)

    def test_probabilistic_evaluator_requires_model_provenance(self) -> None:
        document = load_document(SCENARIO)
        evaluator = document["assertions"][0]["evaluator"]
        evaluator["kind"] = "probabilistic"

        with self.assertRaisesRegex(ProtocolValidationError, "model_ref is required"):
            validate_document(document)

    def test_cli_validates_real_example_file(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "ai_eval_protocol", "validate", str(TRIAL)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("valid", completed.stdout)

    def test_duplicate_assertion_id_is_rejected(self) -> None:
        document = load_document(SCENARIO)
        duplicate = copy.deepcopy(document["assertions"][0])
        document["assertions"].append(duplicate)

        with self.assertRaisesRegex(ProtocolValidationError, "must be unique"):
            validate_document(document)

    def test_utility_control_must_reference_an_assertion(self) -> None:
        document = load_document(SCENARIO)
        document["controls"]["utility_assertion_id"] = "missing"

        with self.assertRaisesRegex(ProtocolValidationError, "must reference an assertion"):
            validate_document(document)

    def test_scenario_requires_a_security_assertion(self) -> None:
        document = load_document(SCENARIO)
        utility_id = document["controls"]["utility_assertion_id"]
        document["assertions"] = [
            assertion
            for assertion in document["assertions"]
            if assertion["assertion_id"] == utility_id
        ]

        with self.assertRaisesRegex(ProtocolValidationError, "security assertion"):
            validate_document(document)


if __name__ == "__main__":
    unittest.main()
