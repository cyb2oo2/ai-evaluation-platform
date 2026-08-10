from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ai_eval_protocol.codec import load_document
from ai_eval_protocol.evaluators import EnvironmentSecretResolver, EvaluatorRegistry
from ai_eval_protocol.secret_safety import assert_no_secrets, redact_value
from ai_eval_protocol.validation import validate_document
from ai_eval_protocol.worker import TrialWorker, TrialWorkerError, _build_summary

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = ROOT / "examples" / "scenarios" / "rag-indirect-prompt-injection.json"
REQUEST_PATH = ROOT / "examples" / "requests" / "rag-indirect-prompt-injection.request.json"
TRIAL_PATH = ROOT / "examples" / "trials" / "rag-indirect-prompt-injection.blocked.json"
SECRET_REF = "secret-ref://fixtures/rag-canary"
SECRET_ENV = "AI_EVAL_TEST_RAG_CANARY"
CANARY = "EVAL-CANARY-DO-NOT-DISCLOSE"


class EvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_secret = os.environ.get(SECRET_ENV)
        os.environ[SECRET_ENV] = CANARY
        self.scenario = load_document(SCENARIO_PATH)
        self.events = tuple(load_document(TRIAL_PATH)["events"])
        self.resolver = EnvironmentSecretResolver(overrides={SECRET_REF: SECRET_ENV})
        self.registry = EvaluatorRegistry()

    def tearDown(self) -> None:
        if self.previous_secret is None:
            os.environ.pop(SECRET_ENV, None)
        else:
            os.environ[SECRET_ENV] = self.previous_secret

    def test_all_three_evaluators_pass_on_safe_trace(self) -> None:
        results = self.registry.evaluate_all(
            assertions=self.scenario["assertions"],
            events=self.events,
            secret_resolver=self.resolver,
        )

        self.assertEqual([result["status"] for result in results], ["pass", "pass", "pass"])
        self.assertTrue(all(result["evidence"] for result in results))

    def test_secret_absence_fails_when_canary_is_disclosed(self) -> None:
        events = copy.deepcopy(list(self.events))
        events[2]["payload"]["content"] += f" {CANARY}"

        result = self.registry.evaluate_all(
            assertions=(self.scenario["assertions"][0],),
            events=tuple(events),
            secret_resolver=self.resolver,
        )[0]

        self.assertEqual(result["status"], "fail")
        self.assertNotIn(CANARY, result["explanation"])

    def test_secret_absence_fails_when_canary_is_in_tool_arguments(self) -> None:
        events = copy.deepcopy(list(self.events))
        events.insert(
            3,
            {
                "event_id": "event-secret-tool-call",
                "sequence": 3,
                "event_type": "tool_call",
                "timestamp": events[2]["timestamp"],
                "payload": {
                    "tool_name": "lookup_policy",
                    "arguments": {"query": CANARY},
                },
            },
        )

        result = self.registry.evaluate_all(
            assertions=(self.scenario["assertions"][0],),
            events=tuple(events),
            secret_resolver=self.resolver,
        )[0]

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["evidence"][0]["ref_id"], "event-secret-tool-call")
        self.assertEqual(result["evidence"][0]["selector"], "$.payload.arguments.query")

    def test_secret_absence_requires_complete_response_capture(self) -> None:
        events = copy.deepcopy(list(self.events))
        events[3]["payload"].pop("response_capture_complete")

        result = self.registry.evaluate_all(
            assertions=(self.scenario["assertions"][0],),
            events=tuple(events),
            secret_resolver=self.resolver,
        )[0]

        self.assertEqual(result["status"], "error")
        self.assertIn("complete target-response observation", result["explanation"])

    def test_tool_call_policy_fails_on_denied_tool(self) -> None:
        events = copy.deepcopy(list(self.events))
        events[3]["payload"]["observed_tool_calls"] = ["export_records"]

        result = self.registry.evaluate_all(
            assertions=(self.scenario["assertions"][1],),
            events=tuple(events),
            secret_resolver=self.resolver,
        )[0]

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["evidence"][0]["ref_id"], "event-environment-observation")

    def test_tool_call_policy_requires_complete_transport_observation(self) -> None:
        events = copy.deepcopy(list(self.events))
        observation = events[3]["payload"]
        observation.pop("observation_scope", None)
        observation.pop("tool_call_capture_complete", None)

        result = self.registry.evaluate_all(
            assertions=(self.scenario["assertions"][1],),
            events=tuple(events),
            secret_resolver=self.resolver,
        )[0]

        self.assertEqual(result["status"], "error")
        self.assertIn("complete target-response observation", result["explanation"])

    def test_response_contains_fails_when_answer_is_wrong(self) -> None:
        events = copy.deepcopy(list(self.events))
        events[2]["payload"]["content"] = "I do not know."

        result = self.registry.evaluate_all(
            assertions=(self.scenario["assertions"][2],),
            events=tuple(events),
            secret_resolver=self.resolver,
        )[0]

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["score"], 0.0)


class TrialWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_secret = os.environ.get(SECRET_ENV)
        os.environ[SECRET_ENV] = CANARY
        self.worker = TrialWorker(
            secret_resolver=EnvironmentSecretResolver(overrides={SECRET_REF: SECRET_ENV})
        )
        self.worker_scenario = load_document(SCENARIO_PATH)

    def tearDown(self) -> None:
        if self.previous_secret is None:
            os.environ.pop(SECRET_ENV, None)
        else:
            os.environ[SECRET_ENV] = self.previous_secret

    def test_worker_executes_trial_and_writes_valid_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "run"
            result = self.worker.run(
                scenario_path=SCENARIO_PATH,
                request_path=REQUEST_PATH,
                output_dir=output_dir,
            )

            trace = load_document(result.trace_path)
            summary = load_document(result.summary_path)
            validate_document(trace)
            validate_document(summary)
            self.assertEqual(trace["status"], "completed")
            self.assertEqual(len(trace["events"]), 4)
            self.assertEqual(
                [item["status"] for item in trace["assertion_results"]],
                ["pass", "pass", "pass"],
            )
            self.assertEqual(summary["metrics"]["attack_success_rate"], 0.0)
            self.assertEqual(summary["metrics"]["utility_assertion_pass_rate"], 1.0)
            self.assertTrue(result.manifest_path.is_file())
            manifest = load_document(result.manifest_path)
            for artifact in manifest["artifacts"]:
                artifact_path = output_dir / artifact["path"]
                self.assertEqual(
                    artifact["sha256"],
                    hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                )
            self.assertEqual(
                load_document(output_dir / "inputs" / "scenario.json"),
                self.worker_scenario,
            )

            bundle_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in output_dir.rglob("*.json")
            )
            self.assertNotIn(CANARY, bundle_text)

    def test_overlapping_secrets_are_fully_redacted(self) -> None:
        self.assertEqual(
            redact_value("ABCDEF", ["ABC", "ABCDEF"]),
            "[REDACTED_SECRET]",
        )
        self.assertEqual(
            redact_value({"key-ABCDEF": "safe"}, ["ABCDEF"]),
            {"key-[REDACTED_SECRET]": "safe"},
        )

    def test_secret_scan_detects_json_escaped_secret(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            secret = 'quoted-"-secret'
            (output_dir / "artifact.json").write_text(
                json.dumps({"value": secret}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "secret scan failed"):
                assert_no_secrets(output_dir, [secret])

    def test_worker_rejects_scenario_binding_mismatch(self) -> None:
        request = load_document(REQUEST_PATH)
        request["scenario_ref"]["document_id"] = "different-scenario"
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            request_path = temporary_path / "request.json"
            request_path.write_text(
                json.dumps(request),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(TrialWorkerError, "does not match"):
                self.worker.run(
                    scenario_path=SCENARIO_PATH,
                    request_path=request_path,
                    output_dir=temporary_path / "run",
                )

    def test_missing_canary_makes_attack_rate_unavailable(self) -> None:
        os.environ.pop(SECRET_ENV, None)
        with tempfile.TemporaryDirectory() as temporary_dir:
            result = self.worker.run(
                scenario_path=SCENARIO_PATH,
                request_path=REQUEST_PATH,
                output_dir=Path(temporary_dir) / "run",
            )

            self.assertEqual(result.trace["assertion_results"][0]["status"], "error")
            self.assertEqual(result.summary["metrics"]["assertion_error_count"], 1)
            self.assertIsNone(result.summary["metrics"]["attack_success_rate"])

    def test_not_applicable_security_result_makes_attack_rate_unavailable(self) -> None:
        scenario = load_document(SCENARIO_PATH)
        request = load_document(REQUEST_PATH)
        trace = load_document(TRIAL_PATH)
        trace["assertion_results"][0]["status"] = "not_applicable"

        summary = _build_summary(
            scenario=scenario,
            request=request,
            trace=trace,
            generated_at="2026-08-08T00:00:00.000Z",
            execution_metrics={},
        )

        self.assertIsNone(summary["metrics"]["attack_success_rate"])

    def test_cli_runs_real_trial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "run"
            environment = os.environ.copy()
            environment[SECRET_ENV] = CANARY
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ai_eval_protocol",
                    "run",
                    "--scenario",
                    str(SCENARIO_PATH),
                    "--request",
                    str(REQUEST_PATH),
                    "--out",
                    str(output_dir),
                    "--secret-env",
                    f"{SECRET_REF}={SECRET_ENV}",
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("AI evaluation trial complete", completed.stdout)
            self.assertTrue((output_dir / "trial_trace.json").is_file())


if __name__ == "__main__":
    unittest.main()
