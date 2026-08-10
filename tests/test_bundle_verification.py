from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ai_eval_protocol.bundle_verification import BundleVerificationError, verify_bundle
from ai_eval_protocol.evaluators import EnvironmentSecretResolver
from ai_eval_protocol.suite import SuiteWorker, _aggregate_case_metrics, _model_results
from ai_eval_protocol.worker import TrialWorker

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = ROOT / "examples" / "scenarios" / "rag-indirect-prompt-injection.json"
REQUEST_PATH = ROOT / "examples" / "requests" / "rag-indirect-prompt-injection.request.json"
SECRET_REF = "secret-ref://fixtures/rag-canary"
SECRET_ENV = "AI_EVAL_VERIFY_RAG_CANARY"
CANARY = "VERIFY-CANARY-DO-NOT-DISCLOSE"


class BundleVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_secret = os.environ.get(SECRET_ENV)
        os.environ[SECRET_ENV] = CANARY
        self.worker = TrialWorker(
            secret_resolver=EnvironmentSecretResolver(overrides={SECRET_REF: SECRET_ENV})
        )

    def tearDown(self) -> None:
        if self.previous_secret is None:
            os.environ.pop(SECRET_ENV, None)
        else:
            os.environ[SECRET_ENV] = self.previous_secret

    def test_verifies_real_trial_bundle_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "trial"
            self._run_trial(output_dir)

            result = verify_bundle(output_dir)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ai_eval_protocol",
                    "verify-bundle",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.bundle_type, "trial")
            self.assertEqual(result.artifact_count, 4)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(json.loads(completed.stdout)["valid"])

    def test_rejects_artifact_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "trial"
            self._run_trial(output_dir)
            trace_path = output_dir / "trial_trace.json"
            trace_path.write_text(trace_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

            with self.assertRaisesRegex(BundleVerificationError, "sha256 does not match"):
                verify_bundle(output_dir)

    def test_rejects_reference_tampering_even_with_updated_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "trial"
            self._run_trial(output_dir)
            summary_path = output_dir / "evaluation_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["scenario_ref"]["document_id"] = "different-scenario"
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._refresh_manifest_entry(output_dir, "evaluation_summary.json")

            with self.assertRaisesRegex(BundleVerificationError, "document_id does not match"):
                verify_bundle(output_dir)

    def test_rejects_transaction_residue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            output_dir = root / "trial"
            self._run_trial(output_dir)
            (root / ".trial.staging-interrupted").mkdir()

            with self.assertRaisesRegex(BundleVerificationError, "transaction residue"):
                verify_bundle(output_dir)

    def test_verifies_real_suite_and_child_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            inputs = root / "suite-inputs"
            (inputs / "scenarios").mkdir(parents=True)
            (inputs / "requests").mkdir()
            shutil.copy2(SCENARIO_PATH, inputs / "scenarios" / "scenario.json")
            shutil.copy2(REQUEST_PATH, inputs / "requests" / "request.json")
            suite = {
                "suite_id": "suite-verify-test",
                "version": "0.1.0",
                "title": "Verifier test suite",
                "cases": [
                    {
                        "case_id": "safe-case",
                        "risk": "indirect_prompt_injection",
                        "scenario": "scenarios/scenario.json",
                        "request": "requests/request.json",
                    }
                ],
            }
            suite_path = inputs / "suite.json"
            suite_path.write_text(json.dumps(suite), encoding="utf-8")
            output_dir = root / "suite-run"
            SuiteWorker(trial_worker=self.worker).run(
                suite_path=suite_path,
                output_dir=output_dir,
            )

            result = verify_bundle(output_dir)

            self.assertEqual(result.bundle_type, "suite")
            self.assertEqual(result.child_bundle_count, 1)
            self.assertEqual(result.artifact_count, 7)

            summary_path = output_dir / "suite_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            case_result = summary["case_results"][0]
            case_result["security_assertion_pass_rate"] = 0.0
            summary["metrics"] = _aggregate_case_metrics(summary["case_results"])
            summary["model_results"] = _model_results(summary["case_results"])
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._refresh_manifest_entry(output_dir, "suite_summary.json")

            with self.assertRaisesRegex(BundleVerificationError, "differs from child trial"):
                verify_bundle(output_dir)

    def _run_trial(self, output_dir: Path) -> None:
        self.worker.run(
            scenario_path=SCENARIO_PATH,
            request_path=REQUEST_PATH,
            output_dir=output_dir,
        )

    def _refresh_manifest_entry(self, output_dir: Path, relative_path: str) -> None:
        manifest_path = output_dir / "bundle_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for artifact in manifest["artifacts"]:
            if artifact["path"] == relative_path:
                artifact_path = output_dir / relative_path
                artifact["sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                break
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
