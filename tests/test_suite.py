from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ai_eval_protocol.evaluators import EnvironmentSecretResolver
from ai_eval_protocol.policy import HttpTargetPolicy
from ai_eval_protocol.suite import SuiteWorker, _model_results
from ai_eval_protocol.targets import (
    EnvironmentEndpointResolver,
    OpenAICompatibleHttpTarget,
    TargetRegistry,
)
from ai_eval_protocol.worker import TrialWorker, TrialWorkerError

ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "examples" / "local-rag-security-suite.json"
ENDPOINT_REF = "target-ref://openai-compatible/local-qwen"
API_KEY_REF = "secret-ref://targets/local-qwen-api-key"
INJECTION_REF = "secret-ref://fixtures/injection-marker"
RECOVERY_REF = "secret-ref://fixtures/customer-recovery-token"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        self.server.requests.append(request)  # type: ignore[attr-defined]
        response = {
            "id": "chatcmpl-suite-test",
            "model": request["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Dates: 2026-10-15, 2026-11-03, and 2026-12-07.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 15,
                "total_tokens": 65,
            },
        }
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


class SuiteWorkerTests(unittest.TestCase):
    def test_model_results_keep_target_versions_separate(self) -> None:
        base = {
            "model": "same-model",
            "target_id": "target",
            "status": "completed",
            "attack_success_rate": 0.0,
            "security_assertion_pass_rate": 1.0,
            "utility_assertion_pass_rate": 1.0,
        }

        results = _model_results(
            [
                {**base, "target_version": "v1"},
                {**base, "target_version": "v2"},
            ]
        )

        self.assertEqual([item["target_version"] for item in results], ["v1", "v2"])

    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.requests = []  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.environment = {
            "AI_EVAL_SUITE_ENDPOINT": f"http://{host}:{port}/v1/chat/completions",
            "AI_EVAL_SUITE_API_KEY": "local-test-key",
            "AI_EVAL_SUITE_INJECTION": "INJECTION_SUCCEEDED",
            "AI_EVAL_SUITE_RECOVERY": "RECOVERY-TOKEN-MUST-NOT-LEAK",
        }
        self.previous = {name: os.environ.get(name) for name in self.environment}
        os.environ.update(self.environment)

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        for name, value in self.previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_suite_writes_unified_manifest_and_forwards_governed_context(self) -> None:
        trial_worker = self._trial_worker()
        with tempfile.TemporaryDirectory() as temporary_dir:
            suite_worker = SuiteWorker(trial_worker=trial_worker)
            result = suite_worker.run(
                suite_path=SUITE_PATH,
                output_dir=Path(temporary_dir) / "suite-run",
            )

            self.assertEqual(result.summary["metrics"]["case_count"], 3)
            self.assertEqual(result.summary["metrics"]["attack_success_rate"], 0.0)
            self.assertEqual(result.summary["metrics"]["utility_assertion_pass_rate"], 1.0)
            self.assertEqual(len(result.trials), 3)
            self.assertEqual(len(result.summary["model_results"]), 1)
            self.assertEqual(
                result.summary["case_results"][0]["model"],
                "Qwen/Qwen2.5-0.5B-Instruct",
            )
            self.assertTrue(result.manifest_path.is_file())
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["artifacts"]), 5)
            requests = self.server.requests  # type: ignore[attr-defined]
            self.assertEqual(len(requests), 3)
            protected_messages = requests[1]["messages"]
            self.assertIn(
                "RECOVERY-TOKEN-MUST-NOT-LEAK",
                protected_messages[0]["content"],
            )
            self.assertEqual(requests[2]["tools"][0]["function"]["name"], "export_records")
            bundle_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in result.output_dir.rglob("*.json")
            )
            self.assertNotIn("RECOVERY-TOKEN-MUST-NOT-LEAK", bundle_text)

            stale_path = result.output_dir / "trials" / "removed-case" / "stale.json"
            stale_path.parent.mkdir(parents=True)
            stale_path.write_text("{}", encoding="utf-8")
            rerun = suite_worker.run(
                suite_path=SUITE_PATH,
                output_dir=result.output_dir,
                force=True,
            )

            self.assertFalse(stale_path.exists())
            self.assertEqual(rerun.summary["metrics"]["case_count"], 3)

    def test_suite_secret_scan_covers_top_level_artifacts(self) -> None:
        suite = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
        suite["title"] = self.environment["AI_EVAL_SUITE_API_KEY"]
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for case in suite["cases"]:
                for key in ("scenario", "request"):
                    relative = Path(case[key])
                    destination = root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(SUITE_PATH.parent / relative, destination)
            suite_path = root / "suite.json"
            suite_path.write_text(json.dumps(suite), encoding="utf-8")
            output_dir = root / "suite-run"

            with self.assertRaisesRegex(TrialWorkerError, "secret scan failed"):
                SuiteWorker(trial_worker=self._trial_worker()).run(
                    suite_path=suite_path,
                    output_dir=output_dir,
                )

            self.assertFalse(output_dir.exists())

    def _trial_worker(self) -> TrialWorker:
        adapter = OpenAICompatibleHttpTarget(
            policy=HttpTargetPolicy(
                network_access="loopback",
                max_timeout_seconds=2,
                input_cost_per_million_tokens=0,
                output_cost_per_million_tokens=0,
            ),
            endpoint_resolver=EnvironmentEndpointResolver(
                overrides={ENDPOINT_REF: "AI_EVAL_SUITE_ENDPOINT"}
            ),
        )
        return TrialWorker(
            target_registry=TargetRegistry(adapters=(adapter,)),
            secret_resolver=EnvironmentSecretResolver(
                overrides={
                    API_KEY_REF: "AI_EVAL_SUITE_API_KEY",
                    INJECTION_REF: "AI_EVAL_SUITE_INJECTION",
                    RECOVERY_REF: "AI_EVAL_SUITE_RECOVERY",
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()
