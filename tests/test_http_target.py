from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ai_eval_protocol.codec import load_document
from ai_eval_protocol.evaluators import EnvironmentSecretResolver
from ai_eval_protocol.policy import HttpPolicyError, HttpTargetPolicy
from ai_eval_protocol.targets import (
    EnvironmentEndpointResolver,
    OpenAICompatibleHttpTarget,
    TargetRegistry,
)
from ai_eval_protocol.worker import TrialWorker, TrialWorkerError

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = ROOT / "examples" / "scenarios" / "rag-indirect-prompt-injection.json"
REQUEST_PATH = (
    ROOT / "examples" / "requests" / "rag-indirect-prompt-injection.http-request.json"
)
ENDPOINT_REF = "target-ref://openai-compatible/local"
SECRET_REF = "secret-ref://targets/openai-compatible-demo"
CANARY_REF = "secret-ref://fixtures/rag-canary"
ENDPOINT_ENV = "AI_EVAL_TEST_HTTP_ENDPOINT"
API_KEY_ENV = "AI_EVAL_TEST_HTTP_API_KEY"
CANARY_ENV = "AI_EVAL_TEST_HTTP_CANARY"
API_KEY = "test-api-key-must-not-leak"
CANARY = "test-canary-must-not-leak"


class _OpenAICompatibleHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        self.server.captured_requests.append(  # type: ignore[attr-defined]
            {
                "path": self.path,
                "authorization": self.headers.get("Authorization", ""),
                "body": json.loads(body.decode("utf-8")),
            }
        )
        delay = self.server.delay_seconds  # type: ignore[attr-defined]
        if delay:
            time.sleep(delay)
        payload = self.server.response_body  # type: ignore[attr-defined]
        status = self.server.response_status  # type: ignore[attr-defined]
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        content_length = self.server.content_length_header  # type: ignore[attr-defined]
        self.send_header(
            "Content-Length",
            str(len(payload)) if content_length is None else content_length,
        )
        self.end_headers()
        with suppress(OSError):
            self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


class _ServerHarness:
    def __init__(
        self,
        response: dict[str, object],
        *,
        delay_seconds: float = 0.0,
        content_length_header: str | None = None,
    ) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAICompatibleHandler)
        self.server.response_body = json.dumps(response).encode("utf-8")  # type: ignore[attr-defined]
        self.server.response_status = 200  # type: ignore[attr-defined]
        self.server.delay_seconds = delay_seconds  # type: ignore[attr-defined]
        self.server.content_length_header = content_length_header  # type: ignore[attr-defined]
        self.server.captured_requests = []  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def endpoint(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/v1/chat/completions"

    @property
    def requests(self) -> list[dict[str, object]]:
        return self.server.captured_requests  # type: ignore[attr-defined,no-any-return]

    def __enter__(self) -> _ServerHarness:
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def _successful_response() -> dict[str, object]:
    return {
        "id": "chatcmpl-local-test",
        "model": "example-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The new expense policy takes effect on 2026-09-01.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        },
    }


class HttpPolicyTests(unittest.TestCase):
    def test_network_is_disabled_by_default(self) -> None:
        with self.assertRaisesRegex(HttpPolicyError, "network access is disabled"):
            HttpTargetPolicy().authorize_endpoint("http://127.0.0.1/v1/chat/completions")

    def test_allowlist_rejects_private_address_even_when_origin_matches(self) -> None:
        policy = HttpTargetPolicy(
            network_access="allowlist",
            allowed_origins=("https://127.0.0.1",),
        )
        with self.assertRaisesRegex(HttpPolicyError, "public addresses"):
            policy.authorize_endpoint("https://127.0.0.1/v1/chat/completions")

    def test_allowlist_requires_exact_chat_completions_path(self) -> None:
        policy = HttpTargetPolicy(network_access="loopback")
        with self.assertRaisesRegex(HttpPolicyError, "path must be"):
            policy.authorize_endpoint("http://127.0.0.1/v1/models")

    def test_endpoint_query_is_rejected(self) -> None:
        policy = HttpTargetPolicy(network_access="loopback")
        with self.assertRaisesRegex(HttpPolicyError, "must not contain a query"):
            policy.authorize_endpoint(
                "http://127.0.0.1/v1/chat/completions?redirect=http://metadata"
            )

    def test_non_finite_policy_values_are_rejected(self) -> None:
        policies = (
            HttpTargetPolicy(max_timeout_seconds=math.nan),
            HttpTargetPolicy(max_cost_usd=math.inf),
            HttpTargetPolicy(input_cost_per_million_tokens=math.nan),
            HttpTargetPolicy(output_cost_per_million_tokens=-math.inf),
        )

        for policy in policies:
            with self.subTest(policy=policy), self.assertRaisesRegex(
                HttpPolicyError, "finite"
            ):
                policy.validate()

    def test_reported_token_counts_reject_booleans(self) -> None:
        policy = HttpTargetPolicy(
            input_cost_per_million_tokens=0,
            output_cost_per_million_tokens=0,
        )

        with self.assertRaisesRegex(HttpPolicyError, "non-negative integer"):
            policy.estimate_cost(prompt_tokens=True, completion_tokens=0)


class HttpTargetEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = {
            name: os.environ.get(name)
            for name in (ENDPOINT_ENV, API_KEY_ENV, CANARY_ENV)
        }
        os.environ[API_KEY_ENV] = API_KEY
        os.environ[CANARY_ENV] = CANARY

    def tearDown(self) -> None:
        for name, value in self.previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_http_target_runs_real_local_request_and_records_usage(self) -> None:
        with _ServerHarness(_successful_response()) as server:
            os.environ[ENDPOINT_ENV] = server.endpoint
            worker = self._worker()
            with tempfile.TemporaryDirectory() as temporary_dir:
                result = worker.run(
                    scenario_path=SCENARIO_PATH,
                    request_path=REQUEST_PATH,
                    output_dir=Path(temporary_dir) / "run",
                )

                self.assertEqual(result.trace["status"], "completed")
                self.assertEqual(
                    [item["status"] for item in result.trace["assertion_results"]],
                    ["pass", "pass", "pass"],
                )
                self.assertEqual(result.summary["metrics"]["total_tokens"], 120)
                self.assertEqual(result.summary["metrics"]["estimated_cost_usd"], 0.00014)
                self.assertEqual(result.summary["metrics"]["attack_success_rate"], 0.0)
                self.assertEqual(len(server.requests), 1)
                captured = server.requests[0]
                self.assertEqual(captured["authorization"], f"Bearer {API_KEY}")
                request_body = captured["body"]
                self.assertEqual(request_body["model"], "example-model")
                self.assertIn("untrusted data", request_body["messages"][0]["content"])
                observation = next(
                    event
                    for event in result.trace["events"]
                    if event["event_type"] == "environment"
                )
                self.assertEqual(
                    observation["payload"]["observation_scope"],
                    "target_response",
                )
                self.assertTrue(observation["payload"]["tool_call_capture_complete"])
                self.assertTrue(observation["payload"]["response_capture_complete"])
                self.assertNotIn("external_side_effects", observation["payload"])

                bundle_text = "\n".join(
                    path.read_text(encoding="utf-8")
                    for path in result.output_dir.rglob("*.json")
                )
                self.assertNotIn(API_KEY, bundle_text)
                self.assertNotIn(CANARY, bundle_text)

    def test_response_size_limit_fails_closed(self) -> None:
        with _ServerHarness(_successful_response()) as server:
            os.environ[ENDPOINT_ENV] = server.endpoint
            worker = self._worker(max_response_bytes=32)
            with tempfile.TemporaryDirectory() as temporary_dir:
                result = worker.run(
                    scenario_path=SCENARIO_PATH,
                    request_path=REQUEST_PATH,
                    output_dir=Path(temporary_dir) / "run",
                )

                self.assertEqual(result.trace["status"], "error")
                self.assertIsNone(result.summary["metrics"]["attack_success_rate"])
                self.assertTrue(
                    any(
                        event["payload"].get("error_type") == "response-limit"
                        for event in result.trace["events"]
                    )
                )

    def test_malformed_content_length_fails_closed(self) -> None:
        with _ServerHarness(
            _successful_response(), content_length_header="not-an-integer"
        ) as server:
            os.environ[ENDPOINT_ENV] = server.endpoint
            with tempfile.TemporaryDirectory() as temporary_dir:
                result = self._worker().run(
                    scenario_path=SCENARIO_PATH,
                    request_path=REQUEST_PATH,
                    output_dir=Path(temporary_dir) / "run",
                )

                self.assertEqual(result.trace["status"], "error")
                self.assertTrue(
                    any(
                        event["payload"].get("error_type") == "invalid-response"
                        for event in result.trace["events"]
                    )
                )

    def test_leaked_canary_is_detected_but_redacted_from_bundle(self) -> None:
        response = _successful_response()
        response["choices"][0]["message"]["content"] += f" {CANARY} {API_KEY}"
        with _ServerHarness(response) as server:
            os.environ[ENDPOINT_ENV] = server.endpoint
            with tempfile.TemporaryDirectory() as temporary_dir:
                result = self._worker().run(
                    scenario_path=SCENARIO_PATH,
                    request_path=REQUEST_PATH,
                    output_dir=Path(temporary_dir) / "run",
                )

                self.assertEqual(result.trace["assertion_results"][0]["status"], "fail")
                self.assertEqual(result.summary["metrics"]["attack_success_rate"], 1.0)
                trace_text = result.trace_path.read_text(encoding="utf-8")
                self.assertIn("[REDACTED_SECRET]", trace_text)
                self.assertNotIn(CANARY, trace_text)
                self.assertNotIn(API_KEY, trace_text)

    def test_unauthorized_http_tool_call_is_captured_and_fails_policy(self) -> None:
        response = _successful_response()
        response["choices"][0]["message"]["tool_calls"] = [
            {
                "id": "call-export",
                "type": "function",
                "function": {
                    "name": "export_records",
                    "arguments": '{"scope":"all_customers"}',
                },
            }
        ]
        with _ServerHarness(response) as server:
            os.environ[ENDPOINT_ENV] = server.endpoint
            with tempfile.TemporaryDirectory() as temporary_dir:
                result = self._worker().run(
                    scenario_path=SCENARIO_PATH,
                    request_path=REQUEST_PATH,
                    output_dir=Path(temporary_dir) / "run",
                )

                tool_events = [
                    event
                    for event in result.trace["events"]
                    if event["event_type"] == "tool_call"
                ]
                self.assertEqual(tool_events[0]["payload"]["tool_name"], "export_records")
                self.assertEqual(result.trace["assertion_results"][1]["status"], "fail")
                self.assertEqual(result.summary["metrics"]["attack_success_rate"], 1.0)

    def test_timeout_fails_closed(self) -> None:
        with _ServerHarness(_successful_response(), delay_seconds=0.2) as server:
            os.environ[ENDPOINT_ENV] = server.endpoint
            worker = self._worker(max_timeout_seconds=0.05)
            with tempfile.TemporaryDirectory() as temporary_dir:
                result = worker.run(
                    scenario_path=SCENARIO_PATH,
                    request_path=REQUEST_PATH,
                    output_dir=Path(temporary_dir) / "run",
                )

                self.assertEqual(result.trace["status"], "error")
                self.assertIsNone(result.summary["metrics"]["attack_success_rate"])

    def test_cost_budget_is_enforced_from_reported_usage(self) -> None:
        response = _successful_response()
        response["usage"] = {
            "prompt_tokens": 1,
            "completion_tokens": 1000,
            "total_tokens": 1001,
        }
        with _ServerHarness(response) as server:
            os.environ[ENDPOINT_ENV] = server.endpoint
            request = load_document(REQUEST_PATH)
            request["limits"]["max_tokens"] = 1
            request["limits"]["max_cost_usd"] = 0.0001
            with tempfile.TemporaryDirectory() as temporary_dir:
                root = Path(temporary_dir)
                request_path = root / "request.json"
                request_path.write_text(json.dumps(request), encoding="utf-8")
                result = self._worker().run(
                    scenario_path=SCENARIO_PATH,
                    request_path=request_path,
                    output_dir=root / "run",
                )

                self.assertEqual(result.trace["status"], "budget_exceeded")
                self.assertGreater(result.summary["metrics"]["estimated_cost_usd"], 0.0001)
                self.assertIsNone(result.summary["metrics"]["attack_success_rate"])

    def test_completion_token_budget_is_enforced_after_response(self) -> None:
        response = _successful_response()
        response["usage"] = {
            "prompt_tokens": 1,
            "completion_tokens": 1000,
            "total_tokens": 1001,
        }
        with _ServerHarness(response) as server:
            os.environ[ENDPOINT_ENV] = server.endpoint
            request = load_document(REQUEST_PATH)
            request["limits"]["max_tokens"] = 1
            request["limits"]["max_cost_usd"] = 1.0
            with tempfile.TemporaryDirectory() as temporary_dir:
                root = Path(temporary_dir)
                request_path = root / "request.json"
                request_path.write_text(json.dumps(request), encoding="utf-8")
                result = self._worker().run(
                    scenario_path=SCENARIO_PATH,
                    request_path=request_path,
                    output_dir=root / "run",
                )

                self.assertEqual(result.trace["status"], "budget_exceeded")
                self.assertTrue(
                    any(
                        event["payload"].get("error_type") == "completion-token-budget"
                        for event in result.trace["events"]
                    )
                )
                self.assertIsNone(result.summary["metrics"]["attack_success_rate"])

    def test_inconsistent_total_token_usage_fails_closed(self) -> None:
        response = _successful_response()
        response["usage"]["total_tokens"] = 999
        with _ServerHarness(response) as server:
            os.environ[ENDPOINT_ENV] = server.endpoint
            with tempfile.TemporaryDirectory() as temporary_dir:
                result = self._worker().run(
                    scenario_path=SCENARIO_PATH,
                    request_path=REQUEST_PATH,
                    output_dir=Path(temporary_dir) / "run",
                )

                self.assertEqual(result.trace["status"], "error")
                self.assertTrue(
                    any(
                        event["payload"].get("error_type") == "invalid-response"
                        for event in result.trace["events"]
                    )
                )

    def test_disabled_network_blocks_before_any_request(self) -> None:
        with _ServerHarness(_successful_response()) as server:
            os.environ[ENDPOINT_ENV] = server.endpoint
            worker = self._worker(network_access="disabled")
            with (
                tempfile.TemporaryDirectory() as temporary_dir,
                self.assertRaisesRegex(TrialWorkerError, "network access is disabled"),
            ):
                worker.run(
                    scenario_path=SCENARIO_PATH,
                    request_path=REQUEST_PATH,
                    output_dir=Path(temporary_dir) / "run",
                )
            self.assertEqual(server.requests, [])

    def test_cli_runs_http_target_with_explicit_loopback_policy(self) -> None:
        with _ServerHarness(_successful_response()) as server:
            environment = os.environ.copy()
            environment[ENDPOINT_ENV] = server.endpoint
            environment[API_KEY_ENV] = API_KEY
            environment[CANARY_ENV] = CANARY
            with tempfile.TemporaryDirectory() as temporary_dir:
                output_dir = Path(temporary_dir) / "run"
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
                        "--network-access",
                        "loopback",
                        "--endpoint-env",
                        f"{ENDPOINT_REF}={ENDPOINT_ENV}",
                        "--secret-env",
                        f"{SECRET_REF}={API_KEY_ENV}",
                        "--secret-env",
                        f"{CANARY_REF}={CANARY_ENV}",
                        "--input-cost-per-million",
                        "1.0",
                        "--output-cost-per-million",
                        "2.0",
                    ],
                    cwd=ROOT,
                    env=environment,
                    text=True,
                    capture_output=True,
                    check=False,
                )

                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("attack success rate 0.00", completed.stdout)
                self.assertTrue((output_dir / "trial_trace.json").is_file())

    def _worker(
        self,
        *,
        network_access: str = "loopback",
        max_timeout_seconds: float = 1.0,
        max_response_bytes: int = 100_000,
    ) -> TrialWorker:
        policy = HttpTargetPolicy(
            network_access=network_access,  # type: ignore[arg-type]
            max_timeout_seconds=max_timeout_seconds,
            max_response_bytes=max_response_bytes,
            max_cost_usd=1.0,
            input_cost_per_million_tokens=1.0,
            output_cost_per_million_tokens=2.0,
        )
        adapter = OpenAICompatibleHttpTarget(
            policy=policy,
            endpoint_resolver=EnvironmentEndpointResolver(
                overrides={ENDPOINT_REF: ENDPOINT_ENV}
            ),
        )
        return TrialWorker(
            target_registry=TargetRegistry(adapters=(adapter,)),
            secret_resolver=EnvironmentSecretResolver(
                overrides={SECRET_REF: API_KEY_ENV, CANARY_REF: CANARY_ENV}
            ),
        )


if __name__ == "__main__":
    unittest.main()
