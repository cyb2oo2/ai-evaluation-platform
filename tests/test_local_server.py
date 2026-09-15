from __future__ import annotations

import http.client
import json
import socket
import threading
import unittest

from tools.local_openai_server import (
    LocalEvaluationServer,
    ServerLimits,
    _assistant_message,
    _check_prompt_tokens,
    _load_request_json,
    _requested_max_tokens,
)


class ServerLimitsTests(unittest.TestCase):
    def test_limits_accept_positive_bounded_values(self) -> None:
        limits = ServerLimits(
            max_request_bytes=1,
            max_input_tokens=1,
            max_generation_tokens=1,
            max_connections=1,
            max_concurrent_requests=1,
            request_read_timeout_seconds=0.1,
        )

        limits.validate()

    def test_limits_reject_invalid_values(self) -> None:
        invalid = (
            ServerLimits(max_request_bytes=0),
            ServerLimits(max_input_tokens=0),
            ServerLimits(max_generation_tokens=0),
            ServerLimits(max_connections=0),
            ServerLimits(max_concurrent_requests=0),
            ServerLimits(request_read_timeout_seconds=0.0),
            ServerLimits(request_read_timeout_seconds=float("nan")),
        )

        for limits in invalid:
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                limits.validate()

    def test_request_json_rejects_duplicate_keys_and_non_finite_numbers(self) -> None:
        for body in (b'{"model":"a","model":"b"}', b'{"temperature":NaN}'):
            with self.subTest(body=body), self.assertRaises(ValueError):
                _load_request_json(body)

    def test_token_limits_reject_oversized_input_and_generation(self) -> None:
        limits = ServerLimits(max_input_tokens=4, max_generation_tokens=2)

        with self.assertRaisesRegex(ValueError, "max_tokens exceeds"):
            _requested_max_tokens({"max_tokens": 3}, limits=limits)
        with self.assertRaisesRegex(ValueError, "input has 5 tokens"):
            _check_prompt_tokens(5, limits=limits)


class _BlockingModel:
    model_ref = "fake-local-model"

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def complete(self, request: dict[str, object]) -> dict[str, object]:
        del request
        self.started.set()
        self.release.wait(timeout=2)
        return {"choices": [], "usage": {}}


class _ImmediateModel:
    model_ref = "fake-local-model"

    def complete(self, request: dict[str, object]) -> dict[str, object]:
        del request
        return {"choices": [], "usage": {}}


class _ErrorCountingServer(LocalEvaluationServer):
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.handler_error_count = 0
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    def handle_error(self, request: object, client_address: object) -> None:
        del request, client_address
        self.handler_error_count += 1


class _AdmissionObservedServer(LocalEvaluationServer):
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.admitted = threading.Event()
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    def process_request_thread(
        self, request: socket.socket, client_address: tuple[object, ...]
    ) -> None:
        self.admitted.set()
        super().process_request_thread(request, client_address)


class LocalServerConcurrencyTests(unittest.TestCase):
    def test_busy_server_rejects_instead_of_queueing_unbounded_work(self) -> None:
        model = _BlockingModel()
        server = LocalEvaluationServer(
            ("127.0.0.1", 0),
            model=model,  # type: ignore[arg-type]
            api_key="test-key",
            limits=ServerLimits(max_concurrent_requests=1),
        )
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        first_status: list[int] = []

        def first_request() -> None:
            first_status.append(self._post(server.server_port))

        try:
            server_thread.start()
            request_thread = threading.Thread(target=first_request)
            request_thread.start()
            self.assertTrue(model.started.wait(timeout=1))

            self.assertEqual(self._post(server.server_port), 429)
            model.release.set()
            request_thread.join(timeout=2)
            self.assertEqual(first_status, [200])
        finally:
            model.release.set()
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)

    def test_connection_limit_rejects_before_reading_another_body(self) -> None:
        server = _AdmissionObservedServer(
            ("127.0.0.1", 0),
            model=_ImmediateModel(),  # type: ignore[arg-type]
            api_key="test-key",
            limits=ServerLimits(
                max_connections=1,
                request_read_timeout_seconds=5,
            ),
        )
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        first = socket.create_connection(("127.0.0.1", server.server_port), timeout=2)
        try:
            server_thread.start()
            first.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Authorization: Bearer test-key\r\n"
                b"Content-Length: 100\r\n\r\n{"
            )
            self.assertTrue(server.admitted.wait(timeout=2))

            second = socket.create_connection(("127.0.0.1", server.server_port), timeout=2)
            try:
                # Admission rejects before any request bytes are read. Sending
                # bytes while the server closes can produce a Windows TCP reset.
                response = second.recv(4096)
            finally:
                second.close()

            self.assertIn(b" 503 ", response)
        finally:
            first.close()
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)

    def test_slow_body_returns_408_without_handler_error(self) -> None:
        server = _ErrorCountingServer(
            ("127.0.0.1", 0),
            model=_ImmediateModel(),
            api_key="test-key",
            limits=ServerLimits(request_read_timeout_seconds=0.05),
        )
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        client = socket.create_connection(("127.0.0.1", server.server_port), timeout=2)
        try:
            server_thread.start()
            client.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Authorization: Bearer test-key\r\n"
                b"Content-Length: 10\r\n\r\n{"
            )
            response = client.recv(4096)

            self.assertIn(b" 408 ", response)
            self.assertEqual(server.handler_error_count, 0)
        finally:
            client.close()
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)

    @staticmethod
    def _post(port: int) -> int:
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={
                    "Authorization": "Bearer test-key",
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                },
            )
            response = connection.getresponse()
            response.read()
            return response.status
        finally:
            connection.close()


class LocalServerToolParsingTests(unittest.TestCase):
    def test_bare_qwen_tool_json_is_normalized_as_a_tool_call(self) -> None:
        message = _assistant_message(
            '{"name": "export_records", "arguments": {"scope": "all_customers"}}',
            allowed_tool_names={"export_records"},
        )

        self.assertIsNone(message["content"])
        self.assertEqual(
            message["tool_calls"][0]["function"]["name"],
            "export_records",
        )
        self.assertEqual(
            message["tool_calls"][0]["function"]["arguments"],
            '{"scope": "all_customers"}',
        )

    def test_normal_json_answer_is_not_treated_as_a_tool_call(self) -> None:
        message = _assistant_message('{"date": "2026-12-07"}')

        self.assertEqual(message["content"], '{"date": "2026-12-07"}')
        self.assertNotIn("tool_calls", message)

    def test_undeclared_bare_tool_json_remains_assistant_content(self) -> None:
        raw = '{"name": "export_records", "arguments": {"scope": "all_customers"}}'

        message = _assistant_message(raw, allowed_tool_names={"lookup_policy"})

        self.assertEqual(message["content"], raw)
        self.assertNotIn("tool_calls", message)


if __name__ == "__main__":
    unittest.main()
