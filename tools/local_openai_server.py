from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass(frozen=True)
class ServerLimits:
    max_request_bytes: int = 256_000
    max_input_tokens: int = 4_096
    max_generation_tokens: int = 512
    max_connections: int = 16
    max_concurrent_requests: int = 1
    request_read_timeout_seconds: float = 10.0

    def validate(self) -> None:
        integer_limits = {
            "max_request_bytes": self.max_request_bytes,
            "max_input_tokens": self.max_input_tokens,
            "max_generation_tokens": self.max_generation_tokens,
            "max_connections": self.max_connections,
            "max_concurrent_requests": self.max_concurrent_requests,
        }
        for label, value in integer_limits.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if (
            not isinstance(self.request_read_timeout_seconds, (int, float))
            or isinstance(self.request_read_timeout_seconds, bool)
            or not math.isfinite(self.request_read_timeout_seconds)
            or self.request_read_timeout_seconds <= 0
        ):
            raise ValueError("request_read_timeout_seconds must be a positive finite number")


class LocalModel:
    def __init__(
        self,
        *,
        model_ref: str,
        device: str,
        dtype_name: str,
        threads: int,
        limits: ServerLimits,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if threads > 0:
            torch.set_num_threads(threads)
        self.torch = torch
        self.model_ref = model_ref
        self.device = device
        self.limits = limits
        self.tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=True)
        dtype = {
            "auto": "auto",
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype_name]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            local_files_only=True,
            dtype=dtype,
        ).to(device)
        self.model.eval()
        self.lock = threading.Lock()

    def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty array")
        max_tokens = _requested_max_tokens(request, limits=self.limits)
        tools = request.get("tools")
        template_args: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools is not None:
            if not isinstance(tools, list):
                raise ValueError("tools must be an array")
            template_args["tools"] = tools
        prompt = self.tokenizer.apply_chat_template(messages, **template_args)
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = int(encoded.input_ids.shape[1])
        _check_prompt_tokens(prompt_tokens, limits=self.limits)
        started = time.perf_counter()
        with self.lock, self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, encoded.input_ids.shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        allowed_tool_names = {
            tool["function"]["name"]
            for tool in (tools or [])
            if isinstance(tool, dict)
            and isinstance(tool.get("function"), dict)
            and isinstance(tool["function"].get("name"), str)
        }
        message = _assistant_message(text, allowed_tool_names=allowed_tool_names)
        completion_tokens = int(generated.shape[0])
        return {
            "id": f"chatcmpl-local-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_ref,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "local_runtime": {
                "device": self.device,
                "duration_seconds": round(time.perf_counter() - started, 6),
            },
        }


def _assistant_message(
    text: str,
    *,
    allowed_tool_names: set[str] | None = None,
) -> dict[str, Any]:
    tool_calls: list[dict[str, Any]] = []
    for index, match in enumerate(_TOOL_CALL_PATTERN.finditer(text)):
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        tool_call = _tool_call_from_value(value, index=index)
        if tool_call is not None:
            tool_calls.append(tool_call)
    content = _TOOL_CALL_PATTERN.sub("", text).strip()
    if not tool_calls and content:
        try:
            bare_value = json.loads(content)
        except json.JSONDecodeError:
            bare_value = None
        bare_call = _tool_call_from_value(bare_value, index=0)
        if (
            bare_call is not None
            and bare_call["function"]["name"] in (allowed_tool_names or set())
        ):
            tool_calls.append(bare_call)
            content = ""
    message: dict[str, Any] = {"role": "assistant", "content": content or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _requested_max_tokens(request: dict[str, Any], *, limits: ServerLimits) -> int:
    value = request.get("max_tokens", request.get("max_completion_tokens", 128))
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if value > limits.max_generation_tokens:
        raise ValueError(f"max_tokens exceeds server limit {limits.max_generation_tokens}")
    return value


def _check_prompt_tokens(prompt_tokens: int, *, limits: ServerLimits) -> None:
    if prompt_tokens > limits.max_input_tokens:
        raise ValueError(
            f"input has {prompt_tokens} tokens; server limit is {limits.max_input_tokens}"
        )


def _tool_call_from_value(value: Any, *, index: int) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not isinstance(value.get("name"), str):
        return None
    arguments = value.get("arguments")
    if not isinstance(arguments, dict):
        return None
    return {
        "id": f"call_local_{index}",
        "type": "function",
        "function": {
            "name": value["name"],
            "arguments": json.dumps(arguments, ensure_ascii=False, sort_keys=True),
        },
    }


def _load_request_json(body: bytes) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number is not allowed: {value}")

    value = json.loads(
        body.decode("utf-8"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("request body must be a JSON object")
    return value


class LocalEvaluationServer(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        model: LocalModel,
        api_key: str,
        limits: ServerLimits,
    ) -> None:
        super().__init__(server_address, Handler)
        self.model = model
        self.api_key = api_key
        self.limits = limits
        self.connection_slots = threading.BoundedSemaphore(limits.max_connections)
        self.inference_slots = threading.BoundedSemaphore(limits.max_concurrent_requests)

    def process_request(
        self,
        request: socket.socket,
        client_address: tuple[Any, ...],
    ) -> None:
        if not self.connection_slots.acquire(blocking=False):
            body = b'{"error":"local connection limit reached"}'
            response = (
                b"HTTP/1.1 503 Service Unavailable\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\n".encode("ascii")
                + b"Connection: close\r\n\r\n"
                + body
            )
            with suppress(OSError):
                request.sendall(response)
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.connection_slots.release()
            raise

    def process_request_thread(
        self,
        request: socket.socket,
        client_address: tuple[Any, ...],
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connection_slots.release()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.server.limits.request_read_timeout_seconds)  # type: ignore[attr-defined]

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except (ConnectionResetError, OSError):
            self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, {"status": "ok", "model": self.server.model.model_ref})  # type: ignore[attr-defined]
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        expected_key = self.server.api_key  # type: ignore[attr-defined]
        if self.headers.get("Authorization") != f"Bearer {expected_key}":
            self._send(401, {"error": "unauthorized"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > self.server.limits.max_request_bytes:  # type: ignore[attr-defined]
                raise ValueError("request body size is outside the configured limit")
            request = _load_request_json(self.rfile.read(length))
        except TimeoutError:
            self.close_connection = True
            self._send(408, {"error": "request body read timed out"})
            return
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._send(400, {"error": str(exc)})
            return
        inference_slots = self.server.inference_slots  # type: ignore[attr-defined]
        if not inference_slots.acquire(blocking=False):
            self._send(429, {"error": "local inference concurrency limit reached"})
            return
        try:
            response = self.server.model.complete(request)  # type: ignore[attr-defined]
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
            return
        except Exception as exc:  # inference errors must cross the HTTP boundary explicitly
            self._send(500, {"error": f"local inference failed: {type(exc).__name__}: {exc}"})
            return
        finally:
            inference_slots.release()
        self._send(200, response)

    def _send(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local OpenAI-compatible Transformers server")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--api-key-env", default="AI_EVAL_LOCAL_API_KEY")
    parser.add_argument("--max-request-bytes", type=int, default=256_000)
    parser.add_argument("--max-input-tokens", type=int, default=4_096)
    parser.add_argument("--max-generation-tokens", type=int, default=512)
    parser.add_argument("--max-connections", type=int, default=16)
    parser.add_argument("--max-concurrent-requests", type=int, default=1)
    parser.add_argument("--request-read-timeout", type=float, default=10.0)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        parser.error("the local evaluation server may only bind to loopback")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        parser.error(f"set {args.api_key_env} before starting the server")
    if not 1 <= args.port <= 65_535:
        parser.error("port must be between 1 and 65535")
    if args.threads < 1:
        parser.error("threads must be a positive integer")
    limits = ServerLimits(
        max_request_bytes=args.max_request_bytes,
        max_input_tokens=args.max_input_tokens,
        max_generation_tokens=args.max_generation_tokens,
        max_connections=args.max_connections,
        max_concurrent_requests=args.max_concurrent_requests,
        request_read_timeout_seconds=args.request_read_timeout,
    )
    try:
        limits.validate()
    except ValueError as exc:
        parser.error(str(exc))
    model = LocalModel(
        model_ref=args.model,
        device=args.device,
        dtype_name=args.dtype,
        threads=args.threads,
        limits=limits,
    )
    server = LocalEvaluationServer(
        (args.host, args.port),
        model=model,
        api_key=api_key,
        limits=limits,
    )
    print(
        f"Local model ready: {args.model} on {args.device}/{args.dtype} at "
        f"http://{args.host}:{args.port}/v1/chat/completions",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
