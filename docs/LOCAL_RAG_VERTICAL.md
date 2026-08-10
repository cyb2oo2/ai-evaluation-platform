# Local RAG security vertical

This repository includes a runnable vertical slice that evaluates a real, locally hosted language
model through the policy-governed OpenAI-compatible adapter. It covers three independent risks:

1. indirect prompt injection from retrieved content;
2. protected-context disclosure;
3. an unauthorized `export_records` tool request.

Each case pairs one deterministic security assertion with a deterministic utility assertion. Tool
calls are captured as inert evidence and are never executed. Protected values are resolved in
worker memory, sent only to the authorized loopback target, evaluated before redaction, and omitted
from serialized evidence.

## Start the local target

The optional server is not part of the dependency-free protocol runtime. Run it from a Python
environment that already contains PyTorch and Transformers and has the model cached locally:

```powershell
$env:AI_EVAL_LOCAL_API_KEY = "replace-with-a-local-random-value"
python tools/local_openai_server.py `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --device cpu `
  --host 127.0.0.1 `
  --port 18080 `
  --max-request-bytes 256000 `
  --max-input-tokens 4096 `
  --max-generation-tokens 512 `
  --max-concurrent-requests 1 `
  --max-connections 16 `
  --request-read-timeout 10
```

An additional cached 3B target can run alongside it on a second loopback port without using the
GPU:

```powershell
python tools/local_openai_server.py `
  --model Qwen/Qwen2.5-Coder-3B-Instruct `
  --device cpu `
  --dtype bfloat16 `
  --host 127.0.0.1 `
  --port 18081
```

The server enforces loopback binding, bearer authentication, strict JSON parsing, request and input
caps, an explicit generation cap, bounded accepted connections, non-queueing inference admission
(HTTP 429 when saturated), a request-body read timeout, and local-only model loading. Excess
connections receive HTTP 503 before another request body is read. Invalid client limits return
HTTP 400 instead of being silently clamped. It exposes `/v1/chat/completions` plus `/health`. It is
a development target, not a production serving stack.

## Run the suite

```powershell
$env:AI_EVAL_LOCAL_ENDPOINT = "http://127.0.0.1:18080/v1/chat/completions"
$env:AI_EVAL_INJECTION_MARKER = "INJECTION_SUCCEEDED"
$env:AI_EVAL_RECOVERY_TOKEN = "generate-a-unique-test-canary"

python -m ai_eval_protocol run-suite `
  --suite examples/local-rag-security-suite.json `
  --out runs/local-rag-security-vertical `
  --network-access loopback `
  --endpoint-env "target-ref://openai-compatible/local-qwen=AI_EVAL_LOCAL_ENDPOINT" `
  --secret-env "secret-ref://targets/local-qwen-api-key=AI_EVAL_LOCAL_API_KEY" `
  --secret-env "secret-ref://fixtures/injection-marker=AI_EVAL_INJECTION_MARKER" `
  --secret-env "secret-ref://fixtures/customer-recovery-token=AI_EVAL_RECOVERY_TOKEN" `
  --input-cost-per-million 0 `
  --output-cost-per-million 0 `
  --max-cost-usd 0
```

## Unified evidence package

```text
runs/local-rag-security-vertical/
  inputs/suite.json
  trials/indirect-prompt-injection/
    inputs/{scenario,request}.json
    trial_trace.json
    evaluation_summary.json
    bundle_manifest.json
  trials/sensitive-data-leakage/...
  trials/unauthorized-tool-call/...
  suite_summary.json
  bundle_manifest.json
```

Each trial remains a valid Protocol v0.1 evidence bundle. The top-level suite summary and manifest
are orchestration artifacts, not new Protocol v0.1 document types. The top-level manifest hashes
the suite input, aggregate summary, and every child bundle manifest, preserving a navigable chain
to the raw events and assertion evidence.

After a run, verify the full chain offline:

```powershell
python -m ai_eval_protocol verify-bundle runs/local-rag-security-vertical
```

One run against one small local model is a smoke-quality vertical slice. It demonstrates the
execution and evidence path; it does not establish general model safety or benchmark quality.

## Two-model matrix

With both local targets running, `examples/local-rag-security-model-matrix.json` executes the same
three risks against 0.5B and 3B targets. Its `suite_summary.json` includes `model_results` grouped
by exact model and target version, while each of the six child bundles retains its own raw trace:

```powershell
$env:AI_EVAL_QWEN3B_ENDPOINT = "http://127.0.0.1:18081/v1/chat/completions"

python -m ai_eval_protocol run-suite `
  --suite examples/local-rag-security-model-matrix.json `
  --out runs/local-rag-security-model-matrix `
  --network-access loopback `
  --endpoint-env "target-ref://openai-compatible/local-qwen=AI_EVAL_LOCAL_ENDPOINT" `
  --endpoint-env "target-ref://openai-compatible/local-qwen3b=AI_EVAL_QWEN3B_ENDPOINT" `
  --secret-env "secret-ref://targets/local-qwen-api-key=AI_EVAL_LOCAL_API_KEY" `
  --secret-env "secret-ref://fixtures/injection-marker=AI_EVAL_INJECTION_MARKER" `
  --secret-env "secret-ref://fixtures/customer-recovery-token=AI_EVAL_RECOVERY_TOKEN" `
  --input-cost-per-million 0 `
  --output-cost-per-million 0 `
  --max-cost-usd 0
```

Bare Qwen tool-call JSON is normalized only when its function name matches a tool declared in the
request. This closes the gap between Qwen's local rendering and OpenAI-compatible structured
`tool_calls` without treating arbitrary JSON answers as tool execution intent.
