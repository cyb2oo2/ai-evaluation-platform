# OpenAI-Compatible HTTP Target

## Compatibility contract

The `openai-compatible-http` adapter implements a deliberately small Chat Completions subset:

```http
POST /v1/chat/completions
Authorization: Bearer <resolved secret>
Content-Type: application/json
```

The request contains `model`, `messages`, and a token ceiling. Target metadata may select
`token_limit_field` as either `max_tokens` (the broad compatibility default) or
`max_completion_tokens`. `seed` is included only when target metadata explicitly sets `send_seed`
to `true`. The response must contain:

```text
choices[0].message.content and/or choices[0].message.tool_calls
usage.prompt_tokens
usage.completion_tokens
usage.total_tokens
```

Tool calls are captured as inert `tool_call` events. The adapter never executes them.

Optional `target.metadata.tools` entries are validated function-tool definitions and forwarded to
the target. Optional `target.metadata.protected_context_refs` entries are secret references that
the Worker resolves and inserts into a protected system message only for the authorized request.
Inline protected values remain forbidden. Both features remain subject to the request-byte,
network, timeout, response, token, tool-call, and cost policies below.

## Trusted reference resolution

Protocol documents contain references instead of URLs and credentials:

```json
{
  "endpoint_ref": "target-ref://openai-compatible/production",
  "secret_ref": "secret-ref://targets/openai-compatible-production"
}
```

By convention these resolve to environment variables:

```text
AI_EVAL_ENDPOINT_OPENAI_COMPATIBLE_PRODUCTION
AI_EVAL_SECRET_TARGETS_OPENAI_COMPATIBLE_PRODUCTION
```

`--endpoint-env REF=ENV_NAME` and `--secret-env REF=ENV_NAME` provide explicit mappings without
putting values in the request or command line.

## Network policy

Network access defaults to `disabled`.

- `disabled`: blocks the adapter before a request is sent.
- `loopback`: permits only endpoints whose resolved addresses are all loopback; HTTP is allowed for
  local integration tests.
- `allowlist`: requires HTTPS, an exact `--allow-origin`, and DNS results containing only public IP
  addresses.

The endpoint path must be exactly `/v1/chat/completions`. URL credentials, fragments, redirects,
and environment-configured HTTP proxies are rejected or disabled. Production deployment should
also enforce egress policy outside the Python process because DNS can change between policy
resolution and connection establishment.

## Resource policy

Two layers apply, and the lower limit wins:

- Trial limits in `trial_request.limits`.
- Trusted Worker limits supplied on the CLI.

Controls include:

- HTTP timeout.
- Maximum request bytes.
- Maximum response bytes, enforced through `Content-Length` and bounded reads.
- Maximum generated tokens sent as `max_tokens` and enforced again against reported completion
  usage.
- Maximum observed tool calls.
- Maximum estimated cost.

Trusted and per-trial numeric limits must be finite and within their declared positive or
non-negative domains. `NaN`, infinity, booleans used as token counts, zero timeouts, and invalid
byte limits fail before execution.

Cost enforcement requires trusted input/output prices per million tokens. Before sending, the
adapter verifies that the output-token ceiling alone cannot exceed the budget. After receiving a
response, it requires `total_tokens = prompt_tokens + completion_tokens`, enforces the completion
token ceiling, and calculates cost from reported usage. Missing or inconsistent usage makes the
trial `error`; a token or cost overrun makes it `budget_exceeded`. Post-response detection cannot
undo charges already incurred, so provider-side project budgets remain necessary.

The adapter marks tool-call capture complete only for the parsed target response. The deterministic
tool policy evaluator therefore proves that the captured response contains no denied call; it does
not claim that no out-of-band tool or environmental side effect occurred.

## CLI example

```powershell
$env:AI_EVAL_ENDPOINT_OPENAI_COMPATIBLE_PRODUCTION = `
  "https://inference.example.com/v1/chat/completions"
$env:AI_EVAL_SECRET_TARGETS_OPENAI_COMPATIBLE_PRODUCTION = "<api-key>"
$env:AI_EVAL_SECRET_FIXTURES_RAG_CANARY = "<evaluation-canary>"

python -m ai_eval_protocol run `
  --scenario examples/scenarios/rag-indirect-prompt-injection.json `
  --request examples/requests/rag-indirect-prompt-injection.http-request.json `
  --out runs/http-rag-demo `
  --network-access allowlist `
  --allow-origin https://inference.example.com `
  --input-cost-per-million 1.00 `
  --output-cost-per-million 2.00
```

The price values above are placeholders. Operators must supply prices matching their chosen
provider and model.
