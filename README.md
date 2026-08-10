# AI Evaluation Platform

An evidence-backed security evaluation platform for models, RAG applications, and tool-using
agents. The repository currently implements the first platform contract: **AI Evaluation
Protocol v0.1**.

The protocol is deliberately provider-neutral and dependency-free. It defines how targets,
scenarios, trials, events, assertions, evidence, and aggregate summaries are represented before
the platform adds model adapters, workers, storage, or a user interface.

## Implemented in v0.1

- A versioned JSON contract for `scenario`, `trial_request`, `trial_trace`, and
  `evaluation_summary` documents.
- Fail-closed validation including unknown-field, duplicate-key, finite-number, limit,
  evidence-reference, evaluator-provenance, and nested credential checks.
- Canonical JSON and SHA-256 fingerprints for reproducible artifacts.
- A first RAG indirect-prompt-injection scenario with paired security and utility assertions.
- A captured trial showing the attack blocked without breaking the benign task.
- A local Trial Worker with a registered safe RAG target and three evidence-backed evaluators:
  `secret-absence`, `tool-call-policy`, and `response-contains`.
- Complete target-response secret checks across assistant content, structured tool calls, nested
  string values, and keys; incomplete capture produces an error rather than a safe result.
- A policy-governed OpenAI-compatible `/v1/chat/completions` target with default-deny networking,
  bounded HTTP, usage capture, and cost enforcement.
- A first executable RAG security vertical spanning indirect prompt injection, protected-context
  leakage, and unauthorized tool use, with one unified evidence package.
- An optional loopback-only Transformers server with bounded connections, request size, input
  tokens, generation, inference concurrency, and request-read time for exercising the adapter with
  a cached local LLM.
- A two-model matrix for running the same risks against Qwen2.5 0.5B and Qwen2.5-Coder 3B while
  preserving per-model provenance and evidence.

## Quick start

```powershell
python -m ai_eval_protocol validate examples/scenarios/rag-indirect-prompt-injection.json
python -m ai_eval_protocol validate examples/requests/rag-indirect-prompt-injection.request.json
python -m ai_eval_protocol validate examples/trials/rag-indirect-prompt-injection.blocked.json
python -m ai_eval_protocol validate examples/summaries/rag-indirect-prompt-injection.summary.json
python -m ai_eval_protocol fingerprint examples/trials/rag-indirect-prompt-injection.blocked.json
python -m pip install -e ".[dev]"
python -m ruff check src tests tools
python -m unittest discover -s tests -v
python tools/validate_examples.py
python tools/run_ci_smoke.py
python -m ai_eval_protocol verify-bundle runs/ci-smoke
```

Run the scenario through the local worker:

```powershell
$env:AI_EVAL_SECRET_FIXTURES_RAG_CANARY = "EVAL-CANARY-DO-NOT-DISCLOSE"
python -m ai_eval_protocol run `
  --scenario examples/scenarios/rag-indirect-prompt-injection.json `
  --request examples/requests/rag-indirect-prompt-injection.request.json `
  --out runs/rag-safe-demo
```

The worker atomically publishes `trial_trace.json`, `evaluation_summary.json`, validated input
snapshots, and a bundle v0.2 manifest containing canonical fingerprints and raw-byte hashes. All
resolved secrets are redacted and the complete staged bundle is scanned before publication.
`verify-bundle` independently checks hashes, protocol documents, references, suite aggregation,
child manifests, and interrupted-transaction residue without contacting a target.

Run the three-risk local RAG suite after starting the optional local model target:

```powershell
python -m ai_eval_protocol run-suite `
  --suite examples/local-rag-security-suite.json `
  --out runs/local-rag-security-vertical `
  --network-access loopback `
  --endpoint-env "target-ref://openai-compatible/local-qwen=AI_EVAL_LOCAL_ENDPOINT" `
  --secret-env "secret-ref://targets/local-qwen-api-key=AI_EVAL_LOCAL_API_KEY" `
  --secret-env "secret-ref://fixtures/injection-marker=AI_EVAL_INJECTION_MARKER" `
  --secret-env "secret-ref://fixtures/customer-recovery-token=AI_EVAL_RECOVERY_TOKEN" `
  --input-cost-per-million 0 --output-cost-per-million 0 --max-cost-usd 0
```

See [docs/LOCAL_RAG_VERTICAL.md](docs/LOCAL_RAG_VERTICAL.md) for server startup, risk cases, and the
evidence-package layout.

When running from a checkout without installing the package:

```powershell
$env:PYTHONPATH = "src"
python -m ai_eval_protocol --help
```

## Repository map

```text
schema/                       JSON Schema contract
src/ai_eval_protocol/         dependency-free models, validation, and CLI
tools/validate_examples.py    complete example and suite-reference validation
tools/run_ci_smoke.py         deterministic run-and-verify CI evidence smoke
docs/PROTOCOL.md              field semantics and lifecycle
docs/ARCHITECTURE.md          executable trial loop and bundle contract
docs/HTTP_TARGET.md           HTTP compatibility and network/resource policy
docs/LOCAL_RAG_VERTICAL.md    local LLM vertical-slice runbook and evidence layout
docs/THREAT_MODEL.md          trust boundaries and non-goals
examples/scenarios/           reusable security scenarios
examples/local-rag-security-suite.json  three-risk suite definition
examples/local-rag-security-model-matrix.json  0.5B/3B six-case comparison suite
examples/trials/              captured protocol examples
tests/                        executable protocol invariants
.github/workflows/ci.yml      cross-platform verification gate
```

CI treats execution and offline verification as hard gates. Retaining the verified smoke through
GitHub artifact storage is best-effort because account-level storage quota is external to the
evaluation result.

## Architectural boundary

The protocol core has no model-provider SDK or runtime dependency. The optional HTTP adapter calls
only explicitly authorized targets and never executes returned tool calls. Credentials are resolved
in Worker memory and redacted before serialization. Scheduling and persistent storage remain
outside this package. VeriSec remains a separate deterministic engine for reviewing generated code
and patches.

See [docs/HTTP_TARGET.md](docs/HTTP_TARGET.md) for external endpoint configuration.
