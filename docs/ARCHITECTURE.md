# Architecture

## Trial loop

```text
scenario + trial request
        -> protocol validation
        -> reference and capability binding
        -> registered target adapter
        -> captured inert events
        -> registered evaluators
        -> evidence-linked assertion results
        -> secret redaction and complete-bundle scan
        -> trial trace
        -> evaluation summary
        -> bundle manifest
        -> offline bundle verification
```

The order is a contract. Evaluators consume captured evidence, never live target output. A target
adapter cannot mark its own assertion as passed, and an evaluator cannot retroactively change the
events it judges.

## Package map

- `codec.py`: canonical JSON, loading, and content fingerprints.
- `models.py`: protocol constants and immutable value types.
- `validation.py`: document shape and cross-field invariants.
- `policy.py`: HTTP network, endpoint, byte, timeout, and cost policy.
- `artifact_io.py`: same-parent staging and replace-on-success output transactions.
- `bundle_verification.py`: offline artifact, reference, aggregation, and transaction-residue
  verification.
- `targets.py`: local safe RAG and OpenAI-compatible HTTP adapters.
- `evaluators.py`: evaluator registry, secret resolution, and the three built-in evaluators.
- `secret_safety.py`: resolved-secret collection, recursive redaction, and bundle scanning.
- `worker.py`: trial orchestration and evidence-bundle writing.
- `suite.py`: multi-risk orchestration and unified evidence-package aggregation.
- `cli.py`: `validate`, `fingerprint`, `run`, `run-suite`, and `verify-bundle` command surface.
- `tools/local_openai_server.py`: optional local Transformers target outside the runtime package.

## Current execution policy

Protocol v0.1 registers the in-process `example-safe-rag` adapter and the policy-governed
`openai-compatible-http` adapter. HTTP access is disabled by default. Explicit loopback or exact
HTTPS-origin policy is required; redirects and proxies are disabled. Neither adapter performs
target-requested tool execution. Retrieved content and generated tool arguments are captured as
JSON data.

Credentials and canaries are resolved from environment variables in worker memory. Evaluators see
the original evidence, then the Worker redacts resolved secret values from events and assertion
results. It scans every staged JSON artifact for every resolved value before publication. Missing
canaries produce an evaluator error instead of silently passing the assertion.

## Trial bundle

Every successful local run writes:

```text
inputs/scenario.json
inputs/request.json
trial_trace.json
evaluation_summary.json
bundle_manifest.json
```

Bundle v0.2 records a canonical protocol fingerprint and a raw-byte SHA-256 for each artifact.
Files are written to a same-parent staging directory and published only after validation, redaction,
secret scanning, and manifest creation succeed. `--force` replaces a previous directory only at
commit time; a failed run preserves the previous complete bundle. If post-swap backup cleanup
fails, the transaction restores the previous bundle and reports failure. The manifest itself is
not yet signed; signed attestations and external artifact storage are deferred.

`verify-bundle` is a pure offline consumer. It recalculates raw hashes and canonical fingerprints,
validates protocol documents and their references, recomputes suite metrics and model groupings,
recurses through child manifests, and rejects staging, backup, or failed-publication residue beside
the requested bundle. Verification provides integrity and internal consistency, not signer
identity.

`run-suite` preserves this per-trial contract under `trials/<case-id>/` and adds a top-level
`suite_summary.json` plus `bundle_manifest.json`. These are orchestration artifacts rather than new
Protocol v0.1 document types. The top-level manifest hashes the suite definition, suite summary,
and each child manifest. Suite case results preserve target and model identity, and
`model_results` groups by model, target ID, and target version so distinct deployments are not
combined under one model label.

## Consumers

- The CLI renders immediate trial and suite outcomes.
- The CLI can independently verify trial and suite bundles before a gate consumes their summaries.
- A future storage service will index summary fields while preserving trace artifacts.
- A future VeriSec adapter will add generated-code findings as evidence-linked assertions.
