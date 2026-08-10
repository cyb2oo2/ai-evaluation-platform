# AI Evaluation Protocol v0.1

## Purpose

The protocol is the stable handoff between scenario authors, target adapters, isolated workers,
evaluators, storage, gates, and reporting. It records what was tested and what evidence supports
the result without assuming a model provider or execution platform.

## Evaluation lifecycle

```text
scenario
  -> trial_request
  -> target execution
  -> ordered events and artifacts
  -> assertion results
  -> trial_trace
  -> repeated trials
  -> evaluation_summary
```

Ordering is part of the evidence meaning:

1. A scenario is versioned before execution.
2. A trial request binds an immutable scenario reference to a target version and resource limits.
3. The worker captures interactions before evaluators interpret them.
4. Assertions cite captured events or hashed artifacts.
5. Summaries aggregate immutable trial references; they do not replace raw evidence.

Validation is closed-world in v0.1: unknown fields, duplicate JSON keys, non-finite numbers,
malformed hashes, invalid resource limits, and credential-like values in protocol metadata are
rejected rather than ignored.

## Documents

### `scenario`

A reusable security experiment. It contains:

- Stable `scenario_id` and `version`.
- Risk references with taxonomy, category, and risk ID.
- Versioned fixtures such as untrusted retrieval documents.
- The initial stimulus presented to the target.
- Evaluator-backed assertions.
- A `controls.utility_assertion_id` identifying the assertion that protects benign utility.

A scenario requires at least one security assertion and one utility assertion. The utility
assertion prevents a system that refuses every request from appearing secure. Fixture content is
part of the scenario fingerprint, so changing an attack payload creates a measurably different
artifact.

### `trial_request`

Binds one target version to one scenario version for one repetition. `request_id` and `version`
provide the stable identity used by a trace reference; `trial_id` identifies the resulting attempt.
Resource limits are explicit: time, tokens, tool calls, response bytes, and estimated cost. `seed`
may be null when the target cannot accept one, but the limitation remains visible.

Targets contain `endpoint_ref` and `secret_ref`, never credentials. References are resolved by an
authorized worker outside the protocol document.

### `trial_trace`

The primary evidence artifact. Events are contiguous and ordered from sequence zero. Supported
event types in v0.1 are:

- `message`: system, user, assistant, or tool messages.
- `retrieval`: retrieved document identifiers and trust labels.
- `tool_call`: requested tool name and arguments, captured only as data.
- `tool_result`: result associated with a tool call.
- `environment`: an explicitly scoped observation; its payload must state what was observed and
  must not imply visibility into effects outside that scope.
- `evaluator`: evaluator diagnostics that do not replace assertion results.
- `error`: target, worker, or evaluator failures.

Every pass or fail assertion result must cite at least one event or artifact. This prevents a score
from being presented as evidence by itself.

### `evaluation_summary`

Aggregates repeated trial references and numeric metrics for one target and scenario. A metric may
be `null` when required evidence was unavailable; consumers must fail closed instead of treating it
as zero. A summary is derived data, and consumers must be able to follow its trial references to
raw evidence.

`attack_success_rate` is available only when all required security assertions have conclusive
`pass` or `fail` outcomes. Missing, errored, or not-applicable security evidence makes it `null`;
it is never silently counted as a safe outcome.

## Evaluator provenance

Every assertion and result embeds an evaluator descriptor:

```json
{
  "evaluator_id": "secret-absence",
  "kind": "deterministic",
  "version": "1.1.0",
  "implementation_ref": "builtin://secret-absence/1.1.0",
  "model_ref": ""
}
```

Evaluator kinds are intentionally distinct:

- `deterministic`: identical evidence and implementation produce the same result.
- `environment`: inspects effects in the controlled execution environment.
- `probabilistic`: uses a model or stochastic classifier and must declare `model_ref`.

Probabilistic results must not be relabeled as deterministic, and a future gate must be able to
apply different thresholds to each kind.

`secret-absence` v1.1 requires an explicit complete target-response observation. It inspects all
assistant-message and structured tool-call string leaves and keys. Missing capture evidence is an
evaluator error, not a pass. `tool-call-policy` has the same scoped-evidence rule for observed tool
calls and makes no claim about out-of-band effects.

## Canonicalization and identity

The reference implementation serializes JSON with sorted keys, UTF-8, no insignificant
whitespace, and no NaN or Infinity. A document fingerprint is:

```text
sha256:<hex digest of canonical JSON bytes>
```

Fingerprints identify exact content. Human-facing IDs and versions identify logical artifacts.

## Versioning

- `protocol_version` is `0.1` for every document in this release.
- Additive compatible changes will advance the minor protocol version.
- Removing fields or changing their meaning requires a new protocol version and migration plan.
- Unknown protocol versions fail closed.

## v0.1 non-goals

- Container or VM execution.
- Scheduler, queue, database, or object storage implementation.
- Authentication and authorization implementation.
- A universal risk taxonomy.
- A universal score combining unrelated risk categories.

## Suite orchestration outside the protocol

The reference implementation can execute several scenario/request pairs with `run-suite`. Every
child output remains a Protocol v0.1 trial bundle. `suite_summary.json` and the top-level
`bundle_manifest.json` deliberately remain orchestration artifacts: adding them as protocol
document types would require a compatible protocol-version change and consumer migration plan.
