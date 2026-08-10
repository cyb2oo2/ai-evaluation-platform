# Threat Model

## Protected assets

- Credentials used to reach evaluation targets.
- Secrets, customer data, and canaries placed in test environments.
- Integrity of scenarios, traces, assertions, and published metrics.
- Worker hosts, internal networks, tools, and persistent storage.
- The distinction between deterministic evidence and probabilistic judgment.

## Untrusted inputs

- Model and agent responses.
- Tool names and tool arguments produced by a target.
- Retrieved RAG documents and external web content.
- Scenario content imported from third parties.
- Target-controlled error messages and metadata.

## Required boundaries

```text
untrusted target output
        |
        v
capture as inert protocol data
        |
        v
policy and evaluator boundary
        |
        +--> approved sandbox action
        +--> deterministic assertion
        +--> isolated probabilistic evaluator
```

- A protocol parser never executes values contained in a document.
- The worker resolves `secret_ref` in memory, redacts all resolved values, and scans every staged
  JSON artifact before publishing a bundle.
- Secret assertions cover assistant messages and structured tool-call keys and values only when a
  complete target-response observation is present; missing capture fails closed.
- Tool calls require an independent allowlist and policy decision before execution.
- Protocol v0.1's local worker never executes target-requested tools; it records them as events.
- Network, filesystem, database, and subprocess effects must occur in an isolated environment.
- HTTP target access is disabled by default; external access requires exact HTTPS origin policy.
- Redirects and environment proxy settings are disabled to reduce SSRF policy bypasses.
- Evaluator prompts must delimit target output as untrusted content.
- Bundle publication is transactional. Retries should use new trial IDs; explicit `--force`
  replaces a prior output only after the new complete bundle passes integrity checks.
- Summaries cannot claim coverage that cannot be traced to trial evidence.
- Offline bundle verification recalculates hashes and derived suite metrics, follows child
  manifests, and rejects interrupted publication residue before evidence is consumed.

## Known v0.1 limitations

- The reference validator checks protocol semantics but is not a sandbox.
- SHA-256 detects mutation but does not provide signer identity; signed attestations are future work.
- Timestamp strings are required but strict RFC 3339 parsing is deferred.
- JSON Schema and reference validation do not prove that an evaluator implementation is honest.
- In-process DNS validation cannot eliminate DNS rebinding races; production workers still require
  infrastructure-level egress controls.
- Usage-based cost enforcement trusts provider-reported token counts and detects final-call cost
  overruns only after the response has been billed.
- HTTP response capture does not observe out-of-band tool execution or environmental side effects;
  those claims require a separately instrumented sandbox.
