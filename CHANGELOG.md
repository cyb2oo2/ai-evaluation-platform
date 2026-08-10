# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- Atomic trial and suite evidence-package publication with raw-byte hashes.
- Bounded local-model serving with input, generation, request, concurrency, and read-time limits.
- Cross-platform CI on Python 3.11 and 3.13.
- Offline `verify-bundle` for trial and suite hashes, references, derived metrics, child bundles,
  and interrupted-publication residue.
- CI discovery of every example and suite reference plus a verified deterministic evidence smoke;
  one representative bundle is retained for seven days when GitHub storage quota is available.

### Changed

- Protocol validation now rejects unknown fields, duplicate keys, non-finite numbers, malformed
  fingerprints, invalid limits, and inline credentials at any nesting depth.
- Security metrics become unavailable when security evidence is missing or inconclusive.
- Tool-call policy results now describe only complete target-response observations.
- HTTP usage validation and post-response token-budget enforcement fail closed.
- Secret absence now covers assistant and structured tool-call keys and values and requires an
  explicit complete-response observation.
- Local serving bounds accepted connections separately from inference concurrency and handles
  slow or reset clients without uncaught handler errors.
- Forced output replacement restores the previous complete bundle if backup cleanup fails.
- CI no longer creates redundant dependency caches or uploads the same smoke bundle from every
  platform matrix entry.

### Security

- Resolved secrets are redacted from events and assertion results, then scanned across the complete
  staged bundle before publication.
