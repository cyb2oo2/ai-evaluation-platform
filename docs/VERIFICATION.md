# Verification record

Rerun September 15, 2026 (Asia/Shanghai), Windows, Python 3.13, from source based on commit `7b8cef8`. Documentation changes do not change the protocol contract. CI provides the commit-specific cross-platform record.

| Command | Observed result |
| --- | --- |
| `python -m ruff check src tests tools` | All checks passed |
| `python -m unittest discover -s tests -v` | 77 tests passed |
| `python -m compileall -q src tests tools` | Exit 0 |
| `python tools/validate_examples.py` | 14 protocol documents and 3 suites validated |
| `python tools/run_ci_smoke.py` | 3 passing assertions, trial bundle with 4 artifacts |
| `python -m ai_eval_protocol verify-bundle runs/ci-smoke` | Valid trial bundle, 0 child bundles |

The smoke uses the registered deterministic safe RAG target, not a downloaded or paid model. HTTP tests use local test servers; they do not call external model APIs. The smoke refreshes its own `runs/ci-smoke` directory. Generated bundles remain outside Git; regenerate them with the commands above.

The trace, summary, scenario/request snapshots, and manifest support offline checking. Hash validity establishes recorded-byte consistency; it does not independently prove that every upstream observation is truthful or that scenario coverage is sufficient. No model-comparison result or production-security claim follows from these checks.
