# Contributions and upstream attribution

**Yibo Chen (cyb2oo2)** designed and maintains this project: the versioned protocol, fail-closed validation, worker and suite lifecycle, deterministic evaluators, governed HTTP adapter, atomic evidence bundles, and offline verifier. Implementation and verification were developed with AI assistance; this does not imply unaided authorship of every line.

The protocol core uses Python's standard library and no model-provider SDK. Ruff is a development tool; setuptools builds the package; GitHub Actions runs cross-platform checks. The optional local-model server relies on PyTorch and Hugging Face Transformers. Qwen2.5 models and their pretrained weights are upstream work and remain subject to their own terms.

Synthetic scenarios and deterministic safe-target fixtures demonstrate the evaluation contract. They are not independent real-world adversarial coverage or evidence that a named language model is robust. The 0.5B/3B model matrix defines a comparison to run; it is not a reported experimental result.

See [architecture](ARCHITECTURE.md), [protocol](PROTOCOL.md), and [threat model](THREAT_MODEL.md) for design details and boundaries. This platform and VeriSec Forge are separate projects with separate evidence.
