"""AI Evaluation Protocol v0.1."""

from ai_eval_protocol.codec import canonical_json, fingerprint, load_document
from ai_eval_protocol.policy import HttpPolicyError, HttpTargetPolicy
from ai_eval_protocol.targets import OpenAICompatibleHttpTarget
from ai_eval_protocol.validation import ProtocolValidationError, validate_document
from ai_eval_protocol.worker import TrialRunResult, TrialWorker, TrialWorkerError

__all__ = [
    "ProtocolValidationError",
    "HttpPolicyError",
    "HttpTargetPolicy",
    "OpenAICompatibleHttpTarget",
    "TrialRunResult",
    "TrialWorker",
    "TrialWorkerError",
    "canonical_json",
    "fingerprint",
    "load_document",
    "validate_document",
]

__version__ = "0.1.0"
