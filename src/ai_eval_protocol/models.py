from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

PROTOCOL_VERSION = "0.1"

DocumentType = Literal["scenario", "trial_request", "trial_trace", "evaluation_summary"]
EvaluatorKind = Literal["deterministic", "probabilistic", "environment"]
AssertionStatus = Literal["pass", "fail", "error", "not_applicable"]
TrialStatus = Literal["completed", "error", "cancelled", "budget_exceeded"]


@dataclass(frozen=True)
class DocumentRef:
    document_id: str
    version: str
    fingerprint: str = ""


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    target_type: str
    version: str
    capabilities: tuple[str, ...] = ()
    endpoint_ref: str = ""
    secret_ref: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluatorSpec:
    evaluator_id: str
    kind: EvaluatorKind
    version: str
    implementation_ref: str
    model_ref: str = ""


@dataclass(frozen=True)
class AssertionSpec:
    assertion_id: str
    objective: str
    evaluator: EvaluatorSpec
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceRef:
    ref_type: Literal["event", "artifact"]
    ref_id: str
    selector: str = ""


@dataclass(frozen=True)
class AssertionResult:
    assertion_id: str
    status: AssertionStatus
    evaluator: EvaluatorSpec
    evidence: tuple[EvidenceRef, ...]
    score: float | None = None
    explanation: str = ""


@dataclass(frozen=True)
class TraceEvent:
    event_id: str
    sequence: int
    event_type: str
    timestamp: str
    payload: dict[str, Any]
