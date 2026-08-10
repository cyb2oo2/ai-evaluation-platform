from __future__ import annotations

import json
import os
from pathlib import Path

from ai_eval_protocol.bundle_verification import verify_bundle
from ai_eval_protocol.evaluators import EnvironmentSecretResolver
from ai_eval_protocol.worker import TrialWorker

ROOT = Path(__file__).resolve().parents[1]
SECRET_REF = "secret-ref://fixtures/rag-canary"
SECRET_ENV = "AI_EVAL_CI_RAG_CANARY"


def main() -> None:
    os.environ[SECRET_ENV] = "CI-CANARY-DO-NOT-DISCLOSE"
    output_dir = ROOT / "runs" / "ci-smoke"
    worker = TrialWorker(
        secret_resolver=EnvironmentSecretResolver(overrides={SECRET_REF: SECRET_ENV})
    )
    result = worker.run(
        scenario_path=ROOT / "examples" / "scenarios" / "rag-indirect-prompt-injection.json",
        request_path=(
            ROOT / "examples" / "requests" / "rag-indirect-prompt-injection.request.json"
        ),
        output_dir=output_dir,
        force=True,
    )
    verified = verify_bundle(output_dir)
    print(
        json.dumps(
            {
                "bundle_type": verified.bundle_type,
                "artifact_count": verified.artifact_count,
                "assertion_statuses": [
                    item["status"] for item in result.trace["assertion_results"]
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
