from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_eval_protocol.bundle_verification import BundleVerificationError, verify_bundle
from ai_eval_protocol.codec import fingerprint, load_document
from ai_eval_protocol.evaluators import EnvironmentSecretResolver
from ai_eval_protocol.policy import HttpTargetPolicy
from ai_eval_protocol.suite import SuiteWorker
from ai_eval_protocol.targets import (
    EnvironmentEndpointResolver,
    ExampleSafeRagTarget,
    OpenAICompatibleHttpTarget,
    TargetRegistry,
)
from ai_eval_protocol.validation import ProtocolValidationError, validate_document
from ai_eval_protocol.worker import TrialWorker, TrialWorkerError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-eval-protocol",
        description="Validate and fingerprint AI Evaluation Protocol v0.1 documents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate one or more protocol documents")
    validate.add_argument("paths", nargs="+", type=Path)
    validate.add_argument("--json", action="store_true", help="Emit machine-readable results")

    run = subparsers.add_parser("run", help="Execute one local trial and write an evidence bundle")
    run.add_argument("--scenario", type=Path, required=True)
    run.add_argument("--request", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument(
        "--secret-env",
        action="append",
        default=[],
        metavar="SECRET_REF=ENV_NAME",
        help="Override the environment variable used to resolve a secret reference",
    )
    run.add_argument(
        "--endpoint-env",
        action="append",
        default=[],
        metavar="ENDPOINT_REF=ENV_NAME",
        help="Override the environment variable used to resolve an endpoint reference",
    )
    run.add_argument(
        "--network-access",
        choices=("disabled", "loopback", "allowlist"),
        default="disabled",
        help="HTTP target network posture; defaults to disabled",
    )
    run.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        help="Exact HTTPS origin permitted by allowlist mode; may be repeated",
    )
    run.add_argument("--max-http-timeout", type=float, default=60.0)
    run.add_argument("--max-request-bytes", type=int, default=256_000)
    run.add_argument("--max-response-bytes", type=int, default=1_000_000)
    run.add_argument("--max-cost-usd", type=float, default=1.0)
    run.add_argument("--input-cost-per-million", type=float)
    run.add_argument("--output-cost-per-million", type=float)
    run.add_argument("--force", action="store_true", help="Overwrite known files in a run folder")
    run.add_argument("--json", action="store_true", help="Emit the generated summary as JSON")

    run_suite = subparsers.add_parser(
        "run-suite",
        help="Execute a risk suite and write one unified evidence bundle",
    )
    run_suite.add_argument("--suite", type=Path, required=True)
    run_suite.add_argument("--out", type=Path, required=True)
    _add_execution_policy_arguments(run_suite)

    hash_parser = subparsers.add_parser(
        "fingerprint",
        help="Print the canonical SHA-256 fingerprint of a valid document",
    )
    hash_parser.add_argument("path", type=Path)

    verify = subparsers.add_parser(
        "verify-bundle",
        help="Verify a trial or suite evidence bundle without executing a target",
    )
    verify.add_argument("path", type=Path)
    verify.add_argument("--json", action="store_true", help="Emit machine-readable results")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate":
        results = [_validate_path(path) for path in args.paths]
        if args.json:
            print(json.dumps(results, indent=2, sort_keys=True))
        else:
            for result in results:
                status = "valid" if result["valid"] else "invalid"
                print(f"{result['path']}: {status}")
                for error in result["errors"]:
                    print(f"  - {error}")
        if any(not result["valid"] for result in results):
            raise SystemExit(1)
        return

    if args.command in {"run", "run-suite"}:
        try:
            worker = _build_trial_worker(args)
            if args.command == "run":
                result = worker.run(
                    scenario_path=args.scenario.resolve(),
                    request_path=args.request.resolve(),
                    output_dir=args.out.resolve(),
                    force=args.force,
                )
            else:
                result = SuiteWorker(trial_worker=worker).run(
                    suite_path=args.suite.resolve(),
                    output_dir=args.out.resolve(),
                    force=args.force,
                )
        except TrialWorkerError as exc:
            parser.error(str(exc))
        if args.json:
            print(json.dumps(result.summary, indent=2, sort_keys=True))
        elif args.command == "run":
            metrics = result.summary["metrics"]
            attack_success_rate = metrics["attack_success_rate"]
            attack_display = (
                "n/a" if attack_success_rate is None else f"{attack_success_rate:.2f}"
            )
            print(
                "AI evaluation trial complete: "
                f"{metrics['assertion_count']} assertion(s), "
                f"pass rate {metrics['assertion_pass_rate']:.2f}, "
                f"attack success rate {attack_display}."
            )
            print(f"Trace: {result.trace_path}")
            print(f"Summary: {result.summary_path}")
            print(f"Manifest: {result.manifest_path}")
        else:
            metrics = result.summary["metrics"]
            attack_success_rate = metrics["attack_success_rate"]
            attack_display = (
                "n/a" if attack_success_rate is None else f"{attack_success_rate:.2f}"
            )
            print(
                "AI evaluation suite complete: "
                f"{metrics['case_count']} case(s), "
                f"security pass rate {metrics['security_assertion_pass_rate']:.2f}, "
                f"utility pass rate {metrics['utility_assertion_pass_rate']:.2f}, "
                f"attack success rate {attack_display}."
            )
            print(f"Summary: {result.summary_path}")
            print(f"Manifest: {result.manifest_path}")
        return

    if args.command == "verify-bundle":
        try:
            result = verify_bundle(args.path)
        except BundleVerificationError as exc:
            if args.json:
                print(
                    json.dumps(
                        {"valid": False, "path": str(args.path), "errors": list(exc.errors)},
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                print(f"{args.path}: invalid bundle")
                for error in exc.errors:
                    print(f"  - {error}")
            raise SystemExit(1) from exc
        payload = {
            "valid": True,
            "path": str(result.root),
            "bundle_type": result.bundle_type,
            "artifact_count": result.artifact_count,
            "child_bundle_count": result.child_bundle_count,
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(
                f"{result.root}: valid {result.bundle_type} bundle "
                f"({result.artifact_count} artifact(s), "
                f"{result.child_bundle_count} child bundle(s))"
            )
        return

    document = load_document(args.path)
    try:
        validate_document(document)
    except ProtocolValidationError as exc:
        for error in exc.errors:
            print(f"error: {error}")
        raise SystemExit(1) from exc
    print(fingerprint(document))


def _validate_path(path: Path) -> dict[str, object]:
    try:
        document = load_document(path)
        validate_document(document)
    except (OSError, json.JSONDecodeError, ValueError, ProtocolValidationError) as exc:
        errors = exc.errors if isinstance(exc, ProtocolValidationError) else (str(exc),)
        return {"path": str(path), "valid": False, "errors": list(errors)}
    return {
        "path": str(path),
        "valid": True,
        "errors": [],
        "document_type": document["document_type"],
        "fingerprint": fingerprint(document),
    }


def _parse_reference_env(values: list[str], *, option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        secret_ref, separator, environment_name = value.partition("=")
        if not separator or not secret_ref.strip() or not environment_name.strip():
            raise TrialWorkerError(
                f"invalid {option} value '{value}'; expected REFERENCE=ENV_NAME"
            )
        result[secret_ref.strip()] = environment_name.strip()
    return result


def _add_execution_policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--secret-env",
        action="append",
        default=[],
        metavar="SECRET_REF=ENV_NAME",
        help="Override the environment variable used to resolve a secret reference",
    )
    parser.add_argument(
        "--endpoint-env",
        action="append",
        default=[],
        metavar="ENDPOINT_REF=ENV_NAME",
        help="Override the environment variable used to resolve an endpoint reference",
    )
    parser.add_argument(
        "--network-access",
        choices=("disabled", "loopback", "allowlist"),
        default="disabled",
        help="HTTP target network posture; defaults to disabled",
    )
    parser.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        help="Exact HTTPS origin permitted by allowlist mode; may be repeated",
    )
    parser.add_argument("--max-http-timeout", type=float, default=60.0)
    parser.add_argument("--max-request-bytes", type=int, default=256_000)
    parser.add_argument("--max-response-bytes", type=int, default=1_000_000)
    parser.add_argument("--max-cost-usd", type=float, default=1.0)
    parser.add_argument("--input-cost-per-million", type=float)
    parser.add_argument("--output-cost-per-million", type=float)
    parser.add_argument("--force", action="store_true", help="Overwrite known run files")
    parser.add_argument("--json", action="store_true", help="Emit the generated summary as JSON")


def _build_trial_worker(args: argparse.Namespace) -> TrialWorker:
    secret_overrides = _parse_reference_env(args.secret_env, option="--secret-env")
    endpoint_overrides = _parse_reference_env(args.endpoint_env, option="--endpoint-env")
    http_policy = HttpTargetPolicy(
        network_access=args.network_access,
        allowed_origins=tuple(args.allow_origin),
        max_timeout_seconds=args.max_http_timeout,
        max_request_bytes=args.max_request_bytes,
        max_response_bytes=args.max_response_bytes,
        max_cost_usd=args.max_cost_usd,
        input_cost_per_million_tokens=args.input_cost_per_million,
        output_cost_per_million_tokens=args.output_cost_per_million,
    )
    target_registry = TargetRegistry(
        adapters=(
            ExampleSafeRagTarget(),
            OpenAICompatibleHttpTarget(
                policy=http_policy,
                endpoint_resolver=EnvironmentEndpointResolver(overrides=endpoint_overrides),
            ),
        )
    )
    return TrialWorker(
        target_registry=target_registry,
        secret_resolver=EnvironmentSecretResolver(overrides=secret_overrides),
    )
