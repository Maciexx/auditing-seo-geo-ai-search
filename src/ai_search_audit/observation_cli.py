"""Safe command boundary; no implicit paid execution from an audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .observation_workflow import load_observation_profile, run_observations


def add_observation_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    observe = subparsers.add_parser("observe-ai", help="explicit bounded OpenAI API web search")
    observe.add_argument("project_ref")
    observe.add_argument("--clients-root", required=True, type=Path)
    observe.add_argument("--source-version", required=True)
    observe.add_argument("--profile", required=True, type=Path)
    observe.add_argument("--paid-authorized", action="store_true")
    observe.add_argument("--preflight-only", action="store_true")
    report = subparsers.add_parser("observation-report", help="read frozen API observations")
    report.add_argument("project_ref")
    report.add_argument("--clients-root", required=True, type=Path)
    report.add_argument("--diagnostic-run", required=True)
    report.add_argument("--view", choices=("client", "technical"), default="client")
    report.add_argument("--projection-version", choices=("1.0.0", "1.1.0"), default="1.0.0")


def observation_command(args: argparse.Namespace) -> int:
    try:
        if args.project_command == "observation-report":
            from .observation_report import (
                load_observation_report,
                render_observation_fragment,
                render_observation_technical_fragment,
            )

            report = load_observation_report(
                args.project_ref,
                clients_root=args.clients_root,
                diagnostic_run_ref=args.diagnostic_run,
            )
            fragment = (
                render_observation_technical_fragment(report)
                if args.view == "technical"
                else render_observation_fragment(report, projection_version=args.projection_version)
            )
            print(fragment, end="")
            return 0
        result = run_observations(
            args.project_ref,
            clients_root=args.clients_root,
            source_version=args.source_version,
            profile=load_observation_profile(args.profile),
            paid_authorized=args.paid_authorized,
            preflight_only=args.preflight_only,
            on_preflight=lambda value: print(
                "OBSERVATION_PREFLIGHT=" + value.model_dump_json(), flush=True
            ),
        )
        metrics = result.collection.run.sample.metrics if result.collection else None
        print(
            "OBSERVATION_RESULT="
            + json.dumps(
                {
                    "published": result.destination is not None,
                    "state": metrics.state.value
                    if result.destination and metrics
                    else "UNAVAILABLE",
                    "attempted": len(result.collection.run.attempts) if result.collection else 0,
                    "eligible": metrics.measured if metrics else 0,
                    "stop_reason": result.collection.stop_reason
                    if result.collection
                    else "preflight_only",
                },
                sort_keys=True,
            )
        )
        if result.destination:
            print(result.destination)
        return 0
    except (ValueError, OSError):
        print(
            "Observation failed: invalid profile, source, evidence or publication.", file=sys.stderr
        )
        return 2
