"""Small CLI boundary; errors never echo profile, credentials or provider bodies."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .measurement_profile import MeasurementPreflight
from .measurement_workflow import (
    UnsupportedMeasurementProfile,
    load_measurement_profile,
    run_measurements,
)


def add_measurement_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    measure = subparsers.add_parser("measure", help="collect a frozen performance sample")
    measure.add_argument("project_ref")
    measure.add_argument("--source-version", required=True)
    measure.add_argument("--clients-root", required=True, type=Path)
    measure.add_argument("--profile", required=True, type=Path)
    report = subparsers.add_parser(
        "measurement-report", help="print client results or an explicit technical view"
    )
    report.add_argument("project_ref")
    report.add_argument("--clients-root", required=True, type=Path)
    report.add_argument("--diagnostic-run", required=True)
    report.add_argument("--view", choices=("client", "technical"), default="client")


def _show_preflight(preflight: MeasurementPreflight) -> None:
    print(f"MEASUREMENT_PREFLIGHT={preflight.model_dump_json()}", flush=True)


def measurement_command(args: argparse.Namespace) -> int:
    if args.project_command == "measurement-report":
        from .measurement_report import (
            load_measurement_report,
            render_measurement_fragment,
            render_measurement_technical_fragment,
        )

        try:
            report = load_measurement_report(
                args.project_ref,
                clients_root=args.clients_root,
                diagnostic_run_ref=args.diagnostic_run,
            )
            renderer = (
                render_measurement_technical_fragment
                if args.view == "technical"
                else render_measurement_fragment
            )
            print(renderer(report), end="")
            return 0
        except (ValueError, OSError):
            print("Measurement report failed: invalid source or performance run.", file=sys.stderr)
            return 2
    try:
        destination = run_measurements(
            args.project_ref,
            clients_root=args.clients_root,
            source_version=args.source_version,
            profile=load_measurement_profile(args.profile),
            on_preflight=_show_preflight,
        )
        from .diagnostic_performance import DiagnosticRunV2
        from .diagnostic_store import DiagnosticStore

        run = (
            DiagnosticStore(destination.parents[2]).load(args.source_version, destination.name).run
        )
        if not isinstance(run, DiagnosticRunV2):
            raise ValueError("performance publication returned the wrong schema")
        summary = {
            "published": True,
            "with_metrics": sum(c.lab is not None or c.field is not None for c in run.collections),
            "states": dict(Counter(c.attempts[-1].state.value for c in run.collections)),
        }
    except UnsupportedMeasurementProfile:
        print(
            "Measurement refused: Gemini execution is not supported by this stage.", file=sys.stderr
        )
        return 2
    except (ValueError, OSError):
        print(
            "Measurement failed: invalid profile, source, evidence or publication.", file=sys.stderr
        )
        return 2
    print("MEASUREMENT_RESULT=" + json.dumps(summary, sort_keys=True))
    print(destination)
    return 0
