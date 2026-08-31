from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .orchestrator import run_public_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-search-audit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="run a bounded public audit")
    audit.add_argument("domain")
    audit.add_argument(
        "--output",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path("audit-output"),
    )
    audit.add_argument("--max-pages", type=int, default=50)
    audit.add_argument("--geo-optimizer-command")
    audit.add_argument("--report-locale", choices=("pl", "en"), default="en")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        run = run_public_audit(
            args.domain,
            output_dir=args.output_dir,
            max_pages=args.max_pages,
            geo_optimizer_command=args.geo_optimizer_command,
            report_locale=args.report_locale,
        )
        print(f"Audit {run.audit_id} written to {args.output_dir}")
    return 0
