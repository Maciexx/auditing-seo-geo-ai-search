from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from .client_delivery import finalize_client_report
from .data_intake import create_owned_intake_dir, discard_owned_intake_dir
from .diagnostic_workflow import prepare_diagnostic_contract, run_diagnostics
from .orchestrator import run_public_audit
from .project_orchestrator import (
    create_project_audit,
    enrich_project,
    refresh_project,
    update_project_context,
    validate_project,
)

_MAX_NORMALIZED_INTAKE_BYTES = 2 * 1024 * 1024
_NORMALIZED_INTAKE_FILENAME = "normalized-intake.json"


def _bounded_max_pages(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 100:
        raise argparse.ArgumentTypeError("max pages must be between 1 and 100")
    return parsed


def _iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def _load_normalized_intake(path: Path, *, intake_dir: Path) -> dict[str, object]:
    if path.parent.absolute() != intake_dir.absolute():
        raise ValueError("normalized intake JSON must be directly inside --intake-dir")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise ValueError("normalized intake JSON must be a regular file")
        if details.st_size > _MAX_NORMALIZED_INTAKE_BYTES:
            raise ValueError("normalized intake JSON exceeds the size limit")
        chunks: list[bytes] = []
        bytes_read = 0
        while bytes_read <= _MAX_NORMALIZED_INTAKE_BYTES:
            chunk = os.read(
                descriptor,
                min(64 * 1024, _MAX_NORMALIZED_INTAKE_BYTES + 1 - bytes_read),
            )
            if not chunk:
                break
            chunks.append(chunk)
            bytes_read += len(chunk)
        if bytes_read > _MAX_NORMALIZED_INTAKE_BYTES:
            raise ValueError("normalized intake JSON exceeds the size limit")
    finally:
        os.close(descriptor)
    payload = json.loads(b"".join(chunks).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("normalized intake JSON must contain an object")
    return payload


def _resolve_project_intake(
    args: argparse.Namespace,
    *,
    required: bool,
) -> tuple[Path | None, dict[str, object] | None]:
    intake_root: Path | None = args.intake_root
    intake_dir: Path | None = args.intake_dir
    normalized_path: Path | None = args.normalized_intake
    if intake_root is not None:
        if intake_dir is not None or normalized_path is not None:
            raise ValueError(
                "--intake-root cannot be combined with --intake-dir or --normalized-intake"
            )
        intake_dir = create_owned_intake_dir(intake_root)
        try:
            print(f"OWNED_INTAKE_DIR={intake_dir}", flush=True)
            if sys.stdin.readline().strip() != "READY":
                raise ValueError("coordinated intake requires READY on stdin")
            normalized_path = intake_dir / _NORMALIZED_INTAKE_FILENAME
            return intake_dir, _load_normalized_intake(
                normalized_path,
                intake_dir=intake_dir,
            )
        except BaseException:
            discard_owned_intake_dir(intake_dir, intake_root=intake_root)
            raise

    if (intake_dir is None) != (normalized_path is None):
        if intake_dir is not None:
            discard_owned_intake_dir(intake_dir, intake_root=intake_dir.parent)
        raise ValueError("--intake-dir and --normalized-intake must be supplied together")
    if intake_dir is None or normalized_path is None:
        if required:
            raise ValueError("project operation requires normalized intake")
        return None, None
    try:
        return intake_dir, _load_normalized_intake(normalized_path, intake_dir=intake_dir)
    except BaseException:
        discard_owned_intake_dir(intake_dir, intake_root=intake_dir.parent)
        raise


def _discard_unconsumed_intake(intake_dir: Path | None) -> None:
    if intake_dir is not None and intake_dir.exists():
        discard_owned_intake_dir(intake_dir, intake_root=intake_dir.parent)


def build_parser() -> argparse.ArgumentParser:
    from .measurement_cli import add_measurement_commands
    from .observation_cli import add_observation_commands

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
    project = subparsers.add_parser("project", help="manage versioned audit projects")
    project_subparsers = project.add_subparsers(dest="project_command", required=True)
    add_measurement_commands(project_subparsers)
    add_observation_commands(project_subparsers)
    diagnose = project_subparsers.add_parser("diagnose", help="collect supplementary diagnostics")
    diagnose.add_argument("project_ref")
    diagnose.add_argument("--source-version", required=True)
    diagnose.add_argument("--clients-root", type=Path, required=True)
    diagnose.add_argument("--intake-root", type=Path, required=True)
    benchmark = project_subparsers.add_parser(
        "benchmark-prepare", help="print the frozen observation worksheet"
    )
    benchmark.add_argument("project_ref")
    benchmark.add_argument("--source-version", required=True)
    benchmark.add_argument("--clients-root", type=Path, required=True)
    create = project_subparsers.add_parser("create", help="create an initial public audit")
    create.add_argument("domain")
    create.add_argument("--clients-root", type=Path, required=True)
    create.add_argument("--project-id", required=True)
    create.add_argument("--client-name", required=True)
    create.add_argument("--report-locale", choices=("pl", "en"), required=True)
    create.add_argument("--max-pages", type=_bounded_max_pages, required=True)
    refresh = project_subparsers.add_parser(
        "refresh", help="append a fresh public evidence draft to an existing project"
    )
    refresh.add_argument("project_ref")
    refresh.add_argument("--clients-root", type=Path, required=True)
    refresh.add_argument("--source-version", required=True)
    refresh.add_argument("--expected-domain", required=True)
    refresh.add_argument("--expected-client-name", required=True)
    refresh.add_argument("--max-pages", type=_bounded_max_pages)
    update = project_subparsers.add_parser(
        "update", help="create a context version from normalized owner input"
    )
    update.add_argument("project_ref")
    update.add_argument("--clients-root", type=Path, required=True)
    update_intake = update.add_mutually_exclusive_group(required=True)
    update_intake.add_argument("--intake-dir", type=Path)
    update_intake.add_argument("--intake-root", type=Path)
    update.add_argument("--normalized-intake", type=Path)
    enrich = project_subparsers.add_parser(
        "enrich", help="create a context version from normalized visibility input"
    )
    enrich.add_argument("project_ref")
    enrich.add_argument("--clients-root", type=Path, required=True)
    enrich_intake = enrich.add_mutually_exclusive_group(required=True)
    enrich_intake.add_argument("--intake-dir", type=Path)
    enrich_intake.add_argument("--intake-root", type=Path)
    enrich.add_argument("--normalized-intake", type=Path)
    validate = project_subparsers.add_parser(
        "validate", help="create a fresh like-for-like validation version"
    )
    validate.add_argument("project_ref")
    validate.add_argument("--clients-root", type=Path, required=True)
    validate.add_argument("--implementation-date", type=_iso_date, required=True)
    validate_intake = validate.add_mutually_exclusive_group()
    validate_intake.add_argument("--intake-dir", type=Path)
    validate_intake.add_argument("--intake-root", type=Path)
    validate.add_argument("--normalized-intake", type=Path)
    validate.add_argument("--baseline-diagnostic-run")
    validate.add_argument("--follow-up-diagnostic-run")
    finalize = project_subparsers.add_parser("finalize", help="publish a reviewed client edition")
    finalize.add_argument("project_ref")
    finalize.add_argument("--clients-root", type=Path, required=True)
    finalize.add_argument("--version-id", required=True)
    finalize.add_argument("--markdown", type=Path, required=True)
    finalize.add_argument("--pdf", type=Path, required=True)
    finalize.add_argument("--reviewed-pdf-sha256", required=True)
    cover = finalize.add_mutually_exclusive_group(required=True)
    cover.add_argument("--hero", type=Path)
    cover.add_argument("--no-hero-reason")
    finalize.add_argument("--hero-source")
    diagnostics = finalize.add_mutually_exclusive_group()
    diagnostics.add_argument("--diagnostic-run")
    diagnostics.add_argument("--diagnostic-runs", nargs="+")
    finalize.add_argument(
        "--observation-projection-version", choices=("1.0.0", "1.1.0"), default="1.0.0"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "project" and args.project_command in {"observe-ai", "observation-report"}:
        from .observation_cli import observation_command

        return observation_command(args)
    if args.command == "project" and args.project_command in {"measure", "measurement-report"}:
        from .measurement_cli import measurement_command

        return measurement_command(args)
    if args.command == "project" and args.project_command in {"diagnose", "benchmark-prepare"}:
        contract = prepare_diagnostic_contract(
            args.project_ref, clients_root=args.clients_root, source_version=args.source_version
        )
        if args.project_command == "benchmark-prepare":
            print(contract.worksheet.model_dump_json(indent=2))
            return 0
        owned = create_owned_intake_dir(args.intake_root)
        try:
            print(f"OWNED_INTAKE_DIR={owned}", flush=True)
            print(f"DIAGNOSTIC_CONTRACT={contract.model_dump_json()}", flush=True)
            if sys.stdin.readline().strip() != "READY":
                raise ValueError("coordinated intake requires READY on stdin")
            destination = run_diagnostics(
                args.project_ref,
                clients_root=args.clients_root,
                source_version=args.source_version,
                owned_dir=owned,
                intake_root=args.intake_root,
            )
        finally:
            _discard_unconsumed_intake(owned)
        print(destination)
        return 0
    if args.command == "project" and args.project_command == "finalize":
        destination = finalize_client_report(
            args.project_ref,
            clients_root=args.clients_root,
            version_id=args.version_id,
            markdown_path=args.markdown,
            pdf_path=args.pdf,
            reviewed_pdf_sha256=args.reviewed_pdf_sha256,
            hero_path=args.hero,
            hero_source=args.hero_source,
            no_hero_reason=args.no_hero_reason,
            **(
                {"diagnostic_run_ref": args.diagnostic_run}
                if args.diagnostic_run is not None
                else {}
            ),
            diagnostic_run_refs=(
                tuple(args.diagnostic_runs) if args.diagnostic_runs is not None else None
            ),
            observation_projection_version=args.observation_projection_version,
        )
        print(destination / "client-report.pdf")
        return 0
    if args.command == "audit":
        run = run_public_audit(
            args.domain,
            output_dir=args.output_dir,
            max_pages=args.max_pages,
            geo_optimizer_command=args.geo_optimizer_command,
            report_locale=args.report_locale,
        )
        print(f"Audit {run.audit_id} written to {args.output_dir}")
    elif args.command == "project" and args.project_command == "create":
        manifest = create_project_audit(
            args.domain,
            clients_root=args.clients_root,
            project_id=args.project_id,
            client_name=args.client_name,
            report_locale=args.report_locale,
            max_pages=args.max_pages,
        )
        print(args.clients_root / manifest.project_id / manifest.versions[-1].relative_path)
    elif args.command == "project" and args.project_command == "refresh":
        manifest = refresh_project(
            args.project_ref,
            clients_root=args.clients_root,
            source_version=args.source_version,
            expected_domain=args.expected_domain,
            expected_client_name=args.expected_client_name,
            max_pages=args.max_pages,
        )
        print(args.clients_root / manifest.project_id / manifest.versions[-1].relative_path)
    elif args.command == "project" and args.project_command == "update":
        intake_dir, payload = _resolve_project_intake(args, required=True)
        assert intake_dir is not None and payload is not None
        try:
            manifest = update_project_context(
                args.project_ref,
                clients_root=args.clients_root,
                intake_dir=intake_dir,
                normalized_intake=payload,
            )
        except BaseException:
            _discard_unconsumed_intake(intake_dir)
            raise
        _discard_unconsumed_intake(intake_dir)
        print(args.clients_root / manifest.project_id / manifest.versions[-1].relative_path)
    elif args.command == "project" and args.project_command == "enrich":
        intake_dir, payload = _resolve_project_intake(args, required=True)
        assert intake_dir is not None and payload is not None
        try:
            manifest = enrich_project(
                args.project_ref,
                clients_root=args.clients_root,
                intake_dir=intake_dir,
                normalized_intake=payload,
            )
        except BaseException:
            _discard_unconsumed_intake(intake_dir)
            raise
        _discard_unconsumed_intake(intake_dir)
        print(args.clients_root / manifest.project_id / manifest.versions[-1].relative_path)
    elif args.command == "project" and args.project_command == "validate":
        if (args.baseline_diagnostic_run is None) != (args.follow_up_diagnostic_run is None):
            _discard_unconsumed_intake(args.intake_dir)
            raise ValueError("baseline and follow-up diagnostic runs must be supplied together")
        intake_dir, payload = _resolve_project_intake(args, required=False)
        try:
            manifest = validate_project(
                args.project_ref,
                clients_root=args.clients_root,
                intake_dir=intake_dir,
                normalized_intake=payload,
                implementation_date=args.implementation_date,
                **(
                    {
                        "baseline_diagnostic_run": args.baseline_diagnostic_run,
                        "follow_up_diagnostic_run": args.follow_up_diagnostic_run,
                    }
                    if args.baseline_diagnostic_run is not None
                    else {}
                ),
            )
        except BaseException:
            _discard_unconsumed_intake(intake_dir)
            raise
        _discard_unconsumed_intake(intake_dir)
        print(args.clients_root / manifest.project_id / manifest.versions[-1].relative_path)
    if args.command == "project":
        print(
            "Evidence bundle saved. Client delivery is pending: prepare the editorial Markdown, "
            "render and visually review it, then run project finalize. Do not deliver the "
            "technical PDF as the client edition.",
            file=sys.stderr,
        )
    return 0
