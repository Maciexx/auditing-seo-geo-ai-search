"""Publish a reviewed editorial edition without changing its evidence bundle.

This is a local, single-writer publication boundary, not a narrative generator.
The reviewed digest is an agent attestation, not an automated semantic review.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from importlib import resources
from pathlib import Path

from pypdf import PdfReader

from .diagnostic_models import DiagnosticBinding, DiagnosticRunReference
from .project_orchestrator import validate_project_bundle
from .project_store import ProjectStore, _rename_directory_no_replace
from .report_models import ClientReportData


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_input(path: Path, *, limit: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("report inputs must be regular files")
        content = stream.read(limit + 1)
    if not content or len(content) > limit:
        raise ValueError("report input is empty or exceeds the size limit")
    return content


def _directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("report destination must not be a symlink")
    path.mkdir(exist_ok=True)
    if not path.is_dir():
        raise ValueError("report destination must be a directory")


def _verify_editorial_render(
    metadata: Mapping[str, object],
    markdown: bytes,
    pdf: bytes,
    hero: bytes | None,
    no_hero_reason: str | None,
) -> None:
    """Reproduce the reviewed PDF; markers alone are not proof of source binding."""
    raw = metadata.get("/ClientEditionRenderOptions")
    if not isinstance(raw, str):
        raise ValueError("editorial render options are missing")
    options = json.loads(raw)
    fields = {
        "client",
        "title",
        "subtitle",
        "date",
        "version",
        "locale",
        "audit-id",
        "author",
        "confidentiality",
    }
    if (
        not isinstance(options, dict)
        or set(options) not in (fields, fields | {"observation-projection-version"})
        or options.get("observation-projection-version", "1.1.0") != "1.1.0"
        or any(not isinstance(value, str) or len(value) > 2000 for value in options.values())
    ):
        raise ValueError("editorial render options are invalid")
    checkout_script = Path(__file__).resolve().parents[2] / "scripts/render_client_pdf.py"
    resource = (
        checkout_script
        if checkout_script.is_file()
        else resources.files("ai_search_audit").joinpath("assets/render_client_pdf.py")
    )
    with (
        resources.as_file(resource) as script,
        tempfile.TemporaryDirectory(prefix="client-verify-") as temporary,
    ):
        directory = Path(temporary)
        source = directory / "source.md"
        output = directory / "render.pdf"
        source.write_bytes(markdown)
        command = [sys.executable, str(script), str(source), str(output)]
        for name, value in options.items():
            command.extend([f"--{name}", value])
        if hero is not None:
            image = directory / "cover.png"
            image.write_bytes(hero)
            command.extend(["--hero", str(image)])
        else:
            command.extend(["--no-hero-reason", no_hero_reason or ""])
        completed = subprocess.run(command, capture_output=True, timeout=30, check=False)
        if completed.returncode or not output.is_file() or output.read_bytes() != pdf:
            raise ValueError("editorial PDF does not match a fresh render of its source and cover")


def finalize_client_report(
    project_ref: str,
    *,
    clients_root: Path,
    version_id: str,
    markdown_path: Path,
    pdf_path: Path,
    reviewed_pdf_sha256: str,
    hero_path: Path | None = None,
    hero_source: str | None = None,
    no_hero_reason: str | None = None,
    diagnostic_run_ref: str | None = None,
    diagnostic_run_refs: tuple[str, ...] | None = None,
    observation_projection_version: str = "1.0.0",
) -> Path:
    """Validate explicit review and source binding; atomically publish a new edition."""
    if observation_projection_version not in {"1.0.0", "1.1.0"}:
        raise ValueError("unsupported observation projection version")
    if diagnostic_run_refs is not None and (
        diagnostic_run_ref is not None
        or not isinstance(diagnostic_run_refs, tuple)
        or not 1 <= len(diagnostic_run_refs) <= 2
        or any(not isinstance(ref, str) for ref in diagnostic_run_refs)
        or len(set(diagnostic_run_refs)) != len(diagnostic_run_refs)
    ):
        raise ValueError("invalid or mixed diagnostic run references")
    if diagnostic_run_ref is not None or diagnostic_run_refs is not None:
        from .diagnostic_workflow import _project_directory, _real_directory_path

        _real_directory_path(clients_root)
    store = ProjectStore(clients_root)
    project_root = store.resolve(project_ref)
    if diagnostic_run_ref is not None or diagnostic_run_refs is not None:
        _project_directory(project_root)
    manifest = store.load(project_root.name)
    selected = next((item for item in manifest.versions if item.version_id == version_id), None)
    if selected is None:
        raise ValueError("audit version is not in the selected project")
    diagnostic_provenance = None
    measurement_fragment = None
    observation_fragment = None
    plural_provenance: list[dict[str, str]] = []
    plural_bindings: list[tuple[DiagnosticRunReference, str, DiagnosticBinding]] = []
    if diagnostic_run_ref is not None:
        from .diagnostic_observations import DiagnosticObservationRun
        from .diagnostic_store import DiagnosticStore
        from .diagnostic_workflow import parse_diagnostic_run_reference

        reference = parse_diagnostic_run_reference(diagnostic_run_ref)
        loaded = DiagnosticStore(project_root).load(reference.source_version, reference.run_id)
        if isinstance(loaded.run, DiagnosticObservationRun):
            raise ValueError(
                "API observation finalization requires a guarded observation projection"
            )
        binding = loaded.run.binding
        if (
            binding.project_id != manifest.project_id
            or binding.source_version != selected.version_id
            or binding.audit_id != selected.audit_id
            or binding.report_locale != manifest.report_locale
            or binding.domain not in manifest.canonical_domains
        ):
            raise ValueError("diagnostic run does not match selected client edition")
        diagnostic_provenance = {
            "path": f"diagnostics/{reference.source_version}/{reference.run_id}",
            "source_audit_id": binding.audit_id,
            "manifest_sha256": loaded.manifest_sha256,
            "content_sha256": next(
                item.sha256 for item in loaded.manifest.files if item.filename == "diagnostics.json"
            ),
        }
        from .diagnostic_performance import DiagnosticRunV2
        from .measurement_report import load_measurement_report, render_measurement_fragment

        if isinstance(loaded.run, DiagnosticRunV2):
            projection = load_measurement_report(
                project_ref, clients_root=clients_root, diagnostic_run_ref=diagnostic_run_ref
            )
            if projection.manifest_sha256 != loaded.manifest_sha256:
                raise ValueError("measurement run changed during finalization")
            measurement_fragment = render_measurement_fragment(projection)
            diagnostic_provenance["measurement_fragment_sha256"] = _digest(
                measurement_fragment.encode("utf-8")
            )
    if diagnostic_run_refs is not None:
        from .diagnostic_observations import DiagnosticObservationRun
        from .diagnostic_performance import DiagnosticRunV2
        from .diagnostic_store import DiagnosticStore
        from .diagnostic_workflow import parse_diagnostic_run_reference
        from .measurement_report import load_measurement_report, render_measurement_fragment
        from .observation_report import load_observation_report, render_observation_fragment

        seen_kinds = set()
        for ref in diagnostic_run_refs:
            reference = parse_diagnostic_run_reference(ref)
            loaded = DiagnosticStore(project_root).load(reference.source_version, reference.run_id)
            binding = loaded.run.binding
            if (
                binding.project_id != manifest.project_id
                or binding.source_version != selected.version_id
                or binding.audit_id != selected.audit_id
                or binding.report_locale != manifest.report_locale
                or binding.domain not in manifest.canonical_domains
            ):
                raise ValueError("diagnostic run does not match selected client edition")
            if isinstance(loaded.run, DiagnosticRunV2):
                kind = "performance"
                projection = load_measurement_report(
                    project_ref, clients_root=clients_root, diagnostic_run_ref=ref
                )
                fragment = render_measurement_fragment(projection)
                measurement_fragment = fragment
                projection_hash = projection.manifest_sha256
            elif isinstance(loaded.run, DiagnosticObservationRun):
                kind = "observation"
                observation_projection = load_observation_report(
                    project_ref, clients_root=clients_root, diagnostic_run_ref=ref
                )
                fragment = render_observation_fragment(
                    observation_projection, projection_version=observation_projection_version
                )
                observation_fragment = fragment
                projection_hash = observation_projection.manifest_sha256
            else:
                raise ValueError(
                    "multi-run delivery supports only performance and API observations"
                )
            if kind in seen_kinds or projection_hash != loaded.manifest_sha256:
                raise ValueError("duplicate diagnostic kind or changed projection")
            seen_kinds.add(kind)
            plural_bindings.append((reference, loaded.manifest_sha256, binding))
            plural_provenance.append(
                {
                    "kind": kind,
                    "schema_version": loaded.run.schema_version,
                    "path": f"diagnostics/{reference.source_version}/{reference.run_id}",
                    "source_audit_id": binding.audit_id,
                    "manifest_sha256": loaded.manifest_sha256,
                    "content_sha256": next(
                        item.sha256
                        for item in loaded.manifest.files
                        if item.filename == "diagnostics.json"
                    ),
                    "client_fragment_sha256": _digest(fragment.encode("utf-8")),
                }
            )
        # The new boundary represents a sole observation or one combined edition.
        if "observation" not in seen_kinds:
            raise ValueError("multi-run delivery requires an observation projection")
        plural_provenance.sort(key=lambda item: {"performance": 0, "observation": 1}[item["kind"]])
    if hero_path is not None:
        if not hero_source or not hero_source.strip() or no_hero_reason is not None:
            raise ValueError("hero requires a source and cannot have a no-hero reason")
    elif not no_hero_reason or not no_hero_reason.strip() or hero_source is not None:
        raise ValueError("an explicit no-hero reason is required without an image")

    markdown = _read_input(markdown_path, limit=2 * 1024 * 1024)
    from .measurement_report import require_measurement_fragment
    from .observation_report import require_observation_fragment

    require_measurement_fragment(markdown.decode("utf-8"), measurement_fragment)
    require_observation_fragment(markdown.decode("utf-8"), observation_fragment)
    pdf_bytes = _read_input(pdf_path, limit=30 * 1024 * 1024)
    if reviewed_pdf_sha256 != _digest(pdf_bytes):
        raise ValueError("PDF changed since visual/evidence review; review it again")
    hero = _read_input(hero_path, limit=20 * 1024 * 1024) if hero_path else None
    pdf = PdfReader(io.BytesIO(pdf_bytes), strict=True)
    metadata: Mapping[str, object] = pdf.metadata or {}
    if metadata.get("/ClientEditionTemplate") != "editorial-v1":
        raise ValueError(
            "an editorial PDF is required; the technical report is not a final edition"
        )
    expected = {
        "/ClientEditionSourceSHA256": _digest(markdown),
        "/ClientEditionHeroSHA256": _digest(hero) if hero else "none",
        "/ClientEditionNoHeroReason": no_hero_reason or "",
        "/ClientEditionLocale": manifest.report_locale,
        "/ClientEditionClient": manifest.client_name,
        "/ClientEditionVersion": selected.version_id,
        "/ClientEditionAuditID": selected.audit_id,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"editorial source or identity mismatch: {field}")
    if not 2 <= len(pdf.pages) <= 100:
        raise ValueError("editorial PDF must contain a cover and content, at most 100 pages")
    if any(
        abs(float(page.mediabox.width) - 595.276) > 1
        or abs(float(page.mediabox.height) - 841.89) > 1
        for page in pdf.pages
    ):
        raise ValueError("editorial PDF must use A4 portrait pages")
    if bool(pdf.pages[0].images) != (hero is not None):
        raise ValueError("cover image is missing or contradicts the declared cover choice")
    if observation_fragment is not None:
        render_options = json.loads(str(metadata.get("/ClientEditionRenderOptions", "null")))
        if (
            not isinstance(render_options, dict)
            or render_options.get("observation-projection-version", "1.0.0")
            != observation_projection_version
        ):
            raise ValueError("renderer observation projection version differs from frozen fragment")
    _verify_editorial_render(metadata, markdown, pdf_bytes, hero, no_hero_reason)

    source_root = project_root / selected.relative_path
    snapshot = validate_project_bundle(
        source_root,
        expected_project_id=manifest.project_id,
        expected_version_number=selected.version_number,
        expected_source_audit_id=selected.source_audit_id,
        expect_owner_context=(source_root / "aggregates/owner-context.json").is_file(),
        expect_visibility_metrics=(source_root / "aggregates/visibility-metrics.json").is_file(),
        expect_validation_comparison=(
            source_root / "aggregates/validation-comparison.json"
        ).is_file(),
        expected_stage=selected.stage,
    )
    canonical = ClientReportData.model_validate_json(
        snapshot.files["engine/client-report-data.json"].content
    )
    if (
        canonical.audit_id != selected.audit_id
        or canonical.report_locale != manifest.report_locale
        or canonical.project is None
        or canonical.project.project_id != manifest.project_id
        or canonical.project.version_id != selected.version_id
        or canonical.project.version_number != selected.version_number
        or canonical.project.source_audit_id != selected.source_audit_id
        or canonical.project.stage != selected.stage
        or canonical.project.report_status != selected.report_status
        or canonical.target_domain not in manifest.canonical_domains
    ):
        raise ValueError("project identity or report status differs from its canonical audit")

    files = {"client-report.pdf": pdf_bytes, "client-report.md": markdown}
    if hero is not None and hero_path is not None:
        suffix = hero_path.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("cover must be a PNG, JPEG or WebP image")
        files[f"cover{suffix}"] = hero
    record = {
        "schema_version": "1.0.0",
        "kind": "client-edition",
        "project_id": manifest.project_id,
        "audit_id": selected.audit_id,
        "audit_version_id": selected.version_id,
        "report_locale": manifest.report_locale,
        "report_status": selected.report_status.value,
        "source_bundle": selected.relative_path,
        "source_audit_sha256": snapshot.files["engine/audit.json"].sha256,
        "source_manifest_sha256": snapshot.files["manifests/output-manifest.json"].sha256,
        "reviewed_pdf_sha256": reviewed_pdf_sha256,
        "review_scope": "agent-attested-evidence-and-visual-review",
        "hero_source": hero_source,
        "no_hero_reason": no_hero_reason,
        "renderer_sha256": metadata.get("/ClientEditionRendererSHA256"),
        "artifacts": {
            name: {"sha256": _digest(data), "bytes": len(data)} for name, data in files.items()
        },
    }
    if diagnostic_provenance is not None:
        if loaded.run.binding.domain != canonical.target_domain:
            raise ValueError("diagnostic run domain does not match selected client edition")
        record["diagnostic_run"] = diagnostic_provenance
    if diagnostic_run_refs is not None:
        for reference, expected_hash, binding in plural_bindings:
            if (
                binding.domain != canonical.target_domain
                or binding.source_sha256 != snapshot.files["engine/audit.json"].sha256
            ):
                raise ValueError("diagnostic source does not match selected client edition")
            fresh = DiagnosticStore(project_root).load(reference.source_version, reference.run_id)
            if fresh.manifest_sha256 != expected_hash or fresh.run.binding != binding:
                raise ValueError("diagnostic run changed during finalization")
        record["schema_version"] = "2.0.0"
        record["diagnostic_runs"] = plural_provenance
    reports = project_root / "reports"
    _directory(reports)
    editions = reports / selected.version_id
    _directory(editions)
    numbers = []
    for path in editions.iterdir():
        match = re.fullmatch(r"edition-([1-9][0-9]*)", path.name)
        if match:
            if path.is_symlink() or not path.is_dir():
                raise ValueError("existing edition must be a real directory, not a symlink")
            numbers.append(int(match.group(1)))
    edition = max(numbers, default=0) + 1
    record["edition"] = edition
    files["delivery.json"] = (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode()
    destination = editions / f"edition-{edition}"
    with tempfile.TemporaryDirectory(prefix=".edition-", dir=editions) as temporary:
        staging = Path(temporary) / "ready"
        staging.mkdir()
        for name, content in files.items():
            (staging / name).write_bytes(content)
        _rename_directory_no_replace(staging, destination)
    return destination
