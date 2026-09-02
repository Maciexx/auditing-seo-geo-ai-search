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
        or set(options) != fields
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
) -> Path:
    """Validate explicit review and source binding; atomically publish a new edition."""
    store = ProjectStore(clients_root)
    project_root = store.resolve(project_ref)
    manifest = store.load(project_root.name)
    selected = next((item for item in manifest.versions if item.version_id == version_id), None)
    if selected is None:
        raise ValueError("audit version is not in the selected project")
    if hero_path is not None:
        if not hero_source or not hero_source.strip() or no_hero_reason is not None:
            raise ValueError("hero requires a source and cannot have a no-hero reason")
    elif not no_hero_reason or not no_hero_reason.strip() or hero_source is not None:
        raise ValueError("an explicit no-hero reason is required without an image")

    markdown = _read_input(markdown_path, limit=2 * 1024 * 1024)
    markdown.decode("utf-8")
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
