from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import ai_search_audit.data_intake as data_intake
from ai_search_audit.data_intake import (
    MAX_METRIC_DIMENSIONS,
    MAX_METRIC_FILTERS,
    MAX_SOURCE_FILTERS,
    DateRange,
    FactApprovalState,
    IntakeCleanupError,
    IntakeOwnershipError,
    IntakeValidationError,
    NonSensitiveExample,
    NormalizedIntake,
    OwnerFactInput,
    ProcessedIntake,
    SourceArtifactDeclaration,
    SourceArtifactProvenance,
    VisibilityMetricPoint,
    VisibilityMetricSeriesInput,
    VisibilitySource,
    consume_intake,
    create_owned_intake_dir,
    discard_owned_intake_dir,
)
from ai_search_audit.models import DataState

NOW = datetime(2026, 9, 1, 9, 30, tzinfo=UTC)
FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "intake"


def source_declaration(
    filename: str,
    content: bytes,
    *,
    source_id: str = "source-1",
    **overrides: Any,
) -> SourceArtifactDeclaration:
    values: dict[str, Any] = {
        "source_id": source_id,
        "filename": filename,
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
        "platform": VisibilitySource.GOOGLE_SEARCH_CONSOLE,
        "report_type": "search-performance",
    }
    values.update(overrides)
    return SourceArtifactDeclaration(**values)


def normalized_intake(
    sources: tuple[SourceArtifactDeclaration, ...],
    **overrides: Any,
) -> NormalizedIntake:
    values: dict[str, Any] = {
        "project_id": "generic-example",
        "canonical_domain": "generic.example",
        "sources": sources,
    }
    values.update(overrides)
    return NormalizedIntake(**values)


def source_provenance(
    *,
    processed_at: datetime = NOW,
    deleted_at: datetime = NOW,
) -> SourceArtifactProvenance:
    source = source_declaration("report.csv", b"example")
    return SourceArtifactProvenance(
        **source.model_dump(),
        processed_at=processed_at,
        deleted_at=deleted_at,
    )


def processed_intake(**overrides: Any) -> ProcessedIntake:
    values: dict[str, Any] = {
        "project_id": "generic-example",
        "canonical_domain": "generic.example",
        "processed_at": NOW,
        "deleted_at": NOW,
        "sources": (source_provenance(),),
    }
    values.update(overrides)
    return ProcessedIntake(**values)


def write_forged_marker(directory: Path, *, nonce: str = "f" * 32) -> None:
    root_details = directory.parent.stat()
    directory_details = directory.stat()
    run_id = directory.name
    marker = {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "nonce": nonce,
        "directory_device": directory_details.st_dev,
        "directory_inode": directory_details.st_ino,
        "root_device": root_details.st_dev,
        "root_inode": root_details.st_ino,
    }
    (directory / ".ai-search-audit-owned-intake.json").write_text(
        json.dumps(marker, sort_keys=True, separators=(",", ":"))
    )


def invalid_normalized_intake() -> NormalizedIntake:
    return normalized_intake((source_declaration("gsc.csv", b"valid"),))


def create_source(
    tmp_path: Path,
    *,
    filename: str = "gsc.csv",
    content: bytes = b"date,clicks\n2026-08-31,0\n",
) -> tuple[Path, NormalizedIntake]:
    inbox = create_owned_intake_dir(tmp_path / "intake", now=NOW)
    artifact = inbox / filename
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(content)
    declaration = source_declaration(filename, content)
    return inbox, normalized_intake((declaration,))


def test_failed_diagnostic_processor_removes_owned_inputs(tmp_path: Path) -> None:
    root = tmp_path / "intake"
    owned = create_owned_intake_dir(root)
    (owned / "source.txt").write_text("synthetic public input")

    def reject(directory_fd: int) -> None:
        assert os.path.isdir(directory_fd)
        raise ValueError("invalid diagnostic")

    with pytest.raises(ValueError, match="invalid diagnostic"):
        data_intake.consume_owned_payload(owned, intake_root=root, processor=reject)
    assert not owned.exists()


def test_owned_payload_returns_processor_result_only_after_cleanup(tmp_path: Path) -> None:
    root = tmp_path / "intake"
    owned = create_owned_intake_dir(root)
    sibling = root / "sibling"
    sibling.mkdir()
    (sibling / "keep.txt").write_text("keep")
    (owned / "source.txt").write_text("synthetic public input")
    descriptors = []
    result = object()

    def process(directory_fd: int) -> object:
        descriptors.append(directory_fd)
        assert owned.exists()
        assert "source.txt" in os.listdir(directory_fd)
        return result

    assert data_intake.consume_owned_payload(owned, intake_root=root, processor=process) is result
    assert not owned.exists()
    assert (sibling / "keep.txt").read_text() == "keep"
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt])
def test_owned_payload_failure_closes_descriptors_and_identity_anchors(tmp_path, error):
    root = tmp_path / "intake"
    owned = create_owned_intake_dir(root)
    issued = next(
        value for value in data_intake._OWNED_CAPABILITIES.values() if value.path == owned
    )
    descriptors = list(issued.identity_anchors)

    def process(directory_fd):
        descriptors.append(directory_fd)
        raise error("processor failed")

    with pytest.raises(error, match="processor failed"):
        data_intake.consume_owned_payload(owned, intake_root=root, processor=process)
    assert not owned.exists()
    assert issued not in data_intake._OWNED_CAPABILITIES.values()
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_owned_payload_failed_cleanup_prevents_return_and_post_consume_publication(
    tmp_path, monkeypatch
):
    root = tmp_path / "intake"
    owned = create_owned_intake_dir(root)
    called = []
    published = []

    def process(directory_fd):
        called.append(directory_fd)
        return "validated-result"

    def fail_cleanup(*args):
        raise OSError("injected deletion failure")

    monkeypatch.setattr(data_intake, "_delete_quarantined_directory", fail_cleanup)
    with pytest.raises(IntakeCleanupError):
        result = data_intake.consume_owned_payload(owned, intake_root=root, processor=process)
        published.append(result)
    assert len(called) == 1
    assert published == []
    with pytest.raises(IntakeOwnershipError):
        data_intake.consume_owned_payload(owned, intake_root=root, processor=process)
    assert len(called) == 1


@pytest.mark.parametrize("target_name", ["intake", "sibling"])
def test_owned_payload_never_claims_root_or_sibling(tmp_path, target_name):
    root = tmp_path / "intake"
    owned = create_owned_intake_dir(root)
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    (sibling / "keep.txt").write_text("keep")

    def forbidden(directory_fd):
        pytest.fail("unowned target must not reach processor")

    with pytest.raises(IntakeOwnershipError):
        data_intake.consume_owned_payload(
            tmp_path / target_name, intake_root=root, processor=forbidden
        )
    assert (sibling / "keep.txt").read_text() == "keep"
    assert owned.exists()
    discard_owned_intake_dir(owned, intake_root=root)


def test_owned_payload_root_replacement_cannot_return_success_or_delete_replacement(tmp_path):
    root = tmp_path / "intake"
    owned = create_owned_intake_dir(root)
    moved_root = tmp_path / "original-intake"
    published = []

    def replace_root(directory_fd):
        root.rename(moved_root)
        root.mkdir()
        (root / "keep.txt").write_text("keep")
        return "validated-result"

    with pytest.raises(IntakeCleanupError, match="root identity"):
        published.append(
            data_intake.consume_owned_payload(owned, intake_root=root, processor=replace_root)
        )
    assert published == []
    assert (root / "keep.txt").read_text() == "keep"
    assert (moved_root / owned.name).is_dir()


def test_owned_payload_sibling_replacement_at_quarantine_is_preserved(tmp_path, monkeypatch):
    root = tmp_path / "intake"
    owned = create_owned_intake_dir(root)
    saved = root / "saved-owned"
    sibling = root / "sibling"
    sibling.mkdir()
    (sibling / "keep.txt").write_text("keep")
    actual_rename = data_intake._rename_to_quarantine
    published = []

    def substitute(root_fd, owned_name, quarantine_fd):
        owned.rename(saved)
        sibling.rename(owned)
        actual_rename(root_fd, owned_name, quarantine_fd)

    monkeypatch.setattr(data_intake, "_rename_to_quarantine", substitute)
    with pytest.raises(IntakeCleanupError, match="identity"):
        published.append(
            data_intake.consume_owned_payload(
                owned, intake_root=root, processor=lambda fd: "result"
            )
        )
    assert published == []
    assert (owned / "keep.txt").read_text() == "keep"
    assert saved.is_dir()


def test_diagnostic_fields_do_not_widen_legacy_intake_contracts():
    for field in (
        "rendered_captures",
        "expected_binding",
        "worksheet",
        "responses",
        "baseline_run",
    ):
        with pytest.raises(ValidationError, match="Extra inputs"):
            normalized_intake((), **{field: {}})
        with pytest.raises(ValidationError, match="Extra inputs"):
            processed_intake(**{field: {}})
    assert "diagnostic" not in {source.value for source in VisibilitySource}


@pytest.mark.parametrize("suffix", (".csv", ".xlsx", ".json", ".pdf", ".png", ".jpg", ".jpeg"))
def test_source_declaration_accepts_declared_source_suffixes(suffix: str) -> None:
    declaration = source_declaration(f"report{suffix.upper()}", b"example")

    assert declaration.filename == f"report{suffix.upper()}"


@pytest.mark.parametrize("suffix", (".exe", ".sh", ".zip", ".tar", ".unknown", ""))
def test_source_declaration_rejects_executable_archive_and_unknown_suffixes(
    suffix: str,
) -> None:
    with pytest.raises(ValidationError, match="supported source suffix"):
        source_declaration(f"report{suffix}", b"example")


def test_source_declaration_retains_normalized_metadata() -> None:
    exported_at = datetime(2026, 8, 31, 18, tzinfo=UTC)
    declaration = source_declaration(
        "exports/search.csv",
        b"example",
        date_range=DateRange(start=date(2026, 8, 1), end=date(2026, 8, 31)),
        filters=("country=PL", "search_type=web"),
        exported_at=exported_at,
    )

    assert declaration.source_id == "source-1"
    assert declaration.filename == "exports/search.csv"
    assert declaration.platform is VisibilitySource.GOOGLE_SEARCH_CONSOLE
    assert declaration.report_type == "search-performance"
    assert declaration.date_range == DateRange(start=date(2026, 8, 1), end=date(2026, 8, 31))
    assert declaration.filters == ("country=PL", "search_type=web")
    assert declaration.exported_at == exported_at
    assert declaration.sha256 == hashlib.sha256(b"example").hexdigest()


def test_source_declaration_rejects_naive_export_and_range_after_export() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        source_declaration(
            "report.csv",
            b"example",
            exported_at=datetime(2026, 9, 1, 8),
        )


def test_report_type_uses_visibility_scope_guard_without_rejecting_owner_context() -> None:
    assert (
        source_declaration(
            "owner.json",
            b"example",
            report_type="owner-context",
        ).report_type
        == "owner-context"
    )
    with pytest.raises(ValidationError, match="visibility-only"):
        source_declaration(
            "report.csv",
            b"example",
            report_type="conversion-report",
        )


@pytest.mark.parametrize(
    "label",
    ("metric=con-version", "metric=re-venue", "metric=con\u200bversion"),
)
def test_visibility_scope_guard_rejects_separator_and_unicode_splitting(label: str) -> None:
    with pytest.raises(ValidationError, match="visibility-only"):
        source_declaration("report.csv", b"example", filters=(label,))


def test_visibility_scope_guard_does_not_false_positive_leadership() -> None:
    declaration = source_declaration(
        "report.csv",
        b"example",
        filters=("topic=leadership",),
    )
    assert declaration.filters == ("topic=leadership",)


@pytest.mark.parametrize(
    "label",
    ("metric=le-ad", "metric=le\u200bad", "metric=l-e-a-d", "metric=c-r-m"),
)
def test_visibility_scope_guard_rejects_split_short_commercial_terms(label: str) -> None:
    with pytest.raises(ValidationError, match="visibility-only"):
        source_declaration("report.csv", b"example", filters=(label,))


@pytest.mark.parametrize("report_type", ("le-ad", "le\u200bad", "l-e-a-d", "c-r-m"))
def test_report_type_rejects_split_short_commercial_terms(report_type: str) -> None:
    with pytest.raises(ValidationError, match="visibility-only"):
        source_declaration("report.csv", b"example", report_type=report_type)


@pytest.mark.parametrize("metric_id", ("le-ad", "le\u200bad", "l-e-a-d", "c-r-m"))
def test_metric_id_rejects_split_short_commercial_terms(metric_id: str) -> None:
    with pytest.raises(ValidationError, match="visibility-only"):
        VisibilityMetricSeriesInput(
            metric_id=metric_id,
            source_id="source-1",
            metric="gsc.clicks",
            unit="clicks",
            state=DataState.UNAVAILABLE,
            coverage=0,
            confidence=0,
        )


@pytest.mark.parametrize(
    "label",
    (
        "metric=le-ad-count",
        "source=c-r-m-report",
        "segment=qualified-le-ad-traffic",
        "le-ad traffic",
        "c-r-m export",
        "metric=con-ver-sion-rate",
        "metric=re-ve-nue-total",
    ),
)
def test_visibility_scope_guard_rejects_forbidden_token_ngrams(label: str) -> None:
    with pytest.raises(ValidationError, match="visibility-only"):
        source_declaration("report.csv", b"example", filters=(label,))


def test_source_declaration_rejects_date_range_after_export() -> None:
    with pytest.raises(ValidationError, match="date range"):
        source_declaration(
            "report.csv",
            b"example",
            date_range=DateRange(start=date(2026, 9, 1), end=date(2026, 9, 2)),
            exported_at=datetime(2026, 9, 1, 8, tzinfo=UTC),
        )


def test_source_provenance_rejects_impossible_trusted_chronology() -> None:
    source = source_declaration(
        "report.csv",
        b"example",
        exported_at=NOW.replace(hour=10),
    )
    with pytest.raises(ValidationError, match="exported_at"):
        SourceArtifactProvenance(
            **source.model_dump(),
            processed_at=NOW,
            deleted_at=NOW,
        )
    ranged = source_declaration(
        "range.csv",
        b"example",
        date_range=DateRange(start=date(2026, 9, 1), end=date(2026, 9, 2)),
    )
    with pytest.raises(ValidationError, match="processed_at"):
        SourceArtifactProvenance(
            **ranged.model_dump(),
            processed_at=NOW,
            deleted_at=NOW,
        )


def test_source_and_series_metadata_bounds_and_uniqueness() -> None:
    source_filters = tuple(f"country=region-{index}" for index in range(MAX_SOURCE_FILTERS))
    assert len(source_declaration("report.csv", b"example", filters=source_filters).filters) == 32
    with pytest.raises(ValidationError, match="32"):
        source_declaration("report.csv", b"example", filters=(*source_filters, "extra=true"))
    with pytest.raises(ValidationError, match="unique"):
        source_declaration("report.csv", b"example", filters=("country=PL", "country=PL"))

    dimensions = tuple(f"dimension-{index}" for index in range(MAX_METRIC_DIMENSIONS))
    filters = tuple(f"filter-{index}=yes" for index in range(MAX_METRIC_FILTERS))
    values = {
        "metric_id": "metric",
        "source_id": "source",
        "metric": "gsc.clicks",
        "unit": "clicks",
        "state": DataState.AVAILABLE,
        "coverage": 1,
        "confidence": 1,
        "points": (VisibilityMetricPoint(period_start=date(2026, 8, 1), value=1),),
    }
    assert len(VisibilityMetricSeriesInput(**values, dimensions=dimensions).dimensions) == 32
    assert len(VisibilityMetricSeriesInput(**values, filters=filters).filters) == 32
    with pytest.raises(ValidationError, match="32"):
        VisibilityMetricSeriesInput(**values, dimensions=(*dimensions, "extra"))
    with pytest.raises(ValidationError, match="32"):
        VisibilityMetricSeriesInput(**values, filters=(*filters, "extra=yes"))


@pytest.mark.parametrize("byte_count", (True, "7"))
def test_source_declaration_rejects_non_strict_byte_count(byte_count: object) -> None:
    with pytest.raises(ValidationError):
        source_declaration("report.csv", b"example", byte_count=byte_count)


@pytest.mark.parametrize(
    "filename",
    (
        "../outside.csv",
        "nested/../../outside.csv",
        "/absolute.csv",
        r"C:\absolute.csv",
        r"nested\report.csv",
        "./report.csv",
        "report.csv/",
    ),
)
def test_source_declaration_rejects_unsafe_paths(filename: str) -> None:
    with pytest.raises(ValidationError, match="safe relative path"):
        source_declaration(filename, b"example")


def test_date_range_rejects_reverse_dates() -> None:
    with pytest.raises(ValidationError, match="end must not precede start"):
        DateRange(start=date(2026, 9, 1), end=date(2026, 8, 1))


def test_normalized_intake_rejects_duplicate_source_ids() -> None:
    first = source_declaration("one.csv", b"one", source_id="duplicate")
    second = source_declaration("two.csv", b"two", source_id="duplicate")

    with pytest.raises(ValidationError, match="unique source_id"):
        normalized_intake((first, second))


def test_normalized_and_processed_intake_publish_explicit_collection_bounds() -> None:
    expected_limits = {
        "owner_facts": 500,
        "metric_series": 200,
        "cited_examples": 100,
        "sources": 32,
    }

    for model in (NormalizedIntake, ProcessedIntake):
        properties = model.model_json_schema()["properties"]
        assert {
            field: properties[field]["maxItems"] for field in expected_limits
        } == expected_limits


def test_normalized_intake_rejects_duplicate_filenames_case_insensitively() -> None:
    first = source_declaration("Report.csv", b"one", source_id="one")
    second = source_declaration("report.CSV", b"two", source_id="two")

    with pytest.raises(ValidationError, match="unique filename"):
        normalized_intake((first, second))


def test_normalized_intake_rejects_unknown_source_references() -> None:
    declaration = source_declaration("report.csv", b"example")

    with pytest.raises(ValidationError, match="declared source_id"):
        normalized_intake(
            (declaration,),
            owner_facts=(
                OwnerFactInput(
                    fact_id="fact-1",
                    field="preferred_brand_name",
                    value="Example",
                    source_id="not-declared",
                ),
            ),
        )


def test_owner_fact_defaults_to_unknown_approval_with_no_conflicts() -> None:
    fact = OwnerFactInput(
        fact_id="fact-1",
        field="preferred_brand_name",
        value=None,
        source_id="source-1",
    )

    assert fact.approval_state is FactApprovalState.UNKNOWN
    assert fact.conflict_ids == ()
    assert fact.resolved_conflict_ids == ()


def test_owner_fact_retains_approved_disjoint_conflict_ids() -> None:
    fact = OwnerFactInput(
        fact_id="fact-1",
        field="preferred_brand_name",
        value="Generic Example",
        source_id="source-1",
        approval_state=FactApprovalState.APPROVED,
        conflict_ids=("conflict-1",),
        resolved_conflict_ids=("conflict-2",),
    )

    assert fact.approval_state is FactApprovalState.APPROVED
    assert fact.conflict_ids == ("conflict-1",)
    assert fact.resolved_conflict_ids == ("conflict-2",)


@pytest.mark.parametrize(
    ("conflict_ids", "resolved_conflict_ids", "message"),
    (
        (("conflict-1", "conflict-1"), (), "unique conflict_ids"),
        ((), ("conflict-1", "conflict-1"), "unique resolved_conflict_ids"),
        (("conflict-1",), ("conflict-1",), "disjoint"),
    ),
)
def test_owner_fact_rejects_duplicate_or_overlapping_conflicts(
    conflict_ids: tuple[str, ...],
    resolved_conflict_ids: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        OwnerFactInput(
            fact_id="fact-1",
            field="preferred_brand_name",
            value="Generic Example",
            source_id="source-1",
            conflict_ids=conflict_ids,
            resolved_conflict_ids=resolved_conflict_ids,
        )


def test_owner_fact_cannot_set_report_status() -> None:
    values = {
        "fact_id": "fact-1",
        "field": "preferred_brand_name",
        "value": "Generic Example",
        "source_id": "source-1",
        "report_status": "CLIENT_VALIDATED",
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        OwnerFactInput.model_validate(values)


def test_owner_fact_rejects_non_scalar_value() -> None:
    values = {
        "fact_id": "fact-1",
        "field": "preferred_brand_name",
        "value": {"raw": "not allowed"},
        "source_id": "source-1",
    }

    with pytest.raises(ValidationError):
        OwnerFactInput.model_validate(values)


def test_consume_intake_validates_files_and_returns_only_retained_data(tmp_path: Path) -> None:
    content = b"date,clicks\n2026-08-31,0\n"
    inbox = create_owned_intake_dir(tmp_path / "intake", now=NOW)
    source_path = inbox / "exports" / "gsc.csv"
    source_path.parent.mkdir()
    source_path.write_bytes(content)
    source = source_declaration("exports/gsc.csv", content)
    metric = VisibilityMetricSeriesInput(
        metric_id="organic-clicks",
        source_id=source.source_id,
        metric="clicks",
        unit="count",
        state=DataState.AVAILABLE,
        coverage=1,
        confidence=0.8,
        points=(
            VisibilityMetricPoint(
                period_start=date(2026, 8, 31),
                period_end=date(2026, 8, 31),
                value=0,
            ),
        ),
    )
    intake = normalized_intake(
        (source,),
        owner_facts=(
            OwnerFactInput(
                fact_id="fact-1",
                field="preferred_brand_name",
                value="Generic Example",
                source_id=source.source_id,
            ),
        ),
        metric_series=(metric,),
        cited_examples=(
            NonSensitiveExample(
                example_id="example-1",
                source_id=source.source_id,
                description="Aggregate reporting example",
                citation="generic.example",
                non_sensitive=True,
            ),
        ),
    )

    processed = consume_intake(inbox, intake, intake_root=tmp_path / "intake", now=NOW)

    assert not inbox.exists()
    assert processed.processed_at == NOW
    assert processed.deleted_at == NOW
    assert processed.sources[0].processed_at == NOW
    assert processed.sources[0].deleted_at == NOW
    assert processed.metric_series[0].points[0].value == 0
    assert processed.owner_facts == intake.owner_facts
    assert processed.cited_examples == intake.cited_examples
    retained = processed.model_dump(mode="json")
    serialized = json.dumps(retained, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "date,clicks" not in serialized
    assert "raw_rows" not in serialized
    assert "raw_bytes" not in serialized
    assert "payload" not in serialized
    assert "blob" not in serialized
    assert "version_id" not in serialized
    assert "latest_audit_id" not in serialized


def test_processor_receives_only_normalized_intake_and_runs_before_deletion(
    tmp_path: Path,
) -> None:
    inbox, intake = create_source(tmp_path)
    received: list[NormalizedIntake] = []

    def processor(value: NormalizedIntake) -> None:
        assert inbox.exists()
        received.append(value)

    processed = consume_intake(
        inbox,
        intake,
        intake_root=tmp_path / "intake",
        now=NOW,
        processor=processor,
    )

    assert received == [intake]
    assert processed.metric_series == intake.metric_series
    assert not inbox.exists()


def test_discard_deletes_abandoned_owned_intake_and_retires_capability(
    tmp_path: Path,
) -> None:
    inbox = create_owned_intake_dir(tmp_path / "intake", now=NOW)
    (inbox / "unused.csv").write_bytes(b"unused")

    discard_owned_intake_dir(inbox, intake_root=tmp_path / "intake")

    assert not inbox.exists()
    with pytest.raises(IntakeOwnershipError):
        discard_owned_intake_dir(inbox, intake_root=tmp_path / "intake")


def test_discard_deletes_owned_intake_with_damaged_marker_content(
    tmp_path: Path,
) -> None:
    inbox = create_owned_intake_dir(tmp_path / "intake", now=NOW)
    marker = next(inbox.iterdir())
    marker.write_text("damaged")
    (inbox / "unused.csv").write_bytes(b"unused")

    discard_owned_intake_dir(inbox, intake_root=tmp_path / "intake")

    assert not inbox.exists()


def test_discard_rejects_forged_directory_without_deleting_it(tmp_path: Path) -> None:
    intake_root = tmp_path / "intake"
    intake_root.mkdir()
    nonce = "e" * 32
    forged = intake_root / f"run-20260901T093000Z-{nonce}"
    forged.mkdir()
    write_forged_marker(forged, nonce=nonce)
    keep = forged / "keep.txt"
    keep.write_text("keep")

    with pytest.raises(IntakeOwnershipError, match="capability"):
        discard_owned_intake_dir(forged, intake_root=intake_root)

    assert keep.read_text() == "keep"


def test_discarded_capability_cannot_be_replayed(tmp_path: Path) -> None:
    intake_root = tmp_path / "intake"
    inbox = create_owned_intake_dir(intake_root, now=NOW)
    run_id = inbox.name
    nonce = run_id.rsplit("-", 1)[-1]
    discard_owned_intake_dir(inbox, intake_root=intake_root)
    inbox.mkdir()
    write_forged_marker(inbox, nonce=nonce)
    keep = inbox / "keep.txt"
    keep.write_text("keep")

    with pytest.raises(IntakeOwnershipError):
        discard_owned_intake_dir(inbox, intake_root=intake_root)

    assert keep.read_text() == "keep"


def test_discard_cleanup_failure_retires_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intake_root = tmp_path / "intake"
    inbox = create_owned_intake_dir(intake_root, now=NOW)
    run_id = inbox.name
    nonce = run_id.rsplit("-", 1)[-1]
    (inbox / "unused.csv").write_bytes(b"unused")

    def fail_delete(
        _root_fd: int,
        _quarantine_name: str,
        _quarantine_identity: tuple[int, int],
    ) -> None:
        raise OSError("deletion blocked")

    monkeypatch.setattr(data_intake, "_delete_quarantined_directory", fail_delete)

    with pytest.raises(IntakeCleanupError, match="could not delete"):
        discard_owned_intake_dir(inbox, intake_root=intake_root)

    assert not inbox.exists()
    inbox.mkdir()
    write_forged_marker(inbox, nonce=nonce)
    with pytest.raises(IntakeOwnershipError, match="capability"):
        discard_owned_intake_dir(inbox, intake_root=intake_root)


def test_failed_processing_deletes_inputs_and_returns_no_retained_result(tmp_path: Path) -> None:
    inbox = create_owned_intake_dir(tmp_path / "intake")
    (inbox / "gsc.csv").write_bytes(b"invalid")
    with pytest.raises(IntakeValidationError):
        consume_intake(inbox, invalid_normalized_intake(), intake_root=tmp_path / "intake")
    assert not inbox.exists()


def test_processor_failure_deletes_inputs_and_propagates_failure(tmp_path: Path) -> None:
    inbox, intake = create_source(tmp_path)

    def fail(_value: NormalizedIntake) -> None:
        raise RuntimeError("processor failed")

    with pytest.raises(RuntimeError, match="processor failed"):
        consume_intake(
            inbox,
            intake,
            intake_root=tmp_path / "intake",
            processor=fail,
        )

    assert not inbox.exists()


def test_model_failure_inside_boundary_deletes_inputs(tmp_path: Path) -> None:
    inbox = create_owned_intake_dir(tmp_path / "intake")
    (inbox / "report.csv").write_bytes(b"example")
    invalid_model = {
        "project_id": "generic-example",
        "canonical_domain": "generic.example",
        "sources": [],
        "credentials": "must-not-be-accepted",
    }

    with pytest.raises(IntakeValidationError, match="normalized intake"):
        consume_intake(
            inbox,
            invalid_model,  # type: ignore[arg-type]
            intake_root=tmp_path / "intake",
        )

    assert not inbox.exists()


def test_boundary_revalidates_constructed_models_before_file_access(tmp_path: Path) -> None:
    intake_root = tmp_path / "intake"
    inbox = create_owned_intake_dir(intake_root)
    outside = intake_root / "outside.csv"
    outside.write_bytes(b"outside")
    unsafe_source = SourceArtifactDeclaration.model_construct(
        source_id="source-1",
        filename="../outside.csv",
        sha256=hashlib.sha256(b"outside").hexdigest(),
        byte_count=len(b"outside"),
        platform=VisibilitySource.MANUAL,
        report_type="manual",
    )
    unsafe_intake = NormalizedIntake.model_construct(
        project_id="generic-example",
        canonical_domain="generic.example",
        sources=(unsafe_source,),
    )

    with pytest.raises(IntakeValidationError, match="normalized intake"):
        consume_intake(inbox, unsafe_intake, intake_root=intake_root)

    assert not inbox.exists()
    assert outside.read_bytes() == b"outside"


def test_missing_source_file_deletes_owned_directory(tmp_path: Path) -> None:
    inbox = create_owned_intake_dir(tmp_path / "intake")
    intake = normalized_intake((source_declaration("missing.csv", b"example"),))

    with pytest.raises(IntakeValidationError, match="missing"):
        consume_intake(inbox, intake, intake_root=tmp_path / "intake")

    assert not inbox.exists()


def test_checksum_mismatch_deletes_owned_directory(tmp_path: Path) -> None:
    inbox, intake = create_source(tmp_path)
    declared_size = intake.sources[0].byte_count
    (inbox / "gsc.csv").write_bytes(b"x" * declared_size)

    with pytest.raises(IntakeValidationError, match="checksum"):
        consume_intake(inbox, intake, intake_root=tmp_path / "intake")

    assert not inbox.exists()


def test_size_mismatch_deletes_owned_directory(tmp_path: Path) -> None:
    content = b"example"
    inbox = create_owned_intake_dir(tmp_path / "intake")
    (inbox / "gsc.csv").write_bytes(content)
    declaration = source_declaration("gsc.csv", content, byte_count=len(content) + 1)

    with pytest.raises(IntakeValidationError, match="byte_count"):
        consume_intake(
            inbox,
            normalized_intake((declaration,)),
            intake_root=tmp_path / "intake",
        )

    assert not inbox.exists()


def test_source_must_be_a_regular_file(tmp_path: Path) -> None:
    inbox = create_owned_intake_dir(tmp_path / "intake")
    (inbox / "report.csv").mkdir()
    intake = normalized_intake((source_declaration("report.csv", b"example"),))

    with pytest.raises(IntakeValidationError, match="regular file"):
        consume_intake(inbox, intake, intake_root=tmp_path / "intake")

    assert not inbox.exists()


def test_source_symlink_is_rejected_without_deleting_target(tmp_path: Path) -> None:
    target = tmp_path / "outside.csv"
    target.write_bytes(b"example")
    inbox = create_owned_intake_dir(tmp_path / "intake")
    (inbox / "report.csv").symlink_to(target)
    intake = normalized_intake((source_declaration("report.csv", b"example"),))

    with pytest.raises(IntakeValidationError, match="symlink"):
        consume_intake(inbox, intake, intake_root=tmp_path / "intake")

    assert not inbox.exists()
    assert target.read_bytes() == b"example"


def test_hardlinked_declared_source_is_rejected_and_not_claimed_deleted(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.csv"
    content = b"example"
    outside.write_bytes(content)
    inbox = create_owned_intake_dir(tmp_path / "intake")
    os.link(outside, inbox / "report.csv")
    intake = normalized_intake((source_declaration("report.csv", content),))
    processor_called = False

    def processor(_value: NormalizedIntake) -> None:
        nonlocal processor_called
        processor_called = True

    with pytest.raises(IntakeCleanupError, match="link"):
        consume_intake(
            inbox,
            intake,
            intake_root=tmp_path / "intake",
            now=NOW,
            processor=processor,
        )

    assert not processor_called
    assert outside.read_bytes() == content
    quarantines = tuple((tmp_path / "intake").glob(".quarantine-*"))
    assert len(quarantines) == 1
    assert (quarantines[0] / "owned" / "report.csv").exists()


def test_source_path_cannot_escape_through_symlinked_parent(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "report.csv").write_bytes(b"example")
    inbox = create_owned_intake_dir(tmp_path / "intake")
    (inbox / "exports").symlink_to(outside, target_is_directory=True)
    intake = normalized_intake((source_declaration("exports/report.csv", b"example"),))

    with pytest.raises(IntakeValidationError, match="symlink"):
        consume_intake(inbox, intake, intake_root=tmp_path / "intake")

    assert not inbox.exists()
    assert (outside / "report.csv").exists()


def test_parent_symlink_swap_during_descriptor_traversal_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "report.csv").write_bytes(b"outside")
    inbox = create_owned_intake_dir(tmp_path / "intake")
    exports = inbox / "exports"
    exports.mkdir()
    content = b"inside"
    (exports / "report.csv").write_bytes(content)
    intake = normalized_intake((source_declaration("exports/report.csv", content),))
    original_open = data_intake._open_directory_component
    swapped = False

    def swap_then_open(parent_fd: int, component: str) -> int:
        nonlocal swapped
        if component == "exports" and not swapped:
            swapped = True
            exports.rename(inbox / "original-exports")
            exports.symlink_to(outside, target_is_directory=True)
        return original_open(parent_fd, component)

    monkeypatch.setattr(data_intake, "_open_directory_component", swap_then_open)

    with pytest.raises(IntakeValidationError, match="symlink"):
        consume_intake(inbox, intake, intake_root=tmp_path / "intake")

    assert not inbox.exists()
    assert (outside / "report.csv").read_bytes() == b"outside"


def test_arbitrary_directory_is_not_deleted(tmp_path: Path) -> None:
    intake_root = tmp_path / "intake"
    arbitrary = intake_root / "arbitrary"
    arbitrary.mkdir(parents=True)
    keep = arbitrary / "keep.txt"
    keep.write_text("keep")
    intake = normalized_intake((source_declaration("missing.csv", b"example"),))

    with pytest.raises(IntakeOwnershipError):
        consume_intake(arbitrary, intake, intake_root=intake_root)

    assert keep.read_text() == "keep"


def test_intake_root_itself_is_never_deleted(tmp_path: Path) -> None:
    intake_root = tmp_path / "intake"
    intake_root.mkdir()
    keep = intake_root / "keep.txt"
    keep.write_text("keep")
    intake = normalized_intake((source_declaration("missing.csv", b"example"),))

    with pytest.raises(IntakeOwnershipError):
        consume_intake(intake_root, intake, intake_root=intake_root)

    assert keep.read_text() == "keep"


def test_clients_sibling_is_never_deleted_on_processing_failure(tmp_path: Path) -> None:
    clients = tmp_path / "clients"
    clients.mkdir()
    keep = clients / "generic.example.json"
    keep.write_text("fabricated")
    inbox = create_owned_intake_dir(tmp_path / "intake")
    (inbox / "gsc.csv").write_bytes(b"invalid")

    with pytest.raises(IntakeValidationError):
        consume_intake(inbox, invalid_normalized_intake(), intake_root=tmp_path / "intake")

    assert keep.read_text() == "fabricated"
    assert not inbox.exists()


def test_deletion_preserves_sibling_and_unrelated_files(tmp_path: Path) -> None:
    intake_root = tmp_path / "intake"
    inbox, intake = create_source(tmp_path)
    sibling = intake_root / "manual-upload"
    sibling.mkdir()
    sibling_file = sibling / "keep.txt"
    sibling_file.write_text("keep")
    root_file = intake_root / "keep.json"
    root_file.write_text("keep")

    consume_intake(inbox, intake, intake_root=intake_root)

    assert sibling_file.read_text() == "keep"
    assert root_file.read_text() == "keep"


def test_copied_marker_does_not_authorize_different_directory(tmp_path: Path) -> None:
    intake_root = tmp_path / "intake"
    owned = create_owned_intake_dir(intake_root)
    marker = next(owned.iterdir())
    forged = intake_root / "forged"
    forged.mkdir()
    (forged / marker.name).write_bytes(marker.read_bytes())
    keep = forged / "keep.txt"
    keep.write_text("keep")
    intake = normalized_intake((source_declaration("missing.csv", b"example"),))

    with pytest.raises(IntakeOwnershipError, match="capability"):
        consume_intake(forged, intake, intake_root=intake_root)

    assert keep.read_text() == "keep"


def test_matching_hand_forged_marker_without_creator_capability_is_never_deleted(
    tmp_path: Path,
) -> None:
    intake_root = tmp_path / "intake"
    intake_root.mkdir()
    nonce = "f" * 32
    forged = intake_root / f"run-20260901T093000Z-{nonce}"
    forged.mkdir()
    write_forged_marker(forged, nonce=nonce)
    keep = forged / "keep.txt"
    keep.write_text("keep")
    intake = normalized_intake((source_declaration("missing.csv", b"example"),))

    with pytest.raises(IntakeOwnershipError, match="capability"):
        consume_intake(forged, intake, intake_root=intake_root)

    assert keep.read_text() == "keep"


def test_damaged_marker_fails_consume_but_retires_and_deletes_issued_directory(
    tmp_path: Path,
) -> None:
    intake_root = tmp_path / "intake"
    inbox = create_owned_intake_dir(intake_root)
    marker = next(inbox.iterdir())
    marker.write_text('{"run_id":"wrong","nonce":"wrong"}')
    keep = inbox / "keep.txt"
    keep.write_text("keep")
    intake = normalized_intake((source_declaration("missing.csv", b"example"),))

    with pytest.raises(IntakeOwnershipError, match="marker"):
        consume_intake(inbox, intake, intake_root=intake_root)

    assert not inbox.exists()
    with pytest.raises(IntakeOwnershipError):
        discard_owned_intake_dir(inbox, intake_root=intake_root)


def test_replayed_marker_does_not_authorize_recreated_directory(tmp_path: Path) -> None:
    intake_root = tmp_path / "intake"
    inbox = create_owned_intake_dir(intake_root)
    marker = next(inbox.iterdir())
    marker_name = marker.name
    marker_bytes = marker.read_bytes()
    marker.unlink()
    inbox.rmdir()
    inbox.mkdir()
    (inbox / marker_name).write_bytes(marker_bytes)
    keep = inbox / "keep.txt"
    keep.write_text("keep")
    intake = normalized_intake((source_declaration("missing.csv", b"example"),))

    with pytest.raises(IntakeOwnershipError, match="capability"):
        consume_intake(inbox, intake, intake_root=intake_root)

    assert keep.read_text() == "keep"


@pytest.mark.parametrize("consumer", ["legacy", "payload"])
def test_recreated_directory_rejected_even_when_path_stats_replay_inode_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, consumer: str
) -> None:
    intake_root = tmp_path / "intake"
    inbox = create_owned_intake_dir(intake_root)
    marker = next(inbox.iterdir())
    old_directory, old_marker = inbox.lstat(), marker.lstat()
    content = marker.read_bytes()
    marker.unlink()
    inbox.rmdir()
    inbox.mkdir()
    marker.write_bytes(content)
    keep = inbox / "keep.txt"
    keep.write_text("keep")
    real_lstat = Path.lstat

    def replayed_lstat(path: Path):
        if path == inbox:
            return old_directory
        if path == marker:
            return old_marker
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", replayed_lstat)
    intake = normalized_intake((source_declaration("missing.csv", b"example"),))
    with pytest.raises(IntakeOwnershipError, match="capability"):
        if consumer == "legacy":
            consume_intake(inbox, intake, intake_root=intake_root)
        else:
            data_intake.consume_owned_payload(
                inbox, intake_root=intake_root, processor=lambda fd: None
            )
    assert keep.read_text() == "keep"


@pytest.mark.parametrize("operation", ["consume", "discard", "invalid"])
def test_issued_inode_anchors_close_after_capability_is_retired(tmp_path: Path, operation: str):
    import ai_search_audit.data_intake as intake_module

    inbox, normalized = create_source(tmp_path)
    issued = next(
        value for value in intake_module._OWNED_CAPABILITIES.values() if value.path == inbox
    )
    anchors = issued.identity_anchors
    assert len(anchors) == 2
    if operation == "discard":
        discard_owned_intake_dir(inbox, intake_root=tmp_path / "intake")
    elif operation == "invalid":
        with pytest.raises(IntakeValidationError):
            consume_intake(inbox, {}, intake_root=tmp_path / "intake")
    else:
        consume_intake(inbox, normalized, intake_root=tmp_path / "intake")
    for descriptor in anchors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize("operation", ["consume", "discard"])
@pytest.mark.parametrize("damage", ["missing-marker", "missing-directory"])
def test_early_ownership_failure_retires_anchors_without_deleting_unverified_files(
    tmp_path: Path, operation: str, damage: str
):
    import ai_search_audit.data_intake as intake_module

    intake_root = tmp_path / "intake"
    inbox = create_owned_intake_dir(intake_root)
    issued = next(
        value for value in intake_module._OWNED_CAPABILITIES.values() if value.path == inbox
    )
    marker = next(inbox.iterdir())
    marker.unlink()
    keep = inbox / "keep.txt"
    if damage == "missing-directory":
        inbox.rmdir()
    else:
        keep.write_text("keep")
    with pytest.raises(IntakeOwnershipError):
        if operation == "consume":
            consume_intake(inbox, {}, intake_root=intake_root)
        else:
            discard_owned_intake_dir(inbox, intake_root=intake_root)
    assert issued not in intake_module._OWNED_CAPABILITIES.values()
    for descriptor in issued.identity_anchors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    if damage == "missing-marker":
        assert keep.read_text() == "keep"


@pytest.mark.parametrize("consumer", ["legacy", "payload"])
def test_consumed_capability_cannot_be_replayed_with_fresh_matching_marker(
    tmp_path: Path,
    consumer: str,
) -> None:
    intake_root = tmp_path / "intake"
    inbox, intake = create_source(tmp_path)
    run_id = inbox.name
    nonce = run_id.rsplit("-", 1)[-1]

    if consumer == "legacy":
        consume_intake(inbox, intake, intake_root=intake_root)
    else:
        data_intake.consume_owned_payload(inbox, intake_root=intake_root, processor=lambda fd: None)

    inbox.mkdir()
    write_forged_marker(inbox, nonce=nonce)
    keep = inbox / "keep.txt"
    keep.write_text("keep")

    with pytest.raises(IntakeOwnershipError, match="capability"):
        if consumer == "legacy":
            consume_intake(
                inbox,
                normalized_intake((source_declaration("missing.csv", b"example"),)),
                intake_root=intake_root,
            )
        else:
            data_intake.consume_owned_payload(
                inbox, intake_root=intake_root, processor=lambda fd: None
            )

    assert keep.read_text() == "keep"


def test_symlinked_marker_does_not_authorize_deletion(tmp_path: Path) -> None:
    intake_root = tmp_path / "intake"
    inbox = create_owned_intake_dir(intake_root)
    marker = next(inbox.iterdir())
    marker_copy = tmp_path / "marker-copy.json"
    marker_copy.write_bytes(marker.read_bytes())
    marker.unlink()
    marker.symlink_to(marker_copy)
    keep = inbox / "keep.txt"
    keep.write_text("keep")
    intake = normalized_intake((source_declaration("missing.csv", b"example"),))

    with pytest.raises(IntakeOwnershipError, match="marker"):
        consume_intake(inbox, intake, intake_root=intake_root)

    assert keep.read_text() == "keep"


def test_symlinked_intake_root_is_rejected(tmp_path: Path) -> None:
    real_root = tmp_path / "real-intake"
    real_root.mkdir()
    linked_root = tmp_path / "linked-intake"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(IntakeOwnershipError, match="symlink"):
        create_owned_intake_dir(linked_root)


def test_symlinked_owned_directory_is_rejected_without_deleting_target(tmp_path: Path) -> None:
    intake_root = tmp_path / "intake"
    owned = create_owned_intake_dir(intake_root)
    linked = intake_root / "linked"
    linked.symlink_to(owned, target_is_directory=True)
    intake = normalized_intake((source_declaration("missing.csv", b"example"),))

    with pytest.raises(IntakeOwnershipError, match="symlink"):
        consume_intake(linked, intake, intake_root=intake_root)

    assert owned.exists()


def test_cleanup_failure_fails_closed_without_trusted_deleted_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inbox, intake = create_source(tmp_path)

    def fail_delete(
        _root_fd: int,
        _quarantine_name: str,
        _quarantine_identity: tuple[int, int],
    ) -> None:
        raise OSError("deletion blocked")

    monkeypatch.setattr(data_intake, "_delete_quarantined_directory", fail_delete)

    with pytest.raises(IntakeCleanupError, match="could not delete") as error:
        consume_intake(inbox, intake, intake_root=tmp_path / "intake", now=NOW)

    assert "deleted_at" not in str(error.value)
    assert not inbox.exists()
    quarantines = tuple((tmp_path / "intake").glob(".quarantine-*"))
    assert len(quarantines) == 1
    assert (quarantines[0] / "owned" / "gsc.csv").exists()


def test_quarantine_open_failure_closes_setup_and_leaves_no_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inbox, intake = create_source(tmp_path)
    original_open = data_intake.os.open

    def fail_quarantine_open(path: object, *args: Any, **kwargs: Any) -> int:
        if isinstance(path, str) and path.startswith(".quarantine-"):
            raise OSError("injected quarantine open failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(data_intake.os, "open", fail_quarantine_open)

    with pytest.raises(IntakeCleanupError, match="could not delete"):
        consume_intake(inbox, intake, intake_root=tmp_path / "intake")

    assert inbox.exists()
    assert tuple((tmp_path / "intake").glob(".quarantine-*")) == ()


def test_owned_descriptor_fstat_failure_closes_every_opened_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intake_root = tmp_path / "intake"
    inbox = create_owned_intake_dir(intake_root, now=NOW)
    owned = data_intake._verify_owned_directory(inbox, intake_root)
    original_open = os.open
    original_close = os.close
    original_fstat = os.fstat
    opened: set[int] = set()
    closed: set[int] = set()
    fstat_calls = 0

    def tracking_open(
        path: str | bytes | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        opened.add(descriptor)
        return descriptor

    def tracking_close(descriptor: int) -> None:
        closed.add(descriptor)
        original_close(descriptor)

    def failing_second_fstat(descriptor: int) -> os.stat_result:
        nonlocal fstat_calls
        fstat_calls += 1
        if fstat_calls == 2:
            raise OSError("injected owned-directory fstat failure")
        return original_fstat(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(data_intake.os, "open", tracking_open)
        patch.setattr(data_intake.os, "close", tracking_close)
        patch.setattr(data_intake.os, "fstat", failing_second_fstat)
        with pytest.raises(OSError, match="injected"):
            data_intake._open_verified_owned_descriptors(owned)

    assert opened
    assert opened == closed
    discard_owned_intake_dir(inbox, intake_root=intake_root)


def test_regular_entry_swap_before_unlink_fails_without_false_deleted_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inbox, intake = create_source(tmp_path)
    original_unlink = data_intake._unlink_opened_regular_entry
    swapped = False

    def swap_then_unlink(
        parent_fd: int,
        entry_name: str,
        entry_fd: int,
        inspected: os.stat_result,
    ) -> None:
        nonlocal swapped
        if entry_name == "gsc.csv" and not swapped:
            swapped = True
            os.rename(
                entry_name,
                "gsc-recovery.csv",
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            replacement_fd = os.open(
                entry_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=parent_fd,
            )
            try:
                os.write(replacement_fd, b"replacement")
            finally:
                os.close(replacement_fd)
        original_unlink(parent_fd, entry_name, entry_fd, inspected)

    monkeypatch.setattr(
        data_intake,
        "_unlink_opened_regular_entry",
        swap_then_unlink,
    )

    with pytest.raises(IntakeCleanupError, match="changed"):
        consume_intake(inbox, intake, intake_root=tmp_path / "intake", now=NOW)

    quarantines = tuple((tmp_path / "intake").glob(".quarantine-*"))
    assert len(quarantines) == 1
    payload = quarantines[0] / "owned"
    assert (payload / "gsc-recovery.csv").exists()
    assert (payload / "gsc.csv").read_bytes() == b"replacement"


def test_held_directory_survivor_after_rmdir_race_prevents_false_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intake_root = tmp_path / "intake"
    content = b"example"
    inbox = create_owned_intake_dir(intake_root, now=NOW)
    exports = inbox / "exports"
    exports.mkdir()
    original_inode = exports.stat().st_ino
    (exports / "report.csv").write_bytes(content)
    intake = normalized_intake((source_declaration("exports/report.csv", content),))
    run_id = inbox.name
    nonce = run_id.rsplit("-", 1)[-1]
    original_rmdir = data_intake._rmdir_relative_directory
    swapped = False

    def swap_then_rmdir(parent_fd: int, entry_name: str) -> None:
        nonlocal swapped
        if entry_name == "exports" and not swapped:
            swapped = True
            os.rename(
                entry_name,
                "exports-survivor",
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.mkdir(entry_name, mode=0o700, dir_fd=parent_fd)
        original_rmdir(parent_fd, entry_name)

    monkeypatch.setattr(
        data_intake,
        "_rmdir_relative_directory",
        swap_then_rmdir,
    )

    with pytest.raises(IntakeCleanupError, match="does not report removal"):
        consume_intake(inbox, intake, intake_root=intake_root, now=NOW)

    quarantines = tuple(intake_root.glob(".quarantine-*"))
    assert len(quarantines) == 1
    survivor = quarantines[0] / "owned" / "exports-survivor"
    assert survivor.stat().st_ino == original_inode

    inbox.mkdir()
    write_forged_marker(inbox, nonce=nonce)
    with pytest.raises(IntakeOwnershipError, match="capability"):
        consume_intake(
            inbox,
            normalized_intake((source_declaration("missing.csv", b"example"),)),
            intake_root=intake_root,
        )


def test_directory_swap_at_quarantine_is_not_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intake_root = tmp_path / "intake"
    inbox, intake = create_source(tmp_path)
    original_owned = intake_root / "original-owned"
    original_rename = data_intake._rename_to_quarantine
    swapped = False

    def swap_then_quarantine(root_fd: int, source_name: str, quarantine_name: str) -> None:
        nonlocal swapped
        if not swapped:
            swapped = True
            inbox.rename(original_owned)
            inbox.mkdir()
            (inbox / "replacement.txt").write_text("replacement")
        original_rename(root_fd, source_name, quarantine_name)

    monkeypatch.setattr(data_intake, "_rename_to_quarantine", swap_then_quarantine)

    with pytest.raises(IntakeCleanupError, match="identity changed"):
        consume_intake(inbox, intake, intake_root=intake_root)

    assert (inbox / "replacement.txt").read_text() == "replacement"
    assert (original_owned / "gsc.csv").exists()


def test_root_swap_to_symlink_during_owned_mkdir_creates_nothing_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intake_root = tmp_path / "intake"
    intake_root.mkdir()
    original_root = tmp_path / "original-intake"
    outside = tmp_path / "outside"
    outside.mkdir()
    original_mkdir = data_intake._mkdir_owned_relative
    swapped = False

    def swap_then_mkdir(root_fd: int, run_name: str) -> None:
        nonlocal swapped
        if not swapped:
            swapped = True
            intake_root.rename(original_root)
            intake_root.symlink_to(outside, target_is_directory=True)
        original_mkdir(root_fd, run_name)

    monkeypatch.setattr(data_intake, "_mkdir_owned_relative", swap_then_mkdir)

    with pytest.raises(IntakeOwnershipError, match="identity changed"):
        create_owned_intake_dir(intake_root, now=NOW)

    assert tuple(outside.iterdir()) == ()
    assert tuple(original_root.iterdir()) == ()


@pytest.mark.parametrize(
    ("model_factory", "extra_field"),
    (
        (lambda: source_declaration("report.csv", b"example"), "token"),
        (
            lambda: OwnerFactInput(
                fact_id="fact-1",
                field="brand_name",
                value="Generic Example",
                source_id="source-1",
            ),
            "user_id",
        ),
        (
            lambda: VisibilityMetricSeriesInput(
                metric_id="clicks",
                source_id="source-1",
                metric="clicks",
                unit="count",
                state=DataState.AVAILABLE,
                coverage=1,
                confidence=0.8,
                points=(VisibilityMetricPoint(period_start=date(2026, 8, 1), value=1),),
            ),
            "raw_rows",
        ),
        (
            lambda: NonSensitiveExample(
                example_id="example-1",
                source_id="source-1",
                description="Safe example",
                citation="generic.example",
                non_sensitive=True,
            ),
            "payload",
        ),
    ),
)
def test_nested_models_forbid_sensitive_or_arbitrary_extra_fields(
    model_factory: Any, extra_field: str
) -> None:
    model = model_factory()
    values = model.model_dump()
    values[extra_field] = "must-not-be-retained"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        type(model).model_validate(values)


def test_normalized_intake_forbids_trusted_timestamps_and_credentials() -> None:
    source = source_declaration("report.csv", b"example")
    values = normalized_intake((source,)).model_dump()
    values.update(
        {
            "processed_at": NOW,
            "deleted_at": NOW,
            "credentials": "must-not-be-accepted",
        }
    )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        NormalizedIntake.model_validate(values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"project_id": "../unsafe"}, "safe identifier"),
        ({"canonical_domain": "https://user:password@generic.example"}, "credentials"),
        (
            {"processed_at": datetime(2026, 9, 1, 9, 30), "deleted_at": NOW},
            "timezone",
        ),
        (
            {
                "processed_at": datetime(2026, 9, 1, 10, tzinfo=UTC),
                "deleted_at": datetime(2026, 9, 1, 9, tzinfo=UTC),
            },
            "deleted_at",
        ),
    ),
)
def test_processed_intake_rejects_untrusted_identity_and_timestamps(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        processed_intake(**overrides)


@pytest.mark.parametrize(
    "mutation",
    (
        "duplicate-source-id",
        "duplicate-filename",
        "duplicate-fact-id",
        "duplicate-metric-id",
        "duplicate-example-id",
        "unknown-reference",
        "source-timestamp-mismatch",
    ),
)
def test_processed_intake_direct_deserialization_reuses_cross_record_invariants(
    mutation: str,
) -> None:
    first_source = source_provenance()
    second_values = first_source.model_dump()
    second_values.update(
        {
            "source_id": "source-2",
            "filename": "second.csv",
        }
    )
    second_source = SourceArtifactProvenance.model_validate(second_values)
    fact = OwnerFactInput(
        fact_id="fact-1",
        field="brand_name",
        value="Generic Example",
        source_id="source-1",
    )
    metric = VisibilityMetricSeriesInput(
        metric_id="metric-1",
        source_id="source-1",
        metric="clicks",
        unit="count",
        state=DataState.AVAILABLE,
        coverage=1,
        confidence=0.8,
        points=(VisibilityMetricPoint(period_start=date(2026, 8, 1), value=0),),
    )
    example = NonSensitiveExample(
        example_id="example-1",
        source_id="source-1",
        description="Safe example",
        citation="generic.example",
        non_sensitive=True,
    )
    values = processed_intake(
        sources=(first_source, second_source),
        owner_facts=(fact,),
        metric_series=(metric,),
        cited_examples=(example,),
    ).model_dump()

    if mutation == "duplicate-source-id":
        values["sources"][1]["source_id"] = "source-1"
    elif mutation == "duplicate-filename":
        values["sources"][1]["filename"] = "REPORT.CSV"
    elif mutation == "duplicate-fact-id":
        values["owner_facts"] = (*values["owner_facts"], values["owner_facts"][0].copy())
    elif mutation == "duplicate-metric-id":
        values["metric_series"] = (
            *values["metric_series"],
            values["metric_series"][0].copy(),
        )
    elif mutation == "duplicate-example-id":
        values["cited_examples"] = (
            *values["cited_examples"],
            values["cited_examples"][0].copy(),
        )
    elif mutation == "unknown-reference":
        values["owner_facts"][0]["source_id"] = "missing-source"
    else:
        values["sources"][0]["processed_at"] = datetime(2026, 9, 1, 9, 29, tzinfo=UTC)

    with pytest.raises(ValidationError):
        ProcessedIntake.model_validate(values)


@pytest.mark.parametrize(
    ("processed_at", "deleted_at", "message"),
    (
        (datetime(2026, 9, 1, 9, 30), NOW, "timezone"),
        (
            datetime(2026, 9, 1, 10, tzinfo=UTC),
            datetime(2026, 9, 1, 9, tzinfo=UTC),
            "deleted_at",
        ),
    ),
)
def test_source_provenance_rejects_naive_or_reversed_trusted_timestamps(
    processed_at: datetime, deleted_at: datetime, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        source_provenance(processed_at=processed_at, deleted_at=deleted_at)


def test_non_sensitive_example_requires_explicit_true_marker() -> None:
    with pytest.raises(ValidationError):
        NonSensitiveExample(
            example_id="example-1",
            source_id="source-1",
            description="Not explicitly safe",
            citation="generic.example",
            non_sensitive=False,  # type: ignore[arg-type]
        )


def test_metric_data_states_remain_distinct() -> None:
    states = tuple(
        VisibilityMetricSeriesInput(
            metric_id=f"metric-{state.value.lower()}",
            source_id="source-1",
            metric="visibility",
            unit="percent",
            state=state,
            coverage=0,
            confidence=0.5,
        ).state
        for state in (DataState.UNAVAILABLE, DataState.UNKNOWN, DataState.FAILED)
    )

    assert states == (DataState.UNAVAILABLE, DataState.UNKNOWN, DataState.FAILED)
    assert len(set(states)) == 3


@pytest.mark.parametrize("value", (True, "0"))
def test_metric_point_rejects_non_strict_numeric_values(value: object) -> None:
    with pytest.raises(ValidationError):
        VisibilityMetricPoint(period_start=date(2026, 8, 1), value=value)


@pytest.mark.parametrize("value", (0, 0.0))
def test_metric_point_accepts_real_numeric_zero(value: int | float) -> None:
    point = VisibilityMetricPoint(period_start=date(2026, 8, 1), value=value)

    assert point.value == 0


@pytest.mark.parametrize("state", (DataState.AVAILABLE, DataState.PARTIAL))
def test_available_metric_states_require_at_least_one_point(state: DataState) -> None:
    with pytest.raises(ValidationError, match="at least one point"):
        VisibilityMetricSeriesInput(
            metric_id="metric-1",
            source_id="source-1",
            metric="clicks",
            unit="count",
            state=state,
            coverage=1,
            confidence=0.8,
        )


@pytest.mark.parametrize(
    "state",
    (
        DataState.UNAVAILABLE,
        DataState.UNKNOWN,
        DataState.FAILED,
        DataState.UNSUPPORTED,
    ),
)
def test_non_numeric_metric_states_reject_points(state: DataState) -> None:
    with pytest.raises(ValidationError, match="must not contain points"):
        VisibilityMetricSeriesInput(
            metric_id="metric-1",
            source_id="source-1",
            metric="clicks",
            unit="count",
            state=state,
            coverage=0,
            confidence=0.5,
            points=(VisibilityMetricPoint(period_start=date(2026, 8, 1), value=0),),
        )


@pytest.mark.parametrize(
    "missing_capability",
    (
        "O_NOFOLLOW",
        "O_DIRECTORY",
        "dir_fd",
        "follow_symlinks",
        "fd",
        "fstat",
    ),
)
def test_creation_fails_before_mutation_without_required_posix_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_capability: str,
) -> None:
    intake_root = tmp_path / "intake"
    if missing_capability in {"O_NOFOLLOW", "O_DIRECTORY"}:
        monkeypatch.setattr(data_intake.os, missing_capability, 0)
    elif missing_capability == "dir_fd":
        monkeypatch.setattr(data_intake.os, "supports_dir_fd", set())
    elif missing_capability == "follow_symlinks":
        monkeypatch.setattr(data_intake.os, "supports_follow_symlinks", set())
    elif missing_capability == "fd":
        monkeypatch.setattr(data_intake.os, "supports_fd", set())
    else:
        monkeypatch.setattr(data_intake.os, "fstat", None)

    with pytest.raises(IntakeOwnershipError, match="POSIX"):
        create_owned_intake_dir(intake_root, now=NOW)

    assert not intake_root.exists()


def test_consume_fails_before_claim_when_platform_primitives_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inbox, intake = create_source(tmp_path)

    with monkeypatch.context() as patch:
        patch.setattr(data_intake.os, "supports_fd", set())
        with pytest.raises(IntakeOwnershipError, match="POSIX"):
            consume_intake(inbox, intake, intake_root=tmp_path / "intake")

    assert inbox.exists()
    discard_owned_intake_dir(inbox, intake_root=tmp_path / "intake")


def test_metric_series_retains_bounded_normalized_labels_without_aggregation() -> None:
    series = VisibilityMetricSeriesInput(
        metric_id="clicks",
        source_id="source-1",
        metric="clicks",
        unit="count",
        state=DataState.AVAILABLE,
        coverage=1,
        confidence=0.8,
        dimensions=("country=PL", "device=mobile"),
        filters=("search_type=web",),
        segment_label="organic search",
        points=(VisibilityMetricPoint(period_start=date(2026, 8, 1), value=0),),
    )

    assert series.dimensions == ("country=PL", "device=mobile")
    assert series.filters == ("search_type=web",)
    assert series.segment_label == "organic search"
    assert series.points[0].value == 0
    assert series.coverage == 1
    assert series.confidence == 0.8


def test_metric_series_rejects_available_values_without_coverage() -> None:
    with pytest.raises(ValidationError, match="coverage"):
        VisibilityMetricSeriesInput(
            metric_id="clicks",
            source_id="source-1",
            metric="gsc.clicks",
            unit="clicks",
            state=DataState.AVAILABLE,
            coverage=0,
            confidence=0.8,
            points=(VisibilityMetricPoint(period_start=date(2026, 8, 1), value=1),),
        )

    with pytest.raises(ValidationError, match="confidence"):
        VisibilityMetricSeriesInput(
            metric_id="clicks",
            source_id="source-1",
            metric="gsc.clicks",
            unit="clicks",
            state=DataState.AVAILABLE,
            coverage=1,
            confidence=0,
            points=(VisibilityMetricPoint(period_start=date(2026, 8, 1), value=1),),
        )


@pytest.mark.parametrize("missing", ("coverage", "confidence"))
def test_metric_series_requires_explicit_quality_metadata(missing: str) -> None:
    payload: dict[str, object] = {
        "metric_id": "clicks",
        "source_id": "source-1",
        "metric": "gsc.clicks",
        "unit": "clicks",
        "state": DataState.AVAILABLE,
        "coverage": 1,
        "confidence": 0.8,
        "points": [{"period_start": "2026-08-01", "value": 1}],
    }
    del payload[missing]

    with pytest.raises(ValidationError, match=missing):
        VisibilityMetricSeriesInput.model_validate(payload)


@pytest.mark.parametrize("fixture_name", ("generic.example", "ecommerce.example", "local.example"))
def test_normalized_intake_fixtures_are_fabricated_and_schema_valid(
    fixture_name: str,
) -> None:
    fixture_path = FIXTURE_ROOT / fixture_name / "normalized-intake.json"
    payload = json.loads(fixture_path.read_text())

    intake = NormalizedIntake.model_validate(payload)

    assert intake.canonical_domain.endswith(".example")
    assert fixture_name in fixture_path.parts
    forbidden_absolute_prefix = "/" + "Users" + "/"
    assert forbidden_absolute_prefix not in fixture_path.read_text()
