"""Client language is a guarded, source-bound projection, never provider prose."""

from datetime import date, timedelta
from importlib import import_module, util

import pytest

from ai_search_audit.diagnostic_performance import DiagnosticRunV2
from ai_search_audit.knowledge import default_registry_root, load_registry
from ai_search_audit.measurement_report import MeasurementReport, render_measurement_fragment
from tests.test_diagnostic_versions import measured_run, performance_run, transient_run
from tests.test_diagnostic_workflow import no_network, project

__all__ = ["no_network", "project"]
TODAY = date(2026, 9, 4)


def report_for(project, *, score=73.0, lcp=4651.0):
    payload = measured_run(project).model_dump(mode="json")
    payload["collections"][0]["lab"]["performance_score"] = score
    payload["collections"][0]["lab"]["metrics"][0]["value"] = lcp
    if score is None or lcp is None:
        payload["collections"][0]["attempts"][0].update(state="PARTIAL", reason="missing_metrics")
    return MeasurementReport(
        reference={"source_version": "public-v1", "run_id": "run-1"},
        manifest_sha256="a" * 64,
        audited_page_count=10,
        run=DiagnosticRunV2.model_validate(payload),
    )


def client_module():
    assert util.find_spec("ai_search_audit.measurement_client"), "guarded client projection missing"
    return import_module("ai_search_audit.measurement_client")


def build(report, *, registry=None, as_of=TODAY):
    return client_module().build_measurement_client(
        report, registry=registry or load_registry(default_registry_root()), as_of=as_of
    )


def test_client_has_readable_measured_score_and_lcp_without_invented_cause(project):
    fragment = render_measurement_fragment(report_for(project), as_of=TODAY)
    assert "73/100" in fragment and "4.65 s" in fragment
    assert "needs improvement" in fragment.lower()
    assert "investigate" in fragment.lower()
    assert "lab" in fragment.lower() and "not" in fragment.lower()
    assert "U1" not in fragment and "diagnostics/" not in fragment
    assert "large image" not in fragment.lower() and "server is" not in fragment.lower()
    assert "fails core web vitals" not in fragment.lower()


@pytest.mark.parametrize(
    "score,lcp,score_band,lcp_band",
    [
        (49, 4001, "poor", "poor"),
        (50, 4000, "needs improvement", "needs improvement"),
        (89, 2501, "needs improvement", "needs improvement"),
        (90, 2500, "good", "good"),
        (99, 4651, "good", "poor"),
        (0, 0, "poor", "good"),
    ],
)
def test_thresholds_and_disagreement_are_independent(project, score, lcp, score_band, lcp_band):
    client = build(report_for(project, score=score, lcp=lcp))
    row = client.lab_rows[0]
    assert score_band in row[3].lower()
    assert lcp_band in row[4].lower()
    assert client.report.run.collections[0].lab.performance_score == score
    assert client.report.run.collections[0].lab.metrics[0].value == lcp


@pytest.mark.parametrize("score,lcp", [(None, 0), (0, None), (0.00000001, 0.00000001)])
def test_null_zero_and_small_positive_remain_distinct(project, score, lcp):
    client = build(report_for(project, score=score, lcp=lcp))
    row = client.lab_rows[0]
    if score is None:
        assert "unavailable" in row[3].lower()
    elif score == 0:
        assert row[3].startswith("0/100")
    else:
        assert not row[3].startswith("0/100") and not row[4].startswith("0 s")
    if lcp is None:
        assert "unavailable" in row[4].lower()


@pytest.mark.parametrize("boundary", ["build", "guard"])
@pytest.mark.parametrize("metric", ["score", "lab_lcp", "field_lcp"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_unchecked_nonfinite_optional_numbers_cannot_become_unknown(
    project, boundary, metric, value
):
    import warnings

    from tests.test_diagnostic_versions import normalized_crux_run

    report = report_for(
        project,
        score=None if metric == "score" else 73.0,
        lcp=None if metric == "lab_lcp" else 4651.0,
    )
    if metric == "field_lcp":
        payload = normalized_crux_run(project).model_dump(mode="python")
        payload["collections"][1]["field"]["metrics"][0]["value"] = None
        payload["collections"][1]["attempts"][0].update(state="PARTIAL", reason="missing_metrics")
        report = report.model_copy(update={"run": DiagnosticRunV2.model_validate(payload)})
    client = build(report)
    module = client_module()
    assert module.validate_measurement_client(client, report=report, as_of=TODAY) == client
    index = 1 if metric == "field_lcp" else 0
    kind = "field" if metric == "field_lcp" else "lab"
    original = report.run.collections[index]
    measurement = getattr(original, kind)
    if metric == "score":
        changed = measurement.model_copy(update={"performance_score": value})
    else:
        first = measurement.metrics[0].model_copy(update={"value": value})
        changed = measurement.model_copy(update={"metrics": (first, *measurement.metrics[1:])})
    collections = list(report.run.collections)
    collections[index] = original.model_copy(update={kind: changed})
    tampered = report.model_copy(
        update={"run": report.run.model_copy(update={"collections": tuple(collections)})}
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            if boundary == "build":
                build(tampered)
            else:
                module.validate_measurement_client(
                    client.model_copy(update={"report": tampered}), report=report, as_of=TODAY
                )
    assert not captured


@pytest.mark.parametrize("boundary", ["build", "guard"])
def test_invalid_optional_value_never_emits_secret_containing_serializer_warnings(
    project, boundary
):
    import warnings

    report = report_for(project, score=None)
    client = build(report)
    first = report.run.collections[0]
    lab = first.lab.model_copy(update={"performance_score": {"secret": "MUST-NOT-PRINT"}})
    run = report.run.model_copy(
        update={"collections": (first.model_copy(update={"lab": lab}), *report.run.collections[1:])}
    )
    tampered = report.model_copy(update={"run": run})
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            if boundary == "build":
                build(tampered)
            else:
                client_module().validate_measurement_client(
                    client.model_copy(update={"report": tampered}), report=report, as_of=TODAY
                )
    assert not captured


def test_frozen_dto_retains_evidence_and_detaches_registry_rules(project):
    registry = load_registry(default_registry_root())
    client = build(report_for(project), registry=registry)
    assert client.report.run.collections[0].lab.evidence_id == "measurement-1"
    assert client.report.manifest_sha256 == "a" * 64
    assert client.as_of == TODAY
    assert len(client.rules) == 2
    assert all(rule.scoring_weight == 0 and rule.evidence_level == "A" for rule in client.rules)
    assert all(rule.verified_at == TODAY and rule.state.value == "CURRENT" for rule in client.rules)
    with pytest.raises(ValueError):
        client.rules[0].confidence = 0
    original = client.rules[0].statement
    next(r for r in registry.rules if r.rule_id == client.rules[0].rule_id).statement = "changed"
    assert client.rules[0].statement == original


def test_missing_rule_fails_closed_and_stale_rule_keeps_only_raw_results(project):
    report = report_for(project)
    registry = load_registry(default_registry_root())
    current = build(report, registry=registry)
    stale = build(
        report,
        registry=registry,
        as_of=max(r.expires_at for r in current.rules) + timedelta(days=1),
    )
    assert "73/100" in stale.lab_rows[0][3] and "4.65 s" in stale.lab_rows[0][4]
    assert "needs improvement" not in stale.lab_rows[0][3].lower()
    assert "poor" not in stale.lab_rows[0][4].lower()
    assert "investigate" not in " ".join(stale.lab_notes).lower()
    assert all(r.state.value == "REQUIRES_VERIFICATION" for r in stale.rules)
    registry.rules = [r for r in registry.rules if r.rule_id != current.rules[0].rule_id]
    with pytest.raises(ValueError, match="rule"):
        build(report, registry=registry)


@pytest.mark.parametrize(
    "mutation",
    ["extra", "narrative", "source", "run", "manifest", "number", "rule", "rule_scope", "as_of"],
)
def test_guard_rejects_unchecked_client_changes(project, mutation):
    report = report_for(project)
    registry = load_registry(default_registry_root())
    client = build(report, registry=registry)
    if mutation == "extra":
        changed = client.model_copy(update={"raw_response": "not allowed"})
    elif mutation == "narrative":
        changed = client.model_copy(update={"lab_notes": ("Guaranteed AI ranking improvement",)})
    elif mutation in {"source", "run", "manifest", "number"}:
        if mutation == "source":
            changed_report = report.model_copy(
                update={
                    "run": report.run.model_copy(
                        update={
                            "binding": report.run.binding.model_copy(
                                update={"source_sha256": "b" * 64}
                            )
                        }
                    )
                }
            )
        elif mutation == "run":
            changed_report = report.model_copy(
                update={"reference": report.reference.model_copy(update={"run_id": "run-2"})}
            )
        elif mutation == "manifest":
            changed_report = report.model_copy(update={"manifest_sha256": "b" * 64})
        else:
            changed_report = report_for(project, score=100)
        changed = client.model_copy(update={"report": changed_report})
    elif mutation in {"rule", "rule_scope"}:
        rule = client.rules[0].model_copy(
            update={"confidence": 1.0} if mutation == "rule" else {"statement": "Ranking guarantee"}
        )
        changed = client.model_copy(update={"rules": (rule, *client.rules[1:])})
    else:
        changed = client.model_copy(update={"as_of": TODAY + timedelta(days=1)})
    with pytest.raises(ValueError):
        client_module().validate_measurement_client(
            changed, report=report, registry=registry, as_of=TODAY
        )


def test_same_visible_results_in_another_run_cannot_reuse_exact_fragment(project):
    report = report_for(project)
    other = report.model_copy(
        update={"reference": report.reference.model_copy(update={"run_id": "run-2"})}
    )
    assert render_measurement_fragment(report) != render_measurement_fragment(other)


def test_failed_local_fallback_is_not_described_as_a_replacement_result(project):
    report = report_for(project).model_copy(update={"run": transient_run(project, version="1.1.0")})
    client = build(report)
    assert "replaces" not in " ".join(client.lab_notes)
    assert "attempt" in " ".join(client.lab_notes).lower()


@pytest.mark.parametrize("different", [False, True])
def test_crux_origin_records_deduplicate_despite_different_retrieval_times(
    project, monkeypatch, capsys, different
):
    from ai_search_audit.cli import main
    from ai_search_audit.diagnostic_performance import performance_collection_range
    from ai_search_audit.diagnostic_store import DiagnosticStore
    from ai_search_audit.performance_providers import PerformanceCollection
    from tests.test_measurement_workflow import command, harness

    harness(monkeypatch, capsys, crux_origin=True)
    assert main(command(project, {"max_pages": 2})) == 0
    report = report_for(project)
    payload = DiagnosticStore(project).load("public-v1", "run-1").run.model_dump(mode="python")
    fields = [c for c in payload["collections"] if c["field"]]
    assert len(fields) == 4
    fields[-1]["field"]["observed_at"] += timedelta(days=1)
    if different:
        fields[-1]["field"]["metrics"][0]["value"] += 1
    collections = tuple(PerformanceCollection.model_validate(c) for c in payload["collections"])
    payload["collection_range"] = performance_collection_range(collections)
    client = build(report.model_copy(update={"run": DiagnosticRunV2.model_validate(payload)}))
    assert len(client.field_rows) == (3 if different else 2)
    assert len([c for c in client.report.run.collections if c.field]) == 4
    assert all("https://studio.example" in row[0] for row in client.field_rows)
    assert "p75" in " ".join(client.field_notes)
    assert "2026-08-01" in client.field_rows[0][1]


@pytest.mark.parametrize("equivalent", ["https://STUDIO.example", "https://studio.example:443"])
def test_equivalent_validated_origin_keys_deduplicate_without_rewriting_evidence(
    project, monkeypatch, capsys, equivalent
):
    from ai_search_audit.cli import main
    from ai_search_audit.diagnostic_store import DiagnosticStore
    from tests.test_measurement_workflow import command, harness

    harness(monkeypatch, capsys, crux_origin=True)
    assert main(command(project, {"max_pages": 2})) == 0
    payload = DiagnosticStore(project).load("public-v1", "run-1").run.model_dump(mode="json")
    field_records = [c["field"] for c in payload["collections"] if c["field"]]
    field_records[-1]["record_key"] = equivalent
    run = DiagnosticRunV2.model_validate(payload)
    client = build(report_for(project).model_copy(update={"run": run}))
    assert len(client.field_rows) == 2
    retained = [c.field.record_key for c in client.report.run.collections if c.field]
    assert len(retained) == 4 and equivalent in retained


def test_origin_key_keeps_conflicting_metrics_periods_devices_and_real_origins_distinct(project):
    from ai_search_audit.performance_models import FieldMeasurement
    from tests.test_diagnostic_versions import normalized_crux_run

    raw = normalized_crux_run(project).collections[1].field.model_dump(mode="python")
    raw.update(scope="origin", record_key="https://studio.example", url_normalization=None)
    record = FieldMeasurement.model_validate(raw)
    key = client_module()._field_key(record)
    variants = []
    for url in ("http://studio.example", "https://studio.example:444", "https://other.example"):
        variants.append(dict(raw, record_key=url, requested_url=url))
    variants.append(dict(raw, device="desktop"))
    variants.append(dict(raw, period={**raw["period"], "first_date": date(2026, 8, 2)}))
    variants.append(dict(raw, metrics=({**raw["metrics"][0], "value": 999.0}, *raw["metrics"][1:])))
    assert all(
        client_module()._field_key(FieldMeasurement.model_validate(v)) != key for v in variants
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://studio.example/a%7Cb%0A%23%23%20Fake",
        "https://studio.example/%5Bclick%5D(https://evil.example)",
    ],
)
def test_page_display_cannot_inject_markdown_layout_or_links(url):
    text = client_module()._page(url, pl=False)
    assert "|" not in text and "\n" not in text
    assert "[click]" not in text


def test_default_date_is_current_utc_not_provider_timestamp(project, monkeypatch):
    module = client_module()
    from datetime import UTC, datetime

    class TrustedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz is UTC
            return cls(2028, 1, 1, tzinfo=UTC)

    monkeypatch.setattr(module, "datetime", TrustedClock)
    client = module.build_measurement_client(report_for(project))
    assert client.as_of == date(2028, 1, 1)
    assert all(rule.state.value == "REQUIRES_VERIFICATION" for rule in client.rules)


def test_partial_field_metrics_are_explicit_and_url_scope_stays_a_page(project):
    from tests.test_diagnostic_versions import normalized_crux_run

    payload = normalized_crux_run(project).model_dump(mode="json")
    payload["collections"][1]["field"]["metrics"][1]["value"] = None
    payload["collections"][1]["attempts"][0].update(state="PARTIAL", reason="missing_metrics")
    report = report_for(project).model_copy(update={"run": DiagnosticRunV2.model_validate(payload)})
    client = build(report)
    assert len(client.field_rows) == 1
    assert client.field_rows[0][0].startswith("Page: /normalized")
    assert client.field_rows[0][3] == "unavailable"
    assert "partial" in " ".join(client.field_notes).lower()


def test_disabled_modules_do_not_imply_no_google_records(project):
    from ai_search_audit.diagnostic_workflow import assemble_performance_run
    from ai_search_audit.measurement_profile import (
        MeasurementProfile,
        prepare_measurement_preflight,
    )
    from tests.test_diagnostic_workflow import _contract

    source = _contract(project).source
    preflight = prepare_measurement_preflight(
        source, MeasurementProfile(pagespeed_insights=False, crux=False)
    )
    run = assemble_performance_run(source, preflight, ())
    client = build(report_for(project).model_copy(update={"run": run}))
    assert not client.lab_rows and not client.field_rows
    assert "disabled" in " ".join(client.lab_notes + client.field_notes).lower()
    assert "no published" not in " ".join(client.field_notes).lower()


def test_lab_redirect_to_another_audited_page_is_visible(project):
    from urllib.parse import urlsplit

    report = report_for(project)
    payload = report.run.model_dump(mode="json")
    final_url = report.run.preflight.selected_pages[1].url
    payload["collections"][0]["lab"]["final_url"] = final_url
    client = build(report.model_copy(update={"run": DiagnosticRunV2.model_validate(payload)}))
    assert "→" in client.lab_rows[0][0]
    assert urlsplit(final_url).path in client.lab_rows[0][0]


def test_successful_local_fallback_is_one_visible_result_per_slot(project, monkeypatch, capsys):
    import json

    from ai_search_audit.cli import main
    from ai_search_audit.diagnostic_store import DiagnosticStore
    from ai_search_audit.lighthouse_runtime import RuntimeResult
    from ai_search_audit.performance_models import LocalRuntimeFingerprint
    from tests.test_lighthouse_provider import fingerprint_data, lhr
    from tests.test_measurement_workflow import command, harness

    harness(monkeypatch, capsys, psi_status=503)
    monkeypatch.setattr("ai_search_audit.performance_providers.time.sleep", lambda value: None)

    class Runtime:
        def run(self, request):
            payload = lhr()
            payload.update(requestedUrl=request.requested_url, finalUrl=request.requested_url)
            payload["configSettings"].update(formFactor=request.device, locale=request.locale)
            return RuntimeResult(
                body=json.dumps(payload).encode(),
                fingerprint=LocalRuntimeFingerprint(**fingerprint_data()),
            )

    monkeypatch.setattr("ai_search_audit.measurement_workflow._local_runtime", Runtime)
    assert main(command(project, {"max_pages": 1, "lighthouse_local": True})) == 0
    run = DiagnosticStore(project).load("public-v1", "run-1").run
    client = build(report_for(project).model_copy(update={"run": run}))
    assert len(client.lab_rows) == 2
    assert all(row[2] == "Local Lighthouse" and "70/100" in row[3] for row in client.lab_rows)
    assert "replaces" in " ".join(client.lab_notes)
    assert len([c for c in client.report.run.collections if c.attempts[0].provider != "crux"]) == 4


@pytest.mark.parametrize("value", [5e-324, 1e308])
def test_extreme_finite_durations_remain_short_and_do_not_underflow_to_zero(value):
    display = client_module()._value(value, seconds=True, pl=False)
    assert display != "0 s"
    assert len(display) < 25


def test_rounding_near_thresholds_is_visibly_approximate(project):
    client = build(report_for(project, score=89.999, lcp=4001))
    assert client.lab_rows[0][3] == "≈90/100; needs improvement"
    assert client.lab_rows[0][4] == "≈4 s; poor"


@pytest.mark.parametrize(
    "reason", ["missing_key", "invalid_key", "http_500", "no_record", "no_data"]
)
def test_crux_unavailable_reasons_are_not_conflated_with_no_records(project, reason):
    payload = performance_run(project).model_dump(mode="python")
    for collection in payload["collections"]:
        if collection["attempts"][0]["provider"] != "crux":
            continue
        first = collection["attempts"][0]
        first.update(reason=reason, http_status=500 if reason == "http_500" else None)
        if reason in {"no_record", "no_data"}:
            from urllib.parse import urlsplit

            parts = urlsplit(first["requested_url"])
            collection["attempts"] = (
                *collection["attempts"],
                dict(
                    first,
                    attempt_id="origin-" + first["attempt_id"],
                    requested_url=f"{parts.scheme}://{parts.netloc}",
                    started_at=first["ended_at"],
                    ended_at=first["ended_at"],
                ),
            )
    from ai_search_audit.diagnostic_performance import performance_collection_range
    from ai_search_audit.performance_providers import PerformanceCollection

    payload["collection_range"] = performance_collection_range(
        tuple(PerformanceCollection.model_validate(c) for c in payload["collections"])
    )
    report = report_for(project).model_copy(update={"run": DiagnosticRunV2.model_validate(payload)})
    client = build(report)
    assert not client.field_rows
    prose = " ".join(client.field_notes).lower()
    assert ("no published" in prose) == (reason in {"no_record", "no_data"})
    assert "readiness" in prose
    if reason in {"no_record", "no_data"}:
        assert "traffic" in prose
    else:
        assert reason not in prose
