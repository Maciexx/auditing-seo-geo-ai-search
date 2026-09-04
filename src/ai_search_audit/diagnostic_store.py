"""Immutable supplementary bundles in the existing single-writer local project scope."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path

from ai_search_audit.benchmark import (
    _source_bound_sample,
    compare_benchmarks,
    validate_benchmark_worksheet,
)
from ai_search_audit.diagnostic_intake import _reject_json_constant, _unique_json_object
from ai_search_audit.diagnostic_models import (
    DiagnosticCaptureEvidence,
    DiagnosticFile,
    DiagnosticManifest,
    DiagnosticRun,
    DiagnosticRunReference,
    DiagnosticSource,
    LoadedDiagnosticRun,
    _lossless_diagnostic_values,
)
from ai_search_audit.diagnostic_observations import (
    DiagnosticObservationManifest,
    DiagnosticObservationRun,
    LoadedDiagnosticObservationRun,
    ObservationAttempt,
    validate_observation_source,
)
from ai_search_audit.diagnostic_performance import (
    DiagnosticManifestV2,
    DiagnosticRunV2,
    LoadedDiagnosticRunV2,
    ordered_collections,
    reject_binary_values,
    validate_performance_source,
)
from ai_search_audit.diagnostic_sources import (
    _diagnostic_validation_operation,
    load_diagnostic_source,
)
from ai_search_audit.diagnostic_workflow import (
    _project_directory,
    _real_directory_path,
    _selected_pages,
)
from ai_search_audit.measurement_profile import MeasurementPreflight, _reject_unserialized_fields
from ai_search_audit.performance_providers import PerformanceCollection
from ai_search_audit.project_store import _rename_directory_no_replace

_MAX_FILE_BYTES = 8 * 1024 * 1024
_INVENTORY = {"diagnostics.json", "evidence.jsonl", "manifest.json"}
StoredRun = DiagnosticRun | DiagnosticRunV2 | DiagnosticObservationRun
LoadedRun = LoadedDiagnosticRun | LoadedDiagnosticRunV2 | LoadedDiagnosticObservationRun


def _schema(value: object) -> str:
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("schema_version"), str)
        or value.get("schema_version") not in {"1.0.0", "2.0.0", "3.0.0"}
    ):
        raise ValueError("unsupported diagnostic schema_version")
    return str(value["schema_version"])


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode()


def _json_load(data: bytes) -> object:
    return json.loads(
        data.decode(), object_pairs_hook=_unique_json_object, parse_constant=_reject_json_constant
    )


def _read_regular(path: Path) -> bytes:
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("diagnostic artifact must be a regular file")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("diagnostic artifact must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            content = stream.read(_MAX_FILE_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(content) > _MAX_FILE_BYTES:
        raise ValueError("diagnostic artifact exceeds size bound")
    return content


class DiagnosticStore:
    """No database or concurrent writer coordination: callers serialize local writes."""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = _real_directory_path(project_root)

    def _source(self, version: str, sources: dict[str, DiagnosticSource]) -> DiagnosticSource:
        _project_directory(self.project_root)
        if version not in sources:
            sources[version] = load_diagnostic_source(
                f"project:{self.project_root.name}",
                clients_root=self.project_root.parent,
                source_version=version,
            )
        return sources[version]

    def _validate_source_bound(
        self,
        run: StoredRun,
        source: DiagnosticSource,
        sources: dict[str, DiagnosticSource],
        seen: frozenset[tuple[str, str]] = frozenset(),
    ) -> None:
        if isinstance(run, DiagnosticObservationRun):
            validate_observation_source(run, source)
            return
        if isinstance(run, DiagnosticRunV2):
            validate_performance_source(run, source)
            return
        if run.binding != source.binding:
            raise ValueError("diagnostic run binding does not match canonical source")
        _selected_pages(source, run.selected_pages)
        validate_benchmark_worksheet(source, run.worksheet)
        if run.benchmark is not None:
            _source_bound_sample(source, run.benchmark)
        if run.baseline is not None:
            baseline = self._load(
                run.baseline.reference.source_version, run.baseline.reference.run_id, seen, sources
            )
            if (
                baseline.manifest_sha256 != run.baseline.manifest_sha256
                or baseline.run.binding != run.baseline.binding
            ):
                raise ValueError("baseline binding or manifest hash mismatch")
            if (
                run.benchmark is None
                or not isinstance(baseline.run, DiagnosticRun)
                or baseline.run.benchmark is None
            ):
                raise ValueError("comparison requires benchmark samples")
            expected = compare_benchmarks(
                baseline.run.benchmark,
                run.benchmark,
                baseline_source=self._source(baseline.run.binding.source_version, sources),
                follow_up_source=source,
            )
            if run.comparison != expected:
                raise ValueError("stored benchmark comparison does not match baseline")

    @_diagnostic_validation_operation
    def publish(self, run: StoredRun) -> Path:
        _reject_unserialized_fields(run)
        if isinstance(run, DiagnosticRunV2):
            reject_binary_values(run)
        values = (
            _lossless_diagnostic_values(run)
            if isinstance(run, DiagnosticObservationRun)
            else run.model_dump(mode="python", serialize_as_any=True, warnings=False)
        )
        version = _schema(values)
        run = (
            DiagnosticRun
            if version == "1.0.0"
            else DiagnosticRunV2
            if version == "2.0.0"
            else DiagnosticObservationRun
        ).model_validate(values)
        sources: dict[str, DiagnosticSource] = {}
        source = self._source(run.binding.source_version, sources)
        self._validate_source_bound(run, source, sources)
        diagnostics = _json_bytes(
            run.model_dump(mode="json", exclude={"captures", "collections", "attempts"})
        )
        if isinstance(run, DiagnosticObservationRun):
            evidence = b"".join(
                _json_bytes(attempt.model_dump(mode="json"))
                for attempt in sorted(run.attempts, key=lambda item: item.attempt_id)
            )
        elif isinstance(run, DiagnosticRunV2):
            evidence = b"".join(
                _json_bytes(collection.model_dump(mode="json"))
                for collection in sorted(
                    run.collections, key=lambda item: item.attempts[0].attempt_id
                )
            )
        else:
            evidence = b"".join(
                _json_bytes(capture.model_dump(mode="json"))
                for capture in sorted(run.captures, key=lambda item: item.evidence_id)
            )
        files = (
            DiagnosticFile(
                filename="diagnostics.json",
                byte_count=len(diagnostics),
                sha256=hashlib.sha256(diagnostics).hexdigest(),
            ),
            DiagnosticFile(
                filename="evidence.jsonl",
                byte_count=len(evidence),
                sha256=hashlib.sha256(evidence).hexdigest(),
            ),
        )
        directory = self.project_root / "diagnostics"
        if not os.path.lexists(directory):
            directory.mkdir()
        _real_directory_path(directory)
        directory /= run.binding.source_version
        if not os.path.lexists(directory):
            directory.mkdir()
        _real_directory_path(directory)
        numbers: list[int] = []
        for child in directory.iterdir():
            if child.name.startswith(".run-"):
                _real_directory_path(child)
                continue  # Interrupted staging is neither promoted nor silently removed.
            if not re.fullmatch(r"run-[1-9][0-9]*", child.name):
                raise ValueError("unexpected diagnostic run directory inventory")
            _real_directory_path(child)
            numbers.append(int(child.name.removeprefix("run-")))
        number = max(numbers, default=0) + 1
        manifest_values = dict(
            binding=run.binding,
            run_number=number,
            files=files,
            input_sha256=run.input_sha256,
            algorithm_sha256=run.algorithm_sha256,
            cleanup=run.cleanup.model_dump(mode="python"),
        )
        manifest = (
            DiagnosticObservationManifest
            if isinstance(run, DiagnosticObservationRun)
            else DiagnosticManifestV2
            if isinstance(run, DiagnosticRunV2)
            else DiagnosticManifest
        ).model_validate(manifest_values)
        manifest_bytes = _json_bytes(manifest.model_dump(mode="json"))
        if any(len(data) > _MAX_FILE_BYTES for data in (diagnostics, evidence, manifest_bytes)):
            raise ValueError("diagnostic artifact exceeds size bound")
        staging = Path(tempfile.mkdtemp(prefix=".run-", dir=directory))
        destination = directory / f"run-{number}"
        try:
            for name, data in (
                ("diagnostics.json", diagnostics),
                ("evidence.jsonl", evidence),
                ("manifest.json", manifest_bytes),
            ):
                with (staging / name).open("xb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
            _rename_directory_no_replace(staging, destination)
        finally:
            if staging.exists():
                # Only this invocation's newly created sibling is owned for cleanup.
                shutil.rmtree(staging)
        return destination

    @_diagnostic_validation_operation
    def load(self, source_version: str, run_name: str) -> LoadedRun:
        return self._load(source_version, run_name, frozenset(), {})

    def _load(
        self,
        source_version: str,
        run_name: str,
        seen: frozenset[tuple[str, str]],
        sources: dict[str, DiagnosticSource],
    ) -> LoadedRun:
        reference = DiagnosticRunReference(source_version=source_version, run_id=run_name)
        key = (source_version, run_name)
        if key in seen or len(seen) >= 32:
            raise ValueError("cyclic or excessive diagnostic baseline references")
        source = self._source(reference.source_version, sources)
        directory = self.project_root / "diagnostics" / source_version / run_name
        _real_directory_path(directory)
        if {path.name for path in directory.iterdir()} != _INVENTORY:
            raise ValueError("diagnostic run inventory mismatch")
        content = {name: _read_regular(directory / name) for name in _INVENTORY}
        manifest_values = _json_load(content["manifest.json"])
        manifest = (
            DiagnosticManifest
            if _schema(manifest_values) == "1.0.0"
            else DiagnosticManifestV2
            if _schema(manifest_values) == "2.0.0"
            else DiagnosticObservationManifest
        ).model_validate(manifest_values)
        if manifest.binding != source.binding or manifest.run_number != int(run_name[4:]):
            raise ValueError("diagnostic manifest source binding or run number mismatch")
        for item in manifest.files:
            data = content[item.filename]
            if len(data) != item.byte_count or hashlib.sha256(data).hexdigest() != item.sha256:
                raise ValueError("diagnostic artifact size or hash mismatch")
        values = _json_load(content["diagnostics.json"])
        if _schema(values) != manifest.schema_version:
            raise ValueError("diagnostic run and manifest schema_version mismatch")
        if not isinstance(values, dict) or {"captures", "collections", "attempts"} & values.keys():
            raise ValueError("invalid diagnostics document inventory")
        if isinstance(manifest, DiagnosticObservationManifest):
            attempts = tuple(
                ObservationAttempt.model_validate(_json_load(line))
                for line in content["evidence.jsonl"].splitlines()
            )
            if tuple(a.attempt_id for a in attempts) != tuple(
                sorted(a.attempt_id for a in attempts)
            ):
                raise ValueError("normalized evidence must be sorted deterministically")
            selected = values.get("selected_prompt_ids")
            if not isinstance(selected, list) or not all(isinstance(p, str) for p in selected):
                raise ValueError("invalid selected prompt inventory")
            order = {prompt: i for i, prompt in enumerate(selected)}
            values["attempts"] = sorted(attempts, key=lambda a: order.get(a.prompt_id, -1))
            observation_run = DiagnosticObservationRun.model_validate(values)
            if (
                observation_run.binding != manifest.binding
                or observation_run.input_sha256 != manifest.input_sha256
                or observation_run.algorithm_sha256 != manifest.algorithm_sha256
                or observation_run.cleanup != manifest.cleanup
            ):
                raise ValueError("diagnostic run manifest provenance mismatch")
            self._validate_source_bound(observation_run, source, sources, seen | {key})
            return LoadedDiagnosticObservationRun(
                run=observation_run,
                manifest=manifest,
                manifest_sha256=hashlib.sha256(content["manifest.json"]).hexdigest(),
            )
        if isinstance(manifest, DiagnosticManifestV2):
            collections = tuple(
                PerformanceCollection.model_validate(_json_load(line))
                for line in content["evidence.jsonl"].splitlines()
            )
            if tuple(c.attempts[0].attempt_id for c in collections) != tuple(
                sorted(c.attempts[0].attempt_id for c in collections)
            ):
                raise ValueError("normalized evidence must be sorted deterministically")
            preflight = MeasurementPreflight.model_validate(values.get("preflight"))
            values["collections"] = ordered_collections(preflight, collections)
            performance_run = DiagnosticRunV2.model_validate(values)
            if (
                performance_run.binding != manifest.binding
                or performance_run.input_sha256 != manifest.input_sha256
                or performance_run.algorithm_sha256 != manifest.algorithm_sha256
                or performance_run.cleanup != manifest.cleanup
            ):
                raise ValueError("diagnostic run manifest provenance mismatch")
            self._validate_source_bound(performance_run, source, sources, seen | {key})
            return LoadedDiagnosticRunV2(
                run=performance_run,
                manifest=manifest,
                manifest_sha256=hashlib.sha256(content["manifest.json"]).hexdigest(),
            )
        evidence = tuple(
            DiagnosticCaptureEvidence.model_validate(_json_load(line))
            for line in content["evidence.jsonl"].splitlines()
        )
        if tuple(item.evidence_id for item in evidence) != tuple(
            sorted(item.evidence_id for item in evidence)
        ):
            raise ValueError("normalized evidence must be sorted deterministically")
        pages = values.get("selected_pages")
        if not isinstance(pages, list):
            raise ValueError("invalid selected page inventory")
        order = {page["url"]: i for i, page in enumerate(pages) if isinstance(page, dict)}
        values["captures"] = sorted(evidence, key=lambda c: (order.get(c.url, -1), c.kind != "raw"))
        run = DiagnosticRun.model_validate(values)
        if (
            run.binding != manifest.binding
            or run.input_sha256 != manifest.input_sha256
            or run.algorithm_sha256 != manifest.algorithm_sha256
            or run.cleanup != manifest.cleanup
        ):
            raise ValueError("diagnostic run manifest provenance mismatch")
        self._validate_source_bound(run, source, sources, seen | {key})
        return LoadedDiagnosticRun(
            run=run,
            manifest=manifest,
            manifest_sha256=hashlib.sha256(content["manifest.json"]).hexdigest(),
        )
