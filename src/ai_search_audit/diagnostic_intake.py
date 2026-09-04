"""Bounded, inert diagnostic input through the existing owned-directory lifecycle."""

from __future__ import annotations

import json
import os
import stat
from typing import NoReturn

from ai_search_audit.data_intake import IntakeValidationError
from ai_search_audit.diagnostic_models import DiagnosticIntake

MAX_DIAGNOSTIC_INTAKE_BYTES = 2 * 1024 * 1024


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate diagnostic JSON key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError("nonstandard diagnostic JSON constant")


def read_diagnostic_payload(owned_fd: int) -> DiagnosticIntake:
    """Read only normalized-intake.json from a borrowed verified owned descriptor.

    Use as a trusted internal processor for consume_owned_payload. HTML and URLs
    remain data, with no execution, external reads or attachment copying. Schema
    validation is local; the coordinator must still verify canonical bindings and
    evidence references before publication, after successful owned-input cleanup.
    """
    try:
        descriptor = os.open(
            "normalized-intake.json",
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=owned_fd,
        )
        try:
            details = os.fstat(descriptor)
            if not stat.S_ISREG(details.st_mode):
                raise IntakeValidationError("diagnostic input must be a regular file")
            data = bytearray()
            while len(data) <= MAX_DIAGNOSTIC_INTAKE_BYTES:
                chunk = os.read(descriptor, MAX_DIAGNOSTIC_INTAKE_BYTES + 1 - len(data))
                if not chunk:
                    break
                data.extend(chunk)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise IntakeValidationError("diagnostic input could not be read safely") from exc
    if len(data) > MAX_DIAGNOSTIC_INTAKE_BYTES:
        raise IntakeValidationError("diagnostic input exceeds total 2 MiB limit")
    try:
        decoded = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
        return DiagnosticIntake.model_validate(decoded)
    except (ValueError, RecursionError) as exc:
        raise IntakeValidationError("diagnostic input failed JSON or schema validation") from exc
