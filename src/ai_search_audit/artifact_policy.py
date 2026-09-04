"""Resolve version policies from saved audit artifacts, not current defaults."""

from typing import Literal

from .models import AuditRun


def classification_version(run: AuditRun) -> Literal["1.0.0", "2.0.0"]:
    value = run.configuration.get("entity_classification_policy", "1.0.0")
    if value == "1.0.0":
        return "1.0.0"
    if value == "2.0.0":
        return "2.0.0"
    raise ValueError("unsupported entity classification policy")


def saved_prompt_version(run: AuditRun) -> str:
    declared = run.configuration.get("prompt_pack_version")
    if "prompt_pack_version" in run.configuration and declared not in ("1.1.0", "2.0.0", "2.1.0"):
        raise ValueError("unsupported prompt pack version")
    versions = {prompt.pack_version for prompt in run.ai_prompts}
    if len(versions) > 1:
        raise ValueError("mixed prompt pack versions")
    value = next(iter(versions), declared if declared is not None else "1.1.0")
    if value not in ("1.1.0", "2.0.0", "2.1.0") or (declared is not None and declared != value):
        raise ValueError("unsupported or mismatched prompt pack version")
    return str(value)
