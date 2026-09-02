"""Evidence-backed SEO and AI Search audit engine."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .orchestrator import run_public_audit

__version__ = "0.2.0"


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name == "run_public_audit":
        from .orchestrator import run_public_audit

        return run_public_audit
    raise AttributeError(name)


__all__ = ["__version__", "run_public_audit"]
