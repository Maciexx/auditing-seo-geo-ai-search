from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field, field_validator


def normalize_target(value: str) -> str:
    candidate = value.strip()
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    parts = urlsplit(candidate)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("target must be an HTTP or HTTPS domain")
    if parts.username or parts.password:
        raise ValueError("credentials are not allowed in target URLs")
    host = parts.hostname.lower()
    port = f":{parts.port}" if parts.port else ""
    path = parts.path or ""
    return urlunsplit((parts.scheme.lower(), f"{host}{port}", path, parts.query, ""))


class AuditConfig(BaseModel):
    domain: str
    max_pages: int = Field(default=50, ge=1, le=100)
    timeout_seconds: float = Field(default=15, gt=0, le=30)
    max_response_bytes: int = Field(default=2_000_000, ge=1, le=5_000_000)
    max_redirects: int = Field(default=5, ge=0, le=10)
    max_sitemaps: int = Field(default=10, ge=1, le=20)
    delay_seconds: float = Field(default=0, ge=0, le=5)
    max_text_chars: int = Field(default=20_000, ge=1, le=100_000)
    user_agent: str = "ai-search-audit-skill/0.1 (+public audit)"
    output_dir: Path = Path("audit-output")
    external_urls: list[str] = Field(default_factory=list)
    geo_optimizer_command: str | None = None

    @field_validator("domain")
    @classmethod
    def normalize_domain(cls, value: str) -> str:
        return normalize_target(value)
