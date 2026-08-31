import pytest

from ai_search_audit.config import AuditConfig, normalize_target


def test_normalize_target_defaults_to_https() -> None:
    assert normalize_target("Example.COM/path#part") == "https://example.com/path"


def test_normalize_target_rejects_credentials() -> None:
    with pytest.raises(ValueError, match="credentials"):
        normalize_target("https://user:secret@example.com")


def test_config_uses_conservative_limits() -> None:
    config = AuditConfig(domain="example.com")
    assert config.max_pages <= 100
    assert config.max_response_bytes <= 5_000_000
    assert config.timeout_seconds <= 30
    assert config.max_sitemaps <= 20
