import json
import logging
import socket
from dataclasses import FrozenInstanceError
from urllib.parse import quote

import httpcore
import httpx
import pytest
from httpcore._trace import Trace
from pydantic import SecretStr, ValidationError

from ai_search_audit.performance_http import (
    HTTPRequestLimits,
    PerformanceHTTPClient,
    PerformanceRequest,
    ProviderHTTPResponse,
)

KEY = "fake-performance-key-NOT-A-CREDENTIAL"


def request(**updates):
    return PerformanceRequest.model_validate(
        {
            "provider": "pagespeed_insights",
            "requested_url": "https://audit.example/a?q=1",
            "device": "mobile",
            "locale": "en",
            **updates,
        }
    )


def public_dns(host):
    return ["93.184.216.34"]


def no_http(request):
    pytest.fail("HTTP must not run")


def no_dns(host):
    pytest.fail("DNS must not run")


def test_missing_key_never_resolves_or_sends():
    result = PerformanceHTTPClient(transport=httpx.MockTransport(no_http), resolver=no_dns).send(
        PerformanceRequest(
            provider="pagespeed_insights",
            requested_url="https://audit.example/",
            device="mobile",
            locale="en",
        ),
        api_key=None,
    )
    assert result.failure == "missing_key"
    assert result.status_code is None
    assert result.body == b""


@pytest.mark.parametrize("key", [None, "", " ", "\t\r\n"])
def test_blank_keys_do_not_perform_io(key):
    result = PerformanceHTTPClient(
        transport=httpx.MockTransport(no_http),
        resolver=no_dns,
    ).send(request(), api_key=SecretStr(key) if key is not None else None)
    assert result == ProviderHTTPResponse(None, failure="missing_key")


@pytest.mark.parametrize("key", ["a b", " a", "a\n", "a\x00b", "é", "\x7f", "x" * 513])
def test_invalid_keys_do_not_perform_io(key):
    result = PerformanceHTTPClient(
        transport=httpx.MockTransport(no_http),
        resolver=no_dns,
    ).send(request(), api_key=SecretStr(key))
    assert result == ProviderHTTPResponse(None, failure="invalid_key")
    assert key not in repr(result)


@pytest.mark.parametrize(
    "field,value",
    [
        ("timeout_seconds", 0.9),
        ("timeout_seconds", 120.1),
        ("timeout_seconds", "60"),
        ("timeout_seconds", True),
        ("timeout_seconds", float("inf")),
        ("timeout_seconds", float("nan")),
        ("max_response_bytes", 0),
        ("max_response_bytes", 8388609),
        ("max_response_bytes", 1.0),
        ("max_response_bytes", "1"),
        ("max_response_bytes", True),
        ("extra", 1),
    ],
)
def test_limits_reject_invalid_values(field, value):
    with pytest.raises(ValidationError):
        HTTPRequestLimits.model_validate({field: value})


def test_models_are_frozen_and_limits_have_bounded_defaults():
    limits = HTTPRequestLimits()
    assert limits.timeout_seconds == 60.0
    assert limits.max_response_bytes == 4194304
    with pytest.raises(ValidationError):
        limits.timeout_seconds = 1.0
    with pytest.raises(ValidationError):
        request().device = "desktop"
    with pytest.raises(FrozenInstanceError):
        ProviderHTTPResponse(None).status_code = 200


@pytest.mark.parametrize(
    "updates",
    [
        {"scope": "origin"},
        {"provider": "crux", "scope": "origin"},
        {"provider": "crux", "scope": "origin", "requested_url": "https://audit.example/?q=1"},
        {"api_key": KEY},
        {"provider": "lighthouse_local"},
        {"locale": "de"},
        {"device": "tablet"},
        {"requested_url": "https://127.0.0.1/"},
        {"requested_url": "https://user:password@audit.example/"},
    ],
)
def test_request_rejects_invalid_scope_or_fields(updates):
    with pytest.raises(ValidationError):
        request(**updates)


@pytest.mark.parametrize(
    "updates",
    [
        {"requested_url": "https://127.0.0.1/"},
        {"device": "tablet"},
        {"scope": "origin"},
        {"locale": "de"},
    ],
)
def test_send_revalidates_unchecked_request_copies(updates):
    tampered = request().model_copy(update=updates)
    with pytest.raises(ValidationError):
        PerformanceHTTPClient(transport=httpx.MockTransport(no_http), resolver=no_dns).send(
            tampered,
            api_key=SecretStr(KEY),
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"timeout_seconds": float("inf")},
        {"timeout_seconds": -1.0},
        {"max_response_bytes": 8388609},
        {"max_response_bytes": True},
    ],
)
def test_send_revalidates_unchecked_limits_copies(updates):
    tampered = HTTPRequestLimits().model_copy(update=updates)
    with pytest.raises(ValidationError):
        PerformanceHTTPClient(
            limits=tampered,
            transport=httpx.MockTransport(no_http),
            resolver=no_dns,
        ).send(request(), api_key=SecretStr(KEY))


@pytest.mark.parametrize(
    "addresses",
    [
        [],
        ["127.0.0.1"],
        ["93.184.216.34", "10.0.0.1"],
        ["93.184.216.34", "::1"],
        ["garbage"],
    ],
)
def test_unsafe_dns_never_sends_to_google(addresses):
    result = PerformanceHTTPClient(
        transport=httpx.MockTransport(no_http),
        resolver=lambda host: addresses,
    ).send(request(), api_key=SecretStr(KEY))
    assert result == ProviderHTTPResponse(None, failure="unsafe_target")


def test_dns_errors_are_sanitized():
    def broken_dns(host):
        raise socket.gaierror(KEY)

    result = PerformanceHTTPClient(
        transport=httpx.MockTransport(no_http),
        resolver=broken_dns,
    ).send(request(), api_key=SecretStr(KEY))
    assert result == ProviderHTTPResponse(None, failure="unsafe_target")


class Chunks(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False
        self.reads = 0

    def __iter__(self):
        for chunk in self.chunks:
            self.reads += 1
            yield chunk

    def close(self):
        self.closed = True


@pytest.mark.parametrize("provider", ["pagespeed_insights", "crux"])
@pytest.mark.parametrize("device", ["mobile", "desktop"])
@pytest.mark.parametrize("locale", ["pl", "en"])
def test_exact_provider_request(provider, device, locale):
    calls = []
    payload = b'{"raw": "provider JSON is not parsed here"}'
    stream = Chunks([payload])

    def handler(outgoing):
        calls.append(outgoing)
        assert outgoing.headers["X-Goog-Api-Key"] == KEY
        assert outgoing.headers["Accept"] == "application/json"
        assert outgoing.headers["Accept-Encoding"] == "identity"
        assert "authorization" not in outgoing.headers
        assert "cookie" not in outgoing.headers
        assert KEY not in str(outgoing.url)
        assert KEY.encode() not in outgoing.content
        assert outgoing.extensions["timeout"] == dict.fromkeys(
            ["connect", "read", "write", "pool"],
            7.0,
        )
        if provider == "pagespeed_insights":
            assert outgoing.method == "GET"
            assert str(outgoing.url).split("?")[0] == (
                "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
            )
            assert dict(outgoing.url.params) == {
                "url": "https://audit.example/a?q=1",
                "strategy": device,
                "locale": locale,
                "category": "performance",
            }
            assert outgoing.content == b""
        else:
            assert outgoing.method == "POST"
            assert (
                str(outgoing.url) == "https://chromeuxreport.googleapis.com/v1/records:queryRecord"
            )
            assert json.loads(outgoing.content) == {
                "url": "https://audit.example/a?q=1",
                "formFactor": "PHONE" if device == "mobile" else "DESKTOP",
                "metrics": [
                    "largest_contentful_paint",
                    "interaction_to_next_paint",
                    "cumulative_layout_shift",
                ],
            }
        return httpx.Response(200, stream=stream)

    result = PerformanceHTTPClient(
        limits=HTTPRequestLimits(timeout_seconds=7.0),
        transport=httpx.MockTransport(handler),
        resolver=public_dns,
    ).send(request(provider=provider, device=device, locale=locale), api_key=SecretStr(KEY))
    assert result == ProviderHTTPResponse(200, payload)
    assert len(calls) == 1
    assert stream.closed


@pytest.mark.parametrize("url", ["https://audit.example", "https://audit.example/"])
def test_crux_origin_sends_only_origin_key(url):
    calls = []

    def handler(outgoing):
        body = json.loads(outgoing.content)
        assert body["origin"] == url
        assert "url" not in body
        calls.append(outgoing)
        return httpx.Response(204, stream=Chunks([]))

    result = PerformanceHTTPClient(
        transport=httpx.MockTransport(handler),
        resolver=public_dns,
    ).send(request(provider="crux", scope="origin", requested_url=url), api_key=SecretStr(KEY))
    assert result == ProviderHTTPResponse(204)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "url,hostname",
    [
        ("https://straße.example/über?q=ß", "xn--strae-oqa.example"),
        ("https://ＥＸＡＭＰＬＥ.example/", "example.example"),
        ("https://AUDIT.example./path", "audit.example"),
    ],
)
def test_dns_uses_modern_normalized_hostname_but_payload_preserves_url(url, hostname):
    hosts = []

    def resolver(host):
        hosts.append(host)
        return public_dns(host)

    def handler(outgoing):
        assert outgoing.url.params["url"] == url
        return httpx.Response(200, stream=Chunks([b"{}"]))

    result = PerformanceHTTPClient(
        transport=httpx.MockTransport(handler),
        resolver=resolver,
    ).send(request(requested_url=url), api_key=SecretStr(KEY))
    assert result.failure is None
    assert hosts == [hostname]


@pytest.mark.parametrize("target", [KEY, quote(KEY, safe="").replace("f", "%66")])
def test_secret_in_target_url_cannot_reach_dns_or_http(target):
    result = PerformanceHTTPClient(
        transport=httpx.MockTransport(no_http),
        resolver=no_dns,
    ).send(request(requested_url=f"https://audit.example/{target}"), api_key=SecretStr(KEY))
    assert result == ProviderHTTPResponse(None, failure="unsafe_target")


@pytest.mark.parametrize("status", [301, 302, 303, 304, 307, 308, 403, 404, 429, 500, 503])
def test_redirects_and_errors_are_one_attempt_without_body(status):
    calls = []
    stream = Chunks([KEY.encode()])

    def handler(outgoing):
        calls.append(outgoing)
        return httpx.Response(status, headers={"Location": "https://evil.example/"}, stream=stream)

    result = PerformanceHTTPClient(
        transport=httpx.MockTransport(handler),
        resolver=public_dns,
    ).send(request(), api_key=SecretStr(KEY))
    assert result == ProviderHTTPResponse(status, failure="redirect" if status < 400 else None)
    assert len(calls) == 1
    assert stream.reads == 0
    assert stream.closed


@pytest.mark.parametrize("location", ["mailto:test@example.com", "http://x:bad/", "http://[bad/"])
def test_malformed_redirect_location_is_not_parsed_or_followed(location):
    calls = []
    stream = Chunks([b"must not read"])

    def handler(outgoing):
        calls.append(outgoing)
        return httpx.Response(302, headers={"Location": location}, stream=stream)

    transport = ClosingTransport(handler)
    result = PerformanceHTTPClient(transport=transport, resolver=public_dns).send(
        request(),
        api_key=SecretStr(KEY),
    )
    assert result == ProviderHTTPResponse(302, failure="redirect")
    assert len(calls) == 1
    assert stream.reads == 0
    assert stream.closed and transport.closed


@pytest.mark.parametrize(
    "error,failure",
    [
        (httpx.ReadTimeout, "timeout"),
        (httpx.ConnectTimeout, "timeout"),
        (httpx.ConnectError, "transport_error"),
        (httpx.RemoteProtocolError, "transport_error"),
    ],
)
def test_http_failures_are_sanitized_and_not_retried(error, failure, caplog, capsys):
    calls = []

    def handler(outgoing):
        calls.append(outgoing)
        raise error(KEY, request=outgoing)

    result = PerformanceHTTPClient(
        transport=httpx.MockTransport(handler),
        resolver=public_dns,
    ).send(request(), api_key=SecretStr(KEY))
    assert result == ProviderHTTPResponse(None, failure=failure)
    assert len(calls) == 1
    assert KEY not in repr(result) + caplog.text + str(capsys.readouterr())


def test_fresh_clients_do_not_reuse_cookies_or_environment_auth(monkeypatch, tmp_path):
    # Environment paths must never be consulted; no credential fixture is written.
    monkeypatch.setenv("NETRC", str(tmp_path / "absent-netrc"))
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "absent-ca"))
    calls = []

    def handler(outgoing):
        assert "cookie" not in outgoing.headers
        assert "authorization" not in outgoing.headers
        calls.append(outgoing)
        return httpx.Response(200, headers={"Set-Cookie": "session=secret"}, stream=Chunks([b"{}"]))

    client = PerformanceHTTPClient(transport=httpx.MockTransport(handler), resolver=public_dns)
    assert client.send(request(), api_key=SecretStr(KEY)).status_code == 200
    assert client.send(request(), api_key=SecretStr(KEY)).status_code == 200
    assert len(calls) == 2


def test_long_accepted_url_is_not_rejected_by_dns_normalization():
    url = "https://audit.example/" + "a" * 3000

    def handler(outgoing):
        assert outgoing.url.params["url"] == url
        return httpx.Response(200, stream=Chunks([b"{}"]))

    result = PerformanceHTTPClient(
        transport=httpx.MockTransport(handler),
        resolver=public_dns,
    ).send(request(requested_url=url), api_key=SecretStr(KEY))
    assert result == ProviderHTTPResponse(200, b"{}")


class ClosingTransport(httpx.MockTransport):
    def __init__(self, handler):
        super().__init__(handler)
        self.closed = False

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    "headers,chunks,failure,reads",
    [
        ({"Content-Length": "9"}, [b"{}"], "response_too_large", 0),
        ({"Content-Length": "99999999999999999999"}, [b"{}"], "response_too_large", 0),
        ({}, [b"1234", b"56789"], "response_too_large", 2),
        ({"Content-Length": "2"}, [b"123456789"], "response_too_large", 1),
        ({"Content-Encoding": "gzip"}, [b"not actually compressed"], "unsupported_encoding", 0),
        ({"Content-Encoding": "br"}, [b"abc"], "unsupported_encoding", 0),
        ({"Content-Encoding": "identity, gzip"}, [b"abc"], "unsupported_encoding", 0),
    ],
)
def test_bounded_response_rejection_and_cleanup(headers, chunks, failure, reads):
    stream = Chunks(chunks)
    transport = ClosingTransport(
        lambda outgoing: httpx.Response(200, headers=headers, stream=stream)
    )
    result = PerformanceHTTPClient(
        limits=HTTPRequestLimits(max_response_bytes=8),
        transport=transport,
        resolver=public_dns,
    ).send(request(), api_key=SecretStr(KEY))
    assert result == ProviderHTTPResponse(200, failure=failure)
    assert stream.reads == reads
    assert stream.closed and transport.closed


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Content-Length": "8"},
        {"Content-Length": "1"},
        {"Content-Length": "bogus"},
        {"Content-Encoding": "identity"},
        {"Content-Encoding": " Identity "},
    ],
)
def test_actual_bytes_at_exact_limit_succeed_without_json_parsing(headers):
    stream = Chunks([b"not ", b"json"])
    transport = ClosingTransport(
        lambda outgoing: httpx.Response(200, headers=headers, stream=stream)
    )
    result = PerformanceHTTPClient(
        limits=HTTPRequestLimits(max_response_bytes=8),
        transport=transport,
        resolver=public_dns,
    ).send(request(), api_key=SecretStr(KEY))
    assert result == ProviderHTTPResponse(200, b"not json")
    assert stream.closed and transport.closed


def test_large_external_chunk_is_bounded_by_actual_size():
    stream = Chunks([b"x" * (65536 * 3)])
    result = PerformanceHTTPClient(
        limits=HTTPRequestLimits(max_response_bytes=65537),
        transport=httpx.MockTransport(lambda outgoing: httpx.Response(200, stream=stream)),
        resolver=public_dns,
    ).send(request(), api_key=SecretStr(KEY))
    assert result == ProviderHTTPResponse(200, failure="response_too_large")
    assert stream.closed


def test_large_successful_raw_chunk_remains_unchanged():
    payload = b"x" * (65536 * 2 + 1)
    result = PerformanceHTTPClient(
        transport=httpx.MockTransport(
            lambda outgoing: httpx.Response(200, stream=Chunks([payload]))
        ),
        resolver=public_dns,
    ).send(request(), api_key=SecretStr(KEY))
    assert result == ProviderHTTPResponse(200, payload)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_deadline_checks_each_tiny_raw_chunk_without_waiting_for_64k():
    clock = Clock()

    def drip():
        for _ in range(10000):
            clock.now += 0.6
            yield b"x"

    stream = Chunks(drip())
    transport = ClosingTransport(lambda outgoing: httpx.Response(200, stream=stream))
    result = PerformanceHTTPClient(
        limits=HTTPRequestLimits(timeout_seconds=1.0),
        transport=transport,
        resolver=public_dns,
        clock=clock,
    ).send(request(), api_key=SecretStr(KEY))
    assert result == ProviderHTTPResponse(200, failure="timeout")
    assert stream.reads == 2
    assert stream.closed and transport.closed


def test_deadline_checks_after_empty_stream_finishes():
    clock = Clock()

    def empty_slow():
        clock.now += 2.0
        yield from []

    stream = Chunks(empty_slow())
    transport = ClosingTransport(lambda outgoing: httpx.Response(200, stream=stream))
    result = PerformanceHTTPClient(
        limits=HTTPRequestLimits(timeout_seconds=1.0),
        transport=transport,
        resolver=public_dns,
        clock=clock,
    ).send(request(), api_key=SecretStr(KEY))
    assert result == ProviderHTTPResponse(200, failure="timeout")
    assert stream.closed and transport.closed


@pytest.mark.parametrize(
    "error,failure",
    [
        (httpx.ReadTimeout, "timeout"),
        (httpx.ReadError, "transport_error"),
    ],
)
def test_partial_stream_error_discards_body_and_closes_all(error, failure, caplog):
    def broken():
        yield b"partial-body"
        raise error(KEY)

    stream = Chunks(broken())
    transport = ClosingTransport(lambda outgoing: httpx.Response(200, stream=stream))
    result = PerformanceHTTPClient(transport=transport, resolver=public_dns).send(
        request(),
        api_key=SecretStr(KEY),
    )
    assert result == ProviderHTTPResponse(200, failure=failure)
    assert stream.closed and transport.closed
    assert KEY not in repr(result) + caplog.text


@pytest.mark.parametrize("chunks", [[KEY.encode()], [KEY[:10].encode(), KEY[10:].encode()]])
def test_success_body_echoing_key_is_discarded(chunks, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    stream = Chunks(chunks)
    transport = ClosingTransport(lambda outgoing: httpx.Response(200, stream=stream))
    result = PerformanceHTTPClient(transport=transport, resolver=public_dns).send(
        request(),
        api_key=SecretStr(KEY),
    )
    assert result == ProviderHTTPResponse(200, failure="sensitive_response")
    assert stream.closed and transport.closed
    assert KEY not in repr(result) + caplog.text + str(capsys.readouterr())


def test_response_repr_omits_entire_transient_body():
    result = ProviderHTTPResponse(200, b"sensitive raw content")
    assert "sensitive raw content" not in repr(result)
    assert "body" not in repr(result)


@pytest.mark.parametrize("key", ["~", "x" * 512])
def test_key_length_boundaries_are_accepted(key):
    def handler(outgoing):
        assert outgoing.headers["X-Goog-Api-Key"] == key
        return httpx.Response(200, stream=Chunks([b"{}"]))

    result = PerformanceHTTPClient(
        transport=httpx.MockTransport(handler),
        resolver=public_dns,
    ).send(request(), api_key=SecretStr(key))
    assert result == ProviderHTTPResponse(200, b"{}")


@pytest.mark.parametrize("status", [200, 302, 403])
def test_all_response_statuses_close_transport(status):
    stream = Chunks([b"{}"])
    transport = ClosingTransport(lambda outgoing: httpx.Response(status, stream=stream))
    result = PerformanceHTTPClient(transport=transport, resolver=public_dns).send(
        request(),
        api_key=SecretStr(KEY),
    )
    assert result.status_code == status
    assert transport.closed and stream.closed


def test_pre_response_transport_exception_closes_client_transport():
    def handler(outgoing):
        raise httpx.ConnectError(KEY)

    transport = ClosingTransport(handler)
    result = PerformanceHTTPClient(transport=transport, resolver=public_dns).send(
        request(),
        api_key=SecretStr(KEY),
    )
    assert result == ProviderHTTPResponse(None, failure="transport_error")
    assert transport.closed


@pytest.mark.parametrize("key", [None, "invalid key"])
def test_preflight_failure_leaves_unused_injected_transport_with_caller(key):
    transport = ClosingTransport(no_http)
    result = PerformanceHTTPClient(transport=transport, resolver=no_dns).send(
        request(),
        api_key=SecretStr(key) if key is not None else None,
    )
    assert result.failure in {"missing_key", "invalid_key"}
    assert not transport.closed


@pytest.mark.parametrize("url", ["https://8.8.8.8/", "https://[2606:4700:4700::1111]/"])
def test_public_ip_literals_do_not_need_dns(url):
    def handler(outgoing):
        assert outgoing.url.params["url"] == url
        return httpx.Response(200, stream=Chunks([b"{}"]))

    result = PerformanceHTTPClient(
        transport=httpx.MockTransport(handler),
        resolver=no_dns,
    ).send(request(requested_url=url), api_key=SecretStr(KEY))
    assert result == ProviderHTTPResponse(200, b"{}")


def test_provider_metadata_never_reaches_http_library_logs(caplog):
    caplog.set_level(logging.DEBUG)

    def handler(outgoing):
        core_request = httpcore.Request(
            "GET", "https://audit.example/", extensions=outgoing.extensions
        )
        kwargs = {"host": "fixed.example", "timeout": 60.0}

        def connect_tcp(*, host, timeout):
            return host, timeout

        with Trace("connect_tcp", logging.getLogger("httpcore.connection"), core_request, kwargs):
            # Started info is actual operation kwargs: deleting it breaks networking.
            assert connect_tcp(**kwargs) == ("fixed.example", 60.0)
        with Trace(
            "receive_response_headers", logging.getLogger("httpcore.http11"), core_request
        ) as trace:
            trace.return_value = (b"HTTP/1.1", 403, KEY.encode(), [(b"Echo", KEY.encode())])
        return httpx.Response(
            403,
            headers={"Echo": KEY},
            extensions={
                "reason_phrase": KEY.encode(),
                "http_version": KEY.encode(),
            },
            stream=Chunks([]),
        )

    result = PerformanceHTTPClient(
        transport=httpx.MockTransport(handler),
        resolver=public_dns,
    ).send(request(), api_key=SecretStr(KEY))
    assert result == ProviderHTTPResponse(403)
    assert KEY not in caplog.text
    logging.getLogger("httpcore.http11").debug("outside performance attempt remains visible")
    assert "outside performance attempt remains visible" in caplog.text


def test_httpcore_exception_trace_is_sanitized_without_global_logger_changes(caplog):
    caplog.set_level(logging.DEBUG)
    logger = logging.getLogger("httpcore.connection")
    state = (logger.level, logger.disabled, list(logger.filters), list(logger.handlers))

    def handler(outgoing):
        core_request = httpcore.Request(
            "GET", "https://audit.example/", extensions=outgoing.extensions
        )
        with Trace("connect_tcp", logger, core_request):
            raise httpx.ConnectError(KEY)

    result = PerformanceHTTPClient(
        transport=httpx.MockTransport(handler),
        resolver=public_dns,
    ).send(request(), api_key=SecretStr(KEY))
    assert result == ProviderHTTPResponse(None, failure="transport_error")
    assert KEY not in caplog.text
    assert (logger.level, logger.disabled, list(logger.filters), list(logger.handlers)) == state
