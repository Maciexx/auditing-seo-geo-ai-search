"""Offline security checks; only DNS and numeric socket dialing are substituted."""

import importlib
import importlib.util
import socket
import stat
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path():
    # macOS sockaddr_un is only 104 bytes; pytest's default path exceeds that.
    with tempfile.TemporaryDirectory(prefix="lh-", dir="/tmp") as directory:
        yield Path(directory).resolve()


def network():
    assert importlib.util.find_spec("ai_search_audit.lighthouse_network") is not None
    return importlib.import_module("ai_search_audit.lighthouse_network")


def request(target="http://audit.example/a?b=1", host="audit.example", extra="", method="GET"):
    return f"{method} {target} HTTP/1.1\r\nHost: {host}\r\n{extra}\r\n".encode("ascii")


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "LOCALHOST.",
        "sub.localhost",
        "127.0.0.1",
        "10.1.2.3",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "0.0.0.0",
        "100.64.0.1",
        "192.0.2.1",
        "224.0.0.1",
        "255.255.255.255",
        "[::]",
        "[::1]",
        "[fc00::1]",
        "[fe80::1]",
        "[ff02::1]",
        "[2001:db8::1]",
        "[::ffff:8.8.8.8]",
        "[2002:0808:0808::1]",
        "[2001:0000:4136:e378:8000:63bf:3fff:fdd2]",
        "2130706433",
        "0x7f000001",
        "0177.0.0.1",
        "127.1",
        "127.000.000.001",
        "0x7f.0.0.1",
        "user:secret@audit.example",
        "audit.example@127.0.0.1",
        "audit.example%00",
        "audit.example\\evil",
        "-bad.example",
        "bad_.example",
        "bad..example",
        "[fe80::1%25eth0]",
        "audit.example:8080",
    ],
)
def test_denies_unsafe_authority_before_dns(host):
    api = network()
    with pytest.raises(api.ProxyPolicyError):
        api.parse_request(request(f"http://{host}/", host), api.ProxyLimits())


@pytest.mark.parametrize(
    "target,host,method",
    [
        ("https://audit.example/", "audit.example", "GET"),
        ("ftp://audit.example/", "audit.example", "GET"),
        ("/relative", "audit.example", "GET"),
        ("http://audit.example/#fragment", "audit.example", "GET"),
        ("audit.example:80", "audit.example:80", "CONNECT"),
        ("audit.example:443/path", "audit.example:443", "CONNECT"),
        ("audit.example", "audit.example", "CONNECT"),
        ("http://audit.example/", "other.example", "GET"),
    ],
)
def test_denies_bad_request_targets(target, host, method):
    api = network()
    with pytest.raises(api.ProxyPolicyError):
        api.parse_request(request(target, host, method=method), api.ProxyLimits())


@pytest.mark.parametrize(
    "extra",
    [
        "Host: audit.example\r\n",
        "Proxy-Authorization: secret\r\n",
        "X-A: a\r\n folded\r\n",
        "Content-Length: -1\r\n",
        "Content-Length: 1, 1\r\n",
        "Content-Length: 1\r\nContent-Length: 1\r\n",
        "Content-Length: +1\r\n",
        "Transfer-Encoding: chunked\r\n",
        "Transfer-Encoding: gzip, chunked\r\n",
        "Content-Length: 1\r\nTransfer-Encoding: chunked\r\n",
        "X-Bad : value\r\n",
        "X-A: secret\x00value\r\n",
        "X-A: a\nB: injected\r\n",
        "Connection: host\r\n",
        "Connection: content-length\r\n",
    ],
)
def test_denies_ambiguous_headers(extra):
    api = network()
    with pytest.raises(api.ProxyPolicyError):
        api.parse_request(request(extra=extra), api.ProxyLimits())


def test_rewrites_path_query_host_and_drops_proxy_connection():
    api = network()
    parsed = api.parse_request(request(extra="Proxy-Connection: keep-alive\r\n"), api.ProxyLimits())
    assert (parsed.host, parsed.port, parsed.connect) == ("audit.example", 80, False)
    assert parsed.upstream_head == (
        b"GET /a?b=1 HTTP/1.1\r\nHost: audit.example\r\nConnection: close\r\n\r\n"
    )


def test_request_repr_does_not_expose_path_query_or_headers():
    api = network()
    parsed = api.parse_request(
        request("http://audit.example/private?secret=token", extra="Cookie: session=secret\r\n"),
        api.ProxyLimits(),
    )
    assert "secret" not in repr(parsed)
    assert "private" not in repr(parsed)
    assert "Cookie" not in repr(parsed)


@pytest.mark.parametrize(
    "host", ["audit.example", "xn--bcher-kva.example", "8.8.8.8", "[2606:4700:4700::1111]"]
)
def test_accepts_public_http_and_connect(host):
    api = network()
    assert api.parse_request(request(f"http://{host}/", host), api.ProxyLimits()).port == 80
    parsed = api.parse_request(
        request(f"{host}:443", f"{host}:443", method="CONNECT"), api.ProxyLimits()
    )
    assert parsed.connect and parsed.port == 443


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_header_bytes", 0),
        ("max_header_bytes", True),
        ("max_header_bytes", 1000001),
        ("max_connections", 0),
        ("max_connections", 1.0),
        ("max_connections", 129),
        ("max_requests", -1),
        ("max_requests", "10"),
        ("max_requests", 100001),
        ("max_transfer_bytes", 0),
        ("max_transfer_bytes", 1073741825),
        ("connection_timeout", 0),
        ("connection_timeout", float("nan")),
        ("connection_timeout", float("inf")),
        ("connection_timeout", True),
        ("connection_timeout", 301),
        ("wall_timeout", 0),
        ("wall_timeout", 901),
    ],
)
def test_limits_are_strict_and_bounded(field, value):
    api = network()
    with pytest.raises(ValueError):
        api.ProxyLimits(**{field: value})


def test_limits_are_frozen():
    api = network()
    with pytest.raises(FrozenInstanceError):
        api.ProxyLimits().max_requests = 999


def answer(ip):
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    return (family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, 80))


def test_dns_rejects_whole_mixed_answer_set_and_rechecks_every_time(monkeypatch):
    api = network()
    answers = iter([[answer("8.8.8.8")], [answer("8.8.8.8"), answer("127.0.0.1")]])
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: next(answers))
    assert api.resolve_public("audit.example", 80) == ("8.8.8.8",)
    with pytest.raises(api.ProxyPolicyError):
        api.resolve_public("audit.example", 80)


@pytest.mark.parametrize("answers", [[], [answer("::ffff:8.8.8.8")], [answer("169.254.169.254")]])
def test_dns_empty_and_unsafe_answers_fail_closed(monkeypatch, answers):
    api = network()
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: answers)
    with pytest.raises(api.ProxyPolicyError):
        api.resolve_public("audit.example", 80)


def test_numeric_connector_never_resolves_hostname_and_closes_failed_socket(monkeypatch):
    api = network()
    calls = []

    class Dial:
        def settimeout(self, timeout):
            calls.append(("timeout", timeout))

        def connect(self, address):
            calls.append(("connect", address))
            raise OSError("secret")

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(socket, "socket", lambda *args: Dial())
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args: pytest.fail("second DNS lookup"))
    with pytest.raises(OSError):
        api.dial_numeric("8.8.8.8", 443, 1.0)
    assert ("connect", ("8.8.8.8", 443)) in calls
    assert calls[-1] == ("close",)
    with pytest.raises(api.ProxyPolicyError):
        api.dial_numeric("audit.example", 443, 1.0)


def proxy():
    assert importlib.util.find_spec("ai_search_audit.lighthouse_proxy") is not None
    return importlib.import_module("ai_search_audit.lighthouse_proxy")


@contextmanager
def running(tmp_path, *, preserved_path=False, **changes):
    api = proxy()
    policy = network()
    limits = policy.ProxyLimits(**{"wall_timeout": 2, "connection_timeout": 0.5, **changes})
    path = tmp_path / "p.sock"
    stop = threading.Event()
    result = []
    worker = threading.Thread(target=lambda: result.append(api.serve_unix(path, limits, stop=stop)))
    worker.start()
    deadline = time.monotonic() + 1
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert path.exists()
    try:
        yield path, worker, result
    finally:
        stop.set()
        worker.join(1)
        assert not worker.is_alive()
        assert path.exists() if preserved_path else not path.exists()


def client(path):
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(2)
    connection.connect(str(path))
    return connection


def receive_all(connection):
    chunks = []
    while True:
        try:
            chunk = connection.recv(4096)
        except ConnectionResetError:
            break
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def upstream_pair(monkeypatch):
    api = proxy()
    upstream, remote = socket.socketpair()
    remote.settimeout(2)
    dials = []
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [answer("8.8.8.8")])

    def dial(address, port, timeout):
        dials.append((address, port))
        return upstream

    monkeypatch.setattr(api, "dial_numeric", dial)
    return upstream, remote, dials


def test_server_rewrites_http_and_streams_fixed_body_without_pipeline(tmp_path, monkeypatch):
    upstream, remote, dials = upstream_pair(monkeypatch)
    with remote, running(tmp_path) as (path, _, _), client(path) as browser:
        browser.sendall(request(extra="Content-Length: 4\r\n", method="POST") + b"body")
        forwarded = remote.recv(4096)
        while not forwarded.endswith(b"body"):
            forwarded += remote.recv(4096)
        assert forwarded.startswith(b"POST /a?b=1 HTTP/1.1\r\n")
        assert b"Connection: close\r\n" in forwarded
        remote.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")
        remote.shutdown(socket.SHUT_WR)
        assert receive_all(browser).endswith(b"OK")
        assert dials == [("8.8.8.8", 80)]
    assert upstream.fileno() == -1


@pytest.mark.parametrize("websocket", [False, True])
def test_connect_and_websocket_tunnels_are_bidirectional(tmp_path, monkeypatch, websocket):
    upstream, remote, dials = upstream_pair(monkeypatch)
    with remote, running(tmp_path) as (path, _, _), client(path) as browser:
        if websocket:
            browser.sendall(request(extra="Connection: Upgrade\r\nUpgrade: websocket\r\n"))
            assert b"Upgrade: websocket\r\n" in remote.recv(4096)
            remote.sendall(b"HTTP/1.1 101 Switching Protocols\r\n\r\n")
            assert b"101" in browser.recv(4096)
        else:
            browser.sendall(request("audit.example:443", "audit.example:443", method="CONNECT"))
            assert browser.recv(4096) == b"HTTP/1.1 200 Connection Established\r\n\r\n"
        browser.sendall(b"browser-data")
        assert remote.recv(4096) == b"browser-data"
        remote.sendall(b"server-data")
        assert browser.recv(4096) == b"server-data"
        browser.shutdown(socket.SHUT_WR)
        assert remote.recv(1) == b""
        remote.shutdown(socket.SHUT_WR)
        assert receive_all(browser) == b""
        assert dials == [("8.8.8.8", 80 if websocket else 443)]
    assert upstream.fileno() == -1


@pytest.mark.parametrize(
    "bad_request",
    [
        request("http://127.0.0.1/secret", "127.0.0.1"),
        request(extra="Proxy-Authorization: secret\r\n"),
        request(extra="Host: other.example\r\n"),
        b"GET http://audit.example/secret HTTP/1.1\r\nHost: audit.example\r\nX: " + b"s" * 5000,
    ],
)
def test_server_rejects_before_connector_and_never_leaks_data(
    tmp_path, monkeypatch, capsys, bad_request
):
    api = proxy()
    monkeypatch.setattr(api, "dial_numeric", lambda *a: pytest.fail("unsafe dialing"))
    with running(tmp_path, max_header_bytes=256) as (path, _, _), client(path) as browser:
        browser.sendall(bad_request)
        response = receive_all(browser)
        assert response == api.DENIED_RESPONSE
        assert b"secret" not in response
    assert capsys.readouterr() == ("", "")


def test_upstream_exception_is_fixed_and_secret_free(tmp_path, monkeypatch, capsys):
    api = proxy()
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [answer("8.8.8.8")])

    def fail(*args):
        raise OSError("https://secret.example/?token=private")

    monkeypatch.setattr(api, "dial_numeric", fail)
    with running(tmp_path) as (path, _, _), client(path) as browser:
        browser.sendall(request())
        assert receive_all(browser) == api.FAILED_RESPONSE
    assert capsys.readouterr() == ("", "")


def test_slow_header_is_bounded_by_absolute_connection_deadline(tmp_path):
    api = proxy()
    with running(tmp_path, connection_timeout=0.08) as (path, _, _), client(path) as browser:
        started = time.monotonic()
        browser.sendall(b"GET ")
        assert receive_all(browser) == api.LIMIT_RESPONSE
        assert time.monotonic() - started < 0.5


def test_checks_deadline_after_dns_before_dial(tmp_path, monkeypatch):
    api = proxy()
    entered = threading.Event()
    release = threading.Event()

    def resolve(*args, **kwargs):
        entered.set()
        release.wait(1)
        return [answer("8.8.8.8")]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(api, "dial_numeric", lambda *a: pytest.fail("dial after deadline"))
    with running(tmp_path, wall_timeout=0.12) as (path, worker, _), client(path) as browser:
        browser.sendall(request())
        assert entered.wait(1)
        worker.join(0.5)
        assert not worker.is_alive()
        release.set()
        receive_all(browser)


def test_concurrency_limit_refuses_extra_client(tmp_path):
    api = proxy()
    with running(tmp_path, max_connections=1) as (path, _, _), client(path) as first:
        first.sendall(b"GET ")
        time.sleep(0.03)
        with client(path) as second:
            assert receive_all(second) == api.LIMIT_RESPONSE


def test_total_request_limit_stops_and_cleans_server(tmp_path):
    with running(tmp_path, max_requests=1) as (path, worker, result), client(path) as browser:
        browser.sendall(b"bad\r\n\r\n")
        receive_all(browser)
        worker.join(0.5)
        assert not worker.is_alive()
        assert result[0].accepted_requests == 1
        assert result[0].policy_denials == 1


def test_global_transfer_limit_closes_stream_and_owned_sockets(tmp_path, monkeypatch):
    upstream, remote, _ = upstream_pair(monkeypatch)
    with (
        remote,
        running(tmp_path, max_transfer_bytes=256) as (path, worker, result),
        client(path) as browser,
    ):
        browser.sendall(request("audit.example:443", "audit.example:443", method="CONNECT"))
        assert b"200" in browser.recv(4096)
        remote.sendall(b"x" * 1024)
        response = receive_all(browser)
        assert len(response) <= 256
        worker.join(0.5)
        assert not worker.is_alive()
        assert result[0].transferred_bytes <= 256
    assert upstream.fileno() == -1


@pytest.mark.parametrize("kind", ["file", "symlink", "socket", "parent-symlink"])
def test_unix_lifecycle_never_overwrites_existing_paths(tmp_path, kind):
    api = proxy()
    path = tmp_path / "existing"
    bound = None
    if kind == "file":
        path.write_text("owned by someone else")
    elif kind == "symlink":
        path.symlink_to(tmp_path / "absent")
    elif kind == "parent-symlink":
        path.symlink_to(tmp_path, target_is_directory=True)
        path = path / "socket"
    else:
        bound = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        bound.bind(str(path))
    try:
        with pytest.raises((OSError, ValueError)):
            api.serve_unix(path, network().ProxyLimits(wall_timeout=0.05))
        if kind == "file":
            assert path.read_text() == "owned by someone else"
        elif kind != "parent-symlink":
            assert path.is_symlink() or path.exists()
    finally:
        if bound:
            bound.close()


def test_owned_socket_has_private_permissions(tmp_path):
    with running(tmp_path) as (path, _, _):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_socket_directory_must_not_be_writable_by_other_users(tmp_path):
    api = proxy()
    tmp_path.chmod(0o777)
    try:
        with pytest.raises((ValueError, OSError)):
            api.serve_unix(tmp_path / "p.sock", network().ProxyLimits(wall_timeout=0.02))
    finally:
        tmp_path.chmod(0o700)


@pytest.mark.parametrize("address", ["64:ff9b::7f00:1", "64:ff9b:1::a00:1", "64:ff9b::a00:1"])
def test_nat64_reserved_prefixes_are_denied(address):
    api = network()
    with pytest.raises(api.ProxyPolicyError):
        api.public_ip(address)


@pytest.mark.parametrize("method", ["GET", "HEAD", "POST", "OPTIONS", "PUT", "PATCH", "DELETE"])
def test_normal_http_methods_are_supported(method):
    api = network()
    parsed = api.parse_request(request(method=method), api.ProxyLimits())
    assert parsed.upstream_head.startswith(method.encode() + b" /a?b=1 ")


@pytest.mark.parametrize("kind", ["connect", "websocket"])
def test_private_tunnels_are_rejected_before_connect(tmp_path, monkeypatch, kind):
    api = proxy()
    monkeypatch.setattr(api, "dial_numeric", lambda *args: pytest.fail("private tunnel dialing"))
    with running(tmp_path) as (path, _, _), client(path) as browser:
        if kind == "connect":
            payload = request("127.0.0.1:443", "127.0.0.1:443", method="CONNECT")
        else:
            payload = request(
                "http://127.0.0.1/",
                "127.0.0.1",
                extra="Connection: Upgrade\r\nUpgrade: websocket\r\n",
            )
        browser.sendall(payload)
        assert receive_all(browser) == api.DENIED_RESPONSE


def test_new_server_connection_rechecks_dns_after_rebinding(tmp_path, monkeypatch):
    api = proxy()
    _, remote, dials = upstream_pair(monkeypatch)
    answers = iter([[answer("8.8.8.8")], [answer("127.0.0.1")]])
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: next(answers))
    with remote, running(tmp_path) as (path, _, _):
        with client(path) as first:
            first.sendall(request())
            assert remote.recv(4096).startswith(b"GET /")
            remote.sendall(
                b"HTTP/1.1 302 Found\r\nLocation: http://audit.example/new\r\n"
                b"Content-Length: 0\r\n\r\n"
            )
            remote.shutdown(socket.SHUT_WR)
            assert b"302 Found" in receive_all(first)
        with client(path) as redirected:
            redirected.sendall(request("http://audit.example/new"))
            assert receive_all(redirected) == api.DENIED_RESPONSE
        assert dials == [("8.8.8.8", 80)]


def test_pipelined_header_prefix_is_rejected_before_connect(tmp_path, monkeypatch):
    api = proxy()
    monkeypatch.setattr(api, "dial_numeric", lambda *args: pytest.fail("pipeline dialing"))
    with running(tmp_path) as (path, _, _), client(path) as browser:
        browser.sendall(request() + request("http://127.0.0.1/", "127.0.0.1"))
        assert receive_all(browser) == api.DENIED_RESPONSE


def test_second_http_request_is_not_forwarded_after_first_header(tmp_path, monkeypatch):
    _, remote, _ = upstream_pair(monkeypatch)
    with remote, running(tmp_path) as (path, _, _), client(path) as browser:
        browser.sendall(request())
        assert remote.recv(4096).startswith(b"GET /")
        browser.sendall(request("http://127.0.0.1/", "127.0.0.1"))
        remote.settimeout(0.05)
        with pytest.raises(TimeoutError):
            remote.recv(4096)
        remote.shutdown(socket.SHUT_WR)
        receive_all(browser)


def test_incomplete_http_body_closes_upstream(tmp_path, monkeypatch):
    upstream, remote, _ = upstream_pair(monkeypatch)
    with remote, running(tmp_path) as (path, _, _), client(path) as browser:
        browser.sendall(request(extra="Content-Length: 20\r\n", method="POST") + b"short")
        browser.shutdown(socket.SHUT_WR)
        assert receive_all(remote).endswith(b"short")
        receive_all(browser)
    assert upstream.fileno() == -1


def test_cleanup_preserves_replacement_file(tmp_path):
    with running(tmp_path, preserved_path=True) as (path, _, _):
        path.unlink()
        path.write_text("replacement")
    assert path.read_text() == "replacement"


def test_numeric_address_fallback_uses_same_dns_result_and_decreasing_timeout(
    tmp_path, monkeypatch
):
    api = proxy()
    upstream, remote = socket.socketpair()
    remote.settimeout(1)
    lookups = []
    dials = []

    def resolve(*args, **kwargs):
        lookups.append(args)
        return [answer("2606:4700:4700::1111"), answer("8.8.8.8")]

    def dial(address, port, timeout):
        dials.append((address, port, timeout))
        if ":" in address:
            time.sleep(0.02)
            raise OSError("network unreachable")
        return upstream

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(api, "dial_numeric", dial)
    with upstream, remote, running(tmp_path) as (path, _, _), client(path) as browser:
        browser.sendall(request("audit.example:443", "audit.example:443", method="CONNECT"))
        assert browser.recv(4096) == api.CONNECTED_RESPONSE
        browser.sendall(b"payload")
        assert remote.recv(4096) == b"payload"
    assert len(lookups) == 1
    assert [item[:2] for item in dials] == [("2606:4700:4700::1111", 443), ("8.8.8.8", 443)]
    assert dials[1][2] < dials[0][2]


def test_all_validated_addresses_fail_with_one_fixed_response(tmp_path, monkeypatch):
    api = proxy()
    dials = []
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [answer("2606:4700:4700::1111"), answer("8.8.8.8")],
    )

    def dial(address, port, timeout):
        dials.append(address)
        raise OSError("secret network metadata")

    monkeypatch.setattr(api, "dial_numeric", dial)
    with running(tmp_path) as (path, _, _), client(path) as browser:
        browser.sendall(request())
        assert receive_all(browser) == api.FAILED_RESPONSE
    assert dials == ["2606:4700:4700::1111", "8.8.8.8"]


def test_connection_deadline_prevents_next_numeric_dial(tmp_path, monkeypatch):
    api = proxy()
    dials = []
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [answer("2606:4700:4700::1111"), answer("8.8.8.8")],
    )

    def dial(address, port, timeout):
        dials.append(address)
        time.sleep(0.08)
        raise OSError("network unreachable")

    monkeypatch.setattr(api, "dial_numeric", dial)
    with running(tmp_path, connection_timeout=0.04) as (path, _, _), client(path) as browser:
        browser.sendall(request())
        assert receive_all(browser) == api.LIMIT_RESPONSE
    assert dials == ["2606:4700:4700::1111"]
