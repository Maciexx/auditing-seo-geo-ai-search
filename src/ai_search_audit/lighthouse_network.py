"""Strict HTTP proxy policy and numeric-only egress for the Lighthouse sidecar.

No browser code runs here. No environment proxy settings or audit credentials are used.
Chunked request bodies are deliberately unsupported; this is a runtime capability limit,
not evidence of a broken audited page. Each ordinary HTTP connection carries one request.
Chromium's plaintext WebSocket CONNECT-to-port-80 behavior is unsupported by the fixed
CONNECT-443 policy; WSS remains supported. Do not widen that policy to obtain an audit result.
"""

from __future__ import annotations

import ipaddress
import math
import re
import socket
from dataclasses import dataclass, field
from urllib.parse import urlsplit


class ProxyPolicyError(ValueError):
    """Fixed, data-free policy failure safe for an untrusted client to receive."""

    def __init__(self) -> None:
        super().__init__("proxy_policy_denied")


@dataclass(frozen=True)
class ProxyLimits:
    max_header_bytes: int = 32768
    max_connections: int = 32
    max_requests: int = 2000
    max_transfer_bytes: int = 134217728
    connection_timeout: float = 30.0
    wall_timeout: float = 120.0

    def __post_init__(self) -> None:
        for value, maximum in (
            (self.max_header_bytes, 1000000),
            (self.max_connections, 128),
            (self.max_requests, 100000),
            (self.max_transfer_bytes, 1073741824),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("invalid_proxy_limits")
        for duration, maximum in ((self.connection_timeout, 300), (self.wall_timeout, 900)):
            if (
                type(duration) not in (int, float)
                or not math.isfinite(duration)
                or not 0 < duration <= maximum
            ):
                raise ValueError("invalid_proxy_limits")


def public_ip(value: str) -> str:
    """Require canonical IP syntax and globally routable, non-transition addresses."""
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        raise ProxyPolicyError() from None
    if "%" in value or not address.is_global or address.is_multicast or address.is_reserved:
        raise ProxyPolicyError()
    if isinstance(address, ipaddress.IPv6Address) and (
        address.ipv4_mapped is not None
        or address.sixtofour is not None
        or address.teredo is not None
    ):
        raise ProxyPolicyError()
    return str(address)


def _host(value: str) -> str:
    if not value or not value.isascii():
        raise ProxyPolicyError()
    try:
        ipaddress.ip_address(value)
    except ValueError:
        pass
    else:
        return public_ip(value)
    hostname = value.lower()
    if hostname.endswith("."):
        hostname = hostname[:-1]
    labels = hostname.split(".")
    if (
        len(hostname) > 253
        or hostname == "localhost"
        or hostname.endswith(".localhost")
        or any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", x) for x in labels)
        # WHATWG numeric aliases, including decimal, octal and hexadecimal forms.
        or re.fullmatch(r"(?:[0-9]+|0x[0-9a-f]+)", labels[-1])
    ):
        raise ProxyPolicyError()
    return hostname


def parse_authority(value: str, port: int, *, explicit_port: bool = False) -> tuple[str, int]:
    """No userinfo, escaped delimiters, scoped IPs or nonstandard ports."""
    if not value or any(c in value for c in "@/%\\?#"):
        raise ProxyPolicyError()
    if value.startswith("["):
        match = re.fullmatch(r"\[([0-9a-fA-F:]+)\](?::([0-9]+))?", value)
        if match is None or ":" not in match[1]:
            raise ProxyPolicyError()
        host, supplied = public_ip(match[1]), match[2]
    else:
        parts = value.split(":")
        if len(parts) > 2:
            raise ProxyPolicyError()
        host, supplied = _host(parts[0]), parts[1] if len(parts) == 2 else None
    if (explicit_port and supplied is None) or (supplied is not None and supplied != str(port)):
        raise ProxyPolicyError()
    return host, port


@dataclass(frozen=True)
class ProxyRequest:
    host: str
    port: int
    connect: bool
    upgrade: bool
    content_length: int
    upstream_head: bytes = field(repr=False)


_TOKEN = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
_METHODS = frozenset({"GET", "HEAD", "POST", "OPTIONS", "PUT", "PATCH", "DELETE"})


def parse_request(raw: bytes, limits: ProxyLimits) -> ProxyRequest:
    """Parse one complete header block; never retain/log full URLs or client headers."""
    if len(raw) > limits.max_header_bytes or not raw.endswith(b"\r\n\r\n"):
        raise ProxyPolicyError()
    try:
        lines = raw[:-4].decode("ascii").split("\r\n")
    except UnicodeDecodeError:
        raise ProxyPolicyError() from None
    if any(any(ord(c) < 32 or ord(c) == 127 for c in line) for line in lines):
        raise ProxyPolicyError()
    start = lines[0].split(" ")
    if len(start) != 3 or start[2] not in ("HTTP/1.0", "HTTP/1.1"):
        raise ProxyPolicyError()
    method, target, version = start
    connect = method == "CONNECT"
    if not connect and method not in _METHODS:
        raise ProxyPolicyError()
    if connect:
        host, port = parse_authority(target, 443, explicit_port=True)
        path = ""
    else:
        try:
            url = urlsplit(target)
        except ValueError:
            raise ProxyPolicyError() from None
        if url.scheme != "http" or not target.startswith("http://") or "#" in target:
            raise ProxyPolicyError()
        host, port = parse_authority(url.netloc, 80)
        path = url.path or "/"
        if url.query or "?" in target:
            path += "?" + url.query
    headers: list[tuple[str, str]] = []
    hosts: list[str] = []
    lengths: list[str] = []
    connection: list[str] = []
    upgrades: list[str] = []
    for line in lines[1:]:
        name, separator, value = line.partition(":")
        if not separator or _TOKEN.fullmatch(name) is None:
            raise ProxyPolicyError()
        value = value.strip(" ")
        lower = name.lower()
        if lower in {"proxy-authorization", "transfer-encoding"}:
            raise ProxyPolicyError()
        if lower == "host":
            hosts.append(value)
        elif lower == "content-length":
            lengths.append(value)
        elif lower == "connection":
            connection.extend(part.strip().lower() for part in value.split(","))
        elif lower == "upgrade":
            upgrades.append(value.lower())
        headers.append((name, value))
    if len(hosts) != 1 or parse_authority(hosts[0], port) != (host, port):
        raise ProxyPolicyError()
    if len(lengths) > 1 or (lengths and re.fullmatch(r"[0-9]{1,10}", lengths[0]) is None):
        raise ProxyPolicyError()
    length = int(lengths[0]) if lengths else 0
    if length > limits.max_transfer_bytes or (connect and length):
        raise ProxyPolicyError()
    if any(token not in {"close", "keep-alive", "upgrade"} for token in connection):
        raise ProxyPolicyError()
    upgrade = upgrades == ["websocket"] and "upgrade" in connection and method == "GET"
    if (upgrades or "upgrade" in connection) and (not upgrade or length):
        raise ProxyPolicyError()
    forward = [
        (name, value)
        for name, value in headers
        if name.lower()
        not in {
            "connection",
            "proxy-connection",
            "keep-alive",
            "upgrade",
        }
    ]
    forward.append(("Connection", "Upgrade" if upgrade else "close"))
    if upgrade:
        forward.append(("Upgrade", "websocket"))
    head = f"{method} {path} {version}\r\n" + "".join(f"{n}: {v}\r\n" for n, v in forward) + "\r\n"
    return ProxyRequest(
        host, port, connect, upgrade, length, b"" if connect else head.encode("ascii")
    )


def resolve_public(host: str, port: int) -> tuple[str, ...]:
    """Fresh resolution per upstream; a single unsafe answer poisons the entire set."""
    host = _host(host)
    if port not in (80, 443):
        raise ProxyPolicyError()
    try:
        ipaddress.ip_address(host)
    except ValueError:
        records = socket.getaddrinfo(
            host, port, socket.AF_UNSPEC, socket.SOCK_STREAM, socket.IPPROTO_TCP
        )
        validated = []
        for record in records:
            address = record[4][0]
            if not isinstance(address, str):
                raise ProxyPolicyError() from None
            validated.append(public_ip(address))
        addresses = tuple(dict.fromkeys(validated))
        if not addresses:
            raise ProxyPolicyError() from None
        return addresses
    return (public_ip(host),)


def dial_numeric(address: str, port: int, timeout: float) -> socket.socket:
    """socket.connect receives only a validated numeric address, never a DNS name."""
    address = public_ip(address)
    if port not in (80, 443):
        raise ProxyPolicyError()
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    upstream = socket.socket(family, socket.SOCK_STREAM)
    try:
        upstream.settimeout(timeout)
        upstream.connect((address, port))
        return upstream
    except BaseException:
        upstream.close()
        raise
