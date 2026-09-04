"""Bounded, per-run AF_UNIX server; importable by the trusted networked sidecar.

The caller must own the socket directory (not writable by browser processes). Threads
are daemonized because libc DNS cannot be cancelled. Shutdown closes owned sockets and
returns without waiting for DNS; the outer sidecar process deadline terminates libc too.
"""

from __future__ import annotations

import os
import select
import socket
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .lighthouse_network import (
    ProxyLimits,
    ProxyPolicyError,
    ProxyRequest,
    dial_numeric,
    parse_request,
    resolve_public,
)

DENIED_RESPONSE = b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
FAILED_RESPONSE = b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
LIMIT_RESPONSE = (
    b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
)
CONNECTED_RESPONSE = b"HTTP/1.1 200 Connection Established\r\n\r\n"


@dataclass(frozen=True)
class ProxyStats:
    """Aggregate counters only; never URL, header, body or exception content."""

    accepted_requests: int
    policy_denials: int
    upstream_failures: int
    limit_denials: int
    transferred_bytes: int


def _close(connection: socket.socket) -> None:
    try:
        connection.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    connection.close()


class _Run:
    def __init__(self, limits: ProxyLimits, stop: threading.Event) -> None:
        self.limits = limits
        self.stop = stop
        self.deadline = time.monotonic() + limits.wall_timeout
        self.lock = threading.Lock()
        self.sockets: set[socket.socket] = set()
        self.threads: set[threading.Thread] = set()
        self.accepted = 0
        self.denied = 0
        self.failed = 0
        self.limited = 0
        self.transferred = 0

    def remaining(self, deadline: float) -> float:
        remaining = min(self.deadline, deadline) - time.monotonic()
        if self.stop.is_set() or remaining <= 0:
            raise TimeoutError()
        return remaining

    def charge(self, count: int) -> None:
        with self.lock:
            if self.transferred + count > self.limits.max_transfer_bytes:
                self.stop.set()
                raise TimeoutError()
            self.transferred += count

    def send(self, connection: socket.socket, data: bytes, deadline: float) -> None:
        connection.settimeout(self.remaining(deadline))
        self.charge(len(data))
        connection.sendall(data)

    def error(self, connection: socket.socket, response: bytes) -> None:
        # Fixed response, short timeout. Error bytes are excluded from stream budget.
        try:
            connection.settimeout(0.02)
            connection.sendall(response)
        except OSError:
            pass

    def read_head(self, connection: socket.socket, deadline: float) -> tuple[bytes, bytes]:
        buffer = bytearray()
        while True:
            connection.settimeout(self.remaining(deadline))
            chunk = connection.recv(min(16384, self.limits.max_header_bytes - len(buffer)))
            if not chunk:
                raise ProxyPolicyError()
            buffer.extend(chunk)
            end = buffer.find(b"\r\n\r\n")
            if end >= 0:
                return bytes(buffer[: end + 4]), bytes(buffer[end + 4 :])
            if len(buffer) >= self.limits.max_header_bytes:
                raise ProxyPolicyError()

    def stream(
        self,
        client: socket.socket,
        upstream: socket.socket,
        parsed: ProxyRequest,
        prefix: bytes,
        deadline: float,
    ) -> None:
        tunnel = parsed.connect or parsed.upgrade
        remaining_body = parsed.content_length
        if prefix:
            self.send(upstream, prefix, deadline)
            remaining_body -= len(prefix)
        readers = {upstream}
        if tunnel or remaining_body:
            readers.add(client)
        while readers:
            ready, _, _ = select.select(list(readers), [], [], self.remaining(deadline))
            if not ready:
                raise TimeoutError()
            for source in ready:
                destination = upstream if source is client else client
                size = 16384 if tunnel or source is upstream else min(16384, remaining_body)
                source.settimeout(self.remaining(deadline))
                chunk = source.recv(size)
                if not chunk:
                    readers.remove(source)
                    if not tunnel:
                        # No second HTTP request, even if the client pipelines one.
                        if source is upstream or remaining_body:
                            return
                    try:
                        destination.shutdown(socket.SHUT_WR)
                    except OSError:
                        pass
                    continue
                self.send(destination, chunk, deadline)
                if source is client and not tunnel:
                    remaining_body -= len(chunk)
                    if remaining_body == 0:
                        readers.remove(client)

    def handle(self, client: socket.socket) -> None:
        deadline = min(self.deadline, time.monotonic() + self.limits.connection_timeout)
        upstream: socket.socket | None = None
        response_started = False
        try:
            head, prefix = self.read_head(client, deadline)
            parsed = parse_request(head, self.limits)
            if not (parsed.connect or parsed.upgrade) and len(prefix) > parsed.content_length:
                raise ProxyPolicyError()
            addresses = resolve_public(parsed.host, parsed.port)
            # DNS is uninterruptible: never dial after it returns past the deadline.
            # Family fallback uses this one validated answer set and the same deadline.
            for address in addresses:
                timeout = self.remaining(deadline)
                try:
                    upstream = dial_numeric(address, parsed.port, timeout)
                except OSError:
                    self.remaining(deadline)
                    continue
                break
            else:
                raise OSError("proxy_upstream_unavailable")
            with self.lock:
                self.remaining(deadline)
                self.sockets.add(upstream)
            if parsed.connect:
                self.send(client, CONNECTED_RESPONSE, deadline)
            else:
                self.send(upstream, parsed.upstream_head, deadline)
            response_started = True
            self.stream(client, upstream, parsed, prefix, deadline)
        except ProxyPolicyError:
            with self.lock:
                self.denied += 1
            if not response_started:
                self.error(client, DENIED_RESPONSE)
        except TimeoutError:
            with self.lock:
                self.limited += 1
            if not response_started:
                self.error(client, LIMIT_RESPONSE)
        except (OSError, ValueError):
            with self.lock:
                self.failed += 1
            if not response_started:
                self.error(client, FAILED_RESPONSE)
        finally:
            if upstream is not None:
                _close(upstream)
            _close(client)
            with self.lock:
                self.sockets.discard(client)
                if upstream is not None:
                    self.sockets.discard(upstream)
                self.threads.discard(threading.current_thread())

    def shutdown(self) -> None:
        self.stop.set()
        with self.lock:
            owned = tuple(self.sockets)
            threads = tuple(self.threads)
        for connection in owned:
            _close(connection)
        # One shared grace period, not a timeout multiplied by thread count.
        deadline = time.monotonic() + 0.1
        for worker in threads:
            worker.join(max(0, deadline - time.monotonic()))

    def stats(self) -> ProxyStats:
        with self.lock:
            return ProxyStats(
                self.accepted, self.denied, self.failed, self.limited, self.transferred
            )


def _directory(path: Path) -> int:
    """Walk without following symlinks and keep an fd for inode-checked cleanup."""
    if not path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError("invalid_proxy_socket_path")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parent.parts[1:]:
            child = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
            raise ValueError("unsafe_proxy_socket_directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def serve_unix(
    socket_path: str | Path, limits: ProxyLimits, *, stop: threading.Event | None = None
) -> ProxyStats:
    """Serve until stopped or budget/deadline exhausted; never replace an existing path.

    DNS worker threads are bounded but may outlive this function; see module caveat.
    The separate browser must have kernel-enforced no-network and only a local relay.
    """
    limits = ProxyLimits(
        **{name: getattr(limits, name) for name in ProxyLimits.__dataclass_fields__}
    )
    path = Path(socket_path)
    directory = _directory(path)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    identity: tuple[int, int] | None = None
    run = _Run(limits, stop if stop is not None else threading.Event())
    try:
        try:
            os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("proxy_socket_path_exists")
        listener.bind(str(path))
        entry = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        identity = (entry.st_dev, entry.st_ino)
        os.chmod(path.name, 0o600, dir_fd=directory, follow_symlinks=False)
        listener.listen(limits.max_connections)
        while not run.stop.is_set() and time.monotonic() < run.deadline:
            with run.lock:
                active = len(run.threads)
            if run.accepted >= limits.max_requests:
                if not active:
                    break
                run.stop.wait(min(0.02, max(0, run.deadline - time.monotonic())))
                continue
            listener.settimeout(min(0.02, max(0.0001, run.deadline - time.monotonic())))
            try:
                client, _ = listener.accept()
            except TimeoutError:
                continue
            run.accepted += 1
            with run.lock:
                full = len(run.threads) >= limits.max_connections
            if full:
                run.limited += 1
                run.error(client, LIMIT_RESPONSE)
                _close(client)
                continue
            worker = threading.Thread(target=run.handle, args=(client,), daemon=True)
            with run.lock:
                run.sockets.add(client)
                run.threads.add(worker)
            try:
                worker.start()
            except BaseException:
                with run.lock:
                    run.threads.discard(worker)
                raise
    finally:
        listener.close()
        run.shutdown()
        try:
            if identity is not None:
                entry = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
                if stat.S_ISSOCK(entry.st_mode) and (entry.st_dev, entry.st_ino) == identity:
                    os.unlink(path.name, dir_fd=directory)
        except FileNotFoundError:
            pass
        finally:
            os.close(directory)
    return run.stats()
