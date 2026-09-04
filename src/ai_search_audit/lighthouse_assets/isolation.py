"""Fixed, pre-navigation Linux kernel probes. No URLs, output files or logs."""

import errno
import json
import os
import re
import socket
from pathlib import Path


def interface_flags() -> dict[str, int]:
    """Enumerate actual namespace interfaces, not unrelated sysfs entries."""
    interfaces = {}
    for _, name in socket.if_nameindex():
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,15}", name) or name in {".", ".."}:
            raise ValueError("invalid_interface_name")
        if name in interfaces:
            raise ValueError("duplicate_interface")
        raw = (Path("/sys/class/net") / name / "flags").read_text().strip()
        if not re.fullmatch(r"0x[0-9a-fA-F]{1,8}", raw):
            raise ValueError("invalid_interface_flags")
        interfaces[name] = int(raw, 16)
    return interfaces


def status_isolated(status: str, interfaces: dict[str, int], routes: str) -> bool:
    fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
    return (
        fields.get("CapEff", "").strip() == "0000000000000000"
        and fields.get("NoNewPrivs", "").strip() == "1"
        and fields.get("Seccomp", "").strip() == "2"
        and bool(interfaces)
        and all(type(flags) is int and 0 <= flags <= 0xFFFFFFFF for flags in interfaces.values())
        and interfaces.get("lo", 0) & 0x9 == 0x9  # IFF_UP | IFF_LOOPBACK
        and not any(flags & 0x1 for name, flags in interfaces.items() if name != "lo")
        and len(routes.strip().splitlines()) <= 1
    )


def verify_network() -> bool:
    if os.getuid() != 1000 or not status_isolated(
        Path("/proc/self/status").read_text(),
        interface_flags(),
        Path("/proc/net/route").read_text(),
    ):
        return False
    # A socket family denied by the outer seccomp profile must fail at creation.
    for family in (40, 38):  # Linux AF_VSOCK, AF_ALG
        try:
            with socket.socket(family, socket.SOCK_STREAM):
                return False
        except OSError as error:
            if error.errno != errno.EPERM:
                return False
    # Positive connection or timeout is not isolation evidence. Require immediate
    # kernel no-route errors, both TCP and UDP, with no name resolution.
    for family, target in (
        (socket.AF_INET, "1.1.1.1"),
        (socket.AF_INET, "10.0.0.1"),
        (socket.AF_INET, "169.254.169.254"),
        (socket.AF_INET6, "2606:4700:4700::1111"),
        (socket.AF_INET6, "fd00::1"),
        (socket.AF_INET6, "fe80::1"),
    ):
        for kind in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
            try:
                with socket.socket(family, kind) as probe:
                    probe.settimeout(0.2)
                    probe.connect((target, 443))
                return False
            except OSError as error:
                if error.errno not in {errno.ENETUNREACH, errno.EHOSTUNREACH, errno.EINVAL}:
                    return False
    return True


if __name__ == "__main__":
    try:
        verified = verify_network()
    except (OSError, ValueError):
        verified = False
    print(json.dumps({"network_isolated": verified}))
