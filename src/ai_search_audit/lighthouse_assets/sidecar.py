"""Only entrypoint of the trusted networked container; never runs browser code."""

import os
import sys

# Image contains only these stdlib-only modules in this package.
sys.path.insert(0, "/opt/audit-lighthouse")
from ai_search_audit.lighthouse_network import ProxyLimits  # noqa: E402
from ai_search_audit.lighthouse_proxy import serve_unix  # noqa: E402

if __name__ == "__main__":
    os.umask(0o077)
    try:
        serve_unix("/run/audit-proxy/proxy.sock", ProxyLimits(wall_timeout=180.0))
    except (OSError, ValueError):
        # Never emit targets, exception details, headers, bodies or credentials.
        sys.exit(1)
