# Local Lighthouse runtime: operator setup

This image is an explicit, separate prerequisite. Collection never builds, pulls,
installs packages or accepts mutable image tags. A missing or mismatched runtime
returns UNAVAILABLE. Do not disable Chromium's sandbox to obtain a measurement.

## Build and register

Use the repository root as the Docker build context. Review this Dockerfile,
the dependency lock and the security assets first. A build downloads the pinned
Node base, public Debian packages and npm packages. No client page is visited.

On native Linux use the local Unix Docker socket. On macOS use Docker Desktop's
local Unix socket. A remote Docker endpoint is not supported. Build natively:
do not use CPU emulation. The launcher checks host, daemon and image architecture.

BuildKit/buildx is required: the Dockerfile-specific context allowlist is not
enforced by the legacy builder. An empty Docker configuration can hide the
buildx plugin from ordinary `docker build` discovery and cause a legacy fallback.
Do not use that fallback. Invoke the installed, trusted buildx executable by its
fixed absolute path and explicitly select the local default builder.

Use a fresh Docker CLI configuration directory whose `config.json` contains only
an empty JSON object, and explicitly select the local socket. Do not change HOME
or the user's global Docker configuration. On Docker Desktop for macOS:

```text
env -i PATH=/usr/local/bin:/usr/bin:/bin DOCKER_HOST=unix://<local-docker-socket> DOCKER_CONFIG=<fresh-empty-config-directory> /Applications/Docker.app/Contents/Resources/cli-plugins/docker-buildx --builder default build --load --pull=false --progress=plain -f src/ai_search_audit/lighthouse_assets/Dockerfile -t audit-lighthouse-local:1 .
docker --host unix://<local-docker-socket> --config <fresh-empty-config-directory> image inspect --format '{{.Id}}' audit-lighthouse-local:1
```

On Linux, use the absolute path of the reviewed buildx executable from the trusted
Docker installation with the same arguments and clean environment. If buildx or
the local default builder is unavailable, stop setup and install/configure that
prerequisite separately. Do not substitute a remote builder or the legacy builder.
The plain-progress output must show the small allowlisted context transfer, not
the whole checkout; stop an unexpectedly broad transfer before continuing.

Give that exact local `sha256:...` identity to `LighthouseRuntimeConfig(image_id=...)`.
The tag is only a setup convenience; it is not accepted by collection. Builds
fail if a pinned Debian package is no longer available. Review and explicitly
update the runtime pins instead of silently substituting a new browser.

The image labels pin the packaged runner, kernel checker, trusted sidecar,
network/proxy modules, dependency lock and seccomp profile. Source changes require
corresponding Dockerfile label changes and a new image build/identity. The offline
asset test rejects stale labels.

## Boundary and prerequisites

Two fresh containers share one freshly named Unix-socket volume. The networked
sidecar runs only the stdlib Python proxy. The networkless browser runs as UID
1000 with no capabilities, no-new-privileges, read-only root, bounded tmpfs,
private IPC, bounded processes/memory/CPU, no host mounts and no daemon socket.
Its socket-volume mount is read-only; the socket directory is UID 1000 mode 0700,
and the proxy socket is mode 0600. CDP uses pipes, not a listening debugging port.
The Docker daemon and operator-selected image are part of the trusted local base.

Before client navigation the runner checks kernel capabilities, no-new-privileges,
seccomp, network interfaces/routes, IPv4/IPv6 TCP/UDP no-route behavior and denied
AF_VSOCK/AF_ALG creation. Chromium's own sandbox page must confirm namespace,
PID/network namespace and seccomp-BPF sandbox protection. A self-check failure
returns UNAVAILABLE, not a judgment about the audited page.

Interface checks enumerate actual namespace interfaces, require loopback to be UP
with the loopback flag, and reject every other UP interface. Built-in DOWN tunnel
devices are permitted only alongside all the other kernel/route/protocol checks.
Missing or malformed interface flags fail closed; unrelated sysfs entries such
as `bonding_masters` are not treated as interfaces.

Only HTTP port 80 and HTTPS/WSS CONNECT port 443 are supported. Chunked request
bodies and Chromium's plaintext WebSocket CONNECT-to-port-80 behavior are not
supported; fixed capability warnings are retained in local measurements.

The parent bounds both process streams and wall time, kills its owned process
group, and removes only containers/volumes bearing this attempt's ownership label.
A daemon outage can prevent cleanup; such a run returns FAILED/cleanup_failed
with no measurement. Resource names start with `audit-lh-`; operator recovery
must inspect the ownership label before deletion. Do not remove unrelated data.

## Provenance and acceptance

The Node base is pinned by digest in Dockerfile. npm dependencies are exact
Lighthouse 13.4.1 and puppeteer-core 25.10.0, with the full generated npm lock and
integrity hashes. Chromium/common are explicitly pinned to
`152.0.7977.75-1~deb12u1`, and Python to `3.11.2-1+b1`. Installation uses the signed
Debian repositories and fails if those exact versions are unavailable; no automatic
version substitution is permitted. Final installed image identity also binds all
transitive system packages. A browser pin update requires fresh runtime acceptance.

`seccomp.json` is derived from
[moby/profiles commit 61eaf32614c7c71b60bd8927d3e6a4ffc8ff1f31](https://github.com/moby/profiles/tree/61eaf32614c7c71b60bd8927d3e6a4ffc8ff1f31):
the upstream `seccomp/default.json` plus one allow entry for `clone`, `setns`,
`unshare` and `chroot`. The outer default-deny, AF_ALG/AF_VSOCK denials and
clone3 ENOSYS behavior without CAP_SYS_ADMIN remain intact. The Apache-2.0 license
is retained in `LICENSE.moby`. The profile's content SHA256 is in the image label
and every local measurement fingerprint.

Offline tests establish projection, process limits and the fixed launcher
contract. They do not establish actual browser isolation. Hostile redirects,
subresources, workers, WebSockets, DNS rebinding, proxy-off fallback denial,
timeout cleanup and a public positive control require the separate explicit
integration acceptance step.

The fingerprint captures image, packages, architecture and runtime policy, not
identical physical CPU performance or host load. Results remain sampled lab
measurements; matching fingerprints alone do not prove hardware equivalence or
controlled numeric comparability.
