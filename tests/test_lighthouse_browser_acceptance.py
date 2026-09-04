"""Explicit operator-only Chromium acceptance; never part of offline CI by default.

Run with AUDIT_LIGHTHOUSE_BROWSER_ACCEPTANCE=1 and AUDIT_LIGHTHOUSE_IMAGE_ID set
to a locally built immutable sha256 image ID. No image pulls/builds or public calls.
The real production launcher, sandbox, proxy parser and public-IP policy execute.
Only DNS answers and upstream HTTP transport are controlled .example fixtures:
socketpair transport does NOT prove actual public-IP numeric dialing or TLS/WSS
success. WSS private denial is tested; Chrome plaintext WS CONNECT:80 remains
unsupported. The separate public provider trial covers genuine public transport.
"""

import json
import os
import re
import tempfile
import time

import pytest

from ai_search_audit import lighthouse_runtime as runtime
from ai_search_audit.performance_models import LighthouseRequest

pytestmark = pytest.mark.skipif(
    os.environ.get("AUDIT_LIGHTHOUSE_BROWSER_ACCEPTANCE") != "1",
    reason="explicit opt-in required for owned Docker/Chromium integration",
)


@pytest.fixture
def immutable_image():
    image = os.environ.get("AUDIT_LIGHTHOUSE_IMAGE_ID", "")
    assert re.fullmatch(r"sha256:[a-f0-9]{64}", image), "immutable local image ID required"
    return image


SIDECAR = r"""
import json, os, socket, sys, threading
sys.path.insert(0, '/opt/audit-lighthouse')
from ai_search_audit import lighthouse_network as network
from ai_search_audit import lighthouse_proxy as proxy
os.umask(0o077)
lock = threading.Lock()
counts = {}
def event(*values):
    with lock:
        with open('/run/audit-proxy/events.jsonl', 'a') as stream:
            stream.write(json.dumps(values) + '\n')
def dns(host, port, *args):
    # No fallback to system DNS or external fixture domains.
    if not host.endswith('.example'):
        raise network.ProxyPolicyError()
    with lock:
        counts[host] = counts.get(host, 0) + 1
        count = counts[host]
    private = host.endswith('-private.example') or (host == 'rebind.example' and count > 1)
    address = '127.0.0.1' if private else '8.8.8.8'
    event('dns', host, address)
    addresses = [address, '127.0.0.1'] if host == 'mixed.example' else [address]
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', (a, port))
            for a in addresses]
socket.getaddrinfo = dns
original_resolve = proxy.resolve_public
def resolve(host, port):
    try:
        return original_resolve(host, port)
    except network.ProxyPolicyError:
        event('denied', host)
        raise
proxy.resolve_public = resolve
def respond(connection):
    try:
        connection.settimeout(5)
        data = b''
        while b'\r\n\r\n' not in data and len(data) < 32768:
            chunk = connection.recv(4096)
            if not chunk:
                return
            data += chunk
        path = data.split(b' ', 2)[1]
        if path == b'/redirect':
            connection.sendall(b'HTTP/1.1 302 Found\r\n'
                b'Location: http://redirect-private.example/\r\n'
                b'Content-Length: 0\r\nConnection: close\r\n\r\n')
            return
        body = b'<html><head><title>Fixture</title></head><body>fixture-ok</body></html>'
        connection.sendall(b'HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n'
            b'Access-Control-Allow-Origin: *\r\nCache-Control: no-store\r\n'
            b'Connection: close\r\nContent-Length: ' + str(len(body)).encode()
            + b'\r\n\r\n' + body)
    except OSError:
        pass
    finally:
        connection.close()
def dial(address, port, timeout):
    # Test transport only: production validation executes BEFORE this function.
    # No private address is allowed even if a regression accidentally reaches here.
    event('dial', address, port)
    network.public_ip(address)
    left, right = socket.socketpair()
    threading.Thread(target=respond, args=(right,), daemon=True).start()
    return left
proxy.dial_numeric = dial
proxy.serve_unix('/run/audit-proxy/proxy.sock', network.ProxyLimits(wall_timeout=120))
"""

BROWSER = r"""
import fs from 'node:fs/promises';
import net from 'node:net';
const {default: puppeteer} = await import(
  '/opt/audit-lighthouse/node_modules/puppeteer-core/lib/puppeteer/puppeteer-core.js');
import {launchBrowser, launchOptions, verifyIsolation} from '/opt/audit-lighthouse/runner.mjs';
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const socket = '/run/audit-proxy/proxy.sock';
for (let i = 0; ; i++) {
  try { if ((await fs.stat(socket)).isSocket()) break; } catch {}
  if (i === 100) throw Error('fixture_not_ready');
  await pause(50);
}
const connections = new Set();
const relay = net.createServer(client => {
  const upstream = net.connect(socket);
  connections.add(client); connections.add(upstream);
  const close = () => {
    client.destroy(); upstream.destroy(); connections.delete(client); connections.delete(upstream);
  };
  client.on('error', close); upstream.on('error', close);
  client.on('close', close); upstream.on('close', close);
  client.setTimeout(10000, close); upstream.setTimeout(10000, close);
  client.pipe(upstream); upstream.pipe(client);
});
await new Promise(resolve => relay.listen(0, '127.0.0.1', resolve));
let browser, direct;
const profiles = [];
const profile = async () => {
  const path = await fs.mkdtemp('/tmp/acceptance-'); profiles.push(path); return path;
};
try {
  browser = await launchBrowser(puppeteer, await profile(), relay.address().port);
  await verifyIsolation(browser);
  // Marker proves the timeout happened AFTER actual Chrome sandbox verification.
  await fs.writeFile('/tmp/acceptance-isolation-verified', 'yes');
  if (TIMEOUT_MODE) await pause(120000);
  const page = await browser.newPage();
  await page.goto('http://fixture.example/', {timeout: 10000});
  const evidence = {positive: await page.evaluate(() => document.body.textContent)};
  const fetchText = host => page.evaluate(async host => {
    try {
      const response = await fetch(`http://${host}/probe`, {cache: 'no-store'});
      return (await response.text()).includes('fixture-ok') ? 'fixture-ok' : response.status;
    } catch { return 'error'; }
  }, host);
  evidence.rebind_first = await fetchText('rebind.example');
  evidence.rebind_second = await fetchText('rebind.example');
  evidence.mixed = await fetchText('mixed.example');
  evidence.subresource = await page.evaluate(() => new Promise(resolve => {
    const image = new Image(); image.onload = () => resolve('loaded');
    image.onerror = () => resolve('error'); image.src = 'http://subresource-private.example/pixel';
    document.body.append(image);
  }));
  evidence.worker = await page.evaluate(() => new Promise(resolve => {
    const blob = new Blob([`fetch('http://worker-private.example/probe')
      .then(r => postMessage(r.status)).catch(() => postMessage('error'))`],
      {type: 'application/javascript'});
    const url = URL.createObjectURL(blob), worker = new Worker(url);
    const close = () => { worker.terminate(); URL.revokeObjectURL(url); };
    worker.onmessage = event => { resolve(event.data); close(); };
    worker.onerror = () => { resolve('worker-error'); close(); };
  }));
  evidence.wss = await page.evaluate(() => new Promise(resolve => {
    const ws = new WebSocket('wss://wss-private.example/socket');
    ws.onopen = () => { ws.close(); resolve('opened'); };
    ws.onerror = () => resolve('error');
  }));
  const redirected = await browser.newPage();
  evidence.redirect = (await redirected.goto('http://fixture.example/redirect',
    {timeout: 10000})).status();
  await redirected.close();
  await browser.close(); browser = null;
  // Deliberate test-only proxy removal, with every sandbox/network flag retained.
  const options = launchOptions(await profile(), relay.address().port);
  options.args = options.args.filter(arg => !arg.startsWith('--proxy-server='));
  direct = await puppeteer.launch(options);
  await verifyIsolation(direct);
  const directPage = await direct.newPage();
  try {
    await directPage.goto('http://8.8.8.8/', {timeout: 3000}); evidence.direct = 'reachable';
  } catch { evidence.direct = 'denied'; }
  evidence.events = (await fs.readFile('/run/audit-proxy/events.jsonl', 'utf8'))
    .trim().split('\n').map(line => JSON.parse(line));
  const version = async name => JSON.parse(await fs.readFile(
    `/opt/audit-lighthouse/node_modules/${name}/package.json`, 'utf8')).version;
  console.log(JSON.stringify({status: 'ok', isolation: true, versions: {
    node: process.versions.node, chrome: (await direct.version()).split('/')[1],
    lighthouse: await version('lighthouse'), puppeteer: await version('puppeteer-core')},
    lhr: evidence}));
} finally {
  if (browser) await browser.close(); if (direct) await direct.close();
  for (const connection of connections) connection.destroy(); relay.close();
  for (const path of profiles) await fs.rm(path, {recursive: true, force: true});
}
"""


def _acceptance(monkeypatch, image, *, timeout=False):
    original = runtime._run_process
    owners = set()
    verified_timeout = []

    def adapt(argv, **kwargs):
        assert argv[:4] == (
            runtime._docker_binary(),
            "--host",
            "unix://" + str(runtime._docker_socket()),
            "--config",
        )
        if argv[5] == "create":
            for required in (
                "--pull=never",
                "--init",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--read-only",
                "--user=1000:1000",
                "--pids-limit=256",
                "--memory=1g",
                "--cpus=2",
                "--ipc=private",
                "--tmpfs=/tmp:rw,noexec,nosuid,size=512m",
            ):
                assert required in argv
            owners.add(argv[argv.index("--label") + 1])
            assert not any("type=bind" in argument for argument in argv)
            assert not any("unconfined" in argument for argument in argv)
            assert not any(argument.startswith("--cap-add") for argument in argv)
            if argv[-1] == runtime.CONTAINER_ROOT + "/sidecar.py":
                assert argv[-2] == image
                assert argv[argv.index("--entrypoint") + 1] == "/usr/bin/python3"
                assert "--network=bridge" in argv
                assert any(
                    argument.startswith("--mount=type=volume,")
                    and argument.endswith(",dst=/run/audit-proxy")
                    for argument in argv
                )
                argv = argv[:-1] + ("-c", SIDECAR)
            else:
                assert argv[-1] == runtime.CONTAINER_ROOT + "/runner.mjs"
                assert "--network=none" in argv
                assert "--security-opt=seccomp=" + str(runtime.ASSETS / "seccomp.json") in argv
                assert argv[-2] == image
                assert argv[argv.index("--entrypoint") + 1] == "/usr/local/bin/node"
                assert any(
                    argument.startswith("--mount=type=volume,")
                    and argument.endswith(",dst=/run/audit-proxy,readonly")
                    for argument in argv
                )
                argv = argv[:-1] + (
                    "--input-type=module",
                    "-e",
                    BROWSER.replace("TIMEOUT_MODE", "true" if timeout else "false"),
                )
        if timeout and "rm" in argv and "--force" in argv:
            # Before production cleanup removes the browser, prove it launched.
            inspection = original(
                argv[:5]
                + ("exec", argv[-1], "/usr/bin/test", "-f", "/tmp/acceptance-isolation-verified"),
                deadline=time.monotonic() + 5,
                max_bytes=4096,
            )
            if inspection.returncode == 0:
                verified_timeout.append(True)
        return original(argv, **kwargs)

    monkeypatch.setattr(runtime, "_run_process", adapt)
    config = runtime.LighthouseRuntimeConfig(
        image_id=image, timeout_seconds=10.0 if timeout else 90.0
    )
    started = time.monotonic()
    result = runtime.LighthouseRuntime(config).run(
        LighthouseRequest(requested_url="http://fixture.example/", device="desktop", locale="en")
    )
    elapsed = time.monotonic() - started
    leftovers = []
    with tempfile.TemporaryDirectory(prefix="lh-acceptance-inspect-") as directory:
        prefix = (
            runtime._docker_binary(),
            "--host",
            "unix://" + str(runtime._docker_socket()),
            "--config",
            directory,
        )
        for owner in owners:
            for kind in ("container", "volume"):
                args = (kind, "ls", "-q") + (("--all",) if kind == "container" else ())
                check = original(
                    prefix + args + ("--filter", "label=" + owner),
                    deadline=time.monotonic() + 10,
                    max_bytes=4096,
                )
                assert check.returncode == 0
                leftovers.extend(check.stdout.decode().split())
    if timeout:
        assert verified_timeout, "timeout must follow actual browser isolation verification"
    return result, leftovers, elapsed


def test_real_browser_hostile_requests_and_public_fixture(immutable_image, monkeypatch):
    result, leftovers, _ = _acceptance(monkeypatch, immutable_image)
    assert result.failure is None
    assert leftovers == []
    evidence = json.loads(result.body)
    assert evidence["positive"] == "fixture-ok"
    assert evidence["rebind_first"] == "fixture-ok"
    assert evidence["rebind_second"] == "error"
    assert evidence["redirect"] == 403
    assert evidence["subresource"] == "error"
    assert evidence["worker"] == "error"
    assert evidence["mixed"] == "error"
    assert evidence["wss"] == "error"
    assert evidence["direct"] == "denied"
    for host in (
        "redirect-private.example",
        "subresource-private.example",
        "worker-private.example",
        "wss-private.example",
        "mixed.example",
        "rebind.example",
    ):
        assert any(event == ["denied", host] for event in evidence["events"])
    assert [event[2] for event in evidence["events"] if event[:2] == ["dns", "rebind.example"]] == [
        "8.8.8.8",
        "127.0.0.1",
    ]
    assert all(event[1] == "8.8.8.8" for event in evidence["events"] if event[0] == "dial")


def test_real_browser_timeout_removes_owned_resources(immutable_image, monkeypatch):
    result, leftovers, elapsed = _acceptance(monkeypatch, immutable_image, timeout=True)
    assert result.failure == "timeout"
    assert leftovers == []
    assert 10 <= elapsed < 40
