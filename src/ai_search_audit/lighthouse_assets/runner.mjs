// Fixed image entrypoint. Raw LHR is transient stdout; no report files or logs.
import fs from 'node:fs/promises';
import net from 'node:net';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {pathToFileURL} from 'node:url';

const ROOT = '/opt/audit-lighthouse';
const SOCKET = '/run/audit-proxy/proxy.sock';
const MAX_OUTPUT = 8388608;
const exec = promisify(execFile);

export function validateRequest(value) {
  if (!value || Array.isArray(value) || typeof value !== 'object' ||
      Object.keys(value).sort().join(',') !== 'device,locale,requested_url' ||
      !['mobile', 'desktop'].includes(value.device) || !['en', 'pl'].includes(value.locale) ||
      typeof value.requested_url !== 'string' || value.requested_url.length > 4096 ||
      /[\s\x00-\x1f\x7f\\#]/.test(value.requested_url)) throw new Error('invalid_request');
  const target = new URL(value.requested_url);
  if (!['http:', 'https:'].includes(target.protocol) || target.username || target.password ||
      target.port || !target.hostname) throw new Error('invalid_request');
  return value;
}

export function launchOptions(profile, port) {
  return {
    executablePath: '/usr/bin/chromium', pipe: true, headless: true, timeout: 20000,
    userDataDir: profile, env: {PATH: '/usr/bin:/bin', TMPDIR: '/tmp', LANG: 'C.UTF-8'},
    args: [`--proxy-server=http://127.0.0.1:${port}`, '--proxy-bypass-list=<-loopback>',
      '--disable-quic', '--disable-background-networking', '--disable-component-update',
      '--disable-sync', '--no-first-run', '--no-default-browser-check'],
  };
}

export async function launchBrowser(puppeteer, profile, port) {
  try { return await puppeteer.launch(launchOptions(profile, port)); }
  catch { throw new Error('runtime_unverified'); }
}

export function sandboxVerified(status) {
  const rows = new Map(status.split('\n').map(line => line.trim().split(/\t+/)));
  return rows.get('Layer 1 Sandbox') === 'Namespace' &&
    rows.get('PID namespaces') === 'Yes' && rows.get('Network namespaces') === 'Yes' &&
    rows.get('Seccomp-BPF sandbox') === 'Yes';
}

export async function verifyIsolation(browser) {
  try {
    const {stdout} = await exec('/usr/bin/python3', [ROOT + '/isolation.py'], {
      timeout: 5000, maxBuffer: 4096, env: {PATH: '/usr/bin:/bin', LANG: 'C.UTF-8'},
    });
    if (JSON.parse(stdout).network_isolated !== true) throw new Error('network_unverified');
  } catch { throw new Error('network_unverified'); }
  let page;
  try {
    page = await browser.newPage();
    await page.goto('chrome://sandbox', {timeout: 5000});
    const status = await page.evaluate(() => document.body.innerText);
    if (!sandboxVerified(status)) {
      throw new Error('sandbox_unverified');
    }
  } catch { throw new Error('sandbox_unverified'); }
  finally { if (page) await page.close().catch(() => {}); }
}

async function relay() {
  // Readiness checks do not send proxy traffic or resolve a target.
  for (let attempt = 0; ; attempt++) {
    try { if ((await fs.stat(SOCKET)).isSocket()) break; } catch {}
    if (attempt >= 100) throw new Error('runtime_error');
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  const connections = new Set();
  const server = net.createServer(client => {
    if (connections.size >= 64) { client.destroy(); return; }
    const upstream = net.connect(SOCKET);
    connections.add(client); connections.add(upstream);
    const close = () => { client.destroy(); upstream.destroy(); connections.delete(client); connections.delete(upstream); };
    client.on('error', close); upstream.on('error', close);
    client.on('close', close); upstream.on('close', close);
    client.setTimeout(30000, close); upstream.setTimeout(30000, close);
    client.pipe(upstream); upstream.pipe(client);
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject); server.listen(0, '127.0.0.1', resolve);
  });
  return {port: server.address().port, close: () => {
    for (const connection of connections) connection.destroy();
    server.close();
  }};
}

async function readRequest() {
  let data = '';
  for await (const chunk of process.stdin) {
    data += chunk;
    if (Buffer.byteLength(data) > 8192) throw new Error('invalid_request');
  }
  return validateRequest(JSON.parse(data));
}

async function main() {
  const timer = setTimeout(() => process.exit(1), 175000);
  let profile, browser, proxy;
  let envelope = {status: 'unavailable', reason: 'runtime_error'};
  try {
    const request = await readRequest();
    const [{default: puppeteer}, {default: lighthouse}] = await Promise.all([
      import('puppeteer-core'), import('lighthouse'),
    ]);
    profile = await fs.mkdtemp('/tmp/audit-lh-profile-');
    proxy = await relay();
    browser = await launchBrowser(puppeteer, profile, proxy.port);
    await verifyIsolation(browser);
    const page = await browser.newPage();
    const desktop = request.device === 'desktop';
    const screenEmulation = desktop
      ? {mobile: false, width: 1350, height: 940, deviceScaleFactor: 1, disabled: false}
      : {mobile: true, width: 412, height: 823, deviceScaleFactor: 1.75, disabled: false};
    const settings = {
      onlyCategories: ['performance'], formFactor: request.device, locale: request.locale,
      throttlingMethod: 'simulate', screenEmulation,
      throttling: {rttMs: desktop ? 40 : 150, throughputKbps: desktop ? 10240 : 1638.4,
        cpuSlowdownMultiplier: desktop ? 1 : 4, requestLatencyMs: 0,
        downloadThroughputKbps: 0, uploadThroughputKbps: 0},
      maxWaitForLoad: 45000, maxWaitForFcp: 30000,
    };
    // Fourth argument is the official Puppeteer Page connection (CDP pipe).
    const result = await lighthouse(request.requested_url, {logLevel: 'silent'},
      {extends: 'lighthouse:default', settings}, page);
    if (!result?.lhr) throw new Error('runtime_error');
    const chrome = (await browser.version()).match(/(?:Chrome|HeadlessChrome)\/([0-9.]+)/)?.[1];
    const packageVersion = async name => JSON.parse(await fs.readFile(
      `${ROOT}/node_modules/${name}/package.json`, 'utf8')).version;
    envelope = {status: 'ok', isolation: true, versions: {
      node: process.versions.node, chrome,
      lighthouse: await packageVersion('lighthouse'), puppeteer: await packageVersion('puppeteer-core'),
    }, lhr: result.lhr};
  } catch (error) {
    envelope = {status: 'unavailable', reason:
      ['runtime_unverified', 'sandbox_unverified', 'network_unverified'].includes(error?.message) ? error.message : 'runtime_error'};
  } finally {
    if (browser) await browser.close().catch(() => {});
    if (proxy) proxy.close();
    if (profile) await fs.rm(profile, {recursive: true, force: true});
    clearTimeout(timer);
  }
  const output = JSON.stringify(envelope);
  if (Buffer.byteLength(output) > MAX_OUTPUT) {
    process.stdout.write('{"status":"unavailable","reason":"response_too_large"}');
  } else process.stdout.write(output);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch(() => { process.exitCode = 1; });
}
