// Screenshot a live page (SSE, WebGL iframes) via Chrome DevTools: node tools/screenshot.mjs URL out.png [wait_ms] [WxH]
// Chrome's plain --screenshot waits for the page to finish loading, which a live dashboard never does.
import { spawn } from 'node:child_process';
import { writeFileSync, rmSync } from 'node:fs';
const [url, out, waitMs = '6000', size = '1600x1000'] = process.argv.slice(2);
const [W, H] = size.split('x').map(Number);
const CH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const port = 9300 + Math.floor(Math.random() * 500), profile = `/tmp/shot-profile-${port}`;
const chrome = spawn(CH, ['--headless=new', `--remote-debugging-port=${port}`, '--enable-unsafe-swiftshader', '--use-angle=swiftshader',
  `--user-data-dir=${profile}`, `--window-size=${W},${H}`, '--no-first-run', '--hide-scrollbars', 'about:blank'], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
const done = code => { try { chrome.kill(); } catch {} setTimeout(() => { rmSync(profile, { recursive: true, force: true }); process.exit(code); }, 300); };
setTimeout(() => { console.error('screenshot timeout'); done(1); }, Number(waitMs) + 30000);
let target;
for (let i = 0; i < 50 && !target; i++) { await sleep(200); try { target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find(t => t.type === 'page'); } catch {} }
const ws = new WebSocket(target.webSocketDebuggerUrl); await new Promise(r => ws.addEventListener('open', r));
let id = 0; const pending = new Map();
ws.addEventListener('message', ev => { const m = JSON.parse(ev.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } });
const send = (method, params = {}) => new Promise(r => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
await send('Page.enable');
await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });
await send('Page.navigate', { url }); await sleep(Number(waitMs));
const r = await send('Page.captureScreenshot', { format: 'png' });
writeFileSync(out, Buffer.from(r.result.data, 'base64')); ws.close(); done(0);
