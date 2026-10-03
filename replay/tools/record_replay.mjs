// Record the viewer's instant replay as frames (frame-exact, via the viewer's manual clock), then encode:
//   python3 -m http.server 8018 -d replay &
//   node replay/tools/record_replay.mjs "http://localhost:8018/viewer/?src=/data/bundles/demo/" /tmp/frames
//   ffmpeg -framerate 30 -i /tmp/frames/f%05d.jpg -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart out.mp4
// Needs Google Chrome (headless). Sequence: 1 s front hold at trial 9.53 s, instant replay (slow-mo,
// swing to side, freeze), 3 s on the callout, Continue (back to front, 0.5x) for 5 s. Edit seekTrial for other clips.
import { spawn } from 'node:child_process';
import { writeFileSync, mkdirSync, rmSync } from 'node:fs';
const [url, dir] = process.argv.slice(2);
rmSync(dir, { recursive: true, force: true }); mkdirSync(dir, { recursive: true });
const CH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const port = 9800 + Math.floor(Math.random() * 100);
const chrome = spawn(CH, ['--headless=new', `--remote-debugging-port=${port}`, '--enable-unsafe-swiftshader', '--use-angle=swiftshader',
  `--user-data-dir=${dir}-profile`, '--window-size=1920,1080', '--no-first-run', 'about:blank'], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let target;
for (let i = 0; i < 50 && !target; i++) { await sleep(200); try { target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find(t => t.type === 'page'); } catch {} }
const ws = new WebSocket(target.webSocketDebuggerUrl); await new Promise(r => ws.addEventListener('open', r));
let id = 0; const pending = new Map();
ws.addEventListener('message', ev => { const m = JSON.parse(ev.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') console.log('[exception]', m.params.exceptionDetails.exception?.description); });
const send = (method, params = {}) => new Promise(r => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const js = async expr => (await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value;
await send('Runtime.enable'); await send('Page.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1920, height: 1080, deviceScaleFactor: 1, mobile: false });
await send('Page.navigate', { url }); await sleep(6000);
const DT = 1000 / 30; let n = 0;
const shoot = async () => { const r = await send('Page.captureScreenshot', { format: 'jpeg', quality: 93 }); writeFileSync(`${dir}/f${String(n++).padStart(5, '0')}.jpg`, Buffer.from(r.result.data, 'base64')); };
const step = async k => { for (let i = 0; i < k; i++) { await js(`window.replay.capture.step(${DT})`); await shoot(); } };
await js(`(()=>{const r=window.replay; r.capture.start(); r.pause(); r.setView('front', 1); r.seekTrial(9.53); r.capture.step(5); return 1})()`);
for (let i = 0; i < 30; i++) await step(1);                       // 1 s hold in the front view
await js(`(()=>{window.replay.instantReplay(); return 1})()`);  // now the swing to the side is visible
let guard = 0;
while (!(await js(`document.querySelector('.rp-call').style.display === 'block'`)) && guard++ < 600) await step(1);
console.log('frozen after', n, 'frames');
await step(90);                                                   // 3 s on the callout
await js(`(()=>{document.querySelector('.rp-call button').click(); return 1})()`);
await step(150);                                                  // 5 s: swing back to front at 0.5x, footprints land
console.log('frames', n); ws.close(); chrome.kill();
