// Instant Replay viewer — mountReplay(element, src, { onSeek })
//
// `src` is a folder URL (trailing slash) holding either
//   - a service bundle: meta.json + faces.bin + verts.bin (floor frame, y up, meters), or
//   - a run folder:     summary.json + <stem>.ply + <stem>_vis.png (SAM camera frame, Tier 0).
// Needs an import map for "three" and "three/addons/" (see index.html).

import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

const COLORS = {
  bg: 0x0e121a, mesh: 0xf1ede4, grid: 0x334155, axis: 0x64748b,
  bos: 0x38bdf8, good: new THREE.Color(0x22c55e), warn: new THREE.Color(0xf59e0b), bad: new THREE.Color(0xef4444),
};
const VIEWS = { front: { az: 0, el: 6 }, side: { az: 90, el: 6 }, top: { az: 0, el: 88 } };
const SPEEDS = [0.25, 0.5, 1];
const TRANSITION_MS = 1400;
const MARGIN_GREEN_M = 0.03;

// ---------- loading ----------

async function fetchOk(url, as = 'json') {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return as === 'json' ? r.json() : r.arrayBuffer();
}

async function pool(items, n, fn) {
  const out = new Array(items.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(n, items.length) }, async () => {
    while (next < items.length) { const i = next++; out[i] = await fn(items[i], i); }
  }));
  return out;
}

async function loadBundle(src, meta, progress) {
  progress(0.3);
  const [facesBuf, vertsBuf] = await Promise.all([fetchOk(src + 'faces.bin', 'buf'), fetchOk(src + 'verts.bin', 'buf')]);
  progress(1);
  const V = meta.vertex_count, all = new Float32Array(vertsBuf);
  const frames = meta.t.map((_, i) => all.subarray(i * V * 3, (i + 1) * V * 3));
  return {
    kind: 'bundle', meta, t: meta.t, frames, index: new Uint32Array(facesBuf),
    vis: meta.frames.map(f => (f.fal_vis_url ? new URL(f.fal_vis_url, new URL(src, location.href)).href : null)),
    aligned: !!meta.quality?.aligned, synthetic: !!meta.quality?.SYNTHETIC, events: meta.events || [],
  };
}

async function loadRunFolder(src, progress) {
  const summary = await fetchOk(src + 'summary.json');
  const list = summary.frames.filter(f => f.usable);
  if (!list.length) throw new Error('run folder has no usable frames');
  const loader = new PLYLoader();
  let done = 0, index = null, V = null;
  const frames = await pool(list, 8, async f => {
    const geo = loader.parse(await fetchOk(src + f.stem + '.ply', 'buf'));
    const p = geo.attributes.position.array;
    if (V === null) { V = p.length / 3; index = geo.index ? geo.index.array : null; }
    if (p.length / 3 !== V) throw new Error(`${f.stem}: vertex count changed (${p.length / 3} vs ${V})`);
    const out = new Float32Array(p.length);
    for (let i = 0; i < p.length; i += 3) { out[i] = p[i]; out[i + 1] = -p[i + 1]; out[i + 2] = -p[i + 2]; } // OpenCV -> y up
    progress(++done / list.length);
    return out;
  });
  return {
    kind: 'run', meta: null, t: list.map(f => f.t_ms), frames, index,
    vis: list.map(f => (f.vis_file ? src + f.vis_file : null)),
    aligned: false, synthetic: !!summary.SYNTHETIC, events: summary.events || [],
  };
}

async function loadSource(src, progress) {
  if (!src.endsWith('/')) src += '/';
  const probe = await fetch(src + 'meta.json').catch(() => null); // GET: not every server allows HEAD
  return probe && probe.ok ? loadBundle(src, await probe.json(), progress) : loadRunFolder(src, progress);
}

// ---------- helpers ----------

const ease = x => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);

function catmullRom(p0, p1, p2, p3, u, out) {
  const u2 = u * u, u3 = u2 * u;
  for (let i = 0; i < out.length; i++) {
    out[i] = 0.5 * (2 * p1[i] + (-p0[i] + p2[i]) * u + (2 * p0[i] - 5 * p1[i] + 4 * p2[i] - p3[i]) * u2 + (-p0[i] + 3 * p1[i] - 3 * p2[i] + p3[i]) * u3);
  }
}

function marginColor(m) {
  if (m === null || m === undefined || Number.isNaN(m)) return new THREE.Color(0x94a3b8);
  if (m <= 0) return COLORS.bad.clone();
  const x = Math.min(m / MARGIN_GREEN_M, 1);
  return x < 0.5 ? COLORS.bad.clone().lerp(COLORS.warn, x * 2) : COLORS.warn.clone().lerp(COLORS.good, (x - 0.5) * 2);
}

function ribbon(width, color, onTop = false) { // thick floor line segment, unit length along x
  const m = new THREE.Mesh(new THREE.BoxGeometry(1, 0.002, width), new THREE.MeshBasicMaterial({ color, depthTest: !onTop }));
  m.renderOrder = onTop ? 5 : 1;
  return m;
}

function placeRibbon(m, a, b, y) { // a, b: [x, z]
  const dx = b[0] - a[0], dz = b[1] - a[1];
  m.position.set((a[0] + b[0]) / 2, y, (a[1] + b[1]) / 2);
  m.scale.x = Math.hypot(dx, dz) + 0.012;
  m.rotation.y = -Math.atan2(dz, dx);
  m.visible = true;
}

// ---------- DOM ----------

const CSS = `
.rp{position:relative;width:100%;height:100%;display:grid;grid-template-columns:1fr auto;grid-template-rows:auto 1fr auto;
  background:#0e121a;color:#f8fafc;font:700 17px/1.2 system-ui,-apple-system,"Segoe UI",sans-serif;overflow:hidden}
.rp-top{grid-column:1/3;display:flex;gap:10px;align-items:center;padding:10px 16px;flex-wrap:wrap}
.rp-title{font-size:20px;margin-right:auto}
.rp-badge{padding:6px 12px;border-radius:8px;font-size:15px;letter-spacing:.04em}
.rp-badge.syn{background:#facc15;color:#111}.rp-badge.raw{background:#7c2d12;color:#fed7aa}.rp-badge.ok{background:#14532d;color:#bbf7d0}
.rp-stage{position:relative;min-width:0;min-height:0}
.rp-stage canvas{display:block;width:100%!important;height:100%!important}
.rp-side{width:min(34vw,420px);padding:0 16px 0 0;display:flex;flex-direction:column;gap:8px;min-height:0}
.rp-side.hidden{display:none}
.rp-side img{width:100%;border:3px solid #334155;border-radius:10px;background:#1e293b;object-fit:contain}
.rp-side small{color:#94a3b8;font-weight:600}
.rp-bar{grid-column:1/3;display:flex;gap:10px;align-items:center;padding:30px 16px 34px;flex-wrap:wrap}
.rp button{font:inherit;color:#f8fafc;background:#1e293b;border:3px solid #475569;border-radius:10px;padding:8px 16px;cursor:pointer}
.rp button.on{background:#facc15;border-color:#facc15;color:#111}
.rp button:focus-visible{outline:3px solid #38bdf8;outline-offset:2px}
.rp-time{min-width:150px;text-align:right;font-variant-numeric:tabular-nums}
.rp-track{position:relative;flex:1 1 300px;height:34px;background:#1e293b;border:3px solid #475569;border-radius:12px;cursor:pointer;touch-action:none}
.rp-fill{position:absolute;inset:0 auto 0 0;background:#334155;border-radius:9px 0 0 9px}
.rp-head{position:absolute;top:-8px;width:8px;height:44px;margin-left:-4px;background:#facc15;border-radius:4px}
.rp-tick{position:absolute;top:-10px;width:6px;height:48px;margin-left:-3px;background:#ef4444;border-radius:3px;cursor:pointer}
.rp-tick span{position:absolute;bottom:52px;left:50%;transform:translateX(-50%);white-space:nowrap;font-size:14px;color:#fecaca}
.rp-tick.low span{bottom:auto;top:52px}
.rp-bar{row-gap:26px}
.rp-group{display:flex;gap:6px}
.rp-load{position:absolute;inset:0;display:grid;place-items:center;font-size:22px;background:#0e121acc}
@media (max-width:760px){.rp{grid-template-columns:1fr}.rp-side{display:none}.rp-top,.rp-bar{grid-column:1}}
`;

function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text) e.textContent = text; return e; }

// ---------- main ----------

export async function mountReplay(element, src, { onSeek, title = 'Instant Replay' } = {}) {
  if (!document.getElementById('rp-css')) { const s = el('style'); s.id = 'rp-css'; s.textContent = CSS; document.head.append(s); }
  const root = el('div', 'rp'), top = el('div', 'rp-top'), stage = el('div', 'rp-stage'), side = el('div', 'rp-side'), bar = el('div', 'rp-bar');
  top.append(el('div', 'rp-title', title));
  root.append(top, stage, side, bar);
  element.replaceChildren(root);
  const loading = el('div', 'rp-load', 'Loading…');
  stage.append(loading);

  let data;
  try {
    data = await loadSource(src, p => { loading.textContent = `Loading ${Math.round(p * 100)}%`; });
  } catch (e) {
    loading.textContent = `Could not load replay: ${e.message}`;
    throw e;
  }
  loading.remove();

  if (data.synthetic) top.append(el('span', 'rp-badge syn', 'SYNTHETIC'));
  top.append(data.aligned ? el('span', 'rp-badge ok', 'FLOOR-ALIGNED') : el('span', 'rp-badge raw', 'CAMERA FRAME · UNALIGNED'));
  if (data.meta?.quality?.ap_real === false) top.append(el('span', 'rp-badge raw', 'FORWARD/BACK = ESTIMATED'));

  // --- three.js scene ---
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  stage.append(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(COLORS.bg);
  const camera = new THREE.PerspectiveCamera(35, 1, 0.05, 50);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x334155, 1.6));
  const key = new THREE.DirectionalLight(0xffffff, 1.8);
  key.position.set(1.5, 3, 2.5);
  scene.add(key);

  const positions = new Float32Array(data.frames[0]);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  if (data.index) geo.setIndex(new THREE.BufferAttribute(data.index, 1));
  geo.computeVertexNormals();
  const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: COLORS.mesh, roughness: 0.55, metalness: 0 }));
  scene.add(mesh);

  // framing: aligned bundles have the floor at y = 0; raw frames use the lowest vertex
  geo.computeBoundingBox();
  const bb = geo.boundingBox, center = bb.getCenter(new THREE.Vector3());
  const floorY = data.aligned ? 0 : bb.min.y;
  const target = new THREE.Vector3(center.x, data.aligned ? 0.9 : center.y, center.z);
  // far enough to fit the body height (same viewing direction as the phone for raw frames)
  const fit = (0.62 * (bb.max.y - Math.min(bb.min.y, floorY))) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  const radius = Math.max(data.aligned ? 3.0 : target.length(), fit);

  // thick grid
  const grid = new THREE.Group();
  for (let k = -6; k <= 6; k++) {
    const w = k === 0 ? 0.014 : 0.007, c = k === 0 ? COLORS.axis : COLORS.grid;
    const a = ribbon(w, c); placeRibbon(a, [target.x - 1.5, target.z + k * 0.25], [target.x + 1.5, target.z + k * 0.25], floorY); grid.add(a);
    const b = ribbon(w, c); placeRibbon(b, [target.x + k * 0.25, target.z - 1.5], [target.x + k * 0.25, target.z + 1.5], floorY); grid.add(b);
  }
  scene.add(grid);

  // COM / BOS overlays (bundles with metrics only)
  const meta = data.meta, hasCom = !!(meta && meta.com), hasBos = !!(meta && meta.bos);
  const overlayY = floorY + 0.004;
  const comDot = new THREE.Mesh(new THREE.CircleGeometry(0.03, 32), new THREE.MeshBasicMaterial({ color: 0x22c55e, depthTest: false }));
  comDot.rotation.x = -Math.PI / 2; comDot.renderOrder = 7; comDot.visible = hasCom; scene.add(comDot);
  const trail = Array.from({ length: 24 }, (_, i) => {
    const m = new THREE.Mesh(new THREE.CircleGeometry(0.012, 16), new THREE.MeshBasicMaterial({ color: 0xfacc15, transparent: true, opacity: 0.9 * (1 - i / 24), depthTest: false }));
    m.rotation.x = -Math.PI / 2; m.renderOrder = 6; m.visible = false; scene.add(m); return m;
  });
  const bosEdges = Array.from({ length: 64 }, () => { const r = ribbon(0.016, COLORS.bos, true); r.visible = false; scene.add(r); return r; });

  // --- camera presets with eased spherical transitions ---
  const view = { az: 0, el: VIEWS.front.el, r: radius };
  let tween = null, viewName = 'front';
  function applyView() {
    const az = THREE.MathUtils.degToRad(view.az), elv = THREE.MathUtils.degToRad(view.el);
    camera.position.set(target.x + view.r * Math.cos(elv) * Math.sin(az), target.y + view.r * Math.sin(elv), target.z + view.r * Math.cos(elv) * Math.cos(az));
    camera.lookAt(target);
  }
  function setView(name) {
    const to = VIEWS[name];
    if (!to) return;
    let daz = ((to.az - view.az + 540) % 360) - 180; // shortest way round
    tween = { t0: performance.now(), from: { ...view }, to: { az: view.az + daz, el: to.el, r: radius } };
    viewName = name;
    viewButtons.forEach((b, n) => b.classList.toggle('on', n === name));
  }

  // --- controls ---
  const playBtn = el('button', '', 'Pause');
  const viewGroup = el('div', 'rp-group'), speedGroup = el('div', 'rp-group');
  const viewButtons = new Map(Object.keys(VIEWS).map((n, i) => {
    const b = el('button', n === 'front' ? 'on' : '', n[0].toUpperCase() + n.slice(1));
    b.title = `${n} view (${i + 1})`; b.onclick = () => setView(n); viewGroup.append(b); return [n, b];
  }));
  let speed = 0.5;
  const speedButtons = SPEEDS.map(s => {
    const b = el('button', s === speed ? 'on' : '', `${s}×`);
    b.onclick = () => { speed = s; speedButtons.forEach(x => x.classList.toggle('on', x === b)); };
    speedGroup.append(b); return b;
  });
  const smoothBtn = el('button', '', 'Smooth');
  let smooth = false;
  smoothBtn.title = 'Catmull-Rom interpolation between keyframes (off = flipbook)';
  smoothBtn.onclick = () => { smooth = !smooth; smoothBtn.classList.toggle('on', smooth); };
  const imgBtn = el('button', 'on', 'Image');
  imgBtn.onclick = () => { side.classList.toggle('hidden'); imgBtn.classList.toggle('on'); resize(); };

  const track = el('div', 'rp-track'), fill = el('div', 'rp-fill'), head = el('div', 'rp-head');
  track.append(fill, head);
  const timeLabel = el('div', 'rp-time');
  bar.append(playBtn, track, timeLabel, viewGroup, speedGroup, smoothBtn, imgBtn);

  const img = el('img'); img.alt = 'fal visualization for the current frame';
  const imgCaption = el('small');
  side.append(img, imgCaption);
  if (!data.vis.some(Boolean)) { side.classList.add('hidden'); imgBtn.classList.remove('on'); }

  // --- time ---
  const t0 = data.t[0], tEnd = data.t[data.t.length - 1] - t0, T = data.t.map(x => x - t0);
  let playhead = 0, playing = true, lastShown = -1;
  data.events.forEach((ev, k) => {
    const x = (ev.t - t0) / tEnd;
    if (x < 0 || x > 1) return;
    const tick = el('div', k % 2 ? 'rp-tick low' : 'rp-tick'); tick.style.left = `${x * 100}%`; // alternate label sides
    tick.append(el('span', '', `${ev.kind}${ev.side ? ' ' + ev.side : ''}`));
    tick.title = `${ev.kind} at ${((ev.t - t0) / 1000).toFixed(2)} s`;
    tick.addEventListener('pointerdown', e => { e.stopPropagation(); seek(Math.max(0, ev.t - t0 - 500)); setPlaying(true); });
    track.append(tick);
  });
  function setPlaying(p) { playing = p; playBtn.textContent = p ? 'Pause' : 'Play'; }
  playBtn.onclick = () => setPlaying(!playing);
  function seek(ms) { playhead = Math.min(Math.max(ms, 0), tEnd); onSeek?.(playhead + t0); }
  const fromPointer = e => { const r = track.getBoundingClientRect(); seek(((e.clientX - r.left) / r.width) * tEnd); };
  track.addEventListener('pointerdown', e => { track.setPointerCapture(e.pointerId); setPlaying(false); fromPointer(e); });
  track.addEventListener('pointermove', e => { if (track.hasPointerCapture(e.pointerId)) fromPointer(e); });
  const onKey = e => {
    if (e.target.closest?.('input,textarea')) return;
    if (e.code === 'Space') { e.preventDefault(); setPlaying(!playing); }
    else if (e.key === '1') setView('front'); else if (e.key === '2') setView('side'); else if (e.key === '3') setView('top');
  };
  addEventListener('keydown', onKey);

  function frameAt(ms) { // i with T[i] <= ms < T[i+1], and fraction u
    let lo = 0, hi = T.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (T[mid] <= ms) lo = mid; else hi = mid; }
    if (ms >= T[T.length - 1]) return [T.length - 1, 0];
    return [lo, (ms - T[lo]) / (T[lo + 1] - T[lo] || 1)];
  }

  function showFrame(i, u) {
    if (smooth && u > 0 && i < data.frames.length - 1) {
      const F = data.frames, n = F.length;
      catmullRom(F[Math.max(i - 1, 0)], F[i], F[i + 1], F[Math.min(i + 2, n - 1)], u, positions);
    } else if (i !== lastShown || smooth) {
      positions.set(data.frames[i]);
    } else return;
    lastShown = smooth ? -1 : i;
    geo.attributes.position.needsUpdate = true;
    geo.computeVertexNormals();
  }

  function updateOverlays(i) {
    const vis = data.vis[i];
    if (vis && img.dataset.src !== vis) { img.src = vis; img.dataset.src = vis; imgCaption.textContent = `fal visualization · frame ${i + 1}/${data.frames.length}`; }
    if (hasCom) {
      const c = meta.com[i];
      comDot.position.set(c[0], overlayY + 0.002, c[2]);
      comDot.material.color.copy(marginColor(meta.margin?.[i]));
      trail.forEach((m, k) => { const j = i - k - 1; m.visible = j >= 0; if (j >= 0) m.position.set(meta.com[j][0], overlayY, meta.com[j][2]); });
    }
    if (hasBos) {
      const h = meta.bos[i] || [];
      bosEdges.forEach((r, k) => { if (k < h.length) placeRibbon(r, h[k], h[(k + 1) % h.length], overlayY); else r.visible = false; });
    }
  }

  // --- loop ---
  function resize() {
    const w = stage.clientWidth, h = stage.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  const ro = new ResizeObserver(resize); ro.observe(stage);
  resize();
  applyView();

  let last = performance.now(), raf = 0;
  function tick(now) {
    const dt = now - last; last = now;
    if (playing) { playhead += dt * speed; if (playhead > tEnd) playhead = 0; }
    if (tween) {
      const x = Math.min((now - tween.t0) / TRANSITION_MS, 1), e = ease(x);
      for (const k of ['az', 'el', 'r']) view[k] = tween.from[k] + (tween.to[k] - tween.from[k]) * e;
      if (x >= 1) tween = null;
    }
    applyView();
    const [i, u] = frameAt(playhead);
    showFrame(i, u);
    updateOverlays(i);
    const frac = tEnd ? playhead / tEnd : 0;
    fill.style.width = head.style.left = `${frac * 100}%`;
    timeLabel.textContent = `${(playhead / 1000).toFixed(1)} / ${(tEnd / 1000).toFixed(1)} s`;
    renderer.render(scene, camera);
    raf = requestAnimationFrame(tick);
  }
  raf = requestAnimationFrame(tick);

  return {
    seek: ms => seek(ms - t0), play: () => setPlaying(true), pause: () => setPlaying(false), setView,
    get view() { return viewName; },
    destroy() { cancelAnimationFrame(raf); ro.disconnect(); removeEventListener('keydown', onKey); renderer.dispose(); element.replaceChildren(); },
  };
}
