// Instant Replay viewer v2 — mountReplay(element, src, { onSeek, jumpTo })
//
// `src`: a service bundle folder (meta.json + faces.bin + verts.bin [+ verts_raw.bin]) or a raw run
// folder (summary.json + <stem>.ply; Tier 0, most overlays unavailable). Built for screen sharing:
// bold colours, thick strokes, slow eased camera moves. Times shown are trial-relative (0 = first frame).
// Needs an import map for "three" and "three/addons/" (see index.html).

import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

const C = {
  bg: 0x070b14, mesh: 0xdbe7f5, rim: 0x7dd3fc, grid: 0x1e293b, gridMajor: 0x3b4b63, axis: 0x64748b,
  bos: 0x38bdf8, green: 0x22c55e, amber: 0xf59e0b, red: 0xef4444, foot: 0xfbbf24, pip: 0x38bdf8,
};
const VIEWS = { front: { az: 0, el: 18 }, side: { az: 90, el: 12 }, top: { az: 0, el: 88 } };
const SPEEDS = [0.25, 0.5, 1];
const MOVE_MS = 2400; // slow, eased camera moves
const MARGIN_GREEN_M = 0.03;
const MINI = 274; // mini-map size (CSS px, inside its 3 px border)

// ------------------------------------------------------------------ loading

async function fetchOk(url, as = 'json') {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return as === 'json' ? r.json() : r.arrayBuffer();
}

function halfToFloat(u16) { // IEEE 754 binary16 -> Float32Array
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    out[i] = e === 0 ? s * 5.960464477539063e-8 * f : e === 31 ? (f ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return out;
}

async function loadBundle(src, meta, progress) {
  const V = meta.vertex_count, F = meta.t.length, f16 = meta.verts_dtype === 'float16';
  const decode = buf => (f16 ? halfToFloat(new Uint16Array(buf)) : new Float32Array(buf));
  const [facesBuf, vertsBuf, rawBuf] = await Promise.all([
    fetchOk(src + 'faces.bin', 'buf'), fetchOk(src + 'verts.bin', 'buf'),
    meta.verts_raw ? fetchOk(src + meta.verts_raw, 'buf').catch(() => null) : null]);
  progress(1);
  const split = all => Array.from({ length: F }, (_, i) => all.subarray(i * V * 3, (i + 1) * V * 3));
  const abs = u => (u ? new URL(u, new URL(src, location.href)).href : null);
  return {
    kind: 'bundle', meta, t: meta.t, frames: split(decode(vertsBuf)), raw: rawBuf ? split(decode(rawBuf)) : null,
    index: new Uint32Array(facesBuf), aligned: !!meta.quality?.aligned, synthetic: !!meta.quality?.SYNTHETIC,
    vis: meta.frames.map(f => abs(f.fal_vis_url)), srcImg: meta.frames.map(f => abs(f.src_url)),
    srcScale: meta.frames.map(f => f.src_scale || 1), touchdowns: meta.touchdowns || [],
    series: meta.series || null, report: src + 'report.html',
  };
}

async function loadRunFolder(src, progress) { // Tier 0: raw SAM run folder, camera frame, no metrics
  const summary = await fetchOk(src + 'summary.json');
  const conv = await fetch(src + 'conventions.json').then(r => (r.ok ? r.json() : null)).catch(() => null);
  const sgn = conv?.mesh?.flip === 'flip_yz' ? 1 : -1; // fal .ply is already y-up; OpenCV-frame meshes are not
  const list = summary.frames.filter(f => f.usable);
  const loader = new PLYLoader();
  let done = 0, index = null;
  const frames = await Promise.all(list.map(async f => {
    const geo = loader.parse(await fetchOk(src + f.stem + '.ply', 'buf'));
    const p = geo.attributes.position.array, out = new Float32Array(p.length);
    if (!index) index = geo.index ? geo.index.array : null;
    for (let i = 0; i < p.length; i += 3) { out[i] = p[i]; out[i + 1] = sgn * p[i + 1]; out[i + 2] = sgn * p[i + 2]; }
    progress(++done / list.length);
    return out;
  }));
  return { kind: 'run', meta: null, t: list.map(f => f.t_ms), frames, raw: null, index, aligned: false,
    synthetic: !!summary.SYNTHETIC, vis: list.map(f => (f.vis_file ? src + f.vis_file : null)), srcImg: [], srcScale: [],
    touchdowns: [], series: null, report: null };
}

async function loadSource(src, progress) {
  if (!src.endsWith('/')) src += '/';
  const probe = await fetch(src + 'meta.json').catch(() => null);
  return probe && probe.ok ? loadBundle(src, await probe.json(), progress) : loadRunFolder(src, progress);
}

// ------------------------------------------------------------------ helpers

const ease = x => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
const lerp = (a, b, u) => a + (b - a) * u;

function catmullRom(p0, p1, p2, p3, u, out) {
  const u2 = u * u, u3 = u2 * u;
  for (let i = 0; i < out.length; i++)
    out[i] = 0.5 * (2 * p1[i] + (-p0[i] + p2[i]) * u + (2 * p0[i] - 5 * p1[i] + 4 * p2[i] - p3[i]) * u2 + (-p0[i] + 3 * p1[i] - 3 * p2[i] + p3[i]) * u3);
}

function marginColor(m) { // metres -> colour: green inside, amber near the edge, red outside
  if (m === null || m === undefined || Number.isNaN(m)) return new THREE.Color(0x94a3b8);
  if (m <= 0) return new THREE.Color(C.red);
  const x = Math.min(m / MARGIN_GREEN_M, 1);
  return x < 0.5 ? new THREE.Color(C.red).lerp(new THREE.Color(C.amber), x * 2) : new THREE.Color(C.amber).lerp(new THREE.Color(C.green), (x - 0.5) * 2);
}
const cssColor = m => '#' + marginColor(m).getHexString();

function canvasTexture(w, h, draw) {
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const tex = new THREE.CanvasTexture(c); tex.anisotropy = 4; tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}
const GLOW = canvasTexture(128, 128, (g, w) => {
  const r = g.createRadialGradient(w / 2, w / 2, 0, w / 2, w / 2, w / 2);
  r.addColorStop(0, 'rgba(255,255,255,1)'); r.addColorStop(0.25, 'rgba(255,255,255,0.55)'); r.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = r; g.fillRect(0, 0, w, w);
});
const SHADOW = canvasTexture(128, 128, (g, w) => {
  const r = g.createRadialGradient(w / 2, w / 2, 0, w / 2, w / 2, w / 2);
  r.addColorStop(0, 'rgba(0,0,0,0.85)'); r.addColorStop(0.6, 'rgba(0,0,0,0.35)'); r.addColorStop(1, 'rgba(0,0,0,0)');
  g.fillStyle = r; g.fillRect(0, 0, w, w);
});
const FOOTPRINT = canvasTexture(128, 320, (g, w, h) => { // glowing sole: heel at the bottom, toes at the top
  g.shadowColor = 'rgba(251,191,36,0.9)'; g.shadowBlur = 18; g.fillStyle = 'rgba(251,191,36,0.95)';
  g.beginPath(); g.ellipse(w / 2, h * 0.78, w * 0.26, h * 0.16, 0, 0, Math.PI * 2); g.fill();
  g.beginPath(); g.ellipse(w / 2 + 4, h * 0.42, w * 0.34, h * 0.26, 0.08, 0, Math.PI * 2); g.fill();
  for (const [x, y, r] of [[0.34, 0.12, 0.09], [0.52, 0.09, 0.07], [0.66, 0.11, 0.06], [0.77, 0.15, 0.05], [0.85, 0.2, 0.045]]) {
    g.beginPath(); g.arc(w * x, h * y, w * r, 0, Math.PI * 2); g.fill();
  }
});

function textSprite(text, { size = 0.06, color = '#f8fafc', bg = 'rgba(2,6,23,0.8)' } = {}) {
  const font = '800 56px system-ui, -apple-system, "Segoe UI", sans-serif';
  const probe = document.createElement('canvas').getContext('2d'); probe.font = font;
  const w = Math.ceil(probe.measureText(text).width) + 48, h = 84;
  const tex = canvasTexture(w, h, g => {
    g.font = font; g.fillStyle = bg; g.beginPath(); g.roundRect(0, 0, w, h, 22); g.fill();
    g.fillStyle = color; g.textBaseline = 'middle'; g.fillText(text, 24, h / 2 + 2);
  });
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  s.scale.set(size * w / h, size, 1); s.renderOrder = 20;
  return s;
}

function floorLabel(text, x, y, z, h = 0.05) { // flat, high-contrast text on the floor
  const font = '800 64px system-ui, sans-serif';
  const probe = document.createElement('canvas').getContext('2d'); probe.font = font;
  const w = Math.ceil(probe.measureText(text).width) + 20;
  const tex = canvasTexture(w, 90, g => { g.font = font; g.fillStyle = '#e2e8f0'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(text, w / 2, 48); });
  const m = new THREE.Mesh(new THREE.PlaneGeometry(h * w / 90, h), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false }));
  m.rotation.x = -Math.PI / 2; m.position.set(x, y + 0.002, z);
  return m;
}

function ribbon(width, color, onTop = false) { // thick floor segment, unit length along x
  const m = new THREE.Mesh(new THREE.BoxGeometry(1, 0.002, width), new THREE.MeshBasicMaterial({ color, depthTest: !onTop, transparent: onTop }));
  m.renderOrder = onTop ? 6 : 1;
  return m;
}
function placeRibbon(m, a, b, y) {
  const dx = b[0] - a[0], dz = b[1] - a[1];
  m.position.set((a[0] + b[0]) / 2, y, (a[1] + b[1]) / 2);
  m.scale.x = Math.hypot(dx, dz) + 0.008; m.rotation.y = -Math.atan2(dz, dx); m.visible = true;
}

function bodyMaterial() { // translucent body, fresnel rim light, optional per-vertex heat colours
  return new THREE.ShaderMaterial({
    uniforms: { uColor: { value: new THREE.Color(C.mesh) }, uRim: { value: new THREE.Color(C.rim) },
      uOpacity: { value: 0.62 }, uHeat: { value: 0 }, uLight: { value: new THREE.Vector3(0.4, 0.75, 0.55).normalize() } },
    vertexShader: `attribute vec3 color; varying vec3 vN; varying vec3 vV; varying vec3 vC;
      void main(){ vec4 mv = modelViewMatrix * vec4(position,1.0); vN = normalize(normalMatrix*normal); vV = -mv.xyz; vC = color; gl_Position = projectionMatrix*mv; }`,
    fragmentShader: `uniform vec3 uColor; uniform vec3 uRim; uniform float uOpacity; uniform float uHeat; uniform vec3 uLight;
      varying vec3 vN; varying vec3 vV; varying vec3 vC;
      void main(){ vec3 n = normalize(vN); vec3 v = normalize(vV); float lam = max(dot(n, uLight), 0.0);
        float fr = pow(1.0 - max(dot(n, v), 0.0), 2.4);
        vec3 base = mix(uColor, vC, uHeat);
        vec3 col = base * (0.30 + 0.62 * lam) + uRim * fr * 1.35;
        gl_FragColor = vec4(col, clamp(uOpacity + fr * 0.55, 0.0, 1.0)); }`,
    transparent: true, depthWrite: true,
  });
}

// ------------------------------------------------------------------ DOM

const CSS = `
.rp{position:relative;width:100%;height:100%;display:grid;grid-template-rows:auto 1fr auto;background:#070b14;color:#f8fafc;
  font:700 17px/1.25 system-ui,-apple-system,"Segoe UI",sans-serif;overflow:hidden}
.rp-top{display:flex;gap:10px;align-items:center;padding:10px 16px;flex-wrap:wrap;z-index:3}
.rp-title{font-size:20px;margin-right:auto}
.rp-badge{padding:6px 12px;border-radius:8px;font-size:14px;letter-spacing:.03em;background:#1e293b;color:#e2e8f0}
.rp-badge.syn{background:#facc15;color:#111}.rp-badge.raw{background:#7c2d12;color:#fed7aa}.rp-badge.ok{background:#14532d;color:#bbf7d0}
.rp-badge.est{background:#78350f;color:#fde68a}.rp-badge.q-good{background:#166534;color:#dcfce7}.rp-badge.q-fair{background:#92400e;color:#fef3c7}.rp-badge.q-poor{background:#7f1d1d;color:#fee2e2}
.rp-stage{position:relative;min-height:0}
.rp-stage canvas.main{display:block;width:100%!important;height:100%!important}
.rp-read{position:absolute;left:14px;top:10px;background:rgba(2,6,23,.84);border:2px solid #1e293b;border-radius:14px;padding:10px 16px 12px;min-width:260px;z-index:2}
.rp-read .k{color:#94a3b8;font-size:13px;letter-spacing:.06em;text-transform:uppercase;margin-top:6px}
.rp-read .v{font-size:28px;font-weight:900}.rp-read .vs{font-size:18px;font-weight:800}
.rp-mini{position:absolute;right:14px;bottom:12px;width:${MINI + 6}px;height:${MINI + 6}px;border:3px solid #334155;border-radius:14px;pointer-events:none;z-index:2;box-sizing:border-box}
.rp-mini span{position:absolute;left:10px;top:6px;font-size:13px;color:#94a3b8;letter-spacing:.06em}
.rp-mini b{position:absolute;right:12px;bottom:8px;font-size:13px;color:#e2e8f0}
.rp-mini i{position:absolute;right:12px;bottom:26px;height:5px;background:#e2e8f0;border-radius:3px}
.rp-pip{position:absolute;right:14px;top:10px;height:calc(100% - ${MINI + 40}px);max-height:560px;min-height:220px;aspect-ratio:9/16;border:3px solid #334155;border-radius:14px;overflow:hidden;background:#000;z-index:2;cursor:zoom-in}
.rp-pip img,.rp-pip canvas{position:absolute;inset:0;width:100%;height:100%}
.rp-pip span{position:absolute;left:8px;top:6px;font-size:12px;background:rgba(2,6,23,.78);padding:2px 7px;border-radius:6px;z-index:2}
.rp-call{position:absolute;left:14px;top:300px;width:min(540px,34%);background:rgba(2,6,23,.94);border:3px solid #ef4444;border-radius:16px;
  padding:14px 20px;font-size:26px;line-height:1.2;font-weight:900;z-index:4;display:none}
.rp-call small{display:block;font-size:15px;color:#cbd5e1;font-weight:700;margin-top:6px}
.rp-call button{margin-top:10px}
.rp-modal{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;place-items:center;z-index:10;cursor:zoom-out}
.rp-modal img{max-width:96vw;max-height:92vh;border-radius:8px}
.rp-bar{display:grid;grid-template-columns:auto 1fr auto;gap:8px 10px;align-items:center;padding:6px 16px 10px;z-index:3}
.rp-tl{position:relative;height:92px;cursor:pointer;touch-action:none}
.rp-tl canvas{width:100%;height:100%;display:block}
.rp-legend{grid-column:1/4;display:flex;gap:18px;flex-wrap:wrap;font-size:13px;color:#cbd5e1;font-weight:700}
.rp-legend i{display:inline-block;width:22px;height:8px;border-radius:4px;vertical-align:middle;margin-right:6px}
.rp-ctl{grid-column:1/4;display:flex;gap:6px;flex-wrap:wrap}
.rp button{font:inherit;font-size:15px;color:#f8fafc;background:#1e293b;border:3px solid #475569;border-radius:10px;padding:7px 12px;cursor:pointer}
.rp button.on{background:#facc15;border-color:#facc15;color:#111}
.rp button.hot{background:#ef4444;border-color:#ef4444}
.rp button:focus-visible{outline:3px solid #38bdf8;outline-offset:2px}
.rp-time{font-variant-numeric:tabular-nums;min-width:120px;text-align:right}
.rp-load{position:absolute;inset:0;display:grid;place-items:center;font-size:22px;background:#070b14cc;z-index:5}
@media (max-width:820px){.rp-pip,.rp-mini{display:none}.rp-read{min-width:0}}
`;
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
const STANCE_TXT = { single_left: 'On left foot', single_right: 'On right foot', double_side_by_side: 'Both feet · side by side',
  double_tandem: 'Both feet · tandem', double_staggered: 'Both feet · staggered' };

// ------------------------------------------------------------------ main

export async function mountReplay(element, src, { onSeek, jumpTo, title = 'Instant Replay' } = {}) {
  if (!document.getElementById('rp-css')) { const s = el('style'); s.id = 'rp-css'; s.textContent = CSS; document.head.append(s); }
  const root = el('div', 'rp'), top = el('div', 'rp-top'), stage = el('div', 'rp-stage'), bar = el('div', 'rp-bar');
  top.append(el('div', 'rp-title', title));
  root.append(top, stage, bar);
  element.replaceChildren(root);
  const loading = el('div', 'rp-load', 'Loading…'); stage.append(loading);
  let data;
  try { data = await loadSource(src, p => { loading.textContent = `Loading ${Math.round(p * 100)}%`; }); }
  catch (e) { loading.textContent = `Could not load replay: ${e.message}`; throw e; }
  loading.remove();

  const meta = data.meta, Q = meta?.quality || {}, A = meta?.analytics || {};
  const t0 = data.t[0], T = data.t.map(x => x - t0), tEnd = T[T.length - 1];
  const rel = ms => (ms - t0) / 1000; // absolute ms -> seconds into the trial

  if (data.synthetic) top.append(el('span', 'rp-badge syn', 'SYNTHETIC'));
  if (A.quality) top.append(el('span', `rp-badge q-${A.quality}`, `quality: ${A.quality}`));
  top.append(data.aligned ? el('span', 'rp-badge ok', 'floor-aligned') : el('span', 'rp-badge raw', 'camera frame · unaligned'));
  if (Q.ap_real === false) top.append(el('span', 'rp-badge est', 'forward/back estimated'));
  if (data.aligned) top.append(el('span', 'rp-badge', Q.patient_height_cm ? `scaled to ${Math.round(Q.patient_height_cm)} cm` : "SAM's scale · no height given"));

  // --- renderer / scene
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.domElement.className = 'main';
  renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.autoClear = false;
  stage.append(renderer.domElement);
  const scene = new THREE.Scene(); scene.background = new THREE.Color(C.bg);
  const camera = new THREE.PerspectiveCamera(30, 1, 0.05, 60);

  const positions = new Float32Array(data.frames[0]);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  if (data.index) geo.setIndex(new THREE.BufferAttribute(data.index, 1));
  const heat = new Float32Array(positions.length).fill(0.8);
  if (meta?.heatmap) { // per-vertex RMS displacement -> blue (still) ... red (moved most)
    const h = meta.heatmap.map(x => x || 0), hi = Math.max(...h) || 1;
    for (let i = 0; i < h.length; i++) {
      const c = new THREE.Color().setHSL(0.62 - 0.62 * Math.min(h[i] / hi, 1), 0.9, 0.55);
      heat[i * 3] = c.r; heat[i * 3 + 1] = c.g; heat[i * 3 + 2] = c.b;
    }
  }
  geo.setAttribute('color', new THREE.BufferAttribute(heat, 3));
  geo.computeVertexNormals();
  const glassMat = bodyMaterial();
  const matteMat = new THREE.MeshStandardMaterial({ color: 0xe5e7eb, roughness: 0.75, metalness: 0 });
  const mesh = new THREE.Mesh(geo, glassMat); mesh.renderOrder = 5; scene.add(mesh);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x1e293b, 1.4));
  const key = new THREE.DirectionalLight(0xffffff, 1.6); key.position.set(1.5, 3, 2.5); scene.add(key);

  geo.computeBoundingBox();
  const bb = geo.boundingBox, center = bb.getCenter(new THREE.Vector3());
  const floorY = data.aligned ? 0 : bb.min.y, oY = floorY + 0.003;

  // --- floor grid: faint 1 cm lines, thick 10 cm lines, large labels every 20 cm
  const R = 0.6;
  if (data.aligned) {
    const minor = [];
    for (let k = -60; k <= 60; k++) if (k % 10) { const v = k / 100; minor.push(-R, floorY, v, R, floorY, v, v, floorY, -R, v, floorY, R); }
    const mg = new THREE.BufferGeometry(); mg.setAttribute('position', new THREE.Float32BufferAttribute(minor, 3));
    scene.add(new THREE.LineSegments(mg, new THREE.LineBasicMaterial({ color: C.grid, transparent: true, opacity: 0.5 })));
    for (let k = -6; k <= 6; k++) {
      const v = k / 10, w = k === 0 ? 0.009 : 0.005, col = k === 0 ? C.axis : C.gridMajor;
      const a = ribbon(w, col); placeRibbon(a, [-R, v], [R, v], floorY); scene.add(a);
      const b = ribbon(w, col); placeRibbon(b, [v, -R], [v, R], floorY); scene.add(b);
      if (k && k % 2 === 0) { scene.add(floorLabel(`${k * 10}`, v, floorY, R + 0.08, 0.07)); scene.add(floorLabel(`${k * 10}`, R + 0.12, floorY, v, 0.07)); }
    }
    scene.add(floorLabel('cm', R + 0.12, floorY, R + 0.08, 0.06));
  } else {
    const g = new THREE.GridHelper(3, 12, C.gridMajor, C.grid); g.position.set(center.x, floorY, center.z); scene.add(g);
  }

  // --- overlays (aligned bundles)
  const hasCom = !!meta?.com, hasBos = !!meta?.bos;
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: SHADOW, transparent: true, depthWrite: false, opacity: 0.85 }));
  shadow.rotation.x = -Math.PI / 2; shadow.renderOrder = 2; shadow.visible = false; scene.add(shadow);
  const bosFill = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ color: C.bos, transparent: true, opacity: 0.3, depthTest: false, side: THREE.DoubleSide }));
  bosFill.renderOrder = 4; scene.add(bosFill);
  const bosEdges = Array.from({ length: 80 }, () => { const r = ribbon(0.008, C.bos, true); r.visible = false; scene.add(r); return r; });
  const trail = Array.from({ length: 30 }, (_, i) => {
    const m = new THREE.Mesh(new THREE.CircleGeometry(0.008, 16), new THREE.MeshBasicMaterial({ color: 0xfacc15, transparent: true, opacity: 0.9 * (1 - i / 30), depthTest: false }));
    m.rotation.x = -Math.PI / 2; m.renderOrder = 7; m.visible = false; scene.add(m); return m;
  });
  const comSphere = new THREE.Mesh(new THREE.SphereGeometry(0.035, 32, 16), new THREE.MeshBasicMaterial({ color: C.green, depthTest: false, transparent: true }));
  comSphere.renderOrder = 12;
  const comGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: GLOW, color: C.green, depthTest: false, transparent: true, blending: THREE.AdditiveBlending, opacity: 0.9 }));
  comGlow.scale.set(0.24, 0.24, 1); comGlow.renderOrder = 11;
  const plumb = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 1, 12), new THREE.MeshBasicMaterial({ color: C.green, depthTest: false, transparent: true, opacity: 0.95 }));
  plumb.renderOrder = 11;
  const plumbDot = new THREE.Mesh(new THREE.CircleGeometry(0.024, 32), new THREE.MeshBasicMaterial({ color: C.green, depthTest: false, transparent: true }));
  plumbDot.rotation.x = -Math.PI / 2; plumbDot.renderOrder = 13;
  const plumbRing = new THREE.Mesh(new THREE.RingGeometry(0.024, 0.032, 32), new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false, transparent: true }));
  plumbRing.rotation.x = -Math.PI / 2; plumbRing.renderOrder = 13;
  const bodyOnly = [comSphere, comGlow, plumb]; // hidden in the mini-map pass
  for (const o of [comSphere, comGlow, plumb, plumbDot, plumbRing]) { o.visible = hasCom; scene.add(o); }

  // --- touchdowns: glowing footprint + ripple + time label; persist once their time has passed
  const tds = data.touchdowns.map(d => {
    const tt = rel(d.t_ms), out = { d, tt, items: [], ripple: null, label: null, num: null };
    if (!d.foot) return out;
    const [hx, hz] = d.foot.heel, [tx, tz] = d.foot.toe, len = Math.max(Math.hypot(tx - hx, tz - hz) * 1.25, 0.2);
    const cx = (hx + tx) / 2, cz = (hz + tz) / 2;
    const fp = new THREE.Mesh(new THREE.PlaneGeometry(len * 0.42, len), new THREE.MeshBasicMaterial({ map: FOOTPRINT, transparent: true, depthWrite: false, opacity: d.uncertain ? 0.45 : 0.95 }));
    fp.rotation.order = 'YXZ'; fp.rotation.y = Math.atan2(tx - hx, tz - hz) + Math.PI; fp.rotation.x = -Math.PI / 2;
    fp.position.set(cx, oY + 0.001, cz); fp.renderOrder = 3; scene.add(fp); out.items.push(fp);
    const rip = new THREE.Mesh(new THREE.RingGeometry(0.85, 1, 64), new THREE.MeshBasicMaterial({ color: C.foot, transparent: true, depthTest: false }));
    rip.rotation.x = -Math.PI / 2; rip.position.copy(fp.position); rip.renderOrder = 8; rip.visible = false; scene.add(rip); out.ripple = rip;
    const lab = textSprite(`${tt.toFixed(1)} s · ${d.label}`, { size: 0.055, color: d.uncertain ? '#fde68a' : '#fef3c7' });
    lab.position.set(cx, 0.13, cz); scene.add(lab); out.label = lab;
    const num = textSprite(`${d.n}`, { size: 0.06, bg: d.uncertain ? 'rgba(146,64,14,0.95)' : 'rgba(217,119,6,0.98)' });
    num.position.set(cx, oY + 0.02, cz); num.layers.set(1); scene.add(num); out.num = num; // mini-map only
    for (const o of [fp, lab, num]) o.visible = false;
    return out;
  });

  // --- onion skin: faint ghosts across the trial
  const ghosts = [];
  for (let k = 0; k < 8; k++) {
    const i = Math.round(k * (data.frames.length - 1) / 7), g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(data.frames[i]), 3));
    if (data.index) g.setIndex(new THREE.BufferAttribute(data.index, 1));
    const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0x7dd3fc, transparent: true, opacity: 0.07, depthWrite: false }));
    m.visible = false; m.renderOrder = 4; scene.add(m); ghosts.push(m);
  }
  let onion = false;

  // --- camera presets: lower, more downward default; tight on body + floor; slow eased moves
  const target = new THREE.Vector3();
  // camera 18° above, aimed at 0.85 m, 4.3 m back: head ~13° above centre, front grid edge ~14° below
  // (inside the ±15° vertical field of view), so body + floor fill the frame without empty sky
  const body = new THREE.Vector3(center.x, data.aligned ? 0.85 : center.y, center.z);
  const fit = (0.6 * (bb.max.y - Math.min(bb.min.y, floorY))) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  const radius = data.aligned ? 4.3 : Math.max(body.length(), fit);
  const presetFor = name => (name === 'top' && data.aligned ? { tx: miniCentre[0], ty: floorY, tz: miniCentre[1], r: 3.0, op: 0.16 }
    : { tx: body.x, ty: body.y, tz: body.z, r: radius, op: 1 });
  let miniCentre = [0, 0];
  const view = { az: 0, el: VIEWS.front.el, tx: body.x, ty: body.y, tz: body.z, r: radius, op: 1 };
  let tween = null, viewName = 'front';
  function applyView() {
    const az = THREE.MathUtils.degToRad(view.az), elv = THREE.MathUtils.degToRad(view.el);
    target.set(view.tx, view.ty, view.tz);
    camera.position.set(target.x + view.r * Math.cos(elv) * Math.sin(az), target.y + view.r * Math.sin(elv), target.z + view.r * Math.cos(elv) * Math.cos(az));
    camera.lookAt(target);
    glassMat.uniforms.uOpacity.value = 0.62 * view.op; matteMat.opacity = view.op; matteMat.transparent = view.op < 0.99;
  }
  function setView(name, ms = MOVE_MS) {
    const to = VIEWS[name]; if (!to) return;
    const daz = ((to.az - view.az + 540) % 360) - 180;
    tween = { t0: clock(), ms, from: { ...view }, to: { az: view.az + daz, el: to.el, ...presetFor(name) } };
    viewName = name; viewButtons.forEach((b, n) => b.classList.toggle('on', n === name));
  }

  // --- mini-map: top-down, zoomed on the feet, rendered into a scissored corner viewport
  const mini = el('div', 'rp-mini'); mini.append(el('span', '', 'TOP VIEW · FEET'));
  const scaleBar = el('i'); mini.append(scaleBar, el('b', '', '10 cm')); stage.append(mini);
  const miniCam = new THREE.OrthographicCamera(-0.3, 0.3, 0.3, -0.3, 0.01, 10); miniCam.layers.enable(1);
  if (hasBos) {
    const xs = [], zs = [];
    meta.bos.forEach(h => (h || []).forEach(([x, z]) => { xs.push(x); zs.push(z); }));
    tds.forEach(({ d }) => d.foot && [d.foot.heel, d.foot.toe].forEach(([x, z]) => { xs.push(x); zs.push(z); }));
    if (xs.length) {
      miniCentre = [(Math.min(...xs) + Math.max(...xs)) / 2, (Math.min(...zs) + Math.max(...zs)) / 2];
      const half = Math.max(0.2, Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...zs) - Math.min(...zs)) / 2 + 0.06);
      Object.assign(miniCam, { left: -half, right: half, top: half, bottom: -half }); miniCam.updateProjectionMatrix();
      scaleBar.style.width = `${(0.1 / (2 * half)) * MINI}px`;
    }
  } else mini.style.display = 'none';
  miniCam.position.set(miniCentre[0], 3, miniCentre[1]); miniCam.up.set(0, 0, -1); miniCam.lookAt(miniCentre[0], 0, miniCentre[1]);

  // --- readout panel
  const read = el('div', 'rp-read'); stage.append(read);
  const rv = {};
  for (const [k, label, cls] of [['margin', 'Margin of stability', 'v'], ['stance', 'Stance', 'vs'], ['lean', 'Trunk lean', 'vs'], ['out', 'Outside BOS so far', 'vs']]) {
    read.append(el('div', 'k', label)); rv[k] = el('div', cls, '–'); read.append(rv[k]);
  }
  if (!meta?.margin) read.style.display = 'none';

  // --- picture-in-picture: the real video frame + OUR unsmoothed mesh projected with SAM's camera
  const cam = meta?.camera, havePip = !!(cam && cam.t && data.srcImg.some(Boolean));
  const pip = el('div', 'rp-pip'), pipImg = el('img'); pipImg.alt = 'video frame with the replay mesh overlaid';
  const pipR = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true }); pipR.setPixelRatio(Math.min(devicePixelRatio, 2));
  pip.append(pipImg, pipR.domElement, el('span', '', 'video + our mesh · click for fal panel'));
  stage.append(pip);
  const pipScene = new THREE.Scene(), pipCam = new THREE.Camera();
  const pipGeo = new THREE.BufferGeometry(), pipPos = new Float32Array(data.frames[0]);
  pipGeo.setAttribute('position', new THREE.BufferAttribute(pipPos, 3));
  pipGeo.setAttribute('color', new THREE.BufferAttribute(heat, 3));
  if (data.index) pipGeo.setIndex(new THREE.BufferAttribute(data.index, 1));
  const pipMat = bodyMaterial();
  pipMat.uniforms.uColor.value = new THREE.Color(C.pip); pipMat.uniforms.uOpacity.value = 0.36; pipMat.uniforms.uRim.value = new THREE.Color(0xffffff);
  const pipMesh = new THREE.Mesh(pipGeo, pipMat); pipMesh.matrixAutoUpdate = false; pipScene.add(pipMesh);
  let pipShown = -1;
  if (!havePip) pip.style.display = 'none';
  const modal = el('div', 'rp-modal'), modalImg = el('img'); modalImg.alt = 'fal visualization'; modal.append(modalImg); document.body.append(modal);
  modal.onclick = () => (modal.style.display = 'none');
  pip.onclick = () => { const v = data.vis[curFrame]; if (v) { modalImg.src = v; modal.style.display = 'grid'; } };
  function updatePip(i) {
    if (!havePip || pip.style.display === 'none' || i === pipShown) return;
    const w = pip.clientWidth, h = pip.clientHeight; if (!w || !h) return;
    pipShown = i;
    if (data.srcImg[i]) pipImg.src = data.srcImg[i];
    const [Wf, Hf] = cam.image_size[i], fx = cam.focal[i] * (w / Wf), fy = cam.focal[i] * (h / Hf), n = 0.05, f = 20;
    pipR.setSize(w, h, false);
    pipCam.projectionMatrix.set(2 * fx / w, 0, 0, 0, 0, 2 * fy / h, 0, 0, 0, 0, -(f + n) / (f - n), -2 * f * n / (f - n), 0, 0, -1, 0);
    pipCam.projectionMatrixInverse.copy(pipCam.projectionMatrix).invert();
    const M = cam.M, t = cam.t[i]; // floor -> OpenCV camera, then flip y/z for OpenGL
    pipMesh.matrix.set(M[0][0], M[0][1], M[0][2], t[0], -M[1][0], -M[1][1], -M[1][2], -t[1], -M[2][0], -M[2][1], -M[2][2], -t[2], 0, 0, 0, 1);
    pipPos.set((data.raw || data.frames)[i]);
    pipGeo.attributes.position.needsUpdate = true; pipGeo.computeVertexNormals();
    pipR.render(pipScene, pipCam);
  }

  const call = el('div', 'rp-call'); stage.append(call);

  // --- controls
  const playBtn = el('button', '', 'Pause');
  const viewButtons = new Map(Object.keys(VIEWS).map((n, i) => {
    const b = el('button', n === 'front' ? 'on' : '', n[0].toUpperCase() + n.slice(1)); b.title = `${n} view (${i + 1})`; b.onclick = () => setView(n); return [n, b];
  }));
  let speed = 0.5, smooth = false;
  const speedButtons = SPEEDS.map(s => { const b = el('button', s === speed ? 'on' : '', `${s}×`); b.onclick = () => setSpeed(s); return b; });
  function setSpeed(s) { speed = s; speedButtons.forEach((b, k) => b.classList.toggle('on', SPEEDS[k] === s)); }
  const toggle = (label, on, fn, tip) => { const b = el('button', on ? 'on' : '', label); b.title = tip || label; b.onclick = () => { on = !on; b.classList.toggle('on', on); fn(on); }; return b; };
  const replayBtn = el('button', 'hot', '▶ Instant replay'); replayBtn.title = 'slow-motion replay of the biggest event (R)';
  replayBtn.onclick = () => instantReplay(defaultEvent());
  const ctl = el('div', 'rp-ctl');
  ctl.append(replayBtn, ...viewButtons.values(), ...speedButtons,
    toggle('Smooth', false, v => (smooth = v), 'interpolate between frames (automatic during instant replay)'),
    toggle('Matte', false, v => { mesh.material = v ? matteMat : glassMat; }, 'solid matte body'),
    toggle('Wire', false, v => { glassMat.wireframe = v; matteMat.wireframe = v; }, 'wireframe (screenshots)'),
    toggle('Heat', false, v => { glassMat.uniforms.uHeat.value = v ? 1 : 0; }, 'per-vertex sway heatmap'),
    toggle('Onion', false, v => { onion = v; ghosts.forEach(g => (g.visible = v)); }, 'ghost poses across the trial'),
    toggle('Video', havePip, v => { pip.style.display = v ? '' : 'none'; pipShown = -1; }, 'video frame with our mesh'));
  if (data.report) { const rb = el('button', '', 'Report ↗'); rb.onclick = () => window.open(data.report, '_blank'); ctl.append(rb); }
  const tl = el('div', 'rp-tl'), tlCanvas = el('canvas'); tl.append(tlCanvas);
  const timeLabel = el('div', 'rp-time');
  const legend = el('div', 'rp-legend');
  legend.innerHTML = '<span><i style="background:linear-gradient(90deg,#22c55e,#f59e0b,#ef4444)"></i>margin of stability (inside → outside)</span>'
    + '<span><i style="background:#ef444455;border:2px solid #ef4444"></i>COM outside the base of support</span>'
    + '<span>👣 touchdown / step · dashed = uncertain · click to replay</span><span>time = seconds into the trial</span>';
  bar.append(playBtn, tl, timeLabel, legend, ctl);

  // --- time
  let playhead = 0, playing = true, curFrame = 0, lastShown = -1, replay = null;
  const margins = meta?.margin ? meta.margin.map(m => (m === null ? NaN : m)) : null;
  function setPlaying(p) { playing = p; playBtn.textContent = p ? 'Pause' : 'Play'; if (p) call.style.display = 'none'; }
  playBtn.onclick = () => { replay = null; setPlaying(!playing); };
  function seek(ms) { playhead = Math.min(Math.max(ms, 0), tEnd); onSeek?.(playhead + t0); }
  function frameAt(ms) {
    let lo = 0, hi = T.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (T[mid] <= ms) lo = mid; else hi = mid; }
    if (ms >= T[T.length - 1]) return [T.length - 1, 0];
    return [lo, (ms - T[lo]) / (T[lo + 1] - T[lo] || 1)];
  }

  // timeline: margin curve (green -> red), shaded outside-BOS spans, footprint icons with labels
  const pad = { l: 8, r: 8, t: 28, b: 18 };
  const xOf = ms => pad.l + (tl.clientWidth - pad.l - pad.r) * (ms / tEnd);
  function drawTimeline() {
    const w = tl.clientWidth, h = tl.clientHeight, dpr = Math.min(devicePixelRatio, 2);
    if (!w || !h) return;
    if (tlCanvas.width !== Math.round(w * dpr)) { tlCanvas.width = Math.round(w * dpr); tlCanvas.height = Math.round(h * dpr); }
    const g = tlCanvas.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
    g.fillStyle = '#0f172a'; g.beginPath(); g.roundRect(0, 0, w, h, 12); g.fill();
    const y0 = pad.t, y1 = h - pad.b;
    if (margins) {
      const ok = margins.filter(Number.isFinite), lo = Math.min(-0.03, ...ok), hi = Math.max(0.06, ...ok);
      const yOf = m => y0 + (y1 - y0) * (hi - m) / (hi - lo);
      g.fillStyle = 'rgba(239,68,68,0.24)';
      for (let i = 0; i < margins.length; i++) if (margins[i] < 0) {
        const a = xOf(i ? (T[i - 1] + T[i]) / 2 : T[i]), b = xOf(i < T.length - 1 ? (T[i] + T[i + 1]) / 2 : T[i]);
        g.fillRect(a, y0, Math.max(b - a, 2), y1 - y0);
      }
      g.strokeStyle = '#475569'; g.lineWidth = 1.5; g.beginPath(); g.moveTo(pad.l, yOf(0)); g.lineTo(w - pad.r, yOf(0)); g.stroke();
      g.lineWidth = 4; g.lineCap = 'round';
      for (let i = 0; i < margins.length - 1; i++) {
        if (!Number.isFinite(margins[i]) || !Number.isFinite(margins[i + 1])) continue;
        g.strokeStyle = cssColor((margins[i] + margins[i + 1]) / 2);
        g.beginPath(); g.moveTo(xOf(T[i]), yOf(margins[i])); g.lineTo(xOf(T[i + 1]), yOf(margins[i + 1])); g.stroke();
      }
    }
    g.font = '700 12px system-ui, sans-serif'; g.fillStyle = '#94a3b8'; g.textAlign = 'center';
    for (let s = 0; s <= tEnd / 1000; s += 2) g.fillText(`${s}`, xOf(s * 1000), h - 4);
    let lastX = -1e9;
    for (const td of tds) {
      const x = xOf(td.tt * 1000);
      g.strokeStyle = td.d.uncertain ? '#fde68a' : '#fbbf24'; g.lineWidth = 2.5; g.setLineDash(td.d.uncertain ? [5, 4] : []);
      g.beginPath(); g.moveTo(x, y0 - 4); g.lineTo(x, y1); g.stroke(); g.setLineDash([]);
      const ns = (td.d.sources || []).length;
      const txt = `👣${td.d.n} ${td.d.label}${ns > 1 ? ` · ${ns} sources` : ''}`;
      g.font = '800 13px system-ui, sans-serif'; const tw = g.measureText(txt).width;
      const lx = Math.min(Math.max(x - 8, lastX + 8), w - tw - 6); lastX = lx + tw;
      g.textAlign = 'left'; g.fillStyle = td.d.uncertain ? '#fde68a' : '#fef3c7'; g.fillText(txt, lx, 17);
    }
    const px = xOf(playhead); g.fillStyle = '#facc15'; g.fillRect(px - 2.5, 4, 5, h - 8);
  }
  const fromPointer = e => { const r = tl.getBoundingClientRect(); seek(((e.clientX - r.left - pad.l) / (r.width - pad.l - pad.r)) * tEnd); };
  tl.addEventListener('pointerdown', e => {
    const r = tl.getBoundingClientRect();
    const hit = e.clientY - r.top < 30 && tds.find(td => Math.abs(xOf(td.tt * 1000) - (e.clientX - r.left)) < 60);
    if (hit) { instantReplay(hit); return; }
    tl.setPointerCapture(e.pointerId); replay = null; setPlaying(false); fromPointer(e);
  });
  tl.addEventListener('pointermove', e => { if (tl.hasPointerCapture(e.pointerId)) fromPointer(e); });

  // --- instant replay: slow motion, swing to the side, freeze on the key frame with the real numbers
  function defaultEvent() { // the landing with the deepest margin (the step-down on IMG_9691)
    return tds.filter(td => Number.isFinite(td.d.min_margin_cm)).sort((a, b) => a.d.min_margin_cm - b.d.min_margin_cm)[0] || tds[0];
  }
  function keyFrameFor(td) { // deepest margin in the 1.5 s before the landing
    let best = null;
    if (margins) for (let i = 0; i < T.length; i++) {
      const s = T[i] / 1000; if (s < td.tt - 1.5 || s > td.tt + 0.2 || !Number.isFinite(margins[i])) continue;
      if (best === null || margins[i] < margins[best]) best = i;
    }
    return best === null ? td.tt * 1000 : T[best];
  }
  function instantReplay(td) {
    if (!td) return;
    const keyMs = keyFrameFor(td), exit = td.d.com_exit_ms !== undefined ? rel(td.d.com_exit_ms) * 1000 : keyMs;
    call.style.display = 'none';
    seek(Math.max(0, Math.min(exit, keyMs) - 1800));
    setSpeed(0.25); smooth = true; setPlaying(true); setView('side');
    replay = { td, keyMs };
  }
  function showCallout(td) {
    const d = td.d, m = d.min_margin_cm;
    const head = Number.isFinite(d.lead_s)
      ? `COM leaves BOS · ${d.lead_s.toFixed(1)} s before ${d.side} foot lands · margin ${m.toFixed(1)} cm`
      : `${d.label} · margin ${Number.isFinite(m) ? (m >= 0 ? '+' : '') + m.toFixed(1) + ' cm' : '–'}`;
    const res = d.resolution_s ? `landing ±${d.resolution_s.toFixed(2)} s` : '';
    const srcs = (d.sources || []).map(x => `${x.source} ${rel(x.t_ms).toFixed(2)} s`).join(' · ');
    call.innerHTML = `${head}<small>${td.tt.toFixed(2)} s into the trial${res ? ' · ' + res : ''}${d.uncertain ? ' · uncertain' : ''}`
      + `${srcs ? `<br>sources: ${srcs}` : ''}</small>`;
    const b = el('button', '', 'Continue ▶'); b.onclick = () => { setSpeed(0.5); smooth = false; setPlaying(true); setView('front'); };
    call.append(b); call.style.borderColor = m < 0 ? '#ef4444' : '#f59e0b'; call.style.display = 'block';
  }

  const onKey = e => {
    if (e.target.closest?.('input,textarea')) return;
    if (e.code === 'Space') { e.preventDefault(); replay = null; setPlaying(!playing); }
    else if (e.key === '1') setView('front'); else if (e.key === '2') setView('side'); else if (e.key === '3') setView('top');
    else if (e.key === 'r' || e.key === 'R') instantReplay(defaultEvent());
    else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      replay = null; setPlaying(false); const [i] = frameAt(playhead);
      seek(T[Math.min(Math.max(i + (e.key === 'ArrowRight' ? 1 : -1), 0), T.length - 1)]);
    }
  };
  addEventListener('keydown', onKey);

  // --- per-frame update
  const comPos = new THREE.Vector3();
  function showFrame(i, u) {
    if (smooth && u > 0 && i < data.frames.length - 1) {
      const F = data.frames, n = F.length;
      catmullRom(F[Math.max(i - 1, 0)], F[i], F[i + 1], F[Math.min(i + 2, n - 1)], u, positions);
    } else if (i !== lastShown || smooth) positions.set(data.frames[i]); else return;
    lastShown = smooth ? -1 : i;
    geo.attributes.position.needsUpdate = true; geo.computeVertexNormals();
  }
  function updateOverlays(i, u) {
    curFrame = smooth && u > 0.5 && i < T.length - 1 ? i + 1 : i;
    const j = curFrame;
    if (hasCom) {
      const a = meta.com[i], b = meta.com[Math.min(i + 1, T.length - 1)], s = smooth ? u : 0;
      comPos.set(lerp(a[0], b[0], s), lerp(a[1], b[1], s), lerp(a[2], b[2], s));
      const col = new THREE.Color(margins[j] < 0 ? C.red : C.green); // inside BOS green, outside red
      comSphere.position.copy(comPos); comGlow.position.copy(comPos);
      plumb.position.set(comPos.x, (comPos.y + floorY) / 2, comPos.z); plumb.scale.y = Math.max(comPos.y - floorY, 0.01);
      plumbDot.position.set(comPos.x, oY + 0.002, comPos.z); plumbRing.position.copy(plumbDot.position);
      for (const o of [comSphere, plumb, plumbDot, comGlow]) o.material.color.copy(col);
      trail.forEach((mm, k) => { const q = j - k - 1; mm.visible = q >= 0; if (q >= 0) mm.position.set(meta.com[q][0], oY, meta.com[q][2]); });
    }
    if (hasBos) {
      const h = meta.bos[j] || [];
      bosEdges.forEach((r, k) => { if (k < h.length) placeRibbon(r, h[k], h[(k + 1) % h.length], oY); else r.visible = false; });
      if (h.length >= 3) {
        const pos = []; let cx = 0, cz = 0, x0 = 1e9, x1 = -1e9, z0 = 1e9, z1 = -1e9;
        for (let k = 1; k < h.length - 1; k++) pos.push(h[0][0], oY, h[0][1], h[k][0], oY, h[k][1], h[k + 1][0], oY, h[k + 1][1]);
        for (const [x, z] of h) { cx += x / h.length; cz += z / h.length; x0 = Math.min(x0, x); x1 = Math.max(x1, x); z0 = Math.min(z0, z); z1 = Math.max(z1, z); }
        bosFill.geometry.dispose(); bosFill.geometry = new THREE.BufferGeometry();
        bosFill.geometry.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); bosFill.visible = true;
        shadow.position.set(cx, floorY + 0.001, cz); shadow.scale.set((x1 - x0) * 1.8 + 0.08, (z1 - z0) * 1.5 + 0.08, 1); shadow.visible = true;
      } else { bosFill.visible = false; shadow.visible = false; }
    }
    const s = playhead / 1000;
    for (const td of tds) { // footprints persist; ripple for 1.6 s after the landing
      const on = s >= td.tt; td.items.forEach(o => (o.visible = on));
      if (td.label) td.label.visible = on; if (td.num) td.num.visible = on;
      if (td.ripple) {
        const age = s - td.tt, live = age >= 0 && age < 1.6;
        td.ripple.visible = live;
        if (live) { const r = 0.05 + age * 0.13; td.ripple.scale.set(r, r, 1); td.ripple.material.opacity = 0.95 * (1 - age / 1.6); }
      }
    }
    if (margins) {
      const m = margins[j], S = data.series;
      rv.margin.textContent = Number.isFinite(m) ? `${m >= 0 ? '+' : ''}${(m * 100).toFixed(1)} cm` : '–';
      rv.margin.style.color = cssColor(m);
      rv.stance.textContent = S?.stance ? (STANCE_TXT[S.stance[j]] || S.stance[j]) : '–';
      rv.lean.textContent = S?.trunk_ml_deg ? `side ${S.trunk_ml_deg[j].toFixed(0)}° · fwd ${S.trunk_ap_deg[j].toFixed(0)}° (est.)` : '–';
      rv.out.textContent = S?.outside_cum_s ? `${S.outside_cum_s[j].toFixed(1)} s` : '–';
    }
    updatePip(j);
  }

  // --- render loop; a manual clock gives frame-exact video capture
  function resize() {
    const w = stage.clientWidth, h = stage.clientHeight; if (!w || !h) return;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); pipShown = -1;
  }
  const ro = new ResizeObserver(resize); ro.observe(stage); resize();
  let manual = false, virtualNow = 0;
  function clock() { return manual ? virtualNow : performance.now(); }
  applyView();
  function render() {
    const w = stage.clientWidth, h = stage.clientHeight;
    renderer.setScissorTest(false); renderer.setViewport(0, 0, w, h); renderer.clear();
    camera.layers.set(0); renderer.render(scene, camera);
    if (mini.style.display === 'none') return;
    const x = w - 14 - 3 - MINI, y = 12 + 3; // bottom-right, inside the border
    renderer.setScissorTest(true); renderer.setScissor(x, y, MINI, MINI); renderer.setViewport(x, y, MINI, MINI);
    renderer.setClearColor(0x020617, 1); renderer.clear();
    const hidden = [mesh, ...bodyOnly, ...tds.map(td => td.label).filter(Boolean), ...ghosts].filter(o => o.visible);
    hidden.forEach(o => (o.visible = false));
    renderer.render(scene, miniCam);
    hidden.forEach(o => (o.visible = true));
    renderer.setScissorTest(false); renderer.setClearColor(C.bg, 1);
  }
  function tick(dt) {
    if (playing) { playhead += dt * speed; if (playhead > tEnd) playhead = replay ? tEnd : 0; }
    if (replay && playhead >= replay.keyMs) { playhead = replay.keyMs; const td = replay.td; replay = null; setPlaying(false); showCallout(td); }
    if (tween) {
      const x = Math.min((clock() - tween.t0) / tween.ms, 1), e = ease(x);
      for (const k of ['az', 'el', 'r', 'tx', 'ty', 'tz', 'op']) view[k] = tween.from[k] + (tween.to[k] - tween.from[k]) * e;
      if (x >= 1) tween = null;
    }
    applyView();
    const [i, u] = frameAt(playhead);
    showFrame(i, u); updateOverlays(i, u);
    timeLabel.textContent = `${(playhead / 1000).toFixed(2)} / ${(tEnd / 1000).toFixed(1)} s`;
    drawTimeline(); render();
  }
  let last = performance.now(), raf = 0;
  function loop(now) { tick(now - last); last = now; raf = requestAnimationFrame(loop); }
  raf = requestAnimationFrame(loop);
  if (jumpTo === 'event' || jumpTo === 'step') setTimeout(() => instantReplay(defaultEvent()), 400);

  return {
    seek: ms => seek(ms - t0), seekTrial: s => seek(s * 1000), play: () => setPlaying(true), pause: () => { replay = null; setPlaying(false); },
    setView, instantReplay: n => instantReplay(n === undefined ? defaultEvent() : tds.find(td => td.d.n === n)),
    toggle: name => [...ctl.querySelectorAll('button')].find(b => b.textContent === name)?.click(),
    get view() { return viewName; }, get frame() { return curFrame; },
    capture: { // deterministic stepping for video export: stops the RAF loop and advances a virtual clock
      start() { cancelAnimationFrame(raf); manual = true; virtualNow = performance.now(); },
      step(dtMs) { virtualNow += dtMs; tick(dtMs); },
      stop() { manual = false; last = performance.now(); raf = requestAnimationFrame(loop); },
    },
    destroy() { cancelAnimationFrame(raf); ro.disconnect(); removeEventListener('keydown', onKey); renderer.dispose(); pipR.dispose(); modal.remove(); element.replaceChildren(); },
  };
}
