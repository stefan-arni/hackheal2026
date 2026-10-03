# Instant Replay — SAM 3D Body lane

Owner: Stefan . Lives in `/replay`. Everything here runs **in the background** and must never block the core product (MediaPipe + optical flow sway metrics, error counts, convergence). If this lane fails, the core still works.

## What it is

After each balance trial, the doctor gets a 3D replay of the patient: full-body mesh from Meta's SAM 3D Body, gravity-aligned floor, center of mass (COM) trail, base of support (BOS), and error moments marked on a timeline. It also produces extra 3D posture data the 2D pipeline can't: whole-body COM relative to BOS, stance verification, trunk/joint angles, and an *estimated* view of forward/back sway.

SAM is **not** the primary sway signal. The 30 Hz MediaPipe pipeline owns the clinical numbers.

### The demo money shot (optimize for this)
Tandem stance, eyes closed, foot touches down at ~12 s → trial ends → replay ready a few seconds later → doctor clicks the error tick → camera swings to side view → COM dot drifts to the edge of the BOS and turns red *just before* the foot comes down. Shown live over Zoom screen share.

## Delivery tiers & checkpoints

| Tier | Contents | Done by |
|---|---|---|
| 0 | Recorded clip → frames → fal → meshes → three.js flipbook with timeline. No teammate dependencies. | hour 4 |
| 1 | Live: frames stream in during trial (via fake phone first), replay ready seconds after trial ends. Floor alignment, metric scale, smoothing, interpolation, event ticks. | hour 8 |
| 2 | COM, BOS hull, stability margin, sway heatmap, side view, SAM noise floor. | hour 10 |
| 3 (stretch) | Extrapolated COM / margin of stability, inverted-pendulum index, previous-trial overlay, joint angles at errors. | if time |

Rule: cut scope, don't slip. Cached replays of demo trials exist by hour 10 regardless.

**Decision gate (hour 1):** reconstruct ~20 frames of a still stance; measure foot-vertex jitter per axis. Depth jitter < ~1 cm → forward/back sway + full stability margin are real features. Several cm → AP is labeled "estimated" and visualization-only; side-to-side metrics still valid.

## Folder layout

```
replay/
  service/        FastAPI app: ingest frames, call fal, post-process, serve bundles
    app.py
    fal_client_wrap.py
    geometry.py   all math (pure functions, unit-tested)
    bundle.py     pack/unpack replay bundles
  viewer/         self-contained three.js module the dashboard mounts
  tools/
    fake_phone.py replays a recorded clip's frames to the service at ~3 fps with timestamps
    extract_frames.sh  ffmpeg -vf fps=3
  notebooks/      CPU-only math exploration (Colab or local), no CUDA
  cache/          precomputed bundles for demo fallback (gitignored except demo bundles)
  tests/
```

`FAL_KEY` lives in `replay/.env` — **gitignored, never committed** (shared repo). The key never reaches the phone or browser.

## Interfaces (contracts with the team)

### In: from the phone/capture page
`POST /replay/{trialId}/frame` (multipart)
- `jpeg`: person crop, **same fixed crop box for the whole trial** (from the first ~1 s, generous margin). Never crop per frame — it creates fake translation.
- `mask` (optional): binary PNG from MediaPipe segmentation (white = person). Passed to SAM as `mask_url` so it skips its own detection and locks onto the right person.
- `t`: capture timestamp in ms, **phone clock** (same clock as the metrics timeline).
- `crop`: `[x0, y0, x1, y1]` in full-frame pixels; `frame_size`: `[W, H]`.
- `kind`: `"uniform"` (~3 fps) or `"burst"` (±0.5 s around an error at ~10 fps, from the phone's 1 s ring buffer).
- optional `gravity`: phone accelerometer gravity vector (cross-check for floor fit).

`POST /replay/{trialId}/end` (JSON)
- `events`: `[{t, kind, side}]` from the metrics pipeline (foot_down, step, hands_off_hips, …)
- `patient_height_cm` from pre-flight
- optional `landmarks_2d`: MediaPipe 2D timeline for reprojection-based outlier rejection

### Out: to the dashboard
- WebSocket message via the team relay: `{ "type": "replay_ready", "trialId": "...", "url": "/replay/<trialId>/" }`
- `GET /replay/{trialId}/meta.json` and `GET /replay/{trialId}/verts.bin`
- Viewer module: `mountReplay(element, bundleUrl, { onSeek })`

### Bundle format
- `meta.json`: `{ trialId, t: [F], faces_count, vertex_count, events, com: [F][3], bos: [F][[x,z]...], margin: [F], heatmap: [V], noise_floor: {x,y,z}, quality: {...}, frames: [{t, fal_vis_url}] }`
- `faces.bin`: Uint32 `[faces_count*3]`
- `verts.bin`: Float32 `[F * V * 3]`, gravity-aligned, metric (meters), origin at stance-foot center. Switch to Float16 if too large.

## fal integration

- Endpoint: `fal-ai/sam-3/3d-body` (~$0.015 per image).
- Inputs: `image_url` (base64 data URI is fine), `mask_url` (optional), `export_meshes: true`, `include_3d_keypoints: false` (otherwise marker spheres get baked into the mesh), `include_mhr_params: false` (lean metadata: keypoints + camera only).
- Outputs used: `meshes[]` (.ply per person), `visualization` (original + keypoints + mesh + side view image — use as a debug/trust panel), `metadata.num_people`, `metadata.keypoint_names` (70 MHR keypoints), and per person: `bbox`, `focal_length`, `pred_cam_t [tx,ty,tz]`, `keypoints_2d`, `keypoints_3d`.
- Use the Python async client with an `asyncio.Semaphore(8)`; check the account's concurrency limit. Verify exact client function names against fal's docs.
- **Pipelining:** submit each frame the moment it arrives *during* the trial, download the .ply concurrently. On `/end`, await stragglers, post-process, publish.
- Drop any frame where `num_people != 1`.
- Send one warm-up request right before the demo (cold starts).

## Math pipeline (geometry.py, in order)

1. **Convention check (hour 1).** Project `keypoints_3d` with pinhole `u = f·X/Z + W/2`, `v = f·Y/Z + H/2` and compare to `keypoints_2d`. If they match as-is, 3D points are already in camera space; if they match only after adding `pred_cam_t`, they're body-relative. Also confirm axis signs (y-down?) and mesh vertices' convention.
2. **Focal normalization.** Camera is fixed, but SAM estimates focal per frame and focal↔depth are coupled. Use median focal `f̄`; per frame `tz' = tz · f̄ / f_i` (tx, ty unchanged).
3. **Outlier rejection.** Reprojection error vs MediaPipe 2D landmarks at nearest timestamp; drop frames above threshold.
4. **Floor fit + gravity alignment.** Stack heel/toe keypoints from stance frames (exclude frames near error events). Center, SVD; normal **n** = right-singular vector with smallest singular value, flipped to point toward the head. Rotate so **n → ŷ** (Rodrigues): `k = (n×ŷ)/|n×ŷ|`, `θ = arccos(n·ŷ)`, `R = I + sinθ·K + (1−cosθ)·K²`. Origin = mean stance-foot center on the floor. Optional cross-check with phone gravity vector.
5. **Metric scale.** `s = H_patient / H_mesh`, `H_mesh` = median floor-to-head-top height. Scale all positions by `s`.
6. **Temporal smoothing** (non-uniform timestamps because of bursts): `V̂_i = Σ_j w_ij V_j / Σ_j w_ij`, `w_ij = exp(−(t_i−t_j)²/2σ²)`, σ ≈ 0.3 s. Mesh topology is fixed across frames — verify vertex count is identical.
7. **COM** per frame: `trimesh.Trimesh(V, F).center_mass` (volumetric, uniform density; discrete divergence theorem via signed tetrahedra: volume `a·(b×c)/6`, centroid `(a+b+c)/4`). Check `is_watertight`.
8. **BOS:** foot vertices within ~1.5 cm of the floor → project to floor → `scipy.spatial.ConvexHull`.
9. **Stability margin:** signed distance from COM floor projection to nearest hull edge (positive inside). Report side-to-side and forward/back components separately.
10. **Sway heatmap:** per-vertex RMS displacement from trial-mean position → vertex colors. Exploratory: displacement growing linearly with height ≈ ankle strategy; trunk/pelvis counter-motion ≈ hip strategy.
11. **Noise floor:** foot-vertex jitter during stance (feet are static) per axis → reported with every metric.
12. *(Stretch)* Extrapolated COM (Hof): `ξ = x + ẋ/ω₀`, `ω₀ = √(g/ℓ)`, ℓ = COM height; use side-to-side velocity from the 30 Hz metrics stream (3 fps is too coarse). Joint angles / trunk lean from `keypoints_3d`.

All of geometry.py is pure functions with unit tests on synthetic data (known plane, known rotation, a cube/sphere for COM, a square for the hull).

## Viewer (viewer/)

- three.js; one `BufferGeometry` whose position attribute is updated every animation frame with **Catmull-Rom interpolation** between keyframes (smooth 60 fps playback).
- Floor grid, BOS polygon, fading COM trail on the floor, COM projection dot colored by stability margin (green → red).
- Camera presets: **front** (matches video), **side** (forward/back sway, labeled "estimated" if the gate says so), **top-down** (3D stabilogram). Slow eased transitions.
- Onion skin: ~8 faint ghost meshes = sway envelope. Heatmap toggle.
- Timeline scrubber with event ticks; clicking a tick seeks to the burst around that error. 0.5× playback default.
- Debug panel: fal `visualization` image for the current frame.
- "Load cached replay" fallback button.
- **Zoom-share friendly:** bold colors, thick lines, no fine detail, slow camera moves. Test performance on the demo laptop *while screen sharing in Zoom*.

## Gotchas

- Coordinate conventions (y-down, mirrored front camera, crop offsets) — do the convention check first.
- Fixed crop per trial; phone-clock timestamps everywhere.
- Don't wait on the capture teammate: build and test the live tier with `tools/fake_phone.py`.
- Privacy: this is the only lane where frames leave the phone. Fixed crop + mask minimize what's sent; pitched as opt-in "3D review".
- Demo network: everything on a hotspot.

## Hour-one checklist

1. fal key in `replay/.env` (gitignored).
2. Record a 20 s tandem stance on a phone with deliberate sway and a foot-down at ~12 s; `ffmpeg -vf fps=3` → frames.
3. Send 3 frames through fal; inspect metadata, latency, vertex count; run the convention check.
4. Send ~20 still-stance frames; measure per-axis foot jitter → decision gate.
5. Start Tier 0.
