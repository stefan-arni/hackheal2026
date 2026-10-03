"""Capture and fal-submission defaults (measured against fal-ai/sam-3/3d-body, Oct 2026).

Latency: ~6.2 s per full-resolution frame (≈92% fal inference), concurrency 10 ->
~1.6 frames/s. tools/live_timing.py: at 1.5 fps uniform with error bursts sent first, the
error replay is ready ~5 s and the full replay ~12 s after a 20 s trial ends.
Downscaling to 1024 px tall cut inference ~23% (5.8 -> 4.4 s) with pose within 0.6 cm and
foot keypoints within 1.5 px of full resolution. SAM's focal (and absolute depth) shifts with
input size, so never mix resolutions within one trial.
"""

LIVE_UNIFORM_FPS = 1.5  # live trials: uniform frames per second
BURST_FPS = 10.0  # error bursts from the phone's ring buffer
BURST_HALF_S = 0.5  # burst covers error ± this many seconds
LIVE_MAX_HEIGHT_PX = 1024  # live trials: frames taller than this are downscaled before fal
OFFLINE_UNIFORM_FPS = 3.0  # recorded / offline segments: full resolution, 3 fps
