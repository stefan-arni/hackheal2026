#!/usr/bin/env bash
# Extract frames from a recorded clip at a fixed rate.
# Usage: tools/extract_frames.sh clip.mp4 [out_dir=data/frames] [fps=3]
# Frame k (1-based) was captured at t = (k-1)/fps seconds.
set -euo pipefail

clip="${1:?usage: extract_frames.sh clip.mp4 [out_dir] [fps]}"
out="${2:-data/frames}"
fps="${3:-3}"

mkdir -p "$out"
ffmpeg -hide_banner -loglevel error -i "$clip" -vf "fps=${fps}" -q:v 2 "$out/frame_%06d.jpg"
echo "wrote $(ls "$out" | wc -l | tr -d ' ') frames to $out at ${fps} fps"
