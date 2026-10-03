#!/usr/bin/env bash
# Live demo chain: iPhone -> posecam (8765) -> replay service (8017) -> fal -> doctor dashboard.
#
#   ./run_demo.sh --mock                 # rehearsal: fal mocked (free), realistic latency model
#   ./run_demo.sh --live --warmup        # the real thing: billed fal calls ($0.018 each)
#   ./run_demo.sh                        # cache-only: frames not already in data/fal_cache are skipped
#
# Options:
#   --live | --mock     fal mode (default cache-only). --live asks before starting.
#   --warmup            one fal call before the session to avoid a cold start (asks first when live)
#   --eyes              also start posecam's eye server (8766)
#   --tunnel            start a cloudflared quick tunnel and print the public dashboard URL
#   --height CM         patient height for metric scaling (default 185; 0 = SAM's own scale)
#   --name TEXT         session name (shown on the session report)
#   --duration S        BESS seconds per stance (default 10: the shortened demo protocol)
#   --countdown S       seconds from pressing a stance in the app to scoring (default 8: walk back into position)
#   --deadline S        publish each trial's replay this long after it ends (default 30)
#   --max-usd X         hard stop: the replay service makes at most X dollars of fal calls (incl. warm-up)
#   --yes               don't ask (live start / warm-up); only after the spend was approved
#   --straggler N:S     mock only: the N-th fal call takes S seconds (default 12:150)
# Ctrl-C stops everything. Logs: replay/data/logs/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPLAY="$ROOT/replay"; POSECAM="$ROOT/posecam"
MODE=cache; WARMUP=0; EYES=0; TUNNEL=0; HEIGHT=185; NAME="demo"; DURATION=10; COUNTDOWN=8; DEADLINE=30; STRAGGLER="12:150"; MAXUSD=""; YES=0
REPLAY_PORT=8017; POSE_PORT=8765; EYE_PORT=8766
while [[ $# -gt 0 ]]; do
  case "$1" in
    --live) MODE=live ;; --mock) MODE=mock ;; --warmup) WARMUP=1 ;; --eyes) EYES=1 ;; --tunnel) TUNNEL=1 ;;
    --height) HEIGHT="$2"; shift ;; --name) NAME="$2"; shift ;; --duration) DURATION="$2"; shift ;;
    --deadline) DEADLINE="$2"; shift ;; --straggler) STRAGGLER="$2"; shift ;;
    --max-usd) MAXUSD="$2"; shift ;; --yes) YES=1 ;; --countdown) COUNTDOWN="$2"; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac; shift
done

LOGS="$REPLAY/data/logs"; mkdir -p "$LOGS"
PIDS=()
cleanup() { trap - INT TERM EXIT; echo; echo "stopping…"; for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup INT TERM EXIT

for port in $REPLAY_PORT $POSE_PORT; do
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then echo "port $port is already in use (old demo still running?)" >&2; exit 1; fi
done

# ---- replay service
ENV=(REPLAY_DEADLINE_S="$DEADLINE" REPLAY_FAL_WORKERS=3)
[[ -n "$MAXUSD" ]] && ENV+=(REPLAY_MAX_USD="$MAXUSD")
case "$MODE" in
  live)
    (cd "$REPLAY" && uv run tools/fal_budget.py status 2>/dev/null | tail -3) || true
    ok=y
    [[ $YES == 1 ]] || read -r -p "fal LIVE: every frame is a billed call (\$0.018; ~50-60 frames per 3-trial session ≈ \$1). Start? [y/N] " ok
    [[ "$ok" == y* ]] || exit 1
    ENV+=(REPLAY_LIVE=1 REPLAY_YES=1) ;;
  mock)
    ENV+=(REPLAY_MOCK_FAL=data/fal_out/demo,data/fal_out/ppcheck REPLAY_MOCK_LATENCY=4.4 REPLAY_MOCK_PARALLEL=3
          REPLAY_MOCK_STRAGGLER="$STRAGGLER") ;;
esac
echo "starting replay service on :$REPLAY_PORT (fal: $MODE)…"
(cd "$REPLAY" && env "${ENV[@]}" uv run uvicorn service.app:app --host 0.0.0.0 --port $REPLAY_PORT >"$LOGS/replay.log" 2>&1) &
PIDS+=($!)
for _ in $(seq 60); do curl -sf "http://localhost:$REPLAY_PORT/viewer/" >/dev/null 2>&1 && break; sleep 0.5; done
curl -sf "http://localhost:$REPLAY_PORT/viewer/" >/dev/null || { echo "replay service did not start; see $LOGS/replay.log" >&2; tail -20 "$LOGS/replay.log" >&2; exit 1; }

SID=$(curl -sf -X POST "http://localhost:$REPLAY_PORT/session" -H 'Content-Type: application/json' -d "{\"name\": \"$NAME\"}" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["sessionId"])')

# ---- fal warm-up (one call)
if [[ $WARMUP == 1 ]]; then
  FRAME=$(ls "$REPLAY"/data/frames/demo/*.jpg 2>/dev/null | head -1 || true)
  if [[ -z "$FRAME" ]]; then echo "warm-up skipped: no frame in replay/data/frames/demo"; else
    go=y
    if [[ $MODE == live && $YES != 1 ]]; then read -r -p "Send 1 fal warm-up call (\$0.018)? [y/N] " go; fi
    if [[ $MODE == cache ]]; then echo "warm-up skipped: cache-only mode never calls fal"; go=n; fi
    if [[ "$go" == y* ]]; then
      echo -n "warm-up: "; curl -sf -X POST "http://localhost:$REPLAY_PORT/replay/warmup?force=1" -F "jpeg=@$FRAME;type=image/jpeg" || echo "failed (see $LOGS/replay.log)"; echo
    fi
  fi
fi

# ---- posecam (MediaPipe 0.10.21 on CPU: newer builds abort without a Metal GPU service)
PC_ARGS=(--port $POSE_PORT --bess-duration "$DURATION" --bess-countdown "$COUNTDOWN" --replay-url "http://localhost:$REPLAY_PORT"
         --publish-url "http://localhost:$REPLAY_PORT" --session-id "$SID")
[[ "$HEIGHT" != 0 ]] && PC_ARGS+=(--patient-height-cm "$HEIGHT")
UVP=(uv run --no-project --python 3.12 --with "mediapipe==0.10.21" --with "numpy<2" --with "opencv-python-headless>=4.9,<4.11" --with "websockets>=13")
echo "starting posecam on :$POSE_PORT…"
(cd "$POSECAM" && "${UVP[@]}" python server.py "${PC_ARGS[@]}" >"$LOGS/posecam.log" 2>&1) &
PIDS+=($!)
if [[ $EYES == 1 ]]; then
  (cd "$POSECAM" && "${UVP[@]}" python eye_server.py --port $EYE_PORT >"$LOGS/eyes.log" 2>&1) &
  PIDS+=($!)
fi
for _ in $(seq 120); do lsof -nP -iTCP:$POSE_PORT -sTCP:LISTEN >/dev/null 2>&1 && break; sleep 0.5; done
lsof -nP -iTCP:$POSE_PORT -sTCP:LISTEN >/dev/null 2>&1 || { echo "posecam did not start; see $LOGS/posecam.log" >&2; tail -20 "$LOGS/posecam.log" >&2; exit 1; }

# ---- optional public tunnel for the doctor
PUBLIC=""
if [[ $TUNNEL == 1 ]]; then
  if command -v cloudflared >/dev/null; then
    cloudflared tunnel --no-autoupdate --url "http://localhost:$REPLAY_PORT" >"$LOGS/tunnel.log" 2>&1 &
    PIDS+=($!)
    for _ in $(seq 40); do PUBLIC=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOGS/tunnel.log" | head -1 || true); [[ -n "$PUBLIC" ]] && break; sleep 0.5; done
  else echo "cloudflared not installed (brew install cloudflared); no public URL"; fi
fi

IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}' || echo "<laptop IP>")
cat <<EOF

================================================================
 session          $SID   (fal: $MODE, $COUNTDOWN s countdown + $DURATION s per stance, replay deadline $DEADLINE s${MAXUSD:+, spend cap \$$MAXUSD})
 doctor dashboard http://localhost:$REPLAY_PORT/dashboard/$SID
EOF
[[ -n "$PUBLIC" ]] && echo " public dashboard $PUBLIC/dashboard/$SID"
cat <<EOF
 session report   http://localhost:$REPLAY_PORT/session/$SID/report
 on the same Wi-Fi: http://$IP:$REPLAY_PORT/dashboard/$SID

 iPhone (posecam iOS test screen, BessTestView):
   Laptop IP  $IP      port $POSE_PORT      (phone and laptop on the same Wi-Fi)
   then: Double -> Tandem -> Single, ~15 s apart

 rehearsal without the phone:
   cd replay && uv run tools/rehearse_protocol.py --session $SID --shots data/reports/rehearsal
 logs: $LOGS/{replay,posecam}.log      Ctrl-C stops everything
================================================================
EOF
wait
