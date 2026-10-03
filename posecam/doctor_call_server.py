from __future__ import annotations

import json
import uuid
from pathlib import Path

from aiohttp import web, WSMsgType


# ============================================================
# CONFIG
# ============================================================

HOST = "0.0.0.0"
PORT = 8088

BASE_DIR = Path(__file__).resolve().parent



# ============================================================
# RUNTIME STATE
# ============================================================

rooms = {}

active_tests = {}

analysis_tasks = set()


def get_room(
    room_id,
):

    if room_id not in rooms:

        rooms[room_id] = {
            "doctor": None,
            "patient": None,
        }

    return rooms[room_id]


async def send_json_safe(
    ws,
    payload,
):

    if ws is None:
        return

    if ws.closed:
        return

    try:

        await ws.send_json(
            payload
        )

    except Exception as exc:

        print(
            "WebSocket send error:",
            exc,
        )


async def send_to_other(
    room_id,
    sender_role,
    payload,
):

    room = get_room(
        room_id
    )

    other_role = (
        "doctor"
        if sender_role == "patient"
        else "patient"
    )

    await send_json_safe(
        room.get(other_role),
        payload,
    )


# ============================================================
# WEBSOCKET
# ============================================================

async def websocket_handler(
    request,
):

    ws = web.WebSocketResponse(
        heartbeat=20
    )

    await ws.prepare(
        request
    )

    room_id = None
    role = None

    try:

        async for message in ws:

            if (
                message.type
                != WSMsgType.TEXT
            ):
                continue

            try:

                data = json.loads(
                    message.data
                )

            except Exception:

                continue

            message_type = data.get(
                "type"
            )

            # -----------------------------------------------
            # JOIN
            # -----------------------------------------------

            if message_type == "join":

                room_id = str(
                    data.get(
                        "room",
                        "television-demo",
                    )
                )

                role = str(
                    data.get(
                        "role",
                        "",
                    )
                )

                if role not in (
                    "doctor",
                    "patient",
                ):
                    continue

                room = get_room(
                    room_id
                )

                old = room.get(
                    role
                )

                if (
                    old is not None
                    and old is not ws
                    and not old.closed
                ):

                    try:
                        await old.close()
                    except Exception:
                        pass

                room[role] = ws

                other = room.get(
                    "patient" if role == "doctor" else "doctor"
                )

                await send_json_safe(
                    ws,
                    {
                        "type":
                            "joined",

                        "room":
                            room_id,

                        "role":
                            role,

                        # is the other side already in the room?
                        # (so the doctor can call a patient who
                        # joined first)
                        "peer_present":
                            other is not None
                            and not other.closed,
                    },
                )

                await send_to_other(
                    room_id,
                    role,
                    {
                        "type":
                            "peer_joined",

                        "role":
                            role,
                    },
                )

                continue

            if (
                room_id is None
                or role is None
            ):
                continue

            # -----------------------------------------------
            # WEBRTC
            # -----------------------------------------------

            if message_type in (
                "offer",
                "answer",
                "ice",
            ):

                await send_to_other(
                    room_id,
                    role,
                    data,
                )

                continue

            # -----------------------------------------------
            # START TEST
            # -----------------------------------------------

            if (
                message_type
                == "start_test"
                and role == "doctor"
            ):

                test_type = str(
                    data.get(
                        "test",
                        "",
                    )
                )

                if test_type not in (
                    "finger",
                    "balance",
                    "eyes",
                ):
                    continue

                # Live only: nothing is stored, the id just
                # labels this test for the phone.
                session_id = uuid.uuid4().hex

                print(
                    "Starting test:",
                    test_type,
                    "room:",
                    room_id,
                    "session:",
                    session_id,
                )

                await send_to_other(
                    room_id,
                    role,
                    {
                        "type": "start_test",
                        "test": test_type,
                        "session_id": session_id,
                    },
                )

                await send_json_safe(
                    ws,
                    {
                        "type": "test_started",
                        "test": test_type,
                        "session_id": session_id,
                    },
                )

                continue

            # -----------------------------------------------
            # STOP TEST
            # -----------------------------------------------

            if (
                message_type
                == "stop_test"
                and role == "doctor"
            ):

                # Tell phone first so it starts
                # finalizing the recording.

                await send_to_other(
                    room_id,
                    role,
                    {
                        "type":
                            "stop_test"
                    },
                )

                continue

            # -----------------------------------------------
            # ANALYZER COMMAND
            # -----------------------------------------------

            if (
                message_type
                == "server_command"
                and role == "doctor"
            ):

                await send_to_other(
                    room_id,
                    role,
                    data,
                )

                continue

            # -----------------------------------------------
            # PATIENT MEASUREMENT
            # -----------------------------------------------

            if (
                message_type
                == "measurement"
                and role == "patient"
            ):

                await send_json_safe(
                    get_room(
                        room_id
                    ).get("doctor"),

                    data,
                )

                continue

    finally:

        if (
            room_id is not None
            and role is not None
        ):

            room = get_room(
                room_id
            )

            if room.get(role) is ws:

                room[role] = None

            await send_to_other(
                room_id,
                role,
                {
                    "type":
                        "peer_left",

                    "role":
                        role,
                },
            )

    return ws


# ============================================================
# DOCTOR DASHBOARD
# ============================================================

DOCTOR_HTML = r"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>TeleVision Doctor</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: #0b0d12;
    color: white;
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

button,
input,
select {
    font: inherit;
}

header {
    height: 64px;
    display: flex;
    align-items: center;
    padding: 0 22px;
    background: #11141b;
    border-bottom: 1px solid #252a34;
}

.logo {
    font-size: 20px;
    font-weight: 800;
}

.status {
    margin-left: auto;
    color: #aeb6c5;
    font-size: 13px;
}

.tabs {
    display: flex;
    gap: 8px;
    padding: 14px 18px 0;
}

.tab {
    background: #171b24;
    color: #bfc6d2;
    border: 1px solid #292f3b;
    padding: 9px 14px;
    border-radius: 10px;
    cursor: pointer;
}

.tab.active {
    background: #2867e8;
    border-color: #2867e8;
    color: white;
}

.page {
    display: none;
}

.page.active {
    display: block;
}

.layout {
    display: grid;
    grid-template-columns:
        minmax(0, 1fr)
        350px;
    gap: 16px;
    padding: 16px 18px 22px;
}

.video-card {
    position: relative;
    background: black;
    border-radius: 18px;
    overflow: hidden;
    min-height: calc(100vh - 145px);
    border: 1px solid #252a34;
}

#patientVideo {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: contain;
    background: black;
}

#doctorVideo {
    position: absolute;
    right: 16px;
    bottom: 16px;
    width: 190px;
    aspect-ratio: 9 / 16;
    object-fit: cover;
    background: #111;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,.35);
}

.panel {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.card {
    background: #151922;
    border: 1px solid #292f3b;
    border-radius: 15px;
    padding: 14px;
}

.card h3 {
    margin: 0 0 12px;
    font-size: 14px;
}

label {
    display: block;
    color: #9fa8b8;
    font-size: 12px;
    margin-bottom: 5px;
}

input,
select {
    width: 100%;
    padding: 9px 10px;
    border-radius: 9px;
    border: 1px solid #343b48;
    background: #0e1117;
    color: white;
}

.row {
    display: flex;
    gap: 8px;
}

.row > * {
    flex: 1;
}

button {
    border: 0;
    border-radius: 9px;
    padding: 9px 12px;
    cursor: pointer;
    background: #2d3441;
    color: white;
}

button.primary {
    background: #2867e8;
}

button.danger {
    background: #c93d45;
}

button.good {
    background: #16865d;
}

.metric {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    padding: 7px 0;
    border-bottom: 1px solid #252a34;
    font-size: 13px;
}

.metric:last-child {
    border-bottom: 0;
}

.metric-name {
    color: #9fa8b8;
}

.metric-value {
    font-weight: 700;
    text-align: right;
}

@media(max-width:900px) {

    .layout {
        grid-template-columns: 1fr;
    }

    .comparison-grid {
        grid-template-columns: 1fr;
    }

    .video-card {
        min-height: 65vh;
    }
}


[hidden] { display: none !important; }   /* hidden wins over display: flex */

/* ---------- header call bar ---------- */
header { gap: 18px; }
.call-bar { display: flex; gap: 8px; align-items: center; }
.call-bar input { width: 170px; padding: 7px 9px; }
.call-bar button { padding: 7px 12px; }

/* ---------- dashboard layout: video | run | results ---------- */
.dash {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 340px 400px;
    gap: 14px;
    padding: 14px 16px 20px;
    align-items: start;
}
.dash .video-card { min-height: calc(100vh - 100px); position: sticky; top: 14px; }
.col { display: flex; flex-direction: column; gap: 12px; }
.card-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 10px; }
.card-head h3 { margin: 0; }
.hint { color: #8f98a8; font-size: 12px; margin-top: 8px; }
button.small { padding: 5px 9px; font-size: 12px; }
button:disabled { opacity: .4; cursor: default; }

/* non-dominant segmented control */
.seg { display: flex; align-items: center; gap: 4px; font-size: 12px; }
.seg-label { color: #9fa8b8; margin-right: 4px; }
.seg-btn { padding: 5px 10px; font-size: 12px; background: #222835; }
.seg-btn.active { background: #2867e8; }

/* stance buttons */
.stances { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.stance {
    display: flex; flex-direction: column; align-items: center; gap: 4px;
    padding: 12px 6px; border-radius: 12px; background: #1d2230;
    border: 2px solid transparent; text-align: center;
}
.stance:hover:not(:disabled) { background: #252c3d; }
.stance .s-name { font-weight: 700; font-size: 14px; }
.stance .s-status { font-size: 11px; color: #9fa8b8; }
.stance .s-score { font-size: 12px; font-weight: 700; }
.stance.running { border-color: #2867e8; }
.stance.done .s-status { color: #3fbf7f; }
.good-txt { color: #3fbf7f; }
.bad-txt { color: #f06a6a; }

/* live block */
.live-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; }
.live-title { font-size: 18px; font-weight: 800; }
.live-sub { color: #9fa8b8; font-size: 12px; margin-top: 2px; }
.live-timer { font-size: 30px; font-weight: 800; font-variant-numeric: tabular-nums; }
.bar { height: 8px; border-radius: 6px; background: #252b38; margin: 10px 0; overflow: hidden; }
.bar-fill { height: 100%; width: 0; background: #2867e8; transition: width .2s linear; }
.errors-row { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.live-errors { font-size: 26px; font-weight: 800; color: #3fbf7f; }
.live-errors.bad { color: #f06a6a; }
.marks { display: flex; flex-direction: column; gap: 5px; }
.mark { padding: 6px 9px; font-size: 12px; background: #7a2f35; }
.live-active { margin-top: 6px; }
.live-active div { color: #f06a6a; font-size: 13px; font-weight: 700; }
.live-metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin: 10px 0; }
.live-metrics div { background: #1b2030; border-radius: 9px; padding: 7px 9px; font-size: 12px; }
.live-metrics > div > span { display: block; color: #9fa8b8; font-size: 11px; }
.live-metrics b { font-size: 14px; }
.live-log { font-size: 12px; font-variant-numeric: tabular-nums; max-height: 130px; overflow-y: auto; }
.live-log div { padding: 2px 0; }
.live-log .nc { color: #7d8595; }
.live-warn { color: #f0a640; font-size: 12px; margin: 6px 0; }
.card.flash { animation: flash .6s ease-out; }
@keyframes flash { from { box-shadow: 0 0 0 3px #f06a6a; } to { box-shadow: 0 0 0 0 transparent; } }

/* video overlays */
.v-badge, .v-stance, .v-feet, .v-depth {
    position: absolute; padding: 6px 12px; border-radius: 999px;
    background: rgba(0,0,0,.65); font-size: 13px; font-weight: 700;
}
.v-badge { top: 14px; left: 14px; display: flex; align-items: center; gap: 7px; font-variant-numeric: tabular-nums; }
.v-badge .dot { width: 9px; height: 9px; border-radius: 50%; background: #7d8595; }
.v-badge.running .dot { background: #f04848; }
.v-badge.countdown .dot { background: #f0a640; }
.v-stance { top: 14px; right: 14px; background: #2867e8; }
.v-feet { bottom: 14px; left: 14px; }
.v-feet.ok { background: rgba(22,134,93,.9); }
.v-feet.wrong { background: rgba(201,61,69,.9); }
.v-depth { bottom: 54px; left: 14px; font-weight: 600; font-size: 12px; }
.v-countdown {
    position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
    font-size: 120px; font-weight: 900; text-shadow: 0 4px 24px rgba(0,0,0,.8); pointer-events: none;
}

/* summary tiles */
.chip { font-size: 12px; font-weight: 700; padding: 4px 9px; border-radius: 999px; background: #2a303c; color: #aeb6c5; }
.chip.ok { background: rgba(63,191,127,.18); color: #3fbf7f; }
.chip.high { background: rgba(240,166,64,.18); color: #f0a640; }
.tiles { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.tile { background: #1b2030; border-radius: 11px; padding: 9px 11px; }
.tile .t-name { color: #9fa8b8; font-size: 12px; }
.tile .t-val { font-size: 22px; font-weight: 800; margin: 2px 0; }
.tile .t-ref { color: #8f98a8; font-size: 11px; }
.range { position: relative; height: 10px; border-radius: 6px; background: #2a303c; margin-top: 6px; }
.range .zone { position: absolute; top: 0; bottom: 0; border-radius: 6px; background: rgba(63,191,127,.55); }
.range .mark-v { position: absolute; top: -3px; width: 3px; height: 16px; background: white; border-radius: 2px; }

/* parameters table */
table.params { width: 100%; border-collapse: collapse; font-size: 12px; }
table.params th, table.params td { padding: 6px 4px; border-bottom: 1px solid #252a34; text-align: right; }
table.params th:first-child, table.params td:first-child { text-align: left; color: #c9d0dc; }
table.params th { color: #9fa8b8; font-weight: 600; font-size: 11px; }
table.params tr.total td { font-weight: 800; }
table.params tr.sep td { border-bottom: 2px solid #343b48; }
table.params .pos { color: #f06a6a; font-weight: 700; }
table.params .na { color: #5d6575; }

details.other summary { cursor: pointer; color: #aeb6c5; font-size: 13px; }
details.other[open] summary { margin-bottom: 10px; }

@media (max-width: 1400px) {
    .dash { grid-template-columns: minmax(0, 1fr) 340px; }
    .dash > .col:last-child { grid-column: 1 / -1; display: grid; grid-template-columns: 1fr 1fr; }
}
@media (max-width: 900px) {
    .dash, .dash > .col:last-child { grid-template-columns: 1fr; }
    .dash .video-card { position: relative; min-height: 60vh; }
    .call-bar { flex-wrap: wrap; }
}

/* system check strip */
.health { display: flex; flex-wrap: wrap; gap: 8px; padding: 10px 16px 0; font-size: 12px; }
.health .h { display: flex; align-items: center; gap: 6px; padding: 5px 10px; border-radius: 999px; background: #161a23; border: 1px solid #262c38; color: #c9d0dc; }
.health .h i { width: 8px; height: 8px; border-radius: 50%; background: #5d6575; display: inline-block; }
.health .h.ok i { background: #3fbf7f; }
.health .h.warn i { background: #f0a640; }
.health .h.bad i { background: #f04848; }
.health .h small { color: #8f98a8; }
/* skeleton over the patient video */
.skeleton { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
/* ---------- speech analysis (clinician-controlled) ---------- */
#speechButton { width: 100%; padding: 11px 12px; font-weight: 700; display: flex; align-items: center; justify-content: center; gap: 8px; }
.speech-dot { width: 9px; height: 9px; border-radius: 50%; background: #fff; animation: speechPulse 1s infinite; }
@keyframes speechPulse { 50% { opacity: .25; } }
.speech-who { color: #9fa8b8; font-size: 12px; margin-bottom: 8px; }
.speech-who b { color: #fff; }
.speech-status { color: #8f98a8; font-size: 12px; margin-top: 8px; min-height: 15px; }
.speech-status.err { color: #f0a640; }
.speech-live { margin-top: 8px; }
.speech-score { display: flex; align-items: baseline; gap: 10px; margin: 12px 0 2px; }
.speech-score b { font-size: 30px; line-height: 1; }
.speech-text { color: #c4cad6; font-size: 12px; margin-bottom: 8px; }
.speech-sub { color: #9fa8b8; font-size: 12px; font-weight: 700; margin: 12px 0 2px; }
.speech-dev .metric-value small { color: #8f98a8; font-weight: 400; }
.speech-dev .lowers { color: #f0a640; }
.speech-note { color: #f0a640; font-size: 12px; margin-top: 6px; }
.v-rec { position: absolute; top: 52px; left: 14px; display: flex; align-items: center; gap: 7px;
         background: rgba(201,61,69,.9); color: #fff; font-size: 12px; font-weight: 700;
         padding: 5px 10px; border-radius: 999px; }
</style>

</head>

<body>

<header>
    <div class="logo">TeleVision Doctor</div>
    <div class="call-bar">
        <input id="roomInput" value="television-demo" title="Patient / Room ID">
        <input id="patientName" placeholder="Patient name">
        <button id="joinButton" class="primary">Join Call</button>
        <button id="endButton" class="danger">End</button>
    </div>
    <div id="connectionStatus" class="status">Disconnected</div>
</header>

<div id="livePage" class="page active">
<div id="health" class="health"></div>
<div class="dash">

    <!-- ===================== VIDEO ===================== -->
    <div class="video-card">
        <video id="patientVideo" autoplay playsinline></video>
        <canvas id="skeleton" class="skeleton"></canvas>
        <video id="doctorVideo" autoplay muted playsinline></video>
        <div id="vRec" class="v-rec" hidden><span class="speech-dot"></span><span id="vRecText">Speech REC</span></div>
        <div id="vBadge" class="v-badge"><span class="dot"></span><span id="vBadgeText">Ready</span></div>
        <div id="vStance" class="v-stance" hidden></div>
        <div id="vCountdown" class="v-countdown" hidden></div>
        <div id="vFeet" class="v-feet" hidden></div>
        <div id="vDepth" class="v-depth" hidden></div>
    </div>

    <!-- ===================== BESS: RUN ===================== -->
    <div class="col">
        <div class="card">
            <div class="card-head">
                <h3>Modified BESS</h3>
                <div class="seg" role="group" aria-label="Non-dominant leg">
                    <span class="seg-label">Non-dominant</span>
                    <button id="ndLeft" class="seg-btn active">Left</button>
                    <button id="ndRight" class="seg-btn">Right</button>
                </div>
            </div>
            <div id="stanceButtons" class="stances"></div>
            <div class="hint" id="stanceHint">Hands on hips, eyes closed. Click a stance to start it.</div>
        </div>

        <div class="card live" id="liveCard">
            <div class="live-head">
                <div>
                    <div id="liveTitle" class="live-title">Ready</div>
                    <div id="liveSub" class="live-sub">No stance running</div>
                </div>
                <div id="liveTimer" class="live-timer">–</div>
            </div>
            <div class="bar"><div id="liveBar" class="bar-fill"></div></div>
            <div class="errors-row">
                <div id="liveErrors" class="live-errors">Errors: 0</div>
                <div class="marks">
                    <button id="markEyes" class="mark" disabled>Eyes opened +1</button>
                    <button id="markHands" class="mark" disabled>Hands off hips +1</button>
                </div>
            </div>
            <div id="liveActive" class="live-active"></div>
            <div class="live-metrics">
                <div><span>Feet</span><b id="mFeet">–</b></div>
                <div><span>Hip angle L / R</span><b id="mHip">–</b></div>
                <div><span>Sway (live)</span><b id="mSway">–</b></div>
                <div><span>Distance</span><b id="mDist">–</b></div>
            </div>
            <div id="liveLog" class="live-log"></div>
            <div id="liveWarn" class="live-warn"></div>
            <div class="row">
                <button id="cancelStance" disabled>Cancel stance</button>
            </div>
        </div>

        <details class="card other">
            <summary>Other tests (finger / eyes)</summary>
            <label>Select Test</label>
            <select id="testSelect">
                <option value="finger">Near Point / Finger</option>
                <option value="balance">Balance / BESS</option>
                <option value="eyes">Eyes</option>
            </select>
            <div class="row" style="margin-top:8px">
                <button id="startTestButton" class="good">Start Test</button>
                <button id="stopTestButton" class="danger">Stop Test</button>
            </div>
            <div class="metric"><span class="metric-name">Current test</span><span id="currentTest" class="metric-value">None</span></div>
            <div class="metric"><span class="metric-name">Finger distance</span><span id="fingerDistance" class="metric-value">—</span></div>
            <div class="metric"><span class="metric-name">Hand</span><span id="handDetected" class="metric-value">—</span></div>
            <div class="metric"><span class="metric-name">Eye score</span><span id="eyeScore" class="metric-value">—</span></div>
            <div class="metric"><span class="metric-name">Eye alert</span><span id="eyeAlert" class="metric-value">—</span></div>
            <button id="recalibrate" style="width:100%;margin-top:8px">Recalibrate</button>
        </details>
    </div>

    <!-- ===================== BESS: RESULTS ===================== -->
    <div class="col">
        <div class="card" id="speechCard">
            <div class="card-head">
                <h3>Speech Analysis</h3>
                <span id="speechChip" class="chip">Not recording</span>
            </div>
            <div class="speech-who">Patient ID <b id="speechPatient">–</b> (the room ID) · patient's call audio only</div>
            <button id="speechButton" class="good">Start speech recording</button>
            <div id="speechStatus" class="speech-status"></div>
            <div id="speechLive" class="speech-live" hidden></div>
            <div id="speechReport" hidden></div>
        </div>

        <div class="card">
            <div class="card-head">
                <h3>BESS Summary</h3>
                <span id="summaryChip" class="chip">In progress</span>
            </div>
            <div id="tiles" class="tiles"></div>
            <div class="hint">Reference ranges are approximate placeholders, and the standard BESS uses 20 s stances.</div>
        </div>

        <div class="card">
            <div class="card-head">
                <h3>Balance Parameters</h3>
                <button id="resetScores" class="small">Reset scores</button>
            </div>
            <table class="params">
                <thead><tr><th>Error</th><th>Feet together</th><th>Tandem</th><th>Single leg</th><th>Total</th></tr></thead>
                <tbody id="paramsBody"></tbody>
            </table>
        </div>
    </div>

</div>
</div>


<script>

const $ = id =>
    document.getElementById(id);


let ws = null;

let pc = null;

let localStream = null;

let currentRoom = null;

let currentTest = null;


// ==========================================================
// TABS
// ==========================================================

// ==========================================================
// WEBSOCKET
// ==========================================================

function wsURL() {

    return (
        (
            location.protocol
            === "https:"
        )
        ? "wss:"
        : "ws:"
    )
    + "//"
    + location.host
    + "/ws";
}


function send(
    payload
) {

    if (
        !ws
        ||
        ws.readyState
            !== WebSocket.OPEN
    ) {

        return;
    }

    ws.send(
        JSON.stringify(
            payload
        )
    );
}


let wantCall = false;          // between Join and End: keep the connection up
let reconnectTimer = null;
let callRetryTimer = null;

function setStatus(text) {
    $("connectionStatus").textContent = text;
}

function connectWebSocket(room) {
    ws = new WebSocket(wsURL());
    ws.onopen = () => {
        send({ type: "join", room: room, role: "doctor" });
    };
    ws.onerror = () => {};    // onclose follows and handles it
    ws.onclose = () => {
        ws = null;
        if (!wantCall) {
            setStatus("Disconnected");
            return;
        }
        // dropped: reconnect by itself (the video call keeps running)
        setStatus("Reconnecting…");
        clearTimeout(reconnectTimer);
        reconnectTimer = setTimeout(() => {
            if (wantCall) {
                connectWebSocket(room);
            }
        }, 1500);
    };
    ws.onmessage = async event => {
        let message;
        try {
            message = JSON.parse(event.data);
        } catch {
            return;
        }
        try {
            await handleMessage(message);
        } catch (error) {
            console.error(error);
        }
    };
}


// ==========================================================
// WEBRTC
// ==========================================================

// The doctor's camera and microphone are optional: without them the doctor
// still receives the patient's video (and can talk if only the mic works).
async function getLocalStream() {
    if (localStream) {
        return localStream;
    }
    for (const want of [{ video: true, audio: true }, { audio: true }]) {
        try {
            localStream = await navigator.mediaDevices.getUserMedia(want);
            return localStream;
        } catch (error) {
            console.warn("No doctor camera/mic for", want, error);
        }
    }
    return null;
}

async function createPeerConnection() {
    if (pc) {
        try { pc.close(); } catch {}
    }
    pc = new RTCPeerConnection({
        iceServers: [{ urls: "stun:stun.l.google.com:19302" }]
    });

    pc.onicecandidate = event => {
        if (!event.candidate) {
            return;
        }
        send({
            type: "ice",
            room: currentRoom,
            role: "doctor",
            candidate: event.candidate.candidate,
            sdpMid: event.candidate.sdpMid,
            sdpMLineIndex: event.candidate.sdpMLineIndex
        });
    };

    const thisPc = pc;
    pc.onconnectionstatechange = () => {
        if (thisPc !== pc) {
            return;               // an older connection
        }
        const state = pc.connectionState;
        setStatus({
            connected: "Patient connected",
            connecting: "Connecting video…",
            new: "Connecting video…",
            disconnected: "Video interrupted…",
            failed: "Video failed, retrying…",
            closed: "Video closed",
        }[state] || state);
        if (state === "connected") {
            startBalanceStream();
        }
        if (state === "failed" && wantCall) {
            clearTimeout(callRetryTimer);
            callRetryTimer = setTimeout(callPatient, 3000);
        }
    };

    pc.ontrack = event => {
        const stream = event.streams[0];
        if (stream) {
            $("patientVideo").srcObject = stream;
            speechOnTrack(stream);
        }
    };

    const stream = await getLocalStream();
    $("doctorVideo").srcObject = stream;
    $("doctorVideo").hidden = !(stream && stream.getVideoTracks().length);
    if (stream) {
        for (const track of stream.getTracks()) {
            pc.addTrack(track, stream);
        }
    }
    // always ask for the patient's audio and video, with or without ours
    const kinds = new Set(stream ? stream.getTracks().map(t => t.kind) : []);
    for (const kind of ["audio", "video"]) {
        if (!kinds.has(kind)) {
            pc.addTransceiver(kind, { direction: "recvonly" });
        }
    }
}

async function makeOffer() {
    if (!pc) {
        return;
    }
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    send({ type: "offer", room: currentRoom, role: "doctor", sdp: offer.sdp });
}

// (Re)call the patient with a fresh connection. The phone starts a fresh
// connection too when it gets a new offer.
async function callPatient() {
    if (!wantCall || !ws) {
        return;
    }
    setStatus("Calling patient…");
    await createPeerConnection();
    await makeOffer();
}


// ==========================================================
// MESSAGES
// ==========================================================

async function handleMessage(message) {
    switch (message.type) {

    case "joined":
        if (message.peer_present) {
            // the patient is already in the room
            if (!pc || pc.connectionState !== "connected") {
                // calling again from scratch: the phone may be a fresh
                // session too, so ask for its balance stream again
                currentTest = null;
                await callPatient();
            } else {
                setStatus("Patient connected");
            }
            setTimeout(startBalanceStream, 800);
        } else {
            setStatus("Waiting for patient");
        }
        break;

    case "peer_joined":
        if (message.role === "patient") {
            // a fresh patient app: start its balance stream again too
            currentTest = null;
            await callPatient();
            setTimeout(startBalanceStream, 800);
        }
        break;

    case "peer_left":
        if (message.role === "patient") {
            setStatus("Patient left. Waiting for patient");
            currentTest = null;
        }
        break;

    case "answer":
        if (!pc) {
            return;
        }
        try {
            await pc.setRemoteDescription(
                new RTCSessionDescription({ type: "answer", sdp: message.sdp }));
        } catch (error) {
            console.error(error);
        }
        break;

    case "ice":
        if (!pc || !message.candidate) {
            return;
        }
        try {
            await pc.addIceCandidate(new RTCIceCandidate({
                candidate: message.candidate,
                sdpMid: message.sdpMid,
                sdpMLineIndex: message.sdpMLineIndex
            }));
        } catch (error) {
            console.error(error);
        }
        break;

    case "measurement":
        handleMeasurement(message);
        break;
    }
}


// ==========================================================
// CALL
// ==========================================================

function joinCall() {
    currentRoom = $("roomInput").value.trim();
    if (!currentRoom) {
        return;
    }
    wantCall = true;
    clearTimeout(reconnectTimer);
    if (ws) {
        ws.onclose = null;
        try { ws.close(); } catch {}
        ws = null;
    }
    setStatus("Connecting…");
    connectWebSocket(currentRoom);
}

function endCall() {
    speechStop();
    wantCall = false;
    clearTimeout(reconnectTimer);
    clearTimeout(callRetryTimer);
    if (pc) {
        pc.close();
        pc = null;
    }
    if (localStream) {
        for (const track of localStream.getTracks()) {
            track.stop();
        }
        localStream = null;
    }
    if (ws) {
        ws.close();
        ws = null;
    }
    currentTest = null;
    $("patientVideo").srcObject = null;
    $("doctorVideo").srcObject = null;
    setStatus("Disconnected");
}

$("joinButton").onclick = joinCall;
$("endButton").onclick = endCall;

// join the room as soon as the page opens
window.addEventListener("load", () => setTimeout(joinCall, 300));


// ==========================================================
// TEST
// ==========================================================

$("startTestButton").onclick =
    () => {

        const test =
            $("testSelect").value;

        currentTest =
            test;

        $("currentTest")
            .textContent =
            test;


        send({

            type:
                "start_test",

            room:
                currentRoom,

            role:
                "doctor",

            test:
                test
        });
    };


$("stopTestButton").onclick =
    () => {

        send({

            type:
                "stop_test",

            room:
                currentRoom,

            role:
                "doctor"
        });

        currentTest =
            null;

        $("currentTest")
            .textContent =
            "None";
    };


// ==========================================================
// ANALYZER COMMANDS
// ==========================================================

function serverCommand(
    test,
    command
) {

    send({

        type:
            "server_command",

        room:
            currentRoom,

        role:
            "doctor",

        test:
            test,

        command:
            command
    });
}


$("recalibrate").onclick =
    () => {

        if (
            currentTest
            !== "balance"
            &&
            currentTest
            !== "eyes"
        ) {

            return;
        }

        serverCommand(
            currentTest,
            {
                type:
                    "recalibrate"
            }
        );
    };


// ==========================================================
// LIVE RESULTS
// ==========================================================

function handleMeasurement(
    message
) {

    if (
        message.test
        === "finger"
    ) {

        const d =
            message.distance_cm;

        $("fingerDistance")
            .textContent =
            typeof d === "number"
            ? d.toFixed(1)
                + " cm"
            : "—";

        $("handDetected")
            .textContent =
            message.hand_detected
            ? "Yes"
            : "No";

        return;
    }


    const result =
        message.result;

    if (!result) {
        return;
    }


    if (
        message.test
        === "balance"
    ) {

        handleBalance(result);
    }


    if (
        message.test
        === "eyes"
    ) {

        const score =
            result.score;

        $("eyeScore")
            .textContent =
            typeof score === "number"
            ? score.toFixed(3)
            : "—";

        $("eyeAlert")
            .textContent =
            result.alerting
            ? "ALERT"
            : "Normal";
    }
}




// ==========================================================
// MODIFIED BESS DASHBOARD
// Live BESS state comes from the pose server's reply for each frame,
// forwarded by the phone as a "measurement" (message.result).
// ==========================================================

const STANCES = [
    ["double", "Feet together"],
    ["tandem", "Tandem"],
    ["single", "Single leg"],
];
const STANCE_NAME = Object.fromEntries(STANCES);

const ERRORS = [
    ["hands_off_hips", "Hands off hips"],
    ["eyes_open", "Eyes opened"],
    ["step_stumble_fall", "Step / stumble / fall"],
    ["hip_angle", "Hip > 30°"],
    ["foot_lift", "Heel / forefoot lifted"],
    ["out_of_position", "Out of position > 5 s"],
];
const ERROR_NAME = Object.fromEntries(ERRORS);

// Approximate placeholders (same as the iPhone app): check against a clinical source.
const REF = { double: [0, 1], tandem: [0, 2], single: [0, 4], total: [0, 6] };

let nondominant = "left";
let bess = null;           // latest BESS status
let session = null;        // scores, by_stance, sway, coverage, error_types
let starting = null;       // stance requested, waiting for the phone to start it
let lastErrors = 0;
let lastResult = null;
let lastLog = [];
let prev = { phase: "idle", stance: null };

const esc = v => String(v).replace(/[&<>"']/g,
    c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[c]));

function fmtTime(sec) {
    sec = Math.max(0, Math.round(sec));
    return Math.floor(sec / 60) + ":" + String(sec % 60).padStart(2, "0");
}

// ---------------- commands ----------------

// Ask the phone to stream to the pose server (the "balance" test), so the
// doctor sees feet / distance / tracking before starting a stance.
// Does nothing if it's already streaming.
function startBalanceStream() {
    if (!currentRoom || currentTest === "balance") {
        return false;
    }
    currentTest = "balance";
    $("currentTest").textContent = "balance";
    $("testSelect").value = "balance";
    send({ type: "start_test", room: currentRoom, role: "doctor", test: "balance" });
    return true;
}

function startStance(stance) {
    if (!currentRoom || starting) {
        return;
    }
    starting = stance;
    lastResult = null;
    $("liveWarn").textContent = "";
    const go = () => serverCommand("balance",
        { type: "bess_start", stance: stance, nondominant: nondominant });

    if (startBalanceStream()) {
        setTimeout(go, 2000);          // the phone connects to the pose server first
    } else {
        go();
    }
    setTimeout(() => {
        if (starting === stance) {
            starting = null;
            $("liveWarn").textContent =
                "The stance didn't start. Is the phone in the call? Try again.";
            renderAll();
        }
    }, 9000);
    renderAll();
}

function setNondominant(side) {
    nondominant = side;
    $("ndLeft").classList.toggle("active", side === "left");
    $("ndRight").classList.toggle("active", side === "right");
}
$("ndLeft").onclick = () => setNondominant("left");
$("ndRight").onclick = () => setNondominant("right");

$("cancelStance").onclick = () => serverCommand("balance", { type: "bess_cancel" });
$("markEyes").onclick = () => serverCommand("balance", { type: "bess_mark", error: "eyes_open" });
$("markHands").onclick = () => serverCommand("balance", { type: "bess_mark", error: "hands_off_hips" });
$("resetScores").onclick = () => {
    if (currentTest === "balance") {
        serverCommand("balance", { type: "bess_reset" });
    }
    session = null;
    lastResult = null;
    lastLog = [];
    renderAll();
};

// ---------------- incoming data ----------------

function handleBalance(r) {
    if (r.type === "pose") {
        noteFrame(r);
        if (r.bess) {
            bess = r.bess;
            if (bess.session) {
                session = bess.session;
            }
            if (starting && bess.phase !== "idle") {
                starting = null;
            }
        }
        renderLive(r);
        renderAll();
    } else if (r.type === "ack" && r.scores) {
        session = r;                    // command reply: session summary (+ sway)
        renderAll();
    } else if (r.type === "result" && r.kind === "bess_done") {
        if (r.session) {
            session = r.session;
        }
        renderAll();
    }
}

// ---------------- rendering ----------------

function renderLive(r) {
    const b = bess || { phase: "idle" };
    const dur = b.duration_s || 0;
    const running = b.phase === "running";

    if (prev.phase === "running" && b.phase === "idle" && prev.stance) {
        const n = session && session.scores ? session.scores[prev.stance] : null;
        lastResult = STANCE_NAME[prev.stance] + ": "
            + (n == null ? "done" : n + " error" + (n === 1 ? "" : "s"));
    }
    prev = { phase: b.phase, stance: b.stance };

    const badge = $("vBadge");
    const cd = $("vCountdown");
    badge.className = "v-badge" + (running ? " running" : b.phase === "countdown" ? " countdown" : "");
    $("vStance").hidden = b.phase === "idle";
    if (b.stance) {
        $("vStance").textContent = STANCE_NAME[b.stance];
    }

    if (b.phase === "countdown") {
        // ~1 s while the start position is recorded (no get-ready countdown)
        $("liveTitle").textContent = STANCE_NAME[b.stance] + ": starting";
        $("liveSub").textContent = b.waiting_for_view
            ? "Waiting to see the whole body, feet included"
            : "Recording the start position";
        $("liveTimer").textContent = "0:00 / " + fmtTime(dur);
        $("vBadgeText").textContent = b.waiting_for_view ? "Feet not in view" : "Starting";
        cd.hidden = true;
        $("liveBar").style.width = "0%";
    } else if (running) {
        const el = Math.max(0, dur - (b.time_left || 0));
        $("liveTitle").textContent = STANCE_NAME[b.stance];
        $("liveSub").textContent = "Scoring";
        $("liveTimer").textContent = fmtTime(el) + " / " + fmtTime(dur);
        $("vBadgeText").textContent = "Running " + fmtTime(el) + " / " + fmtTime(dur);
        cd.hidden = true;
        $("liveBar").style.width = (dur ? Math.min(100, 100 * el / dur) : 0) + "%";
        lastLog = b.log || [];
    } else {
        $("liveTitle").textContent = lastResult
            || (starting ? STANCE_NAME[starting] + ": starting…" : "Ready");
        $("liveSub").textContent = lastResult ? "Stance finished" : "No stance running";
        $("liveTimer").textContent = dur ? fmtTime(dur) : "–";
        $("vBadgeText").textContent = "Ready";
        cd.hidden = true;
        $("liveBar").style.width = lastResult ? "100%" : "0%";
    }

    // errors
    const errors = running ? (b.errors || 0) : null;
    if (running && errors > lastErrors) {
        const card = $("liveCard");
        card.classList.remove("flash");
        void card.offsetWidth;
        card.classList.add("flash");
    }
    lastErrors = running ? errors : 0;
    const le = $("liveErrors");
    le.textContent = "Errors: " + (running ? errors : "–");
    le.classList.toggle("bad", running && errors > 0);
    $("liveActive").innerHTML = running
        ? (b.active || []).map(k => "<div>Now: " + esc(ERROR_NAME[k] || k) + "</div>").join("")
        : "";
    const log = running ? (b.log || []) : (lastResult ? lastLog : []);
    $("liveLog").innerHTML = log.slice().reverse().map(e =>
        '<div class="' + (e.counted ? "" : "nc") + '">' + Number(e.t).toFixed(1) + "s  "
        + (e.counted ? "+1" : " 0") + "  " + esc(e.label) + "</div>").join("");
    if (b.warnings && b.warnings.length) {
        $("liveWarn").textContent = b.warnings.join(" ");
    }

    // controls
    const eyesOn = session && session.error_types
        ? session.error_types.includes("eyes_open") : true;
    $("markEyes").disabled = !running || !eyesOn;
    $("markHands").disabled = !running;
    $("cancelStance").disabled = b.phase === "idle";
    $("ndLeft").disabled = $("ndRight").disabled = b.phase !== "idle";

    // live metrics
    const feet = r.feet ? r.feet.state : null;
    const feetText = { both_down: "Both feet down", left_up: "Left foot up",
                       right_up: "Right foot up", not_visible: "Feet not visible" }[feet] || "–";
    const expected = !b.stance || b.phase === "idle" ? null
        : b.stance === "single" ? (b.nondominant === "left" ? "right_up" : "left_up")
        : "both_down";
    $("mFeet").textContent = feetText;
    $("mFeet").className = expected == null ? "" : (feet === expected ? "good-txt" : "bad-txt");
    const vf = $("vFeet");
    vf.hidden = !feet;
    vf.textContent = feetText;
    vf.className = "v-feet" + (expected == null ? "" : feet === expected ? " ok" : " wrong");

    const hips = b.hip_angles || {};
    const hl = hips.left, hr = hips.right;
    $("mHip").innerHTML = hl == null && hr == null ? "–"
        : [hl, hr].map(a => a == null ? "–"
            : '<span class="' + (a > 30 ? "bad-txt" : "") + '">' + Math.round(a) + "°</span>").join(" / ");
    $("mSway").textContent = b.sway_cm == null ? "–" : b.sway_cm.toFixed(1) + " cm";

    const z = r.depth && r.depth.torso_m ? r.depth.torso_m[2] : null;
    $("mDist").textContent = z ? z.toFixed(2) + " m" : (r.depth ? "body not found" : "no depth");
    $("vDepth").hidden = !r.depth;
    $("vDepth").textContent = z ? "LiDAR " + z.toFixed(1) + " m" : "LiDAR: body not found";
}

function renderStances() {
    const box = $("stanceButtons");
    const active = bess && bess.phase !== "idle";
    const dur = bess && bess.duration_s ? bess.duration_s : null;
    box.innerHTML = "";
    for (const [id, name] of STANCES) {
        const score = session && session.scores ? session.scores[id] : null;
        const current = (active && bess.stance === id) || starting === id;
        const status = current
            ? (starting === id && !active ? "Starting…"
               : bess.phase === "countdown" ? "Get ready" : "Running")
            : (score != null ? "Completed" : "Pending");
        const btn = document.createElement("button");
        btn.className = "stance" + (current ? " running" : "") + (score != null && !current ? " done" : "");
        btn.innerHTML = '<span class="s-name">' + name + "</span>"
            + '<span class="s-status">' + status + "</span>"
            + '<span class="s-score ' + (score == null ? "" : score === 0 ? "good-txt" : "bad-txt") + '">'
            + (score != null ? score + " error" + (score === 1 ? "" : "s") : (dur ? dur + " s" : "&nbsp;"))
            + "</span>";
        btn.disabled = !currentRoom || active || starting !== null;
        btn.title = currentRoom ? "Start " + name : "Join the call first";
        btn.onclick = () => startStance(id);
        box.appendChild(btn);
    }
}

function rangeBar(v, ref, max) {
    const pct = x => Math.max(0, Math.min(100, 100 * x / max));
    return '<div class="range"><div class="zone" style="left:' + pct(ref[0]) + "%;width:"
        + Math.max(2, pct(ref[1]) - pct(ref[0])) + '%"></div>'
        + (v == null ? "" : '<div class="mark-v" style="left:calc(' + pct(v) + '% - 1.5px)"></div>')
        + "</div>";
}

function tile(name, v, ref, max) {
    return '<div class="tile"><div class="t-name">' + name + "</div>"
        + '<div class="t-val ' + (v == null ? "" : v > ref[1] ? "bad-txt" : "good-txt") + '">'
        + (v == null ? "–" : v) + "</div>"
        + '<div class="t-ref">ref. ' + ref[0] + "–" + ref[1] + "</div>"
        + rangeBar(v, ref, max) + "</div>";
}

function renderSummary() {
    const scores = session && session.scores ? session.scores : {};
    const any = STANCES.some(([id]) => scores[id] != null);
    const total = any ? session.total : null;
    let html = tile(session && session.complete ? "Total errors" : "Total (so far)", total, REF.total, 30);
    for (const [id, name] of STANCES) {
        html += tile(name, scores[id] ?? null, REF[id], 10);
    }
    $("tiles").innerHTML = html;
    const chip = $("summaryChip");
    if (!any) {
        chip.textContent = "Not started";
        chip.className = "chip";
    } else if (!session.complete) {
        chip.textContent = "In progress";
        chip.className = "chip";
    } else if (total <= REF.total[1]) {
        chip.textContent = "Within typical range";
        chip.className = "chip ok";
    } else {
        chip.textContent = "Above typical range";
        chip.className = "chip high";
    }
}

function renderTable() {
    const s = session || {};
    const by = s.by_stance || {};
    const types = s.error_types ? ERRORS.filter(([k]) => s.error_types.includes(k)) : ERRORS;
    const cell = v => v == null ? '<td class="na">–</td>'
        : '<td class="' + (v > 0 ? "pos" : "") + '">' + v + "</td>";
    let html = "";
    for (const [k, name] of types) {
        const vals = STANCES.map(([id]) => by[id] ? (by[id][k] ?? 0) : null);
        const done = vals.filter(v => v != null);
        html += "<tr><td>" + name + "</td>" + vals.map(cell).join("")
            + cell(done.length ? done.reduce((a, b) => a + b, 0) : null) + "</tr>";
    }
    const scores = STANCES.map(([id]) => s.scores ? s.scores[id] : null);
    const anyScore = scores.some(v => v != null);
    html += '<tr class="total sep"><td>Errors</td>' + scores.map(cell).join("")
        + cell(anyScore ? s.total : null) + "</tr>";

    const sway = s.sway || {};
    const num = (v, d, unit) => v == null ? '<td class="na">–</td>'
        : "<td>" + Number(v).toFixed(d) + (unit || "") + "</td>";
    const row = (name, key, d) => "<tr><td>" + name + "</td>"
        + STANCES.map(([id]) => num(sway[id] ? sway[id][key] : null, d)).join("")
        + '<td class="na"></td></tr>';
    html += row("Sway velocity (cm/s)", "mean_velocity_cm_s", 1);
    html += row("Sway area (cm²)", "area_95_cm2", 0);
    html += row("Side-to-side RMS (cm)", "rms_ml_cm", 1);
    html += row("Front-back RMS (cm)", "rms_ap_cm", 1);
    const cov = s.coverage || {};
    html += "<tr><td>Tracked</td>"
        + STANCES.map(([id]) => cov[id] == null ? '<td class="na">–</td>'
            : "<td>" + Math.round(cov[id] * 100) + "%</td>").join("")
        + '<td class="na"></td></tr>';
    $("paramsBody").innerHTML = html;
}

function renderAll() {
    renderStances();
    renderSummary();
    renderTable();
}

renderAll();
setInterval(renderStances, 1000);     // picks up joining / leaving the call



// ==========================================================
// SYSTEM CHECK + SKELETON
// Shows which links work (call, phone -> pose server, MediaPipe, LiDAR)
// and draws the MediaPipe skeleton the pose server found.
// ==========================================================

const health = { lastAt: 0, fps: 0, detected: false, depth: null, landmarks: null };

const BONES = [[11, 12], [11, 23], [12, 24], [23, 24], [11, 13], [13, 15], [12, 14], [14, 16],
               [23, 25], [25, 27], [24, 26], [26, 28], [27, 29], [29, 31], [27, 31],
               [28, 30], [30, 32], [28, 32]];

function noteFrame(r) {
    const now = performance.now();
    if (health.lastAt) {
        const inst = 1000 / Math.max(1, now - health.lastAt);
        health.fps = health.fps ? 0.8 * health.fps + 0.2 * inst : inst;
    }
    health.lastAt = now;
    health.detected = !!r.detected;
    health.depth = r.depth ? (r.depth.torso_m ? "ok" : "nobody") : null;
    health.landmarks = r.detected ? r.landmarks : null;
    drawSkeleton();
}

function drawSkeleton() {
    const video = $("patientVideo");
    const canvas = $("skeleton");
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
    }
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    const lm = health.landmarks;
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!lm || lm.length < 33 || !vw || !vh) {
        return;
    }
    // where the video is drawn inside the element (object-fit: contain)
    const scale = Math.min(w / vw, h / vh);
    const dw = vw * scale, dh = vh * scale;
    const ox = (w - dw) / 2, oy = (h - dh) / 2;
    const P = i => [ox + lm[i].x * dw, oy + lm[i].y * dh];
    const seen = i => (lm[i].visibility ?? 1) > 0.5;
    ctx.lineWidth = 4;
    ctx.strokeStyle = "rgba(63,191,127,0.9)";
    for (const [a, b] of BONES) {
        if (seen(a) && seen(b)) {
            const [x1, y1] = P(a), [x2, y2] = P(b);
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        }
    }
    ctx.fillStyle = "#ffd84a";
    for (let i = 11; i < 33; i++) {
        if (seen(i)) {
            const [x, y] = P(i);
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, 2 * Math.PI);
            ctx.fill();
        }
    }
}

function renderHealth() {
    const now = performance.now();
    const fresh = health.lastAt && now - health.lastAt < 2500;
    if (!fresh && health.landmarks) {
        health.landmarks = null;
        drawSkeleton();
    }
    const callState = pc ? pc.connectionState : null;
    const item = (cls, name, note) =>
        '<span class="h ' + cls + '"><i></i>' + name + (note ? " <small>" + note + "</small>" : "") + "</span>";
    const items = [
        item(ws ? "ok" : wantCall ? "warn" : "bad", "Server", ws ? "" : wantCall ? "reconnecting" : "not joined"),
        item(callState === "connected" ? "ok" : callState ? "warn" : "", "Video call",
             callState === "connected" ? "" : callState || "waiting for patient"),
        item(fresh ? "ok" : currentTest === "balance" ? "bad" : "", "Phone → pose server",
             fresh ? health.fps.toFixed(0) + " fps"
                   : currentTest === "balance" ? "no data (is server.py running?)" : "starts when patient joins"),
        item(!fresh ? "" : health.detected ? "ok" : "warn", "MediaPipe pose",
             !fresh ? "" : health.detected ? "person tracked" : "no person in view"),
        item(!fresh ? "" : health.depth === "ok" ? "ok" : health.depth ? "warn" : "warn", "LiDAR depth",
             !fresh ? "" : health.depth === "ok" ? "" : health.depth ? "body not found" : "no depth (2D)"),
    ];
    $("health").innerHTML = items.join("");
}

setInterval(renderHealth, 500);
window.addEventListener("resize", drawSkeleton);
renderHealth();


// ==========================================================
// SPEECH ANALYSIS (clinician starts and ends it)
// ==========================================================
// The patient's voice reaches this page through the WebRTC call. While
// recording, the page taps that remote audio track (never the doctor's mic)
// and streams it as 16-bit PCM to speech_server.py (port 8767), which scores
// it against the patient's prior visits when the clinician ends it.
// Other port: open the page with ?speechPort=8770.

const SPEECH_PORT = new URLSearchParams(location.search).get("speechPort") || "8767";
const speech = {
    phase: "idle",           // idle | connecting | recording | finishing
    ws: null, ctx: null, src: null, node: null,
    pcm: null, pos: 0, chunkSamples: 4000,
    t0: 0, tick: null, endTimer: null,
};

function speechURL() {
    return (location.protocol === "https:" ? "wss:" : "ws:") + "//"
        + (location.hostname || "localhost") + ":" + SPEECH_PORT;
}

function speechEsc(v) {
    return String(v).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function speechFmt(v, unit) {
    return (v === null || v === undefined) ? "–" : speechEsc(v) + (unit || "");
}

function speechJoin(...parts) {
    const keep = parts.filter(x => x && x !== "–");
    return keep.length ? keep.join(" · ") : "–";
}

function speechClock(sec) {
    sec = Math.max(0, Math.floor(sec));
    return Math.floor(sec / 60) + ":" + String(sec % 60).padStart(2, "0");
}

function speechPatientId() {
    return ($("roomInput").value || "").trim() || "unknown";
}

function speechStatus(text, isError) {
    $("speechStatus").textContent = text || "";
    $("speechStatus").classList.toggle("err", !!isError);
}

function speechRender() {
    const b = $("speechButton"), chip = $("speechChip"), p = speech.phase;
    $("speechPatient").textContent = speechPatientId();
    b.disabled = (p === "connecting" || p === "finishing");
    b.className = (p === "recording") ? "danger" : "good";
    if (p === "recording") {
        const t = speechClock((Date.now() - speech.t0) / 1000);
        b.innerHTML = '<span class="speech-dot"></span>End speech recording · ' + t;
        chip.textContent = "Recording " + t;
        chip.className = "chip high";
        $("vRec").hidden = false;
        $("vRecText").textContent = "Speech REC " + t;
    } else {
        b.textContent = { connecting: "Connecting…", finishing: "Analyzing…" }[p] || "Start speech recording";
        chip.textContent = { connecting: "Connecting", finishing: "Analyzing" }[p]
            || (speech.lastReport ? "Report ready" : "Not recording");
        chip.className = speech.lastReport && p === "idle" ? "chip ok" : "chip";
        $("vRec").hidden = true;
    }
}

function patientAudioStream() {
    const s = $("patientVideo").srcObject;
    return (s && s.getAudioTracks && s.getAudioTracks().length) ? s : null;
}

function speechSendPcm(a) {
    if (speech.ws && speech.ws.readyState === WebSocket.OPEN) {
        speech.ws.send(a.buffer.slice(a.byteOffset, a.byteOffset + a.byteLength));
    }
}

function speechPush(f32) {
    const n = speech.chunkSamples;
    for (let i = 0; i < f32.length; i++) {
        if (!speech.pcm) { speech.pcm = new Int16Array(n); speech.pos = 0; }
        const v = Math.max(-1, Math.min(1, f32[i]));
        speech.pcm[speech.pos++] = v < 0 ? v * 32768 : v * 32767;
        if (speech.pos === n) { speechSendPcm(speech.pcm); speech.pcm = null; }
    }
}

function speechFlush() {
    if (speech.pcm && speech.pos) speechSendPcm(speech.pcm.subarray(0, speech.pos));
    speech.pcm = null;
}

// (Re)connect the patient's audio track to the recorder; called again if the
// call reconnects while recording.
function speechAttach(stream) {
    if (!speech.ctx || !stream) return;
    if (speech.src) { try { speech.src.disconnect(); } catch {} }
    speech.src = speech.ctx.createMediaStreamSource(stream);
    speech.src.connect(speech.node);
}

function speechOnTrack(stream) {
    if (speech.phase === "recording" && stream && stream.getAudioTracks().length) {
        speechAttach(stream);
    }
}

function speechStopAudio() {
    clearInterval(speech.tick);
    speech.tick = null;
    if (speech.src) { try { speech.src.disconnect(); } catch {} }
    if (speech.node) { try { speech.node.disconnect(); } catch {} speech.node.onaudioprocess = null; }
    if (speech.ctx) { try { speech.ctx.close(); } catch {} }
    speech.src = speech.node = speech.ctx = null;
}

function speechCloseSocket() {
    clearTimeout(speech.endTimer);
    if (speech.ws) {
        speech.ws.onclose = null;
        try { speech.ws.close(); } catch {}
        speech.ws = null;
    }
}

function speechFail(text) {
    speechStopAudio();
    speechCloseSocket();
    speech.phase = "idle";
    speechStatus(text, true);
    speechRender();
}

async function speechStart() {
    const stream = patientAudioStream();
    if (!stream) {
        speechStatus('No patient audio yet: join the call and wait for "Patient connected".', true);
        return;
    }
    speech.phase = "connecting";
    speechStatus("");
    speechRender();

    let ws;
    try {
        ws = await new Promise((resolve, reject) => {
            const w = new WebSocket(speechURL());
            w.binaryType = "arraybuffer";
            const t = setTimeout(() => { try { w.close(); } catch {} reject(new Error("timeout")); }, 5000);
            w.onopen = () => { clearTimeout(t); resolve(w); };
            w.onerror = () => { clearTimeout(t); reject(new Error("error")); };
        });
    } catch {
        speechFail("Could not reach the speech server at " + speechURL()
            + ". Start it with: python speech_server.py");
        return;
    }
    speech.ws = ws;
    ws.onmessage = e => { try { speechMessage(JSON.parse(e.data)); } catch (err) { console.error(err); } };
    ws.onclose = () => {
        speech.ws = null;
        if (speech.phase === "recording" || speech.phase === "finishing") {
            speechFail("The speech server closed the connection before the report arrived. "
                + "Check the speech_server terminal.");
        }
    };

    try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        await ctx.resume();
        speech.ctx = ctx;
        speech.chunkSamples = Math.round(ctx.sampleRate / 4);          // ~250 ms per message
        speech.node = ctx.createScriptProcessor(4096, 1, 1);
        speech.node.onaudioprocess = e => {
            if (speech.phase === "recording") speechPush(e.inputBuffer.getChannelData(0));
        };
        speech.node.connect(ctx.destination);                          // outputs silence
        ws.send(JSON.stringify({ type: "session_start", patient_id: speechPatientId(),
                                 sample_rate: ctx.sampleRate }));
        speechAttach(stream);
    } catch (err) {
        console.error(err);
        speechFail("Could not record the call audio in this browser (" + err.message + ").");
        return;
    }

    speech.phase = "recording";
    speech.t0 = Date.now();
    speech.tick = setInterval(speechRender, 500);
    $("speechReport").hidden = true;
    $("speechLive").hidden = false;
    $("speechLive").innerHTML = '<div class="hint">Listening… first measurements in about 5 s.</div>';
    speechStatus("Recording the patient's audio from the call.");
    speechRender();
}

function speechStop() {
    if (speech.phase !== "recording") return;
    speech.phase = "finishing";
    speechFlush();
    speechStopAudio();
    if (speech.ws && speech.ws.readyState === WebSocket.OPEN) {
        speech.ws.send(JSON.stringify({ type: "session_end" }));
    }
    speech.endTimer = setTimeout(() => {
        speechFail("No report after 2 minutes. The visit is still stored by the server; check its terminal.");
    }, 120000);
    speechStatus("Analyzing the recording and comparing it with past visits…");
    speechRender();
}

function speechToggle() {
    if (speech.phase === "recording") speechStop();
    else if (speech.phase === "idle") speechStart();
}

function speechLiveHTML(m) {
    const sr = m.speech_rate || {}, v = m.voice || {}, p = m.pauses || {};
    const rows = [
        ["Patient speaking", speechClock(m.speaking_time_s || 0)],
        ["Speech rate", speechJoin(speechFmt(sr.words_per_min, " wpm"), speechFmt(sr.syllables_per_s, " syll/s"))],
        ["Jitter", speechFmt(v.jitter_local_pct, "%")],
        ["Shimmer", speechFmt(v.shimmer_local_pct, "%")],
        ["Pauses", speechFmt(p.count) + (p.mean_s != null ? " (avg " + speechFmt(p.mean_s, " s") + ")" : "")],
    ];
    return rows.map(([k, val]) =>
        '<div class="metric"><span class="metric-name">' + k + '</span><span class="metric-value">' + val + "</span></div>"
    ).join("");
}

function speechReportHTML(r) {
    const s = r.summary || {}, sr = s.speech_rate || {}, v = s.voice || {}, p = s.pauses || {};
    const lx = s.lexical || {}, b = r.baseline || {};
    let h = '<div class="speech-score"><b>' + speechFmt(b.voice_score) + "</b>"
        + (b.provisional ? '<span class="chip">Provisional</span>' : '<span class="chip ok">vs. own baseline</span>')
        + "</div>";
    h += '<div class="speech-text">' + speechFmt(b.summary_text) + "</div>";
    const rows = [
        ["Call length", speechClock(r.call_duration_s || 0) + " (patient speaking " + speechClock(s.speaking_time_s || 0) + ")"],
        ["Speech rate", speechJoin(speechFmt(sr.words_per_min, " wpm"), speechFmt(sr.syllables_per_s, " syll/s"))],
        ["Articulation rate", speechFmt(sr.articulation_rate_syll_per_s, " syll/s")],
        ["Jitter", speechFmt(v.jitter_local_pct, "%")],
        ["Shimmer", speechFmt(v.shimmer_local_pct, "%")],
        ["Pitch", speechFmt(v.f0_mean_hz, " Hz") + " (sd " + speechFmt(v.f0_sd_hz) + ")"],
        ["Pauses", speechFmt(p.count) + " · " + speechFmt(p.per_min, "/min") + " · avg " + speechFmt(p.mean_s, " s")],
        ["Lexical richness", speechJoin(lx.mattr != null ? "MATTR " + speechFmt(lx.mattr) : null, lx.words != null ? speechFmt(lx.words) + " words" : null)],
        ["Lexical density", speechFmt(lx.lexical_density)],
    ];
    h += rows.map(([k, val]) =>
        '<div class="metric"><span class="metric-name">' + k + '</span><span class="metric-value">' + val + "</span></div>"
    ).join("");
    const devs = (b.deviations || []).slice().sort((x, y) => Math.abs(y.z) - Math.abs(x.z)).slice(0, 5);
    if (devs.length) {
        h += '<div class="speech-sub">Compared with past visits</div><div class="speech-dev">';
        h += devs.map(d => {
            const z = (d.z > 0 ? "+" : "") + Number(d.z).toFixed(2);
            const cls = (d.lowers_score && Math.abs(d.z) >= 0.5) ? ' class="lowers"' : "";
            return '<div class="metric"><span class="metric-name">' + speechEsc(d.label) + '</span>'
                + '<span class="metric-value"><span' + cls + ">z " + z + "</span> <small>"
                + speechFmt(d.value) + " vs " + speechFmt(d.baseline_mean) + "</small></span></div>";
        }).join("") + "</div>";
    } else if (b.provisional) {
        h += '<div class="hint">' + speechFmt(b.sessions_needed) + " more usable visit(s) before scores compare against this patient's own baseline.</div>";
    }
    for (const n of (r.notes || [])) h += '<div class="speech-note">' + speechEsc(n) + "</div>";
    const st = r.stored;
    h += '<div class="hint">' + (st ? "Saved (" + speechEsc(st.backend || "ok") + ") · visit " + speechEsc(r.visit_id)
        : "Not saved: " + speechEsc(r.store_error || "no storage")) + "</div>";
    h += '<div class="hint">' + speechFmt(b.disclaimer || "Not a diagnostic output.") + "</div>";
    return h;
}

function speechMessage(m) {
    if (m.type === "error") {
        speechStatus("Speech server: " + m.error, true);
        return;
    }
    if (m.kind === "speech_live" && speech.phase === "recording") {
        $("speechLive").innerHTML = speechLiveHTML(m);
    } else if (m.kind === "speech_report") {
        speech.lastReport = m;
        speechCloseSocket();
        speech.phase = "idle";
        $("speechLive").hidden = true;
        $("speechReport").innerHTML = speechReportHTML(m);
        $("speechReport").hidden = false;
        speechStatus("");
        speechRender();
    }
}

$("speechButton").onclick = speechToggle;
$("roomInput").addEventListener("input", speechRender);
window.addEventListener("load", speechRender);

</script>

</body>

</html>
"""


# ============================================================
# ROUTES
# ============================================================

async def doctor_page(
    request,
):

    return web.Response(
        text=DOCTOR_HTML,
        content_type="text/html",
    )


def create_app():

    app = web.Application(
        client_max_size=
            1024 ** 3
    )

    app.router.add_get(
        "/",
        doctor_page,
    )

    app.router.add_get(
        "/doctor",
        doctor_page,
    )

    app.router.add_get(
        "/ws",
        websocket_handler,
    )

    return app


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "======================================"
    )

    print(
        " TeleVision Doctor (live)"
    )

    print(
        "======================================"
    )

    print(
        f"Dashboard: http://localhost:{PORT}"
    )



    print(
        "======================================"
    )

    print()

    web.run_app(
        create_app(),
        host=HOST,
        port=PORT,
    )
