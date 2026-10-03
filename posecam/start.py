"""Start everything for a BESS session in one go.

    python start.py

Starts the pose server (server.py, port 8765: MediaPipe pose, LiDAR depth, BESS
scoring) and the doctor call server (doctor_call_server.py, port 8088: the video
call and the doctor's dashboard), waits until both are ready, prints the address
to type into the TeleVision app, and opens the doctor page. Ctrl+C stops both.

If speech_server.py is here it also starts the speech server (port 8767) for
the doctor page's Start/End speech recording button. It is optional: if it
can't start (missing packages), the rest keeps running. Skip it with
    python start.py --no-speech
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
SERVERS = [
    ("pose server", "server.py", 8765),
    ("call server", "doctor_call_server.py", 8088),
]
# optional: the doctor page's speech recording button talks to it
SPEECH = ("speech server", "speech_server.py --whisper-model tiny.en", 8767)
DOCTOR_URL = "http://localhost:8088"


def port_in_use(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def lan_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def stop(procs):
    for _, p in procs:
        if p.poll() is None:
            p.terminate()
    for _, p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


def main():
    servers = list(SERVERS)
    optional = set()
    speech_note = None
    if "--no-speech" in sys.argv or not os.path.exists(os.path.join(HERE, SPEECH[1].split()[0])):
        pass
    elif port_in_use(SPEECH[2]):
        speech_note = "already running in another terminal (port 8767)"
    else:
        servers.append(SPEECH)
        optional.add(SPEECH[0])
    busy = [(name, port) for name, _, port in servers if port_in_use(port)]
    if busy:
        for name, port in busy:
            print(f"! The {name} (port {port}) is already running in another terminal.")
        print(f"  Stop {'them' if len(busy) > 1 else 'it'} there with Ctrl+C, "
              "then run `python start.py` again.")
        sys.exit(1)

    procs = []
    for name, script, port in servers:
        procs.append((name, subprocess.Popen([sys.executable, *script.split()], cwd=HERE)))

    try:
        deadline = time.time() + 90
        for (name, _, port), (_, p) in zip(servers, procs):
            while not port_in_use(port):
                if p.poll() is not None and name in optional:
                    print(f"\n! The {name} didn't start (see the error above); "
                          "continuing without speech recording.")
                    break
                if p.poll() is not None:
                    print(f"\n! The {name} stopped while starting (see the error above).")
                    stop(procs)
                    sys.exit(1)
                if time.time() > deadline and name in optional:
                    print(f"\n! The {name} is still starting; the doctor page's speech button "
                          "will work once it's up.")
                    break
                if time.time() > deadline:
                    print(f"\n! The {name} didn't start within 90 s.")
                    stop(procs)
                    sys.exit(1)
                time.sleep(0.3)

        ip = lan_ip()
        print("\n" + "=" * 60)
        print("  Everything is running.")
        print(f"  Doctor:  {DOCTOR_URL}   (opening it now)")
        print(f"  Phone:   TeleVision app -> server IP  {ip}   room  television-demo")
        if SPEECH[0] in optional:
            speech_up = port_in_use(SPEECH[2])
            print("  Speech:  " + ("ready on port 8767 (Start speech recording on the doctor page)"
                                   if speech_up else "not running (see the message above)"))
        elif speech_note:
            print(f"  Speech:  {speech_note}")
        print("  Stop:    Ctrl+C here")
        print("=" * 60 + "\n")
        webbrowser.open(DOCTOR_URL)

        while True:
            for name, p in list(procs):
                if p.poll() is not None and name in optional:
                    print(f"\n! The {name} stopped (see the error above). Speech recording is off; the rest keeps running.")
                    procs.remove((name, p))
                    continue
                if p.poll() is not None:
                    print(f"\n! The {name} stopped unexpectedly (see the error above). Stopping everything.")
                    stop(procs)
                    sys.exit(1)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop(procs)


if __name__ == "__main__":
    main()
