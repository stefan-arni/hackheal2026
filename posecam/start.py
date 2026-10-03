"""Start everything for a BESS session in one go.

    python start.py

Starts the pose server (server.py, port 8765: MediaPipe pose, LiDAR depth, BESS
scoring) and the doctor call server (doctor_call_server.py, port 8088: the video
call and the doctor's dashboard), waits until both are ready, prints the address
to type into the TeleVision app, and opens the doctor page. Ctrl+C stops both.
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
    busy = [(name, port) for name, _, port in SERVERS if port_in_use(port)]
    if busy:
        for name, port in busy:
            print(f"! The {name} (port {port}) is already running in another terminal.")
        print(f"  Stop {'them' if len(busy) > 1 else 'it'} there with Ctrl+C, "
              "then run `python start.py` again.")
        sys.exit(1)

    procs = []
    for name, script, port in SERVERS:
        procs.append((name, subprocess.Popen([sys.executable, script], cwd=HERE)))

    try:
        deadline = time.time() + 90
        for (name, _, port), (_, p) in zip(SERVERS, procs):
            while not port_in_use(port):
                if p.poll() is not None:
                    print(f"\n! The {name} stopped while starting (see the error above).")
                    stop(procs)
                    sys.exit(1)
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
        print("  Stop:    Ctrl+C here")
        print("=" * 60 + "\n")
        webbrowser.open(DOCTOR_URL)

        while True:
            for name, p in procs:
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
