"""Load settings from a .env file into environment variables (no extra packages).

    from env import load_env
    load_env()          # reads .env next to this file, then the current folder

Format (same as python-dotenv for the common cases):
    # comment
    SUPABASE_URL=https://abc.supabase.co
    SUPABASE_SERVICE_KEY="eyJ..."        quotes optional
    export SUPABASE_SPEECH_BUCKET=abc    "export" prefix allowed

Variables already set in the real environment win over the file, so a deployed
server can override .env without editing it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def parse_env(text: str) -> dict[str, str]:
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
            if raw.strip().split("=", 1)[1].strip().startswith('"'):
                val = val.replace("\\n", "\n")
        else:
            val = re.split(r"\s+#", val, maxsplit=1)[0].strip()   # trailing comment
        out[key] = val
    return out


def load_env(*paths: str | Path, override: bool = False) -> list[str]:
    """Load .env files (default: next to this file, then the current folder).
    Returns the files that were read."""
    if not paths:
        paths = (Path(__file__).parent / ".env", Path.cwd() / ".env")
    seen, read = set(), []
    for p in paths:
        p = Path(p).resolve()
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        for k, v in parse_env(p.read_text(encoding="utf-8-sig")).items():
            if override or k not in os.environ:
                os.environ[k] = v
        read.append(str(p))
    return read
