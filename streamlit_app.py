"""Root entrypoint for Streamlit Community Cloud (auto-detected by share.streamlit.io)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

runpy.run_path(str(_SRC / "forgeai" / "app.py"), run_name="__main__")
