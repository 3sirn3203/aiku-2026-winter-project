from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict


def run_python_code(code: str, workdir: str, timeout_sec: int = 300) -> Dict[str, Any]:
    Path(workdir).mkdir(parents=True, exist_ok=True)
    start = time.time()
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=workdir, delete=False) as f:
        f.write(code)
        script_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=workdir,
        )
        return {
            "script_path": script_path,
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "duration_sec": round(time.time() - start, 3),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "script_path": script_path,
            "exit_code": -1,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "duration_sec": round(time.time() - start, 3),
            "error": "timeout",
        }
