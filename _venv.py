"""Run root entry points with the project .venv, even when started as `python3 xxx.py`.

`python3 install.py` creates .venv. Afterwards `python3 predict.py IN OUT`,
`python3 ask.py ...` and `python3 patrol.py` re-launch themselves with
.venv's Python when another interpreter started them, so nobody has to
activate the environment first. Without .venv (e.g. Docker images) or with
SILAB_NO_VENV=1 this is a no-op.
"""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
GUARD = "SILAB_NO_VENV"


def venv_python(venv=None):
    venv = VENV if venv is None else venv
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure():
    python = venv_python()
    if os.environ.get(GUARD) or not python.is_file():
        return
    try:
        if Path(sys.prefix).resolve() == VENV.resolve():
            return
    except OSError:
        return
    os.environ[GUARD] = "1"  # the relaunched process must not relaunch again
    os.environ.setdefault("PYTHONUTF8", "1")  # avoid cp950 defaults on Traditional Chinese Windows
    command = [str(python), *sys.argv]
    if os.name == "nt":  # os.execv does not keep the console attached on Windows
        sys.exit(subprocess.run(command).returncode)
    os.execv(str(python), command)
