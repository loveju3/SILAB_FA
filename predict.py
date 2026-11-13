#!/usr/bin/env python3
"""Assignment-compatible entry point; implementation lives in Solution/."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    import sys
    solution = Path(__file__).resolve().parent / "Solution"
    sys.path.insert(0, str(solution))
    runpy.run_path(str(solution / "predict.py"), run_name="__main__")
