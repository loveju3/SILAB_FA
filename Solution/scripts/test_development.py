#!/usr/bin/env python3
"""Run development tests with dev validation disabled and dev reads blocked."""
import builtins
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    os.environ.pop("SILAB_VALIDATE_DEV", None)
    suite = unittest.TestSuite()
    for folder in (ROOT / "Solution/tests", ROOT / "helper_tool"):
        suite.addTests(unittest.TestLoader().discover(str(folder), pattern="test_*.py"))

    def guard(original):
        def guarded(file, *args, **kwargs):
            if isinstance(file, (str, Path)) and Path(file).name == "dev.jsonl":
                raise AssertionError("開發測試禁止開啟 dev.jsonl")
            return original(file, *args, **kwargs)
        return guarded

    with patch("builtins.open", guard(builtins.open)), patch("io.open", guard(io.open)):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
