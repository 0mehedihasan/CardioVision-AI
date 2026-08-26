#!/usr/bin/env python3
"""
pytest entry point for the verification suites.

The five suites in this directory are executable scripts, not pytest modules,
and that is deliberate: each one stubs whatever part of the ML stack is missing
and prints a per-assertion report, so it runs on a machine where torch was never
installed and tells you exactly which invariant broke. Rewriting them as
``def test_*`` functions would trade that for a traceback.

This wrapper gives ``pytest`` something real to collect: one test per suite,
each running the script in a subprocess and asserting a clean exit. The suites'
own output is captured and printed on failure, so a red run still names the
broken invariant rather than just the exit code.

``[tool.pytest.ini_options] python_files`` in pyproject.toml points only at this
file. Without that, pytest would import the suites as modules and execute their
top-level assertions during collection, where a deliberate ``sys.exit(0)`` — the
architecture suite does this when the LFS checkpoint is unresolved — reads as a
collection error rather than as the skip it is.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

# Ordered cheapest-first, so a broken shared foundation surfaces before four
# suites fail on top of it.
SUITES = (
    "test_case_lifecycle.py",
    "test_ecg_reporting.py",
    "test_ecg_pipeline.py",
    "test_ecg_rendering.py",
    "test_ecg_architecture.py",
)


@pytest.mark.parametrize("suite", SUITES)
def test_suite(suite: str) -> None:
    script = HERE / suite

    assert script.is_file(), f"{suite} is missing from tests/"

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        # Suites that touch the checkpoint or write a temporary database are
        # slow on a cold filesystem, but none of them is minutes-slow. A cap
        # keeps a hung subprocess from taking the whole run with it.
        timeout=600,
        cwd=script.parent.parent,
    )

    if result.returncode != 0:
        # The suite already explained itself; reprinting its report is more
        # useful than any assertion message this wrapper could write.
        print(result.stdout)
        print(result.stderr, file=sys.stderr)

    assert result.returncode == 0, (
        f"{suite} exited {result.returncode} — see the captured report above"
    )
