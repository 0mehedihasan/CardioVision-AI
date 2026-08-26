"""
Command-line entry point: ``cardiovision serve``.

A thin wrapper over uvicorn. It exists for two reasons that a bare uvicorn
command line does not cover.

First, discoverability: ``cardiovision serve`` is one word to remember, whereas
``uvicorn cardiovision.api.app:app --port 8000`` requires knowing the module
path to an application factory.

Second, and more usefully, ``--skip`` maps friendly model names onto the
environment variables the app reads at startup. Skipping MedGemma while
iterating on the imaging pipeline saves loading 8.6 GB of weights, and needing
to remember ``CARDIOVISION_SKIP_MEDGEMMA=1`` to do it is a papercut that gets
paid on every restart.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

from cardiovision import __version__

# Friendly name -> the environment variable the app checks in its lifespan.
SKIPPABLE = {
    "ccta": "CARDIOVISION_SKIP_CCTA",
    "echo": "CARDIOVISION_SKIP_ECHO",
    "ecg": "CARDIOVISION_SKIP_ECG",
    "medgemma": "CARDIOVISION_SKIP_MEDGEMMA",
}

APP_PATH = "cardiovision.api.app:app"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cardiovision",
        description="CardioVision AI — local cardiac imaging and ECG analysis.",
    )
    parser.add_argument(
        "--version", action="version", version=f"CardioVision AI {__version__}",
    )

    subcommands = parser.add_subparsers(dest="command", required=True)

    serve = subcommands.add_parser("serve", help="Run the backend API server.")
    serve.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Interface to bind. Defaults to localhost: there is one shared "
            "password and no transport encryption, so binding to 0.0.0.0 puts "
            "patient data on the network behind an access gate that was never "
            "meant to be one."
        ),
    )
    serve.add_argument("--port", type=int, default=8000, help="Port (default 8000).")
    serve.add_argument(
        "--reload",
        action="store_true",
        help="Reload on source changes. Development only — it reloads the "
             "models too, which takes as long as a cold start.",
    )
    serve.add_argument(
        "--skip",
        action="append",
        choices=sorted(SKIPPABLE),
        default=[],
        metavar="MODEL",
        help=(
            "Do not load this model at startup. Repeatable. Choices: "
            + ", ".join(sorted(SKIPPABLE))
            + "."
        ),
    )
    serve.add_argument(
        "--log-level",
        default="info",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
    )

    subcommands.add_parser(
        "check",
        help="Report which models and checkpoints are available, then exit.",
    )

    return parser


def _serve(args: argparse.Namespace) -> int:
    for name in args.skip:
        os.environ[SKIPPABLE[name]] = "1"

    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is not installed, so the server cannot start.\n"
            "Install the project with its dependencies:\n"
            '    pip install -e ".[dev]"',
            file=sys.stderr,
        )
        return 1

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"[notice] Binding to {args.host}, not localhost. This service has "
            "one shared password and no TLS;\n"
            "         anyone who can reach this port can reach the case "
            "database.",
            file=sys.stderr,
        )

    uvicorn.run(
        APP_PATH,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


def _check() -> int:
    """
    Report what would load, without starting a server.

    Useful as the first thing to run after a fresh clone: it separates "the
    checkpoint is missing" from "the server is broken", which otherwise both
    present as a 503 from the UI.
    """
    from cardiovision.config import (
        CCTA_CHECKPOINT_PATH,
        DEVICE,
        ECG_CHECKPOINT_PATH,
        ECHO_CHECKPOINT_PATH,
        MEDGEMMA_PATH,
        PROJECT_ROOT,
        TORCH_AVAILABLE,
    )

    print(f"CardioVision AI {__version__}")
    print(f"  project root : {PROJECT_ROOT}")
    print(f"  device       : {DEVICE}")
    print(f"  torch        : {'available' if TORCH_AVAILABLE else 'NOT INSTALLED'}")
    print()

    artefacts = (
        ("CCTA checkpoint", CCTA_CHECKPOINT_PATH),
        ("echo checkpoint", ECHO_CHECKPOINT_PATH),
        ("ECG checkpoint", ECG_CHECKPOINT_PATH),
        ("MedGemma weights", MEDGEMMA_PATH),
    )

    missing = 0
    for label, path in artefacts:
        if path.exists():
            size = (
                sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                if path.is_dir() else path.stat().st_size
            )
            print(f"  [ok]      {label}: {size / 1e6:.0f} MB")
        else:
            missing += 1
            print(f"  [missing] {label}: {path}")

    if missing:
        print(
            f"\n{missing} of {len(artefacts)} model artefacts are missing. The "
            "server will still start;\nthe modalities they back will report "
            "themselves unavailable through /api/health."
        )

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "serve":
        return _serve(args)
    if args.command == "check":
        return _check()

    return 1                                            # pragma: no cover


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
