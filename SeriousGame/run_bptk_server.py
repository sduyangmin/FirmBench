#!/usr/bin/env python3
"""
run_bptk_server.py

Start a BPTK-Py BptkServer (Flask REST API) that exposes simulation scenarios
(e.g., the Enterprise Digital Twin scenario manager smEDT) over HTTP.

Usage (recommended):
  # 1) Clone the tutorial repo and install deps as per its README.
  # 2) Run this script from the repo root (the folder that contains ./scenarios):
  python run_bptk_server.py --host 0.0.0.0 --port 5000

Or specify repo root explicitly:
  python run_bptk_server.py --repo-root /path/to/bptk_py_tutorial --host 127.0.0.1 --port 5000

Verify:
  curl http://localhost:5000/healthy
  curl http://localhost:5000/scenarios
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Path to the bptk_py_tutorial repo root (must contain a ./scenarios directory). Default: current dir.",
    )
    p.add_argument("--host", type=str, default="127.0.0.1", help="Flask bind host. Use 0.0.0.0 for remote access.")
    p.add_argument("--port", type=int, default=5000, help="Flask port. Default: 5000")
    p.add_argument(
        "--bearer-token",
        type=str,
        default=os.getenv("BPTK_BEARER_TOKEN", ""),
        help="Optional bearer token for BptkServer. Can also be set via BPTK_BEARER_TOKEN env var.",
    )
    p.add_argument(
        "--cors",
        action="store_true",
        help="Enable permissive CORS (useful if you call the server from a browser).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    repo_root = Path(args.repo_root).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    scenarios_dir = repo_root / "scenarios"
    if not scenarios_dir.exists() or not scenarios_dir.is_dir():
        print(
            f"[ERROR] Could not find scenarios directory at: {scenarios_dir}\n"
            f"Make sure you point --repo-root to the bptk_py_tutorial repo root.",
            file=sys.stderr,
        )
        return 2

    # Important: BPTK-Py scans ./scenarios on startup.
    os.chdir(str(repo_root))

    try:
        # BPTK factory and server
        from BPTK_Py.bptk import bptk  # type: ignore
        from BPTK_Py.server import BptkServer  # type: ignore
    except Exception as e:
        print(
            "[ERROR] Failed to import BPTK_Py. Ensure bptk-py is installed in your environment.\n"
            f"Import error: {e}",
            file=sys.stderr,
        )
        return 3

    def bptk_factory():
        # Instantiating bptk() scans ./scenarios and loads scenario managers/models.
        return bptk()

    app = BptkServer(__name__, bptk_factory=bptk_factory, bearer_token=(args.bearer_token or None))

    if args.cors:
        try:
            from flask_cors import CORS  # type: ignore

            CORS(app)
        except Exception as e:
            print(f"[WARN] --cors requested but flask-cors not installed: {e}", file=sys.stderr)

    print(f"[INFO] Repo root: {repo_root}")
    print(f"[INFO] Scenarios dir: {scenarios_dir}")
    print(f"[INFO] BptkServer listening on http://{args.host}:{args.port}")
    if args.bearer_token:
        print("[INFO] Bearer token auth: ENABLED")
    else:
        print("[INFO] Bearer token auth: DISABLED")

    # Flask run
    app.run(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
