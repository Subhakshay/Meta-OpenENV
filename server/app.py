"""
server/app.py — OpenEnv-compliant entry point.

This module re-exports the FastAPI app from main.py so that the
OpenEnv validator can find it at the expected path (server/app.py).

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import sys
import os

# Ensure the project root is on sys.path so main.py is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import app  # noqa: F401 — re-export


def main():
    """Entry point for `python -m server.app` or pyproject.toml scripts."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
