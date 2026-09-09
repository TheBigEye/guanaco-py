"""Allows the build system to run as ``python -m guanaco <command>``."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
