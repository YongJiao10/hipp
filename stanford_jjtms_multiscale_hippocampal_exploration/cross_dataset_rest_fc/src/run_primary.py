"""CLI entrypoint for the frozen Task 6 real-data run."""

from __future__ import annotations

try:
    from .reporting import main
except ImportError:  # pragma: no cover
    from reporting import main  # type: ignore


if __name__ == "__main__":
    raise SystemExit(main())
