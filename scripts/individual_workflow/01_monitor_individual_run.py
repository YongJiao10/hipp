#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parent / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from routea_individual_gradients_wta import DEFAULT_OUT_ROOT, build_monitor_summary, write_status_tsv, manifest_rows


def latest_heartbeat(out_root: Path) -> Path:
    logs = Path(out_root) / "_runs" / "logs"
    candidates = sorted(logs.glob("*/heartbeat.json"), key=lambda p: p.stat().st_mtime_ns if p.exists() else 0)
    if candidates:
        return candidates[-1]
    return logs / "missing" / "heartbeat.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor RouteA individual gradients+WTA background run.")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--heartbeat", default=None)
    parser.add_argument("--stale-seconds", type=int, default=120)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    heartbeat = Path(args.heartbeat).resolve() if args.heartbeat else latest_heartbeat(out_root)
    rows = manifest_rows(out_root)
    write_status_tsv(out_root / "_runs" / "summaries" / "combo_status.tsv", rows)
    summary = build_monitor_summary(
        out_root=out_root,
        heartbeat_path=heartbeat,
        stale_seconds=int(args.stale_seconds),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"status={summary['status']} ok={summary['ok']} failed={summary['failed']} running={summary['running']} stale={summary['heartbeat_stale']}")
        print(f"heartbeat={summary['heartbeat_path']}")
        print(f"status_tsv={out_root / '_runs' / 'summaries' / 'combo_status.tsv'}")
    return 1 if summary["failed"] or summary["heartbeat_stale"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
