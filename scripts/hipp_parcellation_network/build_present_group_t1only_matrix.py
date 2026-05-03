#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_PRESENT = REPO_ROOT / "scripts" / "hipp_parcellation_network" / "build_present_group_overview_merged_canonical.py"
DEFAULT_PYTHON = "/home/yojiao/.conda/envs/py314/bin/python"
DEFAULT_ROUTE_ROOT = REPO_ROOT / "outputs_experiments" / "hippunfold_recommended_t1w_20260423" / "kong2022"
DEFAULT_PRESENT_DIR = REPO_ROOT / "present_group_t1only"
DEFAULT_MANIFEST_NAME = "group_overview_manifest_t1only_8way.json"
METHODS = ["group-intrinsic-spectral", "group-network-spectral"]
ATLAS = "kong2022"


def build_root_matrix(route_root: Path) -> list[dict[str, str]]:
    return [
        {
            "route_tag": "routea",
            "mode_tag": "allon",
            "group_root": str(route_root / "hipp_group_merged_canonical_atlas_routea_allon"),
        },
        {
            "route_tag": "routea",
            "mode_tag": "alloff",
            "group_root": str(route_root / "hipp_group_merged_canonical_atlas_routea_alloff"),
        },
        {
            "route_tag": "routeb",
            "mode_tag": "allon",
            "group_root": str(route_root / "hipp_group_merged_canonical_atlas_routeb_allon"),
        },
        {
            "route_tag": "routeb",
            "mode_tag": "alloff",
            "group_root": str(route_root / "hipp_group_merged_canonical_atlas_routeb_alloff"),
        },
    ]


def expected_overview_path(present_dir: Path, *, method: str, route_tag: str, mode_tag: str) -> Path:
    return present_dir / f"{ATLAS}_{method}_group_overview_{route_tag}_{mode_tag}.png"


def write_readme(present_dir: Path, items: list[dict[str, str]]) -> None:
    lines = [
        "# present_group_t1only 8-way overviews",
        "",
        "This directory is the single maintained export root.",
        "",
        "| route | mode | method | overview | source_group_root |",
        "|---|---|---|---|---|",
    ]
    for item in items:
        png_name = Path(item["overview_png"]).name
        lines.append(
            f"| {item['route_tag']} | {item['mode_tag']} | {item['method']} | "
            f"[{png_name}]({png_name}) | `{item['source_group_root']}` |"
        )
    (present_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="One-shot exporter for present_group_t1only 8-way overview matrix.")
    p.add_argument("--python-exe", default=DEFAULT_PYTHON)
    p.add_argument("--route-root", default=str(DEFAULT_ROUTE_ROOT))
    p.add_argument("--present-dir", default=str(DEFAULT_PRESENT_DIR))
    p.add_argument("--coverage-threshold", type=float, default=0.80)
    p.add_argument("--manifest-name", default=DEFAULT_MANIFEST_NAME)
    p.add_argument("--scene", default=None, help="Optional override for wb scene path.")
    args = p.parse_args()

    route_root = Path(args.route_root).resolve()
    present_dir = Path(args.present_dir).resolve()
    present_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    combos = build_root_matrix(route_root)
    for combo in combos:
        group_root = Path(combo["group_root"]).resolve()
        if not group_root.exists():
            raise FileNotFoundError(f"Missing group_root: {group_root}")
        route_tag = combo["route_tag"]
        mode_tag = combo["mode_tag"]
        for method in METHODS:
            cmd = [
                args.python_exe,
                str(BUILD_PRESENT),
                "--group-root",
                str(group_root),
                "--present-dir",
                str(present_dir),
                "--atlases",
                ATLAS,
                "--methods",
                method,
                "--coverage-threshold",
                f"{float(args.coverage_threshold):.2f}",
                "--route-tag",
                route_tag,
                "--mode-tag",
                mode_tag,
                "--skip-manifest",
                "--skip-readme",
            ]
            if args.scene:
                cmd.extend(["--scene", str(Path(args.scene).resolve())])
            subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
            overview_png = expected_overview_path(
                present_dir,
                method=method,
                route_tag=route_tag,
                mode_tag=mode_tag,
            ).resolve()
            if not overview_png.exists():
                raise FileNotFoundError(f"Expected overview not found: {overview_png}")
            records.append(
                {
                    "route_tag": route_tag,
                    "mode_tag": mode_tag,
                    "method": method,
                    "source_group_root": str(group_root),
                    "overview_png": str(overview_png),
                }
            )

    if len(records) != 8:
        raise RuntimeError(f"Expected 8 overview outputs, got {len(records)}")

    payload = {
        "present_dir": str(present_dir),
        "count": len(records),
        "items": records,
    }
    (present_dir / args.manifest_name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_readme(present_dir, records)
    print(json.dumps({"present_dir": str(present_dir), "manifest": args.manifest_name, "count": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
