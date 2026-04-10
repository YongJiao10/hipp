#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def _parse_toml_scalar(value: str):
    value = value.strip()
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return value


def load_config(path: Path) -> dict:
    data: dict[str, object] = {}
    current = data
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                section_dict: dict[str, object] = {}
                data[section] = section_dict
                current = section_dict
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            current[key.strip()] = _parse_toml_scalar(value)
    return data


def detect_func_mode(cfg: dict) -> str:
    input_dir = Path(cfg.get("local_input_dir", "data/hippunfold_input"))
    subj = str(cfg["subject"])
    func_dir = input_dir / f"sub-{subj}" / "func"
    dtseries = sorted(func_dir.glob("*.dtseries.nii"))
    if dtseries:
        return "cifti"
    volume = sorted(func_dir.glob("*_bold.nii.gz"))
    if volume:
        return "volume"
    return "missing"


def check_inputs(cfg: dict) -> dict:
    input_dir = Path(cfg.get("local_input_dir", "data/hippunfold_input"))
    subj = str(cfg["subject"])
    func_dir = input_dir / f"sub-{subj}" / "func"
    anat_dir = input_dir / f"sub-{subj}" / "anat"
    report = {
        "input_dir": str(input_dir),
        "exists": input_dir.exists(),
        "anat_files": sorted(p.name for p in anat_dir.glob("*")) if anat_dir.exists() else [],
        "func_files": sorted(p.name for p in func_dir.glob("*")) if func_dir.exists() else [],
        "functional_mode": detect_func_mode(cfg),
    }
    return report


def build_hippunfold_command(cfg: dict) -> list[str]:
    input_dir = Path(cfg.get("local_input_dir", "data/hippunfold_input"))
    out_dir = Path(cfg["output_dir"]) / "hippunfold"
    subj = cfg["subject"]
    cmd = [
        "hippunfold",
        str(input_dir),
        str(out_dir),
        "participant",
        "--modality",
        cfg["hippunfold_modality"],
        "--output-density",
        cfg["surface_density"],
        "--output-spaces",
        "corobl",
        "--participant-label",
        subj,
    ]
    if cfg.get("t1_reg_template", False):
        cmd.append("--t1_reg_template")
    return cmd


def ensure_func_mode_allowed(cfg: dict, mode: str) -> None:
    if mode == "volume" and not cfg.get("allow_volume_reference_fallback", False):
        raise RuntimeError(
            "Detected volume-only rsfMRI input. Volume-based neocortical reference extraction is not enabled. "
            "Per project rule, stop here and report to the user before switching away from the CIFTI-first route."
        )
    if mode == "missing":
        raise RuntimeError("No rsfMRI input found in local_input_dir")


def main() -> int:
    parser = argparse.ArgumentParser(description="Orchestrate the HCP 7T HippoMaps pipeline")
    parser.add_argument("--config", default="config/hippo_pipeline.toml")
    parser.add_argument(
        "action",
        choices=["show-config", "check-inputs", "build-hippunfold-command"],
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.action == "show-config":
        print(json.dumps(cfg, indent=2))
        return 0

    if args.action == "check-inputs":
        report = check_inputs(cfg)
        print(json.dumps(report, indent=2))
        ensure_func_mode_allowed(cfg, report["functional_mode"])
        return 0

    if args.action == "build-hippunfold-command":
        report = check_inputs(cfg)
        ensure_func_mode_allowed(cfg, report["functional_mode"])
        cmd = build_hippunfold_command(cfg)
        print(" ".join(cmd))
        return 0

    raise RuntimeError(f"Unhandled action: {args.action}")


if __name__ == "__main__":
    raise SystemExit(main())
