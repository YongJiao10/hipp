#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
KONG2022_BASE = REPO_ROOT / "outputs_experiments" / "hippunfold_recommended_t1w_20260423" / "kong2022"
FIXEDK_WORK = KONG2022_BASE / "fixedk_work"
PRESENT_FIXEDK = REPO_ROOT / "present_fixedk"
GROUP_ATLAS_SCRIPT = REPO_ROOT / "scripts" / "core_workflow" / "05_01_run_group_atlas.py"
BUILD_PRESENT_SCRIPT = REPO_ROOT / "scripts" / "core_workflow" / "06_01_build_group_overview.py"
MATERIALIZE_LABELS_SCRIPT = REPO_ROOT / "scripts" / "core_workflow" / "06_03_materialize_group_labels.py"
DENSE_BATCH_ROOT = REPO_ROOT / "outputs_experiments" / "hippunfold_recommended_t1w_20260423" / "dense_corobl_batch"
SHARED_ROOT = KONG2022_BASE / "shared_inputs" / "routea_151"
SURFACE_ROOT = KONG2022_BASE / "shared_inputs" / "surface_151"
SUBJECTS_MANIFEST = KONG2022_BASE / "manifests" / "hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt"
GROUP_AVG = KONG2022_BASE / "templates" / "hipp_group_average_t1only_151" / "group_average_surfaces"
SMOOTHINGS = ("2mm", "4mm")
HEMIS = ("L", "R")


@dataclass(frozen=True)
class ModeSpec:
    mode: str
    canonical_root: Path
    subject_fc_mode: str
    subject_fc_top_percent: int | None
    render_mode_tag: str


MODE_SPECS: tuple[ModeSpec, ...] = (
    ModeSpec(
        mode="10perc",
        canonical_root=KONG2022_BASE / "hipp_group_merged_canonical_atlas_routea_10perc",
        subject_fc_mode="top-percent",
        subject_fc_top_percent=10,
        render_mode_tag="10perc",
    ),
    ModeSpec(
        mode="full",
        canonical_root=KONG2022_BASE / "hipp_group_merged_canonical_atlas_routea_full",
        subject_fc_mode="full",
        subject_fc_top_percent=None,
        render_mode_tag="full",
    ),
)


def run_dir(root: Path, smoothing: str, hemi: str) -> Path:
    return root / "network-spectral" / "kong2022" / smoothing / "cov_0p80" / f"hemi_{hemi}"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_copytree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source root: {src}")
    if not dst.exists():
        import shutil

        shutil.copytree(src, dst)


def get_chosen_k(root: Path, smoothing: str, hemi: str) -> int:
    manifest = load_json(run_dir(root, smoothing, hemi) / "group_atlas_manifest.json")
    value = manifest.get("chosen_k_final", -1)
    if value is None or value == "":
        return -1
    return int(value)


def has_forced_selection(root: Path, smoothing: str, hemi: str, k_target: int) -> bool:
    combo_dir = run_dir(root, smoothing, hemi)
    manifest = load_json(combo_dir / "group_atlas_manifest.json")
    if int(manifest.get("chosen_k_final", -1) or -1) != int(k_target):
        return False
    if int(manifest.get("force_final_k", -1) or -1) != int(k_target):
        return False
    import csv

    rows = list(csv.DictReader((combo_dir / "k_selection_summary.tsv").open("r", encoding="utf-8"), delimiter="\t"))
    selected = [int(r["k"]) for r in rows if int(r.get("selected_final", "0")) == 1]
    if len(selected) != 1 or int(selected[0]) != int(k_target):
        return False
    return True


def run_fixed_k_combo(root: Path, spec: ModeSpec, k_target: int, smoothing: str, hemi: str) -> None:
    cmd = [
        sys.executable,
        str(GROUP_ATLAS_SCRIPT),
        "--shared-root",
        str(SHARED_ROOT),
        "--surface-root",
        str(SURFACE_ROOT),
        "--out-root",
        str(root),
        "--subjects-manifest",
        str(SUBJECTS_MANIFEST),
        "--hippunfold-batch-root",
        str(DENSE_BATCH_ROOT),
        "--group-average-surfaces-dir",
        str(GROUP_AVG),
        "--jobs",
        "1",
        "--coverage-thresholds",
        "0.80",
        "--instability-resamples",
        "10",
        "--null-resamples",
        "10",
        "--k-selection-mode",
        "redundancy-aware",
        "--subject-fc-mode",
        spec.subject_fc_mode,
        "--atlases",
        "kong2022",
        "--methods",
        "network-spectral",
        "--smoothings",
        smoothing,
        "--hemis",
        hemi,
        "--force-final-k",
        str(int(k_target)),
        "--k-min",
        "2",
        "--k-max",
        "10",
    ]
    if spec.subject_fc_top_percent is not None:
        cmd.extend(["--subject-fc-top-percent", str(int(spec.subject_fc_top_percent))])
    print("[rerun]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def enforce_fixed_k(root: Path, spec: ModeSpec, k_target: int) -> list[tuple[str, str, int]]:
    unsatisfied: list[tuple[str, str, int]] = []
    for smoothing in SMOOTHINGS:
        for hemi in HEMIS:
            if has_forced_selection(root, smoothing, hemi, int(k_target)):
                continue
            run_fixed_k_combo(root=root, spec=spec, k_target=k_target, smoothing=smoothing, hemi=hemi)
            after = get_chosen_k(root, smoothing, hemi)
            if after != int(k_target) or not has_forced_selection(root, smoothing, hemi, int(k_target)):
                unsatisfied.append((smoothing, hemi, after))
    return unsatisfied


def sync_count_to_fisher_tree(count_root: Path, fisher_root: Path) -> None:
    import shutil

    ensure_copytree(count_root, fisher_root)
    for smoothing in SMOOTHINGS:
        for hemi in HEMIS:
            src = run_dir(count_root, smoothing, hemi)
            dst = run_dir(fisher_root, smoothing, hemi)
            shutil.copytree(src, dst, dirs_exist_ok=True)


def build_fisher_rows(profile: np.ndarray, network_names: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cidx in range(profile.shape[0]):
        row = profile[cidx, :]
        order = np.argsort(row)
        top1_i = int(order[-1])
        top2_i = int(order[-2]) if row.size > 1 else int(order[-1])
        top1 = float(row[top1_i])
        top2 = float(row[top2_i])
        margin = float(top1 - top2)
        rows.append(
            {
                "cluster_id": int(cidx + 1),
                "status": "dominant",
                "top1_network": str(network_names[top1_i]),
                "top1_fisher_z": top1,
                "top2_network": str(network_names[top2_i]),
                "top2_fisher_z": top2,
                "margin": margin,
                "low_margin_lt_0p05": int(margin < 0.05),
            }
        )
    return rows


def convert_to_fisher_assignment(root: Path) -> None:
    for smoothing in SMOOTHINGS:
        for hemi in HEMIS:
            combo_dir = run_dir(root, smoothing, hemi)
            profile = np.load(combo_dir / "cluster_network_profile.npy").astype(np.float32, copy=False)
            names = [str(x) for x in load_json(combo_dir / "group_feature_summary.json")["network_names"]]
            status_payload = {
                "items": build_fisher_rows(profile, names),
                "assignment_metric": "top1_fisher_z",
                "q_threshold": None,
                "top_n_cap": None,
            }
            save_json(combo_dir / "cluster_assignment_status.json", status_payload)
            manifest = load_json(combo_dir / "group_atlas_manifest.json")
            manifest["cluster_assignment_metric"] = "top1_fisher_z"
            save_json(combo_dir / "group_atlas_manifest.json", manifest)

    subprocess.run([sys.executable, str(MATERIALIZE_LABELS_SCRIPT), "--out-root", str(root)], check=True, cwd=str(REPO_ROOT))


def render_root(root: Path, render_mode_tag: str, out_name: str, *, allow_existing_output: bool) -> None:
    PRESENT_FIXEDK.mkdir(parents=True, exist_ok=True)
    out_path = PRESENT_FIXEDK / out_name
    with tempfile.TemporaryDirectory(prefix=f"present_fixedk_{root.name}_") as tmp:
        tmp_dir = Path(tmp)
        cmd = [
            sys.executable,
            str(BUILD_PRESENT_SCRIPT),
            "--group-root",
            str(root),
            "--present-dir",
            str(tmp_dir),
            "--methods",
            "network-spectral",
            "--atlases",
            "kong2022",
            "--route-tag",
            "routea",
            "--mode-tag",
            render_mode_tag,
            "--skip-readme",
            "--skip-manifest",
        ]
        try:
            subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        except subprocess.CalledProcessError:
            if allow_existing_output and out_path.exists():
                print(
                    f"[warn] render failed for root={root.name} mode={render_mode_tag}; keeping existing {out_path}",
                    flush=True,
                )
                return
            raise
        src = tmp_dir / f"kong2022_network-spectral_group_overview_routea_{render_mode_tag}.png"
        import shutil

        shutil.copy2(src, out_path)


def validate_root(
    root: Path,
    expected_k: int,
    spec: ModeSpec,
    *,
    allow_nonfixed_combos: set[tuple[str, str]] | None = None,
) -> None:
    allow_nonfixed_combos = allow_nonfixed_combos or set()
    for smoothing in SMOOTHINGS:
        for hemi in HEMIS:
            manifest = load_json(run_dir(root, smoothing, hemi) / "group_atlas_manifest.json")
            chosen_k = get_chosen_k(root, smoothing, hemi)
            if chosen_k != int(expected_k) and (smoothing, hemi) not in allow_nonfixed_combos:
                raise RuntimeError(f"Unexpected chosen_k_final={chosen_k} at {root} {smoothing} {hemi}")
            sfc = manifest.get("subject_fc_preprocessing", {})
            mode = str(sfc.get("mode", ""))
            mode_tag = str(sfc.get("mode_tag", ""))
            top_percent = sfc.get("top_percent", None)
            if mode == "" or mode_tag == "":
                top_fraction = float(sfc.get("top_fraction", -1.0))
                binarize = bool(sfc.get("binarize", False))
                if abs(top_fraction - 1.0) <= 1e-6 and not binarize:
                    mode = "full"
                    mode_tag = "full"
                    top_percent = None
                elif top_fraction > 0.0 and top_fraction <= 1.0 and binarize:
                    pct = int(round(top_fraction * 100.0))
                    mode = "top-percent"
                    mode_tag = f"{pct}perc"
                    top_percent = pct
            if mode != str(spec.subject_fc_mode):
                raise RuntimeError(f"Unexpected subject_fc mode={mode} at {root} {smoothing} {hemi}")
            if mode_tag != str(spec.mode):
                raise RuntimeError(f"Unexpected subject_fc mode_tag={mode_tag} at {root} {smoothing} {hemi}")
            if spec.subject_fc_top_percent is None:
                if top_percent is not None:
                    raise RuntimeError(f"Expected top_percent=None, got {top_percent} at {root} {smoothing} {hemi}")
            elif int(top_percent) != int(spec.subject_fc_top_percent):
                raise RuntimeError(
                    f"Unexpected top_percent={top_percent} at {root} {smoothing} {hemi}; "
                    f"expected {spec.subject_fc_top_percent}"
                )


def main() -> int:
    summary: dict[str, object] = {"roots": {}, "present_fixedk": str(PRESENT_FIXEDK)}
    for spec in MODE_SPECS:
        for k_target in (2, 3):
            tag = f"{spec.mode}_k{k_target}"
            count_root = FIXEDK_WORK / f"ga_routea_{spec.mode}_k{k_target}_count"
            fisher_root = FIXEDK_WORK / f"ga_routea_{spec.mode}_k{k_target}_fisherz"

            ensure_copytree(spec.canonical_root, count_root)
            unsatisfied = enforce_fixed_k(count_root, spec, k_target)
            allow_nonfixed = {(sm, hemi) for sm, hemi, _ in unsatisfied}
            if unsatisfied:
                print(
                    f"[warn] non-fixed combos remain for mode={spec.mode} k={k_target}: "
                    + ", ".join(f"{sm}/{hemi}->k{after}" for sm, hemi, after in unsatisfied),
                    flush=True,
                )
            validate_root(count_root, k_target, spec, allow_nonfixed_combos=allow_nonfixed)

            sync_count_to_fisher_tree(count_root, fisher_root)
            convert_to_fisher_assignment(fisher_root)
            validate_root(fisher_root, k_target, spec, allow_nonfixed_combos=allow_nonfixed)

            count_png = f"kong2022_network-spectral_group_overview_routea_{spec.mode}_k{k_target}_count.png"
            fisher_png = f"kong2022_network-spectral_group_overview_routea_{spec.mode}_k{k_target}_fisherz.png"
            allow_existing = bool(allow_nonfixed)
            render_root(count_root, spec.render_mode_tag, count_png, allow_existing_output=allow_existing)
            render_root(fisher_root, spec.render_mode_tag, fisher_png, allow_existing_output=allow_existing)
            summary["roots"][tag] = {
                "count_root": str(count_root),
                "fisher_root": str(fisher_root),
                "count_png": str(PRESENT_FIXEDK / count_png),
                "fisher_png": str(PRESENT_FIXEDK / fisher_png),
                "nonfixed_combos": [f"{sm}_{hemi}_k{after}" for sm, hemi, after in unsatisfied],
            }

    pngs = sorted(p.name for p in PRESENT_FIXEDK.glob("kong2022_network-spectral_group_overview_routea_*_k*_*.png"))
    summary["present_png_count"] = len(pngs)
    summary["present_pngs"] = pngs
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
