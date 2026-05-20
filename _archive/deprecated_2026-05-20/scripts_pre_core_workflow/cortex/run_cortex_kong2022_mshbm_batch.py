#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = sys.executable or "python"
RUN_SUBJECT = REPO_ROOT / "scripts" / "cortex" / "run_cortex_mshbm_subject.py"
PLOT_SUBJECT = REPO_ROOT / "scripts" / "cortex" / "plot_cortex_kong2022_mshbm.py"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "hippunfold_input"
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs_experiments" / "cortex_kong2022_areal_mshbm"
DEFAULT_CBIG_CODE_DIR = REPO_ROOT / "external" / "CBIG"
DEFAULT_SCENE = REPO_ROOT / "config" / "manual_wb_scenes" / "cortex_manual.scene"
DEFAULT_SUBJECTS = ["100610", "102311"]
ATLAS_SLUG = "kong2022_areal_mshbm"
MAX_CONCURRENCY = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch run Kong2022_ArealMSHBM subject parcellation, then render one overview PNG per subject."
    )
    p.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    p.add_argument("--jobs", type=int, default=2, help="Worker count, hard-capped at 10.")
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--cbig-code-dir", default=str(DEFAULT_CBIG_CODE_DIR))
    p.add_argument("--group-prior", default=None)
    p.add_argument("--source-scene", default=str(DEFAULT_SCENE))
    p.add_argument("--model", default="gMSHBM", choices=["gMSHBM", "cMSHBM", "dMSHBM"])
    p.add_argument("--num-rois", type=int, default=400)
    p.add_argument("--w", type=int, default=30)
    p.add_argument("--c", type=int, default=30)
    p.add_argument("--beta", type=int, default=50)
    p.add_argument("--run-lengths", nargs="+", type=int, default=[900, 900, 900, 900])
    p.add_argument("--skip-plot", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True)


def subject_paths(data_root: Path, subject: str) -> dict[str, Path]:
    anat = data_root / f"sub-{subject}" / "anat"
    func = data_root / f"sub-{subject}" / "func"
    return {
        "dtseries": func / f"sub-{subject}_task-rest_run-concat.dtseries.nii",
        "left_inflated": anat / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
        "right_inflated": anat / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
        "left_midthickness": anat / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii",
        "right_midthickness": anat / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii",
        "sulc_dscalar": anat / f"sub-{subject}_space-fsLR_den-32k_desc-MSMAll_sulc.dscalar.nii",
    }


def run_subject(
    *,
    subject: str,
    data_root: Path,
    out_root: Path,
    cbig_code_dir: Path,
    group_prior: Path | None,
    source_scene: Path,
    model: str,
    num_rois: int,
    run_lengths: list[int],
    w: int,
    c: int,
    beta: int,
    skip_plot: bool,
    dry_run: bool,
) -> tuple[str, bool, str]:
    paths = subject_paths(data_root, subject)
    log_dir = out_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"sub-{subject}_{ATLAS_SLUG}.log"

    subject_cmd = [
        PYTHON_EXE,
        str(RUN_SUBJECT),
        "--subject",
        subject,
        "--dtseries",
        str(paths["dtseries"]),
        "--left-inflated",
        str(paths["left_inflated"]),
        "--right-inflated",
        str(paths["right_inflated"]),
        "--left-midthickness",
        str(paths["left_midthickness"]),
        "--right-midthickness",
        str(paths["right_midthickness"]),
        "--sulc-dscalar",
        str(paths["sulc_dscalar"]),
        "--cbig-code-dir",
        str(cbig_code_dir),
        "--out-root",
        str(out_root),
        "--model",
        str(model),
        "--num-rois",
        str(int(num_rois)),
        "--w",
        str(int(w)),
        "--c",
        str(int(c)),
        "--beta",
        str(int(beta)),
        "--run-lengths",
        *[str(int(v)) for v in run_lengths],
    ]
    if group_prior is not None:
        subject_cmd.extend(["--group-prior", str(group_prior)])
    if dry_run:
        subject_cmd.append("--dry-run")

    proc = run_cmd(subject_cmd)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(subject_cmd) + "\n")
        f.write((proc.stdout or "") + "\n")
        f.write((proc.stderr or "") + "\n")

    if proc.returncode != 0:
        tail = ((proc.stderr or "") + "\n" + (proc.stdout or ""))[-5000:]
        return subject, False, f"subject run failed\n{tail}"

    if skip_plot or dry_run:
        return subject, True, "ok"

    atlas_dir = out_root / f"sub-{subject}" / ATLAS_SLUG
    dlabel = atlas_dir / f"sub-{subject}_atlas-{ATLAS_SLUG}.dlabel.nii"
    plot_cmd = [
        PYTHON_EXE,
        str(PLOT_SUBJECT),
        "--subject",
        subject,
        "--dlabel",
        str(dlabel),
        "--left-inflated",
        str(paths["left_inflated"]),
        "--right-inflated",
        str(paths["right_inflated"]),
        "--sulc-dscalar",
        str(paths["sulc_dscalar"]),
        "--source-scene",
        str(source_scene),
        "--atlas-dir",
        str(atlas_dir),
    ]
    proc_plot = run_cmd(plot_cmd)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(plot_cmd) + "\n")
        f.write((proc_plot.stdout or "") + "\n")
        f.write((proc_plot.stderr or "") + "\n")

    if proc_plot.returncode != 0:
        tail = ((proc_plot.stderr or "") + "\n" + (proc_plot.stdout or ""))[-5000:]
        return subject, False, f"plot run failed\n{tail}"

    return subject, True, "ok"


def main() -> int:
    args = parse_args()
    if int(args.jobs) > MAX_CONCURRENCY:
        raise ValueError(f"--jobs must be <= {MAX_CONCURRENCY}, got {args.jobs}")

    subjects = [str(s).replace("sub-", "") for s in args.subjects if str(s).strip()]
    if not subjects:
        raise ValueError("No subjects provided")

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()
    cbig_code_dir = Path(args.cbig_code_dir).resolve()
    source_scene = Path(args.source_scene).resolve()
    group_prior = Path(args.group_prior).resolve() if args.group_prior else None

    jobs = max(1, min(int(args.jobs), MAX_CONCURRENCY, len(subjects)))
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {
            pool.submit(
                run_subject,
                subject=s,
                data_root=data_root,
                out_root=out_root,
                cbig_code_dir=cbig_code_dir,
                group_prior=group_prior,
                source_scene=source_scene,
                model=str(args.model),
                num_rois=int(args.num_rois),
                run_lengths=[int(v) for v in args.run_lengths],
                w=int(args.w),
                c=int(args.c),
                beta=int(args.beta),
                skip_plot=bool(args.skip_plot),
                dry_run=bool(args.dry_run),
            ): s
            for s in subjects
        }

        done = 0
        for fut in as_completed(futs):
            done += 1
            subject, ok, msg = fut.result()
            if ok:
                print(f"[done {done}/{len(subjects)}] sub-{subject}", flush=True)
            else:
                failed.append(subject)
                print(f"[FAIL {done}/{len(subjects)}] sub-{subject}\n{msg}", flush=True)

    if failed:
        print(f"[error] failed_subjects={len(failed)} {failed}", flush=True)
        return 1

    print(
        f"[ok] completed={len(subjects)} jobs={jobs} atlas={ATLAS_SLUG} "
        f"model={args.model} num_rois={args.num_rois} "
        f"plot={'no' if args.skip_plot else 'yes'} dry_run={bool(args.dry_run)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
