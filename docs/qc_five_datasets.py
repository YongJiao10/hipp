"""Compare published run-motion screens on the five requested datasets.

Reuse the completed signal measurements; read the original, pre-censoring
motion traces. Production time series, masks, rosters and results are read-only.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/core_workflow"))
from lib.io_utils import load_subject_ids

J = Path("/home/yojiao/LSS/jianglab/3_Data_Working")
HCP = ROOT / "outputs_hcp/inputs/hcp/kong2022"
STANFORD = ROOT / "outputs_stanford/inputs/stanford"
SCHEMES = ("V", "Motion_M05", "Motion_R20", "Motion_F25", "V_M05", "V_R20", "T25")


def read_tsv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path, rows, first):
    columns = first + sorted(set().union(*(r.keys() for r in rows)) - set(first))
    with Path(path).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def state(value):
    return "unassessable" if value is None else "pass" if value else "fail"


def conjunction(values):
    if "fail" in values:
        return "fail"
    return "unassessable" if "unassessable" in values else "pass"


def checked(path):
    path = Path(path)
    if any(p.lower() in {"archive", "_archive", "arch"} for p in path.resolve().parts):
        raise ValueError(f"Archive access not authorized: {path}")
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def run_metrics(dataset, scan, run, path, frames=None, tr=None, *, fmriprep=False):
    row = dict(dataset=dataset, scan=scan, run=run, motion_path=str(path))
    try:
        if fmriprep:
            records = read_tsv(checked(path))
            arrays = {}
            for name in ("rmsd", "framewise_displacement"):
                x = np.array([np.nan if r[name] == "n/a" else float(r[name]) for r in records])
                if len(x) < 2 or not np.isnan(x[0]) or not np.isfinite(x[1:]).all():
                    raise ValueError(f"Expected first-frame n/a and finite subsequent {name}: {path}")
                x[0] = 0.0
                arrays[name] = x
            rms, fd = arrays["rmsd"], arrays["framewise_displacement"]
        else:
            rms = np.loadtxt(checked(path))
            parameters = np.loadtxt(checked(path.parent / "Movement_Regressors.txt"))
            if rms.ndim != 1 or parameters.shape != (len(rms), 12) or not np.isfinite(parameters).all():
                raise ValueError(f"RMS/realignment shape/value mismatch: {path}")
            # HCP translations are mm and rotations are DEGREES. Differentiate
            # the first six position columns, not the derivative columns.
            delta = np.abs(np.diff(parameters[:, :6], axis=0))
            fd = np.r_[0.0, delta[:, :3].sum(1) + 50 * np.pi / 180 * delta[:, 3:].sum(1)]
        if not np.isfinite(rms).all() or (rms < 0).any() or not np.isfinite(fd).all() or (fd < 0).any():
            raise ValueError(f"Nonfinite/negative frame displacement: {path}")
        if frames is not None and len(rms) != frames:
            raise ValueError(f"Declared frame count {frames} != motion length {len(rms)}: {path}")
        n = len(rms)
        bad_rms, bad_fd = int((rms > .25).sum()), int((fd > .5).sum())
        row.update(frames=n, tr_seconds=tr, mean_relative_rms=float(rms.mean()),
                   max_relative_rms=float(rms.max()), mean_fd=float(fd.mean()), max_fd=float(fd.max()),
                   n_rms_gt_0_25=bad_rms, fraction_rms_gt_0_25=bad_rms / n,
                   n_fd_gt_0_5=bad_fd, fraction_fd_gt_0_5=bad_fd / n,
                   Motion_R20=state(bad_rms * 5 <= n), Motion_F25=state(bad_fd * 4 <= n),
                   fd_definition="unfiltered_Power_50mm_lag1_first0")
    except (OSError, ValueError, KeyError) as exc:
        row.update(Motion_R20="unassessable", Motion_F25="unassessable", error=str(exc))
    return row


def new_anatomy(dataset, sid):
    values = []
    scores = {}
    for h in ("L", "R"):
        p = ROOT / f"outputs_{dataset}/inputs/{dataset}/hippunfold/dense_corobl_batch/sub-{sid}/hippunfold/sub-{sid}/qc/sub-{sid}_hemi-{h}_desc-unetf3d_dice.tsv"
        if not p.is_file():
            values.append("unassessable")
            continue
        x = np.loadtxt(checked(p))
        if x.size != 1 or not np.isfinite(x).all():
            raise ValueError(f"Expected one finite Dice score: {p}")
        scores[f"dice_{h}"] = float(x.item())
        values.append(state(x.item() >= .7))
    return conjunction(values), scores


def hcp7t_motion_runs():
    """Read the four exact unpacked runs per member of the 166-person roster."""
    roster = load_subject_ids(ROOT / "manifests/hcp_7t_hippocampus_struct_complete_166.txt")
    declarations = read_tsv(checked(ROOT / "manifests/hcp7t_motion_transfer_20261019.tsv"))
    labels = ("rfMRI_REST1_7T_PA", "rfMRI_REST2_7T_AP",
              "rfMRI_REST3_7T_PA", "rfMRI_REST4_7T_AP")
    files = {}
    for r in declarations:
        key = (r["subject"], r["run_label"], r["filename"])
        if key in files:
            raise ValueError(f"Duplicate HCP7T motion declaration: {key}")
        files[key] = Path(r["lss_path"])
    if {r["subject"] for r in declarations} != set(roster):
        raise ValueError("HCP7T motion declarations do not match the 166-person roster")
    if {r["run_label"] for r in declarations} != set(labels):
        raise ValueError("HCP7T motion declarations have unexpected physical run labels")
    rows = []
    for sid in roster:
        for label in labels:
            rms = files[(sid, label, "Movement_RelativeRMS.txt")]
            parameters = files[(sid, label, "Movement_Regressors.txt")]
            if parameters != rms.parent / "Movement_Regressors.txt":
                raise ValueError(f"HCP7T realignment/RMS directory mismatch: {sid}/{label}")
            # Four 900-frame, 1-second-TR runs match the consumed 3600-frame volume.
            row = run_metrics("HCP7T rest", sid, label, rms, frames=900, tr=1.0)
            if "error" in row:
                raise ValueError(row["error"])
            rows.append(row)
    return rows


def annotate_motion(row, runs):
    row["n_runs"] = len(runs)
    for scheme in ("Motion_R20", "Motion_F25"):
        row[scheme] = conjunction([r[scheme] for r in runs]) if runs else "unassessable"
    if runs and all("mean_relative_rms" in r for r in runs):
        row["mean_relative_rms"] = float(np.mean([float(r["mean_relative_rms"]) for r in runs]))
        row["max_relative_rms"] = max(float(r["max_relative_rms"]) for r in runs)
        row["max_run_fraction_rms_gt_0_25"] = max(float(r["fraction_rms_gt_0_25"]) for r in runs)
        row["Motion_M05"] = state(row["mean_relative_rms"] <= .5)
    else:
        row["Motion_M05"] = "unassessable"
    row["V_M05"] = conjunction([row["V"], row["Motion_M05"]])
    row["V_R20"] = conjunction([row["V"], row["Motion_R20"]])


def update_hcp7t_motion():
    """Refresh HCP7T motion rows while retaining every other cohort's measurements."""
    rows = hcp7t_motion_runs()
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["scan"]].append(r)
    scans = read_tsv(ROOT / "docs/qc_five_datasets_subjects.tsv")
    selected = [r for r in scans if r["dataset"] == "HCP7T rest"]
    if len(selected) != len(grouped) or {r["scan"] for r in selected} != set(grouped):
        raise ValueError("Existing HCP7T comparison rows do not match the motion roster")
    for r in selected:
        annotate_motion(r, grouped[r["scan"]])
    summary = read_tsv(ROOT / "docs/qc_five_datasets_summary.tsv")
    selected_summary = [r for r in summary if r["dataset"] == "HCP7T rest"]
    if len(selected_summary) != len(SCHEMES) or {r["scheme"] for r in selected_summary} != set(SCHEMES):
        raise ValueError("Existing HCP7T summary must declare each comparison scheme once")
    for r in selected_summary:
        c = Counter(s[r["scheme"]] for s in selected)
        r.update(n_total=len(selected), n_pass=c["pass"], n_fail=c["fail"],
                 n_unassessable=c["unassessable"],
                 confirmed_pass_percent=round(100 * c["pass"] / len(selected), 3))
    existing_runs = read_tsv(ROOT / "docs/qc_five_datasets_motion_runs.tsv")
    all_runs = rows + [r for r in existing_runs if r["dataset"] != "HCP7T rest"]
    write_tsv(ROOT / "docs/qc_five_datasets_motion_runs.tsv", all_runs, ["dataset", "scan", "run"])
    write_tsv(ROOT / "docs/qc_five_datasets_subjects.tsv", scans, ["dataset", "scan", "participant"])
    write_tsv(ROOT / "docs/qc_five_datasets_summary.tsv", summary,
              ["dataset", "scheme", "n_total", "n_pass", "n_fail", "n_unassessable"])
    print(f"Updated HCP7T: {len(grouped)} participants / {len(rows)} physical runs")
    for r in selected_summary:
        print(r)
    return selected


def main():
    previous = read_tsv(ROOT / "docs/tsnr_qc_comparison_subjects.tsv")
    native = read_tsv(ROOT / "docs/banda_pdc_qc_subjects.tsv")
    declared_native_runs = read_tsv(ROOT / "docs/banda_pdc_qc_runs.tsv")
    runs, scans = [], []
    by_scan = defaultdict(list)

    for r in hcp7t_motion_runs():
        runs.append(r)
        by_scan[(r["dataset"], r["scan"])].append(r)
    print("Read HCP7T rest: 166 participants / 664 runs", flush=True)

    for task in ("emotion", "wm"):
        dataset = f"HCP3T {task.upper()}"
        ids = load_subject_ids(HCP / f"shared_inputs/task-{task}/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_144.txt")
        for sid in ids:
            declaration = json.loads(checked(HCP / f"manifests/task-{task}/sub-{sid}.json").read_text())
            for r in declaration["runs"]:
                row = run_metrics(dataset, sid, r["run_label"], Path(r["bold"]["path"]).parent / "Movement_RelativeRMS.txt",
                                  declaration["expected_frames_per_run"], .72)
                runs.append(row)
                by_scan[(dataset, sid)].append(row)
        print(f"Read {dataset}: {len(ids)} participants", flush=True)

    for sid in load_subject_ids(STANFORD / "kong2022/manifests/stanford_all_subjects.txt"):
        path = J / f"fmriprep_processed_CausalConnectome/rest_fmriprep/fmriprep/sub-{sid}/func/sub-{sid}_task-rest_desc-confounds_timeseries.tsv"
        row = run_metrics("Stanford", sid, "task-rest", path, fmriprep=True)
        runs.append(row)
        for branch in ("Stanford 22P", "Stanford 36P"):
            by_scan[(branch, sid)].append(row)
    print("Read Stanford: 199 participants, one pre-censoring source run each", flush=True)

    for i, r in enumerate(declared_native_runs, 1):
        row = run_metrics(r["dataset"], r["scan"], r["run"], Path(r["dtseries"]).parent / "Movement_RelativeRMS.txt",
                          int(r["frames"]), float(r["tr_seconds"]))
        runs.append(row)
        by_scan[(r["dataset"], r["scan"])].append(row)
        if i % 400 == 0:
            print(f"Read BANDA/PDC runs: {i}/{len(declared_native_runs)}", flush=True)

    selected = {"HCP7T rest", "HCP3T EMOTION", "HCP3T WM", "Stanford 22P", "Stanford 36P"}
    for old in previous:
        if old["dataset"] not in selected:
            continue
        scans.append(dict(dataset=old["dataset"], scan=old["subject"], participant=old["subject"],
                          representation="den-512_surface", V=old["V"], T25=old["T25"],
                          dice_L=old["dice_L"], dice_R=old["dice_R"], tsnr_definition=old["T25_source"]))
    for old in native:
        ds, sid = old["dataset"], old["scan"]
        anatomy, scores = new_anatomy(ds.lower(), sid)
        scans.append(dict(dataset=ds, scan=sid, participant=old["participant"],
                          representation="native_CIFTI_hippocampal_volume_ROI", Dice=anatomy, **scores,
                          V=conjunction([anatomy, old["Signal_V"]]),
                          T25=conjunction([anatomy, old["Signal_T25"]]),
                          Signal_V=old["Signal_V"], Signal_T25=old["Signal_T25"],
                          tsnr_definition="mean_cleaned_run_over_sampleSD_runwise_intersection"))

    for row in scans:
        rr = by_scan[(row["dataset"], row["scan"])]
        annotate_motion(row, rr)

    task_rows = defaultdict(list)
    pdc_rows = defaultdict(list)
    for r in scans:
        if r["dataset"].startswith("HCP3T "):
            task_rows[r["participant"]].append(r)
        if r["dataset"] == "PDC":
            pdc_rows[r["participant"]].append(r)
    for sid, rr in task_rows.items():
        if {r["dataset"] for r in rr} != {"HCP3T EMOTION", "HCP3T WM"}:
            raise ValueError(f"Both task branches required: {sid}")
        scans.append(dict(dataset="HCP3T both tasks", scan=sid, participant=sid,
                          **{s: conjunction([r[s] for r in rr]) for s in SCHEMES}))
    for sid, rr in pdc_rows.items():
        states = {}
        for scheme in SCHEMES:
            values = [r[scheme] for r in rr]
            states[scheme] = "pass" if "pass" in values else "unassessable" if "unassessable" in values else "fail"
        scans.append(dict(dataset="PDC people, any passing session", scan=sid, participant=sid, **states))

    summary = []
    order = ["HCP7T rest", "HCP3T EMOTION", "HCP3T WM", "HCP3T both tasks", "Stanford 22P", "Stanford 36P", "BANDA", "PDC", "PDC people, any passing session"]
    for dataset in order:
        subset = [r for r in scans if r["dataset"] == dataset]
        for scheme in SCHEMES:
            c = Counter(r[scheme] for r in subset)
            summary.append(dict(dataset=dataset, scheme=scheme, n_total=len(subset),
                                n_pass=c["pass"], n_fail=c["fail"], n_unassessable=c["unassessable"],
                                confirmed_pass_percent=round(100 * c["pass"] / len(subset), 3)))
    write_tsv(ROOT / "docs/qc_five_datasets_motion_runs.tsv", runs, ["dataset", "scan", "run"])
    write_tsv(ROOT / "docs/qc_five_datasets_subjects.tsv", scans, ["dataset", "scan", "participant"])
    write_tsv(ROOT / "docs/qc_five_datasets_summary.tsv", summary, ["dataset", "scheme", "n_total", "n_pass", "n_fail", "n_unassessable"])
    print(f"Wrote {len(runs)} physical runs, {len(scans)} comparison entries, {len(summary)} summary rows", flush=True)
    for r in summary:
        if r["scheme"] in {"Motion_R20", "V_R20", "T25"}:
            print(r, flush=True)


if __name__ == "__main__":
    main()
