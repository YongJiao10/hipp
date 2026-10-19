"""Read-only BANDA/PDC anatomy, motion, and native-CIFTI hippocampal QC.

The ROI comparison is explicitly distinct from den-512 surface QC. The
cleaned CIFTI run itself supplies both temporal mean and sample SD. This is
not the earlier pre-denoising-baseline/detrended rest tSNR definition.
"""
from __future__ import annotations

import csv
import json
import multiprocessing
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/core_workflow"))
from lib.io_utils import load_subject_ids

HEMIS = {"L": "CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT",
         "R": "CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT"}
SCHEMES = ("Dice", "RMS", "Dice_RMS", "Signal_V", "Signal_T25", "ROI_H", "ROI_V", "ROI_M", "ROI_T25")
WORKER_SPECS = {}


def checked(path):
    path = Path(path)
    if any(p.lower() in {"archive", "_archive", "arch"} for p in path.resolve().parts):
        raise ValueError(f"Archive access is not authorized: {path}")
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def conjunction(values):
    if any(v is False for v in values):
        return False
    return None if any(v is None for v in values) else True


def status(value):
    return "unassessable" if value is None else "pass" if value else "fail"


def local(path):
    prefix = "/Shared/jianglab/"
    if not path.startswith(prefix):
        raise ValueError(f"Expected declared Argon Jianglab prefix: {path}")
    return Path("/home/yojiao/LSS/jianglab") / path[len(prefix):]


def specifications():
    specs = []
    for ds in ("banda", "pdc"):
        base = ROOT / f"outputs_{ds}/inputs/{ds}"
        d = json.loads(checked(base / f"kong2022/manifests/{ds}_cortex_inputs.json").read_text())
        roster = base / ("manifests/banda_all_subjects.txt" if ds == "banda" else "manifests/pdc_all_sessions.txt")
        ids = load_subject_ids(roster)
        entries = {}
        for entry in d["subjects"]:
            sid = entry["subject"].removeprefix("sub-")
            if sid in entries:
                raise ValueError(f"Duplicate {ds} scan: {sid}")
            entries[sid] = entry
        if not set(entries) <= set(ids):
            raise ValueError(f"Declared scans outside authoritative {ds} roster")
        people = {sid: sid for sid in ids}
        if ds == "pdc":
            with checked(base / "manifests/session_mapping.tsv").open() as f:
                mappings = list(csv.DictReader(f, delimiter="\t"))
            people = {r["bids_subject_id"]: r["base_subject_id"] for r in mappings}
            if len(people) != len(mappings) or set(people) != set(ids):
                raise ValueError("PDC session mapping must exactly match the authoritative roster")
        specs.append(dict(dataset=ds.upper(), base=base, ids=ids, entries=entries, people=people))
    return specs


def evaluate(spec, sid):
    record = dict(dataset=spec["dataset"], scan=sid, participant=spec["people"][sid])
    errors, dice_states = [], []
    for h in HEMIS:
        p = spec["base"] / f"hippunfold/dense_corobl_batch/sub-{sid}/hippunfold/sub-{sid}/qc/sub-{sid}_hemi-{h}_desc-unetf3d_dice.tsv"
        try:
            x = np.loadtxt(checked(p))
            if x.size != 1 or not np.isfinite(x).all():
                raise ValueError(f"Expected one finite Dice score: {p}")
            record[f"dice_{h}"] = float(x.item())
            dice_states.append(bool(x.item() >= .70))
        except (OSError, ValueError) as exc:
            dice_states.append(None)
            errors.append(f"Dice_{h}:{exc}")
    anatomy = conjunction(dice_states)
    entry = spec["entries"].get(sid)
    if entry is None:
        record.update(Dice=status(anatomy), **{s: "unassessable" for s in SCHEMES if s != "Dice"})
        if anatomy is False:
            record.update({s: "fail" for s in ("Dice", "Dice_RMS", "ROI_H", "ROI_V", "ROI_M", "ROI_T25")})
        record["n_declared_runs"] = 0
        record["errors"] = " | ".join(errors + ["No independent rest runs in the source declaration"])
        return record, []
    runs = entry["runs"]
    if not runs or len({r["run_id"] for r in runs}) != len(runs):
        raise ValueError(f"Missing/duplicate declared run IDs: {sid}")
    record["n_declared_runs"] = len(runs)
    signals = {h: [] for h in HEMIS}
    voxel_orders = {}
    rms_values, run_rows, durations = [], [], []
    for declaration in runs:
        rid = declaration["run_id"]
        path = local(declaration["dtseries"])
        row = dict(dataset=spec["dataset"], scan=sid, participant=record["participant"], run=rid, dtseries=str(path))
        rms, length = None, None
        try:
            a = np.loadtxt(checked(path.parent / "Movement_RelativeRMS.txt"))
            if a.ndim != 1 or a.size < 2 or not np.isfinite(a).all() or (a < 0).any():
                raise ValueError("RMS must be a finite nonnegative vector")
            rms, length = float(a.mean()), len(a)
            row["mean_relative_rms"] = rms
            # Descriptive peak burden only, not a new censoring/gating rule.
            row["max_relative_rms"] = float(a.max())
            row["n_relative_rms_above_0_5"] = int((a > .5).sum())
        except (OSError, ValueError) as exc:
            row["motion_error"] = str(exc)
            errors.append(f"RMS_{rid}:{exc}")
        rms_values.append(rms)
        try:
            im = nib.load(checked(path))
            if len(im.shape) != 2:
                raise ValueError(f"Expected time-by-grayordinate CIFTI: {path}")
            n = im.shape[0]
            if length is not None and length != n:
                raise ValueError(f"CIFTI/RMS frame mismatch {n}/{length}: {path}")
            # Use the declared hippocampal BrainModels directly. Rebuilding
            # all cortical axes for every run is unnecessary for this ROI.
            mapping = im.header.matrix.get_index_map(1)
            time_mapping = im.header.matrix.get_index_map(0)
            tr = float(time_mapping.series_step)
            if time_mapping.series_unit != "SECOND" or tr <= 0:
                raise ValueError(f"Expected a positive TR in seconds: {path}")
            row.update(frames=n, tr_seconds=tr)
            blocks = {}
            for h, name in HEMIS.items():
                matches = [m for m in mapping.brain_models if m.brain_structure == name]
                if len(matches) != 1:
                    raise ValueError(f"Missing/duplicate hippocampal brain model: {path}/{name}")
                model = matches[0]
                if model.model_type != "CIFTI_MODEL_TYPE_VOXELS" or model.index_count < 2:
                    raise ValueError(f"Expected explicit hippocampal volume brain model: {name}")
                sl = slice(model.index_offset, model.index_offset + model.index_count)
                geometry = (np.asarray(model.voxel_indices_ijk),
                            mapping.volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix,
                            mapping.volume.volume_dimensions)
                if h in voxel_orders and not all(np.array_equal(x, y) for x, y in zip(voxel_orders[h], geometry)):
                    raise ValueError(f"Runwise hippocampal voxel/affine mismatch: {sid}/{h}")
                voxel_orders[h] = geometry
                y = np.asarray(im.dataobj[:, sl], dtype=np.float64).T
                blocks[h] = y
                finite = np.isfinite(y).all(1)
                sd = y.std(1, ddof=1)
                good = finite & (sd > 0)
                score = np.full(y.shape[0], np.nan)
                score[good] = y[good].mean(1) / sd[good]
                row[f"n_voxels_{h}"] = y.shape[0]
                row[f"median_mean_{h}"] = float(np.median(y[finite].mean(1))) if finite.any() else ""
                row[f"median_tsnr_{h}"] = float(np.median(score[good])) if good.any() else ""
            for h, y in blocks.items():
                signals[h].append(y)
            durations.append(n * tr)
        except (OSError, ValueError, KeyError, nib.filebasedimages.ImageFileError) as exc:
            row["signal_error"] = str(exc)
            errors.append(f"signal_{rid}:{exc}")
        run_rows.append(row)
    motion = None
    if all(r is not None for r in rms_values):
        record["mean_relative_rms"] = float(np.mean(rms_values))
        motion = bool(record["mean_relative_rms"] <= .50)
    states = {s: [] for s in ("ROI_H", "ROI_V", "ROI_T25")}
    for h in HEMIS:
        if len(signals[h]) != len(runs):
            for s in states:
                states[s].append(None)
            continue
        arrays = signals[h]
        count = arrays[0].shape[0]
        v = np.ones(count, dtype=bool)
        t25 = np.ones(count, dtype=bool)
        for y in arrays:
            sd = y.std(1, ddof=1)
            good = np.isfinite(y).all(1) & (sd > 0)
            score = np.full(count, np.nan)
            score[good] = y[good].mean(1) / sd[good]
            v &= good
            t25 &= good & np.isfinite(score) & (score >= 25)
        full = np.concatenate(arrays, axis=1)
        sd = full.std(1, ddof=1)
        old = np.isfinite(full).all(1) & (sd > 0) & (sd <= 400)
        record[f"n_voxels_{h}"] = count
        for name, mask in (("ROI_H", old), ("ROI_V", v), ("ROI_T25", t25)):
            fraction = float((~mask).mean())
            record[f"{name}_invalid_fraction_{h}"] = fraction
            states[name].append(bool(fraction < .20))
    record["Dice"] = status(anatomy)
    record["RMS"] = status(motion)
    record["Dice_RMS"] = status(conjunction([anatomy, motion]))
    record["Signal_V"] = status(conjunction(states["ROI_V"]))
    record["Signal_T25"] = status(conjunction(states["ROI_T25"]))
    for name, values in states.items():
        record[name] = status(conjunction([anatomy, conjunction(values)]))
    record["ROI_M"] = status(conjunction([anatomy, motion, conjunction(states["ROI_V"])]))
    record["total_run_minutes"] = sum(durations) / 60 if len(durations) == len(runs) else ""
    record["errors"] = " | ".join(errors)
    return record, run_rows


def write(path, rows, first):
    columns = first + sorted(set().union(*(r.keys() for r in rows)) - set(first))
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def evaluate_job(job):
    dataset, sid = job
    return evaluate(WORKER_SPECS[dataset], sid)


def main():
    global WORKER_SPECS
    specs = specifications()
    WORKER_SPECS = {s["dataset"]: s for s in specs}
    jobs = [(s["dataset"], sid) for s in specs for sid in s["ids"]]
    records, runs = [], []
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("fork")) as pool:
        futures = [pool.submit(evaluate_job, job) for job in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            record, rr = f.result()
            records.append(record)
            runs.extend(rr)
            if i % 25 == 0:
                print(f"SCANS {i}/{len(jobs)} elapsed={time.monotonic()-start:.1f}s", flush=True)
    records.sort(key=lambda r: (r["dataset"], r["scan"]))
    runs.sort(key=lambda r: (r["dataset"], r["scan"], r["run"]))
    write(ROOT / "docs/banda_pdc_qc_subjects.tsv", records, ["dataset", "scan", "participant", *SCHEMES])
    write(ROOT / "docs/banda_pdc_qc_runs.tsv", runs, ["dataset", "scan", "participant", "run"])
    summary = []
    for spec in specs:
        rows = [r for r in records if r["dataset"] == spec["dataset"]]
        for scheme in SCHEMES:
            c = Counter(r[scheme] for r in rows)
            assert sum(c.values()) == len(rows)
            passed = c["pass"]
            summary.append(dict(dataset=spec["dataset"], unit="participant" if spec["dataset"] == "BANDA" else "session", scheme=scheme,
                                total=len(rows), passed=passed, failed=c["fail"], unassessable=c["unassessable"],
                                confirmed_pass_percent=round(100 * passed / len(rows), 3)))
        if spec["dataset"] == "PDC":
            by_person = defaultdict(list)
            for r in rows:
                by_person[r["participant"]].append(r)
            for scheme in SCHEMES:
                # Availability count only: at least one session passes. This
                # does not select a session or define a longitudinal cohort.
                states = Counter("pass" if any(r[scheme] == "pass" for r in rr)
                                 else "unassessable" if any(r[scheme] == "unassessable" for r in rr)
                                 else "fail" for rr in by_person.values())
                n = len(by_person)
                summary.append(dict(dataset="PDC", unit="participant_with_at_least_one_passing_session", scheme=scheme,
                                    total=n, passed=states["pass"], failed=states["fail"], unassessable=states["unassessable"],
                                    confirmed_pass_percent=round(100 * states["pass"] / n, 3)))
    write(ROOT / "docs/banda_pdc_qc_summary.tsv", summary, ["dataset", "unit", "scheme", "total", "passed", "failed", "unassessable", "confirmed_pass_percent"])
    for row in summary:
        print(json.dumps(row), flush=True)
    print(f"COMPLETE {len(records)} scans; {len(runs)} runs; elapsed={time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
