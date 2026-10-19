"""Read-only comparison of the QC schemes agreed by Codex and pi Argon.

Run from the repository root with explicit py314 and numerical threads=1.
Only the comparison tables/report and temporary baseline samples are written.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/core_workflow"))
from lib.io_utils import load_subject_ids
from lib.task_quality import task_signal_validity
from lib.tsnr import runwise_tsnr, task_tsnr
from lib.workbench import run_workbench

J = Path("/home/yojiao/LSS/jianglab/3_Data_Working")
HCP = ROOT / "outputs_hcp/inputs/hcp"
SF = ROOT / "outputs_stanford/inputs/stanford"
JJ = ROOT / "outputs_iowa_jj_tms/inputs/iowa_jj_tms"
I7 = J / "fmriprep_processed_Iowa7T"
SCHEMES = ("H", "V", "M", "C", "T25")
BASELINE_CACHE: dict[tuple[str, str], np.ndarray] = {}
SAMPLE_DIR: Path | None = None


def checked(path: Path) -> Path:
    resolved = path.resolve()
    if any(p.lower() in {"archive", "_archive", "arch"} for p in resolved.parts):
        raise ValueError(f"Archive access is not authorized: {path} -> {resolved}")
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def read_json(path: Path):
    return json.loads(checked(path).read_text())


def read_tsv(path: Path) -> list[dict]:
    with checked(path).open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def unique_rows(path: Path) -> dict[str, dict]:
    result = {}
    for row in read_tsv(path):
        sid = row["subject"].removeprefix("sub-")
        if sid in result:
            raise ValueError(f"Duplicate subject {sid}: {path}")
        result[sid] = row
    return result


def conjunction(*values: bool | None) -> bool | None:
    if any(v is False for v in values):
        return False
    return None if any(v is None for v in values) else True


def coverage(fractions) -> bool | None:
    if len(fractions) != 2:
        raise ValueError(f"Expected both hemisphere coverage values: {fractions}")
    return conjunction(*(None if v is None else bool(v < 0.20) for v in fractions))


def load_signal(path: Path) -> np.ndarray:
    img = nib.load(checked(path))
    data = np.column_stack([a.data for a in img.darrays])
    if data.ndim != 2 or data.shape[0] != 512 or data.shape[1] < 2:
        raise ValueError(f"Expected den-512 vertex-by-time signal: {path}: {data.shape}")
    return data.astype(np.float64)


def dice_tsv(path: Path) -> float:
    # HippUnfold writes this score as a headerless one-number TSV.
    value = np.loadtxt(checked(path))
    if value.size != 1 or not np.isfinite(value).all():
        raise ValueError(f"Expected one finite Dice score: {path}")
    return float(value.item())


def saved_t25() -> dict:
    nb = read_json(ROOT / "docs/tsnr_existing_mask_scope_2026-09-26.ipynb")
    full = json.loads("".join(nb["cells"][19]["outputs"][0]["text"]))
    supplement = json.loads("".join(nb["cells"][21]["outputs"][0]["text"]))
    records = [r for r in full["rows"] if r["domain"] == "hipp"] + supplement
    result = {}
    # The original SF/JJ36 rows predate the per-row mapping field. Methods
    # are explicitly declared by notebook section 18 / supplement cell 21.
    # JJ36 sub-1018 L was checked again: enclosing=67 invalid vertices,
    # matching the saved count; trilinear=43. JJ22's corrected rows carry it.
    declared_methods = {"stanford36": "trilinear", "stanford22": "trilinear",
                        "jj36": "enclosing", "jj22": "trilinear"}
    for row in records:
        method = declared_methods[row["branch"]]
        if "mapping_method" in row and row["mapping_method"] != method:
            raise ValueError(f"Saved mapping contradicts declared branch: {row}")
        row["mapping_method"] = method
        key = (row["branch"], row["subject"], row["hemi"])
        if key in result:
            raise ValueError(f"Duplicate saved T25 declaration: {key}")
        result[key] = row
    return result


def cached_t25(spec, sid, hemi, cache, n_frames, validity_difference, current_invalid):
    r = cache[(spec["cache_branch"], sid, hemi)]
    expected_method = "enclosing" if spec["kind"] == "jj" and spec["variant"] == "36" else "trilinear"
    if r["mapping_method"] != expected_method or r["total"] != 512 or sum(r["retained_frames"]) != n_frames:
        raise ValueError(f"Saved T25 variant/mapping/frame mismatch: {spec['name']}/{sid}/{hemi}")
    # The saved mask already includes whole-series signal validity. Requiring
    # runwise validity can add at most `validity_difference` invalid vertices.
    # These union-size bounds can settle the participant gate without knowing
    # the masks' exact overlap; they do not reconstruct a vertex mask.
    saved_invalid = r["total"] - r["new_valid"]
    lower = max(saved_invalid, current_invalid) / 512
    upper = min(512, saved_invalid + validity_difference) / 512
    return lower, upper


def coverage_bounds(bounds) -> bool | None:
    if len(bounds) != 2:
        raise ValueError(f"Expected both hemisphere coverage bounds: {bounds}")
    return conjunction(*(None if lo is None else (True if hi < .20 else False if lo >= .20 else None)
                         for lo, hi in bounds))


def specifications():
    hcp_ids = load_subject_ids(ROOT / "manifests/hcp_7t_hippocampus_struct_complete_166.txt")
    reference = set(load_subject_ids(HCP / "qc/vertex_qc_t1only/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt"))
    excluded = set(load_subject_ids(HCP / "qc/vertex_qc_t1only/manifests/hcp_7t_hippocampus_fmri_qc_excluded_t1only_15.txt"))
    if reference & excluded or reference | excluded != set(hcp_ids):
        raise ValueError("HCP166 must be the disjoint union of the explicit reference151 and excluded15")
    hcp_dice = unique_rows(HCP / "qc/vertex_qc_t1only/qc/hcp7t_t1only_vertex_qc_166_audit.tsv")
    sf_ids = load_subject_ids(SF / "kong2022/manifests/stanford_all_subjects.txt")
    sf_dice = unique_rows(SF / "kong2022/qc/t1only_hippunfold_tsnr_qc.tsv")
    jj_ids = load_subject_ids(JJ / "kong2022/manifests/jj_tms_94_subjects.txt")
    jj_dice = unique_rows(JJ / "kong2022_22p/qc/t1only_hippunfold_tsnr_qc.tsv")
    specs = [dict(name="HCP7T rest", kind="hcp", ids=hcp_ids, dice=hcp_dice, reference=reference)]
    for task in ("emotion", "wm"):
        task_root = HCP / f"kong2022/shared_inputs/task-{task}"
        specs.append(dict(name=f"HCP3T {task.upper()}", kind="task", task=task,
                          ids=load_subject_ids(task_root / "manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_144.txt"),
                          dice=unique_rows(task_root / "qc/t1only_hippunfold_tsnr_qc.tsv"),
                          raw_root=task_root / "surface"))
    for variant in ("36", "22"):
        branch = "kong2022" if variant == "36" else "kong2022_22p"
        specs.append(dict(name=f"Stanford {variant}P", kind="stanford", variant=variant,
                          ids=sf_ids, dice=sf_dice, cache_branch=f"stanford{variant}",
                          raw_root=SF / f"{branch}/shared_inputs/surface"))
        raw_root = (ROOT / "outputs_iowa_jj_tms/work/hcp151_fixed_template_matching/subjects"
                    if variant == "36" else JJ / "kong2022/shared_inputs_22p/surface")
        specs.append(dict(name=f"JJ_TMS {variant}P", kind="jj", variant=variant,
                          ids=jj_ids, dice=jj_dice, cache_branch=f"jj{variant}", raw_root=raw_root))
    # Existing independent subject-product populations, declared before calculation.
    iowa_ids = {"0P": ["06122026"], "22P": ["06112026", "06122026"],
                "32P": ["06122026"], "36P": ["04172026", "06112026", "06122026"]}
    for variant, ids in iowa_ids.items():
        for branch in ("multiband", "multi_echo"):
            specs.append(dict(name=f"Iowa7T {variant} {branch}", kind="iowa", variant=variant,
                              branch=branch, ids=ids,
                              raw_root=I7 / f"DataOrg_PFM_{variant}/hipp_outputs/subjects"))
    for spec in specs:
        if len(spec["ids"]) != len(set(spec["ids"])):
            raise ValueError(f"Duplicate roster IDs: {spec['name']}")
    return specs


def raw_directory(spec, sid: str) -> Path:
    if spec["kind"] == "hcp":
        base = (HCP / "kong2022/shared_inputs/surface" if sid in spec["reference"]
                else HCP / "qc/vertex_qc_t1only/shared_inputs_166")
    else:
        base = spec["raw_root"]
    parent = base / f"sub-{sid}"
    if spec["kind"] == "iowa":
        parent /= spec["branch"]
    return parent / "surface/raw"


def linc_run(path: Path) -> dict:
    rows = read_tsv(path)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one LINC QC row: {path}")
    r = rows[0]
    retained, censored = int(r["num_retained_volumes"]), int(r["num_censored_volumes"])
    rms = float(r["mean_relative_rms"])
    if retained + censored <= 0 or not np.isfinite(rms):
        raise ValueError(f"Invalid LINC motion/count fields: {path}")
    return dict(path=path, retained=retained, censored=censored, rms=rms,
                dummy=int(r["num_dummy_volumes"]), censor_fraction=censored / (retained + censored))


def run_information(spec, sid: str, record: dict) -> tuple[list[int] | None, list[dict], list[Path]]:
    kind = spec["kind"]
    if kind == "hcp":
        record["run_definition"] = "declared_consumed_3600frame_concat"
        record["motion_note"] = "No declared unpacked HCP7T RMS/censor inputs"
        return None, [], []
    if kind == "task":
        d = read_json(HCP / f"kong2022/manifests/task-{spec['task']}/sub-{sid}.json")
        paths = [Path(r["bold"]["path"]) for r in d["runs"]]
        lengths = [d["expected_frames_per_run"]] * len(paths)
        motion = []
        for path, n in zip(paths, lengths):
            p = path.parent / "Movement_RelativeRMS.txt"
            values = np.loadtxt(checked(p))
            if values.shape != (n,) or not np.isfinite(values).all():
                raise ValueError(f"Task relative-RMS shape/value mismatch: {p}")
            motion.append(dict(rms=float(values.mean()), retained=n, censored=0, censor_fraction=0.0))
        record["run_definition"] = "manifest_LR_RL"
        return lengths, motion, paths
    if kind == "stanford":
        qc_rel = ("xcp_d_36P_Head05" if spec["variant"] == "36" else "xcpd_zr/xcpd_vol_22P")
        path = J / f"fmriprep_processed_CausalConnectome/rest_fmriprep/{qc_rel}/sub-{sid}/func/sub-{sid}_task-rest_space-MNI152NLin6Asym_desc-linc_qc.tsv"
        record["run_definition"] = "single_source_rest_run_not_analysis_fourway_splits"
        return None, [linc_run(path)], []
    if kind == "jj":
        if spec["variant"] == "36":
            d = read_json(spec["raw_root"] / f"sub-{sid}/subject_manifest.json")["header_evidence"]
            ids = [int(r["run_id"]) for r in d["runs"]]
            lengths = [int(v) for v in d["run_lengths"]]
        else:
            d = read_json(raw_directory(spec, sid) / "surface_sampling_summary.json")
            ids, lengths = d["run_ids"], d["run_lengths"]
        qc_rel = "xcpd/restfmri" if spec["variant"] == "36" else "xcpd_vol_22P/restfmri"
        runs = [linc_run(J / f"fmriprep_processed_JJ_TMS/{qc_rel}/sub-{sid}/func/sub-{sid}_task-rest_run-{rid:02d}_space-MNI152NLin6Asym_desc-linc_qc.tsv") for rid in ids]
        if lengths != [r["retained"] for r in runs]:
            raise ValueError(f"Declared JJ run lengths disagree with LINC: {spec['name']}/{sid}")
        record["run_definition"] = ",".join(str(i) for i in ids)
        return lengths, runs, []
    # These exact paired acquisitions are declared by the existing source runs.
    stems = (["dir-AP_run-01", "dir-PA_run-01"] if sid == "04172026"
             else ["dir-AP_run-04", "dir-PA_run-03"])
    if spec["branch"] == "multi_echo":
        stems = ["run-01", "run-02"]
    prefix = "xcpd/xcpd_vol_" if spec["branch"] == "multiband" else "xcpd/tedana_output_"
    runs = []
    for stem in stems:
        p = I7 / f"{prefix}{spec['variant']}/sub-{sid}/func/sub-{sid}_task-rest_{stem}_space-MNI152NLin6Asym_desc-linc_qc.tsv"
        info = linc_run(p)
        info["stem"] = stem
        runs.append(info)
    record["run_definition"] = "paired_source_QC:" + ",".join(stems)
    return [r["retained"] for r in runs], runs, []


def baseline_sample(volume: Path, surface: Path) -> np.ndarray:
    global SAMPLE_DIR
    key = (str(checked(volume).resolve()), str(checked(surface).resolve()))
    if key not in BASELINE_CACHE:
        if SAMPLE_DIR is None:
            SAMPLE_DIR = Path(tempfile.mkdtemp(prefix="hipp_tsnr_qc_"))
        dest = SAMPLE_DIR / f"sample_{len(BASELINE_CACHE)}.func.gii"
        print(f"BASELINE {volume.name} / {surface.name}", flush=True)
        run_workbench(["-volume-to-surface-mapping", str(volume), str(surface), str(dest), "-trilinear"])
        BASELINE_CACHE[key] = load_signal(dest)
    return BASELINE_CACHE[key]


def iowa_t25(spec, sid, hemi, raw, motion, record):
    summary = read_json(raw_directory(spec, sid) / "surface_sampling_summary.json")
    entries = [r for r in summary if r["hemi"] == hemi]
    if (len(entries) != 1 or entries[0]["mapping_method"] != "trilinear"
            or entries[0]["n_vertices"] != 512 or entries[0]["n_timepoints"] != raw.shape[1]
            or entries[0]["space"] != "corobl"):
        raise ValueError(f"Missing/ambiguous trilinear surface declaration: {spec['name']}/{sid}/{hemi}")
    # All Iowa variants explicitly use the current shared 36P subject anatomy
    # declared in docs/iowa7t_36p_hipp_outputs_layout.md. This single selected
    # contract replaces historical path strings; it is not an if-missing choice.
    surface = I7 / f"DataOrg_PFM_36P/hipp_outputs/subjects/sub-{sid}/hippunfold/sub-{sid}/surf/sub-{sid}_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    if spec["branch"] == "multiband":
        description = read_json(I7 / f"xcpd/xcpd_vol_{spec['variant']}/dataset_description.json")
        declared_source = description["DatasetLinks"]["preprocessed"]
        # Exact container-mount translations; never choose by file existence.
        source = {"/data/derivatives": "derivatives",
                  "/data/derivatives_synthstrip": "derivatives_synthstrip"}[declared_source]
    else:
        source = "xcpd/tedana_input"
    samples, frames = [], []
    for info in motion:
        stem = f"sub-{sid}_task-rest_{info['stem']}"
        volume = I7 / f"{source}/sub-{sid}/func/{stem}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
        y = baseline_sample(volume, surface)
        outliers = read_tsv(info["path"].parent / f"{stem}_outliers.tsv")
        keep = np.array([i + info["dummy"] for i, r in enumerate(outliers)
                         if float(r["framewise_displacement"]) == 0], dtype=int)
        if len(keep) != info["retained"] or len(outliers) + info["dummy"] != y.shape[1]:
            raise ValueError(f"Iowa baseline/retained-frame mismatch: {volume}")
        samples.append(y)
        frames.append(keep)
    _, valid = runwise_tsnr(samples, frames=frames, threshold=25.0)
    valid &= task_signal_validity(raw, [r["retained"] for r in motion])
    record["T25_source"] = "fresh_matched_baseline_trilinear_retained_frames"
    return float((~valid).mean())


def evaluate(spec, sid: str, cache: dict) -> dict:
    record = dict(dataset=spec["name"], subject=sid, errors="")
    fractions = {s: [] for s in ("H", "V", "T25", "T15", "T20")}
    t25_bounds = []
    dice_values = []
    notes = []
    if spec["kind"] != "iowa":
        d = spec["dice"].get(sid, {})
        dice_values = [float(d[f"dice_{h}"]) if d.get(f"dice_{h}") else None for h in ("L", "R")]
    else:
        # The subject anatomy is explicitly shared across Iowa preprocessing variants.
        for h in ("L", "R"):
            p = I7 / f"DataOrg_PFM_36P/hipp_outputs/subjects/sub-{sid}/hippunfold/sub-{sid}/qc/sub-{sid}_hemi-{h}_desc-unetf3d_dice.tsv"
            try:
                dice_values.append(dice_tsv(p))
            except (OSError, ValueError) as exc:
                dice_values.append(None)
                notes.append(f"Dice:{exc}")
    record.update(dict(zip(("dice_L", "dice_R"), dice_values)))
    anatomy = conjunction(*(None if v is None or not np.isfinite(v) else bool(v >= 0.70) for v in dice_values))
    if anatomy is False:
        record.update({s: "fail" for s in SCHEMES})
        if spec["kind"] == "task":
            record.update(T15="fail", T20="fail")
        record["errors"] = "known_Dice_failure; additional metrics not required for classification"
        return record
    lengths, motion, bolds = None, [], []
    try:
        lengths, motion, bolds = run_information(spec, sid, record)
        if motion:
            record["mean_relative_rms"] = float(np.mean([r["rms"] for r in motion]))
            record["max_censored_fraction"] = max(r["censor_fraction"] for r in motion)
    except (OSError, ValueError, KeyError) as exc:
        notes.append(f"run_or_motion:{exc}")
        # H uses the declared concatenation. V additionally requires the
        # explicit physical-run partition for task, JJ and Iowa products.
    for h in ("L", "R"):
        try:
            path = raw_directory(spec, sid) / f"sub-{sid}_hemi-{h}_space-corobl_den-512_label-hipp_bold.func.gii"
            y = load_signal(path)
            record[f"frames_{h}"] = y.shape[1]
            raw_lengths = lengths if lengths is not None else [y.shape[1]]
            if sum(raw_lengths) != y.shape[1]:
                raise ValueError(f"Declared run lengths do not partition {path}: {raw_lengths}, {y.shape}")
            finite = np.isfinite(y).all(axis=1)
            sd = np.std(y, axis=1, ddof=1)
            h_valid = finite & (sd > 0) & (sd <= 400)
            v_valid = task_signal_validity(y, raw_lengths)
            if spec["kind"] in ("task", "jj", "iowa") and lengths is None:
                fractions["V"].append(None)
            else:
                fractions["V"].append(float((~v_valid).mean()))
            fractions["H"].append(float((~h_valid).mean()))
            record[f"raw_nonfinite_vertices_{h}"] = int((~finite).sum())
            record[f"whole_vs_run_validity_difference_{h}"] = int(np.sum(v_valid != task_signal_validity(y, [y.shape[1]])))
            try:
                if spec["kind"] == "hcp":
                    value = float((~h_valid).mean())
                    record["T25_source"] = "HCP_legacy_10000_over_SD_proxy"
                elif spec["kind"] == "task":
                    scores, _ = task_tsnr(y, raw_lengths, bolds, task=spec["task"], threshold=25.0)
                    for cutoff in (15, 20, 25):
                        mask = v_valid & np.all(np.isfinite(scores) & (scores >= cutoff), axis=0)
                        fractions[f"T{cutoff}"].append(float((~mask).mean()))
                    record[f"median_run_residual_tsnr_{h}"] = ",".join(f"{np.nanmedian(a):.6g}" for a in scores)
                    record["T25_source"] = "fresh_task_OLS_DCT128_residual_SD_ddof1"
                    t25_bounds.append((fractions["T25"][-1], fractions["T25"][-1]))
                    continue
                elif spec["kind"] in ("stanford", "jj"):
                    lo, hi = cached_t25(spec, sid, h, cache, y.shape[1],
                                        record[f"whole_vs_run_validity_difference_{h}"], int((~v_valid).sum()))
                    fractions["T25"].append(lo if lo == hi else None)
                    t25_bounds.append((lo, hi))
                    record["T25_source"] = "saved_2026-09-26_matched_baseline_audit_with_4117_supplement"
                    continue
                else:
                    if not motion:
                        raise ValueError("Declared run QC required for matched baseline tSNR")
                    value = iowa_t25(spec, sid, h, y, motion, record)
                fractions["T25"].append(value)
                t25_bounds.append((value, value))
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                fractions["T25"].append(None)
                t25_bounds.append((None, None))
                if spec["kind"] == "task":
                    fractions["T15"].append(None)
                    fractions["T20"].append(None)
                notes.append(f"T25_{h}:{exc}")
        except (OSError, ValueError, KeyError) as exc:
            for s in fractions:
                fractions[s].append(None)
            t25_bounds.append((None, None))
            notes.append(f"signal_{h}:{exc}")
    for s in fractions:
        for h, v in zip(("L", "R"), fractions[s]):
            record[f"{s}_invalid_fraction_{h}"] = v
    for h, (lo, hi) in zip(("L", "R"), t25_bounds):
        record[f"T25_invalid_fraction_lower_{h}"] = lo
        record[f"T25_invalid_fraction_upper_{h}"] = hi
    h_state = conjunction(anatomy, coverage(fractions["H"]))
    v_state = conjunction(anatomy, coverage(fractions["V"]))
    m_state = conjunction(v_state, None if not motion else bool(record["mean_relative_rms"] <= 0.50))
    c_state = conjunction(m_state, None if not motion else bool(record["max_censored_fraction"] <= 0.50))
    states = dict(H=h_state, V=v_state, M=m_state, C=c_state,
                  T25=conjunction(v_state, coverage_bounds(t25_bounds)))
    if spec["kind"] == "task":
        for s in ("T15", "T20"):
            states[s] = conjunction(anatomy, coverage(fractions[s]))
    for s, value in states.items():
        record[s] = "unassessable" if value is None else ("pass" if value else "fail")
    record["errors"] = " | ".join(notes)
    return record


def write_tables(records):
    columns = ["dataset", "subject", *SCHEMES, "T15", "T20"]
    columns += sorted(set().union(*(r.keys() for r in records)) - set(columns))
    subject_path = ROOT / "docs/tsnr_qc_comparison_subjects.tsv"
    with subject_path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
    summary = []
    by_dataset = defaultdict(list)
    for r in records:
        by_dataset[r["dataset"]].append(r)
    for dataset, rows in by_dataset.items():
        for s in (*SCHEMES, "T15", "T20"):
            states = [r[s] for r in rows if s in r and r[s] != ""]
            if not states:
                continue
            counts = Counter(states)
            total = len(rows)
            if sum(counts.values()) != total:
                raise ValueError(f"Incomplete scheme entries: {dataset}/{s}")
            passed, failed, missing = (counts[x] for x in ("pass", "fail", "unassessable"))
            assert passed + failed + missing == total
            evaluable = total - missing
            summary.append(dict(dataset=dataset, scheme=s, n_total=total, n_pass=passed,
                                n_fail=failed, n_unassessable=missing, n_evaluable=evaluable,
                                confirmed_pass_percent=round(100 * passed / total, 3),
                                pass_percent_evaluable=round(100 * passed / evaluable, 3) if evaluable else ""))
    with (ROOT / "docs/tsnr_qc_comparison_summary.tsv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary)
    for r in summary:
        print(json.dumps(r), flush=True)


def main():
    # Direct checks for the counting decisions that would change the output.
    assert conjunction(False, None) is False
    assert conjunction(True, None) is None
    assert conjunction(True, True) is True
    assert coverage([0.20, 0.0]) is False
    assert coverage([0.199, 0.0]) is True
    assert coverage_bounds([(.19, .21), (0, 0)]) is None
    assert coverage_bounds([(.10, .19), (0, 0)]) is True
    assert coverage_bounds([(.20, .21), (None, None)]) is False
    specs = specifications()
    cache = saved_t25()
    records = []
    started = time.monotonic()
    jobs = [(s, sid) for s in specs if s["kind"] != "iowa" for sid in s["ids"]]
    print(f"START {len(jobs)} primary subject-branch entries; two workers, numerical threads=1", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(evaluate, s, sid, cache): (s["name"], sid) for s, sid in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            records.append(future.result())
            if i % 16 == 0:
                print(f"PRIMARY {i}/{len(jobs)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    # Serial Iowa sampling permits reuse of exactly identical source/surface pairs
    # and avoids concurrent Workbench readers of the same large volume.
    for s in specs:
        if s["kind"] == "iowa":
            for sid in s["ids"]:
                print(f"IOWA {s['name']} {sid}", flush=True)
                records.append(evaluate(s, sid, cache))
    order = {s["name"]: i for i, s in enumerate(specs)}
    records.sort(key=lambda r: (order[r["dataset"]], r["subject"]))
    write_tables(records)
    print(f"COMPLETE {len(records)} subject-branch entries; elapsed={time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
