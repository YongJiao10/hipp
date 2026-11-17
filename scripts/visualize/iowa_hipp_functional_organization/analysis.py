"""Hash-validated Iowa hippocampal functional-organization analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _bootstrap_repository_root():
    """Make repository package imports available to the direct script entrypoint."""
    repository_root = str(Path(__file__).resolve().parents[3])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)


try:
    from . import inputs, metrics
except ImportError:  # Direct script execution keeps the same local module.
    _bootstrap_repository_root()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import inputs
    import metrics


COMPARISONS = ("patient", "pure_mdd", "comorbid_mdd")


def wta_pseudoclusters(labels, label_map):
    """Map only VAN/SMN/DMN WTA labels to A/B/P; every other label is zero."""
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("WTA labels must be one dimensional")
    keys = label_map.get("network_label_keys")
    if not isinstance(keys, dict):
        raise ValueError("WTA label map lacks network_label_keys")
    required = ("VentralAttention", "Somatomotor", "Default")
    if any(name not in keys for name in required):
        raise ValueError("WTA label map lacks VAN, SMN, or DMN")
    selected = [int(keys[name]) for name in required]
    if len(set(selected)) != 3:
        raise ValueError("WTA VAN, SMN, and DMN labels must be distinct")
    out = np.zeros(values.shape, dtype=np.int32)
    out[values == selected[0]] = 1  # VAN = A
    out[values == selected[1]] = 2  # SMN = B
    out[values == selected[2]] = 3  # DMN = P
    return out


def _finite_value(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if math.isfinite(result) else np.nan


def statistics_from_subject_rows(rows):
    """Apply all three declared Welch contrasts to every complete metric key."""
    groups = defaultdict(lambda: defaultdict(dict))
    reasons = defaultdict(lambda: defaultdict(set))
    dimensions = ("theme", "branch", "hemi", "atlas", "system", "metric", "target")
    for row in rows:
        key = tuple(str(row.get(name, "")) for name in dimensions)
        cohort = str(row.get("cohort", ""))
        subject = str(row.get("subject", ""))
        if cohort not in ("hc", *COMPARISONS):
            raise ValueError(f"Unexpected analysis cohort {cohort!r}")
        if not subject:
            raise ValueError(f"Missing subject for statistic {key}")
        if subject in groups[key][cohort]:
            raise ValueError(f"Duplicate subject statistic: key={key}, cohort={cohort}, subject={subject}")
        groups[key][cohort][subject] = _finite_value(row.get("value"))
        if row.get("reason"):
            reasons[key][cohort].add(str(row["reason"]))

    estimates = []
    for key in sorted(groups):
        samples = groups[key]
        hc = list(samples["hc"].values())
        for comparison in COMPARISONS:
            patient = list(samples[comparison].values())
            result = metrics.welch(patient, hc)
            source_reason = "; ".join(sorted(reasons[key]["hc"] | reasons[key][comparison]))
            reason = result["reason"] or source_reason
            estimate = dict(zip(dimensions, key))
            estimate.update(result)
            estimate.update(
                comparison=comparison,
                case_cohort=comparison,
                control_cohort="hc",
                reason=reason,
                significance=metrics.significance(result["p"]),
            )
            estimates.append(estimate)
    return estimates



HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results"
NETWORKS = metrics.NETWORKS
ANALYSIS_COHORTS = ("hc", "patient", "pure_mdd", "comorbid_mdd")
SCALAR_FIELDS = (
    "theme", "branch", "hemi", "atlas", "cohort", "subject", "system",
    "metric", "target", "target_level", "roi_id", "value", "reason",
    "valid_area", "total_area", "n_valid_boundaries",
)
CURVE_FIELDS = (
    "theme", "branch", "hemi", "atlas", "cohort", "subject", "system",
    "metric", "bin", "position", "value", "reason", "valid_area",
    "total_area", "scope",
)
BOUNDARY_FIELDS = (
    "theme", "branch", "hemi", "atlas", "cohort", "subject", "system_pair",
    "position", "reason", "scope",
)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_csv(path, rows, fields):
    rows = list(rows)
    names = list(fields)
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            clean = {
                key: (
                    ""
                    if isinstance(value, float) and not math.isfinite(value)
                    else value
                )
                for key, value in row.items()
            }
            writer.writerow(clean)


def _output_hash(path):
    raw = Path(path).read_bytes()
    return {
        "path": str(Path(path).resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _checked_output_root(path):
    root = Path(path).resolve()
    if not root.is_relative_to(HERE.resolve()):
        raise ValueError(f"Output must stay inside this analysis directory: {root}")
    return root


def detect_system():
    values = [
        os.environ.get("SYSTEM_NAME", ""),
        os.environ.get("CLUSTER_NAME", ""),
        os.environ.get("SGE_CLUSTER_NAME", ""),
        os.uname().nodename,
    ]
    text = " ".join(values).lower()
    hits = [name for name in ("idas", "argon") if name in text]
    if len(hits) != 1:
        raise RuntimeError(
            "Unable to identify exactly one execution system from "
            "SYSTEM_NAME, CLUSTER_NAME, SGE_CLUSTER_NAME, or hostname"
        )
    return hits[0]


THREAD_ENVIRONMENT = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def _single_thread_environment():
    values = {name: os.environ.get(name, "") for name in THREAD_ENVIRONMENT}
    invalid = {name: value for name, value in values.items() if value != "1"}
    if invalid:
        details = ", ".join(f"{name}={value!r}" for name, value in invalid.items())
        raise RuntimeError(
            "This analysis requires one numerical-library thread per process; "
            f"set {details} to '1' before launch"
        )
    return values



def _participant_cohort(inventory):
    rosters = inventory["rosters"]
    mapping = {subject: "hc" for subject in rosters["hc"]}
    for cohort in ("pure_mdd", "comorbid_mdd"):
        for subject in rosters[cohort]:
            if subject in mapping:
                raise ValueError(f"Subject belongs to conflicting cohorts: {subject}")
            mapping[subject] = cohort
    expected = set(rosters["hc"]) | set(rosters["patient"])
    if set(mapping) != expected:
        raise ValueError("Inventory subject roster does not match HC + patient scope")
    return mapping


def _analysis_memberships(base_cohort):
    if base_cohort == "hc":
        return ("hc",)
    if base_cohort in ("pure_mdd", "comorbid_mdd"):
        return ("patient", base_cohort)
    raise ValueError(f"Unsupported base cohort: {base_cohort}")


def _subject_index(inventory):
    index = {}
    for row in inventory["subjects"]:
        key = (row["branch"], row["subject"], row["hemi"])
        if key in index:
            raise ValueError(f"Duplicate inventory subject entry: {key}")
        index[key] = row
    return index


def _group_index(inventory):
    index = {}
    for row in inventory["groups"]:
        key = (row["branch"], row["cohort"], row["hemi"])
        if key in index:
            raise ValueError(f"Duplicate group atlas entry: {key}")
        index[key] = row
    return index


def _reference_index(inventory):
    index = {}
    for row in inventory["references"]:
        key = (row["branch"], row["hemi"])
        if key in index:
            raise ValueError(f"Duplicate HCP gradient reference: {key}")
        index[key] = row
    return index


def _hcp_template_index(inventory):
    """Load the one fixed HCP A/B/P territory for each branch and hemisphere."""
    index = {}
    for row in inventory["hcp_templates"]:
        key = (row["branch"], row["hemi"])
        if key in index:
            raise ValueError(f"Duplicate fixed HCP template: {key}")
        labels = _array(row["labels"], context="fixed HCP template labels").astype(np.int32)
        support_raw = _array(row["support"], context="fixed HCP template support")
        if labels.shape != (512,) or support_raw.shape != (512,):
            raise ValueError(f"Fixed HCP template shape mismatch: {key}")
        if not np.isin(labels, (0, 1, 2, 3)).all() or not np.isin(support_raw, (0, 1, False, True)).all():
            raise ValueError(f"Fixed HCP template values mismatch: {key}")
        support = support_raw.astype(bool)
        if not np.array_equal(labels != 0, support):
            raise ValueError(f"HCP labels/support mismatch: {key}")
        index[key] = (labels, support)
    expected = {(branch, hemi) for branch in ("standard", "ctxreg") for hemi in ("L", "R")}
    if set(index) != expected:
        raise ValueError(f"Fixed HCP template coverage mismatch: expected={sorted(expected)}, got={sorted(index)}")
    return index


def _array(record, *, context):
    path = inputs.verify_record(record)
    try:
        return np.asarray(np.load(path, allow_pickle=False))
    except Exception as exc:
        raise ValueError(f"Cannot load validated {context}: {path}") from exc


def _tm_dominant_maps(inventory, participant_cohort):
    """Reproduce the existing explicit majority-plus-mean-score tie rule."""
    subject_rows = _subject_index(inventory)
    maps = {}
    for branch in ("standard", "ctxreg"):
        for cohort in ANALYSIS_COHORTS:
            subjects = inventory["rosters"][cohort]
            for hemi in ("L", "R"):
                label_stack = []
                score_stack = []
                valid_stack = []
                for subject in sorted(subjects):
                    row = subject_rows[(branch, subject, hemi)]
                    label_stack.append(_array(row["labels"], context="TM labels"))
                    score_stack.append(_array(row["scores"], context="TM template scores"))
                    valid_stack.append(
                        _array(row["valid"], context="TM valid surface mask")
                    )
                labels = np.asarray(label_stack, dtype=np.int32)
                scores = np.asarray(score_stack, dtype=float)
                valid = np.asarray(valid_stack)
                if (
                    labels.shape[1:] != (512,)
                    or scores.shape[1:] != (512, 3)
                    or valid.shape[1:] != (512,)
                ):
                    raise ValueError(
                        f"TM dominant map input shape mismatch: {branch}/{cohort}/{hemi}"
                    )
                if not np.isin(labels, (0, 1, 2, 3)).all() or not np.isin(valid, (0, 1)).all():
                    raise ValueError(
                        f"TM dominant map labels/mask mismatch: {branch}/{cohort}/{hemi}"
                    )
                usable = (
                    valid.astype(bool)
                    & np.isfinite(scores).all(axis=2)
                    & np.isin(labels, (1, 2, 3))
                )
                counts = np.stack(
                    [np.sum(usable & (labels == code), axis=0) for code in (1, 2, 3)],
                    axis=-1,
                )
                n_usable = usable.sum(axis=0)
                score_sum = np.where(usable[:, :, None], scores, 0.0).sum(axis=0)
                mean_scores = np.divide(
                    score_sum,
                    n_usable[:, None],
                    out=np.zeros((512, 3), dtype=float),
                    where=n_usable[:, None] > 0,
                )
                combined = counts.astype(float) + 1e-4 * mean_scores
                dominant = np.argmax(combined, axis=-1).astype(np.int32) + 1
                dominant[n_usable == 0] = 0
                maps[(branch, cohort, hemi)] = dominant
    return maps


def _atlas_supports(dominant, de_novo_support, wta_support):
    """Keep each descriptive atlas on its own declared validity domain."""
    dominant = np.asarray(dominant, dtype=np.int32)
    de_novo_support = np.asarray(de_novo_support)
    wta_support = np.asarray(wta_support)
    if (
        dominant.shape != (512,)
        or de_novo_support.shape != (512,)
        or wta_support.shape != (512,)
    ):
        raise ValueError("Descriptive atlas support arrays must each have 512 vertices")
    if not np.isin(dominant, (0, 1, 2, 3)).all():
        raise ValueError("TM-dominant labels must be 0/A/B/P")
    for name, support in (("de novo", de_novo_support), ("WTA", wta_support)):
        if not np.isin(support, (0, 1)).all():
            raise ValueError(f"Invalid {name} descriptive support mask")
    return {
        "tm_dominant": dominant != 0,
        "de_novo": de_novo_support.astype(bool),
        "wta": wta_support.astype(bool),
    }


def _gradient_reference(inventory):
    refs = {}
    for key, row in _reference_index(inventory).items():
        raw = _array(row["gradients"], context="external HCP gradients")
        support = _array(row["support"], context="external HCP gradient support").astype(bool)
        if raw.shape != (512, 5) or support.shape != (512,):
            raise ValueError(f"External HCP gradient shape mismatch: {key}")
        refs[key] = (raw.astype(float), support)
    return refs


def write_inventory(output_root):
    """Build and verify the selected-source manifest; it never discovers alternatives."""
    root = _checked_output_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    inventory = inputs.make_inventory()
    inputs.validate_inventory(inventory)
    _write_json(root / "input_manifest.json", inventory)

    audit = []
    for row in inventory["subjects"]:
        for artifact, record in row.items():
            if isinstance(record, dict) and "path" in record and "sha256" in record:
                audit.append(
                    {
                        "kind": "subject",
                        "branch": row["branch"],
                        "subject": row["subject"],
                        "hemi": row["hemi"],
                        "artifact": artifact,
                        **record,
                    }
                )
    for row in inventory["groups"] + inventory["references"] + inventory["hcp_templates"]:
        for artifact, record in row.items():
            if isinstance(record, dict) and "path" in record and "sha256" in record:
                audit.append(
                    {
                        "kind": "group_or_reference",
                        "branch": row["branch"],
                        "subject": "",
                        "hemi": row.get("hemi", ""),
                        "artifact": artifact,
                        **record,
                    }
                )
    _write_csv(
        root / "input_audit.csv",
        audit,
        ("kind", "branch", "subject", "hemi", "artifact", "path", "sha256", "size_bytes"),
    )
    return inventory


def load_inventory(output_root):
    root = _checked_output_root(output_root)
    path = root / "input_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run the inventory stage first: {path}")
    inventory = json.loads(path.read_text(encoding="utf-8"))
    inputs.validate_inventory(inventory)
    return inventory



def _metric_row(base, metric_row, *, target="", target_level="hippocampus"):
    return {
        **base,
        "system": metric_row["system"],
        "metric": metric_row["metric"],
        "target": target,
        "target_level": target_level,
        "roi_id": "",
        "value": metric_row.get("value", np.nan),
        "reason": metric_row.get("reason", ""),
        "valid_area": metric_row.get("valid_area", np.nan),
        "total_area": metric_row.get("total_area", np.nan),
        "n_valid_boundaries": metric_row.get("n_valid_boundaries", np.nan),
    }


def _append_metric_rows(destination, base, rows, memberships, *, target="", target_level="hippocampus"):
    for membership in memberships:
        member_base = dict(base, cohort=membership)
        destination.extend(
            _metric_row(member_base, row, target=target, target_level=target_level)
            for row in rows
        )


def _append_curve_rows(destination, base, rows, memberships, *, scope="individual"):
    for membership in memberships:
        for row in rows:
            destination.append(
                {
                    **base,
                    "cohort": membership,
                    "system": row["system"],
                    "metric": row["metric"],
                    "bin": row["bin"],
                    "position": row.get("axis", row.get("position", np.nan)),
                    "value": row.get("value", np.nan),
                    "reason": row.get("reason", ""),
                    "valid_area": row.get("valid_area", np.nan),
                    "total_area": row.get("total_area", np.nan),
                    "scope": scope,
                }
            )


def _append_boundary_rows(destination, base, rows, memberships, *, scope="individual_tm"):
    """Persist every boundary edge instead of selecting one transition per system."""
    for membership in memberships:
        for row in rows:
            destination.append(
                {
                    **base,
                    "cohort": membership,
                    "system_pair": row["system_pair"],
                    "position": row["position"],
                    "reason": row.get("reason", ""),
                    "scope": scope,
                }
            )


def _fit_individual_gradient(fc, valid, reference, reference_support, axis):
    """Fit raw five-component Kong400 gradients and align only in HCP reference space."""
    from scripts.core_workflow.lib.hippomaps_adapter import fit_diffusion_gradients

    source_support = np.asarray(valid, bool) & np.isfinite(fc).all(axis=1)
    raw = np.full((512, 5), np.nan, dtype=np.float32)
    lambdas = np.full(5, np.nan, dtype=float)
    ev = np.full(5, np.nan, dtype=float)
    fit_reason = ""
    if source_support.sum() <= 5:
        fit_reason = "fewer than six finite valid vertices for five-component gradient"
    else:
        try:
            active, lambdas, ev = fit_diffusion_gradients(
                np.asarray(fc[source_support], dtype=np.float32),
                n_components=5,
            )
            if active.shape != (int(source_support.sum()), 5):
                raise RuntimeError(f"unexpected individual gradient shape {active.shape}")
            raw[source_support] = active
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            fit_reason = f"individual gradient noncomputable: {exc}"
    aligned, rows = metrics.aligned_gradient_metrics(
        raw,
        reference,
        axis,
        source_support & np.asarray(reference_support, bool),
    )
    if fit_reason:
        rows = [dict(row, reason=fit_reason) for row in rows]
    return raw, aligned, rows, np.asarray(lambdas), np.asarray(ev), fit_reason


def _worker(task):
    """Numerical worker for one explicit branch/subject/hemisphere tuple."""
    record = task["record"]
    branch = task["branch"]
    subject = task["subject"]
    hemi = task["hemi"]
    base_cohort = task["base_cohort"]
    memberships = _analysis_memberships(base_cohort)
    scores = _array(record["scores"], context="template scores").astype(float)
    labels = _array(record["labels"], context="template labels").astype(np.int32)
    fc = _array(record["fisherz_fc"], context="individual 400-ROI Fisher-z FC").astype(float)
    ts = _array(record["ts"], context="hippocampal timeseries").astype(float)
    roi = _array(record["roi"], context="cortical ROI timeseries").astype(float)
    valid = _array(record["valid"], context="valid surface mask").astype(bool)
    inputs.validate_arrays(ts, roi, valid.astype(np.uint8), f"{branch}/{subject}/{hemi}")
    if scores.shape != (512, 3) or labels.shape != (512,) or fc.shape != (512, 400):
        raise ValueError(f"Subject TM/FC shape mismatch: {branch}/{subject}/{hemi}")
    if not np.isin(labels, (0, 1, 2, 3)).all():
        raise ValueError(f"Subject TM labels outside 0/1/2/3: {branch}/{subject}/{hemi}")

    native_points, faces = inputs.surface(record["native"]["path"])
    unfolded_points, unfolded_faces = inputs.surface(record["unfolded"]["path"])
    if not np.array_equal(faces, unfolded_faces):
        raise ValueError(f"Native/unfolded topology mismatch: {branch}/{subject}/{hemi}")
    area = inputs.vertex_areas(native_points, faces)
    axis = inputs.unfold_axis(unfolded_points)

    base = {
        "theme": "",
        "branch": branch,
        "hemi": hemi,
        "atlas": "",
        "cohort": "",
        "subject": subject,
    }
    scalar_rows = []
    curve_rows = []
    boundary_rows = []

    identity_rows = metrics.identity(scores, labels, area, valid=valid)
    _append_metric_rows(
        scalar_rows,
        dict(base, theme="identity", atlas="tm_individual"),
        identity_rows,
        memberships,
    )
    fixed_hcp_rows = metrics.fixed_hcp_identity(
        scores,
        np.asarray(task["hcp_labels"], dtype=np.int32),
        np.asarray(task["hcp_support"]),
        area,
        valid=valid,
    )
    _append_metric_rows(
        scalar_rows,
        dict(base, theme="identity", atlas="hcp_fixed_territory"),
        fixed_hcp_rows,
        memberships,
    )
    spatial_rows, spatial_curves = metrics.spatial(labels, area, axis, valid=valid)
    _append_metric_rows(
        scalar_rows,
        dict(base, theme="spatial", atlas="tm_individual"),
        spatial_rows,
        memberships,
    )
    _append_curve_rows(
        curve_rows,
        dict(base, theme="spatial", atlas="tm_individual", subject=subject),
        spatial_curves,
        memberships,
    )
    _append_boundary_rows(
        boundary_rows,
        dict(base, theme="spatial", atlas="tm_individual", subject=subject),
        metrics.boundary_positions(labels, axis, faces, valid),
        memberships,
    )
    template_rows, template_curves = metrics.template_axis(scores, area, axis, valid=valid)
    _append_metric_rows(
        scalar_rows,
        dict(base, theme="continuous", atlas="tm_individual"),
        template_rows,
        memberships,
    )
    _append_curve_rows(
        curve_rows,
        dict(base, theme="continuous", atlas="tm_individual", subject=subject),
        template_curves,
        memberships,
    )
    kember_rows = metrics.kember(ts, labels, valid, native_points, faces, radius=4.0)
    _append_metric_rows(
        scalar_rows,
        dict(base, theme="identity", atlas="tm_individual"),
        kember_rows,
        memberships,
    )

    atlas_labels_by_membership = task["atlas_labels"]
    if set(atlas_labels_by_membership) != set(memberships):
        raise ValueError(
            f"Atlas membership contract mismatch: {branch}/{base_cohort}/{hemi}"
        )
    registry = task["registry"]
    networks = np.asarray([row["network"] for row in registry], dtype=object)
    roi_profiles = {}
    profile_cache = {}

    # Patient-26, Pure-MDD, and Comorbid-MDD use separate explicitly declared
    # group atlases. A subgroup participant is therefore re-extracted in the
    # Patient-26 atlas only for that overlapping Patient-26 comparison.
    for membership in memberships:
        label_sets = {
            atlas: np.asarray(values, dtype=np.int32)
            for atlas, values in atlas_labels_by_membership[membership].items()
        }
        if set(label_sets) != {"tm_dominant", "de_novo", "wta"}:
            raise ValueError(f"Incomplete atlas set for {membership}")
        for atlas, atlas_labels in label_sets.items():
            if atlas_labels.shape != (512,) or not np.isin(atlas_labels, (0, 1, 2, 3)).all():
                raise ValueError(
                    f"Invalid {atlas} labels: {branch}/{membership}/{hemi}"
                )

        roi_profiles[membership] = {}
        for atlas, atlas_labels in label_sets.items():
            z, reasons = metrics.regional_fc(ts, roi, atlas_labels, valid)
            net, network_names = metrics.network_fc(z, networks)
            roi_profiles[membership][atlas] = z.astype(np.float32)
            profile_cache[f"{membership}_{atlas}_roi"] = z.astype(np.float32)
            for system_index, system in enumerate(metrics.SYSTEMS):
                for network_index, network in enumerate(network_names):
                    reason = reasons[system_index]
                    if not np.isfinite(net[system_index, network_index]) and not reason:
                        reason = "network requires all registered ROI Fisher-z values"
                    metric_row = {
                        "system": system,
                        "metric": "fisher_z",
                        "value": net[system_index, network_index],
                        "reason": reason,
                    }
                    _append_metric_rows(
                        scalar_rows,
                        dict(base, theme="fingerprints", atlas=atlas),
                        [metric_row],
                        (membership,),
                        target=network,
                        target_level="network",
                    )

    raw_gradients, aligned_gradients, gradient_rows, lambdas, ev, gradient_reason = (
        _fit_individual_gradient(
            fc,
            valid,
            np.asarray(task["reference"], float),
            np.asarray(task["reference_support"], bool),
            axis,
        )
    )
    _append_metric_rows(
        scalar_rows,
        dict(base, theme="continuous", atlas="individual_gradient"),
        gradient_rows,
        memberships,
    )

    cache_path = Path(task["cache_path"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        raw_gradients=raw_gradients,
        aligned_gradients=aligned_gradients,
        lambdas=lambdas,
        ev_fractions=ev,
        **profile_cache,
        gradient_reason=np.asarray([gradient_reason]),
    )
    return {
        "scalar_rows": scalar_rows,
        "curve_rows": curve_rows,
        "boundary_rows": boundary_rows,
        "roi_profiles": roi_profiles,
        "gradient": aligned_gradients,
        "branch": branch,
        "hemi": hemi,
        "subject": subject,
        "base_cohort": base_cohort,
        "axis": axis,
        "transverse": unfolded_points[:, 0],
    }



def _roi_profile_items(profiles, memberships):
    """Yield exact membership-specific atlas profiles without cross-cohort reuse."""
    expected_memberships = tuple(memberships)
    if set(profiles) != set(expected_memberships):
        raise ValueError(
            "ROI-profile membership contract mismatch: "
            f"got {sorted(profiles)}, expected {list(expected_memberships)}"
        )
    expected_atlases = ("tm_dominant", "de_novo", "wta")
    for membership in expected_memberships:
        atlases = profiles[membership]
        if set(atlases) != set(expected_atlases):
            raise ValueError(
                f"ROI-profile atlas contract mismatch for {membership}: "
                f"got {sorted(atlases)}"
            )
        for atlas in expected_atlases:
            z = np.asarray(atlases[atlas], dtype=float)
            if z.shape != (len(metrics.SYSTEMS), 400):
                raise ValueError(
                    f"ROI-profile shape mismatch for {membership}/{atlas}: {z.shape}"
                )
            yield membership, atlas, z
def _roi_statistics(profile_store, registry):
    """Compute and retain every ROI's uncorrected patient-minus-HC effect."""
    estimates = []
    for key in sorted(profile_store):
        branch, hemi, atlas, system = key
        groups = profile_store[key]
        arrays = {}
        for cohort in ANALYSIS_COHORTS:
            values = groups.get(cohort, [])
            arrays[cohort] = np.vstack(values) if values else np.empty((0, 400))
        for roi_index, registry_row in enumerate(registry):
            for comparison in COMPARISONS:
                result = metrics.welch(
                    arrays[comparison][:, roi_index],
                    arrays["hc"][:, roi_index],
                )
                estimates.append(
                    {
                        "theme": "fingerprints",
                        "branch": branch,
                        "hemi": hemi,
                        "atlas": atlas,
                        "system": system,
                        "metric": "fisher_z",
                        "target": registry_row["roi_name"],
                        "target_level": "roi",
                        "roi_id": int(registry_row["roi_id"]),
                        "comparison": comparison,
                        "case_cohort": comparison,
                        "control_cohort": "hc",
                        **result,
                        "significance": metrics.significance(result["p"]),
                    }
                )
    return estimates


def _cortex_area_rows(inventory):
    """Reuse only the verified Iowa canonical7 36P cortical-area subject table."""
    path = inputs.verify_record(inventory["cortex_subjects"])
    expected = {
        "Iowa HC": "hc",
        "Patient": "patient",
        "Pure MDD": "pure_mdd",
        "Comorbid MDD": "comorbid_mdd",
    }
    selected = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["variant"] != "36p"
                or row["method"] != "Kong2022"
                or row["network_classification"] != "canonical7"
                or row["taxonomy"] != "kong_canonical7"
                or row["cohort"] not in expected
            ):
                continue
            component_kind = row.get("component_kind")
            if component_kind in ("noise", "unassigned"):
                if row["network"] != component_kind.title():
                    raise ValueError(
                        f"Malformed canonical7 background component: {row}"
                    )
                continue
            if component_kind != "network" or row["network"] not in inputs.NETWORKS:
                raise ValueError(f"Unexpected canonical7 cortex row: {row}")
            key = (row["cohort"], row["subject"], row["network"])
            if key in selected:
                raise ValueError(f"Duplicate selected cortical-area row: {key}")
            selected[key] = row
    rows = []
    for source_cohort, cohort in expected.items():
        subjects = {subject for name, subject, _network in selected if name == source_cohort}
        expected_subjects = set(inventory["rosters"][cohort])
        if subjects != expected_subjects:
            missing = sorted(expected_subjects - subjects)
            unexpected = sorted(subjects - expected_subjects)
            raise ValueError(
                f"Canonical7 36P cortical-area subject roster mismatch for {source_cohort}: "
                f"missing={missing}, unexpected={unexpected}"
            )
        for subject in subjects:
            names = {
                network for name, sid, network in selected
                if name == source_cohort and sid == subject
            }
            if names != set(inputs.NETWORKS):
                raise ValueError(
                    f"Incomplete canonical7 cortex rows for {source_cohort}/{subject}: {names}"
                )
            for network in sorted(names):
                item = selected[(source_cohort, subject, network)]
                rows.append(
                    {
                        "theme": "spatial",
                        "branch": "shared_36p_cortex",
                        "hemi": "bilateral_cortex",
                        "atlas": "cortex_canonical7",
                        "cohort": cohort,
                        "subject": subject,
                        "system": "cortex",
                        "metric": "area_percent",
                        "target": inputs.NETWORKS[network],
                        "target_level": "network",
                        "roi_id": "",
                        "value": float(item["area_percent"]),
                        "reason": "",
                        "valid_area": np.nan,
                        "total_area": float(item["total_cortical_area_mm2"]),
                        "n_valid_boundaries": np.nan,
                    }
                )
    return rows


def _group_descriptions(inventory, participant_cohort, dominant_maps, gradient_groups):
    """Make non-inferential group atlas curves/maps on declared shared topology."""
    groups = _group_index(inventory)
    subjects = _subject_index(inventory)
    curve_rows = []
    spatial_maps = []

    for branch in ("standard", "ctxreg"):
        for cohort in ANALYSIS_COHORTS:
            for hemi in ("L", "R"):
                entry = groups[(branch, cohort, hemi)]
                group_points, group_faces = inputs.surface(entry["surface"]["path"])
                area = inputs.vertex_areas(group_points, group_faces)

                # The group atlas manifest records topology compatibility. The exact
                # subject is a declared presentation anchor, never a first-found file.
                anchor = subjects[(branch, "sub-1001", hemi)]
                unfolded_points, anchor_faces = inputs.surface(anchor["unfolded"]["path"])
                if not np.array_equal(group_faces, anchor_faces):
                    raise ValueError(
                        f"Group/anchor topology mismatch: {branch}/{cohort}/{hemi}"
                    )
                axis = inputs.unfold_axis(unfolded_points)
                transverse = unfolded_points[:, 0]
                de_novo = _array(entry["labels"], context="de novo group labels").astype(np.int32)
                wta_raw = _array(entry["wta_labels"], context="WTA group labels").astype(np.int32)
                de_novo_support = _array(entry["support"], context="de novo group support")
                wta_support = _array(entry["wta_support"], context="WTA group support")
                wta_map = json.loads(
                    inputs.verify_record(entry["wta_map"]).read_text(encoding="utf-8")
                )
                maps = {
                    "tm_dominant": dominant_maps[(branch, cohort, hemi)],
                    "de_novo": de_novo,
                    "wta": wta_pseudoclusters(wta_raw, wta_map),
                }
                supports = _atlas_supports(
                    maps["tm_dominant"], de_novo_support, wta_support
                )
                for atlas, labels in maps.items():
                    support = supports[atlas]
                    if labels.shape != (512,) or not np.isin(labels, (0, 1, 2, 3)).all():
                        raise ValueError(
                            f"Invalid descriptive {atlas} labels: {branch}/{cohort}/{hemi}"
                        )
                    _rows, curves = metrics.spatial(labels, area, axis, valid=support)
                    _append_curve_rows(
                        curve_rows,
                        {
                            "theme": "spatial",
                            "branch": branch,
                            "hemi": hemi,
                            "atlas": atlas,
                            "subject": "group_atlas",
                        },
                        curves,
                        (f"group_{cohort}",),
                        scope="descriptive_group_atlas",
                    )
                    spatial_maps.append(
                        {
                            "theme": "spatial",
                            "branch": branch,
                            "cohort": cohort,
                            "hemi": hemi,
                            "atlas": atlas,
                            "scope": "descriptive_group_atlas",
                            "display_geometry_subject": "sub-1001",
                            "axis": axis.tolist(),
                            "transverse": transverse.tolist(),
                            "labels": labels.tolist(),
                            "support": support.tolist(),
                        }
                    )

                aligned = gradient_groups.get((branch, hemi, cohort), [])
                if aligned:
                    stack = np.stack(aligned, axis=0)
                    finite = np.isfinite(stack)
                    numerator = np.where(finite, stack, 0.0).sum(axis=0)
                    count = finite.sum(axis=0)
                    group_mean = np.divide(
                        numerator,
                        count,
                        out=np.full(numerator.shape, np.nan, dtype=float),
                        where=count > 0,
                    )
                    spatial_maps.append(
                        {
                            "theme": "continuous",
                            "branch": branch,
                            "cohort": cohort,
                            "hemi": hemi,
                            "atlas": "individual_gradient",
                            "scope": "descriptive_group_mean",
                            "display_geometry_subject": "sub-1001",
                            "axis": axis.tolist(),
                            "transverse": transverse.tolist(),
                            "labels": [0] * 512,
                            "gradients": group_mean[:, :3].tolist(),
                        }
                    )
    return curve_rows, spatial_maps


def _guard_outputs(root, names, overwrite):
    existing = [name for name in names if (root / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite prior analysis outputs without --overwrite: "
            + ", ".join(existing)
        )


def _guard_cache(cache_root, overwrite):
    if cache_root.exists() and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite the provenance cache without --overwrite: "
            f"{cache_root}"
        )


def _write_run_manifest(root, *, stage, outputs, extra=None):
    manifest = {
        "schema": "iowa_hipp_functional_organization_v1",
        "completed_stages": stage,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {name: _output_hash(root / name) for name in outputs},
    }
    if extra:
        manifest.update(extra)
    _write_json(root / "run_manifest.json", manifest)
    return manifest


def run_analysis(output_root=DEFAULT_OUTPUT, *, workers=4, overwrite=False):
    """Derive metrics and statistics from one validated, explicit input inventory."""
    from scripts.core_workflow.lib.hippomaps_adapter import GRADIENT_EIGENSOLVER

    root = _checked_output_root(output_root)
    system = detect_system()
    if system != "idas":
        raise RuntimeError(f"This local py314 analysis is configured for IDAS, got {system}")
    thread_environment = _single_thread_environment()
    workers = int(workers)
    if workers < 1 or workers > 4:
        raise ValueError("This local IDAS analysis supports 1..4 workers")
    inventory = load_inventory(root)
    _guard_outputs(
        root,
        (
            "subject_metrics.csv",
            "curves.csv",
            "boundary_positions.csv",
            "statistics.csv",
            "roi_statistics.csv",
            "spatial_maps.json",
            "roi_registry.json",
            "analysis_config.json",
        ),
        overwrite,
    )

    participant_cohort = _participant_cohort(inventory)
    subjects = _subject_index(inventory)
    groups = _group_index(inventory)
    dominant_maps = _tm_dominant_maps(inventory, participant_cohort)
    references = _gradient_reference(inventory)
    hcp_templates = _hcp_template_index(inventory)
    registry = inventory["registry"]
    if len(registry) != 400:
        raise ValueError("Validated inventory must contain exactly 400 ROI registry rows")

    cache_root = root / "cache" / "individual"
    _guard_cache(cache_root, overwrite)
    tasks = []
    for key in sorted(subjects):
        branch, subject, hemi = key
        record = subjects[key]
        base_cohort = participant_cohort[subject]
        memberships = _analysis_memberships(base_cohort)
        atlas_labels = {}
        for membership in memberships:
            group = groups[(branch, membership, hemi)]
            de_novo = _array(group["labels"], context="de novo group labels")
            wta_raw = _array(group["wta_labels"], context="WTA group labels")
            wta_map = json.loads(
                inputs.verify_record(group["wta_map"]).read_text(encoding="utf-8")
            )
            atlas_labels[membership] = {
                "tm_dominant": dominant_maps[(branch, membership, hemi)],
                "de_novo": de_novo,
                "wta": wta_pseudoclusters(wta_raw, wta_map),
            }
        reference, reference_support = references[(branch, hemi)]
        hcp_labels, hcp_support = hcp_templates[(branch, hemi)]
        tasks.append(
            {
                "record": record,
                "branch": branch,
                "subject": subject,
                "hemi": hemi,
                "base_cohort": base_cohort,
                "atlas_labels": atlas_labels,
                "registry": registry,
                "reference": reference,
                "reference_support": reference_support,
                "hcp_labels": hcp_labels,
                "hcp_support": hcp_support,
                "cache_path": str(cache_root / branch / f"{subject}_hemi-{hemi}.npz"),
            }
        )
    if len(tasks) != len(subjects):
        raise ValueError(
            f"Expected one task per validated branch/subject/hemi inventory row ({len(subjects)}), got {len(tasks)}"
        )

    scalar_rows = []
    curve_rows = []
    boundary_rows = []
    profile_store = defaultdict(lambda: defaultdict(list))
    gradient_groups = defaultdict(list)
    outcomes = []
    if workers == 1:
        outcomes = [_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = [pool.submit(_worker, task) for task in tasks]
            for index, future in enumerate(as_completed(pending), start=1):
                outcomes.append(future.result())
                if index % 24 == 0 or index == len(pending):
                    print(f"Completed {index}/{len(pending)} subject-hemi analyses", flush=True)

    for outcome in outcomes:
        scalar_rows.extend(outcome["scalar_rows"])
        curve_rows.extend(outcome["curve_rows"])
        boundary_rows.extend(outcome["boundary_rows"])
        memberships = _analysis_memberships(outcome["base_cohort"])
        for membership, atlas, z in _roi_profile_items(outcome["roi_profiles"], memberships):
            for system_index, system_name in enumerate(metrics.SYSTEMS):
                key = (outcome["branch"], outcome["hemi"], atlas, system_name)
                profile_store[key][membership].append(z[system_index])
        for membership in memberships:
            gradient_groups[
                (outcome["branch"], outcome["hemi"], membership)
            ].append(outcome["gradient"])

    scalar_rows.extend(_cortex_area_rows(inventory))
    curve_rows_extra, spatial_maps = _group_descriptions(
        inventory, participant_cohort, dominant_maps, gradient_groups
    )
    curve_rows.extend(curve_rows_extra)

    statistics = statistics_from_subject_rows(scalar_rows)
    roi_statistics = _roi_statistics(profile_store, registry)
    _write_csv(root / "subject_metrics.csv", scalar_rows, SCALAR_FIELDS)
    _write_csv(root / "curves.csv", curve_rows, CURVE_FIELDS)
    _write_csv(root / "boundary_positions.csv", boundary_rows, BOUNDARY_FIELDS)
    _write_csv(
        root / "statistics.csv",
        statistics,
        (
            "theme", "branch", "hemi", "atlas", "system", "metric", "target",
            "comparison", "case_cohort", "control_cohort", "mean_patient",
            "sd_patient", "n_patient", "mean_hc", "sd_hc", "n_hc",
            "difference", "ci_low", "ci_high", "g", "p", "df", "reason",
            "significance",
        ),
    )
    _write_csv(
        root / "roi_statistics.csv",
        roi_statistics,
        (
            "theme", "branch", "hemi", "atlas", "system", "metric", "target",
            "target_level", "roi_id", "comparison", "case_cohort",
            "control_cohort", "mean_patient", "sd_patient", "n_patient",
            "mean_hc", "sd_hc", "n_hc", "difference", "ci_low", "ci_high",
            "g", "p", "df", "reason", "significance",
        ),
    )
    _write_json(root / "spatial_maps.json", spatial_maps)
    _write_json(root / "roi_registry.json", registry)
    _write_json(
        root / "analysis_config.json",
        {
            "scope": inventory["scope"],
            "system": system,
            "workers": workers,
            "thread_environment": thread_environment,
            "tests": {
                "primary": "two-sided Welch",
                "covariates": "none",
                "correction": "none",
                "confidence_interval": "95 percent Welch t interval",
                "effect": "Hedges g, patient minus HC",
                "comparisons": list(COMPARISONS),
            },
            "gradient": {
                "features": "individual vertex-to-400-ROI Fisher-z FC",
                "n_components": 5,
                "kernel": "pearson",
                "eigensolver": GRADIENT_EIGENSOLVER,
                "feature_zscore": "columnwise across vertices",
                "brainspace_sparsity": 0.9,
                "alignment": "unscaled, uncentered 5D orthogonal Procrustes to matching external HCP reference",
            },
            "identity_metrics": {
                "individual_tm": ["smax", "margin", "segregation", "boundary_sharpness"],
                "fixed_hcp_territory": {
                    "metrics": ["fixed_hcp_similarity", "fixed_hcp_advantage"],
                    "definition": "area-weighted HCP-template score and matched-template-minus-competitor score within the branch- and hemisphere-matched external HCP A/B/P territory",
                    "provenance": "exact HCP labels/support/prototypes are declared in input_manifest.json; subject scores must use the same prototype file",
                },
            },
            "wta_pseudoclusters": {
                "A": "VentralAttention",
                "B": "Somatomotor",
                "P": "Default",
                "other_networks": "retained in original WTA; excluded from this three-system regional extraction without renormalization",
            },
            "group_map_geometry": {
                "source": "explicit sub-1001 HippUnfold unfolded den-512 anchor",
                "reason": "group atlas topology manifests declare compatible vertex indexing; maps are descriptive only",
                "categorical_support": "TM dominant uses eligible TM source vertices; de novo and WTA each use their own manifest-declared support mask",
            },
            "cortex_area": {
                "source": str(inputs.verify_record(inventory["cortex_subjects"])),
                "selection": "Iowa only; Kong2022 canonical7; 36P; area_percent; no standard/ctxreg duplication",
            },
            "boundary_positions": {
                "atlas": "individual TM only",
                "definition": "every native-mesh edge joining two valid, different A/B/P labels; position is the unfolded AP midpoint",
                "policy": "all edge positions retained; no single boundary is selected for discontinuous or multiple transitions",
            },
        },
    )
    outputs = (
        "input_manifest.json",
        "input_audit.csv",
        "subject_metrics.csv",
        "curves.csv",
        "boundary_positions.csv",
        "statistics.csv",
        "roi_statistics.csv",
        "spatial_maps.json",
        "roi_registry.json",
        "analysis_config.json",
    )
    _write_run_manifest(
        root,
        stage=["inventory", "metrics", "statistics"],
        outputs=outputs,
        extra={
            "input_manifest": _output_hash(root / "input_manifest.json"),
            "cache": {
                "directory": str(cache_root.resolve()),
                "n_subject_hemi_files": len(tasks),
                "selection_policy": "write-only provenance cache; analysis never reads it as fallback",
            },
        },
    )
    return root

IDENTITY_RENDER_OUTPUTS = (
    "hipp_functional_identity_standard.png",
    "hipp_functional_identity_ctxreg.png",
)
RENDER_OUTPUTS = (
    *IDENTITY_RENDER_OUTPUTS,
    "hipp_spatial_organization.png",
    "hipp_network_fingerprints.png",
    "hipp_continuous_gradients.png",
    "iowa_hipp_functional_organization_report.pdf",
    "index.html",
)


def render_identity_report(output_root=DEFAULT_OUTPUT, *, overwrite=False):
    """Render only the validated Standard and Ctx-Reg identity figures."""
    root = _checked_output_root(output_root)
    _guard_outputs(root, IDENTITY_RENDER_OUTPUTS, overwrite)
    try:
        from . import report
    except ImportError:
        import report

    data = report._load(root)
    paths = report._figure_identity(root, "identity", data)
    expected = [root / name for name in IDENTITY_RENDER_OUTPUTS]
    if paths != expected or any(not path.is_file() for path in expected):
        raise RuntimeError("Identity renderer did not produce the declared Standard and Ctx-Reg figures")
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest.get("outputs")
    if not isinstance(declared, dict):
        raise ValueError(f"Malformed analysis manifest outputs: {manifest_path}")
    declared.update({name: _output_hash(root / name) for name in IDENTITY_RENDER_OUTPUTS})
    stages = list(manifest.get("completed_stages", []))
    if "identity_render" not in stages:
        stages.append("identity_render")
    manifest.update(
        completed_stages=stages,
        generated_utc=datetime.now(timezone.utc).isoformat(),
        outputs=declared,
        identity_render={"outputs": list(IDENTITY_RENDER_OUTPUTS)},
    )
    _write_json(manifest_path, manifest)
    return expected


def render_report(output_root=DEFAULT_OUTPUT, *, overwrite=False):
    """Render only from manifest-hashed derivatives, then record output hashes."""
    root = _checked_output_root(output_root)
    _guard_outputs(root, RENDER_OUTPUTS, overwrite)
    try:
        from . import report
    except ImportError:
        import report

    report.render(root)
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing analysis manifest after report rendering: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest.get("outputs")
    if not isinstance(declared, dict):
        raise ValueError(f"Malformed analysis manifest outputs: {manifest_path}")
    declared.update({name: _output_hash(root / name) for name in RENDER_OUTPUTS})
    stages = list(manifest.get("completed_stages", []))
    if "render" not in stages:
        stages.append("render")
    manifest.update(
        completed_stages=stages,
        generated_utc=datetime.now(timezone.utc).isoformat(),
        outputs=declared,
        render={
            "input_policy": "manifest-hash-validated derived statistics only",
            "outputs": list(RENDER_OUTPUTS),
        },
    )
    _write_json(manifest_path, manifest)
    return [root / name for name in RENDER_OUTPUTS]


def _parser():
    parser = argparse.ArgumentParser(
        description="Iowa hippocampal functional-organization analysis and report"
    )
    parser.add_argument(
        "--stage",
        choices=("inventory", "analyze", "identity", "render", "all"),
        default="all",
        help="Run one named stage; identity renders only the two identity PNGs, while all runs inventory → analyze → render.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT),
        help="Output root; it must remain inside this analysis directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        choices=range(1, 5),
        default=4,
        metavar="{1,2,3,4}",
        help="IDAS numerical workers; each process requires one numerical-library thread.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of this stage's existing files; never selects alternate inputs.",
    )
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    root = _checked_output_root(args.out)
    if args.stage in ("inventory", "all"):
        _guard_outputs(root, ("input_manifest.json", "input_audit.csv"), args.overwrite)
        write_inventory(root)
    if args.stage in ("analyze", "all"):
        run_analysis(root, workers=args.workers, overwrite=args.overwrite)
    if args.stage == "identity":
        render_identity_report(root, overwrite=args.overwrite)
    if args.stage in ("render", "all"):
        render_report(root, overwrite=args.overwrite)
    return root


if __name__ == "__main__":
    main()

