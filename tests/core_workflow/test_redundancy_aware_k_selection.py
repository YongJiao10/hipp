#!/usr/bin/env python3
from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _load_numbered_module(script_name: str, alias: str):
    module_path = REPO_ROOT / "scripts" / "core_workflow" / script_name
    spec = importlib.util.spec_from_file_location(alias, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    core_dir = (REPO_ROOT / "scripts" / "core_workflow").resolve()
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


atlas_mod = _load_numbered_module("05_02_run_group_atlas.py", "core_05_02_run_group_atlas")

PROFILE_REDUNDANCY_MAX_SIM = atlas_mod.PROFILE_REDUNDANCY_MAX_SIM
profile_redundancy_metrics = atlas_mod.profile_redundancy_metrics
redundancy_early_stop_triggered = atlas_mod.redundancy_early_stop_triggered
resolve_k_eval_jobs = atlas_mod.resolve_k_eval_jobs
select_k_final_after_redundancy_filter = atlas_mod.select_k_final_after_redundancy_filter


def _redundancy_aware_selected(row: dict[str, int]) -> bool:
    return bool(
        int(row["local_max"]) == 1
        and int(row["within_1se_best"]) == 1
        and int(row["selected_nontrivial"]) == 1
        and int(row["nonredundant_ok"]) == 1
    )


def test_redundancy_detects_duplicate_profiles() -> None:
    profile = np.array(
        [
            [1.0, 0.2, -0.1, 0.0],
            [0.98, 0.18, -0.12, 0.01],
            [-0.3, 0.4, 0.5, -0.2],
        ],
        dtype=np.float32,
    )
    max_sim, redundant_pairs = profile_redundancy_metrics(profile)
    assert max_sim > float(PROFILE_REDUNDANCY_MAX_SIM), (max_sim, redundant_pairs)
    assert redundant_pairs >= 1, (max_sim, redundant_pairs)


def test_redundancy_passes_for_distinct_profiles() -> None:
    profile = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    max_sim, redundant_pairs = profile_redundancy_metrics(profile)
    assert max_sim <= float(PROFILE_REDUNDANCY_MAX_SIM), (max_sim, redundant_pairs)
    assert redundant_pairs == 0, (max_sim, redundant_pairs)


def test_redundant_pair_count_uses_requested_threshold() -> None:
    first = np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0)
    orthogonal = np.array([1.0, -2.0, 1.0]) / np.sqrt(6.0)
    second = 0.85 * first + np.sqrt(1.0 - 0.85**2) * orthogonal
    profile = np.stack([first, second])
    max_sim, at_formal_threshold = profile_redundancy_metrics(profile, threshold=0.90)
    _, at_previous_threshold = profile_redundancy_metrics(profile, threshold=0.80)
    assert np.isclose(max_sim, 0.85)
    assert at_formal_threshold == 0
    assert at_previous_threshold == 1


def test_primary_k_diagnostics_come_from_displayed_partition() -> None:
    rows = [
        {"k": 2, "selected_final": 1, "gradient_clustering": {"k": 2}},
        {"k": 3, "selected_final": 0, "gradient_clustering": {"k": 3}},
    ]
    row = atlas_mod.primary_k_row(rows, 3)
    assert row["gradient_clustering"]["k"] == 3


def test_redundancy_aware_selected_requires_all_four_flags() -> None:
    base = {
        "local_max": 1,
        "within_1se_best": 1,
        "selected_nontrivial": 1,
        "nonredundant_ok": 1,
    }
    assert _redundancy_aware_selected(base) is True
    for key in sorted(base.keys()):
        row = dict(base)
        row[key] = 0
        assert _redundancy_aware_selected(row) is False, key


def test_redundancy_filter_selects_from_remaining_pool_not_redundant_future_k() -> None:
    """If higher-K candidates are redundant, pick from the remaining fair pool.

    Mirrors the failure mode the user pointed out: K=2/K=3 should be compared
    against the nonredundant remainder, not rejected in anticipation of later K
    values that are ultimately ineligible.
    """
    pool = [
        {"k": 2, "group_ari_mean": 0.88, "group_ari_se": 0.01},
        {"k": 3, "group_ari_mean": 0.91, "group_ari_se": 0.01},
        {"k": 4, "group_ari_mean": 0.95, "group_ari_se": 0.01},
    ]
    k, path, within_1se_k_set, survivor_argmax_k = select_k_final_after_redundancy_filter(pool[:2])
    assert survivor_argmax_k == 3, survivor_argmax_k
    assert within_1se_k_set == {3}, within_1se_k_set
    assert k == 3, f"expected K=3 after removing redundant higher-K candidates, got K={k}"
    assert path == "local_max_within_1se_after_redundancy_filter", path


def test_resolve_k_eval_jobs_caps_by_requested_candidates_and_repo_limit() -> None:
    assert resolve_k_eval_jobs(1, 5) == 1
    assert resolve_k_eval_jobs(4, 2) == 2
    assert resolve_k_eval_jobs(99, 20) == 10


def test_redundancy_early_stop_trigger_only_on_redundant_rows() -> None:
    assert redundancy_early_stop_triggered({"nonredundant_ok": 0}, enabled=True) is True
    assert redundancy_early_stop_triggered({"nonredundant_ok": 1}, enabled=True) is False
    assert redundancy_early_stop_triggered({"nonredundant_ok": 0}, enabled=False) is False


def test_resampling_seed_schedules_and_half_fit_offsets() -> None:
    base = 20260420
    assert atlas_mod.split_seed_for_index(0, base) == 20260421
    assert atlas_mod.split_seed_for_index(1, base) == 20260420 + 1 + 13
    assert atlas_mod.split_seed_for_index(19, base) == 20260420 + 1 + 13 * 19
    assert atlas_mod.null_seed_for_index(0, base) == 20260420 + 5000
    assert atlas_mod.null_seed_for_index(1, base) == 20260420 + 5000 + 17
    assert atlas_mod.null_seed_for_index(19, base) == 20260420 + 5000 + 17 * 19
    assert atlas_mod.FORMAL_RESAMPLE_HALF_SEED_OFFSETS == (0, 1)
    assert atlas_mod.half_fit_seeds(20260421, atlas_mod.FORMAL_RESAMPLE_HALF_SEED_OFFSETS) == (20260421, 20260422)


def test_half_group_fc_averages_already_fisher_z_features_once() -> None:
    feature_bank = {
        "a": np.array([[0.20, -0.10], [0.30, 0.40]], dtype=np.float32),
        "b": np.array([[0.40, 0.10], [0.10, 0.20]], dtype=np.float32),
    }
    support = np.array([True, False])
    actual = atlas_mod.compute_half_group_fc(feature_bank, ["a", "b"], support)
    expected = np.array([[0.30, 0.00]], dtype=np.float32)
    assert np.allclose(actual, expected)


def test_compute_resample_and_null_statistics_with_failures() -> None:
    import math

    records = [
        atlas_mod.RepetitionOutcome(i, "split", 1000 + i, 3, 0.60, "ok", "", 100)
        for i in range(18)
    ]
    records.append(atlas_mod.RepetitionOutcome(18, "split", 1018, 3, math.nan, "failed", "graph disconnected", 0))
    records.append(atlas_mod.RepetitionOutcome(19, "split", 1019, 3, math.nan, "failed", "gmm non-convergence", 0))
    stats = atlas_mod.compute_resample_statistics(records, requested=20)
    assert stats["subject_split_requested"] == 20
    assert stats["subject_split_used"] == 18
    assert stats["subject_split_failed"] == 2
    assert math.isclose(float(stats["group_ari_mean"]), 0.60, abs_tol=1e-5)

    null_records = [
        atlas_mod.RepetitionOutcome(i, "null", 2000 + i, 3, 0.05, "ok", "", 100)
        for i in range(20)
    ]
    null_stats = atlas_mod.compute_null_statistics(null_records, requested=20)
    assert null_stats["B_null_requested"] == 20
    assert null_stats["B_null_used_strong"] == 20
    assert null_stats["null_failed"] == 0


def test_failed_left_reference_half_propagates_to_target_half() -> None:
    subjects = ["s1", "s2", "s3", "s4"]
    fb = {s: np.ones((4, 6), dtype=np.float32) for s in subjects}
    vm = {s: np.ones(4, dtype=bool) for s in subjects}
    ref_pair = (
        atlas_mod.FittedHalf(half_index=0, seed=20260421, reason="Diffusion affinity is disconnected"),
        atlas_mod.FittedHalf(half_index=1, seed=20260422, reason="Diffusion affinity is disconnected"),
    )
    _, ha, hb = atlas_mod._prepare_one_split_repetition(
        0,
        subjects,
        fb,
        vm,
        0.8,
        20260420,
        "none",
        "affinity",
        0.9,
        reference_full=None,
        half_fit_seed_offsets=atlas_mod.FORMAL_RESAMPLE_HALF_SEED_OFFSETS,
        reference_pair=ref_pair,
    )
    assert ha.reason == "Left reference failed: Diffusion affinity is disconnected"
    assert hb.reason == "Left reference failed: Diffusion affinity is disconnected"


def main() -> int:
    test_redundancy_detects_duplicate_profiles()
    test_redundancy_passes_for_distinct_profiles()
    test_redundancy_aware_selected_requires_all_four_flags()
    test_redundancy_filter_selects_from_remaining_pool_not_redundant_future_k()
    test_resolve_k_eval_jobs_caps_by_requested_candidates_and_repo_limit()
    test_redundancy_early_stop_trigger_only_on_redundant_rows()
    print("ok: redundancy-aware synthetic checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
