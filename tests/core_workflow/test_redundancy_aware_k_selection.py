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
select_k_final_from_survivors = atlas_mod.select_k_final_from_survivors


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


def _row(k: int, mean: float, se: float, local_max: int) -> dict:
    return {
        "k": k,
        "group_ari_mean": mean,
        "group_ari_se": se,
        "local_max": local_max,
        "selected_final": 1,
    }


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


def test_k_final_prefers_local_max_peak_over_smaller_k() -> None:
    """K=3 should win when K=3 is the local-max peak and K=2 is on a rising slope.

    Mirrors the real 2mm hemi-R failure: K=2 mean=0.9026 (no local max),
    K=3 mean=0.9134 (local max). Smallest-K rule would wrongly pick K=2.
    """
    survivors = [
        _row(k=2, mean=0.9026, se=0.0131, local_max=0),
        _row(k=3, mean=0.9134, se=0.0055, local_max=1),
    ]
    k, path = select_k_final_from_survivors(survivors)
    assert k == 3, f"expected K=3 (local-max peak), got K={k}"
    assert path == "local_max_within_1se", path


def test_k_final_picks_lower_k_when_it_is_the_peak() -> None:
    """K=2 should win when K=2 is the local-max peak and K=3 falls below 1-SE band.

    Mirrors 2mm hemi-L: K=2 mean=0.9590 (local max), K=3 mean=0.9223 (below 1-SE).
    """
    survivors = [
        _row(k=2, mean=0.9590, se=0.0054, local_max=1),
        _row(k=3, mean=0.9223, se=0.0113, local_max=0),
    ]
    k, path = select_k_final_from_survivors(survivors)
    assert k == 2, f"expected K=2 (local-max peak), got K={k}"
    assert path == "local_max_within_1se", path


def test_k_final_parsimony_breaks_true_tie() -> None:
    """When two K values share the same local-max ARI within 1-SE, pick the smaller."""
    survivors = [
        _row(k=2, mean=0.90, se=0.01, local_max=1),
        _row(k=3, mean=0.90, se=0.01, local_max=1),
    ]
    k, path = select_k_final_from_survivors(survivors)
    assert k == 2, f"expected parsimony tiebreak K=2, got K={k}"
    assert path == "local_max_within_1se", path


def test_k_final_within_1se_fallback_when_no_local_max() -> None:
    """Falls back to smallest within-1SE when no survivor has local_max=1."""
    survivors = [
        _row(k=2, mean=0.91, se=0.01, local_max=0),
        _row(k=3, mean=0.92, se=0.01, local_max=0),
    ]
    k, path = select_k_final_from_survivors(survivors)
    # cutoff = 0.92 - 0.01 = 0.91; K=2 (0.91 >= 0.91) and K=3 both in 1-SE → smallest = K=2
    assert k == 2, f"expected K=2 (within-1SE smallest), got K={k}"
    assert path == "within_1se_smallest", path


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


def main() -> int:
    test_redundancy_detects_duplicate_profiles()
    test_redundancy_passes_for_distinct_profiles()
    test_redundancy_aware_selected_requires_all_four_flags()
    test_k_final_prefers_local_max_peak_over_smaller_k()
    test_k_final_picks_lower_k_when_it_is_the_peak()
    test_k_final_parsimony_breaks_true_tie()
    test_k_final_within_1se_fallback_when_no_local_max()
    test_redundancy_filter_selects_from_remaining_pool_not_redundant_future_k()
    test_resolve_k_eval_jobs_caps_by_requested_candidates_and_repo_limit()
    test_redundancy_early_stop_trigger_only_on_redundant_rows()
    print("ok: redundancy-aware synthetic checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
