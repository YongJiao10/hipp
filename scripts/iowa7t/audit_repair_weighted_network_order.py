#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / 'scripts' / 'core_workflow'
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.kong2022_label_utils import canonical_families_8_order  # noqa: E402
from lib.weighted_fisherz import (  # noqa: E402
    assignment_status_from_weighted_network_profile,
)
from lib.workflow_utils import utc_now_iso  # noqa: E402

try:
    from scripts.iowa7t.run_branch_downstream import DEFAULT_OUTPUT_ROOT
except Exception:
    DEFAULT_OUTPUT_ROOT = Path('/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs')

CANONICAL8_NETWORK_ORDER: list[str] = canonical_families_8_order()
SORTED_CANONICAL8_NETWORK_ORDER: list[str] = sorted(CANONICAL8_NETWORK_ORDER)
WEIGHTING_NOTE = 'network_timeseries_weighted_by_roi_n_grayordinates_used_then_fisherz_fc'


def load_core_group_atlas_module():
    module_name = '_iowa7t_weighted_order_core_group_atlas'
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = CORE_WORKFLOW_DIR / '05_02_run_group_atlas.py'
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load core group-atlas module: {module_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path: Path, payload: object) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def validate_order(order: list[str], *, expected_len: int, label: str) -> None:
    if len(order) != expected_len:
        raise ValueError(f'{label} length {len(order)} does not match profile columns {expected_len}')
    if len(set(order)) != len(order):
        raise ValueError(f'{label} contains duplicate names: {order}')


def reorder_profile_columns(profile: np.ndarray, *, current_order: list[str], target_order: list[str]) -> np.ndarray:
    arr = np.asarray(profile, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f'Expected 2D profile, got shape={arr.shape}')
    current = [str(x) for x in current_order]
    target = [str(x) for x in target_order]
    validate_order(current, expected_len=int(arr.shape[1]), label='current_order')
    validate_order(target, expected_len=int(arr.shape[1]), label='target_order')
    if set(current) != set(target):
        raise ValueError(f'Network order sets differ: current={current} target={target}')
    col_index = {name: idx for idx, name in enumerate(current)}
    return arr[:, [col_index[name] for name in target]].astype(np.float32)


def assignment_status_from_profile(profile: np.ndarray, network_names: list[str]) -> tuple[list[dict[str, object]], list[str | None]]:
    rows, dominant = assignment_status_from_weighted_network_profile(profile, network_names)
    return rows, dominant


def infer_order_from_label_map(label_map_path: Path, n_cols: int) -> list[str] | None:
    if not label_map_path.exists():
        return None
    payload = read_json(label_map_path)
    names = payload.get('network_names')
    if isinstance(names, list) and len(names) == n_cols:
        return [str(x) for x in names]
    return None


def cluster_status_needs_rewrite(profile: np.ndarray, status_path: Path, network_names: list[str]) -> bool:
    if not status_path.exists():
        return True
    existing = read_json(status_path).get('items', [])
    if not isinstance(existing, list):
        return True
    rows, _dominant = assignment_status_from_profile(profile, network_names)
    by_cluster = {int(row.get('cluster_id', -1)): row for row in existing if isinstance(row, dict) and 'cluster_id' in row}
    for expected in rows:
        got = by_cluster.get(int(expected['cluster_id']))
        if got is None:
            return True
        if str(got.get('top1_network')) != str(expected.get('top1_network')):
            return True
        if str(got.get('top2_network')) != str(expected.get('top2_network')):
            return True
        for key in ('top1_fisher_z', 'top2_fisher_z'):
            if key in got and key in expected and abs(float(got.get(key, 0.0)) - float(expected.get(key, 0.0))) > 1e-4:
                return True
    return False


def write_gradient_label_products(run_dir: Path, labels: np.ndarray, status_rows: list[dict[str, object]]) -> None:
    core = load_core_group_atlas_module()
    color_map = core.load_canonical_network_color_map()
    functional_labels, functional_meta, label_items = core.functional_atlas_from_clusters(
        group_labels_final=labels.astype(np.int32),
        status_rows=status_rows,
        network_names=CANONICAL8_NETWORK_ORDER,
        color_map=color_map,
    )
    np.save(run_dir / 'group_functional_atlas_labels_weighted_fisherz.npy', functional_labels.astype(np.int32))
    functional_meta['assignment_variant'] = 'weighted_fisherz'
    functional_meta['network_profile_weighting'] = WEIGHTING_NOTE
    write_json(run_dir / 'group_functional_atlas_label_map_weighted_fisherz.json', functional_meta)
    for label_path in sorted((run_dir / 'workbench_assets').glob('*_weighted_fisherz.label.gii')):
        nib.save(core.make_named_label_gifti(functional_labels.astype(np.int32), label_items), str(label_path))


def repair_gradient_dir(run_dir: Path, *, repair: bool) -> dict[str, object]:
    profile_path = run_dir / 'cluster_network_profile_weighted_fisherz.npy'
    render_profile_path = run_dir / 'cluster_network_profile.npy'
    if profile_path.exists():
        profile_source = profile_path
    elif render_profile_path.exists() and (run_dir / 'cluster_assignment_status_weighted_fisherz.json').exists():
        profile_source = render_profile_path
    else:
        return {'kind': 'gradient', 'path': str(run_dir), 'status': 'skipped', 'reason': 'missing weighted profile'}
    profile = np.load(profile_source).astype(np.float32)
    current_order = infer_order_from_label_map(run_dir / 'group_functional_atlas_label_map_weighted_fisherz.json', int(profile.shape[1]))
    if current_order is None:
        current_order = SORTED_CANONICAL8_NETWORK_ORDER if int(profile.shape[1]) == len(CANONICAL8_NETWORK_ORDER) else CANONICAL8_NETWORK_ORDER
    repaired_profile = reorder_profile_columns(profile, current_order=current_order, target_order=CANONICAL8_NETWORK_ORDER)
    status_rows, dominant = assignment_status_from_profile(repaired_profile, CANONICAL8_NETWORK_ORDER)
    changed = current_order != CANONICAL8_NETWORK_ORDER or cluster_status_needs_rewrite(
        repaired_profile,
        run_dir / 'cluster_assignment_status_weighted_fisherz.json',
        CANONICAL8_NETWORK_ORDER,
    )
    result = {'kind': 'gradient', 'path': str(run_dir), 'status': 'needs_repair' if changed else 'ok', 'current_order': current_order, 'target_order': CANONICAL8_NETWORK_ORDER}
    if not changed or not repair:
        return result
    if profile_path.exists():
        np.save(profile_path, repaired_profile.astype(np.float32))
    if render_profile_path.exists():
        np.save(render_profile_path, repaired_profile.astype(np.float32))
    write_json(run_dir / 'cluster_assignment_status_weighted_fisherz.json', {'items': status_rows, 'assignment_metric': 'top1_weighted_network_fisher_z', 'network_order': 'kong2022_canonical8', 'network_profile_weighting': WEIGHTING_NOTE, 'repaired_at': utc_now_iso()})
    write_json(run_dir / 'cluster_dominant_network_weighted_fisherz.json', {'dominant': dominant, 'network_names': CANONICAL8_NETWORK_ORDER, 'assignment_variant': 'weighted_fisherz'})
    label_path = run_dir / 'cluster_labels_k3.npy'
    if not label_path.exists():
        label_path = run_dir / 'group_labels_final.npy'
    if label_path.exists():
        write_gradient_label_products(run_dir, np.load(label_path).astype(np.int32), status_rows)
    result['status'] = 'repaired'
    return result


def iter_gradient_dirs(output_root: Path):
    yield from sorted(Path(output_root).glob('subjects/sub-*/*/atlas/gradient/*/*'))
    yield from sorted(Path(output_root).glob('runs/cache/iowa7t_branch_individual_overviews/sub-*_*/*/sub-*/gradient/*/*'))
    yield from sorted(Path(output_root).glob('runs/cache/iowa7t_branch_individual_overviews/sub-*_*/overview_core_adapter/gradient_weighted_fisherz/sub-*/gradient-spectral/kong2022/*/cov_0p80/g1g2g3/hemi_*'))


def audit(output_root: Path, *, repair: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[Path] = set()
    for run_dir in iter_gradient_dirs(output_root):
        resolved = run_dir.resolve()
        if resolved in seen or not run_dir.is_dir():
            continue
        seen.add(resolved)
        try:
            row = repair_gradient_dir(run_dir, repair=repair)
        except Exception as exc:
            row = {'path': str(run_dir), 'status': 'error', 'error_type': exc.__class__.__name__, 'error': str(exc)}
        if row.get('status') != 'skipped':
            rows.append(row)
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Audit and repair Iowa7T weighted_fisherz network-order products.')
    parser.add_argument('--output-root', default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument('--repair', action='store_true', help='Rewrite affected weighted_fisherz products in canonical8 network order.')
    parser.add_argument('--report-json', default=None, help='Optional path for the audit report JSON.')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_root = Path(args.output_root)
    rows = audit(output_root, repair=bool(args.repair))
    has_open_issue = any(row.get('status') in {'needs_repair', 'error'} for row in rows)
    summary = {'status': 'needs_repair' if has_open_issue else 'ok', 'output_root': str(output_root.resolve()), 'repair': bool(args.repair), 'total_checked': len(rows), 'ok': sum(1 for row in rows if row.get('status') == 'ok'), 'needs_repair': sum(1 for row in rows if row.get('status') == 'needs_repair'), 'repaired': sum(1 for row in rows if row.get('status') == 'repaired'), 'errors': sum(1 for row in rows if row.get('status') == 'error')}
    payload = {'summary': summary, 'items': rows}
    if args.report_json:
        write_json(Path(args.report_json), payload)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary['needs_repair'] or summary['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
