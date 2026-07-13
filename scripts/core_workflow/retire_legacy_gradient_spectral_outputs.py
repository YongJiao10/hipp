#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE_ROOT = REPO_ROOT / "_archive" / "gradient_spectral_relu_cosine_20260713"
LEGACY_DESCRIPTION = "max(cosine_similarity,0)_hadamard_surface_adjacency_diag0"


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _children(path: Path) -> list[Path]:
    return sorted(path.iterdir(), key=lambda item: item.name) if path.is_dir() else []


def _load_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_legacy_group_tree(path: Path) -> bool:
    manifests = sorted(path.rglob("group_atlas_manifest.json"))
    if not manifests:
        return False
    for manifest in manifests:
        payload = _load_object(manifest)
        if payload.get("surface_affinity") == "shifted-cosine":
            return False
        algorithm = str(payload.get("clustering_algorithm", ""))
        description = str(payload.get("affinity_description", ""))
        if "relu_cosine" not in algorithm and description != LEGACY_DESCRIPTION:
            return False
    return True


def _is_legacy_crossfit(path: Path) -> bool:
    manifest = _load_object(path / "crossfit_manifest.json")
    if not manifest:
        return False
    # The retired implementation predates the surface_affinity field.  Every
    # shifted-cosine cross-fit manifest records it explicitly.
    affinity = manifest.get("surface_affinity")
    return affinity is None or affinity == "relu-cosine"


def _text_contains(path: Path, needle: str) -> bool:
    if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".tsv", ".txt", ".csv"}:
        return False
    try:
        return needle in path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _artifact_branch(name: str, dataset: str) -> str:
    lowered = name.lower()
    if "lynch2024_21net" in lowered:
        return "lynch2024_21net_nsi043_full_k3_normalized_wta"
    if "lynch2024_8net" in lowered:
        return "lynch2024_8net_nsi043_full_k3_normalized_wta"
    if "ctxdrop" in lowered:
        return "routea_ctxdrop_full_k3_fisherz"
    if "ctxreg" in lowered:
        return "routea_ctxreg_full_k3_fisherz"
    if "17net" in lowered:
        return "routea_17net_full_k3_fisherz"
    if "routeb" in lowered:
        return "routeb_full_k3_fisherz"
    if dataset == "outputs_stanford" and "symp" in lowered:
        return "routea_symp_weighted_fisherz_full_k3" if "weighted" in lowered else "routea_symp_full_k3_fisherz"
    if dataset == "outputs_stanford" and "hc" in lowered:
        return "routea_hc_weighted_fisherz_full_k3" if "weighted" in lowered else "routea_hc_full_k3_fisherz"
    if "weighted" in lowered:
        return "routea_weighted_fisherz_full_k3"
    return "routea_full_k3_fisherz"


def _text_mentions_gradient(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".tsv"}:
        return False
    try:
        return "gradient-spectral" in path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _add_gradient_only_files(candidates: set[Path], directory: Path) -> None:
    for path in sorted(directory.rglob("*")) if directory.is_dir() else []:
        lowered = path.name.lower()
        if (
            path.is_file()
            and ("gradient-spectral" in lowered or "gradient_spectral" in lowered)
            and "wta-top" not in lowered
            and "wta_top" not in lowered
            and not _text_contains(path, "wta-top")
        ):
            candidates.add(path.absolute())


def collect_candidates(repo_root: Path) -> list[Path]:
    root = Path(repo_root).resolve()
    candidates: set[Path] = set()
    legacy_branches: dict[str, set[str]] = {"outputs_hcp": set(), "outputs_stanford": set()}
    all_gradient_legacy: dict[str, bool] = {}

    for dataset in ("outputs_hcp", "outputs_stanford"):
        group_root = root / dataset / "work" / "group_atlas"
        branches = [branch for branch in _children(group_root) if branch.is_dir()]
        gradient_roots = [branch / "gradient-spectral" for branch in branches if (branch / "gradient-spectral").is_dir()]
        for branch in branches:
            gradient = branch / "gradient-spectral"
            if gradient.is_dir() and _is_legacy_group_tree(gradient):
                candidates.add(gradient.absolute())
                legacy_branches[dataset].add(branch.name)
            # Weighted branches also contain WTA products.  Select only their
            # proven gradient subtree and stale branch-level gradient metadata.
            if "weighted_fisherz" in branch.name and branch.name in legacy_branches[dataset]:
                for item in _children(branch):
                    if item == gradient:
                        continue
                    if item.is_file() and (
                        "gradient-spectral" in item.name.lower() or _text_mentions_gradient(item)
                    ):
                        candidates.add(item.absolute())

        all_gradient_legacy[dataset] = bool(gradient_roots) and all(
            gradient.parent.name in legacy_branches[dataset] for gradient in gradient_roots
        )
        if legacy_branches[dataset]:
            for log_path in _children(group_root / "logs"):
                lowered = log_path.name.lower()
                if log_path.is_file() and (
                    "gradient_spectral" in lowered or "routea_17net_overview" in lowered
                ) and _artifact_branch(log_path.name, dataset) in legacy_branches[dataset] and not _text_contains(log_path, "shifted-cosine"):
                    candidates.add(log_path.absolute())

    canonical_crossfit = root / "outputs_hcp/work/crossfit_template_matching"
    legacy_crossfit = _is_legacy_crossfit(canonical_crossfit)
    if legacy_crossfit:
        candidates.add(canonical_crossfit.absolute())
        execution_log = root / "outputs_hcp/work/crossfit_template_matching_execution.log"
        if execution_log.exists():
            candidates.add(execution_log.absolute())

    # These products are derived from the canonical legacy group/cross-fit
    # outputs.  They are eligible only after the producing source is proven.
    if "routea_full_k3_fisherz" in legacy_branches["outputs_hcp"]:
        derived = (
            "outputs_hcp/work/individual_atlas/gradient_spectral_k3",
            "outputs_hcp/present/overviews/individual/gradient",
        )
        for relative in derived:
            path = root / relative
            if path.exists():
                candidates.add(path.absolute())
        adapter = root / "outputs_hcp/work/individual_atlas/_overview_core_adapter"
        for subject in _children(adapter):
            gradient = subject / "gradient-spectral"
            if gradient.is_dir():
                candidates.add(gradient.absolute())
    if "routea_weighted_fisherz_full_k3" in legacy_branches["outputs_hcp"]:
        directory = root / "outputs_hcp/present/overviews/weighted_fisherz"
        _add_gradient_only_files(candidates, directory)
    stanford_derived = {
        "routea_hc_full_k3_fisherz": "outputs_stanford/present/overviews/routea_hc",
        "routea_symp_full_k3_fisherz": "outputs_stanford/present/overviews/routea_symp",
        "routea_hc_weighted_fisherz_full_k3": "outputs_stanford/present/overviews/routea_hc_weighted_fisherz",
        "routea_symp_weighted_fisherz_full_k3": "outputs_stanford/present/overviews/routea_symp_weighted_fisherz",
    }
    for branch, relative in stanford_derived.items():
        path = root / relative
        if branch in legacy_branches["outputs_stanford"] and path.exists():
            _add_gradient_only_files(candidates, path)
    if legacy_crossfit:
        path = root / "outputs_hcp/present/overviews/crossfit_template_matching"
        if path.exists():
            candidates.add(path.absolute())

    for dataset in ("outputs_hcp", "outputs_stanford"):
        if legacy_branches[dataset]:
            present = root / dataset / "present"
            for relative_dir in ("overviews", "cortex_panels", "subfield_composition"):
                for item in _children(present / relative_dir):
                    if item.is_file() and not _text_contains(item, "shifted-cosine") and (
                        "gradient-spectral" in item.name.lower()
                        or "gradient_spectral" in item.name.lower()
                        or _text_mentions_gradient(item)
                    ) and (
                        _artifact_branch(item.name, dataset) in legacy_branches[dataset]
                        or (all_gradient_legacy[dataset] and "manifest" in item.name.lower())
                    ):
                        candidates.add(item.absolute())
            for item in _children(present):
                if (
                    item.is_file()
                    and _text_mentions_gradient(item)
                    and not _text_contains(item, "shifted-cosine")
                    and all_gradient_legacy[dataset]
                ):
                    candidates.add(item.absolute())

    ordered = sorted(candidates, key=lambda path: (len(path.parts), path.as_posix()))
    deduplicated: list[Path] = []
    for candidate in ordered:
        if any(candidate == parent or _within(candidate, parent) for parent in deduplicated):
            continue
        deduplicated.append(candidate)
    return sorted(deduplicated, key=lambda path: path.relative_to(root).as_posix())


def _signature(path: Path) -> str:
    digest = hashlib.sha256()
    entries = [path] if path.is_file() else sorted(path.rglob("*"))
    for entry in entries:
        relative = "." if entry == path else entry.relative_to(path).as_posix()
        stat = entry.stat()
        digest.update(f"{relative}\0{entry.is_dir()}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        if entry.is_file():
            content_digest = hashlib.sha256()
            with entry.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    content_digest.update(chunk)
            digest.update(content_digest.digest())
    return digest.hexdigest()


def _plan_digest(payload: dict[str, object]) -> str:
    immutable = {
        "schema_version": payload.get("schema_version"),
        "repo_root": payload.get("repo_root"),
        "archive_root": payload.get("archive_root"),
        "items": [
            {
                key: item.get(key)
                for key in ("source", "destination", "kind", "source_signature")
            }
            for item in payload.get("items", [])
            if isinstance(item, dict)
        ],
    }
    encoded = json.dumps(immutable, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_archive_plan(repo_root: Path, archive_root: Path, candidates: list[Path]) -> dict[str, object]:
    root, archive = Path(repo_root).resolve(), Path(archive_root).resolve()
    if not _within(archive, root):
        raise ValueError(f"Archive root must resolve within repository: {archive}")
    items: list[dict[str, object]] = []
    for raw in candidates:
        source = Path(raw).absolute()
        if source.is_symlink():
            raise ValueError(f"Refusing to archive symlink source: {source}")
        resolved = source.resolve()
        if not _within(resolved, root) or _within(resolved, archive):
            raise ValueError(f"Archive source must resolve within active repository: {resolved}")
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        relative = resolved.relative_to(root)
        items.append({
            "source": relative.as_posix(),
            "destination": relative.as_posix(),
            "kind": "directory" if resolved.is_dir() else "file",
            "source_signature": _signature(resolved),
            "status": "pending",
        })
    plan: dict[str, object] = {
        "schema_version": 1,
        "status": "reviewed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(root),
        "archive_root": str(archive),
        "items": items,
    }
    plan["plan_digest"] = _plan_digest(plan)
    return plan


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def apply_archive_plan(plan_path: Path) -> dict[str, object]:
    plan_file = Path(plan_path).resolve()
    manifest = _load_object(plan_file)
    root = Path(str(manifest.get("repo_root", ""))).resolve()
    archive = Path(str(manifest.get("archive_root", ""))).resolve()
    if manifest.get("schema_version") != 1 or not _within(archive, root) or not _within(plan_file, archive):
        raise ValueError(f"Invalid archive plan: {plan_file}")
    if manifest.get("plan_digest") != _plan_digest(manifest):
        raise ValueError("Archive plan digest mismatch; the reviewed inventory was modified")
    items = manifest.get("items")
    if not isinstance(items, list):
        raise ValueError("Archive plan items must be a list")
    manifest["status"] = "in_progress"
    _write_json(plan_file, manifest)
    try:
        for raw in items:
            if not isinstance(raw, dict):
                raise ValueError("Invalid archive plan item")
            source, destination = root / str(raw["source"]), archive / str(raw["destination"])
            if not _within(source, root) or not _within(destination, archive):
                raise ValueError("Archive plan path escaped its approved root")
            if source.exists() and destination.exists():
                raise FileExistsError(f"Archive destination collision: {destination}")
            if destination.exists():
                if _signature(destination) != raw["source_signature"]:
                    raise ValueError(f"Resumed destination signature mismatch: {destination}")
                raw["status"] = "moved"
                _write_json(plan_file, manifest)
                continue
            if not source.exists() or source.is_symlink():
                raise FileNotFoundError(f"Approved source missing or became a symlink: {source}")
            if _signature(source) != raw["source_signature"]:
                raise ValueError(f"Approved source changed after review: {source}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            if _signature(destination) != raw["source_signature"]:
                raise RuntimeError(f"Archive move verification failed: {source} -> {destination}")
            raw["status"] = "moved"
            _write_json(plan_file, manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(plan_file, manifest)
        raise
    manifest["status"] = "ok"
    manifest.pop("error", None)
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(plan_file, manifest)
    return manifest


def archive_candidates(repo_root: Path, archive_root: Path, candidates: list[Path]) -> dict[str, object]:
    """Compatibility helper for tests; production CLI always applies a persisted plan."""
    archive = Path(archive_root).resolve()
    plan_path = archive / "archive_manifest.json"
    _write_json(plan_path, build_archive_plan(repo_root, archive, candidates))
    return apply_archive_plan(plan_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write-plan", type=Path)
    mode.add_argument("--apply-plan", type=Path)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.apply_plan:
        print(json.dumps(apply_archive_plan(args.apply_plan), indent=2, sort_keys=True))
        return 0
    root, archive = Path(args.repo_root).resolve(), Path(args.archive_root).resolve()
    plan = build_archive_plan(root, archive, collect_candidates(root))
    if args.write_plan:
        requested = Path(args.write_plan).resolve()
        if not _within(requested, archive):
            raise ValueError("Reviewed plan must be written inside the approved archive root")
        _write_json(requested, plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
