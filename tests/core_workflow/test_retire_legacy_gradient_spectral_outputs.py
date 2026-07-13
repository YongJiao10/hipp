from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "core_workflow" / "retire_legacy_gradient_spectral_outputs.py"


def load_module():
    spec = importlib.util.spec_from_file_location("retire_legacy_gradient_spectral_outputs", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def touch(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def legacy_manifest(path: Path) -> None:
    touch(
        path / "group_atlas_manifest.json",
        json.dumps(
            {
                "affinity_description": "max(cosine_similarity,0)_hadamard_surface_adjacency_diag0",
                "clustering_algorithm": "spectral_clustering_precomputed_affinity_relu_cosine_diag0",
            }
        ),
    )


def test_collect_candidates_selects_only_legacy_gradient_derived_outputs(tmp_path: Path) -> None:
    module = load_module()
    legacy_manifest(tmp_path / "outputs_hcp/work/group_atlas/routea_full_k3_fisherz/gradient-spectral")
    touch(tmp_path / "outputs_hcp/work/group_atlas/routea_full_k3_fisherz/wta-top/a.npy")
    legacy_manifest(tmp_path / "outputs_hcp/work/group_atlas/routea_weighted_fisherz_full_k3/gradient-spectral")
    touch(tmp_path / "outputs_hcp/work/group_atlas/routea_weighted_fisherz_full_k3/wta-top/keep.npy")
    touch(tmp_path / "outputs_hcp/work/hippomaps_gradients/routea/2mm/hemi_L/group_gradients.npy")
    touch(tmp_path / "outputs_hcp/work/crossfit_template_matching/crossfit_manifest.json", '{"status":"ok"}')
    touch(tmp_path / "outputs_hcp/work/group_atlas/logs/routea_gradient_spectral_old.log")
    touch(tmp_path / "outputs_hcp/work/group_atlas/logs/unrelated_wta.log")
    touch(tmp_path / "outputs_hcp/work/individual_atlas/gradient_spectral_k3/manifest.json")
    touch(tmp_path / "outputs_hcp/work/individual_atlas/wta_top/manifest.json")
    touch(tmp_path / "outputs_hcp/present/overviews/kong2022_gradient-spectral_group_overview_routea_fisherz.png")
    touch(tmp_path / "outputs_hcp/present/overviews/kong2022_wta-top_group_overview_routea.png")
    touch(tmp_path / "outputs_hcp/present/gradients/routea/g1.png")
    touch(tmp_path / "outputs_hcp/reruns/shifted_cosine_crossfit_20260711/group_atlas/new.npy")
    legacy_manifest(tmp_path / "outputs_stanford/work/group_atlas/routea_hc_full_k3_fisherz/gradient-spectral")

    got = {path.relative_to(tmp_path).as_posix() for path in module.collect_candidates(tmp_path)}

    assert "outputs_hcp/work/group_atlas/routea_full_k3_fisherz/gradient-spectral" in got
    assert "outputs_hcp/work/group_atlas/routea_weighted_fisherz_full_k3/gradient-spectral" in got
    assert "outputs_hcp/work/crossfit_template_matching" in got
    assert "outputs_hcp/work/group_atlas/logs/routea_gradient_spectral_old.log" in got
    assert "outputs_hcp/work/individual_atlas/gradient_spectral_k3" in got
    assert "outputs_hcp/present/overviews/kong2022_gradient-spectral_group_overview_routea_fisherz.png" in got
    assert "outputs_stanford/work/group_atlas/routea_hc_full_k3_fisherz/gradient-spectral" in got
    assert not any("wta-top" in path or "wta_top" in path for path in got)
    assert not any("hippomaps_gradients" in path or "/present/gradients/" in f"/{path}/" for path in got)
    assert not any("reruns" in path for path in got)


def test_collect_candidates_rejects_shifted_and_unproven_gradient_trees(tmp_path: Path) -> None:
    module = load_module()
    shifted = tmp_path / "outputs_hcp/work/group_atlas/shifted/gradient-spectral"
    touch(shifted / "group_atlas_manifest.json", '{"surface_affinity":"shifted-cosine"}')
    touch(tmp_path / "outputs_hcp/work/group_atlas/unproven/gradient-spectral/a.npy")
    touch(
        tmp_path / "outputs_hcp/work/crossfit_template_matching/crossfit_manifest.json",
        '{"status":"ok","surface_affinity":"shifted-cosine"}',
    )

    assert module.collect_candidates(tmp_path) == []


def test_artifact_branch_prefers_discovered_taxonomy_mismatch() -> None:
    module = load_module()
    discovered = {"obsolete_taxonomy_nsi043_full_k3_normalized_wta"}

    assert module._artifact_branch(
        "obsolete_taxonomy_gradient_spectral.log",
        "outputs_hcp",
        discovered,
    ) == "obsolete_taxonomy_nsi043_full_k3_normalized_wta"


def test_collect_candidates_accepts_explicit_relu_crossfit(tmp_path: Path) -> None:
    module = load_module()
    crossfit = tmp_path / "outputs_hcp/work/crossfit_template_matching"
    touch(crossfit / "crossfit_manifest.json", '{"surface_affinity":"relu-cosine"}')

    assert module.collect_candidates(tmp_path) == [crossfit]


def test_hcp_provenance_cannot_select_stanford_shifted_products(tmp_path: Path) -> None:
    module = load_module()
    legacy_manifest(tmp_path / "outputs_hcp/work/group_atlas/routea_full_k3_fisherz/gradient-spectral")
    shifted = tmp_path / "outputs_stanford/work/group_atlas/routea_full_k3_fisherz/gradient-spectral"
    touch(shifted / "group_atlas_manifest.json", '{"surface_affinity":"shifted-cosine"}')
    touch(tmp_path / "outputs_stanford/present/overviews/kong2022_gradient-spectral_group_overview_g1g2g3_routea_fisherz.png")
    touch(tmp_path / "outputs_stanford/work/group_atlas/logs/routea_gradient_spectral.log")

    got = {path.relative_to(tmp_path).as_posix() for path in module.collect_candidates(tmp_path)}
    assert got == {"outputs_hcp/work/group_atlas/routea_full_k3_fisherz/gradient-spectral"}


def test_base_routea_legacy_cannot_select_shifted_weighted_or_wta_adapter(tmp_path: Path) -> None:
    module = load_module()
    legacy_manifest(tmp_path / "outputs_hcp/work/group_atlas/routea_full_k3_fisherz/gradient-spectral")
    weighted = tmp_path / "outputs_hcp/work/group_atlas/routea_weighted_fisherz_full_k3/gradient-spectral"
    touch(weighted / "group_atlas_manifest.json", '{"surface_affinity":"shifted-cosine"}')
    touch(tmp_path / "outputs_hcp/present/overviews/weighted_fisherz/new.png")
    touch(tmp_path / "outputs_hcp/work/individual_atlas/_overview_core_adapter/sub-1/gradient-spectral/old.npy")
    touch(tmp_path / "outputs_hcp/work/individual_atlas/_overview_core_adapter/sub-1/wta-top/keep.npy")
    touch(tmp_path / "outputs_hcp/present/overviews/individual/overview_manifest.json", "gradient-spectral wta-top")

    got = {path.relative_to(tmp_path).as_posix() for path in module.collect_candidates(tmp_path)}
    assert "outputs_hcp/present/overviews/weighted_fisherz" not in got
    assert "outputs_hcp/work/individual_atlas/_overview_core_adapter/sub-1/gradient-spectral" in got
    assert not any("wta-top" in path or path.endswith("overview_manifest.json") for path in got)


def test_legacy_weighted_selects_only_gradient_overview_files(tmp_path: Path) -> None:
    module = load_module()
    legacy_manifest(tmp_path / "outputs_hcp/work/group_atlas/routea_weighted_fisherz_full_k3/gradient-spectral")
    directory = tmp_path / "outputs_hcp/present/overviews/weighted_fisherz"
    gradient = directory / "kong_gradient-spectral_weighted.png"
    wta = directory / "kong_wta-top_weighted.png"
    mixed = directory / "overview_manifest.json"
    touch(gradient)
    touch(wta)
    touch(mixed, "gradient-spectral and wta-top")

    got = {path.relative_to(tmp_path).as_posix() for path in module.collect_candidates(tmp_path)}
    assert gradient.relative_to(tmp_path).as_posix() in got
    assert wta.relative_to(tmp_path).as_posix() not in got
    assert mixed.relative_to(tmp_path).as_posix() not in got


def test_archive_candidates_preserves_relative_paths_and_writes_manifest(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "outputs_hcp/work/group_atlas/routea_full_k3_fisherz/gradient-spectral"
    keep = tmp_path / "outputs_hcp/work/group_atlas/routea_full_k3_fisherz/wta-top/keep.npy"
    legacy_manifest(source)
    touch(keep)
    archive_root = tmp_path / "_archive/gradient_spectral_relu_cosine_20260713"

    manifest = module.archive_candidates(tmp_path, archive_root, module.collect_candidates(tmp_path))

    archived = archive_root / source.relative_to(tmp_path)
    assert not source.exists()
    assert keep.exists()
    assert (archived / "group_atlas_manifest.json").exists()
    assert manifest["status"] == "ok"
    assert manifest["items"][0]["source"] == source.relative_to(tmp_path).as_posix()
    on_disk = json.loads((archive_root / "archive_manifest.json").read_text(encoding="utf-8"))
    assert on_disk["status"] == "ok"


def test_apply_plan_rejects_source_changed_after_review(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "outputs_hcp/work/group_atlas/routea/gradient-spectral"
    legacy_manifest(source)
    archive = tmp_path / "_archive/gradient_spectral_relu_cosine_20260713"
    plan_path = archive / "reviewed_plan.json"
    module._write_json(plan_path, module.build_archive_plan(tmp_path, archive, [source]))
    touch(source / "new-after-review.txt")

    try:
        module.apply_archive_plan(plan_path)
    except ValueError as exc:
        assert "changed after review" in str(exc)
    else:
        raise AssertionError("apply must reject a changed reviewed source")


def test_apply_plan_rejects_tampered_inventory_and_binary_content(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "outputs_hcp/work/group_atlas/routea/gradient-spectral"
    legacy_manifest(source)
    binary = source / "labels.npy"
    binary.write_bytes(b"old-binary")
    archive = tmp_path / "_archive/gradient_spectral_relu_cosine_20260713"
    plan_path = archive / "reviewed_plan.json"
    plan = module.build_archive_plan(tmp_path, archive, [source])
    module._write_json(plan_path, plan)
    tampered = json.loads(plan_path.read_text(encoding="utf-8"))
    tampered["items"][0]["destination"] = "outputs_hcp/work/elsewhere"
    module._write_json(plan_path, tampered)
    try:
        module.apply_archive_plan(plan_path)
    except ValueError as exc:
        assert "digest" in str(exc).lower()
    else:
        raise AssertionError("apply must reject a modified reviewed inventory")

    module._write_json(plan_path, plan)
    original_stat = binary.stat()
    binary.write_bytes(b"new-binary")
    binary.touch()
    # Restore mtime to prove that content, not timestamps, is protected.
    import os
    os.utime(binary, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    try:
        module.apply_archive_plan(plan_path)
    except ValueError as exc:
        assert "changed after review" in str(exc)
    else:
        raise AssertionError("apply must hash binary content")


def test_archive_rejects_destination_collision(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "outputs_hcp/work/crossfit_template_matching"
    touch(source / "crossfit_manifest.json")
    archive_root = tmp_path / "_archive/gradient_spectral_relu_cosine_20260713"
    touch(archive_root / source.relative_to(tmp_path) / "collision")

    try:
        module.archive_candidates(tmp_path, archive_root, [source])
    except FileExistsError as exc:
        assert "collision" in str(exc).lower()
    else:
        raise AssertionError("archive must reject destination collisions")


def test_archive_rejects_symlink_source_even_when_target_is_inside_repo(tmp_path: Path) -> None:
    module = load_module()
    target = tmp_path / "outputs_hcp/work/real-gradient"
    touch(target / "manifest.json")
    source = tmp_path / "outputs_hcp/work/group_atlas/routea/gradient-spectral"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.symlink_to(target, target_is_directory=True)

    try:
        module.archive_candidates(
            tmp_path,
            tmp_path / "_archive/gradient_spectral_relu_cosine_20260713",
            [source],
        )
    except ValueError as exc:
        assert "symlink" in str(exc).lower()
    else:
        raise AssertionError("archive must reject symlink sources")
