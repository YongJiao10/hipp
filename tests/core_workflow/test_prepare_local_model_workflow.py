"""Regression tests for the local-model-only HippUnfold workflow overlay."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = (
    REPO_ROOT
    / "scripts/core_workflow/lib/hippunfold_shell/prepare_local_model_workflow.py"
)


def _load_helper():
    spec = importlib.util.spec_from_file_location(
        "prepare_local_model_workflow", HELPER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_overlay_removes_remote_rule_but_retains_local_unpack(
    tmp_path: Path,
) -> None:
    module = _load_helper()
    source = tmp_path / "installed"
    nnunet = source / "workflow/rules/nnunet.smk"
    nnunet.parent.mkdir(parents=True)
    nnunet.write_text(
        "def get_model_tar():\n    return 'model.tar'\n\n"
        "def get_model_dir():\n    return get_model_tar()\n\n"
        "rule download_nnunet_model:\n"
        "    input:\n        url='remote'\n"
        "    output:\n        model_tar='model.tar'\n"
        "    shell:\n        'cp {input} {output}'\n\n"
        "def get_cmd_copy_inputs(wildcards, input):\n    return 'cp'\n\n"
        "rule unpack_nnunet_model:\n"
        "    input:\n        tar=get_model_tar()\n",
        encoding="utf-8",
    )
    (source / "workflow/Snakefile").write_text(
        "include: 'rules/nnunet.smk'\n", encoding="utf-8"
    )
    (source / "resources/etc").mkdir(parents=True)
    (source / "resources/etc/identity.txt").write_text(
        "identity\n", encoding="utf-8"
    )
    model = tmp_path / "model.tar"
    model.write_text("model\n", encoding="utf-8")

    snakefile = module.prepare_local_model_workflow(
        source, tmp_path / "overlay", model
    )

    patched = (snakefile.parent / "rules/nnunet.smk").read_text(
        encoding="utf-8"
    )
    manifest = json.loads(
        (snakefile.parent.parent / "local_model_overlay.json").read_text(
            encoding="utf-8"
        )
    )
    assert "rule download_nnunet_model:" not in patched
    assert "rule unpack_nnunet_model:" in patched
    assert str(model.resolve()) in patched
    assert manifest["remote_rule"] == "removed from overlay"
    assert manifest["local_unpack_rule"] == "retained"
    assert manifest["model_path"] == str(model.resolve())


def test_overlay_rejects_an_unexpected_upstream_rule_layout(tmp_path: Path) -> None:
    module = _load_helper()

    try:
        module._remove_remote_rule("rule unpack_nnunet_model:\n")
    except ValueError as exc:
        assert "exactly one download_nnunet_model" in str(exc)
    else:
        raise AssertionError("Unexpected upstream nnunet.smk layout was accepted")
