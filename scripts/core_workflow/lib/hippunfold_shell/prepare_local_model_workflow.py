#!/home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2/bin/python
"""Create a validated local-model-only HippUnfold workflow overlay.

The helper copies the installed workflow and resources into retained per-run
scratch, removes the stock remote-retrieval rule, binds ``get_model_tar()`` to
the canonical local tar, retains ``unpack_nnunet_model``, and records hashes
for strict acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


REMOTE_RULE_HEADER = "rule download_nnunet_model:\n"
NEXT_DEFINITION = "\n\ndef get_cmd_copy_inputs"
REQUIRED_LOCAL_RULE = "rule unpack_nnunet_model:"
MODEL_FUNCTION_HEADER = "def get_model_tar():\n"
MODEL_FUNCTION_END = "\n\ndef get_model_dir():"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _remove_remote_rule(text: str) -> str:
    if text.count(REMOTE_RULE_HEADER) != 1:
        raise ValueError(
            "Expected exactly one download_nnunet_model rule in installed nnunet.smk"
        )
    start = text.index(REMOTE_RULE_HEADER)
    end = text.find(NEXT_DEFINITION, start)
    if end < 0:
        raise ValueError(
            "Could not locate get_cmd_copy_inputs after download_nnunet_model"
        )
    patched = text[:start] + text[end + 2 :]
    if "download_nnunet_model" in patched:
        raise ValueError("Remote nnUNet model rule remains after patching")
    if REQUIRED_LOCAL_RULE not in patched:
        raise ValueError("Local unpack_nnunet_model rule was removed unexpectedly")
    return patched


def _bind_local_model(text: str, model_path: Path) -> str:
    """Make every nnUNet consumer depend on the validated local tar path."""

    if text.count(MODEL_FUNCTION_HEADER) != 1:
        raise ValueError("Expected exactly one get_model_tar function")
    start = text.index(MODEL_FUNCTION_HEADER)
    end = text.find(MODEL_FUNCTION_END, start)
    if end < 0:
        raise ValueError("Could not locate get_model_dir after get_model_tar")
    replacement = (
        "def get_model_tar():\n"
        f"    return {json.dumps(str(model_path))}"
    )
    return text[:start] + replacement + text[end:]


def prepare_local_model_workflow(
    source_root: Path,
    overlay_root: Path,
    model_path: Path,
) -> Path:
    """Copy and validate a local-model-only workflow overlay; return Snakefile."""

    source_root = source_root.resolve()
    overlay_root = overlay_root.resolve()
    model_path = model_path.resolve()
    source_workflow = source_root / "workflow"
    source_resources = source_root / "resources"
    source_nnunet = source_workflow / "rules" / "nnunet.smk"
    source_snakefile = source_workflow / "Snakefile"

    required = (source_snakefile, source_nnunet, source_resources, model_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required local workflow assets are missing: {missing}")
    if overlay_root == source_root or source_root in overlay_root.parents:
        raise ValueError("Workflow overlay must not overwrite the installed package")

    destination_workflow = overlay_root / "workflow"
    destination_resources = overlay_root / "resources"
    overlay_root.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(
        source_workflow,
        destination_workflow,
        dirs_exist_ok=True,
        symlinks=True,
        ignore=ignore,
    )
    shutil.copytree(
        source_resources,
        destination_resources,
        dirs_exist_ok=True,
        symlinks=True,
        ignore=ignore,
    )

    destination_nnunet = destination_workflow / "rules" / "nnunet.smk"
    source_text = source_nnunet.read_text(encoding="utf-8")
    patched = _bind_local_model(
        _remove_remote_rule(source_text), model_path
    )
    destination_nnunet.write_text(patched, encoding="utf-8")

    snakefile = destination_workflow / "Snakefile"
    manifest = {
        "schema_version": 1,
        "status": "ready",
        "source_root": str(source_root),
        "source_nnunet_sha256": _sha256(source_nnunet),
        "overlay_nnunet_sha256": _sha256(destination_nnunet),
        "model_path": str(model_path),
        "snakefile": str(snakefile),
        "remote_rule": "removed from overlay",
        "local_unpack_rule": "retained",
        "model_binding": "explicit get_model_tar return",
    }
    (overlay_root / "local_model_overlay.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return snakefile


def _installed_hippunfold_root() -> Path:
    import hippunfold

    return Path(hippunfold.__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a full HippUnfold workflow that consumes only the local model tar."
    )
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args(argv)
    try:
        snakefile = prepare_local_model_workflow(
            args.source_root or _installed_hippunfold_root(),
            args.overlay_root,
            args.model_path,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"prepare_local_model_workflow.py: ERROR: {exc}", file=sys.stderr)
        return 2
    print(snakefile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
