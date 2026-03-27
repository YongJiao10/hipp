#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


PREDICT_PATCH_MARKER = "HIPPOMAPS_MACOS_COMPAT_PREDICT"
PREDICT_SIMPLE_PATCH_MARKER = "HIPPOMAPS_MACOS_COMPAT_PREDICT_SIMPLE"


def patch_predict(py_path: Path) -> bool:
    text = py_path.read_text(encoding="utf-8")
    if PREDICT_PATCH_MARKER in text:
        return False

    if "import argparse\n" not in text:
        raise RuntimeError(f"Did not find expected import block in {py_path}")
    text = text.replace(
        "import argparse\n",
        "import argparse\nimport os\nimport platform\n",
        1,
    )

    needle = """def preprocess_multithreaded(trainer, list_of_lists, output_files, num_processes=2, segs_from_prev_stage=None):
    if segs_from_prev_stage is None:
        segs_from_prev_stage = [None] * len(list_of_lists)

    num_processes = min(len(list_of_lists), num_processes)

    classes = list(range(1, trainer.num_classes))
    assert isinstance(trainer, nnUNetTrainer)
    q = Queue(1)
"""
    replacement = f"""def preprocess_multithreaded(trainer, list_of_lists, output_files, num_processes=2, segs_from_prev_stage=None):
    if segs_from_prev_stage is None:
        segs_from_prev_stage = [None] * len(list_of_lists)

    num_processes = min(len(list_of_lists), num_processes)

    classes = list(range(1, trainer.num_classes))
    assert isinstance(trainer, nnUNetTrainer)
    # {PREDICT_PATCH_MARKER}: macOS local testing falls back to inline preprocessing
    # because multiprocessing spawn cannot pickle trainer.preprocess_patient here.
    if platform.system() == "Darwin" or os.environ.get("NNUNET_DISABLE_MULTIPROCESSING") == "1" or num_processes <= 1:
        class _InlineQueue:
            def __init__(self):
                self.items = []

            def put(self, item):
                self.items.append(item)

            def close(self):
                return None

        q = _InlineQueue()
        preprocess_save_to_queue(
            trainer.preprocess_patient,
            q,
            list_of_lists,
            output_files,
            segs_from_prev_stage,
            classes,
            trainer.plans['transpose_forward'],
        )
        for item in q.items:
            if item == "end":
                continue
            yield item
        q.close()
        return

    q = Queue(1)
"""
    if needle not in text:
        raise RuntimeError(f"Did not find expected preprocess_multithreaded block in {py_path}")
    text = text.replace(needle, replacement, 1)
    py_path.write_text(text, encoding="utf-8")
    return True


def patch_predict_simple(py_path: Path) -> bool:
    text = py_path.read_text(encoding="utf-8")
    if PREDICT_SIMPLE_PATCH_MARKER in text:
        return False

    if "import argparse\n" not in text:
        raise RuntimeError(f"Did not find expected import block in {py_path}")
    text = text.replace(
        "import argparse\n",
        "import argparse\nimport os\nimport platform\n",
        1,
    )

    needle = """    args = parser.parse_args()
    input_folder = args.input_folder
"""
    replacement = f"""    args = parser.parse_args()
    # {PREDICT_SIMPLE_PATCH_MARKER}: keep local macOS testing single-process and low-memory.
    if platform.system() == "Darwin" or os.environ.get("NNUNET_DISABLE_MULTIPROCESSING") == "1":
        args.num_threads_preprocessing = 1
        args.num_threads_nifti_save = 1
    input_folder = args.input_folder
"""
    if needle not in text:
        raise RuntimeError(f"Did not find expected argparse block in {py_path}")
    text = text.replace(needle, replacement, 1)
    py_path.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch nnUNet inference for macOS local compatibility")
    parser.add_argument("--conda-prefix", required=True)
    args = parser.parse_args()

    conda_prefix = Path(args.conda_prefix)
    predict_paths = sorted(conda_prefix.glob("*/lib/python*/site-packages/nnunet/inference/predict.py"))
    simple_paths = sorted(conda_prefix.glob("*/lib/python*/site-packages/nnunet/inference/predict_simple.py"))

    patched = []
    for path in predict_paths:
        if patch_predict(path):
            patched.append(str(path))
    for path in simple_paths:
        if patch_predict_simple(path):
            patched.append(str(path))

    if patched:
        print("Patched nnUNet compatibility files:")
        for path in patched:
            print(path)
    else:
        print("No nnUNet compatibility patches were needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
