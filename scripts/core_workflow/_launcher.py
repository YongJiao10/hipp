#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def run_with_spec(step_id: str, purpose: str, target_relpath: str, target_type: str = "python") -> int:
    module_path = Path(__file__).with_name("_entry.py")
    spec = importlib.util.spec_from_file_location("core_workflow_entry", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load core workflow entry helper: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.run_step(
        module.StepSpec(
            step_id=step_id,
            purpose=purpose,
            target_relpath=target_relpath,
            target_type=target_type,
        )
    )
