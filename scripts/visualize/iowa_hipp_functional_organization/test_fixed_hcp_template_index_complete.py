"""Successful-path coverage for the fixed HCP template index."""

import importlib.util
from pathlib import Path
from unittest import mock
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parent


class FixedHcpTemplateIndexCompleteTests(unittest.TestCase):
    def test_complete_fixed_hcp_template_index_has_all_branch_hemi_cells(self):
        path = ROOT / "analysis.py"
        spec = importlib.util.spec_from_file_location("fixed_hcp_index_complete", path)
        analysis = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(analysis)
        templates = []
        for branch in ("standard", "ctxreg"):
            for hemi in ("L", "R"):
                labels = np.zeros(512, dtype=np.int32)
                labels[:3] = (1, 2, 3)
                templates.append(
                    {
                        "branch": branch,
                        "hemi": hemi,
                        "labels": labels,
                        "support": labels != 0,
                    }
                )
        with mock.patch.object(analysis, "_array", side_effect=lambda record, **_kwargs: np.asarray(record)):
            index = analysis._hcp_template_index({"hcp_templates": templates})
        self.assertEqual(set(index), {(branch, hemi) for branch in ("standard", "ctxreg") for hemi in ("L", "R")})


if __name__ == "__main__":
    unittest.main()
