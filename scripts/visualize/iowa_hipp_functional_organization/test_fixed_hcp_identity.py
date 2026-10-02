"""Regression contracts for the fixed HCP territory identity metrics."""

import importlib.util
from pathlib import Path
from unittest import mock
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parent


def _load(name):
    path = ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"fixed_hcp_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixedHcpIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = _load("metrics")
        cls.analysis = _load("analysis")

    @staticmethod
    def _row(rows, system, metric):
        return next(row for row in rows if row["system"] == system and row["metric"] == metric)

    def test_fixed_hcp_metric_uses_reference_territory_and_reports_coverage(self):
        scores = np.array(
            [
                [.5, .2, .1],
                [.9, .1, .2],
                [.2, .8, .4],
                [.1, .3, .7],
            ]
        )
        rows = self.metrics.fixed_hcp_identity(
            scores,
            hcp_labels=np.array([1, 1, 2, 3]),
            hcp_support=np.array([True, True, True, True]),
            area=np.array([2., 3., 5., 7.]),
            valid=np.array([True, False, True, True]),
        )
        similarity = self._row(rows, "A", "fixed_hcp_similarity")
        advantage = self._row(rows, "A", "fixed_hcp_advantage")
        self.assertAlmostEqual(similarity["value"], .5)
        self.assertAlmostEqual(advantage["value"], .3)
        self.assertEqual(similarity["valid_area"], 2)
        self.assertEqual(similarity["total_area"], 5)
        self.assertAlmostEqual(self._row(rows, "B", "fixed_hcp_similarity")["value"], .8)
        with self.assertRaisesRegex(ValueError, "HCP labels/support"):
            self.metrics.fixed_hcp_identity(
                scores,
                hcp_labels=np.array([1, 0, 2, 3]),
                hcp_support=np.array([True, True, True, True]),
                area=np.ones(4),
            )

    def test_hcp_template_index_rejects_label_support_mismatch(self):
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
        templates[0]["labels"][1] = 0
        with mock.patch.object(
            self.analysis, "_array", side_effect=lambda record, **_kwargs: np.asarray(record)
        ):
            with self.assertRaisesRegex(ValueError, "HCP labels/support"):
                self.analysis._hcp_template_index({"hcp_templates": templates})


if __name__ == "__main__":
    unittest.main()
