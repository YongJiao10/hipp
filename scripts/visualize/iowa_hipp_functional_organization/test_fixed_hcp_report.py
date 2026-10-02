"""Report-contract tests for the six fixed-HCP identity panels."""

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent


class FixedHcpReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "report.py"
        spec = importlib.util.spec_from_file_location("fixed_hcp_report", path)
        cls.report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.report)

    @staticmethod
    def _identity_rows():
        systems = {
            ("tm_individual", "smax"): ("whole", "A", "B", "P"),
            ("tm_individual", "margin"): ("whole", "A", "B", "P"),
            ("hcp_fixed_territory", "fixed_hcp_similarity"): ("A", "B", "P"),
            ("hcp_fixed_territory", "fixed_hcp_advantage"): ("A", "B", "P"),
            ("tm_individual", "segregation"): ("A", "B", "P"),
            ("tm_individual", "boundary_sharpness"): ("A", "B", "P"),
        }
        return [
            {
                "theme": "identity",
                "branch": branch,
                "hemi": hemi,
                "atlas": atlas,
                "system": system,
                "metric": metric,
                "target": "",
            }
            for branch in ("standard", "ctxreg")
            for hemi in ("L", "R")
            for (atlas, metric), labels in systems.items()
            for system in labels
        ]

    def test_identity_metric_contract_allows_repeated_subject_cells(self):
        rows = self._identity_rows()
        repeated = [dict(row, subject="sub-1001") for row in rows] + [
            dict(row, subject="sub-1002") for row in rows
        ]
        self.report._identity_metric_scope(repeated)

    def test_identity_metric_contract_requires_the_six_fixed_panel_metrics(self):
        rows = self._identity_rows()
        self.report._identity_metric_scope(rows)
        missing = [row for row in rows if row["metric"] != "fixed_hcp_advantage"]
        with self.assertRaisesRegex(ValueError, "identity metric coverage"):
            self.report._identity_metric_scope(missing)
        retired = rows + [
            {
                "theme": "identity",
                "branch": "standard",
                "hemi": "L",
                "atlas": "tm_individual",
                "system": "A",
                "metric": "all_negative_fraction",
                "target": "",
            }
        ]
        with self.assertRaisesRegex(ValueError, "identity metric coverage"):
            self.report._identity_metric_scope(retired)


if __name__ == "__main__":
    unittest.main()
