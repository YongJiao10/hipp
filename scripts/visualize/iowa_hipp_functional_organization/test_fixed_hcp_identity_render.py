"""Integration contract for the identity-only rendering stage."""

import importlib.util
import json
from pathlib import Path
from unittest import mock
import unittest

from scripts.visualize.iowa_hipp_functional_organization.test_report import ReportTests


ROOT = Path(__file__).resolve().parent


class FixedHcpIdentityRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "analysis.py"
        spec = importlib.util.spec_from_file_location("fixed_hcp_identity_analysis", path)
        cls.analysis = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.analysis)

    def test_identity_stage_renders_only_the_two_identity_figures(self):
        out = ReportTests().fixture(conforming=True)
        with mock.patch.object(self.analysis, "_checked_output_root", return_value=out):
            paths = self.analysis.render_identity_report(out)
        self.assertEqual(
            paths,
            [
                out / "hipp_functional_identity_standard.png",
                out / "hipp_functional_identity_ctxreg.png",
            ],
        )
        self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in paths))
        manifest = json.loads((out / "run_manifest.json").read_text())
        self.assertIn("identity_render", manifest["completed_stages"])
        self.assertEqual(
            set(path.name for path in paths),
            {"hipp_functional_identity_standard.png", "hipp_functional_identity_ctxreg.png"},
        )
        parsed = self.analysis._parser().parse_args(["--stage", "identity", "--workers", "1"])
        self.assertEqual((parsed.stage, parsed.workers), ("identity", 1))


if __name__ == "__main__":
    unittest.main()
