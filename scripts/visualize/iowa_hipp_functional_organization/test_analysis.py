"""Analysis orchestration contracts with small synthetic observations."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest import mock
import subprocess
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parent


class AnalysisContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "analysis.py"
        cls.module_path = path
        if path.exists():
            spec = importlib.util.spec_from_file_location("iowa_organization_analysis", path)
            cls.analysis = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.analysis)

    def setUp(self):
        self.assertTrue(self.module_path.is_file(), "analysis orchestration is missing")

    def test_direct_entrypoint_bootstraps_repository_import(self):
        code = f"""
import importlib
import importlib.util
from pathlib import Path
path = Path({str(self.module_path)!r})
spec = importlib.util.spec_from_file_location("iowa_direct_entrypoint", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
loaded = importlib.import_module("scripts.visualize.lib.multidataset_cohorts")
assert hasattr(loaded, "GridSpec")
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd="/tmp",
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_wta_keeps_only_designated_networks_without_renormalizing(self):
        mapping = {
            "network_label_keys": {
                "Default": 1,
                "Somatomotor": 3,
                "VentralAttention": 5,
                "Visual": 2,
            }
        }
        labels = np.array([1, 3, 5, 2, 8])
        np.testing.assert_array_equal(
            self.analysis.wta_pseudoclusters(labels, mapping), [3, 2, 1, 0, 0]
        )

    def test_tm_dominant_excludes_invalid_source_vertices(self):
        rosters = {
            "hc": ["h1", "h2"],
            "patient": ["p1"],
            "pure_mdd": ["q1"],
            "comorbid_mdd": ["r1"],
        }
        labels = {name: np.ones(512, dtype=np.int32) for names in rosters.values() for name in names}
        scores = {name: np.zeros((512, 3), dtype=float) for names in rosters.values() for name in names}
        valid = {name: np.ones(512, dtype=bool) for names in rosters.values() for name in names}
        labels["h2"][:] = 2
        labels["h1"][0] = 0
        scores["h1"][0] = np.nan
        valid["h1"][0] = False
        subjects = []
        for branch in ("standard", "ctxreg"):
            for cohort, names in rosters.items():
                for hemi in ("L", "R"):
                    for subject in names:
                        subjects.append(
                            {
                                "branch": branch, "hemi": hemi, "subject": subject,
                                "labels": labels[subject], "scores": scores[subject],
                                "valid": valid[subject],
                            }
                        )
        with mock.patch.object(
            self.analysis, "_array", side_effect=lambda record, **_kwargs: np.asarray(record)
        ):
            maps = self.analysis._tm_dominant_maps(
                {"rosters": rosters, "subjects": subjects}, {}
            )
        self.assertEqual(int(maps[("standard", "hc", "L")][0]), 2)

    def test_boundary_rows_preserve_each_position_for_overlapping_memberships(self):
        destination = []
        self.analysis._append_boundary_rows(
            destination,
            {
                "theme": "spatial", "branch": "standard", "hemi": "L",
                "atlas": "tm_individual", "subject": "sub-3001",
            },
            [
                {"system_pair": "A_B", "position": 0.1, "reason": ""},
                {"system_pair": "A_B", "position": 0.6, "reason": ""},
            ],
            ("patient", "pure_mdd"),
        )
        self.assertEqual(len(destination), 4)
        self.assertEqual(
            [(row["cohort"], row["system_pair"], row["position"]) for row in destination],
            [
                ("patient", "A_B", 0.1), ("patient", "A_B", 0.6),
                ("pure_mdd", "A_B", 0.1), ("pure_mdd", "A_B", 0.6),
            ],
        )

    def test_cortex_area_explicitly_excludes_background_components(self):
        source = ROOT.parent / "cortex_network_area" / "stats" / "subject_network_area_metrics.csv"
        rows = self.analysis._cortex_area_rows(
            {
                "cortex_subjects": self.analysis.inputs.fingerprint(source),
                "rosters": self.analysis.inputs.load_rosters(),
            }
        )
        self.assertEqual(len(rows), 119 * 7)
        self.assertEqual({row["target"] for row in rows}, set(self.analysis.metrics.NETWORKS))
        self.assertEqual(
            {row["cohort"] for row in rows},
            {"hc", "patient", "pure_mdd", "comorbid_mdd"},
        )

    def test_cortex_area_rejects_same_count_with_a_substituted_subject(self):
        source = ROOT.parent / "cortex_network_area" / "stats" / "subject_network_area_metrics.csv"
        rosters = {
            cohort: list(subjects)
            for cohort, subjects in self.analysis.inputs.load_rosters().items()
        }
        rosters["hc"][0] = "sub-not-in-declared-hc-roster"
        with self.assertRaisesRegex(ValueError, "subject roster mismatch"):
            self.analysis._cortex_area_rows(
                {
                    "cortex_subjects": self.analysis.inputs.fingerprint(source),
                    "rosters": rosters,
                }
            )

    def test_descriptive_atlases_keep_their_own_support_domains(self):
        dominant = np.zeros(512, dtype=np.int32)
        dominant[[0, 2]] = (1, 3)
        de_novo = np.ones(512, dtype=np.int8)
        wta = np.ones(512, dtype=np.int8)
        de_novo[1] = 0
        wta[0] = 0
        supports = self.analysis._atlas_supports(dominant, de_novo, wta)
        np.testing.assert_array_equal(supports["tm_dominant"][:3], [True, False, True])
        np.testing.assert_array_equal(supports["de_novo"][:3], [True, False, True])
        np.testing.assert_array_equal(supports["wta"][:3], [False, True, True])

    def test_cache_guard_requires_overwrite_when_provenance_cache_exists(self):
        cache_root = Path("/tmp/iowa-analysis-cache-contract")
        with mock.patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(FileExistsError, "provenance cache"):
                self.analysis._guard_cache(cache_root, overwrite=False)

    def test_statistics_retains_all_prespecified_overlapping_comparisons(self):
        rows = [
            dict(
                theme="identity", branch="standard", hemi="L", atlas="tm_dominant",
                cohort="hc", subject="sub-h1", system="whole", metric="smax",
                target="", value=0.0, reason="",
            ),
            dict(
                theme="identity", branch="standard", hemi="L", atlas="tm_dominant",
                cohort="hc", subject="sub-h2", system="whole", metric="smax",
                target="", value=0.2, reason="",
            ),
            *[
                dict(
                    theme="identity", branch="standard", hemi="L", atlas="tm_dominant",
                    cohort=cohort, subject=f"sub-{cohort}{i}", system="whole", metric="smax",
                    target="", value=value, reason="",
                )
                for cohort, values in (("patient", [0.4, 0.6]), ("pure_mdd", [0.4, 0.6]), ("comorbid_mdd", [0.4, 0.6]))
                for i, value in enumerate(values)
            ],
        ]
        estimates = self.analysis.statistics_from_subject_rows(rows)
        self.assertEqual([row["comparison"] for row in estimates], ["patient", "pure_mdd", "comorbid_mdd"])
        self.assertTrue(all(row["n_patient"] == 2 and row["n_hc"] == 2 for row in estimates))
        self.assertTrue(all(row["significance"] in ("", "+", "*", "**", "***", "NA") for row in estimates))

    def test_statistics_never_fills_empty_system_with_zero(self):
        rows = [
            dict(theme="fingerprints", branch="ctxreg", hemi="R", atlas="wta", cohort="hc", subject="h1", system="A", metric="fisherz", target="DMN", value=np.nan, reason="empty valid system"),
            dict(theme="fingerprints", branch="ctxreg", hemi="R", atlas="wta", cohort="hc", subject="h2", system="A", metric="fisherz", target="DMN", value=np.nan, reason="empty valid system"),
            dict(theme="fingerprints", branch="ctxreg", hemi="R", atlas="wta", cohort="patient", subject="p1", system="A", metric="fisherz", target="DMN", value=np.nan, reason="empty valid system"),
            dict(theme="fingerprints", branch="ctxreg", hemi="R", atlas="wta", cohort="patient", subject="p2", system="A", metric="fisherz", target="DMN", value=np.nan, reason="empty valid system"),
            dict(theme="fingerprints", branch="ctxreg", hemi="R", atlas="wta", cohort="pure_mdd", subject="q1", system="A", metric="fisherz", target="DMN", value=np.nan, reason="empty valid system"),
            dict(theme="fingerprints", branch="ctxreg", hemi="R", atlas="wta", cohort="pure_mdd", subject="q2", system="A", metric="fisherz", target="DMN", value=np.nan, reason="empty valid system"),
            dict(theme="fingerprints", branch="ctxreg", hemi="R", atlas="wta", cohort="comorbid_mdd", subject="r1", system="A", metric="fisherz", target="DMN", value=np.nan, reason="empty valid system"),
            dict(theme="fingerprints", branch="ctxreg", hemi="R", atlas="wta", cohort="comorbid_mdd", subject="r2", system="A", metric="fisherz", target="DMN", value=np.nan, reason="empty valid system"),
        ]
        estimates = self.analysis.statistics_from_subject_rows(rows)
        self.assertTrue(all(np.isnan(row["g"]) for row in estimates))
        self.assertTrue(all("fewer than two" in row["reason"] for row in estimates))


    def test_roi_profiles_keep_each_comparison_atlas_separate(self):
        memberships = ("patient", "pure_mdd")
        profiles = {
            membership: {
                atlas: np.full((3, 400), index + 1, dtype=float)
                for index, atlas in enumerate(("tm_dominant", "de_novo", "wta"))
            }
            for membership in memberships
        }
        items = list(self.analysis._roi_profile_items(profiles, memberships))
        self.assertEqual(
            [(membership, atlas) for membership, atlas, _value in items],
            [
                ("patient", "tm_dominant"), ("patient", "de_novo"), ("patient", "wta"),
                ("pure_mdd", "tm_dominant"), ("pure_mdd", "de_novo"), ("pure_mdd", "wta"),
            ],
        )
        malformed = {"patient": profiles["patient"]}
        with self.assertRaisesRegex(ValueError, "membership"):
            list(self.analysis._roi_profile_items(malformed, memberships))

    def test_cli_declares_fixed_default_output_and_worker_cap(self):
        parser = self.analysis._parser()
        defaults = parser.parse_args([])
        self.assertEqual(defaults.stage, "all")
        self.assertEqual(defaults.workers, 4)
        self.assertEqual(defaults.out, str(self.analysis.DEFAULT_OUTPUT))
        render = parser.parse_args(["--stage", "render", "--workers", "1"])
        self.assertEqual((render.stage, render.workers), ("render", 1))


if __name__ == "__main__":
    unittest.main()
