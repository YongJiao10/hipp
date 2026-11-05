"""Report contract tests; retained test outputs stay outside the source tree."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import numpy as np
import unittest

ROOT = Path(__file__).resolve().parent
NETWORKS = ("DMN", "SMN", "VAN", "DAN", "CON", "LAN", "VN")
BRANCHES = ("standard", "ctxreg")
COMPARISONS = ("patient", "pure_mdd", "comorbid_mdd")
COHORTS = ("hc", *COMPARISONS)
HEMIS = ("L", "R")
ATLASES = ("tm_dominant", "de_novo", "wta")
STAT_FIELDS = (
    "theme", "branch", "hemi", "atlas", "system", "metric", "target",
    "comparison", "case_cohort", "control_cohort", "mean_patient",
    "sd_patient", "n_patient", "mean_hc", "sd_hc", "n_hc", "difference",
    "ci_low", "ci_high", "g", "p", "df", "reason", "significance",
)
ROI_FIELDS = (
    "theme", "branch", "hemi", "atlas", "system", "metric", "target",
    "target_level", "roi_id", "comparison", "case_cohort", "control_cohort",
    "mean_patient", "sd_patient", "n_patient", "mean_hc", "sd_hc", "n_hc",
    "difference", "ci_low", "ci_high", "g", "p", "df", "reason", "significance",
)
SUBJECT_FIELDS = (
    "theme", "branch", "hemi", "atlas", "cohort", "subject", "system",
    "metric", "target", "target_level", "roi_id", "value", "reason",
    "valid_area", "total_area", "n_valid_boundaries",
)


class ReportTests(unittest.TestCase):
    def module(self):
        self.assertTrue((ROOT / "report.py").is_file(), "Report renderer must exist")
        spec = importlib.util.spec_from_file_location("functional_report", ROOT / "report.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _write_csv(path, rows, fields):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _analysis_config():
        return {
            "scope": {
                "smoothing": "2mm",
                "variant": "36p",
                "taxonomy": "canonical7",
                "n_roi": 400,
                "density": 512,
            },
            "tests": {
                "primary": "two-sided Welch",
                "covariates": "none",
                "correction": "none",
            },
        }

    @staticmethod
    def _row(**overrides):
        row = dict(
            theme="identity", branch="standard", hemi="L", atlas="tm_individual",
            system="whole", metric="smax", target="", comparison="patient",
            case_cohort="patient", control_cohort="hc", mean_patient=1,
            sd_patient=1, n_patient=3, mean_hc=1, sd_hc=1, n_hc=3,
            difference=0, ci_low=-1, ci_high=1, g=0, p=1, df=4,
            reason="", significance="",
        )
        row.update(overrides)
        return row

    def _conforming_statistics(self):
        identity_choices = [
            ("tm_individual", "smax", system, "")
            for system in ("whole", "A", "B", "P")
        ] + [
            ("tm_individual", "margin", system, "")
            for system in ("whole", "A", "B", "P")
        ] + [
            ("hcp_fixed_territory", metric, system, "")
            for metric in ("fixed_hcp_similarity", "fixed_hcp_advantage")
            for system in ("A", "B", "P")
        ] + [
            ("tm_individual", metric, system, "")
            for metric in ("segregation", "boundary_sharpness")
            for system in ("A", "B", "P")
        ]
        choices = [
            *(("identity", atlas, system, metric, target) for atlas, metric, system, target in identity_choices),
            ("spatial", "tm_individual", "A", "area_fraction", ""),
            ("fingerprints", "tm_dominant", "A", "fisher_z", "DMN"),
            ("continuous", "individual_gradient", "G1", "robust_range", ""),
        ]
        rows = []
        for theme, atlas, system, metric, target in choices:
            for branch in BRANCHES:
                for hemi in HEMIS:
                    for comparison in COMPARISONS:
                        row = self._row(
                            theme=theme, branch=branch, hemi=hemi, atlas=atlas,
                            system=system, metric=metric, target=target,
                            comparison=comparison, case_cohort=comparison,
                        )
                        if (
                            theme == "identity"
                            and branch == "standard"
                            and hemi == "L"
                            and comparison == "patient"
                        ):
                            row.update(g="", p="", reason="no valid area")
                        rows.append(row)
        for comparison in COMPARISONS:
            for target in NETWORKS:
                rows.append(
                    self._row(
                        theme="spatial", branch="shared_36p_cortex",
                        hemi="bilateral_cortex", atlas="cortex_canonical7",
                        system="cortex", metric="area_percent", target=target,
                        comparison=comparison, case_cohort=comparison,
                    )
                )
        return rows

    @staticmethod
    def _conforming_maps():
        axis = np.linspace(0.0, 1.0, 512).tolist()
        maps = []
        for branch in BRANCHES:
            for atlas in ATLASES:
                for cohort in COHORTS:
                    for hemi in HEMIS:
                        transverse = np.linspace(181.25, 200.0, 512)
                        if hemi == "R":
                            transverse = -transverse
                        maps.append(
                            {
                                "theme": "spatial",
                                "branch": branch,
                                "cohort": cohort,
                                "hemi": hemi,
                                "atlas": atlas,
                                "scope": "descriptive_group_atlas",
                                "display_geometry_subject": "sub-1001",
                                "axis": axis,
                                "transverse": transverse.tolist(),
                                "labels": [1] * 512,
                                "support": [True] * 512,
                            }
                        )
        for branch in BRANCHES:
            for cohort in COHORTS:
                for hemi in HEMIS:
                    transverse = np.linspace(181.25, 200.0, 512)
                    if hemi == "R":
                        transverse = -transverse
                    maps.append(
                        {
                            "theme": "continuous",
                            "branch": branch,
                            "cohort": cohort,
                            "hemi": hemi,
                            "atlas": "individual_gradient",
                            "scope": "descriptive_group_mean",
                            "display_geometry_subject": "sub-1001",
                            "axis": axis,
                            "transverse": transverse.tolist(),
                            "labels": [0] * 512,
                            "gradients": np.column_stack(
                                (
                                    np.linspace(0.0, 1.0, 512),
                                    np.linspace(1.0, 0.0, 512),
                                    np.zeros(512),
                                )
                            ).tolist(),
                        }
                    )
        return maps

    @staticmethod
    def _subject_metric_bases(statistics):
        unique = {}
        for row in statistics:
            key = tuple(
                row[name]
                for name in ("theme", "branch", "hemi", "atlas", "system", "metric", "target")
            )
            unique.setdefault(
                key,
                dict(
                    theme=row["theme"], branch=row["branch"], hemi=row["hemi"],
                    atlas=row["atlas"], cohort="hc", subject="fixture",
                    system=row["system"], metric=row["metric"], target=row["target"],
                    target_level="hippocampus", roi_id="", value=0, reason="",
                    valid_area=1, total_area=1, n_valid_boundaries="",
                ),
            )
        return list(unique.values())

    @staticmethod
    def _conforming_roi_statistics():
        rows = []
        for branch in BRANCHES:
            for hemi in HEMIS:
                for atlas in ATLASES:
                    for system in ("A", "B", "P"):
                        for roi_id in range(1, 401):
                            for comparison in COMPARISONS:
                                rows.append(
                                    dict(
                                        theme="fingerprints", branch=branch, hemi=hemi,
                                        atlas=atlas, system=system, metric="fisher_z",
                                        target=f"ROI_{roi_id}", target_level="roi",
                                        roi_id=roi_id, comparison=comparison,
                                        case_cohort=comparison, control_cohort="hc",
                                        mean_patient=0, sd_patient=1, n_patient=3,
                                        mean_hc=0, sd_hc=1, n_hc=3, difference=0,
                                        ci_low=-1, ci_high=1, g=0, p=1, df=4,
                                        reason="", significance="",
                                    )
                                )
        return rows

    @staticmethod
    def _refresh_manifest(out):
        manifest_path = out / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["outputs"] = {
            name: {"sha256": hashlib.sha256((out / name).read_bytes()).hexdigest()}
            for name in manifest["outputs"]
            if (out / name).is_file()
        }
        manifest_path.write_text(json.dumps(manifest))

    def fixture(
        self,
        *,
        include_roi_statistics=True,
        include_boundary_positions=True,
        conforming=False,
    ):
        out = Path(tempfile.mkdtemp(prefix="report_test_", dir=tempfile.gettempdir()))
        if conforming:
            rows = self._conforming_statistics()
            maps = self._conforming_maps()
            roi_rows = self._conforming_roi_statistics()
            subject_rows = self._subject_metric_bases(rows)
        else:
            rows = [
                self._row(branch="36p", atlas="tm_dominant", target="DMN"),
                self._row(
                    branch="36p", atlas="tm_dominant", system="B", metric="margin",
                    target="SMN", mean_patient="", n_patient=0, difference="",
                    ci_low="", ci_high="", g="", p="", df="", reason="no valid area",
                ),
            ]
            maps = []
            roi_rows = []
            subject_rows = []
        self._write_csv(out / "statistics.csv", rows, STAT_FIELDS)
        self._write_csv(out / "subject_metrics.csv", subject_rows, SUBJECT_FIELDS)
        (out / "curves.csv").write_text(
            "theme,branch,hemi,atlas,cohort,subject,system,metric,bin,position,value\n"
        )
        (out / "spatial_maps.json").write_text(json.dumps(maps))
        registry = [
            dict(
                roi_id=i, roi_name=f"ROI_{i}", network=NETWORKS[(i - 1) % len(NETWORKS)],
                hemi="L" if i <= 200 else "R",
            )
            for i in range(1, 401)
        ]
        (out / "roi_registry.json").write_text(json.dumps(registry))
        (out / "analysis_config.json").write_text(json.dumps(self._analysis_config()))
        if include_roi_statistics:
            self._write_csv(out / "roi_statistics.csv", roi_rows, ROI_FIELDS)
        if include_boundary_positions:
            (out / "boundary_positions.csv").write_text(
                "theme,branch,hemi,atlas,cohort,subject,system_pair,position,reason,scope\n"
            )
        files = [
            "statistics.csv", "subject_metrics.csv", "curves.csv", "spatial_maps.json",
            "roi_registry.json", "analysis_config.json",
        ]
        if include_roi_statistics:
            files.append("roi_statistics.csv")
        if include_boundary_positions:
            files.append("boundary_positions.csv")
        (out / "run_manifest.json").write_text(
            json.dumps(
                dict(
                    completed_stages=["statistics"],
                    outputs={
                        name: dict(sha256=hashlib.sha256((out / name).read_bytes()).hexdigest())
                        for name in files
                    },
                )
            )
        )
        self.assertEqual(out.parent, Path(tempfile.gettempdir()))
        return out

    def test_unfolded_display_coordinates_reflect_right_to_common_orientation(self):
        report = self.module()
        axis = np.array([0.0, 0.5, 1.0])
        left = np.array([181.25, 190.625, 200.0])
        right = -left
        left_x, left_y = report._unfolded_display_coordinates(axis, left, "L")
        right_x, right_y = report._unfolded_display_coordinates(axis, right, "R")
        np.testing.assert_allclose(left_x, [0.0, 19.375, 38.75])
        np.testing.assert_allclose(right_x, left_x)
        np.testing.assert_allclose(right_y, left_y)
        with self.assertRaisesRegex(ValueError, "single signed"):
            report._unfolded_display_coordinates(axis, [-1.0, 1.0, 2.0], "R")

    def test_missing_is_distinct_from_zero_and_p_boundaries(self):
        report = self.module()
        self.assertEqual(report.effect_label({"g": "", "p": ""}), "×")
        self.assertEqual(report.effect_label({"g": "0", "p": "1"}), "0.00")
        for p, symbol in [(0.1, ""), (0.05, "+"), (0.01, "*"), (0.001, "**"), (0.0009, "***")]:
            self.assertEqual(report.p_symbol(p), symbol)

    def test_categorical_values_masks_unsupported_vertices(self):
        report = self.module()
        values = report._categorical_map_values(
            {"labels": [1, 2, 3] + [0] * 509, "support": [1, 0, 1] + [1] * 509}
        )
        np.testing.assert_allclose(values[[0, 2]], [1, 3])
        self.assertTrue(np.isnan(values[1]))

    def test_tampered_derived_artifact_rejected(self):
        report = self.module()
        out = self.fixture()
        (out / "curves.csv").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "curves.csv.*SHA256"):
            report.render(out)

    def test_missing_boundary_positions_is_rejected_before_rendering(self):
        report = self.module()
        out = self.fixture(include_boundary_positions=False)
        with self.assertRaisesRegex(ValueError, "boundary_positions.csv"):
            report.render(out)

    def test_missing_roi_statistics_is_rejected_before_rendering(self):
        report = self.module()
        out = self.fixture(include_roi_statistics=False)
        with self.assertRaisesRegex(ValueError, "roi_statistics.csv"):
            report.render(out)

    def test_scope_rejects_noncanonical_branch_before_rendering(self):
        report = self.module()
        with self.assertRaisesRegex(ValueError, "branch"):
            report._load(self.fixture())

    def test_scope_accepts_only_declared_group_atlas_curve_cohorts(self):
        report = self.module()
        report._row_scope(
            [
                {
                    "theme": "spatial", "branch": "standard", "hemi": "L",
                    "atlas": "tm_dominant", "cohort": "group_hc",
                    "scope": "descriptive_group_atlas",
                }
            ],
            artifact="curves.csv",
        )
        with self.assertRaisesRegex(ValueError, "scope"):
            report._row_scope(
                [
                    {
                        "theme": "spatial", "branch": "standard", "hemi": "L",
                        "atlas": "tm_dominant", "cohort": "group_unlisted",
                        "scope": "descriptive_group_atlas",
                    }
                ],
                artifact="curves.csv",
            )

    def test_scope_rejects_subject_metric_without_all_three_statistics(self):
        report = self.module()
        out = self.fixture(conforming=True)
        with (out / "subject_metrics.csv").open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUBJECT_FIELDS)
            writer.writerow(
                dict(
                    theme="identity", branch="standard", hemi="L",
                    atlas="tm_individual", cohort="hc", subject="fixture-extra",
                    system="whole", metric="margin", target="unexpected",
                    target_level="hippocampus", roi_id="", value=0, reason="",
                    valid_area=1, total_area=1, n_valid_boundaries="",
                )
            )
        self._refresh_manifest(out)
        with self.assertRaisesRegex(ValueError, "statistics.csv effect coverage"):
            report._load(out)

    def test_scope_rejects_incomplete_roi_effect_coverage(self):
        report = self.module()
        out = self.fixture(conforming=True)
        lines = (out / "roi_statistics.csv").read_text().splitlines()
        (out / "roi_statistics.csv").write_text("\n".join(lines[:-1]) + "\n")
        self._refresh_manifest(out)
        with self.assertRaisesRegex(ValueError, "ROI effect coverage"):
            report._load(out)

    def test_scope_rejects_missing_descriptive_group_map_cell(self):
        report = self.module()
        out = self.fixture(conforming=True)
        maps = json.loads((out / "spatial_maps.json").read_text())
        maps.pop(next(index for index, item in enumerate(maps) if item["theme"] == "spatial"))
        (out / "spatial_maps.json").write_text(json.dumps(maps))
        self._refresh_manifest(out)
        with self.assertRaisesRegex(ValueError, "descriptive group-map grid"):
            report._load(out)

    def test_render_standalone_filters_full_roi_registry_five_pngs_and_static_pdf(self):
        report = self.module()
        out = self.fixture(conforming=True)
        result = report.render(out)
        html = (out / "index.html").read_text()
        for token in [
            "const fields=['branch','hemi','atlas','metric']", "s.id='filter-'+field",
            "roi-search", "roi-effects", "ROI effect estimates",
            "render all matching ROI effects", "roiStatistics", "ROI_400",
            "no valid area", "anatomical labels", "0.00", "×",
        ]:
            self.assertIn(token, html)
        self.assertNotIn("cdn.", html)
        png_names = sorted(Path(p).name for p in result if str(p).endswith(".png"))
        self.assertEqual(
            png_names,
            [
                "hipp_continuous_gradients.png",
                "hipp_functional_identity_ctxreg.png",
                "hipp_functional_identity_standard.png",
                "hipp_network_fingerprints.png",
                "hipp_spatial_organization.png",
            ],
        )
        pdfs = [Path(path) for path in result if str(path).endswith(".pdf")]
        self.assertEqual(pdfs, [out / "iowa_hipp_functional_organization_report.pdf"])
        self.assertTrue(pdfs[0].read_bytes().startswith(b"%PDF-"))
        for path in result:
            self.assertTrue(Path(path).stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()

