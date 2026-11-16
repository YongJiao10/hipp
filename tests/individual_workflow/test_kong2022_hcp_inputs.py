"""Argon adapter checks; fixtures are retained under /tmp (no cleanup)."""
from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np


REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "scripts/ArealMSHBM_Kong2022/prepare_kong2022_areal_mshbm_subject_inputs.py"
spec = importlib.util.spec_from_file_location("prepare_kong2022_areal_mshbm_subject_inputs", MODULE_PATH)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class HCPAdapterTests(unittest.TestCase):
    def setUp(self):
        self.fixture = Path(tempfile.mkdtemp(prefix="kong2022_hcp_adapter_test_"))
        self.root = self.fixture / "BANDA/imaging"
        self.root.mkdir(parents=True)
        self.source_subject = "BANDA001_MR"
        self.run = adapter.HCP_RUNS[0]
        self.path = adapter.cohort_run_path(self.root, self.source_subject, self.run, "banda")
        self.path.parent.mkdir(parents=True)
        brain = nib.cifti2.BrainModelAxis.from_surface([0, 1], 32492, name="CortexLeft")
        brain += nib.cifti2.BrainModelAxis.from_surface([0, 1], 32492, name="CortexRight")
        header = nib.Cifti2Header.from_axes((nib.cifti2.SeriesAxis(0, 0.8, 3), brain))
        nib.save(nib.Cifti2Image(np.ones((3, 4), dtype=np.float32), header), self.path)
        self.metadata = self.root.parent / "metadata/datastructure_manifest.txt"
        self.metadata.parent.mkdir()
        self.manifest = self.fixture / "cohort.json"

    def metadata_rows(self, runs):
        with self.metadata.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=["associated_file"], delimiter="\t")
            writer.writeheader()
            for run in runs:
                writer.writerow({"associated_file": f"s3://bucket/submission/{self.source_subject}/MNINonLinear/Results/{run}/{run}_Atlas_MSMAll_hp0_clean.dtseries.nii"})

    def build(self):
        self.metadata_rows([self.run, "rfMRI_REST"])
        manifest = adapter.build_cohort_manifest("banda", self.root)
        self.manifest.write_text(json.dumps(manifest))
        return manifest

    def test_visit_identity_preserves_longitudinal_subject(self):
        first = adapter.cohort_identity("pdc", "e0011_01_MR")
        second = adapter.cohort_identity("pdc", "e0011_02_MR")
        self.assertEqual(first["base_subject_id"], second["base_subject_id"])
        self.assertNotEqual(first["subject"], second["subject"])
        self.assertEqual(second["subject"], "sub-e0011s02")

    def test_concat_is_explicitly_excluded_and_non_subject_directory_ignored(self):
        (self.root / "manifests").mkdir()
        (self.root / "BANDA199_MR").mkdir()
        manifest = self.build()
        self.assertEqual(manifest["run_count"], 1)
        self.assertEqual(manifest["excluded_subjects"][0]["subject"], "sub-BANDA199")
        self.assertEqual(len(manifest["excluded_subjects"]), 1)

    def test_missing_declared_run_fails_without_using_concat(self):
        concat = adapter.cohort_run_path(self.root, self.source_subject, "rfMRI_REST", "banda")
        concat.parent.mkdir(parents=True)
        concat.write_bytes(self.path.read_bytes())
        self.metadata_rows([adapter.HCP_RUNS[1], "rfMRI_REST"])
        with self.assertRaises(FileNotFoundError):
            adapter.build_cohort_manifest("banda", self.root)

    def test_duplicate_source_declaration_fails(self):
        self.metadata_rows([self.run, self.run])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            adapter.build_cohort_manifest("banda", self.root)

    def test_variant_and_identity_mismatch_fail(self):
        manifest = self.build()
        manifest["variant"] = "other"
        self.manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "variant"):
            adapter.load_cohort_manifest(self.manifest, "banda", self.root)
        manifest["variant"] = adapter.HCP_VARIANT
        manifest["subjects"][0]["subject"] = "sub-BANDA999"
        self.manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Identity"):
            adapter.load_cohort_manifest(self.manifest, "banda", self.root)

    def test_stale_input_fails(self):
        manifest = self.build()
        with self.path.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "Stale"):
            adapter.check_cohort_file(manifest["subjects"][0]["runs"][0])

    def test_prepare_persists_censor_and_consumed_density_evidence(self):
        self.build()
        output = self.fixture / "output"
        adapter.prepare_cohort_subject(self.manifest, "banda", self.root, "BANDA001", output)
        payload = json.loads((output / "inputs/input_manifest.json").read_text())
        self.assertEqual(payload["source_subject"], self.source_subject)
        self.assertEqual(payload["runs"][0]["timepoints"], 3)
        self.assertEqual(Path(payload["runs"][0]["censor"]).read_text(), "1\n1\n1\n")
        self.assertEqual(payload["runs"][0]["density_evidence"]["cortex_vertices_per_hemisphere"], 32492)
        with self.assertRaisesRegex(ValueError, "Expected one declaration"):
            adapter.prepare_cohort_subject(self.manifest, "banda", self.root, "BANDA999", output)

    def launch(self, **extra):
        env = {**os.environ, "SYSTEM_NAME": "argon", "DATASET": "banda",
               "REPO_ROOT": str(REPO), "DATA_ROOT": str(self.root),
               "COHORT_MANIFEST": str(self.manifest), "PYTHON_EXE": sys.executable,
               "DRY_RUN": "1", "KEEP_TMP": "1", "OVERWRITE": "0",
               "OUT_ROOT": str(self.fixture / "launch"),
               "SUBJECT_IDS_CSV": "BANDA001", "SUBJECT_IDS_FILE": "",
               "UI_TC": "5", "PINC_TC": "15", **extra}
        return subprocess.run(["bash", str(MODULE_PATH.parent / "submit_kong2022_areal_mshbm.sh")],
                              env=env, capture_output=True, text=True)

    def test_launcher_keeps_alphanumeric_subject_and_explicit_dependency(self):
        self.build()
        result = self.launch(HOLD_JID="123,456")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-hold_jid 123,456", result.stderr)
        self.assertIn("total=1", result.stderr)
        wrapper = list((self.fixture / "launch/_manifests").glob("*.job.sh"))
        self.assertEqual(len(wrapper), 1)
        self.assertIn("COHORT_MANIFEST=", wrapper[0].read_text())

    def test_launcher_unknown_subject_and_excess_concurrency_fail(self):
        self.build()
        result = self.launch(SUBJECT_IDS_CSV="BANDA999")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not declared", result.stderr)
        result = self.launch(UI_TC="6", PINC_TC="15")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exceeds argon concurrency cap 20", result.stderr)


class NIHAdapterTests(unittest.TestCase):
    def setUp(self):
        self.fixture = Path(tempfile.mkdtemp(prefix="kong2022_nih_adapter_test_"))
        self.root = self.fixture / "xcpd_surf/restfmri"
        self.roster = self.fixture / "sublist.txt"
        self.roster.write_text("sub-D01\nsub-D02\nsub-D16\n")
        brain = nib.cifti2.BrainModelAxis.from_surface(np.arange(32492), 32492, name="CortexLeft")
        brain += nib.cifti2.BrainModelAxis.from_surface(np.arange(32492), 32492, name="CortexRight")
        header = nib.Cifti2Header.from_axes((nib.cifti2.SeriesAxis(0, 0.8, 2), brain))
        for source_subject in ("D01_ses-1", "D01_ses-2", "D02_ses-1"):
            path = adapter.cohort_run_path(self.root, source_subject, adapter.NIH_RUN_ID, adapter.NIH_DATASET)
            path.parent.mkdir(parents=True)
            nib.save(nib.Cifti2Image(np.ones((2, 64984), dtype=np.float32), header), path)
        (self.root / "sub-D16").mkdir()
        self.original_roster = adapter.NIH_ROSTER
        adapter.NIH_ROSTER = self.roster

    def tearDown(self):
        adapter.NIH_ROSTER = self.original_roster

    def test_manifest_records_excluded_subjects_and_missing_sessions(self):
        manifest = adapter.build_cohort_manifest(adapter.NIH_DATASET, self.root)
        self.assertEqual([entry["subject"] for entry in manifest["subjects"]], ["sub-D01s1", "sub-D01s2", "sub-D02s1"])
        self.assertEqual(manifest["excluded_subjects"], [{"subject": "sub-D16", "reason": "no_session_directory_in_xcpd_surf"}])
        self.assertEqual(manifest["missing_sessions"],
                         [{"subject": "sub-D02", "session": "ses-2", "reason": "no_session_directory_in_xcpd_surf"}])

    def test_prepare_records_existing_session_directory(self):
        manifest_path = self.fixture / "cohort.json"
        manifest_path.write_text(json.dumps(adapter.build_cohort_manifest(adapter.NIH_DATASET, self.root)))
        output = self.fixture / "output"
        adapter.prepare_cohort_subject(manifest_path, adapter.NIH_DATASET, self.root, "D01s2", output)
        payload = json.loads((output / "inputs/input_manifest.json").read_text())
        self.assertEqual(payload["subject_dir"], str(self.root / "sub-D01" / "ses-2"))
        self.assertTrue(Path(payload["subject_dir"]).is_dir())


if __name__ == "__main__":
    unittest.main()
