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
        self.path = adapter.hcp_run_path(self.root, self.source_subject, self.run)
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
        manifest = adapter.build_hcp_manifest("banda", self.root)
        self.manifest.write_text(json.dumps(manifest))
        return manifest

    def test_visit_identity_preserves_longitudinal_subject(self):
        first = adapter.hcp_identity("pdc", "e0011_01_MR")
        second = adapter.hcp_identity("pdc", "e0011_02_MR")
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
        concat = adapter.hcp_run_path(self.root, self.source_subject, "rfMRI_REST")
        concat.parent.mkdir(parents=True)
        concat.write_bytes(self.path.read_bytes())
        self.metadata_rows([adapter.HCP_RUNS[1], "rfMRI_REST"])
        with self.assertRaises(FileNotFoundError):
            adapter.build_hcp_manifest("banda", self.root)

    def test_duplicate_source_declaration_fails(self):
        self.metadata_rows([self.run, self.run])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            adapter.build_hcp_manifest("banda", self.root)

    def test_variant_and_identity_mismatch_fail(self):
        manifest = self.build()
        manifest["variant"] = "other"
        self.manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "variant"):
            adapter.load_hcp_manifest(self.manifest, "banda", self.root)
        manifest["variant"] = adapter.HCP_VARIANT
        manifest["subjects"][0]["subject"] = "sub-BANDA999"
        self.manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Identity"):
            adapter.load_hcp_manifest(self.manifest, "banda", self.root)

    def test_stale_input_fails(self):
        manifest = self.build()
        with self.path.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "Stale"):
            adapter.check_hcp_file(manifest["subjects"][0]["runs"][0])

    def test_prepare_persists_censor_and_consumed_density_evidence(self):
        self.build()
        output = self.fixture / "output"
        adapter.prepare_hcp_subject(self.manifest, "banda", self.root, "BANDA001", output)
        payload = json.loads((output / "inputs/input_manifest.json").read_text())
        self.assertEqual(payload["source_subject"], self.source_subject)
        self.assertEqual(payload["runs"][0]["timepoints"], 3)
        self.assertEqual(Path(payload["runs"][0]["censor"]).read_text(), "1\n1\n1\n")
        self.assertEqual(payload["runs"][0]["density_evidence"]["cortex_vertices_per_hemisphere"], 32492)
        with self.assertRaisesRegex(ValueError, "Expected one declaration"):
            adapter.prepare_hcp_subject(self.manifest, "banda", self.root, "BANDA999", output)

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


if __name__ == "__main__":
    unittest.main()
