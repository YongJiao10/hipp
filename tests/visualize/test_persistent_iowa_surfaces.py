from pathlib import Path
import pytest
from scripts.visualize.cortex_network_area.prepare_iowa_surfaces import apply_manifest, validate_output_paths


def test_persistent_surface_outputs_are_under_dataset_inputs():
    root = Path("/workspace/hipp")
    dest = root / "outputs_iowa_jj_tms/inputs/iowa_jj_tms/cortex_surfaces/fsLR32k"
    validate_output_paths(root, dest, dest / "surface_manifest.json")


def test_temporary_surface_outputs_are_rejected():
    with pytest.raises(ValueError, match="persistent"):
        validate_output_paths(Path("/workspace/hipp"), Path("/tmp/surfaces"), Path("/tmp/surfaces/manifest.json"))


def test_historical_manifest_with_temporary_entry_is_rejected(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"profile":"iowa_fsLR_complete_roster_v1","entries":[{"subject":"sub-1","hemisphere":"L","surface":"/tmp/old.surf.gii","sha256":"x"}]}')
    with pytest.raises(ValueError, match="temporary"):
        apply_manifest([], manifest)
