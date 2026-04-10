from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = REPO_ROOT / "scripts" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from hipp_density_assets import (  # noqa: E402
    DensityAssetError,
    find_surface_asset_strict,
    subject_surf_dir,
)


class HippDensityAssetsStrictTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _touch(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")

    def test_find_surface_asset_success_with_den(self) -> None:
        surf_dir = self.root / "outputs" / "sub-100610" / "surf"
        target = surf_dir / "sub-100610_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii"
        self._touch(target)
        got = find_surface_asset_strict(
            surf_dir=surf_dir,
            subject="100610",
            hemi="L",
            space="corobl",
            density="512",
            suffix="midthickness.surf.gii",
        )
        self.assertEqual(got, target)

    def test_legacy_without_den_rejected(self) -> None:
        surf_dir = self.root / "outputs" / "sub-100610" / "surf"
        self._touch(surf_dir / "sub-100610_hemi-L_space-corobl_label-hipp_midthickness.surf.gii")
        with self.assertRaises(DensityAssetError) as ctx:
            find_surface_asset_strict(
                surf_dir=surf_dir,
                subject="100610",
                hemi="L",
                space="corobl",
                density="512",
                suffix="midthickness.surf.gii",
            )
        msg = str(ctx.exception)
        self.assertIn("Input density = 512", msg)
        self.assertIn("den-512", msg)

    def test_mixed_density_rejected(self) -> None:
        surf_dir = self.root / "outputs" / "sub-100610" / "surf"
        self._touch(surf_dir / "sub-100610_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii")
        self._touch(surf_dir / "sub-100610_hemi-L_space-corobl_den-2mm_label-hipp_midthickness.surf.gii")
        with self.assertRaises(DensityAssetError) as ctx:
            find_surface_asset_strict(
                surf_dir=surf_dir,
                subject="100610",
                hemi="L",
                space="corobl",
                density="512",
                suffix="midthickness.surf.gii",
            )
        self.assertIn("Mixed or mismatched density assets", str(ctx.exception))

    def test_density_mismatch_rejected(self) -> None:
        surf_dir = self.root / "outputs" / "sub-100610" / "surf"
        self._touch(surf_dir / "sub-100610_hemi-L_space-corobl_den-2mm_label-hipp_midthickness.surf.gii")
        with self.assertRaises(DensityAssetError) as ctx:
            find_surface_asset_strict(
                surf_dir=surf_dir,
                subject="100610",
                hemi="L",
                space="corobl",
                density="512",
                suffix="midthickness.surf.gii",
            )
        self.assertIn("requested density=512", str(ctx.exception))

    def test_subject_surf_dir_rejects_noncanonical_work_only(self) -> None:
        hippunfold_root = self.root / "outputs" / "dense_corobl_batch"
        self._touch(hippunfold_root / "work" / "sub-100610" / "surf" / "dummy.txt")
        with self.assertRaises(DensityAssetError) as ctx:
            subject_surf_dir(hippunfold_root, "100610")
        self.assertIn("Only <hippunfold-dir>/sub-<id>/surf is allowed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
