# HCP 7T Hippocampus Gradient and Silhouette Confidence Map Replotting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recompute HippoMaps continuous functional gradients, group atlas parcellation, and vertex-level silhouette confidence maps for HCP 7T (N=166, latest QC <=10 unusable vertices) under PFM 2 mm and 4 mm smoothings, and render presentation-quality PNG figures under `outputs_hcp/present/gradients/pfm/`.

**Architecture:** Use the canonical workflow scripts in strict sequential order: `05_01_compute_hippomaps_gradients.py` generates continuous gradients G1-G3, `05_02_run_group_atlas.py` fits K=3 group atlas and exports active gradient features, `05_04_compute_confidence_maps.py` evaluates vertex-level silhouette confidence values in G1-G3 space, and `06_04_render_hippomaps_gradient_products.py` plus `06_06_render_confidence_maps.py` render standard folded/unfolded surface PNGs and diagnostic figures.

**Tech Stack:** Python 3.14 (`/home/yojiao/.conda/envs/py314/bin/python`), NumPy, SciPy, BrainSpace, scikit-learn, Nibabel, Matplotlib, PIL, and Connectome Workbench via HippoMaps adapter.

---

### Task 1: Pre-flight Verification of Inputs and Environment

**Files:**
- Reference: `outputs_hcp/inputs/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt`
- Reference: `outputs_hcp/inputs/kong2022/templates/hipp_group_average/group_average_surfaces/`
- Reference: `outputs_hcp/inputs/kong2022/shared_inputs/pfm/`

- [ ] **Step 1: Check Python environment and dependencies**

Run:
```bash
/home/yojiao/.conda/envs/py314/bin/python -c "import sys, numpy, scipy, sklearn, brainspace, nibabel; print('Python:', sys.executable); print('All core modules imported successfully.')"
```
Expected: Python interpreter is `/home/yojiao/.conda/envs/py314/bin/python` and modules import cleanly.

- [ ] **Step 2: Verify QC roster count and sample subject inputs**

Run:
```bash
/home/yojiao/.conda/envs/py314/bin/python -c "
from pathlib import Path
manifest = Path('outputs_hcp/inputs/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt')
subjects = [line.strip() for line in manifest.read_text().splitlines() if line.strip()]
assert len(subjects) == 166, f'Expected 166 subjects, got {len(subjects)}'
for sub in subjects[:5]:
    for sm in ('2mm', '4mm'):
        for hemi in ('L', 'R'):
            fc = Path(f'outputs_hcp/inputs/kong2022/shared_inputs/pfm/{sub}/fc/kong2022/{sm}/hemi_{hemi}/hipp_vertex_to_roi_fc_active.npy')
            assert fc.exists(), f'Missing {fc}'
print(f'Verified QC pass roster (N={len(subjects)}) and input FC files.')
"
```
Expected: "Verified QC pass roster (N=166) and input FC files."

- [ ] **Step 3: Commit verification step**

```bash
git status --short
```

---

### Task 2: Compute Continuous Hippocampal Gradients (`05_01`)

**Files:**
- Script: `scripts/core_workflow/05_01_compute_hippomaps_gradients.py`
- Outputs: `outputs_hcp/work/hippomaps_gradients/pfm/<2mm|4mm>/hemi_<L|R>/`

- [ ] **Step 1: Run 05_01 for PFM 2mm and 4mm**

Run:
```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_01_compute_hippomaps_gradients.py \
  --routes pfm \
  --smoothings 2mm 4mm \
  --hemis L R \
  --n-components 5 \
  --selected-components 1,2,3 \
  --gradient-random-state 0 \
  --out-root outputs_hcp/work/hippomaps_gradients
```
Expected: Exits with code 0; generates aligned G1, G2, G3 gradient arrays for 2mm and 4mm (hemi_L and hemi_R).

- [ ] **Step 2: Verify gradient outputs**

Run:
```bash
/home/yojiao/.conda/envs/py314/bin/python -c "
from pathlib import Path
import json, numpy as np
root = Path('outputs_hcp/work/hippomaps_gradients/pfm')
for sm in ('2mm', '4mm'):
    for hemi in ('L', 'R'):
        hdir = root / sm / f'hemi_{hemi}'
        mf = hdir / 'hippomaps_gradient_manifest.json'
        assert mf.exists(), f'Missing manifest in {hdir}'
        data = json.loads(mf.read_text())
        assert len(data['subjects']) == 166, f'Expected 166 subjects in {mf}'
        act = np.load(hdir / 'hippomaps_gradients_active.npy')
        assert act.shape[1] == 5, f'Expected 5 components, got shape {act.shape}'
        ev = np.load(hdir / 'hippomaps_ev_fractions.npy')
        print(f'{sm} hemi-{hemi}: active_shape={act.shape}, EV={ev[:3]}')
print('Gradients verification passed.')
"
```
Expected: "Gradients verification passed." with valid shapes and EV fractions.

---

### Task 3: Build Group Atlas and Extract Gradient Features (`05_02`)

**Files:**
- Script: `scripts/core_workflow/05_02_run_group_atlas.py`
- Outputs: `outputs_hcp/work/group_atlas/pfm/gradient-spectral/kong2022/<2mm|4mm>/cov_0p80/g1g2g3/hemi_<L|R>/`

- [ ] **Step 1: Run 05_02 for PFM 2mm and 4mm**

Run:
```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_02_run_group_atlas.py \
  --shared-root outputs_hcp/inputs/kong2022/shared_inputs/pfm \
  --surface-root outputs_hcp/inputs/kong2022/shared_inputs/surface \
  --reference-root outputs_hcp/inputs/kong2022/shared_inputs/pfm \
  --out-root outputs_hcp/work/group_atlas/pfm \
  --methods gradient-spectral \
  --atlases kong2022 \
  --gradient-root outputs_hcp/work/hippomaps_gradients \
  --route-tag pfm \
  --network-taxonomy canonical7 \
  --smoothings 2mm 4mm \
  --k-min 2 --k-max 8 --default-k 3 --force-final-k 3
```
Expected: Exits with code 0; fits K=3 parcellations and exports `group_gradient_features_active.npy` and `group_labels_final.npy`.

- [ ] **Step 2: Verify group atlas outputs**

Run:
```bash
/home/yojiao/.conda/envs/py314/bin/python -c "
from pathlib import Path
import json, numpy as np
root = Path('outputs_hcp/work/group_atlas/pfm/gradient-spectral/kong2022')
for sm in ('2mm', '4mm'):
    for hemi in ('L', 'R'):
        rdir = root / sm / 'cov_0p80' / 'g1g2g3' / f'hemi_{hemi}'
        lbl = np.load(rdir / 'group_labels_final.npy')
        feat = np.load(rdir / 'group_gradient_features_active.npy')
        mask = np.load(rdir / 'group_support_mask.npy')
        assert np.sum(mask) == len(feat), f'Mismatch in feature/mask vertices'
        unique_lbls = np.unique(lbl[mask])
        assert set(unique_lbls) == {1, 2, 3}, f'Expected clusters {1, 2, 3}, got {unique_lbls}'
        print(f'{sm} hemi-{hemi}: active_vertices={len(feat)}, clusters={unique_lbls}')
print('Group atlas verification passed.')
"
```
Expected: "Group atlas verification passed." with clusters {1, 2, 3}.

---

### Task 4: Compute Silhouette Confidence Maps (`05_04`)

**Files:**
- Script: `scripts/core_workflow/05_04_compute_confidence_maps.py`
- Outputs: `outputs_hcp/work/group_atlas/pfm/gradient-spectral/kong2022/<2mm|4mm>/cov_0p80/g1g2g3/hemi_<L|R>/confidence_silhouette_gradient_feature*`

- [ ] **Step 1: Run 05_04 for 2mm and 4mm**

Run:
```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_04_compute_confidence_maps.py \
  --out-root outputs_hcp/work/group_atlas/pfm \
  --methods gradient-spectral \
  --smoothings 2mm 4mm \
  --hemis L R \
  --coverage-thresholds 0.80 \
  --gradient-components 1,2,3 \
  --metric cosine
```
Expected: Exits with code 0; outputs `.npy` arrays, summaries, and manifests for both 2mm and 4mm (L and R).

- [ ] **Step 2: Verify silhouette confidence computations**

Run:
```bash
/home/yojiao/.conda/envs/py314/bin/python -c "
from pathlib import Path
import json, numpy as np
root = Path('outputs_hcp/work/group_atlas/pfm/gradient-spectral/kong2022')
for sm in ('2mm', '4mm'):
    for hemi in ('L', 'R'):
        rdir = root / sm / 'cov_0p80' / 'g1g2g3' / f'hemi_{hemi}'
        sil_npy = rdir / 'confidence_silhouette_gradient_feature.npy'
        sil_sum = rdir / 'confidence_silhouette_gradient_feature_summary.json'
        assert sil_npy.exists(), f'Missing {sil_npy}'
        assert sil_sum.exists(), f'Missing {sil_sum}'
        summary = json.loads(sil_sum.read_text())
        print(f'{sm} hemi-{hemi} mean silhouette: {summary[\"global_mean\"]:.4f}')
print('Confidence map calculation verified.')
"
```
Expected: "Confidence map calculation verified." with valid mean silhouette values.

---

### Task 5: Render Continuous Gradient Figures (`06_04`)

**Files:**
- Script: `scripts/core_workflow/06_04_render_hippomaps_gradient_products.py`
- Outputs: `outputs_hcp/present/gradients/pfm/<2mm|4mm>/`

- [ ] **Step 1: Run 06_04 for PFM 2mm and 4mm**

Run:
```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/06_04_render_hippomaps_gradient_products.py \
  --gradient-root outputs_hcp/work/hippomaps_gradients \
  --present-root outputs_hcp/present/gradients \
  --routes pfm \
  --smoothings 2mm 4mm \
  --hemis L R \
  --components 1,2,3
```
Expected: Exits with code 0; renders bilateral and hemi-specific gradient surface maps, explained variance bar plots, and endpoint FC contrast figures.

- [ ] **Step 2: Verify rendered gradient PNGs**

Run:
```bash
/home/yojiao/.conda/envs/py314/bin/python -c "
from pathlib import Path
from PIL import Image
pres_root = Path('outputs_hcp/present/gradients/pfm')
for sm in ('2mm', '4mm'):
    sm_dir = pres_root / sm
    expected = [
        f'hippomaps_gradients_pfm_{sm}_bilateral_g1g2g3.png',
        f'hippomaps_gradients_pfm_{sm}_bilateral_g1g2g3_hemi-L.png',
        f'hippomaps_gradients_pfm_{sm}_bilateral_g1g2g3_hemi-R.png',
        f'hippomaps_ev_pfm_{sm}_bilateral.png',
        f'hippomaps_endpoint_fc_contrast_pfm_{sm}_bilateral.png',
    ]
    for fn in expected:
        fp = sm_dir / fn
        assert fp.exists(), f'Missing {fp}'
        assert fp.stat().st_size > 1000, f'File too small: {fp}'
        with Image.open(fp) as img:
            print(f'{fn}: size={img.size}, bytes={fp.stat().st_size}')
print('Gradient presentation figures verified.')
"
```
Expected: "Gradient presentation figures verified." with positive dimensions and file sizes.

---

### Task 6: Render Silhouette Confidence Figures (`06_06`)

**Files:**
- Script: `scripts/core_workflow/06_06_render_confidence_maps.py`
- Outputs: `outputs_hcp/present/gradients/pfm/<2mm|4mm>/`

- [ ] **Step 1: Run 06_06 for PFM 2mm and 4mm**

Run:
```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/06_06_render_confidence_maps.py \
  --out-root outputs_hcp/work/group_atlas/pfm \
  --present-root outputs_hcp/present/gradients \
  --route-tag pfm \
  --smoothings 2mm 4mm \
  --coverage-thresholds 0.80 \
  --gradient-components 1,2,3 \
  --method gradient-spectral
```
Expected: Exits with code 0; renders bilateral and hemi-specific silhouette confidence maps.

- [ ] **Step 2: Verify rendered silhouette confidence PNGs**

Run:
```bash
/home/yojiao/.conda/envs/py314/bin/python -c "
from pathlib import Path
from PIL import Image
pres_root = Path('outputs_hcp/present/gradients/pfm')
for sm in ('2mm', '4mm'):
    sm_dir = pres_root / sm
    expected = [
        f'confidence_silhouette_gradient_feature_pfm_{sm}_bilateral.png',
        f'confidence_silhouette_gradient_feature_pfm_{sm}_bilateral_hemi-L.png',
        f'confidence_silhouette_gradient_feature_pfm_{sm}_bilateral_hemi-R.png',
    ]
    for fn in expected:
        fp = sm_dir / fn
        assert fp.exists(), f'Missing {fp}'
        assert fp.stat().st_size > 1000, f'File too small: {fp}'
        with Image.open(fp) as img:
            print(f'{fn}: size={img.size}, bytes={fp.stat().st_size}')
print('Confidence map figures verified.')
"
```
Expected: "Confidence map figures verified." with positive dimensions and file sizes.

---

### Task 7: Final Comprehensive Verification and Summary Reporting

**Files:**
- Output directory: `outputs_hcp/present/gradients/pfm/`

- [ ] **Step 1: Run end-to-end output audit**

Run:
```bash
/home/yojiao/.conda/envs/py314/bin/python -c "
from pathlib import Path
pres_root = Path('outputs_hcp/present/gradients/pfm')
all_pngs = sorted(list(pres_root.glob('*/*.png')))
print(f'Total generated presentation figures: {len(all_pngs)}')
for p in all_pngs:
    print(' -', p.relative_to(pres_root.parent), f'({p.stat().st_size / 1024:.1f} KB)')
assert len(all_pngs) >= 16, f'Expected at least 16 PNGs across 2mm and 4mm, found {len(all_pngs)}'
print('All outputs successfully generated and verified.')
"
```
Expected: "All outputs successfully generated and verified." listing 16 PNGs.

- [ ] **Step 2: Review working tree and commit implementation plan**

Run:
```bash
git add docs/superpowers/plans/2026-11-06-hcp7t-gradient-confidence-replot.md
git commit -m "docs(plans): add HCP 7T gradient and silhouette confidence replot implementation plan"
```
Expected: Clean git commit of the implementation plan.
