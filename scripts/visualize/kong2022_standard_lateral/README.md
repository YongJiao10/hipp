# Standard Cortical Lateral / Medial Surface Visualizations (Kong2022 & Lynch2024)

This directory contains high-resolution cortical surface visualizations for group standard templates of **Kong2022** and **Lynch2024** functional parcellations on the standard `fsLR-32k` cortical mesh.

## Visualizations Included

1. **Master 2x2 Comparison Figure**:
   - File: `kong_lynch_native_canonical7_comparison.png` (3250 × 2109 px)
   - Layout:
     - **Row 1 (Kong2022)**:
       - **Col 1**: Kong2022 Native 17-Network (Schaefer 400 Parcels)
       - **Col 2**: Kong2022 Merged Canonical 7-Network
     - **Row 2 (Lynch2024)**:
       - **Col 1**: Lynch2024 Native 20-Network (FASTANS PFM Standard Prior)
       - **Col 2**: Lynch2024 Merged Canonical 7-Network
   - Shows 4 views per panel: Left Hemisphere Lateral, Right Hemisphere Lateral, Left Hemisphere Medial, Right Hemisphere Medial.
   - Each condition features its dedicated color legend with constituent sub-network annotations.

2. **Standalone Surface Panels** (1560 × 900 px, matching standard lateral 4-view layout):
   - `kong2022_standard_lateral.png`: Kong2022 17-Network standard group parcellation (original reference).
   - `kong2022_standard_canonical7.png`: Kong2022 Merged Canonical 7-Network standard group parcellation.
   - `lynch2024_standard_20net.png`: Lynch2024 20-Network standard prior parcellation.
   - `lynch2024_standard_canonical7.png`: Lynch2024 Merged Canonical 7-Network standard group parcellation.

## Reproducibility & Regeneration

To regenerate all figures:
```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/kong2022_standard_lateral/render_kong_lynch_native_canonical7.py
```

Options:
- `--out-dir`: Output directory (default: `scripts/visualize/kong2022_standard_lateral`).
- `--renderer`: Workbench scene renderer (default: `OSMesa`).
