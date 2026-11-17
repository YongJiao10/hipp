# Cortical FC heatmaps and chord diagrams

This module renders cortical ROI-to-ROI functional connectivity (FC) from
explicit files. It has no dataset, subject, cohort, acquisition or preprocessing
defaults. The caller selects the input and output paths. It does not select
another input when a requested file is absent.

`fc_visualization.py` contains the reusable heatmap, chord and combined-layout
functions. `render_fc.py` is the generic command-line entrypoint. Run from the
repository root with the project Python interpreter.

## Inputs

- `--fc-matrix`: an `(n_rois, n_rois)` NPY matrix. Off-diagonal values are used
  unchanged; `--colorbar-title` describes their units.
- Alternatively, `--timeseries`: an `(n_rois, n_timepoints)` NPY array. Each ROI
  pair's Pearson correlation is clipped to `[-0.9999, 0.9999]` and converted to
  Fisher-z, preserving the previous cortex FC plotting definition.
- `--parcels`: a TSV whose rows match the array's ROI order exactly. Required
  columns are `parcel_id`, `hemisphere` (`L`/`R` or `LH`/`RH`), `family17`, and
  the selected display column: `canonical7`, `canonical8`, or `family17`.

The current taxonomy views are Kong2022 canonical7 (`--net-type 7net`, default),
canonical8 (`8net`) and native17 (`17net`), independently of which dataset
produced the inputs. ROI ordering is anchored to the native17 families. Labels
and colors use the project's network keys and `config/hipp_network_style.json`;
unknown network colors raise an error.

The diagonal is excluded from plotting and network means. Each chord value is
the mean of the corresponding ROI-pair FC values, including within-network
connections. `--view combined` (default) places the network chord diagram in the
upper-right of the ROI heatmap; `heatmap` and `chord` render separate views.

## Usage

```bash
/home/yojiao/.conda/envs/py314/bin/python -m scripts.visualize.cortex_fc.render_fc \
  --timeseries /absolute/path/to/cortex_roi_parcel_timeseries.npy \
  --parcels /absolute/path/to/cortex_roi_parcels.tsv \
  --net-type 7net --title 'Kong2022 canonical7 cortical FC' \
  --output /absolute/path/to/figures/cortex_fc_canonical7.png

/home/yojiao/.conda/envs/py314/bin/python -m scripts.visualize.cortex_fc.render_fc \
  --fc-matrix /absolute/path/to/cortex_fc.npy \
  --parcels /absolute/path/to/cortex_roi_parcels.tsv \
  --net-type 17net --view heatmap --light-background \
  --title 'Kong2022 native17 cortical FC' --colorbar-title 'Pearson r' \
  --output /absolute/path/to/figures/cortex_fc_native17.png
```

Specify one exact PNG filename per invocation. The renderer creates its parent
directory. Different datasets, preprocessing variants, acquisitions and
taxonomies retain their own output paths; it does not relocate existing figures.

Python callers can import `plot_combined_fc`, `plot_fc_heatmap`, `plot_chord`
and `compute_network_fc` from
`scripts.visualize.cortex_fc.fc_visualization`.
