from pathlib import Path
import tempfile
from importlib import import_module

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_multidataset_cluster_stacked_plots_run():
    pos_mod = import_module("scripts.visualize.multidataset_cluster_confidence_stacked.plot_multidataset_cluster_positive_stacked")
    neg_mod = import_module("scripts.visualize.multidataset_cluster_confidence_negative_stacked.plot_multidataset_cluster_negative_stacked")

    assert hasattr(pos_mod, "render_multidataset_cluster_positive_stacked")
    assert hasattr(neg_mod, "render_multidataset_cluster_negative_stacked")

    with tempfile.TemporaryDirectory() as tmp_dir:
        pos_out = Path(tmp_dir) / "pos_test.png"
        neg_out = Path(tmp_dir) / "neg_test.png"

        pos_mod.render_multidataset_cluster_positive_stacked(pos_out)
        neg_mod.render_multidataset_cluster_negative_stacked(neg_out)

        assert pos_out.exists() and pos_out.stat().st_size > 10000
        assert neg_out.exists() and neg_out.stat().st_size > 10000


def test_render_multidataset_workbench_hippocampus_overlay_runs():
    wb_mod = import_module("scripts.visualize.multidataset_workbench_hippocampus_overlay.render_multidataset_workbench_hippocampus_overlay")
    assert hasattr(wb_mod, "render_all_cohorts_overlay_figure")

    with tempfile.TemporaryDirectory() as tmp_dir:
        wb_out = Path(tmp_dir) / "wb_test.png"
        tsv_out = Path(tmp_dir) / "wb_test.tsv"

        wb_mod.render_all_cohorts_overlay_figure(wb_out, tsv_out)

        assert wb_out.exists() and wb_out.stat().st_size > 50000
        assert tsv_out.exists() and tsv_out.stat().st_size > 1000
