from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys

import nibabel as nib
import numpy as np
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "scripts/core_workflow"))


def stage(name, folder="core_workflow"):
    path=ROOT / "scripts" / folder / name
    spec=importlib.util.spec_from_file_location("review_"+name[:-3],path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def crossfit_config(workflow,tmp_path):
    return workflow.CrossfitConfig(shared_root=tmp_path / "shared",surface_root=tmp_path / "surface",
        template_surfaces_dir=tmp_path / "templates",hippunfold_batch_root=tmp_path / "hippunfold",
        out_root=tmp_path / "out",fold_by_subject={"a":0,"b":1})


def test_crossfit_anchor_is_a_fingerprinted_input(tmp_path):
    workflow=stage("01_01_run_hcp_crossfit_template_matching.py","individual_workflow")
    config=crossfit_config(workflow,tmp_path)
    anchor=SimpleNamespace(template_dir=tmp_path / "anchor")
    anchor.template_dir.mkdir()
    anchor_file=anchor.template_dir / "group_gradients.npy";np.save(anchor_file,np.ones((8,5)))
    paths=workflow._template_input_paths(config,0,"2mm","L",alignment_reference=anchor)
    assert anchor_file.resolve() in [p.resolve() for p in paths.values()]
    for path in paths.values():
        if path == anchor_file:continue
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"input")
    before=workflow.fingerprint_files(paths)
    np.save(anchor_file,np.zeros((8,5)))
    with pytest.raises(ValueError,match="fingerprint|changed|mismatch"):
        workflow.validate_file_fingerprints(before,paths,context="alignment anchor")


def test_crossfit_resume_passes_the_requested_anchor_to_validation(tmp_path,monkeypatch):
    workflow=stage("01_01_run_hcp_crossfit_template_matching.py","individual_workflow")
    config=crossfit_config(workflow,tmp_path)
    path=workflow._fold_template_dir(config.out_root,0,"2mm","L") / "template_manifest.json"
    path.parent.mkdir(parents=True);path.write_text("{}")
    anchor=object();seen=[]
    def load(*args,**kwargs):seen.append(kwargs.get("alignment_reference"));return "loaded"
    monkeypatch.setattr(workflow,"load_fold_template",load)
    assert workflow.fit_or_load_fold_template(config,0,"2mm","L",resume=True,alignment_reference=anchor) == "loaded"
    assert seen == [anchor]


@pytest.mark.parametrize("name",["01_03_run_and_render_hcp_crossfit_disjoint_folds.py","01_04_render_hcp_runs_comparison.py"])
def test_spatial_crossfit_displays_do_not_invent_network_assignments(name,tmp_path):
    module=stage(name,"individual_workflow")
    assert [module.DISPLAY_LABEL_SPECS[i][0] for i in (1,2,3)] == ["Anterior","Middle","Posterior"]
    path=tmp_path / "clusters.label.gii"
    module.write_display_label_gifti(np.array([1,2,3],np.int32),path)
    labels={label.key:label.label for label in nib.load(path).labeltable.labels}
    assert [labels[i] for i in (1,2,3)] == ["Anterior","Middle","Posterior"]


def test_exported_cluster_diagnostics_follow_spatial_relabeling():
    from lib import gradient_clustering as gc
    old=np.array([2]*49+[1]+[3]*50,np.int32)
    new=np.array([1]*49+[2]+[3]*50,np.int32)
    graph=sparse.diags([np.ones(99),np.ones(99)],[-1,1],shape=(100,100)).tocsr()
    diagnostics=gc.cluster_mesh_diagnostics(old,graph,k=3)
    corrected=gc.spatial_cluster_diagnostics(diagnostics,old,new,graph,k=3)
    assert corrected["cluster_vertex_counts"] == {"1":49,"2":1,"3":50}
    assert corrected["small_clusters"] == [2]
    assert corrected["model_cluster_to_spatial_cluster"] == {"1":2,"2":1,"3":3}


@pytest.mark.parametrize("n_networks", [7, 17])
def test_ovr_overview_bar_panel_omits_the_hemisphere_separator(n_networks):
    overview = stage("06_01_build_group_overview.py")
    import matplotlib.pyplot as plt
    left = np.repeat(np.array([[1.], [2.], [3.]]), n_networks, axis=1)
    right = np.repeat(np.array([[4.], [5.], [6.]]), n_networks, axis=1)
    labels = ["L Anterior", "L Middle", "L Posterior", "R Anterior", "R Middle", "R Posterior"]
    profile, positions, row_labels = overview.concat_lr(left, labels[:3], right, labels[3:])
    fig, ax = plt.subplots()
    try:
        overview.plot_ovr_association_panel(
            ax, profile=profile, row_positions=positions, row_labels=row_labels, smoothing="2mm"
        )
        assert [bar.get_height() for bar in ax.patches] == [1., 2., 3., 4., 5., 6.]
        assert [tick.get_text() for tick in ax.get_xticklabels()] == labels
        assert ax.get_ylabel() == "Largest raw beta"
    finally:
        plt.close(fig)
