import subprocess
import sys
from pathlib import Path

from scripts.iowa7t.audit_branch_completion import audit_jobs


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok", encoding="utf-8")


def test_audit_reports_missing_noxcpd_overview_when_36p_counterpart_exists(tmp_path: Path) -> None:
    source = tmp_path / "DataOrg_PFM_36P" / "hipp_outputs"
    target = tmp_path / "DataOrg_PFM_NoXCPD" / "hipp_outputs"
    rel = Path("subjects/sub-06112026/multiband/figures/gradient/sub-06112026_gradient_fisherz_overview.png")
    _touch(source / rel)

    summary = audit_jobs(
        jobs=[("06112026", "multiband")],
        output_root=target,
        compare_output_root=source,
    )

    assert summary["ok"] is False
    assert summary["jobs"][0]["missing_overviews"] == [str(target / rel)]


def test_audit_passes_when_noxcpd_overview_inventory_matches_36p(tmp_path: Path) -> None:
    source = tmp_path / "DataOrg_PFM_36P" / "hipp_outputs"
    target = tmp_path / "DataOrg_PFM_NoXCPD" / "hipp_outputs"
    rels = [
        Path("subjects/sub-06112026/multiband/figures/gradient/sub-06112026_gradient_fisherz_overview.png"),
        Path("subjects/sub-06112026/multiband/figures/wta/sub-06112026_wta_overview.png"),
    ]
    for rel in rels:
        _touch(source / rel)
        _touch(target / rel)

    summary = audit_jobs(
        jobs=[("06112026", "multiband")],
        output_root=target,
        compare_output_root=source,
    )

    assert summary["ok"] is True
    assert summary["jobs"][0]["expected_overviews"] == [str(source / rel) for rel in rels]
    assert summary["jobs"][0]["missing_overviews"] == []


def test_audit_cli_runs_when_invoked_by_script_path(tmp_path: Path) -> None:
    source = tmp_path / "DataOrg_PFM_36P" / "hipp_outputs"
    target = tmp_path / "DataOrg_PFM_NoXCPD" / "hipp_outputs"
    rel = Path("subjects/sub-06112026/multiband/figures/gradient/sub-06112026_gradient_fisherz_overview.png")
    _touch(source / rel)
    _touch(target / rel)
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/iowa7t/audit_branch_completion.py",
            "--output-root",
            str(target),
            "--compare-output-root",
            str(source),
            "--job",
            "06112026:multiband",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"ok": true' in result.stdout
