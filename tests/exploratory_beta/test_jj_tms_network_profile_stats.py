import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/exploratory_beta'))
import render_jj_tms_five_figures as jj_wrapper
import stanford_tms_five_figures as renderer


def synthetic_healthy_profile():
    values = {
        1: {
            'VentralAttention': [3.0, 4.0, 5.0],
            'Default': [1.0, 1.1, 1.2],
            'Control': [1.0, 1.1, 1.2],
        },
        2: {
            'VentralAttention': [0.0, 0.1, 0.2],
            'Default': [0.0, 0.1, 0.2],
            'Control': [0.0, 0.1, 0.2],
        },
        3: {
            'VentralAttention': [-5.0, -4.0, -3.0],
            'Default': [1.0, 1.1, 1.2],
            'Control': [1.0, 1.1, 1.2],
        },
    }
    rows = []
    for cluster, by_network in values.items():
        for network, beta_values in by_network.items():
            for index, beta_mean in enumerate(beta_values):
                rows.append({
                    'subject': f'{network[:2]}{index}',
                    'group': 'healthy',
                    'site_network': network,
                    'hippocampal_hemi': 'L',
                    'cluster': cluster,
                    'parcellation': 'hcp',
                    'beta_mean': beta_mean,
                })
    return pd.DataFrame(rows)


def test_jj_tms_figures_use_iowa_3t_as_display_label():
    assert jj_wrapper.JJ_DATASET_LABEL == 'Iowa 3T'


def test_van_network_comparisons_are_uncorrected_and_encode_direction():
    comparisons = renderer.van_network_comparisons(synthetic_healthy_profile())

    assert len(comparisons) == 6
    assert set(comparisons['comparator']) == {'Default', 'Control'}
    assert set(comparisons['correction']) == {'none'}
    assert set(comparisons['method']) == {'Welch independent-samples t-test'}

    higher = comparisons.loc[comparisons['cluster'] == 1]
    assert set(higher['direction']) == {'van_higher'}
    assert set(higher['marker_color']) == {'#C62828'}
    assert set(higher['significance']) == {'*'}

    nonsignificant = comparisons.loc[comparisons['cluster'] == 2]
    assert set(nonsignificant['direction']) == {'not_significant'}
    assert set(nonsignificant['significance']) == {'ns'}
    assert set(nonsignificant['marker_color']) == {''}

    lower = comparisons.loc[comparisons['cluster'] == 3]
    assert set(lower['direction']) == {'van_lower'}
    assert set(lower['marker_color']) == {'#1565C0'}
    assert set(lower['significance']) == {'*'}


def test_van_annotation_draws_only_significant_comparisons_with_direction_color():
    fig, ax = plt.subplots()
    ax.set_ylim(0, 10)
    comparisons = [
        dict(cluster=1, comparator='Control', significance='*', marker_color='#C62828'),
        dict(cluster=1, comparator='Default', significance='ns', marker_color=''),
        dict(cluster=2, comparator='Control', significance='**', marker_color='#1565C0'),
    ]

    renderer.annotate_van_comparisons(ax, comparisons, renderer.PROFILE_NETWORK_POSITIONS)

    assert len(ax.lines) == 2
    assert {line.get_color() for line in ax.lines} == {'#C62828', '#1565C0'}
    assert [text.get_text() for text in ax.texts] == ['*', '**']
    plt.close(fig)


def test_save_pair_is_png_only_unless_pdf_is_requested(tmp_path):
    fig = plt.figure()
    paths = renderer.save_pair(fig, tmp_path, 'profile')
    assert paths == {'png': str(tmp_path / 'profile.png')}
    assert (tmp_path / 'profile.png').is_file()
    assert not (tmp_path / 'profile.pdf').exists()

    fig = plt.figure()
    paths = renderer.save_pair(fig, tmp_path, 'profile_with_pdf', pdf=True)
    assert paths == {
        'png': str(tmp_path / 'profile_with_pdf.png'),
        'pdf': str(tmp_path / 'profile_with_pdf.pdf'),
    }
    assert (tmp_path / 'profile_with_pdf.pdf').is_file()
