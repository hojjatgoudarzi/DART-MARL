"""Guard the public layout, version metadata, and repeatable output defaults."""
from pathlib import Path
import re

import dart_marl

ROOT = Path(__file__).resolve().parents[1]


def test_release_version_consistency():
    project = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    citation = (ROOT / 'CITATION.cff').read_text(encoding='utf-8')
    version = re.search(r'(?m)^version\s*=\s*"([^"]+)"', project).group(1)
    assert version == dart_marl.__version__
    assert re.search(r'(?m)^version:\s*(\S+)\s*$', citation).group(1) == version


def test_no_obsolete_six_rover_media_or_docs_confusion():
    media = ROOT / 'media'
    assert not (media / 'heldout_returns.png').exists()
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    assert 'media/heldout_returns.png' not in readme
    results = (ROOT / 'docs/RESULTS.md').read_text(encoding='utf-8')
    assert 'not bundled' in results
    assert '786.7 / 639.5' in results and '786.7 / 786.7' in results
    for stem in ('rover_reallocation_vs_fixed_12', 'rover_recovery_12'):
        for ext in ('.gif', '.mp4', '_trace.json'):
            assert (media / (stem+ext)).is_file()


def test_legacy_figure_defaults_do_not_overwrite_tracked_artifacts():
    script = (ROOT / 'scripts/plot_results.py').read_text(encoding='utf-8')
    assert "default='outputs/heldout_returns.png'" in script
    assert "default='media/heldout_returns.png'" not in script


def test_blue_training_plot_remains_paper_ready_and_primary():
    plot = ROOT / 'results/team12/paper_figures/training_overview_2x3.png'
    assert plot.is_file() and plot.stat().st_size > 30_000
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    assert 'results/team12/paper_figures/training_overview_2x3.png' in readme
