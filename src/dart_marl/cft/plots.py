"""CFT quality diagnostics: pointwise calibration is not choice quality."""
from __future__ import annotations

import json
from pathlib import Path


def plot_cft(report_path: str | Path, output: str | Path) -> Path:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = json.loads(Path(report_path).read_text())
    metrics = report['metrics']
    labels = ['Train', 'Validation', 'Test']
    keys = ['train', 'val', 'test']
    x = range(3)
    learned = [metrics[k]['rmse'] for k in keys]
    analytic = [metrics[k]['analytic_rmse'] for k in keys]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2),
                                    gridspec_kw={'width_ratios':[1,1.15]})
    ax1.bar([a-.18 for a in x], analytic, .36, label='Analytic forecast', color='#7696b5')
    ax1.bar([a+.18 for a in x], learned, .36, label='CFT prediction', color='#d3873e')
    ax1.set_xticks(list(x), labels)
    ax1.set_ylabel('Raw H-step return RMSE (lower is better)')
    ax1.grid(axis='y', alpha=.2)
    ax1.legend(frameon=False, fontsize=9)
    regret_l = [metrics[k]['candidate_selection_mean_regret'] or 0 for k in keys]
    regret_a = [metrics[k]['analytic_candidate_selection_mean_regret'] or 0 for k in keys]
    ax2.bar([a-.18 for a in x], regret_a, .36, label='Analytic choice', color='#7696b5')
    ax2.bar([a+.18 for a in x], regret_l, .36, label='CFT choice', color='#d3873e')
    ax2.set_xticks(list(x), labels)
    ax2.set_ylabel('Counterfactual candidate-choice regret (lower is better)')
    ax2.grid(axis='y', alpha=.2)
    ax2.legend(frameon=False, fontsize=9)
    fig.suptitle('legacy_study CFT: prediction quality versus decision quality')
    fig.text(.5,.01, 'Each snapshot forms one paired choice group. Development diagnostic; not multi-controller evidence.',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0,.055,1,.94))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return output
