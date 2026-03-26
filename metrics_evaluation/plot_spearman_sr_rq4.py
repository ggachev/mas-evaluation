"""
Abbildung 5.13: Spearman-Validitätsvergleich SR=1 vs. SR=5 (GPT-OSS-120b).

Zeigt den Spearman-Korrelationskoeffizienten (vs. manueller Gold Standard)
je Metrik für beide Sampling-Konfigurationen. Nur GPT-OSS-120b.

Output: evaluation_results/spearman_sr_rq4.png + .pgf
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

EVAL_DIR  = Path(__file__).parent / 'evaluation_results'
SR1_DIR   = EVAL_DIR / '1_step_gptoss120b'   # SR=1  → Standard
SR5_DIR   = EVAL_DIR / 'default_gptoss120b'  # SR=5  → Reduziert
OUTPUT_DIR = EVAL_DIR

METRICS = ['M2.2', 'M2.3', 'M2.4', 'M2.5', 'M3.1', 'M3.2', 'M4.1', 'M5.1']

METRIC_LABELS = {
    'M2.2': 'M2.2\nTrajectory\nEfficiency',
    'M2.3': 'M2.3\nGlobal\nStrategy',
    'M2.4': 'M2.4\nReasoning\nQuality',
    'M2.5': 'M2.5\nRole\nAdherence',
    'M3.1': 'M3.1\nTool\nSelection',
    'M3.2': 'M3.2\nTool\nExecution',
    'M4.1': 'M4.1\nContext\nUtilization',
    'M5.1': 'M5.1\nComm.\nEfficiency',
}


def load_spearman(csv_path: Path) -> tuple[dict, dict]:
    """Lädt ρ-Werte und Signifikanzniveaus aus einer Spearman-CSV."""
    df = pd.read_csv(csv_path)
    rho  = dict(zip(df['Metric'], df['Spearman_Rho']))
    sig  = dict(zip(df['Metric'], df['Significance']))
    return rho, sig


def annotate_bars(ax, bars, sigs):
    """Zeichnet Signifikanz-Sterne über die Balken."""
    for bar, s in zip(bars, sigs):
        if s and s != 'n.s.':
            y = max(bar.get_height(), 0) + 0.025
            ax.annotate(
                s,
                xy=(bar.get_x() + bar.get_width() / 2, y),
                ha='center', va='bottom', fontsize=13, fontweight='bold',
            )


def main():
    rho_sr1, sig_sr1 = load_spearman(SR1_DIR / 'spearman_correlation_results.csv')
    rho_sr5, sig_sr5 = load_spearman(SR5_DIR / 'spearman_correlation_results.csv')

    vals_sr1 = [rho_sr1.get(m, np.nan) for m in METRICS]
    vals_sr5 = [rho_sr5.get(m, np.nan) for m in METRICS]
    sigs_sr1 = [sig_sr1.get(m, '')     for m in METRICS]
    sigs_sr5 = [sig_sr5.get(m, '')     for m in METRICS]

    x     = np.arange(len(METRICS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 7))

    bars1 = ax.bar(
        x - width / 2, vals_sr1, width,
        label='Standard (SR=1)', color='#1f77b4', alpha=0.85, edgecolor='black',
    )
    bars2 = ax.bar(
        x + width / 2, vals_sr5, width,
        label='Reduziert (SR=5)', color='#ff7f0e', alpha=0.85, edgecolor='black',
    )

    annotate_bars(ax, bars1, sigs_sr1)
    annotate_bars(ax, bars2, sigs_sr5)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=12)
    ax.set_ylabel('Spearman rho (vs. manueller Gold Standard)', fontsize=13)
    ax.axhline(y=0.6, color='green',  linestyle='--', alpha=0.5, linewidth=1.2,
               label='Stark (0,60)')
    ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, linewidth=1.2,
               label='Moderat (0,40)')
    ax.axhline(y=0.0, color='black',  linestyle='-',  alpha=0.25, linewidth=0.8)
    ax.set_ylim(-0.35, 1.05)
    ax.tick_params(axis='y', labelsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda val, _: f'{val:.1f}'.replace('.', ',')
    ))
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    name = 'spearman_sr_rq4'
    fig.savefig(OUTPUT_DIR / f'{name}.png', dpi=150, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f'{name}.pdf', bbox_inches='tight')
    (OUTPUT_DIR / f'{name}.tex').write_text(
        r'\includegraphics[width=\linewidth]{' + name + r'.pdf}' + '\n',
        encoding='utf-8',
    )
    try:
        fig.savefig(OUTPUT_DIR / f'{name}.pgf', bbox_inches='tight')
        print(f"Gespeichert: {name}.png + .pdf + .pgf + .tex")
    except Exception as e:
        print(f"Gespeichert: {name}.png + .pdf + .tex  (PGF fehlgeschlagen: {e})")

    plt.close()


if __name__ == '__main__':
    main()
