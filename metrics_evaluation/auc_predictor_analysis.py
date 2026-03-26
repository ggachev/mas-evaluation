#!/usr/bin/env python3
"""
RQ4 — Abschnitt 5.4.3: Minimales Prädiktor-Subset für den Aufgabenerfolg.

Berechnet die AUC (Area Under the ROC Curve) für jede LLM-Judge-Metrik
als binären Prädiktor von M1.1 (Task Success). Vergleicht ausgewählte Subsets
via Mittelwert-Kombination. Erstellt Systemrang-Tabelle (N=4 Agenten).

Ausgabe: CSV + PNG + PGF (für LaTeX-Einbindung via \\input{}).
Datenquelle: evaluation_results/1_step_gptoss120b/consolidated_results.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mticker
import warnings

warnings.filterwarnings('ignore')

# PGF-Konfiguration für LaTeX-kompatible Ausgabe
matplotlib.rcParams.update({
    'font.size': 11,
    'pgf.texsystem': 'pdflatex',
    'pgf.preamble': r'\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}',
})

# ── Konfiguration ────────────────────────────────────────────────────────────

EVAL_DIR  = Path(__file__).parent / 'evaluation_results'
INPUT_DIR = EVAL_DIR / '1_step_gptoss120b'
INPUT_CSV = INPUT_DIR / 'consolidated_results.csv'

# Nur LLM-Judge Metriken (N=60, alle Agenten); M3.3/M2.1 sind deterministisch
AUC_METRICS = ['M2.2', 'M2.4', 'M2.5', 'M3.1', 'M3.2', 'M4.1']

# Subset-Definitionen für den Vergleich (aufsteigend nach Komplexität)
SUBSETS = {
    '{M2.4}':             ['M2.4'],
    '{M2.4, M2.2}':       ['M2.4', 'M2.2'],
    '{M2.4, M2.2, M4.1}': ['M2.4', 'M2.2', 'M4.1'],
    'Alle 6 Metriken':    ['M2.2', 'M2.4', 'M2.5', 'M3.1', 'M3.2', 'M4.1'],
}

METRIC_LABELS = {
    'M2.2': 'M2.2 Trajectory Efficiency',
    'M2.4': 'M2.4 Stepwise Reasoning Quality',
    'M2.5': 'M2.5 Role Adherence',
    'M3.1': 'M3.1 Tool Selection Quality',
    'M3.2': 'M3.2 Tool Execution Success',
    'M4.1': 'M4.1 Context Utilization',
}

# Gleiche Reihenfolge und Farben wie in descriptive_analysis.py
AGENT_ORDER = ['MetaGPT', 'OpenHands', 'SWE-Agent', 'live-swe-agent']

N_BOOTSTRAP = 1000
RANDOM_SEED = 42


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def german_fmt(val: float, digits: int = 2) -> str:
    """Formatiert einen Float mit deutschem Dezimaltrennzeichen (Komma)."""
    return f"{val:.{digits}f}".replace('.', ',')


def save_fig(fig: plt.Figure, output_dir: Path, name: str) -> None:
    """Speichert eine Figure als PNG, PDF, TEX und PGF."""
    png_path = output_dir / f'{name}.png'
    pdf_path = output_dir / f'{name}.pdf'
    pgf_path = output_dir / f'{name}.pgf'
    tex_path = output_dir / f'{name}.tex'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    tex_path.write_text(
        r'\includegraphics[width=\linewidth]{' + name + r'.pdf}' + '\n',
        encoding='utf-8',
    )
    try:
        fig.savefig(pgf_path, format='pgf', bbox_inches='tight')
        print(f"  Gespeichert: {png_path.name} + {pdf_path.name} + {pgf_path.name} + {tex_path.name}")
    except Exception as e:
        print(f"  Gespeichert: {png_path.name} + {pdf_path.name} + {tex_path.name}  (PGF fehlgeschlagen: {e})")


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """Berechnet 95%-KI für AUC via Bootstrap (Percentile-Methode)."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        y_t = y_true[idx]
        y_s = y_score[idx]
        if len(np.unique(y_t)) < 2:
            continue
        aucs.append(roc_auc_score(y_t, y_s))
    aucs_arr = np.array(aucs)
    return float(np.percentile(aucs_arr, 2.5)), float(np.percentile(aucs_arr, 97.5))


# ── Teil 1: AUC pro Metrik ───────────────────────────────────────────────────

def compute_auc_per_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet AUC und 95%-KI für jede LLM-Judge-Metrik als M1.1-Prädiktor."""
    y_true = df['M1.1'].values
    rows = []
    for metric in AUC_METRICS:
        mask = df[metric].notna().values
        y_t = y_true[mask]
        y_s = df[metric].values[mask]
        auc = roc_auc_score(y_t, y_s)
        ci_lo, ci_hi = bootstrap_auc_ci(y_t, y_s)
        rows.append({
            'Metrik':        metric,
            'Beschreibung':  METRIC_LABELS[metric],
            'N':             int(mask.sum()),
            'AUC':           round(auc, 4),
            'CI_95_lower':   round(ci_lo, 4),
            'CI_95_upper':   round(ci_hi, 4),
        })
    result = pd.DataFrame(rows).sort_values('AUC', ascending=False).reset_index(drop=True)
    return result


# ── Teil 2: Subset-Vergleich ─────────────────────────────────────────────────

def compute_subset_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vergleicht AUC verschiedener Metrik-Subsets.
    Kombination: einfacher Mittelwert der normalisierten [0,1]-Scores.
    """
    y_true = df['M1.1'].values
    rows = []
    for label, metrics in SUBSETS.items():
        combined = df[metrics].mean(axis=1).values
        mask = ~np.isnan(combined)
        y_t = y_true[mask]
        y_s = combined[mask]
        auc = roc_auc_score(y_t, y_s)
        ci_lo, ci_hi = bootstrap_auc_ci(y_t, y_s)
        rows.append({
            'Subset':          label,
            'Anzahl_Metriken': len(metrics),
            'AUC':             round(auc, 4),
            'CI_95_lower':     round(ci_lo, 4),
            'CI_95_upper':     round(ci_hi, 4),
        })
    return pd.DataFrame(rows)


# ── Teil 3: System-Rang-Tabelle ──────────────────────────────────────────────

def compute_agent_ranks(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """
    Berechnet Agenten-Mittelwerte für M1.1, M2.4 und M2.2.
    Gibt Rang-Tabelle und Spearman-ρ zurück (deskriptiv, N=4).
    """
    agg = df.groupby('agent_name').agg(
        Erfolgsrate=('M1.1', 'mean'),
        M2_4=('M2.4', 'mean'),
        M2_2=('M2.2', 'mean'),
    ).reset_index()

    # Reihenfolge
    order_map = {a: i for i, a in enumerate(AGENT_ORDER)}
    agg['_order'] = agg['agent_name'].map(order_map)
    agg = agg.sort_values('_order').drop('_order', axis=1).reset_index(drop=True)

    # Ränge (1 = bester)
    agg['Rang_M1_1'] = agg['Erfolgsrate'].rank(ascending=False, method='min').astype(int)
    agg['Rang_M2_4'] = agg['M2_4'].rank(ascending=False, method='min').astype(int)
    agg['Rang_M2_2'] = agg['M2_2'].rank(ascending=False, method='min').astype(int)

    # Spearman ρ (deskriptiv, N=4)
    rho_m24, _ = spearmanr(agg['Erfolgsrate'], agg['M2_4'])
    rho_m22, _ = spearmanr(agg['Erfolgsrate'], agg['M2_2'])

    # CSV-freundliche Spaltennamen
    out = pd.DataFrame({
        'Agent':             agg['agent_name'],
        'Erfolgsrate (%)':   (agg['Erfolgsrate'] * 100).round(1),
        'M2.4 Mittelwert':   agg['M2_4'].round(3),
        'M2.2 Mittelwert':   agg['M2_2'].round(3),
        'Rang M1.1':         agg['Rang_M1_1'],
        'Rang M2.4':         agg['Rang_M2_4'],
        'Rang M2.2':         agg['Rang_M2_2'],
    })
    return out, round(float(rho_m24), 3), round(float(rho_m22), 3)


# ── Plot: Horizontales AUC-Balkendiagramm ───────────────────────────────────

def plot_auc_barchart(auc_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Horizontales Balkendiagramm der AUC-Werte je Metrik,
    sortiert absteigend. Fehlerbalken = 95%-KI via Bootstrap.
    """
    labels = auc_df['Beschreibung'].tolist()
    aucs   = auc_df['AUC'].tolist()
    ci_lo  = auc_df['CI_95_lower'].tolist()
    ci_hi  = auc_df['CI_95_upper'].tolist()

    xerr = [
        [a - lo for a, lo in zip(aucs, ci_lo)],
        [hi - a for a, hi in zip(aucs, ci_hi)],
    ]

    # Farben: rot = M2.4 (stärkster Prädiktor), blau = M2.2, orange = M4.1, grau = Rest
    color_map = {
        'M2.4 Stepwise Reasoning Quality': '#d62728',
        'M2.2 Trajectory Efficiency':      '#1f77b4',
        'M4.1 Context Utilization':        '#ff7f0e',
    }
    colors = [color_map.get(lbl, '#aec7e8') for lbl in labels]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y_pos = list(range(len(labels)))

    ax.barh(
        y_pos, aucs,
        xerr=xerr,
        color=colors, alpha=0.85, capsize=4,
        error_kw={'linewidth': 1.2, 'capthick': 1.2},
    )

    # Schwellenwert und Zufallslinie
    ax.axvline(x=0.70, color='black',  linestyle='--', linewidth=1.2,
               label='Schwellenwert AUC = 0,70')
    ax.axvline(x=0.50, color='#888888', linestyle=':',  linewidth=1.0,
               label='Zufallsniveau AUC = 0,50')

    # Zahlenwerte rechts hinter dem CI-Oberende platzieren
    for i, (auc, hi) in enumerate(zip(aucs, ci_hi)):
        ax.text(hi + 0.018, i, german_fmt(auc, 2), va='center', fontsize=13)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=13)
    ax.set_xlabel('AUC (95-%-KI via Bootstrap, n = 1 000)', fontsize=13)
    ax.tick_params(axis='x', labelsize=12)
    ax.set_xlim(0.25, 1.02)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: german_fmt(x, 2))
    )
    ax.legend(fontsize=10, loc='lower right')
    ax.invert_yaxis()

    plt.tight_layout()
    save_fig(fig, output_dir, 'auc_predictor_plot')
    plt.close(fig)


# ── Hauptprogramm ────────────────────────────────────────────────────────────

def main() -> None:
    print("Lade Daten ...")
    df = pd.read_csv(INPUT_CSV)
    n_success = int(df['M1.1'].sum())
    print(f"  {len(df)} Trajektorien  |  Erfolge: {n_success}  |  Misserfolge: {len(df) - n_success}")

    # ── Teil 1 ──
    print("\nTeil 1: AUC pro Metrik ...")
    auc_df = compute_auc_per_metric(df)
    out1 = INPUT_DIR / 'auc_predictor_results.csv'
    auc_df.to_csv(out1, index=False)
    print(f"  Gespeichert: {out1.name}")
    print(auc_df.to_string(index=False))

    # ── Teil 2 ──
    print("\nTeil 2: Subset-Vergleich ...")
    subset_df = compute_subset_comparison(df)
    out2 = INPUT_DIR / 'auc_subset_comparison.csv'
    subset_df.to_csv(out2, index=False)
    print(f"  Gespeichert: {out2.name}")
    print(subset_df.to_string(index=False))

    # ── Teil 3 ──
    print("\nTeil 3: System-Rang-Tabelle ...")
    rank_df, rho_m24, rho_m22 = compute_agent_ranks(df)
    out3 = INPUT_DIR / 'agent_rank_comparison.csv'
    rank_df.to_csv(out3, index=False)
    print(f"  Gespeichert: {out3.name}")
    print(rank_df.to_string(index=False))
    print(f"  Spearman rho(Erfolgsrate, M2.4) = {rho_m24}  (deskriptiv, N=4)")
    print(f"  Spearman rho(Erfolgsrate, M2.2) = {rho_m22}  (deskriptiv, N=4)")

    # ── Plot ──
    print("\nErstelle Abbildung ...")
    plot_auc_barchart(auc_df, INPUT_DIR)

    print("\nFertig.")


if __name__ == '__main__':
    main()
