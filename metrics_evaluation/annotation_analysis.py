#!/usr/bin/env python3
"""
RQ2: Analyse der manuellen Expertenbewertungen.

Visualisiert die Verteilung und Konsistenz der manuellen Annotationen
(Likert-Skala 1–5) über alle Agentensysteme und Metriken.
Ausgabe: PNG + PGF (für LaTeX-Einbindung via \\input{}).
Alle Beschriftungen auf Deutsch; Dezimaltrennzeichen = Komma.

Datenquelle: manual_annotations.csv
Optional:    manual_annotations_2.csv (Inter-Rater-Analyse)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
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

# ── Konfiguration ──────────────────────────────────────────────────────────

BASE_DIR         = Path(__file__).parent
ANNOTATIONS_FILE = BASE_DIR / 'manual_annotations.csv'
ANNOTATIONS_2    = BASE_DIR / 'manual_annotations_2.csv'
OUTPUT_DIR       = BASE_DIR / 'evaluation_results'

# Annotierte Metriken: Spaltenname → Kurzlabel (Metrikname bleibt Englisch)
ANNOTATION_COLS = {
    'M2.2_Trajectory_Efficiency_1to5':  'M2.2 Trajectory\nEfficiency',
    'M2.3_Global_Strategy_1to5':        'M2.3 Global Strategy\nConsistency',
    'M2.4_Reasoning_Quality_1to5':      'M2.4 Stepwise\nReasoning Quality',
    'M2.5_Role_Adherence_1to5':         'M2.5 Role\nAdherence',
    'M3.1_Tool_Selection_1to5':         'M3.1 Tool Selection\nQuality',
    'M3.2_Tool_Execution_Quality_1to5': 'M3.2 Tool Execution\nSuccess',
    'M4.1_Context_Utilization_1to5':    'M4.1 Context\nUtilization',
    'M5.1_Communication_1to5_IfMAS':    'M5.1 Communication\nEfficiency',
}

AGENT_ORDER = ['MetaGPT', 'OpenHands', 'SWE-agent', 'live-swe-agent']

AGENT_COLORS = {
    'MetaGPT':        '#2ca02c',
    'OpenHands':      '#1f77b4',
    'SWE-agent':      '#ff7f0e',
    'live-swe-agent': '#d62728',
}


# ── Hilfsfunktionen ────────────────────────────────────────────────────────

def german_fmt(val: float, digits: int = 2) -> str:
    """Formatiert einen Float mit deutschem Dezimaltrennzeichen (Komma)."""
    return f"{val:.{digits}f}".replace('.', ',')


def save_fig(fig: plt.Figure, output_dir: Path, name: str) -> None:
    """Speichert eine Figure als PNG und PGF (für LaTeX: \\input{name.pgf})."""
    png_path = output_dir / f'{name}.png'
    pgf_path = output_dir / f'{name}.pgf'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    try:
        fig.savefig(pgf_path, bbox_inches='tight')
        print(f"  Gespeichert: {png_path.name} + {pgf_path.name}")
    except Exception as e:
        print(f"  Gespeichert: {png_path.name}  (PGF fehlgeschlagen: {e})")


# ── 1. Violin-Plot: Annotationsverteilung ─────────────────────────────────

def create_annotation_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Violin-Plot der manuellen Annotationsverteilung je Metrik.

    Zeigt die Verteilung der Likert-Scores (1–5) aller Annotationen pro Metrik
    inkl. Einzelpunkten (Jitter) und Mittelwert-Beschriftung.
    LaTeX-Einbindung: \\input{annotation_distribution.pgf}
    Verwendung: Abschnitt 5.2.1 (Korrelation Mensch – Maschine)
    """
    cols = [c for c in ANNOTATION_COLS if c in df.columns]
    data_per_metric = [df[c].dropna().values for c in cols]
    labels = [ANNOTATION_COLS[c].replace('\n', ' ') for c in cols]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Violin-Plots zeichnen
    parts = ax.violinplot(
        data_per_metric, positions=range(len(cols)),
        showmedians=True, showextrema=True
    )

    # Farbgebung
    palette = plt.cm.Set2(np.linspace(0, 1, len(cols)))
    for pc, color in zip(parts['bodies'], palette):
        pc.set_facecolor(color)
        pc.set_alpha(0.70)
    for key in ('cmedians', 'cmaxes', 'cmins', 'cbars'):
        parts[key].set_color('black' if key == 'cmedians' else '#666666')
        if key == 'cmedians':
            parts[key].set_linewidth(2.5)

    # Einzelpunkte (jitter) überlagern
    rng = np.random.default_rng(42)
    for i, vals in enumerate(data_per_metric):
        jitter = rng.uniform(-0.13, 0.13, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter, vals,
            color='#222222', alpha=0.55, s=22, zorder=5
        )

    # Mittelwert über jedem Violin beschriften
    for i, vals in enumerate(data_per_metric):
        if len(vals) > 0:
            ax.text(
                i, 5.38, f'Ø {german_fmt(np.mean(vals), 1)}',
                ha='center', va='bottom', fontsize=8, color='#333333'
            )

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, fontsize=9, rotation=22, ha='right')
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(
        ['1 – Sehr schlecht', '2 – Schlecht', '3 – Akzeptabel', '4 – Gut', '5 – Sehr gut'],
        fontsize=9
    )
    ax.set_ylabel('Likert-Score (1–5)', fontsize=11)
    ax.set_ylim(0.5, 5.65)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_fig(fig, output_dir, 'annotation_distribution')
    plt.close()


# ── 2. Heatmap: Mittlere Annotationsscores je Agent ───────────────────────

def create_annotation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Heatmap der mittleren manuellen Scores je Agent × Metrik (Likert 1–5).

    Ergänzt die automatische Leistungs-Heatmap aus descriptive_analysis.py
    um die Perspektive menschlicher Experten.
    LaTeX-Einbindung: \\input{annotation_heatmap_by_agent.pgf}
    Verwendung: Abschnitt 5.2.1 (Korrelation Mensch – Maschine)
    """
    cols = [c for c in ANNOTATION_COLS if c in df.columns]
    labels = [ANNOTATION_COLS[c].replace('\n', ' ') for c in cols]

    matrix = np.full((len(AGENT_ORDER), len(cols)), np.nan)
    for i, agent in enumerate(AGENT_ORDER):
        agent_df = df[df['agent_name'] == agent]
        for j, col in enumerate(cols):
            vals = agent_df[col].dropna()
            if len(vals) > 0:
                matrix[i, j] = vals.mean()

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=1, vmax=5, aspect='auto')

    for i in range(len(AGENT_ORDER)):
        for j in range(len(cols)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, 'n.a.', ha='center', va='center',
                        fontsize=10, color='#888888', fontstyle='italic')
            else:
                color = 'white' if val < 2.0 or val > 4.2 else 'black'
                ax.text(j, i, german_fmt(val, 1), ha='center', va='center',
                        fontsize=12, fontweight='bold', color=color)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, fontsize=9, rotation=22, ha='right')
    ax.set_yticks(range(len(AGENT_ORDER)))
    ax.set_yticklabels(AGENT_ORDER, fontsize=12)

    cbar = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label('Mittlerer Likert-Score', fontsize=10)
    cbar.set_ticks([1, 2, 3, 4, 5])
    cbar.set_ticklabels(['1', '2', '3', '4', '5'])

    plt.tight_layout()
    save_fig(fig, output_dir, 'annotation_heatmap_by_agent')
    plt.close()


# ── 3. Violin-Plot: Annotationsverteilung pro Agent ───────────────────────

def create_annotation_distribution_by_agent(
    df: pd.DataFrame, output_dir: Path
) -> None:
    """
    Violin-Plot der Annotationsverteilung je Metrik, aufgeteilt nach Agent.

    4 Subplots (2×2), je einer pro Agentensystem. Zeigt, wie konsistent
    und wie breit die Skala für jeden Agenten genutzt wurde.
    LaTeX-Einbindung: \\input{annotation_distribution_by_agent.pgf}
    Verwendung: Abschnitt 5.2.1
    """
    cols = [c for c in ANNOTATION_COLS if c in df.columns]
    metric_labels = [ANNOTATION_COLS[c].replace('\n', ' ') for c in cols]
    ytick_labels = ['1 – Sehr schlecht', '2 – Schlecht', '3 – Akzeptabel',
                    '4 – Gut', '5 – Sehr gut']

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    axes = axes.flatten()
    rng = np.random.default_rng(42)

    for ax, agent in zip(axes, AGENT_ORDER):
        agent_df = df[df['agent_name'] == agent]
        color = AGENT_COLORS[agent]

        data_per_metric = [agent_df[c].dropna().values for c in cols]

        # Nur Metriken mit Daten plotten
        valid = [(i, d) for i, d in enumerate(data_per_metric) if len(d) > 0]
        positions = [i for i, _ in valid]
        data_vals = [d for _, d in valid]

        if not data_vals:
            ax.set_title(agent, fontsize=12, fontweight='bold',
                         color=color, pad=8)
            continue

        parts = ax.violinplot(data_vals, positions=positions,
                              showmedians=True, showextrema=True)

        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.55)
        for key in ('cmedians', 'cmaxes', 'cmins', 'cbars'):
            parts[key].set_color('black' if key == 'cmedians' else '#666666')
            if key == 'cmedians':
                parts[key].set_linewidth(2.5)

        # Jitter-Punkte
        for pos, vals in zip(positions, data_vals):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals,
                       color='#111111', alpha=0.6, s=20, zorder=5)

        # Mittelwert beschriften
        for pos, vals in zip(positions, data_vals):
            ax.text(pos, 5.38, f'Ø {german_fmt(np.mean(vals), 1)}',
                    ha='center', va='bottom', fontsize=7.5, color='#333333')

        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(metric_labels, fontsize=8, rotation=22, ha='right')
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(ytick_labels, fontsize=8)
        ax.set_ylim(0.5, 5.65)
        ax.set_title(agent, fontsize=12, fontweight='bold',
                     color=color, pad=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_fig(fig, output_dir, 'annotation_distribution_by_agent')
    plt.close()


# ── 4. Heatmap: Streuung (Std) der Annotationsscores je Agent ─────────────

def create_annotation_spread_heatmap(
    df: pd.DataFrame, output_dir: Path
) -> None:
    """
    Heatmap der Standardabweichung der manuellen Scores je Agent × Metrik.

    Ergänzt annotation_heatmap_by_agent (Mittelwert) um die Konsistenz-
    dimension: niedriger Wert = konsistente Bewertungen über 15 Tasks,
    hoher Wert = heterogenes Agenten-Verhalten oder unsichere Annotation.
    LaTeX-Einbindung: \\input{annotation_spread_heatmap.pgf}
    Verwendung: Abschnitt 5.2.1
    """
    cols = [c for c in ANNOTATION_COLS if c in df.columns]
    labels = [ANNOTATION_COLS[c].replace('\n', ' ') for c in cols]

    matrix_mean = np.full((len(AGENT_ORDER), len(cols)), np.nan)
    matrix_std  = np.full((len(AGENT_ORDER), len(cols)), np.nan)
    matrix_n    = np.full((len(AGENT_ORDER), len(cols)), 0, dtype=int)

    for i, agent in enumerate(AGENT_ORDER):
        agent_df = df[df['agent_name'] == agent]
        for j, col in enumerate(cols):
            vals = agent_df[col].dropna()
            if len(vals) > 1:
                matrix_mean[i, j] = vals.mean()
                matrix_std[i, j]  = vals.std()
                matrix_n[i, j]    = len(vals)
            elif len(vals) == 1:
                matrix_mean[i, j] = vals.mean()
                matrix_std[i, j]  = 0.0
                matrix_n[i, j]    = 1

    # Zwei Subplots nebeneinander: Mittelwert + Std
    fig, (ax_mean, ax_std) = plt.subplots(1, 2, figsize=(22, 4))

    for ax, matrix, title, vmin, vmax, cmap, fmt_str in [
        (ax_mean, matrix_mean,
         'Mittlerer Likert-Score (Ø)', 1, 5, 'RdYlGn', '1'),
        (ax_std, matrix_std,
         'Standardabweichung (σ)', 0, 2, 'YlOrRd', '0,0'),
    ]:
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

        for i in range(len(AGENT_ORDER)):
            for j in range(len(cols)):
                val = matrix[i, j]
                if np.isnan(val):
                    ax.text(j, i, 'n.a.', ha='center', va='center',
                            fontsize=9, color='#888888', fontstyle='italic')
                else:
                    # Kontrast je nach Hintergrundfarbe
                    norm_val = (val - vmin) / (vmax - vmin)
                    text_color = 'white' if norm_val > 0.75 else 'black'
                    ax.text(j, i, german_fmt(val, 1),
                            ha='center', va='center',
                            fontsize=11, fontweight='bold', color=text_color)

        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(labels, fontsize=9, rotation=22, ha='right')
        ax.set_yticks(range(len(AGENT_ORDER)))
        ax.set_yticklabels(AGENT_ORDER, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

        cbar = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
        cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    save_fig(fig, output_dir, 'annotation_spread_heatmap')
    plt.close()


# ── 5. Streudiagramm: Inter-Rater-Übereinstimmung ─────────────────────────

def create_interrater_scatter(
    df1: pd.DataFrame, df2: pd.DataFrame, output_dir: Path
) -> None:
    """
    Streudiagramm Rater 1 vs. Rater 2 je Metrik.

    Visualisiert die paarweise Übereinstimmung beider Annotationen.
    Wird nur erstellt, wenn manual_annotations_2.csv vorhanden ist.
    LaTeX-Einbindung: \\input{interrater_agreement.pgf}
    Verwendung: Abschnitt 5.2.2 (Analyse der Bewertungsdivergenz)
    """
    cols = [c for c in ANNOTATION_COLS if c in df1.columns and c in df2.columns]
    n_cols = len(cols)
    ncols_per_row = 4
    nrows = (n_cols + ncols_per_row - 1) // ncols_per_row

    fig, axes = plt.subplots(nrows, ncols_per_row,
                              figsize=(ncols_per_row * 4, nrows * 4.2))
    axes = np.array(axes).flatten()

    merge_keys = ['task_id', 'agent_name']

    for idx, col in enumerate(cols):
        ax = axes[idx]
        label = ANNOTATION_COLS[col].replace('\n', ' ')

        merged = (
            df1[merge_keys + [col]]
            .merge(df2[merge_keys + [col]], on=merge_keys, suffixes=('_r1', '_r2'))
            .dropna()
        )

        if len(merged) < 3:
            ax.text(0.5, 0.5, 'Zu wenige\nDatenpunkte',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='gray')
            ax.set_title(label, fontsize=9, fontweight='bold')
            continue

        r1 = merged[f'{col}_r1'].values
        r2 = merged[f'{col}_r2'].values

        # Punkte je Agent färben
        for agent in AGENT_ORDER:
            mask = merged['agent_name'] == agent
            if mask.any():
                ax.scatter(r1[mask], r2[mask],
                           color=AGENT_COLORS[agent], s=55, alpha=0.85,
                           label=agent, zorder=5, edgecolors='white',
                           linewidths=0.5)

        ax.plot([0.8, 5.2], [0.8, 5.2], 'k--', alpha=0.35, linewidth=1.2,
                label='Perfekte Übereinstimmung')

        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(0.5, 5.5)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_xlabel('Rater 1', fontsize=9)
        ax.set_ylabel('Rater 2', fontsize=9)
        ax.set_title(label, fontsize=9, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')

        # Spearman-Korrelation als Annotation
        rho, p = spearmanr(r1, r2)
        p_star = ('***' if p < 0.001 else ('**' if p < 0.01 else
                  ('*' if p < 0.05 else 'n.s.')))
        ax.text(0.04, 0.93, f'ρ = {german_fmt(rho)} {p_star}',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#cccccc', alpha=0.9))

    # Gemeinsame Legende einmalig
    handles, lbls = axes[0].get_legend_handles_labels()
    agent_handles = [h for h, l in zip(handles, lbls) if l in AGENT_ORDER]
    agent_labels  = [l for l in lbls if l in AGENT_ORDER]
    if agent_handles:
        fig.legend(agent_handles, agent_labels,
                   loc='lower center', ncol=4, fontsize=9,
                   bbox_to_anchor=(0.5, -0.01),
                   frameon=True, edgecolor='#cccccc')

    # Leere Subplots ausblenden
    for idx in range(len(cols), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save_fig(fig, output_dir, 'interrater_agreement')
    plt.close()


# ── 4. Konsolausgabe: Deskriptive Statistiken ─────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    """Gibt deskriptive Statistiken der manuellen Annotationen aus."""
    cols = [c for c in ANNOTATION_COLS if c in df.columns]

    print("\n  Deskriptive Statistiken (Likert 1–5):")
    print(f"  {'Metrik':<45} {'N':>4} {'Ø':>5} {'Median':>7} "
          f"{'Std':>5} {'Min':>4} {'Max':>4}")
    print("  " + "─" * 76)

    for col in cols:
        vals = df[col].dropna()
        label = ANNOTATION_COLS[col].replace('\n', ' ')
        if len(vals) == 0:
            continue
        print(f"  {label:<45} {len(vals):>4} "
              f"{german_fmt(vals.mean()):>5} "
              f"{german_fmt(float(vals.median())):>7} "
              f"{german_fmt(vals.std()):>5} "
              f"{german_fmt(vals.min(), 0):>4} "
              f"{german_fmt(vals.max(), 0):>4}")

    # Aufteilung nach Agent
    print("\n  Mittlere Scores je Agentensystem:")
    header_parts = ['Agent'.ljust(16)]
    for col in cols:
        header_parts.append(ANNOTATION_COLS[col].replace('\n', ' ')[:8])
    print("  " + " | ".join(header_parts))
    print("  " + "─" * 76)
    for agent in AGENT_ORDER:
        agent_df = df[df['agent_name'] == agent]
        row = [agent.ljust(16)]
        for col in cols:
            vals = agent_df[col].dropna()
            row.append(german_fmt(vals.mean(), 1) if len(vals) > 0 else ' n.a.')
        print("  " + " | ".join(row))


# ── Hauptprogramm ──────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("RQ2: Analyse der manuellen Expertenbewertungen")
    print("=" * 70)

    if not ANNOTATIONS_FILE.exists():
        print(f"Fehler: Annotationsdatei nicht gefunden: {ANNOTATIONS_FILE}")
        return

    df = pd.read_csv(ANNOTATIONS_FILE)
    print(f"\nGeladen: {len(df)} Zeilen | "
          f"Agenten: {df['agent_name'].unique().tolist()} | "
          f"Aufgaben: {df['task_id'].nunique()}")

    # Zweite Annotationsdatei für Inter-Rater-Analyse (optional)
    df2 = None
    if ANNOTATIONS_2.exists():
        df2 = pd.read_csv(ANNOTATIONS_2)
        print(f"Zweite Annotationsdatei geladen: {len(df2)} Zeilen")
    else:
        print(f"Info: Keine zweite Annotationsdatei ({ANNOTATIONS_2.name}) "
              f"– Inter-Rater-Visualisierung wird übersprungen.")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Violin-Plot: Verteilung je Metrik (alle Agenten zusammen)
    print("\n── 1. Annotationsverteilung gesamt (Violin-Plot) ──")
    create_annotation_distribution(df, OUTPUT_DIR)

    # 2. Violin-Plot: Verteilung je Metrik, aufgeteilt nach Agent
    print("\n── 2. Annotationsverteilung pro Agent (Violin-Plot) ──")
    create_annotation_distribution_by_agent(df, OUTPUT_DIR)

    # 3. Heatmap: Mittlere Scores je Agent × Metrik
    print("\n── 3. Heatmap der mittleren Annotationsscores ──")
    create_annotation_heatmap(df, OUTPUT_DIR)

    # 4. Heatmap: Mittelwert + Standardabweichung je Agent × Metrik
    print("\n── 4. Heatmap Mittelwert + Streuung (Std) ──")
    create_annotation_spread_heatmap(df, OUTPUT_DIR)

    # 5. Inter-Rater-Streudiagramm (nur mit zweiter Annotationsdatei)
    if df2 is not None:
        print("\n── 5. Inter-Rater-Übereinstimmung (Streudiagramme) ──")
        create_interrater_scatter(df, df2, OUTPUT_DIR)
    else:
        print("\n── 5. Inter-Rater-Analyse übersprungen (keine zweite Datei) ──")

    # 6. Konsolausgabe
    print_summary(df)

    print("\n" + "=" * 70)
    print("Fertig! Alle Ausgaben in:", OUTPUT_DIR)
    print("PGF-Einbindung in LaTeX:")
    print("  \\input{evaluation_results/annotation_distribution.pgf}")
    print("  \\input{evaluation_results/annotation_distribution_by_agent.pgf}")
    print("  \\input{evaluation_results/annotation_heatmap_by_agent.pgf}")
    print("  \\input{evaluation_results/annotation_spread_heatmap.pgf}")
    if df2 is not None:
        print("  \\input{evaluation_results/interrater_agreement.pgf}")
    print("=" * 70)


if __name__ == "__main__":
    main()
