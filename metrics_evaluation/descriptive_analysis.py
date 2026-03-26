#!/usr/bin/env python3
"""
RQ1: Deskriptive Analyse der Agentensysteme.

Generiert aggregierte Statistiken und Visualisierungen für alle
evaluierten Agenten. Ausgabe: PNG + PGF (für LaTeX-Einbindung via \input{}).
Alle Beschriftungen auf Deutsch; Dezimaltrennzeichen = Komma.

Datenquelle: evaluation_results/1_step_gptoss120b/consolidated_results.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr
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

EVAL_DIR = Path(__file__).parent / 'evaluation_results'
INPUT_CSV = EVAL_DIR / '1_step_gptoss120b' / 'consolidated_results.csv'

# Radar: alle 0-1-Metriken + M1.1 + invertiertes M2.1
RADAR_METRICS = ['M1.1', 'M2.1_inv', 'M2.2', 'M2.4', 'M2.5', 'M3.1', 'M3.2', 'M3.3', 'M4.1']

# Box-Plots: LLM-Judge + deterministische Prozessmetriken
PROCESS_METRICS = ['M2.2', 'M2.3', 'M2.4', 'M2.5', 'M3.1', 'M3.2', 'M3.3', 'M4.1']

# Heatmap: alle vergleichbaren Metriken
HEATMAP_METRICS = ['M1.1', 'M2.1_inv', 'M2.2', 'M2.4', 'M2.5', 'M3.1', 'M3.2', 'M3.3', 'M4.1']

# Korrelationsmatrix: Prozessmetriken der Einzelagenten
CORR_METRICS = ['M2.1_inv', 'M2.2', 'M2.4', 'M2.5', 'M3.1', 'M3.2', 'M3.3', 'M4.1']

# Ressourcen-Metriken
RESOURCE_METRICS = ['M1.2_cost_usd', 'M1.2_tokens', 'M1.2_duration_s', 'M1.2_steps']

# MAS-exklusive Metriken
MAS_METRICS = ['M5.1', 'M5.2', 'M5.3', 'M5.4']

# Aufgabenschwierigkeit (validierter 9-4-2 Split)
TASK_DIFFICULTY = {
    'scikit-learn__scikit-learn-12585': 'easy',
    'pallets__flask-5014':              'easy',
    'psf__requests-2317':               'easy',
    'matplotlib__matplotlib-22719':     'easy',
    'django__django-11099':             'easy',
    'sympy__sympy-13480':               'easy',
    'matplotlib__matplotlib-20676':     'easy',
    'pylint-dev__pylint-4970':          'easy',
    'pytest-dev__pytest-5262':          'easy',
    'django__django-16901':             'medium',
    'scikit-learn__scikit-learn-10297': 'medium',
    'sympy__sympy-20801':               'medium',
    'matplotlib__matplotlib-23299':     'medium',
    'django__django-15128':             'hard',
    'sympy__sympy-18199':               'hard',
}
DIFFICULTY_ORDER = {'easy': 0, 'medium': 1, 'hard': 2}
DIFFICULTY_BG    = {'easy': '#eaf7ec', 'medium': '#fefbe6', 'hard': '#fdecea'}
DIFFICULTY_LABEL = {'easy': 'Leicht (9)',  'medium': 'Mittel (4)', 'hard': 'Schwer (2)'}

# Metrik-Beschriftungen (Metriknamen bleiben auf Englisch)
METRIC_LABELS = {
    'M1.1':          'M1.1 Task\nSuccess Rate',
    'M1.2_cost_usd': 'M1.2 Kosten (USD)',
    'M1.2_tokens':   'M1.2 Token',
    'M1.2_duration_s':'M1.2 Dauer (s)',
    'M1.2_steps':    'M1.2 Schritte',
    'M2.1':          'M2.1 Loop\nDetection',
    'M2.1_inv':      'M2.1 Loop-Free\nRate',
    'M2.2':          'M2.2 Trajectory\nEfficiency',
    'M2.3':          'M2.3 Global Strategy\nConsistency',
    'M2.4':          'M2.4 Stepwise\nReasoning Quality',
    'M2.5':          'M2.5 Role\nAdherence',
    'M3.1':          'M3.1 Tool Selection\nQuality',
    'M3.2':          'M3.2 Tool Execution\nSuccess',
    'M3.3':          'M3.3 Tool Usage\nEfficiency',
    'M4.1':          'M4.1 Context\nUtilization',
    'M5.1':          'M5.1 Communication\nEfficiency',
    'M5.2':          'M5.2 Information\nDiversity (IDS)',
    'M5.3':          'M5.3 Unique Path\nRedundancy (UPR)',
    'M5.4':          'M5.4 Agent Invocation\nDistribution',
}

AGENT_COLORS = {
    'MetaGPT':       '#2ca02c',
    'OpenHands':     '#1f77b4',
    'SWE-Agent':     '#ff7f0e',
    'live-swe-agent':'#d62728',
}

AGENT_ORDER = ['MetaGPT', 'OpenHands', 'SWE-Agent', 'live-swe-agent']

AGENT_DISPLAY_NAMES = {
    'MetaGPT':       'MetaGPT',
    'OpenHands':     'OpenHands',
    'SWE-Agent':     'SWE-Agent',
    'live-swe-agent':'Live-SWE-agent',
}


# ── Hilfsfunktionen ────────────────────────────────────────────────────────

def german_fmt(val: float, digits: int = 2) -> str:
    """Formatiert einen Float mit deutschem Dezimaltrennzeichen (Komma)."""
    return f"{val:.{digits}f}".replace('.', ',')


def german_axis_formatter(digits: int = 2) -> mticker.FuncFormatter:
    """Gibt einen FuncFormatter zurück, der Komma als Dezimaltrennzeichen nutzt."""
    return mticker.FuncFormatter(lambda x, _: german_fmt(x, digits))


def save_fig(fig: plt.Figure, output_dir: Path, name: str) -> None:
    """Speichert eine Figure als PNG, PDF, PGF und TEX.

    Die .tex-Datei referenziert die .pdf (selbst-enthaltend, keine externen
    Abhängigkeiten) und kann per \\input{name.tex} eingebunden werden.
    In Overleaf beide Dateien (.tex + .pdf) hochladen.
    """
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
        fig.savefig(pgf_path, bbox_inches='tight')
        print(f"  Gespeichert: {png_path.name} + {pdf_path.name} + {pgf_path.name} + {tex_path.name}")
    except Exception as e:
        print(f"  Gespeichert: {png_path.name} + {pdf_path.name} + {tex_path.name}  (PGF fehlgeschlagen: {e})")


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt abgeleitete Spalten hinzu."""
    df = df.copy()
    # M2.1 invertieren: 1=Loop detektiert (schlecht) → Loop-Free Rate
    df['M2.1_inv'] = 1 - df['M2.1']
    return df


def get_agent_mean(df: pd.DataFrame, agent: str, metric: str) -> float:
    """Gibt den Mittelwert einer Metrik für einen Agenten zurück (NaN-sicher)."""
    vals = df[df['agent_name'] == agent][metric].dropna()
    return vals.mean() if len(vals) > 0 else np.nan


# ── 1. Aggregation ─────────────────────────────────────────────────────────

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet Mittelwert, Median, Std, Min, Max je Agent und Metrik."""
    all_metrics = (['M1.1'] + RESOURCE_METRICS + ['M2.1', 'M2.1_inv'] +
                   PROCESS_METRICS + MAS_METRICS)
    rows = []

    for agent in AGENT_ORDER:
        agent_df = df[df['agent_name'] == agent]
        for metric in all_metrics:
            if metric not in agent_df.columns:
                continue
            vals = agent_df[metric].dropna()
            if len(vals) == 0:
                continue
            rows.append({
                'Agent':  agent,
                'Metric': metric,
                'Label':  METRIC_LABELS.get(metric, metric).replace('\n', ' '),
                'Mean':   round(vals.mean(), 4),
                'Median': round(vals.median(), 4),
                'Std':    round(vals.std(), 4),
                'Min':    round(vals.min(), 4),
                'Max':    round(vals.max(), 4),
                'N':      len(vals),
            })

    return pd.DataFrame(rows)


def export_table_5_2(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Exportiert Mittelwert und Median der Kategorie-1-Metriken je Agent als CSV.

    Datengrundlage für Tabelle 5.2 im Kapitel 5 der Masterarbeit.
    Ausgabe: table_5_2_performance.csv
    Spalten: Agent | M1.1_Ø | M1.1_Median | Kosten_Ø | Kosten_Median |
             Token_Ø | Token_Median | Dauer_s_Ø | Dauer_s_Median |
             Schritte_Ø | Schritte_Median
    """
    TABLE_METRICS = [
        ('M1.1',          'Erfolgsrate',  ''),
        ('M1.2_cost_usd', 'Kosten',       'USD'),
        ('M1.2_tokens',   'Token',        ''),
        ('M1.2_duration_s','Dauer',       's'),
        ('M1.2_steps',    'Schritte',     ''),
    ]

    rows = []
    for agent in AGENT_ORDER:
        agent_df = df[df['agent_name'] == agent]
        row = {'Agent': agent}
        for col, label, unit in TABLE_METRICS:
            suffix = f' ({unit})' if unit else ''
            vals = agent_df[col].dropna() if col in agent_df.columns else pd.Series([], dtype=float)
            if len(vals) > 0:
                row[f'{label}{suffix}_Mittelwert'] = round(vals.mean(), 4)
                row[f'{label}{suffix}_Median']     = round(vals.median(), 4)
                row[f'{label}{suffix}_Std']        = round(vals.std(), 4)
                row[f'{label}{suffix}_Min']        = round(vals.min(), 4)
                row[f'{label}{suffix}_Max']        = round(vals.max(), 4)
            else:
                for stat in ('Mittelwert', 'Median', 'Std', 'Min', 'Max'):
                    row[f'{label}{suffix}_{stat}'] = np.nan
        rows.append(row)

    result_df = pd.DataFrame(rows)
    out_path = output_dir / 'table_5_2_performance.csv'
    result_df.to_csv(out_path, index=False)
    print(f"  Tabelle 5.2 exportiert: {out_path.name}")

    # Auch als formatierte Konsolausgabe
    print("\n  Tabelle 5.2 – Kategorie-1-Metriken je Agentensystem:")
    print(f"  {'Agent':<16} | {'Erfolg':>9} | {'Kosten (USD)':>13} | "
          f"{'Token':>10} | {'Dauer (s)':>10} | {'Schritte':>9}")
    print("  " + "─" * 78)
    for agent in AGENT_ORDER:
        agent_df = df[df['agent_name'] == agent]
        sr   = agent_df['M1.1'].mean()
        n_ok = int(sr * len(agent_df))
        cost = agent_df['M1.2_cost_usd'].mean()
        tok  = agent_df['M1.2_tokens'].mean()
        dur  = agent_df['M1.2_duration_s'].mean()
        stp  = agent_df['M1.2_steps'].mean()
        print(f"  {agent:<16} | {german_fmt(sr)} ({n_ok}/15) | "
              f"${german_fmt(cost):>11} | "
              f"{tok:>10,.0f} | "
              f"{german_fmt(dur, 0):>10} | "
              f"{german_fmt(stp, 1):>9}")


# ── 2. Radar-Diagramm ──────────────────────────────────────────────────────

def create_radar_chart(df: pd.DataFrame, output_dir: Path) -> None:
    """Radar-Diagramm mit allen vergleichbaren Metriken."""
    metrics = RADAR_METRICS
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for agent in AGENT_ORDER:
        values = [get_agent_mean(df, agent, m) for m in metrics]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2.5, label=AGENT_DISPLAY_NAMES[agent],
                color=AGENT_COLORS[agent], markersize=7)
        ax.fill(angles, values, alpha=0.08, color=AGENT_COLORS[agent])

    # Zeilenumbrüche beibehalten → kürzere Zeilen → bessere Lesbarkeit am Rand
    labels = [METRIC_LABELS.get(m, m) for m in metrics]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8, linespacing=1.3)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0,2', '0,4', '0,6', '0,8', '1,0'], fontsize=8)
    ax.tick_params(pad=20)   # Labels von der Chart-Kante wegdrücken
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.12), fontsize=11)

    plt.tight_layout()
    save_fig(fig, output_dir, 'radar_chart_agents')
    plt.close()


# ── 3. Box-Plots ───────────────────────────────────────────────────────────

def create_boxplots(df: pd.DataFrame, output_dir: Path) -> None:
    """Box-Plots je Metrik, gruppiert nach Agentensystem."""
    metrics = PROCESS_METRICS
    fig, axes = plt.subplots(2, 4, figsize=(20, 11))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        data, labels, colors = [], [], []

        for agent in AGENT_ORDER:
            vals = df[df['agent_name'] == agent][metric].dropna()
            if len(vals) > 0:
                data.append(vals.values)
                labels.append(AGENT_DISPLAY_NAMES[agent])
                colors.append(AGENT_COLORS[agent])

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(METRIC_LABELS.get(metric, metric).replace('\n', ' '),
                     fontsize=14, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=35, labelsize=13)
        ax.tick_params(axis='y', labelsize=13)
        ax.yaxis.set_major_formatter(german_axis_formatter(1))

    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    save_fig(fig, output_dir, 'boxplots_per_metric')
    plt.close()


def create_boxplots_4x2(df: pd.DataFrame, output_dir: Path) -> None:
    """Box-Plots je Metrik im 4×2-Raster (4 Zeilen, 2 Spalten)."""
    metrics = PROCESS_METRICS
    fig, axes = plt.subplots(4, 2, figsize=(12, 20))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        data, labels, colors = [], [], []

        for agent in AGENT_ORDER:
            vals = df[df['agent_name'] == agent][metric].dropna()
            if len(vals) > 0:
                data.append(vals.values)
                labels.append(AGENT_DISPLAY_NAMES[agent])
                colors.append(AGENT_COLORS[agent])

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(METRIC_LABELS.get(metric, metric).replace('\n', ' '),
                     fontsize=14, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=35, labelsize=13)
        ax.tick_params(axis='y', labelsize=13)
        ax.yaxis.set_major_formatter(german_axis_formatter(1))

    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    save_fig(fig, output_dir, 'boxplots_per_metric_4x2')
    plt.close()


# ── 4. Erfolg vs. Misserfolg ───────────────────────────────────────────────

def create_success_failure_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Vergleich der mittleren Metrik-Scores zwischen erfolgreichen und
    gescheiterten Episoden (Mann-Whitney-U-Test).
    Verwendet für RQ3 (prognostische Validität).
    """
    metrics = ['M2.1_inv', 'M2.2', 'M2.4', 'M2.5', 'M3.1', 'M3.2', 'M3.3', 'M4.1']

    success_df = df[df['M1.1'] == 1]
    failure_df = df[df['M1.1'] == 0]
    print(f"\n  Erfolgreich: {len(success_df)} Tasks, Gescheitert: {len(failure_df)} Tasks")

    means_success = [success_df[m].mean() for m in metrics]
    means_failure = [failure_df[m].mean() for m in metrics]

    p_values = []
    for m in metrics:
        s_vals = success_df[m].dropna()
        f_vals = failure_df[m].dropna()
        if len(s_vals) >= 3 and len(f_vals) >= 3:
            _, p = mannwhitneyu(s_vals, f_vals, alternative='two-sided')
            p_values.append(p)
        else:
            p_values.append(np.nan)

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width / 2, means_success, width,
           label=f'Erfolgreich (n={len(success_df)})',
           color='#2ca02c', alpha=0.8, edgecolor='black')
    ax.bar(x + width / 2, means_failure, width,
           label=f'Gescheitert (n={len(failure_df)})',
           color='#d62728', alpha=0.8, edgecolor='black')

    for i, (ms, mf, p) in enumerate(zip(means_success, means_failure, p_values)):
        max_h = max(ms, mf) + 0.03
        if pd.notna(p):
            sig = ('***' if p < 0.001 else ('**' if p < 0.01 else
                   ('*' if p < 0.05 else 'n.s.')))
            ax.text(i, max_h, sig, ha='center', va='bottom', fontsize=13, fontweight='bold')

    labels = [METRIC_LABELS.get(m, m).replace('\n', ' ') for m in metrics]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, rotation=25, ha='right')
    ax.set_ylabel('Mittlerer Score (0–1)', fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_major_formatter(german_axis_formatter(1))

    plt.tight_layout()
    save_fig(fig, output_dir, 'success_vs_failure_comparison')
    plt.close()

    print("\n  | Metrik | Ø Erfolg | Ø Misserfolg | Differenz | p-Wert | Sig. |")
    print("  |--------|----------|--------------|-----------|--------|------|")
    for m, ms, mf, p in zip(metrics, means_success, means_failure, p_values):
        diff = ms - mf
        p_str = german_fmt(p, 4) if pd.notna(p) else "N/A"
        sig = ('***' if pd.notna(p) and p < 0.001 else
               ('**' if pd.notna(p) and p < 0.01 else
               ('*' if pd.notna(p) and p < 0.05 else 'n.s.')))
        label = METRIC_LABELS.get(m, m).replace('\n', ' ')
        print(f"  | {label:30s} | {german_fmt(ms)} | {german_fmt(mf)} | "
              f"{diff:+.3f} | {p_str} | {sig} |")


# ── 5. Heatmap ─────────────────────────────────────────────────────────────

def create_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Agent × Metrik Heatmap mit Kategorie-Separatoren."""
    metrics = HEATMAP_METRICS
    matrix = np.array([
        [get_agent_mean(df, agent, m) for m in metrics]
        for agent in AGENT_ORDER
    ])
    labels = [METRIC_LABELS.get(m, m).replace('\n', ' ') for m in metrics]

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    for i in range(len(AGENT_ORDER)):
        for j in range(len(metrics)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, '—', ha='center', va='center', fontsize=14, color='gray')
            else:
                color = 'white' if val < 0.3 or val > 0.85 else 'black'
                ax.text(j, i, german_fmt(val), ha='center', va='center',
                        fontsize=14, fontweight='bold', color=color)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(labels, fontsize=12, rotation=30, ha='right')
    ax.set_yticks(range(len(AGENT_ORDER)))
    ax.set_yticklabels([AGENT_DISPLAY_NAMES[a] for a in AGENT_ORDER], fontsize=14)

    # Kategorie-Trennlinien
    for sep_x in [0.5, 4.5, 7.5]:
        ax.axvline(x=sep_x, color='black', linewidth=2)

    # Kategorie-Beschriftungen
    for xpos, cat_label in [(0, 'Ergebnis'), (2.5, 'Strategie'), (6, 'Werkzeuge'), (8, 'Wissen')]:
        ax.text(xpos, -0.7, cat_label, ha='center', va='center',
                fontsize=12, fontstyle='italic', color='#555555')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Score (0 = schlecht, 1 = gut)', fontsize=12)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(['0,0', '0,2', '0,4', '0,6', '0,8', '1,0'], fontsize=11)

    plt.tight_layout()
    save_fig(fig, output_dir, 'agent_comparison_heatmap')
    plt.close()


# ── 6. Kosten-Nutzen-Diagramm ──────────────────────────────────────────────

def create_cost_benefit_chart(df: pd.DataFrame, output_dir: Path) -> None:
    """Streudiagramm: Aufgabenerfolg vs. Kosten / Token / Dauer (logarithmische Skala)."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    def fmt_cost(x, _):
        return f'{x:.2f} USD'.replace('.', ',') if x < 1 else f'{x:.1f} USD'.replace('.', ',')

    def fmt_tokens(x, _):
        return f'{x/1e6:.1f} Mio.'.replace('.', ',') if x >= 1e6 else f'{int(x/1e3)} K'

    def fmt_duration(x, _):
        return f'{x:.0f} s'

    comparisons = [
        ('M1.2_cost_usd',   'Ø Kosten (USD)',           fmt_cost,     None),
        ('M1.2_tokens',     'Ø Token-Verbrauch',         fmt_tokens,   [1e5, 5e5, 1e6, 2e6]),
        ('M1.2_duration_s', 'Ø Ausführungsdauer (s)',    fmt_duration, [10, 100, 1000]),
    ]

    for ax, (resource_col, resource_label, x_formatter, xticks) in zip(axes, comparisons):
        for agent in AGENT_ORDER:
            agent_df = df[df['agent_name'] == agent]
            success_rate = agent_df['M1.1'].mean()
            resource_val = agent_df[resource_col].mean()

            ax.scatter(resource_val, success_rate, s=200,
                       color=AGENT_COLORS[agent], edgecolors='black',
                       linewidth=1.5, zorder=5, label=AGENT_DISPLAY_NAMES[agent])
            ax.annotate(AGENT_DISPLAY_NAMES[agent], (resource_val, success_rate),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=11, fontweight='bold')

        ax.set_xlabel(resource_label, fontsize=13)
        ax.set_ylabel('M1.1 Aufgabenerfolgsrate', fontsize=13)
        ax.tick_params(axis='both', labelsize=11)
        ax.tick_params(axis='x', rotation=30)
        ax.set_xscale('log')
        if xticks is not None:
            ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(x_formatter))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_ylim(-0.05, 0.80)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='50%-Schwelle')
        ax.yaxis.set_major_formatter(german_axis_formatter(2))

    plt.tight_layout()
    save_fig(fig, output_dir, 'cost_benefit_scatter')
    plt.close()

    print("\n  | Agent | Erfolg | Ø Kosten | Ø Token | Ø Dauer (s) | Ø Schritte | Loop-Rate |")
    print("  |-------|--------|----------|---------|-------------|------------|-----------|")
    for agent in AGENT_ORDER:
        agent_df = df[df['agent_name'] == agent]
        sr    = agent_df['M1.1'].mean()
        cost  = agent_df['M1.2_cost_usd'].mean()
        tok   = agent_df['M1.2_tokens'].mean()
        dur   = agent_df['M1.2_duration_s'].mean()
        steps = agent_df['M1.2_steps'].mean()
        loops = agent_df['M2.1'].mean()
        print(f"  | {agent:15s} | {german_fmt(sr)} ({int(sr*15)}/15) | "
              f"${german_fmt(cost)} | {tok:,.0f} | {dur:.0f} | "
              f"{steps:.0f} | {german_fmt(loops)} ({int(loops*15)}/15) |")


# ── 7. Multi-Agenten-Metriken ──────────────────────────────────────────────

def create_mas_chart(df: pd.DataFrame, output_dir: Path) -> None:
    """Balkendiagramm der MetaGPT MAS-Metriken (M5.1–M5.4) je Aufgabe,
    sortiert nach Aufgabenschwierigkeit (easy → medium → hard)."""
    metagpt = df[df['agent_name'] == 'MetaGPT'].copy()
    if len(metagpt) == 0:
        print("  Keine MetaGPT-Daten gefunden.")
        return

    # Nach Schwierigkeit sortieren
    metagpt['_diff_order'] = (
        metagpt['issue_name'].map(TASK_DIFFICULTY)
        .map(DIFFICULTY_ORDER)
        .fillna(0)
    )
    metagpt = metagpt.sort_values('_diff_order').reset_index(drop=True)

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))

    for ax, metric in zip(axes, MAS_METRICS):
        mask   = metagpt[metric].notna()
        vals   = metagpt.loc[mask, metric].values
        issues = metagpt.loc[mask, 'issue_name'].values
        diffs  = metagpt.loc[mask, 'issue_name'].map(TASK_DIFFICULTY).fillna('easy').values

        short_issues = [
            i.split('__')[1].replace('-', '\n', 1) if '__' in i else i
            for i in issues
        ]
        bar_colors = ['#2ca02c' if v >= 0.6 else ('#ff7f0e' if v >= 0.4 else '#d62728')
                      for v in vals]

        # Hintergrundbänder je Schwierigkeitsgruppe
        y = 0
        for diff in ['easy', 'medium', 'hard']:
            count = int((diffs == diff).sum())
            if count > 0:
                ax.axhspan(y - 0.5, y + count - 0.5,
                           color=DIFFICULTY_BG[diff], alpha=1.0, zorder=0)
                y += count

        ax.barh(range(len(vals)), vals, color=bar_colors,
                edgecolor='black', alpha=0.85, zorder=2)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(short_issues, fontsize=7)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel('Score', fontsize=10)
        ax.set_title(METRIC_LABELS.get(metric, metric).replace('\n', ' '),
                     fontsize=10, fontweight='bold')

        mean_val = metagpt[metric].mean()
        ax.axvline(x=mean_val, color='black', linestyle='--', linewidth=1.5,
                   label=f'Mittelwert: {german_fmt(mean_val)}')
        ax.legend(fontsize=8)
        ax.grid(axis='x', alpha=0.3, zorder=1)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(german_axis_formatter(1))

    # Gemeinsame Legende für Schwierigkeitsgrad
    from matplotlib.patches import Patch
    diff_handles = [
        Patch(facecolor=DIFFICULTY_BG[d], edgecolor='gray', label=DIFFICULTY_LABEL[d])
        for d in ['easy', 'medium', 'hard']
    ]
    fig.legend(handles=diff_handles, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=True,
               title='Aufgabenschwierigkeit')

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig(fig, output_dir, 'mas_metrics_detail')
    plt.close()


def create_mas_chart_2x2(df: pd.DataFrame, output_dir: Path) -> None:
    """2×2-Version des MAS-Balkendiagramms (M5.1–M5.4), größere Schrift."""
    metagpt = df[df['agent_name'] == 'MetaGPT'].copy()
    if len(metagpt) == 0:
        print("  Keine MetaGPT-Daten gefunden.")
        return

    metagpt['_diff_order'] = (
        metagpt['issue_name'].map(TASK_DIFFICULTY)
        .map(DIFFICULTY_ORDER)
        .fillna(0)
    )
    metagpt = metagpt.sort_values('_diff_order').reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()

    for ax, metric in zip(axes, MAS_METRICS):
        mask   = metagpt[metric].notna()
        vals   = metagpt.loc[mask, metric].values
        issues = metagpt.loc[mask, 'issue_name'].values
        diffs  = metagpt.loc[mask, 'issue_name'].map(TASK_DIFFICULTY).fillna('easy').values

        short_issues = [
            i.split('__')[1].replace('-', '\n', 1) if '__' in i else i
            for i in issues
        ]
        bar_colors = ['#2ca02c' if v >= 0.6 else ('#ff7f0e' if v >= 0.4 else '#d62728')
                      for v in vals]

        y = 0
        for diff in ['easy', 'medium', 'hard']:
            count = int((diffs == diff).sum())
            if count > 0:
                ax.axhspan(y - 0.5, y + count - 0.5,
                           color=DIFFICULTY_BG[diff], alpha=1.0, zorder=0)
                y += count

        ax.barh(range(len(vals)), vals, color=bar_colors,
                edgecolor='black', alpha=0.85, zorder=2)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(short_issues, fontsize=12)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel('Score', fontsize=13)
        ax.set_title(METRIC_LABELS.get(metric, metric).replace('\n', ' '),
                     fontsize=14, fontweight='bold')

        mean_val = metagpt[metric].mean()
        ax.axvline(x=mean_val, color='black', linestyle='--', linewidth=1.5,
                   label=f'Mittelwert: {german_fmt(mean_val)}')
        ax.legend(fontsize=11)
        ax.grid(axis='x', alpha=0.3, zorder=1)
        ax.invert_yaxis()
        ax.tick_params(axis='x', labelsize=14)
        ax.xaxis.set_major_formatter(german_axis_formatter(1))

    from matplotlib.patches import Patch
    diff_handles = [
        Patch(facecolor=DIFFICULTY_BG[d], edgecolor='gray', label=DIFFICULTY_LABEL[d])
        for d in ['easy', 'medium', 'hard']
    ]
    fig.legend(handles=diff_handles, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.01), fontsize=12, frameon=True,
               title='Aufgabenschwierigkeit', title_fontsize=12)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save_fig(fig, output_dir, 'mas_metrics_detail_2x2')
    plt.close()

    print("\n  | Metrik | Ø | Median | Std | Min | Max |")
    print("  |--------|---|--------|-----|-----|-----|")
    for m in MAS_METRICS:
        vals  = metagpt[m].dropna()
        label = METRIC_LABELS.get(m, m).replace('\n', ' ')
        print(f"  | {label:35s} | {german_fmt(vals.mean())} | {german_fmt(vals.median())} | "
              f"{german_fmt(vals.std())} | {german_fmt(vals.min())} | {german_fmt(vals.max())} |")


# ── 8. Metrik-Korrelationsmatrix (RQ4) ────────────────────────────────────

def create_metric_correlation_matrix(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Spearman-Korrelationsmatrix zwischen Prozessmetriken (nur Einzelagenten).
    Verwendung: RQ4 – Identifikation redundanter Metriken im Set.
    LaTeX-Einbindung: \\input{metric_correlation_matrix.pgf}
    """
    single_agents = ['OpenHands', 'SWE-Agent', 'live-swe-agent']
    single_df = df[df['agent_name'].isin(single_agents)]

    metrics = CORR_METRICS
    labels  = [METRIC_LABELS.get(m, m).replace('\n', ' ') for m in metrics]
    n = len(metrics)

    # Paarweise Spearman-Korrelation berechnen
    corr_matrix = np.full((n, n), np.nan)
    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            x = single_df[m1].dropna()
            y = single_df[m2].dropna()
            common_idx = x.index.intersection(y.index)
            if len(common_idx) >= 5:
                rho, _ = spearmanr(x.loc[common_idx], y.loc[common_idx])
                corr_matrix[i, j] = rho

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')

    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, '—', ha='center', va='center', fontsize=13, color='gray')
            else:
                color = 'white' if abs(val) > 0.7 else 'black'
                ax.text(j, i, german_fmt(val), ha='center', va='center',
                        fontsize=13, fontweight='bold', color=color)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=12, rotation=35, ha='right')
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=12)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(r'Spearman-Korrelationskoeffizient $\rho$', fontsize=13)
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    cbar.set_ticklabels(['-1,0', '-0,5', '0,0', '0,5', '1,0'], fontsize=11)

    plt.tight_layout()
    save_fig(fig, output_dir, 'metric_correlation_matrix')
    plt.close()

    # Konsole: stark korrelierte Paare (|ρ| ≥ 0.6, kein Diagonal)
    print("\n  Stark korrelierte Metrik-Paare (|ρ| ≥ 0,60):")
    for i in range(n):
        for j in range(i + 1, n):
            if not np.isnan(corr_matrix[i, j]) and abs(corr_matrix[i, j]) >= 0.6:
                print(f"  {metrics[i]} ↔ {metrics[j]}: ρ = {german_fmt(corr_matrix[i, j])}")


# ── Hauptprogramm ──────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("RQ1: Deskriptive Analyse der Agentensysteme")
    print("Datenquelle:", INPUT_CSV.relative_to(Path(__file__).parent))
    print("=" * 70)

    if not INPUT_CSV.exists():
        print(f"Fehler: Eingabedatei nicht gefunden: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    df = prepare_data(df)
    print(f"\nGeladen: {len(df)} Zeilen | "
          f"Agenten: {df['agent_name'].unique().tolist()} | "
          f"Aufgaben: {df['issue_name'].nunique()}")

    output_dir = EVAL_DIR

    # 1. Aggregierte Statistiken
    print("\n" + "=" * 70)
    print("1. AGGREGIERTE STATISTIKEN")
    print("=" * 70)
    summary_df = compute_summary(df)
    summary_path = output_dir / 'descriptive_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Zusammenfassung gespeichert: {summary_path.name}")

    # 1b. Tabelle 5.2 Export (Kategorie-1-Metriken für Kapitel 5)
    print("\n── 1b. Tabelle 5.2 Export (M1.1 + M1.2) ──")
    export_table_5_2(df, output_dir)

    print("\n  Mittlere Scores je Agent (alle 0–1-Metriken):")
    header = "  | Agent           | M1.1 | M2.1↑ | M2.2 | M2.4 | M2.5 | M3.1 | M3.2 | M3.3 | M4.1 |"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for agent in AGENT_ORDER:
        vals = [german_fmt(get_agent_mean(df, agent, m)) for m in HEATMAP_METRICS]
        print(f"  | {agent:15s} | {' | '.join(vals)} |")

    # 2. Radar-Diagramm
    print("\n── 2. Radar-Diagramm ──")
    create_radar_chart(df, output_dir)

    # 3. Box-Plots
    print("\n── 3. Box-Plots ──")
    create_boxplots(df, output_dir)
    create_boxplots_4x2(df, output_dir)

    # 4. Erfolg vs. Misserfolg (→ RQ3)
    print("\n── 4. Erfolg vs. Misserfolg (für RQ3) ──")
    create_success_failure_comparison(df, output_dir)

    # 5. Heatmap
    print("\n── 5. Leistungs-Heatmap ──")
    create_heatmap(df, output_dir)

    # 6. Kosten-Nutzen
    print("\n── 6. Kosten-Nutzen-Diagramm ──")
    create_cost_benefit_chart(df, output_dir)

    # 7. MAS-Metriken (MetaGPT)
    print("\n── 7. Multi-Agenten-Metriken (MetaGPT) ──")
    create_mas_chart(df, output_dir)
    create_mas_chart_2x2(df, output_dir)

    # 8. Metrik-Korrelationsmatrix (→ RQ4)
    print("\n── 8. Metrik-Korrelationsmatrix (für RQ4) ──")
    create_metric_correlation_matrix(df, output_dir)

    print("\n" + "=" * 70)
    print("Fertig! Alle Ausgaben in:", output_dir)
    print("PGF-Einbindung in LaTeX: \\input{evaluation_results/NAME.pgf}")
    print("=" * 70)


if __name__ == "__main__":
    main()
