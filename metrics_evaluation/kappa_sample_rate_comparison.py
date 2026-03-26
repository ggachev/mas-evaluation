#!/usr/bin/env python3
"""
Weighted Kappa Analysis for LLM Judge Comparison.

Part 1: Sample Rate Comparison (default vs 1_step) per LLM
  - How STABLE is each LLM across sample rates? (Intra-model Kappa)
  - Which config is closer to human? (Spearman side-by-side)

Part 2: Cross-Model Comparison (LLM A vs LLM B) at same config
  - Do different LLMs AGREE with each other? (Inter-model Kappa)
  - Are metrics model-independent or model-sensitive?

Answers RQ5: How robust are evaluation results to model/config variation?
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────

EVAL_DIR = Path(__file__).parent / 'evaluation_results'

# Pairs: (label, default_dir, 1step_dir)
LLM_PAIRS = [
    ('GPT-OSS-120b',   'default_gptoss120b',   '1_step_gptoss120b'),
    ('Qwen3-235b',     'default_qwen3_235b',   '1_step_qwen3_235b'),
    ('GPT-4o-mini-8b', 'default_gpt4omini_8b', '1_step_gpt4omini_8b'),
]

# LLM-judge metrics (same as spearman_correlation.py)
LLM_METRICS = ['M2.2', 'M2.3', 'M2.4', 'M2.5', 'M3.1', 'M3.2', 'M4.1', 'M5.1']

METRIC_DESCRIPTIONS = {
    'M2.2': 'Trajectory Efficiency',
    'M2.3': 'Global Strategy Consistency',
    'M2.4': 'Stepwise Reasoning Quality',
    'M2.5': 'Role Adherence',
    'M3.1': 'Tool Selection Quality',
    'M3.2': 'Tool Execution Success',
    'M4.1': 'Context Utilization',
    'M5.1': 'Communication Efficiency (MAS)',
}

# Cross-model pairs: (label_A, label_B, dir_A, dir_B) — same config
CROSS_MODEL_PAIRS_DEFAULT = [
    ('GPT-OSS-120b', 'Qwen3-235b',     'default_gptoss120b', 'default_qwen3_235b'),
    ('GPT-OSS-120b', 'GPT-4o-mini-8b', 'default_gptoss120b', 'default_gpt4omini_8b'),
    ('Qwen3-235b',   'GPT-4o-mini-8b', 'default_qwen3_235b', 'default_gpt4omini_8b'),
]

CROSS_MODEL_PAIRS_1STEP = [
    ('GPT-OSS-120b', 'Qwen3-235b',     '1_step_gptoss120b', '1_step_qwen3_235b'),
    ('GPT-OSS-120b', 'GPT-4o-mini-8b', '1_step_gptoss120b', '1_step_gpt4omini_8b'),
    ('Qwen3-235b',   'GPT-4o-mini-8b', '1_step_qwen3_235b', '1_step_gpt4omini_8b'),
]

# ── Helpers ────────────────────────────────────────────────────────────────

def discretize_to_bins(values: pd.Series, n_bins: int = 5) -> pd.Series:
    """
    Discretize 0-1 scores into n bins (matching Likert 1-5 scale).
    [0, 0.2] → 1, (0.2, 0.4] → 2, (0.4, 0.6] → 3, (0.6, 0.8] → 4, (0.8, 1.0] → 5
    """
    bins = np.linspace(0, 1, n_bins + 1)
    # pd.cut with right=True: (0, 0.2], (0.2, 0.4], ...
    # Include 0 in first bin by using include_lowest=True
    labels = list(range(1, n_bins + 1))
    return pd.cut(values, bins=bins, labels=labels, include_lowest=True).astype(float)


def interpret_kappa(kappa: float) -> str:
    """Interpret Weighted Kappa (Landis & Koch, 1977)."""
    if kappa < 0:
        return "Poor"
    elif kappa < 0.20:
        return "Slight"
    elif kappa < 0.40:
        return "Fair"
    elif kappa < 0.60:
        return "Moderate"
    elif kappa < 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"


def interpret_significance(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "n.s."


# ── Core ───────────────────────────────────────────────────────────────────

def load_pair(default_dir: str, step1_dir: str) -> pd.DataFrame:
    """Load and merge consolidated_results from two runs."""
    df_default = pd.read_csv(EVAL_DIR / default_dir / 'consolidated_results.csv')
    df_1step = pd.read_csv(EVAL_DIR / step1_dir / 'consolidated_results.csv')

    merged = pd.merge(
        df_default, df_1step,
        on=['agent_name', 'issue_name'],
        suffixes=('_default', '_1step'),
        how='inner'
    )
    return merged


def calculate_kappa_for_pair(merged_df: pd.DataFrame, llm_label: str) -> list:
    """Calculate Weighted Kappa for each metric between default and 1_step."""
    results = []

    for metric in LLM_METRICS:
        col_default = f'{metric}_default'
        col_1step = f'{metric}_1step'

        if col_default not in merged_df.columns or col_1step not in merged_df.columns:
            continue

        # Get paired values, drop NaN
        paired = merged_df[[col_default, col_1step]].dropna()
        n = len(paired)

        if n < 3:
            results.append({
                'LLM': llm_label,
                'Metric': metric,
                'Description': METRIC_DESCRIPTIONS.get(metric, ''),
                'Weighted_Kappa': np.nan,
                'N': n,
                'Interpretation': 'Insufficient data',
                'Mean_Default': np.nan,
                'Mean_1Step': np.nan,
                'Mean_Diff': np.nan,
            })
            continue

        # Discretize both into 5 bins
        bins_default = discretize_to_bins(paired[col_default])
        bins_1step = discretize_to_bins(paired[col_1step])

        # Weighted Kappa (quadratic weights)
        try:
            kappa = cohen_kappa_score(
                bins_default, bins_1step,
                weights='quadratic',
                labels=[1, 2, 3, 4, 5]
            )
        except Exception:
            kappa = np.nan

        mean_default = paired[col_default].mean()
        mean_1step = paired[col_1step].mean()

        results.append({
            'LLM': llm_label,
            'Metric': metric,
            'Description': METRIC_DESCRIPTIONS.get(metric, ''),
            'Weighted_Kappa': round(kappa, 4) if not np.isnan(kappa) else np.nan,
            'N': n,
            'Interpretation': interpret_kappa(kappa) if not np.isnan(kappa) else 'N/A',
            'Mean_Default': round(mean_default, 4),
            'Mean_1Step': round(mean_1step, 4),
            'Mean_Diff': round(mean_1step - mean_default, 4),
        })

    return results


def load_spearman_results() -> dict:
    """Load existing Spearman correlation results from all runs."""
    spearman_data = {}
    for llm_label, default_dir, step1_dir in LLM_PAIRS:
        for config, run_dir in [('default', default_dir), ('1_step', step1_dir)]:
            csv_path = EVAL_DIR / run_dir / 'spearman_correlation_results.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                key = (llm_label, config)
                spearman_data[key] = df.set_index('Metric')['Spearman_Rho'].to_dict()
    return spearman_data


def build_comparison_table(kappa_results: list, spearman_data: dict) -> pd.DataFrame:
    """Build a combined table: Kappa + Spearman default + Spearman 1_step."""
    rows = []
    for r in kappa_results:
        llm = r['LLM']
        metric = r['Metric']

        rho_default = spearman_data.get((llm, 'default'), {}).get(metric, np.nan)
        rho_1step = spearman_data.get((llm, '1_step'), {}).get(metric, np.nan)

        # Determine winner
        if pd.notna(rho_default) and pd.notna(rho_1step):
            if rho_1step > rho_default:
                better = '1_step'
            elif rho_default > rho_1step:
                better = 'default'
            else:
                better = 'tie'
        else:
            better = 'N/A'

        rows.append({
            'LLM': llm,
            'Metric': metric,
            'Description': r['Description'],
            'Kappa_wq': r['Weighted_Kappa'],
            'Kappa_Interp': r['Interpretation'],
            'N': r['N'],
            'Spearman_Default': round(rho_default, 4) if pd.notna(rho_default) else np.nan,
            'Spearman_1Step': round(rho_1step, 4) if pd.notna(rho_1step) else np.nan,
            'Better_Config': better,
            'Mean_Default': r['Mean_Default'],
            'Mean_1Step': r['Mean_1Step'],
            'Mean_Diff': r['Mean_Diff'],
        })

    return pd.DataFrame(rows)


# ── Cross-Model Core ───────────────────────────────────────────────────────

def load_cross_pair(dir_a: str, dir_b: str) -> pd.DataFrame:
    """Load and merge consolidated_results from two different LLM runs."""
    df_a = pd.read_csv(EVAL_DIR / dir_a / 'consolidated_results.csv')
    df_b = pd.read_csv(EVAL_DIR / dir_b / 'consolidated_results.csv')

    merged = pd.merge(
        df_a, df_b,
        on=['agent_name', 'issue_name'],
        suffixes=('_A', '_B'),
        how='inner'
    )
    return merged


def calculate_cross_model_kappa(merged_df: pd.DataFrame, label_a: str, label_b: str,
                                 config_label: str) -> list:
    """Calculate Weighted Kappa between two different LLMs for each metric."""
    results = []
    pair_label = f'{label_a} vs {label_b}'

    for metric in LLM_METRICS:
        col_a = f'{metric}_A'
        col_b = f'{metric}_B'

        if col_a not in merged_df.columns or col_b not in merged_df.columns:
            continue

        paired = merged_df[[col_a, col_b]].dropna()
        n = len(paired)

        if n < 3:
            results.append({
                'Pair': pair_label,
                'Config': config_label,
                'Metric': metric,
                'Description': METRIC_DESCRIPTIONS.get(metric, ''),
                'Weighted_Kappa': np.nan,
                'N': n,
                'Interpretation': 'Insufficient data',
                'Mean_A': np.nan,
                'Mean_B': np.nan,
            })
            continue

        bins_a = discretize_to_bins(paired[col_a])
        bins_b = discretize_to_bins(paired[col_b])

        try:
            kappa = cohen_kappa_score(
                bins_a, bins_b,
                weights='quadratic',
                labels=[1, 2, 3, 4, 5]
            )
        except Exception:
            kappa = np.nan

        results.append({
            'Pair': pair_label,
            'Config': config_label,
            'Metric': metric,
            'Description': METRIC_DESCRIPTIONS.get(metric, ''),
            'Weighted_Kappa': round(kappa, 4) if not np.isnan(kappa) else np.nan,
            'N': n,
            'Interpretation': interpret_kappa(kappa) if not np.isnan(kappa) else 'N/A',
            'Mean_A': round(paired[col_a].mean(), 4),
            'Mean_B': round(paired[col_b].mean(), 4),
        })

    return results


# ── Visualizations ─────────────────────────────────────────────────────────

def create_kappa_barchart(comparison_df: pd.DataFrame, output_dir: Path):
    """Grouped bar chart: Kappa per metric, grouped by LLM."""
    llms = comparison_df['LLM'].unique()
    metrics = [m for m in LLM_METRICS if m in comparison_df['Metric'].values]

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(metrics))
    width = 0.25
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, llm in enumerate(llms):
        subset = comparison_df[comparison_df['LLM'] == llm]
        kappas = []
        for m in metrics:
            row = subset[subset['Metric'] == m]
            k = row['Kappa_wq'].values[0] if len(row) > 0 else np.nan
            kappas.append(k)

        bars = ax.bar(x + i * width, kappas, width, label=llm, color=colors[i], alpha=0.85, edgecolor='black')

        # Value labels
        for bar, k in zip(bars, kappas):
            if pd.notna(k):
                ax.annotate(f'{k:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{m}\n({METRIC_DESCRIPTIONS.get(m, "")})' for m in metrics],
                       fontsize=8, ha='center')
    ax.set_ylabel('Weighted Kappa (quadratic)', fontsize=11)
    ax.set_title('Sample Rate Stability: Weighted Kappa (default vs 1_step) per LLM', fontsize=13)
    ax.legend(title='LLM Judge')
    ax.axhline(y=0.6, color='green', linestyle='--', alpha=0.4, label='Substantial (0.6)')
    ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.4, label='Moderate (0.4)')
    ax.set_ylim(-0.3, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'kappa_sample_rate_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'kappa_sample_rate_comparison.pdf', bbox_inches='tight')
    print(f"\nKappa bar chart saved to {output_dir / 'kappa_sample_rate_comparison.png'}")
    plt.close()


def create_spearman_comparison_chart(comparison_df: pd.DataFrame, output_dir: Path):
    """Side-by-side Spearman: default vs 1_step for each LLM."""
    llms = comparison_df['LLM'].unique()
    metrics = [m for m in LLM_METRICS if m in comparison_df['Metric'].values]

    fig, axes = plt.subplots(1, len(llms), figsize=(6 * len(llms), 6), sharey=True)
    if len(llms) == 1:
        axes = [axes]

    for ax, llm in zip(axes, llms):
        subset = comparison_df[comparison_df['LLM'] == llm]

        rho_default = []
        rho_1step = []
        for m in metrics:
            row = subset[subset['Metric'] == m]
            rho_default.append(row['Spearman_Default'].values[0] if len(row) > 0 else np.nan)
            rho_1step.append(row['Spearman_1Step'].values[0] if len(row) > 0 else np.nan)

        x = np.arange(len(metrics))
        width = 0.35

        ax.bar(x - width / 2, rho_1step, width, label='Standard (SR=1)', color='#1f77b4', alpha=0.8, edgecolor='black')
        ax.bar(x + width / 2, rho_default, width, label='Reduziert (SR=5)', color='#ff7f0e', alpha=0.8, edgecolor='black')

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=9, rotation=45, ha='right')
        ax.set_title(llm, fontsize=12, fontweight='bold')
        ax.set_ylabel('Spearman rho (vs. Mensch)' if ax == axes[0] else '', fontsize=10)
        ax.axhline(y=0.6, color='green', linestyle='--', alpha=0.4)
        ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.4)
        ax.set_ylim(-0.5, 1.05)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'spearman_sample_rate_comparison.png', dpi=150, bbox_inches='tight')
    try:
        plt.savefig(output_dir / 'spearman_sample_rate_comparison.pgf', bbox_inches='tight')
        print(f"Spearman comparison chart saved to {output_dir / 'spearman_sample_rate_comparison.png'} + .pgf")
    except Exception as e:
        print(f"Spearman comparison chart saved to {output_dir / 'spearman_sample_rate_comparison.png'}  (PGF fehlgeschlagen: {e})")
    plt.close()


def create_cross_model_barchart(cross_df: pd.DataFrame, config_label: str, output_dir: Path):
    """Grouped bar chart: Cross-model Kappa per metric, grouped by LLM pair."""
    pairs = cross_df['Pair'].unique()
    metrics = [m for m in LLM_METRICS if m in cross_df['Metric'].values]

    fig, ax = plt.subplots(figsize=(16, 7))

    x = np.arange(len(metrics))
    width = 0.25
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, pair in enumerate(pairs):
        subset = cross_df[cross_df['Pair'] == pair]
        kappas = []
        for m in metrics:
            row = subset[subset['Metric'] == m]
            k = row['Weighted_Kappa'].values[0] if len(row) > 0 else np.nan
            kappas.append(k)

        bars = ax.bar(x + i * width, kappas, width, label=pair, color=colors[i], alpha=0.85, edgecolor='black')

        for bar, k in zip(bars, kappas):
            if pd.notna(k):
                ax.annotate(f'{k:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{m}\n({METRIC_DESCRIPTIONS.get(m, "")})' for m in metrics],
                       fontsize=12, rotation=30, ha='right')
    ax.set_ylabel('Weighted Kappa (quadratic)', fontsize=13)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(title='LLM Pair', fontsize=11, title_fontsize=12)
    ax.axhline(y=0.6, color='green', linestyle='--', alpha=0.4)
    ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.4)
    ax.set_ylim(-0.3, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fname = f'kappa_cross_model_{config_label.lower().replace(" ", "_")}'
    plt.savefig(output_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'{fname}.pdf', bbox_inches='tight')
    (output_dir / f'{fname}.tex').write_text(
        r'\includegraphics[width=\linewidth]{' + fname + r'.pdf}' + '\n',
        encoding='utf-8',
    )
    try:
        plt.savefig(output_dir / f'{fname}.pgf', format='pgf', bbox_inches='tight')
        print(f"Cross-model chart saved: {fname}.png + .pdf + .pgf + .tex")
    except Exception as e:
        print(f"Cross-model chart saved: {fname}.png + .pdf + .tex  (PGF fehlgeschlagen: {e})")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    output_dir = EVAL_DIR
    print("=" * 70)
    print("Weighted Kappa: Sample Rate Comparison (default vs 1_step)")
    print("=" * 70)

    # 1. Calculate Kappa for each LLM pair
    all_kappa_results = []
    for llm_label, default_dir, step1_dir in LLM_PAIRS:
        print(f"\n── {llm_label} ──")
        merged = load_pair(default_dir, step1_dir)
        print(f"  Matched rows: {len(merged)}")
        kappa_results = calculate_kappa_for_pair(merged, llm_label)
        all_kappa_results.extend(kappa_results)

        for r in kappa_results:
            k = r['Weighted_Kappa']
            k_str = f"{k:.4f}" if pd.notna(k) else "N/A"
            print(f"  {r['Metric']}: κ_w={k_str} ({r['Interpretation']}), "
                  f"mean_default={r['Mean_Default']}, mean_1step={r['Mean_1Step']}")

    # 2. Load existing Spearman results
    print("\n── Loading Spearman results ──")
    spearman_data = load_spearman_results()
    for key in sorted(spearman_data.keys()):
        print(f"  Loaded: {key[0]} ({key[1]})")

    # 3. Build combined comparison table
    comparison_df = build_comparison_table(all_kappa_results, spearman_data)

    # 4. Save results
    csv_path = output_dir / 'kappa_sample_rate_results.csv'
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # 5. Print markdown table
    print("\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    print("\n| LLM | Metric | κ_w | Interp. | ρ_default | ρ_1step | Better |")
    print("|-----|--------|-----|---------|-----------|---------|--------|")
    for _, row in comparison_df.iterrows():
        k = f"{row['Kappa_wq']:.3f}" if pd.notna(row['Kappa_wq']) else "N/A"
        rd = f"{row['Spearman_Default']:.3f}" if pd.notna(row['Spearman_Default']) else "N/A"
        r1 = f"{row['Spearman_1Step']:.3f}" if pd.notna(row['Spearman_1Step']) else "N/A"
        print(f"| {row['LLM']} | {row['Metric']} | {k} | {row['Kappa_Interp']} | "
              f"{rd} | {r1} | {row['Better_Config']} |")

    # 6. Summary per LLM
    print("\n" + "=" * 70)
    print("SUMMARY PER LLM")
    print("=" * 70)
    for llm in comparison_df['LLM'].unique():
        subset = comparison_df[comparison_df['LLM'] == llm]
        valid_kappas = subset['Kappa_wq'].dropna()
        wins_1step = (subset['Better_Config'] == '1_step').sum()
        wins_default = (subset['Better_Config'] == 'default').sum()

        print(f"\n  {llm}:")
        if len(valid_kappas) > 0:
            print(f"    Avg |κ_w|: {valid_kappas.abs().mean():.4f}")
            print(f"    Min κ_w:   {valid_kappas.min():.4f} ({subset.loc[valid_kappas.idxmin(), 'Metric']})")
            print(f"    Max κ_w:   {valid_kappas.max():.4f} ({subset.loc[valid_kappas.idxmax(), 'Metric']})")
        print(f"    Spearman wins: default={wins_default}, 1_step={wins_1step}")

    # 7. Visualizations (Part 1)
    print("\n── Creating visualizations (Part 1) ──")
    create_kappa_barchart(comparison_df, output_dir)
    create_spearman_comparison_chart(comparison_df, output_dir)

    # ══════════════════════════════════════════════════════════════════════
    # PART 2: Cross-Model Kappa (LLM A vs LLM B, same config)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 2: Cross-Model Weighted Kappa (LLM A vs LLM B)")
    print("=" * 70)

    all_cross_results = []

    for config_label, pairs_list in [('default (SR=5)', CROSS_MODEL_PAIRS_DEFAULT),
                                      ('1_step (SR=1)', CROSS_MODEL_PAIRS_1STEP)]:
        print(f"\n{'─' * 40}")
        print(f"Config: {config_label}")
        print(f"{'─' * 40}")

        for label_a, label_b, dir_a, dir_b in pairs_list:
            print(f"\n  {label_a} vs {label_b}:")
            merged = load_cross_pair(dir_a, dir_b)
            print(f"    Matched rows: {len(merged)}")
            cross_results = calculate_cross_model_kappa(merged, label_a, label_b, config_label)
            all_cross_results.extend(cross_results)

            for r in cross_results:
                k = r['Weighted_Kappa']
                k_str = f"{k:.4f}" if pd.notna(k) else "N/A"
                print(f"    {r['Metric']}: κ_w={k_str} ({r['Interpretation']})")

    # 8. Save cross-model results
    cross_df = pd.DataFrame(all_cross_results)
    cross_csv_path = output_dir / 'kappa_cross_model_results.csv'
    cross_df.to_csv(cross_csv_path, index=False)
    print(f"\nCross-model results saved to: {cross_csv_path}")

    # 9. Print cross-model table
    print("\n" + "=" * 70)
    print("CROSS-MODEL RESULTS TABLE")
    print("=" * 70)
    print("\n| Config | Pair | Metric | κ_w | Interpretation |")
    print("|--------|------|--------|-----|----------------|")
    for _, row in cross_df.iterrows():
        k = f"{row['Weighted_Kappa']:.3f}" if pd.notna(row['Weighted_Kappa']) else "N/A"
        print(f"| {row['Config']} | {row['Pair']} | {row['Metric']} | {k} | {row['Interpretation']} |")

    # 10. Cross-model summary per pair
    print("\n" + "=" * 70)
    print("CROSS-MODEL SUMMARY")
    print("=" * 70)
    for config_label in cross_df['Config'].unique():
        print(f"\n  Config: {config_label}")
        config_subset = cross_df[cross_df['Config'] == config_label]
        for pair in config_subset['Pair'].unique():
            pair_subset = config_subset[config_subset['Pair'] == pair]
            valid_kappas = pair_subset['Weighted_Kappa'].dropna()
            if len(valid_kappas) > 0:
                print(f"    {pair}:")
                print(f"      Avg κ_w: {valid_kappas.mean():.4f}")
                print(f"      Min κ_w: {valid_kappas.min():.4f} ({pair_subset.loc[valid_kappas.idxmin(), 'Metric']})")
                print(f"      Max κ_w: {valid_kappas.max():.4f} ({pair_subset.loc[valid_kappas.idxmax(), 'Metric']})")

    # 11. Visualizations (Part 2)
    print("\n── Creating visualizations (Part 2) ──")
    for config_label in cross_df['Config'].unique():
        config_subset = cross_df[cross_df['Config'] == config_label]
        create_cross_model_barchart(config_subset, config_label, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
