#!/usr/bin/env python3
"""
RQ1: Descriptive Analysis of Agent Systems.

Generates aggregated statistics, visualizations, and comparisons for all
evaluated agents based on consolidated metric scores.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import matplotlib
import warnings

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.size'] = 11

# ── Configuration ──────────────────────────────────────────────────────────

EVAL_DIR = Path(__file__).parent / 'evaluation_results'
INPUT_CSV = EVAL_DIR / '1_step_gptoss120b' / 'consolidated_results.csv'

# Full radar: all 0-1 metrics + M1.1 + inverted M2.1
RADAR_METRICS = ['M1.1', 'M2.1_inv', 'M2.2', 'M2.4', 'M2.5', 'M3.1', 'M3.2', 'M3.3', 'M4.1']

# Process metrics for boxplots (LLM-judge + deterministic)
PROCESS_METRICS = ['M2.2', 'M2.3', 'M2.4', 'M2.5', 'M3.1', 'M3.2', 'M3.3', 'M4.1']

# Heatmap: all comparable metrics (0-1 scale)
HEATMAP_METRICS = ['M1.1', 'M2.1_inv', 'M2.2', 'M2.4', 'M2.5', 'M3.1', 'M3.2', 'M3.3', 'M4.1']

# Resource metrics
RESOURCE_METRICS = ['M1.2_cost_usd', 'M1.2_tokens', 'M1.2_duration_s', 'M1.2_steps']

# MAS-only metrics
MAS_METRICS = ['M5.1', 'M5.2', 'M5.3', 'M5.4']

METRIC_LABELS = {
    'M1.1': 'M1.1 Task\nSuccess Rate',
    'M1.2_cost_usd': 'M1.2 Cost (USD)',
    'M1.2_tokens': 'M1.2 Tokens',
    'M1.2_duration_s': 'M1.2 Duration (s)',
    'M1.2_steps': 'M1.2 Steps',
    'M2.1': 'M2.1 Loop\nDetection',
    'M2.1_inv': 'M2.1 Loop-Free\nRate',
    'M2.2': 'M2.2 Trajectory\nEfficiency',
    'M2.3': 'M2.3 Global Strategy\nConsistency',
    'M2.4': 'M2.4 Stepwise\nReasoning Quality',
    'M2.5': 'M2.5 Role\nAdherence',
    'M3.1': 'M3.1 Tool Selection\nQuality',
    'M3.2': 'M3.2 Tool Execution\nSuccess',
    'M3.3': 'M3.3 Tool Usage\nEfficiency',
    'M4.1': 'M4.1 Context\nUtilization',
    'M5.1': 'M5.1 Communication\nEfficiency',
    'M5.2': 'M5.2 Information\nDiversity (IDS)',
    'M5.3': 'M5.3 Unique Path\nRedundancy (UPR)',
    'M5.4': 'M5.4 Agent Invocation\nDistribution',
}

AGENT_COLORS = {
    'MetaGPT': '#2ca02c',
    'OpenHands': '#1f77b4',
    'SWE-Agent': '#ff7f0e',
    'live-swe-agent': '#d62728',
}

AGENT_ORDER = ['MetaGPT', 'OpenHands', 'SWE-Agent', 'live-swe-agent']


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns."""
    df = df.copy()
    # Invert M2.1: 1=loop detected (bad) → 0=bad, so invert for "Loop-Free Rate"
    df['M2.1_inv'] = 1 - df['M2.1']
    return df


def get_agent_mean(df: pd.DataFrame, agent: str, metric: str) -> float:
    """Get mean of a metric for an agent, handling NaN."""
    vals = df[df['agent_name'] == agent][metric].dropna()
    return vals.mean() if len(vals) > 0 else np.nan


# ── 1. Aggregation ────────────────────────────────────────────────────────

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, median, std, min, max per agent per metric."""
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
                'Agent': agent,
                'Metric': metric,
                'Label': METRIC_LABELS.get(metric, metric).replace('\n', ' '),
                'Mean': round(vals.mean(), 4),
                'Median': round(vals.median(), 4),
                'Std': round(vals.std(), 4),
                'Min': round(vals.min(), 4),
                'Max': round(vals.max(), 4),
                'N': len(vals),
            })

    return pd.DataFrame(rows)


# ── 2. Full Radar Chart ───────────────────────────────────────────────────

def create_radar_chart(df: pd.DataFrame, output_dir: Path):
    """Full radar chart with all comparable metrics including M1.1 and inverted M2.1."""
    metrics = RADAR_METRICS
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for agent in AGENT_ORDER:
        values = [get_agent_mean(df, agent, m) for m in metrics]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2.5, label=agent,
                color=AGENT_COLORS[agent], markersize=7)
        ax.fill(angles, values, alpha=0.08, color=AGENT_COLORS[agent])

    labels = [METRIC_LABELS.get(m, m).replace('\n', ' ') for m in metrics]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.12), fontsize=11)
    ax.set_title('Agent Performance Profiles (All Metrics)', fontsize=14, pad=25)

    plt.tight_layout()
    plt.savefig(output_dir / 'radar_chart_agents.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'radar_chart_agents.pdf', bbox_inches='tight')
    print(f"Radar chart saved to {output_dir / 'radar_chart_agents.png'}")
    plt.close()


# ── 3. Box Plots ──────────────────────────────────────────────────────────

def create_boxplots(df: pd.DataFrame, output_dir: Path):
    """Box plots per metric grouped by agent."""
    metrics = PROCESS_METRICS
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        data = []
        labels = []
        colors = []

        for agent in AGENT_ORDER:
            vals = df[df['agent_name'] == agent][metric].dropna()
            if len(vals) > 0:
                data.append(vals.values)
                labels.append(agent.replace('live-swe-agent', 'live-swe'))
                colors.append(AGENT_COLORS[agent])

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(METRIC_LABELS.get(metric, metric).replace('\n', ' '), fontsize=10, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=30, labelsize=8)

    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Score Distribution per Agent per Metric', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplots_per_metric.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'boxplots_per_metric.pdf', bbox_inches='tight')
    print(f"Box plots saved to {output_dir / 'boxplots_per_metric.png'}")
    plt.close()


# ── 4. Success vs Failure ─────────────────────────────────────────────────

def create_success_failure_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare metric means for successful vs failed tasks."""
    metrics = ['M2.1_inv', 'M2.2', 'M2.4', 'M2.5', 'M3.1', 'M3.2', 'M3.3', 'M4.1']

    success_df = df[df['M1.1'] == 1]
    failure_df = df[df['M1.1'] == 0]

    print(f"\n  Success: {len(success_df)} tasks, Failure: {len(failure_df)} tasks")

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

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width / 2, means_success, width, label=f'Success (n={len(success_df)})',
           color='#2ca02c', alpha=0.8, edgecolor='black')
    ax.bar(x + width / 2, means_failure, width, label=f'Failure (n={len(failure_df)})',
           color='#d62728', alpha=0.8, edgecolor='black')

    for i, (ms, mf, p) in enumerate(zip(means_success, means_failure, p_values)):
        max_h = max(ms, mf) + 0.03
        if pd.notna(p):
            if p < 0.001:
                sig = '***'
            elif p < 0.01:
                sig = '**'
            elif p < 0.05:
                sig = '*'
            else:
                sig = 'n.s.'
            ax.text(i, max_h, sig, ha='center', va='bottom', fontsize=10, fontweight='bold')

    labels = [METRIC_LABELS.get(m, m).replace('\n', ' ') for m in metrics]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('Mean Score (0-1)', fontsize=11)
    ax.set_title('Process Metrics: Successful vs Failed Tasks (Mann-Whitney U)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'success_vs_failure_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'success_vs_failure_comparison.pdf', bbox_inches='tight')
    print(f"Success vs Failure chart saved to {output_dir / 'success_vs_failure_comparison.png'}")
    plt.close()

    print("\n  | Metric | Mean (Success) | Mean (Failure) | Diff | p-value | Sig. |")
    print("  |--------|----------------|----------------|------|---------|------|")
    for m, ms, mf, p in zip(metrics, means_success, means_failure, p_values):
        diff = ms - mf
        p_str = f"{p:.4f}" if pd.notna(p) else "N/A"
        sig = '***' if pd.notna(p) and p < 0.001 else ('**' if pd.notna(p) and p < 0.01 else
              ('*' if pd.notna(p) and p < 0.05 else 'n.s.'))
        label = METRIC_LABELS.get(m, m).replace('\n', ' ')
        print(f"  | {label} | {ms:.3f} | {mf:.3f} | {diff:+.3f} | {p_str} | {sig} |")


# ── 5. Full Heatmap ──────────────────────────────────────────────────────

def create_heatmap(df: pd.DataFrame, output_dir: Path):
    """Full heatmap: Agents x all metrics, with category separators."""
    metrics = HEATMAP_METRICS

    matrix = []
    for agent in AGENT_ORDER:
        row = [get_agent_mean(df, agent, m) for m in metrics]
        matrix.append(row)

    matrix = np.array(matrix)
    labels = [METRIC_LABELS.get(m, m).replace('\n', ' ') for m in metrics]

    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    for i in range(len(AGENT_ORDER)):
        for j in range(len(metrics)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, '—', ha='center', va='center', fontsize=11, color='gray')
            else:
                color = 'white' if val < 0.3 or val > 0.85 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=11, fontweight='bold', color=color)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(labels, fontsize=9, rotation=25, ha='right')
    ax.set_yticks(range(len(AGENT_ORDER)))
    ax.set_yticklabels(AGENT_ORDER, fontsize=12)
    ax.set_title('Agent Performance Heatmap (Mean Scores, higher = better)', fontsize=13, fontweight='bold')

    # Category separators
    # After M1.1 (idx 0), after M2.1_inv+M2.2+M2.4+M2.5 (idx 4), after M3.1+M3.2+M3.3 (idx 7)
    for sep_x in [0.5, 4.5, 7.5]:
        ax.axvline(x=sep_x, color='black', linewidth=2)

    # Category labels at top
    cat_positions = [(0, 'Ergebnis'), (2.5, 'Strategie'), (6, 'Werkzeuge'), (8, 'Wissen')]
    for xpos, cat_label in cat_positions:
        ax.text(xpos, -0.7, cat_label, ha='center', va='center', fontsize=9,
                fontstyle='italic', color='#555555')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Score (0 = schlecht, 1 = gut)', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'agent_comparison_heatmap.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'agent_comparison_heatmap.pdf', bbox_inches='tight')
    print(f"Heatmap saved to {output_dir / 'agent_comparison_heatmap.png'}")
    plt.close()


# ── 6. Cost-Benefit Visualization ────────────────────────────────────────

def create_cost_benefit_chart(df: pd.DataFrame, output_dir: Path):
    """Visual cost-benefit comparison: success rate vs cost/tokens/duration."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    comparisons = [
        ('M1.2_cost_usd', 'Avg Cost (USD)', 'log'),
        ('M1.2_tokens', 'Avg Tokens', 'log'),
        ('M1.2_duration_s', 'Avg Duration (s)', 'log'),
    ]

    for ax, (resource_col, resource_label, scale) in zip(axes, comparisons):
        for agent in AGENT_ORDER:
            agent_df = df[df['agent_name'] == agent]
            success_rate = agent_df['M1.1'].mean()
            resource_val = agent_df[resource_col].mean()

            ax.scatter(resource_val, success_rate, s=200, color=AGENT_COLORS[agent],
                       edgecolors='black', linewidth=1.5, zorder=5, label=agent)
            ax.annotate(agent, (resource_val, success_rate),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=9, fontweight='bold')

        ax.set_xlabel(resource_label, fontsize=11)
        ax.set_ylabel('M1.1 Success Rate', fontsize=11)
        ax.set_xscale(scale)
        ax.set_ylim(-0.05, 0.75)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    axes[0].set_title('Success vs Cost', fontsize=12, fontweight='bold')
    axes[1].set_title('Success vs Tokens', fontsize=12, fontweight='bold')
    axes[2].set_title('Success vs Duration', fontsize=12, fontweight='bold')

    plt.suptitle('M1.1 Task Success Rate vs M1.2 Resource Efficiency', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'cost_benefit_scatter.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'cost_benefit_scatter.pdf', bbox_inches='tight')
    print(f"Cost-benefit scatter saved to {output_dir / 'cost_benefit_scatter.png'}")
    plt.close()

    # Print table
    print("\n  | Agent | M1.1 Success | Avg Cost | Avg Tokens | Avg Duration | Avg Steps | M2.1 Loop Rate |")
    print("  |-------|-------------|----------|------------|-------------|-----------|----------------|")
    for agent in AGENT_ORDER:
        agent_df = df[df['agent_name'] == agent]
        sr = agent_df['M1.1'].mean()
        cost = agent_df['M1.2_cost_usd'].mean()
        tokens = agent_df['M1.2_tokens'].mean()
        dur = agent_df['M1.2_duration_s'].mean()
        steps = agent_df['M1.2_steps'].mean()
        loops = agent_df['M2.1'].mean()
        print(f"  | {agent:15s} | {sr:.2f} ({int(sr*15)}/15) | ${cost:.4f} | "
              f"{tokens:,.0f} | {dur:.0f}s | {steps:.0f} | {loops:.2f} ({int(loops*15)}/15) |")


# ── 7. MAS Metrics Chart ────────────────────────────────────────────────

def create_mas_chart(df: pd.DataFrame, output_dir: Path):
    """Bar chart for MetaGPT MAS-specific metrics (M5.1-M5.4)."""
    metagpt = df[df['agent_name'] == 'MetaGPT']
    if len(metagpt) == 0:
        return

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for ax, metric in zip(axes, MAS_METRICS):
        vals = metagpt[metric].dropna().values
        issues = metagpt.loc[metagpt[metric].notna(), 'issue_name'].values
        # Shorten issue names
        short_issues = [i.split('__')[1].replace('-', '\n', 1) if '__' in i else i for i in issues]

        colors = ['#2ca02c' if v >= 0.6 else ('#ff7f0e' if v >= 0.4 else '#d62728') for v in vals]
        bars = ax.barh(range(len(vals)), vals, color=colors, edgecolor='black', alpha=0.8)

        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(short_issues, fontsize=7)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel('Score', fontsize=10)
        label = METRIC_LABELS.get(metric, metric).replace('\n', ' ')
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.axvline(x=metagpt[metric].mean(), color='black', linestyle='--', linewidth=1.5,
                    label=f'Mean: {metagpt[metric].mean():.2f}')
        ax.legend(fontsize=8)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

    plt.suptitle('MetaGPT Multi-Agent Metrics (M5.1–M5.4) per Issue', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'mas_metrics_detail.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'mas_metrics_detail.pdf', bbox_inches='tight')
    print(f"MAS metrics chart saved to {output_dir / 'mas_metrics_detail.png'}")
    plt.close()

    # Print summary
    print("\n  | Metric | Mean | Median | Std | Min | Max |")
    print("  |--------|------|--------|-----|-----|-----|")
    for m in MAS_METRICS:
        vals = metagpt[m].dropna()
        label = METRIC_LABELS.get(m, m).replace('\n', ' ')
        print(f"  | {label} | {vals.mean():.3f} | {vals.median():.3f} | "
              f"{vals.std():.3f} | {vals.min():.3f} | {vals.max():.3f} |")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("RQ1: Descriptive Analysis of Agent Systems")
    print("=" * 70)

    if not INPUT_CSV.exists():
        print(f"Error: Input file not found: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    df = prepare_data(df)
    print(f"\nLoaded: {len(df)} rows from {INPUT_CSV.name}")
    print(f"Agents: {df['agent_name'].unique().tolist()}")
    print(f"Issues: {df['issue_name'].nunique()}")

    output_dir = EVAL_DIR

    # 1. Aggregation
    print("\n" + "=" * 70)
    print("1. AGGREGATED STATISTICS")
    print("=" * 70)
    summary_df = compute_summary(df)
    summary_path = output_dir / 'descriptive_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    print("\n  Mean Scores per Agent (all 0-1 metrics):")
    print("\n  | Agent | M1.1 | M2.1_inv | M2.2 | M2.4 | M2.5 | M3.1 | M3.2 | M3.3 | M4.1 |")
    print("  |-------|------|----------|------|------|------|------|------|------|------|")
    for agent in AGENT_ORDER:
        vals = [f"{get_agent_mean(df, agent, m):.2f}" for m in HEATMAP_METRICS]
        print(f"  | {agent:15s} | {' | '.join(vals)} |")

    # 2. Radar Chart (full)
    print("\n── 2. Radar Chart (all metrics) ──")
    create_radar_chart(df, output_dir)

    # 3. Box Plots
    print("\n── 3. Box Plots ──")
    create_boxplots(df, output_dir)

    # 4. Success vs Failure
    print("\n── 4. Success vs Failure ──")
    create_success_failure_comparison(df, output_dir)

    # 5. Full Heatmap
    print("\n── 5. Heatmap (all metrics) ──")
    create_heatmap(df, output_dir)

    # 6. Cost-Benefit Visualization
    print("\n── 6. Cost-Benefit ──")
    create_cost_benefit_chart(df, output_dir)

    # 7. MAS Metrics Detail
    print("\n── 7. MAS Metrics (MetaGPT) ──")
    create_mas_chart(df, output_dir)

    print("\n" + "=" * 70)
    print("Done! All outputs in:", output_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
