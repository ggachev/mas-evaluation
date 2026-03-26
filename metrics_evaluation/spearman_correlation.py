#!/usr/bin/env python3
"""
Spearman Correlation Analysis: Auto Scores vs. Manual Expert Scores

This script calculates the validity (correlation) between automated LLM-based
evaluation scores and manual expert annotations using Spearman's rank correlation.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# Mapping between auto score columns and manual annotation columns
METRIC_MAPPING = {
    'M2.2': 'M2.2_Trajectory_Efficiency_1to5',
    'M2.3': 'M2.3_Global_Strategy_1to5',
    'M2.4': 'M2.4_Reasoning_Quality_1to5',
    'M2.5': 'M2.5_Role_Adherence_1to5',
    'M3.1': 'M3.1_Tool_Selection_1to5',
    'M3.2': 'M3.2_Tool_Execution_Quality_1to5',
    'M4.1': 'M4.1_Context_Utilization_1to5',
    'M5.1': 'M5.1_Communication_1to5_IfMAS',
}

# Metric descriptions for output
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


def interpret_correlation(rho: float) -> str:
    """Interpret Spearman correlation strength."""
    abs_rho = abs(rho)
    if abs_rho >= 0.7:
        strength = "Very Strong"
    elif abs_rho >= 0.6:
        strength = "Strong"
    elif abs_rho >= 0.4:
        strength = "Moderate"
    elif abs_rho >= 0.2:
        strength = "Weak"
    else:
        strength = "Very Weak"

    direction = "positive" if rho >= 0 else "negative"
    return f"{strength} ({direction})"


def interpret_significance(p_value: float) -> str:
    """Interpret p-value significance."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "n.s."


def scale_auto_to_likert(value: float, metric: str = None) -> float:
    """
    Scale auto score from [0,1] to [1,5] Likert scale.
    
    Special handling for M3.2 (Tool Execution):
    - Linear scaling (val * 4 + 1) is too generous.
    - Human Guide defines >50% failure rate as Score 1.
    - So we map 0.5 -> 1.0 and 1.0 -> 5.0.
    - Formula: y = 8(x - 0.5) + 1, clipped to [1, 5].
    """
    if pd.isna(value):
        return np.nan
    
    if metric == 'M3.2':
        # Stricter scaling for execution quality
        # 0.5 success rate becomes 1.0 (Poor)
        # 1.0 success rate becomes 5.0 (Perfect)
        scaled = 8 * (value - 0.5) + 1
        return max(1.0, min(5.0, scaled))
        
    return (value * 4) + 1


def load_and_merge_data(auto_path: Path, manual_path: Path) -> pd.DataFrame:
    """Load both CSVs and merge on common key."""
    # Load auto scores
    auto_df = pd.read_csv(auto_path)
    print(f"Loaded auto scores: {len(auto_df)} rows")

    # Load manual annotations
    manual_df = pd.read_csv(manual_path)
    print(f"Loaded manual annotations: {len(manual_df)} rows")

    # Rename task_id to issue_name for merging
    manual_df = manual_df.rename(columns={'task_id': 'issue_name'})

    # Normalize agent names for consistent merging (handle case differences)
    agent_name_map = {'SWE-agent': 'SWE-Agent'}
    manual_df['agent_name'] = manual_df['agent_name'].replace(agent_name_map)

    # Replace "N/A - Single Agent" strings with NaN (used in M5.1 for non-MAS agents)
    manual_df = manual_df.replace('N/A - Single Agent', np.nan)

    # Convert manual metric columns to numeric (ensures proper dtype after string removal)
    manual_metric_cols = [col for col in manual_df.columns
                          if col.startswith('M') and col not in ('Comments',)]
    for col in manual_metric_cols:
        manual_df[col] = pd.to_numeric(manual_df[col], errors='coerce')

    # Merge on agent_name + issue_name
    merged_df = pd.merge(
        auto_df,
        manual_df,
        on=['agent_name', 'issue_name'],
        how='inner',
        suffixes=('_auto', '_human')
    )
    print(f"Merged dataset: {len(merged_df)} rows")

    return merged_df


def calculate_correlations(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Spearman correlation for each metric pair."""
    results = []

    for auto_col, human_col in METRIC_MAPPING.items():
        # Check if columns exist
        if auto_col not in merged_df.columns:
            print(f"  Warning: Auto column '{auto_col}' not found")
            continue
        if human_col not in merged_df.columns:
            print(f"  Warning: Human column '{human_col}' not found")
            continue

        # Get paired values, filtering out NaN
        paired_data = merged_df[[auto_col, human_col]].dropna()
        n = len(paired_data)

        if n < 3:
            print(f"  {auto_col}: Insufficient data (n={n})")
            results.append({
                'Metric': auto_col,
                'Description': METRIC_DESCRIPTIONS.get(auto_col, ''),
                'Spearman_Rho': np.nan,
                'P_Value': np.nan,
                'N': n,
                'Significance': 'N/A',
                'Interpretation': 'Insufficient data'
            })
            continue

        auto_values = paired_data[auto_col].values
        human_values = paired_data[human_col].values

        # Calculate Spearman correlation
        rho, p_value = spearmanr(auto_values, human_values)

        results.append({
            'Metric': auto_col,
            'Description': METRIC_DESCRIPTIONS.get(auto_col, ''),
            'Spearman_Rho': round(rho, 4),
            'P_Value': round(p_value, 6),
            'N': n,
            'Significance': interpret_significance(p_value),
            'Interpretation': interpret_correlation(rho)
        })

        print(f"  {auto_col}: ρ={rho:.4f}, p={p_value:.4f}, n={n}")

    return pd.DataFrame(results)


def create_scatterplots(merged_df: pd.DataFrame, output_dir: Path):
    """Create scatter plots for each metric pair."""
    # Calculate grid dimensions
    n_metrics = len(METRIC_MAPPING)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for idx, (auto_col, human_col) in enumerate(METRIC_MAPPING.items()):
        ax = axes[idx]

        if auto_col not in merged_df.columns or human_col not in merged_df.columns:
            ax.set_visible(False)
            continue

        # Get paired data
        paired_data = merged_df[[auto_col, human_col, 'agent_name']].dropna()

        if len(paired_data) < 3:
            ax.text(0.5, 0.5, f'{auto_col}\nInsufficient data',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 6)
            ax.set_ylim(0, 1.1)
            continue

        # Scale auto scores to 1-5 for visualization
        auto_scaled = paired_data[auto_col].apply(lambda x: scale_auto_to_likert(x, metric=auto_col))
        human_values = paired_data[human_col]

        # Color by agent
        agent_colors = {
            'OpenHands': '#1f77b4',
            'SWE-Agent': '#ff7f0e',
            'MetaGPT': '#2ca02c',
            'live-swe-agent': '#d62728'
        }
        colors = [agent_colors.get(a, '#999999') for a in paired_data['agent_name']]

        # Scatter plot
        ax.scatter(human_values, auto_scaled, c=colors, alpha=0.7, s=60, edgecolors='white')

        # Add diagonal line (perfect correlation)
        ax.plot([1, 5], [1, 5], 'k--', alpha=0.3, label='Perfect agreement')

        # Calculate and display correlation
        rho, p = spearmanr(paired_data[auto_col], human_values)
        sig = interpret_significance(p)

        ax.set_xlabel('Human Score (1-5)', fontsize=10)
        ax.set_ylabel('Auto Score (scaled 1-5)', fontsize=10)
        ax.set_title(f'{auto_col}: {METRIC_DESCRIPTIONS.get(auto_col, "")}\nρ={rho:.3f} {sig}', fontsize=11)
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(0.5, 5.5)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.grid(True, alpha=0.3)

        # Add jitter to avoid overlapping points
        ax.set_aspect('equal')

    # Hide unused axes
    for idx in range(len(METRIC_MAPPING), len(axes)):
        axes[idx].set_visible(False)

    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=c, markersize=10, label=a)
                       for a, c in agent_colors.items()]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_scatterplots.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'correlation_scatterplots.pdf', bbox_inches='tight')
    print(f"\nScatterplots saved to {output_dir / 'correlation_scatterplots.png'}")
    plt.close()


def create_correlation_heatmap(results_df: pd.DataFrame, output_dir: Path):
    """Create a bar chart showing correlation coefficients."""
    fig, ax = plt.subplots(figsize=(13, 7))

    valid_results = results_df.dropna(subset=['Spearman_Rho'])

    if len(valid_results) == 0:
        print("No valid correlations to plot")
        return

    x = range(len(valid_results))
    rhos = valid_results['Spearman_Rho'].values

    colors = []
    for rho in rhos:
        if abs(rho) >= 0.6:
            colors.append('#2ca02c')
        elif abs(rho) >= 0.4:
            colors.append('#ff7f0e')
        else:
            colors.append('#d62728')

    bars = ax.bar(x, rhos, color=colors, edgecolor='black', alpha=0.8)

    for bar, rho, sig in zip(bars, rhos, valid_results['Significance']):
        height = bar.get_height()
        ax.annotate(f'{rho:.2f}{sig}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=13, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({d})" for m, d in
                        zip(valid_results['Metric'], valid_results['Description'])],
                       rotation=30, ha='right', fontsize=12)
    ax.set_ylabel('Spearman rho', fontsize=14)
    ax.axhline(y=0.6, color='green', linestyle='--', linewidth=1.5,
               alpha=0.6, label='Stark (0,60)')
    ax.axhline(y=0.4, color='orange', linestyle='--', linewidth=1.5,
               alpha=0.6, label='Moderat (0,40)')
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    name = 'correlation_barchart'
    fig.savefig(output_dir / f'{name}.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / f'{name}.pdf', bbox_inches='tight')
    (output_dir / f'{name}.tex').write_text(
        r'\includegraphics[width=\linewidth]{' + name + r'.pdf}' + '\n',
        encoding='utf-8',
    )
    print(f"Bar chart saved to {output_dir / f'{name}.png'} + .pdf + .tex")
    plt.close()


def main():
    script_dir = Path(__file__).parent

    # Define input paths
    auto_path = script_dir / 'evaluation_results' / 'consolidated_results.csv'
    manual_path = script_dir / 'manual_annotations.csv'
    output_dir = script_dir / 'evaluation_results'

    # Verify files exist
    if not auto_path.exists():
        print(f"Error: Auto scores file not found: {auto_path}")
        return
    if not manual_path.exists():
        print(f"Error: Manual annotations file not found: {manual_path}")
        return

    print("=" * 60)
    print("Spearman Correlation Analysis")
    print("=" * 60)

    # Load and merge data
    print("\n1. Loading and merging data...")
    merged_df = load_and_merge_data(auto_path, manual_path)

    # Calculate correlations
    print("\n2. Calculating Spearman correlations...")
    results_df = calculate_correlations(merged_df)

    # Save results to CSV
    results_csv_path = output_dir / 'spearman_correlation_results.csv'
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nResults saved to: {results_csv_path}")

    # Print results as markdown table
    print("\n" + "=" * 60)
    print("RESULTS TABLE")
    print("=" * 60)
    print("\n| Metric | Description | ρ | p-value | N | Sig. | Interpretation |")
    print("|--------|-------------|-----|---------|---|------|----------------|")
    for _, row in results_df.iterrows():
        rho = f"{row['Spearman_Rho']:.4f}" if pd.notna(row['Spearman_Rho']) else "N/A"
        p = f"{row['P_Value']:.4f}" if pd.notna(row['P_Value']) else "N/A"
        print(f"| {row['Metric']} | {row['Description']} | {rho} | {p} | {row['N']} | {row['Significance']} | {row['Interpretation']} |")

    # Create visualizations
    print("\n3. Creating visualizations...")
    create_scatterplots(merged_df, output_dir)
    create_correlation_heatmap(results_df, output_dir)

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    valid_rhos = results_df['Spearman_Rho'].dropna()
    if len(valid_rhos) > 0:
        print(f"  Average |ρ|: {valid_rhos.abs().mean():.4f}")
        print(f"  Highest ρ:   {valid_rhos.max():.4f} ({results_df.loc[valid_rhos.idxmax(), 'Metric']})")
        print(f"  Lowest ρ:    {valid_rhos.min():.4f} ({results_df.loc[valid_rhos.idxmin(), 'Metric']})")

        sig_count = (results_df['P_Value'] < 0.05).sum()
        print(f"  Significant (p<0.05): {sig_count}/{len(valid_rhos)} metrics")

    print("\nDone!")


if __name__ == "__main__":
    main()
