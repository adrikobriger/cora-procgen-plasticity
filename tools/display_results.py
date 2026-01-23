"""
Display analysis results in a pretty formatted table.
Works with the output from analyze_results.py.
"""
import argparse
import pandas as pd
import json


def display_pretty_table(csv_path, json_path=None):
    """Display results in a formatted table."""
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Load JSON for additional metadata if available
    metadata = {}
    if json_path:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data:
                    metadata = data[0]
        except:
            pass
    
    print("\n" + "="*120)
    print("CONTINUAL RL RESULTS SUMMARY")
    print("="*120)
    
    # Display metadata
    if metadata:
        statistic = metadata.get('statistic', 'median')
        n_seeds = metadata.get('n_seeds', 'N/A')
        bootstrap = metadata.get('bootstrap_samples', 'N/A')
        last_k = metadata.get('last_k_rank', 10)
        print(f"Seeds per method: {n_seeds} | Bootstrap: {bootstrap} | Statistic: {statistic.upper()} | CI: 95%")
    else:
        statistic = df['statistic'].iloc[0] if 'statistic' in df.columns else 'median'
        n_seeds = df['n_seeds'].iloc[0] if 'n_seeds' in df.columns else 'N/A'
        last_k = df['last_k_rank'].iloc[0] if 'last_k_rank' in df.columns else 10
        print(f"Seeds per method: {n_seeds} | Statistic: {statistic.upper()} | CI: 95%")
    
    print("="*140)
    
    # Extract key columns
    display_cols = []
    rename_map = {}
    
    # Method name
    if 'method' in df.columns:
        display_cols.append('method')
        rename_map['method'] = 'Method'
    
    # Peak IQM Return (most important)
    peak_col = f'peak_iqm_return_{statistic}'
    if peak_col in df.columns:
        df['Peak Return'] = df.apply(
            lambda row: f"{row[peak_col]:.3f} [{row['peak_iqm_return_ci_low']:.3f}, {row['peak_iqm_return_ci_high']:.3f}]",
            axis=1
        )
        display_cols.append('Peak Return')
    
    # Final IQM Return
    final_col = f'final_iqm_return_{statistic}'
    if final_col in df.columns:
        df['Final Return'] = df.apply(
            lambda row: f"{row[final_col]:.3f} [{row['final_iqm_return_ci_low']:.3f}, {row['final_iqm_return_ci_high']:.3f}]",
            axis=1
        )
        display_cols.append('Final Return')
    
    # Forgetting
    forget_col = f'max_isolated_forgetting_{statistic}'
    if forget_col in df.columns:
        df['Max Forgetting'] = df.apply(
            lambda row: f"{row[forget_col]:.3f} [{row['max_isolated_forgetting_ci_low']:.3f}, {row['max_isolated_forgetting_ci_high']:.3f}]",
            axis=1
        )
        display_cols.append('Max Forgetting')
    
    # Effective Rank
    rank_col = f'final_effective_rank_{statistic}'
    if rank_col in df.columns:
        df['Rank'] = df.apply(
            lambda row: f"{row[rank_col]:.2f} [{row['final_effective_rank_ci_low']:.2f}, {row['final_effective_rank_ci_high']:.2f}]",
            axis=1
        )
        display_cols.append('Rank')
    
    # Dormant Fraction
    dorm_col = f'final_dormant_frac_{statistic}'
    if dorm_col in df.columns:
        df['Dormant %'] = df.apply(
            lambda row: f"{row[dorm_col]*100:.1f} [{row['final_dormant_frac_ci_low']*100:.1f}, {row['final_dormant_frac_ci_high']*100:.1f}]",
            axis=1
        )
        display_cols.append('Dormant %')
    
    # Display table
    print("\n" + df[display_cols].to_string(index=False))
    
        print("\n" + "="*120)
        print("NOTES")
        print("="*120)
        print(f"Format: {statistic.upper()} [95% CI_low, CI_high]")
        print(f"Rank uses average of last {last_k} effective-rank points.")
        print("="*120 + "\n")
    
    # Summary ranking
    print("="*120)
    print("RANKING BY PEAK RETURN (Higher is Better)")
    print("="*120)
    
    df_sorted = df.sort_values(peak_col, ascending=False)
    ranking_cols = ['method']
    if peak_col in df.columns:
        ranking_cols.append(peak_col)
    if forget_col in df.columns:
        ranking_cols.append(forget_col)
    if dorm_col in df.columns:
        ranking_cols.append(dorm_col)
    
    ranking_display = df_sorted[ranking_cols].copy()
    ranking_display.columns = ['Method', f'Peak Return ({statistic})', 'Max Forgetting', 'Dormant Frac']
    ranking_display = ranking_display.round(3)
    
    print("\n" + ranking_display.to_string(index=False))
    print("\n" + "="*120 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Display analysis results in a compact table')
    parser.add_argument('--csv', type=str, default='results/summary_table.csv',
                       help='Path to summary_table.csv')
    parser.add_argument('--json', type=str, default='results/summary_table.json',
                       help='Path to summary_table.json (optional, for metadata)')
    
    args = parser.parse_args()
    display_pretty_table(args.csv, args.json)


if __name__ == '__main__':
    main()
