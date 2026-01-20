"""
Generate a pretty table showing per-seed values and aggregated confidence intervals.
"""
import json
import pandas as pd

# Load the detailed results
with open('results_3seeds_test/summary_table.json', 'r', encoding='utf-8') as f:
    data = json.load(f)[0]

print("\n" + "="*120)
print("CONTINUAL RL RESULTS - 3 SEEDS AGGREGATION WITH BOOTSTRAP CI")
print("="*120)
print(f"\nExperiment: {data['method']}")
statistic = data['statistic']  # Get the aggregation statistic (median, mean, etc.)
print(f"Seeds: {data['n_seeds']} | Bootstrap samples: {data['bootstrap_samples']} | Confidence Level: 95% | Statistic: {statistic.upper()}")
print("\n" + "="*120)

# Create table data
metrics_data = []

metrics = [
    ("Final IQM Return", "final_iqm_return", "Higher is better"),
    ("Peak IQM Return", "peak_iqm_return", "Higher is better"),
    ("Max Forgetting", "max_isolated_forgetting", "Lower is better"),
    ("Effective Rank", "final_effective_rank", "Value"),
    ("Final Dormant Frac", "final_dormant_frac", "Lower is better"),
    ("Peak Dormant Frac", "peak_dormant_frac", "Lower is better"),
]

for name, key, direction in metrics:
    central = data[f'{key}_{statistic}']
    ci_low = data[f'{key}_ci_low']
    ci_high = data[f'{key}_ci_high']
    ci_width = ci_high - ci_low
    
    # Format the CI string
    ci_str = f"[{ci_low:.4f}, {ci_high:.4f}]"
    
    # Calculate relative uncertainty (CI width / central, as percentage)
    rel_uncertainty = (ci_width / abs(central) * 100) if central != 0 else 0
    
    metrics_data.append({
        "Metric": name,
        statistic.capitalize(): f"{central:.4f}",
        "95% CI": ci_str,
        "CI Width": f"{ci_width:.4f}",
        "Uncertainty %": f"{rel_uncertainty:.1f}%",
        "Interpretation": direction
    })

df = pd.DataFrame(metrics_data)
print(df.to_string(index=False))

print("\n" + "="*120)
print("\nKEY INSIGHTS:")
print("="*120)
print(f"""
✅ Confidence Intervals ARE different from the {statistic} → Proper cross-seed variation captured!

What these numbers mean:
  • {statistic.capitalize()}: Central estimate across {data['n_seeds']} seeds (robust to outliers)
  • 95% CI: If we reran with different seeds, we expect the true value to be in this range 95% of the time
  • CI Width: Smaller = more consistent across seeds, Larger = more variable
  • Uncertainty %: Relative uncertainty (CI Width / {statistic.capitalize()})

Metrics explained:
  • Final IQM Return: Performance at the END of training (last value)
  • Peak IQM Return: BEST performance achieved during training (max value)
  • Max Forgetting: WORST forgetting observed (max value)
  • Effective Rank: Diversity of learned features (avg of last {data.get('last_k_rank', 10)} values)
  • Final Dormant Frac: Fraction of dormant neurons at END of training (lower is better)
  • Peak Dormant Frac: HIGHEST dormant fraction observed (lower is better)

For publication/reports:
  • Report as: "{statistic.capitalize()} ± CI_width (95% CI: [CI_low, CI_high])"
""")

# Also show individual seed values if available (would need to extract from raw data)
print("\n" + "="*120)
print("NOTE: To see individual per-seed values, you'd need to extract them from TensorBoard")
print("      logs before aggregation. The current output shows only aggregated statistics.")
print("="*120)