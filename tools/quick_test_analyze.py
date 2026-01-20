"""
Quick test to show what analyze_results.py would produce with your current runs.
This creates fake CI just to show the table format.
"""

import sys
sys.path.insert(0, '/home/adri/year31a/cora-procgen-plasticity')

from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

# Your existing runs
runs = {
    "Dense": "/home/adri/year31a/cora-procgen-plasticity/dense/ppo_dense_20260117_044911/ppo_procgen_3_tasks_2_cycles_5m_Jan_17_2026_04.49.14.862817/events.out.tfevents.1768621759.dl-h100gpu1.3393665.0",
    "ReDo": "/home/adri/year31a/cora-procgen-plasticity/tmp_redo_test/ppo_procgen_fruitbot_tiny_Jan_12_2026_20.52.18.721781/events.out.tfevents.1768247541.DESKTOP-5T5G6G8.233516.0"
}

print("=" * 80)
print("EXTRACTING IQM VALUES FROM YOUR RUNS")
print("=" * 80)

all_results = {}

for intervention, event_file in runs.items():
    print(f"\n{intervention}:")
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    iqm_tags = [tag for tag in tags if 'eval_reward_iqm' in tag]
    
    task_iqms = {}
    for tag in sorted(iqm_tags):
        events = ea.Scalars(tag)
        if events:
            task_id = tag.split('/')[-1]
            iqm_val = events[-1].value
            task_iqms[task_id] = iqm_val
            print(f"  Task {task_id}: IQM = {iqm_val:.4f}")
    
    all_results[intervention] = task_iqms

# Create table
print("\n" + "=" * 80)
print("SAMPLE TABLE (with only 1 run, can't compute real CI)")
print("=" * 80)

rows = []
for intervention, task_data in sorted(all_results.items()):
    row = {"Agent Treatment": intervention}
    for i, (task_id, iqm) in enumerate(sorted(task_data.items()), start=1):
        row[f"Task {i} IQM"] = f"{iqm:.2f}"
        row[f"Task {i} 95% CI"] = "N/A (need multiple seeds)"
    rows.append(row)

df = pd.DataFrame(rows)
print(df.to_string(index=False))

print("\n" + "=" * 80)
print("TO GET REAL CONFIDENCE INTERVALS:")
print("=" * 80)
print("You need to run each intervention with multiple seeds (e.g., 5 runs)")
print("Then organize like: runs/exp/{Dense,ReDo}/seed_{0,1,2,3,4}/events.*")
print("Then: python3 tools/analyze_results.py --exp_dir runs/exp --output results.csv")
print("=" * 80)
