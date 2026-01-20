"""
Quick script to inspect what metrics are stored in a TensorBoard event file.
Usage: python tools/inspect_tensorboard.py <path_to_events_file>
"""

import sys
from tensorboard.backend.event_processing import event_accumulator

if len(sys.argv) < 2:
    print("Usage: python tools/inspect_tensorboard.py <path_to_events_file>")
    sys.exit(1)

event_file = sys.argv[1]

print(f"Reading: {event_file}\n")

ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()

print("=" * 80)
print("AVAILABLE SCALAR TAGS:")
print("=" * 80)

tags = ea.Tags()['scalars']
for tag in sorted(tags):
    events = ea.Scalars(tag)
    if events:
        first_val = events[0].value
        last_val = events[-1].value
        n_events = len(events)
        print(f"{tag}")
        print(f"  └─ {n_events} logged values, first={first_val:.4f}, last={last_val:.4f}")

print("\n" + "=" * 80)
print("EVAL REWARD IQM VALUES (what the analysis script uses):")
print("=" * 80)

iqm_tags = [tag for tag in tags if 'eval_reward_iqm' in tag]
if iqm_tags:
    for tag in sorted(iqm_tags):
        events = ea.Scalars(tag)
        if events:
            final_value = events[-1].value
            task_id = tag.split('/')[-1] if '/' in tag else "unknown"
            print(f"  Task: {task_id:15s} → IQM = {final_value:.4f}")
else:
    print("  (No IQM values found - check if eval was run)")

print("\n" + "=" * 80)
print("HOW IT WORKS:")
print("=" * 80)
print("1. During training, task_base.py logged metrics to TensorBoard")
print("2. TensorBoard stored everything in this events.out.tfevents file")
print("3. The analysis script (analyze_results.py) reads these files")
print("4. It extracts the FINAL IQM value for each task from EACH run")
print("5. Then computes mean IQM and 95% CI across all runs/seeds")
print("=" * 80)
