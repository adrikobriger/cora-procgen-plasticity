"""
Debug script to scan TensorBoard runs and check for missing tags.
Helps diagnose grouping issues and missing forgetting metrics.
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator
import sys


def extract_seed_from_path(path_str):
    """
    Extract seed number from path.
    Looks for patterns like: seed_0, seed_1, seed_2, etc.
    """
    match = re.search(r'seed[_-]?(\d+)', path_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def extract_method_from_path(path_str):
    """
    Extract method/intervention name from path.
    Looks for common intervention names in the path hierarchy.
    """
    path_parts = Path(path_str).parts
    
    # Common intervention names
    known_methods = [
        'dense', 'gmp', 'partial_reinit', 'redo', 'reset', 'set',
        'clear', 'ewc', 'pnc', 'impala', 'ppo', 'sane'
    ]
    
    # Search for known method in path parts
    for part in path_parts:
        part_lower = part.lower()
        for method in known_methods:
            if method in part_lower:
                return method
    
    # If not found, return first non-seed folder name
    for part in path_parts:
        if 'seed' not in part.lower() and part not in ['.', '..']:
            return part
    
    return 'unknown'


def find_event_files(runs_dir):
    """Find all TensorBoard event files in directory."""
    runs_path = Path(runs_dir)
    event_files = list(runs_path.glob('**/events.out.tfevents.*'))
    return event_files


def load_tags(event_file_path):
    """
    Load available scalar tags from event file.
    Returns (tags_list, error_message)
    """
    try:
        ea = event_accumulator.EventAccumulator(str(event_file_path))
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        return tags, None
    except Exception as e:
        return [], str(e)


def find_similar_tags(tags, keyword):
    """Find tags containing the keyword (case-insensitive)."""
    keyword_lower = keyword.lower()
    return [tag for tag in tags if keyword_lower in tag.lower()]


def main():
    parser = argparse.ArgumentParser(
        description='Debug TensorBoard runs - check grouping and missing tags'
    )
    parser.add_argument('--runs-dir', type=str, required=True,
                       help='Root directory containing TensorBoard runs')
    parser.add_argument('--forgetting-tag', type=str, default='forgetting/isolated_avg',
                       help='Expected forgetting tag name')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed per-run information')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Scanning TensorBoard runs in: {args.runs_dir}")
    print(f"{'='*80}\n")
    
    # Find all event files
    event_files = find_event_files(args.runs_dir)
    print(f"Found {len(event_files)} event files\n")
    
    if not event_files:
        print("No event files found!")
        return
    
    # Group by method
    method_to_seeds = defaultdict(set)
    method_to_runs = defaultdict(list)
    runs_with_forgetting = 0
    runs_without_forgetting = 0
    
    # Process each event file
    for event_file in event_files:
        path_str = str(event_file.relative_to(args.runs_dir))
        
        # Extract metadata
        seed = extract_seed_from_path(path_str)
        method = extract_method_from_path(path_str)
        
        # Load tags
        tags, error = load_tags(event_file)
        
        if error:
            if args.verbose:
                print(f"[SKIP] {path_str}")
                print(f"       Error: {error}\n")
            continue
        
        # Check for forgetting tag
        has_forgetting = args.forgetting_tag in tags
        if has_forgetting:
            runs_with_forgetting += 1
        else:
            runs_without_forgetting += 1
        
        # Store info
        method_to_seeds[method].add(seed)
        method_to_runs[method].append({
            'path': path_str,
            'seed': seed,
            'has_forgetting': has_forgetting,
            'tags': tags
        })
        
        # Print detailed info if verbose
        if args.verbose:
            print(f"[RUN] {path_str}")
            print(f"      Method: {method}, Seed: {seed}")
            print(f"      Tags found: {len(tags)}")
            print(f"      Has '{args.forgetting_tag}': {has_forgetting}")
            
            if not has_forgetting:
                # Find similar tags
                similar = find_similar_tags(tags, 'forget')
                if similar:
                    print(f"      Similar tags: {', '.join(similar)}")
                else:
                    print(f"      No tags containing 'forget'")
            print()
    
    # Summary by method
    print(f"\n{'='*80}")
    print(f"GROUPING SUMMARY")
    print(f"{'='*80}\n")
    
    for method in sorted(method_to_seeds.keys()):
        seeds = sorted([s for s in method_to_seeds[method] if s is not None])
        n_runs = len(method_to_runs[method])
        print(f"Method: {method}")
        print(f"  Seeds: {seeds} (n={len(seeds)})")
        print(f"  Total runs: {n_runs}")
        
        # Check forgetting availability
        runs = method_to_runs[method]
        with_forgetting = sum(1 for r in runs if r['has_forgetting'])
        without_forgetting = len(runs) - with_forgetting
        print(f"  Runs with forgetting tag: {with_forgetting}/{len(runs)}")
        print(f"  Runs without forgetting tag: {without_forgetting}/{len(runs)}")
        print()
    
    # Overall summary
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*80}\n")
    
    total_valid_runs = runs_with_forgetting + runs_without_forgetting
    print(f"Total valid runs: {total_valid_runs}")
    print(f"Runs with '{args.forgetting_tag}': {runs_with_forgetting} ({100*runs_with_forgetting/total_valid_runs:.1f}%)")
    print(f"Runs without '{args.forgetting_tag}': {runs_without_forgetting} ({100*runs_without_forgetting/total_valid_runs:.1f}%)")
    
    # Sample tags from first run
    if method_to_runs:
        first_method = next(iter(method_to_runs.keys()))
        first_run = method_to_runs[first_method][0]
        print(f"\n{'='*80}")
        print(f"SAMPLE TAGS (from first run)")
        print(f"{'='*80}\n")
        print(f"Path: {first_run['path']}")
        print(f"Total tags: {len(first_run['tags'])}\n")
        
        # Group tags by category
        tag_categories = defaultdict(list)
        for tag in sorted(first_run['tags']):
            category = tag.split('/')[0] if '/' in tag else 'other'
            tag_categories[category].append(tag)
        
        for category in sorted(tag_categories.keys()):
            print(f"{category}/ ({len(tag_categories[category])} tags):")
            for tag in tag_categories[category][:5]:  # Show first 5
                print(f"  - {tag}")
            if len(tag_categories[category]) > 5:
                print(f"  ... and {len(tag_categories[category]) - 5} more")
            print()
    
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
