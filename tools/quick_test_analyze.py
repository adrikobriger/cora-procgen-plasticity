"""
Deprecated helper.

Use tools/analyze_results.py to generate results in results/ and
tools/display_results.py to view the summary table.
"""

def main() -> None:
    print("This script is deprecated.")
    print("Use: python tools/analyze_results.py --runs-dir <runs> --out-dir results")
    print("Then: python tools/display_results.py --csv results/summary_table.csv")


if __name__ == "__main__":
    main()
