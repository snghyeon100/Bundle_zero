import pandas as pd
import os
import argparse
import sys

def analyze_difficulty(result_csv, meta_csv):
    """
    Analyze result statistics based on problem difficulty.
    """
    if not os.path.exists(result_csv):
        print(f"Error: Result file not found: {result_csv}")
        return
    if not os.path.exists(meta_csv):
        print(f"Error: Difficulty meta file not found: {meta_csv}")
        return

    # Load data
    df_results = pd.read_csv(result_csv)
    df_meta = pd.read_csv(meta_csv)

    # Ensure index column for merging if it's missing or named differently
    # Most difficulty meta files have an 'index' column matching the row number
    if 'index' not in df_results.columns:
        df_results['index'] = df_results.index
    
    # Merge on index
    df = pd.merge(df_meta, df_results, on='index')

    if 'difficulty' not in df.columns:
        print("Error: 'difficulty' column not found in merged data.")
        return
    
    if 'hit' not in df.columns:
        print("Error: 'hit' column not found in result data.")
        return

    print("=" * 60)
    print(f"Difficulty Analysis Report")
    print(f"Result File: {os.path.basename(result_csv)}")
    print(f"Meta File:   {os.path.basename(meta_csv)}")
    print("=" * 60)

    # 1. Basic Stats per Difficulty
    stats = df.groupby('difficulty')['hit'].agg(['count', 'sum']).reset_index()
    stats.columns = ['Difficulty', 'Total', 'Hits']
    stats['Misses'] = stats['Total'] - stats['Hits']
    stats['Accuracy (%)'] = (stats['Hits'] / stats['Total'] * 100).round(1)

    print("\n[1] Performance by Difficulty Level")
    print("-" * 60)
    print(stats.to_string(index=False))

    # 2. Error Distribution
    total_misses = stats['Misses'].sum()
    if total_misses > 0:
        stats['Error Contribution (%)'] = (stats['Misses'] / total_misses * 100).round(1)
        print("\n[2] Error Distribution (Where do most failures occur?)")
        print("-" * 60)
        error_dist = stats[['Difficulty', 'Misses', 'Error Contribution (%)']].sort_values(by='Error Contribution (%)', ascending=False)
        print(error_dist.to_string(index=False))

    # 3. Summary Statistics
    total_samples = len(df)
    overall_acc = (df['hit'].sum() / total_samples * 100) if total_samples > 0 else 0
    print("\n[3] Summary")
    print("-" * 60)
    print(f"Total Evaluated Samples: {total_samples}")
    print(f"Overall Accuracy:       {overall_acc:.1f}%")
    
    # Calculate weighted difficulty (average difficulty of failed cases)
    failed_df = df[df['hit'] == 0]
    if not failed_df.empty:
        avg_failed_diff = failed_df['difficulty'].mean()
        print(f"Avg Difficulty of Failures: {avg_failed_diff:.2f}")
    
    passed_df = df[df['hit'] == 1]
    if not passed_df.empty:
        avg_passed_diff = passed_df['difficulty'].mean()
        print(f"Avg Difficulty of Successes: {avg_passed_diff:.2f}")

    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Analyze result CSV by problem difficulty.")
    parser.add_argument("--csv", required=True, help="Path to the results CSV file")
    parser.add_argument("--meta", required=True, help="Path to the problem_difficulty_meta.csv file")
    
    args = parser.parse_args()
    
    analyze_difficulty(args.csv, args.meta)

if __name__ == "__main__":
    main()
