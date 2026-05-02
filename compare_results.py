import pandas as pd
import os
import json

file1 = r"results\pog_dense\results_pog_dense_20260420_203840.csv"
file2 = r"results\pog_dense\results_pog_dense_HN_C10_T5_20260426_194737.csv"

# Load dataframes
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Create a sample index since problem_difficulty_meta.csv uses row index
df1['sample_idx'] = df1.index
df2['sample_idx'] = df2.index

# Define columns to keep from file1 (base information)
base_cols = [
    'bundle_id', 'sample_idx', 'true_option_char', 'input_indices', 'candidate_indices', 
    'input_str', 'target_str'
]

# Select and rename columns for A (file1) and B (file2)
df1_sub = df1[base_cols + ['prediction', 'raw_response', 'hit']].rename(
    columns={'prediction': 'prediction_A', 'raw_response': 'raw_response_A', 'hit': 'hit_A'}
)
df2_sub = df2[['bundle_id', 'prediction', 'raw_response', 'hit']].rename(
    columns={'prediction': 'prediction_B', 'raw_response': 'raw_response_B', 'hit': 'hit_B'}
)

# Merge predictions
df_merged = pd.merge(df1_sub, df2_sub, on='bundle_id')

# Load difficulty metadata if exists
# Assuming the difficulty file is in the same dataset folder or a known path
# If file1 is from pog, check pog folder
dataset_folder = os.path.dirname(file1)
# Just in case we are in pog_dense but the meta is in pog
diff_file = os.path.join("results", "pog", "problem_difficulty_meta.csv")
if os.path.exists(diff_file):
    print(f">>> Loading difficulty metadata from {diff_file}")
    df_diff = pd.read_csv(diff_file)
    if 'index' in df_diff.columns:
        df_diff = df_diff.rename(columns={'index': 'sample_idx'})
    df_merged = pd.merge(df_merged, df_diff[['sample_idx', 'difficulty', 'reason']], on='sample_idx', how='left')

# Categorize into groups
def categorize(row):
    if row['hit_A'] == 1 and row['hit_B'] == 1:
        return 'Both_Hit'
    elif row['hit_A'] == 0 and row['hit_B'] == 0:
        return 'Both_Fail'
    elif row['hit_A'] == 1 and row['hit_B'] == 0:
        return 'A_Hit_Only'
    else:
        return 'B_Hit_Only'

df_merged['group'] = df_merged.apply(categorize, axis=1)

# Summary Stats
total = len(df_merged)
both_hit = len(df_merged[df_merged['group'] == 'Both_Hit'])
both_miss = len(df_merged[df_merged['group'] == 'Both_Fail'])
a_hit_only = len(df_merged[df_merged['group'] == 'A_Hit_Only'])
b_hit_only = len(df_merged[df_merged['group'] == 'B_Hit_Only'])

print("="*50)
print("📊 PROMPT COMPARISON & ERROR ANALYSIS PREP")
print("="*50)
print(f"Total Evaluated Bundles : {total}")
print(f"File A Hits             : {df_merged['hit_A'].sum()} / {total} ({(df_merged['hit_A'].sum()/total)*100:.2f}%)")
print(f"File B Hits             : {df_merged['hit_B'].sum()} / {total} ({(df_merged['hit_B'].sum()/total)*100:.2f}%)")
print("-" * 50)
print(f"✅ Both Correct       : {both_hit} ({(both_hit/total)*100:.2f}%)")
print(f"❌ Both Incorrect     : {both_miss} ({(both_miss/total)*100:.2f}%)")
print(f"⬆️ File A Only Correct: {a_hit_only} ({(a_hit_only/total)*100:.2f}%)")
print(f"⭐ File B Only Correct: {b_hit_only} ({(b_hit_only/total)*100:.2f}%)")
print("="*50)

# Save rich dataset for LLM Analysis
out_dir = "analysis"
os.makedirs(out_dir, exist_ok=True)

out_csv = os.path.join(out_dir, "analysis_ready_data.csv")
df_merged.to_csv(out_csv, index=False)
print(f"💾 Comprehensive analysis dataset saved to: {out_csv}")
print("   -> Now ready for 'analyze_error_patterns.py' to process!")
