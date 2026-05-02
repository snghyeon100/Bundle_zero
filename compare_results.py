import pandas as pd
import os


file1 = r"results\pog_dense\results_pog_dense_20260420_203840.csv"
file2 = r"results\pog_dense\results_pog_dense_HN_C10_T5_20260426_194737.csv"
# Load dataframes
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Assuming bundle_id is consistent. If not, use index since it's the exact same test set
df_merged = pd.merge(df1[['bundle_id', 'hit']], df2[['bundle_id', 'hit']], on='bundle_id', suffixes=('_file1', '_file2'))

print("="*40)
print("📊 PROMPT COMPARISON ANALYSIS")
print(f"File 1   vs   File 2")
print("="*40)

# Calculate overlap masks
both_hit_mask = (df_merged['hit_file1'] == 1) & (df_merged['hit_file2'] == 1)
both_miss_mask = (df_merged['hit_file1'] == 0) & (df_merged['hit_file2'] == 0)
only_file1_hit_mask = (df_merged['hit_file1'] == 1) & (df_merged['hit_file2'] == 0)
only_file2_hit_mask = (df_merged['hit_file1'] == 0) & (df_merged['hit_file2'] == 1)

both_hit = both_hit_mask.sum()
both_miss = both_miss_mask.sum()
only_file1_hit = only_file1_hit_mask.sum()
only_file2_hit = only_file2_hit_mask.sum()
total = len(df_merged)

print(f"Total Evaluated Bundles : {total}")
print(f"File 1 Hits             : {df_merged['hit_file1'].sum()} / {total} ({(df_merged['hit_file1'].sum()/total)*100:.2f}%)")
print(f"File 2 Hits             : {df_merged['hit_file2'].sum()} / {total} ({(df_merged['hit_file2'].sum()/total)*100:.2f}%)")
print("-"*40)
print(f"✅ Both Correct       : {both_hit} ({(both_hit/total)*100:.2f}%)")
print(f"❌ Both Incorrect     : {both_miss} ({(both_miss/total)*100:.2f}%)")
print(f"⬆️ File 1 Only Correct: {only_file1_hit} ({(only_file1_hit/total)*100:.2f}%)")
print(f"⭐ File 2 Only Correct: {only_file2_hit} ({(only_file2_hit/total)*100:.2f}%)")
print("="*40)

# Save bundle_ids for case study
import json
out_dir = "analysis"
os.makedirs(out_dir, exist_ok=True)

# Convert int64 to native int for JSON serialization
case_study_dict = {
    "both_hit_ids": [int(x) for x in df_merged[both_hit_mask]['bundle_id'].tolist()],
    "both_miss_ids": [int(x) for x in df_merged[both_miss_mask]['bundle_id'].tolist()],
    "only_file1_hit_ids": [int(x) for x in df_merged[only_file1_hit_mask]['bundle_id'].tolist()],
    "only_file2_hit_ids": [int(x) for x in df_merged[only_file2_hit_mask]['bundle_id'].tolist()]
}

out_file = os.path.join(out_dir, "case_study_ids.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(case_study_dict, f, indent=4)
print(f"💾 Case study bundle_ids successfully saved to: {out_file}")
