import pandas as pd
import os


file1 = r"results\pog_dense\results_pog_dense_20260420_214637.csv"
file2 = r"results\pog_dense\results_pog_dense_20260420_203840.csv"
# Load dataframes
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Assuming bundle_id is consistent. If not, use index since it's the exact same test set
df_merged = pd.merge(df1[['bundle_id', 'hit']], df2[['bundle_id', 'hit']], on='bundle_id', suffixes=('_basic', '_stylist'))

print("="*40)
print("📊 PROMPT COMPARISON ANALYSIS")
print(f"Basic   vs   Stylist")
print("="*40)

# Calculate overlap
both_hit = ((df_merged['hit_basic'] == 1) & (df_merged['hit_stylist'] == 1)).sum()
both_miss = ((df_merged['hit_basic'] == 0) & (df_merged['hit_stylist'] == 0)).sum()
only_basic_hit = ((df_merged['hit_basic'] == 1) & (df_merged['hit_stylist'] == 0)).sum()
only_stylist_hit = ((df_merged['hit_basic'] == 0) & (df_merged['hit_stylist'] == 1)).sum()
total = len(df_merged)

print(f"Total Evaluated Bundles : {total}")
print(f"Basic Prompt Hits       : {df_merged['hit_basic'].sum()} / {total} ({(df_merged['hit_basic'].sum()/total)*100:.2f}%)")
print(f"Stylist Prompt Hits     : {df_merged['hit_stylist'].sum()} / {total} ({(df_merged['hit_stylist'].sum()/total)*100:.2f}%)")
print("-"*40)
print(f"✅ Both Correct       : {both_hit} ({(both_hit/total)*100:.2f}%)")
print(f"❌ Both Incorrect     : {both_miss} ({(both_miss/total)*100:.2f}%)")
print(f"⬆️ Basic Only Correct : {only_basic_hit} ({(only_basic_hit/total)*100:.2f}%)")
print(f"⭐ Stylist Only Correct: {only_stylist_hit} ({(only_stylist_hit/total)*100:.2f}%)")
print("="*40)
