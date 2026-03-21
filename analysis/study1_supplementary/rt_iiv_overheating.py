"""
TGC Model: Study 1 Supplementary — RT Intra-Individual Variability (Overheating)
==================================================================================
Tests Prediction 4 of the TGC model: noise-induced transition is evidenced by a
spike in RT intra-individual variability (IIV = within-subject RT standard deviation)
at the highest cognitive load (Load 13), reflecting stochastic jumps between
attractor states near the fold boundary.

Environment variables:
    DS003838_DIR — path to ds003838-download dataset (default: ./ds003838-download)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

DATASET_ROOT = os.environ.get('DS003838_DIR', './ds003838-download')

print("--- RT-IIV Overheating analysis (Prediction 4: noise-induced transition) ---")
print("Computing RT intra-individual variability...")

all_data = []

# Load behavioral data for all subjects
for root, dirs, files in os.walk(DATASET_ROOT):
    for file in files:
        if file.endswith('beh.tsv'):
            filepath = os.path.join(root, file)
            try:
                df = pd.read_csv(filepath, sep='\t')

                # Flexible column name search (rt, response_time, reaction_time)
                cols = df.columns.str.lower()
                rt_col = None
                if 'rt' in cols: rt_col = df.columns[cols.get_loc('rt')]
                elif 'response_time' in cols: rt_col = df.columns[cols.get_loc('response_time')]
                elif 'reaction_time' in cols: rt_col = df.columns[cols.get_loc('reaction_time')]

                if rt_col and 'condition' in df.columns:
                    df['participant_id'] = file.split('_')[0]
                    # Convert to numeric and exclude non-positive values
                    df['rt_clean'] = pd.to_numeric(df[rt_col], errors='coerce')
                    df = df[df['rt_clean'] > 0]
                    all_data.append(df[['participant_id', 'condition', 'rt_clean']])
            except Exception as e:
                pass

# Safety net if no data could be loaded
if len(all_data) == 0:
    print("Error: no data loaded. Column names in beh.tsv may differ from expected.")
    # Print column names from first available file as diagnostic hint
    for root, dirs, files in os.walk(DATASET_ROOT):
        for file in files:
            if file.endswith('beh.tsv'):
                sample_df = pd.read_csv(os.path.join(root, file), sep='\t')
                print(f"Hint: columns found in dataset -> {list(sample_df.columns)}")
                break
        break
    exit()

master_df = pd.concat(all_data, ignore_index=True).dropna()

# 1. Compute RT standard deviation per subject per condition (IIV)
rt_variability = master_df.groupby(['participant_id', 'condition'])['rt_clean'].std().reset_index()
rt_variability.rename(columns={'rt_clean': 'RT_StdDev'}, inplace=True)
rt_variability = rt_variability.dropna()

# 2. Summary statistics
summary = rt_variability.groupby('condition')['RT_StdDev'].agg(['mean', 'sem']).reset_index()
print("\nRT variability (noise proxy) by cognitive load:")
print(summary)

# 3. Statistical test: Load 13 vs Load 9 (paired)
load9_var = rt_variability[rt_variability['condition'] == 9].set_index('participant_id')['RT_StdDev']
load13_var = rt_variability[rt_variability['condition'] == 13].set_index('participant_id')['RT_StdDev']

# Restrict to subjects with data at both loads for paired test
common_subjects = load9_var.index.intersection(load13_var.index)
load9_var_matched = load9_var.loc[common_subjects]
load13_var_matched = load13_var.loc[common_subjects]

t_stat, p_val = stats.ttest_rel(load13_var_matched, load9_var_matched, nan_policy='omit')
print(f"\nStatistical test (Load 13 vs Load 9): t = {t_stat:.3f}, p = {p_val:.5e}")

if p_val < 0.05 and t_stat > 0:
    print("Result: RT-IIV significantly higher at Load 13 — consistent with noise-induced transition.")
else:
    print("Result: IIV spike not statistically significant.")

# 4. Visualization
plt.figure(figsize=(8, 6))

# Mean ± SEM line plot
plt.errorbar(summary['condition'], summary['mean'], yerr=summary['sem'],
             fmt='-o', color='darkorange', linewidth=3, markersize=10, capsize=5, label='RT Variability (Std Dev)')

# Individual subject trajectories (spaghetti)
for subj in common_subjects:
    subj_data = rt_variability[rt_variability['participant_id'] == subj]
    plt.plot(subj_data['condition'], subj_data['RT_StdDev'], color='gray', alpha=0.15)

plt.title('Evidence of Overheating:\nSpike in Intra-Individual RT Variability', fontsize=14, fontweight='bold')
plt.xlabel('Cognitive Load (Digit Span)', fontsize=12)
plt.ylabel('Reaction Time Standard Deviation (s)', fontsize=12)
plt.xticks([5, 9, 13])
plt.grid(True, linestyle=':', alpha=0.7)

# Annotate p-value
plt.text(11, summary['mean'].max() * 0.9, f'Load 13 > 9\np = {p_val:.3e}',
         fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

plt.legend()
plt.tight_layout()
plt.show()