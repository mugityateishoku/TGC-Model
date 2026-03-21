"""
TGC Model: Study 1 Supplementary — Macroscopic Hysteresis via Previous-Trial Load
==================================================================================
Tests whether accuracy on Load 9 trials depends on the preceding load condition
(Load 5 vs. Load 13), as predicted by the TGC hysteresis mechanism (Prediction 2).

If the system has memory (hysteresis), Load 9 accuracy should be higher when
preceded by Load 5 (cool state) than Load 13 (overheated state).

Environment variables:
    DS003838_DIR — path to ds003838-download dataset (default: ./ds003838-download)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_ROOT = os.environ.get('DS003838_DIR', './ds003838-download')

print("--- Extracting hysteresis signal from ds003838 ---")

all_data = []

# Load behavioral data for all subjects
for root, dirs, files in os.walk(DATASET_ROOT):
    for file in files:
        if file.endswith('beh.tsv'):
            filepath = os.path.join(root, file)
            try:
                df = pd.read_csv(filepath, sep='\t')
                if 'condition' in df.columns and 'partialScore' in df.columns:
                    df['Accuracy'] = df['partialScore'] / df['condition'] * 100
                    # Previous trial load (n-1), key for hysteresis test
                    df['prev_condition'] = df['condition'].shift(1)
                    all_data.append(df[['participant_id', 'condition', 'prev_condition', 'Accuracy']])
            except Exception as e:
                pass

master_df = pd.concat(all_data, ignore_index=True).dropna()

# Focus on Load 9 trials (critical boundary in TGC model)
target_df = master_df[master_df['condition'] == 9].copy()

# Classify by preceding load: from 5 (cool) vs. from 13 (overheated)
def classify_history(prev):
    if prev == 5: return 'From 5\n(Cool Engine)'
    if prev == 13: return 'From 13\n(Overheated)'
    return None

target_df['History'] = target_df['prev_condition'].apply(classify_history)
hysteresis_df = target_df.dropna(subset=['History'])

# Statistical summary
summary = hysteresis_df.groupby('History')['Accuracy'].agg(['mean', 'sem', 'count'])
print("\nEffect of prior state on Load 9 accuracy (hysteresis test):")
print(summary)

# Plot
plt.figure(figsize=(8, 6))
sns.barplot(x='History', y='Accuracy', data=hysteresis_df, palette=['#3498db', '#e74c3c'], errorbar=('ci', 68), capsize=0.1)

plt.title('Macroscopic Hysteresis in Working Memory (TGC Model)\nAccuracy on Load 9 depends on previous state', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy on Load 9 (%)', fontsize=12)
plt.xlabel('Previous Trial Load (The Ghost of the Past)', fontsize=12)
plt.ylim(0, 100)

# Reference line: overall Load 9 mean
plt.axhline(hysteresis_df['Accuracy'].mean(), color='black', linestyle='--', alpha=0.5, label='Average Load 9 Accuracy')

plt.legend()
plt.tight_layout()
plt.show()
from scipy import stats

# Extract raw accuracy for each history group
data_from_5 = hysteresis_df[hysteresis_df['History'] == 'From 5\n(Cool Engine)']['Accuracy'].values
data_from_13 = hysteresis_df[hysteresis_df['History'] == 'From 13\n(Overheated)']['Accuracy'].values

# Welch's independent-samples t-test
t_stat, p_val = stats.ttest_ind(data_from_5, data_from_13, equal_var=False)

print(f"\nStatistical test results:")
print(f"t = {t_stat:.3f}")
print(f"p = {p_val:.5e}")

if p_val < 0.05:
    print("Result: p < 0.05 — significant hysteresis effect confirmed.")
else:
    print("Result: p >= 0.05 — difference not statistically significant.")