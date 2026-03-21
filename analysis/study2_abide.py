"""
TGC Model — Study 2: ABIDE Resting-State fMRI 1/f Slope
=========================================================
Harada (2026): Thermostatic Gain Control Model, Section 3
NON-CONFIRMATORY BOUNDARY CHECK — NULL RESULT EXPECTED

Dataset: ABIDE I (Autism Brain Imaging Data Exchange)
    http://fcon_1000.projects.nitrc.org/indi/abide/
    Preprocessed: http://preprocessed-connectomes-project.org/abide/

Analysis: Resting-state fMRI 1/f spectral slope across ~111 ROIs
    (Harvard-Oxford atlas), compared ASD vs TD.
Result: 0/111 ROIs survived FDR correction.
Interpretation: Uninformative — resting-state fMRI lacks sensitivity
    to task-evoked gain dynamics. See Section 4 for planned protocol.

Environment variables:
    ABIDE_DATA_DIR   — path to downloaded ABIDE ROI time series
    ABIDE_PHENOTYPE  — path to phenotypic CSV

Usage:
    python analysis/study2_abide.py
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats

print("--- Study 2: ABIDE resting-state fMRI 1/f slope (full-brain ROI scan) ---")

# 1. Configuration
DATA_DIR = os.environ.get('ABIDE_DATA_DIR', './abide_data')
PHENOTYPE_FILE = os.environ.get('ABIDE_PHENOTYPE', 'Phenotypic_V1_0b_preprocessed1.csv')

try:
    pheno_df = pd.read_csv(PHENOTYPE_FILE)
    pheno_dict = dict(zip(pheno_df['SUB_ID'], pheno_df['DX_GROUP']))
except FileNotFoundError:
    print(f"Error: Phenotype file not found.
    exit()

file_pattern = os.path.join(DATA_DIR, '**', '*_rois_ho.1D')
file_list = glob.glob(file_pattern, recursive=True)

if not file_list:
    print("Error: No data files found. Check DATA_DIR.
    exit()

print("Processing: scanning all brain ROIs for 1/f slope...

# 2. 全領域の傾きを計算・保存
roi_data = {} # 領域ごとのデータを格納する辞書

for file in file_list:
    filename = os.path.basename(file)
    match = re.search(r'\d{5,7}', filename)
    if not match: continue
    sub_id = int(match.group())
    if sub_id not in pheno_dict: continue
    
    group_name = 'ASD' if pheno_dict[sub_id] == 1 else 'TD (Control)'
    
    try:
        data = np.loadtxt(file)
    except Exception:
        continue
        
    num_rois = data.shape[1]
    for roi in range(num_rois):
        time_series = data[:, roi]
        
        if np.all(time_series == 0) or np.var(time_series) == 0:
            continue
            
        f, pxx = signal.welch(time_series, fs=0.5, nperseg=len(time_series)//2)
        valid_idx = (f > 0.01) & (f < 0.1)
        f_valid = f[valid_idx]
        pxx_valid = pxx[valid_idx]
        
        if len(f_valid) > 2 and np.all(pxx_valid > 0):
            slope, _, _, _, _ = stats.linregress(np.log10(f_valid), np.log10(pxx_valid))
            if not np.isnan(slope) and not np.isinf(slope):
                if roi not in roi_data:
                    roi_data[roi] = []
                roi_data[roi].append({'SUB_ID': sub_id, 'Group': group_name, 'Slope': slope})

# 3. 各領域で総当たり戦（t検定）
p_values = {}
t_stats = {}

for roi, records in roi_data.items():
    df_roi = pd.DataFrame(records)
    asd_slopes = df_roi[df_roi['Group'] == 'ASD']['Slope']
    td_slopes = df_roi[df_roi['Group'] == 'TD (Control)']['Slope']
    
    # 十分なデータがある領域のみ検定
    if len(asd_slopes) > 5 and len(td_slopes) > 5:
        t_stat, p_val = stats.ttest_ind(asd_slopes, td_slopes, equal_var=False, nan_policy='omit')
        p_values[roi] = p_val
        t_stats[roi] = t_stat

if not p_values:
    print("Error: No comparable ROI data found.
    exit()

# 4. 最も有意差が出たトップ領域（震源地）を特定
best_roi = min(p_values, key=p_values.get)
best_p = p_values[best_roi]
best_t = t_stats[best_roi]

print("\n=========================================")
print(f"Scan complete. Most divergent ROI found:
print("=========================================")
print(f"P値: {best_p:.5e} (t値: {best_t:.3f})")

best_df = pd.DataFrame(roi_data[best_roi])
asd_mean = best_df[best_df['Group'] == 'ASD']['Slope'].mean()
td_mean = best_df[best_df['Group'] == 'TD (Control)']['Slope'].mean()

print(f"ASD群 平均 Slope = {asd_mean:.4f}")
print(f"TD群  平均 Slope = {td_mean:.4f}")

if best_p < 0.05:
    if asd_mean > td_mean:
        print("Significant: ASD group shows flatter slope (higher noise / lower Omega).
    else:
        print("Significant but reversed: ASD slope is steeper.
else:
    print("No ROI reached p < 0.05 (null result, as expected).

# 5. 可視化
plt.figure(figsize=(8, 6))
sns.violinplot(x='Group', y='Slope', data=best_df, palette=['#e74c3c', '#3498db'], inner="quartile")
plt.title(f'E/I Balance Proxy ($\Omega$) in Most Divergent Region (ROI #{best_roi})', fontsize=14, fontweight='bold')
plt.ylabel('1/f Spectral Slope (Flatter = Higher Noise / Lower $\Omega$)', fontsize=12)
plt.xlabel('Diagnostic Group', fontsize=12)

plt.text(0.5, best_df['Slope'].max()*0.95, f'p = {best_p:.4f}', 
         horizontalalignment='center', fontsize=12, fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

plt.tight_layout()
plt.show()