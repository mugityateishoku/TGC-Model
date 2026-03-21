import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_ROOT = 'ds003838-download'
TARGET_SUB = 'sub-013'

print("--- 🚀 究極の解析: Event-Related Pupillometry を実行中 ---\n")

pupil_file = None
events_file = None

for root, dirs, files in os.walk(DATASET_ROOT):
    if TARGET_SUB in root:
        for f in files:
            if f.endswith('pupil.tsv'):
                pupil_file = os.path.join(root, f)
            elif f.endswith('pupil/sub-013_task-memory_events.tsv') or (f.endswith('events.tsv') and 'pupil' in root):
                events_file = os.path.join(root, f)

if not pupil_file or not events_file:
    print("エラー: ファイルが見つかりません。")
    exit()

# 1. データの読み込み
pupil_cols = ['pupil_timestamp', 'diameter_3d', 'confidence', 'blink']
try:
    df_pupil = pd.read_csv(pupil_file, sep='\t', usecols=pupil_cols)
except ValueError:
    pupil_cols = ['pupil_timestamp', 'diameter', 'confidence', 'blink']
    df_pupil = pd.read_csv(pupil_file, sep='\t', usecols=pupil_cols)
    df_pupil.rename(columns={'diameter': 'diameter_3d'}, inplace=True)

df_events = pd.read_csv(events_file, sep='\t')

# 2. 瞳孔データの洗浄
clean_pupil = df_pupil[(df_pupil['confidence'] > 0.8) & (df_pupil['blink'] == 0)].copy()

# 3. トリガーコードを用いたベースライン補正（Delta Pupilの計算）
# 試行開始のトリガーコード
start_triggers = {500105: 5, 500109: 9, 500113: 13}
trial_starts = df_events[df_events['label'].isin(start_triggers.keys())].copy()

results = []

for _, row in trial_starts.iterrows():
    onset = row['timestamp']
    load = start_triggers[int(row['label'])]
    
    # ベースライン: 試行開始の直前 1.0秒間
    base_mask = (clean_pupil['pupil_timestamp'] >= onset - 1.0) & (clean_pupil['pupil_timestamp'] < onset)
    
    # タスク中: 試行開始から、数字をすべて聞き終わるまでの時間
    # (Pavlov et al. の実験デザイン: 1数字あたり約2秒の提示間隔)
    task_duration = load * 2.0 
    task_mask = (clean_pupil['pupil_timestamp'] >= onset) & (clean_pupil['pupil_timestamp'] <= onset + task_duration)
    
    baseline_pupil = clean_pupil.loc[base_mask, 'diameter_3d'].mean()
    task_pupil = clean_pupil.loc[task_mask, 'diameter_3d'].mean()
    
    # ベースライン補正（純粋なタスク誘発性の瞳孔反応）
    if pd.notnull(baseline_pupil) and pd.notnull(task_pupil):
        delta_pupil = task_pupil - baseline_pupil
        results.append({'Load': load, 'Delta_Pupil': delta_pupil})

df_results = pd.DataFrame(results)

# 4. 統計サマリーの計算
summary = df_results.groupby('Load')['Delta_Pupil'].agg(['mean', 'sem']).reset_index()
print("【ベースライン補正後の純粋な瞳孔応答 (Delta Pupil)】")
print(summary)

# 5. 逆U字型（メルトダウン）の可視化
plt.figure(figsize=(9, 6))

# Load 5 -> 9 -> 13 のプロット
plt.errorbar(summary['Load'], summary['mean'], yerr=summary['sem'], 
             fmt='-o', color='crimson', linewidth=4, markersize=12, capsize=6)

# 理論的なアトラクタ崩壊の注釈
plt.title('Catastrophic Meltdown of LC-NE Gain (TGC Model)', fontsize=18, fontweight='bold')
plt.xlabel('Cognitive Load $E$ (Digit Span)', fontsize=14)
plt.ylabel('Task-Evoked Pupil Response ($\Delta$ Diameter)', fontsize=14)

# 領域の色分け
plt.axvspan(4, 7, color='green', alpha=0.1)
plt.text(5, plt.ylim()[1]*0.9, 'Focus\n(Bistable)', ha='center', color='green', fontweight='bold')

plt.axvspan(7, 11, color='orange', alpha=0.1)
plt.text(9, plt.ylim()[1]*0.9, 'Critical\n(Max Gain)', ha='center', color='orange', fontweight='bold')

plt.axvspan(11, 14, color='red', alpha=0.1)
plt.text(13, plt.ylim()[1]*0.9, 'Overload\n(Attractor Collapse)', ha='center', color='red', fontweight='bold')

plt.xticks([5, 9, 13], fontsize=12)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()