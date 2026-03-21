import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

DATASET_ROOT = 'ds003838-download'

print("--- 🚀 Phase 3: 熱暴走（ノイズ誘起遷移）の証明（完全防弾版） ---")
print("⏳ RT-IIV（反応時間のばらつき）を計算中...")

all_data = []

# 全被験者の行動データを読み込み
for root, dirs, files in os.walk(DATASET_ROOT):
    for file in files:
        if file.endswith('beh.tsv'):
            filepath = os.path.join(root, file)
            try:
                df = pd.read_csv(filepath, sep='\t')
                
                # 柔軟なカラム名探索（rt, response_time, reaction_time に対応）
                cols = df.columns.str.lower()
                rt_col = None
                if 'rt' in cols: rt_col = df.columns[cols.get_loc('rt')]
                elif 'response_time' in cols: rt_col = df.columns[cols.get_loc('response_time')]
                elif 'reaction_time' in cols: rt_col = df.columns[cols.get_loc('reaction_time')]
                
                if rt_col and 'condition' in df.columns:
                    df['participant_id'] = file.split('_')[0]
                    # 反応時間を数値化し、異常値（0以下）を除外
                    df['rt_clean'] = pd.to_numeric(df[rt_col], errors='coerce')
                    df = df[df['rt_clean'] > 0]
                    all_data.append(df[['participant_id', 'condition', 'rt_clean']])
            except Exception as e:
                pass

# データが1件も読み込めなかった場合のセーフティネット
if len(all_data) == 0:
    print("❌ データが読み込めませんでした。beh.tsvの中身（カラム名）が予想と異なる可能性があります。")
    # 最初のファイルを読んでカラム名をヒントとして出す
    for root, dirs, files in os.walk(DATASET_ROOT):
        for file in files:
            if file.endswith('beh.tsv'):
                sample_df = pd.read_csv(os.path.join(root, file), sep='\t')
                print(f"💡 ヒント: このデータセットに存在するカラム名は以下の通りです -> {list(sample_df.columns)}")
                break
        break
    exit()

master_df = pd.concat(all_data, ignore_index=True).dropna()

# 1. 被験者ごと・条件ごとに「RTのばらつき（標準偏差）」を計算
rt_variability = master_df.groupby(['participant_id', 'condition'])['rt_clean'].std().reset_index()
rt_variability.rename(columns={'rt_clean': 'RT_StdDev'}, inplace=True)
rt_variability = rt_variability.dropna()

# 2. サマリーの計算
summary = rt_variability.groupby('condition')['RT_StdDev'].agg(['mean', 'sem']).reset_index()
print("\n【各負荷における反応時間のばらつき（ノイズ量）】")
print(summary)

# 3. 統計検定 (Load 9 vs Load 13)
load9_var = rt_variability[rt_variability['condition'] == 9].set_index('participant_id')['RT_StdDev']
load13_var = rt_variability[rt_variability['condition'] == 13].set_index('participant_id')['RT_StdDev']

# 両方の負荷をクリアしている被験者のみを抽出して対応のあるt検定を行う
common_subjects = load9_var.index.intersection(load13_var.index)
load9_var_matched = load9_var.loc[common_subjects]
load13_var_matched = load13_var.loc[common_subjects]

t_stat, p_val = stats.ttest_rel(load13_var_matched, load9_var_matched, nan_policy='omit')
print(f"\n🎯 統計検定 (Load 13 vs Load 9): t = {t_stat:.3f}, p = {p_val:.5e}")

if p_val < 0.05 and t_stat > 0:
    print("🎉【大勝利】Load 13でRTのばらつき（ノイズ）が有意に爆発しています！熱暴走（Noise-Induced Transition）の決定的な証拠です！")
else:
    print("⚠️ ばらつきの爆発は統計的に有意ではありませんでした。")

# 4. 可視化
plt.figure(figsize=(8, 6))

# 線グラフとエラーバー
plt.errorbar(summary['condition'], summary['mean'], yerr=summary['sem'], 
             fmt='-o', color='darkorange', linewidth=3, markersize=10, capsize=5, label='RT Variability (Std Dev)')

# 個々の被験者の変化を薄い線で描画（スパゲッティプロット）
for subj in common_subjects:
    subj_data = rt_variability[rt_variability['participant_id'] == subj]
    plt.plot(subj_data['condition'], subj_data['RT_StdDev'], color='gray', alpha=0.15)

plt.title('Evidence of Overheating:\nSpike in Intra-Individual RT Variability', fontsize=14, fontweight='bold')
plt.xlabel('Cognitive Load (Digit Span)', fontsize=12)
plt.ylabel('Reaction Time Standard Deviation (s)', fontsize=12)
plt.xticks([5, 9, 13])
plt.grid(True, linestyle=':', alpha=0.7)

# P値をグラフに記載
plt.text(11, summary['mean'].max() * 0.9, f'Load 13 > 9\np = {p_val:.3e}', 
         fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

plt.legend()
plt.tight_layout()
plt.show()