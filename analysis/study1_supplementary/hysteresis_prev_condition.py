import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_ROOT = 'ds003838-download'

print("--- 🚀 ds003838に眠る「ヒステリシスの背骨」を抽出中 ---")

all_data = []

# 全被験者の行動データを読み込み
for root, dirs, files in os.walk(DATASET_ROOT):
    for file in files:
        if file.endswith('beh.tsv'):
            filepath = os.path.join(root, file)
            try:
                df = pd.read_csv(filepath, sep='\t')
                if 'condition' in df.columns and 'partialScore' in df.columns:
                    df['Accuracy'] = df['partialScore'] / df['condition'] * 100
                    # ★超重要: 直前の負荷（n-1試行目）を取得
                    df['prev_condition'] = df['condition'].shift(1)
                    all_data.append(df[['participant_id', 'condition', 'prev_condition', 'Accuracy']])
            except Exception as e:
                pass

master_df = pd.concat(all_data, ignore_index=True).dropna()

# 現在の負荷が「9桁（限界臨界点）」のデータだけを狙い撃ち
target_df = master_df[master_df['condition'] == 9].copy()

# 5桁から来たか、13桁から来たかでグループ分け
def classify_history(prev):
    if prev == 5: return 'From 5\n(Cool Engine)'
    if prev == 13: return 'From 13\n(Overheated)'
    return None

target_df['History'] = target_df['prev_condition'].apply(classify_history)
hysteresis_df = target_df.dropna(subset=['History'])

# 統計サマリーの計算
summary = hysteresis_df.groupby('History')['Accuracy'].agg(['mean', 'sem', 'count'])
print("\n【同じ9桁タスクにおける、直前の状態（履歴）の影響】")
print(summary)

# グラフ化
plt.figure(figsize=(8, 6))
sns.barplot(x='History', y='Accuracy', data=hysteresis_df, palette=['#3498db', '#e74c3c'], errorbar=('ci', 68), capsize=0.1)

plt.title('Macroscopic Hysteresis in Working Memory (TGC Model)\nAccuracy on Load 9 depends on previous state', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy on Load 9 (%)', fontsize=12)
plt.xlabel('Previous Trial Load (The Ghost of the Past)', fontsize=12)
plt.ylim(0, 100)

# 9桁の全体平均線を引く
plt.axhline(hysteresis_df['Accuracy'].mean(), color='black', linestyle='--', alpha=0.5, label='Average Load 9 Accuracy')

plt.legend()
plt.tight_layout()
plt.show()
from scipy import stats

# 青いバー（From 5）と赤いバー（From 13）の生データを抽出
data_from_5 = hysteresis_df[hysteresis_df['History'] == 'From 5\n(Cool Engine)']['Accuracy'].values
data_from_13 = hysteresis_df[hysteresis_df['History'] == 'From 13\n(Overheated)']['Accuracy'].values

# 独立標本のt検定を実行
t_stat, p_val = stats.ttest_ind(data_from_5, data_from_13, equal_var=False)

print(f"\n【統計的有意差の検定結果】")
print(f"t値: {t_stat:.3f}")
print(f"p値: {p_val:.5e}")

if p_val < 0.05:
    print("🎉 結論: p < 0.05 です！ この差は誤差ではありません。完全なる『ヒステリシスの統計的証明』です。")
else:
    print("⚠️ 結論: p >= 0.05 です。残念ながら誤差の範囲内（偶然の産物）である可能性が残ります。")