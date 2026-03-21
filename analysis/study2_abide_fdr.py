import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests

# 先ほど取得したp=0.0257という結果を含む、p値のリスト（シミュレーション）
# ※実際の環境では先ほどの p_values 辞書をそのまま使ってくれ。
# ここではテストとして、110領域中、最小が0.0257であるリストを想定。
p_vals = np.random.uniform(0.1, 1.0, 110)
p_vals[77] = 0.0257

# FDR (Benjamini-Hochberg) 補正
reject, pvals_corrected, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')

print(f"補正前の最小 p値 (ROI #77): {p_vals[77]:.5f}")
print(f"FDR補正後の p値: {pvals_corrected[77]:.5f}")

if pvals_corrected[77] < 0.05:
    print("驚異的！FDR補正後も有意差が残りました！")
else:
    print("やはりFDR補正をかけると有意差は消滅しました（探索的仮説にとどめるべきです）。")

# ROI #77の解剖学的名称（Harvard-Oxford Cortical/Subcortical Atlas）
# ※アトラスのインデックス番号はツールによって微妙にズレるため、
# 典型的なFSL/AFNI系のインデックスを想定したヒント。
print("\n【ROI #77 の解剖学的名称のヒント】")
print("Harvard-Oxfordアトラスでは、70番台後半は通常『後頭葉（Occipital Lobe）』や『小脳（Cerebellum）』、あるいは『皮質下（Subcortical）の線条体・視床』付近に該当します。")
print("正確な部位を特定するには、AFNIやFSLのアトラス表（XML等）で '77' 番のラベルを確認する必要があります。")