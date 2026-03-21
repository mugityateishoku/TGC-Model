"""
TGC Model — Comprehensive Plausibility Check Pipeline (Windows対応版)
ds003838 (OpenNeuro)

修正点 vs. 前バージョン:
  1. 出力パスをWindows対応に変更（カレントディレクトリに出力）
  2. RT列の自動検出を強化（rt, response_time, ReactionTime など）
  3. RT-IIV NaN問題の修正: 列存在チェックと数値変換を厳密化
  4. Pupil列がbeh.tsvにない場合の適切なスキップ処理
  5. 列名デバッグ出力を追加（最初の1ファイルの列名を表示）
  6. 統計サマリーCSVをカレントディレクトリに保存
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import friedmanchisquare, levene, spearmanr
from pathlib import Path

# ── 出力ディレクトリ（カレントディレクトリ、Windows対応）────
OUTPUT_DIR = Path('.')
OUTPUT_DIR.mkdir(exist_ok=True)

try:
    import diptest
    HAS_DIPTEST = True
except ImportError:
    HAS_DIPTEST = False
    print("diptest未インストール: pip install diptest")

np.random.seed(42)

plt.rcParams.update({
    'font.family':   'DejaVu Sans',
    'font.size':     10,
    'axes.labelsize':11,
    'axes.titlesize':12,
    'axes.linewidth':1.2,
})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 0: データロード
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET_ROOT = 'ds003838-download'

print(f"📂 データセット読み込み中: {DATASET_ROOT}")
all_data   = []
file_count = 0
first_file = True

# RT列の候補（優先順）
RT_COLS     = ['rt', 'response_time', 'ReactionTime', 'RT',
               'reaction_time', 'responseTime']
PUPIL_COLS  = ['pupil_diameter', 'pupil', 'phasic_arousal_e',
               'pupilDiameter', 'pupil_size']
CORR_COLS   = ['correct', 'Correct', 'isCorrect', 'accuracy']

for root, dirs, files in os.walk(DATASET_ROOT):
    for file in sorted(files):
        if not file.endswith('beh.tsv'):
            continue
        filepath = os.path.join(root, file)
        try:
            df_raw = pd.read_csv(filepath, sep='\t')

            # ── 最初のファイルで列名を確認 ────────────────────
            if first_file:
                print(f"\n📋 列名サンプル ({file}):")
                print(f"   {list(df_raw.columns)}\n")
                first_file = False

            # ── participant_id ────────────────────────────────
            if 'participant_id' not in df_raw.columns:
                import re
                match = re.search(r'sub-(\w+)', filepath)
                df_raw['participant_id'] = (
                    f'sub-{match.group(1)}' if match else 'unknown')

            # ── condition（負荷レベル）────────────────────────
            if 'condition' not in df_raw.columns:
                print(f"  スキップ（condition列なし）: {file}")
                continue
            df_raw['condition'] = pd.to_numeric(
                df_raw['condition'], errors='coerce')
            df_raw = df_raw[df_raw['condition'] > 0].copy()

            # ── Accuracy計算 ──────────────────────────────────
            if 'partialScore' in df_raw.columns:
                df_raw['Accuracy'] = np.clip(
                    pd.to_numeric(df_raw['partialScore'], errors='coerce')
                    / df_raw['condition'] * 100,
                    0, 100)
            else:
                corr_col = next(
                    (c for c in CORR_COLS if c in df_raw.columns), None)
                if corr_col:
                    df_raw['Accuracy'] = (
                        pd.to_numeric(df_raw[corr_col], errors='coerce') * 100)
                else:
                    print(f"  スキップ（Accuracy列なし）: {file}")
                    continue

            # ── RT ────────────────────────────────────────────
            rt_col = next(
                (c for c in RT_COLS if c in df_raw.columns), None)
            if rt_col:
                df_raw['rt'] = pd.to_numeric(df_raw[rt_col], errors='coerce')
                # 秒→ms変換（中央値が10未満なら秒単位と判断）
                rt_median = df_raw['rt'].median()
                if pd.notna(rt_median) and rt_median < 10:
                    df_raw['rt'] *= 1000
                    print(f"  RT単位を秒→msに変換: {file}")
            else:
                df_raw['rt'] = np.nan

            # ── Pupil ─────────────────────────────────────────
            pupil_col = next(
                (c for c in PUPIL_COLS if c in df_raw.columns), None)
            df_raw['pupil_diameter'] = (
                pd.to_numeric(df_raw[pupil_col], errors='coerce')
                if pupil_col else np.nan)

            # ── trial_index ───────────────────────────────────
            df_raw['trial_index'] = np.arange(len(df_raw))

            keep = ['participant_id', 'condition', 'Accuracy',
                    'rt', 'pupil_diameter', 'trial_index']
            all_data.append(df_raw[keep].copy())
            file_count += 1

        except Exception as e:
            print(f"  スキップ: {file} ({e})")

if not all_data:
    print("❌ データを抽出できませんでした。")
    sys.exit(1)

master_df = pd.concat(all_data, ignore_index=True)
master_df = master_df.dropna(subset=['condition', 'Accuracy'])
n_subs    = master_df['participant_id'].nunique()
print(f"✅ {file_count}ファイル, {len(master_df)}試行, {n_subs}被験者")

# RT・Pupilのカバレッジを確認
rt_coverage    = master_df['rt'].notna().mean() * 100
pupil_coverage = master_df['pupil_diameter'].notna().mean() * 100
print(f"   RT カバレッジ:    {rt_coverage:.1f}%")
print(f"   Pupil カバレッジ: {pupil_coverage:.1f}%")

if rt_coverage < 5:
    print("   ⚠️  RTデータがほぼ存在しません。beh.tsvの列名を確認してください。")
if pupil_coverage < 5:
    print("   ⚠️  Pupilデータはbeh.tsvにない可能性があります（生理データは別ファイル）。")

DATA_NOTE = "ds003838 (OpenNeuro)"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1: 集計
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
loads_all = sorted(master_df['condition'].unique())
print(f"\n負荷条件: {[int(L) for L in loads_all]}")

sub_load_acc = (
    master_df
    .groupby(['participant_id', 'condition'])['Accuracy']
    .mean().reset_index()
    .rename(columns={'Accuracy': 'acc_mean'}))

sub_load_iiv = (
    master_df
    .groupby(['participant_id', 'condition'])['rt']
    .std().reset_index()
    .rename(columns={'rt': 'rt_iiv'}))

sub_load_pupil = (
    master_df
    .groupby(['participant_id', 'condition'])['pupil_diameter']
    .mean().reset_index()
    .rename(columns={'pupil_diameter': 'pupil_mean'}))

merged = (sub_load_acc
          .merge(sub_load_iiv,   on=['participant_id','condition'], how='left')
          .merge(sub_load_pupil, on=['participant_id','condition'], how='left'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2: 統計検定
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
stats_results = {}
print("\n" + "="*60)
print("統計検定結果")
print("="*60)

# (a) Friedman検定
print("\n(a) Friedman検定 — 負荷間Accuracy差")
pivot_acc = sub_load_acc.pivot(
    index='participant_id', columns='condition',
    values='acc_mean').dropna()
if pivot_acc.shape[1] >= 2:
    stat_f, p_f = friedmanchisquare(
        *[pivot_acc[c].values for c in pivot_acc.columns])
    print(f"  χ²({pivot_acc.shape[1]-1}) = {stat_f:.3f},  p = {p_f:.4f}")
    stats_results['friedman_acc'] = {'stat': stat_f, 'p': p_f}

# (b) Dip検定
print("\n(b) Hartigan Dip検定 — 二峰性（Bistability, Prediction 1）")
for load in loads_all:
    acc_vals = master_df[master_df['condition']==load]['Accuracy'].dropna().values
    if HAS_DIPTEST and len(acc_vals) >= 10:
        d, p_d = diptest.diptest(acc_vals)
        sig = "✅ p<.05" if p_d < 0.05 else "n.s."
        print(f"  Load {int(load):2d}: dip={d:.4f}, p={p_d:.4f}  {sig}")
        stats_results[f'dip_load{int(load)}'] = {'dip': d, 'p': p_d}
    else:
        kurt = stats.kurtosis(acc_vals)
        print(f"  Load {int(load):2d}: 尖度={kurt:.3f}")

# (c) Levene検定
print("\n(c) Levene検定 — 分散の負荷間差")
groups_acc = [
    master_df[master_df['condition']==L]['Accuracy'].dropna().values
    for L in loads_all
    if len(master_df[master_df['condition']==L]) >= 5]
if len(groups_acc) >= 2:
    stat_l, p_l = levene(*groups_acc)
    print(f"  F = {stat_l:.3f},  p = {p_l:.4f}  "
          f"{'✅ 分散に有意差あり' if p_l < 0.05 else 'n.s.'}")
    stats_results['levene_acc'] = {'stat': stat_l, 'p': p_l}

# (d) Spearman: 負荷 × RT-IIV
print("\n(d) Spearman相関 — 負荷 × RT-IIV")
iiv_by_load = merged.groupby('condition')['rt_iiv'].mean().dropna()
if len(iiv_by_load) >= 3:
    rho_d, p_d2 = spearmanr(iiv_by_load.index, iiv_by_load.values)
    sig = "✅ 有意な単調増加" if (p_d2 < 0.05 and rho_d > 0) else "n.s./逆方向"
    print(f"  ρ = {rho_d:.3f},  p = {p_d2:.4f}  {sig}")
    stats_results['spearman_iiv'] = {'rho': rho_d, 'p': p_d2}
else:
    print("  ⚠️  RTデータが不足（RT列が見つからない可能性）")
    print(f"  RT-IIVの有効データ数: {merged['rt_iiv'].notna().sum()}")

# (e) Pupil × RT-IIV
print("\n(e) Spearman相関 — Pupil × RT-IIV（Overheating）")
pupil_iiv = merged.dropna(subset=['pupil_mean', 'rt_iiv'])
if len(pupil_iiv) >= 10:
    rho_p, p_p = spearmanr(pupil_iiv['pupil_mean'], pupil_iiv['rt_iiv'])
    sig = "✅ 同時増大を確認" if (p_p < 0.05 and rho_p > 0) else "n.s./逆方向"
    print(f"  ρ = {rho_p:.3f},  p = {p_p:.4f}  {sig}")
    stats_results['overheating_corr'] = {'rho': rho_p, 'p': p_p}
else:
    print(f"  ⚠️  有効データが{len(pupil_iiv)}行（Pupilデータはbeh.tsvに含まれない可能性）")
    print("      → physio/*.tsv または eye tracking ファイルを確認してください")

# (f) 順序効果
print("\n(f) 順序効果検定 — Hysteresis探索（Prediction 2、参考値）")
print("    ⚠️  ds003838はcounterbalanced設計ではないため参考値のみ")
if 9 in loads_all:
    df9       = master_df[master_df['condition']==9].copy()
    med_idx   = df9['trial_index'].median()
    early     = df9[df9['trial_index'] <= med_idx]['Accuracy']
    late      = df9[df9['trial_index'] >  med_idx]['Accuracy']
    if len(early) >= 5 and len(late) >= 5:
        stat_h, p_h = stats.mannwhitneyu(early, late, alternative='greater')
        cohen_d = (early.mean()-late.mean()) / np.sqrt(
            (early.std()**2+late.std()**2)/2)
        print(f"  早期 vs. 後期 (load=9):  "
              f"U={stat_h:.0f}, p={p_h:.4f}, d={cohen_d:.3f}")
        stats_results['hysteresis_proxy'] = {
            'U': stat_h, 'p': p_h, 'cohen_d': cohen_d}

# 統計サマリー保存（カレントディレクトリ）
stats_df = pd.DataFrame([{'検定': k, **v} for k, v in stats_results.items()])
csv_path = OUTPUT_DIR / 'TGC_stats_summary.csv'
stats_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n✅ 統計サマリー保存: {csv_path}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3: Figure 8 — ポテンシャル地形
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
target_loads = [L for L in [5, 9, 13] if L in loads_all]
fig_colors   = {5:'#009E73', 9:'#E69F00', 13:'#D55E00'}
fig_titles   = {
    5:  '(a)  Load 5 — Baseline\n(stable high-gain attractor)',
    9:  '(b)  Load 9 — Critical\n(bistability / attractor splitting)',
    13: '(c)  Load 13 — Overload\n(meltdown / monostable collapse)',
}

fig8, axes8 = plt.subplots(1, len(target_loads),
                            figsize=(5.5*len(target_loads), 5.0),
                            sharey=False)
if len(target_loads) == 1:
    axes8 = [axes8]

fig8.suptitle(
    'Figure 8.  Behavioral Attractor Collapse under Cognitive Load (TGC Model)\n'
    r'Pseudo-potential $V(x) \propto -\ln \hat{P}(x)$  '
    '[heuristic — see text for caveats]',
    fontsize=12, y=1.02)

from scipy.signal import argrelmin

for ax, load in zip(axes8, target_loads):
    subset = master_df[master_df['condition']==load]['Accuracy'].dropna().values
    np.random.seed(42 + int(load))
    subset_j = np.clip(subset + np.random.normal(0, 1.2, len(subset)), 0, 100)

    kde    = stats.gaussian_kde(subset_j, bw_method=0.18)
    x_rng  = np.linspace(0, 100, 300)
    p_x    = np.maximum(kde(x_rng), 1e-6)
    v_x    = -np.log(p_x)
    v_x   -= v_x.min()

    color  = fig_colors.get(int(load), '#999999')
    ax.plot(x_rng, v_x, color=color, lw=2.8, zorder=3)
    ax.fill_between(x_rng, v_x, v_x.max(),
                    color=color, alpha=0.08, zorder=2)

    for m in argrelmin(v_x, order=15)[0]:
        ax.scatter(x_rng[m], v_x[m], color=color, s=90, zorder=5,
                   edgecolors='white', linewidths=1.2)

    dip_str = ""
    if HAS_DIPTEST and len(subset) >= 10:
        d_v, p_dv = diptest.diptest(subset)
        dip_str = f"\nDip={d_v:.3f}, p={p_dv:.3f}"

    ax.text(0.03, 0.97,
            f"n={len(subset)}\nM={np.mean(subset):.1f}%"
            f"\nVar={np.var(subset):.1f}{dip_str}",
            transform=ax.transAxes, va='top', ha='left',
            fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.3',
                      fc='white', ec='grey', alpha=0.85))

    ax.set_title(fig_titles.get(int(load), f'Load {int(load)}'),
                 fontsize=11.5, pad=8)
    ax.set_xlabel(r'Accuracy / Memory State $x$ (%)', fontsize=11)
    if ax is axes8[0]:
        ax.set_ylabel(r'$V(x) \propto -\ln\hat{P}(x)$  (a.u.)', fontsize=11)
    ax.set_xlim(-2, 102)
    ax.grid(axis='y', alpha=0.25)
    ax.set_ylim(bottom=0)

fig8.text(0.5, -0.04,
          f'Data: {DATA_NOTE}  |  '
          'Caution: binary accuracy violates continuity assumption  |  '
          'Stationarity not verified',
          ha='center', fontsize=8, color='grey', style='italic')

plt.tight_layout()
fig8_path = OUTPUT_DIR / 'Figure8_potential_landscape.pdf'
fig8.savefig(fig8_path, dpi=300, bbox_inches='tight')
fig8.savefig(OUTPUT_DIR / 'Figure8_potential_landscape.tiff',
             dpi=300, bbox_inches='tight')
print(f"✅ Figure8 保存: {fig8_path}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4: Figure 5 — Overheatingシグネチャ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COLLAPSE_LOAD = max(loads_all)
C_PUPIL    = '#0072B2'
C_RT       = '#E69F00'
C_COLLAPSE = '#D55E00'
C_ANNOT    = '#CC79A7'

fig5, axes5 = plt.subplots(1, 2, figsize=(11, 4.8),
                            constrained_layout=True)

for ax, (col, ylabel, title, color, marker) in zip(axes5, [
    ('pupil_mean',
     'Mean pupil diameter (a.u.)\n[LC-NE proxy / primary beta proxy]',
     '(A)  Phasic Pupil Dilation',
     C_PUPIL, 'o'),
    ('rt_iiv',
     'RT intra-individual variability\n[SD, ms / secondary beta proxy]',
     '(B)  RT-IIV',
     C_RT, 's'),
]):
    grp    = merged.groupby('condition')[col]
    mean_s = grp.mean()
    sem_s  = grp.sem()
    valid_s = mean_s.dropna()

    if len(valid_s) == 0:
        ax.text(0.5, 0.5,
                f'データなし\n({col}がbeh.tsvに含まれない)\n'
                '生理データファイルを確認してください',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='grey',
                bbox=dict(boxstyle='round', fc='lightyellow', ec='grey'))
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Cognitive load', fontsize=11)
        continue

    n_s = grp.count().max()
    ax.errorbar(valid_s.index, valid_s.values,
                yerr=sem_s[valid_s.index].values,
                marker=marker, ms=7, lw=1.8, capsize=4,
                color=color, ecolor=color,
                markeredgecolor='white', markeredgewidth=0.8,
                zorder=4, label=f'Mean +- SEM  (n={n_s})')

    if COLLAPSE_LOAD in valid_s.index:
        ax.axvline(COLLAPSE_LOAD, color=C_COLLAPSE,
                   lw=1.6, ls='--', zorder=3,
                   label=f'TGC collapse boundary (Load {int(COLLAPSE_LOAD)})')
        y_max  = (mean_s + sem_s).max()
        y_min  = (mean_s - sem_s).min()
        y_span = y_max - y_min or 1
        ax.annotate('Overheating\nprediction:\npeak here',
                    xy=(COLLAPSE_LOAD, valid_s[COLLAPSE_LOAD]),
                    xytext=(COLLAPSE_LOAD - 2.0, y_max + y_span*0.28),
                    arrowprops=dict(arrowstyle='->', color=C_ANNOT,
                                    lw=1.4, mutation_scale=12),
                    fontsize=8.5, color=C_ANNOT, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3',
                              fc='white', ec=C_ANNOT, alpha=0.88))

    valid_pair = merged[['condition', col]].dropna()
    if len(valid_pair) >= 5:
        rho_v, p_v = spearmanr(valid_pair['condition'], valid_pair[col])
        ax.text(0.03, 0.04,
                f'rho = {rho_v:.2f},  p = {p_v:.3f}',
                transform=ax.transAxes, fontsize=9, color=color,
                bbox=dict(boxstyle='round,pad=0.3',
                          fc='white', ec=color, alpha=0.85))

    ax.set_xlabel('Cognitive load (digit span)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xticks([int(L) for L in sorted(merged['condition'].unique())])
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis='y', alpha=0.25)

fig5.text(0.5, -0.04,
          f'Data: {DATA_NOTE}  |  Plausibility check only (not confirmatory)',
          ha='center', fontsize=8, color='grey', style='italic')
fig5.suptitle(
    'Figure 5.  Overheating Signature: LC-NE Arousal and Beta Instability\n'
    '(TGC Prediction: synchronous peak of sigma proxy and RT-IIV at critical load)',
    fontsize=11.5)

fig5_path = OUTPUT_DIR / 'Figure5_overheating.pdf'
fig5.savefig(fig5_path, dpi=300, bbox_inches='tight')
fig5.savefig(OUTPUT_DIR / 'Figure5_overheating.tiff',
             dpi=300, bbox_inches='tight')
print(f"✅ Figure5 保存: {fig5_path}")

print("\n" + "="*60)
print("全解析完了")
print("="*60)
print(f"出力先: {OUTPUT_DIR.resolve()}")
print("  TGC_stats_summary.csv")
print("  Figure5_overheating.pdf / .tiff")
print("  Figure8_potential_landscape.pdf / .tiff")