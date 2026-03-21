"""
TGC Model: Study 1 Section 3.1.4 — EEG 1/f Aperiodic Slope
============================================================
ds003838 task-memory EEG (n=67) の 1/f 非周期的スロープを
Load 5/9/13 条件ごとに計算し、TGCモデルの Ω 予測を検証する。

予測: Load が上がるにつれてスロープが平坦化（Ω低下）
     ただし崩壊後（Load 13）は逆転の可能性あり

実行: python tgc_eeg_study1.py
必要: pip install mne numpy scipy matplotlib pandas
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.signal import welch
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
EEG_DIR     = os.environ.get('DS003838_DIR', './ds003838-download')
FIGURES_DIR = os.environ.get('TGC_FIGURES_DIR', './figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

SFREQ       = 1000.0   # ds003838 の サンプリング周波数 (Hz)
EPOCH_TMIN  = 0.0      # エポック開始 (秒, encode onset から)
EPOCH_TMAX  = 1.8      # エポック終了 (秒, 各桁の提示間隔)

# 1/f フィット帯域 (EEG の非周期成分に適した範囲)
SLOPE_FMIN  = 1.0
SLOPE_FMAX  = 40.0

# Load 条件のイベントコード (ds003838 events.tsv の trial_type)
# 6PPPLLLC 形式: P=presentation, L=load digits, C=conditions
LOAD_MAP = {5: "Load5", 9: "Load9", 13: "Load13"}

# 前頭-中心部チャンネル（LC-NE proxy として最適）
FRONTAL_CH = ["Fz", "FCz", "Cz", "F3", "F4", "FC3", "FC4"]


# ══════════════════════════════════════════════
# 1. イベント読み込みと Load 条件の特定
# ══════════════════════════════════════════════
def get_load_onsets(events_tsv: str):
    """
    events.tsv から Load 5/9/13 の最初の桁 (first) の onset を返す。
    trial_type 例:
      "control 01/13: listen to digit 1 (first) in 13 digit sequence"
      "memorize 01/09: listen to digit 1 (first) in 9 digit sequence"
    → "/05", "/09", "/13" でロードを判定
    → "(first)" を含む行だけをエポック開始点として使用
    """
    df = pd.read_csv(events_tsv, sep="\t")
    onsets = {5: [], 9: [], 13: []}

    for _, row in df.iterrows():
        tt = str(row.get("trial_type", ""))
        onset = float(row["onset"])
        # "(first)" = シーケンス開始点のみ使用
        if "(first)" not in tt:
            continue
        for load in [5, 9, 13]:
            if f"/{load:02d}:" in tt or f"/{load:02d} " in tt:
                onsets[load].append(onset)
                break
    return onsets


# ══════════════════════════════════════════════
# 2. .set ファイルから 1/f スロープを計算
# ══════════════════════════════════════════════
def compute_slope_from_segment(data: np.ndarray, sfreq: float) -> float:
    """
    EEG セグメント (channels × samples) の平均 1/f スロープを計算。
    各チャンネルの PSD を平均してからフィット。
    """
    psds = []
    for ch_data in data:
        if np.std(ch_data) < 1e-10:
            continue
        freqs, psd = welch(ch_data, fs=sfreq,
                           nperseg=min(len(ch_data), int(sfreq * 2)))
        psds.append(psd)

    if not psds:
        return np.nan

    mean_psd = np.mean(psds, axis=0)
    mask = (freqs >= SLOPE_FMIN) & (freqs <= SLOPE_FMAX) & (freqs > 0)
    if mask.sum() < 5:
        return np.nan

    slope, *_ = stats.linregress(
        np.log10(freqs[mask]),
        np.log10(np.clip(mean_psd[mask], 1e-30, None))
    )
    return slope


def analyze_subject(set_file: str):
    """
    1 被験者の task-memory EEG を Load 条件別に解析。
    preload=False で必要な区間だけ読み出し高速化。
    """
    try:
        import mne
        events_tsv = set_file.replace("_eeg.set", "_events.tsv")
        if not os.path.exists(events_tsv):
            return None

        onsets = get_load_onsets(events_tsv)
        if not any(onsets.values()):
            return None

        # preload=False: メタデータのみ読み込み（高速）
        raw = mne.io.read_raw_eeglab(set_file, preload=False, verbose=False)
        sfreq = raw.info["sfreq"]
        n_times = raw.n_times

        # 前頭チャンネルのインデックスを取得
        ch_names_up = [c.upper() for c in raw.ch_names]
        frontal_idx = [i for i, c in enumerate(ch_names_up)
                       if c in [f.upper() for f in FRONTAL_CH]]
        if not frontal_idx:
            frontal_idx = list(range(min(16, len(ch_names_up))))
        picks = np.array(frontal_idx)

        results = {}
        for load, onset_list in onsets.items():
            if not onset_list:
                results[load] = np.nan
                continue

            segments = []
            for onset in onset_list:
                idx_start = int((onset + EPOCH_TMIN) * sfreq)
                idx_end   = int((onset + EPOCH_TMAX) * sfreq)
                if idx_start < 0 or idx_end > n_times:
                    continue
                # 必要な区間だけ読み出し
                seg = raw.get_data(picks=picks, start=idx_start, stop=idx_end)
                segments.append(seg)

            if not segments:
                results[load] = np.nan
                continue

            combined = np.concatenate(segments, axis=1)
            results[load] = compute_slope_from_segment(combined, sfreq)

        return results

    except Exception as e:
        sub = os.path.basename(os.path.dirname(os.path.dirname(set_file)))
        print(f"  ⚠ {sub}: {e}")
        return None


# ══════════════════════════════════════════════
# 3. 全被験者解析
# ══════════════════════════════════════════════
def _worker(args):
    """並列処理用ワーカー"""
    i, total, f = args
    sub = os.path.basename(os.path.dirname(os.path.dirname(f)))
    result = analyze_subject(f)
    return sub, result

def run_all():
    try:
        import mne
        print(f"  MNE version: {mne.__version__}")
    except ImportError:
        print("❌ MNE が見つかりません: pip install mne")
        return None

    set_files = sorted(glob.glob(
        os.path.join(EEG_DIR, "sub-*", "eeg", "*task-memory_eeg.set")))
    n = len(set_files)
    print(f"✅ task-memory EEG: {n} 件")

    import multiprocessing as mp
    n_workers = min(4, mp.cpu_count())
    print(f"  並列処理: {n_workers} workers")

    args = [(i, n, f) for i, f in enumerate(set_files)]
    rows = []

    with mp.Pool(n_workers) as pool:
        for done, (sub, result) in enumerate(pool.imap(_worker, args), 1):
            if result and any(not np.isnan(v) for v in result.values()):
                row = {"subject": sub}
                row.update({f"slope_load{k}": v for k, v in result.items()})
                rows.append(row)
                slopes = [f"L{k}={v:.3f}" for k, v in result.items() if not np.isnan(v)]
                print(f"  [{done:2d}/{n}] {sub}: {', '.join(slopes)}")
            else:
                print(f"  [{done:2d}/{n}] {sub}: skip")

    if not rows:
        print("❌ 有効なデータがありません")
        return None

    df = pd.DataFrame(rows)
    out = os.path.join(FIGURES_DIR, "eeg_slopes_ds003838.csv")
    df.to_csv(out, index=False)
    print(f"\n✅ 保存: {out}  (n={len(df)})")
    return df


# ══════════════════════════════════════════════
# 4. 統計解析
# ══════════════════════════════════════════════
def run_stats(df: pd.DataFrame):
    print("\n" + "=" * 55)
    print("統計解析: Load 条件間の 1/f スロープ比較")
    print("=" * 55)

    loads = [5, 9, 13]
    cols  = [f"slope_load{l}" for l in loads]
    data  = [df[c].dropna().values for c in cols]

    for l, d in zip(loads, data):
        print(f"  Load {l:2d}: mean={np.mean(d):.4f}, "
              f"SD={np.std(d):.4f}, n={len(d)}")

    # 線形傾向検定 (Load 5 → 9 → 13)
    complete = df[cols].dropna()
    if len(complete) >= 5:
        # 被験者内傾向: 各人のスロープ差
        trend = stats.f_oneway(*[complete[c].values for c in cols])
        print(f"\n  One-way ANOVA (Load 5/9/13): F={trend.statistic:.3f}, "
              f"p={trend.pvalue:.4f}")

        # ペアワイズ
        for (l1, c1), (l2, c2) in [
            ((5, cols[0]), (9, cols[1])),
            ((9, cols[1]), (13, cols[2])),
            ((5, cols[0]), (13, cols[2]))]:
            paired = complete[[c1, c2]].dropna()
            t, p = stats.ttest_rel(paired[c1], paired[c2])
            d_val = (paired[c1].mean() - paired[c2].mean()) / paired[c1].std()
            print(f"  Load {l1} vs Load {l2}: "
                  f"t={t:.3f}, p={p:.4f}, d={d_val:.3f} "
                  f"(n={len(paired)})")

    return complete


# ══════════════════════════════════════════════
# 5. 可視化
# ══════════════════════════════════════════════
def plot_results(df: pd.DataFrame):
    loads = [5, 9, 13]
    cols  = [f"slope_load{l}" for l in loads]
    complete = df[cols].dropna()

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "TGC Model Study 1 — EEG 1/f Aperiodic Slope vs. Cognitive Load\n"
        "ds003838 task-memory (n=67) | Frontal-Central Channels | NON-CONFIRMATORY",
        fontsize=12, fontweight="bold")

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.45, wspace=0.35,
                           top=0.90, bottom=0.07)

    colors = ["#2196F3", "#FF9800", "#F44336"]
    means  = [complete[c].mean() for c in cols]
    sems   = [complete[c].sem()  for c in cols]

    # ── Panel A: 平均スロープ ± SEM ──
    ax_A = fig.add_subplot(gs[0, 0])
    ax_A.errorbar(loads, means, yerr=sems, fmt="o-",
                  color="#37474F", lw=2, ms=8, capsize=5)
    ax_A.fill_between(loads,
                      [m - s for m, s in zip(means, sems)],
                      [m + s for m, s in zip(means, sems)],
                      alpha=0.15, color="#37474F")
    ax_A.set_xticks(loads)
    ax_A.set_xlabel("Cognitive Load (digits)", fontsize=10)
    ax_A.set_ylabel("1/f Slope (flatter = lower \u03a9)", fontsize=10)
    ax_A.set_title("Mean 1/f Slope \u00b1 SEM\nby Load Condition", fontsize=11)
    ax_A.annotate(
        "TGC prediction:\nflattening toward fold boundary\n(non-monotonic if collapse at Load 13)",
        xy=(0.05, 0.05), xycoords="axes fraction",
        fontsize=8, style="italic", color="darkblue",
        bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))

    # ── Panel B: 個人別スパゲッティプロット ──
    ax_B = fig.add_subplot(gs[0, 1])
    for _, row in complete.iterrows():
        vals = [row[c] for c in cols]
        ax_B.plot(loads, vals, color="gray", alpha=0.25, lw=0.8)
    ax_B.errorbar(loads, means, yerr=sems, fmt="o-",
                  color="black", lw=2.5, ms=8, capsize=5, zorder=10)
    ax_B.set_xticks(loads)
    ax_B.set_xlabel("Cognitive Load (digits)", fontsize=10)
    ax_B.set_ylabel("1/f Slope", fontsize=10)
    ax_B.set_title(f"Individual Trajectories (n={len(complete)})", fontsize=11)

    # ── Panel C: 分布 (violin) ──
    ax_C = fig.add_subplot(gs[1, 0])
    data_list = [complete[c].values for c in cols]
    parts = ax_C.violinplot(data_list, positions=loads, widths=2,
                             showmeans=True, showmedians=False)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    ax_C.set_xticks(loads)
    ax_C.set_xlabel("Cognitive Load (digits)", fontsize=10)
    ax_C.set_ylabel("1/f Slope", fontsize=10)
    ax_C.set_title("Distribution by Load Condition", fontsize=11)

    # ── Panel D: 結果サマリーテキスト ──
    ax_D = fig.add_subplot(gs[1, 1])
    ax_D.axis("off")

    # 差分方向の確認
    diff_5_9  = means[1] - means[0]
    diff_9_13 = means[2] - means[1]
    diff_5_13 = means[2] - means[0]

    direction_5_9  = "flatter (TGC consistent)" if diff_5_9 > 0 else "steeper"
    direction_9_13 = "flatter (TGC consistent)" if diff_9_13 > 0 else "steeper/reversed"
    pattern = "NON-MONOTONIC" if (diff_5_9 > 0) != (diff_9_13 > 0) else "MONOTONIC"

    summary = (
        f"RESULTS SUMMARY\n"
        f"{'='*35}\n"
        f"n (complete cases) = {len(complete)}\n\n"
        f"Load  5 slope: {means[0]:.4f} (SEM={sems[0]:.4f})\n"
        f"Load  9 slope: {means[1]:.4f} (SEM={sems[1]:.4f})\n"
        f"Load 13 slope: {means[2]:.4f} (SEM={sems[2]:.4f})\n\n"
        f"Load 5\u21929:  {'+' if diff_5_9>0 else ''}{diff_5_9:.4f} ({direction_5_9})\n"
        f"Load 9\u219213: {'+' if diff_9_13>0 else ''}{diff_9_13:.4f} ({direction_9_13})\n"
        f"Load 5\u219213: {'+' if diff_5_13>0 else ''}{diff_5_13:.4f}\n\n"
        f"Pattern: {pattern}\n\n"
        f"TGC prediction:\n"
        f"  Flattening before fold boundary\n"
        f"  + possible steepening after collapse\n"
        f"  = non-monotonic peak at Load 9"
    )
    ax_D.text(0.05, 0.95, summary, transform=ax_D.transAxes,
              fontsize=9, va="top", fontfamily="monospace",
              bbox=dict(boxstyle="round", fc="#F5F5F5", alpha=0.9))

    out = os.path.join(FIGURES_DIR, "tgc_eeg_study1_1f.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"\n✅ Figure saved: {out}")
    plt.show()


# ══════════════════════════════════════════════
# 6. メイン
# ══════════════════════════════════════════════
def main():
    print("=" * 55)
    print("TGC Model Study 1: EEG 1/f Slope Analysis")
    print("ds003838 task-memory EEG  n=67")
    print("=" * 55)

    # MNE がない場合のインストール案内
    try:
        import mne
    except ImportError:
        print("\n❌ MNE が必要です:")
        print("  pip install mne")
        return

    df = run_all()
    if df is None:
        return

    run_stats(df)
    plot_results(df)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()  # Windows 対応
    main()