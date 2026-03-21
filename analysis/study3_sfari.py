"""
TGC Model — Study 3: SFARI ASSR E/I Balance in ASD (ds006780)
===============================================================
Harada (2026): Thermostatic Gain Control Model, Section 3
NON-CONFIRMATORY BOUNDARY CHECK — NULL RESULT

Dataset: OpenNeuro ds006780 (SFARI auditory steady-state response)
    https://openneuro.org/datasets/ds006780

Analysis: 40 Hz ASSR amplitude as Omega proxy, ASD vs TD comparison.
    Theoretical basis: 40 Hz ASSR reflects gamma-band E/I balance.
    TGC prediction: reduced ASSR in ASD (lower Omega).
Result: No significant group difference (d = 0.12, p = 0.41).
Interpretation: Uninformative — ASSR indexes auditory cortex, not
    prefrontal E/I balance indexed by Omega.

Environment variables:
    SFARI_DATA_DIR — path to downloaded ds006780 dataset

Usage:
    python analysis/study3_sfari.py

Requires: pip install mne numpy scipy matplotlib pandas
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.signal import welch
import multiprocessing as mp
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
SFARI_DIR   = os.environ.get('SFARI_DATA_DIR', './ds006780')
FIGURES_DIR = os.environ.get('TGC_FIGURES_DIR', './figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

SFREQ_EXPECTED = 512.0   # BioSemi ActiveTwo
ASSR_FREQS     = [27.0, 40.0]  # Hz - 定常状態応答周波数
ASSR_BW        = 1.0     # ± Hz のバンド幅でパワーを積算
EPOCH_TMIN     = 0.0     # onset からの開始 (秒)
EPOCH_TMAX     = 0.5     # onset からの終了 (秒, 各刺激 500ms)

# 前頭-中心部チャンネル (ASSR は前頭〜中心部で最大)
FRONTAL_CH = ["Fz", "FCz", "Cz", "F3", "F4", "FC1", "FC2", "FC5", "FC6"]

# イベントコード
EVENT_CODES = {
    11: "40Hz_Standard",
    12: "40Hz_Oddball",
    21: "27Hz_Standard",
    22: "27Hz_Oddball",
}


# ══════════════════════════════════════════════
# 1. Phenotypic 読み込み
# ══════════════════════════════════════════════
def load_phenotypic():
    tsv = os.path.join(SFARI_DIR, "participants.tsv")
    df = pd.read_csv(tsv, sep="\t")
    df["participant_id"] = df["participant_id"].str.strip()
    print(f"✅ participants: {len(df)} 件")
    print(f"   group 分布: {df['group'].value_counts().to_dict()}")
    return df[["participant_id","group","age","sex","fsiq","ados_css"]].copy()


# ══════════════════════════════════════════════
# 2. ASSR パワー計算
# ══════════════════════════════════════════════
def compute_assr_power(raw, picks, events_tsv: str, target_freq: float):
    """
    ASSR パワーを計算。
    全エポックを連結してから一括 FFT することで周波数分解能を最大化。
    512Hz × 500ms = 256サンプル/エポック → 2Hz/bin (粗すぎ)
    170エポック連結 → 43520サンプル → 0.012Hz/bin (十分)
    """
    df_ev = pd.read_csv(events_tsv, sep="\t")
    sfreq = raw.info["sfreq"]
    n_times = raw.n_times

    # 対象周波数の Standard イベントを取得 (value列優先)
    target_label = f"{int(target_freq)}_Hz_Standard"
    if "value" in df_ev.columns and "trial_type" in df_ev.columns:
        mask = df_ev["trial_type"].str.contains(
            f"{int(target_freq)}_Hz_Standard", na=False)
        sub_ev = df_ev[mask]
    else:
        return np.nan

    if len(sub_ev) == 0:
        return np.nan

    # エポック抽出 (500ms × 全試行)
    n_epoch = int(0.5 * sfreq)  # 256 samples @ 512Hz
    segments = []
    for _, row in sub_ev.iterrows():
        start = int(float(row["onset"]) * sfreq)
        stop  = start + n_epoch
        if start < 0 or stop > n_times:
            continue
        seg = raw.get_data(picks=picks, start=start, stop=stop)  # ch × time
        segments.append(seg)

    if len(segments) < 10:
        return np.nan

    # ── 全エポックを連結して FFT (高周波数分解能) ──
    concat = np.concatenate(segments, axis=1)  # ch × (n_epochs × n_epoch)
    n_total = concat.shape[1]

    # 各チャンネルの FFT 振幅
    fft_amp = np.abs(np.fft.rfft(concat, axis=1)) / n_total
    freqs   = np.fft.rfftfreq(n_total, 1.0 / sfreq)

    # ターゲット周波数ビン ± 0.5Hz の最大値をシグナルとする
    bw = 0.5
    mask = np.abs(freqs - target_freq) <= bw
    if mask.sum() == 0:
        mask = np.array([np.argmin(np.abs(freqs - target_freq))])

    # ノイズフロア: ターゲット ± 2〜5Hz の平均
    noise_mask = (np.abs(freqs - target_freq) >= 2) &                  (np.abs(freqs - target_freq) <= 5)

    # チャンネル平均振幅 (単位: V, BioSemi BDF)
    signal = fft_amp[:, mask].max(axis=1).mean()

    # μV に変換して返す (1V = 1e6 μV)
    return float(signal * 1e6)


def analyze_subject(set_file: str):
    """1 被験者の ASSR 解析"""
    try:
        import mne
        events_tsv = set_file.replace("_eeg.bdf", "_events.tsv")
        if not os.path.exists(events_tsv):
            return None

        # BDF 読み込み (preload=False で高速化)
        raw = mne.io.read_raw_bdf(set_file, preload=False, verbose=False)
        sfreq = raw.info["sfreq"]

        # チャンネル選択
        ch_names_up = [c.upper() for c in raw.ch_names]
        frontal_idx = [i for i, c in enumerate(ch_names_up)
                       if c in [f.upper() for f in FRONTAL_CH]]
        if not frontal_idx:
            frontal_idx = list(range(min(16, len(ch_names_up))))
        picks = np.array(frontal_idx)

        result = {}
        for freq in ASSR_FREQS:
            result[f"assr_{int(freq)}hz"] = compute_assr_power(
                raw, picks, events_tsv, freq)

        return result

    except Exception as e:
        return {"error": str(e)}


def _worker(args):
    sub_id, bdf_file = args
    result = analyze_subject(bdf_file)
    return sub_id, result


# ══════════════════════════════════════════════
# 3. 全被験者解析
# ══════════════════════════════════════════════
def run_all(pheno: pd.DataFrame):
    bdf_files = sorted(glob.glob(
        os.path.join(SFARI_DIR, "sub-*", "eeg", "*ASSR*.bdf")))
    print(f"✅ ASSR .bdf: {len(bdf_files)} 件")

    # 被験者IDとファイルのペア
    def get_sub_id(f):
        return os.path.basename(os.path.dirname(os.path.dirname(f)))

    args = [(get_sub_id(f), f) for f in bdf_files]

    n_workers = min(4, mp.cpu_count())
    print(f"  並列処理: {n_workers} workers")

    rows = []
    with mp.Pool(n_workers) as pool:
        for done, (sub_id, result) in enumerate(pool.imap(_worker, args), 1):
            if result and "error" not in result:
                row = {"participant_id": sub_id}
                row.update(result)
                rows.append(row)
                vals = ", ".join(f"{k}={v:.2f}" for k, v in result.items()
                                 if not (isinstance(v, float) and np.isnan(v)))
                print(f"  [{done:3d}/{len(args)}] {sub_id}: {vals}")
            elif result and "error" in result:
                print(f"  [{done:3d}/{len(args)}] {sub_id}: ⚠ {result['error'][:60]}")
            else:
                print(f"  [{done:3d}/{len(args)}] {sub_id}: skip")

    if not rows:
        print("❌ 有効データなし")
        return None

    df_raw = pd.DataFrame(rows)
    assr_cols = [c for c in df_raw.columns if c.startswith("assr_")]

    # ── Step1: run レベルで外れ値を NaN に (3IQR) ──
    for col in assr_cols:
        valid = df_raw[col].dropna()
        q1, q3 = valid.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 3*iqr, q3 + 3*iqr
        mask = (df_raw[col] < lo) | (df_raw[col] > hi)
        n_out = mask.sum()
        if n_out > 0:
            bad = df_raw.loc[mask, ["participant_id", col]].values
            print(f"  run外れ値除去 {col}: {n_out}件 (>{hi:.3f})")
            for sub_id, val in bad:
                print(f"    {sub_id}: {val:.3f}")
            df_raw.loc[mask, col] = np.nan

    # ── Step2: 複数 run を被験者ごとに平均 ──
    df = df_raw.groupby("participant_id")[assr_cols].mean().reset_index()
    print(f"\n  run 平均後: {len(df_raw)} run → {len(df)} 被験者")

    # ── Step3: 被験者レベルでも念のため 3IQR フィルタ ──
    for col in assr_cols:
        valid = df[col].dropna()
        if len(valid) < 4:
            continue
        q1, q3 = valid.quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (df[col] < q1 - 3*iqr) | (df[col] > q3 + 3*iqr)
        if mask.sum() > 0:
            print(f"  被験者外れ値除去 {col}: {mask.sum()}件")
            df.loc[mask, col] = np.nan

    df = df.merge(pheno, on="participant_id", how="inner")
    out = os.path.join(FIGURES_DIR, "sfari_assr_results.csv")
    df.to_csv(out, index=False)
    print(f"✅ 保存: {out}  (n={len(df)})")
    return df


# ══════════════════════════════════════════════
# 4. 統計解析
# ══════════════════════════════════════════════
def run_stats(df: pd.DataFrame):
    print("\n" + "=" * 55)
    print("統計解析: ASD vs TD ASSR 振幅比較")
    print("=" * 55)

    # ASD と TD のみ (SIB は除外)
    df_asd = df[df["group"] == "ASD"]
    df_td  = df[df["group"] == "TD"]
    print(f"  ASD n={len(df_asd)}, TD n={len(df_td)}")

    results = {}
    for freq in ASSR_FREQS:
        col = f"assr_{int(freq)}hz"
        if col not in df.columns:
            continue
        v_asd = df_asd[col].dropna().values
        v_td  = df_td[col].dropna().values
        if len(v_asd) < 3 or len(v_td) < 3:
            continue

        t, p = stats.ttest_ind(v_asd, v_td, equal_var=False)
        pooled = np.sqrt((v_asd.std()**2 + v_td.std()**2) / 2)
        d = (v_asd.mean() - v_td.mean()) / pooled if pooled > 0 else np.nan
        direction = "ASD < TD (TGC consistent)" if v_asd.mean() < v_td.mean() else "ASD > TD"

        print(f"\n  {int(freq)}Hz ASSR:")
        print(f"    ASD: {v_asd.mean():.4f} ± {v_asd.std():.4f}")
        print(f"    TD:  {v_td.mean():.4f} ± {v_td.std():.4f}")
        print(f"    t={t:.3f}, p={p:.4f}, d={d:.3f}")
        print(f"    Direction: {direction}")
        results[freq] = {"t": t, "p": p, "d": d,
                         "mean_ASD": v_asd.mean(), "mean_TD": v_td.mean()}
    return results


# ══════════════════════════════════════════════
# 5. 可視化
# ══════════════════════════════════════════════
def plot_results(df: pd.DataFrame, stats_results: dict):
    df_plot = df[df["group"].isin(["ASD", "TD"])].copy()
    colors  = {"ASD": "#F44336", "TD": "#2196F3"}

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "TGC Model: Study 3 — ASSR E/I Balance in ASD vs TD (SFARI ds006780)\n"
        "40Hz & 27Hz ASSR Amplitude as \u03a9 Proxy | NON-CONFIRMATORY",
        fontsize=12, fontweight="bold")

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.5, wspace=0.4,
                           top=0.90, bottom=0.07)

    for col_idx, freq in enumerate(ASSR_FREQS):
        col = f"assr_{int(freq)}hz"
        if col not in df_plot.columns:
            continue

        # Panel 上段: violin + strip
        ax = fig.add_subplot(gs[0, col_idx])
        groups = ["TD", "ASD"]
        data   = [df_plot[df_plot["group"]==g][col].dropna().values for g in groups]
        vp = ax.violinplot(data, positions=[0,1], widths=0.6,
                           showmeans=True, showmedians=False)
        for patch, g in zip(vp["bodies"], groups):
            patch.set_facecolor(colors[g])
            patch.set_alpha(0.6)
        # 個人データ点
        for i, (g, d) in enumerate(zip(groups, data)):
            jitter = np.random.normal(0, 0.05, len(d))
            ax.scatter(np.full(len(d), i) + jitter, d,
                      color=colors[g], alpha=0.4, s=15, zorder=3)
        ax.set_xticks([0,1])
        ax.set_xticklabels(groups)
        ax.set_ylabel("ASSR Amplitude (FFT)", fontsize=9)
        ax.set_title(f"{int(freq)}Hz ASSR\nASD vs TD", fontsize=11)

        # 統計注記
        if freq in stats_results:
            r = stats_results[freq]
            sig = "**" if r["p"] < 0.01 else ("*" if r["p"] < 0.05 else "n.s.")
            direction_str = "↓ (TGC✓)" if r["mean_ASD"] < r["mean_TD"] else "↑"
            ax.set_xlabel(
                f"p={r['p']:.4f} {sig}  d={r['d']:.3f}\nASD {direction_str}",
                fontsize=8.5)
        ax.annotate("TGC: ASD amplitude lower\n(E/I imbalance = lower \u03a9)",
                    xy=(0.03, 0.97), xycoords="axes fraction",
                    fontsize=7.5, va="top", style="italic", color="darkblue")

    # Panel 下段左: 40Hz vs 27Hz 相関
    ax_corr = fig.add_subplot(gs[1, 0])
    col40 = "assr_40hz"
    col27 = "assr_27hz"
    if col40 in df_plot.columns and col27 in df_plot.columns:
        for g in ["TD", "ASD"]:
            sub = df_plot[df_plot["group"]==g][[col40, col27]].dropna()
            ax_corr.scatter(sub[col27], sub[col40],
                           color=colors[g], alpha=0.5, s=20, label=g)
        ax_corr.set_xlabel("27Hz ASSR", fontsize=9)
        ax_corr.set_ylabel("40Hz ASSR", fontsize=9)
        ax_corr.set_title("40Hz vs 27Hz ASSR\nCorrelation", fontsize=11)
        ax_corr.legend(fontsize=8)

    # Panel 下段中: 年齢との関係
    ax_age = fig.add_subplot(gs[1, 1])
    if "age" in df_plot.columns and col40 in df_plot.columns:
        for g in ["TD", "ASD"]:
            sub = df_plot[df_plot["group"]==g][["age", col40]].dropna()
            ax_age.scatter(sub["age"], sub[col40],
                          color=colors[g], alpha=0.5, s=20, label=g)
        ax_age.set_xlabel("Age (years)", fontsize=9)
        ax_age.set_ylabel("40Hz ASSR", fontsize=9)
        ax_age.set_title("40Hz ASSR vs Age", fontsize=11)
        ax_age.legend(fontsize=8)

    # Panel 下段右: サマリー
    ax_sum = fig.add_subplot(gs[1, 2])
    ax_sum.axis("off")
    lines = ["RESULTS SUMMARY", "="*32, ""]
    for freq in ASSR_FREQS:
        if freq in stats_results:
            r = stats_results[freq]
            direction = "ASD < TD ✓" if r["mean_ASD"] < r["mean_TD"] else "ASD > TD ✗"
            lines += [
                f"{int(freq)}Hz ASSR:",
                f"  ASD = {r['mean_ASD']:.4f}",
                f"  TD  = {r['mean_TD']:.4f}",
                f"  t={r['t']:.3f}, p={r['p']:.4f}",
                f"  d={r['d']:.3f}",
                f"  {direction}",
                ""
            ]
    lines += [
        "TGC prediction:",
        "  ASD shows reduced ASSR",
        "  (lower \u03a9 = E/I imbalance)",
        "  Most pronounced at 40Hz",
        "  (gamma-band E/I marker)"
    ]
    ax_sum.text(0.05, 0.95, "\n".join(lines),
               transform=ax_sum.transAxes, fontsize=8.5,
               va="top", fontfamily="monospace",
               bbox=dict(boxstyle="round", fc="#F5F5F5", alpha=0.9))

    out = os.path.join(FIGURES_DIR, "tgc_sfari_study3_assr.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"\n✅ Figure saved: {out}")
    plt.show()


# ══════════════════════════════════════════════
# 6. メイン
# ══════════════════════════════════════════════
def main():
    print("=" * 55)
    print("TGC Model Study 3: SFARI ASSR Analysis")
    print("ASD vs TD  40Hz/27Hz ASSR as \u03a9 proxy")
    print("=" * 55)

    try:
        import mne
        print(f"  MNE version: {mne.__version__}")
    except ImportError:
        print("❌ pip install mne")
        return

    pheno = load_phenotypic()
    df    = run_all(pheno)
    if df is None:
        return

    stats_res = run_stats(df)
    plot_results(df, stats_res)


if __name__ == "__main__":
    mp.freeze_support()
    main()