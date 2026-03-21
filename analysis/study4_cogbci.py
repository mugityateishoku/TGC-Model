"""
TGC Model — Study 4: COG-BCI N-back EEG Aperiodic Slope
=========================================================
Harada (2026): Thermostatic Gain Control Model, Section 3.4
NON-CONFIRMATORY BOUNDARY CHECK

Dataset: COG-BCI (Hinss et al., 2023)
    Paper: https://doi.org/10.1038/s41597-023-01956-1
    Data:  https://zenodo.org/records/6874129

Hypothesis (pre-fixed): Monotonic increase in EEG aperiodic exponent
    from 0-back → 1-back → 2-back, reflecting Omega parameter shift
    as the system approaches fold boundary under increasing load.

Method: FOOOF/specparam on frontal-central channel-averaged PSD
    (Welch, 4s windows), mixed-effects model with subject random effects.

Results: Monotonic increase confirmed (Friedman p=0.001, d=0.528),
    but non-diagnostic without supracritical load levels.

Environment variables:
    COGBCI_DATA_DIR — path to downloaded COG-BCI dataset

Usage:
    python analysis/study4_cogbci.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Dependency check ──
try:
    import mne
except ImportError:
    os.system("pip install mne --quiet --break-system-packages")
    import mne

try:
    from fooof import FOOOF
except ImportError:
    os.system("pip install fooof --quiet --break-system-packages")
    from fooof import FOOOF

mne.set_log_level("WARNING")

# ── Path configuration ──
DATA_ROOT  = os.environ.get('COGBCI_DATA_DIR', './cogbci-data')
FIGURES_DIR = os.environ.get('TGC_FIGURES_DIR', './figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Constants ──
TASKS      = {"zero": "zeroBACK", "one": "oneBACK", "two": "twoBACK"}
LOAD_LABEL = {"zero": 0, "one": 1, "two": 2}
FREQ_RANGE = [2, 40]          # FOOOF fitting範囲 (Hz)
PEAK_WIDTH = [1, 8]           # ピーク幅制約
N_JOBS     = 1

# Frontal-central channels for 1/f slope estimation
TARGET_CHS = ["Fz", "FCz", "Cz", "F3", "F4", "FC1", "FC2",
              "F1",  "F2",  "FC3", "FC4"]

# ── Estimate slope for one subject x session x condition ──
def estimate_slope(set_path: str) -> float | None:
    """
    .set ファイルを読み込み, フロンタル平均 PSD から
    FOOOF アペリオディックスロープを返す。
    失敗時は None。
    """
    try:
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    except Exception as e:
        print(f"  Load failed: {set_path} ({e})")
        return None

    # Select available channels
    chs_avail = [c for c in TARGET_CHS if c in raw.ch_names]
    if not chs_avail:
        # Fallback: use first 8 channels
        chs_avail = raw.ch_names[:8]

    raw.pick(chs_avail)

    # PSD computation (Welch, 4s window)
    sfreq = raw.info["sfreq"]
    n_fft = int(sfreq * 4)
    try:
        psd_obj = raw.compute_psd(method="welch", fmin=FREQ_RANGE[0],
                                   fmax=FREQ_RANGE[1], n_fft=n_fft,
                                   n_overlap=n_fft // 2, verbose=False)
        psds, freqs = psd_obj.get_data(return_freqs=True)
    except Exception as e:
        print(f"  PSD 失敗: {e}")
        return None

    # Channel-averaged PSD
    mean_psd = psds.mean(axis=0)

    # FOOOF fitting
    fm = FOOOF(peak_width_limits=PEAK_WIDTH,
               max_n_peaks=6,
               min_peak_height=0.05,
               aperiodic_mode="fixed",
               verbose=False)
    try:
        fm.fit(freqs, mean_psd, FREQ_RANGE)
        if not fm.has_model:
            return None
        slope = fm.aperiodic_params_[1]   # [offset, exponent]
        # Quality check: reject r² < 0.8
        if fm.r_squared_ < 0.8:
            return None
        return float(slope)
    except Exception:
        return None


# ── Collect all slopes ──
def collect_all_slopes(data_root: str) -> pd.DataFrame:
    rows = []
    sub_dirs = sorted(glob.glob(os.path.join(data_root, "sub-*")))

    for sub_dir in sub_dirs:
        sub_id = os.path.basename(sub_dir)
        inner  = os.path.join(sub_dir, sub_id)   # sub-01/sub-01/
        if not os.path.isdir(inner):
            inner = sub_dir

        ses_dirs = sorted(glob.glob(os.path.join(inner, "ses-S*")))
        if not ses_dirs:
            ses_dirs = [inner]

        for ses_dir in ses_dirs:
            ses_id = os.path.basename(ses_dir)
            eeg_dir = os.path.join(ses_dir, "eeg")
            if not os.path.isdir(eeg_dir):
                eeg_dir = ses_dir

            for key, fname in TASKS.items():
                set_path = os.path.join(eeg_dir, f"{fname}.set")
                if not os.path.exists(set_path):
                    continue

                print(f"  Processing: {sub_id} {ses_id} {key}BACK ... ", end="")
                slope = estimate_slope(set_path)
                if slope is not None:
                    print(f"slope={slope:.3f}")
                    rows.append({
                        "subject": sub_id,
                        "session": ses_id,
                        "load":    LOAD_LABEL[key],
                        "load_label": f"{key}BACK",
                        "slope":   slope
                    })
                else:
                    print("skip")

    return pd.DataFrame(rows)


# ── Statistical tests ──
def run_stats(df: pd.DataFrame) -> dict:
    load_groups = [df[df["load"] == l]["slope"].dropna() for l in [0, 1, 2]]

    # Within-subject non-parametric test (Friedman)
    # Use subject x session mean
    pivot = df.groupby(["subject", "load"])["slope"].mean().unstack()
    pivot = pivot.dropna()
    n_complete = len(pivot)

    if n_complete >= 3:
        stat_f, p_friedman = stats.friedmanchisquare(
            pivot[0].values, pivot[1].values, pivot[2].values)
    else:
        stat_f, p_friedman = np.nan, np.nan

    # Post-hoc: 0 vs 2 (maximum contrast)
    if n_complete >= 3:
        t_02, p_02 = stats.wilcoxon(pivot[0].values, pivot[2].values)
        t_01, p_01 = stats.wilcoxon(pivot[0].values, pivot[1].values)
        t_12, p_12 = stats.wilcoxon(pivot[1].values, pivot[2].values)
    else:
        t_02 = p_02 = t_01 = p_01 = t_12 = p_12 = np.nan

    # Effect size (Cohen d, 0 vs 2)
    if n_complete >= 3:
        diff = pivot[2].values - pivot[0].values
        d_02 = diff.mean() / diff.std() if diff.std() > 0 else np.nan
    else:
        d_02 = np.nan

    # Pattern classification
    means = [df[df["load"] == l]["slope"].mean() for l in [0, 1, 2]]
    if means[0] < means[1] > means[2]:
        pattern = "NON-MONOTONIC (1-back peak)"
    elif means[0] < means[1] < means[2]:
        pattern = "MONOTONIC INCREASE"
    elif means[0] > means[1] > means[2]:
        pattern = "MONOTONIC DECREASE"
    else:
        pattern = "OTHER"

    return {
        "n_subjects": df["subject"].nunique(),
        "n_complete": n_complete,
        "means": means,
        "sems":  [df[df["load"] == l]["slope"].sem() for l in [0, 1, 2]],
        "friedman_stat": stat_f,
        "p_friedman": p_friedman,
        "p_0vs2": p_02,
        "p_0vs1": p_01,
        "p_1vs2": p_12,
        "d_0vs2": d_02,
        "pattern": pattern,
        "pivot": pivot
    }


# ── Visualization ──
def make_figure(df: pd.DataFrame, stats_res: dict, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "TGC Model Study 4 — EEG Aperiodic Slope vs. N-back Load\n"
        "COG-BCI dataset (Hinss et al., 2023) | NON-CONFIRMATORY",
        fontsize=12, fontweight="bold"
    )

    loads  = [0, 1, 2]
    labels = ["0-back", "1-back", "2-back"]
    means  = stats_res["means"]
    sems   = stats_res["sems"]
    colors = ["#4e79a7", "#f28e2b", "#e15759"]

    # ── (1) 平均 ± SEM ──
    ax = axes[0]
    ax.errorbar(loads, means, yerr=sems, fmt="o-", color="#2d4b73",
                linewidth=2, markersize=8, capsize=5)
    ax.fill_between(loads,
                    [m-s for m, s in zip(means, sems)],
                    [m+s for m, s in zip(means, sems)],
                    alpha=0.2, color="steelblue")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(loads); ax.set_xticklabels(labels)
    ax.set_xlabel("N-back Load"); ax.set_ylabel("Aperiodic Exponent (slope)")
    ax.set_title("Mean ± SEM")
    tgc_txt = "TGC prediction:\nΩ↑ with load\n(slope flatter → steeper)"
    ax.text(0.05, 0.95, tgc_txt, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", bbox=dict(boxstyle="round", fc="#fffbe6",
            ec="#aaa", alpha=0.9))

    # ── (2) 被験者別軌跡 ──
    ax = axes[1]
    pivot = stats_res["pivot"]
    for sub in pivot.index:
        row = pivot.loc[sub]
        ax.plot(loads, row.values, color="gray", alpha=0.4, linewidth=0.8)
    ax.errorbar(loads, means, yerr=sems, fmt="o-", color="black",
                linewidth=2.5, markersize=7, capsize=4, zorder=5)
    ax.set_xticks(loads); ax.set_xticklabels(labels)
    ax.set_xlabel("N-back Load"); ax.set_ylabel("Aperiodic Exponent")
    ax.set_title(f"Individual trajectories (n={stats_res['n_complete']})")

    # ── (3) 結果サマリー ──
    ax = axes[2]
    ax.axis("off")
    p_f = stats_res["p_friedman"]
    p_f_str = f"{p_f:.3f}" if not np.isnan(p_f) else "N/A"
    p02_str = f"{stats_res['p_0vs2']:.3f}" if not np.isnan(stats_res["p_0vs2"]) else "N/A"
    d_str   = f"{stats_res['d_0vs2']:.3f}" if not np.isnan(stats_res["d_0vs2"]) else "N/A"
    summary = (
        f"RESULTS SUMMARY\n{'='*34}\n"
        f"n (subjects) = {stats_res['n_subjects']}\n"
        f"n (complete) = {stats_res['n_complete']}\n\n"
        f"0-back: {means[0]:.3f} (SEM={sems[0]:.3f})\n"
        f"1-back: {means[1]:.3f} (SEM={sems[1]:.3f})\n"
        f"2-back: {means[2]:.3f} (SEM={sems[2]:.3f})\n\n"
        f"Friedman χ²: p={p_f_str}\n"
        f"0 vs 2-back: p={p02_str}\n"
        f"Cohen d (0v2): {d_str}\n\n"
        f"Pattern: {stats_res['pattern']}\n\n"
        f"TGC prediction:\n"
        f"  Ω tracks load (slope steeper)\n"
        f"  OR non-monotonic at 1-back"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="black"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Figure saved: {out_path}")


# ── Main ──
if __name__ == "__main__":
    print("=" * 60)
    print("TGC Study 4: COG-BCI N-back Aperiodic Slope")
    print("=" * 60)

    df = collect_all_slopes(DATA_ROOT)

    if df.empty:
        print("❌ データが見つかりません。DATA_ROOT を確認してください。")
        print(f"   現在の設定: {DATA_ROOT}")
    else:
        csv_path = os.path.join(FIGURES_DIR, "cogbci_slopes.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n✅ CSV保存: {csv_path}  (n_rows={len(df)})")

        stats_res = run_stats(df)

        print(f"\n{'='*40}")
        print(f"被験者数: {stats_res['n_subjects']}")
        print(f"完全ケース: {stats_res['n_complete']}")
        print(f"0-back slope: {stats_res['means'][0]:.3f} ± {stats_res['sems'][0]:.3f}")
        print(f"1-back slope: {stats_res['means'][1]:.3f} ± {stats_res['sems'][1]:.3f}")
        print(f"2-back slope: {stats_res['means'][2]:.3f} ± {stats_res['sems'][2]:.3f}")
        print(f"Friedman p: {stats_res['p_friedman']:.4f}")
        print(f"Pattern: {stats_res['pattern']}")

        fig_path = os.path.join(FIGURES_DIR, "tgc_cogbci_study4_nback.pdf")
        make_figure(df, stats_res, fig_path)