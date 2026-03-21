"""
TGC Model: Study 1 Section 3.1.2 — Pupil Overheating Signature
===============================================================
ds003838 task-memory 瞳孔径 (n=86) を Load 5/9/13 条件別に解析。
TGC予測: Load9でピーク→Load13で低下（非単調 = Overheating シグネチャ）

pupil.tsv 形式: Pupil Labs, timestamp ベースで events.tsv と同期
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import multiprocessing as mp
warnings.filterwarnings("ignore")

EEG_DIR     = os.environ.get('DS003838_DIR', './ds003838-download')
FIGURES_DIR = os.environ.get('TGC_FIGURES_DIR', './figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# 瞳孔計測の時間窓: シーケンス開始から何秒後を使うか
# digit span: 1桁/2秒 × Load数 → Load5=10s, Load9=18s, Load13=26s
# 維持期間（最後の桁提示後）の瞳孔径を使用
BASELINE_WIN = (-1.0, 0.0)   # onset前1秒をベースライン
MEASURE_WIN  = (0.5, None)   # onset後0.5秒〜シーケンス終了まで


def load_pupil(pupil_tsv: str) -> pd.DataFrame:
    """pupil.tsv を読み込み、有効な瞳孔径データを返す"""
    available = pd.read_csv(pupil_tsv, sep="\t", nrows=0).columns.tolist()
    diam_col = "diameter_3d" if "diameter_3d" in available else "diameter"
    ts_col   = "pupil_timestamp" if "pupil_timestamp" in available else available[0]
    conf_col = "confidence" if "confidence" in available else None
    usecols  = [c for c in [ts_col, diam_col, conf_col] if c]

    # on_bad_lines='skip' で壊れた行を無視
    df = pd.read_csv(pupil_tsv, sep="\t", low_memory=False,
                     usecols=usecols, on_bad_lines="skip")
    df[ts_col]   = pd.to_numeric(df[ts_col],   errors="coerce")
    df[diam_col] = pd.to_numeric(df[diam_col], errors="coerce")

    if conf_col:
        df[conf_col] = pd.to_numeric(df[conf_col], errors="coerce")
        df = df[df[conf_col] >= 0.8]

    df = df[[ts_col, diam_col]].dropna()
    df.columns = ["timestamp", "diameter"]
    df = df.groupby("timestamp")["diameter"].mean().reset_index()
    return df if len(df) > 100 else None


def get_sequence_onsets(events_tsv: str):
    """
    Load 5/9/13 のシーケンス開始 onset (秒) を返す。
    memorize 条件のみ使用（control は除外）。
    """
    df = pd.read_csv(events_tsv, sep="\t")
    onsets = {5: [], 9: [], 13: []}
    durations = {5: [], 9: [], 13: []}

    for _, row in df.iterrows():
        tt = str(row.get("trial_type", ""))
        if "(first)" not in tt:
            continue
        if "memory" not in tt.lower():  # trial_type は "memory XX/YY correct/error: ..."
            continue
        onset = float(row["onset"])
        for load in [5, 9, 13]:
            if f"/{load:02d}" in tt:
                onsets[load].append(onset)
                # シーケンス長 = load × 2秒 (各桁2秒間隔)
                durations[load].append(load * 2.0)
                break
    return onsets, durations


def extract_pupil_per_load(pupil_df: pd.DataFrame,
                            onsets: dict, durations: dict,
                            eeg_t0_unix: float):
    """
    各 Load 条件のベースライン補正済み瞳孔径を計算。
    per-trial baseline: 各シーケンス onset の -2〜0秒。
    memory トライアルは onset~771s 以降のため pupil 範囲内に入る。
    """
    results = {}
    for load in [5, 9, 13]:
        if not onsets[load]:
            results[load] = np.nan
            continue

        trial_means = []
        for onset, dur in zip(onsets[load], durations[load]):
            t0_unix = eeg_t0_unix + onset

            # ベースライン: onset -2〜0秒
            bl = pupil_df[
                (pupil_df["timestamp"] >= t0_unix - 2.0) &
                (pupil_df["timestamp"] <  t0_unix)
            ]["diameter"]

            # シグナル: onset +2〜6秒（固定窓）
            # 全条件でベースラインから同じ時間後を計測 → 時間効果を除去
            # Load5=5桁×2s=10s なので +2〜6s は確実にシーケンス内
            sig = pupil_df[
                (pupil_df["timestamp"] >= t0_unix + 2.0) &
                (pupil_df["timestamp"] <  t0_unix + 6.0)
            ]["diameter"]

            if len(bl) < 5 or len(sig) < 5:
                continue

            bl_mean = bl.mean()
            if bl_mean <= 0:
                continue

            pct = (sig.mean() - bl_mean) / bl_mean * 100
            trial_means.append(pct)

        results[load] = np.mean(trial_means) if trial_means else np.nan

    return results


def get_eeg_t0(pupil_tsv: str, events_tsv: str) -> float:
    """
    EEG t=0 の Unix 時刻を推定。
    pupil 最初の timestamp - EEG 最初の有効 onset = t0_unix
    """
    df = pd.read_csv(pupil_tsv, sep="\t", low_memory=False,
                     usecols=["pupil_timestamp"], nrows=100)
    pupil_first = pd.to_numeric(df["pupil_timestamp"],
                                errors="coerce").dropna().iloc[0]

    df_ev = pd.read_csv(events_tsv, sep="\t")
    valid = df_ev[(df_ev["onset"] > 1.0) &
                  (~df_ev["trial_type"].str.contains("STATUS", na=False))]
    if len(valid) == 0:
        return pupil_first

    first_eeg_onset = valid["onset"].iloc[0]
    return pupil_first - first_eeg_onset


def analyze_subject(args):
    sub_id, pupil_tsv, events_tsv, eeg_json = args
    try:
        pupil_df = load_pupil(pupil_tsv)
        if pupil_df is None or len(pupil_df) < 100:
            return sub_id, None

        onsets, durations = get_sequence_onsets(events_tsv)
        if not any(onsets.values()):
            return sub_id, None

        eeg_t0 = get_eeg_t0(pupil_tsv, events_tsv)
        result = extract_pupil_per_load(pupil_df, onsets, durations, eeg_t0)
        return sub_id, result

    except Exception as e:
        return sub_id, {"error": str(e)}


def run_all():
    # pupil.tsv を持つ被験者を列挙
    pupil_files = sorted(glob.glob(
        os.path.join(EEG_DIR, "sub-*", "pupil", "*task-memory_pupil.tsv")))
    print(f"✅ pupil.tsv: {len(pupil_files)} 件")

    args_list = []
    missing_events = []
    for pf in pupil_files:
        sub_id  = os.path.basename(os.path.dirname(os.path.dirname(pf)))
        sub_dir = os.path.dirname(os.path.dirname(pf))
        eeg_dir = os.path.join(sub_dir, "eeg")
        events_tsv = os.path.join(eeg_dir, f"{sub_id}_task-memory_events.tsv")
        eeg_json   = os.path.join(eeg_dir, f"{sub_id}_task-memory_eeg.json")
        if os.path.exists(events_tsv):
            args_list.append((sub_id, pf, events_tsv, eeg_json))
        else:
            missing_events.append(sub_id)

    print(f"  events.tsv 照合済み: {len(args_list)} 件")
    if missing_events:
        print(f"  ⚠ events.tsv なし ({len(missing_events)}件): "
              f"{missing_events[:5]}{'...' if len(missing_events)>5 else ''}")

    rows = []
    for done, args in enumerate(args_list, 1):
        sub_id, result = analyze_subject(args)
        if result and "error" not in result:
            valid = {k: v for k, v in result.items()
                     if not (isinstance(v, float) and np.isnan(v))}
            if valid:
                row = {"subject": sub_id}
                row.update({f"pupil_load{k}": v for k, v in result.items()})
                rows.append(row)
                vals = ", ".join(f"L{k}={v:+.2f}%" for k, v in valid.items())
                print(f"  [{done:2d}/{len(args_list)}] {sub_id}: {vals}")
            else:
                print(f"  [{done:2d}/{len(args_list)}] {sub_id}: no valid trials")
        elif result and "error" in result:
            print(f"  [{done:2d}/{len(args_list)}] {sub_id}: ⚠ {result['error'][:60]}")
        else:
            print(f"  [{done:2d}/{len(args_list)}] {sub_id}: skip")

    if not rows:
        print("❌ 有効データなし")
        return None

    df = pd.DataFrame(rows)

    # 3×IQR 外れ値を NaN に
    pupil_cols = [c for c in df.columns if c.startswith("pupil_load")]
    for col in pupil_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 3*iqr, q3 + 3*iqr
        mask = (df[col] < lo) | (df[col] > hi)
        if mask.sum() > 0:
            print(f"  外れ値除去 {col}: {mask.sum()}件 "
                  f"(範囲外: {df.loc[mask, 'subject'].values})")
            df.loc[mask, col] = np.nan

    out = os.path.join(FIGURES_DIR, "pupil_loads_ds003838.csv")
    df.to_csv(out, index=False)
    print(f"\n✅ 保存: {out}  (n={len(df)})")
    return df


def run_stats(df: pd.DataFrame):
    print("\n" + "=" * 55)
    print("統計: Load 条件間の瞳孔径比較")
    print("=" * 55)
    loads = [5, 9, 13]
    cols  = [f"pupil_load{l}" for l in loads]
    complete = df[cols].dropna()
    print(f"  complete cases: {len(complete)}")

    means = [complete[c].mean() for c in cols]
    sems  = [complete[c].sem()  for c in cols]

    for l, m, s in zip(loads, means, sems):
        print(f"  Load {l:2d}: {m:+.3f}% (SEM={s:.3f})")

    # 線形傾向
    if len(complete) >= 5:
        f, p_anova = stats.f_oneway(*[complete[c].values for c in cols])
        print(f"\n  ANOVA: F={f:.3f}, p={p_anova:.4f}")

        for (l1,c1),(l2,c2) in [((5,cols[0]),(9,cols[1])),
                                  ((9,cols[1]),(13,cols[2])),
                                  ((5,cols[0]),(13,cols[2]))]:
            paired = complete[[c1,c2]].dropna()
            t, p = stats.ttest_rel(paired[c1], paired[c2])
            d = (paired[c1].mean()-paired[c2].mean()) / paired[c1].std()
            print(f"  L{l1}→L{l2}: t={t:.3f}, p={p:.4f}, d={d:.3f} (n={len(paired)})")

    # TGC パターン判定
    diff_5_9  = means[1] - means[0]
    diff_9_13 = means[2] - means[1]
    if diff_5_9 > 0 and diff_9_13 < 0:
        pattern = "✅ NON-MONOTONIC PEAK AT LOAD 9 (TGC Overheating signature)"
    elif diff_5_9 > 0:
        pattern = "→ Monotonic increase (partial TGC consistent)"
    elif diff_9_13 < 0:
        pattern = "→ Load13 suppression only"
    else:
        pattern = "✗ No predicted pattern"
    print(f"\n  Pattern: {pattern}")
    return complete, means, sems


def plot_results(df: pd.DataFrame, complete, means, sems):
    loads = [5, 9, 13]
    cols  = [f"pupil_load{l}" for l in loads]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "TGC Model Study 1 — Pupil Diameter vs. Cognitive Load\n"
        "ds003838 task-memory memorize condition | NON-CONFIRMATORY",
        fontsize=12, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.90, bottom=0.07)

    colors = ["#2196F3","#FF9800","#F44336"]

    # Panel A: 平均 ± SEM
    ax_A = fig.add_subplot(gs[0,0])
    ax_A.errorbar(loads, means, yerr=sems, fmt="o-",
                  color="#37474F", lw=2.5, ms=9, capsize=5)
    ax_A.fill_between(loads,
                      [m-s for m,s in zip(means,sems)],
                      [m+s for m,s in zip(means,sems)],
                      alpha=0.15, color="#37474F")
    ax_A.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax_A.set_xticks(loads)
    ax_A.set_xlabel("Cognitive Load (digits)", fontsize=10)
    ax_A.set_ylabel("Pupil diameter change (%)\nvs. baseline", fontsize=10)
    ax_A.set_title("Mean ± SEM\nMemorize condition", fontsize=11)
    ax_A.annotate("TGC prediction:\nPeak at Load 9\n(pre-collapse Overheating)",
                  xy=(0.05,0.95), xycoords="axes fraction",
                  fontsize=8.5, va="top", style="italic", color="darkblue",
                  bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))

    # Panel B: スパゲッティ
    ax_B = fig.add_subplot(gs[0,1])
    for _, row in complete.iterrows():
        vals = [row[c] for c in cols]
        color = "#F44336" if vals[1] == max(vals) else "gray"
        ax_B.plot(loads, vals, color=color,
                  alpha=0.5 if color=="gray" else 0.7, lw=0.9)
    ax_B.errorbar(loads, means, yerr=sems, fmt="o-",
                  color="black", lw=2.5, ms=8, capsize=5, zorder=10)
    ax_B.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax_B.set_xticks(loads)
    ax_B.set_xlabel("Cognitive Load (digits)", fontsize=10)
    ax_B.set_ylabel("Pupil diameter change (%)", fontsize=10)
    n_peak9 = sum(1 for _, row in complete.iterrows()
                  if row[cols[1]] == max(row[cols].values))
    ax_B.set_title(f"Individual trajectories (n={len(complete)})\n"
                   f"Red = Load9 peak ({n_peak9}/{len(complete)} subj.)", fontsize=11)

    # Panel C: バイオリン
    ax_C = fig.add_subplot(gs[1,0])
    data_list = [complete[c].values for c in cols]
    parts = ax_C.violinplot(data_list, positions=loads, widths=2,
                             showmeans=True, showmedians=False)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    ax_C.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax_C.set_xticks(loads)
    ax_C.set_xlabel("Cognitive Load (digits)", fontsize=10)
    ax_C.set_ylabel("Pupil diameter change (%)", fontsize=10)
    ax_C.set_title("Distribution", fontsize=11)

    # Panel D: サマリー
    ax_D = fig.add_subplot(gs[1,1])
    ax_D.axis("off")
    diff_5_9  = means[1]-means[0]
    diff_9_13 = means[2]-means[1]
    n_peak9 = sum(1 for _, row in complete.iterrows()
                  if row[cols[1]] == max(row[cols].values))
    pattern = "NON-MONOTONIC ✅" if diff_5_9>0 and diff_9_13<0 else "OTHER"

    summary = (
        f"RESULTS SUMMARY\n{'='*35}\n"
        f"n (complete) = {len(complete)}\n\n"
        f"Load  5: {means[0]:+.3f}% (SEM={sems[0]:.3f})\n"
        f"Load  9: {means[1]:+.3f}% (SEM={sems[1]:.3f})\n"
        f"Load 13: {means[2]:+.3f}% (SEM={sems[2]:.3f})\n\n"
        f"L5\u21929:  {diff_5_9:+.3f}%\n"
        f"L9\u219213: {diff_9_13:+.3f}%\n\n"
        f"Load9 peak: {n_peak9}/{len(complete)} subjects\n\n"
        f"Pattern: {pattern}\n\n"
        f"TGC prediction:\n"
        f"  \u03b2 tracks load (pupil dilation)\n"
        f"  Peak before fold boundary (Load9)\n"
        f"  Collapse/suppression at Load13"
    )
    ax_D.text(0.05, 0.95, summary, transform=ax_D.transAxes,
              fontsize=9, va="top", fontfamily="monospace",
              bbox=dict(boxstyle="round", fc="#F5F5F5", alpha=0.9))

    out = os.path.join(FIGURES_DIR, "tgc_pupil_study1_n86.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"\n✅ Figure saved: {out}")
    plt.show()


def main():
    print("="*55)
    print("TGC Model Study 1: Pupil Diameter Analysis")
    print("ds003838 memorize condition  n~86")
    print("="*55)

    df = run_all()
    if df is None:
        return

    complete, means, sems = run_stats(df)
    plot_results(df, complete, means, sems)


if __name__ == "__main__":
    mp.freeze_support()
    main()