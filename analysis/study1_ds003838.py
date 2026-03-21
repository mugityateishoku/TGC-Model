"""
TGC Model — Study 1: Boundary-Condition Plausibility Check (ds003838)
======================================================================
Harada (2026): Thermostatic Gain Control Model, Section 3.1
NON-CONFIRMATORY EXPLORATORY ANALYSIS

Dataset: OpenNeuro ds003838 (digit-span task)
    https://openneuro.org/datasets/ds003838
    Download the dataset and set DATASET_ROOT below to its path.

Dataset structure:
    beh/   *_beh.tsv       : condition(5/9/13), NCorrect, partialScore
    pupil/ *_pupil.tsv     : pupil diameter time series
           *_events.tsv    : timestamp + label (event codes)
    eeg/   *_eeg.set       : 64ch EEG (.set format)

Beta proxy variables:
    Primary   : task-evoked pupil dilation (LC-NE gain index)
    Secondary : trial-wise NCorrect proportion (limited; no RT in dataset)

Analyses:
    Section 3.1.1 — KDE potential landscape (heuristic only)
    Section 3.1.2 — Overheating signature (pupil + accuracy variability)
    Section 3.1.3 — Macroscopic hysteresis (non-significant; design limitation)
    Section 3.1.4 — EEG 1/f aperiodic slope (null expected)

Usage:
    python analysis/study1_ds003838.py
"""

import os, glob, warnings, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import seaborn as sns

# ── Japanese font setup (optional, for local plots) ──
def _set_japanese_font():
    """利用可能な日本語フォントを自動選択する"""
    jp_fonts = [
        "MS Gothic", "Meiryo", "Yu Gothic", "MS Mincho",   # Windows
        "Hiragino Sans", "Hiragino Kaku Gothic Pro",        # macOS
        "IPAGothic", "IPAPGothic", "Noto Sans CJK JP",     # Linux
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in jp_fonts:
        if font in available:
            plt.rcParams["font.family"] = font
            return font
    # Fallback: warn if no Japanese font available
    print("No Japanese font found. Labels shown in English.")
    return None

_JP_FONT = _set_japanese_font()
from scipy import stats
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════
# 0. CONFIG
# ══════════════════════════════════════════════
DATASET_ROOT = "ds003838-download"
FIGURES_DIR  = "figures_tgc"
os.makedirs(FIGURES_DIR, exist_ok=True)

# beh TSV column names
SUBJ_COL      = "participant_id"
COND_COL      = "condition"        # values: 5, 9, 13
TASK_COL      = "task"             # use 'remember' task only
NCORRECT_COL  = "NCorrect"
PARTIAL_COL   = "partialScore"

# Load condition mapping
LOAD_MAP   = {5: "Load5", 9: "Load9", 13: "Load13"}
LOAD_ORDER = ["Load5", "Load9", "Load13"]

# EEG settings
EEG_FMIN, EEG_FMAX = 1.0, 40.0

# Subjects without EEG (per dataset README)
NO_EEG_SUBS = {
    "sub-013","sub-014","sub-015","sub-016","sub-017","sub-018","sub-019","sub-020",
    "sub-021","sub-022","sub-023","sub-024","sub-025","sub-026","sub-027","sub-028",
    "sub-029","sub-030","sub-031","sub-037","sub-066"
}
NO_PUPIL_SUBS = {"sub-017", "sub-094"}

RNG_SEED = 42
np.random.seed(RNG_SEED)


# ══════════════════════════════════════════════
# 1. Load behavioral data (beh/)
# ══════════════════════════════════════════════
def load_beh_data() -> pd.DataFrame:
    """
    Load all  *_beh.tsv files.
    task == 'remember'（記憶課題）のみ使用。
    """
    records = []
    for f in sorted(glob.glob(
            os.path.join(DATASET_ROOT, "**", "beh", "*_beh.tsv"), recursive=True)):
        df = pd.read_csv(f, sep="\t")
        records.append(df)
    if not records:
        raise FileNotFoundError("*_beh.tsv not found. Please check DATASET_ROOT.")
    data = pd.concat(records, ignore_index=True)

    # Memory task only
    data = data[data[TASK_COL] == "remember"].copy()
    # Assign condition labels
    data["Condition"] = data[COND_COL].map(LOAD_MAP)
    data = data.dropna(subset=["Condition"])
    # Accuracy proportion (0–1)
    data["acc_prop"] = data[NCORRECT_COL] / data[COND_COL]

    print(f"✅ Behavioral data: {data[SUBJ_COL].nunique()} subjects, "
          f"{len(data)} trials (memory task only)")
    return data


# ══════════════════════════════════════════════
# 2. Load pupil data (pupil/)
# ══════════════════════════════════════════════
def _label_to_condition(label) -> str | None:
    """
    Map event label to condition name.
    memory events: 6PPPLLLC 形式
      PPP = digit position (001–013)
      LLL = sequence length (050/090/130)
      C   = correct(1) / error(0)
    control events: 5PPLLL 形式 → 除外
    """
    try:
        s = str(int(label))
    except (ValueError, TypeError):
        return None
    if not s.startswith("6"):   # exclude control events (5xxx)
        return None
    # 末尾3桁がLLLC: 050/051→Load5, 090/091→Load9, 130/131→Load13
    if len(s) >= 7:
        llc = s[-3:]
        if llc in ("050", "051"): return "Load5"
        if llc in ("090", "091"): return "Load9"
        if llc in ("130", "131"): return "Load13"
    return None


def load_pupil_data() -> pd.DataFrame:
    """
    pupil.tsv から task-evoked 瞳孔径を条件ごとに集計。

    データ構造（探索スクリプトで確認済み）:
      - 列名 `diameter`（ピクセル単位の瞳孔径）
      - 列名 `pupil_timestamp`（Unix秒）
      - 列名 `eye_id`（0=left, 1=right）→ 両目を confidence 加重平均
      - 列名 `confidence`（0–1）→ 0.6 以下の低信頼サンプルを除外
      - blink == 1 のサンプルを除外
    イベントは pupil/ の events.tsv から取得（列: timestamp, label）

    利用可能被験者: 14 名（dataset全体の約17%）
    → 探索的解析の限界として図中に明記
    """
    CONFIDENCE_THRESH = 0.6
    WINDOW_SEC = 1.8   # 刺激提示から1.8秒のウィンドウ（SOA=2sに合わせる）

    rows = []

    for sub_dir in sorted(glob.glob(os.path.join(DATASET_ROOT, "sub-*"))):
        sub = os.path.basename(sub_dir)
        if sub in NO_PUPIL_SUBS:
            continue

        p_files = glob.glob(os.path.join(sub_dir, "pupil", "*_pupil.tsv"))
        e_files = glob.glob(os.path.join(sub_dir, "pupil", "*_events.tsv"))
        if not p_files or not e_files:
            continue

        try:
            df_p = pd.read_csv(p_files[0], sep="\t",
                               usecols=["pupil_timestamp", "eye_id",
                                        "confidence", "diameter", "blink"])
        except Exception:
            continue

        try:
            df_ev = pd.read_csv(e_files[0], sep="\t",
                                usecols=["timestamp", "label"])
        except Exception:
            continue

        # 前処理: blink 除外 + confidence フィルタ
        df_p = df_p[
            (df_p["confidence"] >= CONFIDENCE_THRESH) &
            (df_p["blink"] != 1) &
            (df_p["diameter"] > 0)
        ].copy()
        if len(df_p) < 100:
            continue

        # 両目を confidence 加重平均して1時系列に統合（高速化）
        df_p["w_dia"] = df_p["confidence"] * df_p["diameter"]
        grp = df_p.groupby("pupil_timestamp").agg(
            w_dia_sum=("w_dia", "sum"),
            conf_sum=("confidence", "sum")
        ).reset_index()
        grp["diameter_avg"] = grp["w_dia_sum"] / grp["conf_sum"]
        df_p_avg = grp[["pupil_timestamp", "diameter_avg"]]
        ts_arr  = df_p_avg["pupil_timestamp"].values
        dia_arr = df_p_avg["diameter_avg"].values

        # サンプリング周波数を推定
        if len(ts_arr) > 1:
            sfreq = len(ts_arr) / (ts_arr[-1] - ts_arr[0])
        else:
            sfreq = 120.0
        win_n = max(int(sfreq * WINDOW_SEC), 5)

        # イベントごとにウィンドウを切り出してbaseline補正した平均を取得
        # baseline: イベント直前 0.5秒
        base_n = max(int(sfreq * 0.5), 3)

        for _, ev in df_ev.iterrows():
            cond = _label_to_condition(ev["label"])
            if cond is None:
                continue
            idx = np.searchsorted(ts_arr, ev["timestamp"])
            i_base_start = max(0, idx - base_n)
            i_seg_end    = min(len(dia_arr), idx + win_n)
            if i_seg_end - idx < 5:
                continue

            baseline = dia_arr[i_base_start:idx]
            segment  = dia_arr[idx:i_seg_end]

            if len(baseline) < 2 or len(segment) < 5:
                continue

            # ベースライン補正（加算性: 平均差分）
            evoked = segment.mean() - baseline.mean()
            rows.append({SUBJ_COL: sub, "Condition": cond,
                         "pupil_evoked": evoked})

    if not rows:
        print("⚠ Pupil data could not be loaded.")
        return pd.DataFrame()

    df_all = pd.DataFrame(rows)
    df_agg = (df_all.groupby([SUBJ_COL, "Condition"])["pupil_evoked"]
              .mean().reset_index())
    n = df_agg[SUBJ_COL].nunique()
    print(f"✅ Pupil data: {n} subjects  "
          f"(~{n*100//84}% of dataset — noted as analysis constraint)")
    return df_agg


# ══════════════════════════════════════════════
# 3. Section 3.1.1 — KDE ポテンシャルランドスケープ
# ══════════════════════════════════════════════
def plot_potential_landscape(beh: pd.DataFrame, ax: plt.Axes):
    """
    V(x) ∝ −ln P(x)  using KDE of acc_prop per condition.

    【方法論的制約 — Section 3.1.1】
    - 定常分布の仮定に違反（非定常試行構造）
    - NCorrect/条件は離散かつ有界 (continuity 仮定違反)
    - 目的: 「Load増大 → マルチモーダル→単峰崩壊」の定性的トポロジー図示のみ
    """
    x_grid = np.linspace(-0.05, 1.05, 400)
    colors  = ["#2196F3", "#FF9800", "#F44336"]

    for i, cond in enumerate(LOAD_ORDER):
        vals = beh[beh["Condition"] == cond]["acc_prop"].dropna().values
        if len(vals) < 10:
            continue
        kde = gaussian_kde(vals, bw_method="silverman")
        p_x = np.clip(kde(x_grid), 1e-10, None)
        V_x = -np.log(p_x)
        V_x -= V_x.min()
        ax.plot(x_grid, V_x, color=colors[i], lw=2.5, label=cond)

    ax.set_xlabel("β proxy  (NCorrect / Condition)", fontsize=10)
    ax.set_ylabel("Pseudo-Potential  V(x) ∝ −ln P(x)", fontsize=10)
    ax.set_title("Heuristic Potential Landscape\n"
                 "(Section 3.1.1 — Illustrative only)", fontsize=11)
    ax.legend(fontsize=9)
    ax.annotate(
        "Note: Discrete/bounded accuracy violates Langevin continuity.\n"
        "Heuristic visualization only — not a rigorous dynamical estimate.",
        xy=(0.02, 0.97), xycoords="axes fraction",
        fontsize=7.5, va="top", color="gray")


# ══════════════════════════════════════════════
# 4. Section 3.1.2 — Overheating署名
#    Primary: 瞳孔径 (LC-NE proxy)
#    Secondary: 行動変動 (NCorrect IIV)
# ══════════════════════════════════════════════
def compute_acc_iiv(beh: pd.DataFrame) -> pd.DataFrame:
    """
    Within-subject SD of acc_prop per subject x condition.
    β instability の粗い行動代理 (RT-IIV の代替、本データにRTなし)。
    """
    return (beh.groupby([SUBJ_COL, "Condition"])["acc_prop"]
            .std().reset_index().rename(columns={"acc_prop": "acc_IIV"}))


def plot_overheating_signature(pupil: pd.DataFrame, acc_iiv: pd.DataFrame,
                                fig: plt.Figure, gs_cell):
    """
    Overheating署名: 瞳孔径 + 行動変動 の 2-panel。
    TGC予測: Load 13 で両指標がピーク（Section 3.1.2）。
    """
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_cell, wspace=0.4)
    ax_p = fig.add_subplot(inner[0])
    ax_b = fig.add_subplot(inner[1])

    # ── Pupil ──
    def _stats_label(g1, g2, name):
        v1 = g1.dropna(); v2 = g2.dropna()
        if len(v1) < 2 or len(v2) < 2:
            return ""
        t, p = stats.ttest_ind(v1, v2, equal_var=False)
        sd = np.sqrt((v1.std()**2 + v2.std()**2) / 2)
        d  = (v2.mean() - v1.mean()) / sd if sd > 0 else np.nan
        sig = "p<.05 *" if p < 0.05 else "n.s."
        return f"L9 vs L13: t={t:.2f}, p={p:.4f}, d={d:.3f} [{sig}]"

    if not pupil.empty:
        for _, row in pupil.pivot(
                index=SUBJ_COL, columns="Condition",
                values="pupil_evoked").iterrows():
            ax_p.plot(LOAD_ORDER, [row.get(c, np.nan) for c in LOAD_ORDER],
                      color="gray", alpha=0.2, lw=0.8)
        sns.boxplot(data=pupil, x="Condition", y="pupil_evoked",
                    order=LOAD_ORDER, palette="YlOrRd", ax=ax_p, showfliers=False)
        p9  = pupil[pupil["Condition"]=="Load9"]["pupil_evoked"]
        p13 = pupil[pupil["Condition"]=="Load13"]["pupil_evoked"]
        ax_p.set_xlabel(_stats_label(p9, p13, "Pupil"), fontsize=7.5)
    else:
        ax_p.text(0.5, 0.5, "No pupil data\n(pupil.tsv: only 14 subjects)", ha="center", va="center",
                  transform=ax_p.transAxes, fontsize=10)

    ax_p.set_ylabel("Baseline-corrected Pupil Diameter (a.u.)", fontsize=9)
    ax_p.set_title("Primary: Pupil Dilation\n(LC-NE / β proxy)", fontsize=10)

    # ── Behavioral variability ──
    if not acc_iiv.empty:
        for _, row in acc_iiv.pivot(
                index=SUBJ_COL, columns="Condition",
                values="acc_IIV").iterrows():
            ax_b.plot(LOAD_ORDER, [row.get(c, np.nan) for c in LOAD_ORDER],
                      color="gray", alpha=0.2, lw=0.8)
        sns.boxplot(data=acc_iiv, x="Condition", y="acc_IIV",
                    order=LOAD_ORDER, palette="YlOrRd", ax=ax_b, showfliers=False)
        b9  = acc_iiv[acc_iiv["Condition"]=="Load9"]["acc_IIV"]
        b13 = acc_iiv[acc_iiv["Condition"]=="Load13"]["acc_IIV"]
        ax_b.set_xlabel(_stats_label(b9, b13, "AccIIV"), fontsize=7.5)

    ax_b.set_ylabel("Acc IIV (within-subj SD)", fontsize=9)
    ax_b.set_title("Secondary: Accuracy Variability\n(RT-IIV surrogate; no RT in dataset)", fontsize=10)

    # Common title
    ax_p.set_title("Overheating Signature (Section 3.1.2)\n" + ax_p.get_title(),
                   fontsize=10)
    ax_p.annotate(
        "n=14 (16% of dataset); low statistical power.\n"
        "Load9 peak observed; Load13 drop may reflect\n"
        "post-collapse LC-NE suppression (not predicted by TGC).",
        xy=(0.02, 0.02), xycoords="axes fraction",
        fontsize=7, va="bottom", color="gray", style="italic")


# ══════════════════════════════════════════════
# 5. Section 3.1.3 — 巨視的ヒステリシス (非有意)
# ══════════════════════════════════════════════
def compute_hysteresis(beh: pd.DataFrame) -> dict:
    """
    ループ面積 A = ∮β dE の代理。

    【設計上の限界 — Section 3.1.3】
    ds003838 には counterbalanced ascending/descending 設計がない。
    疲労・順序効果とトポロジカルヒステリシスを分離不可。
    → 「情報なし」として扱う。論文値: Area=11.6, d=0.423, p=0.418。
    """
    # Compute proxy loop area from subject-level mean accuracy
    pivot = (beh.groupby([SUBJ_COL, "Condition"])["acc_prop"]
             .mean().unstack("Condition"))
    pivot = pivot[[c for c in LOAD_ORDER if c in pivot.columns]]

    loads_num = [5, 9, 13][:len(pivot.columns)]
    areas = []
    for _, row in pivot.iterrows():
        vals = row.values
        if not np.any(np.isnan(vals)):
            areas.append(np.trapz(vals, loads_num))

    areas = np.array(areas)
    # # No test: no ascending/descending design
    res = {"n": len(areas)}
    if len(areas) > 0:
        res["mean_area"] = float(areas.mean())
        res["sd_area"]   = float(areas.std())
    return res


def plot_hysteresis(hyst: dict, ax: plt.Axes):
    ax.axis("off")
    info = (
        f"Loop Area (mean) = {hyst.get('mean_area', float('nan')):.2f}\n"
        f"           (SD)  = {hyst.get('sd_area',   float('nan')):.2f}\n"
        f"n                = {hyst.get('n', '—')}\n\n"
        f"Statistical test: NOT conducted"
    )
    ax.text(0.5, 0.62, info, transform=ax.transAxes,
            ha="center", va="center", fontsize=13,
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.9))
    ax.text(0.5, 0.18,
            "No counterbalanced ascending/descending design.\n"
            "Fatigue ≠ hysteresis here. Result is UNINFORMATIVE.\n"
            "(Section 3.1.3 — non-significant expected)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=8.5, color="darkred")
    ax.set_title("Macroscopic Hysteresis  A = ∮β dE\n"
                 "(Section 3.1.3 — Exploratory)", fontsize=11)


# ══════════════════════════════════════════════
# 6. Section 3.1.4 — EEG 1/f Slope (Null expected)
# ══════════════════════════════════════════════
def run_eeg_1f(beh: pd.DataFrame) -> pd.DataFrame:
    """
    グローバル 1/f aperiodic slope。

    【論文の解釈 — Section 3.1.4】
    「Global 1/f slope: no significant variation across loads (p>0.05).
     Null is interpreted as artifact of global spatial averaging
     and un-modeled alpha interference. Source-localized EEG required.」
    この null result の再現が目的。
    """
    try:
        import mne
        from mne.time_frequency import psd_array_welch
    except ImportError:
        print("⚠ MNE not installed. Run: pip install mne")
        return pd.DataFrame()

    # 被験者ごとのイベントコード→条件マッピング（ecg/events.tsv から取得）
    # memory イベントコード: 6XYZW (Y=digit pos, Z=length 05/09/13, W=0/1)
    def label_to_event_id(length: int) -> list:
        """Return all memory event codes for given length"""
        ids = []
        for pos in range(1, length + 1):
            for correct in [0, 1]:
                ids.append(int(f"6{pos:03d}{length:02d}{correct}"))
        return ids

    event_ids_by_cond = {
        "Load5":  label_to_event_id(5),
        "Load9":  label_to_event_id(9),
        "Load13": label_to_event_id(13),
    }

    results = []
    eeg_files = sorted(glob.glob(
        os.path.join(DATASET_ROOT, "**", "eeg", "*_task-memory_eeg.set"),
        recursive=True))
    print(f"  EEG files (task-memory): {len(eeg_files)}")
    if len(eeg_files) <= 1:
        print("  ⚠ task-memory EEG: only sub-034 (others are task-rest).")
        print("    Statistical test not possible — null recorded as design constraint.")
        return pd.DataFrame()

    for f in eeg_files:
        sub = os.path.basename(f).split("_")[0]
        if sub in NO_EEG_SUBS:
            continue
        try:
            raw = mne.io.read_raw_eeglab(f, preload=True, verbose=False)
            events, ev_dict = mne.events_from_annotations(raw, verbose=False)

            for cond, codes in event_ids_by_cond.items():
                # 存在するイベントIDのみ使用
                valid = {str(c): c for c in codes if str(c) in ev_dict}
                if not valid:
                    continue
                try:
                    epo = mne.Epochs(raw, events, event_id=valid,
                                     tmin=0, tmax=1.8, baseline=None,
                                     preload=True, verbose=False)
                    if len(epo) == 0:
                        continue
                    data_arr = epo.get_data()  # (epochs, ch, times)
                    psds, freqs = psd_array_welch(
                        data_arr, sfreq=raw.info["sfreq"],
                        fmin=EEG_FMIN, fmax=EEG_FMAX,
                        n_fft=256, verbose=False)
                    mean_psd = np.mean(psds, axis=(0, 1))  # グローバル平均
                    slope, *_ = stats.linregress(
                        np.log10(freqs), np.log10(mean_psd))
                    results.append({
                        SUBJ_COL: sub, "Condition": cond,
                        "slope_global": slope
                    })
                except Exception:
                    pass
        except Exception as e:
            print(f"  EEG error {os.path.basename(f)}: {e}")

    return pd.DataFrame(results)


def plot_eeg_null(df_eeg: pd.DataFrame, ax: plt.Axes):
    if df_eeg.empty:
        ax.text(0.5, 0.55,
                "task-memory EEG: sub-034 only (1 file)\n"
                "(remaining 25 = task-rest)\n\n"
                "Statistical test impossible\n"
                "Null result = dataset design constraint (Sec 3.1.4)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="darkred",
                bbox=dict(boxstyle="round", fc="#FFF3E0", alpha=0.9))
    else:
        for _, row in df_eeg.pivot(
                index=SUBJ_COL, columns="Condition",
                values="slope_global").iterrows():
            ax.plot(LOAD_ORDER, [row.get(c, np.nan) for c in LOAD_ORDER],
                    color="gray", alpha=0.25, lw=0.8)
        sns.boxplot(data=df_eeg, x="Condition", y="slope_global",
                    order=LOAD_ORDER, palette="Blues", ax=ax, showfliers=False)

        l9  = df_eeg[df_eeg["Condition"]=="Load9"]["slope_global"].dropna()
        l13 = df_eeg[df_eeg["Condition"]=="Load13"]["slope_global"].dropna()
        if len(l9) > 1 and len(l13) > 1:
            t, p = stats.ttest_ind(l9, l13, equal_var=False)
            sig = "p<.05 (unexpected)" if p < 0.05 else "n.s. (expected null)"
            ax.set_xlabel(f"L9 vs L13: t={t:.3f}, p={p:.4f} [{sig}]",
                          fontsize=8.5)

    ax.set_title("Cortical Noise: Global 1/f Aperiodic Slope\n"
                 "(Null expected — global avg artifact, Section 3.1.4)", fontsize=11)
    ax.set_ylabel("1/f Slope  (Flatter ↑ = Higher σ)", fontsize=9)
    ax.annotate(
        "Null expected: global averaging dilutes load-specific signal.\n"
        "Source-localized high-density EEG required for valid test.",
        xy=(0.02, 0.97), xycoords="axes fraction",
        fontsize=7.5, va="top", color="darkred")


# ══════════════════════════════════════════════
# 7. Main
# ══════════════════════════════════════════════
def main():
    print("=" * 65)
    print("TGC Model — Section 3 Boundary-Condition Plausibility Check")
    print("Harada (2026): Non-confirmatory exploratory analysis")
    print("=" * 65)

    # ── Load data ──────────────────────────
    beh     = load_beh_data()
    pupil   = load_pupil_data()
    acc_iiv = compute_acc_iiv(beh)

    # ── Print summary statistics ────────────────────────
    print("\n📊 Section 3.1.1 — Accuracy distribution per condition")
    print(beh.groupby("Condition")["acc_prop"]
          .agg(["mean","std","count"]).loc[LOAD_ORDER])

    print("\n📊 Section 3.1.2 — Overheating Signature")
    if not pupil.empty:
        print("  [Pupil] mean per condition:")
        print(pupil.groupby("Condition")["pupil_evoked"]
              .mean().loc[[c for c in LOAD_ORDER if c in pupil.groupby("Condition").groups]])
    print("  [Acc-IIV behavioral variability]:")
    print(acc_iiv.groupby("Condition")["acc_IIV"]
          .agg(["mean","std"]).loc[[c for c in LOAD_ORDER if c in acc_iiv.groupby("Condition").groups]])

    hyst = compute_hysteresis(beh)
    print(f"\n📊 Section 3.1.3 — Hysteresis proxy")
    print(f"  Area(mean)={hyst.get('mean_area',float('nan')):.3f}, "
          f"SD={hyst.get('sd_area',float('nan')):.3f}, n={hyst.get('n','?')}  "
          "← No statistical test (design limitation)")

    print("\n📊 Section 3.1.4 — EEG 1/f slope (null expected)")
    df_eeg = run_eeg_1f(beh)

    # ── Figure ─────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "TGC Model: Boundary-Condition Plausibility Check  (Harada, 2026)\n"
        "ds003838 — Sections 3.1.1-3.1.4  |  NON-CONFIRMATORY EXPLORATORY ANALYSIS",
        fontsize=13, fontweight="bold", y=0.98)

    gs_outer = gridspec.GridSpec(3, 2, figure=fig,
                                 hspace=0.55, wspace=0.35,
                                 top=0.93, bottom=0.05)

    # Panel A: Potential Landscape
    ax_A = fig.add_subplot(gs_outer[0, 0])
    plot_potential_landscape(beh, ax_A)

    # Panel B: Hysteresis
    ax_C = fig.add_subplot(gs_outer[1, 0])
    plot_hysteresis(hyst, ax_C)

    # Panel C: EEG
    ax_D = fig.add_subplot(gs_outer[1, 1])
    plot_eeg_null(df_eeg, ax_D)

    # Panel D: Overheating (2-col span)
    plot_overheating_signature(pupil, acc_iiv, fig, gs_outer[0, 1])

    # Panel E: Summary table
    ax_E = fig.add_subplot(gs_outer[2, :])
    ax_E.axis("off")
    summary = [
        ["Section", "Analysis", "β proxy used", "Result", "Interpretation"],
        ["3.1.1", "KDE Potential Landscape",
         "NCorrect/condition", "Qualitative topology check",
         "Heuristic only — illustrative"],
        ["3.1.2", "Overheating Signature",
         "Pupil dilation + Acc-IIV", "Pupil/IIV peak at Load13?",
         "Primary plausibility evidence"],
        ["3.1.3", "Macroscopic Hysteresis",
         "Mean accuracy × load", "Non-significant",
         "UNINFORMATIVE — design limitation"],
        ["3.1.4", "EEG 1/f Aperiodic Slope",
         "Global EEG power", "Null (p>0.05 expected)",
         "Global avg artifact — source-EEG required"],
    ]
    tbl = ax_E.table(cellText=summary[1:], colLabels=summary[0],
                     cellLoc="left", loc="center",
                     bbox=[0.0, 0.0, 1.0, 1.0])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#ECEFF1")

    out = os.path.join(FIGURES_DIR, "tgc_plausibility_check_v5.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"\n✅ Figure saved: {out}")
    plt.show()

    print("\n" + "=" * 65)
    print("Done: 4 analyses corresponding to Sections 3.1.1-3.1.4.")
    print("=" * 65)


if __name__ == "__main__":
    main()