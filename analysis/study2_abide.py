"""
TGC Model - Study 2: ABIDE Resting-State fMRI 1/f Slope
=======================================================
Harada (2026): Thermostatic Gain Control Model, Section 3
NON-CONFIRMATORY BOUNDARY CHECK - NULL RESULT EXPECTED

Dataset: ABIDE I (Autism Brain Imaging Data Exchange)
    http://fcon_1000.projects.nitrc.org/indi/abide/
    Preprocessed: http://preprocessed-connectomes-project.org/abide/

Analysis: resting-state fMRI 1/f spectral slope across Harvard-Oxford
ROI time series, comparing ASD and TD groups. The expected result is
uninformative for TGC because resting-state fMRI is not a task-evoked
gain-dynamics measurement.

Environment variables:
    ABIDE_DATA_DIR   - path to downloaded ABIDE ROI time series
    ABIDE_PHENOTYPE  - path to phenotypic CSV
    TGC_FIGURES_DIR  - output directory for figures and CSV files

Usage:
    python analysis/study2_abide.py
"""

import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal, stats
from statsmodels.stats.multitest import multipletests


DATA_DIR = os.environ.get("ABIDE_DATA_DIR", "./abide_data")
PHENOTYPE_FILE = os.environ.get(
    "ABIDE_PHENOTYPE", "Phenotypic_V1_0b_preprocessed1.csv"
)
FIGURES_DIR = os.environ.get("TGC_FIGURES_DIR", "./figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def _subject_id_from_filename(path):
    match = re.search(r"\d{5,7}", os.path.basename(path))
    return int(match.group()) if match else None


def _load_phenotype(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Phenotype file not found. Set ABIDE_PHENOTYPE to the ABIDE "
            "Phenotypic_V1_0b_preprocessed1.csv file."
        )
    pheno = pd.read_csv(path)
    required = {"SUB_ID", "DX_GROUP"}
    missing = required - set(pheno.columns)
    if missing:
        raise ValueError(f"Phenotype file is missing columns: {sorted(missing)}")
    return dict(zip(pheno["SUB_ID"], pheno["DX_GROUP"]))


def _estimate_slope(time_series, tr_seconds=2.0):
    if np.all(time_series == 0) or np.var(time_series) == 0:
        return None

    fs = 1.0 / tr_seconds
    nperseg = max(8, len(time_series) // 2)
    freqs, power = signal.welch(time_series, fs=fs, nperseg=nperseg)
    valid = (freqs > 0.01) & (freqs < 0.1) & (power > 0)
    if valid.sum() < 3:
        return None

    slope, _, _, _, _ = stats.linregress(
        np.log10(freqs[valid]), np.log10(power[valid])
    )
    return float(slope) if np.isfinite(slope) else None


def collect_roi_slopes():
    pheno_dict = _load_phenotype(PHENOTYPE_FILE)
    files = glob.glob(os.path.join(DATA_DIR, "**", "*_rois_ho.1D"), recursive=True)
    if not files:
        raise FileNotFoundError(
            "No ABIDE ROI files found. Set ABIDE_DATA_DIR to the directory "
            "containing *_rois_ho.1D files."
        )

    records = []
    for path in sorted(files):
        sub_id = _subject_id_from_filename(path)
        if sub_id is None or sub_id not in pheno_dict:
            continue

        try:
            data = np.loadtxt(path)
        except Exception as exc:
            print(f"Skipping unreadable file: {path} ({exc})")
            continue

        if data.ndim != 2:
            continue

        group = "ASD" if pheno_dict[sub_id] == 1 else "TD"
        for roi in range(data.shape[1]):
            slope = _estimate_slope(data[:, roi])
            if slope is not None:
                records.append(
                    {"SUB_ID": sub_id, "Group": group, "ROI": roi, "Slope": slope}
                )

    if not records:
        raise RuntimeError("No comparable ROI slope estimates were produced.")
    return pd.DataFrame(records)


def test_rois(slopes):
    rows = []
    for roi, df_roi in slopes.groupby("ROI"):
        asd = df_roi.loc[df_roi["Group"] == "ASD", "Slope"]
        td = df_roi.loc[df_roi["Group"] == "TD", "Slope"]
        if len(asd) <= 5 or len(td) <= 5:
            continue
        t_stat, p_value = stats.ttest_ind(
            asd, td, equal_var=False, nan_policy="omit"
        )
        rows.append(
            {
                "ROI": roi,
                "n_ASD": len(asd),
                "n_TD": len(td),
                "ASD_mean": asd.mean(),
                "TD_mean": td.mean(),
                "t": t_stat,
                "p": p_value,
            }
        )

    if not rows:
        raise RuntimeError("No ROI had enough ASD and TD observations for testing.")

    results = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    _, q_values, _, _ = multipletests(results["p"], alpha=0.05, method="fdr_bh")
    results["q_fdr_bh"] = q_values
    results["significant_fdr_05"] = results["q_fdr_bh"] < 0.05
    return results


def plot_best_roi(slopes, results):
    best = results.iloc[0]
    best_roi = int(best["ROI"])
    best_df = slopes[slopes["ROI"] == best_roi]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(
        x="Group",
        y="Slope",
        data=best_df,
        palette=["#e74c3c", "#3498db"],
        inner="quartile",
        ax=ax,
    )
    ax.set_title(
        f"ABIDE 1/f Slope in Most Divergent ROI #{best_roi}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Diagnostic Group")
    ax.set_ylabel("1/f Spectral Slope")
    ax.text(
        0.5,
        0.95,
        f"p = {best['p']:.4g}, FDR q = {best['q_fdr_bh']:.4g}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
    )
    fig.tight_layout()

    out = os.path.join(FIGURES_DIR, "tgc_abide_study2_best_roi.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    return out


def main():
    print("--- Study 2: ABIDE resting-state fMRI 1/f slope ---")
    slopes = collect_roi_slopes()
    results = test_rois(slopes)

    csv_path = os.path.join(FIGURES_DIR, "abide_roi_slope_fdr_results.csv")
    results.to_csv(csv_path, index=False)
    fig_path = plot_best_roi(slopes, results)

    best = results.iloc[0]
    n_sig = int(results["significant_fdr_05"].sum())
    print(f"Saved ROI results: {csv_path}")
    print(f"Saved best-ROI figure: {fig_path}")
    print(
        f"Best ROI #{int(best['ROI'])}: p={best['p']:.4g}, "
        f"FDR q={best['q_fdr_bh']:.4g}"
    )
    print(f"FDR-significant ROIs at q<0.05: {n_sig}/{len(results)}")
    if n_sig == 0:
        print("Result: null after FDR correction, as expected for this boundary check.")


if __name__ == "__main__":
    main()
