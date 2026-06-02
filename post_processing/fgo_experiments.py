# fgo_experiments.py
#
# Mirrors eskf_experiments.py but uses PlotterCSVJoint from the FGO pipeline.
#
# Assumed directory structure:
#
# results/
#   exp1_noise/
#       sigma_0.005/
#           run_01/
#               rov_est.csv
#               asv_est.csv
#               ...
#
#   exp2_scaling/
#       alpha_0.25/
#           run_01/
#               ...
#
#   exp3_dropout/
#       dropout_0.10/
#           run_01/
#               ...
#
# Produces:
#   - aggregate statistics CSVs
#   - publication-ready comparison plots
#   - experiment summary plots
#
# ------------------------------------------------------------

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plotting import PlotterCSVJoint


# ============================================================
# Helpers
# ============================================================

def _find_file(directory: Path, pattern: str):
    matches = list(directory.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find '{pattern}' inside {directory}")
    return matches[0]


def _numeric_sort_key(text):
    m = re.search(r"([-+]?[0-9]*\.?[0-9]+)", str(text))
    if m:
        return float(m.group(1))
    return text


def _savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _aggregate_runs(df, parameter_name):
    grouped = (
        df.groupby(parameter_name)
        .agg(
            ate_rms_mean=("ate_rms", "mean"),
            ate_rms_std=("ate_rms", "std"),
            final_error_mean=("final_error", "mean"),
            final_error_std=("final_error", "std"),
            tanees_mean=("mean_tanees_pos", "mean"),
            tanees_std=("mean_tanees_pos", "std"),
        )
        .reset_index()
    )

    grouped = grouped.sort_values(parameter_name)
    return grouped


# ============================================================
# Experiment 1
# Measurement noise sensitivity
# ============================================================

def run_experiment_1_noise_sensitivity(
    root_dir,
    rov_gt_csv,
    asv_gt_csv,
    save_dir,
    gnss_csv=None,
):
    """
    Sweep over sigma values.

    Folder convention:
        sigma_0.005/
        sigma_0.01/
        ...
    """

    root_dir = Path(root_dir)
    save_dir = Path(save_dir)

    all_rows = []

    sigma_dirs = sorted(
        [p for p in root_dir.iterdir() if p.is_dir()],
        key=lambda p: _numeric_sort_key(p.name),
    )

    for sigma_dir in sigma_dirs:

        sigma = float(re.findall(r"[-+]?[0-9]*\.?[0-9]+", sigma_dir.name)[0])

        run_dirs = sorted([p for p in sigma_dir.iterdir() if p.is_dir()])

        for run_dir in run_dirs:

            try:
                plotter = PlotterCSVJoint(
                    rov_gt_csv=rov_gt_csv,
                    asv_gt_csv=asv_gt_csv,
                    rov_est_csv=_find_file(run_dir, "*rov*.csv"),
                    asv_est_csv=_find_file(run_dir, "*asv*.csv"),
                    gnss_csv=gnss_csv,
                    scenario_name=f"Sigma={sigma}",
                )

                stats_df = plotter.export_statistics()

                rov_stats = stats_df[stats_df["platform"] == "ROV"].iloc[0]

                row = rov_stats.to_dict()
                row["sigma"] = sigma
                row["run"] = run_dir.name

                all_rows.append(row)

            except Exception as exc:
                print(f"[WARN] Failed {run_dir}: {exc}")

    df = pd.DataFrame(all_rows)

    save_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = save_dir / "experiment1_raw.csv"
    df.to_csv(raw_csv, index=False)

    agg = _aggregate_runs(df, "sigma")

    agg_csv = save_dir / "experiment1_aggregate.csv"
    agg.to_csv(agg_csv, index=False)

    # --------------------------------------------------------
    # ATE RMS vs sigma
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.errorbar(
        agg["sigma"],
        agg["ate_rms_mean"],
        yerr=agg["ate_rms_std"],
        marker="o",
        capsize=4,
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"Measurement noise $\sigma$")
    ax.set_ylabel("ATE RMS [m]")
    ax.set_title("FGO Measurement Noise Sensitivity")

    _savefig(fig, save_dir / "exp1_ate_vs_sigma.png")

    # --------------------------------------------------------
    # ANEES vs sigma
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.errorbar(
        agg["sigma"],
        agg["tanees_mean"],
        yerr=agg["tanees_std"],
        marker="o",
        capsize=4,
    )

    ax.axhline(3.0, linestyle="--")

    ax.set_xscale("log")
    ax.set_xlabel(r"Measurement noise $\sigma$")
    ax.set_ylabel("ANEES")
    ax.set_title("FGO Consistency vs Measurement Noise")

    _savefig(fig, save_dir / "exp1_anees_vs_sigma.png")

    return df, agg


# ============================================================
# Experiment 2
# Robust cost scaling sweep
# ============================================================

def run_experiment_2_alpha_sweep(
    root_dir,
    rov_gt_csv,
    asv_gt_csv,
    save_dir,
    gnss_csv=None,
):
    """
    Sweep over robust loss scaling alpha.

    Folder convention:
        alpha_0.25/
        alpha_0.5/
        ...
    """

    root_dir = Path(root_dir)
    save_dir = Path(save_dir)

    all_rows = []

    alpha_dirs = sorted(
        [p for p in root_dir.iterdir() if p.is_dir()],
        key=lambda p: _numeric_sort_key(p.name),
    )

    for alpha_dir in alpha_dirs:

        alpha = float(re.findall(r"[-+]?[0-9]*\.?[0-9]+", alpha_dir.name)[0])

        run_dirs = sorted([p for p in alpha_dir.iterdir() if p.is_dir()])

        for run_dir in run_dirs:

            try:
                plotter = PlotterCSVJoint(
                    rov_gt_csv=rov_gt_csv,
                    asv_gt_csv=asv_gt_csv,
                    rov_est_csv=_find_file(run_dir, "*rov*.csv"),
                    asv_est_csv=_find_file(run_dir, "*asv*.csv"),
                    gnss_csv=gnss_csv,
                    scenario_name=f"Alpha={alpha}",
                )

                stats_df = plotter.export_statistics()

                rov_stats = stats_df[stats_df["platform"] == "ROV"].iloc[0]

                row = rov_stats.to_dict()
                row["alpha"] = alpha
                row["run"] = run_dir.name

                all_rows.append(row)

            except Exception as exc:
                print(f"[WARN] Failed {run_dir}: {exc}")

    df = pd.DataFrame(all_rows)

    save_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(save_dir / "experiment2_raw.csv", index=False)

    agg = _aggregate_runs(df, "alpha")

    agg.to_csv(save_dir / "experiment2_aggregate.csv", index=False)

    # --------------------------------------------------------
    # ATE RMS vs alpha
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.errorbar(
        agg["alpha"],
        agg["ate_rms_mean"],
        yerr=agg["ate_rms_std"],
        marker="o",
        capsize=4,
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"Robust scaling parameter $\alpha$")
    ax.set_ylabel("ATE RMS [m]")
    ax.set_title("FGO Robust Cost Scaling Sweep")

    _savefig(fig, save_dir / "exp2_ate_vs_alpha.png")

    # --------------------------------------------------------
    # Final error vs alpha
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.errorbar(
        agg["alpha"],
        agg["final_error_mean"],
        yerr=agg["final_error_std"],
        marker="o",
        capsize=4,
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"Robust scaling parameter $\alpha$")
    ax.set_ylabel("Final position error [m]")
    ax.set_title("Final Error vs Robust Scaling")

    _savefig(fig, save_dir / "exp2_final_vs_alpha.png")

    return df, agg


# ============================================================
# Experiment 3
# Measurement dropout robustness
# ============================================================

def run_experiment_3_dropout(
    root_dir,
    rov_gt_csv,
    asv_gt_csv,
    save_dir,
    gnss_csv=None,
):
    """
    Folder convention:
        dropout_0.00/
        dropout_0.10/
        ...
    """

    root_dir = Path(root_dir)
    save_dir = Path(save_dir)

    all_rows = []

    dropout_dirs = sorted(
        [p for p in root_dir.iterdir() if p.is_dir()],
        key=lambda p: _numeric_sort_key(p.name),
    )

    for dropout_dir in dropout_dirs:

        dropout = float(re.findall(r"[-+]?[0-9]*\.?[0-9]+", dropout_dir.name)[0])

        run_dirs = sorted([p for p in dropout_dir.iterdir() if p.is_dir()])

        for run_dir in run_dirs:

            try:
                plotter = PlotterCSVJoint(
                    rov_gt_csv=rov_gt_csv,
                    asv_gt_csv=asv_gt_csv,
                    rov_est_csv=_find_file(run_dir, "*rov*.csv"),
                    asv_est_csv=_find_file(run_dir, "*asv*.csv"),
                    gnss_csv=gnss_csv,
                    scenario_name=f"Dropout={dropout}",
                )

                stats_df = plotter.export_statistics()

                rov_stats = stats_df[stats_df["platform"] == "ROV"].iloc[0]

                row = rov_stats.to_dict()
                row["dropout"] = dropout
                row["run"] = run_dir.name

                all_rows.append(row)

            except Exception as exc:
                print(f"[WARN] Failed {run_dir}: {exc}")

    df = pd.DataFrame(all_rows)

    save_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(save_dir / "experiment3_raw.csv", index=False)

    agg = _aggregate_runs(df, "dropout")

    agg.to_csv(save_dir / "experiment3_aggregate.csv", index=False)

    # --------------------------------------------------------
    # ATE RMS vs dropout
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.errorbar(
        agg["dropout"],
        agg["ate_rms_mean"],
        yerr=agg["ate_rms_std"],
        marker="o",
        capsize=4,
    )

    ax.set_xlabel("Measurement dropout probability")
    ax.set_ylabel("ATE RMS [m]")
    ax.set_title("FGO Robustness to Measurement Dropout")

    _savefig(fig, save_dir / "exp3_ate_vs_dropout.png")

    # --------------------------------------------------------
    # ANEES vs dropout
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.errorbar(
        agg["dropout"],
        agg["tanees_mean"],
        yerr=agg["tanees_std"],
        marker="o",
        capsize=4,
    )

    ax.axhline(3.0, linestyle="--")

    ax.set_xlabel("Measurement dropout probability")
    ax.set_ylabel("ANEES")
    ax.set_title("Consistency under Measurement Dropout")

    _savefig(fig, save_dir / "exp3_anees_vs_dropout.png")

    return df, agg


# ============================================================
# Optional:
# ESKF vs FGO comparison utilities
# ============================================================

def compare_estimators(
    eskf_csv,
    fgo_csv,
    parameter_name,
    metric="ate_rms_mean",
    save_path=None,
):
    """
    Compare aggregate ESKF and FGO results.

    Inputs:
        eskf_csv: aggregate CSV from eskf_experiments.py
        fgo_csv : aggregate CSV from fgo_experiments.py
    """

    eskf = pd.read_csv(eskf_csv)
    fgo = pd.read_csv(fgo_csv)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        eskf[parameter_name],
        eskf[metric],
        marker="o",
        label="ESKF",
    )

    ax.plot(
        fgo[parameter_name],
        fgo[metric],
        marker="s",
        label="FGO",
    )

    if parameter_name in ["sigma", "alpha"]:
        ax.set_xscale("log")

    ax.set_xlabel(parameter_name)
    ax.set_ylabel(metric)
    ax.legend()

    fig.tight_layout()

    if save_path:
        _savefig(fig, Path(save_path))

    return fig