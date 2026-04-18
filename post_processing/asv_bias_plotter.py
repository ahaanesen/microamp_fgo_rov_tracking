#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sympy import true

from microampere_ros2ws.src.microamp_fgo_rov_tracking.post_processing.joint_3d_plotter import _load_csv


BIAS_COLUMN_CANDIDATES = {
    "gyro_x": ["gyro_bias_x", "bias_gyro_x", "bgx"],
    "gyro_y": ["gyro_bias_y", "bias_gyro_y", "bgy"],
    "gyro_z": ["gyro_bias_z", "bias_gyro_z", "bgz"],
    "acc_x": ["accel_bias_x", "bias_acc_x", "bax"],
    "acc_y": ["accel_bias_y", "bias_acc_y", "bay"],
    "acc_z": ["accel_bias_z", "bias_acc_z", "baz"],
}


def _resolve_bias_columns(df: pd.DataFrame) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for canonical, candidates in BIAS_COLUMN_CANDIDATES.items():
        for col in candidates:
            if col in df.columns:
                resolved[canonical] = col
                break
    return resolved


def plot_bias(csv_path: Path, output: Path | None = None, show: bool = True) -> None:
    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        raise ValueError(f"CSV '{csv_path}' must contain a 'time' column.")

    resolved = _resolve_bias_columns(df)
    if not resolved:
        supported = ", ".join(sum(BIAS_COLUMN_CANDIDATES.values(), []))
        raise ValueError(
            "No ASV bias columns found in CSV. "
            f"Expected one of: {supported}."
        )

    t = df["time"].to_numpy(dtype=float)
    t = t - t[0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    gyro_keys = ["gyro_x", "gyro_y", "gyro_z"]
    accel_keys = ["acc_x", "acc_y", "acc_z"]

    for k in gyro_keys:
        if k in resolved:
            axes[0].plot(t, df[resolved[k]].to_numpy(dtype=float), label=resolved[k])
    axes[0].set_title("Estimated gyroscope bias")
    axes[0].set_ylabel("bias [rad/s]")
    axes[0].grid(True)
    axes[0].legend()

    for k in accel_keys:
        if k in resolved:
            axes[1].plot(t, df[resolved[k]].to_numpy(dtype=float), label=resolved[k])
    axes[1].set_title("Estimated accelerometer bias")
    axes[1].set_ylabel("bias [m/s²]")
    axes[1].set_xlabel("time since start [s]")
    axes[1].grid(True)
    if any(k in resolved for k in accel_keys):
        axes[1].legend()

    fig.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")

    if show:
        plt.show(block=True)


def main() -> None:
    rov_gt_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/simulation_data/rov_ground_truth.csv",
    asv_gt_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/simulation_data/asv_ground_truth.csv",
    rov_est_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/estimated_data/scenario3/rov_estimated_20260418_131139.csv",
    asv_est_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/estimated_data/scenario3/boat_estimated_20260418_131139.csv",
    save_dir="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/plots/scenario3",
    save_path = Path(save_dir)
    asv_est_df = _load_csv(asv_est_csv)

    plot_bias(Path(asv_est_csv), save_path, true)


if __name__ == "__main__":
    main()
