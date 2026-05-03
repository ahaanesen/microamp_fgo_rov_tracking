from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

mpl.rcParams["axes.grid"] = True
mpl.rcParams["legend.loc"] = "lower right"
mpl.rcParams["legend.fontsize"] = "small"



def _load_csv(path):
    return pd.read_csv(path)


def _gt_time(df):
    if "t_ros_ns" in df.columns:
        return df["t_ros_ns"].to_numpy(dtype=float) * 1e-9
    if "t_ros_sec" in df.columns:
        return df["t_ros_sec"].to_numpy(dtype=float)
    raise KeyError("Ground-truth CSV must contain either 't_ros_ns' or 't_ros_sec'.")


def _fix_est_time(est_t, gt_t):
    """
    Some estimate files store epoch seconds with a 1e-9 scaling error,
    so values around 1.7 appear instead of 1.7e9. Detect and fix that.
    """
    est_t = np.asarray(est_t, dtype=float)
    gt_t = np.asarray(gt_t, dtype=float)

    if len(est_t) == 0 or len(gt_t) == 0:
        return est_t

    gt_scale = np.nanmedian(np.abs(gt_t))
    est_scale = np.nanmedian(np.abs(est_t))

    if gt_scale > 1e8 and est_scale < 1e3:
        est_t = est_t * 1e9

    return est_t


def _xyz(df, x, y, z):
    return np.stack([df[x], df[y], df[z]], axis=1).astype(float)


def _interp(src_t, src_xyz, tgt_t):
    """
    Interpolate 3D positions from src_t onto tgt_t.
    Samples outside the valid source time range are left as NaN.
    """
    src_t = np.asarray(src_t, dtype=float)
    tgt_t = np.asarray(tgt_t, dtype=float)
    src_xyz = np.asarray(src_xyz, dtype=float)

    if len(src_t) == 0:
        raise ValueError("Source time array is empty.")
    if len(src_t) != len(src_xyz):
        raise ValueError("src_t and src_xyz must have the same length.")
    if src_xyz.shape[1] != 3:
        raise ValueError("src_xyz must have shape (N, 3).")

    order = np.argsort(src_t)
    src_t = src_t[order]
    src_xyz = src_xyz[order]

    # Remove duplicate timestamps to keep np.interp well-defined.
    uniq_t, uniq_idx = np.unique(src_t, return_index=True)
    src_t = uniq_t
    src_xyz = src_xyz[uniq_idx]

    t_min, t_max = src_t[0], src_t[-1]
    mask = (tgt_t >= t_min) & (tgt_t <= t_max)

    out = np.full((len(tgt_t), 3), np.nan, dtype=float)
    for i in range(3):
        out[mask, i] = np.interp(tgt_t[mask], src_t, src_xyz[:, i])
    return out


def _valid_rows(a, b):
    return np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)


def _position_error(gt_xyz, est_xyz):
    """
    Returns:
        dist: Euclidean position error per valid timestep, shape (M,)
        err: component-wise error per valid timestep, shape (M, 3)
        valid: boolean mask over original rows, shape (N,)
    """
    valid = _valid_rows(gt_xyz, est_xyz)
    err = est_xyz[valid] - gt_xyz[valid]
    dist = np.linalg.norm(err, axis=1)
    return dist, err, valid


def _path_length(xyz: np.ndarray) -> float:
    if len(xyz) < 2:
        return 0.0
    diffs = np.diff(xyz, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _error_statistics(gt_xyz, est_xyz):
    valid = _valid_rows(gt_xyz, est_xyz)
    gt_xyz = gt_xyz[valid]
    est_xyz = est_xyz[valid]

    if len(gt_xyz) == 0:
        return {
            "num_valid_samples": 0,
            "mean_error": np.nan,
            "median_error": np.nan,
            "p95_error": np.nan,
            "max_error": np.nan,
            "final_error": np.nan,
            "ate_rms": np.nan,
            "mean_abs_n": np.nan,
            "mean_abs_e": np.nan,
            "mean_abs_d": np.nan,
            "std_n": np.nan,
            "std_e": np.nan,
            "std_d": np.nan,
            "path_length_gt": np.nan,
            "path_length_est": np.nan,
            "path_length_error_pct": np.nan,
        }

    err = est_xyz - gt_xyz
    pos_err = np.linalg.norm(err, axis=1)

    path_length_gt = _path_length(gt_xyz)
    path_length_est = _path_length(est_xyz)

    if path_length_gt > 0:
        path_length_error_pct = 100.0 * abs(path_length_est - path_length_gt) / path_length_gt
    else:
        path_length_error_pct = np.nan

    return {
        "num_valid_samples": int(len(gt_xyz)),
        "mean_error": float(np.mean(pos_err)),
        "median_error": float(np.median(pos_err)),
        "p95_error": float(np.percentile(pos_err, 95)),
        "max_error": float(np.max(pos_err)),
        "final_error": float(pos_err[-1]),
        "ate_rms": float(np.sqrt(np.mean(pos_err**2))),
        "mean_abs_n": float(np.mean(np.abs(err[:, 0]))),
        "mean_abs_e": float(np.mean(np.abs(err[:, 1]))),
        "mean_abs_d": float(np.mean(np.abs(err[:, 2]))),
        "std_n": float(np.std(err[:, 0])),
        "std_e": float(np.std(err[:, 1])),
        "std_d": float(np.std(err[:, 2])),
        "path_length_gt": float(path_length_gt),
        "path_length_est": float(path_length_est),
        "path_length_error_pct": float(path_length_error_pct),
    }


@dataclass
class PlotterCSVJoint:
    rov_gt_csv: str
    asv_gt_csv: str
    rov_est_csv: str
    asv_est_csv: str

    scenario_name: str = "Joint scenario"
    save_dir: str | None = None

    def _load(self):
        rov_gt_df = _load_csv(self.rov_gt_csv)
        asv_gt_df = _load_csv(self.asv_gt_csv)
        rov_est_df = _load_csv(self.rov_est_csv)
        asv_est_df = _load_csv(self.asv_est_csv)

        if "rov_id" in rov_gt_df.columns and "rov_id" in rov_est_df.columns:
            rov_id = int(rov_est_df["rov_id"].mode().iloc[0])
            rov_gt_df = rov_gt_df[rov_gt_df["rov_id"] == rov_id].copy()
            rov_est_df = rov_est_df[rov_est_df["rov_id"] == rov_id].copy()

        self.rov_gt_t = _gt_time(rov_gt_df)
        self.asv_gt_t = _gt_time(asv_gt_df)

        if "time" not in rov_est_df.columns or "time" not in asv_est_df.columns:
            raise KeyError("Estimated CSVs must contain a 'time' column.")

        self.rov_est_t = _fix_est_time(rov_est_df["time"].to_numpy(dtype=float), self.rov_gt_t)
        self.asv_est_t = _fix_est_time(asv_est_df["time"].to_numpy(dtype=float), self.asv_gt_t)

        self.rov_gt = _xyz(rov_gt_df, "x_n", "y_e", "z_d")
        self.asv_gt = _xyz(asv_gt_df, "x_n", "y_e", "z_d")

        self.rov_est = _xyz(rov_est_df, "x", "y", "z")
        self.asv_est = _xyz(asv_est_df, "x", "y", "z")

        # Store full relative times for trajectory plots.
        self.rov_gt_t_rel = self.rov_gt_t - self.rov_gt_t[0]
        self.asv_gt_t_rel = self.asv_gt_t - self.asv_gt_t[0]
        self.rov_est_t_rel = self.rov_est_t - self.rov_est_t[0]
        self.asv_est_t_rel = self.asv_est_t - self.asv_est_t[0]

        # Create overlap-aware time bases so the GT-only initialization edges are ignored.
        rov_mask = (self.rov_gt_t >= self.rov_est_t.min()) & (self.rov_gt_t <= self.rov_est_t.max())
        asv_mask = (self.asv_gt_t >= self.asv_est_t.min()) & (self.asv_gt_t <= self.asv_est_t.max())

        self.rov_gt_eval_t = self.rov_gt_t[rov_mask]
        self.asv_gt_eval_t = self.asv_gt_t[asv_mask]
        self.rov_gt_eval = self.rov_gt[rov_mask]
        self.asv_gt_eval = self.asv_gt[asv_mask]

        self.rov_gt_eval_t_rel = self.rov_gt_eval_t - self.rov_gt_eval_t[0]
        self.asv_gt_eval_t_rel = self.asv_gt_eval_t - self.asv_gt_eval_t[0]

        self.rov_est_i = _interp(self.rov_est_t, self.rov_est, self.rov_gt_eval_t)
        self.asv_est_i = _interp(self.asv_est_t, self.asv_est, self.asv_gt_eval_t)

    def plot3d(self):
        self._load()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            *self.rov_gt.T,
            label="ROV ground truth",
            linestyle="--",
            color="C1",
            alpha=0.8,
        )
        ax.scatter(*self.rov_gt[0], marker="x", color="red", s=60)

        ax.plot(
            *self.rov_est.T,
            label="ROV estimate",
            color="C0",
            alpha=0.85,
        )

        ax.plot(
            *self.asv_gt.T,
            label="ASV ground truth",
            linestyle="-.",
            color="C2",
            alpha=0.7,
        )
        ax.scatter(*self.asv_gt[0], marker="^", color="C2", s=60)

        ax.plot(
            *self.asv_est.T,
            label="ASV estimate",
            color="C3",
            alpha=0.85,
        )

        ax.set_xlabel("North [m]", labelpad=10)
        ax.set_ylabel("East [m]", labelpad=10)
        ax.set_zlabel("Down [m]", labelpad=10)

        ax.invert_zaxis()
        ax.view_init(elev=15, azim=-110)
        ax.set_box_aspect(None)

        ax.xaxis.pane.set_edgecolor("black")
        ax.yaxis.pane.set_edgecolor("black")
        ax.zaxis.pane.set_edgecolor("black")

        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)

        ax.grid(True)
        ax.set_title(f"{self.scenario_name} - 3D Trajectories")
        ax.legend(loc="upper right")

        fig.tight_layout()
        return fig

    def plot_position_error(self):
        self._load()

        rov_err_norm, _, rov_valid = _position_error(self.rov_gt_eval, self.rov_est_i)
        asv_err_norm, _, asv_valid = _position_error(self.asv_gt_eval, self.asv_est_i)

        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

        axs[0].plot(
            self.rov_gt_eval_t_rel[rov_valid],
            rov_err_norm,
            label="ROV position error",
            color="C0",
        )
        axs[0].set_ylabel("Error [m]")
        axs[0].set_title("ROV Position Error")
        axs[0].grid(True)
        axs[0].legend()

        axs[1].plot(
            self.asv_gt_eval_t_rel[asv_valid],
            asv_err_norm,
            label="ASV position error",
            color="C3",
        )
        axs[1].set_xlabel("Time [s]")
        axs[1].set_ylabel("Error [m]")
        axs[1].set_title("ASV Position Error")
        axs[1].grid(True)
        axs[1].legend()

        fig.suptitle(f"{self.scenario_name} - Position Errors")
        fig.tight_layout()
        return fig

    def plot_position_error_components(self, platform="ROV"):
        self._load()

        platform = platform.upper()
        if platform == "ROV":
            gt_t = self.rov_gt_eval_t_rel
            gt = self.rov_gt_eval
            est_i = self.rov_est_i
        elif platform == "ASV":
            gt_t = self.asv_gt_eval_t_rel
            gt = self.asv_gt_eval
            est_i = self.asv_est_i
        else:
            raise ValueError("platform must be either 'ROV' or 'ASV'")

        valid = _valid_rows(gt, est_i)

        t = gt_t[valid]
        err = est_i[valid] - gt[valid]

        labels = ["North", "East", "Down"]
        colors = ["C0", "C1", "C2"]

        fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

        for i in range(3):
            axs[i].plot(t, err[:, i], color=colors[i])
            axs[i].set_ylabel(f"{labels[i]} error [m]")
            axs[i].grid(True)

        axs[-1].set_xlabel("Time [s]")
        fig.suptitle(f"{self.scenario_name} - {platform} Position Error Components")
        fig.tight_layout()
        return fig

    def export_statistics(self):
        self._load()

        rows = []

        rov_stats = _error_statistics(self.rov_gt_eval, self.rov_est_i)
        rov_stats.update({"scenario": self.scenario_name, "platform": "ROV"})
        rows.append(rov_stats)

        asv_stats = _error_statistics(self.asv_gt_eval, self.asv_est_i)
        asv_stats.update({"scenario": self.scenario_name, "platform": "ASV"})
        rows.append(asv_stats)

        return pd.DataFrame(rows)

    def save_all(self):
        fig1 = self.plot3d()
        fig2 = self.plot_position_error()
        fig3 = self.plot_position_error_components("ROV")
        fig4 = self.plot_position_error_components("ASV")
        stat_csv = self.export_statistics()

        if not self.save_dir:
            raise ValueError("save_dir must be set to save plots and statistics.")

        path = Path(self.save_dir)
        path.mkdir(parents=True, exist_ok=True)

        fig1.savefig(path / "traj_3d.png", dpi=150, bbox_inches="tight")
        fig2.savefig(path / "position_error.png", dpi=150, bbox_inches="tight")
        fig3.savefig(path / "rov_position_error_components.png", dpi=150, bbox_inches="tight")
        fig4.savefig(path / "asv_position_error_components.png", dpi=150, bbox_inches="tight")
        stat_csv.to_csv(path / "fgo_statistics.csv", index=False)

        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)
        return path, stat_csv

    def show(self):
        if self.save_dir:
            self.save_all()

        plt.show(block=True)


if __name__ == "__main__":
    plotter1 = PlotterCSVJoint(
        rov_gt_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/simulation_data/8a/rov_ground_truth.csv",
        asv_gt_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/simulation_data/8a/asv_ground_truth.csv",
        rov_est_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/debugging/test/rov_estimated_my_experiment_01.csv",
        asv_est_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/debugging/test/boat_estimated_my_experiment_01.csv",
        scenario_name="FGO - Scenario 1: Bearing-only",
        save_dir="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/debugging/test/plots_s1",
    )
    plotter1.show()

    # plotter2 = PlotterCSVJoint(
    #     rov_gt_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/simulation_data/circle2/rov_ground_truth.csv",
    #     asv_gt_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/simulation_data/circle2/asv_ground_truth.csv",
    #     rov_est_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/estimated_data/0_02cv_sigma/rov_estimated_s2.csv",
    #     asv_est_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/estimated_data/0_02cv_sigma/boat_estimated_s2.csv",
    #     scenario_name="FGO - Scenario 2: Bearing + range",
    #     save_dir="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/plots/scenario2",
    # )
    # plotter2.show()

    # plotter3 = PlotterCSVJoint(
    #     rov_gt_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/simulation_data/circle2/rov_ground_truth.csv",
    #     asv_gt_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/simulation_data/circle2/asv_ground_truth.csv",
    #     rov_est_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/estimated_data/0_02cv_sigma/rov_estimated_s3.csv",
    #     asv_est_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/estimated_data/0_02cv_sigma/boat_estimated_s3.csv",
    #     scenario_name="FGO - Scenario 3: Bearing + range + depth",
    #     save_dir="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/plots/scenario3",
    # )
    # plotter3.show()
