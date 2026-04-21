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
    return df["t_ros_sec"].to_numpy(dtype=float)


def _xyz(df, x, y, z):
    return np.stack([df[x], df[y], df[z]], axis=1).astype(float)


def _interp(src_t, src_xyz, tgt_t):
    # Clip target times to the valid range of the source
    t_min, t_max = src_t[0], src_t[-1]
    mask = (tgt_t >= t_min) & (tgt_t <= t_max)
    out = np.full((len(tgt_t), 3), np.nan)
    for i in range(3):
        out[mask, i] = np.interp(tgt_t[mask], src_t, src_xyz[:, i])
    return out

# def _rmse(gt_xyz, est_xyz):
#     err = est_xyz - gt_xyz
#     return np.sqrt(np.mean(err**2, axis=1)), err

def _position_error(gt_xyz, est_xyz):
    err = est_xyz - gt_xyz
    dist = np.linalg.norm(err, axis=1)   # true 3D Euclidean error per timestep
    return dist, err

def _path_length(xyz: np.ndarray) -> float:
    diffs = np.diff(xyz, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _error_statistics(gt_xyz, est_xyz):
    err = est_xyz - gt_xyz
    rmse_t = np.sqrt(np.mean(err**2, axis=1))

    stats = {
        "mean_rmse": float(np.mean(rmse_t)),
        "median_rmse": float(np.median(rmse_t)),
        "p95_rmse": float(np.percentile(rmse_t, 95)),
        "max_rmse": float(np.max(rmse_t)),
        "final_rmse": float(rmse_t[-1]),
        "ate_rms": float(np.sqrt(np.mean(rmse_t**2))),
        "mean_abs_n": float(np.mean(np.abs(err[:, 0]))),
        "mean_abs_e": float(np.mean(np.abs(err[:, 1]))),
        "mean_abs_d": float(np.mean(np.abs(err[:, 2]))),
        "std_n": float(np.std(err[:, 0])),
        "std_e": float(np.std(err[:, 1])),
        "std_d": float(np.std(err[:, 2])),
    }

    stats["path_length_gt"] = _path_length(gt_xyz)
    stats["path_length_est"] = _path_length(est_xyz)
    stats["path_length_error_pct"] = (
        100.0
        * abs(stats["path_length_est"] - stats["path_length_gt"])
        / stats["path_length_gt"]
    )

    return stats


@dataclass
class PlotterCSVJoint:
    rov_gt_csv: str
    asv_gt_csv: str
    rov_est_csv: str
    asv_est_csv: str

    scenario_name: str = "Joint scenario"
    save_dir: str | None = None

    def _load(self):
        # Load CSVs
        rov_gt_df = _load_csv(self.rov_gt_csv)
        asv_gt_df = _load_csv(self.asv_gt_csv)
        rov_est_df = _load_csv(self.rov_est_csv)
        asv_est_df = _load_csv(self.asv_est_csv)

        # Extract times
        self.rov_gt_t = _gt_time(rov_gt_df)
        self.asv_gt_t = _gt_time(asv_gt_df)
        self.rov_est_t = rov_est_df["time"].to_numpy()
        self.asv_est_t = asv_est_df["time"].to_numpy()

        # Extract positions (NED)
        self.rov_gt = _xyz(rov_gt_df, "x_n", "y_e", "z_d")
        self.asv_gt = _xyz(asv_gt_df, "x_n", "y_e", "z_d")

        self.rov_est = _xyz(rov_est_df, "x", "y", "z")
        self.asv_est = _xyz(asv_est_df, "x", "y", "z")

        # Align by relative time (offset each to start at zero)
        # Should fix this when creating the CSVs
        self.rov_gt_t -= self.rov_gt_t[0]
        self.asv_gt_t -= self.asv_gt_t[0]
        self.rov_est_t -= self.rov_est_t[0]
        self.asv_est_t -= self.asv_est_t[0]

    def plot3d(self):
        self._load()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # ------------------------
        # ROV
        # ------------------------
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

        # ------------------------
        # ASV
        # ------------------------
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

        # ------------------------
        # Styling (same as yours)
        # ------------------------
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

        ax.set_title(f"FGO: {self.scenario_name} — 3D Trajectories")
        ax.legend(loc="upper right")

        fig.tight_layout()
        return fig

    def plot_rmse(self):
        self._load()

        # Interpolate estimates to GT timelines
        rov_est_i = _interp(self.rov_est_t, self.rov_est, self.rov_gt_t)
        asv_est_i = _interp(self.asv_est_t, self.asv_est, self.asv_gt_t)

        rov_rmse, rov_err = _position_error(self.rov_gt, rov_est_i)
        asv_rmse, asv_err = _position_error(self.asv_gt, asv_est_i)

        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

        # ------------------------
        # ROV RMSE
        # ------------------------
        axs[0].plot(self.rov_gt_t, rov_rmse, label="ROV position RMSE", color="C0")
        axs[0].set_ylabel("RMSE [m]")
        axs[0].set_title("ROV Position RMSE")
        axs[0].grid(True)
        axs[0].legend()

        # ------------------------
        # ASV RMSE
        # ------------------------
        axs[1].plot(self.asv_gt_t, asv_rmse, label="ASV position RMSE", color="C3")
        axs[1].set_xlabel("Time [s]")
        axs[1].set_ylabel("RMSE [m]")
        axs[1].set_title("ASV Position RMSE")
        axs[1].grid(True)
        axs[1].legend()

        fig.suptitle(f"FGO RMSE — {self.scenario_name}")
        fig.tight_layout()
        return fig
    
    def plot_position_error_components(self):
        self._load()

        rov_est_i = _interp(self.rov_est_t, self.rov_est, self.rov_gt_t)
        err = rov_est_i - self.rov_gt

        labels = ["North", "East", "Down"]
        colors = ["C0", "C1", "C2"]

        fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

        for i in range(3):
            axs[i].plot(self.rov_gt_t, err[:, i], color=colors[i])
            axs[i].set_ylabel(f"{labels[i]} error [m]")
            axs[i].grid(True)

        axs[-1].set_xlabel("Time [s]")
        fig.suptitle(f"ROV Position Error Components — {self.scenario_name}")
        fig.tight_layout()
        return fig

    def export_statistics(self):
        self._load()

        rov_est_i = _interp(self.rov_est_t, self.rov_est, self.rov_gt_t)
        asv_est_i = _interp(self.asv_est_t, self.asv_est, self.asv_gt_t)

        rows = []

        rov_stats = _error_statistics(self.rov_gt, rov_est_i)
        rov_stats.update(
            {"scenario": self.scenario_name, "platform": "ROV"}
        )
        rows.append(rov_stats)

        asv_stats = _error_statistics(self.asv_gt, asv_est_i)
        asv_stats.update(
            {"scenario": self.scenario_name, "platform": "ASV"}
        )
        rows.append(asv_stats)

        df = pd.DataFrame(rows)

        # if self.save_dir:
        #     path = Path(self.save_dir)
        #     path.mkdir(parents=True, exist_ok=True)
        #     df.to_csv(path / "statistics.csv", index=False)

        return df

    def show(self):
        fig1 = self.plot3d()
        fig2 = self.plot_rmse()
        stat_csv = self.export_statistics()

        if self.save_dir:
            path = Path(self.save_dir)
            path.mkdir(parents=True, exist_ok=True)
            fig1.savefig(path / "traj_3d.png", dpi=150, bbox_inches="tight")
            fig2.savefig(path / "rmse.png", dpi=150, bbox_inches="tight")
            stat_csv.to_csv(path / "statistics.csv", index=False)

        plt.show(block=True)

plotter1 = PlotterCSVJoint(
    rov_gt_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/simulation_data/circle2/rov_ground_truth.csv",
    asv_gt_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/simulation_data/circle2/asv_ground_truth.csv",
    rov_est_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/estimated_data/0_02cv_sigma/rov_estimated_s1.csv",
    asv_est_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/estimated_data/0_02cv_sigma/boat_estimated_s1.csv",
    scenario_name="Scenario 1: Bearing-only",
    save_dir="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/plots/scenario1",
)
plotter1.show()

plotter2 = PlotterCSVJoint(
    rov_gt_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/simulation_data/circle2/rov_ground_truth.csv",
    asv_gt_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/simulation_data/circle2/asv_ground_truth.csv",
    rov_est_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/estimated_data/0_02cv_sigma/rov_estimated_s2.csv",
    asv_est_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/estimated_data/0_02cv_sigma/boat_estimated_s2.csv",
    scenario_name="Scenario 2: Bearing + range",
    save_dir="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/plots/scenario2",
)
plotter2.show()

plotter3 = PlotterCSVJoint(
    rov_gt_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/simulation_data/circle2/rov_ground_truth.csv",
    asv_gt_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/simulation_data/circle2/asv_ground_truth.csv",
    rov_est_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/estimated_data/0_02cv_sigma/rov_estimated_s3.csv",
    asv_est_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/estimated_data/0_02cv_sigma/boat_estimated_s3.csv",
    scenario_name="Scenario 3: Bearing + range + depth",
    save_dir="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/plots/scenario3",
)
plotter3.show()