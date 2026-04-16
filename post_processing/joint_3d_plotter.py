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
    out = np.empty((len(tgt_t), 3))
    for i in range(3):
        out[:, i] = np.interp(tgt_t, src_t, src_xyz[:, i])
    return out


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

    def show(self):
        fig = self.plot3d()

        if self.save_dir:
            path = Path(self.save_dir)
            path.mkdir(parents=True, exist_ok=True)
            fig.savefig(path / "3d_joint_new.png", dpi=150, bbox_inches="tight")

        plt.show(block=True)

plotter = PlotterCSVJoint(
    rov_gt_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/simulation_data/rov_ground_truth.csv",
    asv_gt_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/simulation_data/asv_ground_truth.csv",
    rov_est_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/estimated_data/scenario3/rov_estimated_20260416_090058.csv",
    asv_est_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/estimated_data/scenario3/boat_estimated_20260416_090058.csv",
    scenario_name="Scenario 3",
    save_dir="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/plots/scenario3",
)
plotter.show()