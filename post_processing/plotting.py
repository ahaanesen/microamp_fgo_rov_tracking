from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

mpl.rcParams["axes.grid"] = True
mpl.rcParams["legend.loc"] = "lower right"
mpl.rcParams["legend.fontsize"] = "small"

# Chi-squared 95 % confidence bounds for 3 DOF (used for NEES and NIS)
try:
    from scipy.stats import chi2 as _chi2
    _CHI2_3_LO = float(_chi2.ppf(0.025, 3))
    _CHI2_3_HI = float(_chi2.ppf(0.975, 3))
except ImportError:
    _CHI2_3_LO = 0.352   # chi2(3, 0.025)
    _CHI2_3_HI = 9.348   # chi2(3, 0.975)


# ============================================================
# CSV / array helpers
# ============================================================

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


def _interp_cov6(src_t, src_cov6, tgt_t):
    """
    Interpolate the 6 upper-triangle elements [xx, xy, xz, yy, yz, zz]
    of a symmetric 3×3 covariance matrix onto tgt_t.
    Samples outside the source range are left as NaN.
    """
    src_t    = np.asarray(src_t,    dtype=float)
    src_cov6 = np.asarray(src_cov6, dtype=float)
    tgt_t    = np.asarray(tgt_t,    dtype=float)

    order = np.argsort(src_t)
    src_t    = src_t[order]
    src_cov6 = src_cov6[order]

    uniq_t, uniq_idx = np.unique(src_t, return_index=True)
    src_cov6 = src_cov6[uniq_idx]

    t_min, t_max = uniq_t[0], uniq_t[-1]
    mask = (tgt_t >= t_min) & (tgt_t <= t_max)

    out = np.full((len(tgt_t), 6), np.nan, dtype=float)
    for i in range(6):
        out[mask, i] = np.interp(tgt_t[mask], uniq_t, src_cov6[:, i])
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


def _cov6_has_data(cov6) -> bool:
    """True only if the covariance array exists and contains non-trivial values."""
    if cov6 is None:
        return False
    finite = cov6[np.isfinite(cov6)]
    return bool(finite.size > 0 and np.any(np.abs(finite) > 1e-12))


# ============================================================
# NEES / NIS
# ============================================================

def _compute_nees(errors: np.ndarray, cov6: np.ndarray) -> np.ndarray:
    """
    Compute per-timestep NEES = err^T P^{-1} err.

    Parameters
    ----------
    errors : (N, 3)  estimation errors (est − gt)
    cov6   : (N, 6)  upper-triangle of 3×3 position covariance
                     columns: [xx, xy, xz, yy, yz, zz]

    Returns
    -------
    nees : (N,) — NaN for non-finite or singular rows.
    """
    valid = np.isfinite(errors).all(axis=1) & np.isfinite(cov6).all(axis=1)
    nees  = np.full(len(errors), np.nan)
    if not valid.any():
        return nees

    ev = errors[valid]
    cv = cov6[valid]

    nees_valid = np.full(len(ev), np.nan)
    for k in range(len(ev)):
        P = np.array([
            [cv[k, 0], cv[k, 1], cv[k, 2]],
            [cv[k, 1], cv[k, 3], cv[k, 4]],
            [cv[k, 2], cv[k, 4], cv[k, 5]],
        ])
        try:
            nees_valid[k] = ev[k] @ np.linalg.inv(P) @ ev[k]
        except np.linalg.LinAlgError:
            pass

    nees[valid] = nees_valid
    return nees


def _compute_nis(innovations: np.ndarray, cov6: np.ndarray,
                 R_diag: np.ndarray) -> np.ndarray:
    """
    Compute per-measurement NIS = ν^T S^{-1} ν   where   S = P_pos + R.

    Parameters
    ----------
    innovations : (N, 3)  z − H x   (GNSS NED minus ASV position estimate)
    cov6        : (N, 6)  position covariance at measurement times
    R_diag      : (3,)    diagonal of the GNSS noise covariance R

    Returns
    -------
    nis : (N,) — NaN for non-finite or singular rows.

    Note: ISAM2 is a smoother, so x here is the *posterior* (the measurement
    was already fused). The resulting NIS reflects smoother residuals, not
    classical filter innovations, and will be smaller than a forward filter NIS.
    """
    valid = np.isfinite(innovations).all(axis=1) & np.isfinite(cov6).all(axis=1)
    nis   = np.full(len(innovations), np.nan)
    if not valid.any():
        return nis

    iv = innovations[valid]
    cv = cov6[valid]
    R  = np.diag(R_diag)

    nis_valid = np.full(len(iv), np.nan)
    for k in range(len(iv)):
        P = np.array([
            [cv[k, 0], cv[k, 1], cv[k, 2]],
            [cv[k, 1], cv[k, 3], cv[k, 4]],
            [cv[k, 2], cv[k, 4], cv[k, 5]],
        ])
        S = P + R
        try:
            nis_valid[k] = iv[k] @ np.linalg.inv(S) @ iv[k]
        except np.linalg.LinAlgError:
            pass

    nis[valid] = nis_valid
    return nis


def _chi2_panel(ax, t, values, color, title):
    """Draw a NEES or NIS time-series with chi² bounds on ax."""
    win = max(1, len(values) // 20)
    avg = np.convolve(
        np.where(np.isfinite(values), values, 0.0),
        np.ones(win) / win,
        mode="same",
    )
    ax.plot(t, values, alpha=0.25, color=color, linewidth=0.8, label="value")
    ax.plot(t, avg,    color=color, linewidth=1.5,
            label=f"moving avg (w={win})")
    ax.axhline(3.0,        color="black", linestyle="--", linewidth=1.0,
               label="expected (dof=3)")
    ax.axhline(_CHI2_3_HI, color="red",   linestyle=":",  linewidth=1.0,
               label=f"95 % upper ({_CHI2_3_HI:.2f})")
    ax.axhline(_CHI2_3_LO, color="green", linestyle=":",  linewidth=1.0,
               label=f"95 % lower ({_CHI2_3_LO:.2f})")
    mean_val = float(np.nanmean(values))
    ax.set_title(f"{title}  (mean = {mean_val:.2f})")
    ax.set_ylabel("value")
    ax.legend(loc="upper right")


# ============================================================
# Error statistics
# ============================================================

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


# ============================================================
# Plotter
# ============================================================

@dataclass
class PlotterCSVJoint:
    rov_gt_csv: str
    asv_gt_csv: str
    rov_est_csv: str
    asv_est_csv: str

    scenario_name: str = "Joint scenario"
    save_dir: str | None = None

    # Optional: path to the GNSS CSV logged by csv_logger_node (enables NIS).
    gnss_csv: str | None = None
    # GNSS noise sigmas in metres — must match the values used in the FGO node.
    gps_sigma_ne: float = 0.3
    gps_sigma_d:  float = 0.5
    # Optional extras stored for reference; not used by current plot methods.
    usbl_csv:    str | None = None
    scenario_id: int | None = None

    def _load(self):
        rov_gt_df  = _load_csv(self.rov_gt_csv)
        asv_gt_df  = _load_csv(self.asv_gt_csv)
        rov_est_df = _load_csv(self.rov_est_csv)
        asv_est_df = _load_csv(self.asv_est_csv)

        # ── ROV: filter by rov_id and mark unavailable if CSV is empty ──────────
        self._rov_available = True
        if "rov_id" in rov_gt_df.columns and "rov_id" in rov_est_df.columns:
            mode_result = rov_est_df["rov_id"].mode()
            if mode_result.empty:
                self._rov_available = False
            else:
                rov_id     = int(mode_result.iloc[0])
                rov_gt_df  = rov_gt_df[rov_gt_df["rov_id"] == rov_id].copy()
                rov_est_df = rov_est_df[rov_est_df["rov_id"] == rov_id].copy()

        if len(rov_est_df) == 0:
            self._rov_available = False

        self.asv_gt_t = _gt_time(asv_gt_df)

        if "time" not in asv_est_df.columns:
            raise KeyError("Estimated ASV CSV must contain a 'time' column.")

        self.asv_est_t = _fix_est_time(asv_est_df["time"].to_numpy(dtype=float), self.asv_gt_t)
        self.asv_gt    = _xyz(asv_gt_df,  "x_n", "y_e", "z_d")
        self.asv_est   = _xyz(asv_est_df, "x",   "y",   "z")

        self.asv_gt_t_rel  = self.asv_gt_t  - self.asv_gt_t[0]
        self.asv_est_t_rel = self.asv_est_t - self.asv_est_t[0]

        asv_mask = (self.asv_gt_t >= self.asv_est_t.min()) & (self.asv_gt_t <= self.asv_est_t.max())
        self.asv_gt_eval_t     = self.asv_gt_t[asv_mask]
        self.asv_gt_eval       = self.asv_gt[asv_mask]
        self.asv_gt_eval_t_rel = self.asv_gt_eval_t - self.asv_gt_eval_t[0]
        self.asv_est_i         = _interp(self.asv_est_t, self.asv_est, self.asv_gt_eval_t)

        # ── ROV arrays (only when data is present) ────────────────────────────
        if self._rov_available:
            self.rov_gt_t = _gt_time(rov_gt_df)
            if "time" not in rov_est_df.columns:
                raise KeyError("Estimated ROV CSV must contain a 'time' column.")
            self.rov_est_t     = _fix_est_time(rov_est_df["time"].to_numpy(dtype=float), self.rov_gt_t)
            self.rov_gt        = _xyz(rov_gt_df,  "x_n", "y_e", "z_d")
            self.rov_est       = _xyz(rov_est_df, "x",   "y",   "z")
            self.rov_gt_t_rel  = self.rov_gt_t  - self.rov_gt_t[0]
            self.rov_est_t_rel = self.rov_est_t - self.rov_est_t[0]
            rov_mask = ((self.rov_gt_t >= self.rov_est_t.min()) &
                        (self.rov_gt_t <= self.rov_est_t.max()))
            self.rov_gt_eval_t     = self.rov_gt_t[rov_mask]
            self.rov_gt_eval       = self.rov_gt[rov_mask]
            self.rov_gt_eval_t_rel = self.rov_gt_eval_t - self.rov_gt_eval_t[0]
            self.rov_est_i         = _interp(self.rov_est_t, self.rov_est, self.rov_gt_eval_t)
        else:
            self.rov_gt_t          = np.array([])
            self.rov_est_t         = np.array([])
            self.rov_gt            = np.empty((0, 3))
            self.rov_est           = np.empty((0, 3))
            self.rov_gt_t_rel      = np.array([])
            self.rov_est_t_rel     = np.array([])
            self.rov_gt_eval_t     = np.array([])
            self.rov_gt_eval       = np.empty((0, 3))
            self.rov_gt_eval_t_rel = np.array([])
            self.rov_est_i         = np.empty((0, 3))

        # ── Position covariance (for NEES) ────────────────────────────────────
        # Columns written by csv_logger_node:
        #   pos_cov_xx, pos_cov_xy, pos_cov_xz, pos_cov_yy, pos_cov_yz, pos_cov_zz
        cov_cols = ['pos_cov_xx', 'pos_cov_xy', 'pos_cov_xz',
                    'pos_cov_yy', 'pos_cov_yz', 'pos_cov_zz']

        if all(c in asv_est_df.columns for c in cov_cols):
            asv_cov6_raw       = asv_est_df[cov_cols].to_numpy(dtype=float)
            self.asv_cov6_i    = _interp_cov6(self.asv_est_t, asv_cov6_raw, self.asv_gt_eval_t)
            self._asv_cov6_raw = asv_cov6_raw   # kept for NIS (interpolated to GNSS times)
        else:
            self.asv_cov6_i    = None
            self._asv_cov6_raw = None

        if self._rov_available and all(c in rov_est_df.columns for c in cov_cols):
            rov_cov6_raw    = rov_est_df[cov_cols].to_numpy(dtype=float)
            self.rov_cov6_i = _interp_cov6(self.rov_est_t, rov_cov6_raw, self.rov_gt_eval_t)
        else:
            self.rov_cov6_i = None

        # ── GNSS measurements (for NIS) ───────────────────────────────────────
        if self.gnss_csv:
            gnss_df      = _load_csv(self.gnss_csv)
            self.gnss_t  = _fix_est_time(gnss_df["time"].to_numpy(float), self.asv_gt_t)
            self.gnss_ned = _xyz(gnss_df, "ned_n", "ned_e", "ned_d")
        else:
            self.gnss_t   = None
            self.gnss_ned = None

    # ── 3D trajectory ─────────────────────────────────────────────────────────

    def plot3d(self):
        self._load()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        if self._rov_available:
            ax.plot(*self.rov_gt.T,  label="ROV ground truth", linestyle="--", color="C1", alpha=0.8)
            ax.scatter(*self.rov_gt[0], marker="x", color="red", s=60)
            ax.plot(*self.rov_est.T, label="ROV estimate",     color="C0",    alpha=0.85)
        ax.plot(*self.asv_gt.T,  label="ASV ground truth", linestyle="-.", color="C2", alpha=0.7)
        ax.scatter(*self.asv_gt[0], marker="^", color="C2", s=60)
        ax.plot(*self.asv_est.T, label="ASV estimate",     color="C3",    alpha=0.85)

        ax.set_xlabel("North [m]", labelpad=10)
        ax.set_ylabel("East [m]",  labelpad=10)
        ax.set_zlabel("Down [m]",  labelpad=10)
        ax.invert_zaxis()
        ax.view_init(elev=15, azim=-110)
        ax.set_box_aspect(None)
        ax.xaxis.pane.set_edgecolor("black");  ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_edgecolor("black");  ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_edgecolor("black");  ax.zaxis.pane.set_alpha(0.1)
        ax.grid(True)
        ax.set_title(f"{self.scenario_name} - 3D Trajectories")
        ax.legend(loc="upper right")
        fig.tight_layout()
        return fig

    # ── Position error ────────────────────────────────────────────────────────

    def plot_position_error(self):
        self._load()

        asv_err_norm, _, asv_valid = _position_error(self.asv_gt_eval, self.asv_est_i)

        n_rows = 2 if self._rov_available else 1
        fig, axs = plt.subplots(n_rows, 1, figsize=(10, 3 * n_rows + 1), sharex=False,
                                squeeze=False)

        if self._rov_available:
            rov_err_norm, _, rov_valid = _position_error(self.rov_gt_eval, self.rov_est_i)
            axs[0, 0].plot(self.rov_gt_eval_t_rel[rov_valid], rov_err_norm,
                           label="ROV position error", color="C0")
            axs[0, 0].set_ylabel("Error [m]")
            axs[0, 0].set_title("ROV Position Error")
            axs[0, 0].grid(True)
            axs[0, 0].legend()
            asv_ax = axs[1, 0]
        else:
            asv_ax = axs[0, 0]

        asv_ax.plot(self.asv_gt_eval_t_rel[asv_valid], asv_err_norm,
                    label="ASV position error", color="C3")
        asv_ax.set_xlabel("Time [s]")
        asv_ax.set_ylabel("Error [m]")
        asv_ax.set_title("ASV Position Error")
        asv_ax.grid(True)
        asv_ax.legend()

        fig.suptitle(f"{self.scenario_name} - Position Errors")
        fig.tight_layout()
        return fig

    def plot_position_error_components(self, platform="ROV"):
        self._load()

        platform = platform.upper()
        if platform == "ROV":
            if not self._rov_available:
                raise ValueError("No ROV estimates available — skipping ROV component error plot.")
            gt_t, gt, est_i = self.rov_gt_eval_t_rel, self.rov_gt_eval, self.rov_est_i
        elif platform == "ASV":
            gt_t, gt, est_i = self.asv_gt_eval_t_rel, self.asv_gt_eval, self.asv_est_i
        else:
            raise ValueError("platform must be either 'ROV' or 'ASV'")

        valid = _valid_rows(gt, est_i)
        t     = gt_t[valid]
        err   = est_i[valid] - gt[valid]

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

    # ── NEES ──────────────────────────────────────────────────────────────────

    def plot_nees(self):
        """
        Plot Normalized Estimation Error Squared (NEES) for each platform
        that has published non-zero position covariance.

        A consistent estimator has NEES ~ chi²(3):  mean ≈ 3,  95 % of
        individual samples inside [{lo:.2f}, {hi:.2f}].
        """.format(lo=_CHI2_3_LO, hi=_CHI2_3_HI)
        self._load()

        has_asv = _cov6_has_data(self.asv_cov6_i)
        has_rov = _cov6_has_data(self.rov_cov6_i)

        if not has_asv and not has_rov:
            raise ValueError(
                "No non-zero covariance found in any estimated CSV. "
                "Make sure the FGO node publishes covariance and the CSV was "
                "recorded with the updated csv_logger_node."
            )

        panels = []
        if has_asv:
            valid  = (_valid_rows(self.asv_gt_eval, self.asv_est_i)
                      & np.isfinite(self.asv_cov6_i).all(axis=1))
            errors = self.asv_est_i[valid] - self.asv_gt_eval[valid]
            nees   = _compute_nees(errors, self.asv_cov6_i[valid])
            panels.append(("ASV", self.asv_gt_eval_t_rel[valid], nees, "C3"))
        if has_rov:
            valid  = (_valid_rows(self.rov_gt_eval, self.rov_est_i)
                      & np.isfinite(self.rov_cov6_i).all(axis=1))
            errors = self.rov_est_i[valid] - self.rov_gt_eval[valid]
            nees   = _compute_nees(errors, self.rov_cov6_i[valid])
            panels.append(("ROV", self.rov_gt_eval_t_rel[valid], nees, "C0"))

        fig, axs = plt.subplots(len(panels), 1,
                                figsize=(10, 4 * len(panels)), squeeze=False)
        for ax, (name, t, nees, color) in zip(axs[:, 0], panels):
            _chi2_panel(ax, t, nees, color, f"{name} Position NEES")

        axs[-1, 0].set_xlabel("Time [s]")
        fig.suptitle(f"{self.scenario_name} - NEES")
        fig.tight_layout()
        return fig

    # ── NIS ───────────────────────────────────────────────────────────────────

    def plot_nis(self):
        """
        Plot Normalized Innovation Squared (NIS) for GNSS vs the ASV
        posterior position estimate.

        Innovation:  ν = z_gnss − x_est  (both in NED frame).
        Innovation covariance:  S = P_pos + R_gnss.

        Because ISAM2 is a smoother, x_est here is the *posterior* — the GNSS
        measurement has already been fused.  The NIS therefore reflects
        smoother residuals, not forward-filter innovations, and will generally
        be smaller than chi²(3).  It is still a useful consistency indicator:
        large spikes reveal measurement outliers or modelling errors.
        """
        self._load()

        if self.gnss_t is None:
            raise ValueError(
                "gnss_csv is not set.  Pass the path to the GNSS CSV produced "
                "by csv_logger_node to enable NIS plots."
            )
        if self._asv_cov6_raw is None:
            raise ValueError(
                "No covariance columns found in the ASV estimated CSV. "
                "Make sure the FGO node publishes covariance."
            )

        # Restrict to GNSS samples within the ASV estimate time window.
        t_min = self.asv_est_t.min()
        t_max = self.asv_est_t.max()
        mask  = (self.gnss_t >= t_min) & (self.gnss_t <= t_max)
        if not mask.any():
            raise ValueError(
                "No GNSS measurements overlap with the ASV estimate time range."
            )

        gnss_t   = self.gnss_t[mask]
        gnss_ned = self.gnss_ned[mask]

        # Interpolate ASV position and covariance to GNSS measurement times.
        asv_pos_at_gnss  = _interp(self.asv_est_t, self.asv_est, gnss_t)
        asv_cov6_at_gnss = _interp_cov6(self.asv_est_t, self._asv_cov6_raw, gnss_t)

        innovations = gnss_ned - asv_pos_at_gnss
        R_diag = np.array([
            self.gps_sigma_ne ** 2,
            self.gps_sigma_ne ** 2,
            self.gps_sigma_d  ** 2,
        ])
        nis   = _compute_nis(innovations, asv_cov6_at_gnss, R_diag)
        t_rel = gnss_t - gnss_t[0]

        fig, ax = plt.subplots(figsize=(10, 4))
        _chi2_panel(ax, t_rel, nis, "C1",
                    "ASV GNSS NIS  [smoother posterior]")
        ax.set_xlabel("Time [s]")
        fig.suptitle(f"{self.scenario_name} - NIS")
        fig.tight_layout()
        return fig

    # ── Statistics ────────────────────────────────────────────────────────────

    def export_statistics(self):
        self._load()
        rows = []

        if self._rov_available:
            rov_stats = _error_statistics(self.rov_gt_eval, self.rov_est_i)
            if _cov6_has_data(self.rov_cov6_i):
                valid = (_valid_rows(self.rov_gt_eval, self.rov_est_i)
                         & np.isfinite(self.rov_cov6_i).all(axis=1))
                nees  = _compute_nees(
                    self.rov_est_i[valid] - self.rov_gt_eval[valid],
                    self.rov_cov6_i[valid],
                )
                rov_stats["mean_nees"] = float(np.nanmean(nees))
            rov_stats.update({"scenario": self.scenario_name, "platform": "ROV"})
            rows.append(rov_stats)

        asv_stats = _error_statistics(self.asv_gt_eval, self.asv_est_i)
        if _cov6_has_data(self.asv_cov6_i):
            valid = (_valid_rows(self.asv_gt_eval, self.asv_est_i)
                     & np.isfinite(self.asv_cov6_i).all(axis=1))
            nees  = _compute_nees(
                self.asv_est_i[valid] - self.asv_gt_eval[valid],
                self.asv_cov6_i[valid],
            )
            asv_stats["mean_nees"] = float(np.nanmean(nees))
        asv_stats.update({"scenario": self.scenario_name, "platform": "ASV"})
        rows.append(asv_stats)

        return pd.DataFrame(rows)

    # ── Save all ──────────────────────────────────────────────────────────────

    def save_all(self):
        if not self.save_dir:
            raise ValueError("save_dir must be set to save plots and statistics.")

        path = Path(self.save_dir)
        path.mkdir(parents=True, exist_ok=True)

        fig1 = self.plot3d()
        fig2 = self.plot_position_error()
        fig4 = self.plot_position_error_components("ASV")
        stat_csv = self.export_statistics()

        fig1.savefig(path / "traj_3d.png",                       dpi=150, bbox_inches="tight")
        fig2.savefig(path / "position_error.png",                dpi=150, bbox_inches="tight")
        fig4.savefig(path / "asv_position_error_components.png", dpi=150, bbox_inches="tight")
        stat_csv.to_csv(path / "fgo_statistics.csv", index=False)
        plt.close(fig1); plt.close(fig2); plt.close(fig4)

        try:
            fig3 = self.plot_position_error_components("ROV")
            fig3.savefig(path / "rov_position_error_components.png", dpi=150, bbox_inches="tight")
            plt.close(fig3)
        except ValueError as exc:
            print(f"[plotting] Skipping ROV component error: {exc}")

        try:
            fig5 = self.plot_nees()
            fig5.savefig(path / "nees.png", dpi=150, bbox_inches="tight")
            plt.close(fig5)
        except ValueError as exc:
            print(f"[plotting] Skipping NEES: {exc}")

        if self.gnss_csv:
            try:
                fig6 = self.plot_nis()
                fig6.savefig(path / "nis.png", dpi=150, bbox_inches="tight")
                plt.close(fig6)
            except ValueError as exc:
                print(f"[plotting] Skipping NIS: {exc}")

        return path, stat_csv

    def show(self):
        if self.save_dir:
            self.save_all()
        plt.show(block=True)


# ============================================================
# Quick local test
# ============================================================
if __name__ == "__main__":
    plotter1 = PlotterCSVJoint(
        rov_gt_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/simulation_data/8a/rov_ground_truth.csv",
        asv_gt_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/simulation_data/8a/asv_ground_truth.csv",
        rov_est_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/debugging/test/rov_estimated_my_experiment_01.csv",
        asv_est_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/debugging/test/boat_estimated_my_experiment_01.csv",
        scenario_name="FGO - Scenario 1: Bearing-only",
        save_dir="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/debugging/test/plots_s1",
        gnss_csv="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/debugging/test/gnss_my_experiment_01.csv",
    )
    plotter1.show()
