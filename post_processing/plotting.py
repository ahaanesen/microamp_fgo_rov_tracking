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
    _CHI2_3_LO = 0.352
    _CHI2_3_HI = 9.348


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

    uniq_t, uniq_idx = np.unique(src_t, return_index=True)
    src_t = uniq_t
    src_xyz = src_xyz[uniq_idx]

    t_min, t_max = src_t[0], src_t[-1]
    mask = (tgt_t >= t_min) & (tgt_t <= t_max)

    out = np.full((len(tgt_t), 3), np.nan, dtype=float)
    for i in range(3):
        out[mask, i] = np.interp(tgt_t[mask], src_t, src_xyz[:, i])
    return out

def _rmse(gt: np.ndarray, est: np.ndarray):
    """
    Per-time-step RMSE across components.
    gt, est: (N, d)
    returns:
      rmse: (N,)
      err:  (N, d)
    """
    err = est - gt
    rmse = np.sqrt(np.mean(err**2, axis=1))
    return rmse, err

def _interp_cov6(src_t, src_cov6, tgt_t):
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
    if cov6 is None:
        return False
    finite = cov6[np.isfinite(cov6)]
    return bool(finite.size > 0 and np.any(np.abs(finite) > 1e-12))


# ============================================================
# NEES / NIS
# ============================================================

def _compute_nees(errors: np.ndarray, cov6: np.ndarray) -> np.ndarray:
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

    nees_pos_vals, nees_vel_vals = [], []
    # Estimate position covariance from residuals (sample covariance) and
    # compute per-sample NEES = e.T @ P^{-1} @ e. Use pseudo-inverse if needed.
    try:
        if len(err) > 0:
            # sample covariance (unbiased)
            P = np.cov(err, rowvar=False, bias=False)
            # ensure shape (3,3)
            if P.shape == (3, 3) and np.all(np.isfinite(P)):
                try:
                    Pinv = np.linalg.inv(P)
                except np.linalg.LinAlgError:
                    Pinv = np.linalg.pinv(P)
                for k in range(len(err)):
                    e = err[k]
                    try:
                        nees_pos_vals.append(float(e @ Pinv @ e))
                    except Exception:
                        nees_pos_vals.append(np.nan)
    except Exception:
        nees_pos_vals = []

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
        "mean_tanees_pos": float(np.nanmean(nees_pos_vals)) if len(nees_pos_vals) > 0 else np.nan,
        "mean_tanees_vel": float(np.nanmean(nees_vel_vals)) if len(nees_vel_vals) > 0 else np.nan,
        # "mean_abs_n": float(np.mean(np.abs(err[:, 0]))),
        # "mean_abs_e": float(np.mean(np.abs(err[:, 1]))),
        # "mean_abs_d": float(np.mean(np.abs(err[:, 2]))),
        # "std_n": float(np.std(err[:, 0])),
        # "std_e": float(np.std(err[:, 1])),
        # "std_d": float(np.std(err[:, 2])),
        # "path_length_gt": float(path_length_gt),
        # "path_length_est": float(path_length_est),
        # "path_length_error_pct": float(path_length_error_pct),
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

    gnss_csv: str | None = None
    gps_sigma_ne: float = 0.3
    gps_sigma_d:  float = 0.5
    usbl_csv:    str | None = None
    scenario_id: int | None = None

    # --- velocity column names in estimate CSVs ---
    asv_vel_cols: tuple = ("vx", "vy", "vz")
    rov_vel_cols: tuple = ("vx", "vy", "vz")

    def _load(self):
        rov_gt_df  = _load_csv(self.rov_gt_csv)
        asv_gt_df  = _load_csv(self.asv_gt_csv)
        rov_est_df = _load_csv(self.rov_est_csv)
        asv_est_df = _load_csv(self.asv_est_csv)

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

        # velocities
        self.asv_vel = _xyz(asv_est_df, *self.asv_vel_cols)

        self.asv_gt_t_rel  = self.asv_gt_t  - self.asv_gt_t[0]
        self.asv_est_t_rel = self.asv_est_t - self.asv_est_t[0]

        asv_mask = (self.asv_gt_t >= self.asv_est_t.min()) & (self.asv_gt_t <= self.asv_est_t.max())
        self.asv_gt_eval_t     = self.asv_gt_t[asv_mask]
        self.asv_gt_eval       = self.asv_gt[asv_mask]
        self.asv_gt_eval_t_rel = self.asv_gt_eval_t - self.asv_gt_eval_t[0]
        self.asv_est_i         = _interp(self.asv_est_t, self.asv_est, self.asv_gt_eval_t)
        self.asv_vel_i         = _interp(self.asv_est_t, self.asv_vel, self.asv_gt_eval_t)

        if self._rov_available:
            self.rov_gt_t = _gt_time(rov_gt_df)
            if "time" not in rov_est_df.columns:
                raise KeyError("Estimated ROV CSV must contain a 'time' column.")
            self.rov_est_t     = _fix_est_time(rov_est_df["time"].to_numpy(dtype=float), self.rov_gt_t)
            self.rov_gt        = _xyz(rov_gt_df,  "x_n", "y_e", "z_d")
            self.rov_est       = _xyz(rov_est_df, "x",   "y",   "z")
            self.rov_vel       = _xyz(rov_est_df, *self.rov_vel_cols)

            self.rov_gt_t_rel  = self.rov_gt_t  - self.rov_gt_t[0]
            self.rov_est_t_rel = self.rov_est_t - self.rov_est_t[0]
            rov_mask = ((self.rov_gt_t >= self.rov_est_t.min()) &
                        (self.rov_gt_t <= self.rov_est_t.max()))
            self.rov_gt_eval_t     = self.rov_gt_t[rov_mask]
            self.rov_gt_eval       = self.rov_gt[rov_mask]
            self.rov_gt_eval_t_rel = self.rov_gt_eval_t - self.rov_gt_eval_t[0]
            self.rov_est_i         = _interp(self.rov_est_t, self.rov_est, self.rov_gt_eval_t)
            self.rov_vel_i         = _interp(self.rov_est_t, self.rov_vel, self.rov_gt_eval_t)
        else:
            self.rov_gt_t          = np.array([])
            self.rov_est_t         = np.array([])
            self.rov_gt            = np.empty((0, 3))
            self.rov_est           = np.empty((0, 3))
            self.rov_vel           = np.empty((0, 3))
            self.rov_gt_eval_t     = np.array([])
            self.rov_gt_eval       = np.empty((0, 3))
            self.rov_est_i         = np.empty((0, 3))
            self.rov_vel_i         = np.empty((0, 3))

        cov_cols = ['pos_cov_xx', 'pos_cov_xy', 'pos_cov_xz',
                    'pos_cov_yy', 'pos_cov_yz', 'pos_cov_zz']

        if all(c in asv_est_df.columns for c in cov_cols):
            asv_cov6_raw       = asv_est_df[cov_cols].to_numpy(dtype=float)
            self.asv_cov6_i    = _interp_cov6(self.asv_est_t, asv_cov6_raw, self.asv_gt_eval_t)
            self._asv_cov6_raw = asv_cov6_raw
        else:
            self.asv_cov6_i    = None
            self._asv_cov6_raw = None

        if self._rov_available and all(c in rov_est_df.columns for c in cov_cols):
            rov_cov6_raw    = rov_est_df[cov_cols].to_numpy(dtype=float)
            self.rov_cov6_i = _interp_cov6(self.rov_est_t, rov_cov6_raw, self.rov_gt_eval_t)
        else:
            self.rov_cov6_i = None

        if self.gnss_csv:
            gnss_df      = _load_csv(self.gnss_csv)
            self.gnss_t  = _fix_est_time(gnss_df["time"].to_numpy(float), self.asv_gt_t)
            self.gnss_ned = _xyz(gnss_df, "ned_n", "ned_e", "ned_d")
        else:
            self.gnss_t   = None
            self.gnss_ned = None

    # ── 3D trajectory ─────────────────────────────────────────

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

    # ── RMSE (pos+vel per platform) ───────────────────────────

    def plot_rmse_rov(self):
        self._load()
        if not self._rov_available:
            return None

        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

        rmse_pos, _ = _rmse(self.rov_gt_eval, self.rov_est_i)
        rmse_vel, _ = _rmse(np.gradient(self.rov_gt_eval, self.rov_gt_eval_t, axis=0), self.rov_vel_i)

        axs[0].plot(self.rov_gt_eval_t_rel, rmse_pos, color="C0", label="ROV pos RMSE")
        axs[0].set_title("ROV Position RMSE")
        axs[0].set_ylabel("RMSE [m]")
        axs[0].grid(True); axs[0].legend()

        axs[1].plot(self.rov_gt_eval_t_rel, rmse_vel, color="C1", label="ROV vel RMSE")
        axs[1].set_title("ROV Velocity RMSE")
        axs[1].set_ylabel("RMSE [m/s]")
        axs[1].set_xlabel("Time [s]")
        axs[1].grid(True); axs[1].legend()

        fig.suptitle(f"{self.scenario_name} — ROV RMSE")
        fig.tight_layout()
        return fig

    def plot_rmse_asv(self):
        self._load()

        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

        rmse_pos, _ = _rmse(self.asv_gt_eval, self.asv_est_i)
        rmse_vel, _ = _rmse(np.gradient(self.asv_gt_eval, self.asv_gt_eval_t, axis=0), self.asv_vel_i)

        axs[0].plot(self.asv_gt_eval_t_rel, rmse_pos, color="C3", label="ASV pos RMSE")
        axs[0].set_title("ASV Position RMSE")
        axs[0].set_ylabel("RMSE [m]")
        axs[0].grid(True); axs[0].legend()

        axs[1].plot(self.asv_gt_eval_t_rel, rmse_vel, color="C4", label="ASV vel RMSE")
        axs[1].set_title("ASV Velocity RMSE")
        axs[1].set_ylabel("RMSE [m/s]")
        axs[1].set_xlabel("Time [s]")
        axs[1].grid(True); axs[1].legend()

        fig.suptitle(f"{self.scenario_name} — ASV RMSE")
        fig.tight_layout()
        return fig

    # ── NEES (pos+vel) ─────────────────────────────────────────

    def plot_nees(self):
        self._load()

        has_asv = _cov6_has_data(self.asv_cov6_i)
        has_rov = _cov6_has_data(self.rov_cov6_i)

        if not has_asv and not has_rov:
            raise ValueError("No non-zero covariance found in estimated CSVs.")

        fig, axs = plt.subplots(2, 1, figsize=(10, 8), squeeze=False)

        if has_asv:
            valid = (_valid_rows(self.asv_gt_eval, self.asv_est_i)
                     & np.isfinite(self.asv_cov6_i).all(axis=1))
            errors = self.asv_est_i[valid] - self.asv_gt_eval[valid]
            nees = _compute_nees(errors, self.asv_cov6_i[valid])
            _chi2_panel(axs[0, 0], self.asv_gt_eval_t_rel[valid], nees, "C3", "ASV Position NEES")

        if has_rov:
            valid = (_valid_rows(self.rov_gt_eval, self.rov_est_i)
                     & np.isfinite(self.rov_cov6_i).all(axis=1))
            errors = self.rov_est_i[valid] - self.rov_gt_eval[valid]
            nees = _compute_nees(errors, self.rov_cov6_i[valid])
            _chi2_panel(axs[1, 0], self.rov_gt_eval_t_rel[valid], nees, "C0", "ROV Position NEES")

        axs[-1, 0].set_xlabel("Time [s]")
        fig.suptitle(f"{self.scenario_name} - NEES")
        fig.tight_layout()
        return fig

    # ── NIS ───────────────────────────────────────────────────

    def plot_nis(self):
        self._load()

        if self.gnss_t is None:
            raise ValueError("gnss_csv is not set.")
        if self._asv_cov6_raw is None:
            raise ValueError("No covariance columns found in ASV estimated CSV.")

        t_min = self.asv_est_t.min()
        t_max = self.asv_est_t.max()
        mask  = (self.gnss_t >= t_min) & (self.gnss_t <= t_max)
        if not mask.any():
            raise ValueError("No GNSS measurements overlap with the ASV estimate time range.")

        gnss_t   = self.gnss_t[mask]
        gnss_ned = self.gnss_ned[mask]

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

    # ── Statistics ────────────────────────────────────────────

    def export_statistics(self):
        self._load()
        rows = []

        if self._rov_available:
            rov_stats = _error_statistics(self.rov_gt_eval, self.rov_est_i)
            rov_stats.update({"scenario": self.scenario_name, "platform": "ROV"})
            rows.append(rov_stats)

        asv_stats = _error_statistics(self.asv_gt_eval, self.asv_est_i)
        asv_stats.update({"scenario": self.scenario_name, "platform": "ASV"})
        rows.append(asv_stats)

        return pd.DataFrame(rows)

    # ── Save all ─────────────────────────────────────────────

    def save_all(self):
        if not self.save_dir:
            raise ValueError("save_dir must be set to save plots and statistics.")

        path = Path(self.save_dir)
        path.mkdir(parents=True, exist_ok=True)

        fig1 = self.plot3d()
        fig_rmse_asv = self.plot_rmse_asv()
        fig_rmse_rov = self.plot_rmse_rov()
        fig_nees = self.plot_nees()
        stat_csv = self.export_statistics()

        fig1.savefig(path / "traj_3d.png", dpi=150, bbox_inches="tight")
        if fig_rmse_asv:
            fig_rmse_asv.savefig(path / "rmse_asv.png", dpi=150, bbox_inches="tight")
        if fig_rmse_rov:
            fig_rmse_rov.savefig(path / "rmse_rov.png", dpi=150, bbox_inches="tight")
        if fig_nees:
            fig_nees.savefig(path / "nees.png", dpi=150, bbox_inches="tight")
        stat_csv.to_csv(path / "fgo_statistics.csv", index=False)

        plt.close(fig1)
        if fig_rmse_asv:
            plt.close(fig_rmse_asv)
        if fig_rmse_rov:
            plt.close(fig_rmse_rov)
        if fig_nees:
            plt.close(fig_nees)

        if self.gnss_csv:
            try:
                fig_nis = self.plot_nis()
                fig_nis.savefig(path / "nis.png", dpi=150, bbox_inches="tight")
                plt.close(fig_nis)
            except ValueError as exc:
                print(f"[plotting] Skipping NIS: {exc}")

        return path, stat_csv

    def show(self):
        if self.save_dir:
            self.save_all()
        plt.show(block=True)