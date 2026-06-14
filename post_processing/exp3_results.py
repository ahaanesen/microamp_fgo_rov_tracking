'''Experiment 3: TDMA slot length comparison.'''
import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np

'''
cd /ros2_ws/microampere_ros2ws
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 src/microamp_fgo_rov_tracking/post_processing/exp3_results.py
'''

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent.parent
RUN_SCRIPT = SCRIPT_DIR / "run_all_scenarios.py"

DEFAULT_TRAJECTORY = "linear_turns"

# linear_turns_init 
#       rov_initial_range_guess:  223.749 # initial guess on range from ASV to ROV [m]
  
TRUE_RANGE_FIGURE8 = np.sqrt(269)  # pre-computed true range for this scenario
TRUE_RANGE_LINEAR_TURNS = 223.749

TDMA_SLOT_LENGTHS = [5.0, 10.0, 20.0, 30.0, 60.0, 120.0]


def _default_dataset_prefix(trajectory: str) -> str:
  return str(PACKAGE_ROOT / "post_processing" / "simulation_data" / f"{trajectory}_delay_no_loss_tdma_slot_")


def _default_output_root(trajectory: str) -> Path:
  return PACKAGE_ROOT / "post_processing" / "results" / trajectory / "exp3_tdma_sweep"


def _dataset_prefix_from_seed(dataset_dir: str | None, trajectory: str) -> str:
  if dataset_dir is None:
    return _default_dataset_prefix(trajectory)

  seed_dir = Path(dataset_dir).resolve()
  match = re.match(r"(?P<prefix>.*_tdma_slot_)\d+p\d+$", str(seed_dir))
  if not match:
    raise ValueError(
      "--dataset-dir must point to a TDMA slot folder ending like "
      f"{trajectory}_delay_no_loss_tdma_slot_5p0"
    )
  return match.group("prefix")


def _true_range_for_trajectory(trajectory: str) -> float:
  return TRUE_RANGE_FIGURE8 if trajectory == "figure8" else TRUE_RANGE_LINEAR_TURNS


def _run_scenario(
  dataset_dir: Path,
  output_root: Path,
  sigma: float,
  tdma_interval: float,
  startup_delay: float,
  trajectory: str,
) -> Path:
  run_name = f"tdma_{tdma_interval:.1f}"
  run_root = output_root / run_name
  range_guess = _true_range_for_trajectory(trajectory)
  cmd = [
      sys.executable,
      str(RUN_SCRIPT),
      # "--chosen-scenarios",
      # "1",
      "--dataset-dir",
      str(dataset_dir),
      "--output-root",
      str(output_root),
      "--run-name",
      run_name,
      "--startup-delay",
      str(startup_delay),
      "--fgo-params",
      f"rov_cv_continous_sigma={sigma}",
      "init_with_gt=false",
      f"rov_initial_range_guess={range_guess}"
  ]
  subprocess.run(cmd, check=True, cwd=WORKSPACE_ROOT)
  return run_root


def _load_stats(run_root: Path, scenario_ids: list[int]) -> list[pd.Series]:
  stats_rows: list[pd.Series] = []
  missing_paths: list[Path] = []

  for scenario_id in scenario_ids:
    stats_path = run_root / f"scenario{scenario_id}" / "plots" / "fgo_statistics.csv"
    if not stats_path.exists():
      missing_paths.append(stats_path)
      continue

    df = pd.read_csv(stats_path)
    added_platform = False
    if "platform" in df.columns:
      for platform_name in ("ROV", "ASV"):
        platform_rows = df[df["platform"].astype(str).str.upper() == platform_name]
        if not platform_rows.empty:
          stats_rows.append(platform_rows.iloc[0])
          added_platform = True

    if not added_platform:
      stats_rows.append(df.iloc[0])

  if missing_paths:
    missing_list = ", ".join(str(path) for path in missing_paths)
    raise FileNotFoundError(f"Missing stats file(s): {missing_list}")

  return stats_rows


def _build_result_row(
  stats: pd.Series,
  *,
  sigma: float,
  tdma_interval: float,
  experiment: str,
  estimator: str,
  divergence_threshold: float,
) -> dict:
  scenario = str(stats.get("scenario", "scenario1"))
  final_error = float(stats.get("final_error", float("nan")))
  rmse = float(stats.get("ate_rms", float("nan")))
  mean_nees = float(stats.get("mean_tanees_pos", float("nan")))
  platform = str(stats.get("platform", "")).strip().upper()
  converged = bool(final_error < divergence_threshold) if pd.notna(final_error) else False

  return {
    "estimator": estimator,
    "experiment": experiment,
    "scenario": scenario,
    "platform": platform if platform else "UNKNOWN",
    "sigma_a": sigma,
    "tdma_interval": tdma_interval,
    "init_range_scale": str(1),
    "rmse": rmse,
    "ate": rmse,
    "final_error": final_error,
    "mean_nees": mean_nees,
    "converged": converged,
  }


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 3: TDMA slot length comparison")
    parser.add_argument("--trajectory", choices=["figure8", "linear_turns"], default=DEFAULT_TRAJECTORY, help="Trajectory to analyze")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Optional seed TDMA dataset directory, e.g. ..._tdma_slot_5p0. Sibling slot folders are inferred.",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--startup-delay", type=float, default=2.0)
    parser.add_argument("--runs-per-sigma", type=int, default=1)
    parser.add_argument("--estimator", default="FGO")
    parser.add_argument("--experiment", default="exp3_tdma_sweep")
    parser.add_argument("--divergence-threshold", type=float, default=10.0)
    args = parser.parse_args()
    trajectory = args.trajectory
    dataset_prefix = _dataset_prefix_from_seed(args.dataset_dir, trajectory)
    output_root = Path(args.output_root).resolve() if args.output_root else _default_output_root(trajectory).resolve()

    sigma = 0.02  # Fixed sigma for this experiment
    
    rows = []
    scenario_ids = [1, 2, 3]
    for tdma_interval in TDMA_SLOT_LENGTHS:
        # for scenario_id in [1, 2, 3]:  # Run each TDMA slot length on all three scenarios
        # Slot folders are named like figure8_delay_no_loss_tdma_slot_5p0.
        slot_str = f"{tdma_interval:.1f}".replace(".0", "")
        dataset_dir = Path(f"{dataset_prefix}{slot_str}p0").resolve()
        if not dataset_dir.exists():
          raise FileNotFoundError(f"Missing TDMA dataset directory: {dataset_dir}")
        print(f"Running tdma_interval={tdma_interval}...", flush=True)
        print(f"Dataset dir: {dataset_dir}", flush=True)
        run_root = _run_scenario(dataset_dir, output_root, sigma, tdma_interval, args.startup_delay, trajectory)
        print(f"Run root: {run_root}")
        # run_name = f"tdma_{tdma_interval:.1f}"
        # run_root = output_root / run_name        
        stats_rows = _load_stats(run_root, scenario_ids)

        for stats in stats_rows:
            row = _build_result_row(
                stats,
                sigma=sigma,
                tdma_interval=tdma_interval,
                experiment=args.experiment,
                estimator=args.estimator,
                divergence_threshold=args.divergence_threshold,
                )
            rows.append(row)

    df = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_root / "experiment3_runs_complete.csv", index=False)

    print(f"Done. Results saved in {output_root}")
    return 0


if __name__ == "__main__":
  raise SystemExit(main())
