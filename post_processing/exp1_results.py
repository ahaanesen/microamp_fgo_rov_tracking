"""Experiment 1: ROV CV sigma sweep with GT initialization."""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

'''
cd /ros2_ws/microampere_ros2ws
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 src/microamp_fgo_rov_tracking/post_processing/exp1_results.py
'''


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent.parent
RUN_SCRIPT = SCRIPT_DIR / "run_all_scenarios.py"

DEFAULT_DATASET = (
  PACKAGE_ROOT / "post_processing" / "simulation_data" / "figure8_delay_no_loss_tdma_slot_5p0"
)
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "post_processing" / "results" / "exp1_noise_sweep_joined_stats"

SIGMA_VALUES = [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]


def _run_scenario(dataset_dir: Path, output_root: Path, sigma: float, startup_delay: float) -> Path:
  run_name = f"sigma_{sigma}"
  run_root = output_root / run_name
  cmd = [
    sys.executable,
    str(RUN_SCRIPT),
    "--chosen-scenarios",
    "1",
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
    "init_with_gt=true",
  ]
  subprocess.run(cmd, check=True, cwd=WORKSPACE_ROOT)
  return run_root


def _load_stats(run_root: Path) -> dict:
  stats_path = run_root / "scenario1" / "plots" / "fgo_statistics.csv"
  if not stats_path.exists():
    raise FileNotFoundError(f"Missing stats file: {stats_path}")

  df = pd.read_csv(stats_path)
  result = {}
  
  if "platform" in df.columns:
    rov_rows = df[df["platform"].astype(str).str.upper() == "ROV"]
    if not rov_rows.empty:
      result["ROV"] = rov_rows.iloc[0]
    
    asv_rows = df[df["platform"].astype(str).str.upper() == "ASV"]
    if not asv_rows.empty:
      result["ASV"] = asv_rows.iloc[0]
  
  if not result:
    result["ROV"] = df.iloc[0]
  
  return result


def _build_result_row(
  stats: pd.Series,
  *,
  sigma: float,
  experiment: str,
  estimator: str,
  divergence_threshold: float,
) -> dict:
  scenario = str(stats.get("scenario", "scenario1"))
  platform = str(stats.get("platform", "")).strip().upper()
  final_error = float(stats.get("final_error", float("nan")))
  rmse = float(stats.get("ate_rms", float("nan")))
  mean_nees = float(stats.get("mean_tanees_pos", float("nan")))
  converged = bool(final_error < divergence_threshold) if pd.notna(final_error) else False

  return {
    "estimator": estimator,
    "experiment": experiment,
    "scenario": scenario,
    "platform": platform if platform else "UNKNOWN",
    "sigma_a": sigma,
    "tdma_interval": str(5.0),  # Fixed TDMA interval for this experiment
    "init_range_scale": str("GT init"),  # GT initialization
    "rmse": rmse,
    "ate": rmse,
    "final_error": final_error,
    "mean_nees": mean_nees,
    "converged": converged,
  }


def main() -> int:
  parser = argparse.ArgumentParser(description="Experiment 1: ROV CV sigma sweep")
  parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
  parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
  parser.add_argument("--startup-delay", type=float, default=2.0)
  parser.add_argument("--runs-per-sigma", type=int, default=1)
  parser.add_argument("--estimator", default="FGO")
  parser.add_argument("--experiment", default="exp1_noise_sweep")
  parser.add_argument("--divergence-threshold", type=float, default=10.0)
  args = parser.parse_args()

  dataset_dir = Path(args.dataset_dir).resolve()
  output_root = Path(args.output_root).resolve()

  rows = []
  for sigma in SIGMA_VALUES:
    print(f"Running sigma={sigma}...", flush=True)
    # run_root = _run_scenario(dataset_dir, output_root, sigma, args.startup_delay)
    run_name = f"sigma_{sigma}"
    run_root = output_root / run_name
    # stats = _load_stats(run_root)

    # row = _build_result_row(
    #   stats,
    #   sigma=sigma,
    #   experiment=args.experiment,
    #   estimator=args.estimator,
    #   divergence_threshold=args.divergence_threshold,
    # )
    # rows.append(row)
    stats_by_platform = _load_stats(run_root)
    for platform, stats in stats_by_platform.items():
        row = _build_result_row(
            stats,
            sigma=sigma,
            experiment=args.experiment,
            estimator=args.estimator,
            divergence_threshold=args.divergence_threshold,
        )
        rows.append(row)

  df = pd.DataFrame(rows)
  output_root.mkdir(parents=True, exist_ok=True)
  df.to_csv(output_root / "experiment1_runs_joined.csv", index=False)

  

  print(f"Done. Results saved in {output_root}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

