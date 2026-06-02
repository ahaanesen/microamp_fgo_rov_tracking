'''Experiment 3: TDMA slot length comparison.'''
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

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

DEFAULT_DATASET = (
  PACKAGE_ROOT / "post_processing" / "simulation_data" / "figure8_delay_no_loss_tdma_slot_5p0"
)
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "post_processing" / "results" / "exp3_tdma_sweep2"

TRUE_RANGE = np.sqrt(269)  # pre-computed true range for this scenario

TDMA_SLOT_LENGTHS = [5.0, 10.0, 20.0, 30.0, 60.0, 120.0]

def _run_scenario(dataset_dir: Path, output_root: Path, scenario_id: list, sigma: float, tdma_interval: float, startup_delay: float) -> Path:
  run_name = f"tdma_{tdma_interval:.1f}"
  run_root = output_root / run_name
  cmd = [
    sys.executable,
    str(RUN_SCRIPT),
    # "--chosen-scenarios",
    # str(scenario_id), # All are default
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
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--startup-delay", type=float, default=2.0)
    parser.add_argument("--runs-per-sigma", type=int, default=1)
    parser.add_argument("--estimator", default="FGO")
    parser.add_argument("--experiment", default="exp3_tdma_sweep")
    parser.add_argument("--divergence-threshold", type=float, default=10.0)
    args = parser.parse_args()

    dataset_dir_without_tdma = PACKAGE_ROOT / "post_processing" / "simulation_data" / "figure8_delay_no_loss_tdma_slot_"
    output_root = Path(args.output_root).resolve()
    sigma = 0.02  # Fixed sigma for this experiment
    
    rows = []
    scenario_ids = [1, 2, 3]
    for tdma_interval in TDMA_SLOT_LENGTHS:
        # for scenario_id in [1, 2, 3]:  # Run each TDMA slot length on all three scenarios
        # Slot folders are named like figure8_delay_no_loss_tdma_slot_5p0.
        slot_str = f"{tdma_interval:.1f}".replace(".0", "")
        dataset_dir = Path(f"{dataset_dir_without_tdma}{slot_str}p0").resolve()
        print(f"Running tdma_interval={tdma_interval}...", flush=True)
        run_root = _run_scenario(dataset_dir, output_root, [1,2,3], sigma, tdma_interval, args.startup_delay)
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

