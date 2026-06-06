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
python3 src/microamp_fgo_rov_tracking/post_processing/exp4_results.py

'''

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent.parent
RUN_SCRIPT = SCRIPT_DIR / "run_all_scenarios.py"

# TRAJECTORY_NAME = "figure8"
TRAJECTORY_NAME = "linear_turns"  # For test

# linear_turns_init 
#       rov_initial_range_guess:  223.749 # initial guess on range from ASV to ROV [m]
  

DEFAULT_DATASET = (
  PACKAGE_ROOT / "post_processing" / "simulation_data" / f"{TRAJECTORY_NAME}_delay_tdma_loss_loss_0p0"
)
print(f"Default dataset: {DEFAULT_DATASET}")
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT/"post_processing"/"results"/f"{TRAJECTORY_NAME}"/"exp4_packet_loss_sweep"



TRUE_RANGE_FIGURE8 = np.sqrt(269)  # pre-computed true range for this scenario
TRUE_RANGE_LINEAR_TURNS = 223.749
TDMA_SLOT_LENGTH = 5.0

# PACKET_LOSS_PROB = [0.1, 0.3] # For test
PACKET_LOSS_PROB = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]  # packet loss probabilities 

def _run_scenario(dataset_dir: Path, output_root: Path, scenario_id: list, sigma: float, packet_loss_prob: float, startup_delay: float, rov_initial_range_guess: float) -> Path:
  run_name = f"packet_loss_{packet_loss_prob:.1f}"
  run_root = output_root / run_name

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
    f"rov_initial_range_guess={rov_initial_range_guess}",
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
  packet_loss_prob: float,
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
    "packet_loss_prob": packet_loss_prob,
    "init_range_scale": str(1),
    "rmse": rmse,
    "ate": rmse,
    "final_error": final_error,
    "mean_nees": mean_nees,
    "converged": converged,
  }


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 4: Packet Loss Comparison")
    parser.add_argument("--trajectory", choices=["figure8", "linear_turns"], default="linear_turns", help="Trajectory to analyze")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--startup-delay", type=float, default=2.0)
    parser.add_argument("--runs-per-sigma", type=int, default=1)
    parser.add_argument("--estimator", default="FGO")
    parser.add_argument("--experiment", default="exp4_packet_loss_sweep")
    parser.add_argument("--divergence-threshold", type=float, default=10.0)
    args = parser.parse_args()
    
    dataset_dir_without_loss = PACKAGE_ROOT / "post_processing" / "simulation_data" / f"{TRAJECTORY_NAME}_delay_tdma_loss_loss_"

    output_root = Path(args.output_root).resolve()
    sigma = 0.02  # Fixed sigma for this experiment
    
    rows = []
    scenario_ids = [1, 2, 3]
    for packet_loss_prob in PACKET_LOSS_PROB:
        # for scenario_id in [1, 2, 3]:  # Run each TDMA slot length on all three scenarios
        # Slot folders are named like figure8_delay_no_loss_tdma_slot_5p0.
        slot_str = f"{packet_loss_prob:.1f}".replace("0.", "")
        dataset_dir = Path(f"{dataset_dir_without_loss}0p{slot_str}").resolve()
        print(f"Running packet_loss_prob={packet_loss_prob}...", flush=True)
        print(f"Dataset dir: {dataset_dir}", flush=True)
        run_root = _run_scenario(dataset_dir, output_root, [1,2,3], sigma, packet_loss_prob, args.startup_delay, TRUE_RANGE_LINEAR_TURNS if TRAJECTORY_NAME == "linear_turns" else TRUE_RANGE_FIGURE8)    
        stats_rows = _load_stats(run_root, scenario_ids)

        for stats in stats_rows:
            row = _build_result_row(
                stats,
                sigma=sigma,
                tdma_interval=TDMA_SLOT_LENGTH,
                packet_loss_prob=packet_loss_prob,
                experiment=args.experiment,
                estimator=args.estimator,
                divergence_threshold=args.divergence_threshold,
                )
            rows.append(row)

    df = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_root / "experiment4_runs_complete.csv", index=False)

    print(f"Done. Results saved in {output_root}")
    return 0


if __name__ == "__main__":
  raise SystemExit(main())

