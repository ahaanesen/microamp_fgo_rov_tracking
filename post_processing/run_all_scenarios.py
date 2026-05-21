#!/usr/bin/env python3

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import yaml

from plotting import PlotterCSVJoint


""" 
Navigate to /ros2_ws/microampere_ros2ws and run:
    source /opt/ros/humble/setup.bash
    source install/setup.bash

  python3 src/microamp_fgo_rov_tracking/post_processing/run_all_scenarios.py \
  --dataset-dir src/microamp_fgo_rov_tracking/post_processing/simulation_data/8a \
  --output-root src/microamp_fgo_rov_tracking/post_processing/batch_runs \
  --run-name 8_cv_0_02 \
  --startup-delay 2.0

cd /ros2_ws/microampere_ros2ws
source /opt/ros/humble/setup.bash
source install/setup.bash

python3 src/microamp_fgo_rov_tracking/post_processing/run_all_scenarios.py \
  --dataset-dir src/microamp_fgo_rov_tracking/post_processing/simulation_data/circular_delay_loss_tdma \
  --output-root src/microamp_fgo_rov_tracking/post_processing/batch_runs \
  --run-name d2105_t1640_circular_d_l_CHOL_relin5_CV_0_02_cv_fac_tdma_rovPriorVelSigma_0_1 \
  --startup-delay 2.0 \
  --fgo-params use_rov_depth_prior=false rov_cv_continous_sigma=0.02 rov_prior_vel_sigma=0.1 \
 """

SCENARIOS = {
    1: "FGO - Scenario 1: Bearing-only",
    2: "FGO - Scenario 2: Bearing + range",
    3: "FGO - Scenario 3: Bearing + range + depth",
}

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET = PACKAGE_ROOT / "post_processing" / "simulation_data" / "sinusoidal_no_delay_no_loss"
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "post_processing" / "batch_runs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FGO scenarios 1-3 on the same bag dataset and plot the results."
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_DATASET),
        help="Directory containing metadata.yaml, the rosbag file, and ground-truth CSVs.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where per-scenario CSVs, plots, and summary statistics will be written.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run folder name. Defaults to the dataset directory name.",
    )
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=2.0,
        help="Seconds to wait after starting the ROS node and logger before playing the bag.",
    )
    parser.add_argument(
        "--fgo-params",
        nargs="+",
        default=[],
        help="FGO parameter overrides as key=value pairs (e.g., accel_noise=0.001 gyro_noise=0.00005)",
    )
    parser.add_argument(
        "--fgo-config-override",
        type=str,
        default=None,
        help="Path to a YAML file with FGO parameter overrides (merged with --fgo-params)",
    )
    return parser.parse_args()


def parse_fgo_params(fgo_params_list: list[str], config_override_path: str | None = None) -> dict[str, str]:
    """
    Parse FGO parameter overrides from command line and optional config file.
    Returns a dict of param_name -> param_value strings suitable for ROS -p flags.
    """
    params = {}
    
    # Parse from config override file if provided
    if config_override_path:
        config_override_path = Path(config_override_path)
        if not config_override_path.exists():
            raise FileNotFoundError(f"FGO config override file not found: {config_override_path}")
        try:
            config_data = yaml.safe_load(config_override_path.read_text())
            # Support nested fgo: { param: value } or flat param: value
            if "fgo" in config_data and isinstance(config_data["fgo"], dict):
                for key, val in config_data["fgo"].items():
                    params[key] = str(val)
            else:
                for key, val in config_data.items():
                    params[key] = str(val)
        except Exception as e:
            raise ValueError(f"Failed to parse FGO config override file: {e}")
    
    # Parse from command-line key=value pairs (these override config file)
    for param_pair in fgo_params_list:
        if "=" not in param_pair:
            raise ValueError(f"FGO parameter must be in format key=value, got: {param_pair}")
        key, val = param_pair.split("=", 1)
        params[key.strip()] = val.strip()
    
    return params


def load_bag_path(dataset_dir: Path) -> Path:
    # relative path: microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/simulation_data/8a/metadata.yaml
    # python expects: /ros2_ws/src/microamp_fgo_rov_tracking/post_processing/simulation_data/8a/metadata.yaml
    metadata_path = dataset_dir / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.yaml in {dataset_dir}")

    metadata = yaml.safe_load(metadata_path.read_text())
    relative_paths = metadata["rosbag2_bagfile_information"]["relative_file_paths"]
    if not relative_paths:
        raise ValueError(f"No bag files listed in {metadata_path}")

    bag_file = dataset_dir / relative_paths[0]
    if not bag_file.exists():
        raise FileNotFoundError(f"Bag file referenced by metadata does not exist: {bag_file}")
    return dataset_dir


def check_ground_truth_files(dataset_dir: Path) -> tuple[Path, Path]:
    rov_gt = dataset_dir / "rov_ground_truth.csv"
    asv_gt = dataset_dir / "asv_ground_truth.csv"
    if not rov_gt.exists() or not asv_gt.exists():
        raise FileNotFoundError(
            f"Expected ground-truth CSVs in {dataset_dir}, found rov={rov_gt.exists()} asv={asv_gt.exists()}"
        )
    return rov_gt, asv_gt


def make_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"
    return env


def start_process(cmd: list[str], env: dict[str, str], cwd: Path) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def stop_process(proc: subprocess.Popen | None, timeout: float = 10.0) -> None:
    if proc is None or proc.poll() is not None:
        return

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=timeout)


def wait_for_process(proc: subprocess.Popen, name: str) -> None:
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"{name} exited with code {return_code}")


def run_single_scenario(
    scenario_id: int,
    scenario_name: str,
    bag_path: Path,
    rov_gt_csv: Path,
    asv_gt_csv: Path,
    output_root: Path,
    startup_delay: float,
    fgo_params: dict[str, str] | None = None,
) -> dict[str, str]:
    env = make_env()
    scenario_dir = output_root / f"scenario{scenario_id}"
    csv_dir = scenario_dir / "estimated_data"
    plot_dir = scenario_dir / "plots"
    run_name = f"s{scenario_id}"

    scenario_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    tracker_cmd = [
        "ros2",
        "run",
        "microamp_fgo_rov_tracking",
        "fgo_tracking_node",
        "--ros-args",
        "-p",
        f"scenario_id:={scenario_id}",
        "-p",
        "use_sim_time:=true",
    ]
    
    # Add FGO parameter overrides
    if fgo_params:
        for param_name, param_value in fgo_params.items():
            tracker_cmd.extend(["-p", f"fgo.{param_name}:={param_value}"])
    logger_cmd = [
        "python3",
        str(SCRIPT_DIR / "csv_logger_node.py"),
        "--output-dir",
        str(csv_dir),
        "--run-name",
        run_name,
    ]
    bag_cmd = [
        "ros2",
        "bag",
        "play",
        str(bag_path),
        "--clock",
        "--rate",
        "10",
    ]

    tracker_proc = None
    logger_proc = None
    bag_proc = None

    try:
        tracker_proc = start_process(tracker_cmd, env, PACKAGE_ROOT.parent.parent)
        logger_proc = start_process(logger_cmd, env, PACKAGE_ROOT.parent.parent)
        time.sleep(startup_delay)

        if tracker_proc.poll() is not None:
            raise RuntimeError(f"Tracker node exited early with code {tracker_proc.returncode}")
        if logger_proc.poll() is not None:
            raise RuntimeError(f"CSV logger exited early with code {logger_proc.returncode}")

        bag_proc = start_process(bag_cmd, env, PACKAGE_ROOT.parent.parent)
        wait_for_process(bag_proc, f"ros2 bag play for scenario {scenario_id}")
        time.sleep(1.0)
    finally:
        stop_process(logger_proc)
        stop_process(tracker_proc)

    rov_est_csv = csv_dir / f"rov_estimated_{run_name}.csv"
    asv_est_csv = csv_dir / f"boat_estimated_{run_name}.csv"
    gnss_csv = csv_dir / f"gnss_{run_name}.csv"
    usbl_csv = csv_dir / f"usbl_{run_name}.csv"

    if not rov_est_csv.exists() or not asv_est_csv.exists():
        raise FileNotFoundError(
            f"Expected estimate CSVs were not created for scenario {scenario_id}: "
            f"{rov_est_csv}, {asv_est_csv}"
        )

    plotter = PlotterCSVJoint(
        rov_gt_csv=str(rov_gt_csv),
        asv_gt_csv=str(asv_gt_csv),
        rov_est_csv=str(rov_est_csv),
        asv_est_csv=str(asv_est_csv),
        scenario_name=scenario_name,
        save_dir=str(plot_dir),
        gnss_csv=str(gnss_csv) if gnss_csv.exists() else None,
        usbl_csv=str(usbl_csv) if usbl_csv.exists() else None,
        scenario_id=scenario_id,
        gps_sigma_ne=float(fgo_params.get("gps_sigma_ne", 0.3)) if fgo_params else 0.3,
        gps_sigma_d=float(fgo_params.get("gps_sigma_d",  0.5))  if fgo_params else 0.5,
    )
    _, stats_df = plotter.save_all()

    stats_df = stats_df.copy()
    stats_df["scenario_id"] = scenario_id
    stats_df.to_csv(plot_dir / "fgo_statistics.csv", index=False)

    return {
        "scenario_id": str(scenario_id),
        "scenario_name": scenario_name,
        "rov_est_csv": str(rov_est_csv),
        "asv_est_csv": str(asv_est_csv),
        "plot_dir": str(plot_dir),
    }


def main() -> int:
    args = parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_root = Path(args.output_root).resolve()
    run_name = args.run_name or dataset_dir.name
    run_root = output_root / run_name

    bag_path = load_bag_path(dataset_dir)
    rov_gt_csv, asv_gt_csv = check_ground_truth_files(dataset_dir)

    # Parse FGO parameter overrides
    fgo_params = parse_fgo_params(args.fgo_params, args.fgo_config_override)
    if fgo_params:
        print(f"FGO parameter overrides: {fgo_params}", flush=True)

    run_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for i, (scenario_id, scenario_name) in enumerate(SCENARIOS.items()):
        if i > 0:
            time.sleep(5.0)  # Let DDS deregister dead nodes before starting the next scenario
        print(f"Running scenario {scenario_id}: {scenario_name}", flush=True)
        scenario_summary = run_single_scenario(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            bag_path=bag_path,
            rov_gt_csv=rov_gt_csv,
            asv_gt_csv=asv_gt_csv,
            output_root=run_root,
            startup_delay=args.startup_delay,
            fgo_params=fgo_params,
        )
        summary_rows.append(scenario_summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_root / "run_summary.csv", index=False)

    stats_frames = []
    consistency_frames = []
    for scenario_id in SCENARIOS:
        stat_path = run_root / f"scenario{scenario_id}" / "plots" / "fgo_statistics.csv"
        stats_frames.append(pd.read_csv(stat_path))
        consistency_path = run_root / f"scenario{scenario_id}" / "plots" / "consistency_statistics.csv"
        if consistency_path.exists():
            consistency_frames.append(pd.read_csv(consistency_path))
    pd.concat(stats_frames, ignore_index=True).to_csv(run_root / "all_statistics.csv", index=False)
    if consistency_frames:
        pd.concat(consistency_frames, ignore_index=True).to_csv(
            run_root / "all_consistency_statistics.csv",
            index=False,
        )

    print(f"Finished. Outputs saved in {run_root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
