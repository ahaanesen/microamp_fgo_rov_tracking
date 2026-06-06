#!/usr/bin/env bash
set -e

FIGURE8="figure8"
LINEAR_TURNS="linear_turns"

TRAJECTORIES = (
  FIGURE8,
  LINEAR_TURNS
)
PACKAGE_ROOT = SCRIPT_DIR.parent

for TRAJECTORY_NAME in "${TRAJECTORIES[@]}"; do
  echo "Processing trajectory: ${TRAJECTORY_NAME}"
  DEFAULT_DATASET = PACKAGE_ROOT / "post_processing" / "simulation_data" / f"{TRAJECTORY_NAME}_delay_no_loss_tdma_slot_5p0"
  DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT/"post_processing"/"results"/f"{TRAJECTORY_NAME}"/"exp3_tdma_sweep"

  python3 exp3_results.py --trajectory "${TRAJECTORY_NAME}" --dataset-dir "${DEFAULT_DATASET}" --output-root "${DEFAULT_OUTPUT_ROOT}" --startup-delay 2.0
  python3 exp4_results.py --trajectory "${TRAJECTORY_NAME}" --dataset-dir "${DEFAULT_DATASET}" --output-root "${DEFAULT_OUTPUT_ROOT}" --startup-delay 2.0
done



