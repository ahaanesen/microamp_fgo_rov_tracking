#!/usr/bin/env python3
"""
Export one self-contained FGO dataset folder containing both:
  - a ROS 2 MCAP bag for replay into another ROS 2/FGO node
  - ASV/ROV ground-truth CSV files for plotting/error analysis

The resulting folder has the structure expected by the other ROS 2 node, e.g.:

  <out>/
    asv_ground_truth.csv
    rov_ground_truth.csv
    gt_metadata.json
    metadata.yaml
    <bag_name>_0.mcap

Usage example:

PYTHONPATH=src:$PYTHONPATH python3 export_fgo_dataset_combined.py \
    --out /tmp/circular_no_delay_no_loss \
    --duration 300 \
    --dt 0.01 \
    --seed 42 \
    --trajectory-type circular \
    --rov-id 1 \
    --epoch-sec 1700000000 \
    --datum-lat 60.3913 \
    --datum-lon 5.3221 \
    --datum-h 0.0 \
    --usbl-rate 0.2 \
    --range-rate 0.2 \
    --depth-rate 0.2 \
    --write-acoustic-rx false \
    --acoustic-delay false \
    --acoustic-jitter-std 0.1 \
    --usbl-miss-prob 0.1 \
    --range-miss-prob 0.1 \
    --depth-miss-prob 0.0 \
    --overwrite

Exit docker enviroment to copy from docker/tmp to local folder:
docker cp eskf_humble:/tmp/circular_no_delay_no_loss ./datasets/circular_no_delay_no_loss
"""

import argparse
import csv
import json
import math
import os
import shutil
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

import rclpy
from rclpy.serialization import serialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

from builtin_interfaces.msg import Time as TimeMsg
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3, Point

# ESKF repo imports
from generate_trajectories import generate_trajectories
from generate_measurements import MeasurementGenerator


# ----------------------------
# WGS84 helpers (NED <-> LLA)
# ----------------------------

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def lla_to_ecef(lat_rad: float, lon_rad: float, h_m: float) -> np.ndarray:
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)

    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + h_m) * cos_lat * cos_lon
    y = (n + h_m) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + h_m) * sin_lat
    return np.array([x, y, z], dtype=float)


def ecef_to_lla(x: float, y: float, z: float) -> Tuple[float, float, float]:
    # Iterative solution, stable for this local conversion use case.
    lon = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    h = 0.0
    for _ in range(8):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        h = p / max(math.cos(lat), 1e-12) - n
        lat = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
    return lat, lon, h


def ned_to_ecef_delta(n: float, e: float, d: float, lat0_rad: float, lon0_rad: float) -> np.ndarray:
    s_lat = math.sin(lat0_rad)
    c_lat = math.cos(lat0_rad)
    s_lon = math.sin(lon0_rad)
    c_lon = math.cos(lon0_rad)

    # NED -> ECEF. This is the corrected transpose direction for exporting GNSS.
    r_ned_to_ecef = np.array(
        [
            [-s_lat * c_lon, -s_lon, -c_lat * c_lon],
            [-s_lat * s_lon, c_lon, -c_lat * s_lon],
            [c_lat, 0.0, -s_lat],
        ],
        dtype=float,
    )
    return r_ned_to_ecef @ np.array([n, e, d], dtype=float)


def ned_to_lla(n: float, e: float, d: float, lat0_deg: float, lon0_deg: float, h0_m: float) -> Tuple[float, float, float]:
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    ecef0 = lla_to_ecef(lat0, lon0, h0_m)
    ecef = ecef0 + ned_to_ecef_delta(n, e, d, lat0, lon0)
    lat, lon, h = ecef_to_lla(float(ecef[0]), float(ecef[1]), float(ecef[2]))
    return math.degrees(lat), math.degrees(lon), h


# ----------------------------
# ROS / dataset time helpers
# ----------------------------


def sim_time_to_ros_time(sim_t: float, epoch_sec: int) -> TimeMsg:
    t = epoch_sec + sim_t
    sec = int(t)
    nsec = int((t - sec) * 1e9)
    msg = TimeMsg()
    msg.sec = sec
    msg.nanosec = nsec
    return msg


def sim_time_to_ros_stamp(sim_t: float, epoch_sec: int) -> Tuple[int, int]:
    t = epoch_sec + sim_t
    sec = int(t)
    nsec = int((t - sec) * 1e9)
    return sec, nsec


def to_ns(sim_t: float, epoch_sec: int) -> int:
    return int((epoch_sec + sim_t) * 1e9)


# ----------------------------
# Data utilities / senfuslib compatibility
# ----------------------------


def as_pairs(ts):
    """Convert senfuslib TimeSequence-like objects to (t, value) pairs."""
    if hasattr(ts, "items"):
        return list(ts.items())
    if hasattr(ts, "_items"):
        return list(ts._items)
    if hasattr(ts, "data"):
        return list(ts.data)
    return list(ts)


def meas_as_array(m) -> np.ndarray:
    """Convert senfuslib NamedArray-like measurements to a flat numpy array."""
    return np.asarray(m, dtype=float).reshape(-1)


def yaw_from_rotmat(r_nb: np.ndarray) -> float:
    return float(math.atan2(r_nb[1, 0], r_nb[0, 0]))


def str2bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "yes", "y", "on")


@dataclass
class Event:
    t: float
    kind: str
    payload: object


# ----------------------------
# Ground-truth CSV export
# ----------------------------


def write_ground_truth_csvs(out_dir: str, asv_tseq, rov_tseq, args) -> None:
    asv_csv = os.path.join(out_dir, "asv_ground_truth.csv")
    rov_csv = os.path.join(out_dir, "rov_ground_truth.csv")
    meta_json = os.path.join(out_dir, "gt_metadata.json")

    with open(asv_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "t_sim_sec",
                "t_ros_sec",
                "t_ros_ns",
                "x_n",
                "y_e",
                "z_d",
                "vx_n",
                "vy_e",
                "vz_d",
                "yaw_rad",
            ]
        )
        for t, s in asv_tseq.items():
            r_nb = s.ori.as_rotmat()
            yaw = yaw_from_rotmat(r_nb)
            t_ros_sec, t_ros_nsec = sim_time_to_ros_stamp(float(t), args.epoch_sec)
            t_ns = to_ns(float(t), args.epoch_sec)
            w.writerow(
                [
                    float(t),
                    t_ros_sec + t_ros_nsec * 1e-9,
                    t_ns,
                    float(s.pos[0]),
                    float(s.pos[1]),
                    float(s.pos[2]),
                    float(s.vel[0]),
                    float(s.vel[1]),
                    float(s.vel[2]),
                    yaw,
                ]
            )

    with open(rov_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "t_sim_sec",
                "t_ros_sec",
                "t_ros_ns",
                "rov_id",
                "x_n",
                "y_e",
                "z_d",
                "vx_n",
                "vy_e",
                "vz_d",
            ]
        )
        for t, s in rov_tseq.items():
            t_ros_sec, t_ros_nsec = sim_time_to_ros_stamp(float(t), args.epoch_sec)
            t_ns = to_ns(float(t), args.epoch_sec)
            w.writerow(
                [
                    float(t),
                    t_ros_sec + t_ros_nsec * 1e-9,
                    t_ns,
                    int(args.rov_id),
                    float(s.pos[0]),
                    float(s.pos[1]),
                    float(s.pos[2]),
                    float(s.vel[0]),
                    float(s.vel[1]),
                    float(s.vel[2]),
                ]
            )

    with open(meta_json, "w") as f:
        json.dump(
            {
                "duration_sec": args.duration,
                "dt_sec": args.dt,
                "seed": args.seed,
                "trajectory_type": args.trajectory_type,
                "rov_id": args.rov_id,
                "epoch_sec": args.epoch_sec,
                "frame": "NED",
                "datum": {
                    "lat_deg": args.datum_lat,
                    "lon_deg": args.datum_lon,
                    "h_m": args.datum_h,
                },
                "measurement_timing": {
                    "acoustic_delay": str2bool(args.acoustic_delay),
                    "acoustic_jitter_std_sec": args.acoustic_jitter_std,
                    "usbl_miss_prob": args.usbl_miss_prob,
                    "range_miss_prob": args.range_miss_prob,
                    "depth_miss_prob": args.depth_miss_prob,
                    "sound_speed_mps": args.sound_speed,
                },
                "timestamp_alignment": {
                    "t_ros_sec_definition": "epoch_sec + t_sim_sec",
                    "t_ros_ns_definition": "int((epoch_sec + t_sim_sec) * 1e9)",
                    "stamp_split": "sec=int(t), nsec=int((t-sec)*1e9)",
                },
                "notes": "Ground-truth CSVs and MCAP bag were generated in one run from the same trajectories and seed.",
            },
            f,
            indent=2,
        )

    print(f"[OK] Wrote {asv_csv}")
    print(f"[OK] Wrote {rov_csv}")
    print(f"[OK] Wrote {meta_json}")


# ----------------------------
# ROS bag export
# ----------------------------


def write_rosbag(out_dir: str, asv_gt, rov_gt, args) -> None:
    write_acoustic = str2bool(args.write_acoustic_rx)

    # Dynamic message classes avoid hard dependency at import time.
    imu_msg = get_message("sensor_msgs/msg/Imu")
    gnss_msg = get_message("blueboat_interfaces/msg/GNSSNavPvt")
    usbl_msg = get_message("blueboat_interfaces/msg/USBL")
    acoustic_msg = get_message("blueboat_interfaces/msg/AcousticCommReceive")

    mg = MeasurementGenerator(asv_gt, rov_gt)

    gnss_lever_arm = np.array([0.3, 0.3, 0.1], dtype=float)
    lever_arm = np.array([0.0, 0.0, 1.2], dtype=float)  # match env.usbl_offset_z default

    imu_seq = mg.generate_imu_asv(
        accm_std=args.imu_acc_std,
        gyro_std=args.imu_gyro_std,
        rate_hz=args.imu_rate,
    )
    gnss_seq = mg.generate_gnss_asv(
        std_ne=args.gnss_std_ne,
        std_d=args.gnss_std_d,
        lever_arm=gnss_lever_arm,
        rate_hz=args.gnss_rate,
    )
    acoustic_delay = str2bool(args.acoustic_delay)
    usbl_seq = mg.generate_usbl(
        std_rad=args.usbl_std_rad,
        lever_arm=lever_arm,
        rate_hz=args.usbl_rate,
        acoustic_delay=acoustic_delay,
        jitter_std=args.acoustic_jitter_std,
        miss_prob=args.usbl_miss_prob,
        sound_speed=args.sound_speed,
    )
    range_seq = mg.generate_range(
        std_m=args.range_std_m,
        lever_arm=lever_arm,
        rate_hz=args.range_rate,
        acoustic_delay=acoustic_delay,
        jitter_std=args.acoustic_jitter_std,
        miss_prob=args.range_miss_prob,
        sound_speed=args.sound_speed,
    )
    depth_seq = mg.generate_depth(
        std_m=args.depth_std_m,
        rate_hz=args.depth_rate,
        miss_prob=args.depth_miss_prob,
    )

    imu_pairs = as_pairs(imu_seq)
    gnss_pairs = as_pairs(gnss_seq)
    usbl_pairs = as_pairs(usbl_seq)
    range_pairs = as_pairs(range_seq)
    depth_pairs = as_pairs(depth_seq)

    range_times = np.array([t for t, _ in range_pairs], dtype=float)
    range_vals = [z for _, z in range_pairs]
    depth_times = np.array([t for t, _ in depth_pairs], dtype=float)
    depth_vals = [z for _, z in depth_pairs]

    if len(range_pairs) == 0:
        raise RuntimeError("No range measurements were generated. Reduce --range-miss-prob or increase duration/rate.")
    if len(depth_pairs) == 0:
        raise RuntimeError("No depth measurements were generated. Reduce --depth-miss-prob or increase duration/rate.")

    def nearest(times: np.ndarray, vals: List, t: float):
        idx = int(np.argmin(np.abs(times - t)))
        return vals[idx]

    events: List[Event] = []
    for t, z in imu_pairs:
        events.append(Event(float(t), "imu", z))
    for t, z in gnss_pairs:
        events.append(Event(float(t), "gnss", z))
    for t, z in usbl_pairs:
        events.append(Event(float(t), "usbl", z))
    events.sort(key=lambda e: e.t)

    storage_options = rosbag2_py.StorageOptions(uri=out_dir, storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    writer = rosbag2_py.SequentialWriter()
    writer.open(storage_options, converter_options)

    writer.create_topic(
        rosbag2_py.TopicMetadata(
            name=args.topic_imu,
            type="sensor_msgs/msg/Imu",
            serialization_format="cdr",
        )
    )
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            name=args.topic_gnss,
            type="blueboat_interfaces/msg/GNSSNavPvt",
            serialization_format="cdr",
        )
    )
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            name=args.topic_usbl,
            type="blueboat_interfaces/msg/USBL",
            serialization_format="cdr",
        )
    )
    if write_acoustic:
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=args.topic_acoustic,
                type="blueboat_interfaces/msg/AcousticCommReceive",
                serialization_format="cdr",
            )
        )

    for e in events:
        stamp = sim_time_to_ros_time(e.t, args.epoch_sec)
        stamp_ns = to_ns(e.t, args.epoch_sec)

        if e.kind == "imu":
            m = imu_msg()
            m.header = Header()
            m.header.stamp = stamp
            m.header.frame_id = "base_link"
            m.linear_acceleration = Vector3(
                x=float(e.payload.acc[0]),
                y=float(e.payload.acc[1]),
                z=float(e.payload.acc[2]),
            )
            m.angular_velocity = Vector3(
                x=float(e.payload.avel[0]),
                y=float(e.payload.avel[1]),
                z=float(e.payload.avel[2]),
            )
            writer.write(args.topic_imu, serialize_message(m), stamp_ns)

        elif e.kind == "gnss":
            ned_pos = meas_as_array(e.payload)
            n, e_m, d = float(ned_pos[0]), float(ned_pos[1]), float(ned_pos[2])
            lat, lon, h = ned_to_lla(n, e_m, d, args.datum_lat, args.datum_lon, args.datum_h)

            m = gnss_msg()
            m.header = Header()
            m.header.stamp = stamp
            m.header.frame_id = "gnss_link"
            m.lat = lat
            m.lon = lon
            m.height = h
            m.fix_type = 3
            m.gnss_fix_ok = True
            m.h_acc = args.gnss_std_ne
            if hasattr(m, "v_acc"):
                m.v_acc = args.gnss_std_d
            writer.write(args.topic_gnss, serialize_message(m), stamp_ns)

        elif e.kind == "usbl":
            usbl_arr = meas_as_array(e.payload)
            az_rad = float(usbl_arr[0])
            el_rad = float(usbl_arr[1])

            rng = nearest(range_times, range_vals, e.t)
            dep = nearest(depth_times, depth_vals, e.t)
            rng_m = float(meas_as_array(rng)[0])
            dep_m = float(meas_as_array(dep)[0])

            tof_sec = max(rng_m / args.sound_speed, 0.0)
            if acoustic_delay:
                # The new MeasurementGenerator can timestamp USBL at reception time.
                # Recover an approximate transmit time for the message fields.
                t_recv_sec = float(args.epoch_sec) + float(e.t)
                t_sent_sec = t_recv_sec - tof_sec
            else:
                # Keep the previous behaviour: bag/header time is the measurement epoch,
                # while acoustic modem fields still represent sent/received UTC-like times.
                t_sent_sec = float(args.epoch_sec) + float(e.t)
                t_recv_sec = t_sent_sec + tof_sec
            t_sent_us = int(round(t_sent_sec * 1e6))
            t_recv_us = int(round(t_recv_sec * 1e6))

            m = usbl_msg()
            m.header = Header()
            m.header.stamp = stamp
            m.header.frame_id = "usbl_link"
            m.rov_id = int(args.rov_id)
            m.azimuth = math.degrees(az_rad)
            m.elevation = math.degrees(el_rad)
            m.t_sent = t_sent_us
            m.t_received = t_recv_us
            m.position = Vector3(x=0.0, y=0.0, z=dep_m)
            writer.write(args.topic_usbl, serialize_message(m), stamp_ns)

            if write_acoustic:
                a = acoustic_msg()
                a.header = Header()
                a.header.stamp = stamp
                a.header.frame_id = "acoustic_link"
                a.node_id = int(args.rov_id)
                a.t_sent = t_sent_us
                a.t_received = t_recv_us
                a.position = Point(x=0.0, y=0.0, z=dep_m)
                writer.write(args.topic_acoustic, serialize_message(a), stamp_ns)

    print(f"[OK] Wrote MCAP bag to: {out_dir}")
    print("[OK] Topics:")
    print(f"  - {args.topic_imu}")
    print(f"  - {args.topic_gnss}")
    print(f"  - {args.topic_usbl}")
    if write_acoustic:
        print(f"  - {args.topic_acoustic}")


# ----------------------------
# CLI
# ----------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a combined FGO ROS 2 MCAP bag and ground-truth CSV dataset folder."
    )
    parser.add_argument("--out", type=str, required=True, help="Output dataset folder")
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--dt", type=float, default=0.1, help="Trajectory dt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trajectory-type",
        type=str,
        default="circular",
        choices=("circular", "figure_8", "sinusoidal"),
        help="Trajectory generator mode passed to generate_trajectories().",
    )

    parser.add_argument("--datum-lat", type=float, default=60.3913)
    parser.add_argument("--datum-lon", type=float, default=5.3221)
    parser.add_argument("--datum-h", type=float, default=0.0)

    parser.add_argument("--sound-speed", type=float, default=1500.0)
    parser.add_argument("--rov-id", type=int, default=1)
    parser.add_argument("--epoch-sec", type=int, default=1700000000)

    parser.add_argument("--imu-rate", type=float, default=100.0)
    parser.add_argument("--gnss-rate", type=float, default=1.0)
    parser.add_argument("--usbl-rate", type=float, default=1.0)
    parser.add_argument("--depth-rate", type=float, default=1.0)
    parser.add_argument("--range-rate", type=float, default=1.0)

    parser.add_argument(
        "--acoustic-delay",
        type=str,
        default="false",
        help="If true, USBL/range timestamps are shifted by acoustic time-of-flight in the measurement generator.",
    )
    parser.add_argument("--acoustic-jitter-std", type=float, default=0.0, help="Gaussian timing jitter std [s] for USBL/range reception timestamps.")
    parser.add_argument("--usbl-miss-prob", type=float, default=0.0, help="Probability of dropping each USBL measurement.")
    parser.add_argument("--range-miss-prob", type=float, default=0.0, help="Probability of dropping each range measurement.")
    parser.add_argument("--depth-miss-prob", type=float, default=0.0, help="Probability of dropping each depth measurement.")

    parser.add_argument("--imu-acc-std", type=float, default=1.167e-3)
    parser.add_argument("--imu-gyro-std", type=float, default=4.36e-5)
    parser.add_argument("--gnss-std-ne", type=float, default=0.3)
    parser.add_argument("--gnss-std-d", type=float, default=0.5)
    parser.add_argument("--usbl-std-rad", type=float, default=0.01745)
    parser.add_argument("--range-std-m", type=float, default=0.5)
    parser.add_argument("--depth-std-m", type=float, default=0.3)

    parser.add_argument("--h-acc-mm", type=int, default=300)  # kept for CLI compatibility
    parser.add_argument("--v-acc-mm", type=int, default=500)  # kept for CLI compatibility

    parser.add_argument("--write-acoustic-rx", type=str, default="false")
    parser.add_argument("--overwrite", action="store_true", help="Delete --out first if it already exists")

    parser.add_argument("--topic-imu", type=str, default="microampere/imu/data")
    parser.add_argument("--topic-gnss", type=str, default="microampere/gnss/nav_pvt")
    parser.add_argument("--topic-usbl", type=str, default="microampere/sensors/usbl")
    parser.add_argument("--topic-acoustic", type=str, default="microampere/acoustic/receive")

    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.out):
        if not args.overwrite:
            raise FileExistsError(
                f"Output folder already exists: {args.out}. "
                "Use --overwrite or choose a new --out folder."
            )
        shutil.rmtree(args.out)

    parent = os.path.dirname(os.path.abspath(args.out))
    if parent:
        os.makedirs(parent, exist_ok=True)

    np.random.seed(args.seed)

    # Generate trajectories once so bag and ground-truth CSVs are guaranteed to match.
    asv_gt, rov_gt, _imu_gt = generate_trajectories(
        duration=args.duration,
        dt=args.dt,
        trajectory_type=args.trajectory_type,
    )

    # rosbag2 creates args.out and writes metadata.yaml + *.mcap into it.
    write_rosbag(args.out, asv_gt, rov_gt, args)

    # Add plotting ground-truth files into the exact same dataset folder.
    write_ground_truth_csvs(args.out, asv_gt, rov_gt, args)

    print("[OK] Combined dataset folder contents should now include:")
    print("  - metadata.yaml")
    print("  - *.mcap")
    print("  - asv_ground_truth.csv")
    print("  - rov_ground_truth.csv")
    print("  - gt_metadata.json")


if __name__ == "__main__":
    rclpy.init()
    try:
        main()
    finally:
        rclpy.shutdown()
