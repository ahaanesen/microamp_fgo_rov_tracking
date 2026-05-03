#!/usr/bin/env python3

import argparse
import math

import rclpy
from rclpy.node import Node
import csv
from pathlib import Path
from datetime import datetime

from nav_msgs.msg import Odometry
from blueboat_interfaces.msg import ROVState, GNSSNavPvt, USBL

""" Usage example:
    python3 src/microamp_fgo_rov_tracking/post_processing/csv_logger_node.py --output-dir src/microamp_fgo_rov_tracking/post_processing/debugging/test --run-name my_experiment_01
"""

boat_state_topic = "/state/boat"
rov_state_topic = "/state/rov"
gnss_topic = "microampere/gnss/nav_pvt"
usbl_topic = "microampere/sensors/usbl"


class _NedConverter:
    """WGS-84 ECEF → local NED, matching NedConverter.hpp."""
    _A = 6378137.0
    _E2 = (1.0 / 298.257223563) * (2.0 - 1.0 / 298.257223563)

    def __init__(self):
        self._datum_set = False
        self._sin_lat0 = self._cos_lat0 = 0.0
        self._sin_lon0 = self._cos_lon0 = 0.0
        self._x0 = self._y0 = self._z0 = 0.0

    def set_datum(self, lat_deg, lon_deg, h_m):
        lat0 = math.radians(lat_deg)
        lon0 = math.radians(lon_deg)
        self._sin_lat0 = math.sin(lat0)
        self._cos_lat0 = math.cos(lat0)
        self._sin_lon0 = math.sin(lon0)
        self._cos_lon0 = math.cos(lon0)
        N0 = self._A / math.sqrt(1.0 - self._E2 * self._sin_lat0 ** 2)
        self._x0 = (N0 + h_m) * self._cos_lat0 * self._cos_lon0
        self._y0 = (N0 + h_m) * self._cos_lat0 * self._sin_lon0
        self._z0 = (N0 * (1.0 - self._E2) + h_m) * self._sin_lat0
        self._datum_set = True

    def gnss_to_ned(self, lat_deg, lon_deg, h_m):
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        sl, cl = math.sin(lat), math.cos(lat)
        slo, clo = math.sin(lon), math.cos(lon)
        N = self._A / math.sqrt(1.0 - self._E2 * sl ** 2)
        x = (N + h_m) * cl * clo
        y = (N + h_m) * cl * slo
        z = (N * (1.0 - self._E2) + h_m) * sl
        dx, dy, dz = x - self._x0, y - self._y0, z - self._z0
        n = -self._sin_lat0*self._cos_lon0*dx - self._sin_lat0*self._sin_lon0*dy + self._cos_lat0*dz
        e = -self._sin_lon0*dx + self._cos_lon0*dy
        d = -self._cos_lat0*self._cos_lon0*dx - self._cos_lat0*self._sin_lon0*dy - self._sin_lat0*dz
        return n, e, d


class CSVLogger(Node):
    def __init__(self, output_dir: Path, run_name: str | None = None):
        super().__init__('csv_logger')

        timestamp = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)

        self.rov_path  = output_dir / f'rov_estimated_{timestamp}.csv'
        self.asv_path  = output_dir / f'boat_estimated_{timestamp}.csv'
        self.gnss_path = output_dir / f'gnss_{timestamp}.csv'
        self.usbl_path = output_dir / f'usbl_{timestamp}.csv'

        self.rov_file = open(self.rov_path, 'w', newline='')
        self.rov_writer = csv.writer(self.rov_file)
        self.rov_writer.writerow([
            'time', 'rov_id', 'x', 'y', 'z', 'vx', 'vy', 'vz',
            'pos_cov_xx', 'pos_cov_xy', 'pos_cov_xz',
            'pos_cov_yy', 'pos_cov_yz', 'pos_cov_zz',
        ])

        self.boat_file = open(self.asv_path, 'w', newline='')
        self.boat_writer = csv.writer(self.boat_file)
        self.boat_writer.writerow([
            'time', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'yaw', 'yaw_rate',
            'pos_cov_xx', 'pos_cov_xy', 'pos_cov_xz',
            'pos_cov_yy', 'pos_cov_yz', 'pos_cov_zz',
        ])

        self.gnss_file = open(self.gnss_path, 'w', newline='')
        self.gnss_writer = csv.writer(self.gnss_file)
        self.gnss_writer.writerow([
            'time', 'lat_deg', 'lon_deg', 'h_m', 'ned_n', 'ned_e', 'ned_d', 'h_acc_m', 'v_acc_m',
        ])

        self.usbl_file = open(self.usbl_path, 'w', newline='')
        self.usbl_writer = csv.writer(self.usbl_file)
        self.usbl_writer.writerow([
            'time', 'rov_id', 'message_index', 'position_x', 'position_y', 'position_z',
            't_sent_us', 't_received_us', 't_sent_sec', 't_received_sec', 'tof_sec',
            'azimuth_deg', 'elevation_deg',
        ])

        self._ned = _NedConverter()

        self.create_subscription(ROVState, rov_state_topic, self.rov_callback, 10)
        self.create_subscription(Odometry, boat_state_topic, self.boat_callback, 10)
        self.create_subscription(GNSSNavPvt, gnss_topic, self.gnss_callback, 10)
        self.create_subscription(USBL, usbl_topic, self.usbl_callback, 10)

        self.get_logger().info(
            f"CSV Logger started. ASV: {self.asv_path} | ROV: {self.rov_path} | GNSS: {self.gnss_path} | USBL: {self.usbl_path}"
        )

    def rov_callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        c = msg.position_covariance
        self.rov_writer.writerow([
            t, msg.rov_id,
            msg.position.x, msg.position.y, msg.position.z,
            msg.velocity.x, msg.velocity.y, msg.velocity.z,
            c[0], c[1], c[2],
            c[4], c[5],
            c[8],
        ])

    def boat_callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))

        # pose.covariance is a flat 36-element array, order [x, y, z, rx, ry, rz]
        # Position 3×3 upper triangle: indices 0,1,2 / 7,8 / 14
        c = msg.pose.covariance
        self.boat_writer.writerow([
            t,
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
            yaw,
            msg.twist.twist.angular.z,
            c[0],  c[1],  c[2],   # pos_cov_xx, pos_cov_xy, pos_cov_xz
            c[7],  c[8],          # pos_cov_yy, pos_cov_yz
            c[14],                # pos_cov_zz
        ])

    def gnss_callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if not self._ned._datum_set:
            self._ned.set_datum(msg.lat, msg.lon, msg.height)

        n, e, d = self._ned.gnss_to_ned(msg.lat, msg.lon, msg.height)
        self.gnss_writer.writerow([
            t, msg.lat, msg.lon, msg.height,
            n, e, d,
            msg.h_acc, msg.v_acc,
        ])

    def usbl_callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        t_sent_sec = msg.t_sent * 1e-6
        t_received_sec = msg.t_received * 1e-6
        self.usbl_writer.writerow([
            t,
            msg.rov_id,
            msg.message_index,
            msg.position.x,
            msg.position.y,
            msg.position.z,
            msg.t_sent,
            msg.t_received,
            t_sent_sec,
            t_received_sec,
            t_received_sec - t_sent_sec,
            msg.azimuth,
            msg.elevation,
        ])

    def destroy_node(self):
        self.rov_file.close()
        self.boat_file.close()
        self.gnss_file.close()
        self.usbl_file.close()
        super().destroy_node()


def parse_args():
    parser = argparse.ArgumentParser(description="Log ASV and ROV state estimates to CSV.")
    parser.add_argument(
        "--output-dir",
        default="microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/estimated_data/0_02cv_sigma",
        help="Directory where the estimate CSV files will be stored.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional suffix used in output filenames instead of a timestamp.",
    )
    return parser.parse_args()


def main(args=None):
    cli_args = parse_args()
    rclpy.init(args=args)
    node = CSVLogger(Path(cli_args.output_dir), cli_args.run_name)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
