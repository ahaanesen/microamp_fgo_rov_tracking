#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import csv
from pathlib import Path
from datetime import datetime

# Import your message types
from blueboat_interfaces.msg import GNSSNavPvt, USBL
from sensor_msgs.msg import Imu

# Topics to subscribe to
gnss_topic = "/microampere/gnss/nav_pvt"
imu_topic = "/microampere/imu/data"
usbl_topic = "/microampere/sensors/usbl"

# Path to save CSV files (optional, can be current directory)
csv_path = Path("microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/simulation_data/8")

# This node subscribes to the ASV and ROV state topics and logs the data to CSV files for later analysis and plotting.
# Run directly in python as is not packaged
class CSVLogger(Node):

    def __init__(self):
        super().__init__('csv_logger')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path.mkdir(parents=True, exist_ok=True)

        # Open CSV files
        self.gnss_file = open(csv_path / f'gnss_{timestamp}.csv', 'w', newline='')
        self.gnss_writer = csv.writer(self.gnss_file)
        self.gnss_writer.writerow([
            'time', 'latitude', 'longitude', 'altitude', 'fix_type', 'h_acc',
        ])

        self.imu_file = open(csv_path / f'imu_{timestamp}.csv', 'w', newline='')
        self.imu_writer = csv.writer(self.imu_file)
        self.imu_writer.writerow([
            'time', 'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
            'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z',
        ])

        self.usbl_file = open(csv_path / f'usbl_{timestamp}.csv', 'w', newline='')
        self.usbl_writer = csv.writer(self.usbl_file)
        self.usbl_writer.writerow([
            'time', 'rov_id', 't_sent', 't_received', 'x', 'y', 'z', 'channels', 'rssi', 'azimuth', 'elevation'
        ])

        # Subscribers
        self.create_subscription(
            GNSSNavPvt,
            gnss_topic,
            self.gnss_callback,
            10
        )
        self.create_subscription(
            Imu,
            imu_topic,
            self.imu_callback,
            10
        )
        self.create_subscription(
            USBL,
            usbl_topic,
            self.usbl_callback,
            10
        )

        self.get_logger().info("CSV Logger started.")

    def gnss_callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        self.gnss_writer.writerow([
            t,
            msg.lat,
            msg.lon,
            msg.height,
            msg.fix_type,
            msg.h_acc,
        ])

    def imu_callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        self.imu_writer.writerow([
            t,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ])

    def usbl_callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        self.usbl_writer.writerow([
            t,
            msg.rov_id,
            msg.t_sent,
            msg.t_received,
            msg.position.x,
            msg.position.y,
            msg.position.z,
            msg.channels,
            msg.rssi,
            msg.azimuth,
            msg.elevation
        ])

    def destroy_node(self):
        self.gnss_file.close()
        self.imu_file.close()
        self.usbl_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CSVLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
