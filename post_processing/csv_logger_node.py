#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import csv
from pathlib import Path
from datetime import datetime

# Import your message types
from blueboat_interfaces.msg import BoatState, ROVState

# Topics to subscribe to
boat_state_topic = "/state/boat"
rov_state_topic = "/state/rov"

# Path to save CSV files (optional, can be current directory)
csv_path = Path("microampere_ros2ws/src/microamp_fgo_rov_tracking/post_processing/estimated_data/scenario3")

# This node subscribes to the ASV and ROV state topics and logs the data to CSV files for later analysis and plotting.
# Run directly in python as is not packaged
class CSVLogger(Node):

    def __init__(self):
        super().__init__('csv_logger')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path.mkdir(parents=True, exist_ok=True)

        # Open CSV files
        self.rov_file = open(csv_path / f'rov_estimated_{timestamp}.csv', 'w', newline='')
        self.rov_writer = csv.writer(self.rov_file)
        self.rov_writer.writerow([
            'time', 'rov_id', 'x', 'y', 'z', 'vx', 'vy', 'vz'
        ])

        self.boat_file = open(csv_path / f'boat_estimated_{timestamp}.csv', 'w', newline='')
        self.boat_writer = csv.writer(self.boat_file)
        self.boat_writer.writerow([
            'time', 'x', 'y', 'yaw', 'surge', 'sway', 'yaw_rate'
        ])

        # Subscribers
        self.create_subscription(
            ROVState,
            rov_state_topic,
            self.rov_callback,
            10
        )

        self.create_subscription(
            BoatState,
            boat_state_topic,
            self.boat_callback,
            10
        )

        self.get_logger().info("CSV Logger started.")

    def rov_callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        self.rov_writer.writerow([
            t,
            msg.rov_id,
            msg.position.x,
            msg.position.y,
            msg.position.z,
            msg.velocity.x,
            msg.velocity.y,
            msg.velocity.z
        ])

    def boat_callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        self.boat_writer.writerow([
            t,
            msg.x,
            msg.y,
            msg.yaw,
            msg.surge,
            msg.sway,
            msg.yaw_r
        ])

    def destroy_node(self):
        self.rov_file.close()
        self.boat_file.close()
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
