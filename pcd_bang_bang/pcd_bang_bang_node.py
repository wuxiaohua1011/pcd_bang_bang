#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from carla_msgs.msg import CarlaEgoVehicleControl
from sensor_msgs.msg import PointCloud2
import socket
import numpy as np


class PCDBangBangNode(Node):
    def __init__(self):
        super().__init__("pcd_bang_bang_node")
        self.subscription = self.create_subscription(
            PointCloud2,
            "/pointcloud",
            self.on_pcd_received,
            1,
        )
        self.subscription  # prevent unused variable warning

    def on_pcd_received(self, msg):
        # self.get_logger().info("hi")
        pass


def main(args=None):
    rclpy.init(args=args)

    node = PCDBangBangNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
