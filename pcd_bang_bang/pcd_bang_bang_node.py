#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from carla_msgs.msg import CarlaEgoVehicleControl
from sensor_msgs.msg import PointCloud2
import socket
import numpy as np
from typing import Optional, Tuple

## The code below is "ported" from
# https://github.com/ros/common_msgs/tree/noetic-devel/sensor_msgs/src/sensor_msgs
# I'll make an official port and PR to this repo later:
# https://github.com/ros2/common_interfaces
import sys
from collections import namedtuple
import ctypes
import math
import struct
from sensor_msgs.msg import PointCloud2, PointField
import cv2

_DATATYPES = {}
_DATATYPES[PointField.INT8] = ("b", 1)
_DATATYPES[PointField.UINT8] = ("B", 1)
_DATATYPES[PointField.INT16] = ("h", 2)
_DATATYPES[PointField.UINT16] = ("H", 2)
_DATATYPES[PointField.INT32] = ("i", 4)
_DATATYPES[PointField.UINT32] = ("I", 4)
_DATATYPES[PointField.FLOAT32] = ("f", 4)
_DATATYPES[PointField.FLOAT64] = ("d", 8)


def read_points(cloud, field_names=None, skip_nans=False, uvs=[]):
    """
    Read points from a L{sensor_msgs.PointCloud2} message.
    @param cloud: The point cloud to read from.
    @type  cloud: L{sensor_msgs.PointCloud2}
    @param field_names: The names of fields to read. If None, read all fields. [default: None]
    @type  field_names: iterable
    @param skip_nans: If True, then don't return any point with a NaN value.
    @type  skip_nans: bool [default: False]
    @param uvs: If specified, then only return the points at the given coordinates. [default: empty list]
    @type  uvs: iterable
    @return: Generator which yields a list of values for each point.
    @rtype:  generator
    """
    assert isinstance(cloud, PointCloud2), "cloud is not a sensor_msgs.msg.PointCloud2"
    fmt = _get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
    width, height, point_step, row_step, data, isnan = (
        cloud.width,
        cloud.height,
        cloud.point_step,
        cloud.row_step,
        cloud.data,
        math.isnan,
    )
    unpack_from = struct.Struct(fmt).unpack_from

    if skip_nans:
        if uvs:
            for u, v in uvs:
                p = unpack_from(data, (row_step * v) + (point_step * u))
                has_nan = False
                for pv in p:
                    if isnan(pv):
                        has_nan = True
                        break
                if not has_nan:
                    yield p
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    p = unpack_from(data, offset)
                    has_nan = False
                    for pv in p:
                        if isnan(pv):
                            has_nan = True
                            break
                    if not has_nan:
                        yield p
                    offset += point_step
    else:
        if uvs:
            for u, v in uvs:
                yield unpack_from(data, (row_step * v) + (point_step * u))
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    yield unpack_from(data, offset)
                    offset += point_step


def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = ">" if is_bigendian else "<"

    offset = 0
    for field in (
        f
        for f in sorted(fields, key=lambda f: f.offset)
        if field_names is None or f.name in field_names
    ):
        if offset < field.offset:
            fmt += "x" * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print(
                "Skipping unknown PointField datatype [%d]" % field.datatype,
                file=sys.stderr,
            )
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt


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
        self.max_dist = 4
        # occupancy map
        self.scaling_factor = 100
        self.occu_map = np.zeros(
            shape=(
                math.ceil(self.max_dist * self.scaling_factor),
                math.ceil(self.max_dist * self.scaling_factor),
            ),
            dtype=np.float32,
        )
        self.cx = self.occu_map.shape[0] // 2
        self.cy = 0
        self.kernel_size = 2

    def on_pcd_received(self, msg):
        pcd_as_numpy_array = np.array(list(read_points(msg)))[:, :3]
        self.update_occu_map(pcd_as_numpy_array)
        # self.smoothen_occu_map()

        left, center, right = self.find_obstacle_l_c_r(
            self.occu_map,
            left_occ_thresh=0.2,
            center_occ_thresh=0.2,
            right_occ_thresh=0.2,
            debug=True,
        )

    def smoothen_occu_map(self):
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        self.occu_map = cv2.morphologyEx(
            self.occu_map, cv2.MORPH_OPEN, kernel
        )  # erosion followed by dilation
        self.occu_map = cv2.dilate(
            self.occu_map, kernel, iterations=2
        )  # to further filter out some noise

    def update_occu_map(self, points3d: np.ndarray) -> np.ndarray:
        """
        Convert point cloud to occupancy map by first doing an affine transformation,
        then use the log odd update for updating the occupancy map
        Note that this method will update self.occu_map as well as returning the updated occupancy map.
        Args:
            scaling_factor: scaling factor for the affine transformation from pcd to occupancy map
            cx: x-axis constant for the affine transformation from pcd to occupancy map
            cz: z-axis constant for the affine transformation from pcd to occupancy map
        Returns:
            the updated occupancy map
        """
        points = points3d[:, [0, 2]]
        points *= self.scaling_factor
        points = points.astype(int)
        points[:, 0] += self.cx
        points[:, 1] += self.cy
        points[:, 0] = np.clip(points[:, 0], 0, self.occu_map.shape[0] - 1)
        points[:, 1] = np.clip(points[:, 1], 0, self.occu_map.shape[1] - 1)
        self.occu_map -= 0.05
        self.occu_map[points[:, 1], points[:, 0]] += 0.9  # ground
        self.occu_map = self.occu_map.clip(0, 1)

    def find_obstacle_l_c_r(
        self,
        occu_map,
        left_occ_thresh=0.5,
        center_occ_thresh=0.5,
        right_occ_thresh=0.5,
        debug=False,
    ) -> Tuple[Tuple[bool, float], Tuple[bool, float], Tuple[bool, float]]:
        """
        Given an occupancy map `occu_map`, find whether in `left`, `center`, `right` which is/are occupied and
        also its probability for occupied.
        Args:
            occu_map: occupancy map
            left_occ_thresh: threshold occupied --> lower ==> more likely to be occupied
            center_occ_thresh: threshold occupied --> lower ==> more likely to be occupied
            right_occ_thresh: threshold occupied --> lower ==> more likely to be occupied
            debug: if true, draw out where the algorithm is looking at
        Returns:
            there tuples of bool and float representing occupied or not and its relative probability.
        """
        backtorgb = cv2.cvtColor(occu_map, cv2.COLOR_GRAY2RGB)

        height, width, channel = backtorgb.shape
        left_rec_start = (40 * width // 100, 10 * height // 100)
        left_rec_end = (45 * width // 100, 40 * height // 100)

        mid_rec_start = (48 * width // 100, 10 * height // 100)
        mid_rec_end = (52 * width // 100, 40 * height // 100)

        right_rec_start = (55 * width // 100, 10 * height // 100)
        right_rec_end = (60 * width // 100, 40 * height // 100)
        right = self.is_occupied(
            m=occu_map,
            start=right_rec_start,
            end=right_rec_end,
            percent_free=left_occ_thresh,
        )
        center = self.is_occupied(
            m=occu_map,
            start=mid_rec_start,
            end=mid_rec_end,
            percent_free=center_occ_thresh,
        )
        left = self.is_occupied(
            m=occu_map,
            start=left_rec_start,
            end=left_rec_end,
            percent_free=right_occ_thresh,
        )
        if debug:
            backtorgb = cv2.rectangle(
                backtorgb, left_rec_start, left_rec_end, (0, 0, 255), 1
            )

            backtorgb = cv2.rectangle(
                backtorgb, mid_rec_start, mid_rec_end, (0, 255, 0), 1
            )

            backtorgb = cv2.rectangle(
                backtorgb, right_rec_start, right_rec_end, (255, 0, 0), 1
            )

            cv2.imshow("Occupancy Map", backtorgb)
            cv2.waitKey(1)
        print(left, center, right)
        return left, center, right

    @staticmethod
    def is_occupied(
        m, start, end, percent_free, threshold_free=0.8
    ) -> Tuple[bool, float]:
        """
        Return the whether the area in `m` specified with `start` and `end` is occupied or not
        based on a ratio threshold.
        If the number of free spots / total area is less than threshold,
        then it means that this place is probability occupied.
        Args:
            m: 2D numpy array of occupancy map (free map to be exact)
            start: starting bounding box
            end: ending bounding box
            threshold: ratio to determine free or not
        Returns:
            bool -> true if occupied, false if free.
        """
        cropped = m[start[1] : end[1], start[0] : end[0]]
        area = (end[1] - start[1]) * (end[0] - start[0])
        spots_free = (cropped > threshold_free).sum()
        ratio = round(spots_free / area, 3)
        return (
            ratio < percent_free,
            ratio,
        )  # if spots_free/area < threshold, then this place is occupied


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
