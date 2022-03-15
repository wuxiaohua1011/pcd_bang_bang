from asyncio import base_subprocess
from email.mime import base

from sqlalchemy import true
from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory
import launch_ros
from pathlib import Path
import launch


def generate_launch_description():
    base_path = os.path.realpath(get_package_share_directory("pcd_bang_bang"))

    bang_bang_controller_node = Node(
        package="pcd_bang_bang",
        executable="pcd_bang_bang_node",
        name="pcd_bang_bang_node",
        output="screen",
        emulate_tty=True,
    )
    return LaunchDescription(
        [
            launch.actions.DeclareLaunchArgument(
                name="initial_voxel_size", default_value="0.5"
            ),
            launch.actions.DeclareLaunchArgument(
                name="distance_threshold", default_value="1.0"
            ),
            launch.actions.DeclareLaunchArgument(name="ransac_n", default_value="10"),
            launch.actions.DeclareLaunchArgument(
                name="num_iterations", default_value="1000"
            ),
            launch.actions.DeclareLaunchArgument(
                name="max_distance", default_value="2.0"
            ),
            launch.actions.IncludeLaunchDescription(
                launch.launch_description_sources.PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("roar_ground_plane_detector"),
                        "ground_plane_detector.launch.py",
                    )
                ),
                launch_arguments={
                    "lidar_topic": "/pointcloud",
                    "display_axis_scale": "2.0",
                    "should_show": "True",
                    "initial_voxel_size": launch.substitutions.LaunchConfiguration(
                        "initial_voxel_size"
                    ),
                    "distance_threshold": launch.substitutions.LaunchConfiguration(
                        "distance_threshold"
                    ),
                    "num_iterations": launch.substitutions.LaunchConfiguration(
                        "num_iterations"
                    ),
                    "max_distance": launch.substitutions.LaunchConfiguration(
                        "max_distance"
                    ),
                }.items(),
            ),
            bang_bang_controller_node,
        ]
    )
