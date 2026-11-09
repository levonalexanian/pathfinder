import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def make_demo_launch(robot_pkg, robot_launch_file):
    def generate_launch_description():
        robot_share = get_package_share_directory(robot_pkg)
        bringup_share = get_package_share_directory("pathfinder_bringup")

        headless = LaunchConfiguration("headless")
        default_algorithm = LaunchConfiguration("default_algorithm")
        launch_rerun = LaunchConfiguration("launch_rerun")
        robot_radius = LaunchConfiguration("robot_radius")
        max_plan_time_sec = LaunchConfiguration("max_plan_time_sec")

        declare_headless = DeclareLaunchArgument(
            "headless", default_value="false",
            description="If true, run gz sim server-only with headless rendering",
        )
        declare_default_algorithm = DeclareLaunchArgument(
            "default_algorithm", default_value="astar",
            description="Default planning algorithm name",
        )
        declare_launch_rerun = DeclareLaunchArgument(
            "launch_rerun", default_value="true",
            description="Launch the Rerun bridge (web viewer on :9090)",
        )
        declare_robot_radius = DeclareLaunchArgument(
            "robot_radius", default_value="0.25",
            description="Robot radius (m) used to inflate occupied voxels",
        )
        declare_max_plan_time_sec = DeclareLaunchArgument(
            "max_plan_time_sec", default_value="5.0",
            description="Hard time budget for a single plan (seconds)",
        )

        robot_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(robot_share, "launch", robot_launch_file)
            ),
            launch_arguments={
                "headless": headless,
                "default_algorithm": default_algorithm,
                "launch_rerun": launch_rerun,
            }.items(),
        )

        algos_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(bringup_share, "launch", "algos.launch.py")
            ),
            launch_arguments={
                "use_sim_time": "true",
                "robot_radius": robot_radius,
                "max_plan_time_sec": max_plan_time_sec,
            }.items(),
        )

        return LaunchDescription([
            declare_headless,
            declare_default_algorithm,
            declare_launch_rerun,
            declare_robot_radius,
            declare_max_plan_time_sec,
            robot_launch,
            algos_launch,
        ])

    return generate_launch_description
