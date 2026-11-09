import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    default_map = os.path.join(os.getcwd(), "maps", "demo_world.bt")

    use_sim_time = LaunchConfiguration("use_sim_time")
    map_file = LaunchConfiguration("map_file")
    default_algorithm = LaunchConfiguration("default_algorithm")
    launch_rerun = LaunchConfiguration("launch_rerun")
    urdf_path = LaunchConfiguration("urdf_path")

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time", default_value="false",
        description="Use simulation (Gazebo) clock if true",
    )
    declare_map_file = DeclareLaunchArgument(
        "map_file", default_value=default_map,
        description="Path to the .bt octomap file to publish",
    )
    declare_default_algorithm = DeclareLaunchArgument(
        "default_algorithm", default_value="astar",
        description="Default planning algorithm name",
    )
    declare_launch_rerun = DeclareLaunchArgument(
        "launch_rerun", default_value="true",
        description="Launch the Rerun bridge (web viewer on :9090)",
    )

    declare_urdf_path = DeclareLaunchArgument(
        "urdf_path", default_value="",
        description="Robot URDF to show in Rerun (empty: draw a point)",
    )

    map_publisher = Node(
        package="pathfinder_core",
        executable="map_publisher_node",
        name="map_publisher",
        output="screen",
        parameters=[{
            "map_file": map_file,
            "map_frame": "map",
            "use_sim_time": use_sim_time,
        }],
    )

    voxel_map_visualizer = Node(
        package="pathfinder_core",
        executable="voxel_map_visualizer_node",
        name="voxel_map_visualizer",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    scheduler = Node(
        package="pathfinder_core",
        executable="scheduler_node",
        name="scheduler",
        output="screen",
        parameters=[{
            "default_algorithm": default_algorithm,
            "use_sim_time": use_sim_time,
        }],
    )

    rerun_bridge = Node(
        package="pathfinder_rerun",
        executable="rerun_bridge",
        name="rerun_bridge",
        output="screen",
        parameters=[{
            "urdf_path": urdf_path,
            "use_sim_time": use_sim_time,
        }],
        condition=IfCondition(PythonExpression(["'", launch_rerun, "' == 'true'"])),
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_map_file,
        declare_default_algorithm,
        declare_launch_rerun,
        declare_urdf_path,
        map_publisher,
        voxel_map_visualizer,
        scheduler,
        rerun_bridge,
    ])
