import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    LaunchConfiguration,
    PythonExpression,
)
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("pathfinder_robot_car")
    bringup_share = get_package_share_directory("pathfinder_bringup")
    ros_gz_sim_share = get_package_share_directory("ros_gz_sim")

    world_path = os.path.join(pkg_share, "gazebo", "worlds", "car_world.sdf")
    car_sdf_path = os.path.join(pkg_share, "gazebo", "car.sdf")
    xacro_path = os.path.join(pkg_share, "urdf", "car.urdf.xacro")

    headless = LaunchConfiguration("headless")
    default_algorithm = LaunchConfiguration("default_algorithm")
    launch_rerun = LaunchConfiguration("launch_rerun")
    use_sim_time = LaunchConfiguration("use_sim_time")
    map_file = LaunchConfiguration("map_file")

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
    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time", default_value="true",
        description="Use simulation (Gazebo) clock if true",
    )
    declare_map_file = DeclareLaunchArgument(
        "map_file",
        default_value=os.path.join(os.getcwd(), "maps", "demo_world.bt"),
        description="Path to the .bt octomap file to publish",
    )

    gz_args = PythonExpression([
        "'", world_path, " -r'",
        " + (' -s --headless-rendering' if '", headless, "' == 'true' else '')",
    ])

    base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, "launch", "base.launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "default_algorithm": default_algorithm,
            "launch_rerun": launch_rerun,
            "urdf_path": os.path.join(pkg_share, "urdf", "car.urdf"),
            "map_file": map_file,
        }.items(),
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": gz_args}.items(),
    )

    robot_description = Command(["xacro ", xacro_path])

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": True,
        }],
    )

    spawn_car = Node(
        package="ros_gz_sim",
        executable="create",
        name="spawn_pathfinder_car",
        output="screen",
        arguments=[
            "-file", car_sdf_path,
            "-name", "pathfinder_car",
            "-x", "0.5", "-y", "0.5", "-z", "0.145",
        ],
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ros_gz_bridge",
        output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/model/pathfinder_car/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose",
        ],
        parameters=[{"use_sim_time": True}],
    )

    pose_relay = Node(
        package="pathfinder_core",
        executable="gz_pose_relay_node",
        name="gz_pose_relay",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "source_topic": "/model/pathfinder_car/pose",
            "parent_frame": "map",
            "child_frame": "base_link",
        }],
    )

    path_follower = Node(
        package="pathfinder_robot_car",
        executable="path_follower_node",
        name="path_follower",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    set_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=os.path.join(pkg_share, "gazebo"),
    )

    return LaunchDescription([
        declare_headless,
        declare_default_algorithm,
        declare_launch_rerun,
        declare_use_sim_time,
        declare_map_file,
        set_resource_path,
        base_launch,
        gz_sim,
        robot_state_publisher,
        spawn_car,
        bridge,
        pose_relay,
        path_follower,
    ])
