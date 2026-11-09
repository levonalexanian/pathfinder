"""ROS 2 -> Rerun bridge for the PathFinder demo.

Subscribes to the demo's visualization topics and logs them to a Rerun
recording. By default it hosts a gRPC server plus the Rerun web viewer, so the
scene is visible at http://localhost:9090 with nothing installed on the host.
Set ``mode:=connect`` to stream into an already-running native viewer instead,
or ``mode:=save`` (with ``save_path``) to record an .rrd file.
"""

import os

import numpy as np
import rclpy
import rerun as rr
import rerun.blueprint as rrb
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from pathfinder_msgs.msg import PlannerStatus
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import MarkerArray

from pathfinder_rerun.markers import marker_array_ops

WORLD = "world"
MAP_ENTITY = f"{WORLD}/map"
SEARCH_ENTITY = f"{WORLD}/search"
ROBOT_ENTITY = f"{WORLD}/robot"
PATH_ENTITY = f"{WORLD}/planned_path"
GOAL_ENTITY = f"{WORLD}/goal"

PATH_COLOR = [40, 140, 255, 255]
GOAL_COLOR = [255, 60, 60, 255]


def default_blueprint():
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin=WORLD, name="Scene",
                # The URDF loader also logs collision shapes; they duplicate the visuals.
                contents=["+ /world/**", "- /world/robot/**/collision_geometries/**"],
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="planner", name="Planner"),
                rrb.TextLogView(origin="log", name="Log"),
            ),
            column_shares=[3, 1],
        ),
        collapse_panels=True,
    )


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class RerunBridge(Node):
    def __init__(self):
        super().__init__("rerun_bridge")
        self.declare_parameter("mode", "serve")  # serve | connect | save
        self.declare_parameter("save_path", "pathfinder.rrd")
        self.declare_parameter("grpc_port", 9876)
        self.declare_parameter("web_port", 9090)
        self.declare_parameter("connect_url", "rerun+http://127.0.0.1:9876/proxy")
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tf_rate_hz", 30.0)

        self._map_frame = self.get_parameter("map_frame").value
        self._base_frame = self.get_parameter("base_frame").value
        self._tf_period = 1.0 / max(1e-3, float(self.get_parameter("tf_rate_hz").value))
        self._last_tf_t = -1e9
        self._last_map_key = None
        self._last_status = None

        self._start_rerun()

        latched = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(MarkerArray, "/voxel_map_markers", self._on_map, latched)
        self.create_subscription(
            MarkerArray, "/search_visualization", self._on_search, latched)
        self.create_subscription(Path, "/planned_path", self._on_path, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal, 10)
        self.create_subscription(PlannerStatus, "/planner_status", self._on_status, 10)
        self.create_subscription(TFMessage, "/tf", self._on_tf, 100)

    def _start_rerun(self):
        bp = default_blueprint()
        rr.init("pathfinder")
        mode = self.get_parameter("mode").value
        if mode == "save":
            path = self.get_parameter("save_path").value
            rr.save(path, default_blueprint=bp)
            self.get_logger().info(f"recording to {path}")
        elif mode == "connect":
            url = self.get_parameter("connect_url").value
            rr.connect_grpc(url, default_blueprint=bp)
            self.get_logger().info(f"streaming to Rerun viewer at {url}")
        else:
            uri = rr.serve_grpc(
                grpc_port=int(self.get_parameter("grpc_port").value), default_blueprint=bp)
            web_port = int(self.get_parameter("web_port").value)
            rr.serve_web_viewer(web_port=web_port, open_browser=False, connect_to=uri)
            self.get_logger().info(
                f"Rerun web viewer: http://localhost:{web_port}  (gRPC: {uri})")

        rr.log(WORLD, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        urdf = self.get_parameter("urdf_path").value
        if urdf:
            if os.path.isfile(urdf):
                rr.log_file_from_path(urdf, entity_path_prefix=ROBOT_ENTITY, static=True)
            else:
                self.get_logger().warn(f"urdf_path {urdf!r} not found; robot shown as a point")
        if not urdf or not os.path.isfile(urdf):
            rr.log(f"{ROBOT_ENTITY}/marker", rr.Points3D([[0, 0, 0]], radii=[0.2]), static=True)

    def _set_time(self, stamp):
        rr.set_time("ros_time", duration=stamp_seconds(stamp))

    def _on_map(self, msg):
        # The map visualizer republishes the same boxes every second; only log
        # when the content actually changes, and log it static (timeless).
        key = tuple(
            (m.ns, m.id, m.type, round(m.pose.position.x, 4), round(m.pose.position.y, 4),
             round(m.pose.position.z, 4), round(m.scale.x, 4), round(m.scale.y, 4),
             round(m.scale.z, 4))
            for m in msg.markers)
        if key == self._last_map_key:
            return
        self._last_map_key = key
        for path, arch in marker_array_ops(msg.markers, MAP_ENTITY):
            rr.log(path, arch, static=True)

    def _on_search(self, msg):
        if msg.markers:
            self._set_time(msg.markers[0].header.stamp)
        for path, arch in marker_array_ops(msg.markers, SEARCH_ENTITY):
            rr.log(path, arch)

    def _on_path(self, msg):
        self._set_time(msg.header.stamp)
        pts = np.array(
            [[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in msg.poses],
            dtype=np.float32)
        if len(pts) < 2:
            rr.log(PATH_ENTITY, rr.Clear(recursive=False))
            return
        rr.log(PATH_ENTITY, rr.LineStrips3D([pts], radii=[0.03], colors=[PATH_COLOR]))
        rr.log("log", rr.TextLog(f"path published: {len(pts)} waypoints", level="INFO"))

    def _on_goal(self, msg):
        self._set_time(msg.header.stamp)
        p = msg.pose.position
        rr.log(GOAL_ENTITY, rr.Points3D([[p.x, p.y, p.z]], radii=[0.12], colors=[GOAL_COLOR]))
        rr.log(PATH_ENTITY, rr.Clear(recursive=False))
        rr.log("log", rr.TextLog(f"goal ({p.x:.2f}, {p.y:.2f}, {p.z:.2f})", level="INFO"))

    def _on_status(self, msg):
        status = (msg.active_algorithm, msg.planning, msg.last_plan_duration_ms)
        if status == self._last_status:
            return
        prev = self._last_status
        self._last_status = status
        rr.set_time("ros_time", duration=stamp_seconds(self.get_clock().now().to_msg()))
        if prev is None or prev[0] != msg.active_algorithm:
            rr.log("log", rr.TextLog(f"active algorithm: {msg.active_algorithm}", level="INFO"))
        if prev is None or prev[2] != msg.last_plan_duration_ms:
            rr.log("planner/plan_duration_ms", rr.Scalars([msg.last_plan_duration_ms]))

    def _on_tf(self, msg):
        for tf in msg.transforms:
            if tf.header.frame_id != self._map_frame or tf.child_frame_id != self._base_frame:
                continue
            t = stamp_seconds(tf.header.stamp)
            if t - self._last_tf_t < self._tf_period:
                continue
            self._last_tf_t = t
            rr.set_time("ros_time", duration=t)
            tr, q = tf.transform.translation, tf.transform.rotation
            rr.log(
                ROBOT_ENTITY,
                rr.Transform3D(
                    translation=[tr.x, tr.y, tr.z],
                    quaternion=rr.Quaternion(xyzw=[q.x, q.y, q.z, q.w])),
            )


def main(args=None):
    rclpy.init(args=args)
    node = RerunBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
