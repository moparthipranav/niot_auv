"""ROS 2 node and tree for the buoy-touch mission.

YOLO runs in ``auv_vision``. This node adapts its stamped detections and vehicle
sensors to the behavior-tree blackboard, then publishes the mixed buoy command.
"""

import json
import math

import py_trees
import py_trees_ros
import rclpy
from geometry_msgs.msg import Twist, Vector3
from py_trees.common import Status
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String

from auv_mission_planner.buoy_task_behaviors import (
    ApproachBuoy,
    BuoySearchSweep,
    CenterBuoyPosition,
    CheckBuoyVisibility,
    CommandMixer,
    MatchBuoyDepth,
)

TICK_PERIOD_SEC = 0.1
BUOY_COLORS = ("red", "green", "yellow")
DEFAULT_COLOR_CLASS_NAMES = {
    "red": "red,red_buoy,buoy_red,red buoy",
    "green": "green,green_buoy,buoy_green,green buoy",
    "yellow": "yellow,yellow_buoy,buoy_yellow,yellow buoy",
}


class _TouchBuoySequence(py_trees.composites.Sequence):
    """Tick the final mixer on every cycle, including while another child runs."""

    def __init__(self, mixer):
        super().__init__(name="TouchBuoy", memory=False)
        self.mixer = mixer

    def tick(self):
        mixer_ticked = False
        for node in super().tick():
            if node is self.mixer:
                mixer_ticked = True
            yield node
        if not mixer_ticked:
            # Standard sequences stop at RUNNING; the mixer still publishes the current axes.
            yield from self.mixer.tick()


def create_buoy_tree(cmd_publisher, node=None, target_color="red", use_sway=False):
    """Create the target-acquisition, concurrent-servo and contact tree."""
    if not isinstance(target_color, str) or not target_color.strip():
        raise ValueError("target_color must be a non-empty color string")

    initializer = py_trees.blackboard.Client(name="TouchBuoyBlackboardInitializer")
    initializer.register_key(key="target_color", access=py_trees.common.Access.WRITE)
    initializer.target_color = target_color.strip().lower()
    for key in ("cmd_surge", "cmd_sway", "cmd_heave", "cmd_yaw"):
        initializer.register_key(key=key, access=py_trees.common.Access.WRITE)
        setattr(initializer, key, 0.0)

    acquire_target = py_trees.composites.Selector(name="AcquireTarget", memory=False)
    acquire_target.add_children(
        [
            CheckBuoyVisibility(node=node),
            BuoySearchSweep(node=node),
        ]
    )

    # Unsynchronised SuccessOnAll re-ticks aligned children so they keep correcting.
    servo_and_approach = py_trees.composites.Parallel(
        name="ServoAndApproach",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False),
    )
    servo_and_approach.add_children(
        [
            CenterBuoyPosition(use_sway=use_sway),
            MatchBuoyDepth(),
            ApproachBuoy(node=node),
        ]
    )

    mixer = CommandMixer(cmd_publisher)
    root = _TouchBuoySequence(mixer=mixer)
    root.add_children([acquire_target, servo_and_approach, mixer])
    return root


def _normalise_class_name(class_name):
    """Normalize common YOLO label separators for color-class matching."""
    return "_".join(class_name.strip().lower().replace("-", " ").split())


class BuoyMissionNode(Node):
    """Adapt vision/navigation topics to the tree and publish buoy-task commands."""

    def __init__(self):
        super().__init__("buoy_mission_node")
        self.declare_parameter("target_buoy_color", "red")
        self.declare_parameter("use_sway", False)
        self.declare_parameter("wait_for_active_task", True)
        for color, default_names in DEFAULT_COLOR_CLASS_NAMES.items():
            self.declare_parameter(f"{color}_class_names", default_names)

        self.target_color = str(
            self.get_parameter("target_buoy_color").value
        ).strip().lower()
        if self.target_color not in BUOY_COLORS:
            raise ValueError(
                "target_buoy_color must be one of: %s" % ", ".join(BUOY_COLORS)
            )
        self.color_class_names = {}
        for color in BUOY_COLORS:
            configured_names = str(
                self.get_parameter(f"{color}_class_names").value
            ).split(",")
            self.color_class_names[color] = {
                _normalise_class_name(name)
                for name in configured_names
                if name.strip()
            }

        # The ROS callbacks are the writers for sensor and vision blackboard inputs.
        self.blackboard = py_trees.blackboard.Client(name="BuoyMissionROSInputs")
        blackboard_keys = (
            "target_color",
            "target_found",
            "detected_color",
            "detection_stamp",
            "norm_x",
            "norm_y",
            "buoy_bbox_fraction",
            "current_depth",
            "distance_traveled",
            "cmd_surge",
            "cmd_sway",
            "cmd_heave",
            "cmd_yaw",
        )
        for key in blackboard_keys:
            self.blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)
            setattr(self.blackboard, key, None)
        self.blackboard.target_color = self.target_color

        self.distance_traveled = 0.0
        self.last_dvl_time_ns = None
        self.dvl_valid = False
        self.wait_for_active_task = bool(
            self.get_parameter("wait_for_active_task").value
        )
        self.target_color_config_received = not self.wait_for_active_task
        self.active_task_name = ""
        self.task_active = not self.wait_for_active_task
        self.task_result_published = False
        self._warned_unlabeled_buoy = False

        self.cmd_pub = self.create_publisher(Twist, "/control/requests/buoy", 10)
        self.task_completed_pub = self.create_publisher(
            String, "/mission/task_completed", 10
        )
        self.task_failed_pub = self.create_publisher(
            String, "/mission/task_failed", 10
        )

        root = create_buoy_tree(
            self.cmd_pub,
            node=self,
            target_color=self.target_color,
            use_sway=bool(self.get_parameter("use_sway").value),
        )
        self.command_mixer = root.children[-1]
        self.tree = py_trees_ros.trees.BehaviourTree(
            root=root,
            unicode_tree_debug=True,
        )
        self.tree.setup(timeout=15, node=self)

        self.create_subscription(String, "/vision/detections", self._detections_cb, 10)
        self.create_subscription(Float32, "/sensors/depth", self._depth_cb, 10)
        self.create_subscription(Bool, "/sensors/dvl_valid", self._dvl_valid_cb, 10)
        self.create_subscription(
            Vector3, "/sensors/dvl_velocity", self._dvl_velocity_cb, 10
        )
        self.create_subscription(
            String, "/mission/active_task", self._active_task_cb, 10
        )
        self.create_subscription(
            String, "/mission/target_buoy_color", self._target_color_cb, 10
        )
        self.timer = self.create_timer(TICK_PERIOD_SEC, self.tick_tree)

        self.get_logger().info(
            "Buoy mission task ready: target=%s, waiting_for_active_task=%s"
            % (self.target_color, self.wait_for_active_task)
        )

    def _set_missing_detection(self, stamp=None):
        self.blackboard.target_found = False
        if stamp is not None and math.isfinite(stamp):
            self.blackboard.detection_stamp = stamp

    def _detections_cb(self, message):
        """Select the requested color from the detector's stamped JSON detections."""
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError as error:
            self.get_logger().error("Invalid /vision/detections JSON: %s" % error)
            self._set_missing_detection()
            return

        if not isinstance(payload, dict):
            self.get_logger().error(
                "/vision/detections must contain stamped metadata and a detections list"
            )
            self._set_missing_detection()
            return

        stamp = payload.get("stamp")
        frame_width = payload.get("frame_width")
        frame_height = payload.get("frame_height")
        detections = payload.get("detections")
        if (
            not isinstance(stamp, (int, float))
            or not math.isfinite(stamp)
            or not isinstance(frame_width, int)
            or not isinstance(frame_height, int)
            or frame_width <= 0
            or frame_height <= 0
            or not isinstance(detections, list)
        ):
            self.get_logger().error("Invalid stamped detection metadata")
            self._set_missing_detection()
            return

        selected = None
        other_color = None
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            class_name = detection.get("class_name")
            if not isinstance(class_name, str):
                continue
            normalized_name = _normalise_class_name(class_name)
            detected_color = next(
                (
                    color
                    for color, class_names in self.color_class_names.items()
                    if normalized_name in class_names
                ),
                None,
            )
            if detected_color is None:
                if normalized_name in ("buoy", "target_buoy"):
                    if not self._warned_unlabeled_buoy:
                        self.get_logger().warning(
                            "YOLO reports generic 'buoy' labels; configure color-specific "
                            "class names so the selected buoy can be distinguished"
                        )
                        self._warned_unlabeled_buoy = True
                continue

            confidence = detection.get("confidence", 0.0)
            coordinates = tuple(
                detection.get(key) for key in ("x1", "y1", "x2", "y2")
            )
            if (
                not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or not all(
                    isinstance(value, (int, float)) and math.isfinite(value)
                    for value in coordinates
                )
            ):
                continue
            candidate = (confidence, detected_color, coordinates)
            if detected_color == self.target_color:
                if selected is None or confidence > selected[0]:
                    selected = candidate
            elif other_color is None or confidence > other_color[0]:
                other_color = candidate

        self.blackboard.detection_stamp = float(stamp)
        if selected is None:
            self._set_missing_detection(float(stamp))
            if other_color is not None:
                self.blackboard.detected_color = other_color[1]
            return

        _, detected_color, (x1, y1, x2, y2) = selected
        box_width = max(0.0, min(float(frame_width), x2) - max(0.0, x1))
        box_height = max(0.0, min(float(frame_height), y2) - max(0.0, y1))
        if box_width <= 0.0 or box_height <= 0.0:
            self._set_missing_detection(float(stamp))
            return

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        self.blackboard.target_found = True
        self.blackboard.detected_color = detected_color
        self.blackboard.norm_x = max(
            -1.0,
            min(1.0, (center_x - frame_width / 2.0) / (frame_width / 2.0)),
        )
        self.blackboard.norm_y = max(
            -1.0,
            min(1.0, (center_y - frame_height / 2.0) / (frame_height / 2.0)),
        )
        self.blackboard.buoy_bbox_fraction = min(
            1.0, box_width * box_height / (frame_width * frame_height)
        )

    def _depth_cb(self, message):
        self.blackboard.current_depth = float(message.data)

    def _dvl_valid_cb(self, message):
        self.dvl_valid = bool(message.data)

    def _dvl_velocity_cb(self, message):
        """Integrate valid body-forward DVL speed into task-relative distance."""
        now_ns = self.get_clock().now().nanoseconds
        if (
            self.task_active
            and self.last_dvl_time_ns is not None
            and self.dvl_valid
        ):
            elapsed = (now_ns - self.last_dvl_time_ns) / 1_000_000_000
            if 0.0 < elapsed <= 1.0 and math.isfinite(message.x):
                self.distance_traveled += float(message.x) * elapsed
        self.last_dvl_time_ns = now_ns
        self.blackboard.distance_traveled = self.distance_traveled

    def _active_task_cb(self, message):
        """Run only while the mission sequencer owns the buoy task."""
        self.active_task_name = message.data.strip().lower()
        self._update_task_activation()

    def _target_color_cb(self, message):
        """Use the same selected color as the mission sequencer."""
        target_color = message.data.strip().lower()
        if target_color not in BUOY_COLORS:
            self.get_logger().error(
                "Ignoring unsupported mission buoy color: %s" % target_color
            )
            return
        self.target_color = target_color
        self.blackboard.target_color = target_color
        self.target_color_config_received = True
        self._update_task_activation()

    def _update_task_activation(self):
        task_active = (
            self.active_task_name == "buoy"
            and self.target_color_config_received
        )
        if task_active == self.task_active:
            return
        self.task_active = task_active
        self.task_result_published = False
        if task_active:
            self.distance_traveled = 0.0
            self.last_dvl_time_ns = None
            self.blackboard.distance_traveled = 0.0
        self.tree.root.stop(Status.INVALID)
        self._publish_zero_command()

    def _publish_zero_command(self):
        for key in ("cmd_surge", "cmd_sway", "cmd_heave", "cmd_yaw"):
            setattr(self.blackboard, key, 0.0)
        list(self.command_mixer.tick())

    def tick_tree(self):
        if not self.task_active:
            return
        if self.tree.root.status in (Status.SUCCESS, Status.FAILURE):
            return

        self.tree.tick()
        if self.tree.root.status == Status.SUCCESS:
            self._publish_zero_command()
            self.task_completed_pub.publish(String(data="buoy"))
            self.task_result_published = True
        elif self.tree.root.status == Status.FAILURE and not self.task_result_published:
            self._publish_zero_command()
            self.task_failed_pub.publish(String(data="buoy"))
            self.task_result_published = True


def main(args=None):
    """Run the buoy mission task node."""
    rclpy.init(args=args)
    node = BuoyMissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.tree.shutdown()
        node.destroy_node()
        rclpy.shutdown()
