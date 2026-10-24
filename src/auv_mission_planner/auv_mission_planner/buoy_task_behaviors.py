"""Reactive behaviors for touching a selected buoy.

Blackboard inputs (written by the ROS subscribers in ``buoy_task_tree``):
    target_found (bool): whether YOLO currently reports a buoy detection.
    target_color (str): requested buoy color, for example ``"red"``.
    detected_color (str): color/class of the current YOLO detection.
    detection_stamp (float): timestamp in seconds in the tree's clock domain
        (ROS time when a node is supplied, otherwise monotonic time).
    norm_x (float): normalized horizontal camera-frame error (-1 to 1).
    norm_y (float): normalized vertical camera-frame error (-1 to 1).
    buoy_bbox_fraction (float): YOLO bounding-box area/frame-area fraction [0, 1].
    current_depth (float): depth sensor reading in metres.
    distance_traveled (float): DVL/INS odometry in metres.

The behavior tree owns cmd_surge, cmd_sway, cmd_heave and cmd_yaw (float,
metres/second for linear axes and radians/second for yaw); the behaviors write
these blackboard commands and CommandMixer is the sole Twist publisher.
"""

import logging
import math
import time

import py_trees
from geometry_msgs.msg import Twist
from py_trees.common import Status


DETECTION_MAX_AGE_SEC = 0.5
DEFAULT_MISS_LIMIT = 5
TICK_PERIOD_SEC = 0.1


def clamp(value, limit):
    """Limit a signed velocity command to the symmetric interval [-limit, limit]."""
    if not math.isfinite(limit) or limit < 0:
        raise ValueError("limit must be finite and non-negative")
    if not math.isfinite(value):
        raise ValueError("velocity command must be finite")
    return max(-limit, min(limit, value))


def _clock_seconds(node):
    """Use ROS time when available, otherwise use monotonic time."""
    if node is not None:
        return node.get_clock().now().nanoseconds / 1_000_000_000
    return time.monotonic()


def _set_command(blackboard, key, value):
    """Write a bounded axis command; the mixer clamps again at the output boundary."""
    limits = {
        "cmd_surge": 0.4,
        "cmd_sway": 0.3,
        "cmd_heave": 0.3,
        "cmd_yaw": 0.5,
    }
    setattr(blackboard, key, clamp(value, limits[key]))


class CheckBuoyVisibility(py_trees.behaviour.Behaviour):
    """Validate target color and freshness, retaining position during short dropouts."""

    def __init__(self, node=None, miss_limit=DEFAULT_MISS_LIMIT, name="CheckBuoyVisibility"):
        super().__init__(name)
        if miss_limit < 1:
            raise ValueError("miss_limit must be at least 1")
        self.node = node
        self.miss_limit = miss_limit
        self.misses = 0
        self.last_position = None
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        for key in (
            "target_found",
            "target_color",
            "detected_color",
            "detection_stamp",
            "norm_x",
            "norm_y",
            "buoy_bbox_fraction",
        ):
            access = py_trees.common.Access.READ
            if key in ("norm_x", "norm_y"):
                # WRITE also grants read access; debounce can restore cached camera coordinates.
                access = py_trees.common.Access.WRITE
            self.blackboard.register_key(key=key, access=access)

    def _miss(self):
        self.misses += 1
        if self.misses >= self.miss_limit:
            return Status.FAILURE
        if self.last_position is not None:
            self.blackboard.norm_x, self.blackboard.norm_y = self.last_position
            return Status.SUCCESS
        # Keep the selector on this check until the debounce window expires; then it may search.
        return Status.RUNNING

    def update(self):
        try:
            target_found = self.blackboard.target_found
            target_color = self.blackboard.target_color
            detected_color = self.blackboard.detected_color
            detection_stamp = self.blackboard.detection_stamp
            norm_x = self.blackboard.norm_x
            norm_y = self.blackboard.norm_y
            bbox_fraction = self.blackboard.buoy_bbox_fraction
        except KeyError:
            return self._miss()

        # Missing values are handled explicitly; malformed values remain visible as real errors.
        if (
            target_found is None
            or target_color is None
            or detected_color is None
            or detection_stamp is None
            or norm_x is None
            or norm_y is None
            or bbox_fraction is None
        ):
            return self._miss()
        # Other colored buoys are misses even if YOLO reports them in the same frame.
        if not target_found or detected_color.lower() != target_color.lower():
            return self._miss()

        # Reject malformed camera values as misses, not as plausible servo commands.
        if not all(
            math.isfinite(value)
            for value in (detection_stamp, norm_x, norm_y, bbox_fraction)
        ):
            return self._miss()
        if not 0.0 <= bbox_fraction <= 1.0:
            return self._miss()
        # Reject stale observations using the same clock domain as the publisher timestamp.
        age = _clock_seconds(self.node) - detection_stamp
        if age < 0 or age > DETECTION_MAX_AGE_SEC:
            return self._miss()

        self.misses = 0
        self.last_position = (norm_x, norm_y)
        return Status.SUCCESS


class _AxisBehaviour(py_trees.behaviour.Behaviour):
    """Shared command registration; each servo writes only its own axes."""

    def __init__(self, name, command_keys):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.command_keys = command_keys
        for key in command_keys:
            self.blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)

    def stop_axes(self):
        for key in self.command_keys:
            _set_command(self.blackboard, key, 0.0)

    def terminate(self, new_status):
        # A behavior only clears its own blackboard axes; publishers are centralized in the mixer.
        self.stop_axes()


class CenterBuoyPosition(_AxisBehaviour):
    """Point yaw at the target, with optional fine sway correction."""

    def __init__(
        self,
        use_sway=False,
        yaw_sign=-1.0,
        yaw_gain=0.5,
        sway_gain=0.25,
        threshold=0.05,
        name="CenterBuoyPosition",
    ):
        super().__init__(name, ("cmd_yaw", "cmd_sway"))
        self.use_sway = use_sway
        # Verify this camera/yaw sign convention in simulation and in-water trials.
        self.yaw_sign = yaw_sign
        self.yaw_gain = yaw_gain
        self.sway_gain = sway_gain
        self.threshold = threshold
        self.blackboard.register_key(key="norm_x", access=py_trees.common.Access.READ)

    def update(self):
        self.stop_axes()
        try:
            norm_x = self.blackboard.norm_x
        except KeyError:
            return Status.FAILURE
        if norm_x is None:
            return Status.FAILURE
        if abs(norm_x) < self.threshold:
            return Status.SUCCESS

        _set_command(self.blackboard, "cmd_yaw", self.yaw_sign * self.yaw_gain * norm_x)
        if self.use_sway:
            _set_command(self.blackboard, "cmd_sway", -self.sway_gain * norm_x)
        return Status.RUNNING


class MatchBuoyDepth(_AxisBehaviour):
    """Align the buoy vertically in the camera image using heave only."""

    def __init__(self, gain=0.4, threshold=0.05, name="MatchBuoyDepth"):
        super().__init__(name, ("cmd_heave",))
        self.gain = gain
        self.threshold = threshold
        self.last_current_depth = None
        self.blackboard.register_key(key="norm_y", access=py_trees.common.Access.READ)
        # Keep the real depth reading available for diagnostics, but never compare it to pixel
        # error.
        self.blackboard.register_key(key="current_depth", access=py_trees.common.Access.READ)

    def update(self):
        self.stop_axes()
        try:
            norm_y = self.blackboard.norm_y
        except KeyError:
            return Status.FAILURE
        if norm_y is None:
            return Status.FAILURE

        # This sensor value is telemetry only: mixing metres with normalized pixels is invalid.
        try:
            current_depth = self.blackboard.current_depth
        except KeyError:
            current_depth = None
        if current_depth is None:
            self.last_current_depth = None
        else:
            self.last_current_depth = current_depth

        if abs(norm_y) < self.threshold:
            return Status.SUCCESS
        # Camera-frame vertical error is converted directly to heave, never to absolute depth.
        _set_command(self.blackboard, "cmd_heave", -self.gain * norm_y)
        return Status.RUNNING


class ApproachBuoy(_AxisBehaviour):
    """Close visually on the buoy, then push forward briefly to make contact."""

    def __init__(
        self,
        contact_distance=0.35,
        close_threshold=0.18,
        push_duration=2.0,
        speed_gain=1.2,
        push_speed=0.15,
        node=None,
        name="ApproachBuoy",
    ):
        super().__init__(name, ("cmd_surge",))
        # Preserve the distance argument for callers; bbox size and a timed push decide contact.
        self.contact_distance = contact_distance
        self.close_threshold = close_threshold
        self.push_duration = push_duration
        self.speed_gain = speed_gain
        self.push_speed = push_speed
        self.node = node
        if not 0.0 < close_threshold <= 1.0:
            raise ValueError("close_threshold must be in (0, 1]")
        if push_duration <= 0:
            raise ValueError("push_duration must be positive")
        self.push_started_at = None
        self.push_started_tick = None
        self.tick_count = 0
        self.last_distance_traveled = None
        for key in (
            "buoy_bbox_fraction",
            "target_found",
            "target_color",
            "detected_color",
            "detection_stamp",
            "distance_traveled",
        ):
            self.blackboard.register_key(key=key, access=py_trees.common.Access.READ)

    def initialise(self):
        self.push_started_at = None
        self.push_started_tick = None

    def _target_is_fresh(self):
        try:
            target_found = self.blackboard.target_found
            target_color = self.blackboard.target_color
            detected_color = self.blackboard.detected_color
            detection_stamp = self.blackboard.detection_stamp
        except KeyError:
            return False
        if (
            target_found is None
            or target_color is None
            or detected_color is None
            or detection_stamp is None
        ):
            return False
        if not target_found or detected_color.lower() != target_color.lower():
            return False
        if not math.isfinite(detection_stamp):
            return False
        age = _clock_seconds(self.node) - detection_stamp
        return 0 <= age <= DETECTION_MAX_AGE_SEC

    def update(self):
        self.tick_count += 1
        self.stop_axes()
        if not self._target_is_fresh():
            # Do not keep surging or touch based on a stale bounding box.
            return Status.RUNNING

        try:
            bbox_fraction = self.blackboard.buoy_bbox_fraction
        except KeyError:
            return Status.FAILURE
        if bbox_fraction is None or not math.isfinite(bbox_fraction):
            return Status.FAILURE
        if not 0.0 <= bbox_fraction <= 1.0:
            return Status.FAILURE

        # DVL/INS odometry is kept as real telemetry, while visual size controls this approach.
        try:
            distance_traveled = self.blackboard.distance_traveled
        except KeyError:
            distance_traveled = None
        if distance_traveled is None:
            self.last_distance_traveled = None
        else:
            self.last_distance_traveled = distance_traveled

        # The bbox fraction is a visual range proxy; surge closes the remaining fraction error.
        if bbox_fraction >= self.close_threshold:
            if self.push_started_at is None and self.push_started_tick is None:
                # Start a dedicated contact phase only after reaching the visual close threshold.
                self.push_started_at = _clock_seconds(self.node) if self.node is not None else None
                self.push_started_tick = self.tick_count

            if self.node is not None:
                elapsed = _clock_seconds(self.node) - self.push_started_at
            else:
                elapsed = (self.tick_count - self.push_started_tick) * TICK_PERIOD_SEC
            if elapsed >= self.push_duration:
                return Status.SUCCESS
            # Maintain forward contact for the configured duration, then stop in terminate().
            _set_command(self.blackboard, "cmd_surge", self.push_speed)
            return Status.RUNNING

        self.push_started_at = None
        self.push_started_tick = None
        remaining_fraction = self.close_threshold - bbox_fraction
        _set_command(self.blackboard, "cmd_surge", self.speed_gain * remaining_fraction)
        return Status.RUNNING


class BuoySearchSweep(_AxisBehaviour):
    """Rotate in place for a bounded number of timed search sweeps."""

    def __init__(
        self,
        timeout_sec=10.0,
        max_attempts=3,
        yaw_speed=0.35,
        node=None,
        tick_period=TICK_PERIOD_SEC,
        name="BuoySearchSweep",
    ):
        super().__init__(name, ("cmd_surge", "cmd_sway", "cmd_heave", "cmd_yaw"))
        if timeout_sec <= 0 or max_attempts < 1 or tick_period <= 0:
            raise ValueError("timeout_sec, max_attempts and tick_period must be positive")
        self.timeout_sec = timeout_sec
        self.max_attempts = max_attempts
        self.yaw_speed = yaw_speed
        self.node = node
        self.tick_period = tick_period
        self.start_time = None
        self.start_tick = 0
        self.tick_count = 0
        self.attempts = 0

    def initialise(self):
        self.start_time = _clock_seconds(self.node) if self.node is not None else None
        self.start_tick = self.tick_count
        self.attempts = 0

    def update(self):
        self.tick_count += 1
        self.stop_axes()
        if self.node is not None:
            elapsed = _clock_seconds(self.node) - self.start_time
        else:
            elapsed = (self.tick_count - self.start_tick) * self.tick_period

        if elapsed >= self.timeout_sec:
            self.attempts += 1
            if self.attempts >= self.max_attempts:
                message = (
                    "Buoy search exhausted %d attempts without finding the target"
                    % self.max_attempts
                )
                if self.node is not None:
                    self.node.get_logger().error(message)
                else:
                    logging.getLogger(__name__).error(message)
                return Status.FAILURE
            self.start_time = _clock_seconds(self.node) if self.node is not None else None
            self.start_tick = self.tick_count

        # Sweep yaw only: avoid translating toward nearby non-target buoys or obstacles.
        _set_command(self.blackboard, "cmd_yaw", self.yaw_speed)
        return Status.RUNNING


class CommandMixer(py_trees.behaviour.Behaviour):
    """Combine blackboard axis requests and publish exactly one Twist per tick."""

    def __init__(self, cmd_publisher, name="CommandMixer"):
        super().__init__(name)
        self.cmd_publisher = cmd_publisher
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        for key in ("cmd_surge", "cmd_sway", "cmd_heave", "cmd_yaw"):
            self.blackboard.register_key(key=key, access=py_trees.common.Access.READ)

    def update(self):
        twist = Twist()
        # Clamp at the final output boundary as well as when each behavior writes its axis.
        twist.linear.x = clamp(self.blackboard.cmd_surge, 0.4)
        twist.linear.y = clamp(self.blackboard.cmd_sway, 0.3)
        twist.linear.z = clamp(self.blackboard.cmd_heave, 0.3)
        twist.angular.z = clamp(self.blackboard.cmd_yaw, 0.5)
        self.cmd_publisher.publish(twist)
        return Status.SUCCESS
