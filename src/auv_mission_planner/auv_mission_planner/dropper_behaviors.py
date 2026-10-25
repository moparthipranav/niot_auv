import py_trees
from py_trees.common import Status
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import math
import time


def fresh_bin_detection(blackboard, clock, max_age, require_marker=False):
    """Return a usable bin measurement, rejecting stale or invalid data."""
    try:
        detection = blackboard.bin_detection
        stamp = detection.header.stamp
        timestamp = stamp.sec + stamp.nanosec * 1e-9
        age = clock() - timestamp
        values = (
            detection.x,
            detection.y,
            detection.z,
            detection.yaw,
            detection.confidence,
        )
    except (AttributeError, KeyError):
        return None

    if (age < 0 or age > max_age
            or not all(math.isfinite(float(value)) for value in values)
            or not detection.detected):
        return None
    if require_marker and (
            not detection.marker_detected or detection.confidence < 0.2):
        return None
    return detection


class ConfirmBinDetection(py_trees.behaviour.Behaviour):
    """Require a fresh target marker before alignment begins."""
    def __init__(self, cmd_publisher, target_marker=None, hold_time=1.0,
                 clock=time.time, max_age=0.75, name="ConfirmBinDetection"):
        super().__init__(name)
        self.cmd_pub = cmd_publisher
        self.target_marker = target_marker
        self.hold_time = hold_time
        self.clock = clock
        self.max_age = max_age
        self.start_time = None
        self.confirmed = False
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(
            key="bin_detection", access=py_trees.common.Access.READ
        )

    def initialise(self):
        if self.start_time is None:
            self.start_time = self.clock()

    def update(self):
        detection = fresh_bin_detection(
            self.blackboard, self.clock, self.max_age
        )
        target_seen = (
            detection is not None
            and (
                self.target_marker is None
                or (
                    detection.marker_label == self.target_marker
                    and detection.marker_detected
                    and detection.confidence >= 0.2
                )
            )
        )
        if self.confirmed:
            if not target_seen:
                self.cmd_pub.publish(Twist())
                self.feedback_message = (
                    f"Waiting for target marker {self.target_marker}"
                )
                return Status.RUNNING
            return Status.SUCCESS
        if not target_seen:
            self.start_time = None
            if self.confirmed:
                self.cmd_pub.publish(Twist())
                self.feedback_message = (
                    f"Waiting for target marker {self.target_marker}"
                )
                return Status.RUNNING
            self.feedback_message = (
                f"Target marker {self.target_marker} lost before confirmation"
            )
            return Status.FAILURE
        if self.start_time is None:
            self.start_time = self.clock()
        if self.clock() - self.start_time >= self.hold_time:
            self.confirmed = True
            return Status.SUCCESS
        return Status.RUNNING

    def terminate(self, new_status):
        if new_status == Status.INVALID:
            self.start_time = None
            self.confirmed = False


def vehicle_is_stable(blackboard, clock, max_age):
    """Return whether a true stability signal is recent."""
    try:
        stable = blackboard.vehicle_stable
        received_at = blackboard.vehicle_stable_received_at
    except (AttributeError, KeyError):
        return False
    age = clock() - received_at
    return stable and 0 <= age <= max_age


class CheckBinAligned(py_trees.behaviour.Behaviour):
    """Succeed when the bin is within the accepted horizontal alignment range."""
    def __init__(self, tolerance=0.05, clock=time.time, max_age=0.75,
                 name="BinAligned"):
        super().__init__(name)
        self.tolerance = tolerance
        self.clock = clock
        self.max_age = max_age
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(
            key="bin_detection", access=py_trees.common.Access.READ
        )

    def update(self):
        detection = fresh_bin_detection(
            self.blackboard, self.clock, self.max_age
        )
        if detection is None:
            self.feedback_message = "No fresh bin detection"
            return Status.FAILURE

        norm_x = float(detection.x)
        norm_y = float(detection.y)
        if (abs(norm_x) <= self.tolerance
                and abs(norm_y) <= self.tolerance):
            self.feedback_message = (
                f"Aligned within tolerance {self.tolerance:.2f}; "
                "horizontal thruster commands are not needed"
            )
            return Status.SUCCESS

        self.feedback_message = (
            f"Outside alignment range: x={norm_x:.3f}, y={norm_y:.3f}"
        )
        return Status.FAILURE


class CheckBinDetected(py_trees.behaviour.Behaviour):
    """Require a fresh bin measurement before mission actions."""
    def __init__(self, clock=time.time, max_age=0.75, latch_on_detection=False,
                 name="IsBinDetected"):
        super().__init__(name)
        self.clock = clock
        self.max_age = max_age
        self.latch_on_detection = latch_on_detection
        self.detected_once = False
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(
            key="bin_detection", access=py_trees.common.Access.READ
        )

    def update(self):
        detected = fresh_bin_detection(
            self.blackboard, self.clock, self.max_age
        ) is not None
        if detected and self.latch_on_detection:
            self.detected_once = True
        if detected or (self.latch_on_detection and self.detected_once):
            return Status.SUCCESS
        return Status.FAILURE

    def terminate(self, new_status):
        if new_status == Status.INVALID:
            self.detected_once = False


class CheckTargetBin(py_trees.behaviour.Behaviour):
    """Latch acquisition only after the requested marker is identified."""
    def __init__(self, target_marker, clock=time.time, max_age=0.75,
                 name="IsTargetBinDetected"):
        super().__init__(name)
        self.target_marker = target_marker
        self.clock = clock
        self.max_age = max_age
        self.target_acquired = False
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(
            key="bin_detection", access=py_trees.common.Access.READ
        )

    def update(self):
        detection = fresh_bin_detection(
            self.blackboard, self.clock, self.max_age
        )
        if (
            detection is not None
            and (
                self.target_marker is None
                or (
                    detection.marker_detected
                    and detection.marker_label == self.target_marker
                    and detection.confidence >= 0.2
                )
            )
        ):
            self.target_acquired = True
        if self.target_acquired:
            return Status.SUCCESS
        return Status.FAILURE

    def terminate(self, new_status):
        if new_status == Status.INVALID:
            self.target_acquired = False


class WaitForBinMarker(py_trees.behaviour.Behaviour):
    """Hold position after alignment until a confident marker is detected."""
    def __init__(self, cmd_publisher, target_marker=None, clock=time.time, max_age=0.75,
                 name="WaitForBinMarker"):
        super().__init__(name)
        self.cmd_pub = cmd_publisher
        self.target_marker = target_marker
        self.clock = clock
        self.max_age = max_age
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(
            key="bin_detection", access=py_trees.common.Access.READ
        )

    def update(self):
        detection = fresh_bin_detection(
            self.blackboard, self.clock, self.max_age
        )
        if detection is None:
            self.cmd_pub.publish(Twist())
            self.feedback_message = "Waiting for fresh bin detection"
            return Status.RUNNING
        if (
            detection.marker_detected
            and detection.confidence >= 0.2
            and (
                self.target_marker is None
                or detection.marker_label == self.target_marker
            )
        ):
            return Status.SUCCESS
        self.cmd_pub.publish(Twist())
        if self.target_marker is not None:
            self.feedback_message = (
                f"Waiting for target marker {self.target_marker}"
            )
        return Status.RUNNING


class CenterBinSway(py_trees.behaviour.Behaviour):
    """Action node: Regulates lateral sway (linear.y) to center over the bin horizontally"""
    def __init__(self, cmd_publisher, deadband=0.05, clock=time.time,
                 max_age=0.75, name="CenterBinSway"):
        super().__init__(name)
        self.cmd_pub = cmd_publisher
        self.deadband = deadband
        self.clock = clock
        self.max_age = max_age
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(
            key="bin_detection", access=py_trees.common.Access.READ
        )

    def update(self):
        detection = fresh_bin_detection(
            self.blackboard, self.clock, self.max_age
        )
        if detection is None:
            self.cmd_pub.publish(Twist())
            return Status.RUNNING

        norm_x = float(detection.x)
        if abs(norm_x) <= self.deadband:
            return Status.SUCCESS

        twist = Twist()
        twist.linear.y = -0.35 * norm_x
        self.cmd_pub.publish(twist)
        return Status.RUNNING

    def terminate(self, new_status):
        self.cmd_pub.publish(Twist())

class CenterBinSurge(py_trees.behaviour.Behaviour): # Check this behaviour as we are using a downward camera
    """Action node: Regulates forward/backward surge(linear.x) to center over the bin vertically."""
    def __init__(self, cmd_publisher, deadband=0.05, clock=time.time,
                 max_age=0.75, name="CenterBinSurge"):
        super().__init__(name)
        self.cmd_pub = cmd_publisher
        self.deadband = deadband
        self.clock = clock
        self.max_age = max_age
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(
            key="bin_detection", access=py_trees.common.Access.READ
        )

    def update(self):
        detection = fresh_bin_detection(
            self.blackboard, self.clock, self.max_age
        )
        if detection is None:
            self.cmd_pub.publish(Twist())
            return Status.RUNNING

        norm_y = float(detection.y)
        if abs(norm_y) <= self.deadband:
            return Status.SUCCESS

        twist = Twist()
        twist.linear.x = -0.30 * norm_y
        self.cmd_pub.publish(twist)
        return Status.RUNNING

    def terminate(self, new_status):
        self.cmd_pub.publish(Twist())

class DescendToDropAltitude(py_trees.behaviour.Behaviour):
    """Action node: Adjusts vertical heave to lower the AUV to the optimal drop altitude. The optimal drop altitude depends on the weight and buoyancy of the payload."""
    def __init__(self, cmd_publisher, target_altitude=0.40, clock=time.time,
                 max_age=0.75, name="DescendToDropAltitude"):
        super().__init__(name)
        self.cmd_pub = cmd_publisher
        self.target_altitude = target_altitude
        self.clock = clock
        self.max_age = max_age
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(
            key="bin_detection", access=py_trees.common.Access.READ
        )

    def update(self):
        detection = fresh_bin_detection(
            self.blackboard, self.clock, self.max_age
        )
        if detection is None:
            self.cmd_pub.publish(Twist())
            return Status.RUNNING

        current_altitude = float(detection.z)
        if current_altitude <= 0:
            self.cmd_pub.publish(Twist())
            self.feedback_message = "Waiting for a valid bin range"
            return Status.RUNNING
        if current_altitude <= self.target_altitude:
            return Status.SUCCESS

        twist = Twist()
        twist.linear.z = -0.20
        self.cmd_pub.publish(twist)
        return Status.RUNNING

    def terminate(self, new_status):
        if new_status != Status.INVALID:
            self.cmd_pub.publish(Twist())


class WaitForVehicleStable(py_trees.behaviour.Behaviour):
    """Hold position until recent stable-state and bin data are available."""
    def __init__(self, cmd_publisher, clock=time.time, max_age=0.75,
                 name="WaitForVehicleStable"):
        super().__init__(name)
        self.cmd_pub = cmd_publisher
        self.clock = clock
        self.max_age = max_age
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(
            key="bin_detection", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="vehicle_stable", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="vehicle_stable_received_at",
            access=py_trees.common.Access.READ
        )

    def update(self):
        detection = fresh_bin_detection(
            self.blackboard, self.clock, self.max_age
        )
        if (detection is None or not vehicle_is_stable(
                self.blackboard, self.clock, self.max_age)):
            self.cmd_pub.publish(Twist())
            return Status.RUNNING
        return Status.SUCCESS


class ReleasePayloadGripper(py_trees.behaviour.Behaviour):
    """Action node: Sends release signal to gripper actuator. (please clarify the gripper mechanism)"""
    def __init__(self, cmd_publisher, gripper_publisher, target_marker=None,
                 release_hold_time=2.0, clock=time.time, max_age=0.75,
                 name="ReleasePayloadGripper"):
        super().__init__(name)
        self.cmd_pub = cmd_publisher
        self.gripper_pub = gripper_publisher
        self.target_marker = target_marker
        self.release_hold_time = release_hold_time
        self.clock = clock
        self.max_age = max_age
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(
            key="bin_detection", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="vehicle_stable", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="vehicle_stable_received_at",
            access=py_trees.common.Access.READ
        )
        self.start_time = None
        self.released = False

    def initialise(self):
        if self.released:
            return
        self.start_time = None

    def update(self):
        if self.released:
            return Status.SUCCESS
        if fresh_bin_detection(
                self.blackboard, self.clock, self.max_age,
                require_marker=True) is None:
            self.gripper_pub.publish(Bool(data=False))
            self.cmd_pub.publish(Twist())
            self.start_time = None
            self.feedback_message = "Waiting for fresh bin marker"
            return Status.RUNNING
        detection = fresh_bin_detection(
            self.blackboard, self.clock, self.max_age,
            require_marker=True,
        )
        if (
            self.target_marker is not None
            and detection.marker_label != self.target_marker
        ):
            self.gripper_pub.publish(Bool(data=False))
            self.cmd_pub.publish(Twist())
            self.start_time = None
            self.feedback_message = (
                f"Waiting for target marker {self.target_marker}"
            )
            return Status.RUNNING
        if not vehicle_is_stable(
                self.blackboard, self.clock, self.max_age):
            self.gripper_pub.publish(Bool(data=False))
            self.cmd_pub.publish(Twist())
            self.start_time = None
            self.feedback_message = "Waiting for stable vehicle"
            return Status.RUNNING
        if self.start_time is None:
            self.start_time = self.clock()
            self.gripper_pub.publish(Bool(data=True))
        elapsed = self.clock() - self.start_time
        if elapsed >= self.release_hold_time:
            self.gripper_pub.publish(Bool(data=False))
            self.released = True
            return Status.SUCCESS

        self.gripper_pub.publish(Bool(data=True))
        return Status.RUNNING

    def terminate(self, new_status):
        self.gripper_pub.publish(Bool(data=False))

class BinSearchSweep(py_trees.behaviour.Behaviour):
    """Search expanding square legs until the requested bin is acquired."""
    def __init__(self, cmd_publisher, clock=time.time, forward_speed=0.12,
                 turn_speed=0.25, initial_leg_duration=4.0,
                 name="BinSearchSweep"):
        super().__init__(name)
        self.cmd_pub = cmd_publisher
        self.clock = clock
        self.forward_speed = forward_speed
        self.turn_speed = turn_speed
        self.initial_leg_duration = initial_leg_duration
        self.phase_start = None
        self.phase = "forward"
        self.leg_duration = initial_leg_duration
        self.legs_at_duration = 0

    def initialise(self):
        if self.phase_start is None:
            self.phase_start = self.clock()

    def update(self):
        elapsed = self.clock() - self.phase_start
        twist = Twist()
        if self.phase == "forward":
            if elapsed + 1e-9 >= self.leg_duration:
                self.phase = "turn"
                self.phase_start = self.clock()
                self.legs_at_duration += 1
                if self.legs_at_duration == 2:
                    self.legs_at_duration = 0
                    self.leg_duration += self.initial_leg_duration
                return Status.RUNNING
            twist.linear.x = self.forward_speed
        else:
            turn_duration = (math.pi / 2) / self.turn_speed
            if elapsed + 1e-9 >= turn_duration:
                self.phase = "forward"
                self.phase_start = self.clock()
                return Status.RUNNING
            twist.angular.z = self.turn_speed
        self.cmd_pub.publish(twist)
        return Status.RUNNING

    def terminate(self, new_status):
        self.cmd_pub.publish(Twist())
        if new_status == Status.INVALID:
            self.phase_start = None
            self.phase = "forward"
            self.leg_duration = self.initial_leg_duration
            self.legs_at_duration = 0
