import py_trees
from py_trees.common import Status
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import math
import time


def fresh_bin_detection(blackboard, clock, max_age):
    """Return a usable marker measurement, rejecting stale or invalid data."""
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
            or not detection.detected
            or not detection.marker_detected
            or detection.confidence < 0.2):
        return None
    return detection


def vehicle_is_stable(blackboard, clock, max_age):
    """Return whether a true stability signal is recent."""
    try:
        stable = blackboard.vehicle_stable
        received_at = blackboard.vehicle_stable_received_at
    except (AttributeError, KeyError):
        return False
    age = clock() - received_at
    return stable and 0 <= age <= max_age


class CheckBinDetected(py_trees.behaviour.Behaviour):
    """Require a fresh bin and marker measurement before mission actions."""
    def __init__(self, clock=time.time, max_age=0.75, name="IsBinDetected"):
        super().__init__(name)
        self.clock = clock
        self.max_age = max_age
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(
            key="bin_detection", access=py_trees.common.Access.READ
        )

    def update(self):
        return (
            Status.SUCCESS
            if fresh_bin_detection(self.blackboard, self.clock, self.max_age)
            
            is not None
            else Status.FAILURE
        )

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
            return Status.FAILURE

        norm_x = float(detection.x)
        if abs(norm_x) < self.deadband:
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
            return Status.FAILURE

        norm_y = float(detection.y)
        if abs(norm_y) < self.deadband:
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
            return Status.FAILURE

        current_altitude = float(detection.z)
        if current_altitude <= 0:
            return Status.FAILURE
        if current_altitude <= self.target_altitude:
            return Status.SUCCESS

        twist = Twist()
        twist.linear.z = -0.20
        self.cmd_pub.publish(twist)
        return Status.RUNNING

    def terminate(self, new_status):
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
    def __init__(self, gripper_publisher, release_hold_time=2.0,
                 clock=time.time, max_age=0.75,
                 name="ReleasePayloadGripper"):
        super().__init__(name)
        self.gripper_pub = gripper_publisher
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
                self.blackboard, self.clock, self.max_age) is None:
            self.gripper_pub.publish(Bool(data=False))
            return Status.FAILURE
        if not vehicle_is_stable(
                self.blackboard, self.clock, self.max_age):
            self.gripper_pub.publish(Bool(data=False))
            return Status.FAILURE
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
    """Action node: Rotational search pattern to find the bin using the bottom camera to locate the bin."""
    def __init__(self, cmd_publisher, timeout_sec=25.0, name="BinSearchSweep"):
        super().__init__(name)
        self.cmd_pub = cmd_publisher
        self.timeout = timeout_sec
        self.start_time = None

    def initialise(self):
        self.start_time = time.time()

    def update(self):
        if (time.time() - self.start_time) > self.timeout:
            return Status.FAILURE # Search timed out

        twist = Twist()
        twist.angular.z = 0.25 # Rotational scan rate
        self.cmd_pub.publish(twist)
        return Status.RUNNING

    def terminate(self, new_status):
        self.cmd_pub.publish(Twist())
