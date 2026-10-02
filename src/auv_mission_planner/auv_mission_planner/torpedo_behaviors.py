"""Behaviours for the torpedo firing task in NIOT SAVe competition."""
import py_trees
from py_trees.common import Status
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import time

class CheckTorpedoTargetVisible(py_trees.behaviour.Behaviour):
    """Condition node: Verifies if YOLO/Vision detects the target cutout board. """
    def __init__(self, name="IsTorpedoCutoutVisible"):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="torpedo_found", access=py_trees.common.Access.READ)

    def update(self):
        try:
            if self.blackboard.torpedo_found.data:
                return Status.SUCCESS
        except (KeyError, AttributeError):
            pass

        return Status.FAILURE

class CenterTorpedoYaw(py_trees.behaviour.Behaviour): # Update this behaviour for sway motion
    """Action node: Aligns vehicle yaw with the target cutout horizontal center."""
    def __init__(self, cmd_publisher, deadband=0.04, name="CenterTorpedoYaw"):
        super().__init__(name)
        self.cmd_pub = cmd_publisher
        self.deadband = deadband
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="torpedo_pose", access=py_trees.common.Access.READ)

    def update(self):
        try:
            # Torpedo Pose: [norm_x, norm_y, distance_z]
            norm_x = float(self.blackboard.torpedo_pose.data[0])
            if abs(norm_x) < self.deadband:
                return Status.SUCCESS

            twist = Twist()
            twist.angular.z = -0.05 * norm_x # P-control yaw alignment (Using the concept of a Proportional Controller where Kp = -0.05)
            self.cmd_pub.publish(twist)
            return Status.RUNNING
        except (KeyError, AttributeError, IndexError):  # Proper Error handling can be patched later
            return Status.FAILURE

    def terminate(self, new_status):
        self.cmd_pub.publish(Twist())  # Stop the Torpedo

class MatchTorpedoDepth(py_trees.behaviour.Behaviour):
    """Action node: Aligns vehicle depth (heave) with the target cutout elevation."""
    def __init__(self, cmd_publisher, deadband=0.4, name="MatchTorpedoDepth"):
        super().__init__()
        self.cmd_pub = cmd_publisher
        self.deadband = deadband
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key("torpedo_pose", access=py_trees.common.Access.READ)

    def update(self):
        try:
            norm_y = float(self.blackboard.torpedo_pose.data[1])
            if abs(norm_y) < self.deadband:
                return Status.SUCCESS

            twist = Twist()
            twist.linear.z = -0.4 * norm_y
            self.cmd_pub.publish(twist)
            return Status.RUNNING
        except (KeyError, AttributeError, IndexError):
            return Status.FAILURE

    def terminate(self, new_status):
        self.cmd_pub.publish(Twist())

class ApproachFiringStandOff(py_trees.behaviour.Behaviour):
    """Action node: Approaches the target until opimal firing stand-off distance is reached."""
    def __init__(self, cmd_publisher, standoff_distance=0.50, name="ApproachFiringDistance"):
        super().__init__(name)
        self.cmd_pub = cmd_publisher
        self.standoff_distance = standoff_distance
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="torpedo_pose", access=py_trees.common.Access.READ)

    def update(self):
        try:
            dist = float(self.blackboard.torpedo_pose.data[2])
            if dist <= self.standoff_distance:
                return Status.SUCCESS

            twist = Twist()
            twist.linear.x = 0.25 # Controlled safe firing stand-off range
            self.cmd_pub.publish(twist)
            return Status.RUNNING
        except(KeyError, AttributeError, IndexError):
            return Status.FAILURE

    def terminate(self, new_status):
        self.cmd_pub.publish(Twist())

class FireTorpedoActuator(py_trees.behaviour.Behaviour):
    """Action node: Sends trigger pulse to the torpedo launch mechanism"""
    def __init__(self, fire_publisher, duration=1.5, name="FireTorpedoActuator"):
        super().__init__(name=name)
        self.fire_pub = fire_publisher
        self.duration = duration
        self.start_time = None

    def initialise(self):
        self.start_time = time.time()
        msg = Bool(data=True)
        self.fire_pub.publish(msg)

    def update(self):
        elapsed = time.time() - self.start_time
        if elapsed >= self.duration:
            # Release the trigger
            self.fire_pub.publish(Bool(data=False))
            return Status.SUCCESS

        self.fire_pub.publish(Bool(data=True))
        return Status.RUNNING

    def terminate(self, new_status):
        self.fire_pub.publish(Bool(data=False))

class TorpedoSearchSweep(py_trees.behaviour.Behaviour):
    """Action node: Rotational Search pattern to locate the torpedo target board.""" # Later patch this behaviour to use advanced search patterns such as A* or Dijkstra's algorithm.
    def __init__(self, cmd_publisher, timeout_sec=25.0, name="TorpedoSearchSweep"):
        super().__init__(name=name)
        self.cmd_pub = cmd_publisher
        self.timeout_sec = timeout_sec
        self.start_time = None

    def initialise(self):
        self.start_time = time.time()

    def update(self):
        if (time.time() - self.start_time) > self.timeout_sec:
            return Status.FAILURE

        twist = Twist()
        twist.angular.z = 0.25 # Sweep yaw rotation
        self.cmd_pub.publish(twist)
        return Status.RUNNING

    def terminate(self, new_status):
        self.cmd_pub.publish(Twist())
