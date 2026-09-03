import py_trees
from py_trees.common import Status
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import time


class CheckGateDetected(py_trees.behaviour.Behaviour):
    """Condition node: Checks if the YOLO algorithm detects the L-gate."""
    def __init__(self, name="IsGateDetected"):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="gate_found", access=py_trees.common.Access.READ)

    def update(self):
        try:
            if self.blackboard.gate_found.data:
                return Status.SUCCESS
        except (KeyError, AttributeError):
            pass
        return Status.FAILURE

# For now yaw motion is only being programmed. After testing the sway motion, the sway behaviours will be added
class CenterGateYaw(py_trees.behaviour.Behaviour):
    """Action node: Aligns vehicle yaw with the horizontal center of the gate."""
    def __init__(self, cmd_publisher, deadband=0.06, name="CenterGateYaw"):
        super().__init__(name)
        self.cmd_pub = cmd_publisher
        self.deadband = deadband
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="gate_pose", access=py_trees.common.Access.READ)

    def update(self):
        try:
            norm_x = self.blackboard.gate_pose.data[0]
            if abs(norm_x) < self.deadband:
                return Status.SUCCESS

            twist = Twist()
            twist.angular.z = -0.55 * float(norm_x)  # Ensure -0.55 gain
            self.cmd_pub.publish(twist)
            return Status.RUNNING
        except (KeyError, AttributeError, IndexError):
            return Status.FAILURE

    def terminate(self, new_status):
        self.cmd_pub.publish(Twist())
        
class MatchGateDepth(py_trees.behaviour.Behaviour):
    """Action Node: Aligns the vehicle depth (heave) with the vertical midpoint of the gate. """
    def __init__(self, cmd_publisher, deadband=0.06, name="MatchGateDepth"):
        super().__init__(name)
        self.cmd_pub = cmd_publisher
        self.deadband = deadband
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="gate_pose", access=py_trees.common.Access.READ)

    def update(self):
        try:
            norm_y = self.blackboard.gate_pose.data[1]
            if abs(norm_y) < self.deadband:
                return Status.SUCCESS

            twist = Twist()
            twist.linear.z = -0.4 * norm_y # Correct vertical deviation
            self.cmd_pub.publish(twist)
            return Status.RUNNING
        except (KeyError, AttributeError, IndexError):
            return Status.FAILURE

    def terminate(self, new_status):
        self.cmd_pub.publish(Twist())

class PassThroughGate(py_trees.behaviour.Behaviour):
    """Action Node: Drives straight through the gate for a fixed duration."""
    def __init__(self, cmd_publisher, pass_duration=10.0, cruise_speed=0.4, name="PassThroughGate"):
        super().__init__(name)
        self.cmd_pub = cmd_publisher
        self.pass_duration = pass_duration
        self.cruise_speed = cruise_speed
        self.start_time = None

    def initialise(self):
        self.start_time = time.time()   # This will be instantiated for every tick

    def update(self):
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.pass_duration:
            return Status.SUCCESS

        twist = Twist()
        twist.linear.x = self.cruise_speed
        self.cmd_pub.publish(twist)
        return Status.RUNNING

    def terminate(self, new_status):
        self.cmd_pub.publish(Twist())

class GateSearchSweep(py_trees.behaviour.Behaviour):
    """Action node: Fallback rotational scan pattern to find the gate""" # Algorithms like A*, DP and other path searching methodologies can also be applied
    def __init__(self, cmd_publisher, timeout_sec=25.0, name="GateSearchSweep"):
        super().__init__(name)
        self.cmd_pub = cmd_publisher
        self.timeout = timeout_sec
        self.start_time = None

    def initialise(self):
        self.start_time = time.time()

    def update(self):
        if (time.time() - self.start_time) > self.timeout:
            return Status.FAILURE

        twist = Twist()
        twist.angular.z = 0.25 # Sweep yaw
        self.cmd_pub.publish(twist)
        return Status.RUNNING

    def terminate(self, new_status):
        self.cmd_pub.publish(Twist())



