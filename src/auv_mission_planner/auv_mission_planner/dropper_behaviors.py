import py_trees
from py_trees.common import Status
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import time

class CheckBinDetected(py_trees.behaviour.Behaviour):
    """Condition Node: Checks if the bottom camera has detected the designated bin ('X' or 'O')"""
    def __init__(self, name="IsBinDetected"):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="bin_found", access=py_trees.common.Access.READ)

    def update(self):
        try:
            if self.blackboard.bin_found.data:
                return Status.SUCCESS
        except (KeyError, AttributeError):
            pass
        return Status.FAILURE

class CenterBinSway(py_trees.behaviour.Behaviour):
    """Action node: Regulates lateral sway (linear.y) to center over the bin horizontally"""
    def __init__(self, cmd_publisher, deadband=0.05, name="CenterBinSway"):
        super().__init__(name)
        self.cmd_pub = cmd_publisher
        self.deadband = deadband
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="bin_pose", access=py_trees.common.Access.READ)

    def update(self):
        try:
            # bin_pose: [norm_x, norm_y, altiute_z]
            norm_x = float(self.blackboard.bin_pose.data[0])
            if abs(norm_x) < self.deadband:
                return Status.SUCCESS

            twist = Twist()
            twist.linear.y = -0.35 * norm_x
            self.cmd_pub.publish(twist)
            return Status.RUNNING
        except (KeyError, AttributeError, IndexError):
            return Status.FAILURE

    def terminate(self, new_status):
        self.cmd_pub.publish(Twist())

class CenterBinSurge(py_trees.behaviour.Behaviour): # Check this behaviour as we are using a downward camera
    """Action node: Regulates forward/backward surge(linear.x) to center over the bin vertically."""
    def __init__(self, cmd_publisher, deadband=0.05, name="CenterBinSurge"):
        super().__init__(name)
        self.cmd_pub = cmd_publisher
        self.deadband = deadband
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="bin_pose", access=py_trees.common.Access.READ)

    def update(self):
        try:
            norm_y = float(self.blackboard.bin_pose.data[1])
            if abs(norm_y) < self.deadband:
                return Status.SUCCESS

            twist = Twist()
            twist.linear.x = -0.30 * norm_y
            self.cmd_pub.publish(twist)
            return Status.RUNNING
        except (KeyError, AttributeError, IndexError):
            return Status.FAILURE

    def terminate(self, new_status):
        self.cmd_pub.publish(Twist())

class DescendToDropAltitude(py_trees.behaviour.Behaviour):
    """Action node: Adjusts vertical heave to lower the AUV to the optimal drop altitude. The optimal drop altitude depends on the weight and buoyancy of the payload."""
    def __init__(self, cmd_publisher, target_altitude=0.40, name="DescendToDropAltitude"):
        super().__init__(name)
        self.cmd_pub = cmd_publisher
        self.target_altitude = target_altitude
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="bin_pose", access=py_trees.common.Access.READ)

    def update(self):
        try:
            current_altitude = float(self.blackboard.bin_pose.data[2])
            if current_altitude <= self.target_altitude:
                return Status.SUCCESS

            twist = Twist()
            twist.linear.z = -0.20 # Controlled descent heave
            self.cmd_pub.publisher(twist)
            return Status.RUNNING
        except (KeyError, AttributeError, IndexError):
            return Status.FAILURE

    def terminate(self, new_status):
        self.cmd_pub.publish(Twist())

class ReleasePayloadGripper(py_trees.behaviour.Behaviour):
    """Action node: Sends release signal to gripper actuator. (please clarify the gripper mechanism)"""
    def __init__(self, gripper_publisher, release_hold_time=2.0, name="ReleasePayloadGripper"): # release_hold_time sets how long the release trigger signal remains active. 
        super().__init__(name)
        self.gripper_pub = gripper_publisher
        self.release_hold_time = release_hold_time
        self.start_time = None

    def initialise(self):
        self.start_time = time.time()
        self.gripper_pub.publisher(Bool(data=True))

    def update(self):
        elapsed = time.time() - self.start_time
        if elapsed >= self.release_hold_time:
            self.gripper_pub.publish(Bool(data=False))
            return Status.SUCCESS

        self.gripper_pub.publish(Bool(data=True))
        return Status.RUNNING

    def terminate(self, new_status):
        self.gripper_pub.publisher(Bool(data=False))

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
        