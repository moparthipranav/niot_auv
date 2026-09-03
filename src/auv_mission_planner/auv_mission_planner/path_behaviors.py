import py_trees
from py_trees.common import Status
import rclpy
from geometry_msgs.msg import Twist # Twist is a standard message type used to represent velocity in space. 
import time

class CheckMarkerVisibility(py_trees.behaviour.Behaviour):
    """Checks if the bottom camera has a lock on the orange marker or track. """
    def __init__(self, name="IsMarkerVisible"):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="marker_found", access=py_trees.common.Access.READ) # Giving a read only access to "marker_found" variable

    def update(self):
        try:
            if self.blackboard.marker_found.data:
                return Status.SUCCESS
        except (KeyError, AttributeError):
            pass
        return Status.FAILURE

class CenterXYPosition(py_trees.behaviour.Behaviour):
    """Corrects lateral and forward offsets over the path strip."""
    def __init__(self, cmd_publisher, name):
        super().__init__(name)
        self.cmd_pub = cmd_publisher
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="marker_pose", access=py_trees.common.Access.READ)

    def update(self):
        try:
            pose = self.blackboard.marker_pose
            x_offset = pose.x

            if abs(x_offset) < 0.05: # Reconfigure this threshold value
                return Status.SUCCESS

            twist = Twist()
            twist.linear.y = -0.3 * x_offset # Correcting the sway veloctiy
            self.cmd_pub.publish(twist)
            return Status.RUNNING
        except (KeyError, AttributeError):
            return Status.FAILURE

    def terminate(self, new_status):
        self.cmd_pub.publish(Twist())


class AlignHeading(py_trees.behaviour.Behaviour):
    """Aligns vehicle yaw with the orientation of the orange path."""
    def __init__(self, cmd_publisher, name):
        super().__init__(name)
        self.cmd_pub = cmd_publisher
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="marker_pose", access=py_trees.common.Access.READ)

    def update(self):
        try:
            pose = self.blackboard.marker_pose
            heading_err = pose.z # Angle error in radians

            # Alignment
            if abs(heading_err) < 0.05:
                return Status.SUCCESS

            twist = Twist()
            twist.angular.z = 0.5 * heading_err # Proportional yaw control
            self.cmd_pub.publish(twist)
            return Status.RUNNING
        except (KeyError, AttributeError):
            return Status.FAILURE

    def terminate(self, new_status):
        self.cmd_pub.publish(Twist())

class TraversePath(py_trees.behaviour.Behaviour):
    """Surges forward along the aligned vector of the path marker. """
    def __init__(self, cmd_publisher, name):
        super().__init__(name)
        self.cmd_pub = cmd_publisher

    def update(self):
        twist = Twist()
        twist.linear.x = 0.35 # Controls the forward cruise speed (m/s)
        self.cmd_pub.publish(twist)
        return Status.RUNNING

    def terminate(self, new_status):
        self.cmd_pub.publish(Twist())

class BottomCamSearch(py_trees.behaviour.Behaviour):
    """Searching to re-acquire the path marker"""
    def __init__(self, cmd_publisher, timeout = 20.0, name="BottomCamSearch"):
        super().__init__(name)
        self.cmd_pub = cmd_publisher
        self.timeout = timeout
        self.start_time = None

    def initialise(self):
        self.start_time = time.time()

    def update(self):

        if (time.time() - self.start_time) > self.timeout: # Timeout happens
            return Status.FAILURE

        twist = Twist()
        twist.angular.z = 0.25 # Sweep Yaw
        self.cmd_pub.publish(twist)
        return Status.RUNNING

    def terminate(self, new_status):
        self.cmd_pub.publish(Twist())

        