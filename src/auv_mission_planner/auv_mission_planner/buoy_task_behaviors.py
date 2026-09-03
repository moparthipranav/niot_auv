import py_trees
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from py_trees.common import Status
import time

class CheckBuoyVisibility(py_trees.behaviour.Behaviour):
    """Checks if the target buoy is visible based on the blackboard variable 'target_found'."""
    def __init__(self, name="CheckBuoyVisibility"): # The initialization happends only once when the behavior is created, not on every tick.
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=self.name)  # Creates a client for the blackboard to access shared data.
        self.blackboard.register_key(key="target_found", access=py_trees.common.Access.READ)

    def update(self):
        try:
            if self.blackboard.target_found:
                return Status.SUCCESS
        except (KeyError, AttributeError):
            pass
        return Status.FAILURE

class CenterBuoyPosition(py_trees.behaviour.Behaviour): # once verify if sway motion is possible, then implement it in the update function
    """Centers the AUV on the target buoy based on the blackboard variable 'buoy_position'. Matches the AUV's position to the buoy's position in the camera frame on a horizontal axis"""
    def __init__(self, cmd_publisher, name="CenterBuoyPosition"): # The cmd_publisher is passed to the behavior to allow it to send movement commands to the AUV.
        super().__init__(name)
        self.cmd_publisher = cmd_publisher
        self.blackboard = py_trees.blackboard.Client(name=self.name)  # Creates a client for the blackboard to access shared data.
        self.blackboard.register_key(key="buoy_position", access=py_trees.common.Access.READ)

    def update(self):
        try:
            # target_position is expected to be a dictionary with 'x' and 'y' keys representing the buoy's position in the camera frame.
            norm_x = self.blackboard.buoy_position.data.x

            # Checking if the buoy is centered within a threshold of 0.05 units in the camera frame.
            if abs(norm_x) < 0.05:
                return Status.SUCCESS

            twist = Twist() # Get the current position of the buoy from the blackboard and create a Twist message to command the AUV's movement.
            twist.angular.z = -0.5 * norm_x  # Proportional control to center the buoy in the camera frame.
            self.cmd_publisher.publish(twist)  # Publish the Twist message to command the A
            return Status.RUNNING  # Return RUNNING to indicate that the behavior is still in progress.
        except (KeyError, AttributeError):
            return Status.FAILURE  # Return FAILURE if the buoy position is not available on the blackboard.

    def terminate(self, new_status):
        self.cmd_publisher.publish(Twist())  # Stop the AUV's movement when the behavior is terminated, regardless of the new status.

class MatchBuoyDepth(py_trees.behaviour.Behaviour):
    """Matches the AUV's depth to the target buoy's depth based on the blackboard variable 'buoy_depth'. Uses the AUV's depth sensor data to adjust its vertical position to align with the buoy's depth. Uses the heave moiton"""
    def __init__(self, cmd_publisher, name="MatchBuoyDepth"):
        super().__init__(name)
        self.cmd_publisher = cmd_publisher
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="buoy_depth", access=py_trees.common.Access.READ)

    def update(self):
        try:
            target_depth = self.blackboard.buoy_depth.data.z  # Get the target depth from the blackboard.
            current_depth = 0.0  # Placeholder for current depth; in practice, this would be obtained from a depth sensor.

            # Check if the AUV is within a threshold of 0.1 units of the target depth.
            if abs(current_depth - target_depth) < 0.1:
                return Status.SUCCESS

            twist = Twist()
            twist.linear.z = 0.5 * (target_depth - current_depth)  # Proportional control to match the depth.
            self.cmd_publisher.publish(twist)  # Publish the Twist message to command the AUV's vertical movement.
            return Status.RUNNING
        except (KeyError, AttributeError):
            return Status.FAILURE

    def terminate(self, new_status):
        self.cmd_publisher.publish(Twist())  # Stop vertical movement when the behavior is terminated.

class ApproachBuoy(py_trees.behaviour.Behaviour):
    """Approaches the target buoy by moving forward based on the blackboard variable 'target_distance'. Uses the AUV's forward motion to close the distance to the buoy."""
    def __init__(self, cmd_publisher, name="ApproachBuoy",contact_distance=0.35):
        super().__init__(name)
        self.cmd_publisher = cmd_publisher
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="target_distance", access=py_trees.common.Access.READ)

    def update(self):
        try:
            target_distance = self.blackboard.target_distance.data  # Get the target distance from the blackboard. This can also be obtained from the yaml file in which an esimtated distance to the buoy is provided. The AUV will use this distance to determine how far it needs to move forward to reach the buoy.
            current_distance = 0.0 # Placeholder for current distance; in practice, this would be obtained from using DVL and INS and calulcating the distance traveled from the starting point.
            if abs(current_distance - target_distance) < self.contact_distance:  # Check if the AUV is within a threshold of the target distance. Verify if the contact distance is appropriate for the AUV to consider it has reached the buoy.
                return Status.SUCCESS

            twist = Twist()
            twist.linear.x = 0.5 * (target_distance - current_distance)  # Proportional control to approach the buoy. The AUV will move forward at a speed proportional to the distance remaining to the target distance.
            self.cmd_publisher.publish(twist)  # Publish the Twist message to command the AUV's forward movement.

            return Status.RUNNING
        except (KeyError, AttributeError):
            return Status.FAILURE

    def terminate(self, new_status):
        self.cmd_publisher.publish(Twist())  # Stop forward movement when the behavior is terminated.

class BuoySearchSweep(py_trees.behaviour.Behaviour):
    """Performs a search sweep pattern to locate the target buoy. This behavior is executed when the buoy is not visible, and it commands the AUV to perform a predefined search pattern."""
    def __init__(self, cmd_publisher, name="BuoySearchSweep"):
        super().__init__(name)
        self.cmd_publisher = cmd_publisher
        self.start_time = None

    def initialise(self):
        self.start_time = time.time()  # Record the start time of the search sweep.

    def update(self):
        elapsed_time = time.time() - self.start_time  # Calculate the elapsed time since the start of the search sweep.
        if elapsed_time < 10.0:  # Perform the search sweep for 10 seconds.
            twist = Twist()
            twist.linear.x = 0.2  # Move forward at a constant speed.
            twist.angular.z = 0.5  # Rotate to create a sweeping motion.
            self.cmd_publisher.publish(twist)  # Publish the Twist message to command the AUV's movement.
            return Status.RUNNING
        else:
            return Status.FAILURE  # Return FAILURE after completing the search sweep.

    def terminate(self, new_status):
        self.cmd_publisher.publish(Twist())  # Stop movement when the behavior is terminated.

