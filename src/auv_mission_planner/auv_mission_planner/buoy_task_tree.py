import py_trees
import py_trees_ros
import rclpy
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default   # ROS2 jazzy requires this update to the import statement for QoS profiles.
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32MultiArray
from py_trees.common import Status

from auv_mission_planner.buoy_task_behaviors import (
    BuoySearchSweep,
    CheckBuoyVisibility,
    CenterBuoyPosition,
    MatchBuoyDepth,
    ApproachBuoy,
    ContactBuoy,
)

def create_buoy_tree(cmd_publisher):
    root = py_trees.composites.Selector(name="BuoyTaskRoot", memory=False)

    # 1. Approach & Contact Branch
    interact_sequence = py_trees.composites.Sequence(name="InteractingPhase", memory=True)
    is_buoy_visible = CheckBuoyVisibility(name="IsBuoyVisible")
    
    align_sequence = py_trees.composites.Sequence(name="AlignAUV", memory=True)
    center_yaw = CenterBuoyPosition(cmd_publisher, name="CenterYaw")
    match_depth = MatchBuoyDepth(cmd_publisher, name="MatchDepth")
    align_sequence.add_children([center_yaw, match_depth])

    approach_touch = ApproachBuoy(cmd_publisher, contact_distance=0.35, name="ApproachAndTouch")
    interact_sequence.add_children([is_buoy_visible, align_sequence, approach_touch])

    # 2. Fallback Search Phase
    search_sequence = py_trees.composites.Sequence(name="FallbackSearchPhase", memory=True)
    search_sweep = BuoySearchSweep(cmd_publisher, timeout_sec=20.0, name="BuoySearchSweep")
    search_sequence.add_children([search_sweep])

    # Assemble Root
    root.add_children([interact_sequence, search_sequence])
    return root

class BuoyMissionNode(rclpy.node.Node):

    def __init__(self):
        super().__init__('buoy_mission_node')

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Build py_trees_ros Tree
        tree_root = create_buoy_tree(self.cmd_pub)
        self.tree = py_trees_ros.trees.BehaviourTree(
            root=tree_root,
            unicode_tree_debug=True
        )

        # Blackboard Subscriber 1: Buoy Found Flag (with QoS profile)
        self.bb_buoy_found = py_trees_ros.subscribers.ToBlackboard( # ROS2 messages arrive asynchronously, so we use a subscriber to update the blackboard with the latest buoy found status.
            # The ToBlackboard subscriber listens to the /vision/target_found topic, which publishes a Bool message indicating whether the target buoy has been detected by the vision system. The subscriber updates the blackboard variable "target_found" with the latest value from the topic, allowing the behavior tree to make decisions based on the current visibility of the buoy.
            name="SubBuoyFound",
            topic_name="/vision/target_found",
            topic_type=Bool,
            qos_profile=qos_profile_system_default,
            blackboard_variables={"target_found": None},
            clearing_policy=py_trees.common.ClearingPolicy.NEVER
        )

        # Blackboard Subscriber 2: Buoy Pose / Angle (with QoS profile)
        self.bb_target_pose = py_trees_ros.subscribers.ToBlackboard(
            # The ToBlackboard subscriber listens to the /vision/target_pose topic, which publishes a Float32MultiArray message containing the pose of the target buoy. The subscriber updates the blackboard variable "target_pose" with the latest value from the topic, allowing the behavior tree to make decisions based on the current position of the buoy.
            name="SubBuoyPose",
            topic_name="/vision/target_pose",
            topic_type=Float32MultiArray,
            qos_profile=qos_profile_system_default,
            blackboard_variables={"target_pose": None},
            clearing_policy=py_trees.common.ClearingPolicy.NEVER
        )

        self.tree.setup(timeout=15, node=self)  # Setup the behavior tree with a timeout of 15 seconds. This allows for initialization of the tree and its components.
        self.timer = self.create_timer(0.1, self.tick_tree)  # Create a timer to tick the behavior tree at a frequency of 10 Hz (every 0.1 seconds).

    def tick_tree(self):
        self.tree.tick()  # Tick the behavior tree to update its state and execute behaviors based on the current conditions and blackboard data.


