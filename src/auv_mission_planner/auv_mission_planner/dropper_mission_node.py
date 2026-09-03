import rclpy
import py_trees_ros
import py_trees
from rclpy.qos import qos_profile_system_default
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import Twist

from auv_mission_planner.dropper_behaviors import (
    CheckBinDetected,
    CenterBinSway,
    CenterBinSurge,
    DescendToDropAltitude,
    ReleasePayloadGripper,
    BinSearchSweep
)

def create_dropper_tree(cmd_publisher, gripper_publisher):
    # Dynamic Selector (memory=False ensures reactivity on sensor loss)
    root = py_trees.composites.Selector(name="DropperMissionRoot", memory=False)

    # 1. Main Drop Sequence
    drop_seq = py_trees.composites.Sequence(name="DropSequence", memory=True)
    is_bin_detected = CheckBinDetected(name="IsBinDetected")

    align_seq = py_trees.composites.Sequence(name="CenterOverBin", memory=True)
    center_sway = CenterBinSway(cmd_publisher, deadband=0.05, name="CenterBinSway")
    center_surge = CenterBinSurge(cmd_publisher, deadband=0.05, name="CenterBinSurge")
    align_seq.add_children([center_sway, center_surge])

    descend_alt = DescendToDropAltitude(cmd_publisher, target_altitude=0.40, name="DescendToDropAltitude")
    release_payload = ReleasePayloadGripper(gripper_publisher, release_hold_time=2.0, name="ReleasePayloadGripper")

    drop_seq.add_children([is_bin_detected, align_seq, descend_alt, release_payload])

    # 2. Fallback Search Branch
    search_seq = py_trees.composites.Sequence(name="SearchPhaseRoot", memory=True)
    search_sweep = BinSearchSweep(cmd_publisher, timeout_sec=25.0, name="BinSearchSweep")
    search_seq.add_children([search_sweep])

    root.add_children([drop_seq, search_seq])
    return root

class DropperMissionNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('dropper_mission_node')

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.gripper_pub = self.create_publisher(Bool, '/actuators/gripper_open', 10)

        # Build Behavior Tree
        tree_root = create_dropper_tree(self.cmd_pub, self.gripper_pub)
        self.tree = py_trees_ros.trees.BehaviourTree(
            root=tree_root,
            unicode_tree_debug=True
        )

        # py_trees_ros Blackboard Subscribers
        self.bb_bin_found = py_trees_ros.subscribers.ToBlackboard(
            name="SubBinFound",
            topic_name="/vision/bin_found",
            topic_type=Bool,
            qos_profile=qos_profile_system_default,
            blackboard_variables={"bin_found": None},
            clearing_policy=py_trees.common.ClearingPolicy.NEVER
        )

        self.bb_bin_pose = py_trees_ros.subscribers.ToBlackboard(
            name="SubBinPose",
            topic_name="/vision/bin_pose",
            topic_type=Float32MultiArray,
            qos_profile=qos_profile_system_default,
            blackboard_variables={"bin_pose": None},
            clearing_policy=py_trees.common.ClearingPolicy.NEVER
        )

        self.tree.setup(timeout=15.0, node=self)
        self.timer = self.create_timer(0.1, self.tick_tree)  # 10 Hz Execution

    def tick_tree(self):
        self.tree.tick()


def main(args=None):
    rclpy.init(args=args)
    node = DropperMissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.tree.shutdown()
        node.destroy_node()
        rclpy.shutdown()