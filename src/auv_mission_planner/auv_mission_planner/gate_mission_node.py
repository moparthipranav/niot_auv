import rclpy
from rclpy.qos import qos_profile_system_default
import py_trees
import py_trees_ros
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from py_trees.common import Status

from auv_mission_planner.gate_behaviors import (
    CheckGateDetected,
    CenterGateYaw,
    MatchGateDepth,
    PassThroughGate,
    GateSearchSweep
)

def create_gate_tree(cmd_publisher):
    # Dynamic Selector (memory=False ensures condition re-evaluation on every tick)
    root = py_trees.composites.Selector(name="LGateMissionRoot", memory=False)

    # 1. Main Pass-Through Branch
    traverse_seq = py_trees.composites.Sequence(name="TraverseSequence", memory=True)
    is_gate_detected = CheckGateDetected(name="IsGateDetected")

    align_seq = py_trees.composites.Sequence(name="AlignToGate", memory=True)
    center_yaw = CenterGateYaw(cmd_publisher, name="CenterGateYaw")
    match_depth = MatchGateDepth(cmd_publisher, name="MatchGateDepth")
    align_seq.add_children([center_yaw, match_depth])

    pass_through = PassThroughGate(cmd_publisher, pass_duration=12.0, cruise_speed=0.4, name="PassThroughGate")
    traverse_seq.add_children([is_gate_detected, align_seq, pass_through])

    # 2. Fallback Search Branch
    search_seq = py_trees.composites.Sequence(name="SearchPhaseRoot", memory=True)
    search_sweep = GateSearchSweep(cmd_publisher, timeout_sec=25.0, name="GateSearchSweep")
    search_seq.add_children([search_sweep])

    root.add_children([traverse_seq, search_seq])
    return root

class GateMissionNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('gate_mission_node')
        self.cmd_pub = self.create_publisher(Twist, '/control/requests/gate', 10)
        self.task_completed_pub = self.create_publisher(
            String, '/mission/task_completed', 10
        )

        # Build py_trees_ros Behaviour Tree
        tree_root = create_gate_tree(self.cmd_pub)
        self.tree = py_trees_ros.trees.BehaviourTree(
            root=tree_root,
            unicode_tree_debug=True
        )

        # Blackboard Subscribers with QoS profiles
        self.bb_gate_found = py_trees_ros.subscribers.ToBlackboard(
            name="SubGateFound",
            topic_name="/vision/gate_found",
            topic_type=Bool,
            qos_profile=qos_profile_system_default,
            blackboard_variables={"gate_found": None},
            clearing_policy=py_trees.common.ClearingPolicy.NEVER
        )

        self.bb_gate_pose = py_trees_ros.subscribers.ToBlackboard(
            name="SubGatePose",
            topic_name="/vision/gate_pose",
            topic_type=Float32MultiArray,
            qos_profile=qos_profile_system_default,
            blackboard_variables={"gate_pose": None},
            clearing_policy=py_trees.common.ClearingPolicy.NEVER
        )

        self.tree.setup(timeout=15.0, node=self)
        self.timer = self.create_timer(0.1, self.tick_tree) # 10 Hz Execution

    def tick_tree(self):
        self.tree.tick()
        if self.tree.root.status == Status.SUCCESS:
            self.task_completed_pub.publish(String(data='gate'))

def main(args=None):
    rclpy.init(args=args)
    node = GateMissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.tree.shutdown()
        node.destroy_node()
        rclpy.shutdown()
