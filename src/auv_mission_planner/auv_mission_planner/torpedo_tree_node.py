import rclpy
from rclpy.qos import qos_profile_system_default
import py_trees
import py_trees_ros
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import Twist

from auv_mission_planner.torpedo_behaviors import (
    CheckTorpedoTargetVisible,
    CenterTorpedoYaw,
    MatchTorpedoDepth,
    ApproachFiringStandOff,
    FireTorpedoActuator,
    TorpedoSearchSweep
)

def create_torpedo_tree(cmd_publisher, fire_publisher):
    # Dynamic Selector (memory=False ensures condition re-evaluation on every tick)
    root = py_trees.composites.Selector(name="TorpedoMissionRoot", memory=False)

    # 1. Main Strike Sequence
    strike_seq = py_trees.composites.Sequence(name="StrikeSequence", memory=True)
    is_target_detected = CheckTorpedoTargetVisible(name="IsTorpedoTargetVisible")

    align_seq = py_trees.composites.Sequence(name="AlignCutout", memory=True)
    center_yaw = CenterTorpedoYaw(cmd_publisher, deadband=0.04, name="CenterTorpedoYaw")
    match_depth = MatchTorpedoDepth(cmd_publisher, deadband=0.04, name="MatchTorpedoDepth")
    align_seq.add_children([center_yaw, match_depth])

    approach_standoff = ApproachFiringStandOff(cmd_publisher, standoff_distance=0.90, name="ApproachFiringStandOff")
    fire_actuator = FireTorpedoActuator(fire_publisher, pulse_duration=1.5, name="FireTorpedoActuator")

    strike_seq.add_children([is_target_detected, align_seq, approach_standoff, fire_actuator])

    # 2. Fallback Search Branch
    search_seq = py_trees.composites.Sequence(name="FallbackSearchPhase", memory=True)
    search_sweep = TorpedoSearchSweep(cmd_publisher, timeout_sec=25.0, name="TorpedoSearchSweep")
    search_seq.add_children([search_sweep])

    root.add_children([strike_seq, search_seq])
    return root

class TorpedoMissionNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('torpedo_mission_node')
        
        # Actuator and Velocity Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.fire_pub = self.create_publisher(Bool, '/actuators/torpedo_fire', 10)

        # Build Behavior Tree
        tree_root = create_torpedo_tree(self.cmd_pub, self.fire_pub)
        self.tree = py_trees_ros.trees.BehaviourTree(
            root=tree_root,
            unicode_tree_debug=True
        )

        # Blackboard Subscribers
        self.bb_torpedo_found = py_trees_ros.subscribers.ToBlackboard(
            name="SubTorpedoFound",
            topic_name="/vision/torpedo_found",
            topic_type=Bool,
            qos_profile=qos_profile_system_default,
            blackboard_variables={"torpedo_found": None},
            clearing_policy=py_trees.common.ClearingPolicy.NEVER
        )

        self.bb_torpedo_pose = py_trees_ros.subscribers.ToBlackboard(
            name="SubTorpedoPose",
            topic_name="/vision/torpedo_pose",
            topic_type=Float32MultiArray,
            qos_profile=qos_profile_system_default,
            blackboard_variables={"torpedo_pose": None},
            clearing_policy=py_trees.common.ClearingPolicy.NEVER
        )

        self.tree.setup(timeout=15.0, node=self)
        self.timer = self.create_timer(0.1, self.tick_tree)  # 10 Hz Tick

    def tick_tree(self):
        self.tree.tick()


def main(args=None):
    rclpy.init(args=args)
    node = TorpedoMissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.tree.shutdown()
        node.destroy_node()
        rclpy.shutdown()