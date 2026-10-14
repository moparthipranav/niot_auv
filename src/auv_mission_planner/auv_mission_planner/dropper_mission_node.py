import time
import rclpy
import py_trees_ros
import py_trees
from rclpy.qos import qos_profile_system_default
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from auv_interfaces.msg import BinDetection

from auv_mission_planner.dropper_behaviors import (
    CheckBinDetected,
    CheckBinAligned,
    CenterBinSway,
    CenterBinSurge,
    DescendToDropAltitude,
    WaitForVehicleStable,
    ReleasePayloadGripper,
    BinSearchSweep
)

def create_dropper_tree(cmd_publisher, gripper_publisher, clock=None,
                        max_detection_age=0.75, alignment_tolerance=0.05):
    clock = clock or time.time
    # Dynamic Selector (memory=False ensures reactivity on sensor loss)
    root = py_trees.composites.Selector(name="DropperMissionRoot", memory=False)

    # 1. Main Drop Sequence
    drop_seq = py_trees.composites.Sequence(name="DropSequence", memory=False)
    is_bin_detected = CheckBinDetected(
        clock=clock, max_age=max_detection_age, name="IsBinDetected"
    )

    align_seq = py_trees.composites.Sequence(name="CenterOverBin", memory=False)
    center_sway = CenterBinSway(
        cmd_publisher, deadband=alignment_tolerance, clock=clock,
        max_age=max_detection_age, name="CenterBinSway"
    )
    center_surge = CenterBinSurge(
        cmd_publisher, deadband=alignment_tolerance, clock=clock,
        max_age=max_detection_age, name="CenterBinSurge"
    )
    align_seq.add_children([center_sway, center_surge])

    alignment_check = CheckBinAligned(
        tolerance=alignment_tolerance, clock=clock,
        max_age=max_detection_age,
    )
    align_selector = py_trees.composites.Selector(
        name="AlignToBin", memory=False
    )
    align_selector.add_children([alignment_check, align_seq])

    descend_alt = DescendToDropAltitude(
        cmd_publisher, target_altitude=0.40, clock=clock,
        max_age=max_detection_age, name="DescendToDropAltitude"
    )
    wait_for_stability = WaitForVehicleStable(
        cmd_publisher, clock=clock, max_age=max_detection_age
    )
    release_payload = ReleasePayloadGripper(
        gripper_publisher, release_hold_time=2.0, clock=clock,
        max_age=max_detection_age, name="ReleasePayloadGripper"
    )

    drop_seq.add_children(
        [is_bin_detected, align_selector, descend_alt, wait_for_stability,
         release_payload]
    )

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
        self.cmd_pub = self.create_publisher(Twist, '/control/requests/dropper', 10)
        self.gripper_pub = self.create_publisher(Bool, '/actuators/gripper_open', 10)

        # Build Behavior Tree
        tree_root = create_dropper_tree(
            self.cmd_pub,
            self.gripper_pub,
            clock=lambda: self.get_clock().now().nanoseconds * 1e-9,
        )
        self.tree = py_trees_ros.trees.BehaviourTree(
            root=tree_root,
            unicode_tree_debug=True
        )

        self.blackboard = py_trees.blackboard.Client(
            name="DropperMissionNode"
        )
        self.blackboard.register_key(
            key="bin_detection", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="vehicle_stable", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="vehicle_stable_received_at",
            access=py_trees.common.Access.WRITE
        )
        self.stability_subscription = self.create_subscription(
            Bool,
            '/state/vehicle_stable',
            self.vehicle_stability_callback,
            qos_profile_system_default,
        )
        self.bin_detection_subscription = self.create_subscription(
            BinDetection,
            '/vision/bin_detection',
            self.bin_detection_callback,
            qos_profile_system_default,
        )

        self.tree.setup(timeout=15.0, node=self)
        self.timer = self.create_timer(0.1, self.tick_tree)  # 10 Hz Execution

    def tick_tree(self):
        self.tree.tick()

    def bin_detection_callback(self, message):
        self.blackboard.bin_detection = message

    def vehicle_stability_callback(self, message):
        self.blackboard.vehicle_stable = message.data
        self.blackboard.vehicle_stable_received_at = (
            self.get_clock().now().nanoseconds * 1e-9
        )


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
