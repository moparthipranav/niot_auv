import rclpy
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default
import py_trees
import py_trees_ros
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist, Vector3

from auv_mission_planner.path_behaviors import (
    CheckMarkerVisibility,
    CenterXYPosition,
    AlignHeading,
    TraversePath,
    BottomCamSearch,
)

def create_path_tree(cmd_publisher):
    root = py_trees.composites.Selector(name="PathFollowingRoot", memory=False)

    # 1. Track & Traverse Branch
    track_sequence = py_trees.composites.Sequence(name="TrackAndTraverse", memory=True)
    is_marker_visible = CheckMarkerVisibility(name="IsMarkerVisible")
    
    align_sequence = py_trees.composites.Sequence(name="AlignAUV", memory=True)
    center_xy = CenterXYPosition(cmd_publisher, name="CenterXY")
    align_yaw = AlignHeading(cmd_publisher, name="AlignHeading")
    align_sequence.add_children([center_xy, align_yaw])

    traverse_action = TraversePath(cmd_publisher, name="TraversePath")
    track_sequence.add_children([is_marker_visible, align_sequence, traverse_action])

    # 2. Fallback Search Branch
    search_sequence = py_trees.composites.Sequence(name="SearchPhaseRoot", memory=True)
    search_pattern = BottomCamSearch(cmd_publisher, timeout=25.0, name="BottomCamSearch")
    search_sequence.add_children([search_pattern])

    root.add_children([track_sequence, search_sequence])
    return root

class PathMissionNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('path_mission_node')
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Build py_trees_ros Tree
        tree_root = create_path_tree(self.cmd_pub)
        self.tree = py_trees_ros.trees.BehaviourTree(
            root=tree_root,
            unicode_tree_debug=True
        )

        # Blackboard Subscriber 1: Marker Found Flag (with QoS profile)
        self.bb_marker_found = py_trees_ros.subscribers.ToBlackboard(
            name="SubMarkerFound",
            topic_name="/vision/marker_found",
            topic_type=Bool,
            qos_profile=qos_profile_system_default,
            blackboard_variables={"marker_found": None},
            clearing_policy=py_trees.common.ClearingPolicy.NEVER
        )

        # Blackboard Subscriber 2: Marker Pose / Angle (with QoS profile)
        self.bb_marker_pose = py_trees_ros.subscribers.ToBlackboard(
            name="SubMarkerPose",
            topic_name="/vision/marker_pose",
            topic_type=Vector3,
            qos_profile=qos_profile_system_default,
            blackboard_variables={"marker_pose": None},
            clearing_policy=py_trees.common.ClearingPolicy.NEVER
        )

        # Setup tree with ROS node interface
        self.tree.setup(timeout=15.0, node=self)
        self.timer = self.create_timer(0.1, self.tick_tree)  # 10 Hz tick loop

    def tick_tree(self):
        self.tree.tick()

def main(args=None):
    rclpy.init(args=args)
    node = PathMissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.tree.shutdown()
        node.destroy_node()
        rclpy.shutdown()