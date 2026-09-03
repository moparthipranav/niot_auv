import unittest
import py_trees
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist, Vector3
from auv_mission_planner.path_tree_node import create_path_tree

class TestPathBehaviorTree(unittest.TestCase):
    def setUp(self):
        """Runs before every individual test case."""
        # 1. Clean Blackboard state across tests
        py_trees.blackboard.Blackboard.clear()

        # 2. Mock command publisher to capture output /cmd_vel Twist messages
        self.last_cmd = Twist()

        class MockPublisher:
            def __init__(self, outer):
                self.outer = outer
            def publish(self, msg):
                self.outer.last_cmd = msg

        self.cmd_pub = MockPublisher(self)

        # 3. Build the Behavior Tree
        self.root = create_path_tree(self.cmd_pub)
        self.tree = py_trees.trees.BehaviourTree(self.root)

        # 4. Register write access to exact Blackboard keys
        self.bb = py_trees.blackboard.Client(name="UnitTestClient")
        self.bb.register_key(key="marker_found", access=py_trees.common.Access.WRITE)
        self.bb.register_key(key="marker_pose", access=py_trees.common.Access.WRITE)

    def tearDown(self):
        """Runs after every individual test case."""
        py_trees.blackboard.Blackboard.clear()

    def test_search_fallback_when_marker_not_found(self):
        """Test: When marker is not detected, the tree must run the search sweep."""
        # Arrange
        self.bb.marker_found = Bool(data=False)

        # Act: Use self.tree.tick() (NOT tick_once)
        self.tree.tick()

        # Assert: Search branch must be active and commanding yaw sweep
        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertAlmostEqual(self.last_cmd.angular.z, 0.25)
        self.assertAlmostEqual(self.last_cmd.linear.x, 0.0)

    def test_centering_sway_when_marker_offset_large(self):
        """Test: When marker is off-center (> 5% deadband), vehicle must command sway velocity."""
        # Arrange
        self.bb.marker_found = Bool(data=True)
        self.bb.marker_pose = Vector3(x=0.4, y=0.0, z=0.0)

        # Act
        self.tree.tick()

        # Assert: CenterXY must emit linear.y = -0.3 * 0.4 = -0.12 m/s
        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertAlmostEqual(self.last_cmd.linear.y, -0.12)
        self.assertAlmostEqual(self.last_cmd.linear.x, 0.0)

    def test_heading_alignment_when_marker_missaligned(self):
        """Test: When centered (x < 0.05) but misaligned in yaw, vehicle must command yaw rate."""
        # Arrange
        self.bb.marker_found = Bool(data=True)
        self.bb.marker_pose = Vector3(x=0.02, y=0.0, z=0.2)

        # Act
        self.tree.tick()

        # Assert: AlignHeading must emit angular.z = 0.5 * 0.2 = 0.1 rad/s
        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertAlmostEqual(self.last_cmd.angular.z, 0.1)
        self.assertAlmostEqual(self.last_cmd.linear.y, 0.0)

    def test_traversal_forward_cruise_when_aligned(self):
        """Test: When within both XY and heading deadbands, vehicle must cruise forward."""
        # Arrange
        self.bb.marker_found = Bool(data=True)
        self.bb.marker_pose = Vector3(x=0.01, y=0.0, z=0.01)

        # Act
        self.tree.tick()

        # Assert: TraversePath must emit linear.x = 0.35 m/s
        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertAlmostEqual(self.last_cmd.linear.x, 0.35)
        self.assertAlmostEqual(self.last_cmd.angular.z, 0.0)

    def test_search_timeout_triggers_failure(self):
        """Test: If search runs longer than timeout limit, it returns FAILURE."""
        # Arrange
        self.bb.marker_found = Bool(data=False)
        self.tree.tick()

        # Locate BottomCamSearch leaf node in Fallback branch
        search_node = self.tree.root.children[1].children[0]
        search_node.start_time -= 30.0  # Force timeout expiration

        # Act
        self.tree.tick()

        # Assert
        self.assertEqual(search_node.status, py_trees.common.Status.FAILURE)

if __name__ == '__main__':
    unittest.main()