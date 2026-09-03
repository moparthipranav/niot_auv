import unittest
import py_trees
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import Twist
from auv_mission_planner.gate_mission_node import create_gate_tree

class TestGateTreeBehaviour(unittest.TestCase):

    def setUp(self):
        """Runs before every individual test case."""
        # 1. Clean Blackboard state across test runs
        py_trees.blackboard.Blackboard.clear()

        # 2. Mock command publisher for capturing /cmd_vel Twist messages
        self.last_cmd = Twist()

        class MockPublisher:
            def __init__(self, outer):
                self.outer = outer

            def publish(self, msg):
                self.outer.last_cmd = msg

        self.cmd_pub = MockPublisher(self)

        # 3. Instantiate Behavior Tree
        self.root = create_gate_tree(self.cmd_pub)
        self.tree = py_trees.trees.BehaviourTree(self.root)

        # 4. Register write access to Blackboard keys
        self.bb = py_trees.blackboard.Client(name="UnitTestClient")
        self.bb.register_key(key="gate_found", access=py_trees.common.Access.WRITE)
        self.bb.register_key(key="gate_pose", access=py_trees.common.Access.WRITE)

    def tearDown(self):
        """Runs after every individual test case."""
        py_trees.blackboard.Blackboard.clear()

    def test_search_fallback_when_gate_not_found(self):
        """Test: When gate is not detected, vehicle must execute rotational sweep."""
        # Arrange
        self.bb.gate_found = Bool(data=False)

        # Act
        self.tree.tick()

        # Assert: Search branch active; emits yaw sweep, zero surge
        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertAlmostEqual(self.last_cmd.angular.z, 0.25)
        self.assertAlmostEqual(self.last_cmd.linear.x, 0.0)

    def test_centering_yaw_when_horizontal_offset_large(self):
        """Test: When gate is horizontally off-center (|x| > deadband 0.06), vehicle commands corrective yaw."""
        # Arrange: norm_x = 0.4, norm_y = 0.0, distance_z = 2.0
        self.bb.gate_found = Bool(data=True)
        self.bb.gate_pose = Float32MultiArray(data=[0.4, 0.0, 2.0])

        # Act
        self.tree.tick()

        # Assert: P-gain yaw (-0.55 * 0.4 = -0.22 rad/s)
        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertAlmostEqual(self.last_cmd.angular.z, -0.22)
        self.assertAlmostEqual(self.last_cmd.linear.x, 0.0)

    def test_depth_alignment_when_depth_misaligned(self):
        """Test: When yaw is centered (|x| < 0.06) but depth is off (|y| > 0.06), vehicle commands heave."""
        # Arrange: norm_x = 0.02 (centered), norm_y = 0.3 (gate is lower), distance_z = 2.0
        self.bb.gate_found = Bool(data=True)
        self.bb.gate_pose = Float32MultiArray(data=[0.02, 0.3, 2.0])

        # Act
        self.tree.tick()

        # Assert: P-gain heave (-0.4 * 0.3 = -0.12 m/s)
        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertAlmostEqual(self.last_cmd.linear.z, -0.12)
        self.assertAlmostEqual(self.last_cmd.linear.x, 0.0)

    def test_pass_through_when_fully_aligned(self):
        """Test: When both yaw and depth are within deadbands (< 0.06), vehicle cruises forward."""
        # Arrange: norm_x = 0.02, norm_y = 0.01, distance_z = 2.0
        self.bb.gate_found = Bool(data=True)
        self.bb.gate_pose = Float32MultiArray(data=[0.02, 0.01, 2.0])

        # Act
        self.tree.tick()

        # Assert: PassThroughGate executes with surge speed 0.4 m/s
        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertAlmostEqual(self.last_cmd.linear.x, 0.4)
        self.assertAlmostEqual(self.last_cmd.angular.z, 0.0)

    def test_search_timeout_triggers_failure(self):
        """Test: If search sweep exceeds timeout threshold, it returns FAILURE."""
        # Arrange: Target not detected
        self.bb.gate_found = Bool(data=False)
        self.tree.tick()

        # Fast-forward start time beyond 25.0s timeout limit
        search_node = self.tree.root.children[1].children[0]
        search_node.start_time -= 30.0

        # Act
        self.tree.tick()

        # Assert
        self.assertEqual(search_node.status, py_trees.common.Status.FAILURE)

if __name__ == '__main__':
    unittest.main()