import unittest
import py_trees
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import Twist
from auv_mission_planner.dropper_mission_node import create_dropper_tree

class TestDropperTreeBehaviour(unittest.TestCase):

    def setUp(self):
        """Runs before every individual test case."""
        py_trees.blackboard.Blackboard.clear()

        self.last_cmd = Twist()
        self.last_gripper = Bool(data=False)

        class MockCmdPublisher:
            def __init__(self, outer):
                self.outer = outer
            def publish(self, msg):
                self.outer.last_cmd = msg

        class MockGripperPublisher:
            def __init__(self, outer):
                self.outer = outer
            def publish(self, msg):
                self.outer.last_gripper = msg

        self.cmd_pub = MockCmdPublisher(self)
        self.gripper_pub = MockGripperPublisher(self)

        self.root = create_dropper_tree(self.cmd_pub, self.gripper_pub)
        self.tree = py_trees.trees.BehaviourTree(self.root)

        self.bb = py_trees.blackboard.Client(name="UnitTestClient")
        self.bb.register_key(key="bin_found", access=py_trees.common.Access.WRITE)
        self.bb.register_key(key="bin_pose", access=py_trees.common.Access.WRITE)

    def tearDown(self):
        """Runs after every individual test case."""
        py_trees.blackboard.Blackboard.clear()

    def test_search_fallback_when_bin_not_found(self):
        """Test: When bin is not detected, vehicle executes search yaw sweep."""
        self.bb.bin_found = Bool(data=False)

        self.tree.tick()

        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertAlmostEqual(self.last_cmd.angular.z, 0.25)
        self.assertAlmostEqual(self.last_cmd.linear.x, 0.0)
        self.assertFalse(self.last_gripper.data)

    def test_sway_centering_when_lateral_offset_large(self):
        """Test: When bin is offset laterally (|x| > 0.05 deadband), vehicle commands sway."""
        # Arrange: norm_x = 0.4 (right), norm_y = 0.0, altitude_z = 1.0m
        self.bb.bin_found = Bool(data=True)
        self.bb.bin_pose = Float32MultiArray(data=[0.4, 0.0, 1.0])

        self.tree.tick()

        # -0.35 * 0.4 = -0.14 m/s sway
        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertAlmostEqual(self.last_cmd.linear.y, -0.14)
        self.assertAlmostEqual(self.last_cmd.linear.x, 0.0)

    def test_surge_centering_when_forward_offset_large(self):
        """Test: When sway is centered (|x| < 0.05) but surge is off, vehicle commands surge."""
        # Arrange: norm_x = 0.01 (centered), norm_y = 0.3 (ahead), altitude_z = 1.0m
        self.bb.bin_found = Bool(data=True)
        self.bb.bin_pose = Float32MultiArray(data=[0.01, 0.3, 1.0])

        self.tree.tick()

        # -0.30 * 0.3 = -0.09 m/s surge
        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertAlmostEqual(self.last_cmd.linear.x, -0.09)
        self.assertAlmostEqual(self.last_cmd.linear.y, 0.0)

    def test_descent_when_centered_and_high(self):
        """Test: When fully centered in XY but high (altitude > 0.40m), vehicle descends."""
        # Arrange: norm_x = 0.01, norm_y = 0.01, altitude_z = 0.90m
        self.bb.bin_found = Bool(data=True)
        self.bb.bin_pose = Float32MultiArray(data=[0.01, 0.01, 0.90])

        self.tree.tick()

        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertAlmostEqual(self.last_cmd.linear.z, -0.20)
        self.assertFalse(self.last_gripper.data)

    def test_release_gripper_when_at_drop_altitude(self):
        """Test: When centered and at drop altitude (altitude <= 0.40m), open gripper."""
        # Arrange: norm_x = 0.01, norm_y = 0.01, altitude_z = 0.35m
        self.bb.bin_found = Bool(data=True)
        self.bb.bin_pose = Float32MultiArray(data=[0.01, 0.01, 0.35])

        self.tree.tick()

        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertTrue(self.last_gripper.data)

    def test_search_timeout_triggers_failure(self):
        """Test: If search sweep exceeds timeout limit, it returns FAILURE."""
        self.bb.bin_found = Bool(data=False)
        self.tree.tick()

        search_node = self.tree.root.children[1].children[0]
        search_node.start_time -= 30.0  # Force timeout expiration

        self.tree.tick()

        self.assertEqual(search_node.status, py_trees.common.Status.FAILURE)

if __name__ == '__main__':
    unittest.main()