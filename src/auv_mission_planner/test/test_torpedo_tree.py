import unittest
import py_trees
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import Twist
from auv_mission_planner.torpedo_tree_node import create_torpedo_tree

class TestTorpedoTreeBehaviour(unittest.TestCase):

    def setUp(self):
        """Runs before every individual test case."""
        py_trees.blackboard.Blackboard.clear()

        self.last_cmd = Twist()
        self.last_fire = Bool(data=False)

        class MockCmdPublisher:
            def __init__(self, outer):
                self.outer = outer
            def publish(self, msg):
                self.outer.last_cmd = msg

        class MockFirePublisher:
            def __init__(self, outer):
                self.outer = outer
            def publish(self, msg):
                self.outer.last_fire = msg

        self.cmd_pub = MockCmdPublisher(self)
        self.fire_pub = MockFirePublisher(self)

        self.root = create_torpedo_tree(self.cmd_pub, self.fire_pub)
        self.tree = py_trees.trees.BehaviourTree(self.root)

        self.bb = py_trees.blackboard.Client(name="UnitTestClient")
        self.bb.register_key(key="torpedo_found", access=py_trees.common.Access.WRITE)
        self.bb.register_key(key="torpedo_pose", access=py_trees.common.Access.WRITE)

    def tearDown(self):
        """Runs after every individual test case."""
        py_trees.blackboard.Blackboard.clear()

    def test_search_fallback_when_target_not_found(self):
        """Test: When target is not detected, vehicle commands a yaw sweep."""
        self.bb.torpedo_found = Bool(data=False)

        self.tree.tick()

        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertAlmostEqual(self.last_cmd.angular.z, 0.25)
        self.assertAlmostEqual(self.last_cmd.linear.x, 0.0)
        self.assertFalse(self.last_fire.data)

    def test_centering_yaw_when_horizontal_offset_large(self):
        """Test: When cutout is off-center (|x| > 0.04 deadband), vehicle commands yaw correction."""
        self.bb.torpedo_found = Bool(data=True)
        self.bb.torpedo_pose = Float32MultiArray(data=[0 .3, 0.0, 2.0])

        self.tree.tick()

        # -0.55 * 0.3 = -0.165 rad/s
        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertAlmostEqual(self.last_cmd.angular.z, -0.165)
        self.assertAlmostEqual(self.last_cmd.linear.x, 0.0)

    def test_depth_alignment_when_depth_misaligned(self):
        """Test: When yaw is aligned (|x| < 0.04) but depth is off, vehicle commands heave correction."""
        self.bb.torpedo_found = Bool(data=True)
        self.bb.torpedo_pose = Float32MultiArray(data=[0.01, 0.25, 2.0])

        self.tree.tick()

        # -0.4 * 0.25 = -0.1 m/s
        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertAlmostEqual(self.last_cmd.linear.z, -0.1)
        self.assertAlmostEqual(self.last_cmd.linear.x, 0.0)

    def test_approach_when_aligned_and_far(self):
        """Test: When yaw and depth are centered but outside firing stand-off (dist > 0.90m), surge forward."""
        self.bb.torpedo_found = Bool(data=True)
        self.bb.torpedo_pose = Float32MultiArray(data=[0.01, 0.01, 1.8])

        self.tree.tick()

        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertAlmostEqual(self.last_cmd.linear.x, 0.25)
        self.assertFalse(self.last_fire.data)

    def test_fire_actuator_when_in_standoff_range(self):
        """Test: When within stand-off range (dist <= 0.90m), trigger pulse is sent to launcher."""
        self.bb.torpedo_found = Bool(data=True)
        self.bb.torpedo_pose = Float32MultiArray(data=[0.01, 0.01, 0.85])

        self.tree.tick()

        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertTrue(self.last_fire.data)

    def test_search_timeout_triggers_failure(self):
        """Test: If search sweep exceeds timeout threshold, it returns FAILURE."""
        self.bb.torpedo_found = Bool(data=False)
        self.tree.tick()

        search_node = self.tree.root.children[1].children[0]
        search_node.start_time -= 30.0

        self.tree.tick()

        self.assertEqual(search_node.status, py_trees.common.Status.FAILURE)

if __name__ == '__main__':
    unittest.main()

    