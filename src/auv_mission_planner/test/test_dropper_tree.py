import unittest
import time
import py_trees
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from auv_interfaces.msg import BinDetection
from auv_mission_planner.dropper_mission_node import create_dropper_tree


class TestDropperTreeBehaviour(unittest.TestCase):

    def setUp(self):
        """Runs before every individual test case."""
        py_trees.blackboard.Blackboard.clear()

        self.last_cmd = Twist()
        self.cmd_history = []
        self.last_gripper = Bool(data=False)
        self.gripper_history = []
        self.now = time.time()

        class MockCmdPublisher:
            def __init__(self, outer):
                self.outer = outer
            def publish(self, msg):
                self.outer.last_cmd = msg
                self.outer.cmd_history.append(msg)

        class MockGripperPublisher:
            def __init__(self, outer):
                self.outer = outer
            def publish(self, msg):
                self.outer.last_gripper = msg
                self.outer.gripper_history.append(msg.data)

        self.cmd_pub = MockCmdPublisher(self)
        self.gripper_pub = MockGripperPublisher(self)

        self.root = create_dropper_tree(
            self.cmd_pub, self.gripper_pub, clock=lambda: self.now,
            detection_confirm_sec=0.0,
        )
        self.tree = py_trees.trees.BehaviourTree(self.root)

        self.bb = py_trees.blackboard.Client(name="UnitTestClient")
        self.bb.register_key(
            key="bin_detection", access=py_trees.common.Access.WRITE
        )
        self.bb.register_key(
            key="vehicle_stable", access=py_trees.common.Access.WRITE
        )
        self.bb.register_key(
            key="vehicle_stable_received_at",
            access=py_trees.common.Access.WRITE
        )

    def set_detection(self, x=0.01, y=0.01, z=1.0, age=0.0,
                      detected=True, marker_detected=True):
        detection = BinDetection()
        timestamp = self.now - age
        detection.header.stamp.sec = int(timestamp)
        detection.header.stamp.nanosec = int(
            (timestamp - int(timestamp)) * 1e9
        )
        detection.detected = detected
        detection.marker_detected = marker_detected
        detection.x = x
        detection.y = y
        detection.z = z
        detection.yaw = 0.0
        detection.confidence = 0.9 if marker_detected else 0.0
        self.bb.bin_detection = detection
        self.set_vehicle_stable(True)

    def set_vehicle_stable(self, stable, age=0.0):
        self.bb.vehicle_stable = stable
        self.bb.vehicle_stable_received_at = self.now - age

    def tearDown(self):
        """Runs after every individual test case."""
        py_trees.blackboard.Blackboard.clear()

    def test_search_fallback_when_bin_not_found(self):
        """Test: When bin is not detected, vehicle executes search yaw sweep."""
        self.set_detection(detected=False, marker_detected=False)

        self.tree.tick()

        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertAlmostEqual(self.last_cmd.angular.z, 0.25)
        self.assertAlmostEqual(self.last_cmd.linear.x, 0.0)
        self.assertFalse(self.last_gripper.data)

    def test_detection_must_persist_before_aligning_without_marker(self):
        """A stable bin detection starts alignment before marker detection."""
        self.root = create_dropper_tree(
            self.cmd_pub, self.gripper_pub, clock=lambda: self.now,
            detection_confirm_sec=1.0,
        )
        self.tree = py_trees.trees.BehaviourTree(self.root)
        self.set_detection(x=0.4, y=0.0, z=2.5, marker_detected=False)

        self.tree.tick()
        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertEqual(self.last_cmd, Twist())

        self.now += 1.0
        self.set_detection(x=0.4, y=0.0, z=2.5, marker_detected=False)
        self.tree.tick()
        self.assertAlmostEqual(self.last_cmd.linear.y, -0.14)
        self.assertEqual(self.last_cmd.linear.z, 0.0)

        self.set_detection(x=0.0, y=0.0, z=2.5, marker_detected=False)
        self.tree.tick()
        self.assertEqual(self.last_cmd, Twist())
        self.assertFalse(self.last_gripper.data)

    def test_sway_centering_when_lateral_offset_large(self):
        """Test: When bin is offset laterally (|x| > 0.05 deadband), vehicle commands sway."""
        # Arrange: norm_x = 0.4 (right), norm_y = 0.0, altitude_z = 1.0m
        self.set_detection(x=0.4, y=0.0)

        self.tree.tick()

        # -0.35 * 0.4 = -0.14 m/s sway
        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertAlmostEqual(self.last_cmd.linear.y, -0.14)
        self.assertAlmostEqual(self.last_cmd.linear.x, 0.0)

    def test_surge_centering_when_forward_offset_large(self):
        """Test: When sway is centered (|x| < 0.05) but surge is off, vehicle commands surge."""
        # Arrange: norm_x = 0.01 (centered), norm_y = 0.3 (ahead), altitude_z = 1.0m
        self.set_detection(x=0.01, y=0.3)

        self.tree.tick()

        # -0.30 * 0.3 = -0.09 m/s surge
        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertAlmostEqual(self.last_cmd.linear.x, -0.09)
        self.assertAlmostEqual(self.last_cmd.linear.y, 0.0)

    def test_bin_within_alignment_tolerance_skips_horizontal_commands(self):
        """Offsets within tolerance count as aligned without perfect centering."""
        self.set_detection(x=0.049, y=-0.05, z=0.90)

        self.tree.tick()

        alignment_check = self.tree.root.children[0].children[2].children[0]
        self.assertEqual(
            alignment_check.status, py_trees.common.Status.SUCCESS
        )
        self.assertIn("horizontal thruster commands are not needed",
                      alignment_check.feedback_message)
        self.assertTrue(all(
            command.linear.x == 0.0 and command.linear.y == 0.0
            for command in self.cmd_history
        ))
        self.assertAlmostEqual(self.last_cmd.linear.z, -0.20)

    def test_descent_when_centered_and_high(self):
        """Test: When fully centered in XY but high (altitude > 0.40m), vehicle descends."""
        # Arrange: norm_x = 0.01, norm_y = 0.01, altitude_z = 0.90m
        self.set_detection(z=0.90)

        self.tree.tick()

        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertAlmostEqual(self.last_cmd.linear.z, -0.20)
        self.assertFalse(self.last_gripper.data)

    def test_realigns_when_marker_moves_during_descent(self):
        self.set_detection(z=0.90)
        self.tree.tick()
        self.assertAlmostEqual(self.last_cmd.linear.z, -0.20)

        self.set_detection(x=0.4, z=0.90)
        self.tree.tick()

        self.assertAlmostEqual(self.last_cmd.linear.y, -0.14)
        self.assertAlmostEqual(self.last_cmd.linear.z, 0.0)

    def test_release_gripper_when_at_drop_altitude(self):
        """Test: When centered and at drop altitude (altitude <= 0.40m), open gripper."""
        # Arrange: norm_x = 0.01, norm_y = 0.01, altitude_z = 0.35m
        self.set_detection(z=0.35)

        self.tree.tick()

        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertTrue(self.last_gripper.data)

    def test_invalid_range_does_not_release_gripper(self):
        """A missing/invalid range must not trigger a payload release."""
        self.set_detection(z=0.0)

        self.tree.tick()

        self.assertFalse(self.last_gripper.data)
        self.assertAlmostEqual(self.last_cmd.angular.z, 0.25)

    def test_payload_release_is_one_shot(self):
        self.set_detection(z=0.35)
        release_node = self.tree.root.children[0].children[6]
        release_node.release_hold_time = 0.0

        self.tree.tick()
        self.tree.tick()

        self.assertEqual(self.gripper_history.count(True), 1)
        self.assertFalse(self.last_gripper.data)

    def test_unstable_vehicle_holds_position_without_releasing(self):
        self.set_detection(z=0.35)
        self.set_vehicle_stable(False)

        self.tree.tick()

        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertEqual(self.last_cmd, Twist())
        self.assertFalse(self.last_gripper.data)

    def test_stale_pose_stops_an_active_release_pulse(self):
        self.set_detection(z=0.35)
        release_node = self.tree.root.children[0].children[6]
        release_node.release_hold_time = 10.0

        self.tree.tick()
        self.assertTrue(self.last_gripper.data)

        detection = self.bb.bin_detection
        detection.header.stamp.sec -= 2
        self.bb.bin_detection = detection
        self.tree.tick()

        self.assertFalse(self.last_gripper.data)

    def test_unstable_vehicle_stops_an_active_release_pulse(self):
        self.set_detection(z=0.35)
        release_node = self.tree.root.children[0].children[6]
        release_node.release_hold_time = 10.0

        self.tree.tick()
        self.assertTrue(self.last_gripper.data)

        self.set_vehicle_stable(False)
        self.tree.tick()

        self.assertFalse(self.last_gripper.data)

    def test_search_timeout_triggers_failure(self):
        """Test: If search sweep exceeds timeout limit, it returns FAILURE."""
        self.set_detection(detected=False, marker_detected=False)
        self.tree.tick()

        search_node = self.tree.root.children[1].children[0]
        search_node.start_time -= 30.0  # Force timeout expiration

        self.tree.tick()

        self.assertEqual(search_node.status, py_trees.common.Status.FAILURE)

    def test_stale_detection_cannot_drive_or_release(self):
        self.set_detection(x=0.4, z=0.35, age=2.0)

        self.tree.tick()

        self.assertAlmostEqual(self.last_cmd.angular.z, 0.25)
        self.assertAlmostEqual(self.last_cmd.linear.y, 0.0)
        self.assertFalse(self.last_gripper.data)

    def test_bin_without_marker_aligns_but_does_not_release(self):
        self.set_detection(x=0.4, z=0.35, marker_detected=False)

        self.tree.tick()

        self.assertAlmostEqual(self.last_cmd.angular.z, 0.0)
        self.assertAlmostEqual(self.last_cmd.linear.y, -0.14)
        self.assertAlmostEqual(self.last_cmd.linear.z, 0.0)
        self.assertFalse(self.last_gripper.data)

if __name__ == '__main__':
    unittest.main()