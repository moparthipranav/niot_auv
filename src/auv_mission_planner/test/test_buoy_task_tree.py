import time
import unittest

from auv_mission_planner.buoy_task_tree import create_buoy_tree
from geometry_msgs.msg import Twist
import py_trees


class TestBuoyTaskTree(unittest.TestCase):
    def setUp(self):
        py_trees.blackboard.Blackboard.clear()
        self.last_cmd = Twist()

        class MockPublisher:
            def __init__(self, outer):
                self.outer = outer

            def publish(self, message):
                self.outer.last_cmd = message

        self.root = create_buoy_tree(MockPublisher(self), target_color='green')
        self.tree = py_trees.trees.BehaviourTree(self.root)
        self.blackboard = py_trees.blackboard.Client(name='BuoyTreeTest')
        for key in (
            'target_found',
            'target_color',
            'detected_color',
            'detection_stamp',
            'norm_x',
            'norm_y',
            'buoy_bbox_fraction',
            'distance_traveled',
        ):
            self.blackboard.register_key(
                key=key, access=py_trees.common.Access.WRITE
            )
        self.blackboard.target_found = True
        self.blackboard.target_color = 'green'
        self.blackboard.detected_color = 'green'
        self.blackboard.detection_stamp = time.monotonic()
        self.blackboard.norm_x = 0.0
        self.blackboard.norm_y = 0.0
        self.blackboard.buoy_bbox_fraction = 0.18
        self.blackboard.distance_traveled = 0.0

    def tearDown(self):
        py_trees.blackboard.Blackboard.clear()

    def tick_with_fresh_detection(self):
        self.blackboard.detection_stamp = time.monotonic()
        self.tree.tick()

    def test_touch_then_retreat_rise_and_pass_buoy(self):
        approach = self.root.children[0].children[1].children[2]
        self.assertEqual(approach.push_duration, 4.0)

        self.tick_with_fresh_detection()
        self.assertEqual(self.root.status, py_trees.common.Status.RUNNING)
        self.assertAlmostEqual(self.last_cmd.linear.x, 0.15)

        self.blackboard.target_found = False
        self.blackboard.buoy_bbox_fraction = 0.0
        self.blackboard.norm_x = 0.8
        self.blackboard.norm_y = 0.8
        for _ in range(40):
            self.tree.tick()

        self.assertAlmostEqual(self.last_cmd.linear.x, -0.15)
        self.assertAlmostEqual(self.last_cmd.linear.z, 0.0)
        self.assertAlmostEqual(self.last_cmd.angular.z, 0.0)

        for _ in range(10):
            self.tick_with_fresh_detection()

        rise = self.root.children[2]
        rise_commands = []
        for _ in range(40):
            self.tick_with_fresh_detection()
            if rise.status == py_trees.common.Status.RUNNING:
                rise_commands.append(self.last_cmd.linear.z)
            else:
                break

        self.assertGreaterEqual(len(rise_commands), 20)
        self.assertTrue(
            all(
                earlier < later
                for earlier, later in zip(rise_commands, rise_commands[1:])
            )
        )
        self.assertGreaterEqual(rise_commands[0], 0.1)
        self.assertLess(rise_commands[0], 0.12)
        self.assertGreater(rise_commands[-1], 0.28)
        self.assertLessEqual(rise_commands[-1], 0.3)
        self.assertEqual(rise.status, py_trees.common.Status.SUCCESS)

        self.assertAlmostEqual(self.last_cmd.linear.x, 0.2)
        self.assertAlmostEqual(self.last_cmd.linear.z, 0.0)

        for _ in range(20):
            if self.root.status == py_trees.common.Status.SUCCESS:
                break
            self.tick_with_fresh_detection()

        self.assertEqual(self.root.status, py_trees.common.Status.SUCCESS)
        self.assertAlmostEqual(self.last_cmd.linear.x, 0.0)
        self.assertAlmostEqual(self.last_cmd.linear.z, 0.0)


if __name__ == '__main__':
    unittest.main()
