"""Integration tests for ROS mission command routing."""

import time
import unittest

from geometry_msgs.msg import Twist

from auv_mission_planner.buoy_task_tree import BuoyMissionNode
from auv_mission_planner.dropper_mission_node import DropperMissionNode
from auv_mission_planner.gate_mission_node import GateMissionNode
from auv_mission_planner.path_tree_node import PathMissionNode
from auv_mission_planner.torpedo_tree_node import TorpedoMissionNode

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from reactive_control_layer.control_layer import ReactiveControlLayer

from std_msgs.msg import Bool, String, UInt8


class TestReactiveControlCommunication(unittest.TestCase):
    """Verify mission publishers are consumed by the reactive controller."""

    @classmethod
    def setUpClass(cls):
        """Initialize ROS once for this test class."""
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Shut down the ROS context after the test class."""
        rclpy.shutdown()

    def test_mission_commands_reach_controller(self):
        """Forward each mission command and verify M64 conversion."""
        controller = ReactiveControlLayer()
        probe = Node('reactive_control_test_probe')
        executor = SingleThreadedExecutor()
        mission_nodes = []
        selected_commands = []
        m64_commands = []

        enable_publisher = probe.create_publisher(Bool, '/control/enable', 10)
        source_publisher = probe.create_publisher(
            String, '/control/select_source', 10
        )
        probe.create_subscription(
            Twist,
            '/control/selected_cmd_vel',
            selected_commands.append,
            10,
        )
        probe.create_subscription(
            UInt8,
            '/m64/command',
            m64_commands.append,
            10,
        )

        mission_types = (
            ('path', PathMissionNode),
            ('gate', GateMissionNode),
            ('buoy', BuoyMissionNode),
            ('dropper', DropperMissionNode),
            ('torpedo', TorpedoMissionNode),
        )

        try:
            for _, node_type in mission_types:
                mission_nodes.append(node_type())

            executor.add_node(controller)
            executor.add_node(probe)
            self.assertTrue(self._spin_until(
                executor,
                lambda: (
                    enable_publisher.get_subscription_count() > 0
                    and source_publisher.get_subscription_count() > 0
                    and (
                        controller.selected_twist_pub.get_subscription_count() > 0
                    )
                    and controller.command_pub.get_subscription_count() > 0
                ),
            ))
            enable_publisher.publish(Bool(data=True))
            self.assertTrue(
                self._spin_until(executor, lambda: controller.enabled)
            )

            for index, ((source, _), mission) in enumerate(
                zip(mission_types, mission_nodes), start=1
            ):
                self.assertTrue(
                    self._spin_until(
                        executor,
                        lambda: mission.cmd_pub.get_subscription_count() > 0,
                    )
                )
                source_publisher.publish(String(data=source))
                self.assertTrue(
                    self._spin_until(
                        executor,
                        lambda: controller.selected_source == source,
                    )
                )

                expected_speed = index / 10.0
                command = Twist()
                command.linear.x = expected_speed
                mission.cmd_pub.publish(command)
                selected_commands.clear()

                forwarded = self._spin_until(
                    executor,
                    lambda: any(
                        abs(message.linear.x - expected_speed) < 1e-6
                        for message in selected_commands
                    ),
                )
                self.assertTrue(
                    forwarded,
                    f'{source} mission command was not forwarded',
                )
                if index == 1:
                    m64_forwarded = self._spin_until(
                        executor,
                        lambda: any(
                            message.data == ord('W')
                            for message in m64_commands
                        ),
                    )
                    self.assertTrue(
                        m64_forwarded,
                        'forward command was not converted to the '
                        'M64 W command',
                    )
        finally:
            executor.shutdown()
            for node in mission_nodes:
                node.tree.shutdown()
                node.destroy_node()
            controller.destroy_node()
            probe.destroy_node()

    @staticmethod
    def _spin_until(executor, predicate, timeout_sec=2.0):
        """Spin until a condition succeeds or its deadline is reached."""
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
            if predicate():
                return True
        return predicate()


if __name__ == '__main__':
    unittest.main()
