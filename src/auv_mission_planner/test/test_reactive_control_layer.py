"""Integration tests for ROS mission command routing."""

import json
import struct
import time
import unittest

from geometry_msgs.msg import Twist
from auv_interfaces.msg import BinDetection

from auv_mission_planner.buoy_task_tree import BuoyMissionNode
from auv_mission_planner.dropper_mission_node import DropperMissionNode
from auv_mission_planner.gate_mission_node import GateMissionNode
from auv_mission_planner.path_tree_node import PathMissionNode
from auv_mission_planner.torpedo_tree_node import TorpedoMissionNode

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from reactive_control_layer.control_layer import ReactiveControlLayer, crc8

from std_msgs.msg import Bool, String


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

    def test_mission_commands_reach_controller_and_serial_output(self):
        """Forward mission requests and verify the controller's Pi frame."""
        controller = ReactiveControlLayer()
        probe = Node('reactive_control_test_probe')
        executor = SingleThreadedExecutor()
        mission_nodes = []
        selected_commands = []

        class RecordingSerial:
            def __init__(self):
                self.frames = []

            def write(self, frame):
                self.frames.append(bytes(frame))
                return len(frame)

            def close(self):
                pass

            @property
            def in_waiting(self):
                return 0

            def read(self, size):
                return b''

        serial = RecordingSerial()
        controller.serial_port = serial

        enable_publisher = probe.create_publisher(Bool, '/control/enable', 10)
        source_publisher = probe.create_publisher(
            String, '/control/select_source', 10
        )
        active_task_publisher = probe.create_publisher(
            String, '/mission/active_task', 10
        )
        bin_detection_publisher = probe.create_publisher(
            BinDetection, '/vision/bin_detection', 10
        )
        probe.create_subscription(
            Twist,
            '/control/selected_cmd_vel',
            selected_commands.append,
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
                    frame_forwarded = self._spin_until(
                        executor,
                        lambda: any(
                            frame[0:2] == b'\xaa\x55'
                            and struct.unpack('>hhhhhhB', frame[2:15])[0] == 100
                            and crc8(frame[2:15]) == frame[15]
                            for frame in serial.frames
                        ),
                    )
                    self.assertTrue(
                        frame_forwarded,
                        'forward request was not encoded into a valid '
                        'Raspberry Pi serial frame',
                    )

            dropper = mission_nodes[3]
            executor.add_node(dropper)
            self.assertTrue(self._spin_until(
                executor,
                lambda: (
                    bin_detection_publisher.get_subscription_count() > 0
                    and active_task_publisher.get_subscription_count() > 0
                ),
            ))
            active_task_publisher.publish(String(data='dropper'))
            self.assertTrue(self._spin_until(
                executor, lambda: dropper.active_task == 'dropper'
            ))
            source_publisher.publish(String(data='dropper'))
            self.assertTrue(self._spin_until(
                executor,
                lambda: controller.selected_source == 'dropper',
            ))

            detection = BinDetection()
            detection.header.stamp = probe.get_clock().now().to_msg()
            detection.detected = True
            detection.marker_detected = True
            detection.x = 0.4
            detection.y = 0.0
            detection.z = 0.9
            detection.confidence = 0.9
            selected_commands.clear()

            def publish_detection_and_check_alignment():
                detection.header.stamp = probe.get_clock().now().to_msg()
                bin_detection_publisher.publish(detection)
                return any(
                    abs(message.linear.y + 0.14) < 1e-6
                    for message in selected_commands
                )

            correction_forwarded = self._spin_until(
                executor,
                publish_detection_and_check_alignment,
            )
            self.assertTrue(
                correction_forwarded,
                'dropper alignment correction did not reach the controller',
            )
            correction_frame_forwarded = self._spin_until(
                executor,
                lambda: any(
                    frame[0:2] == b'\xaa\x55'
                    and struct.unpack('>hhhhhhB', frame[2:15])[1] == -140
                    and crc8(frame[2:15]) == frame[15]
                    for frame in serial.frames
                ),
            )
            self.assertTrue(
                correction_frame_forwarded,
                'dropper alignment correction was not encoded into a valid '
                'Raspberry Pi serial frame',
            )

            detection = BinDetection()
            detection.header.stamp = probe.get_clock().now().to_msg()
            bin_detection_publisher.publish(detection)
            selected_commands.clear()
            search_forwarded = self._spin_until(
                executor,
                lambda: any(
                    abs(message.angular.z - 0.25) < 1e-6
                    for message in selected_commands
                ),
            )
            self.assertTrue(
                search_forwarded,
                'dropper fallback search command did not reach the controller',
            )
            search_frame_forwarded = self._spin_until(
                executor,
                lambda: any(
                    frame[0:2] == b'\xaa\x55'
                    and struct.unpack('>hhhhhhB', frame[2:15])[5] == 250
                    and crc8(frame[2:15]) == frame[15]
                    for frame in serial.frames
                ),
            )
            self.assertTrue(
                search_frame_forwarded,
                'dropper search command was not encoded into a valid '
                'Raspberry Pi serial frame',
            )
        finally:
            executor.shutdown()
            for node in mission_nodes:
                node.tree.shutdown()
                node.destroy_node()
            controller.destroy_node()
            probe.destroy_node()

    def test_buoy_detection_command_reaches_controller(self):
        """Turn a stamped YOLO detection into a buoy command consumed by control."""
        controller = ReactiveControlLayer()
        mission = BuoyMissionNode()
        probe = Node('buoy_task_integration_probe')
        executor = SingleThreadedExecutor()
        selected_commands = []

        class RecordingSerial:
            def write(self, frame):
                return len(frame)

            def close(self):
                pass

            @property
            def in_waiting(self):
                return 0

            def read(self, size):
                return b''

        controller.serial_port = RecordingSerial()
        enable_publisher = probe.create_publisher(Bool, '/control/enable', 10)
        source_publisher = probe.create_publisher(
            String, '/control/select_source', 10
        )
        active_task_publisher = probe.create_publisher(
            String, '/mission/active_task', 10
        )
        target_color_publisher = probe.create_publisher(
            String, '/mission/target_buoy_color', 10
        )
        detection_publisher = probe.create_publisher(
            String, '/vision/detections', 10
        )
        probe.create_subscription(
            Twist,
            '/control/selected_cmd_vel',
            selected_commands.append,
            10,
        )

        try:
            executor.add_node(controller)
            executor.add_node(mission)
            executor.add_node(probe)
            self.assertTrue(
                self._spin_until(
                    executor,
                    lambda: (
                        enable_publisher.get_subscription_count() > 0
                        and active_task_publisher.get_subscription_count() > 0
                        and target_color_publisher.get_subscription_count() > 0
                        and detection_publisher.get_subscription_count() > 0
                        and mission.cmd_pub.get_subscription_count() > 0
                    ),
                )
            )
            enable_publisher.publish(Bool(data=True))
            source_publisher.publish(String(data='buoy'))
            target_color_publisher.publish(String(data='red'))
            active_task_publisher.publish(String(data='buoy'))
            self.assertTrue(
                self._spin_until(
                    executor,
                    lambda: controller.enabled
                    and controller.selected_source == 'buoy'
                    and mission.task_active,
                )
            )

            stamp = probe.get_clock().now().nanoseconds / 1_000_000_000
            detection = {
                'stamp': stamp,
                'frame_width': 640,
                'frame_height': 480,
                'detections': [
                    {
                        'class_name': 'red_buoy',
                        'confidence': 0.95,
                        'x1': 320,
                        'y1': 200,
                        'x2': 480,
                        'y2': 280,
                    },
                    {
                        'class_name': 'green_buoy',
                        'confidence': 0.99,
                        'x1': 100,
                        'y1': 180,
                        'x2': 260,
                        'y2': 300,
                    },
                ],
            }
            detection_publisher.publish(String(data=json.dumps(detection)))

            self.assertTrue(
                self._spin_until(
                    executor,
                    lambda: any(
                        message.angular.z < -0.1
                        and message.linear.x > 0.1
                        for message in selected_commands
                    ),
                ),
                'red buoy servo command did not reach the control layer',
            )
        finally:
            executor.shutdown()
            mission.tree.shutdown()
            controller.destroy_node()
            mission.destroy_node()
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
