"""Tests for the mission sequencer's complete task progression."""

import unittest

from auv_interfaces.msg import BinDetection
from auv_mission_planner.mission_sequencer_node import (
    MissionSequencerNode,
    MissionState,
)
import rclpy
from std_msgs.msg import String


class TestMissionSequencer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_full_mission_progresses_on_detection_and_task_completion(self):
        node = MissionSequencerNode()
        try:
            node._transition_to(MissionState.DIVE)
            node.current_depth = node.target_depth
            node._sequencer_tick()
            self.assertEqual(node.state, MissionState.TRANSIT_TO_BUOY)

            bin_detection = BinDetection()
            bin_detection.detected = True
            node._bin_seen_cb(bin_detection)
            self.assertTrue(node.bin_detected)

            node.buoy_detected = True
            node._sequencer_tick()
            self.assertEqual(node.state, MissionState.BUOY_INTERACTION)

            node._task_completed_cb(String(data='gate'))
            self.assertEqual(node.last_completed_task, '')
            node._task_completed_cb(String(data='buoy'))
            node._sequencer_tick()
            self.assertEqual(node.state, MissionState.TRANSIT_TO_GATE)

            node.gate_detected = True
            node._sequencer_tick()
            self.assertEqual(node.state, MissionState.GATE_PASS)

            node._task_completed_cb(String(data='gate'))
            node._sequencer_tick()
            self.assertEqual(node.state, MissionState.TRANSIT_TO_FORK)

            node.distance_traveled = node.state_start_distance + 3.0
            node._sequencer_tick()
            self.assertEqual(node.state, MissionState.BRANCH_TASK_1)

            node._task_completed_cb(String(data=node.first_task))
            node._sequencer_tick()
            self.assertEqual(node.state, MissionState.INTER_STATION_TRANSIT)

            node.distance_traveled = node.state_start_distance + node.inter_station_dist
            node._sequencer_tick()
            self.assertEqual(node.state, MissionState.BRANCH_TASK_2)

            second_task = 'dropper' if node.first_task == 'torpedo' else 'torpedo'
            node._task_completed_cb(String(data=second_task))
            node._sequencer_tick()
            self.assertEqual(node.state, MissionState.TRANSIT_TO_OCTAGON)

            node.octagon_detected = True
            node._sequencer_tick()
            self.assertEqual(node.state, MissionState.OCTAGON_SURFACE)

            node.current_depth = 0.0
            node._sequencer_tick()
            self.assertEqual(node.state, MissionState.COMPLETED)
        finally:
            node.destroy_node()


if __name__ == '__main__':
    unittest.main()
