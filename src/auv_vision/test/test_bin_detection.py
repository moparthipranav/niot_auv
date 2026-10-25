"""Unit tests for OpenCV bin detection and range estimation."""

import unittest

from auv_vision.bin_detection import (
    annotate_bin_detection,
    estimate_bin_detection,
    estimate_bin_pose,
)

import cv2

import numpy as np


class TestBinDetection(unittest.TestCase):
    """Tests for bin pose extraction from image contours."""

    def test_estimates_offsets_and_range_from_bin_contour(self):
        """Return expected normalized offsets and width-based range."""
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[:, :] = (255, 0, 0)
        cv2.rectangle(image, (130, 80), (250, 200), (255, 255, 255), -1)

        result = estimate_bin_pose(image, real_bin_width=0.30, fx=600.0)

        self.assertIsNotNone(result)
        norm_x, norm_y, range_m, _ = result
        self.assertAlmostEqual(norm_x, 0.1875, places=2)
        self.assertAlmostEqual(norm_y, 0.1667, places=2)
        self.assertAlmostEqual(range_m, 1.5, places=1)

    def test_rejects_unconfigured_bin_width(self):
        """Reject range estimation without a physical bin width."""
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        self.assertIsNone(
            estimate_bin_pose(image, real_bin_width=0.0, fx=600.0)
        )

    def test_reports_marker_pose_and_classifier_confidence(self):
        """Report marker offsets and calibrated range as compact measurements."""
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[:, :] = (255, 0, 0)
        cv2.rectangle(image, (20, 20), (300, 220), (255, 255, 255), -1)
        cv2.line(image, (100, 80), (220, 160), (0, 0, 255), 12)
        cv2.line(image, (100, 160), (220, 80), (0, 0, 255), 12)

        result = estimate_bin_detection(
            image, real_bin_width=0.30, fx=600.0
        )

        self.assertIsNotNone(result)
        self.assertTrue(result['detected'])
        self.assertTrue(result['marker_detected'])
        self.assertAlmostEqual(result['x'], 0.0, places=1)
        self.assertAlmostEqual(result['y'], 0.0, places=1)
        self.assertGreater(result['z'], 0.0)
        self.assertGreaterEqual(result['confidence'], 0.2)
        self.assertLessEqual(result['confidence'], 1.0)
        self.assertIn(result['marker_label'], ('X', 'O'))

        annotated = annotate_bin_detection(image, result)
        self.assertEqual(annotated.shape, image.shape)
        self.assertFalse(np.array_equal(annotated, image))

    def test_reports_bin_without_marker(self):
        """Keep bin visibility independent from marker visibility."""
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[:, :] = (255, 0, 0)
        cv2.rectangle(image, (20, 20), (300, 220), (255, 255, 255), -1)

        result = estimate_bin_detection(
            image, real_bin_width=0.30, fx=600.0
        )

        self.assertIsNotNone(result)
        self.assertTrue(result['detected'])
        self.assertFalse(result['marker_detected'])
        self.assertEqual(result['confidence'], 0.0)
        self.assertEqual(result['marker_label'], 'none')

    def test_detects_far_bin_under_partial_shadow(self):
        image = np.full((480, 640, 3), (70, 45, 30), dtype=np.uint8)
        cv2.rectangle(image, (286, 200), (350, 264), (235, 235, 235), -1)
        cv2.rectangle(image, (286, 232), (350, 264), (105, 105, 105), -1)
        cv2.line(image, (296, 210), (340, 254), (0, 0, 255), 5)
        cv2.line(image, (296, 254), (340, 210), (0, 0, 255), 5)

        result = estimate_bin_detection(
            image, real_bin_width=0.42, fx=468.0
        )

        self.assertIsNotNone(result)
        self.assertTrue(result['detected'])
        self.assertTrue(result['marker_detected'])
        self.assertEqual(result['marker_label'], 'X')
        self.assertGreater(result['confidence'], 0.2)
        self.assertGreater(result['z'], 2.0)

    def test_classifies_o_marker_for_target_selection(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[:, :] = (255, 0, 0)
        cv2.rectangle(image, (20, 20), (300, 220), (255, 255, 255), -1)
        cv2.circle(image, (160, 120), 48, (0, 0, 255), 12)

        result = estimate_bin_detection(
            image, real_bin_width=0.30, fx=600.0
        )

        self.assertIsNotNone(result)
        self.assertTrue(result['marker_detected'])
        self.assertEqual(result['marker_label'], 'O')

    def test_selects_requested_marker_from_adjacent_bins(self):
        image = np.full((480, 640, 3), (70, 45, 30), dtype=np.uint8)
        cv2.rectangle(image, (70, 150), (220, 300), (245, 245, 245), -1)
        cv2.rectangle(image, (240, 150), (390, 300), (245, 245, 245), -1)
        cv2.circle(image, (145, 225), 36, (0, 0, 255), 8)
        cv2.line(image, (275, 190), (345, 260), (0, 0, 255), 8)
        cv2.line(image, (275, 260), (345, 190), (0, 0, 255), 8)

        result = estimate_bin_detection(
            image, real_bin_width=0.42, fx=468.0, target_marker='X'
        )

        self.assertIsNotNone(result)
        self.assertEqual(len(result['bin_candidates']), 2)
        self.assertEqual(result['marker_label'], 'X')
        self.assertTrue(result['marker_detected'])
        self.assertGreater(result['x'], -0.1)
        self.assertLess(result['x'], 0.1)

    def test_selects_requested_o_bin_when_adjacent_x_is_larger(self):
        image = np.full((480, 640, 3), (70, 45, 30), dtype=np.uint8)
        cv2.rectangle(image, (30, 140), (220, 330), (245, 245, 245), -1)
        cv2.rectangle(image, (240, 140), (500, 400), (245, 245, 245), -1)
        cv2.circle(image, (125, 235), 44, (0, 0, 255), 10)
        cv2.line(image, (300, 200), (440, 340), (0, 0, 255), 12)
        cv2.line(image, (300, 340), (440, 200), (0, 0, 255), 12)

        result = estimate_bin_detection(
            image, real_bin_width=0.42, fx=468.0, target_marker='O'
        )

        self.assertIsNotNone(result)
        self.assertEqual(len(result['bin_candidates']), 2)
        self.assertEqual(result['marker_label'], 'O')
        self.assertLess(result['x'], -0.4)

    def test_visualizes_frames_without_a_bin(self):
        """Make missing detections visible without mutating the camera frame."""
        image = np.zeros((240, 320, 3), dtype=np.uint8)

        annotated = annotate_bin_detection(image, None)

        self.assertFalse(np.array_equal(annotated, image))
        self.assertEqual(image.sum(), 0)


if __name__ == '__main__':
    unittest.main()
