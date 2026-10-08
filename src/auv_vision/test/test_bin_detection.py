"""Unit tests for OpenCV bin detection and range estimation."""

import unittest

from auv_vision.bin_detection import (
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


if __name__ == '__main__':
    unittest.main()
