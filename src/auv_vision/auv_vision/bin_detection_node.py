"""ROS 2 node publishing compact classical-CV bin measurements."""

import math

from auv_interfaces.msg import BinDetection
from auv_vision.bin_detection import estimate_bin_detection

import cv2
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class BinDetectionNode(Node):
    """Publish a fresh bin/marker measurement for every received camera frame."""

    def __init__(self):
        super().__init__('bin_detection_node')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('bin_width', 0.0)
        self.declare_parameter('fx', 620.5)
        self.declare_parameter('pose_topic', '/vision/bin_detection')

        self.bin_width = float(self.get_parameter('bin_width').value)
        self.fx = float(self.get_parameter('fx').value)
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(
            BinDetection, str(self.get_parameter('pose_topic').value), 10
        )
        self.subscription = self.create_subscription(
            Image,
            str(self.get_parameter('image_topic').value),
            self.image_callback,
            10,
        )
        if (not math.isfinite(self.bin_width) or self.bin_width <= 0
                or not math.isfinite(self.fx) or self.fx <= 0):
            self.get_logger().warning(
                'Set valid bin_width (metres) and fx (pixels) parameters '
                'to enable range-based descent; the tree will not release '
                'payload with an invalid range.'
            )

    def image_callback(self, image_msg):
        """Publish a detection or an explicit not-detected result per frame."""
        result = BinDetection()
        result.header = image_msg.header
        if result.header.stamp.sec == 0 and result.header.stamp.nanosec == 0:
            result.header.stamp = self.get_clock().now().to_msg()

        try:
            image = self.bridge.imgmsg_to_cv2(
                image_msg, desired_encoding='bgr8'
            )
            measurement = estimate_bin_detection(
                image, self.bin_width, self.fx
            )
        except (CvBridgeError, cv2.error) as error:
            self.get_logger().error(f'Bin detection failed for camera frame: {error}')
            self.publisher.publish(result)
            return

        if measurement is not None:
            result.detected = measurement['detected']
            result.marker_detected = measurement['marker_detected']
            result.x = float(measurement['x'])
            result.y = float(measurement['y'])
            result.z = float(measurement['z'])
            result.yaw = float(measurement['yaw'])
            result.confidence = float(measurement['confidence'])

        self.publisher.publish(result)


def main(args=None):
    rclpy.init(args=args)
    node = BinDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
