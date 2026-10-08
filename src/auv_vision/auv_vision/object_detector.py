# The Object Detector node subscribes to the camera_publisher node to perform yolov11 object detection algorthim on the images. 

import json
import math
import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from rclpy.node import Node
import numpy as np
import time
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool, Float32MultiArray, String
from auv_vision.inference_client import InferenceClient, InferenceError

# Camera Parameters
FX = 620.5
FY = 618.7
CX = 320
CY = 240

class CameraCoordinateTransformer:

    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
    
    def estimate_depth(self, real_width, pixel_width):

        "Estimate depth using pinhole camera model"

        return (self.fx * real_width) / pixel_width

    def image_to_camera(self, u, v, depth):
        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        Z = depth

        return np.array([X, Y, Z])

transformer = CameraCoordinateTransformer(FX, FY, CX, CY)

# Creating the ObjectDetection Subscriber which subscribes to the camera_publisher
class ObjectDetectionSubscriber(Node):

    def __init__(self):
        super().__init__('object_detection_subscriber')

        # Declare the ROS parameters 
        self.declare_parameter('inference_host', '127.0.0.1')
        self.declare_parameter('inference_port', 8765)
        self.declare_parameter('inference_timeout', 5.0)
        self.declare_parameter('jpeg_quality', 85)
        self.declare_parameter('real_buoy_width', 0.20) # Change this value later
        self.declare_parameter('real_bin_width', 0.0)
        self.declare_parameter('fx', 620.5)
        self.declare_parameter('fy', 618.7)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 240.0)

        # Retrieve the parameter values
        real_buoy_width = self.get_parameter('real_buoy_width').value
        self.real_buoy_width = float(real_buoy_width)
        self.real_bin_width = float(self.get_parameter('real_bin_width').value)
        fx = self.get_parameter('fx').value
        fy = self.get_parameter('fy').value
        cx = self.get_parameter('cx').value
        cy = self.get_parameter('cy').value

        # Initialize the Camera Coordinate Transform
        self.transform = CameraCoordinateTransformer(fx, fy, cx, cy)
        if not math.isfinite(self.real_bin_width) or self.real_bin_width <= 0:
            self.get_logger().warning(
                'Set real_bin_width to publish a calibrated YOLO bin range; '
                'classical bin measurements come from bin_detection.'
            )

        # Initialize the cv bridge
        self.bridge = CvBridge()

        self.inference_client = InferenceClient(
            str(self.get_parameter('inference_host').value),
            int(self.get_parameter('inference_port').value),
            float(self.get_parameter('inference_timeout').value),
            int(self.get_parameter('jpeg_quality').value),
        )

        # Setting up the subscribers and the publishers.
        # Subscribe to incoming raw frames from camera_publisher node
        self.subscription_ = self.create_subscription(
            Image, 
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.detections_publisher = self.create_publisher(
            String, '/vision/detections', 10
        )
        self.debug_publisher = self.create_publisher(
            Image, '/vision/debug_image', 10
        )

        self.mission_class_names = {
            'marker': self._class_aliases(
                self._declare_class_parameter('marker_class', 'path'),
                ('marker', 'track'),
            ),
            'gate': self._class_aliases(
                self._declare_class_parameter('gate_class', 'gate'),
                ('l_gate',),
            ),
            'bin': self._class_aliases(
                self._declare_class_parameter('bin_class', 'bin'),
                ('dropper_bin',),
            ),
            'target': self._class_aliases(
                self._declare_class_parameter('target_class', 'buoy'),
                ('target_buoy',),
            ),
            'torpedo': self._class_aliases(
                self._declare_class_parameter('torpedo_class', 'torpedo'),
                ('torpedo_target', 'cutout'),
            ),
        }
        self.found_publishers = {
            mission_object: self.create_publisher(
                Bool, f'/vision/{topic_name}_found', 10
            )
            for mission_object, topic_name in (
                ('marker', 'marker'),
                ('gate', 'gate'),
                ('bin', 'bin'),
                ('target', 'target'),
                ('torpedo', 'torpedo'),
            )
        }
        self.pose_publishers = {
            'marker': self.create_publisher(Vector3, '/vision/marker_pose', 10),
            'gate': self.create_publisher(
                Float32MultiArray, '/vision/gate_pose', 10
            ),
            'bin': self.create_publisher(
                Float32MultiArray, '/vision/bin_pose', 10
            ),
            'target': self.create_publisher(
                Float32MultiArray, '/vision/target_pose', 10
            ),
            'torpedo': self.create_publisher(
                Float32MultiArray, '/vision/torpedo_pose', 10
            ),
        }

        # FPS tracking variable
        self.previous_frame_time = time.perf_counter()
        self.smoothed_fps = 0.0

        self.get_logger().info("Object Detection Node active and listening on '/camera/image_raw'. ")

    def _declare_class_parameter(self, name, default):
        self.declare_parameter(name, default)
        return str(self.get_parameter(name).value).strip().lower()

    @staticmethod
    def _class_aliases(class_name, aliases):
        return {class_name, *aliases}

    def image_callback(self, msg:Image):
        """Callback triggered on every frame received from the camera node."""
        try:
            # Convert the raw ros2 msg Image into OpenCv cv::Mat
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        try:
            remote_detections = self.inference_client.detect(cv_image)
        except (InferenceError, OSError, ValueError) as e:
            self.get_logger().error(f"YOLO inference server request failed: {e}")
            remote_detections = []
        self.detections_publisher.publish(
            String(data=json.dumps(remote_detections))
        )
        detections = {}
        annotated_frame = cv_image.copy()
        frame_height, frame_width = cv_image.shape[:2]

        # Keep depth, camera coordinates, mission selection and ROS publishing local.
        for box in remote_detections:
            x1, y1, x2, y2 = (
                float(box[key]) for key in ('x1', 'y1', 'x2', 'y2')
            )
            confidence = float(box['confidence'])
            class_id = int(box['class_id'])
            class_name = str(box['class_name']).strip().lower()
            track_id = box.get('track_id')

            # Computing the center pixel coordinates and width
            u = (x1 + x2) / 2.0
            v = (y1 + y2) / 2.0
            width_pixels = x2 - x1
            if width_pixels <= 0:
                continue
            mission_object = next(
                (
                    name for name, aliases in self.mission_class_names.items()
                    if class_name in aliases
                ),
                None,
            )
            real_width = (
                self.real_bin_width
                if mission_object == 'bin'
                else self.real_buoy_width
            )
            if not math.isfinite(real_width) or real_width <= 0:
                continue
            depth = self.transform.estimate_depth(real_width, width_pixels)
            if not math.isfinite(depth) or depth <= 0:
                continue
            X, Y, Z = self.transform.image_to_camera(u, v, depth)
            norm_x = (u - frame_width / 2.0) / (frame_width / 2.0)
            norm_y = (v - frame_height / 2.0) / (frame_height / 2.0)

            if mission_object is not None and (
                mission_object not in detections
                or confidence > detections[mission_object][0]
            ):
                detections[mission_object] = (
                    confidence,
                    norm_x,
                    norm_y,
                    float(Z),
                )

            self.get_logger().debug(
                f"Class: {class_name} ({class_id}) | Track: {track_id} | "
                f"Conf: {confidence:.2f} | "
                f"Pixel: ({u:.1f}, {v:.1f}) | Camera XYZ: ({X:.3f}, {Y:.3f}, {Z:.3f}) m"
            )
            cv2.rectangle(
                annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                (0, 255, 0), 2,
            )
            cv2.putText(
                annotated_frame,
                f"{class_name} {confidence:.2f} XYZ: ({X:.2f}, {Y:.2f}, {Z:.2f})m",
                (int(x1), max(20, int(y1) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        for mission_object, publisher in self.found_publishers.items():
            detection = detections.get(mission_object)
            publisher.publish(Bool(data=detection is not None))
            if detection is None:
                continue

            _, norm_x, norm_y, depth = detection
            if mission_object == 'marker':
                self.pose_publishers[mission_object].publish(
                    Vector3(x=float(norm_x), y=float(norm_y), z=0.0)
                )
            else:
                self.pose_publishers[mission_object].publish(
                    Float32MultiArray(data=[float(norm_x), float(norm_y), depth])
                )

        # Calculate the current FPS metric
        current_time = time.perf_counter()
        frame_time = current_time - self.previous_frame_time
        self.previous_frame_time = current_time
        instantaneous_fps = 1.0 / frame_time if frame_time > 0 else 0.0
        self.smoothed_fps = instantaneous_fps if self.smoothed_fps == 0 else 0.9 * self.smoothed_fps + 0.1 * instantaneous_fps

        # Overlay FPS on debug frame
        cv2.putText(
            annotated_frame,
            f"FPS: {self.smoothed_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        debug_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        debug_msg.header = msg.header
        self.debug_publisher.publish(debug_msg)

        # Write methods to send data to the Mission planner (Action server and client model)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
