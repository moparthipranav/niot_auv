# The Object Detector node subscribes to the camera_publisher node to perform yolov11 object detection algorthim on the images. 

import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from rclpy.node import Node
from ultralytics import YOLO
import numpy as np
import os
from typing import Union, cast
import time
from ultralytics.engine.results import Results

# Camera Parameters
FX = 620.5
FY = 618.7
CX = 320
CY = 240

REAL_BUOY_WIDTH = 0.20


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
        self.declare_parameter('model_path', 'model/best.pt')
        self.declare_parameter('real_buoy_width', 0.20) # Change this value later
        self.declare_parameter('fx', 620.5)
        self.declare_parameter('fy', 618.7)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 240.0)

        # Retrieve the parameter values
        model_path_param = self.get_parameter('model_path').value
        real_buoy_width = self.get_parameter('real_buoy_width').value
        fx = self.get_parameter('fx').value
        fy = self.get_parameter('fy').value
        cx = self.get_parameter('cx').value
        cy = self.get_parameter('cy').value

        # Initialize the Camera Coordinate Transform
        self.transform = CameraCoordinateTransformer(fx, fy, cx, cy)

        # Initialize the cv bridge
        self.bridge = CvBridge()

        # Resolving the model path
        if not os.path.isabs(model_path_param):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_path = os.path.abspath(os.path.join(script_dir, "..", model_path_param)) # Check this when deploying the code
        else:
            self.model_path = model_path_param

        # Load the YOLO model
        self.get_logger().info("Loading the YOLO model from: {self.model_path}")
        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
            self.get_logger().info("YOLO model loaded successfully!")
        else:
            self.get_logger().warn(f"Model path '{self.model_path}' not found! Loading default... 'yolov8n'")
            self.model = YOLO('yolov8n.pt')

        # Setting up the subscribers and the publishers.
        # Subscribe to incoming raw frames from camera_publisher node
        self.subscription_ = self.create_subscription(
            Image, 
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # FPS tracking variable
        self.previous_frame_time = time.perf_counter()
        self.smoothed_fps = 0.0

        self.get_logger().info("Object Detection Node active and listening on '/camera/image_raw'. ")

    def image_callback(self, msg:Image):
        """Callback triggered on every frame received from the camera node."""
        try:
            # Convert the raw ros2 msg Image into OpenCv cv::Mat
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        # Perform YOLO inference 
        results_iter = self.model.predict(cv_image, stream=False, verbose=False)
        results_list = list(results_iter)

        target_detected = False
        target_xyz = [0.0, 0.0, 0.0]

        if results_list:
            first_result = cast(Results, results_list[0])
            annotated_frame = first_result.plot()

            # Process detections
            for box in first_result.boxes:#type:ignore
                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                confidence = float(box.conf.item())

                # Computing the center pixel coordinates and width
                u = (x1 + x2) / 2.0
                v = (y1 + y2) / 2.0
                width_pixels = x2 - x1

                depth = self.transform.estimate_depth(self.real_buoy_width, width_pixels)
                X, Y, Z = self.transform.image_to_camera(u, v, depth)

                target_detected = True
                target_xyz = [float(X), float(Y), float(Z)]

                # Print telemetry to node logs
                self.get_logger().debug(
                    f"Class: {self.model.names[int(box.cls)]} | Conf: {confidence:.2f} | "
                    f"Pixel: ({u:.1f}, {v:.1f}) | Camera XYZ: ({X:.3f}, {Y:.3f}, {Z:.3f}) m"
                )

                # Draw 3D coordinate text overlay on debug image
                cv2.putText(
                    annotated_frame,
                    f"XYZ: ({X:.2f}, {Y:.2f}, {Z:.2f})m",
                    (int(x1), max(20, int(y1) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )
        else:
            annotated_frame = cv_image.copy()

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

