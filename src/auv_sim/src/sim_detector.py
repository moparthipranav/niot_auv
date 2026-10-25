#!/usr/bin/env python3
"""HSV colour-threshold buoy detector for Gazebo (stand-in for untrained YOLO).

Publishes std_msgs/String (JSON) on /vision/detections in the same layout as the
real object_detector:
  {"stamp": <float sec>, "frame_width": W, "frame_height": H,
   "detections": [{"class_name": "red_buoy", "confidence": 0.95,
                   "x1":..,"y1":..,"x2":..,"y2":.., "class_id": 0, "track_id": 12}]}
One message per camera frame, containing every buoy seen (empty list if none).
"""
import json
import cv2, numpy as np, rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

RANGES = {   # OpenCV HSV (H 0-179). Red wraps around -> two ranges.
    "red":    [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "green":  [((40, 80, 70), (80, 255, 255))],
    "yellow": [((20, 100, 100), (35, 255, 255))],
}
# GUESS: match these ids to your real model's class ids if the mission node uses them.
CLASS_ID = {"red": 0, "green": 1, "yellow": 2}


class SimDetector(Node):
    def __init__(self):
        super().__init__("sim_detector")
        self.declare_parameter("target_color", "red")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("min_area_px", 30)
        self.declare_parameter("detect_all_colors", True)   # False = target colour only
        self.declare_parameter("confidence", 0.95)
        self.bridge = CvBridge()
        self.dbg = self.create_publisher(Image, "/vision/debug_image", 1)
        self.create_subscription(
            Image, self.get_parameter("image_topic").value, self.on_image, 1)
        self.pub = self.create_publisher(String, "/vision/detections", 1)

    def on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        frame = frame.copy()
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        min_area = self.get_parameter("min_area_px").value
        conf = self.get_parameter("confidence").value
        colors = (list(RANGES) if self.get_parameter("detect_all_colors").value
                  else [self.get_parameter("target_color").value])

        detections = []
        for color in colors:
            mask = np.zeros((h, w), np.uint8)
            for lo, hi in RANGES[color]:
                mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
            if not cnts:
                continue
            x, y, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
            detections.append({
                "class_name": f"{color}_buoy",
                "confidence": conf,
                "x1": int(x), "y1": int(y), "x2": int(x + bw), "y2": int(y + bh),
                "class_id": CLASS_ID[color],
                "track_id": CLASS_ID[color],   # one buoy per colour -> stable id
            })

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 255, 255), 2)
            cv2.putText(frame, f"{color}_buoy", (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Stamp = "now" from the node clock (not the image header), so freshness
        # checks in the mission node see a current time.
        stamp = self.get_clock().now().nanoseconds / 1e9
        self.pub.publish(String(data=json.dumps({
            "stamp": stamp,
            "frame_width": int(w),
            "frame_height": int(h),
            "detections": detections,
        })))

        cv2.line(frame, (w // 2, 0), (w // 2, h), (200, 200, 200), 1)   # image centre line
        self.dbg.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8", header=msg.header))


def main():
    rclpy.init(); rclpy.spin(SimDetector()); rclpy.shutdown()


if __name__ == "__main__":
    main()
