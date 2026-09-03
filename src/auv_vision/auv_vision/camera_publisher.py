import rclpy
import cv2
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

# Creating a ros2 published node for publishing camera images

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        
        # Declare the parameters for the node to use
        self.declare_parameter('video_source', 0)
        self.declare_parameter('publish_rate', 30.0) # publishes 30 frames per second

        source = self.get_parameter('video_source').value
        rate = self.get_parameter('publish_rate').value

        # Creating a publisher node on the topic '/camera/image_raw' with queue_size 10 which limits the amount of messages queued messages if the receiver is not accepting them
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)

        # Initialize the OpenCV capture device and CvBridge
        self.cap = cv2.VideoCapture(source)
        self.bridge = CvBridge()

        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open camera/video source: {source}")
            return
        
        timer_period = 1.0 / rate
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info(f"Camera Publisher initialized. Publishing at {rate} FPS on '/camera/image_raw'...")
    
    def timer_callback(self):   # Creates the message and publishes it.
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warn("Frame read failed or end of file reached. Reopening source...")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return 
        
        # Converting Opencv numpy arrays into ros2/Image format as ros follows a strict message interface. (Check if you need to add the interface details in the package.xml, setup.config or setup.py files)
        ros_image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8') # color or pixel depth conversion. bgr8 will give color image with blue-green-red color order
        ros_image_msg.header.stamp = self.get_clock().now().to_msg() # Building a header into the ros_image_msg. This indicates the timestamp of the shot taken and published. ROS header
        ros_image_msg.header.frame_id = 'camera_link'

        self.publisher_.publish(ros_image_msg)
    
def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

        




        

