# Two container vision deployment

The ROS container keeps `camera_publisher`, `object_detector`, depth/XYZ math,
debug overlays, and mission topics. The YOLO container runs the unchanged
Ultralytics model and tracking call. The containers exchange one JPEG frame and
a JSON detection list per request over TCP port **8765**. Both should use host
networking on the Jetson, so `127.0.0.1` is the default server address.

## Build and start the Jetson YOLO container

From the repository root, with Docker configured for the Jetson Nano:

```sh
docker build -f docker/yolo/Dockerfile -t niot-auv-yolo:jetson .
docker run --rm -it --network host \
  -v "$PWD/models:/models:ro" \
  niot-auv-yolo:jetson --model /models/best.pt
```

Build on the Nano itself for its native architecture. When building from an
amd64 PC for the Nano, use Docker Buildx with `--platform linux/arm64` and
ensure the Jetson base image is available for that platform.

Change the host `models` path and model filename to the existing model's
location. This image adds only `inference_server.py` to the supplied Jetson
Ultralytics image; it does not install ROS or replace Python, PyTorch,
Ultralytics, CUDA, or TensorRT.

## Build and start the ROS container

Build the ROS 2 Jazzy workspace in the ROS image as you currently do, including
`auv_vision` and `auv_mission_planner`. Start the ROS container with host
networking and the camera device/source it uses, then source the workspace:

```sh
docker run --rm -it --network host --device=/dev/video0 \
  -v /path/to/ros-workspace:/workspace \
  YOUR_ROS_JAZZY_IMAGE bash
```

Inside that container, in terminals with the ROS environment and workspace
sourced, start the nodes:

```sh
ros2 run auv_vision camera_publisher
ros2 run auv_vision object_detector --ros-args \
  -p inference_host:=127.0.0.1 -p inference_port:=8765
```

Start the appropriate existing `auv_mission_planner` node in another sourced
terminal. The detector publishes `/vision/detections` as a `std_msgs/String`
containing the JSON detection list, `/vision/debug_image`, and the existing
`/vision/...` found and pose topics used by the mission planner.
With Docker host networking, both containers share the host network namespace;
the ROS detector connects to the YOLO server at `127.0.0.1:8765`. The listener
binds to `0.0.0.0:8765`.

## Test the TCP link on a PC

The mock server requires Python OpenCV and NumPy, but imports no Ultralytics or
CUDA libraries. In a PC environment that has `cv2` and `numpy`:

```sh
python3 src/auv_vision/auv_vision/inference_server.py --mock
```

In another terminal, submit any readable image:

```sh
python3 src/auv_vision/auv_vision/inference_client.py sample.jpg
```

The mock server decodes the JPEG and returns `[]`; this checks framing and TCP
connectivity only. Inference with the Jetson GPU and the deployed model must be
checked on the Jetson.

The wire format is a 4-byte unsigned network-order payload length followed by
JPEG bytes. The response uses the same 4-byte length followed by UTF-8 JSON:
an array of detections, each with `class_id`, `class_name`, `confidence`,
`x1`, `y1`, `x2`, and `y2` (plus the optional tracking ID).

## Test bin alignment and payload release

The classical OpenCV detector is a separate ROS 2 node. It processes every
`/camera/image_raw` frame and publishes one `auv_interfaces/msg/BinDetection`
message to `/vision/bin_detection`, including explicit `detected` and
`marker_detected` flags when either target is absent. `x` and `y` are normalized
image offsets of the marker center (range -1 to 1), `z` is the estimated
camera-to-bin range in metres, `yaw` is the bin's image-plane orientation in
radians, and `confidence` is the marker classifier's vote confidence. The
message header carries the camera measurement timestamp.

Configure `bin_width` to the measured physical bin width and `fx` to the
camera's calibrated horizontal focal length in pixels for the actual image
size. The width-based range is approximate and assumes the bin's measured
width is visible without substantial perspective foreshortening. Range is
zero when calibration is missing or invalid; the mission tree can still
observe the image offsets, but will not descend or release on an invalid range.
The tree requires both the bin and marker to be detected, a
confidence of at least 0.2, and a measurement no older than 0.75 seconds before
issuing alignment, descent, or release commands.
After reaching the configured drop range, it also waits for a recent
`std_msgs/msg/Bool` on `/state/vehicle_stable`; false or missing/stale
stability data keeps the vehicle stopped and prevents payload release. The
stability publisher is a separate vehicle-state input and must be running for
the mission to proceed to release.

Build and source the ROS workspace on the Jetson, then run these commands in
separate sourced terminals. Replace the example bin width with its measured
value and use the correct camera index:

```sh
cd /path/to/niot_auv
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-up-to auv_vision auv_mission_planner
source install/setup.bash
```

In separate terminals with the ROS workspace sourced:

```sh
ros2 run auv_vision camera_publisher --ros-args \
  -p video_source:=/videos/test.mp4
ros2 run auv_vision bin_detection --ros-args \
  -p bin_width:=0.30 -p fx:=620.5
ros2 run auv_mission_planner dropper_mission_node
```

First observe detections and requested motion without enabling the actuators:

```sh
ros2 topic echo /vision/bin_detection
ros2 topic echo /control/requests/dropper
ros2 run rqt_image_view rqt_image_view
```

The dropper tree searches when the bin or marker is not detected or the last
measurement is stale, centers over the marker, descends to a 0.40 m measured
range, then pulses `/actuators/gripper_open`. Release is one-shot. The tree
publishes movement requests on `/control/requests/dropper` and the release
command on `/actuators/gripper_open`; the reactive control layer must remain
disabled until the commands have been observed and the vehicle is secured.

If the reactive control layer is on the Raspberry Pi, run it there on the same
ROS 2 domain and LAN, with the M64 modem parameters configured for the Pi's
serial port. For example, in a sourced terminal on the Pi:

```sh
ros2 run auv_mission_planner reactive_control_layer --ros-args \
  -p enabled:=false -p use_m64_modem:=true \
  -p m64_port:=/dev/ttyUSB3
```

Use the Pi's actual modem port, do not start a second control layer on the
Jetson, and confirm that the Pi receives `/control/requests/dropper` and
`/actuators/gripper_open` before enabling it.

For a physical test, secure the AUV and clear the actuator mechanism first.
Start the reactive control layer disabled on the actuator computer. Stop the
observer-only dropper node before enabling the controller; then publish enable
from any ROS 2 terminal on the shared domain and launch one fresh dropper node:

```sh
ros2 topic pub --once /control/enable std_msgs/msg/Bool "{data: true}"
ros2 run auv_mission_planner dropper_mission_node
```
