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
