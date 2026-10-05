# Keep the established Jetson CUDA/PyTorch/Ultralytics environment as the base.
FROM ultralytics/ultralytics:latest-jetson-jetpack4

COPY src/auv_vision/auv_vision/inference_server.py /opt/auv/inference_server.py

EXPOSE 8765
ENTRYPOINT ["python3", "/opt/auv/inference_server.py"]
