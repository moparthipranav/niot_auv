"""Single client-at-a-time YOLO inference server using JPEG and TCP."""

import argparse
import json
import os
import socket
import struct

import cv2
import numpy as np


HOST = '0.0.0.0'
PORT = 8765
MAX_MESSAGE_SIZE = 64 * 1024 * 1024


def _receive_exact(sock, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError('client closed the connection')
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


def _send_json(sock, payload):
    data = json.dumps(payload, separators=(',', ':')).encode('utf-8')
    sock.sendall(struct.pack('!I', len(data)) + data)


class InferenceServer:
    def __init__(self, model_path, mock=False):
        self.mock = mock
        self.model = None
        self.model_path = model_path
        if not mock:
            from ultralytics import YOLO

            print('Loading YOLO model: {}'.format(model_path), flush=True)
            self.model = YOLO(model_path)
            print('YOLO model loaded', flush=True)

    def detect(self, jpeg_bytes):
        image_array = np.frombuffer(jpeg_bytes, dtype='uint8')
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('request did not contain a valid JPEG image')
        if self.mock:
            return []

        # Keep tracking and inference options; debug output is rendered/published
        # by the ROS node, so the server does not open a GUI or write video files.
        results_list = self.model.track(
            image,
            persist=True,
            show=False,
            save=False,
            verbose=False,
        )
        detections = []
        if results_list:
            result = results_list[0]
            for box in result.boxes:
                class_id = int(box.cls.item())
                names = self.model.names
                class_name = names[class_id] if not isinstance(names, dict) else names[class_id]
                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                detections.append({
                    'class_id': class_id,
                    'class_name': str(class_name),
                    'confidence': float(box.conf.item()),
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2),
                    'track_id': int(box.id.item()) if box.id is not None else None,
                })
        return detections

    def serve_client(self, client):
        while True:
            try:
                header = client.recv(4)
                if not header:
                    return
                # TCP may return fewer than four bytes in its first read.
                if len(header) < 4:
                    header += _receive_exact(client, 4 - len(header))
                request_size = struct.unpack('!I', header)[0]
                if request_size == 0 or request_size > MAX_MESSAGE_SIZE:
                    _send_json(client, {'error': 'invalid image size'})
                    return
                jpeg_bytes = _receive_exact(client, request_size)
                try:
                    _send_json(client, self.detect(jpeg_bytes))
                except Exception as error:
                    _send_json(client, {'error': str(error)})
            except (ConnectionError, OSError):
                return

    def serve_forever(self, host=HOST, port=PORT):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((host, port))
            server.listen(1)
            print('Inference server listening on {}:{}'.format(host, port), flush=True)
            while True:
                client, address = server.accept()
                print('Client connected: {}'.format(address), flush=True)
                with client:
                    self.serve_client(client)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default=HOST)
    parser.add_argument('--port', type=int, default=PORT)
    parser.add_argument(
        '--model',
        default=os.environ.get('YOLO_MODEL', '/models/best.pt'),
        help='path to the existing YOLO model inside the container',
    )
    parser.add_argument(
        '--mock', action='store_true',
        help='accept and decode JPEG images, then return an empty detection list',
    )
    args = parser.parse_args()
    InferenceServer(args.model, mock=args.mock).serve_forever(args.host, args.port)


if __name__ == '__main__':
    main()
