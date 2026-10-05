"""Small JPEG-over-TCP client for the remote YOLO inference server."""

import json
import socket
import struct

import cv2


MAX_MESSAGE_SIZE = 64 * 1024 * 1024


class InferenceError(Exception):
    """Raised when the inference server returns an error or bad response."""


def _receive_exact(sock, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise InferenceError('inference server closed the connection')
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


class InferenceClient:
    def __init__(self, host='127.0.0.1', port=8765, timeout=5.0, jpeg_quality=85):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.jpeg_quality = jpeg_quality

    def detect(self, image):
        ok, encoded = cv2.imencode(
            '.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            raise InferenceError('OpenCV could not encode the camera frame as JPEG')
        image_bytes = encoded.tobytes()
        request = struct.pack('!I', len(image_bytes)) + image_bytes

        with socket.create_connection(
            (self.host, self.port), timeout=self.timeout
        ) as sock:
            sock.settimeout(self.timeout)
            sock.sendall(request)
            response_size = struct.unpack('!I', _receive_exact(sock, 4))[0]
            if response_size > MAX_MESSAGE_SIZE:
                raise InferenceError('inference response exceeds size limit')
            response = json.loads(_receive_exact(sock, response_size).decode('utf-8'))

        if isinstance(response, dict) and 'error' in response:
            raise InferenceError(response['error'])
        if not isinstance(response, list):
            raise InferenceError('inference server returned an invalid response')
        return response


def main():
    """Send one local image to the server as a quick connection check."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', help='image file to send')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    image = cv2.imread(args.image)
    if image is None:
        parser.error('could not read image: {}'.format(args.image))
    detections = InferenceClient(args.host, args.port).detect(image)
    print(json.dumps(detections, indent=2))


if __name__ == '__main__':
    main()
