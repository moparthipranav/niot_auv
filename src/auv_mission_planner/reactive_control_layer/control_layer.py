"""Single-owner reactive control layer for the NIOT AUV.

Mission nodes submit Twist requests; this node alone selects and converts one
request into an M64 acoustic command.  The M64 protocol accepts one discrete,
latched command, so simultaneous Twist axes cannot be sent together.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, String, UInt8


CMD_FORWARD = ord('W')
CMD_BACKWARD = ord('S')
CMD_LEFT = ord('A')
CMD_RIGHT = ord('D')
CMD_DOWN = ord('G')
CMD_GRIPPER_RELEASE = ord('F')
CMD_UP = ord('T')
CMD_SWAY = ord('L')
CMD_SWAY_REVERSE = ord('K')
CMD_STOP = ord('Q')

MAGIC_COMMAND = 0xA5
SIG1 = 0x4D
SIG2 = 0x64


@dataclass
class Request:
    command: Twist
    received_at_ns: int


def crc8(data: bytes) -> int:
    """CRC-8/0x07, matching AUV_M64_NEW.PY."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def build_m64_frame(command: int, sequence: int) -> bytes:
    """Create the exact eight-byte command frame consumed by the AUV M64."""
    body = bytes([MAGIC_COMMAND, command, sequence >> 8, sequence & 0xFF, 0])
    return body + bytes([crc8(body), SIG1, SIG2])


def command_from_twist(command: Twist, deadband: float = 0.01) -> int:
    """Map the strongest Twist axis to the M64's single-command interface."""
    candidates = (
        (abs(command.linear.x), CMD_FORWARD if command.linear.x >= 0 else CMD_BACKWARD),
        (abs(command.linear.y), CMD_SWAY if command.linear.y >= 0 else CMD_SWAY_REVERSE),
        (abs(command.linear.z), CMD_UP if command.linear.z >= 0 else CMD_DOWN),
        (abs(command.angular.z), CMD_LEFT if command.angular.z >= 0 else CMD_RIGHT),
    )
    magnitude, m64_command = max(candidates, key=lambda candidate: candidate[0])
    return m64_command if magnitude >= deadband else CMD_STOP


class ReactiveControlLayer(Node):
    """Arbitrate mission commands and safely drive the M64 control interface."""

    SOURCES = ('manual', 'path', 'gate', 'buoy', 'dropper', 'torpedo')
    DEFAULT_PRIORITIES = {
        'manual': 100,
        'torpedo': 50,
        'dropper': 40,
        'gate': 30,
        'buoy': 20,
        'path': 10,
    }

    def __init__(self):
        super().__init__('reactive_control_layer')
        self.declare_parameter('enabled', False)
        self.declare_parameter('input_timeout_sec', 0.5)
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('deadband', 0.01)
        self.declare_parameter('use_m64_modem', False)
        self.declare_parameter('m64_port', '/dev/ttyUSB3')
        self.declare_parameter('m64_baud', 115200)
        self.declare_parameter('m64_channel', 4)

        self.enabled = bool(self.get_parameter('enabled').value)
        self.selected_source = ''
        self.requests: Dict[str, Request] = {}
        self.last_m64_command: Optional[int] = None
        self.sequence = 0
        self.modem = None
        self._gripper_was_requested = False

        self.command_pub = self.create_publisher(UInt8, '/m64/command', 10)
        self.selected_twist_pub = self.create_publisher(Twist, '/control/selected_cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/control/status', 10)

        self.create_subscription(Bool, '/control/enable', self._enable_callback, 10)
        self.create_subscription(String, '/control/select_source', self._select_source_callback, 10)
        self.create_subscription(Bool, '/actuators/gripper_open', self._gripper_callback, 10)
        for source in self.SOURCES:
            topic = '/control/request' if source == 'manual' else f'/control/requests/{source}'
            self.create_subscription(Twist, topic, self._request_callback(source), 10)

        if self.get_parameter('use_m64_modem').value:
            self._connect_m64()

        rate = float(self.get_parameter('control_rate_hz').value)
        self.timer = self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info('RCL ready: disabled until /control/enable is true.')

    def _request_callback(self, source):
        def callback(command):
            self.requests[source] = Request(command, self.get_clock().now().nanoseconds)
        return callback

    def _enable_callback(self, message: Bool):
        self.enabled = message.data
        if not self.enabled:
            self._send_m64(CMD_STOP)

    def _select_source_callback(self, message: String):
        source = message.data.strip().lower()
        if source and source not in self.SOURCES:
            self.get_logger().warning(f'Ignoring unknown source: {source}')
            return
        self.selected_source = source

    def _gripper_callback(self, message: Bool):
        # The M64 receiver latches gripper state.  Send only the rising edge;
        # a False message is a release of the ROS request, not a close command.
        if self.enabled and message.data and not self._gripper_was_requested:
            self._send_m64(CMD_GRIPPER_RELEASE)
        self._gripper_was_requested = message.data

    def _active_request(self):
        now_ns = self.get_clock().now().nanoseconds
        timeout_ns = int(float(self.get_parameter('input_timeout_sec').value) * 1e9)
        fresh = {
            source: request for source, request in self.requests.items()
            if now_ns - request.received_at_ns <= timeout_ns
        }
        if self.selected_source:
            return (self.selected_source, fresh[self.selected_source]) if self.selected_source in fresh else (None, None)
        if not fresh:
            return None, None
        source = max(
            fresh,
            key=lambda name: (self.DEFAULT_PRIORITIES[name], fresh[name].received_at_ns),
        )
        return source, fresh[source]

    def _tick(self):
        source, request = self._active_request()
        command = request.command if self.enabled and request else Twist()
        m64_command = command_from_twist(command, float(self.get_parameter('deadband').value))
        self.selected_twist_pub.publish(command)
        self._send_m64(m64_command)
        state = 'enabled' if self.enabled else 'disabled'
        self.status_pub.publish(String(data=f'{state}; source={source or "none"}; m64={chr(m64_command)}'))

    def _connect_m64(self):
        try:
            from wlmodem import WlModem
            self.modem = WlModem(
                self.get_parameter('m64_port').value,
                baudrate=int(self.get_parameter('m64_baud').value),
            )
            if not self.modem.connect() or not self.modem.cmd_configure(
                'a', int(self.get_parameter('m64_channel').value)
            ):
                raise RuntimeError('could not configure M64 modem')
            self.get_logger().info('M64 modem connected as topside role A.')
        except Exception as exc:
            self.modem = None
            self.get_logger().error(f'M64 modem unavailable; publishing only: {exc}')

    def _send_m64(self, command: int):
        if command == self.last_m64_command:
            return
        self.last_m64_command = command
        self.command_pub.publish(UInt8(data=command))
        if self.modem is not None:
            try:
                self.sequence = (self.sequence + 1) & 0xFFFF
                if not self.modem.cmd_queue_packet(build_m64_frame(command, self.sequence)):
                    self.get_logger().error('M64 rejected command frame')
            except Exception as exc:
                self.get_logger().error(f'M64 transmission failed: {exc}')

    def destroy_node(self):
        self._send_m64(CMD_STOP)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ReactiveControlLayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
