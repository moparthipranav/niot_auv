"""Single-owner reactive control layer for the NIOT SAVe AUV (6 DOF).

Mission nodes submit Twist requests on per-task topics.  This node
arbitrates them by priority and freshness, then merges the winning
mission Twist with the always-on stabilizer's roll/pitch corrections.

The merged Twist is:
  1. Published on /control/selected_cmd_vel (for logging / debugging).
  2. Sent as a compact serial frame to the Raspberry Pi over UART.
     The Pi reads this frame and drives the ESCs accordingly.

Stabilizer Merging:
  The stabilizer publishes corrective roll (angular.x) and pitch (angular.y)
  on /control/requests/stabilizer.  These are ALWAYS overlaid onto the
  winning mission Twist, so the AUV stays level regardless of which task
  is active.  Mission behaviors should NOT set angular.x or angular.y.

Serial Frame Protocol (Jetson → Pi):
  [HEADER_1][HEADER_2][surge_hi][surge_lo][sway_hi][sway_lo]
  [heave_hi][heave_lo][roll_hi][roll_lo][pitch_hi][pitch_lo]
  [yaw_hi][yaw_lo][gripper][CRC8]

  - HEADER: 0xAA 0x55 (sync bytes)
  - surge/sway/heave/roll/pitch/yaw: signed int16, velocity × 1000 (milli-units)
  - gripper: 0x01 = open/release, 0x00 = closed/idle
  - CRC8: CRC-8/0x07 over bytes[2..14] (the 13 data bytes)

  Total: 16 bytes per frame at 20 Hz = 320 bytes/sec (well within UART capacity).
"""

import struct
from dataclasses import dataclass
from typing import Dict, Optional

import rclpy
from geometry_msgs.msg import Twist, Vector3
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


# Serial frame constants — Jetson → Pi (thruster commands)
CMD_HEADER_1 = 0xAA
CMD_HEADER_2 = 0x55
CMD_FRAME_SIZE = 16

# Serial frame constants — Pi → Jetson (sensor data)
SENSOR_HEADER_1 = 0xBB
SENSOR_HEADER_2 = 0x66
SENSOR_HEADER = bytes([SENSOR_HEADER_1, SENSOR_HEADER_2])
SENSOR_FRAME_SIZE = 20   # 2 header + 17 payload + 1 CRC


@dataclass
class Request:
    command: Twist
    received_at_ns: int


def crc8(data: bytes) -> int:
    """CRC-8/0x07 over the data payload for serial frame integrity."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def float_to_int16(value: float) -> int:
    """Convert a float velocity to a signed int16 (milli-units).

    Clamps to [-32767, 32767] to stay within int16 range.
    Example: 0.55 → 550, -0.30 → -300
    """
    raw = int(value * 1000)
    return max(-32767, min(32767, raw))


def build_serial_frame(twist: Twist, gripper_open: bool) -> bytes:
    """Build the 16-byte serial frame sent to the Raspberry Pi.

    Frame: [0xAA][0x55][surge_hi][surge_lo][sway_hi][sway_lo]
           [heave_hi][heave_lo][roll_hi][roll_lo][pitch_hi][pitch_lo]
           [yaw_hi][yaw_lo][gripper][CRC8]
    """
    surge = float_to_int16(twist.linear.x)
    sway  = float_to_int16(twist.linear.y)
    heave = float_to_int16(twist.linear.z)
    roll  = float_to_int16(twist.angular.x)
    pitch = float_to_int16(twist.angular.y)
    yaw   = float_to_int16(twist.angular.z)
    gripper_byte = 0x01 if gripper_open else 0x00

    # Pack 6 signed int16 values (big-endian) + 1 gripper byte = 13 data bytes
    payload = struct.pack('>hhhhhhB', surge, sway, heave, roll, pitch, yaw, gripper_byte)
    checksum = crc8(payload)

    return bytes([CMD_HEADER_1, CMD_HEADER_2]) + payload + bytes([checksum])


class ReactiveControlLayer(Node):
    """Arbitrate mission Twist commands and send them to the Raspberry Pi over serial UART."""

    # 'stabilizer' is handled specially — it is NOT in SOURCES because it doesn't
    # compete in priority arbitration.  Its roll/pitch are always merged into the
    # winning mission Twist.
    SOURCES = ('manual', 'path', 'transit', 'gate', 'buoy', 'dropper', 'torpedo')
    DEFAULT_PRIORITIES = {
        'manual': 100,
        'torpedo': 50,
        'dropper': 40,
        'gate': 30,
        'buoy': 20,
        'transit': 15,
        'path': 10,
    }

    def __init__(self):
        super().__init__('reactive_control_layer')

        # ── Parameters ──────────────────────────────────────────────
        self.declare_parameter('enabled', False)
        self.declare_parameter('input_timeout_sec', 0.5)
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('serial_port', '/dev/ttyTHS1')   # Jetson Nano UART TX/RX pins
        self.declare_parameter('serial_baud', 115200)

        # ── State ───────────────────────────────────────────────────
        self.enabled: bool = bool(self.get_parameter('enabled').value)
        self.selected_source: str = ''
        self.requests: Dict[str, Request] = {}
        self.gripper_open: bool = False
        self.serial_port = None
        self.stabilizer_twist: Optional[Twist] = None      # Latest stabilizer correction
        self.stabilizer_received_ns: int = 0                # Timestamp of last stabilizer msg
        self.sensor_rx_buffer = bytearray()                 # Buffer for incoming sensor frames

        # ── Publishers ──────────────────────────────────────────────
        self.selected_twist_pub = self.create_publisher(
            Twist, '/control/selected_cmd_vel', 10
        )
        self.status_pub = self.create_publisher(
            String, '/control/status', 10
        )

        # Sensor data publishers (data received from Pi via serial RX)
        self.dvl_velocity_pub = self.create_publisher(Vector3, '/sensors/dvl_velocity', 10)
        self.dvl_valid_pub = self.create_publisher(Bool, '/sensors/dvl_valid', 10)
        self.dvl_altitude_pub = self.create_publisher(Float32, '/sensors/dvl_altitude', 10)
        self.depth_pub = self.create_publisher(Float32, '/sensors/depth', 10)
        self.heading_pub = self.create_publisher(Float32, '/sensors/heading', 10)
        self.pitch_pub = self.create_publisher(Float32, '/sensors/pitch', 10)
        self.roll_pub = self.create_publisher(Float32, '/sensors/roll', 10)

        # ── Subscriptions ───────────────────────────────────────────
        self.create_subscription(Bool, '/control/enable', self._enable_callback, 10)
        self.create_subscription(String, '/control/select_source', self._select_source_callback, 10)
        self.create_subscription(Bool, '/actuators/gripper_open', self._gripper_callback, 10)

        for source in self.SOURCES:
            topic = '/control/request' if source == 'manual' else f'/control/requests/{source}'
            self.create_subscription(Twist, topic, self._make_request_callback(source), 10)

        # Stabilizer has its own dedicated subscription (always-on, not arbitrated)
        self.create_subscription(
            Twist, '/control/requests/stabilizer', self._stabilizer_callback, 10
        )

        # ── Serial Connection ───────────────────────────────────────
        self._connect_serial()

        # ── Control Loop ────────────────────────────────────────────
        rate = float(self.get_parameter('control_rate_hz').value)
        self.timer = self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info('Control layer ready — disabled until /control/enable is True.')

    # ── Callbacks ───────────────────────────────────────────────────

    def _make_request_callback(self, source: str):
        """Return a closure that timestamps and stores incoming Twist commands."""
        def callback(command: Twist):
            self.requests[source] = Request(command, self.get_clock().now().nanoseconds)
        return callback

    def _enable_callback(self, message: Bool):
        self.enabled = message.data
        if not self.enabled:
            # Immediately send a zero-velocity frame to stop the vehicle
            self._send_serial(Twist(), gripper_open=False)
        state = 'ENABLED' if self.enabled else 'DISABLED'
        self.get_logger().info(f'Control layer {state}.')

    def _select_source_callback(self, message: String):
        source = message.data.strip().lower()
        if source and source not in self.SOURCES:
            self.get_logger().warning(f'Ignoring unknown source: "{source}"')
            return
        self.selected_source = source
        self.get_logger().info(f'Source lock: {"ANY (priority)" if not source else source}')

    def _gripper_callback(self, message: Bool):
        self.gripper_open = message.data

    def _stabilizer_callback(self, command: Twist):
        """Store the latest stabilizer roll/pitch correction (always-on, not arbitrated)."""
        self.stabilizer_twist = command
        self.stabilizer_received_ns = self.get_clock().now().nanoseconds

    # ── Arbitration ─────────────────────────────────────────────────

    def _active_request(self):
        """Select the highest-priority, non-stale Twist request."""
        now_ns = self.get_clock().now().nanoseconds
        timeout_ns = int(float(self.get_parameter('input_timeout_sec').value) * 1e9)

        fresh = {
            source: request for source, request in self.requests.items()
            if now_ns - request.received_at_ns <= timeout_ns
        }

        # If operator locked to a specific source, only use that
        if self.selected_source:
            if self.selected_source in fresh:
                return self.selected_source, fresh[self.selected_source]
            return None, None

        if not fresh:
            return None, None

        # Pick by (priority DESC, timestamp DESC) — highest priority wins,
        # ties broken by most recent command
        source = max(
            fresh,
            key=lambda name: (self.DEFAULT_PRIORITIES[name], fresh[name].received_at_ns),
        )
        return source, fresh[source]

    # ── Control Loop Tick ───────────────────────────────────────────

    def _tick(self):
        source, request = self._active_request()

        # If disabled or no active request → zero velocity
        command = request.command if self.enabled and request else Twist()

        # ── Merge stabilizer roll/pitch into the mission Twist ──────
        # The stabilizer's angular.x (roll) and angular.y (pitch) are
        # always applied, regardless of which mission source won.
        # Mission behaviors should never set angular.x or angular.y.
        command = self._merge_stabilizer(command)

        # Publish the merged Twist for debugging / visualization
        self.selected_twist_pub.publish(command)

        # Send over serial to Raspberry Pi
        self._send_serial(command, gripper_open=self.gripper_open)

        # Read incoming sensor frames from Pi (DVL, depth, heading, etc.)
        self._read_sensor_frames()

        # Publish status string for monitoring
        state = 'enabled' if self.enabled else 'disabled'
        src_label = source or 'none'
        self.status_pub.publish(String(
            data=f'{state}; source={src_label}; '
                 f'surge={command.linear.x:.2f} sway={command.linear.y:.2f} '
                 f'heave={command.linear.z:.2f} '
                 f'roll={command.angular.x:.3f} pitch={command.angular.y:.3f} '
                 f'yaw={command.angular.z:.2f}'
        ))

    def _merge_stabilizer(self, mission_twist: Twist) -> Twist:
        """Overlay the stabilizer's roll/pitch onto the mission's Twist.

        The stabilizer is always-on and not subject to priority arbitration.
        If the stabilizer data is stale (>0.5s), roll/pitch are left at zero
        as a safety measure.
        """
        if self.stabilizer_twist is None:
            return mission_twist

        now_ns = self.get_clock().now().nanoseconds
        timeout_ns = int(float(self.get_parameter('input_timeout_sec').value) * 1e9)
        if now_ns - self.stabilizer_received_ns > timeout_ns:
            # Stabilizer data is stale — don't apply corrections
            return mission_twist

        # Merge: mission owns surge/sway/heave/yaw, stabilizer owns roll/pitch
        merged = Twist()
        merged.linear.x = mission_twist.linear.x
        merged.linear.y = mission_twist.linear.y
        merged.linear.z = mission_twist.linear.z
        merged.angular.x = self.stabilizer_twist.angular.x    # Roll from stabilizer
        merged.angular.y = self.stabilizer_twist.angular.y    # Pitch from stabilizer
        merged.angular.z = mission_twist.angular.z
        return merged

    # ── Serial Communication ───────────────────────────────────────

    def _connect_serial(self):
        """Open the UART serial port to the Raspberry Pi."""
        try:
            import serial
            port = self.get_parameter('serial_port').value
            baud = int(self.get_parameter('serial_baud').value)
            self.serial_port = serial.Serial(port, baud, timeout=0.01)
            self.get_logger().info(f'Serial connected: {port} @ {baud} baud.')
        except Exception as exc:
            self.serial_port = None
            self.get_logger().error(
                f'Serial port unavailable (Twist still published to ROS): {exc}'
            )

    def _send_serial(self, twist: Twist, gripper_open: bool):
        """Build and transmit the serial frame to the Raspberry Pi."""
        frame = build_serial_frame(twist, gripper_open)
        if self.serial_port is not None:
            try:
                self.serial_port.write(frame)
            except Exception as exc:
                self.get_logger().error(f'Serial write failed: {exc}')

    def _read_sensor_frames(self):
        """Read and parse incoming sensor frames (0xBB 0x66) from the Pi.

        Sensor Frame (20 bytes):
          [0xBB][0x66][dvl_vx][dvl_vy][dvl_vz][dvl_valid]
          [altitude][depth][heading][pitch][roll][CRC8]

        Publishes parsed data as individual ROS topics for downstream
        nodes (navigation, depth hold, stabilization) to subscribe to.
        """
        if self.serial_port is None:
            return

        try:
            # Read all available bytes (non-blocking)
            available = self.serial_port.in_waiting
            if available > 0:
                incoming = self.serial_port.read(available)
                self.sensor_rx_buffer.extend(incoming)
        except Exception as exc:
            self.get_logger().error(f'Serial read failed: {exc}')
            return

        # Try to parse complete sensor frames from the buffer
        while len(self.sensor_rx_buffer) >= SENSOR_FRAME_SIZE:
            # Search for sensor frame header (0xBB 0x66)
            header_pos = self.sensor_rx_buffer.find(SENSOR_HEADER)
            if header_pos == -1:
                # No header found — discard everything
                self.sensor_rx_buffer.clear()
                break
            if header_pos > 0:
                # Discard bytes before header
                del self.sensor_rx_buffer[:header_pos]
            if len(self.sensor_rx_buffer) < SENSOR_FRAME_SIZE:
                break

            # Extract candidate frame
            candidate = bytes(self.sensor_rx_buffer[:SENSOR_FRAME_SIZE])
            payload = candidate[2:19]       # 17 data bytes
            received_crc = candidate[19]

            if crc8(payload) == received_crc:
                # Valid frame — parse sensor data
                self._publish_sensor_data(payload)
                del self.sensor_rx_buffer[:SENSOR_FRAME_SIZE]
            else:
                # Bad CRC — skip one byte and retry
                del self.sensor_rx_buffer[:1]

    def _publish_sensor_data(self, payload: bytes):
        """Parse a 17-byte sensor payload and publish as ROS topics.

        Payload format ('>hhhBhhhhh'):
          dvl_vx:    int16, ×1000 (m/s)
          dvl_vy:    int16, ×1000 (m/s)
          dvl_vz:    int16, ×1000 (m/s)
          dvl_valid: uint8, 0x01 = valid
          altitude:  int16, ×1000 (m)
          depth:     int16, ×1000 (m)
          heading:   int16, ×100  (degrees)
          pitch:     int16, ×100  (degrees)
          roll:      int16, ×100  (degrees)
        """
        dvl_vx_raw, dvl_vy_raw, dvl_vz_raw, dvl_valid_byte, \
            altitude_raw, depth_raw, heading_raw, pitch_raw, roll_raw = \
            struct.unpack('>hhhBhhhhh', payload)

        # Convert back to floats
        dvl_vx = dvl_vx_raw / 1000.0
        dvl_vy = dvl_vy_raw / 1000.0
        dvl_vz = dvl_vz_raw / 1000.0
        dvl_valid = dvl_valid_byte == 0x01
        altitude = altitude_raw / 1000.0
        depth = depth_raw / 1000.0
        heading = heading_raw / 100.0
        pitch = pitch_raw / 100.0
        roll = roll_raw / 100.0

        # Publish DVL velocity (body-frame forward/sideways/vertical)
        vel_msg = Vector3()
        vel_msg.x = dvl_vx
        vel_msg.y = dvl_vy
        vel_msg.z = dvl_vz
        self.dvl_velocity_pub.publish(vel_msg)

        # Publish DVL validity
        self.dvl_valid_pub.publish(Bool(data=dvl_valid))

        # Publish DVL altitude (distance to pool floor)
        self.dvl_altitude_pub.publish(Float32(data=altitude))

        # Publish depth (Bar30)
        self.depth_pub.publish(Float32(data=depth))

        # Publish heading, pitch, roll (Ellipse-N INS)
        self.heading_pub.publish(Float32(data=heading))
        self.pitch_pub.publish(Float32(data=pitch))
        self.roll_pub.publish(Float32(data=roll))

    # ── Lifecycle ───────────────────────────────────────────────────

    def destroy_node(self):
        # Send stop frame before shutting down
        self._send_serial(Twist(), gripper_open=False)
        if self.serial_port is not None:
            self.serial_port.close()
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
