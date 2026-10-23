"""Master Mission Sequencer Node for NIOT SAVe AUV.

Orchestrates the entire competition mission run:
1. DIVE: Commands initial dive to operational depth (default 1.2m).
2. TRANSIT_1: Follows path markers while front camera concurrently scans for buoys.
   Transitions immediately to BUOY as soon as front camera spots the buoys,
   or if safety distance threshold is reached.
3. BUOY_INTERACTION: Smart 3-buoy task (touch configured target color, avoid non-targets).
4. TRANSIT_2: Path follow while front camera scans for L-Gate.
5. GATE_PASS: Aligns and traverses through the L-Gate.
6. TRANSIT_3: Follows path to fork point.
7. FORK_TASK_1: Executes pre-configured branch (Torpedo or Dropper) as selected by ROS parameter.
8. INTER_STATION_TRANSIT: Direct waypoint navigation between stations without path markers.
9. FORK_TASK_2: Executes the second branch task.
10. TRANSIT_OCTAGON: Navigates to the surfacing octagon zone.
11. OCTAGON_SURFACE: Aligns within the octagon and surfaces cleanly.
"""

from enum import auto, Enum
import time

from auv_interfaces.msg import BinDetection
from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


class MissionState(Enum):
    IDLE = auto()
    DIVE = auto()
    TRANSIT_TO_BUOY = auto()
    BUOY_INTERACTION = auto()
    TRANSIT_TO_GATE = auto()
    GATE_PASS = auto()
    TRANSIT_TO_FORK = auto()
    BRANCH_TASK_1 = auto()
    INTER_STATION_TRANSIT = auto()
    BRANCH_TASK_2 = auto()
    TRANSIT_TO_OCTAGON = auto()
    OCTAGON_SURFACE = auto()
    COMPLETED = auto()
    ABORTED = auto()


class MissionSequencerNode(Node):
    """Autonomous State Machine Coordinator for SAVe AUV."""

    def __init__(self):
        super().__init__('mission_sequencer_node')

        # ── Declare Parameters ──────────────────────────────────────
        self.declare_parameter('auto_start', False)
        self.declare_parameter('operational_depth', 1.20)     # meters
        self.declare_parameter('transit_speed', 0.35)          # m/s
        self.declare_parameter('target_buoy_color', 'red')     # 'red', 'green', 'yellow'
        self.declare_parameter(
            'first_task_after_gate', 'torpedo'
        )  # torpedo or dropper
        self.declare_parameter('max_path_dist_to_buoy', 7.5)   # safety timeout (meters)
        self.declare_parameter('max_path_dist_to_gate', 8.0)
        self.declare_parameter(
            'inter_station_distance', 4.0
        )  # distance between torpedo and dropper (meters)

        self.auto_start = bool(self.get_parameter('auto_start').value)
        self.target_depth = float(self.get_parameter('operational_depth').value)
        self.transit_speed = float(self.get_parameter('transit_speed').value)
        self.target_buoy_color = str(self.get_parameter('target_buoy_color').value).lower()
        self.first_task = str(self.get_parameter('first_task_after_gate').value).lower()
        if self.first_task not in ('torpedo', 'dropper'):
            raise ValueError(
                'first_task_after_gate must be torpedo or dropper'
            )
        self.max_dist_buoy = float(self.get_parameter('max_path_dist_to_buoy').value)
        self.max_dist_gate = float(self.get_parameter('max_path_dist_to_gate').value)
        self.inter_station_dist = float(self.get_parameter('inter_station_distance').value)

        # ── State Machine Tracking ──────────────────────────────────
        self.state = MissionState.IDLE
        self.active_task = 'idle'
        self.state_start_time = time.time()
        self.state_start_distance = 0.0

        # Sensor & Nav Inputs
        self.current_depth: float = 0.0
        self.current_heading: float = 0.0
        self.distance_traveled: float = 0.0

        # Vision Inputs
        self.buoy_detected: bool = False
        self.gate_detected: bool = False
        self.bin_detected: bool = False
        self.torpedo_detected: bool = False
        self.octagon_detected: bool = False

        # Task Completion Signals
        self.last_completed_task: str = ''

        # ── Publishers ──────────────────────────────────────────────
        self.state_pub = self.create_publisher(String, '/mission/state', 10)
        self.active_task_pub = self.create_publisher(String, '/mission/active_task', 10)
        self.select_source_pub = self.create_publisher(String, '/control/select_source', 10)
        self.control_enable_pub = self.create_publisher(Bool, '/control/enable', 10)
        self.vision_mode_pub = self.create_publisher(String, '/vision/mode', 10)

        # Navigation control topics
        self.nav_depth_pub = self.create_publisher(Float32, '/nav/set_target_depth', 10)
        self.nav_heading_pub = self.create_publisher(Float32, '/nav/set_target_heading', 10)
        self.nav_speed_pub = self.create_publisher(Float32, '/nav/set_transit_speed', 10)
        self.nav_transit_enable_pub = self.create_publisher(Bool, '/nav/enable_transit', 10)
        self.nav_depth_hold_enable_pub = self.create_publisher(Bool, '/nav/enable_depth_hold', 10)
        self.nav_reset_dist_pub = self.create_publisher(Bool, '/nav/reset_distance', 10)

        # Direct transit twist
        self.transit_cmd_pub = self.create_publisher(Twist, '/control/requests/transit', 10)

        # ── Subscriptions ───────────────────────────────────────────
        self.create_subscription(Bool, '/mission/start', self._start_cb, 10)
        self.create_subscription(Bool, '/mission/abort', self._abort_cb, 10)
        self.create_subscription(String, '/mission/task_completed', self._task_completed_cb, 10)

        # Sensors & Nav
        self.create_subscription(Float32, '/sensors/depth', self._depth_cb, 10)
        self.create_subscription(Float32, '/sensors/heading', self._heading_cb, 10)
        self.create_subscription(Float32, '/nav/distance_traveled', self._dist_cb, 10)

        # Vision detections
        self.create_subscription(Bool, '/vision/target_found', self._buoy_seen_cb, 10)
        self.create_subscription(Bool, '/vision/gate_found', self._gate_seen_cb, 10)
        self.create_subscription(
            BinDetection, '/vision/bin_detection', self._bin_seen_cb, 10
        )
        self.create_subscription(Bool, '/vision/torpedo_found', self._torpedo_seen_cb, 10)
        self.create_subscription(Bool, '/vision/octagon_found', self._octagon_seen_cb, 10)

        # ── Sequencer Tick Loop (10 Hz) ─────────────────────────────
        self.timer = self.create_timer(0.1, self._sequencer_tick)

        self.get_logger().info(
            f'MissionSequencerNode ready. Config: target_depth={self.target_depth}m, '
            f'target_buoy={self.target_buoy_color}, first_branch={self.first_task}'
        )

        if self.auto_start:
            self._transition_to(MissionState.DIVE)

    # ── Callbacks ───────────────────────────────────────────────────

    def _start_cb(self, msg: Bool):
        if msg.data and self.state in (
            MissionState.IDLE,
            MissionState.COMPLETED,
            MissionState.ABORTED,
        ):
            self.get_logger().info('Mission START commanded!')
            self._transition_to(MissionState.DIVE)

    def _abort_cb(self, msg: Bool):
        if msg.data:
            self.get_logger().warn('Mission ABORT commanded!')
            self._transition_to(MissionState.ABORTED)

    def _task_completed_cb(self, msg: String):
        completed_task = msg.data.strip().lower()
        expected_task = {
            MissionState.BUOY_INTERACTION: 'buoy',
            MissionState.GATE_PASS: 'gate',
            MissionState.BRANCH_TASK_1: self.first_task,
            MissionState.BRANCH_TASK_2: (
                'dropper' if self.first_task == 'torpedo' else 'torpedo'
            ),
        }.get(self.state)
        if completed_task != expected_task:
            return
        if completed_task != self.last_completed_task:
            self.get_logger().info(f'Received task completion: {completed_task}')
        self.last_completed_task = completed_task

    def _depth_cb(self, msg: Float32):
        self.current_depth = msg.data

    def _heading_cb(self, msg: Float32):
        self.current_heading = msg.data

    def _dist_cb(self, msg: Float32):
        self.distance_traveled = msg.data

    def _buoy_seen_cb(self, msg: Bool):
        self.buoy_detected = msg.data

    def _gate_seen_cb(self, msg: Bool):
        self.gate_detected = msg.data

    def _bin_seen_cb(self, msg: BinDetection):
        self.bin_detected = msg.detected

    def _torpedo_seen_cb(self, msg: Bool):
        self.torpedo_detected = msg.data

    def _octagon_seen_cb(self, msg: Bool):
        self.octagon_detected = msg.data

    # ── State Machine Transitions ───────────────────────────────────

    def _set_active_task(self, task_name: str):
        self.active_task = task_name
        self.active_task_pub.publish(String(data=task_name))

    def _transition_to(self, new_state: MissionState):
        self.get_logger().info(f'Transition: {self.state.name} ──▶ {new_state.name}')
        self.state = new_state
        self.state_start_time = time.time()
        self.state_start_distance = self.distance_traveled
        self.last_completed_task = ''

        # Enter-state actions
        if new_state == MissionState.DIVE:
            self.control_enable_pub.publish(Bool(data=True))
            self.select_source_pub.publish(String(data='transit'))
            self._set_active_task('dive')
            self.nav_depth_pub.publish(Float32(data=self.target_depth))
            self.nav_depth_hold_enable_pub.publish(Bool(data=True))
            self.nav_speed_pub.publish(Float32(data=0.0))
            self.nav_transit_enable_pub.publish(Bool(data=True))

        elif new_state == MissionState.TRANSIT_TO_BUOY:
            self.select_source_pub.publish(String(data='path'))
            self._set_active_task('path_transit_to_buoy')
            self.vision_mode_pub.publish(String(data='path_and_buoy_scan'))
            self.nav_reset_dist_pub.publish(Bool(data=True))

        elif new_state == MissionState.BUOY_INTERACTION:
            self.select_source_pub.publish(String(data='buoy'))
            self._set_active_task('buoy')
            self.vision_mode_pub.publish(String(data='buoy_full'))

        elif new_state == MissionState.TRANSIT_TO_GATE:
            self.select_source_pub.publish(String(data='path'))
            self._set_active_task('path_transit_to_gate')
            self.vision_mode_pub.publish(String(data='path_and_gate_scan'))
            self.nav_reset_dist_pub.publish(Bool(data=True))

        elif new_state == MissionState.GATE_PASS:
            self.select_source_pub.publish(String(data='gate'))
            self._set_active_task('gate')
            self.vision_mode_pub.publish(String(data='gate_full'))

        elif new_state == MissionState.TRANSIT_TO_FORK:
            self.select_source_pub.publish(String(data='path'))
            self._set_active_task('path_transit_to_fork')
            self.nav_reset_dist_pub.publish(Bool(data=True))

        elif new_state == MissionState.BRANCH_TASK_1:
            task_name = self.first_task
            self.select_source_pub.publish(String(data=task_name))
            self._set_active_task(task_name)
            self.vision_mode_pub.publish(String(data=f'{task_name}_full'))

        elif new_state == MissionState.INTER_STATION_TRANSIT:
            # Direct dead reckoning travel between Torpedo and Dropper
            self.select_source_pub.publish(String(data='transit'))
            self._set_active_task('inter_station_transit')
            self.nav_speed_pub.publish(Float32(data=self.transit_speed))
            self.nav_transit_enable_pub.publish(Bool(data=True))
            self.nav_reset_dist_pub.publish(Bool(data=True))

        elif new_state == MissionState.BRANCH_TASK_2:
            second_task = 'dropper' if self.first_task == 'torpedo' else 'torpedo'
            self.select_source_pub.publish(String(data=second_task))
            self._set_active_task(second_task)
            self.vision_mode_pub.publish(String(data=f'{second_task}_full'))

        elif new_state == MissionState.TRANSIT_TO_OCTAGON:
            self.select_source_pub.publish(String(data='transit'))
            self._set_active_task('transit_to_octagon')
            self.vision_mode_pub.publish(String(data='octagon_scan'))
            self.nav_speed_pub.publish(Float32(data=self.transit_speed))
            self.nav_transit_enable_pub.publish(Bool(data=True))

        elif new_state == MissionState.OCTAGON_SURFACE:
            self.select_source_pub.publish(String(data='transit'))
            self._set_active_task('surface')
            # Surface to 0.0m
            self.nav_depth_pub.publish(Float32(data=0.0))
            self.nav_speed_pub.publish(Float32(data=0.0))

        elif new_state in (MissionState.COMPLETED, MissionState.ABORTED):
            self.control_enable_pub.publish(Bool(data=False))
            self._set_active_task('idle')
            self.select_source_pub.publish(String(data=''))
            self.nav_transit_enable_pub.publish(Bool(data=False))

    # ── Master Sequencer Tick Loop ──────────────────────────────────

    def _sequencer_tick(self):
        # Always publish state string for UI and logging
        self.state_pub.publish(String(data=self.state.name))
        self.active_task_pub.publish(String(data=self.active_task))

        elapsed_sec = time.time() - self.state_start_time
        dist_in_state = self.distance_traveled - self.state_start_distance

        if self.state == MissionState.IDLE:
            pass

        elif self.state == MissionState.DIVE:
            # Condition: reached operational depth (within ±10 cm)
            if abs(self.current_depth - self.target_depth) <= 0.10 or elapsed_sec > 15.0:
                self._transition_to(MissionState.TRANSIT_TO_BUOY)

        elif self.state == MissionState.TRANSIT_TO_BUOY:
            # Key User Requirement:
            # As soon as front camera detects buoys, stop path following and switch!
            # Or if DVL distance reaches safety threshold (e.g., 7.5m)
            if self.buoy_detected or dist_in_state >= self.max_dist_buoy or elapsed_sec > 30.0:
                self._transition_to(MissionState.BUOY_INTERACTION)

        elif self.state == MissionState.BUOY_INTERACTION:
            # Wait for buoy task completion
            if self.last_completed_task == 'buoy' or elapsed_sec > 45.0:
                self._transition_to(MissionState.TRANSIT_TO_GATE)

        elif self.state == MissionState.TRANSIT_TO_GATE:
            # Front camera detects gate or DVL distance reached
            if self.gate_detected or dist_in_state >= self.max_dist_gate or elapsed_sec > 35.0:
                self._transition_to(MissionState.GATE_PASS)

        elif self.state == MissionState.GATE_PASS:
            # Wait for gate pass completion
            if self.last_completed_task == 'gate' or elapsed_sec > 40.0:
                self._transition_to(MissionState.TRANSIT_TO_FORK)

        elif self.state == MissionState.TRANSIT_TO_FORK:
            # Reach fork point (e.g. 3m travel or path splits)
            if dist_in_state >= 3.0 or elapsed_sec > 20.0:
                self._transition_to(MissionState.BRANCH_TASK_1)

        elif self.state == MissionState.BRANCH_TASK_1:
            if self.last_completed_task == self.first_task or elapsed_sec > 50.0:
                self._transition_to(MissionState.INTER_STATION_TRANSIT)

        elif self.state == MissionState.INTER_STATION_TRANSIT:
            # Direct travel between stations (user requirement: no path following needed here)
            if dist_in_state >= self.inter_station_dist or elapsed_sec > 25.0:
                self._transition_to(MissionState.BRANCH_TASK_2)

        elif self.state == MissionState.BRANCH_TASK_2:
            second_task = 'dropper' if self.first_task == 'torpedo' else 'torpedo'
            if self.last_completed_task == second_task or elapsed_sec > 50.0:
                self._transition_to(MissionState.TRANSIT_TO_OCTAGON)

        elif self.state == MissionState.TRANSIT_TO_OCTAGON:
            if self.octagon_detected or dist_in_state >= 5.0 or elapsed_sec > 30.0:
                self._transition_to(MissionState.OCTAGON_SURFACE)

        elif self.state == MissionState.OCTAGON_SURFACE:
            # Ascend to surface (< 0.15m)
            if self.current_depth <= 0.15 or elapsed_sec > 20.0:
                self.get_logger().info('Vehicle reached surface! Mission Completed.')
                self._transition_to(MissionState.COMPLETED)

        elif self.state in (MissionState.COMPLETED, MissionState.ABORTED):
            pass


def main(args=None):
    rclpy.init(args=args)
    node = MissionSequencerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
