"""Gazebo + bridge + sim_detector + your buoy_mission_node + reactive_control_layer.

  ros2 launch auv_sim sim.launch.py target_color:=red
  ros2 launch auv_sim sim.launch.py target_color:=green fake_sensors:=true
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def topic_pub(topic, msg_type, data_expr, rate="1", condition=None):
    """Publish a constant at `rate` Hz (stands in for mission_sequencer_node)."""
    return ExecuteProcess(
        cmd=["ros2", "topic", "pub", "-r", rate, topic, msg_type, data_expr],
        output="log",
        condition=condition)

def generate_launch_description():
    target_color = LaunchConfiguration("target_color")
    active_task = LaunchConfiguration("active_task")
    select_source = LaunchConfiguration("select_source")
    fake_sensors = LaunchConfiguration("fake_sensors")

    world = os.path.join(get_package_share_directory("auv_sim"), "worlds", "pool.sdf")
    gz = ExecuteProcess(cmd=["gz", "sim", "-r", world], output="screen")

    bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge", output="screen",
        arguments=[
            "/model/auv/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",   # ROS -> GZ
            "/camera/image@sensor_msgs/msg/Image[gz.msgs.Image",          # GZ -> ROS
        ],
        remappings=[("/camera/image", "/camera/image_raw")])

    detector = Node(
        package="auv_sim", executable="sim_detector.py", output="screen",
        parameters=[{"target_color": target_color,
                     "image_topic": "/camera/image_raw"}])

    mission = Node(package="auv_mission_planner", executable="buoy_mission_node",
                   output="screen")

    # Your control layer's output goes straight into Gazebo.
    control = Node(
        package="auv_mission_planner", executable="reactive_control_layer",
        output="screen",
        remappings=[("/control/selected_cmd_vel", "/model/auv/cmd_vel")])

    # ---- stand-ins for mission_sequencer_node ----
    # GUESS: the string values below ("buoy") must match what your nodes expect.
    triggers = [
        topic_pub("/mission/active_task", "std_msgs/msg/String",
                  ["{data: '", active_task, "'}"]),
        topic_pub("/mission/target_buoy_color", "std_msgs/msg/String",
                  ["{data: '", target_color, "'}"]),
        topic_pub("/control/enable", "std_msgs/msg/Bool", "{data: true}"),
        topic_pub("/control/select_source", "std_msgs/msg/String",
                  ["{data: '", select_source, "'}"]),
    ]

    sensors = [
        topic_pub("/sensors/depth", "std_msgs/msg/Float32", "{data: 2.0}", "10",
                  condition=IfCondition(fake_sensors)),
        topic_pub("/sensors/dvl_valid", "std_msgs/msg/Bool", "{data: true}", "10",
                  condition=IfCondition(fake_sensors)),
        topic_pub("/sensors/dvl_velocity", "geometry_msgs/msg/Vector3",
                  "{x: 0.0, y: 0.0, z: 0.0}", "10",
                  condition=IfCondition(fake_sensors)),
    ]

    return LaunchDescription([
        DeclareLaunchArgument("target_color", default_value="red"),
        DeclareLaunchArgument("active_task", default_value="buoy"),
        DeclareLaunchArgument("select_source", default_value="buoy"),
        DeclareLaunchArgument("fake_sensors", default_value="false"),
        gz, bridge,
        TimerAction(period=5.0, actions=[detector, control]),
        TimerAction(period=7.0, actions=[mission]),
        TimerAction(period=9.0, actions=triggers + sensors),
    ])
