"""Run the bin vision and dropper mission against the Gazebo test world.

Launch with:
  ros2 launch auv_sim dropper_test.launch.py

Set visualize:=false to disable the annotated bin-detection image viewer.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def topic_pub(topic, message_type, data, rate='1'):
    return ExecuteProcess(
        cmd=['ros2', 'topic', 'pub', '-r', rate, topic, message_type, data],
        output='log',
    )


def generate_launch_description():
    bin_width = LaunchConfiguration('bin_width')
    fx = LaunchConfiguration('fx')
    drop_altitude = LaunchConfiguration('drop_altitude')
    target_marker = LaunchConfiguration('target_marker')
    visualize = LaunchConfiguration('visualize')
    world = os.path.join(
        get_package_share_directory('auv_sim'),
        'worlds',
        'dropper_test.sdf',
    )

    return LaunchDescription([
        DeclareLaunchArgument('bin_width', default_value='0.42'),
        DeclareLaunchArgument('fx', default_value='468.0'),
        DeclareLaunchArgument(
            'drop_altitude',
            default_value='1.0',
            description='Minimum camera-to-bin range at payload release (m).',
        ),
        DeclareLaunchArgument(
            'target_marker',
            default_value='X',
            description='Marker to drop the payload into: X or O.',
        ),
        DeclareLaunchArgument(
            'visualize',
            default_value='true',
            description='Open an image viewer for annotated bin detections.',
        ),
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', world],
            output='screen',
        ),
        TimerAction(
            period=3.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        'gz', 'topic', '-t', '/dropper/payload/attach',
                        '-m', 'gz.msgs.Empty', '-p', '',
                    ],
                    output='screen',
                ),
            ],
        ),
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                '/model/auv/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                '/dropper/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/actuators/gripper_open@std_msgs/msg/Bool]gz.msgs.Boolean',
            ],
            output='screen',
        ),
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='auv_vision',
                    executable='bin_detection',
                    parameters=[{
                        'use_sim_time': True,
                        'image_topic': '/dropper/camera/image',
                        'bin_width': bin_width,
                        'fx': fx,
                        'target_marker': target_marker,
                    }],
                    output='screen',
                ),
                Node(
                    package='auv_mission_planner',
                    executable='reactive_control_layer',
                    parameters=[{'use_sim_time': True}],
                    remappings=[
                        ('/control/selected_cmd_vel', '/model/auv/cmd_vel'),
                    ],
                    output='screen',
                ),
            ],
        ),
        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            arguments=['/vision/bin_detection/debug_image'],
            condition=IfCondition(visualize),
            output='screen',
        ),
        TimerAction(
            period=7.0,
            actions=[
                Node(
                    package='auv_mission_planner',
                    executable='dropper_mission_node',
                    parameters=[{
                        'use_sim_time': True,
                        'drop_altitude': drop_altitude,
                        'target_marker': target_marker,
                    }],
                    output='screen',
                ),
            ],
        ),
        TimerAction(
            period=15.0,
            actions=[
                topic_pub(
                    '/mission/active_task',
                    'std_msgs/msg/String',
                    "{data: 'dropper'}",
                ),
                topic_pub(
                    '/control/enable',
                    'std_msgs/msg/Bool',
                    '{data: true}',
                ),
                topic_pub(
                    '/control/select_source',
                    'std_msgs/msg/String',
                    "{data: 'dropper'}",
                ),
                topic_pub(
                    '/state/vehicle_stable',
                    'std_msgs/msg/Bool',
                    '{data: true}',
                    rate='10',
                ),
            ],
        ),
    ])
