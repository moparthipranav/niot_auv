from setuptools import find_packages, setup

package_name = 'auv_mission_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pranav',
    maintainer_email='venkatasaipranav.m24@iiits.in',
    description='Mission Planning for the NIOT SAVe competition AUV',
    license='SIDDHI',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'path_mission_node = auv_mission_planner.path_tree_node:main',
            'gate_mission_node = auv_mission_planner.gate_mission_node:main',
            'buoy_mission_node = auv_mission_planner.buoy_task_tree:main',
            'dropper_mission_node = auv_mission_planner.dropper_mission_node:main',
            'torpedo_mission_node = auv_mission_planner.torpedo_tree_node:main',
            'mission_sequencer_node = auv_mission_planner.mission_sequencer_node:main',
            'reactive_control_layer = reactive_control_layer.control_layer:main',
        ],
    },
)
