from setuptools import find_packages, setup

package_name = 'auv_vision'

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
    description='Vision assistance for SRI Ratnakara. An Autnomous Underwater Vehicle designed for NIOT SAVe competition',
    license='SIDDHI',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_publisher = auv_vision.camera_publisher:main',
            'object_detector = auv_vision.object_detector:main',
        ],
    },
)
