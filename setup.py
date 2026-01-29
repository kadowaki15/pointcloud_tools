from setuptools import find_packages, setup

package_name = 'pointcloud_tools'

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
    maintainer='kadowaki',
    maintainer_email='reopi0812@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'pointcloud_diff = pointcloud_tools.pointcloud_diff:main',
        'ego_compensator = scripts.ego_compensator:main',
        'dynamic_cluster_node = scripts.dynamic_cluster_node:main',
        'tracker = scripts.tracker:main',
        'intent_estimator = scripts.intent_estimator:main',
        'safety_interface = scripts.safety_interface:main',
        'static_obstacle_detector = scripts.static_obstacle_detector:main',
        'rgb_motion_gate_filter= scripts.rgb_motion_gate_filter:main',
       
        ],
    },
)
