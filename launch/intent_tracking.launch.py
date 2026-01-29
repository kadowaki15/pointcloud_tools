from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pointcloud_tools',
            executable='ego_compensator',
            name='ego_compensator',
            output='screen',
        ),
        Node(
            package='pointcloud_tools',
            executable='dynamic_cluster_node',
            name='dynamic_cluster_node',
            output='screen',
        ),
        Node(
            package='pointcloud_tools',
            executable='tracker',
            name='tracker',
            output='screen',
        ),
        Node(
            package='pointcloud_tools',
            executable='intent_estimator',
            name='intent_estimator',
            output='screen',
        ),
        #Node(
         #   package='pointcloud_tools',
          #  executable='safety_interface',
           # name='safety_interface',
            #output='screen',
        #),
        Node(
            package='pointcloud_tools',
            executable='static_obstacle_detector',
            name='static_obstacle_detector',
            output='screen',
        ),
        Node(
            package='pointcloud_tools',
            executable='rgb_motion_gate_filter',
            name='rgb_motion_gate_filter',
            output='screen',
        ),
       
    ])
