# ~/colcon_ws/src/robotis_mujoco_menagerie/robotis_omy/launch/omy.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        # 1) MuJoCo + OMY 모델 구동
        Node(
            package='mujoco_ros2_control',
            executable='mujoco_ros2_control_node',
            name='mujoco',
            parameters=[
                '/home/ohseohyun/colcon_ws/src/robotis_mujoco_menagerie/robotis_omy/omy.xml'
            ],
            output='screen'
        ),

        # 2) ros2_control_node (controller_manager) 구동
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            name='controller_manager',
            parameters=[
                'config/control.yaml',
                '/home/ohseohyun/colcon_ws/src/robotis_mujoco_menagerie/robotis_omy/omy.xml'
            ],
            output='screen'
        ),

        # 3) admittance_controller 자동 로드 & 활성화
        Node(
            package='controller_manager',
            executable='spawner',
            name='spawn_admittance',
            arguments=[
                'admittance_controller',
                '--controller-manager', '/controller_manager'
            ],
            output='screen'
        ),
    ])

