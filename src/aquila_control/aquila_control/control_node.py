import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray


class ControlNode(Node):

    def __init__(self):
        super().__init__('control_node')

        # Receive movement commands
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Publish commands for the 4 motors
        self.motor_publisher = self.create_publisher(
            Float32MultiArray,
            '/motor_commands',
            10
        )

        # Safety watchdog
        self.last_command_time = self.get_clock().now()

        self.timer = self.create_timer(
            0.1,
            self.safety_check
        )

        self.get_logger().info(
            'Aquila Control Node Started!'
        )

    def cmd_vel_callback(self, msg):

        linear = msg.linear.x
        angular = msg.angular.z

        # Differential drive calculation
        left = linear - angular
        right = linear + angular

        # Limit values to [-1.0, 1.0]
        left = max(-1.0, min(1.0, left))
        right = max(-1.0, min(1.0, right))

        # 4 wheel commands
        motor_commands = Float32MultiArray()

        motor_commands.data = [
            left,    # Front Left
            right,   # Front Right
            left,    # Rear Left
            right    # Rear Right
        ]

        self.motor_publisher.publish(motor_commands)

        self.last_command_time = self.get_clock().now()

        self.get_logger().info(
            f'FL={left:.2f} '
            f'FR={right:.2f} '
            f'RL={left:.2f} '
            f'RR={right:.2f}'
        )

    def safety_check(self):

        elapsed = (
            self.get_clock().now()
            - self.last_command_time
        ).nanoseconds / 1e9

        # Stop motors if no command received
        if elapsed > 1.0:

            motor_commands = Float32MultiArray()

            motor_commands.data = [
                0.0,
                0.0,
                0.0,
                0.0
            ]

            self.motor_publisher.publish(motor_commands)


def main(args=None):

    rclpy.init(args=args)

    node = ControlNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        # Make sure motors stop when node closes
        motor_commands = Float32MultiArray()

        motor_commands.data = [
            0.0,
            0.0,
            0.0,
            0.0
        ]

        node.motor_publisher.publish(motor_commands)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
