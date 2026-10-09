#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition
import math

class BasePilot(Node):
    """
    BasePilot provides the lifecycle of ROS 2, arming, QoS, and publishing/subscribing
    to PX4 messages for offboard control.
    """
    def __init__(self, node_name="base_pilot"):
        super().__init__(node_name)
        
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.offboard_pub = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", qos)
        self.trajectory_pub = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos)
        self.vehicle_cmd_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos)

        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", self._pos_callback, qos)

        self.current_pos_ned = [0.0, 0.0, 0.0]
        self.px4_timestamp = 0
        self.nav_state = 0
        self.setpoint_counter = 0

    def _pos_callback(self, msg: VehicleLocalPosition) -> None:
        self.current_pos_ned = [msg.x, msg.y, msg.z]
        self.px4_timestamp = msg.timestamp

    @staticmethod
    def gazebo_to_ned(coord: list[float]) -> list[float]:
        return [coord[1], coord[0], -coord[2]]
        
    @staticmethod
    def yaw_gazebo_to_ned(yaw: float) -> float:
        return math.pi / 2.0 - yaw

    def publish_offboard_mode(self) -> None:
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = self.px4_timestamp
        self.offboard_pub.publish(msg)

    def publish_trajectory_setpoint(self, x: float, y: float, z: float, yaw: float) -> None:
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.velocity = [float("nan")] * 3
        msg.acceleration = [float("nan")] * 3
        msg.jerk = [float("nan")] * 3
        msg.yaw = float(yaw)
        msg.yawspeed = float("nan")
        msg.timestamp = self.px4_timestamp
        self.trajectory_pub.publish(msg)

    def publish_vehicle_command(self, command: int, param1: float = 0.0, param2: float = 0.0) -> None:
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self.px4_timestamp
        self.vehicle_cmd_pub.publish(msg)

    def handle_arming(self) -> None:
        if self.nav_state == 0:
            self.setpoint_counter += 1
            if self.setpoint_counter >= 50:
                self.nav_state = 1
                self.setpoint_counter = 0
                self.get_logger().info("Iniciando secuencia de despegue y re-intentos de armado...")

        if self.nav_state == 1:
            self.setpoint_counter += 1
            if self.setpoint_counter == 1 or self.setpoint_counter % 125 == 0:
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0, 21196.0)
                self.get_logger().info("Comandando ARM & OFFBOARD (reintento FORZADO)...")
