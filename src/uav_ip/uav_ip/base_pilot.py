#!/usr/bin/env python3
import math
from typing import Optional, Sequence
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition


class BasePilot(Node):
    """
    BasePilot provides the lifecycle of ROS 2, arming, QoS, coordinate frame
    conversions (Gazebo ENU <-> PX4 NED), and publishing/subscribing to PX4
    messages for offboard control.
    """
    def __init__(self, node_name: str = "base_pilot"):
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
        self.current_pos_ned = [float(msg.x), float(msg.y), float(msg.z)]
        self.px4_timestamp = msg.timestamp

    @staticmethod
    def gazebo_to_ned(coord: Sequence[float]) -> list[float]:
        """Convierte posición o vector libre desde Gazebo ENU [E, N, U] a PX4 NED [N, E, D]."""
        return [float(coord[1]), float(coord[0]), -float(coord[2])]

    @staticmethod
    def ned_to_gazebo(coord: Sequence[float]) -> list[float]:
        """Convierte posición o vector libre desde PX4 NED [N, E, D] a Gazebo ENU [E, N, U]."""
        return [float(coord[1]), float(coord[0]), -float(coord[2])]

    @staticmethod
    def yaw_gazebo_to_ned(yaw: float) -> float:
        """Convierte ángulo yaw desde Gazebo ENU (0=Este, CCW) a PX4 NED (0=Norte, CW)."""
        return math.atan2(math.sin(math.pi / 2.0 - yaw), math.cos(math.pi / 2.0 - yaw))

    @staticmethod
    def yaw_ned_to_gazebo(yaw_ned: float) -> float:
        """Convierte ángulo yaw desde PX4 NED (0=Norte, CW) a Gazebo ENU (0=Este, CCW)."""
        return math.atan2(math.sin(math.pi / 2.0 - yaw_ned), math.cos(math.pi / 2.0 - yaw_ned))

    def publish_offboard_mode(self, use_velocity: bool = False) -> None:
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = bool(use_velocity)
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = self.px4_timestamp
        self.offboard_pub.publish(msg)

    def publish_trajectory_setpoint(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        velocity: Optional[Sequence[float]] = None,
        yawspeed: float = float("nan"),
    ) -> None:
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        if velocity is not None:
            msg.velocity = [float(velocity[0]), float(velocity[1]), float(velocity[2])]
        else:
            msg.velocity = [float("nan")] * 3
        msg.acceleration = [float("nan")] * 3
        msg.jerk = [float("nan")] * 3
        msg.yaw = float(yaw)
        msg.yawspeed = float(yawspeed)
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
