#!/usr/bin/env python3
import sys
import rclpy
import math
from pathlib import Path
import yaml
from uav_ip.base_pilot import BasePilot
from px4_msgs.msg import TrajectorySetpoint, OffboardControlMode, VehicleCommand

class TrackingController(BasePilot):
    """
    TrackingController implements Feedback control for primitives like orbit.
    """
    def __init__(self):
        super().__init__("tracking_controller")
        
        self.declare_parameter("waypoints_file", "")
        wp_file = self.get_parameter("waypoints_file").get_parameter_value().string_value
        
        self.target_gazebo = [5.0, 0.0, 2.0]
        self.radius = 2.0
        self.omega = 0.5
        
        if wp_file:
            wp_path = Path(wp_file)
            if wp_path.is_file():
                with open(wp_path, "r") as fh:
                    cfg = yaml.safe_load(fh)
                
                target_lm = None
                for prim in cfg.get("primitives", []):
                    if prim["type"] == "orbit":
                        lm_id = prim["target"]
                        for lm in cfg.get("landmarks", []):
                            if lm["id"] == lm_id:
                                target_lm = lm
                                break
                        break
                
                if target_lm:
                    self.target_gazebo = [float(target_lm["x"]), float(target_lm["y"]), float(target_lm["z"])]
                    self.get_logger().info(f"Targeting landmark {target_lm['id']} for orbit.")
        
        self.start_time = None
        self.timer = self.create_timer(0.02, self._control_loop)
        
    def publish_velocity_setpoint(self, vx: float, vy: float, vz: float, yaw: float) -> None:
        msg = TrajectorySetpoint()
        msg.position = [float("nan")] * 3
        msg.velocity = [float(vx), float(vy), float(vz)]
        msg.acceleration = [float("nan")] * 3
        msg.jerk = [float("nan")] * 3
        msg.yaw = float(yaw)
        msg.yawspeed = float("nan")
        msg.timestamp = self.px4_timestamp
        
        mode_msg = OffboardControlMode()
        mode_msg.position = False
        mode_msg.velocity = True
        mode_msg.acceleration = False
        mode_msg.attitude = False
        mode_msg.body_rate = False
        mode_msg.timestamp = self.px4_timestamp
        
        self.offboard_pub.publish(mode_msg)
        self.trajectory_pub.publish(msg)

    def _control_loop(self):
        if self.px4_timestamp == 0:
            return
            
        self.publish_offboard_mode()
            
        if self.nav_state == 0:
            # Send current position as idle setpoint to allow offboard to engage
            self.publish_trajectory_setpoint(self.current_pos_ned[0], self.current_pos_ned[1], self.current_pos_ned[2], self.yaw_gazebo_to_ned(math.pi/2))
                
        elif self.nav_state == 1:
            # Approaching the orbit starting point
            cx = self.target_gazebo[0] + self.radius
            cy = self.target_gazebo[1]
            cz = self.target_gazebo[2]
            
            target_ned = self.gazebo_to_ned([cx, cy, cz])
            self.publish_trajectory_setpoint(target_ned[0], target_ned[1], target_ned[2], self.yaw_gazebo_to_ned(math.pi/2))
            
            # Check distance
            dx = target_ned[0] - self.current_pos_ned[0]
            dy = target_ned[1] - self.current_pos_ned[1]
            dz = target_ned[2] - self.current_pos_ned[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            if dist < 0.5:
                self.nav_state = 2
                self.start_time = self.get_clock().now()
                self.get_logger().info("Punto de inicio alcanzado. Iniciando orbit PID...")
                
        elif self.nav_state == 2:
            t = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            
            cx = self.target_gazebo[0] + self.radius * math.cos(self.omega * t)
            cy = self.target_gazebo[1] + self.radius * math.sin(self.omega * t)
            cz = self.target_gazebo[2]
            
            target_ned = self.gazebo_to_ned([cx, cy, cz])
            
            Kp = 2.0
            vx = Kp * (target_ned[0] - self.current_pos_ned[0])
            vy = Kp * (target_ned[1] - self.current_pos_ned[1])
            vz = Kp * (target_ned[2] - self.current_pos_ned[2])
            
            yaw_gazebo = math.atan2(self.target_gazebo[1] - self.current_pos_ned[1], self.target_gazebo[0] - self.current_pos_ned[0])
            yaw_ned = self.yaw_gazebo_to_ned(yaw_gazebo)
            
            self.publish_velocity_setpoint(vx, vy, vz, yaw_ned)

        self.handle_arming()

def main(args=None):
    rclpy.init(args=args)
    node = TrackingController()
    try:
        rclpy.spin(node)
    except SystemExit as e:
        node.get_logger().info(f"Nodo terminado con codigo {e.code}.")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == "__main__":
    main()
