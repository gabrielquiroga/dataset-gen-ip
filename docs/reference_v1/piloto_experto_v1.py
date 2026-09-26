#!/usr/bin/env python3
"""
Expert Pilot v1 - Original con waypoints hardcodeados.
Archivo de referencia. El código activo está en uav_ip/expert_pilot.py (v2).

Mundo asociado: circuito_aros_01.sdf (= tf_circuito_aros.sdf en PX4)
Lanzamiento original:
    cd ~/PX4-Autopilot
    PX4_GZ_WORLD=tf_circuito_aros make px4_sitl gz_x500_depth

    # Terminal 2:
    MicroXRCEAgent udp4 -p 8888

    # Terminal 3:
    ros2 run ros_gz_bridge parameter_bridge "/depth_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked"

    # Terminal 4:
    python3 piloto_experto.py
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition
import math

class ExpertPilot(Node):
    def __init__(self):
        super().__init__('expert_pilot')
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.offboard_ctrl_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.pos_sub = self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.pos_callback, qos_profile)

        self.timer = self.create_timer(0.02, self.control_loop)

        # --- CORREDORES DE VUELO ---
        # Obligamos al dron a alinearse antes de la puerta y mantener 
        # la linea recta hasta haberla cruzado por completo.
        self.waypoints_gazebo = [
            [0.0, 0.0, 1.5],     # WP 0: Punto estricto de despegue
            
            # Aro 1 (Centro en X=5)
            [4.0, 0.0, 1.5],     # WP 1: Alineacion previa al Aro 1
            [6.0, 0.0, 1.5],     # WP 2: Salida recta (Garantiza cruzar limpio)
            
            # Aro 2 (Centro en X=10)
            [9.0, -1.5, 2.5],    # WP 3: Alineacion previa al Aro 2
            [11.0, -1.5, 2.5],   # WP 4: Salida recta
            
            # Aro 3 (Centro en X=15)
            [14.0, 0.0, 1.0],    # WP 5: Alineacion previa al Aro 3
            [16.0, 0.0, 1.0],    # WP 6: Salida recta
            
            # --- FASE DE ATERRIZAJE (Stop & Drop) ---
            [20.0, 0.0, 1.5],    # WP 7: Freno aereo. Centrarse sobre la plataforma.
            [20.0, 0.0, 0.3]     # WP 8: Descenso vertical estricto a 30cm
        ]
        
        self.waypoints_ned = [self.gazebo_to_ned(wp) for wp in self.waypoints_gazebo]
        
        self.current_wp = 0
        self.current_pos_ned = [0.0, 0.0, 0.0]
        
        # Maquina de estados: 0=Init, 1=Despegando, 2=Navegando, 3=Aterrizando, 4=Completado
        self.nav_state = 0 
        self.setpoint_counter = 0
        self.px4_timestamp = 0
        self.target_yaw = math.pi / 2.0

    def gazebo_to_ned(self, coord):
        return [coord[1], coord[0], -coord[2]]

    def pos_callback(self, msg):
        self.current_pos_ned = [msg.x, msg.y, msg.z]
        self.px4_timestamp = msg.timestamp 

    def control_loop(self):
        if self.px4_timestamp == 0:
            return

        self.publish_offboard_control_mode()
        
        # Inicializamos el target con la posicion actual por seguridad
        target = [self.current_pos_ned[0], self.current_pos_ned[1], self.current_pos_ned[2]]
        
        # Logica de la Maquina de Estados
        if self.nav_state == 0:
            target = [self.current_pos_ned[0], self.current_pos_ned[1], self.current_pos_ned[2]]
        
        elif self.nav_state == 1: # DESPEGUE ESTRICTO
            target = self.waypoints_ned[0]
            self.target_yaw = math.pi / 2.0 
            
            if abs(self.current_pos_ned[2] - target[2]) < 0.2:
                self.current_wp = 1
                self.nav_state = 2
                self.get_logger().info("Despegue completado. Iniciando circuito hacia Aro 1...")

        elif self.nav_state == 2: # NAVEGACION POR AROS Y STOP & DROP
            target = self.waypoints_ned[self.current_wp]
            
            # Yaw dinamico mirando al objetivo
            dist_xy = math.sqrt((target[0] - self.current_pos_ned[0])**2 + (target[1] - self.current_pos_ned[1])**2)
            if dist_xy > 0.5:
                self.target_yaw = math.atan2(target[1] - self.current_pos_ned[1], target[0] - self.current_pos_ned[0])

            # Diferenciar navegacion normal vs aterrizaje
            es_ultimo_wp = (self.current_wp == len(self.waypoints_ned) - 1)

            if es_ultimo_wp:
                # REGLA ESTRICTA (Stop & Drop)
                dist_z = abs(self.current_pos_ned[2] - target[2])
                
                if dist_xy < 0.15 and dist_z < 0.15:
                    self.get_logger().info("Centro perfecto a 30cm alcanzado. Iniciando Auto Land.")
                    self.nav_state = 3
            else:
                # REGLA FLUIDA (Aros)
                dist_3d = math.sqrt((self.current_pos_ned[0] - target[0])**2 + 
                                    (self.current_pos_ned[1] - target[1])**2 + 
                                    (self.current_pos_ned[2] - target[2])**2)
                if dist_3d < 0.3:
                    self.current_wp += 1
                    self.get_logger().info(f"=== Cruzando objetivo. Rumbo al WP {self.current_wp} ===")

        elif self.nav_state == 3: # ATERRIZAJE
            target = self.waypoints_ned[-1]
            self.get_logger().info("=== Ejecutando comando VEHICLE_CMD_NAV_LAND ===")
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.nav_state = 4
            
        elif self.nav_state == 4:
            target = self.waypoints_ned[-1]
            if self.current_pos_ned[2] > -0.05:
                self.get_logger().info("Contacto fisico. PX4 desarmara automaticamente.")
                self.nav_state = 5 

        # Enviar las ordenes calculadas
        if self.nav_state < 4:
            self.publish_trajectory_setpoint(target[0], target[1], target[2], self.target_yaw)

        # Secuencia de encendido inicial (a los 2 segundos)
        if self.setpoint_counter == 100:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0) 
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0) 
            self.nav_state = 1
            self.get_logger().info("Motores armados. Elevacion vertical iniciada.")

        if self.setpoint_counter < 101:
            self.setpoint_counter += 1

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = self.px4_timestamp
        self.offboard_ctrl_pub.publish(msg)

    def publish_trajectory_setpoint(self, x, y, z, yaw):
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.velocity = [float('nan'), float('nan'), float('nan')]
        msg.acceleration = [float('nan'), float('nan'), float('nan')]
        msg.jerk = [float('nan'), float('nan'), float('nan')]
        msg.yaw = float(yaw)
        msg.yawspeed = float('nan')
        msg.timestamp = self.px4_timestamp
        self.trajectory_pub.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
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

def main(args=None):
    rclpy.init(args=args)
    node = ExpertPilot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
