#!/usr/bin/env python3
import math
from pathlib import Path
from typing import Optional
import yaml
import rclpy
from px4_msgs.msg import VehicleCommand
from uav_ip.base_pilot import BasePilot


class WaypointController(BasePilot):
    """
    WaypointController implements smooth Feed-Forward trajectory control for primitives
    (traverse, slalom left/right, overfly, and high-resolution centripetal orbit).

    Elimina las detenciones bruscas en puntos intermedios ('pre', 'center', 'orbit')
    combinando discretización suave de arcos con términos de velocidad feed-forward
    tangencial hacia el siguiente waypoint.
    """
    APPROACH_OFFSET = 1.2
    TAKEOFF_HEIGHT = 1.5
    HOVER_HEIGHT = 1.5
    TOUCHDOWN_HEIGHT = 0.3

    # Velocidades de paso feed-forward (m/s) para evitar frenazos a mitad de maniobra
    TRANSIT_SPEED_FF = 1.4
    ORBIT_SPEED_FF = 1.2
    ORBIT_ARC_STEPS = 12

    def __init__(self):
        super().__init__("waypoint_controller")

        self.declare_parameter("waypoints_file", "")
        wp_file = self.get_parameter("waypoints_file").get_parameter_value().string_value
        if not wp_file:
            self.get_logger().fatal("Parameter 'waypoints_file' not provided.")
            raise SystemExit(1)

        wp_path = Path(wp_file)
        if not wp_path.is_file():
            self.get_logger().fatal(f"Waypoint file not found: {wp_file}")
            raise SystemExit(1)

        self.waypoints_gazebo = self._build_waypoints(wp_path)
        self.waypoints_ned = self.waypoints_gazebo
        self.current_wp = 0
        self.target_yaw = math.pi / 2.0

        self.timer = self.create_timer(0.02, self._control_loop)

    def _build_waypoints(self, yaml_path: Path) -> list:
        with open(yaml_path, "r") as fh:
            cfg = yaml.safe_load(fh)

        OFFSET = self.APPROACH_OFFSET
        wps = []

        # Pose inicial aérea post-despegue (admite aleatorización suave por episodio)
        start_cfg = cfg.get("start_pose") or {}
        start_x = float(start_cfg.get("x", 0.0))
        start_y = float(start_cfg.get("y", 0.0))
        start_z = float(start_cfg.get("z", self.TAKEOFF_HEIGHT))
        start_yaw = float(start_cfg.get("yaw", 0.0))

        # Paso 0: ascenso vertical limpio desde (0, 0) hasta la altitud de salida
        wps.append({
            "pos": [0.0, 0.0, start_z],
            "type": "takeoff",
            "yaw": start_yaw,
            "vel_ff": None,
        })

        # Si start_pose tiene desplazamiento lateral Y o X respecto al origen (0,0),
        # colocarse en start_pose antes de iniciar el avance por el circuito
        if math.hypot(start_x, start_y) > 0.05:
            wps.append({
                "pos": [start_x, start_y, start_z],
                "type": "start_pose",
                "yaw": start_yaw,
                "vel_ff": None,
            })

        landmarks_list = cfg.get("landmarks") or []
        if not landmarks_list and "hoops" in cfg:
            landmarks_list = []
            for h in cfg["hoops"]:
                h_copy = dict(h)
                h_copy["id"] = h_copy.get("name", "hoop")
                h_copy["type"] = "gate"
                landmarks_list.append(h_copy)

        landmarks_map = {lm["id"]: lm for lm in landmarks_list}

        primitives = cfg.get("primitives", [])
        if not primitives and landmarks_list:
            sorted_lms = sorted(landmarks_list, key=lambda l: float(l.get("x", 0.0)))
            primitives = [{"type": "traverse", "target": lm["id"]} for lm in sorted_lms]

        def get_prim_x(prim):
            tid = prim.get("target")
            if tid in landmarks_map:
                return float(landmarks_map[tid]["x"])
            return 0.0

        primitives = sorted(primitives, key=get_prim_x)

        for prim in primitives:
            target_id = prim.get("target")
            if target_id not in landmarks_map:
                continue

            lm = landmarks_map[target_id]
            cx, cy, cz = float(lm["x"]), float(lm["y"]), float(lm["z"])
            yaw = float(lm.get("yaw", 0.0))
            fwd_x = math.cos(yaw)
            fwd_y = math.sin(yaw)

            # Vector de velocidad feed-forward en marco ENU a lo largo de la normal del landmark
            v_pass_enu = [
                self.TRANSIT_SPEED_FF * fwd_x,
                self.TRANSIT_SPEED_FF * fwd_y,
                0.0,
            ]

            p_type = prim["type"]
            if p_type == "traverse":
                wps.append({
                    "pos": [cx - OFFSET * fwd_x, cy - OFFSET * fwd_y, cz],
                    "type": "pre",
                    "primitive": p_type,
                    "target": target_id,
                    "yaw": yaw,
                    "vel_ff": None,
                })
                wps.append({
                    "pos": [cx, cy, cz],
                    "type": "center",
                    "primitive": p_type,
                    "target": target_id,
                    "yaw": yaw,
                    "vel_ff": v_pass_enu,
                })
                wps.append({
                    "pos": [cx + OFFSET * fwd_x, cy + OFFSET * fwd_y, cz],
                    "type": "post",
                    "primitive": p_type,
                    "target": target_id,
                    "yaw": yaw,
                    "vel_ff": None,
                })
            elif p_type == "overfly":
                OVER_Z = 2.5
                wps.append({
                    "pos": [cx - OFFSET * fwd_x, cy - OFFSET * fwd_y, cz + OVER_Z],
                    "type": "pre",
                    "primitive": p_type,
                    "target": target_id,
                    "yaw": yaw,
                    "vel_ff": None,
                })
                wps.append({
                    "pos": [cx, cy, cz + OVER_Z],
                    "type": "center",
                    "primitive": p_type,
                    "target": target_id,
                    "yaw": yaw,
                    "vel_ff": v_pass_enu,
                })
                wps.append({
                    "pos": [cx + OFFSET * fwd_x, cy + OFFSET * fwd_y, cz + OVER_Z],
                    "type": "post",
                    "primitive": p_type,
                    "target": target_id,
                    "yaw": yaw,
                    "vel_ff": None,
                })
            elif p_type == "slalom":
                side = str(prim.get("side", "right")).lower()
                sign = 1.0 if side == "right" else -1.0
                right_x = math.sin(yaw) * sign
                right_y = -math.cos(yaw) * sign
                LAT_OFFSET = 1.5
                wps.append({
                    "pos": [cx - OFFSET * fwd_x + right_x * LAT_OFFSET, cy - OFFSET * fwd_y + right_y * LAT_OFFSET, cz],
                    "type": "pre",
                    "primitive": p_type,
                    "target": target_id,
                    "yaw": yaw,
                    "vel_ff": None,
                })
                wps.append({
                    "pos": [cx + right_x * LAT_OFFSET, cy + right_y * LAT_OFFSET, cz],
                    "type": "center",
                    "primitive": p_type,
                    "target": target_id,
                    "yaw": yaw,
                    "vel_ff": v_pass_enu,
                })
                wps.append({
                    "pos": [cx + OFFSET * fwd_x + right_x * LAT_OFFSET, cy + OFFSET * fwd_y + right_y * LAT_OFFSET, cz],
                    "type": "post",
                    "primitive": p_type,
                    "target": target_id,
                    "yaw": yaw,
                    "vel_ff": None,
                })
            elif p_type == "orbit":
                R = 1.5
                right_x = math.sin(yaw)
                right_y = -math.cos(yaw)
                # Discretización suave del arco de 270° (de ángulo local pi a -pi/2 pasando por -pi/2, 0, +pi/2)
                # En la base local (fwd, right):
                #   s = 0.0 -> (-R*fwd)              [frente al poste, mirando a +fwd]
                #   s = 1/3 -> (+R*right)            [flanco derecho, mirando a -right]
                #   s = 2/3 -> (+R*fwd)              [detrás del poste, mirando a -fwd]
                #   s = 1.0 -> (-R*right)            [flanco izquierdo, mirando a +right]
                num_steps = self.ORBIT_ARC_STEPS
                total_sweep = 1.5 * math.pi  # 270 grados
                for k in range(num_steps + 1):
                    alpha = (k / float(num_steps)) * total_sweep
                    # Vector radial desde el centro del poste:
                    # k=0 (alpha=0) -> -fwd
                    # alpha=pi/2    -> +right
                    # alpha=pi      -> +fwd
                    # alpha=3pi/2   -> -right
                    rad_fwd = -math.cos(alpha)
                    rad_right = math.sin(alpha)
                    px = cx + R * (rad_fwd * fwd_x + rad_right * right_x)
                    py = cy + R * (rad_fwd * fwd_y + rad_right * right_y)

                    # Tangente unitaria de avance a lo largo de la órbita (derivada respecto a alpha)
                    tan_fwd = math.sin(alpha)
                    tan_right = math.cos(alpha)
                    tx = tan_fwd * fwd_x + tan_right * right_x
                    ty = tan_fwd * fwd_y + tan_right * right_y

                    # Yaw centrípeto mirando siempre hacia el eje central del poste (cx, cy)
                    centripetal_yaw = math.atan2(cy - py, cx - px)

                    wp_kind = "pre" if k == 0 else "orbit_arc"
                    # El punto de entrada pre no lleva feed-forward para posicionamiento exacto
                    vel_ff_enu = None if k == 0 else [
                        self.ORBIT_SPEED_FF * tx,
                        self.ORBIT_SPEED_FF * ty,
                        0.0,
                    ]
                    wps.append({
                        "pos": [px, py, cz],
                        "type": wp_kind,
                        "primitive": p_type,
                        "target": target_id,
                        "yaw": centripetal_yaw,
                        "vel_ff": vel_ff_enu,
                    })

                # Salida tangencial progresiva hacia +X reanudando orientación del pasillo
                wps.append({
                    "pos": [cx + R * fwd_x - right_x * R, cy + R * fwd_y - right_y * R, cz],
                    "type": "post",
                    "primitive": p_type,
                    "target": target_id,
                    "yaw": yaw,
                    "vel_ff": None,
                })

        lx, ly = float(cfg["landing"]["x"]), float(cfg["landing"]["y"])
        wps.append({"pos": [lx, ly, self.HOVER_HEIGHT], "type": "landing_hover", "yaw": 0.0, "vel_ff": None})
        wps.append({"pos": [lx, ly, self.TOUCHDOWN_HEIGHT], "type": "landing_touchdown", "yaw": 0.0, "vel_ff": None})
        return wps

    def _control_loop(self):
        if self.px4_timestamp == 0:
            return

        target = list(self.current_pos_ned)
        vel_ff_ned: Optional[list[float]] = None

        if self.nav_state == 0:
            pass
        elif self.nav_state == 1:
            wp_info = self.waypoints_ned[0]
            target = self.gazebo_to_ned(wp_info["pos"])
            self.target_yaw = self.yaw_gazebo_to_ned(wp_info["yaw"])

            if abs(self.current_pos_ned[2] - target[2]) < 0.2:
                self.current_wp = 1
                self.nav_state = 2
                next_wp_type = self.waypoints_ned[1]["type"] if len(self.waypoints_ned) > 1 else "FINAL"
                self.get_logger().info(
                    f"Despegue completado. Iniciando circuito hacia waypoint [1] ({next_wp_type})..."
                )
        elif self.nav_state == 2:
            wp_info = self.waypoints_ned[self.current_wp]
            target = self.gazebo_to_ned(wp_info["pos"])
            wp_type = wp_info["type"]
            wp_prim = wp_info.get("primitive", "traverse")
            wp_yaw = wp_info["yaw"]

            if wp_info.get("vel_ff") is not None:
                vel_ff_ned = self.gazebo_to_ned(wp_info["vel_ff"])

            curr_enu = self.ned_to_gazebo(self.current_pos_ned)
            wp_pos = wp_info["pos"]
            dx_enu = curr_enu[0] - wp_pos[0]
            dy_enu = curr_enu[1] - wp_pos[1]
            dz_enu = curr_enu[2] - wp_pos[2]
            dist_xy = math.hypot(dx_enu, dy_enu)
            dist_3d = math.hypot(dx_enu, dy_enu, dz_enu)

            # Proyección geométrica sobre el rumbo local del waypoint
            fwd_x = math.cos(wp_yaw)
            fwd_y = math.sin(wp_yaw)
            along_track = dx_enu * fwd_x + dy_enu * fwd_y
            cross_track = abs(-dx_enu * fwd_y + dy_enu * fwd_x)
            alt_err = abs(dz_enu)

            self.target_yaw = self.yaw_gazebo_to_ned(wp_yaw)

            wp_reached = False
            if wp_type == "landing_touchdown":
                if dist_xy < 0.25 and alt_err < 0.25:
                    self.get_logger().info("Aterrizaje alcanzado. Comandando AUTO_LAND...")
                    self.nav_state = 3
            elif wp_type == "landing_hover":
                wp_reached = (dist_xy < 0.40 and alt_err < 0.40)
            elif wp_type == "start_pose":
                wp_reached = (dist_3d < 0.50 or (along_track > 0.0 and cross_track < 0.65 and alt_err < 0.50))
            elif wp_type == "pre":
                if wp_prim == "traverse":
                    wp_reached = (dist_3d < 0.50 or (along_track > -0.15 and cross_track < 0.42 and alt_err < 0.42))
                else:
                    wp_reached = (dist_3d < 0.60 or (along_track > -0.15 and cross_track < 0.60 and alt_err < 0.50))
            elif wp_type == "center":
                # Para traverse (ventana libre +-0.75m, radio UAV 0.28m -> ventana segura +-0.47m),
                # exigir cross_track < 0.38m y alt_err < 0.42m impide falsos cruces por fuera del marco.
                if wp_prim == "traverse":
                    wp_reached = ((dist_3d < 0.45 or along_track >= -0.05) and cross_track < 0.38 and alt_err < 0.42)
                else:
                    wp_reached = ((dist_3d < 0.65 or along_track >= -0.08) and cross_track < 0.65 and alt_err < 0.55)
            elif wp_type == "orbit_arc":
                wp_reached = (dist_3d < 0.60 and alt_err < 0.50)
            elif wp_type == "post":
                if wp_prim == "traverse":
                    wp_reached = (dist_3d < 0.65 or (along_track >= -0.05 and cross_track < 0.55 and alt_err < 0.55))
                else:
                    wp_reached = (dist_3d < 0.75 or (along_track >= -0.05 and cross_track < 0.75 and alt_err < 0.60))

            if wp_reached:
                self.current_wp += 1
                next_wp_type = (
                    self.waypoints_ned[self.current_wp]["type"]
                    if self.current_wp < len(self.waypoints_ned)
                    else "FINAL"
                )
                self.get_logger().info(
                    f"Waypoint alcanzado: [{self.current_wp - 1}/{len(self.waypoints_ned) - 1}] ({wp_type}/{wp_prim}). "
                    f"Avanzando a [{self.current_wp}] ({next_wp_type})."
                )
        elif self.nav_state == 3:
            target = self.gazebo_to_ned(self.waypoints_ned[-1]["pos"])
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.nav_state = 4
        elif self.nav_state == 4:
            target = self.gazebo_to_ned(self.waypoints_ned[-1]["pos"])
            if not hasattr(self, "land_start_time"):
                self.land_start_time = self.get_clock().now()
            elapsed = (self.get_clock().now() - self.land_start_time).nanoseconds / 1e9
            curr_enu = self.ned_to_gazebo(self.current_pos_ned)
            landed_ok = (self.landed_detected and elapsed >= 3.0) or (curr_enu[2] < 0.22 and elapsed >= 4.0)
            if landed_ok:
                self.get_logger().info(
                    f"Aterrizaje confirmado (landed_detected={self.landed_detected}, z={curr_enu[2]:.2f}m, t={elapsed:.1f}s)."
                )
                self.nav_state = 5
            elif elapsed >= 10.0:
                self.get_logger().warning(
                    f"Timeout de confirmación de aterrizaje ({elapsed:.1f}s, z={curr_enu[2]:.2f}m). Finalizando controlador."
                )
                self.nav_state = 5
        elif self.nav_state == 5:
            raise SystemExit(0)

        self.publish_offboard_mode(use_velocity=(vel_ff_ned is not None))
        if self.nav_state < 4:
            self.publish_trajectory_setpoint(
                target[0], target[1], target[2], self.target_yaw, velocity=vel_ff_ned
            )

        self.handle_arming()


def main(args=None):
    rclpy.init(args=args)
    node = WaypointController()
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
