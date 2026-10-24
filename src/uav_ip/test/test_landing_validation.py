from pathlib import Path
from typing import Any, Dict, List
import math
import subprocess
import sys
import numpy as np
import pytest

from uav_ip.validate_episode import (
    LANDING_WINDOW_S,
    validate_episode_data,
)


def _make_flight_with_rate(
    rate_hz: float = 50.0,
    total_duration_s: float = 6.0,
    rest_start_s: float = 4.5,
) -> Dict[str, Any]:
    """
    Genera un episodio sintetico parametrizado por frecuencia de muestreo (`rate_hz`)
    sobre una duracion fisica fija (`total_duration_s`), con secuencia temporal completa
    de `VehicleLandDetected` a 5 Hz.
    """
    dt_s = 1.0 / float(rate_hz)
    num_samples = int(round(total_duration_s * rate_hz)) + 1
    base_unix_s = 1_700_000_000.0
    ts_s = base_unix_s + np.arange(num_samples, dtype=np.float64) * dt_s

    odom_list: List[Dict[str, Any]] = []
    s = math.sqrt(0.5)

    for t_val in ts_s:
        rel_t = float(t_val - base_unix_s)
        if rel_t < 0.5:
            frac = rel_t / 0.5
            x = 0.0
            z = 0.1 + 1.4 * frac
            vx, vz = 0.0, 2.8
        elif rel_t < 3.5:
            frac = (rel_t - 0.5) / 3.0
            x = 8.0 * frac
            z = 1.5
            vx, vz = 8.0 / 3.0, 0.0
        elif rel_t < rest_start_s:
            frac = (rel_t - 3.5) / max(1e-3, (rest_start_s - 3.5))
            x = 8.0
            z = 1.5 * (1.0 - frac) + 0.05 * frac
            vx, vz = 0.0, -0.5
        else:
            x = 8.0
            z = 0.05
            vx, vz = 0.0, 0.0

        t_us = int(round(t_val * 1e6))
        t_ns = int(round(t_val * 1e9))
        odom_list.append(
            {
                "timestamp": t_ns,
                "timestamp_us": t_us,
                "timestamp_sample_us": t_us,
                "position": [0.0, float(x), float(-z)],
                "q": [s, 0.0, 0.0, s],
                "velocity": [0.0, float(vx), float(-vz)],
                "pose_frame": 1,
                "velocity_frame": 1,
                "reset_counter": 0,
            }
        )

    # Secuencia temporal del detector cada 0.20 s (False en vuelo, True desde rest_start_s)
    land_list: List[Dict[str, Any]] = []
    ld_times = np.arange(0.0, total_duration_s + 1e-6, 0.20)
    for rel_t in ld_times:
        abs_t = base_unix_s + float(rel_t)
        is_landed = bool(rel_t >= rest_start_s - 1e-6)
        land_list.append(
            {
                "timestamp": int(round(abs_t * 1e9)),
                "timestamp_us": int(round(abs_t * 1e6)),
                "landed": is_landed,
                "ground_contact": is_landed,
                "maybe_landed": is_landed,
                "freefall": False,
                "in_ground_effect": is_landed,
                "at_rest": is_landed,
            }
        )

    waypoints_cfg = {
        "landmarks": [
            {"id": "gate_1", "type": "square_gate", "x": 5.0, "y": 0.0, "z": 1.5, "yaw": 0.0}
        ],
        "primitives": [{"type": "traverse", "target": "gate_1"}],
        "landing": {"x": 8.0, "y": 0.0, "z": 0.05},
    }
    return {
        "telemetry": {"odometry": odom_list, "local_pos": [], "land_detected": land_list},
        "waypoints_cfg": waypoints_cfg,
    }


def test_single_landed_true_message_does_not_prove_persistence():
    """Regresion C4: un unico mensaje `landed=True` al final no prueba persistencia."""
    fx = _make_flight_with_rate(rate_hz=50.0)
    last_ld = fx["telemetry"]["land_detected"][-1]
    fx["telemetry"]["land_detected"] = [last_ld]

    rep = validate_episode_data(fx["telemetry"], fx["waypoints_cfg"], pilot_exit_code=0)
    assert rep["validation_status"] == "rejected"
    assert rep["metrics"]["landed_detector_confirmed"] is False
    assert "px4_land_detector_not_confirmed" in rep["rejection_reasons"]


def test_two_stale_detector_messages_are_rejected():
    """Regresion C4: dos mensajes `landed=True` antiguos (> LANDING_MAX_DETECTOR_AGE_S) se rechazan."""
    fx = _make_flight_with_rate(rate_hz=50.0)
    t_end_ns = fx["telemetry"]["odometry"][-1]["timestamp"]
    # Dos mensajes landed=True pero emitidos 2.5 s y 2.0 s antes de t_end (> 1.5 s)
    t1_ns = t_end_ns - int(2.5 * 1e9)
    t2_ns = t_end_ns - int(2.0 * 1e9)
    fx["telemetry"]["land_detected"] = [
        {"timestamp": t1_ns, "timestamp_us": t1_ns // 1000, "landed": True, "ground_contact": True},
        {"timestamp": t2_ns, "timestamp_us": t2_ns // 1000, "landed": True, "ground_contact": True},
    ]

    rep = validate_episode_data(fx["telemetry"], fx["waypoints_cfg"], pilot_exit_code=0)
    assert rep["validation_status"] == "rejected"
    assert rep["metrics"]["landed_detector_confirmed"] is False
    assert any(r.startswith("stale_land_detector_message") for r in rep["rejection_reasons"])


def test_missing_or_zero_detector_timestamp_is_invalid():
    """Regresion C4: timestamp ausente o cero en `land_detected` es dato invalido, nunca edad cero."""
    fx = _make_flight_with_rate(rate_hz=50.0)
    fx["telemetry"]["land_detected"][-1]["timestamp"] = 0

    rep = validate_episode_data(fx["telemetry"], fx["waypoints_cfg"], pilot_exit_code=0)
    assert rep["validation_status"] == "rejected"
    assert rep["metrics"]["landed_detector_confirmed"] is False
    assert "invalid_land_detector_timestamp" in rep["rejection_reasons"]


def test_landed_false_inside_window_invalidates_persistence():
    """Regresion C4: un `landed=False` dentro de la ventana final [t_end - landing_window_s, t_end] invalida."""
    fx = _make_flight_with_rate(rate_hz=50.0)
    # Penultimo mensaje (en t_end - 0.20 s, dentro de la ventana de 0.50 s) cae a False
    fx["telemetry"]["land_detected"][-2]["landed"] = False

    rep = validate_episode_data(fx["telemetry"], fx["waypoints_cfg"], pilot_exit_code=0)
    assert rep["validation_status"] == "rejected"
    assert rep["metrics"]["landed_detector_confirmed"] is False
    assert "px4_land_detector_not_confirmed" in rep["rejection_reasons"]


def test_ground_contact_never_substitutes_for_landed():
    """Regresion C4: `ground_contact=True` con `landed=False` se rechaza siempre."""
    fx = _make_flight_with_rate(rate_hz=50.0)
    for msg in fx["telemetry"]["land_detected"]:
        msg["ground_contact"] = True
        msg["maybe_landed"] = True
        msg["landed"] = False

    rep = validate_episode_data(fx["telemetry"], fx["waypoints_cfg"], pilot_exit_code=0)
    assert rep["validation_status"] == "rejected"
    assert rep["metrics"]["landed_detector_confirmed"] is False
    assert "px4_land_detector_not_confirmed" in rep["rejection_reasons"]


def test_same_flight_at_50hz_and_125hz_uses_equivalent_duration():
    """
    Regresion C4: el mismo vuelo muestreado a 50 Hz y a 125 Hz evalua exactamente la misma
    duracion fisica (`landing_window_s = 0.50 s`), ajustando el numero de muestras de la ventana.
    """
    fx_50 = _make_flight_with_rate(rate_hz=50.0)
    fx_125 = _make_flight_with_rate(rate_hz=125.0)

    rep_50 = validate_episode_data(fx_50["telemetry"], fx_50["waypoints_cfg"], pilot_exit_code=0)
    rep_125 = validate_episode_data(fx_125["telemetry"], fx_125["waypoints_cfg"], pilot_exit_code=0)

    assert rep_50["validation_status"] == "accepted", rep_50["rejection_reasons"]
    assert rep_125["validation_status"] == "accepted", rep_125["rejection_reasons"]

    dur_50 = rep_50["metrics"]["landing_window_duration_s"]
    dur_125 = rep_125["metrics"]["landing_window_duration_s"]
    assert dur_50 == pytest.approx(LANDING_WINDOW_S, abs=0.025)
    assert dur_125 == pytest.approx(LANDING_WINDOW_S, abs=0.010)
    assert abs(dur_50 - dur_125) <= 0.02
    assert rep_125["metrics"]["landing_window_sample_count"] > rep_50["metrics"]["landing_window_sample_count"] * 2


def test_flyby_near_pad_followed_by_rest_outside_is_rejected():
    """
    Regresion C4:
      1. Pasar sobre la plataforma y terminar en reposo fuera de ella se rechaza (`landing_off_pad`).
      2. Estar fuera de la plataforma al principio de la ventana de 0.5 s y entrar solo en las
         ultimas muestras tambien se rechaza porque toda la ventana debe cumplir posicion y reposo.
    """
    # Caso 1: paso cercano al pad seguido de reposo afuera (x=9.2, a 1.2 m del centro en x=8.0)
    fx_out = _make_flight_with_rate(rate_hz=50.0)
    for item in fx_out["telemetry"]["odometry"][-30:]:
        item["position"][1] = 9.2  # ENU x = 9.2 m (pad en 8.0 m -> err = 1.2 m > 0.75 m)

    rep_out = validate_episode_data(fx_out["telemetry"], fx_out["waypoints_cfg"], pilot_exit_code=0)
    assert rep_out["validation_status"] == "rejected"
    assert any(r.startswith("landing_off_pad") for r in rep_out["rejection_reasons"])

    # Caso 2: las ultimas 10 muestras estan en el pad (x=8.0), pero al inicio de la ventana de 0.5 s
    # (muestra -20, es decir t_end - 0.40 s) estaba fuera del pad (x=9.1) y moviendose
    fx_late = _make_flight_with_rate(rate_hz=50.0)
    for item in fx_late["telemetry"]["odometry"][-25:-10]:
        item["position"][1] = 9.1
        item["velocity"][1] = 0.8

    rep_late = validate_episode_data(fx_late["telemetry"], fx_late["waypoints_cfg"], pilot_exit_code=0)
    assert rep_late["validation_status"] == "rejected"
    assert any(r.startswith("landing_off_pad") for r in rep_late["rejection_reasons"])
    assert any(r.startswith("landing_not_at_rest") for r in rep_late["rejection_reasons"])


def test_waypoint_controller_freshness_persistence_and_timeout_process_exit_code(tmp_path: Path):
    """
    Regresion C4 sobre `BasePilot` y `WaypointController`:
      - Verifica que `WaypointController` en `nav_state == 4` reinicia el contador de persistencia
        si `landed_detected` cae a `False` antes de cumplir `landing_persistence_s`.
      - Verifica mediante un subproceso real que un timeout de confirmacion de aterrizaje
        termina el proceso con codigo de retorno `1` y emite la marca de confirmacion.
    """
    wp_path = tmp_path / "waypoints.yaml"
    wp_path.write_text(
        "landmarks:\n"
        "  - {id: gate_1, type: square_gate, x: 4.0, y: 0.0, z: 1.5, yaw: 0.0}\n"
        "primitives:\n"
        "  - {type: traverse, target: gate_1}\n"
        "landing: {x: 8.0, y: 0.0, z: 0.05}\n",
        encoding="utf-8",
    )
    pkg_src = str(Path(__file__).resolve().parents[1])

    script = f"""
import sys
import types
sys.path.insert(0, {pkg_src!r})

try:
    import rclpy
    from rclpy.time import Time
    HAS_ROS = True
except ModuleNotFoundError:
    HAS_ROS = False
    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda args=None: None
    rclpy.ok = lambda: True
    rclpy.shutdown = lambda: None
    rclpy.spin = lambda node: None

    class _FakeTime:
        def __init__(self, nanoseconds=10_000_000_000, clock_type=None):
            self.nanoseconds = int(nanoseconds)
            self.clock_type = clock_type
        def __sub__(self, other):
            return _FakeTime(nanoseconds=self.nanoseconds - other.nanoseconds)

    Time = _FakeTime
    node_mod = types.ModuleType("rclpy.node")
    class _FakeLogger:
        def info(self, msg): pass
        def error(self, msg): pass
        def fatal(self, msg): pass
    class _FakeClock:
        def __init__(self): self._ns = 10_000_000_000
        def now(self): return _FakeTime(self._ns)
    class _FakeParamVal:
        def __init__(self, val):
            self.string_value = str(val) if isinstance(val, str) else ""
            self.double_value = float(val) if isinstance(val, (int, float)) else 0.0
    class _FakeParam:
        def __init__(self, val): self._v = _FakeParamVal(val)
        def get_parameter_value(self): return self._v
    class Node:
        def __init__(self, name):
            self._params = {{"waypoints_file": {str(wp_path)!r}, "landing_persistence_s": 3.0, "landing_timeout_s": 1.0}}
            self._clock = _FakeClock()
            self._logger = _FakeLogger()
        def declare_parameter(self, name, default):
            if name not in self._params or not self._params[name]:
                self._params[name] = default
        def get_parameter(self, name): return _FakeParam(self._params[name])
        def create_publisher(self, *a, **kw):
            class _Pub:
                def publish(self, m): pass
            return _Pub()
        def create_subscription(self, *a, **kw): return None
        def create_timer(self, *a, **kw): return None
        def get_clock(self): return self._clock
        def get_logger(self): return self._logger
        def destroy_node(self): pass
    node_mod.Node = Node
    qos_mod = types.ModuleType("rclpy.qos")
    for k in ("QoSProfile", "ReliabilityPolicy", "HistoryPolicy", "DurabilityPolicy"):
        setattr(qos_mod, k, type(k, (), {{"BEST_EFFORT": 1, "RELIABLE": 2, "KEEP_LAST": 1, "TRANSIENT_LOCAL": 1, "VOLATILE": 2, "__init__": lambda self, **kw: None}}))
    sys.modules["rclpy"] = rclpy
    sys.modules["rclpy.node"] = node_mod
    sys.modules["rclpy.qos"] = qos_mod

    px4_mod = types.ModuleType("px4_msgs")
    px4_msg_mod = types.ModuleType("px4_msgs.msg")
    class _DummyMsg:
        VEHICLE_CMD_DO_SET_MODE = 176
        VEHICLE_CMD_COMPONENT_ARM_DISARM = 400
        VEHICLE_CMD_NAV_LAND = 21
        NAVIGATION_STATE_OFFBOARD = 14
        ARMING_STATE_ARMED = 2
        def __init__(self):
            self.position = [0.0, 0.0, 0.0]
            self.velocity = [0.0, 0.0, 0.0]
    for mname in ("OffboardControlMode", "TrajectorySetpoint", "VehicleCommand", "VehicleLocalPosition", "VehicleStatus", "VehicleLandDetected"):
        setattr(px4_msg_mod, mname, _DummyMsg)
    sys.modules["px4_msgs"] = px4_mod
    sys.modules["px4_msgs.msg"] = px4_msg_mod

from uav_ip.waypoint_controller import WaypointController

if HAS_ROS:
    rclpy.init(args=["--ros-args", "-p", f"waypoints_file:={str(wp_path)}"])
node = WaypointController()
node.px4_timestamp = 1000
node.waypoints_ned = [{{"pos": [8.0, 0.0, 0.05], "yaw": 0.0, "type": "landing_touchdown"}}]
node.current_pos_ned = [0.0, 8.0, -0.05]  # ENU (8.0, 0.0, 0.05)
node.nav_state = 4
node.landing_persistence_s = 3.0
node.landing_timeout_s = 1.0

# 1. Comprobar que una caida transitoria de landed_detected reinicia landed_persistence_start_time
now = node.get_clock().now()
node.last_pos_recv_time = now
node.last_land_detected_recv_time = now
node.landed_detected = True
node._control_loop()
assert node.landed_persistence_start_time is not None, "Debe iniciar contador cuando condiciones se cumplen"

# Si landed_detected cae a False, el contador se reinicia a None
node.landed_detected = False
node._control_loop()
assert node.landed_persistence_start_time is None, "Debe reiniciar contador si landed_detected cae a False"

# 2. Forzar que land_start_time haya superado landing_timeout_s sin persistencia completa
node.land_start_time = Time(nanoseconds=max(1, now.nanoseconds - int(2.0 * 1e9)), clock_type=getattr(now, "clock_type", None))
try:
    node._control_loop()  # Pasa a nav_state = 5 con exit_code = 1
    node._control_loop()  # Lanza SystemExit(1)
except SystemExit as e:
    print(f"LANDING_TIMEOUT_CONFIRMED_EXIT_{{int(e.code)}}", flush=True)
    if HAS_ROS and rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(int(e.code))
sys.exit(0)
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 1, f"Se esperaba codigo 1 en timeout de aterrizaje, stderr={proc.stderr}"
    assert "LANDING_TIMEOUT_CONFIRMED_EXIT_1" in proc.stdout

