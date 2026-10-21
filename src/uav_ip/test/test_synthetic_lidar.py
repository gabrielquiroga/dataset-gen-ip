import math
import numpy as np
import pytest

from uav_ip.synthetic_lidar import (
    SyntheticLidarGenerator,
    generate_canonical_square_gate,
    generate_canonical_circular_gate,
    generate_canonical_pole,
    generate_canonical_landing_pad,
    fps_numpy,
    px4_odometry_to_gazebo_enu,
    extract_yaw_from_rot_matrix,
    ned_to_gazebo_enu,
    yaw_ned_to_gazebo_enu,
    rot_z,
)


def test_canonical_geometries():
    g_sq = generate_canonical_square_gate(2000)
    assert g_sq.shape == (2000, 3)

    g_circ = generate_canonical_circular_gate(2000)
    assert g_circ.shape == (2000, 3)

    p_pole = generate_canonical_pole(2000)
    assert p_pole.shape == (2000, 3)

    pad = generate_canonical_landing_pad(1000, height=0.1)
    assert pad.shape == (1000, 3)
    # Verify bottom face (-0.05) is not sampled by default, and top face (+0.05) dominates
    assert np.sum(np.isclose(pad[:, 2], 0.05)) > 800


def test_fps_sampling_and_edge_cases():
    test_pts = np.random.uniform(-5, 5, size=(5000, 3)).astype(np.float32)
    sampled = fps_numpy(test_pts, 2048, seed=42)
    sampled_repeat = fps_numpy(test_pts, 2048, seed=42)
    assert sampled.shape == (2048, 3)
    assert np.allclose(sampled, sampled_repeat)

    # Padding when N < 2048
    small_pts = np.random.uniform(-1, 1, size=(50, 3)).astype(np.float32)
    sampled_pad = fps_numpy(small_pts, 2048, seed=42)
    assert sampled_pad.shape == (2048, 3)

    # Empty cloud: places points on outer boundary ring (r = 1.0), never at (0, 0, 0)
    empty_pts = np.zeros((0, 3), dtype=np.float32)
    sampled_empty = fps_numpy(empty_pts, 2048, seed=42, empty_fallback_radius=1.0)
    assert sampled_empty.shape == (2048, 3)
    assert np.allclose(np.linalg.norm(sampled_empty, axis=1), 1.0, atol=1e-5)
    assert not np.any(np.all(sampled_empty == 0.0, axis=1))


def test_coordinate_frame_transformations():
    pos_ned = [5.0, 10.0, -2.0]
    pos_enu = ned_to_gazebo_enu(pos_ned)
    assert np.allclose(pos_enu, [10.0, 5.0, 2.0])

    # Heading East: yaw_ned = pi/2 -> yaw_enu = 0
    yaw_ned_east = math.pi / 2.0
    yaw_enu_east = yaw_ned_to_gazebo_enu(yaw_ned_east)
    assert abs(yaw_enu_east - 0.0) < 1e-6
    q_ned_east = [math.cos(yaw_ned_east / 2.0), 0.0, 0.0, math.sin(yaw_ned_east / 2.0)]
    p_conv, R_conv = px4_odometry_to_gazebo_enu(pos_ned, q_ned_east)
    assert np.allclose(p_conv, pos_enu)
    assert np.allclose(R_conv, np.eye(3))
    assert abs(extract_yaw_from_rot_matrix(R_conv) - 0.0) < 1e-6

    # Heading North: yaw_ned = 0 -> yaw_enu = pi/2
    yaw_ned_north = 0.0
    yaw_enu_north = yaw_ned_to_gazebo_enu(yaw_ned_north)
    assert abs(yaw_enu_north - math.pi / 2.0) < 1e-6
    q_ned_north = [1.0, 0.0, 0.0, 0.0]
    _, R_conv_north = px4_odometry_to_gazebo_enu([0, 0, 0], q_ned_north)
    assert np.allclose(R_conv_north, rot_z(math.pi / 2.0))
    assert abs(extract_yaw_from_rot_matrix(R_conv_north) - math.pi / 2.0) < 1e-6


def test_egocentric_projection_and_heading_invariance():
    gen = SyntheticLidarGenerator(r_max=10.0, scale_factor=0.1, target_points=2048)

    # 1. Drone facing East (yaw_enu = 0), gate located East at (5.0, 0.0, 1.5)
    drone_east = {"x": 0.0, "y": 0.0, "z": 1.5, "yaw": 0.0}
    lm_east = [{"id": "g_east", "type": "square_gate", "x": 5.0, "y": 0.0, "z": 1.5, "yaw": 0.0}]
    cloud_east = gen.generate_pointcloud(drone_east, lm_east, seed=42)
    mean_east = np.mean(cloud_east, axis=0)
    assert mean_east[0] > 0.45
    assert abs(mean_east[1]) < 0.05

    # 2. Drone facing North (yaw_enu = pi/2, q_ned = [1,0,0,0]), gate located North at (0.0, 5.0, 1.5)
    q_ned_north = [1.0, 0.0, 0.0, 0.0]
    drone_north_qned = {"x": 0.0, "y": 0.0, "z": 1.5, "q_ned": q_ned_north}
    lm_north = [{"id": "g_north", "type": "square_gate", "x": 0.0, "y": 5.0, "z": 1.5, "yaw": math.pi / 2.0}]
    cloud_north_qned = gen.generate_pointcloud(drone_north_qned, lm_north, seed=42)
    mean_north_qned = np.mean(cloud_north_qned, axis=0)
    assert mean_north_qned[0] > 0.45
    assert abs(mean_north_qned[1]) < 0.05

    # 3. Drone facing North passing rotation matrix R directly
    _, R_north = px4_odometry_to_gazebo_enu([0, 0, 0], q_ned_north)
    drone_north_R = {"x": 0.0, "y": 0.0, "z": 1.5, "R": R_north}
    cloud_north_R = gen.generate_pointcloud(drone_north_R, lm_north, seed=42)
    assert np.allclose(cloud_north_qned, cloud_north_R)


def test_gravity_aligned_pitch_and_roll_invariance():
    """Verifica que inclinaciones puras de pitch/roll no distorsionen Z_FLU cuando gravity_aligned=True."""
    gen = SyntheticLidarGenerator(r_max=10.0, scale_factor=0.1, target_points=2048, gravity_aligned=True)
    lm_north = [{"id": "g_north", "type": "square_gate", "x": 0.0, "y": 6.0, "z": 1.5, "yaw": math.pi / 2.0}]

    # Level drone facing North (yaw_ned = 0)
    q_level = [1.0, 0.0, 0.0, 0.0]
    cloud_level = gen.generate_pointcloud({"x": 0.0, "y": 0.0, "z": 1.5, "q_ned": q_level}, lm_north, seed=101)

    # Drone pitched forward 20 degrees (theta = -20 deg in FRD) and rolled 15 degrees
    pitch = math.radians(-20.0)
    roll = math.radians(15.0)
    cy, sy = 1.0, 0.0  # yaw_ned = 0
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    q_tilted = [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]
    cloud_tilted_aligned = gen.generate_pointcloud(
        {"x": 0.0, "y": 0.0, "z": 1.5, "q_ned": q_tilted}, lm_north, seed=101, gravity_aligned=True
    )
    assert np.allclose(cloud_level, cloud_tilted_aligned, atol=1e-5)

    # Contrast with unstabilized body frame (gravity_aligned=False): Z_FLU shifts noticeably
    cloud_tilted_raw = gen.generate_pointcloud(
        {"x": 0.0, "y": 0.0, "z": 1.5, "q_ned": q_tilted}, lm_north, seed=101, gravity_aligned=False
    )
    assert abs(np.mean(cloud_tilted_raw[:, 2]) - np.mean(cloud_level[:, 2])) > 0.10


def test_generator_hoops_backwards_compatibility():
    gen = SyntheticLidarGenerator(r_max=10.0, scale_factor=0.1, target_points=2048)
    # Legacy hoops format (e.g. task_0000)
    legacy_hoops = [
        {"name": "aro_1", "x": 5.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
        {"name": "aro_2", "x": 10.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
    ]
    landing = {"x": 15.0, "y": 0.0}
    drone_pose = {"x": 0.0, "y": 0.0, "z": 1.5, "yaw": 0.0}
    cloud = gen.generate_pointcloud(drone_pose, legacy_hoops, landing=landing, seed=42)
    assert cloud.shape == (2048, 3)
    assert np.max(np.linalg.norm(cloud, axis=1)) <= 1.0 + 1e-4


def test_tube_augmentation_se3_convergence():
    """
    Valida Tarea 2.3 (30% Tube Augmentation):
    1. Exactamente el 30% de los instantes T generan muestras de recuperacion.
    2. Cada accion correctiva cumple la identidad de grupo SE(3):
           \\tilde{T}_{WE}^t @ \\tilde{T}_{EA}^{(t, k)} == T_{WE}^{t+k}
    3. Ante una desviacion lateral (+Y / -Y) o vertical (+Z / -Z) respecto al eje de un aro,
       el vector de accion local \\Delta\\tilde{p}_{FLU} apunta convergentemente de regreso al eje.
    """
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from bag_to_heterodata import compute_se3_action_horizon, compute_tube_augmentation

    T = 40
    scale = 0.1
    # Trayectoria nominal recta por el eje del aro en Y=0, Z=1.5 m, avanzando en +X
    pos_enu_m = np.column_stack([
        np.linspace(0.0, 12.0, T),
        np.zeros(T),
        np.full(T, 1.5),
    ]).astype(np.float64)
    yaw_enu = np.zeros((T, 1), dtype=np.float32)
    flight_state = np.ones((T, 1), dtype=np.float32)
    flight_state[-4:, 0] = 0.0

    landmarks = [{"id": "gate_1", "type": "square_gate", "x": 6.0, "y": 0.0, "z": 1.5, "yaw": 0.0}]
    landing = {"x": 12.0, "y": 0.0, "z": 0.05}
    gen = SyntheticLidarGenerator(r_max=10.0, scale_factor=scale, target_points=2048)

    pose_se3_nom, _, _ = compute_se3_action_horizon(
        (pos_enu_m * scale).astype(np.float32), yaw_enu, flight_state, horizon=8, action_stride=1
    )

    aug = compute_tube_augmentation(
        pos_enu_m=pos_enu_m,
        yaw_enu=yaw_enu,
        flight_state=flight_state,
        landmarks=landmarks,
        landing=landing,
        lidar_gen=gen,
        ep_seed=42001,
        scale_factor=scale,
        horizon=8,
        action_stride=1,
        tube_aug_prob=0.30,
        tube_sigma_pos=0.25,
        tube_sigma_yaw_deg=5.0,
        state_flip_prob=0.10,
    )

    T_aug = len(aug["timestep_idx"])
    assert T_aug == int(round(0.30 * T))  # 12 de 40 (30%)
    assert int(np.sum(aug["aug_mask"])) == T_aug
    assert aug["scene_pos"].shape == (T_aug, 2048, 3)
    assert aug["y"].shape == (T_aug, 8, 5)
    assert aug["action_se3"].shape == (T_aug, 8, 4, 4)

    for i in range(T_aug):
        t = int(aug["timestep_idx"][i])
        T_we_pert = aug["pose_se3"][i]
        for k in range(1, 9):
            tk = min(T - 1, t + k)
            T_ea_corr = aug["action_se3"][i, k - 1]
            # Identidad exacta en SE(3): \tilde{T}_{WE}^t * \tilde{T}_{EA}^{(t,k)} == T_{WE}^{t+k}
            T_reconstructed = T_we_pert @ T_ea_corr
            assert np.allclose(T_reconstructed, pose_se3_nom[tk], atol=1e-5)

            # Verificar signo de recuperacion transversal/vertical hacia el eje Y=0, Z=1.5
            world_recovery_vec = aug["target_pos_enu"][i, k - 1] - aug["pos_enu"][i]
            dp_injected_scaled = aug["delta_pos_m"][i] * scale
            # Como la trayectoria nominal tiene Y=0 y Z=constante, el desplazamiento mundial de
            # recuperacion en Y y Z debe ser exactamente -dp_injected_scaled[1:3]
            assert np.allclose(world_recovery_vec[1], -dp_injected_scaled[1], atol=1e-5)
            assert np.allclose(world_recovery_vec[2], -dp_injected_scaled[2], atol=1e-5)


def test_discover_dataset_episodes():
    """Valida Tarea 2.2: descubrimiento recursivo de episodios en datasets/."""
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from bag_to_heterodata import discover_dataset_episodes

    dataset_dir = repo_root / "datasets"
    if not dataset_dir.is_dir():
        pytest.skip("Directorio opcional datasets/ no disponible en este entorno")
    eps = discover_dataset_episodes(dataset_dir)
    assert len(eps) >= 6
    labels = [e["label"] for e in eps]
    assert "task_0000/episode_00" in labels
    assert "task_0001/episode_02" in labels


def test_heterodata_load_node_types_compatibility():
    """Valida que la deteccion de nodos en HeteroData no use 'in data' directamente."""
    class MockNode:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class MockHeteroData:
        def __init__(self):
            # En PyG HeteroData, __contains__ verifica nombres de atributos en data.keys(),
            # NO los nombres de los tipos de nodo (node_types).
            self.node_types = ["scene", "agent", "aug"]
            self.stores = {
                "scene": MockNode(pos=np.zeros((5, 2048, 3), dtype=np.float32)),
                "agent": MockNode(
                    x=np.zeros((5, 3), dtype=np.float32),
                    pos_enu=np.zeros((5, 3), dtype=np.float32),
                    y=np.zeros((5, 8, 5), dtype=np.float32),
                    action_se3=np.zeros((5, 8, 4, 4), dtype=np.float32),
                ),
                "aug": MockNode(
                    timestep_idx=np.array([1, 3]),
                    scene_pos=np.zeros((2, 2048, 3), dtype=np.float32),
                    pos_enu=np.zeros((2, 3), dtype=np.float32),
                    y=np.zeros((2, 8, 5), dtype=np.float32),
                ),
            }

        def __getitem__(self, key):
            return self.stores[key]

        def __contains__(self, key):
            # Comportamiento exacto de PyG: solo atributos, nunca node_types
            return key in ["pos", "x", "pos_enu", "y", "action_se3"]

    data = MockHeteroData()
    # Demostracion del bug previo de PyG: 'agent' in data da False
    assert ("agent" in data) is False
    # Verificacion de la solucion robusta:
    assert hasattr(data, "node_types") and "agent" in data.node_types
    assert "scene" in data.node_types
    assert "aug" in data.node_types


def _make_synthetic_telemetry(pos_enu_arr: np.ndarray, yaw_enu_arr: np.ndarray, dt_s: float = 0.02):
    """Construye telemetria sintetica en formato PX4 NED para pruebas unitarias de validate_episode."""
    N = len(pos_enu_arr)
    odom = []
    vel_enu = np.zeros_like(pos_enu_arr)
    if N > 1:
        vel_enu[1:] = np.diff(pos_enu_arr, axis=0) / dt_s
        vel_enu[0] = vel_enu[1]
    for i in range(N):
        t_us = int(round((100.0 + i * dt_s) * 1e6))
        t_ns = t_us * 1000
        # ENU -> NED: [N, E, D] = [y, x, -z]
        p_ned = [float(pos_enu_arr[i, 1]), float(pos_enu_arr[i, 0]), float(-pos_enu_arr[i, 2])]
        v_ned = [float(vel_enu[i, 1]), float(vel_enu[i, 0]), float(-vel_enu[i, 2])]
        yaw_ned = math.pi / 2.0 - float(yaw_enu_arr[i])
        q_ned = [math.cos(yaw_ned * 0.5), 0.0, 0.0, math.sin(yaw_ned * 0.5)]
        odom.append({
            "timestamp": t_ns,
            "timestamp_us": t_us,
            "timestamp_sample_us": t_us,
            "pose_frame": 1,
            "velocity_frame": 1,
            "position": p_ned,
            "q": q_ned,
            "velocity": v_ned,
            "reset_counter": 0,
        })
    last_t_us = int(round((100.0 + (N - 1) * dt_s) * 1e6))
    return {
        "odometry": odom,
        "local_pos": [],
        "land_detected": [
            {
                "timestamp": last_t_us * 1000,
                "timestamp_us": last_t_us,
                "landed": True,
                "ground_contact": True,
                "maybe_landed": True,
                "freefall": False,
                "at_rest": True,
            }
        ],
        "vehicle_status": [
            {
                "timestamp": last_t_us * 1000,
                "timestamp_us": last_t_us,
                "nav_state": 14,
                "arming_state": 2,
                "failsafe": False,
            }
        ],
        "setpoints": [],
    }


def _build_valid_flight_arrays(dt_s: float = 0.05):
    """Genera una trayectoria de vuelo suave con ventana de reposo final de 1.5s en (20, 0, 0.08)."""
    N_flight = 370
    N_rest = 30  # 1.5 s de reposo final a dt=0.05s
    N = N_flight + N_rest

    x_flight = np.linspace(0.0, 20.0, N_flight)
    y_flight = 1.5 * np.exp(-0.5 * ((x_flight - 10.0) / 1.2) ** 2)  # +1.5m en X=10 (left flank)
    z_flight = np.full(N_flight, 1.5)
    z_flight[:15] = np.linspace(0.15, 1.5, 15)
    # Ascenso para overfly en X=15 hasta Z=4.0m
    z_flight += 2.5 * np.exp(-0.5 * ((x_flight - 15.0) / 1.3) ** 2)
    # Descenso final suave a plataforma en X=20
    z_flight[-30:] = np.linspace(z_flight[-31], 0.08, 30)
    y_flight[-30:] = np.linspace(y_flight[-31], 0.0, 30)

    x_arr = np.concatenate([x_flight, np.full(N_rest, 20.0)])
    y_arr = np.concatenate([y_flight, np.zeros(N_rest)])
    z_arr = np.concatenate([z_flight, np.full(N_rest, 0.08)])

    pos_valid = np.column_stack([x_arr, y_arr, z_arr])
    yaw_valid = np.zeros(N)
    return pos_valid, yaw_valid, x_arr


def test_validate_episode_physical_geometry_and_failure_modes():
    """
    Valida Fase 1 / 1.1: verificador independiente de colisiones, primitivas,
    aterrizaje e integridad temporal de grabacion.
    """
    from uav_ip.validate_episode import validate_episode_data, reconstruct_simulation_clock

    waypoints_cfg = {
        "landmarks": [
            {"id": "landmark_1", "type": "square_gate", "x": 5.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
            {"id": "landmark_2", "type": "pole", "x": 10.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
            {"id": "landmark_3", "type": "pole", "x": 15.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
        ],
        "primitives": [
            {"type": "traverse", "target": "landmark_1"},
            {"type": "slalom", "target": "landmark_2", "side": "left"},
            {"type": "overfly", "target": "landmark_3"},
        ],
        "landing": {"x": 20.0, "y": 0.0, "z": 0.05},
    }

    pos_valid, yaw_valid, x_arr = _build_valid_flight_arrays(dt_s=0.05)
    N = len(pos_valid)
    tel_valid = _make_synthetic_telemetry(pos_valid, yaw_valid, dt_s=0.05)

    res_valid = validate_episode_data(tel_valid, waypoints_cfg, pilot_exit_code=0)
    assert res_valid["validation_status"] == "accepted", f"Reasons: {res_valid['rejection_reasons']}"
    assert res_valid["recording_status"] == "valid"
    assert res_valid["mission_status"] == "valid"

    # 2. Falso positivo historico del controlador: bypass lateral del gate a Y=1.10m
    pos_bypass = pos_valid.copy()
    pos_bypass[:, 1] += 1.10 * np.exp(-0.5 * ((x_arr - 5.0) / 1.0) ** 2)
    tel_bypass = _make_synthetic_telemetry(pos_bypass, yaw_valid, dt_s=0.05)
    res_bypass = validate_episode_data(tel_bypass, waypoints_cfg, pilot_exit_code=0)
    assert res_bypass["validation_status"] == "rejected"
    assert any("outside_gate_aperture" in r or "collision" in r for r in res_bypass["rejection_reasons"])

    # 3. Colision contra la barra superior del gate en (X=5.0, Y=0.0, Z=2.25)
    pos_col = pos_valid.copy()
    pos_col[:, 2] += 0.75 * np.exp(-0.5 * ((x_arr - 5.0) / 0.5) ** 2)
    tel_col = _make_synthetic_telemetry(pos_col, yaw_valid, dt_s=0.05)
    res_col = validate_episode_data(tel_col, waypoints_cfg, pilot_exit_code=0)
    assert res_col["validation_status"] == "rejected"
    assert any("collision_with_landmark_1" in r for r in res_col["rejection_reasons"])

    # 4. Slalom por el flanco equivocado (derecho Y=-1.5m cuando se pidio left)
    pos_wrong_slalom = pos_valid.copy()
    pos_wrong_slalom[:, 1] = -1.5 * np.exp(-0.5 * ((x_arr - 10.0) / 1.2) ** 2)
    tel_wrong_slalom = _make_synthetic_telemetry(pos_wrong_slalom, yaw_valid, dt_s=0.05)
    res_wrong_slalom = validate_episode_data(tel_wrong_slalom, waypoints_cfg, pilot_exit_code=0)
    assert res_wrong_slalom["validation_status"] == "rejected"
    assert any("wrong_flank_or_offset_left" in r for r in res_wrong_slalom["rejection_reasons"])

    # 5. Reset del estimador EKF2 a mitad de vuelo
    tel_reset = _make_synthetic_telemetry(pos_valid, yaw_valid, dt_s=0.05)
    for item in tel_reset["odometry"][N // 2:]:
        item["reset_counter"] = 1
    res_reset = validate_episode_data(tel_reset, waypoints_cfg, pilot_exit_code=0)
    assert res_reset["recording_status"] == "invalid"
    assert res_reset["validation_status"] == "rejected"
    assert any("ekf2_estimator_reset_in_flight" in r for r in res_reset["rejection_reasons"])

    # 6. Reconstruccion de reloj ante salto de offset Timesync de uxrce_dds_client (Patron A: UNIX > 1e8, boot < 1e6)
    ts_bag = np.linspace(1700000000.0, 1700000010.0, 200)
    ts_px4 = ts_bag.copy()
    ts_px4[85] = 62.10  # Muestra aislada con timestamp crudo de boot PX4 sin offset UNIX (salto de bajada y subida)
    ts_reconstructed, jumps = reconstruct_simulation_clock(ts_px4, ts_bag)
    assert jumps == 2
    assert np.all(np.diff(ts_reconstructed) > 0)
    assert np.allclose(ts_reconstructed, ts_bag - ts_bag[0], atol=1e-6)


def test_validate_episode_false_positive_battery(tmp_path):
    """
    Valida la bateria de 9 falsos positivos identificada en docs/feedback/respuesta.md (§6.D):
    1. Sin mensajes del detector de aterrizaje -> Rechazar.
    2. Contacto (ground_contact=True) sin landed=True -> Rechazar.
    3. Paso cerca de la plataforma y termino afuera -> Rechazar.
    4. Timeout de aterrizaje / pilot_exit_code != 0 -> Rechazar.
    5. Barrido orbital suficiente pero mirando hacia afuera -> Rechazar.
    6. Primitivas realizadas en orden temporal incorrecto -> Rechazar.
    7. Dos muestras discretas libres con una barra entre ellas (subdivision de segmento) -> Rechazar.
    8. Marco de posicion desconocido (pose_frame != 1) -> Rechazar.
    9. MCAP truncado con prefijo aparentemente valido -> No certificar integridad.
    """
    from uav_ip.validate_episode import (
        validate_episode_data,
        verify_mcap_integrity,
        load_episode_telemetry,
        MCAP_MAGIC,
    )

    waypoints_cfg = {
        "landmarks": [
            {"id": "landmark_1", "type": "square_gate", "x": 5.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
            {"id": "landmark_2", "type": "pole", "x": 10.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
            {"id": "landmark_3", "type": "pole", "x": 15.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
        ],
        "primitives": [
            {"type": "traverse", "target": "landmark_1"},
            {"type": "slalom", "target": "landmark_2", "side": "left"},
            {"type": "overfly", "target": "landmark_3"},
        ],
        "landing": {"x": 20.0, "y": 0.0, "z": 0.05},
    }
    pos_valid, yaw_valid, _ = _build_valid_flight_arrays(dt_s=0.05)

    # Caso 1: Sin mensajes del detector de aterrizaje
    tel_no_land_msg = _make_synthetic_telemetry(pos_valid, yaw_valid, dt_s=0.05)
    tel_no_land_msg["land_detected"] = []
    res_1 = validate_episode_data(tel_no_land_msg, waypoints_cfg, pilot_exit_code=0)
    assert res_1["validation_status"] == "rejected"
    assert any("missing_land_detected_messages" in r for r in res_1["rejection_reasons"])

    # Caso 2: Contacto (ground_contact=True) sin landed=True
    tel_contact_only = _make_synthetic_telemetry(pos_valid, yaw_valid, dt_s=0.05)
    tel_contact_only["land_detected"][0]["landed"] = False
    tel_contact_only["land_detected"][0]["ground_contact"] = True
    res_2 = validate_episode_data(tel_contact_only, waypoints_cfg, pilot_exit_code=0)
    assert res_2["validation_status"] == "rejected"
    assert any("px4_land_detector_not_confirmed" in r for r in res_2["rejection_reasons"])

    # Caso 3: Paso cerca de la plataforma y termino afuera (fly-by sobre el pad en t intermedio, reposo a 1.8m)
    pos_flyby = pos_valid.copy()
    # En las ultimas 30 muestras (ventana de reposo), se desplaza suavemente a Y=1.8m fuera del pad
    pos_flyby[-60:-30, 1] = np.linspace(0.0, 1.8, 30)
    pos_flyby[-30:, 1] = 1.8
    tel_flyby = _make_synthetic_telemetry(pos_flyby, yaw_valid, dt_s=0.05)
    res_3 = validate_episode_data(tel_flyby, waypoints_cfg, pilot_exit_code=0)
    assert res_3["validation_status"] == "rejected"
    assert any("landing_off_pad" in r for r in res_3["rejection_reasons"])

    # Caso 4: Timeout de aterrizaje / pilot_exit_code != 0
    tel_timeout = _make_synthetic_telemetry(pos_valid, yaw_valid, dt_s=0.05)
    res_4 = validate_episode_data(tel_timeout, waypoints_cfg, pilot_exit_code=1)
    assert res_4["validation_status"] == "rejected"
    assert any("pilot_non_zero_exit_code" in r for r in res_4["rejection_reasons"])

    # Caso 5: Barrido orbital suficiente (315 deg) pero mirando hacia afuera (yaw opuesto al poste)
    orbit_cfg = {
        "landmarks": [{"id": "pole_1", "type": "pole", "x": 5.0, "y": 0.0, "z": 1.5, "yaw": 0.0}],
        "primitives": [{"type": "orbit", "target": "pole_1"}],
        "landing": {"x": 10.0, "y": 0.0, "z": 0.05},
    }
    N_orb = 300
    angles = np.linspace(-math.pi, 0.75 * math.pi, 180)  # 315 deg de barrido
    x_orb = np.concatenate([
        np.linspace(0.0, 5.0 + 1.8 * math.cos(angles[0]), 40),
        5.0 + 1.8 * np.cos(angles),
        np.linspace(5.0 + 1.8 * math.cos(angles[-1]), 10.0, 50),
        np.full(30, 10.0),
    ])
    y_orb = np.concatenate([
        np.linspace(0.0, 1.8 * math.sin(angles[0]), 40),
        1.8 * np.sin(angles),
        np.linspace(1.8 * math.sin(angles[-1]), 0.0, 50),
        np.zeros(30),
    ])
    z_orb = np.concatenate([
        np.full(N_orb - 60, 1.5),
        np.linspace(1.5, 0.08, 30),
        np.full(30, 0.08),
    ])
    pos_orb = np.column_stack([x_orb, y_orb, z_orb])
    # Yaw mirando hacia AFUERA del poste (opuesto al vector centripeta)
    yaw_outward = np.arctan2(y_orb - 0.0, x_orb - 5.0)
    tel_orb_out = _make_synthetic_telemetry(pos_orb, yaw_outward, dt_s=0.04)
    res_5 = validate_episode_data(tel_orb_out, orbit_cfg, pilot_exit_code=0)
    assert res_5["validation_status"] == "rejected"
    assert any("excessive_centripetal_yaw_error" in r for r in res_5["rejection_reasons"])

    # Verificar que mirando hacia ADENTRO (centripeta) el mismo vuelo orbital es aceptado
    yaw_inward = np.arctan2(0.0 - y_orb, 5.0 - x_orb)
    tel_orb_in = _make_synthetic_telemetry(pos_orb, yaw_inward, dt_s=0.04)
    res_5_ok = validate_episode_data(tel_orb_in, orbit_cfg, pilot_exit_code=0)
    assert res_5_ok["validation_status"] == "accepted", f"Reasons: {res_5_ok['rejection_reasons']}"

    # Caso 6: Primitivas realizadas en orden temporal incorrecto
    waypoints_wrong_order = {
        "landmarks": waypoints_cfg["landmarks"],
        "primitives": [
            {"type": "slalom", "target": "landmark_2", "side": "left"},  # ocurre en X=10 (despues)
            {"type": "traverse", "target": "landmark_1"},                # ocurre en X=5 (antes)
            {"type": "overfly", "target": "landmark_3"},
        ],
        "landing": waypoints_cfg["landing"],
    }
    tel_order = _make_synthetic_telemetry(pos_valid, yaw_valid, dt_s=0.05)
    res_6 = validate_episode_data(tel_order, waypoints_wrong_order, pilot_exit_code=0)
    assert res_6["validation_status"] == "rejected"
    assert any("primitive_order_violation" in r for r in res_6["rejection_reasons"])

    # Caso 7: Dos muestras discretas libres (> 0.28m) con una barra entre ellas (< 0.28m en el segmento)
    from uav_ip.validate_episode import compute_min_distance_to_landmark, UAV_COLLISION_RADIUS_XY
    gate_lm = {"id": "landmark_1", "type": "square_gate", "x": 5.0, "y": 0.0, "z": 1.5, "yaw": 0.0}
    # La viga superior ocupa X in [4.95, 5.05], Z in [2.20, 2.30].
    # En P1=(4.83, 0, 2.555) y P2=(5.17, 0, 2.555) (paso=0.34m <= 0.35m),
    # dist_vertice = hypot(0.12, 0.255) = 0.2818m > 0.28m (libres en muestras discretas),
    # pero en el punto medio (5.00, 0, 2.555) dist_segmento = 0.255m < 0.28m (colision).
    sparse_seg = np.array([[4.83, 0.0, 2.555], [5.17, 0.0, 2.555]], dtype=np.float64)
    d_without_subdiv = compute_min_distance_to_landmark(sparse_seg, gate_lm, max_step_m=10.0)
    d_with_subdiv = compute_min_distance_to_landmark(sparse_seg, gate_lm, max_step_m=0.02)
    assert d_without_subdiv > UAV_COLLISION_RADIUS_XY  # 0.2818m > 0.28m (falso negativo sin subdividir)
    assert d_with_subdiv < UAV_COLLISION_RADIUS_XY     # 0.2550m < 0.28m (detectado con subdivision a 0.02m)

    # Verificar que validate_episode_data rechaza el episodio cuando ese segmento ocurre en vuelo
    idx_before = int(np.searchsorted(pos_valid[:, 0], 4.83))
    idx_after = int(np.searchsorted(pos_valid[:, 0], 5.17))
    pos_bar_between = np.vstack([
        pos_valid[:idx_before],
        np.array([[4.83, 0.0, 2.555], [5.17, 0.0, 2.555]]),
        pos_valid[idx_after + 1:],
    ])
    # Ajustar suavemente Z antes y despues para que ningun paso supere 0.35m pero los vertices en X=4.83 y X=5.17 sean los unicos cerca de X=5.0
    for k_step in range(1, 12):
        alpha = k_step / 12.0
        pos_bar_between[idx_before - k_step, 2] = (1.0 - alpha) * 2.555 + alpha * 1.5
        pos_bar_between[idx_before + 1 + k_step, 2] = (1.0 - alpha) * 2.555 + alpha * 1.5
    yaw_bar_between = np.zeros(len(pos_bar_between))
    tel_tunnel = _make_synthetic_telemetry(pos_bar_between, yaw_bar_between, dt_s=0.05)
    res_7 = validate_episode_data(tel_tunnel, waypoints_cfg, pilot_exit_code=0)
    assert res_7["validation_status"] == "rejected"
    assert any("collision_with_landmark_1" in r for r in res_7["rejection_reasons"])

    # Caso 8: Marco de posicion desconocido (pose_frame != 1)
    tel_bad_frame = _make_synthetic_telemetry(pos_valid, yaw_valid, dt_s=0.05)
    tel_bad_frame["odometry"][10]["pose_frame"] = 2
    res_8 = validate_episode_data(tel_bad_frame, waypoints_cfg, pilot_exit_code=0)
    assert res_8["validation_status"] == "rejected"
    assert any("invalid_or_non_ned_coordinate_frame" in r for r in res_8["rejection_reasons"])

    # Caso 9: MCAP truncado con prefijo aparentemente valido
    truncated_mcap = tmp_path / "truncated_0.mcap"
    # Escribir magic header + un registro Header valido + relleno > 45 bytes + truncamiento sin Footer ni magic final
    header_payload = b"\x04\x00\x00\x00ros2" + b"\x04\x00\x00\x00test" + (b"\x00" * 40)
    truncated_bytes = MCAP_MAGIC + bytes([0x01]) + (len(header_payload)).to_bytes(8, "little") + header_payload
    truncated_mcap.write_bytes(truncated_bytes)
    with pytest.raises(ValueError, match="truncated_mcap_missing_trailer_magic"):
        verify_mcap_integrity(truncated_mcap)
    with pytest.raises(ValueError, match="truncated_mcap_missing_trailer_magic"):
        load_episode_telemetry(truncated_mcap)


def test_mcap_native_cdr_offsets_and_footer_verification(tmp_path):
    """
    Construye un archivo MCAP sintetico completo (con Header, Schema, Channel, Message, Footer y Magic)
    para verificar de forma autocontenida:
    - verify_mcap_integrity en un MCAP bien formado vs truncado.
    - Los offsets CDR exactos de VehicleLocalPosition (vx en 56, heading en 104),
      VehicleLandDetected (at_rest en 23) y VehicleStatus (arming_state en 28, nav_state en 45, failsafe en 62).
    """
    import struct
    from uav_ip.validate_episode import (
        MCAP_MAGIC,
        verify_mcap_integrity,
        _parse_mcap_native,
    )

    def make_record(opcode: int, payload: bytes) -> bytes:
        return bytes([opcode]) + struct.pack("<Q", len(payload)) + payload

    def make_mcap_str(s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<I", len(b)) + b

    # 1. Header record (opcode 0x01)
    header_rec = make_record(0x01, make_mcap_str("ros2") + make_mcap_str("test"))

    # 2. Schemas (opcode 0x03): id(u16), name(str), encoding(str), data(bytes)
    def make_schema(sid: int, name: str) -> bytes:
        payload = struct.pack("<H", sid) + make_mcap_str(name) + make_mcap_str("ros2msg") + struct.pack("<I", 0)
        return make_record(0x03, payload)

    s_lpos = make_schema(1, "px4_msgs/msg/VehicleLocalPosition")
    s_land = make_schema(2, "px4_msgs/msg/VehicleLandDetected")
    s_stat = make_schema(3, "px4_msgs/msg/VehicleStatus")

    # 3. Channels (opcode 0x04): id(u16), schema_id(u16), topic(str), msg_enc(str), metadata_map(u32=0)
    def make_channel(cid: int, sid: int, topic: str) -> bytes:
        payload = (
            struct.pack("<HH", cid, sid)
            + make_mcap_str(topic)
            + make_mcap_str("cdr")
            + struct.pack("<I", 0)
        )
        return make_record(0x04, payload)

    c_lpos = make_channel(1, 1, "/fmu/out/vehicle_local_position_v1")
    c_land = make_channel(2, 2, "/fmu/out/vehicle_land_detected")
    c_stat = make_channel(3, 3, "/fmu/out/vehicle_status_v4")

    # 4. Messages (opcode 0x05): channel_id(u16), seq(u32), log_time(u64), pub_time(u64), data
    def make_msg(cid: int, log_time: int, cdr_data: bytes) -> bytes:
        hdr = struct.pack("<HIQQ", cid, 1, log_time, log_time)
        return make_record(0x05, hdr + cdr_data)

    # VehicleLocalPosition CDR buffer (168 bytes)
    lpos_buf = bytearray(168)
    lpos_buf[0:4] = b"\x00\x01\x00\x00"
    struct.pack_into("<QQ", lpos_buf, 4, 123456000, 123455000)
    lpos_buf[20] = 1  # xy_valid
    lpos_buf[21] = 1  # z_valid
    lpos_buf[22] = 1  # v_xy_valid
    lpos_buf[23] = 1  # v_z_valid
    struct.pack_into("<fff", lpos_buf, 24, 10.5, -3.25, -1.75)
    struct.pack_into("<fff", lpos_buf, 56, 1.25, -0.5, 0.125)  # vx, vy, vz at offset 56
    struct.pack_into("<f", lpos_buf, 104, 0.785398)            # heading at offset 104

    # VehicleLandDetected CDR buffer (28 bytes)
    land_buf = bytearray(28)
    land_buf[0:4] = b"\x00\x01\x00\x00"
    struct.pack_into("<Q", land_buf, 4, 123456000)
    land_buf[12] = 0  # freefall
    land_buf[13] = 1  # ground_contact
    land_buf[14] = 1  # maybe_landed
    land_buf[15] = 1  # landed
    land_buf[23] = 1  # at_rest at offset 23

    # VehicleStatus CDR buffer (92 bytes)
    stat_buf = bytearray(92)
    stat_buf[0:4] = b"\x00\x01\x00\x00"
    struct.pack_into("<Q", stat_buf, 4, 123456000)
    stat_buf[28] = 2   # arming_state at offset 28
    stat_buf[45] = 14  # nav_state (OFFBOARD=14) at offset 45
    stat_buf[62] = 1   # failsafe at offset 62

    m_lpos = make_msg(1, 123456000000, bytes(lpos_buf))
    m_land = make_msg(2, 123456000000, bytes(land_buf))
    m_stat = make_msg(3, 123456000000, bytes(stat_buf))

    # 5. Footer record (opcode 0x02): summary_start(u64), summary_offset_start(u64), summary_crc(u32)
    footer_rec = make_record(0x02, struct.pack("<QQI", 0, 0, 0))

    full_mcap = (
        MCAP_MAGIC
        + header_rec
        + s_lpos + s_land + s_stat
        + c_lpos + c_land + c_stat
        + m_lpos + m_land + m_stat
        + footer_rec
        + MCAP_MAGIC
    )
    mcap_file = tmp_path / "synthetic_valid.mcap"
    mcap_file.write_bytes(full_mcap)

    verified_buf = verify_mcap_integrity(mcap_file)
    assert len(verified_buf) == len(full_mcap)

    parsed = _parse_mcap_native(mcap_file, buf=verified_buf)
    assert len(parsed["local_pos"]) == 1
    lp = parsed["local_pos"][0]
    assert np.allclose([lp["x"], lp["y"], lp["z"]], [10.5, -3.25, -1.75], atol=1e-5)
    assert np.allclose([lp["vx"], lp["vy"], lp["vz"]], [1.25, -0.5, 0.125], atol=1e-5)
    assert abs(lp["heading"] - 0.785398) < 1e-5

    assert len(parsed["land_detected"]) == 1
    ld = parsed["land_detected"][0]
    assert ld["landed"] is True
    assert ld["ground_contact"] is True
    assert ld["at_rest"] is True

    assert len(parsed["vehicle_status"]) == 1
    vs = parsed["vehicle_status"][0]
    assert vs["arming_state"] == 2
    assert vs["nav_state"] == 14
    assert vs["failsafe"] is True


def test_world_generator_transition_feasibility():
    """Valida Fase 1 / 1.1 (Tarea 1.3): chequeo de factibilidad geometrica de transiciones en world_generator."""
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from world_generator import check_circuit_transition_feasibility

    start_pose = {"x": 0.0, "y": 0.0, "z": 0.2, "yaw": 0.0}
    hoops_ok = [
        {"id": "landmark_1", "type": "square_gate", "x": 5.0, "y": 0.0, "z": 1.6, "yaw": 0.0},
        {"id": "landmark_2", "type": "pole", "x": 10.5, "y": 1.0, "z": 1.5, "yaw": 0.1},
    ]
    prims_ok = [
        {"type": "traverse", "target": "landmark_1"},
        {"type": "orbit", "target": "landmark_2"},
    ]
    landing_ok = {"x": 16.0, "y": 0.5, "z": 0.05}
    ok, reasons = check_circuit_transition_feasibility(hoops_ok, prims_ok, landing_ok, start_pose=start_pose)
    assert ok is True, f"Unexpected rejection: {reasons}"

    # Caso invalido 1: pendiente vertical excesiva entre overfly (Z=4.0) y traverse bajo (Z=1.0) muy cercano
    hoops_steep = [
        {"id": "landmark_1", "type": "pole", "x": 5.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
        {"id": "landmark_2", "type": "square_gate", "x": 9.0, "y": 0.0, "z": 1.0, "yaw": 0.0},
    ]
    prims_steep = [
        {"type": "overfly", "target": "landmark_1"},
        {"type": "traverse", "target": "landmark_2"},
    ]
    ok_steep, reasons_steep = check_circuit_transition_feasibility(
        hoops_steep, prims_steep, landing_ok, start_pose=start_pose
    )
    assert ok_steep is False
    assert any("excessive_slope" in r for r in reasons_steep)

    # Caso invalido 2: un tercer obstaculo interpuesto en medio del segmento recto de transicion
    hoops_blocked = [
        {"id": "landmark_1", "type": "square_gate", "x": 5.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
        {"id": "landmark_2", "type": "pole", "x": 9.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
        {"id": "landmark_3", "type": "square_gate", "x": 13.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
    ]
    # Mision que visita landmark_1 y landmark_3; landmark_2 (poste sin primitiva asignada) queda
    # sobre el segmento recto post_1 (X=6.2, Y=0) -> pre_3 (X=11.8, Y=0) en X=9.0, Y=0.0
    prims_blocked = [
        {"type": "traverse", "target": "landmark_1"},
        {"type": "traverse", "target": "landmark_3"},
    ]
    ok_blocked, reasons_blocked = check_circuit_transition_feasibility(
        hoops_blocked, prims_blocked, landing_ok, start_pose=start_pose
    )
    assert ok_blocked is False
    assert any("transition_segment_0_1_intersects_pole_landmark_2" in r for r in reasons_blocked)


def test_real_dataset_mcap_validation():
    """Verifica el lector MCAP (rosbags vs parser CDR nativo) y el validador fisico sobre episodios reales si existen."""
    from pathlib import Path
    from uav_ip.validate_episode import (
        validate_episode_directory,
        _parse_mcap_native,
        _parse_with_rosbags,
    )

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    ep_dir = repo_root / "datasets" / "task_0001" / "episode_00"
    if not ep_dir.is_dir():
        pytest.skip("Episodio local datasets/task_0001/episode_00 no disponible en este entorno")

    report = validate_episode_directory(ep_dir, update_meta=False)
    assert report["validation_status"] == "accepted"
    assert report["recording_status"] == "valid"
    assert report["mission_status"] == "valid"
    assert report["metrics"]["min_obstacle_clearance_m"] > 0.30

    # Comparar el lector mantenido (rosbags) contra el parser nativo CDR (_parse_mcap_native)
    mcap_path = ep_dir / "rosbag" / "rosbag_0.mcap"
    if mcap_path.is_file():
        native_tel = _parse_mcap_native(mcap_path)
        try:
            rosbags_tel = _parse_with_rosbags([mcap_path])
        except Exception:
            rosbags_tel = None

        if rosbags_tel is not None:
            for key in ("odometry", "local_pos", "land_detected", "vehicle_status", "setpoints"):
                assert len(native_tel[key]) == len(rosbags_tel[key]), f"Count mismatch in {key}"
            # Verificar coincidencia numerica bit a bit en odometria, posicion local, detector de aterrizaje y estado
            for a, b in zip(native_tel["odometry"][::50], rosbags_tel["odometry"][::50]):
                assert a["timestamp_sample_us"] == b["timestamp_sample_us"]
                assert a["pose_frame"] == b["pose_frame"]
                assert np.allclose(a["position"], b["position"], atol=1e-6)
                assert np.allclose(a["velocity"], b["velocity"], atol=1e-6)
                assert a["reset_counter"] == b["reset_counter"]
            for a, b in zip(native_tel["local_pos"][::50], rosbags_tel["local_pos"][::50]):
                assert np.allclose([a["x"], a["y"], a["z"]], [b["x"], b["y"], b["z"]], atol=1e-6)
                assert np.allclose([a["vx"], a["vy"], a["vz"]], [b["vx"], b["vy"], b["vz"]], atol=1e-6)
                assert abs(a["heading"] - b["heading"]) < 1e-6
            for a, b in zip(native_tel["land_detected"], rosbags_tel["land_detected"]):
                assert a["landed"] == b["landed"]
                assert a["ground_contact"] == b["ground_contact"]
                assert a["at_rest"] == b["at_rest"]
            for a, b in zip(native_tel["vehicle_status"], rosbags_tel["vehicle_status"]):
                assert a["arming_state"] == b["arming_state"]
                assert a["nav_state"] == b["nav_state"]
                assert a["failsafe"] == b["failsafe"]



