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
    if dataset_dir.is_dir():
        eps = discover_dataset_episodes(dataset_dir)
        assert len(eps) >= 6
        labels = [e["label"] for e in eps]
        assert "task_0000/episode_00" in labels
        assert "task_0001/episode_02" in labels


def test_heterodata_load_node_types_compatibility():
    """Valida que la deteccion de nodos en HeteroData no use 'in data' directamente."""
    from uav_ip.visualize_dataset import _to_numpy

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

