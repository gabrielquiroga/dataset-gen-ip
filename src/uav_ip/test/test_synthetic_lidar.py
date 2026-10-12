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

    pad = generate_canonical_landing_pad(1000)
    assert pad.shape == (1000, 3)


def test_fps_sampling_and_edge_cases():
    test_pts = np.random.uniform(-5, 5, size=(5000, 3)).astype(np.float32)
    sampled = fps_numpy(test_pts, 2048, seed=42)
    assert sampled.shape == (2048, 3)

    # Padding when N < 2048
    small_pts = np.random.uniform(-1, 1, size=(50, 3)).astype(np.float32)
    sampled_pad = fps_numpy(small_pts, 2048, seed=42)
    assert sampled_pad.shape == (2048, 3)

    # Empty cloud
    empty_pts = np.zeros((0, 3), dtype=np.float32)
    sampled_empty = fps_numpy(empty_pts, 2048, seed=42)
    assert sampled_empty.shape == (2048, 3)
    assert np.all(sampled_empty == 0.0)


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

    # Heading North: yaw_ned = 0 -> yaw_enu = pi/2
    yaw_ned_north = 0.0
    yaw_enu_north = yaw_ned_to_gazebo_enu(yaw_ned_north)
    assert abs(yaw_enu_north - math.pi / 2.0) < 1e-6
    q_ned_north = [1.0, 0.0, 0.0, 0.0]
    _, R_conv_north = px4_odometry_to_gazebo_enu([0, 0, 0], q_ned_north)
    assert np.allclose(R_conv_north, rot_z(math.pi / 2.0))


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
