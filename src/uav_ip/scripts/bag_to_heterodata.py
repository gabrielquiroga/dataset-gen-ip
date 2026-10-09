#!/usr/bin/env python3
"""
bag_to_heterodata.py - Converts ROS2 bag (.mcap) files into PyTorch Geometric HeteroData format.

This script parses the odometry and point cloud messages, extracts the actions,
and packages them into a format ready for Instant Policy training.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
try:
    from rosbags.rosbag2 import Reader
    from rosbags.serde import deserialize_cdr
except ImportError:
    Reader = None
    print("Warning: rosbags not installed. Please 'pip install rosbags'")
try:
    from torch_geometric.data import HeteroData
except ImportError:
    HeteroData = None
    print("Warning: torch_geometric not installed.")

# Relevant Topics
TOPIC_ODOM = "/fmu/out/vehicle_odometry"
TOPIC_PCL = "/depth_camera/points"
TOPIC_SETPOINT = "/fmu/in/trajectory_setpoint"

def fps_numpy(points: np.ndarray, n_samples: int) -> np.ndarray:
    """Farthest Point Sampling in numpy."""
    N = len(points)
    if N <= n_samples:
        indices = np.arange(N)
        if N < n_samples and N > 0:
            pad = np.random.choice(N, n_samples - N, replace=True)
            indices = np.concatenate([indices, pad])
        elif N == 0:
            return np.zeros((n_samples, 3), dtype=np.float32)
        return points[indices]
    
    selected = np.zeros(n_samples, dtype=int)
    distances = np.full(N, np.inf)
    current = np.random.randint(0, N)
    
    for i in range(n_samples):
        selected[i] = current
        p = points[current]
        dists = np.sum((points - p)**2, axis=1)
        distances = np.minimum(distances, dists)
        current = np.argmax(distances)
        
    return points[selected]

def process_pointcloud(msg) -> np.ndarray:
    """
    Parses PointCloud2, transforms to local drone frame, filters, and downsamples via FPS.
    """
    x_offset = y_offset = z_offset = None
    for field in msg.fields:
        if field.name == 'x': x_offset = field.offset
        if field.name == 'y': y_offset = field.offset
        if field.name == 'z': z_offset = field.offset
        
    if x_offset is None or y_offset is None or z_offset is None:
        return np.zeros((2048, 3), dtype=np.float32)
        
    point_step = msg.point_step
    # Ensure data length is perfectly divisible by point_step to avoid reshape errors
    raw_data = np.frombuffer(msg.data, dtype=np.uint8)
    valid_len = (len(raw_data) // point_step) * point_step
    data = raw_data[:valid_len].reshape(-1, point_step)
    
    x = data[:, x_offset:x_offset+4].copy().view(dtype=np.float32).flatten()
    y = data[:, y_offset:y_offset+4].copy().view(dtype=np.float32).flatten()
    z = data[:, z_offset:z_offset+4].copy().view(dtype=np.float32).flatten()
    
    points = np.vstack((x, y, z)).T
    
    # Remove NaNs and Infs
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    
    if len(points) == 0:
        return np.zeros((2048, 3), dtype=np.float32)

    # Transform from camera optical frame (z fwd, x right, y down) 
    # to drone local frame FLU (x fwd, y left, z up)
    x_flu = points[:, 2]
    y_flu = -points[:, 0]
    z_flu = -points[:, 1]
    points_flu = np.vstack((x_flu, y_flu, z_flu)).T
    
    # Filtering distant objects (>10m), sky, and drone body (<0.3m)
    distances = np.linalg.norm(points_flu, axis=1)
    mask = (distances > 0.3) & (distances < 10.0)
    points_filtered = points_flu[mask]
    
    if len(points_filtered) == 0:
        points_filtered = points_flu # Fallback if everything was filtered out
        
    # Downsample with Farthest Point Sampling to 2048 points
    return fps_numpy(points_filtered, 2048)

def parse_bag(bag_path: Path):
    """
    Parses a ROS2 bag and extracts odometry, point clouds, and setpoints.
    """
    print(f"Parsing bag: {bag_path}")
    if Reader is None:
        return {}
    
    data = {
        'odometry': [],
        'pointclouds': [],
        'setpoints': []
    }
    
    with Reader(bag_path) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == TOPIC_ODOM:
                msg = deserialize_cdr(rawdata, connection.msgtype)
                # Store position, orientation (quaternion), velocity
                data['odometry'].append({
                    'timestamp': timestamp,
                    'position': msg.position if hasattr(msg, 'position') else [0,0,0],
                    'orientation': msg.q if hasattr(msg, 'q') else [1,0,0,0],
                    'velocity': msg.velocity if hasattr(msg, 'velocity') else [0,0,0]
                })
            elif connection.topic == TOPIC_PCL:
                msg = deserialize_cdr(rawdata, connection.msgtype)
                data['pointclouds'].append({
                    'timestamp': timestamp,
                    'data': process_pointcloud(msg)
                })
            elif connection.topic == TOPIC_SETPOINT:
                msg = deserialize_cdr(rawdata, connection.msgtype)
                data['setpoints'].append({
                    'timestamp': timestamp,
                    'position': msg.position if hasattr(msg, 'position') else [0,0,0],
                    'yaw': msg.yaw if hasattr(msg, 'yaw') else 0.0
                })
    return data

def build_heterodata(parsed_data, output_path: Path):
    """
    Builds a PyTorch Geometric HeteroData object from parsed bag data.
    """
    print(f"Building HeteroData and saving to {output_path}")
    if HeteroData is None or not parsed_data:
        return
    
    data = HeteroData()
    
    # Extract states (e.g. odometry positions)
    if parsed_data['odometry']:
        positions = [odom['position'] for odom in parsed_data['odometry']]
        data['agent'].x = torch.tensor(positions, dtype=torch.float32)
    else:
        data['agent'].x = torch.empty((0, 3), dtype=torch.float32)
        
    # Extract actions (e.g. setpoints)
    if parsed_data['setpoints']:
        targets = [sp['position'] for sp in parsed_data['setpoints']]
        data['agent'].y = torch.tensor(targets, dtype=torch.float32)
    else:
        data['agent'].y = torch.empty((0, 3), dtype=torch.float32)
        
    # Extract point clouds
    if parsed_data['pointclouds']:
        pcs = [pc['data'] for pc in parsed_data['pointclouds']]
        # pcs is list of (2048, 3) arrays
        data['scene'].pos = torch.tensor(np.stack(pcs), dtype=torch.float32)
    else:
        data['scene'].pos = torch.empty((0, 2048, 3), dtype=torch.float32)
        
    # Define edges (e.g., temporal sequence)
    num_nodes = data['agent'].x.size(0)
    if num_nodes > 1:
        edges_src = torch.arange(0, num_nodes - 1, dtype=torch.long)
        edges_dst = torch.arange(1, num_nodes, dtype=torch.long)
        data['agent', 'temporal', 'agent'].edge_index = torch.stack([edges_src, edges_dst], dim=0)
    else:
        data['agent', 'temporal', 'agent'].edge_index = torch.empty((2, 0), dtype=torch.long)
        
    torch.save(data, output_path)

def main():
    parser = argparse.ArgumentParser(description="Convert ROS2 bag to HeteroData")
    parser.add_argument("--bag-dir", type=Path, required=True, help="Directory containing the .mcap bag")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_processed"), help="Output directory")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    parsed = parse_bag(args.bag_dir)
    
    out_file = args.output_dir / (args.bag_dir.name + ".pt")
    build_heterodata(parsed, out_file)
    print("Conversion complete.")

if __name__ == "__main__":
    main()
