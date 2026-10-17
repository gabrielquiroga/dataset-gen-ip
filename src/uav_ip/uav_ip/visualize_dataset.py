#!/usr/bin/env python3
"""
visualize_dataset.py — Herramienta de Visualizacion 3D para HeteroData y LiDAR Sintetico.

Permite inspeccionar graficamente en 3D:
1. Archivos generados por bag_to_heterodata.py (.pt de PyTorch Geometric o .npz de NumPy).
2. Nubes de puntos LiDAR 360° generadas directamente desde waypoints.yaml (sin necesidad de volar).

Uso:
  # Opcion 1: Visualizar directamente desde waypoints.yaml para verificar el LiDAR
  python3 visualize_dataset.py --waypoints src/uav_ip/generated_worlds/test_001/waypoints.yaml --drone-pose 0.0 0.0 1.5 0.0 --output lidar_view.png

  # Opcion 2: Visualizar un archivo de datos .pt o .npz generado por bag_to_heterodata.py
  python3 visualize_dataset.py --file test_001.pt --step 0 --output grafo_view.png
  python3 visualize_dataset.py --file test_001.npz --step 10 --output grafo_view.png
"""

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

# Configurar backend de Matplotlib segun disponibilidad de entorno grafico (WSLg / X11)
has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
if not has_display:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

# Agregar directorio local al sys.path para importar synthetic_lidar
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

try:
    from uav_ip.synthetic_lidar import SyntheticLidarGenerator
except ImportError:
    try:
        from synthetic_lidar import SyntheticLidarGenerator
    except ImportError:
        SyntheticLidarGenerator = None


def _to_numpy(val):
    """Convierte un tensor de PyTorch o arreglo a np.ndarray."""
    if val is None:
        return None
    if hasattr(val, "detach"):
        return val.detach().cpu().numpy()
    return np.asarray(val)


def _extract_local_horizon_flu(agent_y: np.ndarray, action_se3: np.ndarray = None) -> np.ndarray:
    """
    Extrae el horizonte de traslacion relativo (T, H, 3) en el marco local Horizontal FLU
    a partir de agent_y (T, H, >=3) o de action_se3 (T, H, 4, 4).
    Si el archivo es legado (T, 3), retorna None.
    """
    if action_se3 is not None and action_se3.ndim == 4 and action_se3.shape[-2:] == (4, 4):
        return action_se3[:, :, :3, 3].astype(np.float32)
    if agent_y is not None and agent_y.ndim == 3 and agent_y.shape[-1] >= 3:
        return agent_y[:, :, :3].astype(np.float32)
    return None


def load_dataset_file(file_path: Path, return_extra: bool = False):
    """
    Carga un archivo .pt (HeteroData) o .npz (NumPy) de forma retrocompatible y extrae:
    - agent_x: (T, 3) posiciones del dron en ENU (escaladas s=0.1)
    - agent_y: (T, 8, 5) horizonte de acciones relativas SE(3) o (T, 3) en archivos legados
    - scene_pos: (T, 2048, 3) nubes de puntos LiDAR egocentricas (escaladas s=0.1)
    - extra (opcional si return_extra=True): dict con 'action_horizon_flu', 'keyframe_indices',
      'flight_state', 'setpoint_raw', 'yaw_enu', 'keypoints_body'.
    """
    ext = file_path.suffix.lower()
    extra = {
        "action_horizon_flu": None,
        "keyframe_indices": None,
        "flight_state": None,
        "setpoint_raw": None,
        "yaw_enu": None,
        "keypoints_body": None,
    }

    if ext == ".npz":
        data = np.load(file_path)
        agent_x = data["pos_enu"] if "pos_enu" in data else data["agent_x"]
        agent_y = data["agent_y"] if "agent_y" in data else None
        scene_pos = data["scene_pos"]
        action_se3 = data["action_se3"] if "action_se3" in data else None
        extra["action_horizon_flu"] = _extract_local_horizon_flu(agent_y, action_se3)
        if "keyframe_indices" in data:
            extra["keyframe_indices"] = np.asarray(data["keyframe_indices"], dtype=np.int64)
        if "flight_state" in data:
            extra["flight_state"] = np.asarray(data["flight_state"], dtype=np.float32)
        if "setpoint_raw" in data:
            extra["setpoint_raw"] = np.asarray(data["setpoint_raw"], dtype=np.float32)
        if "yaw_enu" in data:
            extra["yaw_enu"] = np.asarray(data["yaw_enu"], dtype=np.float32)
        if "keypoints_body" in data:
            extra["keypoints_body"] = np.asarray(data["keypoints_body"], dtype=np.float32)
        return (agent_x, agent_y, scene_pos, extra) if return_extra else (agent_x, agent_y, scene_pos)

    elif ext == ".pt":
        try:
            import torch
            from torch_geometric.data import HeteroData  # noqa: F401
        except ImportError:
            raise ImportError(
                "Para visualizar archivos .pt se requiere 'torch' y 'torch_geometric'. "
                "Instala las librerias con: pip install torch torch_geometric\n"
                "O genera/visualiza el archivo en formato .npz."
            )

        data = torch.load(file_path, map_location="cpu", weights_only=False)
        if hasattr(data, "__getitem__") and "agent" in data and "scene" in data:
            ag = data["agent"]
            agent_x = _to_numpy(getattr(ag, "pos_enu", getattr(ag, "x", None)))
            agent_y = _to_numpy(getattr(ag, "y", None))
            scene_pos = _to_numpy(data["scene"].pos)
            action_se3 = _to_numpy(getattr(ag, "action_se3", None))
            extra["action_horizon_flu"] = _extract_local_horizon_flu(agent_y, action_se3)
            extra["keyframe_indices"] = _to_numpy(getattr(ag, "keyframe_indices", None))
            extra["flight_state"] = _to_numpy(getattr(ag, "flight_state", None))
            extra["setpoint_raw"] = _to_numpy(getattr(ag, "setpoint_raw", None))
            extra["yaw_enu"] = _to_numpy(getattr(ag, "yaw_enu", None))
            extra["keypoints_body"] = _to_numpy(getattr(ag, "keypoints_body", None))
        elif isinstance(data, dict):
            agent_x = _to_numpy(data.get("pos_enu", data.get("agent_x", data.get("x"))))
            agent_y = _to_numpy(data.get("agent_y", data.get("y")))
            scene_pos = _to_numpy(data.get("scene_pos", data.get("pos")))
            action_se3 = _to_numpy(data.get("action_se3"))
            extra["action_horizon_flu"] = _extract_local_horizon_flu(agent_y, action_se3)
            extra["keyframe_indices"] = _to_numpy(data.get("keyframe_indices"))
            extra["flight_state"] = _to_numpy(data.get("flight_state"))
            extra["setpoint_raw"] = _to_numpy(data.get("setpoint_raw"))
            extra["yaw_enu"] = _to_numpy(data.get("yaw_enu"))
            extra["keypoints_body"] = _to_numpy(data.get("keypoints_body"))
        else:
            raise ValueError(f"Estructura no reconocida en {file_path}")

        return (agent_x, agent_y, scene_pos, extra) if return_extra else (agent_x, agent_y, scene_pos)

    else:
        raise ValueError(f"Extension de archivo no soportada: {ext}. Debe ser .pt o .npz")


def plot_pointcloud_and_trajectory(
    cloud: np.ndarray,
    agent_x: np.ndarray = None,
    agent_y: np.ndarray = None,
    current_step: int = 0,
    scale: float = 0.1,
    title: str = "Visualizacion LiDAR 360° Egocentrico (Horizontal FLU)",
    output_image: Path = None,
    interactive: bool = True,
    extra: dict = None
):
    """
    Genera un panel de 3 vistas (3D Perspectiva, Vista Superior XY, Vista Lateral XZ),
    incluyendo el horizonte de acciones relativas SE(3) (T=8) si esta disponible.
    """
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    horizon_step = None
    if extra and extra.get("action_horizon_flu") is not None:
        h_all = extra["action_horizon_flu"]
        if 0 <= current_step < len(h_all):
            # Anteceder (0,0,0) para dibujar la polilinea saliendo del centro del dron
            horizon_step = np.vstack([np.zeros((1, 3), dtype=np.float32), h_all[current_step]])

    # 1. Subplot 3D Perspectiva
    ax3d = fig.add_subplot(131, projection="3d")

    dists = np.linalg.norm(cloud, axis=1)
    scatter = ax3d.scatter(
        cloud[:, 0], cloud[:, 1], cloud[:, 2],
        c=dists, cmap="viridis", s=6, alpha=0.7, label=f"LiDAR ({len(cloud)} pts)"
    )

    ax3d.scatter([0], [0], [0], color="red", s=80, marker="^", label="Dron (0,0,0)")
    if horizon_step is not None:
        ax3d.plot(
            horizon_step[:, 0], horizon_step[:, 1], horizon_step[:, 2],
            "m-o", markersize=4, linewidth=2.0, label=f"Horizonte SE(3) (T={len(horizon_step)-1})"
        )

    axis_len = 0.3  # en espacio escalado (equivale a 3m sin escalar)
    ax3d.quiver(0, 0, 0, axis_len, 0, 0, color="r", arrow_length_ratio=0.15, label="+X (Adelante)")
    ax3d.quiver(0, 0, 0, 0, axis_len, 0, color="g", arrow_length_ratio=0.15, label="+Y (Izquierda)")
    ax3d.quiver(0, 0, 0, 0, 0, axis_len, color="b", arrow_length_ratio=0.15, label="+Z (Arriba)")

    ax3d.set_xlabel("X FLU (Adelante) [m·s]")
    ax3d.set_ylabel("Y FLU (Izquierda) [m·s]")
    ax3d.set_zlabel("Z FLU (Arriba) [m·s]")
    ax3d.set_title("Perspectiva 3D (Marco Dron FLU)")
    ax3d.legend(loc="upper right", fontsize=8)
    fig.colorbar(scatter, ax=ax3d, fraction=0.03, pad=0.04, label="Distancia relativa")

    # 2. Subplot 2D Vista Superior (Plano XY)
    ax_xy = fig.add_subplot(132)
    ax_xy.scatter(cloud[:, 0], cloud[:, 1], c=dists, cmap="viridis", s=8, alpha=0.6)
    ax_xy.scatter([0], [0], color="red", s=100, marker="^", label="Dron")
    if horizon_step is not None:
        ax_xy.plot(
            horizon_step[:, 0], horizon_step[:, 1],
            "m-o", markersize=4, linewidth=2.0, label="Horizonte SE(3)"
        )
    ax_xy.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax_xy.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax_xy.quiver(0, 0, axis_len, 0, color="r", scale=1, scale_units="xy", angles="xy", label="Frente (+X)")
    ax_xy.quiver(0, 0, 0, axis_len, color="g", scale=1, scale_units="xy", angles="xy", label="Izq (+Y)")
    ax_xy.set_xlabel("X FLU [m·s] (Adelante ->)")
    ax_xy.set_ylabel("Y FLU [m·s] (<- Izq / Der ->)")
    ax_xy.set_title("Vista Superior (Plano XY)")
    ax_xy.grid(True, linestyle=":", alpha=0.6)
    ax_xy.axis("equal")
    ax_xy.legend(loc="upper left", fontsize=8)

    # 3. Subplot 2D Vista Lateral (Plano XZ)
    ax_xz = fig.add_subplot(133)
    ax_xz.scatter(cloud[:, 0], cloud[:, 2], c=dists, cmap="viridis", s=8, alpha=0.6)
    ax_xz.scatter([0], [0], color="red", s=100, marker="^", label="Dron")
    if horizon_step is not None:
        ax_xz.plot(
            horizon_step[:, 0], horizon_step[:, 2],
            "m-o", markersize=4, linewidth=2.0, label="Horizonte SE(3)"
        )
    ax_xz.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax_xz.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax_xz.quiver(0, 0, axis_len, 0, color="r", scale=1, scale_units="xy", angles="xy", label="Frente (+X)")
    ax_xz.quiver(0, 0, 0, axis_len, color="b", scale=1, scale_units="xy", angles="xy", label="Arriba (+Z)")
    ax_xz.set_xlabel("X FLU [m·s] (Adelante ->)")
    ax_xz.set_ylabel("Z FLU [m·s] (Arriba ->)")
    ax_xz.set_title("Vista Lateral (Plano XZ)")
    ax_xz.grid(True, linestyle=":", alpha=0.6)
    ax_xz.axis("equal")
    ax_xz.legend(loc="upper left", fontsize=8)

    plt.tight_layout()

    if output_image:
        output_image = Path(output_image)
        output_image.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_image, dpi=180)
        print(f"\n[OK] Grafica guardada con exito en: {output_image.resolve()}")

    if interactive and has_display:
        print("[INFO] Mostrando ventana interactiva de Matplotlib...")
        plt.show()
    elif not has_display and not output_image:
        default_out = Path("visualizacion_lidar.png")
        plt.savefig(default_out, dpi=180)
        print(f"\n[INFO] Entorno sin pantalla grafica detectado. Grafica guardada automaticamente en: {default_out.resolve()}")


def plot_trajectory_global(
    agent_x: np.ndarray,
    agent_y: np.ndarray = None,
    output_image: Path = None,
    interactive: bool = True,
    extra: dict = None
):
    """
    Grafica la trayectoria global del vuelo (posicion real, keyframes L=10 y setpoints).
    Soporta tanto el nuevo formato SE(3) como archivos legados (T, 3).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(agent_x[:, 0], agent_x[:, 1], agent_x[:, 2], "b-o", markersize=3, label="Trayectoria ejecutada (pos_enu)")

    # Si existe setpoint_raw en extra o agent_y es legado (T, 3), graficarlo
    sp_raw = extra.get("setpoint_raw") if extra else None
    if sp_raw is not None and sp_raw.ndim == 2 and sp_raw.shape[1] == 3:
        ax.plot(sp_raw[:, 0], sp_raw[:, 1], sp_raw[:, 2], "r--", alpha=0.5, label="Setpoint controlador (raw)")
    elif agent_y is not None and agent_y.ndim == 2 and agent_y.shape[1] == 3:
        ax.plot(agent_y[:, 0], agent_y[:, 1], agent_y[:, 2], "r--", alpha=0.7, label="Acciones / Setpoints (agent.y)")

    # Resaltar keyframes geometricos (L=10) si estan presentes
    kf_idx = extra.get("keyframe_indices") if extra else None
    if kf_idx is not None and len(kf_idx) > 0:
        valid_kf = [int(i) for i in kf_idx if 0 <= int(i) < len(agent_x)]
        if valid_kf:
            kf_pts = agent_x[valid_kf]
            ax.scatter(
                kf_pts[:, 0], kf_pts[:, 1], kf_pts[:, 2],
                color="gold", edgecolors="black", s=140, marker="*", zorder=10,
                label=f"Keyframes contexto (L={len(valid_kf)})"
            )

    ax.scatter([agent_x[0, 0]], [agent_x[0, 1]], [agent_x[0, 2]], color="green", s=100, label="Inicio")
    ax.scatter([agent_x[-1, 0]], [agent_x[-1, 1]], [agent_x[-1, 2]], color="black", s=100, marker="x", label="Fin")

    ax.set_xlabel("X [m·s]")
    ax.set_ylabel("Y [m·s]")
    ax.set_zlabel("Z [m·s]")
    ax.set_title("Trayectoria del Vuelo Registrada en HeteroData (s=0.1)")
    ax.legend()
    plt.tight_layout()

    if output_image:
        output_image = Path(output_image)
        plt.savefig(output_image, dpi=180)
        print(f"[OK] Trayectoria global guardada en: {output_image.resolve()}")

    if interactive and has_display:
        plt.show()


def animate_pointcloud_and_trajectory(
    agent_x: np.ndarray,
    agent_y: np.ndarray,
    scene_pos: np.ndarray,
    scale: float = 0.1,
    title: str = "Vuelo Animado Instant Policy (HeteroData)",
    output_anim: Path = None,
    fps: int = 15,
    interactive: bool = True,
    extra: dict = None
):
    """
    Genera una animacion sincronizada de la trayectoria global (con L=10 keyframes)
    y las nubes LiDAR 360° egocentricas (con el horizonte de acciones relativas T=8).
    """
    T = len(scene_pos)
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    horizon_all = extra.get("action_horizon_flu") if extra else None
    kf_idx = extra.get("keyframe_indices") if extra else None
    flight_state = extra.get("flight_state") if extra else None

    # 1. Subplot 3D Perspectiva: Trayectoria Global
    ax3d = fig.add_subplot(131, projection="3d")
    ax3d.plot(agent_x[:, 0], agent_x[:, 1], agent_x[:, 2], color="gray", linestyle="--", alpha=0.5, label="Ruta completa")
    if kf_idx is not None and len(kf_idx) > 0:
        valid_kf = [int(i) for i in kf_idx if 0 <= int(i) < len(agent_x)]
        if valid_kf:
            kf_pts = agent_x[valid_kf]
            ax3d.scatter(
                kf_pts[:, 0], kf_pts[:, 1], kf_pts[:, 2],
                color="gold", edgecolors="black", s=90, marker="*", alpha=0.9,
                label=f"Keyframes (L={len(valid_kf)})"
            )
    line_history, = ax3d.plot([], [], [], "b-", linewidth=2, label="Recorrido actual")
    drone_marker = ax3d.scatter([agent_x[0, 0]], [agent_x[0, 1]], [agent_x[0, 2]], color="red", s=100, marker="^", label="Dron")

    ax3d.set_xlabel("X ENU [m·s]")
    ax3d.set_ylabel("Y ENU [m·s]")
    ax3d.set_zlabel("Z ENU [m·s]")
    ax3d.set_title("Trayectoria 3D Global")

    x_min, x_max = float(np.min(agent_x[:, 0])) - 0.2, float(np.max(agent_x[:, 0])) + 0.2
    y_min, y_max = float(np.min(agent_x[:, 1])) - 0.2, float(np.max(agent_x[:, 1])) + 0.2
    z_min, z_max = min(0.0, float(np.min(agent_x[:, 2])) - 0.1), float(np.max(agent_x[:, 2])) + 0.2
    ax3d.set_xlim(x_min, x_max)
    ax3d.set_ylim(y_min, y_max)
    ax3d.set_zlim(z_min, z_max)
    ax3d.legend(loc="upper left", fontsize=8)

    # 2. Subplot 2D: Vista Superior Egocentrica (Plano XY)
    ax_xy = fig.add_subplot(132)
    cloud_0 = scene_pos[0]
    dists_0 = np.linalg.norm(cloud_0, axis=1)
    scat_xy = ax_xy.scatter(cloud_0[:, 0], cloud_0[:, 1], c=dists_0, cmap="viridis", s=8, alpha=0.7)
    ax_xy.scatter([0], [0], color="red", s=100, marker="^", label="Dron (0,0)")
    line_h_xy, = ax_xy.plot([], [], "m-o", markersize=3, linewidth=1.8, label="Horizonte T=8")
    ax_xy.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax_xy.axvline(0, color="gray", linestyle="--", alpha=0.5)
    axis_len = 0.3
    ax_xy.quiver(0, 0, axis_len, 0, color="r", scale=1, scale_units="xy", angles="xy", label="+X Adelante")
    ax_xy.quiver(0, 0, 0, axis_len, color="g", scale=1, scale_units="xy", angles="xy", label="+Y Izquierda")
    ax_xy.set_xlim(-1.1, 1.1)
    ax_xy.set_ylim(-1.1, 1.1)
    ax_xy.set_xlabel("X FLU [m·s] (Adelante ->)")
    ax_xy.set_ylabel("Y FLU [m·s] (<- Izq / Der ->)")
    ax_xy.set_title("LiDAR 360° Egocéntrico (Plano XY)")
    ax_xy.grid(True, linestyle=":", alpha=0.6)
    ax_xy.axis("equal")
    ax_xy.legend(loc="upper left", fontsize=8)

    # 3. Subplot 2D: Vista Lateral Egocentrica (Plano XZ)
    ax_xz = fig.add_subplot(133)
    scat_xz = ax_xz.scatter(cloud_0[:, 0], cloud_0[:, 2], c=dists_0, cmap="viridis", s=8, alpha=0.7)
    ax_xz.scatter([0], [0], color="red", s=100, marker="^", label="Dron (0,0)")
    line_h_xz, = ax_xz.plot([], [], "m-o", markersize=3, linewidth=1.8, label="Horizonte T=8")
    ax_xz.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax_xz.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax_xz.quiver(0, 0, axis_len, 0, color="r", scale=1, scale_units="xy", angles="xy", label="+X Adelante")
    ax_xz.quiver(0, 0, 0, axis_len, color="b", scale=1, scale_units="xy", angles="xy", label="+Z Arriba")
    ax_xz.set_xlim(-1.1, 1.1)
    ax_xz.set_ylim(-0.8, 0.8)
    ax_xz.set_xlabel("X FLU [m·s] (Adelante ->)")
    ax_xz.set_ylabel("Z FLU [m·s] (Arriba ->)")
    ax_xz.set_title("LiDAR 360° Egocéntrico (Plano XZ)")
    ax_xz.grid(True, linestyle=":", alpha=0.6)
    ax_xz.axis("equal")
    ax_xz.legend(loc="upper left", fontsize=8)

    plt.tight_layout()

    def update(frame):
        line_history.set_data(agent_x[:frame+1, 0], agent_x[:frame+1, 1])
        line_history.set_3d_properties(agent_x[:frame+1, 2])
        drone_marker._offsets3d = ([agent_x[frame, 0]], [agent_x[frame, 1]], [agent_x[frame, 2]])

        cloud = scene_pos[frame]
        dists = np.linalg.norm(cloud, axis=1)

        scat_xy.set_offsets(cloud[:, :2])
        scat_xy.set_array(dists)

        scat_xz.set_offsets(np.column_stack([cloud[:, 0], cloud[:, 2]]))
        scat_xz.set_array(dists)

        if horizon_all is not None and frame < len(horizon_all):
            h_pts = np.vstack([np.zeros((1, 3), dtype=np.float32), horizon_all[frame]])
            line_h_xy.set_data(h_pts[:, 0], h_pts[:, 1])
            line_h_xz.set_data(h_pts[:, 0], h_pts[:, 2])

        alt_m = agent_x[frame, 2] / scale
        x_m = agent_x[frame, 0] / scale
        s_str = ""
        if flight_state is not None and frame < len(flight_state):
            s_val = float(flight_state[frame].reshape(-1)[0])
            s_str = f" | Estado s_t={int(round(s_val))}"
        fig.suptitle(
            f"{title} | Paso {frame}/{T-1} | X: {x_m:.1f} m | Altitud: {alt_m:.2f} m{s_str}",
            fontsize=14, fontweight="bold"
        )
        return line_history, drone_marker, scat_xy, scat_xz, line_h_xy, line_h_xz

    anim = FuncAnimation(fig, update, frames=T, interval=1000 // max(1, fps), blit=False)

    if output_anim:
        output_anim = Path(output_anim)
        output_anim.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Renderizando animacion a {output_anim.resolve()} (FPS={fps}, frames={T})...")
        anim.save(output_anim, writer=PillowWriter(fps=fps))
        print(f"[OK] Animacion guardada exitosamente en: {output_anim.resolve()}")

    if interactive and has_display:
        print("[INFO] Mostrando reproduccion animada interactiva...")
        plt.show()
    elif not has_display and not output_anim:
        default_out = Path("animacion_vuelo.gif")
        print(f"[INFO] Entorno sin pantalla detectado. Guardando GIF animado en: {default_out.resolve()}...")
        anim.save(default_out, writer=PillowWriter(fps=fps))
        print(f"[OK] Animacion guardada en: {default_out.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualizador 3D para HeteroData y LiDAR Sintetico 360°"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file", "-f",
        type=Path,
        help="Ruta al archivo .pt (HeteroData) o .npz (NumPy) generado por bag_to_heterodata.py"
    )
    group.add_argument(
        "--waypoints", "-w",
        type=Path,
        help="Ruta a waypoints.yaml para visualizar el LiDAR sintetico directamente sin volar"
    )

    parser.add_argument(
        "--drone-pose",
        nargs=4,
        type=float,
        metavar=("X", "Y", "Z", "YAW"),
        default=[0.0, 0.0, 1.5, 0.0],
        help="Pose simulada del dron [X Y Z Yaw(rad)] para modo --waypoints (default: 0 0 1.5 0)"
    )
    parser.add_argument(
        "--step", "-t",
        type=int,
        default=0,
        help="Indice del paso temporal a visualizar en archivos .pt / .npz (default: 0)"
    )
    parser.add_argument(
        "--animate", "-a",
        action="store_true",
        help="Reproducir o guardar una animacion continua tipo video del vuelo completo y del LiDAR"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Cuadros por segundo para la animacion (default: 15)"
    )
    parser.add_argument(
        "--r-max",
        type=float,
        default=10.0,
        help="Alcance del LiDAR en metros (default: 10.0)"
    )
    parser.add_argument(
        "--scale", "-s",
        type=float,
        default=0.1,
        help="Factor de escala metrica s (default: 0.1)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Ruta donde guardar la imagen resultante (.png) o animacion (.gif)"
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="No abrir ventana interactiva, solo generar y guardar la imagen o animacion"
    )

    args = parser.parse_args()

    # 1. Modo A: Inspeccionar archivo generado por bag_to_heterodata (.pt / .npz)
    if args.file:
        if not args.file.is_file():
            print(f"ERROR: Archivo no encontrado: {args.file}")
            sys.exit(1)

        print(f"Cargando dataset desde: {args.file}...")
        agent_x, agent_y, scene_pos, extra = load_dataset_file(args.file, return_extra=True)

        T = len(scene_pos)
        step = max(0, min(args.step, T - 1))
        cloud = scene_pos[step]

        print("\n--- Informacion del Archivo HeteroData ---")
        print(f"Pasos temporales (T):           {T}")
        print(f"Paso seleccionado:              t = {step}")
        print(f"Dimensiones de nubes LiDAR:     {scene_pos.shape}  (T, puntos, 3)")
        print(f"Dimensiones de trayectoria x:   {agent_x.shape}    (T, 3)")
        if agent_y is not None:
            print(f"Dimensiones de acciones y:      {agent_y.shape}")
        if extra.get("keyframe_indices") is not None:
            print(f"Keyframes de contexto (L={len(extra['keyframe_indices'])}):   {extra['keyframe_indices'].tolist()}")
        if extra.get("keypoints_body") is not None:
            print(f"Keypoints cuerpo UAV:           {extra['keypoints_body'].shape}")
        print(f"Rango de nube t={step}:         Min={np.min(cloud, axis=0)}, Max={np.max(cloud, axis=0)}")
        print(f"Distancia maxima del LiDAR:     {np.max(np.linalg.norm(cloud, axis=1)):.3f} (en espacio s={args.scale})")
        print("------------------------------------------")

        is_anim = args.animate or (args.output is not None and args.output.suffix.lower() in [".gif", ".mp4"])

        if is_anim:
            title = f"Vuelo Animado: {args.file.name}"
            animate_pointcloud_and_trajectory(
                agent_x=agent_x,
                agent_y=agent_y,
                scene_pos=scene_pos,
                scale=args.scale,
                title=title,
                output_anim=args.output,
                fps=args.fps,
                interactive=not args.no_gui,
                extra=extra
            )
        else:
            title = f"HeteroData: {args.file.name} (Paso t={step}/{T-1})"
            plot_pointcloud_and_trajectory(
                cloud=cloud,
                agent_x=agent_x,
                agent_y=agent_y,
                current_step=step,
                scale=args.scale,
                title=title,
                output_image=args.output,
                interactive=not args.no_gui,
                extra=extra
            )

    # 2. Modo B: Inspeccionar LiDAR sintetico directamente desde waypoints.yaml
    elif args.waypoints:
        if not args.waypoints.is_file():
            print(f"ERROR: Archivo no encontrado: {args.waypoints}")
            sys.exit(1)

        if SyntheticLidarGenerator is None:
            print("ERROR: No se pudo importar SyntheticLidarGenerator desde synthetic_lidar.py")
            sys.exit(1)

        import yaml
        with open(args.waypoints, "r") as f:
            cfg = yaml.safe_load(f)

        raw_lms = cfg.get("landmarks") or cfg.get("hoops") or []
        landmarks = []
        for lm in raw_lms:
            lm_c = dict(lm)
            if "id" not in lm_c and "name" in lm_c:
                lm_c["id"] = lm_c["name"]
            if "type" not in lm_c:
                lm_c["type"] = "square_gate"
            landmarks.append(lm_c)
        landing = cfg.get("landing", None)

        gen = SyntheticLidarGenerator(
            r_max=args.r_max,
            scale_factor=args.scale,
            target_points=2048
        )

        drone_pose = {
            "x": args.drone_pose[0],
            "y": args.drone_pose[1],
            "z": args.drone_pose[2],
            "yaw": args.drone_pose[3]
        }

        print(f"Generando nube sintetica desde {args.waypoints}...")
        print(f"Pose del dron (ENU): X={drone_pose['x']}, Y={drone_pose['y']}, Z={drone_pose['z']}, Yaw={drone_pose['yaw']} rad")
        print(f"Landmarks cargados: {len(landmarks)}")

        cloud = gen.generate_pointcloud(drone_pose, landmarks, landing)

        print("\n--- Informacion de la Nube Sintetica ---")
        print(f"Shape:               {cloud.shape}")
        print(f"Limites Min (FLU):   {np.min(cloud, axis=0)}")
        print(f"Limites Max (FLU):   {np.max(cloud, axis=0)}")
        print(f"Distancia Maxima:    {np.max(np.linalg.norm(cloud, axis=1)):.3f}")
        print("---------------------------------------")

        title = f"LiDAR Sintetico 360° desde {args.waypoints.name} (Dron en {args.drone_pose[:3]})"
        plot_pointcloud_and_trajectory(
            cloud=cloud,
            scale=args.scale,
            title=title,
            output_image=args.output,
            interactive=not args.no_gui
        )


if __name__ == "__main__":
    main()
