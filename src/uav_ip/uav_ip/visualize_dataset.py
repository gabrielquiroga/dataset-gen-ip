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
    from synthetic_lidar import SyntheticLidarGenerator
except ImportError:
    SyntheticLidarGenerator = None


def load_dataset_file(file_path: Path):
    """
    Carga un archivo .pt (HeteroData) o .npz (NumPy) y extrae:
    - agent_x: (T, 3) posiciones del dron (escaladas s=0.1)
    - agent_y: (T, 3) acciones / setpoints futuros (escaladas s=0.1)
    - scene_pos: (T, 2048, 3) nubes de puntos LiDAR egocentricas (escaladas s=0.1)
    """
    ext = file_path.suffix.lower()

    if ext == ".npz":
        data = np.load(file_path)
        agent_x = data["agent_x"]
        agent_y = data["agent_y"]
        scene_pos = data["scene_pos"]
        return agent_x, agent_y, scene_pos

    elif ext == ".pt":
        try:
            import torch
            from torch_geometric.data import HeteroData
        except ImportError:
            # Si torch no esta en el interprete actual, intentar cargar con numpy si fue guardado via pickle
            raise ImportError(
                "Para visualizar archivos .pt se requiere 'torch' y 'torch_geometric'. "
                "Instala las librerias con: pip install torch torch_geometric\n"
                "O genera/visualiza el archivo en formato .npz."
            )

        data = torch.load(file_path, map_location="cpu", weights_only=False)
        if hasattr(data, "__getitem__") and "agent" in data and "scene" in data:
            # Objeto HeteroData
            agent_x = data["agent"].x.detach().cpu().numpy()
            agent_y = data["agent"].y.detach().cpu().numpy()
            scene_pos = data["scene"].pos.detach().cpu().numpy()
        elif isinstance(data, dict):
            agent_x = data.get("agent_x", data.get("x")).detach().cpu().numpy()
            agent_y = data.get("agent_y", data.get("y")).detach().cpu().numpy()
            scene_pos = data.get("scene_pos", data.get("pos")).detach().cpu().numpy()
        else:
            raise ValueError(f"Estructura no reconocida en {file_path}")

        return agent_x, agent_y, scene_pos

    else:
        raise ValueError(f"Extension de archivo no soportada: {ext}. Debe ser .pt o .npz")


def plot_pointcloud_and_trajectory(
    cloud: np.ndarray,
    agent_x: np.ndarray = None,
    agent_y: np.ndarray = None,
    current_step: int = 0,
    scale: float = 0.1,
    title: str = "Visualizacion LiDAR 360° Egocentrico (FLU)",
    output_image: Path = None,
    interactive: bool = True
):
    """
    Genera un panel de 3 vistas (3D Perspectiva, Vista Superior XY, Vista Lateral XZ).
    """
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1. Subplot 3D Perspectiva
    ax3d = fig.add_subplot(131, projection="3d")
    
    # Graficar nube LiDAR egocentrica
    # Asignar color segun distancia al dron (origen 0,0,0)
    dists = np.linalg.norm(cloud, axis=1)
    scatter = ax3d.scatter(
        cloud[:, 0], cloud[:, 1], cloud[:, 2],
        c=dists, cmap="viridis", s=6, alpha=0.7, label=f"LiDAR ({len(cloud)} pts)"
    )

    # Graficar origen del dron (0,0,0) y ejes locales FLU
    ax3d.scatter([0], [0], [0], color="red", s=80, marker="^", label="Dron (0,0,0)")
    
    # Ejes locales (Forward: Rojo, Left: Verde, Up: Azul)
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
        # Si no hay display y no paso --output, guardar por defecto en visualizacion_lidar.png
        default_out = Path("visualizacion_lidar.png")
        plt.savefig(default_out, dpi=180)
        print(f"\n[INFO] Entorno sin pantalla grafica detectado. Grafica guardada automaticamente en: {default_out.resolve()}")


def plot_trajectory_global(
    agent_x: np.ndarray,
    agent_y: np.ndarray,
    output_image: Path = None,
    interactive: bool = True
):
    """
    Grafica la trayectoria global del vuelo (posicion real vs setpoints).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(agent_x[:, 0], agent_x[:, 1], agent_x[:, 2], "b-o", markersize=3, label="Trayectoria ejecutada (agent.x)")
    if agent_y is not None:
        ax.plot(agent_y[:, 0], agent_y[:, 1], agent_y[:, 2], "r--", alpha=0.7, label="Acciones / Setpoints (agent.y)")

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
    interactive: bool = True
):
    """
    Genera una animacion sincronizada de la trayectoria global y las nubes LiDAR 360° egocentricas.
    Subplots sincronizados en cada frame t:
    1. Perspectiva 3D: Trayectoria global + posicion actual del dron.
    2. Plano Superior 2D (XY): Vista del sensor desde el dron (adelante/lados).
    3. Plano Lateral 2D (XZ): Vista de altura y sobrevuelo (overfly/slalom).
    """
    T = len(scene_pos)
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1. Subplot 3D Perspectiva: Trayectoria Global
    ax3d = fig.add_subplot(131, projection="3d")
    ax3d.plot(agent_x[:, 0], agent_x[:, 1], agent_x[:, 2], color="gray", linestyle="--", alpha=0.5, label="Ruta completa")
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

        alt_m = agent_x[frame, 2] / scale
        x_m = agent_x[frame, 0] / scale
        fig.suptitle(f"{title} | Paso {frame}/{T-1} | X: {x_m:.1f} m | Altitud: {alt_m:.2f} m", fontsize=14, fontweight="bold")
        return line_history, drone_marker, scat_xy, scat_xz

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
        agent_x, agent_y, scene_pos = load_dataset_file(args.file)

        T = len(scene_pos)
        step = max(0, min(args.step, T - 1))
        cloud = scene_pos[step]

        print("\n--- Informacion del Archivo HeteroData ---")
        print(f"Pasos temporales (T):           {T}")
        print(f"Paso seleccionado:              t = {step}")
        print(f"Dimensiones de nubes LiDAR:     {scene_pos.shape}  (T, puntos, 3)")
        print(f"Dimensiones de trayectoria x:   {agent_x.shape}    (T, 3)")
        print(f"Dimensiones de setpoints y:     {agent_y.shape}    (T, 3)")
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
                interactive=not args.no_gui
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
                interactive=not args.no_gui
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
