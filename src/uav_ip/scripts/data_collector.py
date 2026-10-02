#!/usr/bin/env python3
"""
data_collector.py — Orquestador Automatico de Recoleccion de Datos.

Automatiza el bucle doble (N tareas x M episodios) para generar el dataset
de Instant Policy:
  1. Genera mundo aleatorio (world_generator.py)
  2. Lanza PX4 SITL + Gazebo con el mundo generado
  3. Lanza MicroXRCEAgent, ros_gz_bridge
  4. Graba rosbag con los topicos relevantes
  5. Ejecuta el piloto experto
  6. Detecta finalizacion exitosa (exit code 0)
  7. Cierra todo limpiamente y repite

Uso:
    python3 data_collector.py
    python3 data_collector.py --num-tasks 5 --episodes-per-task 10
    python3 data_collector.py --resume  # continua desde la ultima tarea incompleta

IMPORTANTE: Ejecutar DESDE DENTRO de WSL2 (no desde PowerShell).
            Requiere: source /opt/ros/jazzy/setup.bash
                      source ~/ros2_ws/install/setup.bash
                      export ROS_DOMAIN_ID=33
"""

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


# ============================================================================
# Configuracion
# ============================================================================

# Rutas del proyecto
ROS2_WS = Path.home() / "ros2_ws"
UAV_IP_PKG = ROS2_WS / "src" / "uav_ip"
WORLD_GENERATOR = UAV_IP_PKG / "scripts" / "world_generator.py"
GENERATED_WORLDS = UAV_IP_PKG / "generated_worlds"
PX4_DIR = Path.home() / "PX4-Autopilot"

# Directorio de salida del dataset
DATASET_DIR = ROS2_WS / "datasets"

# Topicos a grabar
RECORD_TOPICS = [
    "/fmu/out/vehicle_odometry",
    "/fmu/out/vehicle_local_position_v1",
    "/fmu/in/trajectory_setpoint",
    "/depth_camera/points",
]

# Timeouts (segundos)
PX4_STARTUP_TIMEOUT = 50       # Tiempo para que PX4 + Gazebo arranquen
XRCE_STARTUP_DELAY = 3         # Espera tras lanzar XRCE agent
BRIDGE_STARTUP_DELAY = 3       # Espera tras lanzar el bridge
PRE_FLIGHT_SETTLE = 5          # Espera antes de iniciar el vuelo
EPISODE_TIMEOUT = 180          # Timeout maximo por episodio (3 min)
POST_EPISODE_COOLDOWN = 5      # Espera tras cerrar procesos
PROCESS_KILL_TIMEOUT = 10      # Tiempo para que un proceso muera con SIGINT

# Variables de entorno para PX4
PX4_MODEL_POSE = "0,0,0.2,0,0,0"
GZ_VERSION = "harmonic"
PX4_VEHICLE = "gz_x500_depth"


# ============================================================================
# Gestion de procesos
# ============================================================================

class ProcessManager:
    """Gestiona procesos hijos con cleanup robusto via process groups."""

    def __init__(self):
        self.processes: dict[str, subprocess.Popen] = {}

    def start(
        self,
        name: str,
        cmd: list[str] | str,
        env: Optional[dict] = None,
        cwd: Optional[Path] = None,
        shell: bool = False,
        log_file: Optional[Path] = None,
    ) -> subprocess.Popen:
        """Inicia un proceso en un nuevo process group.

        Args:
            name: Nombre identificador del proceso.
            cmd: Comando a ejecutar.
            env: Variables de entorno (se fusionan con os.environ).
            cwd: Directorio de trabajo.
            shell: Si usar shell=True.
            log_file: Si se especifica, stdout/stderr van a este archivo.
        """
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            stdout = open(log_file, "w")
            stderr = subprocess.STDOUT
        else:
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL

        proc = subprocess.Popen(
            cmd,
            env=full_env,
            cwd=cwd,
            shell=shell,
            stdout=stdout,
            stderr=stderr,
            # Crear nuevo process group para poder matar todo el arbol
            preexec_fn=os.setpgrp,
        )

        self.processes[name] = proc
        return proc

    def stop(self, name: str, sig: int = signal.SIGINT) -> Optional[int]:
        """Detiene un proceso enviando una señal y esperando.

        Args:
            name: Nombre del proceso.
            sig: Señal a enviar (SIGINT para cierre limpio, SIGKILL para forzar).

        Returns:
            Exit code del proceso, o None si no existia.
        """
        proc = self.processes.pop(name, None)
        if proc is None or proc.poll() is not None:
            return proc.returncode if proc else None

        try:
            # Enviar señal a todo el process group
            os.killpg(os.getpgid(proc.pid), sig)
            proc.wait(timeout=PROCESS_KILL_TIMEOUT)
        except subprocess.TimeoutExpired:
            # Forzar si no murio con la señal suave
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=5)
            except Exception:
                pass
        except ProcessLookupError:
            pass  # Ya murio

        return proc.returncode

    def stop_all(self) -> None:
        """Detiene todos los procesos en orden inverso al de creacion."""
        names = list(reversed(self.processes.keys()))
        for name in names:
            self.stop(name)

    def is_running(self, name: str) -> bool:
        """Verifica si un proceso sigue corriendo."""
        proc = self.processes.get(name)
        return proc is not None and proc.poll() is None

    def wait(self, name: str, timeout: float) -> Optional[int]:
        """Espera a que un proceso termine.

        Returns:
            Exit code, o None si hizo timeout.
        """
        proc = self.processes.get(name)
        if proc is None:
            return None
        try:
            proc.wait(timeout=timeout)
            return proc.returncode
        except subprocess.TimeoutExpired:
            return None


# ============================================================================
# Orquestador principal
# ============================================================================

class DataCollector:
    """Orquestador de recoleccion de datos para Instant Policy."""

    def __init__(
        self,
        num_tasks: int = 2,
        episodes_per_task: int = 15,
        base_seed: int = 0,
        dataset_dir: Path = DATASET_DIR,
        resume: bool = False,
        episode_timeout: int = EPISODE_TIMEOUT,
        px4_startup_wait: int = PX4_STARTUP_TIMEOUT,
    ):
        self.num_tasks = num_tasks
        self.episodes_per_task = episodes_per_task
        self.base_seed = base_seed
        self.dataset_dir = dataset_dir
        self.resume = resume
        self.episode_timeout = episode_timeout
        self.px4_startup_wait = px4_startup_wait
        self.pm = ProcessManager()

        # Contadores globales
        self.total_success = 0
        self.total_failed = 0
        self.total_timeout = 0

        # Handler para Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self._interrupted = False

    def _signal_handler(self, signum, frame):
        """Maneja Ctrl+C para limpiar procesos."""
        if self._interrupted:
            # Segunda vez: forzar salida
            print("\n[FORCE] Segunda interrupcion. Forzando salida...")
            self.pm.stop_all()
            sys.exit(1)
        self._interrupted = True
        print("\n[STOP] Interrupcion recibida. Limpiando procesos...")
        self.pm.stop_all()
        self._print_summary()
        sys.exit(0)

    # ----------------------------------------------------------------
    #  Flujo principal
    # ----------------------------------------------------------------

    def run(self) -> None:
        """Ejecuta el bucle completo de recoleccion."""
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.time()

        print("=" * 60)
        print("  INSTANT POLICY — Data Collector")
        print(f"  Tareas: {self.num_tasks}  |  Episodios/tarea: "
              f"{self.episodes_per_task}")
        print(f"  Total vuelos: {self.num_tasks * self.episodes_per_task}")
        print(f"  Dataset dir: {self.dataset_dir}")
        print("=" * 60)

        for task_idx in range(self.num_tasks):
            if self._interrupted:
                break

            task_id = f"task_{task_idx:04d}"
            task_dir = self.dataset_dir / task_id
            seed = self.base_seed + task_idx

            # Verificar si ya esta completa (resume)
            if self.resume and self._is_task_complete(task_dir):
                print(f"\n[SKIP] {task_id} ya completa. Saltando...")
                continue

            print(f"\n{'='*60}")
            print(f"  TAREA {task_idx + 1}/{self.num_tasks}: {task_id} "
                  f"(seed={seed})")
            print(f"{'='*60}")

            # Generar mundo
            if not self._generate_world(task_id, seed):
                print(f"[ERROR] Fallo generando mundo para {task_id}. "
                      "Saltando tarea.")
                continue

            # Copiar waypoints al directorio de la tarea
            task_dir.mkdir(parents=True, exist_ok=True)
            wp_src = GENERATED_WORLDS / task_id / "waypoints.yaml"
            wp_dst = task_dir / "waypoints.yaml"
            if wp_src.exists():
                import shutil
                shutil.copy2(wp_src, wp_dst)

            # Ejecutar episodios
            for ep_idx in range(self.episodes_per_task):
                if self._interrupted:
                    break

                ep_id = f"episode_{ep_idx:02d}"
                ep_dir = task_dir / ep_id

                # Resume: saltar episodios completos
                if self.resume and self._is_episode_complete(ep_dir):
                    print(f"  [SKIP] {ep_id} ya completo.")
                    continue

                print(f"\n  --- {task_id} / {ep_id} "
                      f"({ep_idx + 1}/{self.episodes_per_task}) ---")

                result = self._run_episode(task_id, ep_id, ep_dir)

                if result == "success":
                    self.total_success += 1
                    print(f"  [OK] {ep_id} completado exitosamente.")
                elif result == "timeout":
                    self.total_timeout += 1
                    print(f"  [TIMEOUT] {ep_id} excedio el limite de "
                          f"{EPISODE_TIMEOUT}s.")
                else:
                    self.total_failed += 1
                    print(f"  [FAIL] {ep_id} fallo (exit code: {result}).")

                # Cooldown entre episodios
                if not self._interrupted:
                    time.sleep(POST_EPISODE_COOLDOWN)

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  RECOLECCION COMPLETADA en {elapsed/60:.1f} min")
        self._print_summary()

    # ----------------------------------------------------------------
    #  Generacion de mundo
    # ----------------------------------------------------------------

    def _generate_world(self, task_id: str, seed: int) -> bool:
        """Ejecuta world_generator.py para crear el mundo."""
        print(f"  Generando mundo {task_id} (seed={seed})...")
        result = subprocess.run(
            [
                sys.executable,
                str(WORLD_GENERATOR),
                "--task-id", task_id,
                "--seed", str(seed),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  [ERROR] world_generator.py fallo:\n{result.stderr}")
            return False

        # Mostrar resumen
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                print(f"    {line.strip()}")
        return True

    # ----------------------------------------------------------------
    #  Ejecucion de un episodio
    # ----------------------------------------------------------------

    def _run_episode(
        self, task_id: str, ep_id: str, ep_dir: Path
    ) -> str:
        """Ejecuta un episodio completo.

        Returns:
            "success", "timeout", o el exit code como string.
        """
        ep_dir.mkdir(parents=True, exist_ok=True)
        log_dir = ep_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        waypoints_file = GENERATED_WORLDS / task_id / "waypoints.yaml"

        try:
            # 1. Lanzar XRCE-DDS Agent
            print("    Lanzando MicroXRCEAgent...")
            self._start_xrce_agent(log_dir / "xrce.log")
            time.sleep(XRCE_STARTUP_DELAY)

            # 2. Lanzar PX4 + Gazebo
            print(f"    Lanzando PX4 (world={task_id})...")
            self._start_px4(task_id, log_dir / "px4.log")
            time.sleep(self.px4_startup_wait)

            if not self.pm.is_running("px4"):
                return "px4_crash"

            # 3. Lanzar gz_bridge
            print("    Lanzando ros_gz_bridge...")
            self._start_gz_bridge(log_dir / "bridge.log")
            time.sleep(BRIDGE_STARTUP_DELAY)

            # 4. Espera de estabilizacion
            print(f"    Esperando {PRE_FLIGHT_SETTLE}s de estabilizacion...")
            time.sleep(PRE_FLIGHT_SETTLE)

            # 5. Iniciar grabacion de rosbag
            print("    Iniciando grabacion rosbag...")
            self._start_bag_record(ep_dir, log_dir / "bag.log")
            time.sleep(1)  # Dar tiempo al bag recorder para inicializar

            # 6. Lanzar piloto experto
            print("    Lanzando piloto experto...")
            self._start_expert_pilot(
                waypoints_file, log_dir / "pilot.log"
            )

            # 7. Esperar a que el piloto termine (o timeout)
            print(f"    Volando... (timeout={self.episode_timeout}s)")
            exit_code = self.pm.wait("pilot", timeout=self.episode_timeout)

            if exit_code is None:
                # Timeout
                return "timeout"
            elif exit_code == 0:
                # Guardar metadata del episodio
                self._save_episode_metadata(
                    ep_dir, task_id, ep_id, "success"
                )
                return "success"
            else:
                return str(exit_code)

        except Exception as e:
            print(f"    [EXCEPTION] {e}")
            return f"exception: {e}"

        finally:
            # Cleanup: detener todo en orden inverso
            print("    Cerrando procesos...")
            # Bag: SIGINT para cierre limpio del archivo
            self.pm.stop("bag", signal.SIGINT)
            time.sleep(1)
            self.pm.stop("pilot", signal.SIGINT)
            self.pm.stop("bridge", signal.SIGINT)
            self.pm.stop("xrce", signal.SIGINT)
            # PX4: SIGINT primero, luego cleanup
            self.pm.stop("px4", signal.SIGINT)
            # Matar cualquier proceso zombie de gazebo
            self._cleanup_gazebo()

    # ----------------------------------------------------------------
    #  Arranque de procesos individuales
    # ----------------------------------------------------------------

    def _start_px4(self, world_name: str, log_file: Path) -> None:
        """Lanza PX4 SITL con Gazebo."""
        env = {
            "GZ_VERSION": GZ_VERSION,
            "PX4_GZ_WORLD": world_name,
            "PX4_GZ_MODEL_POSE": PX4_MODEL_POSE,
        }
        self.pm.start(
            "px4",
            f"make px4_sitl {PX4_VEHICLE}",
            env=env,
            cwd=PX4_DIR,
            shell=True,
            log_file=log_file,
        )

    def _start_xrce_agent(self, log_file: Path) -> None:
        """Lanza MicroXRCEAgent."""
        self.pm.start(
            "xrce",
            ["MicroXRCEAgent", "udp4", "-p", "8888"],
            log_file=log_file,
        )

    def _start_gz_bridge(self, log_file: Path) -> None:
        """Lanza ros_gz_bridge para la nube de puntos."""
        self.pm.start(
            "bridge",
            [
                "ros2", "run", "ros_gz_bridge", "parameter_bridge",
                "/depth_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            ],
            log_file=log_file,
        )

    def _start_bag_record(self, ep_dir: Path, log_file: Path) -> None:
        """Inicia ros2 bag record."""
        bag_path = ep_dir / "rosbag"
        cmd = [
            "ros2", "bag", "record",
            "-o", str(bag_path),
            "--storage", "mcap",
            "--topics",
        ] + RECORD_TOPICS

        self.pm.start(
            "bag",
            cmd,
            log_file=log_file,
        )

    def _start_expert_pilot(
        self, waypoints_file: Path, log_file: Path
    ) -> None:
        """Lanza el nodo del piloto experto."""
        self.pm.start(
            "pilot",
            [
                "ros2", "run", "uav_ip", "expert_pilot",
                "--ros-args",
                "-p", f"waypoints_file:={waypoints_file}",
            ],
            log_file=log_file,
        )

    # ----------------------------------------------------------------
    #  Utilidades
    # ----------------------------------------------------------------

    def _cleanup_gazebo(self) -> None:
        """Mata procesos residuales de gazebo que pueden quedar huerfanos."""
        try:
            subprocess.run(
                ["pkill", "-f", "gz sim"],
                capture_output=True,
                timeout=5,
            )
            subprocess.run(
                ["pkill", "-f", "ruby.*gz"],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass

    def _save_episode_metadata(
        self,
        ep_dir: Path,
        task_id: str,
        ep_id: str,
        status: str,
    ) -> None:
        """Guarda metadata del episodio en JSON."""
        meta = {
            "task_id": task_id,
            "episode_id": ep_id,
            "status": status,
            "timestamp": datetime.datetime.now().isoformat(),
            "topics_recorded": RECORD_TOPICS,
            "episode_timeout_s": self.episode_timeout,
        }
        meta_path = ep_dir / "episode_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def _is_task_complete(self, task_dir: Path) -> bool:
        """Verifica si una tarea tiene todos sus episodios completos."""
        if not task_dir.is_dir():
            return False
        complete = 0
        for ep_idx in range(self.episodes_per_task):
            ep_dir = task_dir / f"episode_{ep_idx:02d}"
            if self._is_episode_complete(ep_dir):
                complete += 1
        return complete >= self.episodes_per_task

    def _is_episode_complete(self, ep_dir: Path) -> bool:
        """Verifica si un episodio fue completado exitosamente."""
        meta_path = ep_dir / "episode_meta.json"
        if not meta_path.exists():
            return False
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            return meta.get("status") == "success"
        except Exception:
            return False

    def _print_summary(self) -> None:
        """Imprime resumen de la recoleccion."""
        total = self.total_success + self.total_failed + self.total_timeout
        print(f"\n  Resumen: {total} episodios ejecutados")
        print(f"    Exitosos:  {self.total_success}")
        print(f"    Fallidos:  {self.total_failed}")
        print(f"    Timeouts:  {self.total_timeout}")
        if total > 0:
            rate = self.total_success / total * 100
            print(f"    Tasa de exito: {rate:.1f}%")
        print("=" * 60)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Orquestador automatico de recoleccion de datos "
        "para Instant Policy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Prerequisitos (ejecutar antes en la misma terminal):\n"
            "  source /opt/ros/jazzy/setup.bash\n"
            "  source ~/ros2_ws/install/setup.bash\n"
            "  export ROS_DOMAIN_ID=33\n"
            "\n"
            "Ejemplos:\n"
            "  python3 data_collector.py\n"
            "  python3 data_collector.py --num-tasks 5 --episodes-per-task 10\n"
            "  python3 data_collector.py --resume\n"
        ),
    )
    parser.add_argument(
        "--num-tasks", type=int, default=2,
        help="Numero de pseudo-tareas (mundos aleatorios). Default: 2.",
    )
    parser.add_argument(
        "--episodes-per-task", type=int, default=15,
        help="Episodios por tarea. Default: 15.",
    )
    parser.add_argument(
        "--base-seed", type=int, default=0,
        help="Semilla base para la generacion de mundos. "
        "Tarea N usa seed = base_seed + N. Default: 0.",
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=DATASET_DIR,
        help=f"Directorio de salida del dataset. Default: {DATASET_DIR}",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Continuar desde la ultima tarea/episodio incompleto.",
    )
    parser.add_argument(
        "--episode-timeout", type=int, default=EPISODE_TIMEOUT,
        help=f"Timeout maximo por episodio en segundos. Default: {EPISODE_TIMEOUT}.",
    )
    parser.add_argument(
        "--px4-startup-wait", type=int, default=PX4_STARTUP_TIMEOUT,
        help=f"Tiempo de espera para que PX4+Gazebo arranquen. "
        f"Default: {PX4_STARTUP_TIMEOUT}.",
    )

    args = parser.parse_args()

    # Validar prerequisitos
    _check_prerequisites()

    collector = DataCollector(
        num_tasks=args.num_tasks,
        episodes_per_task=args.episodes_per_task,
        base_seed=args.base_seed,
        dataset_dir=args.dataset_dir,
        resume=args.resume,
        episode_timeout=args.episode_timeout,
        px4_startup_wait=args.px4_startup_wait,
    )
    collector.run()


def _check_prerequisites() -> None:
    """Verifica que las dependencias esten disponibles."""
    errors = []

    # Verificar ROS 2
    if subprocess.run(
        ["which", "ros2"], capture_output=True
    ).returncode != 0:
        errors.append(
            "ros2 no encontrado. Ejecuta: source /opt/ros/jazzy/setup.bash"
        )

    # Verificar ROS_DOMAIN_ID
    if os.environ.get("ROS_DOMAIN_ID") != "33":
        errors.append(
            "ROS_DOMAIN_ID no es 33. Ejecuta: export ROS_DOMAIN_ID=33"
        )

    # Verificar colcon build
    install_dir = ROS2_WS / "install" / "uav_ip"
    if not install_dir.is_dir():
        errors.append(
            "Paquete uav_ip no instalado. Ejecuta:\n"
            "  cd ~/ros2_ws && colcon build --packages-select uav_ip && "
            "source install/setup.bash"
        )

    # Verificar PX4
    if not PX4_DIR.is_dir():
        errors.append(f"PX4-Autopilot no encontrado en {PX4_DIR}")

    # Verificar world_generator.py
    if not WORLD_GENERATOR.is_file():
        errors.append(f"world_generator.py no encontrado en {WORLD_GENERATOR}")

    # Verificar MicroXRCEAgent
    if subprocess.run(
        ["which", "MicroXRCEAgent"], capture_output=True
    ).returncode != 0:
        errors.append("MicroXRCEAgent no encontrado en PATH.")

    if errors:
        print("ERROR: Prerequisitos no cumplidos:\n")
        for e in errors:
            print(f"  - {e}")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
