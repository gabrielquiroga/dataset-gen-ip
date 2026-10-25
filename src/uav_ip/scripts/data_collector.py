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
from typing import Any, Dict, List, Optional

# Asegurar que el paquete local `uav_ip` sea importable desde scripts/
ROS2_WS = Path.home() / "ros2_ws"
UAV_IP_PKG = ROS2_WS / "src" / "uav_ip"
if str(UAV_IP_PKG) not in sys.path:
    sys.path.insert(0, str(UAV_IP_PKG))

from uav_ip.validate_episode import (  # noqa: E402
    VALIDATOR_VERSION,
    get_effective_validation_parameters,
    validate_episode_directory,
)


# ============================================================================
# Configuracion
# ============================================================================

# Rutas del proyecto
WORLD_GENERATOR = UAV_IP_PKG / "scripts" / "world_generator.py"
TASK_GENERATOR = UAV_IP_PKG / "scripts" / "task_generator.py"
GENERATED_WORLDS = UAV_IP_PKG / "generated_worlds"
PX4_DIR = Path.home() / "PX4-Autopilot"

# Directorio de salida del dataset
DATASET_DIR = ROS2_WS / "datasets"

# Topicos a grabar (telemetria ligera PX4 < 5 MB/demo, sin PointCloud2 pesada)
RECORD_TOPICS = [
    "/fmu/out/vehicle_odometry",
    "/fmu/in/trajectory_setpoint",
    "/fmu/out/vehicle_land_detected",
    "/fmu/out/vehicle_status_v4",
    "/fmu/out/vehicle_local_position_v1",
    "/fmu/out/timesync_status",
    "/fmu/in/vehicle_command",
]

REQUIRED_ACTIVE_PX4_TOPICS = [
    "/fmu/out/vehicle_odometry",
    "/fmu/out/vehicle_status_v4",
    "/fmu/out/vehicle_land_detected",
]
PX4_TOPIC_READY_TIMEOUT = 25.0

# Timeouts (segundos)
PX4_STARTUP_TIMEOUT = 50       # Tiempo para que PX4 + Gazebo arranquen
XRCE_STARTUP_DELAY = 3         # Espera tras lanzar XRCE agent
BRIDGE_STARTUP_DELAY = 3       # Espera tras lanzar el bridge
PRE_FLIGHT_SETTLE = 5          # Espera antes de iniciar el vuelo
EPISODE_TIMEOUT = 360          # Timeout maximo por episodio (6 min)
POST_EPISODE_COOLDOWN = 5      # Espera tras cerrar procesos
PROCESS_KILL_TIMEOUT = 10      # Tiempo para que un proceso muera con SIGINT

# Variables de entorno para PX4
# Spawn en (X=0, Y=0) fijo para preservar la alineacion exacta del origen EKF2 (NED)
# con el origen del mundo Gazebo (ENU). La variacion de pose inicial ocurre en vuelo (start_pose).
PX4_MODEL_POSE_GROUND = "0,0,0.15,0,0,0"    # Cuando no hay plataforma_salida (--no-takeoff-platform)
PX4_MODEL_POSE_PLATFORM = "0,0,0.35,0,0,0"  # Cuando se conserva plataforma_salida (altura 0.10m)
GZ_VERSION = "harmonic"
PX4_VEHICLE = "gz_x500"        # Vehiculo estandar gz_x500 (acelera RTF; observacion privilegiada)

# Binario de PX4 SITL (generado por `make px4_sitl gz_x500`).
# Se lanza directamente en modo daemon (-d) en lugar de via `make`:
#   - `make` usa USES_TERMINAL y arranca el shell interactivo `pxh>`, que lee
#     stdin. Sin TTY, pxh entra en un bucle que llena el log (~190 MB/vuelo);
#     con la TTY heredada desde un process group en segundo plano, el kernel
#     envia SIGTTIN y PX4 queda DETENIDO (sintoma: dron en Gazebo sin armar).
#   - `-d` desactiva pxh: PX4 corre como servidor puro, ideal para scripts.
PX4_BUILD_DIR = PX4_DIR / "build" / "px4_sitl_default"
PX4_BINARY = PX4_BUILD_DIR / "bin" / "px4"
PX4_ROOTFS = PX4_BUILD_DIR / "rootfs"

# Procesos que pueden quedar huerfanos de ejecuciones anteriores
# (p.ej. test_flight.sh interrumpido con Ctrl+C) y bloquear recursos
# como el puerto UDP 8888 del agente XRCE.
STALE_PROCESS_PATTERNS = [
    ["pkill", "-9", "-f", "gz sim"],
    ["pkill", "-9", "-f", "ruby.*gz"],
    ["pkill", "-9", "-f", "bin/px4"],
    ["pkill", "-9", "-f", "MicroXRCEAgent"],
    ["pkill", "-9", "-f", "ros_gz_bridge"],
    ["pkill", "-9", "-f", "waypoint_controller"],
    ["pkill", "-9", "-f", "tracking_controller"],
    ["pkill", "-9", "-f", "ros2 bag"],
]


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
        # Logs de nodos Python visibles en tiempo real (tail -f)
        full_env["PYTHONUNBUFFERED"] = "1"
        if env:
            full_env.update(env)

        log_handle = None
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_handle = open(log_file, "w")
            stdout = log_handle
            stderr = subprocess.STDOUT
        else:
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL

        proc = subprocess.Popen(
            cmd,
            env=full_env,
            cwd=cwd,
            shell=shell,
            # IMPORTANTE: nunca heredar la terminal como stdin. Un proceso
            # en otro process group que lee de la TTY recibe SIGTTIN y el
            # kernel lo detiene (estado 'T'). Esto es lo que hace bash con
            # `cmd &` en scripts no interactivos (stdin -> /dev/null).
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            # Nueva sesion (=> nuevo process group, sin TTY de control):
            # permite matar todo el arbol con killpg y evita que el Ctrl+C
            # de la terminal llegue a los hijos antes que nuestro handler.
            start_new_session=True,
        )

        # El hijo ya tiene su propia copia del descriptor
        if log_handle:
            log_handle.close()

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
        no_takeoff_platform: bool = True,
        num_landmarks: Optional[int] = None,
        min_landmarks: int = 2,
        max_landmarks: int = 5,
        include_circular_gates: bool = False,
        max_retries: int = 2,
    ):
        self.num_tasks = num_tasks
        self.episodes_per_task = episodes_per_task
        self.base_seed = base_seed
        self.dataset_dir = dataset_dir
        self.resume = resume
        self.episode_timeout = episode_timeout
        self.px4_startup_wait = px4_startup_wait
        self.no_takeoff_platform = no_takeoff_platform
        self.num_landmarks = num_landmarks
        self.min_landmarks = min_landmarks
        self.max_landmarks = max_landmarks
        self.include_circular_gates = include_circular_gates
        self.max_retries = max(0, int(max_retries))
        self.pm = ProcessManager()

        # Contadores globales (intentos originales vs reintentos)
        self.total_success = 0
        self.total_failed = 0
        self.total_timeout = 0
        self.primary_attempts_total = 0
        self.primary_attempts_accepted = 0
        self.retry_attempts_total = 0
        self.retry_attempts_accepted = 0

        # Handler para Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self._interrupted = False

    def _signal_handler(self, signum, frame):
        """Maneja Ctrl+C para limpiar procesos agresivamente."""
        if self._interrupted:
            # Segunda vez: forzar salida inmediatamente sin esperar
            print("\n[FORCE] Segunda interrupcion. Forzando salida...")
            self._kill_stale_processes()
            sys.exit(1)
            
        self._interrupted = True
        print("\n[STOP] Interrupcion recibida. Limpiando procesos...")
        # Intentar cerrar el bag limpiamente primero
        self.pm.stop("bag", signal.SIGINT)
        # Luego matar todo lo demas sin piedad para no colgar la terminal
        self._kill_stale_processes()
        self._print_summary()
        sys.exit(0)

    @staticmethod
    def get_attempt_episode_id(ep_idx: int, attempt_idx: int = 0) -> str:
        """
        Retorna el identificador de carpeta para un intento de episodio:
          - Intento original (`attempt_idx == 0`): `episode_XX`
          - Reintentos (`attempt_idx >= 1`): `episode_XX_attempt_YY`
        Nunca sobrescribe el intento original ni reintentos previos.
        """
        base_id = f"episode_{ep_idx:02d}"
        if attempt_idx <= 0:
            return base_id
        return f"{base_id}_attempt_{attempt_idx:02d}"

    @staticmethod
    def get_attempt_seed(base_seed: int, task_idx: int, ep_idx: int, attempt_idx: int = 0) -> int:
        """Calcula una semilla determinista derivada para el intento."""
        return (base_seed + task_idx) * 1000 + ep_idx + 1 + max(0, int(attempt_idx)) * 100000

    def _next_available_attempt_index(self, task_dir: Path, ep_idx: int, start_attempt: int = 0) -> int:
        """
        Encuentra el primer indice de intento (`>= start_attempt`) cuyo directorio no contiene
        aun un intento registrado (`episode_meta.json`), garantizando que nunca se sobrescriba
        un intento previo (exitoso o fallido).
        """
        att = max(0, int(start_attempt))
        while True:
            cand_id = self.get_attempt_episode_id(ep_idx, att)
            cand_dir = task_dir / cand_id
            if not (cand_dir / "episode_meta.json").exists():
                return att
            att += 1

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
              f"{self.episodes_per_task}  |  Max reintentos/ep: {self.max_retries}")
        print(f"  Validador integrado: v{VALIDATOR_VERSION}")
        print(f"  Dataset dir: {self.dataset_dir}")
        print("=" * 60)

        for task_idx in range(self.num_tasks):
            if self._interrupted:
                break

            task_id = f"task_{task_idx:04d}"
            task_dir = self.dataset_dir / task_id
            task_seed = self.base_seed + task_idx

            # Verificar si ya esta completa (resume)
            if self.resume and self._is_task_complete(task_dir):
                print(f"\n[SKIP] {task_id} ya completa. Saltando...")
                continue

            print(f"\n{'='*60}")
            print(f"  TAREA {task_idx + 1}/{self.num_tasks}: {task_id} "
                  f"(task_seed={task_seed})")
            print(f"{'='*60}")

            task_dir.mkdir(parents=True, exist_ok=True)

            # 1. Generar configuracion semantica de la tarea (sin reemplazo)
            if not self._generate_task_config(task_id, task_seed, task_idx=task_idx):
                print(f"[ERROR] Fallo generando configuracion semantica para {task_id}. "
                      "Saltando tarea.")
                continue

            # Copiar definicion semantica (task_config.yaml) a la carpeta de la tarea
            cfg_src = GENERATED_WORLDS / task_id / "task_config.yaml"
            cfg_dst = task_dir / "task_config.yaml"
            if cfg_src.exists():
                import shutil
                shutil.copy2(cfg_src, cfg_dst)

            # 2. Ejecutar episodios (cada uno con su propia randomizacion espacial)
            for ep_idx in range(self.episodes_per_task):
                if self._interrupted:
                    break

                base_ep_id = self.get_attempt_episode_id(ep_idx, 0)

                # Resume: saltar slots de episodio que ya cuentan con un intento aceptado
                if self.resume and self._is_episode_slot_complete(task_dir, ep_idx):
                    print(f"  [SKIP] {base_ep_id} ya cuenta con un intento aceptado.")
                    continue

                slot_accepted = False
                for retry_num in range(self.max_retries + 1):
                    if self._interrupted:
                        break

                    attempt_idx = self._next_available_attempt_index(task_dir, ep_idx, start_attempt=retry_num)
                    ep_id = self.get_attempt_episode_id(ep_idx, attempt_idx)
                    ep_dir = task_dir / ep_id
                    ep_seed = self.get_attempt_seed(self.base_seed, task_idx, ep_idx, attempt_idx)

                    attempt_label = (
                        "intento original"
                        if attempt_idx == 0
                        else f"reintento #{attempt_idx} de {base_ep_id}"
                    )
                    print(
                        f"\n  --- {task_id} / {ep_id} [{attempt_label}] "
                        f"({ep_idx + 1}/{self.episodes_per_task}) (ep_seed={ep_seed}) ---"
                    )

                    if attempt_idx == 0:
                        self.primary_attempts_total += 1
                    else:
                        self.retry_attempts_total += 1

                    # Generar mundo fisico y waypoints aleatorios para este intento
                    if not self._generate_episode_world(task_id, ep_seed):
                        print(f"  [ERROR] Fallo generando mundo para {task_id}/{ep_id}.")
                        ep_dir.mkdir(parents=True, exist_ok=True)
                        self._save_episode_metadata(
                            ep_dir,
                            task_id,
                            ep_id,
                            "world_gen_failed",
                            task_seed=task_seed,
                            ep_seed=ep_seed,
                            attempt_idx=attempt_idx,
                            original_episode_id=base_ep_id,
                            validation_status="rejected",
                            rejection_reasons=["world_generation_failed"],
                        )
                        self.total_failed += 1
                        continue

                    result = self._run_episode(
                        task_id,
                        ep_id,
                        ep_dir,
                        task_seed=task_seed,
                        ep_seed=ep_seed,
                        attempt_idx=attempt_idx,
                        original_episode_id=base_ep_id,
                    )

                    # Cooldown entre intentos/episodios
                    if not self._interrupted:
                        time.sleep(POST_EPISODE_COOLDOWN)

                    if result == "accepted":
                        self.total_success += 1
                        if attempt_idx == 0:
                            self.primary_attempts_accepted += 1
                        else:
                            self.retry_attempts_accepted += 1
                        slot_accepted = True
                        print(f"  [ACCEPTED] {ep_id} validado y aceptado.")
                        break
                    elif result == "timeout":
                        self.total_timeout += 1
                        print(
                            f"  [REJECTED/TIMEOUT] {ep_id} excedio el limite de "
                            f"{self.episode_timeout}s (conservado para auditoria)."
                        )
                    else:
                        self.total_failed += 1
                        print(
                            f"  [REJECTED] {ep_id} rechazado ({result}). "
                            f"Intento conservado intacto en {ep_dir.name}."
                        )

                if not slot_accepted and not self._interrupted:
                    print(
                        f"  [WARN] No se logro un intento aceptado para {task_id}/{base_ep_id} "
                        f"tras {self.max_retries + 1} intento(s)."
                    )

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  RECOLECCION COMPLETADA en {elapsed/60:.1f} min")
        self._print_summary()

    # ----------------------------------------------------------------
    #  Generacion de tarea y mundo por episodio
    # ----------------------------------------------------------------

    def _generate_task_config(
        self, task_id: str, task_seed: int, task_idx: Optional[int] = None
    ) -> bool:
        """Ejecuta task_generator.py para definir la semantica fija de la tarea sin reemplazo."""
        idx_str = f", task_index={task_idx}" if task_idx is not None else ""
        print(f"  Generando configuracion semantica de tarea {task_id} (task_seed={task_seed}{idx_str})...")
        cmd_task = [
            sys.executable,
            str(TASK_GENERATOR),
            "--task-id", task_id,
            "--seed", str(task_seed),
            "--shuffle-seed", str(self.base_seed),
            "--min-landmarks", str(self.min_landmarks),
            "--max-landmarks", str(self.max_landmarks),
        ]
        if self.num_landmarks is not None:
            cmd_task.extend(["--num-landmarks", str(self.num_landmarks)])
        if self.include_circular_gates:
            cmd_task.append("--include-circular-gates")
        if task_idx is not None:
            cmd_task.extend(["--task-index", str(task_idx)])

        result_task = subprocess.run(cmd_task, capture_output=True, text=True)
        if result_task.returncode != 0:
            print(f"  [ERROR] task_generator.py fallo:\n{result_task.stderr}")
            return False
        return True

    def _generate_episode_world(self, task_id: str, ep_seed: int) -> bool:
        """Ejecuta world_generator.py para generar poses fisicas aleatorias en este episodio."""
        print(f"    Generando mundo fisico para episodio (ep_seed={ep_seed})...")
        cmd_world = [
            sys.executable,
            str(WORLD_GENERATOR),
            "--task-id", task_id,
            "--seed", str(ep_seed),
        ]
        if self.no_takeoff_platform:
            cmd_world.append("--no-takeoff-platform")

        result_world = subprocess.run(
            cmd_world,
            capture_output=True,
            text=True,
        )
        if result_world.returncode != 0:
            print(f"    [ERROR] world_generator.py fallo:\n{result_world.stderr}")
            return False

        # Mostrar resumen de pose inicial y landmarks generados
        for line in result_world.stdout.strip().split("\n"):
            line_str = line.strip()
            if line_str and ("Pose inicial" in line_str or "landmark_" in line_str or "Plataforma:" in line_str):
                print(f"      {line_str}")
        return True

    # ----------------------------------------------------------------
    #  Verificacion activa de topicos de PX4 antes de volar
    # ----------------------------------------------------------------

    def _wait_for_px4_topics(self, timeout_s: float = PX4_TOPIC_READY_TIMEOUT) -> bool:
        """
        Espera activamente a que los topicos esenciales de PX4 esten registrados y que
        `/fmu/out/vehicle_odometry` este publicando muestras antes de grabar el bag y
        lanzar el piloto experto.
        """
        deadline = time.time() + float(timeout_s)
        full_env = os.environ.copy()
        full_env["ROS_DOMAIN_ID"] = os.environ.get("ROS_DOMAIN_ID", "33")

        topics_visible = False
        while time.time() < deadline:
            if not self.pm.is_running("px4") or not self.pm.is_running("xrce"):
                return False
            try:
                res = subprocess.run(
                    ["ros2", "topic", "list"],
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                    env=full_env,
                )
                if res.returncode == 0:
                    active_topics = set(line.strip() for line in res.stdout.splitlines() if line.strip())
                    if all(req in active_topics for req in REQUIRED_ACTIVE_PX4_TOPICS):
                        topics_visible = True
                        break
            except Exception:
                pass
            time.sleep(1.0)

        if not topics_visible:
            return False

        # Verificar que /fmu/out/vehicle_odometry tenga al menos un publicador activo y emita mensajes
        remaining = max(3.0, deadline - time.time())
        try:
            res_echo = subprocess.run(
                ["ros2", "topic", "echo", "--once", "/fmu/out/vehicle_odometry"],
                capture_output=True,
                text=True,
                timeout=min(remaining, 10.0),
                env=full_env,
            )
            if res_echo.returncode == 0 and res_echo.stdout.strip():
                return True
        except Exception:
            pass

        # Fallback: comprobar publisher count > 0 en ros2 topic info
        try:
            res_info = subprocess.run(
                ["ros2", "topic", "info", "/fmu/out/vehicle_odometry"],
                capture_output=True,
                text=True,
                timeout=5.0,
                env=full_env,
            )
            if res_info.returncode == 0 and "Publisher count: 0" not in res_info.stdout:
                return True
        except Exception:
            pass

        return False

    # ----------------------------------------------------------------
    #  Ejecucion de un episodio
    # ----------------------------------------------------------------

    def _run_episode(
        self,
        task_id: str,
        ep_id: str,
        ep_dir: Path,
        task_seed: Optional[int] = None,
        ep_seed: Optional[int] = None,
        attempt_idx: int = 0,
        original_episode_id: Optional[str] = None,
    ) -> str:
        """Ejecuta y valida automaticamente un episodio completo.

        Returns:
            "accepted", "timeout", o descripcion del motivo de rechazo/error.
        """
        ep_dir.mkdir(parents=True, exist_ok=True)
        log_dir = ep_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        waypoints_file = GENERATED_WORLDS / task_id / "waypoints.yaml"

        # Copiar waypoints.yaml exclusivos de este episodio para auditoria y postproceso
        ep_wp_dst = ep_dir / "waypoints.yaml"
        if waypoints_file.exists():
            import shutil
            shutil.copy2(waypoints_file, ep_wp_dst)

        print(f"    Logs del episodio: {log_dir}")
        print(f"      (monitorear piloto: tail -f {log_dir}/pilot.log)")
        print(f"      (monitorear PX4/SITL: tail -f {log_dir}/px4.log)")

        # 0. Matar procesos huerfanos de ejecuciones previas
        self._kill_stale_processes()

        exit_code: Optional[int] = None
        raw_status: str = "unknown_error"

        try:
            # 1. Lanzar XRCE-DDS Agent
            print("    Lanzando MicroXRCEAgent...")
            self._start_xrce_agent(log_dir / "xrce.log")
            time.sleep(XRCE_STARTUP_DELAY)

            if not self.pm.is_running("xrce"):
                print("    [ERROR] MicroXRCEAgent murio al iniciar "
                      "(¿puerto 8888 ocupado?). Ver xrce.log")
                self._print_log_tail(log_dir / "xrce.log", "XRCE Log (ultimas lineas)")
                self._save_episode_metadata(
                    ep_dir,
                    task_id,
                    ep_id,
                    "xrce_crash",
                    task_seed=task_seed,
                    ep_seed=ep_seed,
                    attempt_idx=attempt_idx,
                    original_episode_id=original_episode_id,
                    validation_status="rejected",
                    rejection_reasons=["xrce_agent_startup_crash"],
                )
                return "xrce_crash"

            # 2. Lanzar PX4 + Gazebo
            print(f"    Lanzando PX4 (world={task_id}, model={PX4_VEHICLE})...")
            self._start_px4(task_id, log_dir / "px4.log")
            time.sleep(self.px4_startup_wait)

            if not self.pm.is_running("px4"):
                print("    [ERROR] PX4 murio al iniciar. Ver px4.log")
                self._print_log_tail(log_dir / "px4.log", "PX4 Log (ultimas lineas)")
                self._save_episode_metadata(
                    ep_dir,
                    task_id,
                    ep_id,
                    "px4_crash",
                    task_seed=task_seed,
                    ep_seed=ep_seed,
                    attempt_idx=attempt_idx,
                    original_episode_id=original_episode_id,
                    validation_status="rejected",
                    rejection_reasons=["px4_startup_crash"],
                )
                return "px4_crash"

            # 3. Lanzar gz_bridge (solo requerido si se utiliza modelo con camara de profundidad)
            if PX4_VEHICLE == "gz_x500_depth":
                print("    Lanzando ros_gz_bridge (camara de profundidad)...")
                self._start_gz_bridge(log_dir / "bridge.log")
                time.sleep(BRIDGE_STARTUP_DELAY)
            else:
                print("    [INFO] Vehiculo gz_x500: Observacion privilegiada activa (sin camara de profundidad).")

            # 4. Verificar activamente que los topicos de PX4 esten publicando antes de grabar/volar
            print("    Verificando publicacion activa de topicos PX4...")
            if not self._wait_for_px4_topics(timeout_s=PX4_TOPIC_READY_TIMEOUT):
                print("    [ERROR] Los topicos de PX4 no comenzaron a publicar a tiempo.")
                self._save_episode_metadata(
                    ep_dir,
                    task_id,
                    ep_id,
                    "px4_topics_inactive",
                    task_seed=task_seed,
                    ep_seed=ep_seed,
                    attempt_idx=attempt_idx,
                    original_episode_id=original_episode_id,
                    validation_status="rejected",
                    rejection_reasons=["px4_topics_not_publishing_before_takeoff"],
                )
                return "px4_topics_inactive"

            print(f"    Esperando {PRE_FLIGHT_SETTLE}s de estabilizacion...")
            time.sleep(PRE_FLIGHT_SETTLE)

            # 5. Iniciar grabacion de rosbag
            print("    Iniciando grabacion rosbag...")
            self._start_bag_record(ep_dir, log_dir / "bag.log")
            time.sleep(1)  # Dar tiempo al bag recorder para inicializar

            # 6. Lanzar piloto experto
            print("    Lanzando piloto experto...")
            self._start_pilot(
                ep_wp_dst, log_dir / "pilot.log"
            )

            # 7. Esperar a que el piloto termine (o timeout)
            print(f"    Volando... (timeout={self.episode_timeout}s)")
            exit_code = self.pm.wait("pilot", timeout=self.episode_timeout)

            if exit_code is None:
                raw_status = "timeout"
                print(f"\n    [TIMEOUT] El episodio excedio los {self.episode_timeout}s.")
                self._print_log_tail(log_dir / "px4.log", "PX4 Log (ultimas lineas)")
                self._print_log_tail(log_dir / "pilot.log", "Pilot Log (ultimas lineas)")
            elif exit_code == 0:
                raw_status = "success"
            else:
                raw_status = f"pilot_exit_{exit_code}"

            # 8. Cerrar limpiamente el bag ANTES de ejecutar el validador post-vuelo
            print("    Cerrando rosbag limpiamente (SIGINT)...")
            self.pm.stop("bag", signal.SIGINT)

            # Guardar metadata base con version y parametros congelados del validador
            self._save_episode_metadata(
                ep_dir,
                task_id,
                ep_id,
                raw_status,
                task_seed=task_seed,
                ep_seed=ep_seed,
                attempt_idx=attempt_idx,
                original_episode_id=original_episode_id,
            )

            # 9. Invocar automaticamente el validador sobre el episodio cerrado
            print(f"    Ejecutando validate_episode_directory (v{VALIDATOR_VERSION})...")
            val_report = validate_episode_directory(
                ep_dir,
                pilot_exit_code=exit_code,
                update_meta=True,
            )
            val_status = val_report.get("validation_status", "rejected")
            rej_reasons = val_report.get("rejection_reasons", [])

            if raw_status == "timeout" and "pilot_timeout" not in rej_reasons:
                rej_reasons = ["pilot_timeout"] + list(rej_reasons)
                val_status = "rejected"
                meta_path = ep_dir / "episode_meta.json"
                if meta_path.is_file():
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        meta_obj = json.load(mf)
                    meta_obj["validation_status"] = "rejected"
                    meta_obj["rejection_reasons"] = rej_reasons
                    with open(meta_path, "w", encoding="utf-8") as mf:
                        json.dump(meta_obj, mf, indent=2)

            if val_status == "accepted" and raw_status == "success":
                return "accepted"
            if raw_status == "timeout":
                return "timeout"
            return f"rejected ({', '.join(rej_reasons) if rej_reasons else raw_status})"

        except Exception as e:
            print(f"    [EXCEPTION] {e}")
            return f"exception: {e}"

        finally:
            # Cleanup: detener todo agresivamente
            print("    Cerrando procesos...")
            # Bag: SIGINT para cierre limpio del archivo (no-op si ya se detuvo en el paso 8)
            self.pm.stop("bag", signal.SIGINT)
            
            # En lugar de pedir amablemente con stop_all() y causar cuelgues,
            # aniquilamos los procesos manejados instantaneamente con SIGKILL.
            for name, proc in list(self.pm.processes.items()):
                if name == "bag":
                    continue
                try:
                    proc.kill()
                except Exception:
                    pass
            self.pm.processes.clear()
            
            # Y finalmente ejecutamos la misma estrategia de pkill -9 de kill_sim.sh
            # para destruir cualquier hijo rebelde (como el servidor ruby de Gazebo)
            self._kill_stale_processes()

    # ----------------------------------------------------------------
    #  Arranque de procesos individuales
    # ----------------------------------------------------------------

    def _start_px4(self, world_name: str, log_file: Path) -> None:
        """Lanza PX4 SITL con Gazebo en modo daemon.

        Replica lo que hace `make px4_sitl gz_x500_depth` (ver
        PX4-Autopilot/src/modules/simulation/gz_bridge/CMakeLists.txt),
        pero con `-d` para no arrancar el shell interactivo pxh.
        """
        # Escribir comandos extras de inicializacion (ej. forzar chequeos)
        extras_path = PX4_ROOTFS / "etc" / "extras.txt"
        extras_path.parent.mkdir(parents=True, exist_ok=True)
        with open(extras_path, "w") as f:
            f.write("param set NAV_DLL_ACT 0\n")
            f.write("param set EKF2_GPS_CHECK 0\n")  # Relajar chequeo de GPS por recomendacion web
            f.write("param set MPC_TKO_SPEED 3.0\n") # Acelerar el despegue para el dataset

            f.write("param set MPC_XY_CRUISE 3.0\n")  # Velocidad de crucero deseada (m/s)
            f.write("param set MPC_XY_VEL_MAX 4.0\n")   # Velocidad maxima absoluta (m/s)

            f.write("commander arm -f\n")

        spawn_pose = (
            PX4_MODEL_POSE_GROUND if self.no_takeoff_platform else PX4_MODEL_POSE_PLATFORM
        )
        env = {
            "ROS_DOMAIN_ID": os.environ.get("ROS_DOMAIN_ID", "33"),
            "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST",
            "GZ_PARTITION": "sim",
            "GZ_VERSION": GZ_VERSION,
            "GZ_IP": "127.0.0.1",
            "PX4_SIM_MODEL": PX4_VEHICLE,
            "PX4_GZ_WORLD": world_name,
            "PX4_GZ_MODEL_POSE": spawn_pose,
        }
        self.pm.start(
            "px4",
            [str(PX4_BINARY), "-d"],
            env=env,
            cwd=PX4_ROOTFS,
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

    def _start_pilot(
        self, waypoints_file: Path, log_file: Path
    ) -> None:
        """Lanza el nodo del piloto experto."""
        self.pm.start(
            "pilot",
            [
                "ros2", "run", "uav_ip", "waypoint_controller",
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

    def _kill_stale_processes(self) -> None:
        """Mata procesos huerfanos del stack (PX4, XRCE, bridge, Gazebo).

        Un MicroXRCEAgent huerfano ocupa el puerto 8888 y hace fallar al
        nuevo agente con 'bind error errno 98'.
        """
        killed_any = False
        for cmd in STALE_PROCESS_PATTERNS:
            try:
                res = subprocess.run(cmd, capture_output=True, timeout=5)
                killed_any |= res.returncode == 0
            except Exception:
                pass
        if killed_any:
            print("    [WARN] Se encontraron procesos huerfanos y se "
                  "terminaron. Esperando liberacion de recursos...")
            time.sleep(3)

    def _save_episode_metadata(
        self,
        ep_dir: Path,
        task_id: str,
        ep_id: str,
        status: str,
        task_seed: Optional[int] = None,
        ep_seed: Optional[int] = None,
        attempt_idx: int = 0,
        original_episode_id: Optional[str] = None,
        validation_status: Optional[str] = None,
        rejection_reasons: Optional[List[str]] = None,
    ) -> None:
        """Guarda metadata del episodio en JSON con parametros congelados del validador."""
        spawn_pose = (
            PX4_MODEL_POSE_GROUND
            if self.no_takeoff_platform
            else PX4_MODEL_POSE_PLATFORM
        )
        meta: Dict[str, Any] = {
            "task_id": task_id,
            "episode_id": ep_id,
            "original_episode_id": original_episode_id or ep_id.split("_attempt_")[0],
            "attempt_index": int(attempt_idx),
            "is_retry": bool(attempt_idx > 0),
            "task_seed": task_seed,
            "episode_seed": ep_seed,
            "status": status,
            "timestamp": datetime.datetime.now().isoformat(),
            "vehicle_model": PX4_VEHICLE,
            "spawn_pose": spawn_pose,
            "no_takeoff_platform": self.no_takeoff_platform,
            "embodiment": "uav_lidar_360_privileged",
            "scale_factor": 0.1,
            "lidar_r_max": 10.0,
            "topics_recorded": RECORD_TOPICS,
            "episode_timeout_s": self.episode_timeout,
            "validator_version": VALIDATOR_VERSION,
            "validation_parameters": get_effective_validation_parameters(),
        }
        if validation_status is not None:
            meta["validation_status"] = validation_status
        if rejection_reasons is not None:
            meta["rejection_reasons"] = list(rejection_reasons)

        meta_path = ep_dir / "episode_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _is_task_complete(self, task_dir: Path) -> bool:
        """Verifica si una tarea tiene todos sus slots de episodio aceptados."""
        if not task_dir.is_dir():
            return False
        complete = 0
        for ep_idx in range(self.episodes_per_task):
            if self._is_episode_slot_complete(task_dir, ep_idx):
                complete += 1
        return complete >= self.episodes_per_task

    def _is_episode_slot_complete(self, task_dir: Path, ep_idx: int) -> bool:
        """
        Verifica si el slot `episode_XX` cuenta con un intento aceptado (ya sea en
        `episode_XX` o en alguno de sus reintentos `episode_XX_attempt_YY`).
        """
        base_dir = task_dir / self.get_attempt_episode_id(ep_idx, 0)
        if self._is_episode_complete(base_dir):
            return True
        for att_dir in sorted(task_dir.glob(f"episode_{ep_idx:02d}_attempt_*")):
            if att_dir.is_dir() and self._is_episode_complete(att_dir):
                return True
        return False

    def _is_episode_complete(self, ep_dir: Path) -> bool:
        """Verifica si un directorio de episodio fue validado y aceptado (`validation_status == 'accepted'`)."""
        meta_path = ep_dir / "episode_meta.json"
        if not meta_path.exists():
            return False
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if "validation_status" in meta:
                return meta.get("validation_status") == "accepted"
            return meta.get("status") == "success"
        except Exception:
            return False

    def _print_log_tail(self, log_file: Path, title: str, lines: int = 15) -> None:
        """Imprime las ultimas lineas de un log para mayor visibilidad."""
        if log_file.exists():
            print(f"\n    === {title} ===")
            try:
                import collections
                with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                    tail = collections.deque(f, maxlen=lines)
                for line in tail:
                    print(f"      {line}", end="")
                print("    ====================\n")
            except Exception as e:
                print(f"      [WARN] No se pudo leer {log_file.name}: {e}")

    def _print_summary(self) -> None:
        """Imprime resumen de la recoleccion diferenciando intento original vs post-reintento."""
        total = self.total_success + self.total_failed + self.total_timeout
        print(f"\n  Resumen: {total} intentos ejecutados")
        print(f"    Aceptados: {self.total_success}")
        print(f"    Rechazados:{self.total_failed}")
        print(f"    Timeouts:  {self.total_timeout}")
        if self.primary_attempts_total > 0:
            p_rate = self.primary_attempts_accepted / self.primary_attempts_total * 100.0
            post_rate = self.total_success / self.primary_attempts_total * 100.0
            print(
                f"    Exito en 1er intento (original): "
                f"{self.primary_attempts_accepted}/{self.primary_attempts_total} ({p_rate:.1f}%)"
            )
            print(
                f"    Exito tras reintentos (slots):   "
                f"{self.total_success}/{self.primary_attempts_total} ({post_rate:.1f}%)"
            )
        elif total > 0:
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

    parser.add_argument(
        "--with-takeoff-platform", action="store_false", dest="no_takeoff_platform",
        help="Incluir la plataforma de salida (por defecto no se incluye).",
    )
    parser.add_argument(
        "--num-landmarks", type=int, default=None,
        help="Fijar cantidad exacta de landmarks en todas las tareas (anula min/max).",
    )
    parser.add_argument(
        "--min-landmarks", type=int, default=2,
        help="Cantidad minima de landmarks acotados por tarea (default: 2).",
    )
    parser.add_argument(
        "--max-landmarks", type=int, default=5,
        help="Cantidad maxima de landmarks acotados por tarea (default: 5).",
    )
    parser.add_argument(
        "--include-circular-gates", action="store_true",
        help="Incluir aros circulares (circular_gate) en el catalogo de tareas.",
    )
    parser.add_argument(
        "--max-retries", type=int, default=2,
        help="Maximo de reintentos con semilla derivada cuando un episodio es rechazado "
        "(guardados como episode_XX_attempt_YY sin sobrescribir el intento original). Default: 2.",
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
        no_takeoff_platform=args.no_takeoff_platform,
        num_landmarks=args.num_landmarks,
        min_landmarks=args.min_landmarks,
        max_landmarks=args.max_landmarks,
        include_circular_gates=args.include_circular_gates,
        max_retries=args.max_retries,
    )
    
    try:
        collector.run()
    except KeyboardInterrupt:
        print("\n[STOP] Interrupcion global recibida (Ctrl+C). Limpiando todos los procesos...")
        collector.pm.stop_all()
        collector._kill_stale_processes()
        sys.exit(130)


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
    if "uav_ip" not in os.environ.get("AMENT_PREFIX_PATH", ""):
        errors.append(
            "Paquete uav_ip no instalado. Ejecuta:\n"
            "  cd ~/ros2_ws && colcon build --packages-select uav_ip && "
            "source install/setup.bash"
        )

    # Verificar PX4
    if not PX4_DIR.is_dir():
        errors.append(f"PX4-Autopilot no encontrado en {PX4_DIR}")
    elif not PX4_BINARY.is_file():
        errors.append(
            f"Binario PX4 no encontrado en {PX4_BINARY}. Compilalo una vez:\n"
            f"  cd ~/PX4-Autopilot && make px4_sitl {PX4_VEHICLE}"
        )

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
