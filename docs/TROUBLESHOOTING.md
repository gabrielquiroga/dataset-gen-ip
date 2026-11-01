# Troubleshooting y Problemas Frecuentes (UAV Instant Policy)

*Guía de resolución de incidencias técnicas en WSL2, ROS 2 Jazzy, Gazebo Harmonic, PX4 SITL y pipelines ML · Actualizado a Noviembre de 2026*

---

## 1. Dron no logra armarse y reporta `High Gyro Bias` o `GPS Vert Speed Drift too high`

**Síntoma:** Al lanzar PX4, en la terminal comienzan a aparecer alertas continuas sobre el giroscopio y el GPS, seguidas por `Arming denied: Resolve system health failures first`. El nodo `expert_pilot` se queda infinitamente enviando comandos `ARM & OFFBOARD (reintento)`.

**Causa:** Colisión y explosión del motor de físicas al iniciar (`Spawn Penetration`). La plataforma de salida en Gazebo (`plataforma_salida`) tiene una altura hasta `Z=0.1`. Si la variable `PX4_GZ_MODEL_POSE` define una altura de spawn inferior a las patas del dron (por ejemplo, `Z=0.2`), el dron "nace" penetrando la estructura estática. El motor ODE intenta expulsarlo de inmediato aplicando fuerzas colosales, lo que hace que el dron salga volando hacia arriba girando sin control. Esto satura los sensores simulados.

**Solución:**
Asegurar que el dron inicie en $(X, Y) = (0, 0)$ (para que el origen local NED de EKF2 coincida exactamente con el origen ENU de Gazebo) y con altura suficiente para apoyarse suavemente por gravedad sin penetrar la geometría. Tanto `data_collector.py` como `test_flight.sh` usan constantes unificadas:
```bash
# Modo por defecto (sin plataforma de salida, --no-takeoff-platform):
export PX4_GZ_MODEL_POSE="0,0,0.15,0,0,0"

# Con plataforma de salida activa (Z_plataforma = 0.10 m) o en test_flight.sh:
export PX4_GZ_MODEL_POSE="0,0,0.35,0,0,0"
```
*(Nota: La variación de pose inicial entre episodios se realiza en vuelo mediante el waypoint `start_pose` antes de `takeoff_exit`, preservando la alineación exacta del marco EKF2 con el mundo de Gazebo).*

---

## 2. Dron aparece pero los sensores no publican (`Preflight Fail: ekf2 missing data`)

**Síntoma:** El simulador Gazebo se abre y ves el dron perfectamente colocado. Sin embargo, la terminal de PX4 arroja advertencias de `barometer 0 missing`, `ekf2 missing data` o `Found 0 compass`.

**Causa:** Procesos `gz sim` zombies. Al usar `Ctrl+C` para matar un lanzamiento previo de PX4 (`make px4_sitl`), el servidor en segundo plano de Gazebo no se cierra completamente. Cuando vuelves a iniciar PX4, el script detecta que Gazebo ya corre e inyecta un *nuevo* dron en ese mundo "huérfano" cuyos nodos de transporte (`gz-transport`) están pausados o corruptos. PX4 no recibe datos de los sensores.

**Solución manual:**
Antes de realizar lanzamientos manuales, usar el script de limpieza provisto para matar procesos zombies:
```bash
bash src/uav_ip/scripts/kill_sim.sh
```
*(El orquestador `data_collector.py` ya ejecuta estas rutinas de limpieza automáticamente en cada episodio y entre reintentos).*

---

## 3. PX4 arranca perfectamente pero no responde al piloto experto

**Síntoma:** La terminal de PX4 no muestra errores (llega a `home set`) pero el dron jamás inicia vuelo, a pesar de que la terminal de `expert_pilot` indica "Comandando ARM".

**Causa:** Desincronización del Dominio ROS 2 (`ROS_DOMAIN_ID`). PX4 en sí mismo no es un nodo de ROS; depende del puente `MicroXRCEAgent`. Si el agente no se ejecuta con el dominio 33, escuchará en el Dominio 0. Sin embargo, el `expert_pilot` publica sus comandos en el Dominio 33. Al no verse, PX4 no crea *suscriptores* (DataReaders) y las peticiones de armado caen al vacío.

**Solución:**
Asegurar la inyección explícita de las variables de red antes de ejecutar el puente:
```bash
export ROS_DOMAIN_ID=33
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
MicroXRCEAgent udp4 -p 8888
```
*(El `ProcessManager` de `data_collector.py` inyecta estas variables automáticamente en todos sus subprocesos).*

---

## 4. Simulador (Gazebo) y ROS están muy desfasados de tiempo en WSL2

**Síntoma:** Pérdida de paquetes, errores de QoS o TF timeouts.
**Causa:** Conflicto de interfaces de red de Windows hacia WSL2 que confunde a Gazebo Transport y FastDDS.
**Solución:**
Aislar la simulación al loopback local:
```bash
export GZ_IP=127.0.0.1
export GZ_PARTITION=sim
```

---

## 5. Episodio Rechazado por Salto de Reloj o Ambigüedad (`ambiguous_clock_step_pattern_b`)

**Síntoma:** `validate_episode.py` o `data_collector.py` rechazan un episodio con `recording_status: invalid` y la causa `ambiguous_clock_step_pattern_b` o `telemetry_dropout_gap`.

**Causa:**
En PX4 SITL, el cliente micro-XRCE (`uxrce_dds_client`) recalibra periódicamente el offset UNIX durante el vuelo (habitualmente cada 15–17 s), introduciendo saltos de $0.25\text{–}0.48\text{ s}$ en los timestamps exportados por `VehicleOdometry`.

**Solución y Diagnóstico:**
1. `data_collector.py` graba automáticamente `/fmu/out/timesync_status` (`px4_msgs/msg/TimesyncStatus`).
2. El validador `validate_episode.py` desambigua analíticamente: si el salto en odometría coincide en una ventana de $\pm 0.05\text{ s}$ con un salto equivalente en `TimesyncStatus.estimated_offset`, el reloj se repara automáticamente (`timesync_offset_adjustment_repaired`) y el episodio se acepta (`usable_for_validation = True`).
3. Si el episodio sigue rechazado con `ambiguous_clock_step_pattern_b`:
   * Verifica que `/fmu/out/timesync_status` esté presente en el rosbag (`ros2 bag info <bag_dir>`).
   * Si no se debe a un ajuste de offset de red sino a una congelación genuina de la simulación o caída de telemetría superior a $0.15\text{ s}$, el rechazo es legítimo y `data_collector.py` lanzará un reintento no destructivo (`episode_XX_attempt_01`).

---

## 6. Error de Dependencias de Machine Learning: PyTorch o PyG no encontrados

**Síntoma:** Al ejecutar `bag_to_heterodata.py` o `audit_dataset.py --require-pt`, el script falla con:
`RuntimeError: PyTorch y/o torch_geometric no están disponibles (--require-torch activo)`
o genera silenciosamente archivos comprimidos `.npz` en lugar de grafos `.pt`.

**Causa:**
En Ubuntu 24.04 (Noble Numbat), la política PEP 668 prohíbe instalar librerías pesadas de PyTorch con `pip` en el Python del sistema (`/usr/bin/python3`), para proteger el entorno de ROS 2 Jazzy.
El entorno virtual de ML se encuentra instalado en `~/venv_ip` (un nivel por encima de `ros2_ws`).

**Solución:**
* **Opción A (Recomendada, ejecución directa):** Invocar los scripts de postproceso directamente con el binario de Python de `venv_ip`:
  ```bash
  ~/venv_ip/bin/python3 src/uav_ip/scripts/bag_to_heterodata.py --dataset-dir datasets_sitl_v2 --require-torch
  ~/venv_ip/bin/python3 src/uav_ip/scripts/audit_dataset.py --dataset-dir datasets_sitl_v2 --require-pt
  ~/venv_ip/bin/python3 src/uav_ip/scripts/kinematic_pseudo_generator.py --require-torch
  ```
* **Opción B (Activación de entorno):** Activar el virtualenv en una terminal dedicada:
  ```bash
  source ~/venv_ip/bin/activate
  python3 src/uav_ip/scripts/bag_to_heterodata.py --dataset-dir datasets_sitl_v2 --require-torch
  ```
* Si el entorno no existe aún, se puede provisionar en 2 minutos con:
  ```bash
  bash src/uav_ip/scripts/setup_venv_ip.sh
  ```

---

## 7. Compatibilidad de Nombres de Parámetros CLI (Alias Tolerantes)

**Síntoma:** Al ejecutar un script antiguo o manual, aparece el error:
`error: unrecognized arguments: --seed ...` o `--output-dir ...`

**Causa:**
En versiones históricas iniciales se usaban nombres específicos (`--base-seed` y `--dataset-dir` en `data_collector.py`; `--exclude-signatures-from` en generadores cinemáticos).

**Solución:**
A partir de la versión actual, todos los scripts de `src/uav_ip/scripts/` admiten alias de argumentos de forma tolerante y transparente:
* `data_collector.py`: admite tanto `--seed` como `--base-seed`; y tanto `--output-dir` como `--dataset-dir`.
* `bag_to_heterodata.py`: admite tanto `--dataset-dir` como `--output` / `--output-dir`, y admite el flag `--overwrite`.
* `kinematic_pseudo_generator.py`: admite tanto `--exclude-signatures-from` como `--exclude-tasks-from`, y acepta indistintamente rutas de carpetas o archivos `.json` individuales.

---

## 8. Ejecución de Tests Unitarios (`pytest`)

**Síntoma:** Al ejecutar `pytest` desde la raíz de `ros2_ws/`, aparecen errores de importación como `ModuleNotFoundError: No module named 'uav_ip'`.

**Causa:**
Falta registrar los directorios del paquete en el `sys.path` del entorno de test.

**Solución:**
El proyecto cuenta con un archivo canónico [`pyproject.toml`](/pyproject.toml) en la raíz que configura automáticamente:
```toml
[tool.pytest.ini_options]
pythonpath = ["src/uav_ip", "src/uav_ip/scripts"]
testpaths = ["src/uav_ip/test"]
```
Simplemente ejecuta:
```bash
pytest src/uav_ip/test -q
```
La suite completa (63 tests) debe pasar en aproximadamente 11 segundos con 0 errores y 0 tests omitidos.
