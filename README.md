# UAV Instant Policy: Data Collection & Synthetic LiDAR Pipeline

## 📖 Descripción del Proyecto

Este proyecto implementa un *pipeline* automatizado de recolección de datos en simulación para entrenar un modelo de Aprendizaje por Imitación en Contexto (*In-Context Imitation Learning* - ICIL), basado en la arquitectura **Instant Policy** (Vosylius y Johns, ICLR 2025).

El sistema utiliza **ROS 2 Jazzy**, **Gazebo Harmonic** y **PX4 Autopilot** para simular un VANT (Vehículo Aéreo No Tripulado) que navega procedimentalmente a través de circuitos dinámicos de landmarks (aros cuadrados, toroides, postes cilíndricos y plataformas de aterrizaje).

<p align="center">
   <img src='docs\assets\vuelo_animado.gif' width="850" alt="Animación de vuelo y LiDAR sintético" />
   <br>
   <em>Figura 1: Visualización sincronizada del vuelo en 3D (izquierda), proyección egocéntrica del LiDAR en plano XY (centro) y perfil vertical en plano XZ (derecha) mostrando la maniobra de sobrepaso.</em>
</p>

---

## 🗂️ Estructura del Espacio de Trabajo

```text
ros2_ws/
├── src/
│   ├── px4_msgs/                  # Definiciones de tópicos (VehicleOdometry, TrajectorySetpoint, etc.)
│   ├── px4_offboard/              # Tutoriales y utilidades offboard de referencia
│   └── uav_ip/
│       ├── package.xml
│       ├── setup.py
│       ├── worlds/                # Plantillas SDF de Gazebo (circuito_aros_01.sdf)
│       ├── generated_worlds/      # Tareas generadas procedimentalmente (.sdf + waypoints.yaml)
│       ├── scripts/
│       │   ├── setup_venv_ip.sh       # Configurador automatizado de virtualenv ML (~/venv_ip)
│       │   ├── task_generator.py      # Generador de recetas semánticas de tareas (sin reemplazo)
│       │   ├── world_generator.py     # Generador de mundos SDF y waypoints.yaml
│       │   ├── synthetic_lidar.py     # Thin wrapper CLI para LiDAR sintético 360°
│       │   ├── audit_dataset.py       # Auditoría semántica, métricas de vuelo y particionado Train/Val/Held-Out
│       │   ├── bag_to_heterodata.py   # Conversor batch multiproceso a HeteroData PyG (.pt) con Tube Augmentation
│       │   ├── visualize_dataset.py   # Thin wrapper CLI para visualización 3D y análisis de recuperación
│       │   ├── data_collector.py      # Orquestador automático del dataset (doble bucle)
│       │   ├── test_flight.sh         # Script rápido para vuelo de prueba
│       │   └── kill_sim.sh            # Limpieza agresiva de procesos huérfanos
│       └── uav_ip/                    # Módulo Python principal (Punto Único de Verdad)
│           ├── __init__.py
│           ├── base_pilot.py          # Máquina de estados, QoS y protocolo seguro de armado
│           ├── waypoint_controller.py # Piloto experto feed-forward (traverse, overfly, slalom)
│           ├── tracking_controller.py # Piloto experto feedback (orbit)
│           ├── synthetic_lidar.py     # Generador LiDAR analítico (Horizontal FLU, kernel C SIMD FPS)
│           ├── validation_geometry.py # Geometría compartida, mediciones SDF x500 y tolerancias derivadas
│           ├── validate_episode.py    # Validador post-vuelo (configuración, reloj, MCAP/CDR, misión)
│           └── visualize_dataset.py   # Visualizador 3D, animación GIF e inspección Tube Augmentation
│   ├── datasets/                      # Salida del dataset estructurado por tarea y episodio
├── docs/                          # Documentación técnica canónica
│   ├── ESTRATEGIA_TEORICA.md      # Marco teórico, preservación de Markov, escala s=0.1 y taxonomía
│   ├── PLAN_IMPLEMENTACION.md     # Hoja de ruta, hitos de desarrollo (0-4) y backlog operativo
│   ├── DECISIONES_DISENO.md       # Registro de decisiones de arquitectura (ADRs) y enfoques descartados
│   └── TROUBLESHOOTING.md         # Resolución de incidencias comunes (WSL2, PX4, Gazebo)
├── build/
├── install/
└── log/
```

---

## ⚙️ Prerrequisitos del Sistema

- **Sistema Operativo:** Ubuntu 24.04 LTS (o Windows 11 con WSL2).
- **Simulación:** PX4 Autopilot (rama `main` compilada para SITL: `make px4_sitl gz_x500`).
- **Middleware:** ROS 2 Jazzy con `MicroXRCEAgent` (`udp4 -p 8888`).
- **Python:** Python 3.12+ con librerías:
  - `rclpy`, `px4_msgs`, `sensor_msgs` (Ecosistema ROS 2 en el sistema)
  - `numpy`, `yaml`, `scipy` (Generación geométrica y procedural)
  - `rosbags`, `torch`, `torch_geometric` (Procesamiento offline a HeteroData)

### Configuración del Entorno Virtual de Machine Learning (`~/venv_ip`)
Para aislar las dependencias de PyTorch y PyTorch Geometric sin interferir con la instalación de ROS 2 ni con la política PEP 668 de Ubuntu 24.04:

```bash
# Ejecutar el script automatizado de configuración:
bash src/uav_ip/scripts/setup_venv_ip.sh

# Activar el entorno virtual para procesamiento y entrenamiento ML:
source ~/venv_ip/bin/activate
```
*(Nota: El script enlaza automáticamente el módulo `uav_ip` dentro del entorno virtual y soporta tanto instalaciones directas con `pip` como entornos base con `--system-site-packages`).*

---

## 🚀 Procedimiento 1: Ejecución y Auditoría de una Demostración Individual

Para validar y auditar una demostración de punta a punta (inspeccionando trayectorias, telemetría y el grafo generado), sigue estos pasos:

### Paso 1: Compilar y Generar el Mundo Procedural
En una terminal con ROS 2 configurado (`source /opt/ros/jazzy/setup.bash` y `export ROS_DOMAIN_ID=33`):

```bash
cd ~/ros2_ws
colcon build --packages-select uav_ip --symlink-install
source install/setup.bash

# 1. Generar la receta semántica de la tarea
python3 src/uav_ip/scripts/task_generator.py --task-id test_001 --seed 42

# 2. Generar el mundo en Gazebo y exportar waypoints.yaml
python3 src/uav_ip/scripts/world_generator.py --task-id test_001 --seed 42 --output-dir src/uav_ip/generated_worlds
```

<p align="center">
   <img src='docs\assets\gazebo_world_init.png' width="60%">
   <br>
   <em>Figura 2: Mundo procedural generado con landmarks (aros cuadrados, toroides y postes cilíndricos).</em>
</p>

### Paso 2: Lanzar el Simulador PX4 SITL (Terminal 1)
```bash
cd ~/PX4-Autopilot
export GZ_VERSION=harmonic
export PX4_GZ_WORLD=test_001
export PX4_GZ_MODEL_POSE="0,0,0.3,0,0,0"

# Lanzar con el modelo estándar ligero gz_x500
make px4_sitl gz_x500
```

### Paso 3: Lanzar el Agente XRCE-DDS (Terminal 2)
```bash
export ROS_DOMAIN_ID=33
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
MicroXRCEAgent udp4 -p 8888
```
*(Nota: Con `gz_x500` y observación privilegiada, **ya no se necesita** ejecutar `ros_gz_bridge` para cámaras).*

### Paso 4: Grabar Rosbag y Volar con el Piloto Experto (Terminal 3 y 4)
En la **Terminal 3** (Grabador de rosbag ligero, $< 5\text{ MB}$):
```bash
cd ~/ros2_ws && source install/setup.bash
export ROS_DOMAIN_ID=33

# Borrar rosbag previo si existe
rm -rf rosbag_test_001

# Grabar tópicos de telemetría compacta
ros2 bag record -o rosbag_test_001 \
  /fmu/out/vehicle_odometry \
  /fmu/out/timesync_status \
  /fmu/in/trajectory_setpoint \
  /fmu/out/vehicle_land_detected \
  /fmu/out/vehicle_status_v4 \
  /fmu/out/vehicle_local_position_v1 \
  /fmu/in/vehicle_command
```

En la **Terminal 4** (Piloto experto modular):
```bash
cd ~/ros2_ws && source install/setup.bash
export ROS_DOMAIN_ID=33

# Ejecutar el controlador hacia los waypoints
ros2 run uav_ip waypoint_controller --ros-args \
  -p waypoints_file:=src/uav_ip/generated_worlds/test_001/waypoints.yaml
```

Cuando el dron aterrice en la plataforma verde, el piloto experto finalizará automáticamente con exit code 0. Detén el grabador en la Terminal 3 con `Ctrl+C`.

### Paso 5: Generar el Grafo HeteroData con LiDAR Sintético 360°
```bash
cd ~/ros2_ws
python3 src/uav_ip/scripts/bag_to_heterodata.py \
  --bag-dir rosbag_test_001 \
  --waypoints src/uav_ip/generated_worlds/test_001/waypoints.yaml \
  --output test_001.pt \
  --scale 0.1 \
  --fps-points 2048
```
*(Nota: Por defecto, `bag_to_heterodata.py` recorta automáticamente los tiempos estáticos en tierra antes del despegue y tras el aterrizaje. Puedes desactivarlo con `--no-trim-idle` o cambiar el umbral con `--trim-start liftoff`).*

El archivo `test_001.pt` resultante contendrá el grafo heterogéneo con:
- `data['scene'].pos`: nubes de 2048 puntos de los landmarks visibles a $\le 10\text{ m}$ en marco egocéntrico **Horizontal FLU** (estabilizado por gravedad con roll=0, pitch=0), escaladas a $s = 0.1$.
- `data['agent'].x`: poses $SE(3)$ del dron ($x, y, z$ escalados por $s = 0.1$, cuaternión $q_w, q_x, q_y, q_z$) más los 6 keypoints del cuadricóptero para visualización y aprendizaje geométrico.
- `data['agent'].y`: horizonte de $T=8$ acciones relativas $T_{EA}^{(t,k)} \in SE(3)$ muestreadas de la trayectoria de vuelo real a $\sim 10\text{ Hz}$.
- `data['aug']`: nodos de aumento sintético para recuperación (*30% Tube Augmentation*) con poses perturbadas $\tilde{T}_{WE}$, nubes sintéticas recalculadas $\tilde{P}_{\text{scene}}$, perturbación del 10% en el estado binario de vuelo $\tilde{s}_t$, y secuencias de acción correctivas orientadas a retornar a la trayectoria nominal.
- Demos de contexto reducidas a $L=10$ keyframes temporales representativos.

### Paso 6: Visualizar e Inspeccionar los Datos en 3D (Instantáneo, Animación y Recuperación)
Para auditar visualmente tanto la nube LiDAR egocéntrica como la trayectoria del dron y las maniobras de recuperación:

```bash
cd ~/ros2_ws

# 1. Reproducir el vuelo completo como ANIMACION tipo video (GUI o GIF):
python3 src/uav_ip/scripts/visualize_dataset.py --file test_001.pt --animate --output vuelo_animado.gif

# 2. Visualizar la recuperación y Tube Augmentation (poses perturbadas y trayectorias correctivas):
python3 src/uav_ip/scripts/visualize_dataset.py --file test_001.pt --show-aug --output tube_recovery.png

# 3. Visualizar un fotograma estático específico (ej. paso t=10):
python3 src/uav_ip/scripts/visualize_dataset.py --file test_001.pt --step 10 --output vista_lidar.png

# 4. O verificar directamente el LiDAR sintético sobre los waypoints sin necesidad de volar:
python3 src/uav_ip/scripts/visualize_dataset.py \
  --waypoints src/uav_ip/generated_worlds/test_001/waypoints.yaml \
  --drone-pose 0.0 0.0 1.5 0.0 \
  --output test_lidar.png

# También puede verificarse eligiendo el step de la trayectoria
python3 src/uav_ip/scripts/visualize_dataset.py \
  --file test_001.pt \
  --step 110 \
  --output vista_lidar_t110.png

```
*(Si ejecutas en WSL2 con WSLg o servidor gráfico, `--animate` reproducirá el vuelo en una ventana interactiva continua; si estás en entorno headless o especificas `.gif`, guardará la animación completa en video GIF).*

<p align="center">
   <img src='docs\assets\gazebo_world_1.png' width=30% alt="Mundo en Gazebo" />
   <img src='docs\assets\lidar_sintetico_1(t_110).png' width=68% alt="LiDAR 360° en t=110" />
   <br>
   <em>Figura 3: Correspondencia entre el entorno físico simulado en Gazebo (izquierda) y la observación LiDAR sintética procesada para la red neuronal (derecha).</em>

</p>

-----

## 🤖 Procedimiento 2: Generación Automatizada del Dataset Completo

Para generar cientos o miles de demostraciones de forma completamente autónoma ($N_{\text{tasks}} \times N_{\text{demos}}$), se utiliza el orquestador `data_collector.py`.

### 1. Ejecución del Colector Automático
En una única terminal:
```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=33

# Ejemplo: 5 pseudo-tareas distintas, con 10 episodios por tarea (50 vuelos en total)
python3 src/uav_ip/scripts/data_collector.py \
  --num-tasks 5 \
  --episodes-per-task 10 \
  --base-seed 100 \
  --dataset-dir datasets
```

### Opciones Disponibles de `data_collector.py`
- `--num-tasks <N>`: Número de pseudo-tareas únicas a generar (mundos con variaciones espaciales aleatorias).
- `--episodes-per-task <M>`: Demostraciones por cada tarea (para conformar $N-1$ demos de contexto y 1 de query).
- `--min-landmarks <min>`: Cantidad mínima de landmarks acotados por tarea (default: 2).
- `--max-landmarks <max>`: Cantidad máxima de landmarks acotados por tarea (default: 5).
- `--num-landmarks <K>`: Forzar cantidad exacta fija de landmarks en todas las tareas (anula min/max).
- `--base-seed <S>`: Semilla base para reproducibilidad estricta.
- `--resume`: Si se interrumpió la recolección, reanuda automáticamente omitiendo tareas y episodios ya completados con éxito.
- `--episode-timeout <seg>`: Timeout de seguridad por vuelo (default: 360s).

### Estructura de Salida del Dataset
`data_collector.py` organiza los datos de forma limpia y autocontenida:
```text
datasets/
├── task_0000/
│   ├── waypoints.yaml             # Definición ground-truth de la tarea y landmarks
│   ├── episode_00/
│   │   ├── rosbag/                # Rosbag .mcap (< 5 MB con telemetría ligera)
│   │   ├── waypoints.yaml         # Copia local para postproceso inmediato
│   │   ├── episode_meta.json      # Metadatos del vuelo (timestamp, status, tópicos)
│   │   └── logs/                  # Logs individuales de px4, pilot y xrce
│   ├── episode_01/
│   └── ...
└── task_0001/
    └── ...
```

### 2. Validación Post-Vuelo de Episodios (`validate_episode.py`)
Verificador de configuración, grabación y geometría de misión independiente del piloto. Lee `waypoints.yaml`, `episode_meta.json` y el `.mcap` de cada episodio y decide `accepted`/`rejected` con estados desglosados (`configuration_status`, `recording_status`, `mission_status`, `validation_status`) y `rejection_reasons`, devolviendo código de salida `0` (todos aceptados), `1` (al menos un rechazo) o `2` (error de uso/argumentos):

```bash
# Reporte en terminal sobre todo el dataset (no modifica archivos)
python3 src/uav_ip/scripts/validate_episode.py --dataset-dir datasets
# Validar un único episodio en formato JSON
python3 src/uav_ip/scripts/validate_episode.py --episode-dir datasets/task_0001/episode_00 --json
# Persistir resultados de forma atómica en cada episode_meta.json
python3 src/uav_ip/scripts/validate_episode.py --dataset-dir datasets --update-meta
```

**Integración en línea y desambiguación analítica del reloj (Fase 2):**
- **Invocación automática:** `data_collector.py` invoca automáticamente `validate_episode_directory(ep_dir, update_meta=True)` tras cerrar limpiamente el bag con `SIGINT`.
- **Desambiguación del Patrón B:** Si `VehicleOdometry` experimenta un salto de reloj en SITL por recalibración periódica de offset UNIX de `uxrce_dds_client`, se correlaciona analíticamente en una ventana de $\pm 0.05\text{ s}$ contra `TimesyncStatus.estimated_offset`. Si la magnitud coincide, se repara el reloj en `reconstructed_time_s` (`timesync_offset_adjustment_repaired`) y el episodio se clasifica como válido (`usable_for_validation = True`).
- **Conservación estricta de intentos:** Los intentos fallidos se conservan intactos con sus motivos de rechazo. Los reintentos con semilla derivada se guardan en `episode_XX_attempt_YY` sin sobrescribir jamás el intento original `episode_XX`.
- **Filtro conservador bajo hipótesis explícitas:** Envolvente esférica 3D de 0,30 m (0,25 m de radio físico XY de chasis/motores `x500` + 0,05 m de margen de seguridad), cota inferior de discretización $d_{\text{inferior}} = \max(0, d_{\text{muestreada}} - h/2)$ con paso $h = 0,02\text{ m}$, y obstáculos estáticos. Criterios, hipótesis y limitaciones: [`docs/JUSTIFICACION_REVISION_TECNICA.md`](docs/JUSTIFICACION_REVISION_TECNICA.md).

### 3. Auditoría en 3 Capas del Dataset y Partición Formal (`audit_dataset.py`)
Una vez finalizada la recolección (o durante una pausa), se ejecuta el script de auditoría en 3 capas para verificar la integridad del dataset y generar las particiones oficiales:

```bash
python3 src/uav_ip/scripts/audit_dataset.py --dataset-dir datasets
```

Este comando realiza las siguientes validaciones estructuradas en 3 capas:
- **Capa 1 (Vuelos crudos y metadatos):**
  - Reporta tasas de aceptación por intento original (`original_attempt_acceptance_rate`) por separado de la tasa post-reintentos (`post_retry_slot_acceptance_rate`).
  - Desglose exhaustivo de motivos de rechazo (`rejection_reasons`).
  - Detección de colisiones de firmas semánticas y control de variabilidad espacial de waypoints.
- **Capa 2 (Tensores de aprendizaje `.pt`):**
  - Audita finitud estricta (cero NaNs e Infs en todos los tensores).
  - Verificación dimensional: `scene.pos` $(T, 2048, 3)$, `agent.action_se3` $(T, 8, 4, 4)$, `agent.valid_mask` $(T, 8)$.
  - Ortogonalidad estricta en $SO(3)$ ($R^\top R = I_3$, $\det R = +1.0$) en matrices de rotación y transformaciones relativas.
  - Verificación del horizonte físico continuo de $0.8\text{ s}$ ($\Delta t = 0.1\text{ s}$) y cota esférica sólida $\|\delta p\|_2 \le 0.60\text{ m}$ en Tube Augmentation.
- **Capa 3 (Particionado oficial y splits):**
  - Exige un umbral mínimo de episodios aceptados ($\ge 5$, configurable con `--min-accepted-episodes`) antes de incluir una tarea en los splits.
  - Genera particiones mutuamente excluyentes en `datasets/splits/`:
    - `train_tasks.json` ($\sim 70\%$ de las firmas semánticas).
    - `val_tasks.json` ($\sim 15\%$ de las firmas).
    - `held_out_tasks.json` ($\sim 15\%$ de firmas nunca vistas en entrenamiento, para evaluar generalización *in-context*).
    - `dataset_summary.json` (resumen estadístico global de las 3 capas).

*(Para realizar un chequeo rápido en terminal sin generar archivos JSON, puedes agregar el flag `--no-export`).*

### 4. Conversión por Lotes a Formato PyTorch Geometric
Una vez validado el dataset, puedes convertir automáticamente los episodios aceptados a grafos `HeteroData` (`data.pt`) utilizando el modo batch multiproceso de `bag_to_heterodata.py`:

```bash
# Activar entorno ML
source ~/venv_ip/bin/activate

# Conversión masiva en paralelo (ej. 8 trabajadores / núcleos):
python3 src/uav_ip/scripts/bag_to_heterodata.py \
  --dataset-dir datasets \
  --workers 8 \
  --fps-points 2048 \
  --scale 0.1
```

**Ventajas y Características del Procesamiento por Lotes (Fase 2):**
- **Filtrado Estricto de Postproceso:** Ignora por defecto cualquier episodio con `validation_status != "accepted"` en `episode_meta.json` (desactivable con `--include-failed`).
- **Remuestreo Temporal Uniforme a 10 Hz:** Interpola poses y observaciones a $\Delta t = 0.1\text{ s}$ exactos sobre `reconstructed_time_s`, desenvolviendo el ángulo de guiñada (`np.unwrap`) para evitar singularidades en $\pm \pi$, garantizando que el horizonte $T=8$ cubra exactamente $0.8\text{ s}$ físicos.
- **Máscara de Validez de Acción (`valid_mask`):** Incorpora un tensor booleano $(T, 8)$ que marca como válidos los pasos con trayectoria futura real y distingue los últimos 7 timesteps donde el horizonte queda clampeado al aterrizaje estacionario.
- **Tube Augmentation con Seguridad Geométrica:** Muestreo en esfera euclídea sólida $\|\delta p\|_2 \le 0.60\text{ m}$, descartando perturbaciones que violen el clearance de colisión ($\ge 0.30\text{ m}$) contra landmarks o que salgan de la apertura libre en maniobras `traverse` de aros.
- **Procesamiento Multiproceso y Aceleración SIMD C:** Distribuye los episodios uniformemente entre núcleos (`--workers`), con submuestreo FPS compilado con SIMD vectorizado (`-O3 -march=native -ffast-math`).
- **Keyframes de Contexto:** Extrae automáticamente $L=10$ hitos temporales para estructurar el condicionamiento *in-context*.

---
### 🎯 Embodiment y Observación Privilegiada (LiDAR 360°)
- **Embodiment:** El dron se modela conceptualmente con un sensor **LiDAR 3D omnidireccional 360°** (análogo en el mundo real a un *Livox Mid-360*), implementado mediante **observación privilegiada** en modo `range` ($R_{\text{max}} = 10.0\text{ m}$).
- **Preservación de la Hipótesis de Markov:** La visión 360° elimina la ceguera retrospectiva (*backward blindness*) al cruzar aros o superar obstáculos, reteniendo simultáneamente referencias espaciales al frente y detrás sin requerir modelos recurrentes en Fase 1.
- **Simulación Ligera (`gz_x500`):** Se utiliza el modelo estándar `gz_x500` de PX4 en lugar de `gz_x500_depth`, acelerando drásticamente el *Real-Time Factor* (RTF) al eliminar el renderizado de cámaras de profundidad en Gazebo.
- **Eficiencia de Almacenamiento:** Los rosbags registran exclusivamente telemetría ligera de PX4 (**$< 5\text{ MB}$ por demo**, frente a los 5–6 GB del enfoque obsoleto con cámara).
- **Escala Métrica ($s = 0.1$):** Se aplica $s = 0.1$ a coordenadas y deltas para evitar el aliasing espacial en la codificación posicional sinusoidal de Instant Policy.

---

## 📚 Documentación y Decisiones Arquitectónicas

Para profundizar en los fundamentos teóricos y evitar reabrir discusiones de diseño resueltas, consulta:
- [ESTRATEGIA_TEORICA.md](/docs/ESTRATEGIA_TEORICA.md): Marco teórico, preservación de Markov, formulación de grafos y taxonomía de pseudo-tareas.
- [PLAN_IMPLEMENTACION.md](/docs/PLAN_IMPLEMENTACION.md): Hoja de ruta por hitos (0 a 4), estado de componentes de software y criterios de aceptación.
- [DECISIONES_DISENO.md](/docs/DECISIONES_DISENO.md): Registro de decisiones arquitectónicas (ADRs), descarte de cámara de profundidad, marcos de referencia NED/ENU y escala métrica $s = 0.1$.
- [TROUBLESHOOTING.md](/docs/TROUBLESHOOTING.md): Guía de solución para errores de EKF2, penetración de spawn en Gazebo y puertos XRCE-DDS.