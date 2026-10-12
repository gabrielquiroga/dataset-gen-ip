# UAV Instant Policy: Data Collection & Synthetic LiDAR Pipeline

## 📖 Descripción del Proyecto
Este proyecto implementa un *pipeline* automatizado de recolección de datos en simulación para entrenar un modelo de Aprendizaje por Imitación en Contexto (*In-Context Imitation Learning* - ICIL), basado en la arquitectura **Instant Policy** (Vosylius y Johns, ICLR 2025).

El sistema utiliza **ROS 2 Jazzy**, **Gazebo Harmonic** y **PX4 Autopilot** para simular un VANT (Vehículo Aéreo No Tripulado) que navega procedimentalmente a través de circuitos dinámicos de landmarks (aros cuadrados, toroides, postes cilíndricos y plataformas de aterrizaje).

### 🎯 Embodiment y Observación Privilegiada (LiDAR 360°)
- **Embodiment:** El dron se modela conceptualmente con un sensor **LiDAR 3D omnidireccional 360°** (análogo en el mundo real a un *Livox Mid-360*), implementado mediante **observación privilegiada** en modo `range` ($R_{\text{max}} = 10.0\text{ m}$).
- **Preservación de la Hipótesis de Markov:** La visión 360° elimina la ceguera retrospectiva (*backward blindness*) al cruzar aros o superar obstáculos, reteniendo simultáneamente referencias espaciales al frente y detrás sin requerir modelos recurrentes en Fase 1.
- **Simulación Ligera (`gz_x500`):** Se utiliza el modelo estándar `gz_x500` de PX4 en lugar de `gz_x500_depth`, acelerando drásticamente el *Real-Time Factor* (RTF) al eliminar el renderizado de cámaras de profundidad en Gazebo.
- **Eficiencia de Almacenamiento:** Los rosbags registran exclusivamente telemetría ligera de PX4 (**$< 5\text{ MB}$ por demo**, frente a los 5–6 GB del enfoque obsoleto con cámara).
- **Escala Métrica ($s = 0.1$):** Se aplica $s = 0.1$ a coordenadas y deltas para evitar el aliasing espacial en la codificación posicional sinusoidal de Instant Policy.

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
│       │   ├── task_generator.py      # Generador de recetas semánticas de tareas
│       │   ├── world_generator.py     # Generador de mundos SDF y waypoints.yaml
│       │   ├── synthetic_lidar.py     # Generador de nubes LiDAR 360° sintéticas (FPS 2048 pts)
│       │   ├── bag_to_heterodata.py   # Conversor de rosbag a grafos PyTorch Geometric (.pt)
│       │   ├── data_collector.py      # Orquestador automático del dataset (doble bucle)
│       │   ├── test_flight.sh         # Script rápido para vuelo de prueba
│       │   └── kill_sim.sh            # Limpieza agresiva de procesos huérfanos
│       └── uav_ip/                    # Módulo Python principal
│           ├── __init__.py
│           ├── base_pilot.py          # Máquina de estados, QoS y protocolo seguro de armado
│           ├── waypoint_controller.py # Piloto experto feed-forward (traverse, overfly, slalom)
│           ├── tracking_controller.py # Piloto experto feedback (orbit)
│           └── synthetic_lidar.py     # Módulo empotrado de generación LiDAR
├── datasets/                      # Salida del dataset estructurado por tarea y episodio
├── docs/                          # Documentación técnica y fundamentos
│   ├── DECISIONES_DISENO.md       # Memoria persistente de decisiones y fundamentación matemática
│   ├── PLAN_ESTRATEGIA.md         # Estrategia general adaptada de Instant Policy
│   ├── DATASET_REQUIREMENTS.md    # Requisitos arquitectónicos del dataset
│   └── TROUBLESHOOTING.md         # Resolución de incidencias comunes
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
  - `rclpy`, `px4_msgs`, `sensor_msgs` (Ecosistema ROS 2)
  - `numpy`, `yaml` (Generación geométrica y procedural)
  - `rosbags`, `torch`, `torch_geometric` (Procesamiento offline a HeteroData)

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
  -p waypoints_file:=/home/gabrielq/ros2_ws/src/uav_ip/generated_worlds/test_001/waypoints.yaml
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
El archivo `test_001.pt` resultante contendrá el grafo con:
- `data['scene'].pos`: nubes de 2048 puntos de los landmarks visibles a $\le 10\text{ m}$ en marco FLU egocéntrico escaladas a $s = 0.1$.
- `data['agent'].x`: trayectoria real ejecutada ($s = 0.1$).
- `data['agent'].y`: acciones / setpoints futuros ($s = 0.1$).

### Paso 6: Visualizar e Inspeccionar los Datos en 3D
Para auditar visualmente tanto la nube LiDAR egocéntrica como la trayectoria del dron:

```bash
cd ~/ros2_ws

# 1. Visualizar la nube LiDAR de un instante específico (ej. paso t=10) del archivo .pt o .npz:
python3 src/uav_ip/scripts/visualize_dataset.py --file test_001.pt --step 10 --output vista_lidar.png

# 2. O verificar directamente el LiDAR sintético sobre los waypoints sin necesidad de volar:
python3 src/uav_ip/scripts/visualize_dataset.py \
  --waypoints src/uav_ip/generated_worlds/test_001/waypoints.yaml \
  --drone-pose 0.0 0.0 1.5 0.0 \
  --output test_lidar.png
```
*(Si ejecutas en WSL2 con WSLg o servidor gráfico, se abrirá una ventana 3D interactiva de Matplotlib. Si estás en entorno headless, se guardará automáticamente la imagen `.png` en alta resolución).*

---

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

### 2. Conversión por Lotes a Formato PyTorch Geometric
Una vez recolectadas las tareas, puedes convertir todos los episodios a `.pt` iterando con `bag_to_heterodata.py`:
```bash
# Ejemplo de procesamiento por lotes en bash:
for ep_dir in datasets/task_*/episode_*/; do
  if [ -d "$ep_dir/rosbag" ]; then
    echo "Procesando $ep_dir..."
    python3 src/uav_ip/scripts/bag_to_heterodata.py \
      --bag-dir "$ep_dir/rosbag" \
      --waypoints "$ep_dir/waypoints.yaml" \
      --output "$ep_dir/data.pt"
  fi
done
```

---

## 📚 Documentación y Decisiones Arquitectónicas

Para profundizar en los fundamentos teóricos y evitar reabrir discusiones de diseño resueltas, consulta:
- [DECISIONES_DISENO.md](file:///home/gabrielq/ros2_ws/docs/DECISIONES_DISENO.md): Detalle técnico sobre el dilema de la cámara vs. LiDAR 360°, la observación privilegiada, la unificación analítica de marcos NED/ENU, la monotonicidad en pasillos 1D, el armado offboard seguro y la escala $s = 0.1$.
- [PLAN_ESTRATEGIA.md](file:///home/gabrielq/ros2_ws/docs/PLAN_ESTRATEGIA.md): Plan maestro de pseudo-tareas, taxonomía, formulación de grafos y fases de escalado Hito 0 a Hito C.
- [DATASET_REQUIREMENTS.md](file:///home/gabrielq/ros2_ws/docs/DATASET_REQUIREMENTS.md): Requisitos matemáticos y estructurales indispensables para la convergencia de Instant Policy.
- [TROUBLESHOOTING.md](file:///home/gabrielq/ros2_ws/docs/TROUBLESHOOTING.md): Guía de solución para errores de EKF2, penetración de spawn en Gazebo y puertos XRCE-DDS.