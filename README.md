# UAV Instant Policy: Data Collection Pipeline

## 📖 Descripción del Proyecto
Este proyecto implementa un *pipeline* automatizado de recolección de datos en simulación para entrenar un modelo de Aprendizaje por Imitación en Contexto (*In-Context Imitation Learning*), basado en la arquitectura **Instant Policy**. 

El sistema utiliza **ROS 2**, **Gazebo Harmonic** y **PX4 Autopilot** para simular un VANT (Vehículo Aéreo No Tripulado) que navega a través de circuitos dinámicos. El objetivo principal es generar cientos de "pseudo-demostraciones" (archivos `.mcap`) que vinculen Nubes de Puntos 3D, telemetría y desplazamientos reales en el espacio SE(3), preparados específicamente para el entrenamiento con Redes Neuronales de Grafos (GNN).

## 🗂️ Estructura del Espacio de Trabajo
Dado que estamos interactuando con ecosistemas de compilación estrictos (como `colcon` de ROS 2), el proyecto debe estructurarse como un paquete de ROS 2 dentro de tu `ros2_ws`:

```text
ros2_ws/
├── src/
│   ├── px4_msgs/                  # Definiciones de tópicos (VehicleOdometry, TrajectorySetpoint, etc.)
│   ├── px4_offboard/              # Scripts de vuelo, control PID/MPC y waypoints
│   └── uav_ip/
│       ├── package.xml
│       ├── setup.py
│       ├── resource/
│       ├── worlds/                 # Archivos .sdf exportados desde Gazebo (Ej. circuito_aros.sdf)
│       ├── scripts/                # Utilidades para copiar worlds a PX4 y lanzar el bridge
│       └── uav_ip/                 # Módulo Python principal
│           ├── __init__.py
│           ├── expert_pilot.py     # Nodo ROS 2: Controlador experto (MPC/PID)
│           ├── env_randomizer.py   # Nodo ROS 2: Mueve obstáculos y reinicia simulaciones
│           └── dataset_extractor.py# Script (sin ROS): Convierte .mcap a tensores de PyTorch
├── build/
├── install/
└── log/
```
*(Nota: Los archivos `.sdf` de la carpeta `worlds` deben ser copiados mediante un script bash a `~/PX4-Autopilot/Tools/simulation/gz/worlds/` antes de ejecutar el simulador).*

## ⚙️ Prerrequisitos del Sistema
- **SO:** Windows 11 con WSL2 (Ubuntu 24.04 LTS).
- **Simulación:** PX4 Autopilot (Main branch compilado para SITL) y Gazebo Harmonic.
- **Middleware:** ROS 2 Jazzy con `micro_ros_agent` (XRCE-DDS).
- **Python:** Python 3.12+ con las siguientes librerías (`pip install --break-system-packages`):
  - `rclpy`, `px4_msgs`, `sensor_msgs` (Dependencias ROS)
  - `rosbags`, `open3d`, `numpy`, `torch` (Dependencias de ML)

## 🚀 Flujo de Ejecución (Pipeline)

1. **Preparar el entorno:** En cada terminal, asegúrate de cargar ROS 2 y fijar el dominio de red.
   ```bash
   source /opt/ros/jazzy/setup.bash
   export ROS_DOMAIN_ID=33
   ```
   Se configuró un script de bash para automatizar esta carga y configurar algunas variables de entorno de ROS2 y Gazebo, que se pueden omitir en pasos siguientes. 
   ```bash
   load_ros
   ```
   Las variables configuradas por defecto son:
   - ROS_DOMAIN_ID:            33
   - DISCOVERY_RANGE:          LOCALHOST
   - GZ_VERSION:               harmonic
   - GZ_IP:                    127.0.0.1
   - GZ_PARTITION:             sim
   - PX4_GZ_WORLD:             tf_circuito_aros
2. **Lanzar Simulación (Terminal 1):** Iniciar PX4 con el mundo personalizado.
   ```bash
   cd ~/PX4-Autopilot
   export GZ_VERSION=harmonic
   export PX4_GZ_WORLD=tf_circuito_aros
   make px4_sitl gz_x500_depth
   ```
3. **Lanzar Agent y Bridge (Terminal 2 y 3):** Conectar PX4 con ROS 2.
   ```bash
   MicroXRCEAgent udp4 -p 8888
   # En otra terminal:
   ros2 run ros_gz_bridge parameter_bridge "/depth_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked"
   ```
4. **Ejecutar Recolección (Terminal 4):** Lanza el randomizador y el piloto experto.
   ```bash
   colcon build --packages-select uav_imitation_learning
   source install/setup.bash
   ros2 run uav_imitation_learning env_randomizer
   ```