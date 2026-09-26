# Memoria de Estado del Proyecto

*Este archivo rastrea el progreso del proyecto, las decisiones arquitectonicas tomadas y las tareas pendientes. Debe ser actualizado por el agente conforme se avance.*

## Logros Completados
- **Setup Base:** Instalacion exitosa de Ubuntu 24.04 (WSL2), ROS 2 Jazzy, Gazebo Harmonic y PX4 SITL.
- **Comunicacion:** Enlace DDS establecido mediante Micro XRCE-DDS Agent.
- **Piloto Experto (v1):** Script Python creado (`expert_pilot.py`) que publica `TrajectorySetpoint` sincronizado con el timestamp de PX4, cruzando exitosamente aros fijos.
- **Integracion de Sensores:** `ros_gz_bridge` configurado correctamente para extraer `PointCloudPacked` hacia `sensor_msgs/PointCloud2`.
- **Validacion de Datos:** Grabacion exitosa de rosbags (`.mcap` de 6+ GiB) conteniendo Odometria, Acciones y Nubes de Puntos, validados visualmente con Foxglove Studio.
- **Piloto Experto (v2 - Parametrico):** Refactorizacion completa de `expert_pilot.py` para leer waypoints dinamicamente desde un archivo YAML (parametro ROS 2 `waypoints_file`). Genera automaticamente waypoints pre-alineacion y post-salida para cada aro usando trigonometria (cos/sin del yaw del aro). Incluye salida limpia con `SystemExit(0)` para deteccion por el orquestador. Validado con backward-compatibility test: 9/9 waypoints coinciden con el circuito original. Probado en simulacion: vuelo exitoso identico al v1.
- **Infraestructura de Paquete:** `setup.py` actualizado con entry_point `expert_pilot` y data_files para instalar mundos SDF y YAML. Build exitoso con `colcon build`.

## Decisiones Arquitectonicas
- **Separacion ROS 2 / ML:** El dominio ROS 2 vive en `uav_ip` (colcon). El dominio ML vivira en `ml_pipeline/` (uv + venv aislado).
- **Inyeccion Segura de Mundos:** Copiar SDF a `~/PX4-Autopilot/Tools/simulation/gz/worlds/` y usar `PX4_GZ_WORLD`. Cero modificaciones al codigo fuente de PX4.
- **Esquema YAML de Circuito:** Los hoops se definen como {name, x, y, z, yaw} en frame Gazebo ENU. El piloto auto-genera waypoints de approach/exit a APPROACH_OFFSET metros del centro.
- **Senal de Mision Completa:** El piloto lanza `SystemExit(0)` al aterrizar. El orquestador detecta exit code 0.
- **Dataset:** 2 tareas x 15 episodios por tarea (prioriza variacion intra-tarea, alineado al in-context learning del paper).

## Gotchas Criticos (NO OLVIDAR)
- **PX4_GZ_MODEL_POSE:** La plataforma de despegue en el SDF esta a Z=0.1m (pose Z=0.05 + mitad del box 0.05). El dron por defecto spawna en Z=0, quedando DENTRO de la plataforma y no puede despegar. SOLUCION: exportar `PX4_GZ_MODEL_POSE="0,0,0.2,0,0,0"` antes de lanzar PX4. El orquestador DEBE setear esta variable.

## Topicos PX4 Verificados (del rosbag existente)
- `/fmu/in/trajectory_setpoint` (TrajectorySetpoint)
- `/fmu/out/vehicle_odometry` (VehicleOdometry) - SIN sufijo _v1
- `/fmu/out/vehicle_local_position_v1` (VehicleLocalPosition) - CON sufijo _v1
- `/depth_camera/points` (PointCloud2)

## Variables de Entorno para Lanzamiento PX4
```bash
export GZ_VERSION=harmonic
export PX4_GZ_WORLD=tf_circuito_aros        # o el nombre del mundo generado
export PX4_GZ_MODEL_POSE="0,0,0.2,0,0,0"    # Evita colision con plataforma de salida
# Luego: make px4_sitl gz_x500_depth
```

## Tareas Activas / Inmediatas
1. **Paso 1: world_generator.py** - Script que toma circuito_aros_01.sdf como plantilla, aplica variaciones aleatorias a poses de aros, y exporta nuevo .sdf + waypoints.yaml.
2. **Paso 2: data_collector.py** - Orquestador con bucle doble (tasks x episodios).
3. **Paso 3: ml_pipeline/extractor.py** - Pipeline de extraccion con uv + rosbags.

## Deuda Tecnica
- **Segmentacion:** Heuristica espacial para borrar suelo de la nube de puntos.
- **Marco Egocentrico:** Transformar odometria global a desplazamientos relativos en base_link.
- **Keyframing:** Heuristica de distancia acumulada + cambio angular para extraer ~10-15 keyframes por episodio.
