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
- **PX4_GZ_MODEL_POSE y Explosión de Físicas:** La plataforma de despegue en el SDF es una caja que se extiende hasta Z=0.1m. Si el dron (`x500_depth`) spawnea en `Z=0.2`, su tren de aterrizaje penetra la caja, lo que causa una explosión en el motor de físicas al resolver la colisión. Esto genera errores de `High Gyro Bias` y `GPS Vert Speed Drift too high` en PX4, denegando el armado (Arming denied). **SOLUCIÓN:** Usar `PX4_GZ_MODEL_POSE="0,0,0.8,0,0,0"` para asegurar que el dron nazca en el aire y caiga suavemente.
- **Zombies de Gazebo (`gz sim`):** Al matar el proceso de PX4 (`make px4_sitl` o el ejecutable `px4`), el servidor de Gazebo en segundo plano **no muere**. Si se lanza PX4 nuevamente, se inyectará en el mundo zombie corrupto, causando fallas masivas de lectura de sensores (`ekf2 missing data`, `barometer 0 missing`, etc). **SOLUCIÓN:** Siempre matar los huérfanos con `pkill -f "gz sim" ; pkill -f "ruby.*gz" ; pkill -f "bin/px4"` antes de un nuevo lanzamiento. El orquestador ya incluye esta limpieza automática.
- **Comunicación y Particiones:** Para que el stack completo se comunique sin conflictos y en la misma red, se debe exportar explícitamente `ROS_DOMAIN_ID=33`, `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` y `GZ_PARTITION=sim`. Si el `MicroXRCEAgent` se levanta sin el Domain ID correcto, el piloto experto jamás logrará comunicarse con PX4.

## Topicos PX4 Verificados (del rosbag existente)
- `/fmu/in/trajectory_setpoint` (TrajectorySetpoint)
- `/fmu/out/vehicle_odometry` (VehicleOdometry) - SIN sufijo _v1
- `/fmu/out/vehicle_local_position_v1` (VehicleLocalPosition) - CON sufijo _v1
- `/depth_camera/points` (PointCloud2)

## Variables de Entorno para Lanzamiento PX4
```bash
export GZ_VERSION=harmonic
export PX4_GZ_WORLD=tf_circuito_aros        # o el nombre del mundo generado
export PX4_GZ_MODEL_POSE="0,0,0.8,0,0,0"    # Altura de Z=0.8 obligatoria para evitar High Gyro Bias
export GZ_PARTITION=sim
export GZ_IP=127.0.0.1
export ROS_DOMAIN_ID=33
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
# Luego: make px4_sitl gz_x500_depth
```

## Tareas Activas / Inmediatas
1. **Fase de Planeación:** (COMPLETADO) - Creación del `docs/PLAN_DETALLADO_IMPLEMENTACION.md` que traza la hoja de ruta hacia un motor de pseudo-tareas con hitos incrementales para el Hito 0 (Pipeline core) y Hito A (Controladores dinámicos).
2. **Paso 1: task_generator.py & world_generator.py** (PENDIENTE) - Refactorizar para generar secuencias concretas de primitivas a partir del catálogo (distintas formas/postes) en vez de solo 3 aros estáticos.
3. **Paso 2: waypoint_controller.py & tracking_controller.py** (PENDIENTE) - Expandir el control para interpretar las secuencias dinámicas (Feed-forward para traverse/overfly y Feedback para orbit).
4. **Paso 3: data_collector.py** (EN PROGRESO) - Orquestador automático con bucle doble (tasks x episodios).
5. **Paso 4: ml_pipeline/extractor.py** - Pipeline de extraccion con uv + rosbags (`bag_to_heterodata.py` base completado).

## Deuda Tecnica
- **Segmentacion:** Heuristica espacial para borrar suelo de la nube de puntos.
- **Marco Egocentrico:** Transformar odometria global a desplazamientos relativos en base_link.
- **Keyframing:** Heuristica de distancia acumulada + cambio angular para extraer ~10-15 keyframes por episodio.
