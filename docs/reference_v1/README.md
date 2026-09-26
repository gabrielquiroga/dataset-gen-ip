# Referencia v1 — Piloto Experto Original

Este directorio contiene la version original del piloto experto con waypoints
hardcodeados, guardado como referencia en caso de necesidad de rollback.

## Archivos

- `piloto_experto_v1.py` — Codigo original con waypoints fijos
- Este `README.md`

## Mundo asociado

El mundo SDF asociado es `circuito_aros_01.sdf` (copiado a PX4 como
`tf_circuito_aros.sdf`). Ambos estan en:
- `src/uav_ip/worlds/circuito_aros_01.sdf`
- `~/PX4-Autopilot/Tools/simulation/gz/worlds/tf_circuito_aros.sdf`

## Como lanzar (el flujo original manual)

```bash
# Terminal 1: PX4 + Gazebo
cd ~/PX4-Autopilot
export GZ_VERSION=harmonic
export PX4_GZ_WORLD=tf_circuito_aros
export PX4_GZ_MODEL_POSE="0,0,0.2,0,0,0"   # CRITICO: sin esto el dron colisiona con la plataforma
make px4_sitl gz_x500_depth

# Terminal 2: XRCE-DDS Agent
MicroXRCEAgent udp4 -p 8888

# Terminal 3: Bridge de nube de puntos
source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=33
ros2 run ros_gz_bridge parameter_bridge "/depth_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked"

# Terminal 4: Piloto experto v1
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=33
python3 piloto_experto_v1.py
```

> **NOTA:** La plataforma de salida esta a Z=0.1m (pose 0.05 + mitad del box 0.05).
> Sin `PX4_GZ_MODEL_POSE` el dron spawna dentro de ella y no puede despegar.

## Waypoints hardcodeados (frame Gazebo ENU)

| WP | X | Y | Z | Descripcion |
|----|---|---|---|-------------|
| 0 | 0.0 | 0.0 | 1.5 | Despegue |
| 1 | 4.0 | 0.0 | 1.5 | Pre-Aro 1 |
| 2 | 6.0 | 0.0 | 1.5 | Post-Aro 1 |
| 3 | 9.0 | -1.5 | 2.5 | Pre-Aro 2 |
| 4 | 11.0 | -1.5 | 2.5 | Post-Aro 2 |
| 5 | 14.0 | 0.0 | 1.0 | Pre-Aro 3 |
| 6 | 16.0 | 0.0 | 1.0 | Post-Aro 3 |
| 7 | 20.0 | 0.0 | 1.5 | Hover sobre plataforma |
| 8 | 20.0 | 0.0 | 0.3 | Descenso pre-aterrizaje |

## Poses de los aros (frame Gazebo ENU)

| Modelo | X | Y | Z | Yaw |
|--------|---|---|---|-----|
| aro_1 | 5.0 | 0.0 | 1.5 | 0.0 |
| aro_2 | 10.0 | -1.5 | 2.5 | 0.0 |
| aro_3 | 15.0 | 0.0 | 1.0 | 0.0 |
| plataforma_llegada | 20.0 | 0.0 | 0.05 | 0.0 |
