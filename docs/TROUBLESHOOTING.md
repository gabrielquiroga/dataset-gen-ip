# Troubleshooting y Problemas Frecuentes (UAV Instant Policy)

Este documento sirve como registro de los problemas técnicos mas críticos encontrados durante la integración de PX4 SITL, Gazebo Harmonic y ROS 2, junto con sus soluciones definitivas.

---

## 1. Dron no logra armarse y reporta `High Gyro Bias` o `GPS Vert Speed Drift too high`

**Síntoma:** Al lanzar PX4, en la terminal comienzan a aparecer alertas continuas sobre el giroscopio y el GPS, seguidas por `Arming denied: Resolve system health failures first`. El nodo `expert_pilot` se queda infinitamente enviando comandos `ARM & OFFBOARD (reintento)`.

**Causa:** Colisión y explosión del motor de físicas al iniciar (`Spawn Penetration`). La plataforma de salida en Gazebo (`plataforma_salida`) tiene una altura hasta `Z=0.1`. Si la variable `PX4_GZ_MODEL_POSE` define una altura de spawn inferior a las patas del dron (por ejemplo, `Z=0.2`), el dron "nace" penetrando la estructura estática. El motor ODE intenta expulsarlo de inmediato aplicando fuerzas colosales, lo que hace que el dron salga volando hacia arriba girando sin control (invisble en la UI ya que es muy veloz). Esto satura los sensores simulados.

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

**Causa:** `gz sim` Zombie. Al usar `Ctrl+C` para matar un lanzamiento previo de PX4 (`make px4_sitl`), el servidor en segundo plano de Gazebo no se cierra. Cuando vuelves a iniciar PX4, el script detecta que Gazebo ya corre e inyecta un *nuevo* dron en ese mundo "huérfano" cuyos nodos de transporte (`gz-transport`) están pausados o corruptos. PX4 no recibe datos de los sensores.

**Solución manual:**
Antes de realizar lanzamientos manuales, usar el script de limpieza provisto para matar procesos zombies:
```bash
~/ros2_ws/src/uav_ip/scripts/kill_sim.sh
```
*(El script `data_collector.py` ya ejecuta estas rutinas de limpieza automáticamente en cada episodio).*

---

## 3. PX4 arranca perfectamente pero no responde al piloto experto

**Síntoma:** La terminal de PX4 no muestra errores (llega a `home set`) pero el dron jamás inicia vuelo, a pesar de que la terminal de `expert_pilot` indica "Comandando ARM".

**Causa:** Desincronización del Dominio ROS 2 (`ROS_DOMAIN_ID`). PX4 en sí mismo no es un nodo de ROS; depende del puente `MicroXRCEAgent`. Si el script `load_ros` no se ejecuta en la terminal antes de levantar el Agente, el Agente escuchará en el Dominio 0. Sin embargo, el `expert_pilot` publica sus comandos en el Dominio 33. Al no verse, PX4 no crea *suscriptores* (DataReaders) y las peticiones de armado caen al vacío.

**Solución:**
Asegurar la inyección explícita de las variables de red antes de ejecutar el puente:
```bash
export ROS_DOMAIN_ID=33
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
MicroXRCEAgent udp4 -p 8888
```
*(El `ProcessManager` de `data_collector.py` ha sido diseñado para inyectar estas variables por la fuerza en todos sus subprocesos).*

---

## 4. Simulador (Gazebo) y ROS están muy desfasados de tiempo en WSL2

**Síntoma:** Pérdida de paquetes, errores de QoS o TF timeouts.
**Causa:** Conflicto de interfaces de red de Windows hacia WSL2 que confunde a Gazebo Transport y FastDDS.
**Solución:**
Aislar la simulación al loopback:
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
1. A partir de la Fase 2, `data_collector.py` graba automáticamente `/fmu/out/timesync_status` (`px4_msgs/msg/TimesyncStatus`).
2. El validador `validate_episode.py` desambigua analíticamente: si el salto en odometría coincide en una ventana de $\pm 0.05\text{ s}$ con un salto equivalente en `TimesyncStatus.estimated_offset`, el reloj se repara automáticamente (`timesync_offset_adjustment_repaired`) y el episodio se acepta (`usable_for_validation = True`).
3. Si el episodio sigue rechazado con `ambiguous_clock_step_pattern_b`:
   - Verifica que `/fmu/out/timesync_status` esté presente en el rosbag (`ros2 bag info <bag_dir>`).
   - Los episodios grabados antes de Fase 1.1/Fase 2 carecen de este tópico y deben ser re-recolectados.
   - Si no se debe a un ajuste de offset de red sino a una congelación genuina de la simulación o caída de telemetría superior a $0.15\text{ s}$, el rechazo es legítimo y `data_collector.py` lanzará un reintento no destructivo (`episode_XX_attempt_01`).

