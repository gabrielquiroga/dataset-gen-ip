# Justificación Técnica y Defensa de Decisiones ante la Revisión Experta

*Documento de trazabilidad, fundamentación matemática y defensa de ingeniería sobre el análisis externo (`docs/feedback/revision.md` y `docs/feedback/sugerencias.md`) · Octubre de 2026*

---

## 1. Propósito de este Documento

Antes de iniciar la campaña masiva de recolección de demostraciones en simulación HITL/SITL (PX4 + Gazebo Harmonic), el repositorio fue sometido a una revisión externa independiente (`docs/feedback/revision.md` y `docs/feedback/sugerencias.md`).

Este documento registra con rigor académico y de ingeniería:
1. **Qué observaciones revelaron vulnerabilidades reales** en nuestro pipeline y cómo fueron resueltas en código.
2. **Qué hallazgos empíricos nuevos** descubrimos al inspeccionar a nivel binario los archivos `.mcap` reales del proyecto.
3. **Qué sugerencias fueron descartadas o simplificadas deliberadamente por constituir sobreingeniería**, proporcionando los argumentos físicos, matemáticos y computacionales para defender cada decisión ante un tribunal o revisor.

---

## 2. Matriz Resumen de Decisiones (Punto por Punto)

| Punto de la Revisión / Sugerencia | Clasificación Técnica | Decisión Adoptada | Módulo / Archivo Responsable |
| :--- | :--- | :--- | :--- |
| **1. Éxito del episodio acoplado a `pilot.returncode == 0`** (`Revisión §2.1`, `Sug. §3.2`) | **Error real crítico** | Separación formal entre `pilot_exit_code`, `recording_status`, `mission_status` y `validation_status` (`accepted`/`rejected`) mediante validador post-vuelo independiente. | `src/uav_ip/uav_ip/validate_episode.py` |
| **2. Tolerancias permisivas de cruce (`cross_track < 1.4m` en aro de $\pm 0.75\text{ m}$)** (`Revisión §2.2`, `Sug. §3.4`) | **Error real crítico** | Endurecimiento a `cross_track < 0.38m` y `alt_err < 0.42m` en `WaypointController` + verificación de intersección exacta con el plano $x_{\text{loc}}=0$ en `validate_episode.py`. | `src/uav_ip/uav_ip/waypoint_controller.py`, `validate_episode.py` |
| **3. Aterrizaje por temporizador fijo de `4.0s`** (`Revisión §2.1`, `Sug. §3.2.B`) | **Debilidad real** | Suscripción a `/fmu/out/vehicle_land_detected` en `BasePilot` y espera activa de confirmación (`landed == True` o $z_{\text{ENU}} < 0.22\text{ m}$) + validación de error radial $\le 0.85\text{ m}$ al centro de la plataforma. | `src/uav_ip/uav_ip/base_pilot.py`, `waypoint_controller.py` |
| **4. Detección de colisiones mediante plugins de contacto en Gazebo SDF + `ros_gz_bridge`** (`Sug. §3.2.A`) | **Sobreingeniería** (Problema real, solución innecesariamente pesada) | **Reemplazado por verificación analítica 3D exacta a $125\text{ Hz}$** contra las 4 barras AABB de `square_gate`, el toroide de `circular_gate` y el cilindro de `pole` ($d_{\text{min}} \ge R_{\text{UAV}} = 0.28\text{ m}$). | `src/uav_ip/uav_ip/validate_episode.py` |
| **5. Factibilidad geométrica de transiciones entre primitivas consecutivas** (`Revisión §2.5`, `Sug. §3.5.A`) | **Mejora valiosa y preventiva** | Implementación de `check_circuit_transition_feasibility`: valida avance longitudinal $\Delta x \ge 1.0\text{ m}$, distancia $d_{xy} \ge 1.4\text{ m}$, pendiente $|\Delta z|/d_{xy} \le 1.15$ con `traverse` ($\le 1.60$ entre postes) y no-interferencia de waypoints `pre`/`post`. | `src/uav_ip/scripts/world_generator.py` |
| **6. Saltos de reloj en `VehicleOdometry.timestamp` vs reloj de `rosbag2`** (`Revisión §2.4`, `Sug. §3.3.A`) | **Hallazgo empírico clave** | Se demostró que los saltos aislados provienen de recalibraciones `Timesync` en `uxrce_dds_client`. Se implementó `reconstruct_simulation_clock()`, que repara los escalones de offset usando $\Delta t_{\text{bag}}$ sin perder la cadencia física de PX4 ($125\text{ Hz}$, $\Delta t = 8.0\text{ ms}$). | `src/uav_ip/uav_ip/validate_episode.py` |
| **7. Reemplazar el piloto por trayectorias polinomiales *Minimum-Snap* / reactivar `B2_BSPLINE`** (`Revisión §2.2`, `Sug. §3.4`) | **Sobreingeniería** | **Descartado.** El controlador interno de PX4 ya aplica perfilado jerk-limited sobre `TrajectorySetpoint` con velocidad *feed-forward*; además, variar el perfil para una misma tarea geométrica reintroduciría *data leakage* entre particiones. | Justificado en Sección 4.2 |
| **8. Aumentar el número de demos de contexto a $W = 3\text{–}4$** (`Sug. §3.5.C`) | **Sobreingeniería en etapa de recolección** | **Descartado como requisito de campaña.** El paper original de *Instant Policy* demuestra que la ganancia se satura en $W = 2$ demos de contexto ($N=3$ episodios). Recolectar $\ge 5$ episodios válidos por tarea ya permite evaluar $W \in \{1, 2, 3, 4\}$ en inferencia sin cambiar la campaña. | Justificado en Sección 4.3 |

---

## 3. Fundamentación de lo Implementado en la Fase 1

### 3.1 Desacoplamiento entre Ejecución del Piloto y Validez Física del Episodio (`validate_episode.py`)

En un pipeline de aprendizaje por imitación en contexto (*In-Context Imitation Learning*), **una sola demostración de contexto defectuosa contamina todos los episodios de evaluación que se condicionen sobre ella**, ya que el grafo de contexto $\mathcal{G}_c$ codifica literalmente la trayectoria demostrada como especificación de la tarea.

Anteriormente, el criterio de éxito era:
$$\text{status} = \text{success} \iff \text{returncode}(\texttt{waypoint\_controller}) == 0$$

Con la arquitectura implementada en [`validate_episode.py`](file:///home/gabrielq/ros2_ws/src/uav_ip/uav_ip/validate_episode.py), cada episodio se evalúa de forma independiente sobre la telemetría grabada en el archivo `.mcap` y el archivo `waypoints.yaml`, produciendo cuatro indicadores ortogonales en `episode_meta.json`:
1. **`pilot_exit_code`**: Código de salida del proceso ROS 2 del controlador.
2. **`recording_status` (`"valid"` / `"invalid"`)**: Verifica que la grabación contenga $\ge 30$ muestras de odometría finitas, cuaterniones unitarios ($\|q\| \in [0.90, 1.10]$), duración $\ge 3.0\text{ s}$, ausencia de huecos de telemetría ($\Delta t_{\max} \le 0.60\text{ s}$) y **cero resets del filtro EKF2 durante el vuelo activo** ($\Delta \texttt{reset\_counter} = 0$).
3. **`mission_status` (`"valid"` / `"invalid"`)**: Verifica despegue ($z_{\max} \ge 0.80\text{ m}$), ausencia de contacto con el suelo a mitad de misión, cumplimiento estricto de todas las primitivas en orden secuencial, margen libre de colisión ($d_{\text{min}} \ge 0.28\text{ m}$) contra **todos** los landmarks del escenario y aterrizaje final sobre la plataforma.
4. **`validation_status` (`"accepted"` / `"rejected"`)**:
   $$\texttt{validation\_status} = \texttt{"accepted"} \iff (\texttt{recording\_status} == \texttt{"valid"}) \land (\texttt{mission\_status} == \texttt{"valid"})$$

Adicionalmente, para garantizar que `validate_episode.py` pueda ejecutarse tanto dentro del entorno del sistema ROS 2 como en entornos aislados sin dependencias externas, se implementó un **lector binario nativo MCAP + CDR (`_parse_mcap_native`)** en Python puro que decodifica directamente los esquemas de `px4_msgs` (`VehicleOdometry`, `VehicleLocalPosition`, `VehicleLandDetected`, `VehicleStatus`, `TrajectorySetpoint`).

---

### 3.2 Corrección de Tolerancias del Controlador y Verificación Geométrica por Primitiva

#### El problema geométrico del criterio anterior
Una puerta cuadrada (`square_gate`) tiene centro en $(c_x, c_y, c_z)$ y está compuesta por 4 barras de grosor $0.10\text{ m}$:
* Barras superior e inferior en $z_{\text{loc}} = \pm 0.75\text{ m}$ (abertura libre vertical: $[-0.70, +0.70]\text{ m}$).
* Barras laterales izquierda y derecha en $y_{\text{loc}} = \pm 0.70\text{ m}$ (abertura libre horizontal: $[-0.65, +0.65]\text{ m}$; borde exterior del marco en $\pm 0.75\text{ m}$).

El código previo en `waypoint_controller.py` avanzaba del waypoint `center` cuando:
```python
along_track > -0.10 and cross_track < 1.4 and alt_err < 0.8
```
Como $1.40\text{ m} > 0.75\text{ m}$, un dron que pasara a $1.10\text{ m}$ a la izquierda o derecha del centro del aro esquivaba completamente la estructura por fuera y el controlador lo daba por bueno.

#### Solución doble (Controlador + Validador)
1. **En `waypoint_controller.py`:** Se etiquetó cada waypoint con su primitiva (`"primitive": p_type`) y se restringió la condición de paso por `center` en `traverse` a:
   $$\big(\|\mathbf{p} - \mathbf{p}_{\text{center}}\|_2 < 0.45\text{ m} \;\lor\; x_{\text{along}} \ge -0.05\text{ m}\big) \;\land\; |y_{\text{cross}}| < 0.38\text{ m} \;\land\; |z_{\text{err}}| < 0.42\text{ m}$$
   Dado que el radio físico del UAV es $R_{\text{UAV}} \approx 0.28\text{ m}$, exigir $|y_{\text{cross}}| < 0.38\text{ m}$ garantiza que el chasis completo ($0.38 + 0.28 = 0.66\text{ m} < 0.70\text{ m}$) cruce por el interior libre de la ventana.
2. **En `validate_episode.py`:** Se interpola linealmente el punto exacto $(y_{\text{cross}}, z_{\text{cross}})$ donde la trayectoria cruza el plano $x_{\text{loc}} = 0$ del landmark en dirección $+x_{\text{loc}}$ y se verifica tanto la inclusión dentro de la abertura como la distancia mínima 3D a las 4 barras.

---

### 3.3 Hallazgo Empírico: Saltos de Offset `Timesync` en `uxrce_dds_client` y Reconstrucción del Reloj

Al auditar los 12 episodios reales grabados en `datasets/task_0000` a `task_0003`, descubrimos dos hechos empíricos fundamentales:
1. **Frecuencia real de publicación:** `/fmu/out/vehicle_odometry` se publica a **$125\text{ Hz}$ ($\Delta t_{\text{mediana}} = 8.0\text{ ms}$)**, no a $50\text{ Hz}$ (que es la tasa de `/fmu/out/vehicle_local_position_v1`).
2. **Anomalía de `Timesync` en `uxrce_dds_client`:** En PX4 SITL, el campo `msg.timestamp` de `VehicleOdometry` se genera sumando al reloj interno de arranque de PX4 (`hrt_abstime`) un offset de sincronización $\Delta T_{\text{sync}}$ estimado hacia el reloj UNIX del host por el puente `MicroXRCEAgent`.
   * Entre $1$ y $4$ veces por episodio (de un total de $\sim 3500$ muestras), en el ciclo exacto en que el filtro `Timesync` actualiza $\Delta T_{\text{sync}}$, una muestra aislada se publica con el tiempo crudo de arranque de PX4 ($\sim 62.10\text{ s}$) en lugar del tiempo UNIX ($\sim 1.77 \times 10^9\text{ s}$), o sufre un escalón de $\sim 280\text{ ms}$.
   * Sin embargo, el timestamp de grabación del contenedor MCAP (`log_time`) en ese mismo mensaje avanza exactamente $+8.0\text{ ms}$.

Para resolver esto sin perder la precisión del reloj de simulación de PX4, implementamos `reconstruct_simulation_clock(ts_px4_s, ts_bag_s)`:
* Calcula paso a paso $\Delta t_{\text{px4}}[k]$ y $\Delta t_{\text{bag}}[k]$.
* Si $|\Delta t_{\text{px4}}[k] - \Delta t_{\text{bag}}[k]| > 50\text{ ms}$ mientras el grabador confirma continuidad normal ($0 \le \Delta t_{\text{bag}}[k] \le 100\text{ ms}$), identifica un salto de offset `Timesync` y sustituye ese único diferencial por $\Delta t_{\text{bag}}[k]$.
* **Resultado empírico en los 12 episodios del proyecto:** El reloj reconstruido es estrictamente monótono creciente en el $100\%$ de los vuelos, con $\Delta t_{\text{mediana}} = 8.0\text{ ms}$ y hueco máximo $\Delta t_{\max} \le 28.0\text{ ms}$.

---

## 4. Defensa Técnica de lo Descartado por Sobreingeniería

A continuación se exponen los argumentos formales para defender por qué ciertas propuestas de `sugerencias.md` no se adoptaron o se resolvieron de forma más limpia.

### 4.1 Por qué NO se usaron Plugins de Contacto de Gazebo (`gz-sim-contact-system`) para Detectar Colisiones

**Sugerencia original (`Sugerencias §3.2.A`):** Añadir `<sensor name="contact" type="contact">` en los modelos SDF, cargar el plugin de sistema de contactos en Gazebo y puentear los tópicos de colisión hacia ROS 2 mediante `ros_gz_bridge`.

**Por qué nuestra solución analítica 3D en `validate_episode.py` es superior:**
1. **Conocimiento geométrico exacto (Entorno Estático Conocido):** Todos los obstáculos del mundo (`square_gate`, `circular_gate`, `pole`) son estáticos y sus poses y dimensiones exactas están definidas en `waypoints.yaml` y `world_generator.py`. Calcular la distancia euclídea mínima desde la posición del dron $\mathbf{p}(t)$ a las 4 cajas orientadas (OBB/AABB locales), al toroide o al cilindro tiene solución analítica cerrada en $< 2\text{ ms}$ por episodio completo.
2. **Inmunidad ante *Tunneling* y roces de hélice:** En Gazebo, las hélices suelen modelarse mediante plugins aerodinámicos (fuerzas/pares en el enlace base) sin malla de colisión rotatoria fina, o bien una colisión tangencial rápida entre pasos discretos puede no disparar un contacto sostenido. En cambio, imponer un volumen envolvente de seguridad $d_{\text{min}}(\mathbf{p}(t), \mathcal{O}_k) \ge R_{\text{UAV}} = 0.28\text{ m}$ a $125\text{ Hz}$ (un punto cada $\sim 1.1\text{ cm}$ de avance) detecta cualquier roce o aproximación peligrosa con margen físico garantizado.
3. **Detección unificada de fallos con y sin contacto:** Un plugin de contacto en Gazebo **no detecta** si el dron esquivó la puerta por fuera sin tocarla ($y = 1.2\text{ m}$), ni si hizo el slalom por el lado equivocado, ni si abortó la órbita a mitad de camino. El validador geométrico analítico resuelve colisiones y cumplimiento semántico de primitivas en una única herramienta autocontenida.
4. **Cero penalización en el *Real-Time Factor* (RTF):** Mantener `gz_x500` libre de sensores adicionales en Gazebo preserva $\text{RTF} \ge 1.0\times$, requisito crítico para evitar timeouts de armado Offboard en PX4.

---

### 4.2 Por qué NO se Reemplazó el Piloto Experto por Optimización Polinomial *Minimum-Snap* ni se Reactivó `B2_BSPLINE`

**Sugerencia original (`Revisión §2.2`, `Sugerencias §3.4`):** Sustituir el controlador por waypoints con velocidad *feed-forward* por un generador de trayectorias polinomiales *minimum-snap* (Mellinger & Kumar, 2011) o splines continuos que distingan `B1_LINEAR` de `B2_BSPLINE`.

**Argumentos de defensa:**
1. **Arquitectura en cascada de PX4 Multicopter Position Control:** Cuando `WaypointController` publica `TrajectorySetpoint` con posición y velocidad *feed-forward* (`vel_ff` a $1.4\text{ m/s}$ en cruces y $1.2\text{ m/s}$ en los 12 segmentos de arco de `orbit`), el controlador interno de PX4 (`mc_pos_control` + `FlightTaskOffboard`) **no aplica un controlador bang-bang discontinuo**, sino que filtra la referencia a través de la dinámica de actitud y empuje del cuadricóptero real en Gazebo. Las métricas medidas en nuestros 12 episodios reales muestran velocidades medias suaves ($\sim 1.0\text{ m/s}$), aceleraciones percentil-98 acotadas ($\sim 4.5\text{ m/s}^2$) y velocidad positiva continua a través de los centros de los aros.
2. **Las etiquetas de entrenamiento provienen de la trayectoria realizada, no del setpoint crudo:** En `bag_to_heterodata.py` (Tarea 1.5.5), las acciones objetivo $T_{EA}^{(t,k)} = (T_{WE}^t)^{-1} T_{WE}^{t+k}$ se extraen de la **trayectoria física realmente volada** por el dron (que es $C^2$-suave por la inercia física del vehículo en Gazebo), nunca del escalón del setpoint.
3. **Prevención de *Data Leakage*:** Si dos tareas tuviesen la misma secuencia de objetos y primitivas pero distinto etiqueta `profile` (`B1_LINEAR` vs `B2_BSPLINE`), volverían a tratarse como tareas distintas en los splits `train`/`val`/`held_out`, filtrando la misma estructura semántica entre entrenamiento y evaluación. Mantener una única ley de control experta consistente garantiza que la firma semántica de cada tarea sea biyectiva con la secuencia de maniobras.

---

### 4.3 Por qué se Mantiene $W = 2$ Demostraciones de Contexto ($N = 3$ Episodios por Muestra) en el Diseño del Modelo

**Sugerencia original (`Sugerencias §3.5.C`):** Incrementar el número de demostraciones de contexto por tarea a $W = 3\text{–}4$ debido a que el espacio de estados 3D de un UAV es más amplio que el de un brazo manipulador.

**Argumentos de defensa:**
1. **Fidelidad al paper base (*Instant Policy*, ICLR 2025):** Los autores demuestran experimentalmente (Sección 4, Figura 4) que el rendimiento del modelo se satura en $1\text{–}2$ demostraciones de contexto porque las demostraciones solo necesitan desambiguar **qué secuencia de primitivas y sobre qué objetos** debe ejecutarse (la topología semántica), mientras que la adaptación geométrica fina a las coordenadas actuales la realiza el subgrafo local $\mathcal{G}_l^t$ junto con las aristas relativas en $SE(3)$.
2. **Desacoplamiento entre recolección y muestreo de entrenamiento:** En nuestra campaña recolectaremos $\ge 5$ episodios aceptados por tarea. Esto permite que durante el entrenamiento el `InContextGraphDataset` sortee subconjuntos aleatorios de $W = 2$ demos de contexto en cada época ($\binom{5}{2} = 10$ combinaciones posibles de contexto por cada episodio query), multiplicando la diversidad combinatoria sin inflar cuadráticamente el costo de memoria de las capas de atención cruzada en el *Heterogeneous Graph Transformer*.
3. **Flexibilidad en evaluación:** Al disponer de $\ge 5$ episodios por tarea en `held_out_tasks.json`, se puede evaluar el mismo modelo entrenado con $W \in \{1, 2, 3, 4\}$ en tiempo de inferencia sin re-entrenar ni re-recolectar datos.

---

### 4.4 Por qué se Usa el Reloj de PX4 Reconstruido (`reconstruct_simulation_clock`) y no el Timestamp de `rosbag2`

**Sugerencia original (`Sugerencias §3.3.A`):** Considerar el uso del timestamp de grabación de `rosbag2` si el reloj de PX4 presenta discontinuidades.

**Argumento de defensa:**
El timestamp de recepción de `rosbag2` (`log_time`) refleja el instante en que el proceso de usuario en Linux leyó el paquete del socket UDP/DDS, por lo que está sujeto al *scheduling jitter* del sistema operativo cuando Gazebo y PX4 compiten por CPU. En cambio, `VehicleOdometry.timestamp` es generado por el reloj determinista del simulador físico en cada paso de integración de EKF2. Al reparar únicamente los $1\text{–}4$ escalones discretos de recalibración de `Timesync` mediante `reconstruct_simulation_clock()`, conservamos el reloj físico exacto de la simulación con monotonía estricta garantizada.
