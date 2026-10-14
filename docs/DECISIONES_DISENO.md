# Registro de Decisiones de Diseño y Memoria Arquitectónica: UAV Instant Policy

*Documento de persistencia técnica y fundamentación metodológica · Octubre de 2026*

Este documento consolida y formaliza de manera permanente todas las decisiones arquitectónicas, matemáticas y de ingeniería tomadas durante la adaptación de **Instant Policy** (Vosylius y Johns, ICLR 2025) a la navegación autónoma de drones (PX4 + ROS 2 + Gazebo). Su propósito es servir como punto único de verdad para evitar reabrir dilemas ya resueltos o reintroducir hipótesis inviables.

---

## 1. El Dilema de la Cámara Frontal vs. LiDAR 360° y la Hipótesis de Markov

### 1.1 Contexto y Problema Original
Inicialmente se contempló usar el modelo `gz_x500_depth` (cuadricóptero x500 equipado con una cámara de profundidad frontal OAK-D simulada con campo de visión horizontal $\approx 80^\circ$).

Durante la ejecución de maniobras de vuelo 3D (como atravesar aros, hacer slalom o rodear postes):
1. **Ceguera retrospectiva ("Backward Blindness"):** En el instante exacto en que el dron cruza el centro de un aro o supera un poste, el objeto sale instantáneamente del frustum de la cámara hacia atrás.
2. **Violación de la Hipótesis de Markov:** Instant Policy modela la política de imitación como una distribución reactiva condicional $P(A_t \mid O_t, \mathcal{D}_{\text{context}})$, donde $O_t$ es un snapshot sensorial del instante presente. Si el landmark que define la referencia espacial desaparece del cono frontal, la observación $O_t$ queda vacía o desorientada hasta que el siguiente landmark entra en el campo de visión.
3. **Inestabilidad del Transformer / GNN:** En el Apéndice G de Instant Policy, los autores identificaron que la falta de cobertura sensorial o la contradicción espacial en demostraciones genera bloqueos, oscilaciones erráticas o detención prematura. Para una cámara frontal, resolver esto requeriría introducir memoria recurrente (RNN, Transformer con ventana causal de observaciones pasadas), agregando una complejidad desmedida en la Fase 1.

### 1.2 Decisión Adoptada
- Se reformuló conceptualmente el *embodiment* del dron: el vehículo se modela como un UAV equipado con un sensor **LiDAR 3D omnidireccional 360°** (análogo en el mundo real a un *Livox Mid-360* o sensor de domo esférico), con un radio de detección de $R_{\text{max}} = 10.0\text{ m}$.
- Gracias a la visión omnidireccional de 360°, el dron mantiene simultáneamente en su observación egocéntrica los landmarks situados al frente, a los costados y en la retaguardia (dentro de los 10 metros).
- **Preservación estricta de la Hipótesis de Markov:** El estado actual del mundo relevante para la acción futura está completamente capturado en el snapshot presente, eliminando la necesidad de recurrencia o memoria histórica en la red durante la Fase 1.

---

## 2. Justificación de la Observación Privilegiada vs. Captura de Sensores

### 2.1 El Costo Inviable de la Captura Masiva de Cámara
La primera versión del pipeline intentó grabar directamente el tópico de ROS 2 `/depth_camera/points` emitido por Gazebo mediante `ros_gz_bridge`. El análisis empírico reveló:
- Cada nube de profundidad de $640 \times 480$ contiene $307.200$ puntos 3D sin comprimir (`sensor_msgs/PointCloud2`).
- A una tasa moderada de 15 a 30 Hz, una única trayectoria de 40–50 segundos producía entre $600$ y $1.000$ mensajes, generando entre **5 y 6 GB por demostración**.
- Para un dataset de entrenamiento estándar (ej. $100\text{ tareas} \times 15\text{ demos} = 1.500\text{ vuelos}$), el espacio en disco hubiera requerido entre **7.5 y 9 Terabytes**.
- En términos computacionales, el renderizado de la cámara de profundidad en Gazebo Harmonic colapsaba el *Real-Time Factor* (RTF) a valores de $\times0.1$ a $\times0.2$, tornando prohibitiva la recolección masiva.

### 2.2 Decisión: Observación Privilegiada Sintética
- El dataset almacena exclusivamente **telemetría ligera** de PX4 (`vehicle_odometry`, `trajectory_setpoint`, `vehicle_local_position_v1`, `vehicle_status_v4`, `vehicle_command`, `vehicle_land_detected`) y los metadatos de poses de los landmarks (`waypoints.yaml`). Cada vuelo pesa **$< 5\text{ MB}$** (reducción $> 1.000\times$).
- La simulación SITL se ejecuta con el modelo estándar **`gz_x500`** (sin sensor de profundidad), disparando el RTF a $\times1.0$ (o superior en modo *fast-forward*).
- Las nubes de puntos de 2048 puntos se generan **sintéticamente bajo demanda** (al vuelo o en el script `bag_to_heterodata.py`) mediante el generador modular `synthetic_lidar.py`.
- **Segmentación perfecta garantizada:** La observación privilegiada elimina por construcción el suelo, paredes, el propio chasis del dron y distractores, entregando exactamente los objetos de interés tal como exige Instant Policy (Apéndice E).

---

## 3. Unificación de Marcos de Referencia (Gazebo Ground Truth vs. PX4 EKF2)

### 3.1 El Riesgo de Offsets Silenciosos
Un error clásico y destructivo en la simulación robótica con PX4 SITL es mezclar coordenadas de Gazebo con la odometría de PX4 sin conversión matemática estricta:
- Gazebo opera en el marco global **ENU** (East-North-Up), donde el origen $(0,0,0)$ es el centro del mundo simulado.
- PX4 y su estimador EKF2 operan en **NED** (North-East-Down), inicializando su origen local en el punto donde se inicializó el EKF2 en el arranque del microcontrolador.
- Además, el cuerpo del dron en PX4 utiliza convención **FRD** (Forward-Right-Down), mientras que la convención estándar en ROS y robótica espacial es **FLU** (Forward-Left-Up).
- Si las poses de los landmarks vienen de Gazebo y la pose del dron viene de PX4 sin compensar estas convenciones, la nube de puntos egocéntrica calculada como $P_{\text{dron}} = T_{w,d}^{-1} \cdot P_w$ acumula una rotación y un offset sistemáticos que impiden que el modelo correlacione la acción con la geometría observada.

### 3.2 Formalización de la Transformación
Se implementó en `synthetic_lidar.py` y `bag_to_heterodata.py` la transformación canónica exacta:
1. **Traslación:**
   $$\begin{bmatrix} x_{\text{ENU}} \\ y_{\text{ENU}} \\ z_{\text{ENU}} \end{bmatrix} = \begin{bmatrix} y_{\text{NED}} \\ x_{\text{NED}} \\ -z_{\text{NED}} \end{bmatrix}$$
2. **Rotación:**
   Sea $R_{\text{NED}}^{\text{FRD}}$ la matriz de rotación del cuerpo a la tierra en PX4 (obtenida del cuaternión de `vehicle_odometry`). La orientación del dron de cuerpo FLU a mundo ENU se calcula unívocamente mediante:
   $$R_{\text{ENU}}^{\text{FLU}} = M_{\text{world}} \cdot R_{\text{NED}}^{\text{FRD}} \cdot M_{\text{body}}$$
   donde:
   $$M_{\text{world}} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & -1 \end{bmatrix}, \quad M_{\text{body}} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \end{bmatrix}$$
3. Con estas matrices ortonormales, la proyección de cualquier punto del mundo $P_w$ al marco egocéntrico del dron FLU se expresa de forma vectorizada como:
   $$P_{\text{dron}} = (P_w - p_d) \cdot R_{\text{ENU}}^{\text{FLU}}$$

---

## 4. Ordenamiento Monotónico de Waypoints en Pasillos 1D

### 4.1 Problema de Permutación Arbitraria
En las primeras iteraciones procedimentales se barajó permutar libremente el orden de visita de los landmarks (ej. visitar aro 3, luego aro 1, luego aro 2) en un circuito lineal a lo largo del eje $+X$.
- Sin embargo, en un pasillo unidireccional, hacer que el dron regrese hacia atrás implica volar a través de un espacio ocupado por aros intermedios sin un planificador reactivo de evitación de colisiones.
- En Instant Policy, las demostraciones no modelan colisiones; el método asume demostraciones **factibles y libres de colisión por construcción** (Sec. 5, Apéndice G). Demos inconsistentes o con trayectorias caóticas inducen al transformer a oscilar y congelarse.

### 4.2 Decisión Adoptada
- Para circuitos en pasillos 1D, la lógica topológica válida es **`A1_DIRECT`**: avance monótonamente creciente en el eje $+X$ ($X_0 < X_1 < X_2 < X_3$).
- El generador de tareas ordena los waypoints asegurando que cada segmento conecte landmarks consecutivos sin requerir maniobras de retroceso a través de estructuras ya superadas.
- Variantes topológicas con cruces complejos (recorridos inversos o cerrados) quedan diferidas a mundos 2D/3D con áreas abiertas de retorno libre de colisión.

---

## 5. Transición Segura a Modo Offboard en PX4

### 5.1 Protocolo de Failsafe de PX4
El firmware PX4 cuenta con un mecanismo estricto de protección:
- El piloto automático **rechaza** la transición a modo `Offboard` y aborta el armado si el nodo externo no ha estado publicando comandos de setpoint a una tasa superior a 2 Hz durante un período previo continuo (típicamente $\ge 500\text{ ms}$).
- Si se envía un comando `VEHICLE_CMD_DO_SET_MODE` antes de iniciar la ráfaga de setpoints, PX4 emite `Failsafe mode activated` o `Arming denied: Offboard lost`.

### 5.2 Solución Implementada en `BasePilot`
Se diseñó una máquina de estados determinista en `base_pilot.py`:
1. **Estado `nav_state = 0` (Calentamiento):** El nodo se suscribe a `/fmu/out/vehicle_local_position_v1` y publica 50 setpoints continuos a 50 Hz fijados en la posición actual de spawn en reposo antes de solicitar cualquier cambio de modo.
2. **Estado `nav_state = 1` (Armado):** Una vez verificado el flujo estable de 50 mensajes, se comanda el modo Offboard (`VEHICLE_CMD_DO_SET_MODE`) y se solicita el armado forzado (`VEHICLE_CMD_COMPONENT_ARM_DISARM`).
3. Esto garantiza un 100% de éxito en la transición sin abortos de pre-vuelo ni cuelgues del planificador.

---

## 6. Factor de Escala Métrica ($s = 0.1$) para el Encoder Sinusoidal

### 6.1 Fundamento Matemático del Aliasing Sinusoidal
Instant Policy codifica las distancias relativas entre nodos de grafos (posiciones relativas entre puntos de la escena y efector) mediante un embedding posicional de Fourier con bandas de frecuencia periódicas:
$$f_k(\Delta p) = \sin(2^k \pi \Delta p)$$
- En esta formulación, el período base de la función sinusoidal es:
  $$T_{\text{base}} = \frac{2\pi}{\pi} = 2.0$$
- **Consecuencia en Escala Métrica Real:** Si dos waypoints o landmarks difieren en $2.0\text{ m}$, $4.0\text{ m}$, $6.0\text{ m}$ u $8.0\text{ m}$ en cualquier eje cartesiano, la red neuronal recibe **exactamente el mismo vector de características posicionales**, sufriendo un fenómeno catastrófico de *aliasing espacial*.
- La red original fue entrenada en entornos de manipulación sobremesa (ShapeNet), donde los objetos y las trayectorias operan en el rango $[-1.0, 1.0]\text{ m}$.

### 6.2 Decisión de Escala Adoptada
- Se aplica un factor de escala métrica constante **$s = 0.1$** a:
  1. Todas las coordenadas de las nubes de puntos sintéticas: $P_{\text{scaled}} = 0.1 \cdot P_{\text{dron}}$.
  2. Todas las posiciones del dron almacenadas en el grafo: $x_{\text{scaled}} = 0.1 \cdot x_{\text{enu}}$.
  3. Todas las acciones futuras / deltas de trayectoria: $y_{\text{scaled}} = 0.1 \cdot y_{\text{enu}}$.
- Con $s = 0.1$:
  - Un rango de alcance de LiDAR de $10.0\text{ m}$ se mapea exactamente al intervalo $[-1.0, 1.0]$.
  - Un aro de $1.5\text{ m}$ de envergadura se mapea a $0.15\text{ m}$ (análogo a una taza o mango de mesa en ShapeNet).
  - Se elimina por completo el aliasing sinusoidal, permitiendo reutilizar el encoder geométrico preentrenado sin degradación ni reentrenamiento destructivo.

---

## 7. Desacoplamiento de Entornos y Gestión de Dependencias (ROS 2 SITL vs. ML Pipeline)

### 7.1 Contexto y Conflicto de Dependencias
El sistema operativo base es **Ubuntu 24.04 LTS (Noble)** bajo WSL2, ejecutando **ROS 2 Jazzy Jalisco**.
1. **PEP 668 (Externally Managed Environment):** Ubuntu 24.04 bloquea de forma predeterminada cualquier comando `pip install` a nivel de sistema para prevenir la sobreescritura de paquetes `.deb` críticos del sistema operativo y de ROS 2.
2. **Incompatibilidad de Ecosistemas:** El stack de simulación (PX4 + Gazebo + ROS 2) depende de versiones específicas de Python del sistema (`/usr/bin/python3`), mientras que el pipeline de Machine Learning (PyTorch, PyG, CUDA, cuDNN, Torch-Scatter, Torch-Sparse) posee árboles de dependencias pesados y dinámicos que entran en conflicto con las librerías compartidas de ROS.
3. **Independencia de Rosbags:** La librería `rosbags` (de Ternaris) es una implementación escrita en Python puro para la lectura y deserialización directa de archivos `.mcap` y `.db3`. No depende de la API nativa de ROS 2 ni requiere hacer `source install/setup.bash`.

### 7.2 Decisión Adoptada: Desacoplamiento Estricto
Se adopta una separación bicameral estricta de entornos:
1. **Entorno Simulación y Vuelo (Host / ROS 2):**
   * Corre en el intérprete del sistema con ROS 2 Jazzy sourced (`source /opt/ros/jazzy/setup.bash`).
   * Responsabilidades: Ejecución de PX4 SITL, Gazebo Harmonic, nodos de control (`base_pilot.py`, `waypoint_controller.py`), generadores procedimentales (`task_generator.py`, `world_generator.py`) y orquestación (`data_collector.py`).
   * Registro: Graba los rosbags `.mcap` con telemetría ligera.
2. **Entorno Machine Learning y Entrenamiento (Virtualenv Aislado):**
   * Corre en un entorno virtual aislado (`~/venv_ip` o entorno Conda).
   * Dependencias: `torch`, `torch_geometric`, `rosbags`, `scipy`, `numpy`, `matplotlib`.
   * Responsabilidades: Procesamiento de rosbags a grafos HeteroData (`bag_to_heterodata.py`), extracción de keyframes, aumentos de datos, entrenamiento del modelo de difusión sobre grafos e inferencia offline.
   * Ventaja: Aislamiento total de dependencias sin riesgo de corromper la instalación de ROS 2 Jazzy.

---

## 8. Muestreo de Tareas sin Reemplazo y Partición Held-Out Estricta

### 8.1 El Problema de la Colisión Combinatoria
En Instant Policy, la red aprende a inferir la regla semántica de una tarea a partir de $N-1$ demostraciones de contexto para predecir las acciones de una demostración objetivo. Para evaluar la capacidad real de aprendizaje *in-context*, el protocolo exige evaluar en **tareas Held-Out** (secuencias de tareas nunca vistas durante el entrenamiento).

En la configuración de pasillo 1D con 3 landmarks:
* Cada landmark puede ser `gate` (1 acción: `traverse`) o `pole` (3 acciones: `overfly`, `slalom`, `orbit`).
* El espacio combinatorio total de firmas semánticas es exactamente de **64 tareas únicas**:
  $$\text{Firma Semántica} = [(\text{lm}_1, \text{prim}_1), (\text{lm}_2, \text{prim}_2), (\text{lm}_3, \text{prim}_3)]$$
* Si las tareas se generan mediante un muestreo aleatorio independiente con reemplazo (como hacía la primera versión de `task_generator.py`), la **Paradoja del Cumpleaños** provoca que:
  * Con 15 tareas generadas, la probabilidad de al menos un duplicado supere el $80\%$.
  * Con 20 tareas, la probabilidad de duplicados supere el $96\%$.
* Si existen tareas idénticas etiquetadas con distinto ID (`task_0001` y `task_0018`), se corre el riesgo severo de filtrar datos de entrenamiento en el conjunto de test/held-out (*data leakage*).

### 8.2 Decisión Adoptada: Muestreo sin Reemplazo y Partición Semántica
1. **Generación Exhaustiva o Muestreo sin Reemplazo:**
   * El generador de tareas debe muestrear del conjunto discreto de las 64 firmas válidas **sin repetición**, garantizando que cada `task_XXXX` represente una estructura semántica estrictamente única.
2. **Doble Nivel de Diversidad:**
   * **Diversidad Inter-Tarea:** Viene dada por la firma semántica única (`task_config.yaml`).
   * **Diversidad Intra-Tarea:** Viene dada por la variación continua de las poses 3D $(x, y, z, \text{yaw})$ de los objetos en cada episodio (`waypoints.yaml`).
3. **Auditoría Automatizada (`audit_dataset.py`):**
   * Se establece la necesidad de un script que verifique la ausencia de firmas duplicadas en el dataset y genere los archivos formales de partición: `train_tasks.json`, `val_tasks.json` y `held_out_tasks.json`.

---

## 9. Resumen de Parámetros Críticos

| Parámetro | Valor | Fundamentación |
| :--- | :--- | :--- |
| **Vehículo SITL** | `gz_x500` | Máxima aceleración RTF; prescinde de sensor de profundidad |
| **Observación** | Privilegiada (`range` 360°) | Sin ruido de suelo; reduce rosbag a $< 5\text{ MB}$ |
| **Alcance LiDAR ($R_{\text{max}}$)** | $10.0\text{ m}$ | Suficiente para anticipar y retener el landmark activo y previo |
| **Puntos por Nube ($N$)** | $2048$ | Estándar de Instant Policy; submuestreado con FPS |
| **Factor de Escala ($s$)** | $0.1$ | Previene aliasing en codificación posicional sinusoidal |
| **Marcos** | ENU (Mundo) / FLU (Cuerpo) | Unificación analítica estricta con odometría PX4 NED/FRD |
| **Lógica Pasillo 1D** | `A1_DIRECT` | Trayectorias monótonas libres de colisión por construcción |
| **Entorno de Simulación** | ROS 2 Jazzy (Sistema) | Control de vuelo, PX4 SITL, Gazebo Harmonic |
| **Entorno de ML** | Python venv (`~/venv_ip`) | PyTorch, PyG, rosbags puro, sin interferir con PEP 668 |

---

## 10. Registro Histórico de Enfoques Descartados y Evolución Técnica

*Esta sección documenta formalmente las arquitecturas, experimentos y decisiones implementadas en versiones anteriores que fueron evaluadas, contrastadas y descartadas, sirviendo como memoria técnica para el trabajo final / tesis.*

### 10.1 Descarte del Sensor Físico de Profundidad (`gz_x500_depth` + `PointCloud2`)
* **Qué se implementó (Septiembre 2026):**
  * Modelo de vehículo `gz_x500_depth` equipado con una cámara de profundidad OAK-D simulada en Gazebo Harmonic.
  * Puente `ros_gz_bridge` configurado para transformar `/depth_camera/points` (`gz.msgs.PointCloudPacked`) en tópicos de ROS 2 (`sensor_msgs/PointCloud2`).
  * Grabación directa de los mensajes de nube densa en rosbags `.mcap`.
* **Hallazgos Empíricos y Métricas:**
  * Cada nube contenía $307.200$ puntos 3D sin comprimir por fotograma.
  * A 20 Hz, un vuelo de 45 segundos generaba entre **5 y 6 GB por demostración**.
  * El renderizado GPU de profundidad en Gazebo colapsaba el *Real-Time Factor* (RTF) a $\times0.1 - \times0.2$, haciendo que un vuelo de 20 s tardara más de 3 minutos en simularse.
  * Para $1.000$ demostraciones, el dataset hubiera requerido $\approx 6\text{ TB}$ de almacenamiento.
* **Problemas Teóricos Irreparables:**
  * **Violación de Markov por Ceguera Retrospectiva:** La cámara frontal de $80^\circ$ pierde inmediatamente la referencia visual del aro en cuanto el dron lo cruza, vaciando la observación $O_t$ hasta avistar el siguiente objeto.
  * **Contaminación del Suelo:** La nube cruda incluía el suelo, las paredes y el propio chasis del dron. Segmentar analíticamente el suelo con planos RANSAC resultaba inestable ante inclinaciones de *pitch/roll* del dron.
* **Resolución Definitiva:** Descartado en su totalidad el 11 de octubre de 2026. Reemplazado por el modelo estándar `gz_x500` con **observación privilegiada LiDAR 360° sintética** vía `synthetic_lidar.py` (rosbags $< 5\text{ MB}$, RTF $\ge 1.0$, segmentación analítica perfecta, 2048 puntos FPS).

### 10.2 Descarte del Piloto Monolítico (`expert_pilot.py`)
* **Qué se implementó (Septiembre 2026):**
  * Un script único monolítico `expert_pilot.py` que mezclaba el ciclo de vida de ROS 2, la máquina de estados de armado, el parsing de tópicos, el cálculo trigonométrico de waypoints y el control PID.
* **Motivo del Descarte:**
  * Rigidez arquitectónica: No permitía desacoplar maniobras feed-forward (cruce directo de aros) de controladores de tracking en feedback continuo (órbita de postes).
* **Resolución Definitiva:** Refactorizado en el patrón *Strategy* mediante `base_pilot.py` (ciclo de vida, QoS y seguridad de armado), `waypoint_controller.py` (trayectorias feed-forward) y `tracking_controller.py` (seguimiento continuo).

### 10.3 Descarte de Permutaciones Topológicas No Monótonas en Pasillo 1D
* **Qué se evaluó (Principios de Octubre 2026):**
  * Permitir que el generador de tareas ordenara libremente los landmarks en un pasillo lineal $+X$ (ej. visitar aro 1, luego aro 3, y regresar al aro 2).
* **Motivo del Descarte:**
  * Instant Policy asume demostraciones factibles y libres de colisión por construcción (no aprende a evitar colisiones de forma reactiva, sino a imitar desvíos demostrados).
  * En un pasillo lineal estrecho, retroceder hacia un aro previo implica volar a través de obstáculos existentes sin un planificador local de evitación, provocando colisiones sistemáticas.
* **Resolución Definitiva:** Restringir la lógica del pasillo 1D a `A1_DIRECT` (avance monótono). Maniobras con retornos libres se difieren a circuitos cerrados 2D/3D con áreas abiertas.

### 10.4 Descarte de la Generación de Mundo Único por Tarea
* **Qué se implementó inicialmente:**
  * `data_collector.py` ejecutaba `_generate_world` una única vez antes del bucle de episodios de cada tarea.
* **Fallo Identificado:**
  * Todos los episodios de una misma tarea compartían exactamente las mismas coordenadas físicas $(x, y, z, \text{yaw})$.
  * Esto violaba el principio central de In-Context Imitation Learning, donde las demos de contexto y la demo objetivo deben compartir la misma regla semántica pero exhibir variabilidad espacial continua.
* **Resolución Definitiva:** Separación estricta entre `_generate_task_config` (semántica fija en `task_config.yaml`) y `_generate_episode_world` (aleatorización espacial en cada episodio con semilla determinista única).
