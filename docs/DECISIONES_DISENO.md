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
| **Puntos por Nube ($N$)** | $2048$ | Estándar de Instant Policy; submuestreado con FPS (kernel SIMD C en $\sim 6\text{ ms}$) |
| **Factor de Escala ($s$)** | $0.1$ | Previene aliasing en codificación posicional sinusoidal |
| **Marcos** | ENU (Mundo) / Horizontal FLU (Egocéntrico) | Estabilizado por gravedad ($\text{roll}=0, \text{pitch}=0$), unificado con PX4 NED/FRD |
| **Lógica Pasillo 1D** | `A1_DIRECT` | Trayectorias monótonas libres de colisión por construcción |
| **Horizonte de Acción ($T$)** | $8$ pasos ($\approx 0.8\text{ s}$ a $10\text{ Hz}$) | Desplazamientos relativos $T_{EA}^{(t,k)} \in SE(3)$ de la trayectoria real |
| **Keyframes de Contexto ($L$)** | $10$ pasos | Submuestreo uniforme de trayectorias de contexto para el prompt geométrico |
| **Tube Augmentation** | $30\%$ de pasos con $\sigma_p = 0.25\text{ m}, \sigma_\psi = 5^\circ$ | Mitigación del *covariate shift* con flujo correctivo monótono $SE(3)$ en `data['aug']` |
| **Perturbación de Estado Binario** | $10\%$ de probabilidad ($\tilde{s}_t = 1 - s_t$) | Robustez en transiciones despegue/aterrizaje; evita sobreajuste a estados discretos |
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

### 10.5 Descarte de la Plantilla Rígida de 3 Aros y Transición a Inyección Dinámica Modular
* **Qué se implementó inicialmente:**
  * El archivo de simulación `circuito_aros_01.sdf` contenía tres modelos fijos: `<model name="aro_1">`, `aro_2` y `aro_3`.
  * `world_generator.py` buscaba obligatoriamente estos tres nombres para renombrarlos o modificar sus mallas.
* **Limitaciones Identificadas:**
  * Imposibilidad de variar la complejidad de las tareas: Tareas de 2 objetos dejaban un "aro fantasma" abandonado en el mundo; tareas de 4 o más objetos provocaban una excepción `ValueError` al no existir `aro_4` en el XML.
* **Resolución Definitiva (Octubre 2026):**
  * Desacoplamiento total entre el entorno base de Gazebo (física, iluminación, plano de tierra, plataforma de salida y plataforma de llegada) y los landmarks.
  * Implementación de constructores modulares de XML en Python (`_create_gate_element` y `_create_pole_element`) que inyectan dinámicamente $N$ elementos en tiempo de ejecución.
  * Reposicionamiento automático de la plataforma de aterrizaje a $X_N + \Delta X_{\text{land}}$, garantizando que cualquier circuito de longitud acotada $N \in [2, 5]$ sea físicamente factible.
  * Catálogo combinatorio dinámico balanceado e intercalado en `task_generator.py`, garantizando muestreo sin reemplazo estricto (cero duplicados) a través de todas las longitudes.

---

## 11. Marco Egocéntrico Estabilizado por Gravedad (*Horizontal FLU*)

### 11.1 Contexto y Limitación del Marco Rígido de Cuerpo (FLU)
Un cuadricóptero es un sistema subactuado: para acelerar horizontalmente (hacia adelante o a los lados), debe inclinarse físicamente (*pitch* y *roll*).
* Si las observaciones LiDAR y las acciones se expresan rígidamente en el marco del cuerpo del dron (Body FLU), cualquier aceleración o frenado inclina el sistema de coordenadas respecto al plano horizontal.
* Esto provoca que los puntos del suelo o landmarks verticales aparezcan inclinados durante maniobras rápidas, acoplando de forma espuria la dinámica de actitud transitoria con la geometría del entorno.
* Más crítico aún: en robótica de manipulación (dominio nativo de Instant Policy), el efector final no depende de inclinarse contra la gravedad para desplazarse horizontalmente.

### 11.2 Decisión Adoptada: Marco Horizontal FLU
Se define el marco egocéntrico **Horizontal FLU** ($H$):
1. **Origen:** Coincidente con la posición 3D del dron $(x, y, z)$ en ENU.
2. **Orientación:** Comparte el ángulo de guiñada (*yaw*, $\psi$) del dron, pero fuerza $\text{roll} = 0$ y $\text{pitch} = 0$:
   $$R_{\text{ENU}}^H = R_z(\psi)$$
3. **Alineación con la Gravedad:** El eje $+Z_H$ coincide exactamente con la vertical opuesta al vector gravitatorio terrestre (Up).
4. **Beneficios Teóricos y Prácticos:**
   * La geometría observada de los landmarks verticales (postes, aros) permanece invariante ante inclinaciones de aceleración del cuadricóptero.
   * Elimina la confusión del modelo neuronal entre actitud transitoria y geometría espacial del mundo.
   * Facilita el desacoplamiento entre el planificador de trayectorias (que emite referencias en el plano horizontal y vertical puro) y el controlador de actitud de bajo nivel de PX4.

---

## 12. Arquitectura de Poses Relativas $SE(3)$, Horizonte $T=8$ y $L=10$ Keyframes

### 12.1 Limitaciones del Esquema Original (Setpoints Escalonados y Coordenadas Globales)
La versión previa de `bag_to_heterodata.py` extraía como acciones futuras los setpoints brutos de PX4 (`trajectory_setpoint`), que presentaban dos patologías:
1. Eran escalones constantes que permanecían idénticos durante decenas de pasos hasta alcanzar la esfera de aceptación del waypoint, perdiendo la suavidad continua de la dinámica real.
2. Se almacenaban como posiciones absolutas en el marco del mundo, forzando a la red a memorizar localizaciones globales en lugar de aprender transformaciones geométricas relativas invariantes a traslación/rotación.

### 12.2 Decisión Adoptada: Horizonte Relativo de Trayectoria Real en $SE(3)$
1. **Poses $SE(3)$ y Keypoints:** Cada estado del agente se representa como una pose rígida tridimensional $T_{WE} \in SE(3)$ conformada por posición $p \in \mathbb{R}^3$ (escalada por $s = 0.1$) y orientación en cuaternión unitario $q = (q_w, q_x, q_y, q_z)$, complementada con 6 keypoints geométricos del cuadricóptero (centro, proa, popa, babor, estribor y mástil superior).
2. **Horizonte de Acción ($T=8$ pasos):** Para cada instante temporal $t$, se muestrea la trayectoria de vuelo real a una tasa de $\sim 10\text{ Hz}$ en los pasos futuros $t+k$ ($k=1, \dots, 8$, cubriendo $\approx 0.8\text{ s}$ de dinámica futura):
   $$T_{EA}^{(t,k)} = (T_{WE}^t)^{-1} \cdot T_{WE}^{t+k} \in SE(3)$$
   donde $\Delta p = 0.1 \cdot (R_{WE}^t)^T (p^{t+k} - p^t)$ y $\Delta q = (q^t)^{-1} \otimes q^{t+k}$.
3. **Keyframes de Contexto ($L=10$ pasos):** Las $N-1$ demostraciones de contexto que condicionan a la red se subsamplean uniformemente a $L=10$ hitos temporales representativos a lo largo de toda la duración del episodio, preservando la coherencia global de la tarea con un tamaño de grafo acotado.

---

## 13. Aumento de Datos para Recuperación (*30% Tube Augmentation*)

### 13.1 El Problema del *Covariate Shift* en Vuelo Autónomo
En aprendizaje por imitación estándar (*Behavioral Cloning*), el modelo aprende únicamente sobre trayectorias nominales óptimas ejecutadas por el piloto experto. Si durante la ejecución autónoma el vehículo sufre una pequeña perturbación aerodinámica o error de inferencia:
* El dron entra en un estado espacial fuera de la distribución de entrenamiento ($s \notin \mathcal{D}_{\text{train}}$).
* Al carecer de datos de corrección, los errores se acumulan de forma exponencial (*compounding errors*), conduciendo a colisiones o divergencia irreversible.

### 13.2 Decisión Adoptada: Aumento Sintético de Tubo (*Tube Augmentation*)
Aprovechando la naturaleza analítica de nuestro generador LiDAR privilegiado y la geometría de las trayectorias de vuelo, se implementó en `bag_to_heterodata.py`:
1. **Muestreo de Perturbaciones (30% de los timesteps):**
   * Perturbación de traslación: $\delta p \sim \mathcal{N}(0, \sigma_p^2 I_3)$ con $\sigma_p = 0.25\text{ m}$ (acotada a $\|\delta p\| \le 0.60\text{ m}$).
   * Perturbación de orientación (yaw): $\delta \psi \sim \mathcal{N}(0, \sigma_\psi^2)$ con $\sigma_\psi = 5^\circ$ (acotada a $|\delta \psi| \le 15^\circ$).
   * Pose perturbada: $\tilde{T}_{WE}^t = T_{WE}^t \cdot \Delta T_{\text{pert}}$.
2. **Observación y Acción de Recuperación:**
   * Se evalúa la nube LiDAR analítica $\tilde{P}_{\text{scene}}^t$ desde la pose perturbada $\tilde{T}_{WE}^t$ en el marco Horizontal FLU.
   * Se calculan las acciones relativas dirigidas hacia los hitos nominales futuros:
     $$\tilde{T}_{EA}^{(t,k)} = (\tilde{T}_{WE}^t)^{-1} \cdot T_{WE}^{t+k} \in SE(3)$$
   * Esta transformación garantiza que la acción predicha guíe convergente y suavemente al dron de regreso a la trayectoria nominal de referencia.
3. **Perturbación del Estado Binario (10% flip):**
   * En un 10% de las muestras aumentadas se invierte el estado de vuelo ($\tilde{s}_t = 1 - s_t$), forzando a la red a discernir la fase de vuelo a partir de la proximidad geométrica al suelo y landmarks, evitando atajos espurios basados en el bit discreto.
4. **Almacenamiento Separado en `data['aug']`:**
   * Los datos aumentados se almacenan bajo el tipo de nodo `aug` en el `HeteroData`, garantizando que las demostraciones de contexto nominales en `data['agent']` permanezcan inalteradas.

---

## 14. Aceleración Computacional: Kernel SIMD C para FPS y Procesamiento Batch

### 14.1 Cuello de Botella de Farthest Point Sampling (FPS)
El submuestreo de nubes LiDAR de $M \approx 4096$ puntos a $N = 2048$ puntos mediante FPS en NumPy puro requería $\approx 43\text{ ms}$ por fotograma. Para un vuelo de 100 pasos, esto sumaba $> 4.3\text{ s}$ solo en FPS, haciendo que procesar un dataset de 1000 episodios tomara más de una hora.

### 14.2 Decisión Adoptada: Kernel SIMD en C y Multiprocesamiento
1. **Kernel C Vectorizado Dinámico:** Se implementó en `src/uav_ip/uav_ip/synthetic_lidar.py` (`_get_fps_c_kernel`):
   * Disposición de datos en Estructura de Arreglos (SoA: `xs`, `ys`, `zs` contiguos) para optimizar el acceso a caché L1/L2.
   * Compilación dinámica bajo demanda con `gcc -O3 -march=native -ffast-math -shared -fPIC`.
   * Redujo el tiempo de cómputo por nube de $\approx 43\text{ ms}$ a **$\approx 6\text{ ms}$** ($\sim 7\times$ de aceleración).
2. **Procesamiento por Lotes Multiproceso:**
   * Soporte en `bag_to_heterodata.py` mediante `--dataset-dir` y `--workers`.
   * Utiliza `multiprocessing.Pool` para paralelizar la conversión de episodios entre todos los núcleos disponibles de la CPU.
   * Procesamiento de 100 episodios en $\sim 5\text{ segundos}$ en un procesador moderno de 16 hilos.

---

## 15. Patrón Arquitectónico: Módulo Central `uav_ip` y Thin Wrappers Ejecutables

### 15.1 El Riesgo de Duplicación de Código
En versiones tempranas existían copias independientes de archivos como `synthetic_lidar.py` y `visualize_dataset.py` tanto en `src/uav_ip/scripts/` como en `src/uav_ip/uav_ip/`. Esto generaba riesgo inminente de desincronización, donde correcciones en un archivo no se propagaban al otro.

### 15.2 Decisión Adoptada: Punto Único de Verdad
1. Toda la lógica de negocio, algoritmos matemáticos y clases residen de manera unificada en `src/uav_ip/uav_ip/`.
2. Los scripts homónimos en `src/uav_ip/scripts/` se transformaron en **thin wrappers** mínimos (menos de 10 líneas) que importan `main()` o las funciones respectivas desde `uav_ip`:
   ```python
   #!/usr/bin/env python3
   from uav_ip.synthetic_lidar import main
   if __name__ == '__main__':
       main()
   ```
3. Esto garantiza que la ejecución directa desde terminal (`python3 src/uav_ip/scripts/synthetic_lidar.py`), la importación como módulo de Python (`from uav_ip.synthetic_lidar import ...`) y la ejecución vía ROS 2 (`ros2 run uav_ip ...`) ejecuten exactamente la misma base de código validada.

---

## 16. Verificación Física Independiente Post-Vuelo, Tolerancias de Apertura y Factibilidad de Transiciones

### 16.1 El Problema de Acoplar el Éxito del Episodio al Código de Salida del Controlador
En la versión previa del pipeline, `data_collector.py` etiquetaba un episodio con `"status": "success"` únicamente si el proceso `waypoint_controller` terminaba con `returncode == 0`. Una revisión técnica exhaustiva identificó cuatro riesgos críticos en ese diseño:
1. **Falsos cruces de puerta (`traverse`):** El criterio de avance del waypoint `center` en `waypoint_controller.py` permitía `(along_track > -0.10 and cross_track < 1.4 and alt_err < 0.8)`. Dado que la ventana interior libre de una puerta cuadrada mide $1.5\text{ m} \times 1.5\text{ m}$ (semiancho libre $\pm 0.75\text{ m}$, semiancho exterior del marco $\pm 0.85\text{ m}$), una desviación lateral de $0.85\text{–}1.39\text{ m}$ rodeaba la puerta por fuera y aun así era marcada como alcanzada por el piloto.
2. **Ausencia de verificación de colisiones:** Ni Gazebo ni el piloto verificaban si el radio físico de los rotores del cuadricóptero `gz_x500` ($R_{\text{UAV}} \approx 0.28\text{ m}$) intersectaba las barras del aro o el cilindro de un poste.
3. **Finalización de aterrizaje por temporizador fijo:** El piloto salía con `SystemExit(0)` exactamente $4.0\text{ s}$ después de enviar `VEHICLE_CMD_NAV_LAND`, sin comprobar `/fmu/out/vehicle_land_detected` ni la altura final real sobre la plataforma.
4. **Saltos de offset en reloj `Timesync` de XRCE-DDS:** En PX4 SITL conectado mediante `MicroXRCEAgent`, el cliente `uxrce_dds_client` recalibra periódicamente el offset entre el reloj de arranque de PX4 y el reloj UNIX del host. En $1\text{–}4$ muestras aisladas por episodio (de $\sim 3500$ mensajes a $125\text{ Hz}$), `VehicleOdometry.timestamp` cae transitoriamente al tiempo crudo de boot ($\sim 60\text{ s}$) o salta $\sim 280\text{ ms}$, mientras que el reloj de grabación MCAP (`log_time`) avanza monótonamente a $\Delta t = 8.0\text{ ms}$.

### 16.2 Decisión Adoptada: Validador Físico Analítico Independiente (`validate_episode.py`)
En lugar de instrumentar sensores de contacto en Gazebo (*contact plugins* en XML SDF y puentes adicionales `ros_gz_bridge`), los cuales dependen de que el motor de colisiones discreto no sufra *tunneling* y no detectan errores semánticos (como pasar por fuera de un aro o por el flanco opuesto de un poste), se diseñó el validador independiente [`src/uav_ip/uav_ip/validate_episode.py`](file:///home/gabrielq/ros2_ws/src/uav_ip/uav_ip/validate_episode.py):
1. **Separación de Estados en `episode_meta.json`:**
   * `pilot_exit_code`: Código de retorno del proceso `waypoint_controller`.
   * `recording_status` (`"valid"` / `"invalid"`): Integridad del archivo `.mcap`, presencia y cantidad mínima de mensajes en `/fmu/out/vehicle_odometry`, ausencia de `NaN`/cuaterniones degenerados, continuidad temporal ($\Delta t_{\max} \le 0.60\text{ s}$) y **cero resets del estimador EKF2 en vuelo** ($\Delta\text{reset\_counter} = 0$ durante la ventana aérea).
   * `mission_status` (`"valid"` / `"invalid"`): Cumplimiento geométrico de despegue, primitivas, margen libre de colisión contra todos los obstáculos y aterrizaje sobre la plataforma.
   * `validation_status` (`"accepted"` / `"rejected"`): Aceptado sí y solo si `recording_status == "valid"` y `mission_status == "valid"`, acompañado de `rejection_reasons`, `primitive_validation` y `validation_metrics`.
2. **Geometría Analítica Exacta de Colisión y Primitivas a $125\text{ Hz}$:**
   * Como los landmarks son cuerpos rígidos estáticos de dimensiones conocidas exactamente en `waypoints.yaml`, la distancia mínima desde la trayectoria 3D $\mathbf{p}(t)$ a las 4 cajas AABB de cada `square_gate`, al toroide $(R=0.75\text{ m}, r=0.05\text{ m})$ de cada `circular_gate` y al cilindro $(r=0.10\text{ m}, h=3.0\text{ m})$ de cada `pole` se evalúa analíticamente con precisión milimétrica en todos los instantes de odometría ($125\text{ Hz}$, espaciado espacial $\sim 1\text{ cm}$), exigiendo $d_{\text{min}} \ge R_{\text{UAV}} = 0.28\text{ m}$.
   * Para **`traverse`**: se interpola la intersección exacta de la trayectoria con el plano local $x_{\text{loc}} = 0$ del aro y se exige $|y_{\text{cross}}| \le 0.58\text{ m}$, $|z_{\text{cross}}| \le 0.58\text{ m}$ (dentro de la abertura libre de $\pm 0.70\text{ m}$).
   * Para **`slalom`**: se verifica el cruce de $x_{\text{loc}} = 0$ por el semiplano lateral correcto en el marco FLU del poste ($y_{\text{cross}} \in [+0.45, +2.60]\text{ m}$ para `left`, $y_{\text{cross}} \in [-2.60, -0.45]\text{ m}$ para `right`).
   * Para **`overfly`**: se verifica el paso sobre el poste con $z_{\text{cross}} \ge 3.30\text{ m}$ ($0.30\text{ m}$ por encima de la cima física de $3.0\text{ m}$) y alineación lateral $|y_{\text{cross}}| \le 0.85\text{ m}$.
   * Para **`orbit`**: se mide la progresión angular antihoraria acumulada $\Delta\theta_{\text{ccw}} = \max_t (\theta(t) - \min_{s \le t} \theta(s))$ alrededor del eje del poste, exigiendo $\Delta\theta_{\text{ccw}} \ge 210^\circ$ y error medio de guiñada centrípeta $\le 45^\circ$.
3. **Reconstrucción Monótona del Reloj de Simulación (`reconstruct_simulation_clock`):**
   * Cuando un paso $\Delta t_{\text{px4}} = t_{\text{px4}}[i] - t_{\text{px4}}[i-1]$ difiere del paso del grabador $\Delta t_{\text{bag}}$ en más de $50\text{ ms}$ mientras $\Delta t_{\text{bag}} \in [0, 100]\text{ ms}$ (firma inequívoca de actualización de `time_offset` en `uxrce_dds_client`), el incremento anómalo se sustituye por $\Delta t_{\text{bag}}$, preservando el reloj físico de PX4 libre de saltos de sincronización.
4. **Endurecimiento del Controlador y Factibilidad de Transiciones:**
   * En `waypoint_controller.py`, el cruce `center` de una primitiva `traverse` exige ahora `cross_track < 0.38 m` y `alt_err < 0.42 m`, impidiendo que el piloto dé por superada una puerta si no está alineado con su ventana central, y el estado `nav_state == 4` aguarda confirmación de `/fmu/out/vehicle_land_detected` (`landed == True`) o $z_{\text{ENU}} < 0.22\text{ m}$.
   * En `world_generator.py`, `check_circuit_transition_feasibility` verifica antes de aceptar un mundo aleatorio que todo par de primitivas consecutivas cumpla progresión longitudinal $\Delta x = x_{\text{pre}, i+1} - x_{\text{post}, i} \ge 1.0\text{ m}$, distancia horizontal $d_{xy} \ge 1.4\text{ m}$ y pendiente vertical $|\Delta z| / d_{xy} \le 1.15$ cuando interviene un `traverse` ($\le 1.60$ entre postes abiertos).

