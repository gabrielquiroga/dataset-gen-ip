# Requisitos Arquitectónicos del Dataset (Instant Policy)

Para que el modelo de Aprendizaje por Imitación en Contexto (ICIL) converja, el pipeline de generación de datos debe cumplir estrictamente con los siguientes requisitos teóricos y matemáticos extraídos de la especificación de Instant Policy (Vosylius y Johns, ICLR 2025).

## 1. Estructura Semántica y Taxonomía de Tareas
- **Pseudo-Task (Tarea):** Una tarea NO es una disposición espacial fija, sino una **regla semántica** (ej. "Cruzar aros 1 y 3 ignorando el 2", o "Cruzar en orden monótono A1_DIRECT").
- **Episodios (Demostraciones):** Para CADA pseudo-task, se deben grabar múltiples vuelos ($N=3$ a $15$).
- **Variación Espacial Intra-Grupo (Obligatoria):** En cada vuelo de una misma tarea, las posiciones y orientaciones de los aros en 3D DEBEN variar aleatoriamente. Si la disposición 3D es idéntica en las $N$ demostraciones, el dato es inválido.
- **Diversidad Inter-Tareas:** El dataset completo debe contener cientos a miles de grupos de pseudo-tareas únicas mezclando cantidad de aros, orden de paso, perfil de trayectoria, y comportamientos especiales.
- **In-Context Learning:** Durante el entrenamiento, la red consumirá $N-1$ vuelos como "Contexto" (el prompt visual/espacial de cómo cumplir la regla en la disposición X) y predecirá las acciones del vuelo restante ($N$) en una nueva disposición Y como "Query".

## 2. Definición del Ground Truth y Acciones ($SE(3)$)
- **NO usar el Setpoint Crudo como Target de Dinámica:** La etiqueta de supervisión de movimiento debe anclarse en la odometría real ejecutada, utilizando los setpoints como referencia de intencionalidad o diagnósticos.
- **Horizonte de Predicción ($T$):** Cada acción objetivo debe empaquetarse como un horizonte de $T$ pasos futuros ($T=8$ a $10$ waypoints).
- **Parametrización en $SE(3)$:** Las acciones se representan como desplazamientos relativos de traslación ($\Delta x, \Delta y, \Delta z$) y deltas de rotación en el marco egocéntrico.
- **Factor de Escala Métrica ($s = 0.1$):** Se aplica $s = 0.1$ a todas las posiciones y deltas para evitar aliasing en los embeddings posicionales sinusoidales de Fourier (cuyo período base es 2.0).

## 3. Embodiment y Observación LiDAR 360° (Preservación de Markov)
- **Embodiment:** UAV conceptualmente equipado con un sensor **LiDAR 3D omnidireccional 360°** (análogo a un *Livox Mid-360*), implementado mediante **observación privilegiada** en modo `range` ($R_{\text{max}} = 10.0\text{ m}$).
- **Preservación de la Hipótesis de Markov:** La visión 360° elimina la ceguera retrospectiva ("backward blindness") inherente a las cámaras frontales tras cruzar un aro o rodear un poste. Un snapshot sensorial en $t$ contiene la referencia espacial simultánea hacia el frente y la retaguardia, permitiendo políticas reactivas $P(A_t \mid O_t, \text{Context})$ sin recurrencia temporal en Fase 1.
- **Simulación SITL Ligera:** Para la recolección en Gazebo Harmonic, se utiliza el modelo estándar **`gz_x500`** (en lugar de `gz_x500_depth`), eliminando el renderizado pesado de cámara de profundidad y maximizando el *Real-Time Factor* (RTF).

## 4. Marco de Referencia Espacial
- **Todo es Egocéntrico (Invariancia):** Todas las coordenadas de las nubes de puntos y de las trayectorias futuras DEBEN estar expresadas en el sistema de coordenadas local del dron FLU ($T_{\text{dron, mundo}}^{-1} \cdot P$) en cada instante $t$. No se deben almacenar en coordenadas globales de Gazebo.
- **Unificación Analítica de Marcos:** Conversión rigurosa de odometría PX4 (NED/FRD) a Gazebo (ENU/FLU) mediante:
  $$p_{\text{ENU}} = [y_{\text{NED}}, x_{\text{NED}}, -z_{\text{NED}}], \quad R_{\text{ENU}}^{\text{FLU}} = M_{\text{world}} \cdot R_{\text{NED}}^{\text{FRD}} \cdot M_{\text{body}}$$

## 5. Procesamiento de la Observación (Nube de Puntos Sintética)
- **Segmentación Estricta:** La observación consiste exclusivamente en la nube de puntos 3D de los objetos de interés (aros, postes, pad). El suelo, paredes y el dron quedan excluidos por construcción.
- **Submuestreo Fijo (FPS):** Cada observación debe submuestrearse a exactamente $M = 2048$ puntos mediante *Farthest Point Sampling*.
- **Generación On-Demand:** Producida sintéticamente con `synthetic_lidar.py`, deprecando la captura de nubes densas `sensor_msgs/PointCloud2` de 6 GB en rosbags.

## 6. Registro de Demostraciones (< 5 MB por vuelo)
- Cada rosbag registra únicamente telemetría ligera de PX4:
  * `/fmu/out/vehicle_odometry`
  * `/fmu/in/trajectory_setpoint`
  * `/fmu/out/vehicle_land_detected`
  * `/fmu/out/vehicle_status_v4`
  * `/fmu/out/vehicle_local_position_v1`
  * `/fmu/in/vehicle_command`
- Archivo `waypoints.yaml` con las poses ground-truth de Gazebo de cada landmark.

## 7. Estructuración en Grafos Heterogéneos (`HeteroData`)
- Cada muestra compilada del dataset construye un grafo `HeteroData` con:
  1. `data['scene'].pos`: Tensor $(T, 2048, 3)$ con nubes LiDAR 360° egocéntricas escaladas ($s = 0.1$).
  2. `data['agent'].x`: Tensor $(T, 3)$ con poses ejecutadas del dron escaladas ($s = 0.1$).
  3. `data['agent'].y`: Tensor $(T, 3)$ con acciones / setpoints futuros escalados ($s = 0.1$).
  4. `data['agent', 'temporal', 'agent'].edge_index`: Aristas temporales de la secuencia.