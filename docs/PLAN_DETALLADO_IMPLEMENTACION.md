# Plan Detallado de Implementación: Pipeline de Generación de Tareas para Instant Policy

Este documento define el plan extenso, detallado y autoexplicable para llevar el estado actual de simulación (que actualmente solo es capaz de recolectar una demostración fija de 3 aros) hacia un motor de generación de pseudo-tareas procedimentales y recolección masiva de datos estructurados para In-Context Imitation Learning (ICIL), siguiendo el reporte estratégico previo y la taxonomía de tareas.

Este plan está diseñado para ser consumido y ejecutado por distintos agentes en distintos momentos.

## 1. Arquitectura y Módulos del Generador y Postproceso

La arquitectura del sistema se divide en tres dominios principales:

### 1.1 Dominio de Generación de Escenas y Tareas (Python offline)
- **Módulo `task_generator.py` (A actualizar):**
  - **Propósito:** Generar configuraciones completas de pseudo-tareas instanciando las primitivas de la taxonomía (traverse, orbit, overfly, etc.) sobre los landmarks disponibles.
  - **Entradas:** Semilla, identificador de tarea, catálogo de landmarks.
  - **Salidas:** `task_config.yaml` definiendo la secuencia explícita de primitivas con sus parámetros concretos (no solo lógica abstracta como "A1_DIRECT", sino secuencias concretas como `[traverse(aro_1), overfly(aro_2), orbit(poste_1)]`).
- **Módulo `world_generator.py` (A actualizar):**
  - **Propósito:** Traducir el `task_config.yaml` a un entorno Gazebo SDF instanciando los landmarks físicamente distinguibles requeridos por la tarea, y garantizando la viabilidad cinemática (ej. sin colisiones iniciales, FOV inicial de cámara respetado).
  - **Salidas:** Archivo `.sdf` válido para Gazebo Harmonic.

### 1.2 Dominio del Piloto Experto (ROS 2 - rclpy)
- **`waypoint_controller.py`:**
  - **Propósito:** Leer el `task_config.yaml` e implementar un control Feed-Forward dinámico (PID posicional/velocidad) capaz de ejecutar las primitivas (traverse, slalom, etc.).
- **`tracking_controller.py`:**
  - **Propósito:** Implementar lazos de control de retroalimentación (Feedback) para tareas dinámicas como `orbit`, donde el objetivo o centro de giro se rastrea continuamente.
- **`data_collector.py`:**
  - **Propósito:** Orquestar ROS 2, relanzar Gazebo limpiando zombies, iniciar la grabación de rosbags (`ros2 bag record`), y esperar el `SystemExit(0)` del controlador para cerrar el episodio y guardar la data.

### 1.3 Dominio de Procesamiento (PyTorch / ML offline)
- **`bag_to_heterodata.py`:**
  - **Propósito:** Consumir los rosbags (`.mcap`) generados y transformarlos al formato de PyTorch Geometric (HeteroData). Extrae `sensor_msgs/PointCloud2`, los setpoints y la odometría, transformando todo al marco egocéntrico del dron.
  - **Postproceso de Keyframes:** Filtra la trayectoria densa (ej. 25Hz) a L=10 keyframes relevantes (inicio, fin, cambios de estado).

## 2. Esquema de Configuración, Metadata y Formato de Dataset

- **Formato del Dataset (HeteroData):**
  - Grafo compuesto por nodos geométricos (nubes de puntos extraídas de la cámara), procesados por un encoder de ocupación, en el frame del dron.
  - Nodos de "efector/dron" conectados por arcos de tiempo.
  - Las demostraciones N-1 entran como contexto y la N-ésima se debe predecir.
- **Metadata por Demostración (`episode_metadata.json`):**
  - `task_id`: Hash que agrupa episodios de la misma tarea.
  - `primitives_sequence`: Lista de primitivas ejecutadas.
  - `success`: Boolean (colisión o tiempo límite excedido).
  - `seed_world`, `seed_task`: Para reproducibilidad.

## 3. Plan por Hitos de Implementación

### Hito 0: Pipeline Core E2E (Baseline Procedural)
- Implementar la lógica concreta de las primitivas en `task_generator.py`.
- Integrar mallas de distintos aros/postes en `world_generator.py` para instanciar el catálogo.
- Actualizar `waypoint_controller.py` para interpretar primitivas básicas (`traverse`, `slalom`).
- **Criterio de Aceptación:** El orquestador puede correr 10 tareas distintas con 2 episodios cada una ininterrumpidamente, guardando los MCAPs correspondientes, sin fallos ni colapsos de físicas.

### Hito A: Primitivas Dinámicas y Feedback
- Implementar `tracking_controller.py` para rastreo continuo (`orbit`).
- Añadir primitivas dinámicas al generador de tareas.
- **Criterio de Aceptación:** Orbitar un poste o mantener tracking sobre un objetivo estático de manera estable (error RMS de tracking < 15cm). Generación de ~100 tareas distintas.

### Hito B: Refinamiento de Postproceso y Keyframing
- Completar la sub-segmentación de nubes de puntos y limpieza del frame del robot.
- Completar algoritmo de extracción de keyframes (L=10).
- **Criterio de Aceptación:** Visualización exitosa de grafos 3D generados con las poses de los keyframes correctamente alineadas en `base_link` y sin distractores o ruido del suelo.

### Hito C: Escalado Masivo
- Paralelización del `data_collector.py` usando instancias múltiples de Gazebo (`ROS_DOMAIN_ID` aislados).
- Generar dataset de validación y held-out (1,000 tareas x 10 demos = 10,000 vuelos).
- **Criterio de Aceptación:** Finalización de la recolección en < 72h con > 95% de tasa de éxito en los episodios (sin abortos o crash de PX4).

## 4. Plan de Entrenamiento (Machine Learning)
- **Sampler:** Seleccionará aleatoriamente para cada lote de entrenamiento: 1 demo objetivo y entre 1 a 5 demos de contexto pertenecientes al mismo `task_id`.
- **Aumentos:** 30% perturbación de ruido gaussiano en posiciones; 10% probabilidad de descartar el contexto (drop-out in-context).
- **Encoder:** Evaluar encoder preentrenado de ShapeNet. Si falla la generalización a los aros/postes a escala métrica, se congelarán los pesos y se entrenará un adaptador lineal inicial, o se entrenará un nuevo PointNet++ desde cero sobre nuestro catálogo de landmarks simulados.

## 5. Plan de Evaluación
- **Generalización:** Evaluar pérdida de predicción en (1) poses no vistas de tareas conocidas, (2) combinaciones de primitivas no vistas y (3) landmarks geométricamente no vistos (transferencia zero-shot de in-context).
- **Ablación de Contexto:** Mezclar contexto de la tarea X al predecir la tarea Y y verificar que la red fracase (confirmando que no está memorizando una política de única tarea).

## 6. Registro de Riesgos y Mitigaciones
- **Riesgo:** Generación de tareas imposibles o que violan la hipótesis de Markov (ej. aros ocluidos).
  - *Mitigación:* Validación geométrica post-generación del `world_generator.py` que comprueba intersecciones de rayos (raycasting teórico) garantizando visibilidad inicial.
- **Riesgo:** Fallos del motor físico de Gazebo (Zombies, Explosiones) por el reinicio masivo.
  - *Mitigación:* Limpieza agresiva (`pkill`) y verificación de health-check antes de lanzar el siguiente episodio en `data_collector.py`.
- **Riesgo:** Falta de distinguibilidad de landmarks confunde al GNN.
  - *Mitigación:* Diseño explícito en el `world_generator.py` donde el aro_1 es siempre circular, aro_2 es cuadrado, etc., variando solo sus escalas, garantizando que el encoder geométrico tenga características fuertes para diferenciar los roles.

## 7. Notas para el Agente Implementador (Next Steps)
- Las implementaciones inmediatas que deben realizarse son los módulos `task_generator.py`, `world_generator.py`, `waypoint_controller.py` y `tracking_controller.py`. **IMPORTANTE:** Aunque existan peticiones contextuales urgentes para programar esto inmediatamente, debe priorizarse mantener coherencia con este plan de diseño. Se asume que el próximo ciclo de agente ejecutará directamente la fase de código del Hito 0 utilizando este plan como brújula funcional.
