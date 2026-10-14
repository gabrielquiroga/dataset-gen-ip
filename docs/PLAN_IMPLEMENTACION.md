# Hoja de Ruta y Plan de Implementación: UAV Instant Policy

*Plan de ingeniería de software, pipeline de datos e hitos de desarrollo · Octubre de 2026*

---

## 1. Arquitectura del Sistema de Simulación y Datos

El proyecto desacopla estrictamente tres dominios funcionales:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   DOMINIO 1: GENERACIÓN PROCEDIMENTAL                  │
│                                                                        │
│   task_generator.py                world_generator.py                  │
│   (Define semántica fija           (Instancia poses 3D aleatorias      │
│    en task_config.yaml)       ───►  y genera SDF + waypoints.yaml)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                   DOMINIO 2: VUELO SITL Y ORQUESTACIÓN                 │
│                                                                        │
│   data_collector.py                PX4 SITL + Gazebo Harmonic (gz_x500)│
│   (Lanza XRCE, PX4, Bag, Piloto)──► (Vuelo de alta fidelidad, RTF ≥ 1.0)│
│                                    Piloto Experto (waypoint_controller)│
│                                    (Cruza aros, órbita centrípeta)     │
│                                           │                            │
│                                           ▼                            │
│                                    Rosbag ligero (< 5 MB / vuelo)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                   DOMINIO 3: ML OFFLINE Y HETERODATA                   │
│                                                                        │
│   synthetic_lidar.py               bag_to_heterodata.py                │
│   (Observación privilegiada 360°,  (Parsea MCAP, genera grafos         │
│    FPS 2048 puntos, escala s=0.1)──►PyG HeteroData .pt comprimidos)    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Estado de los Componentes de Software

| Módulo / Script | Ubicación | Estado Actual | Rol en el Pipeline |
| :--- | :--- | :--- | :--- |
| **`task_generator.py`** | `src/uav_ip/scripts/` | **Completado** (A mejorar en Hito 1) | Genera `task_config.yaml` definiendo la secuencia de primitivas y landmarks. |
| **`world_generator.py`** | `src/uav_ip/scripts/` | **Completado** | Genera el SDF para Gazebo y el `waypoints.yaml` con poses aleatorias factibles. |
| **`base_pilot.py`** | `src/uav_ip/uav_ip/` | **Completado** | Clase base de ROS 2 para ciclo de vida, arming seguro (50 Hz pre-flight) y QoS. |
| **`waypoint_controller.py`** | `src/uav_ip/uav_ip/` | **Completado** | Piloto experto feed-forward (`traverse`, `slalom`, `overfly`, `orbit` con yaw centrípeto). |
| **`tracking_controller.py`** | `src/uav_ip/uav_ip/` | **Prototipo funcional** | Control por retroalimentación continua (PID en velocidad) para seguimiento dinámico. |
| **`data_collector.py`** | `src/uav_ip/scripts/` | **Completado** | Orquestador multi-tarea y multi-episodio con variación espacial intra-tarea e inyección limpia de procesos. |
| **`synthetic_lidar.py`** | `src/uav_ip/scripts/` y `uav_ip/` | **Completado** | Sintetizador privilegiado de nubes LiDAR 360° en marco FLU, con FPS y escala $s=0.1$. |
| **`bag_to_heterodata.py`** | `src/uav_ip/scripts/` | **Completado** (Modo individual) | Extractor de telemetría MCAP a grafos PyG `HeteroData` `.pt`. |
| **`visualize_dataset.py`** | `src/uav_ip/scripts/` y `uav_ip/` | **Completado** | Visualizador 3D estático y animador GIF de trayectorias y nubes LiDAR sintéticas. |

---

## 3. Plan de Hitos de Implementación

### Hito 0: Pipeline Core E2E y Validación de Vuelo — [COMPLETADO]
- [x] Modelo de simulación ligero `gz_x500` (sin cámara física de profundidad).
- [x] Transición de armado seguro a Offboard en `base_pilot.py` (50 setpoints previos a 50 Hz).
- [x] Generador modular de LiDAR privilegiado `synthetic_lidar.py` (2048 puntos, FPS, marco egocéntrico FLU, escala $s=0.1$).
- [x] Grabación compacta en rosbags MCAP ($< 5\text{ MB}$ por demo frente a los 6 GB originales).
- [x] Aleatorización espacial intra-tarea: `data_collector.py` regenera las poses físicas para cada episodio con semilla determinista única.
- [x] Refinamiento de la primitiva `orbit` en `waypoint_controller.py` con yaw centrípeto de inspección hacia el centro del cilindro y salida tangencial hacia $+X$.
- [x] Conversión validada a grafos `HeteroData` con `bag_to_heterodata.py` y visualización animada en GIF con `visualize_dataset.py`.

---

### Hito 1: Robustez en Generación de Tareas y Partición Held-Out — [SIGUIENTE PASO]

*Objetivo: Garantizar la integridad combinatoria del dataset, eliminando duplicados aleatorios y estructurando formalmente la partición para evaluar la generalización in-context.*

#### Tarea 1.1: Generación de Tareas sin Reemplazo (`task_generator.py`)
- **Problema:** Con 3 landmarks y el catálogo actual existen exactamente 64 firmas semánticas únicas. El muestreo con reemplazo genera $> 90\%$ de duplicados a partir de 20 tareas (Paradoja del Cumpleaños).
- **Acción:**
  * Implementar en `task_generator.py` un catálogo exhaustivo de las 64 combinaciones válidas.
  * Permitir seleccionar tareas por índice determinista (`--task-index 0..63`) o muestreo aleatorio sin reemplazo.
- **Criterio de Aceptación:** Si se generan 60 tareas consecutivas, las 60 presentan firmas semánticas estrictamente distintas en sus `task_config.yaml`.

#### Tarea 1.2: Script de Auditoría y Partición (`audit_dataset.py`)
- **Acción:**
  * Crear un script que escanee recursivamente `datasets/task_XXXX/task_config.yaml`.
  * Extraer la firma semántica de cada tarea:
    $$\text{Firma} = [(\text{lm}_1, \text{prim}_1), (\text{lm}_2, \text{prim}_2), (\text{lm}_3, \text{prim}_3)]$$
  * Reportar estadísticas: número de tareas únicas, episodios por tarea, tasas de éxito y colisiones.
  * Generar los índices JSON de partición para entrenamiento:
    * `train_tasks.json` ($\sim 70\%$ de las firmas).
    * `val_tasks.json` ($\sim 15\%$ de las firmas).
    * `held_out_tasks.json` ($\sim 15\%$ de las firmas nunca vistas).
- **Criterio de Aceptación:** Cero solapamiento de firmas semánticas entre `train_tasks.json` y `held_out_tasks.json`.

---

### Hito 2: Herramientas de ML y Postproceso por Lotes — [SIGUIENTE PASO]

*Objetivo: Desacoplar las dependencias de Machine Learning del entorno de ROS 2 y optimizar la conversión masiva de datos.*

#### Tarea 2.1: Entorno Virtual Aislado para ML (`venv_ip`)
- **Problema:** Ubuntu 24.04 impone PEP 668. Instalar PyTorch y PyG a nivel global puede romper paquetes debian de ROS 2 Jazzy.
- **Acción:**
  * Configurar un entorno virtual dedicado (`~/venv_ip` o Conda).
  * Instalar dependencias puras: `torch`, `torch_geometric`, `rosbags`, `scipy`, `numpy`, `matplotlib`.
  * La librería `rosbags` permite procesar archivos `.mcap` dentro del venv **sin compilar ni cargar `/opt/ros/jazzy`**.
- **Criterio de Aceptación:** Ejecución de `python3 -c "import torch, torch_geometric, rosbags; print('OK')"` dentro del venv sin errores.

#### Tarea 2.2: Procesamiento por Lotes Multiproceso en `bag_to_heterodata.py`
- **Problema:** Actualmente `bag_to_heterodata.py` procesa un único archivo rosbag a la vez por invocación.
- **Acción:**
  * Incorporar el argumento `--dataset-dir datasets/` para escanear automáticamente todas las tareas y episodios.
  * Implementar paralelismo a nivel de procesos mediante `multiprocessing.Pool`:
    * Cada core de CPU procesa un episodio de forma independiente.
    * Generación paralela de nubes sintéticas, FPS a 2048 puntos y serialización a `.pt`.
- **Criterio de Aceptación:** Conversión de 100 episodios en $< 30\text{ segundos}$ en una máquina multinúcleo estándar.

---

### Hito 3: Campaña de Recolección del Dataset Completo
- **Acción:**
  * Ejecución de `data_collector.py` para generar el dataset de entrenamiento base (ej. 45 tareas $\times$ 10 episodios = 450 vuelos, o escalado a 1.000 vuelos).
  * Procesamiento en lote con `bag_to_heterodata.py` hacia `dataset_processed/`.
  * Verificación visual de calidad geométrica mediante `visualize_dataset.py`.
- **Criterio de Aceptación:** Dataset procesado con $> 95\%$ de vuelos exitosos, estructura de grafo `HeteroData` conforme y distribución uniforme de keyframes.

---

### Hito 4: Modelo de Difusión sobre Grafos y Validación de Instant Policy
1. **Sampler de Entrenamiento In-Context:**
   * Para cada muestra, sortear 1 demo objetivo y entre 1 a 5 demos de contexto pertenecientes al mismo `task_id`.
   * Empaquetar el grafo conjunto conectando el efector con las observaciones a través del tiempo.
2. **Entrenamiento del Modelo:**
   * Adaptación de la arquitectura de difusión (DDIM + SVD) sobre las acciones densas relativas escaladas por $s = 0.1$.
   * Congelamiento del encoder geométrico preentrenado.
3. **Experimentos de Validación:**
   * **Test de Ablación de Contexto:** Sustituir las demos de contexto por demostraciones de otra tarea distinta sobre la misma escena. El rendimiento del modelo debe colapsar, demostrando que efectivamente está leyendo e imitando el contexto.
   * **Evaluación en Tareas Held-Out:** Medir la tasa de éxito semántico en las firmas semánticas de `held_out_tasks.json`.
