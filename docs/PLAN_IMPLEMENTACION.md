# Hoja de Ruta y Plan de Implementación: UAV Instant Policy

*Plan maestro de ingeniería de software, correcciones matemáticas, pipeline de datos y aprendizaje profundo · Actualizado a Octubre de 2026*

---

## 1. Arquitectura General del Sistema (4 Dominios Desacoplados)

El proyecto estructura el flujo completo desde la generación procedimental hasta el control en lazo cerrado en cuatro dominios funcionales estrictamente desacoplados:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     DOMINIO 1: GENERACIÓN PROCEDIMENTAL                      │
│                                                                              │
│   task_generator.py                   world_generator.py                     │
│   (Define firma semántica única       (Instancia poses 3D aleatorias y       │
│    sin reemplazo en task_config.yaml)  genera .sdf + waypoints.yaml)         │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────────────────┐
│                     DOMINIO 2: VUELO SITL Y ORQUESTACIÓN                     │
│                                                                              │
│   data_collector.py                   PX4 SITL + Gazebo Harmonic (gz_x500)   │
│   (Lanza XRCE, PX4 -d, Bag, Piloto)─► (Vuelo de alta fidelidad, RTF ≥ 1.0)   │
│                                       Piloto Experto (waypoint_controller)   │
│                                       (Trayectorias suaves + yaw centrípeto) │
│                                                    │                         │
│                                                    ▼                         │
│                                       Rosbag ligero MCAP (< 5 MB / vuelo)    │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────────────────┐
│               DOMINIO 3: POSTPROCESO OFFLINE Y GRAFOS HETERODATA             │
│                                                                              │
│   synthetic_lidar.py                  bag_to_heterodata.py                   │
│   (Observación privilegiada 360° en   (Extrae poses SE(3), acciones relativas│
│    marco Horizontal FLU, FPS 2048,  ─► T_EA a horizonte T=8, keyframes L=10, │
│    escala s=0.1, determinista)         30% tube augmentation y guarda .pt)   │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────────────────┐
│             DOMINIO 4: DIFUSIÓN SOBRE GRAFOS Y CONTROL CLOSED-LOOP           │
│                                                                              │
│   In-Context Graph Sampler            Red de Difusión Hetero-GNN (σ, φ, ψ)   │
│   (N-1 demos contexto [L=10] +      ─►(Encoder escena φ_e + 6 keypoints UAV  │
│    1 observación actual O_t)           + flujo desacoplado DDIM + SVD SE(3)) │
│                                                    │                         │
│                                                    ▼                         │
│                                       Nodo ROS 2 de Inferencia Closed-Loop   │
│                                       (Rollout + filtro B-spline -> PX4)     │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Estado de los Componentes de Software

| Módulo / Script | Ubicación | Estado Actual | Observaciones y Deuda Técnica Identificada |
| :--- | :--- | :--- | :--- |
| **`task_generator.py`** | `src/uav_ip/scripts/` | **Completado** | Muestreo combinatorio sin reemplazo, firma semántica limpia, `slalom` bilateral (`left`/`right`) y soporte para `circular_gate`. |
| **`world_generator.py`** | `src/uav_ip/scripts/` | **Completado** | Inyección dinámica SDF operativa con `circular_gate` poligonal de 16 segmentos (`get_circular_gate_cylinders_spec()`), validación de factibilidad geométrica de transiciones (`check_circuit_transition_feasibility`), alineación de `plataforma_llegada` y variación de `start_pose`. |
| **`base_pilot.py`** | `src/uav_ip/uav_ip/` | **Completado** | Ciclo de vida ROS 2, pre-flujo de 50 setpoints a 50 Hz para armado Offboard seguro, suscripción a `/fmu/out/vehicle_land_detected`, registro de frescura de mensajes y conversión base NED/ENU. |
| **`waypoint_controller.py`** | `src/uav_ip/uav_ip/` | **Completado** | Ejecuta `traverse`, `slalom`, `overfly`, `orbit` con velocidad *feed-forward* en `center`/arcos, tolerancias de paso derivadas de `validation_geometry.py` (`|y| ≤ 0.33m, |z| ≤ 0.38m` en cuadrada y `r ≤ 0.3656m` en circular) y aterrizaje confirmado por persistencia continua (`3.0 s`) de `landed` (código 1 propagado si vence `10.0 s`). |
| **`tracking_controller.py`** | `src/uav_ip/uav_ip/` | **Completado** | Conversión consistente entre coordenadas Gazebo ENU y PX4 NED en el cálculo de rumbo. |
| **`data_collector.py`** | `src/uav_ip/scripts/` | **Completado** | Orquestador multi-tarea/episodio robusto con `PX4_GZ_MODEL_POSE` unificado y variación de pose inicial (integración en línea del validador pendiente en Fase 2). |
| **`validation_geometry.py`** | `src/uav_ip/uav_ip/` | **Completado** | Fuente única de verdad geométrica, mediciones trazables del SDF `x500` (`UAV_COLLISION_RADIUS_XY = 0.30 m`), derivación de tolerancias (`derive_validation_tolerances()`) y especificación 1 a 1 de los 16 cilindros de `circular_gate`. |
| **`validate_episode.py`** | `src/uav_ip/uav_ip/` y `scripts/` | **Completado** | Validador de configuración, grabación y misión independiente del piloto (sin degradación silenciosa entre `rosbags` y parser nativo MCAP/CDR, verificación de CRC32, `VehicleOdometry` obligatoria, reloj estructurado `C3`, aterrizaje en ventana temporal común `C4`, cota inferior de discretización `C7` y secuencia progresiva de primitivas `C8`). |
| **`synthetic_lidar.py`** | `src/uav_ip/uav_ip/` y `scripts/` | **Completado** | Sintetizador LiDAR 360° en marco *Horizontal FLU* ($s=0.1$, FPS 2048 determinista acelerado ~7x con kernel C SIMD), cuota acotada en `landing_pad` y *thin wrapper* en `scripts/`. |
| **`audit_dataset.py`** | `src/uav_ip/scripts/` | **Completado** | Auditoría universal de firmas semánticas, variabilidad espacial y particionado `train` / `val` / `held_out` sin fugas. |
| **`bag_to_heterodata.py`** | `src/uav_ip/scripts/` | **Completado** | Exporta poses $SE(3)$, horizonte relativo $T_{EA}$ ($T=8$), $L=10$ keyframes, 6 keypoints del UAV, *30% Tube Augmentation* de recuperación y procesamiento multiproceso (`--dataset-dir`, `--workers`). |
| **`visualize_dataset.py`** | `src/uav_ip/uav_ip/` y `scripts/` | **Completado** | Visualizador 3D estático, animador GIF y verificador gráfico de *Tube Augmentation* (`--show-aug`) con *thin wrapper* en `scripts/`. |
| **`setup_venv_ip.sh`** | `src/uav_ip/scripts/` | **Completado** | Aprovisionamiento automatizado del entorno virtual aislado `~/venv_ip` e instalación editable de `uav_ip`. |

---

## 3. Plan de Hitos de Implementación (Paso a Paso)

### Hito 0: Pipeline Core E2E y Validación de Vuelo Base — [COMPLETADO]
- [x] Modelo de simulación ligero `gz_x500` (sin cámara física de profundidad), con $\text{RTF} \ge 1.0\times$ observado en la máquina de desarrollo.
- [x] Transición de armado seguro a Offboard en `base_pilot.py` (50 setpoints previos a 50 Hz).
- [x] Generador base de LiDAR privilegiado `synthetic_lidar.py` (2048 puntos, FPS, escala $s=0.1$).
- [x] Grabación compacta en rosbags MCAP ($< 5\text{ MB}$ por demo frente a los 6 GB originales).
- [x] Aleatorización espacial intra-tarea: `data_collector.py` regenera las poses físicas para cada episodio con semilla determinista única.
- [x] Refinamiento de la primitiva `orbit` en `waypoint_controller.py` con yaw centrípeto de inspección hacia el centro del cilindro y salida tangencial hacia $+X$.
- [x] Prototipo inicial de conversión a grafos `HeteroData` con `bag_to_heterodata.py` y visualización animada en GIF con `visualize_dataset.py`.

---

### Hito 1: Generación Combinatoria sin Reemplazo y Auditoría Base — [COMPLETADO]
- [x] **Tarea 1.1: Generación de Tareas sin Reemplazo (`task_generator.py`):**
  * Catálogo combinatorio dinámico e intercalado (`get_balanced_multilength_catalog`) para $N \in [2, 5]$ landmarks.
  * Muestreo determinista sin reemplazo vía `--task-index` y permutación barajada con `--shuffle-seed`.
- [x] **Tarea 1.2: Script de Auditoría y Partición (`audit_dataset.py`):**
  * Escaneo recursivo de `datasets/task_XXXX/task_config.yaml`, detección de duplicados, validación de variabilidad espacial intra-tarea y exportación de splits (`train_tasks.json`, `val_tasks.json`, `held_out_tasks.json`).

---

### Hito 1.5: Corrección de Bugs Críticos, Fidelidad Matemática $SE(3)$ y Calidad de Demostraciones — [COMPLETADO]

*Objetivo: Corregir las discrepancias matemáticas y de ingeniería detectadas en la auditoría técnica antes de ejecutar la recolección masiva de vuelos (Hito 3) y el entrenamiento del modelo (Hito 4).*

#### Bloque A: Correcciones que Afectan la Simulación y Recolección en Gazebo/PX4 — [COMPLETADO]

##### Tarea 1.5.1: Corrección de Firma Semántica (`profile`) y Ampliación de Catálogo en `task_generator.py` y `world_generator.py` — [x] COMPLETADO
- **Problema Detectado:**
  1. `task_generator.py` asigna aleatoriamente `profile: B1_LINEAR` o `B2_BSPLINE` y lo incluye en `get_task_signature()`, pero `waypoint_controller.py` ignora ese campo. Dos tareas con idéntica secuencia de objetos y primitivas pero distinto `profile` reciben firmas distintas, provocando una **fuga de datos silenciosa (*data leakage*)** entre Train y Held-Out.
  2. La primitiva `slalom` carece del parámetro `side` en `DEFAULT_ACTION_CATALOG`, por lo que `waypoint_controller.py` ejecuta el 100% de los slaloms por la derecha (`side="right"`).
  3. El tipo `circular_gate` está implementado en `synthetic_lidar.py`, pero nunca se instancia en `world_generator.py`.
- **Acciones de Implementación:**
  * En `task_generator.py`: fijar `profile = "B1_LINEAR"` y excluirlo de `get_task_signature` hasta que exista un controlador diferenciado para garantizar unicidad semántica real.
  * Expandir `DEFAULT_ACTION_CATALOG` parametrizando `slalom` con `side="left"` y `side="right"` (`("pole", "slalom", {"side": "left"})` y `("pole", "slalom", {"side": "right"})`) para que el modelo aprenda desvíos bilaterales en contexto.
  * Añadir soporte en `world_generator.py` (`_create_circular_gate_element` mediante anillo poligonal de 16 cilindros en el plano $Y\text{-}Z$) y flag `--include-circular-gates` para evaluar generalización a geometrías novedosas.
- **Criterio de Aceptación:** `audit_dataset.py` verifica que no existan dos tareas con la misma secuencia de `(landmark, primitiva, side)` en particiones distintas, y los vuelos de `slalom` incluyen pases por izquierda y derecha.

##### Tarea 1.5.2: Suavizado Cinemático en `waypoint_controller.py` y Corrección de Marco en `tracking_controller.py` — [x] COMPLETADO
- **Problema Detectado:**
  1. En `waypoint_controller.py`, el dron conmuta discretamente entre `pre` $\to$ `center` $\to$ `post` al entrar en una esfera de tolerancia, haciendo que el controlador de posición de PX4 desacelere al llegar a cada punto intermedio y vuelva a acelerar. Según el Apéndice G de *Instant Policy*, las detenciones o cambios bruscos de velocidad a mitad de una maniobra degradan severamente la inferencia.
  2. En `tracking_controller.py` (línea 112), se resta `self.current_pos_ned` (marco NED) directamente de `self.target_gazebo` (marco ENU) sin conversión previa.
- **Acciones de Implementación:**
  * En `base_pilot.py` y `waypoint_controller.py`: habilitar `use_velocity` y vectores de velocidad *feed-forward* (`vel_ff` a `TRANSIT_SPEED_FF = 1.4` m/s en `pre` y `center` de `traverse`, `overfly` y `slalom`, con radios de paso *fly-by* ampliados) y discretizar `orbit` en `ORBIT_ARC_STEPS = 12` puntos de arco suave con velocidad tangencial `ORBIT_SPEED_FF = 1.2` m/s y yaw centrípeto continuo.
  * En `tracking_controller.py`: convertir `self.current_pos_ned` a ENU mediante `self.ned_to_gazebo(self.current_pos_ned)` antes de calcular `math.atan2`.
- **Criterio de Aceptación:** El perfil de velocidad $|v_{xy}(t)|$ registrado en `/fmu/out/vehicle_odometry` durante el cruce de un aro o slalom no cae a cero en los puntos `pre` y `center`.

##### Tarea 1.5.3: Unificación de `PX4_GZ_MODEL_POSE` y Variación de Pose Inicial en `data_collector.py` — [x] COMPLETADO
- **Problema Detectado:**
  1. En `data_collector.py` (línea 613), `"PX4_GZ_MODEL_POSE"` está comentado; en `test_flight.sh` vale `"0,0,0.2,0,0,0"` (valor que `TROUBLESHOOTING.md` advierte que causa explosión física si está la plataforma de salida), y en `TROUBLESHOOTING.md` figura `"0,0,0.8,0,0,0"`.
  2. El dron inicia todos los episodios exactamente en $(0, 0, 1.5, \text{yaw}=0)$, desaprovechando la variación de pose inicial recomendada por *Instant Policy* (Sec. 3.4).
- **Acciones de Implementación:**
  * Unificar las constantes `PX4_MODEL_POSE_GROUND = "0,0,0.15,0,0,0"` y `PX4_MODEL_POSE_PLATFORM = "0,0,0.35,0,0,0"` en `data_collector.py`, `test_flight.sh` y `TROUBLESHOOTING.md`, manteniendo el despegue físico en $(X, Y) = (0, 0)$ para no desalinear el origen NED de EKF2 respecto a Gazebo ENU.
  * Agregar una perturbación aleatoria acotada y determinista (por `ep_seed` en `generate_start_pose`) al waypoint aéreo inicial (`start_pose` previo a `takeoff_exit`): $y_0 \in [-0.8, 0.8]\text{ m}$, $z_0 \in [1.2, 1.8]\text{ m}$, $\psi_0 \in [-0.20, 0.20]\text{ rad}$, guardándola en `waypoints.yaml`.
- **Criterio de Aceptación:** Cada episodio de una misma tarea arranca el circuito desde una posición y orientación inicial ligeramente distinta sin fallos de pre-armado en PX4.

---

#### Bloque B: Correcciones Matemáticas en Observación LiDAR, Grafos `HeteroData` y Empaquetado — [COMPLETADO]

##### Tarea 1.5.4: Marco Egocéntrico Estabilizado por Gravedad (*Horizontal FLU*) y Refinamiento de `synthetic_lidar.py` — [x] COMPLETADO
- **Problema Detectado:**
  1. **Acoplamiento Pitch/Roll en cuadricópteros subactuados:** `px4_odometry_to_gazebo_enu` proyecta la nube de puntos usando la matriz 3D completa $R_{\text{ENU}}^{\text{FLU}}$ (que incluye el *roll* $\phi$ y *pitch* $\theta$ instantáneos del chasis). Al acelerar o frenar a $3\text{ m/s}$, el dron cabecea $\pm 20^\circ$, haciendo que un aro situado a $8\text{ m}$ al frente aparente subir o bajar $\pm 2.7\text{ m}$ en $Z_{\text{FLU}}$, distorsionando la geometría respecto al espacio plano de control $(x, y, z, \text{yaw})$.
  2. **Sobredensidad de `landing_pad`:** La plataforma de aterrizaje se muestrea como una caja de $2.0 \times 2.0 \times 0.1\text{ m}$ ($8\text{ m}^2$ de área total, incluyendo cara inferior), robando gran parte de los 2048 puntos FPS cuando el dron ejecuta la última primitiva a $4\text{–}6\text{ m}$ de ella.
  3. **No-determinismo en pre-submuestreo y colapso en $(0,0,0)$:** `fps_numpy` usa `seed=None` por defecto al reducir nubes de $>4096$ puntos y devuelve `np.zeros((2048, 3))` (el centro de masa del dron) si la nube queda vacía ($N=0$).
- **Acciones de Implementación:**
  * Implementar por defecto la proyección al **marco egocéntrico estabilizado por gravedad (*Horizontal FLU / Gravity-Aligned Frame*)**: extraer el *yaw* $\psi_{\text{ENU}} = \operatorname{atan2}(R_{1,0}, R_{0,0})$ de $R_{\text{ENU}}^{\text{FLU}}$ y rotar la escena exclusivamente con $R_z(\psi_{\text{ENU}})$ (dejando `gravity_aligned: bool = True` como parámetro configurable).
  * En `generate_canonical_landing_pad`: muestrear exclusivamente la cara superior y bordes laterales (excluyendo la cara inferior apoyada en el suelo) con una cuota acotada de $1000$ puntos para no eclipsar a los landmarks principales.
  * Pasar una semilla determinista (`seed = (ep_seed * 10000 + step_idx) % (2**31 - 1)`) a `generate_pointcloud` / `fps_numpy`.
  * En caso de nube vacía ($N=0$), colocar los puntos dummy en un anillo sobre la frontera exterior del rango ($R = s \cdot R_{\text{max}} = 1.0$), nunca en $(0,0,0)$.
- **Criterio de Aceptación:** Suite `pytest` en `test_synthetic_lidar.py` ampliada verificando invarianza de la nube ante inclinaciones puras de *pitch/roll* del dron con `gravity_aligned=True`.

##### Tarea 1.5.5: Rediseño Matemático de `bag_to_heterodata.py` acorde a *Instant Policy* ($SE(3)$, Horizonte $T=8$, Keyframes $L=10$ y 6 Keypoints) y Actualización de `visualize_dataset.py` — [x] COMPLETADO
- **Problema Detectado:**
  1. `bag_to_heterodata.py` asigna como acción objetivo (`data['agent'].y`) la coordenada global en ENU del setpoint discreto publicado por `WaypointController` (una función escalón que apunta a metros de distancia y salta discontinuamente), contradiciendo el Apéndice H de *Instant Policy*.
  2. `data['agent'].x` y `data['agent'].y` están en el marco global ENU mientras que `data['agent'].pos` está en el marco local del dron, y **no se guarda la orientación del dron ($R_{WE}$ / yaw)** ni el estado binario de vuelo $s_t \in \{0, 1\}$, haciendo imposible reconstruir los desplazamientos relativos en $SE(3)$ o los 6 nodos keypoint del dron.
- **Acciones de Implementación:**
  1. **Re-muestreo y Estado Completo del Agente:**
     * Guardar en `data['agent']`:
       - `pos_enu` (y `x`): $(T_{\text{steps}}, 3)$ posición en ENU escalada por $s=0.1$.
       - `yaw_enu`, `rot_enu` y `pose_se3`: $(T_{\text{steps}}, 1)$, $(T_{\text{steps}}, 3, 3)$ y $(T_{\text{steps}}, 4, 4)$ orientación estabilizada $R_z(\psi_t)$ y matriz homogénea $T_{WE}^t$ (más cuaternión completo `q_ned` para auditoría).
       - `flight_state`: $(T_{\text{steps}}, 1)$ estado binario $s_t \in \{0, 1\}$ ($1$ en vuelo activo, $0$ durante descenso final/touchdown sobre la plataforma).
       - `keypoints_body`: $(6, 3)$ coordenadas canónicas de los 6 nodos keypoint del UAV en el marco local (centro, proa $+X$, popa $-X$, babor $+Y$, estribor $-Y$, cenit $+Z$) para construir los nodos del agente $\{p_g^i\}_{i=1}^6$ requeridos por el algoritmo SVD de *Instant Policy* (Sec. 3.2 y 3.3).
  2. **Horizonte de Acciones Relativas ($T = 8$) sobre la Trayectoria Ejecutada:**
     * Para cada instante $t$, calcular la secuencia de $T=8$ transformaciones futuras relativas en el marco local estabilizado del dron a partir de la trayectoria realmente volada (con paso uniforme):
       $$T_{EA}^{(t, k)} = (T_{WE}^t)^{-1} T_{WE}^{t+k} \in SE(3), \quad k = 1, \dots, T$$
     * Almacenar tanto las matrices relativas `data['agent'].action_se3` de forma $(T_{\text{steps}}, 8, 4, 4)$ como la descomposición desacoplada `data['agent'].y` $(\Delta p_{t+k}, \Delta \psi_{t+k}, s_{t+k})$ de forma $(T_{\text{steps}}, 8, 5)$ escalada por $s=0.1$.
     * Guardar el setpoint crudo de PX4 en `data['agent'].setpoint_raw` exclusivamente con fines de diagnóstico del controlador.
  3. **Extracción Geométrica de Keyframes de Contexto ($L = 10$) y Soporte en `visualize_dataset.py`:**
     * Implementar `extract_trajectory_keyframes(pos_enu, yaw_enu, flight_state, num_keyframes=10)` combinando inicio/fin, transición de aterrizaje $s_t$, máximos de curvatura 3D $\kappa(t)$ y $|\dot{\psi}(t)|$, y cobertura por longitud de arco (1D FPS), guardando `data['agent'].keyframe_indices` de forma $(10,)$.
     * Actualizar `visualize_dataset.py` de forma retrocompatible para renderizar el horizonte local $T=8$ en las vistas egocéntricas y los $L=10$ keyframes sobre la trayectoria 3D global.
- **Criterio de Aceptación:** Todo archivo `.pt`/`.npz` generado contiene `scene.pos` $(T_{\text{steps}}, 2048, 3)$, poses $SE(3)$ completas, horizonte de acciones relativas $T=8$ en marco local, estado $s_t$ e índices de $L=10$ keyframes verificados visualmente en `visualize_dataset.py`.

##### Tarea 1.5.6: Sincronización Estructural entre `src/uav_ip/uav_ip/` y `src/uav_ip/scripts/` (Patrón *Thin Wrapper*) — [x] COMPLETADO
- **Contexto y Justificación:**
  * La carpeta `src/uav_ip/uav_ip/` responde al estándar oficial de paquetes `ament_python` de ROS 2 (instalable vía `colcon build` para importar como `from uav_ip.synthetic_lidar import ...`), mientras que `src/uav_ip/scripts/` permite ejecutar scripts directamente por CLI (`python3 src/uav_ip/scripts/...`) o importarlos desde `bag_to_heterodata.py` dentro del virtualenv de ML (`~/venv_ip`) sin tener ROS 2 cargado.
  * Sin embargo, mantener dos copias completas de 713 líneas (`synthetic_lidar.py`) y 517 líneas (`visualize_dataset.py`) introduce riesgo de divergencia silenciosa al aplicar las mejoras de las Tareas 1.5.4 y 1.5.5.
- **Estrategia de Resolución (*Single Source of Truth + Thin Shim*):**
  1. Mantener el código fuente completo **únicamente** dentro del módulo de Python `src/uav_ip/uav_ip/synthetic_lidar.py` y `src/uav_ip/uav_ip/visualize_dataset.py`.
  2. Reemplazar el contenido de `src/uav_ip/scripts/synthetic_lidar.py` y `src/uav_ip/scripts/visualize_dataset.py` por un *thin wrapper* (~25 líneas) que inyecte dinámicamente `src/uav_ip/` en `sys.path`, re-exporte toda la API (`from uav_ip.synthetic_lidar import *`) y ejecute `main()` si `__name__ == "__main__"`.
- **Criterio de Aceptación:** `pytest` y la invocación directa desde `scripts/` y `ros2 run` pasan el 100% de las pruebas sin código duplicado.

---

### Hito 2: Entorno Aislado de ML (`venv_ip`), Aumento de Recuperación y Procesamiento Multiproceso — [COMPLETADO]

*Objetivo: Desacoplar el entorno de Machine Learning respecto de ROS 2 Jazzy (PEP 668), acelerar el postproceso en paralelo e inyectar el aumento de datos correctivo exigido por Instant Policy.*

#### Tarea 2.1: Entorno Virtual Aislado para ML (`~/venv_ip`) — [x] COMPLETADO
- **Acción:**
  * Crear y configurar `~/venv_ip` con `torch`, `torch_geometric`, `rosbags`, `scipy`, `numpy`, `matplotlib`, `pyyaml` mediante el script idempotente `src/uav_ip/scripts/setup_venv_ip.sh` (con soporte automático para entornos sin `ensurepip`).
  * Instalar el paquete `uav_ip` en modo editable sin dependencias de ROS (`pip install --no-build-isolation --no-deps -e ~/ros2_ws/src/uav_ip`) para que `uav_ip` sea nativamente importable en el venv, más auto-detección de `~/venv_ip` en `bag_to_heterodata.py` y `visualize_dataset.py`.
- **Criterio de Aceptación:** `~/venv_ip/bin/python3 -c "import torch, torch_geometric, rosbags, uav_ip; print('OK')"` ejecuta sin errores.

#### Tarea 2.2: Procesamiento por Lotes Multiproceso en `bag_to_heterodata.py` — [x] COMPLETADO
- **Acción:**
  * Añadir `--dataset-dir datasets/` (`discover_dataset_episodes` y `process_dataset_directory`) para descubrir y procesar recursivamente todos los `task_XXXX/episode_YY`.
  * Implementar paralelismo mediante `multiprocessing.Pool` (`--workers N`) y aceleración SIMD en C para `fps_numpy` (`_get_fps_c_kernel`, reduciendo el costo por paso de ~43 ms a ~6 ms) para sintetizar nubes LiDAR, calcular FPS y exportar `.pt` en paralelo en todos los núcleos de CPU.
- **Criterio de Aceptación:** Conversión de 100 episodios en $< 30\text{ segundos}$ en máquina multinúcleo estándar (~5 segundos estimados en 16 núcleos).

#### Tarea 2.3: Aumento de Datos Sintético para Recuperación (*30% Tube Augmentation*) — [x] COMPLETADO
- **Fundamento ([Paper, Apéndices D y E]):**
  * En *Instant Policy*, al 30% de las muestras se les inyectan perturbaciones locales de pose junto con la acción correctiva que devuelve al robot a la trayectoria de referencia, y al 10% se les perturba el estado binario. Sin esto, la política sufre de *covariate shift* en vuelo real cerrado.
- **Acción:**
  * Aprovechando que `SyntheticLidarGenerator` es analítico, implementar en `bag_to_heterodata.py` `compute_tube_augmentation`: para una fracción estratificada $p_{\text{aug}} = 0.30$ de los instantes $t$, muestrear una desviación $\delta p \sim \mathcal{N}(0, \sigma_p^2)$ ($\sigma_p = 0.25\text{ m}$) y $\delta\psi \sim \mathcal{N}(0, \sigma_\psi^2)$ ($\sigma_\psi = 5^\circ$), más inversión del 10% del estado binario de entrada $\tilde{s}_t$.
  * Evaluar `lidar_gen.generate_pointcloud` en la pose perturbada $\tilde{T}_{WE}^t$, calcular las acciones objetivo hacia las poses nominales futuras:
    $$\tilde{T}_{EA}^{(t, k)} = (\tilde{T}_{WE}^t)^{-1} T_{WE}^{t+k}$$
    y almacenarlas en los nodos `data['aug']`, añadiendo visualización dedicada (`--show-aug` / `plot_tube_augmentation_recovery`) en `visualize_dataset.py`.
- **Criterio de Aceptación:** Verificación matemática (`test_tube_augmentation_se3_convergence`) y gráfica (`--show-aug`) de que ante una desviación lateral o vertical respecto al centro de un aro, el flujo de acción objetivo $\tilde{T}_{EA}$ apunta convergentemente de regreso al eje del aro.

---

### Hito 2.5: Blindaje Pre-Campaña contra Falsos Positivos (Revisión Experta) — [EN PROGRESO: FASE 1 Y 1.1 COMPLETADAS]

*Objetivo: reducir la probabilidad de aceptar vuelos con colisión, cruces por fuera del aro, aterrizajes incompletos o grabaciones corruptas, con criterios explícitos y verificables, antes de la recolección masiva. El validador es un filtro conservador bajo hipótesis declaradas, no una certificación física (ver [`JUSTIFICACION_REVISION_TECNICA.md`](JUSTIFICACION_REVISION_TECNICA.md)).*

#### Fase 1: Validador Independiente, Endurecimiento del Piloto y Factibilidad de Transiciones — [x] COMPLETADO (corregido en Fase 1.1)
- [x] **Tarea 2.5.1: Validador de episodio (`src/uav_ip/uav_ip/validate_episode.py`):** estados separados (`pilot_exit_code`, `recording_status`, `mission_status`, `validation_status`), `rejection_reasons` y métricas, verificación geométrica por primitiva y distancia a todos los landmarks.
- [x] **Tarea 2.5.2: Tolerancias del piloto y confirmación de aterrizaje (`waypoint_controller.py`, `base_pilot.py`).**
- [x] **Tarea 2.5.3: Factibilidad de transiciones (`world_generator.py`).**

#### Fase 1.1: Correcciones por la Segunda Revisión (`docs/feedback/respuesta.md` y `docs/feedback/plan_f1.md`) — [x] COMPLETADO (`C1–C10`)
- [x] **CLI, descubrimiento y contrato de configuración/procedencia (`C1 + C2`):**
  * `scripts/validate_episode.py` propaga `sys.exit(main())` con códigos `0` (aceptado), `1` (rechazo) y `2` (error de uso/argumentos mutuamente excluyentes/rutas vacías o inexistentes).
  * `discover_episode_dirs()` descubre todas las carpetas `task_*/episode_*` y reporta `missing_bag` cuando falta `.mcap`.
  * `validate_waypoints_config()` valida `waypoints.yaml` antes de evaluar la misión (`configuration_status`).
  * Preservación de procedencia de `pilot_exit_code` (`pilot_status_source`) sin inventar éxito por defecto (`unknown_pilot_exit_code`) y escritura atómica de `episode_meta.json` con `--update-meta`.
- [x] **Reloj con diagnóstico estructurado (`C3`):**
  * `reconstruct_simulation_clock()` devuelve un `ClockReconstructionResult` con máscaras de reparación, retroceso/duplicado y ambigüedad, `usable_for_validation` y motivos.
  * El Patrón B se marca como ambiguo (`ambiguous_clock_step_pattern_b`) y se rechaza para campaña sin comprimir entregas en ráfaga.
  * El Patrón A preserva el intervalo entre vecinos UNIX válidos (`raw_px4[k+1] - raw_px4[k-1]`).
  * Se eliminó la fabricación de timestamps con `i * 0.01`; relojes ausentes o no finitos provocan rechazo.
- [x] **Aterrizaje sobre ventana temporal común y frescura del piloto (`C4`):**
  * Sustitución de "últimas 25 muestras" por `landing_window_s = 0.50 s` (parámetro de ingeniería inicial a calibrar), evaluando error XY ≤ 0,75 m, z ∈ [0,0; 0,25] m y velocidad ≤ 0,25 m/s en toda la ventana.
  * Persistencia real de `VehicleLandDetected.landed == True` (nunca `ground_contact`) en toda la ventana con al menos 2 mensajes distintos, timestamps finitos `> 0` y vigencia ≤ 1,5 s.
  * `BasePilot` registra `last_pos_recv_time` y `last_land_detected_recv_time`; `WaypointController` exige frescura, posición XY en plataforma, `z ≤ 0.25 m`, `landed == True` y un contador de persistencia continua (`landing_persistence_s = 3.0 s`) que se reinicia si alguna condición cae, saliendo con código `1` propagado al proceso al vencer `landing_timeout_s = 10.0 s`.
- [x] **Lectura de bags sin degradación silenciosa y política de odometría (`C5 + C6`):**
  * `load_episode_telemetry(bag_path, backend=...)` registra `reader_info` (`backend`, `backend_version`, `integrity_checks`, `schema_revision`, `schemas_verified`). Si `rosbags` está instalado y falla al deserializar, propaga el rechazo sin degradar en silencio al parser nativo.
  * `verify_mcap_integrity()` verifica *magic* inicial/final, `Footer`, recorrido completo del stream de registros y `uncompressed_crc` (`zlib.crc32`) en cada `Chunk` sin comprimir; `_parse_mcap_native()` verifica presencia y compatibilidad textual de esquemas de `px4_msgs`, encabezado CDR y los 5 contadores de reset de `VehicleLocalPosition`.
  * `VehicleOdometry` es obligatoria para aceptación de campaña (`missing_required_vehicle_odometry`); `VehicleLocalPosition` queda como diagnóstico o exige sus contadores de reset presentes y verificados (`unverified_local_position_reset_counters`).
- [x] **Geometría compartida, medición del SDF `x500` y cota inferior de discretización (`C7`):**
  * Módulo compartido `src/uav_ip/uav_ip/validation_geometry.py` importado por `world_generator.py`, `waypoint_controller.py` y `validate_episode.py`.
  * `X500_SDF_MEASUREMENT` documenta las medidas trazables de `x500_base/model.sdf` (`UAV_PHYSICAL_RADIUS_XY = 0.25 m`, `UAV_SAFETY_MARGIN_XY = 0.05 m`, `UAV_COLLISION_RADIUS_XY = 0.30 m`, más `X500_FULL_ROTOR_TIP_RADIUS_M = 0.3857 m`).
  * Ventana de `traverse`: `|y| ≤ 0.35 m` y `|z| ≤ 0.40 m` (`square_gate`), `r ≤ 0.3856 m` (`circular_gate`); piloto con `is_traverse_center_reached()` (`|y| ≤ 0.33 m, |z| ≤ 0.38 m` en cuadrada y `r ≤ 0.3656 m` en circular).
  * Cota inferior de discretización $d_{\text{inferior}} = \max(0, d_{\text{muestreada}} - h/2)$ con paso $h = 0,02\text{ m}$ ($h/2 = 0,01\text{ m}$) y geometría compartida 1 a 1 de los 16 cilindros rectos de `circular_gate` (`get_circular_gate_cylinders_spec()`).
- [x] **Secuencia temporal progresiva de primitivas (`C8`):**
  * Cada primitiva $m$ se busca desde el `completion_index` de la primitiva $m-1$, reportando `start_index`, `event_index` y `completion_index` y rechazando ejecuciones prematuras fuera de turno (`primitive_order_violation`).
- [x] **Suite modular de pruebas y trazabilidad documental (`C9 + C10`):**
  * 56 tests en 8 módulos (`test_validation_cli.py`, `test_validation_contract.py`, `test_validation_clock.py`, `test_landing_validation.py`, `test_mcap_reader.py`, `test_validation_geometry.py`, `test_primitive_validation.py`, `test_synthetic_lidar.py`).

#### Fase 2: Integración en la Recolección, Filtrado de Postproceso y Auditoría en 3 Capas — [ ] PENDIENTE
- [ ] **Tarea 2.5.4: Integrar el validador en `data_collector.py`:**
  * Esperar *topics* activos antes de lanzar el piloto e invocar `validate_episode_directory(..., update_meta=True)` al cerrar el bag.
  * **Conservar los intentos fallidos** con sus `rejection_reasons`.
  * Si hay reintentos con semilla derivada, registrarlos como intentos nuevos y no sobrescribir el original.
- [ ] **Tarea 2.5.5: Filtrado y re-muestreo en `bag_to_heterodata.py`:**
  * Ignorar los episodios `rejected` y usar el reloj reconstruido.
  * Interpolar uniformemente en el tiempo (10 Hz) en lugar de diezmar por índice.
  * Verificar la distancia a obstáculos en las poses perturbadas del *Tube Augmentation*.
- [ ] **Tarea 2.5.6: Auditoría en 3 capas (`audit_dataset.py`):**
  * Vuelos crudos: tasas de aceptación y rechazo por causa.
  * Tensores `.pt`: finitud, formas, ortogonalidad $SO(3)$ y horizonte $T=8$.
  * Mínimo de episodios aceptados por tarea antes de incluirla en los *splits*.
- [ ] **Congelar los umbrales del validador** (con versión registrada en `episode_meta.json`) antes de medir el lote de evaluación.

#### Fase 3: Piloto en Escala de Pre-Campaña (10 Tareas $\times$ 5 Episodios = 50 Vuelos) — [ ] PENDIENTE
- [ ] **Tarea 2.5.7: Ejecución y reporte del piloto pre-campaña:**
  * Reportar la **tasa de aceptación por intento original**, por separado de la tasa tras reintentos. Reintentar con otra semilla desplaza la distribución hacia escenarios que el experto resuelve, así que no debe presentarse como éxito sobre los escenarios pedidos.
  * Inspeccionar manualmente una muestra de episodios aceptados y rechazados.
  * Objetivo orientativo: aceptación primaria ≥ 85 %.

---

### Hito 3: Campaña de Recolección del Dataset Completo y Auditoría Formal — [PENDIENTE]

*Objetivo: Generar el corpus definitivo de demostraciones en simulación de alta fidelidad libre de colisiones y fugas de información.*

- [ ] **Tarea 3.1: Campaña de Vuelo en SITL:**
  * Ejecutar `data_collector.py` sobre el catálogo de tareas sin reemplazo (ej. $45\text{–}64$ tareas únicas $\times 10\text{–}15$ episodios = $450\text{–}960$ vuelos en PX4 SITL).
- [ ] **Tarea 3.2: Auditoría y Partición Oficial (`audit_dataset.py`):**
  * Verificar $0$ colisiones semánticas, tasa de éxito de vuelo $> 95\%$, variabilidad espacial 3D intra-tarea y exportar `datasets/splits/{train,val,held_out}_tasks.json`.
- [ ] **Tarea 3.3: Generación Masiva de Grafos `.pt` y Validación Visual:**
  * Procesar todo el dataset con `bag_to_heterodata.py --dataset-dir datasets/` y auditar muestras aleatorias de cada primitiva (`traverse`, `slalom_left`, `slalom_right`, `overfly`, `orbit`) con `visualize_dataset.py --animate`.

---

### Hito 4: Arquitectura de Difusión sobre Grafos Heterogéneos, Entrenamiento y Validación en Lazo Cerrado — [PENDIENTE]

*Objetivo: Entrenar y validar la política de difusión sobre grafos capaz de inferir la tarea en contexto a partir de $N-1$ demostraciones y comandar el cuadricóptero en PX4 SITL.*

#### Tarea 4.1: Encoder Geométrico Local ($\phi_e$) y Escalado de Centroides ($M=16 \to 32$) — [ ] PENDIENTE
- **Acción:**
  * Integrar el encoder PointNet++ (2 capas *Set Abstraction* con embeddings sinusoidales de Fourier) preentrenado como red de ocupación implícita.
  * Evaluar empíricamente el número de centroides muestreados por FPS en la capa SA: comparar $M=16$ (valor original de mesa en *Instant Policy*) frente a $M=32$ centroides para asegurar que múltiples landmarks simultáneos dentro de $R_{\text{max}}=10\text{ m}$ y barras delgadas ($0.1\text{ m}$) conserven suficientes nodos descriptivos por objeto.

#### Tarea 4.2: Construcción del Grafo Heterogéneo Conjunto ($\mathcal{G}_l, \mathcal{G}_c, \mathcal{G}_a$) y *In-Context Sampler* — [ ] PENDIENTE
- **Acción:**
  * Implementar el `InContextGraphDataset` que, en cada paso de entrenamiento, sortee una tarea de `train_tasks.json`, extraiga $N-1$ episodios como contexto ($N \in [2, 4]$, reducidos a sus $L=10$ keyframes) y un instante $t$ del $N$-ésimo episodio como *query*.
  * Construir los tres niveles de subgrafos con aristas direccionales codificadas mediante embeddings sinusoidales escalados ($s=0.1$):
    1. **Subgrafo Local ($\mathcal{G}_l^t$):** Conecta los $M$ nodos de escena con los 6 nodos keypoint del dron $\{p_g^i\}_{i=1}^6$.
    2. **Subgrafo de Contexto ($\mathcal{G}_c$):** Conecta temporalmente los 6 keypoints del dron a lo largo de los $L$ keyframes de cada demo y proyecta aristas de atención cruzada desde los keyframes de contexto hacia los 6 keypoints actuales.
    3. **Subgrafo de Acción ($\mathcal{G}_a$):** Instancia los 6 keypoints imaginados en las $T=8$ poses futuras ruidosas $\mathcal{G}_l^a(T_{EA}^{-1} \times P^t, T_{WE}^t, a_g)$, conectando los nodos actuales con los futuros.

#### Tarea 4.3: Modelo de Difusión Hetero-GNN ($\sigma, \varphi, \psi$), Flujo Desacoplado y Paso SVD — [ ] PENDIENTE
- **Acción:**
  * Implementar las tres redes secuenciales de *Heterogeneous Graph Transformer* ($\sigma \to \varphi \to \psi$, cada una de 2 capas, dimensión oculta 1024, 16 cabezas de 64 dims, LayerNorm y conexiones residuales, según Apéndice C del paper).
  * Parametrizar la adición de ruido en el espacio tangente $\mathfrak{se}(3)$ (`Logmap` / `Expmap`) con normalización estricta al rango $[-1, 1]$ (Apéndices B y E).
  * Entrenar la red para predecir el **flujo desacoplado** por nodo keypoint:
    $$\nabla\hat{p} = \nabla\hat{p}_t + \nabla\hat{p}_r = (t_{EA}^0 - t_{EA}^k) + (R_{EA}^0 p_{kp} - R_{EA}^k p_{kp})$$
  * En inferencia, ejecutar muestreo **DDIM** ($K = 4\text{–}8$ pasos de denoising) extrayendo en cada paso la actualización rígida $T_{k-1, k} \in SE(3)$ (o proyectada al subgrupo plano 4-DoF $(x, y, z, \text{yaw})$) mediante **Descomposición en Valores Singulares (SVD de Arun et al., 1987)**.

#### Tarea 4.4: Nodo ROS 2 de Inferencia en Lazo Cerrado y Suavizado Dinámico (Fases 1 a 3) — [ ] PENDIENTE
- **Acción:**
  * Implementar `instant_policy_pilot.py` (heredando de `BasePilot`): en cada ciclo de control (10–20 Hz), recibe la odometría de PX4, sintetiza la observación LiDAR 360° en el marco *Horizontal FLU*, ejecuta los $K$ pasos DDIM condicionando en las $N-1$ demos de contexto y publica la acción en PX4.
  * Implementar progresión de control:
    - **Fase 1:** Ejecución directa en horizonte deslizante (*receding horizon*) de los primeros $1\text{–}2$ pasos del horizonte $T=8$ predicho.
    - **Fase 2:** Ajuste de curva suave **B-spline** sobre los $T=8$ pasos predichos para emitir setpoints de posición y velocidad continuos a 50 Hz hacia el controlador de PX4.
    - **Fase 3 (Opcional / Avanzada):** Guiado por restricciones cinemáticas y de barrera (*Constraint-Guided Diffusion* / filtro QP) para garantizar márgenes de seguridad frente a postes y marcos de aros.

#### Tarea 4.5: Protocolo de Evaluación Experimental y Estudios de Ablación — [ ] PENDIENTE
- **Experimentos Obligatorios de Validación Científica:**
  1. **Evaluación In-Distribution vs. Held-Out:** Medir tasa de éxito por primitiva y tasa de éxito de circuito completo en `val_tasks.json` (firmas vistas en poses nuevas) y en `held_out_tasks.json` (combinaciones semánticas de primitivas jamás vistas en entrenamiento).
  2. **Test de Ablación de Contexto (*Context Swap Test*):** Sobre un mismo mundo físico en Gazebo, pasar a la red dos contextos distintos (ej. Contexto A demuestra `slalom_left` + `overfly`; Contexto B demuestra `orbit` + `slalom_right`). Verificar que el dron cambie radicalmente su trayectoria obedeciendo al contexto proporcionado sin tocar sus pesos.
  3. **Sensibilidad a Hiperparámetros de Inferencia:** Evaluar tasa de éxito variando el número de demostraciones de contexto $N \in \{1, 2, 3, 4\}$, la longitud de keyframes $L \in \{5, 10, 15\}$ y los pasos de difusión DDIM $K \in \{2, 4, 8\}$.
  4. **Generalización a Geometrías No Vistas:** Evaluar en circuitos con `circular_gate` proporcionando demostraciones con `square_gate` (y viceversa).
