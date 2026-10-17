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
| **`task_generator.py`** | `src/uav_ip/scripts/` | **Funcional (Requiere Ajustes)** | Muestreo combinatorio sin reemplazo operativo. **Pendiente:** eliminar `profile` fantasma de la firma semántica, agregar `side` (`left`/`right`) en `slalom` y variante `circular_gate`. |
| **`world_generator.py`** | `src/uav_ip/scripts/` | **Funcional (Requiere Ajustes)** | Inyección dinámica SDF operativa. **Pendiente:** constructor SDF para `circular_gate` y alineación geométrica de `plataforma_llegada`. |
| **`base_pilot.py`** | `src/uav_ip/uav_ip/` | **Completado** | Ciclo de vida ROS 2, pre-flujo de 50 setpoints a 50 Hz para armado Offboard seguro y conversión base NED/ENU. |
| **`waypoint_controller.py`** | `src/uav_ip/uav_ip/` | **Funcional (Requiere Mejora)** | Ejecuta `traverse`, `slalom`, `overfly`, `orbit`. **Pendiente:** suavizar transición entre waypoints para evitar frenazos discretos y admitir variación de pose inicial. |
| **`tracking_controller.py`** | `src/uav_ip/uav_ip/` | **Prototipo (Con Bug)** | Control por velocidad. **Bug identificado:** mezcla coordenadas Gazebo ENU y PX4 NED en el cálculo de `yaw_gazebo` (línea 112). |
| **`data_collector.py`** | `src/uav_ip/scripts/` | **Funcional (Requiere Ajustes)** | Orquestador multi-tarea/episodio robusto. **Pendiente:** descomentar y unificar `PX4_GZ_MODEL_POSE` con `TROUBLESHOOTING.md` y `test_flight.sh`. |
| **`synthetic_lidar.py`** | `src/uav_ip/uav_ip/` y `scripts/` | **Funcional (Requiere Corrección)** | Sintetizador LiDAR 360° ($s=0.1$, FPS 2048). **Pendiente:** proyectar en marco estabilizado por gravedad (*Horizontal FLU*), acotar puntos de `landing_pad`, fijar semilla en `fps_numpy`, tratar nube vacía ($N=0$) y eliminar duplicación de archivo vía *thin wrapper*. |
| **`audit_dataset.py`** | `src/uav_ip/scripts/` | **Completado** | Auditoría universal de firmas semánticas, variabilidad espacial y particionado `train` / `val` / `held_out` sin fugas. |
| **`bag_to_heterodata.py`** | `src/uav_ip/scripts/` | **Requiere Rediseño Crítico** | Actualmente guarda setpoints discretos en marco global ENU como `agent.y` y omite la rotación $R_{WE}$ y el estado binario $s_t$. Debe reescribirse para exportar desplazamientos relativos $T_{EA} \in SE(3)$ a horizonte $T=8$, extraer $L=10$ keyframes y soportar multiproceso. |
| **`visualize_dataset.py`** | `src/uav_ip/uav_ip/` y `scripts/` | **Completado (Sujeto a Refactor)** | Visualizador 3D estático y animador GIF. **Pendiente:** eliminar duplicación de código en `scripts/` vía *thin wrapper* y visualizar vectores de acción relativa $T=8$ y keyframes. |

---

## 3. Plan de Hitos de Implementación (Paso a Paso)

### Hito 0: Pipeline Core E2E y Validación de Vuelo Base — [COMPLETADO]
- [x] Modelo de simulación ligero `gz_x500` (sin cámara física de profundidad), garantizando $\text{RTF} \ge 1.0\times$.
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

### Hito 1.5: Corrección de Bugs Críticos, Fidelidad Matemática $SE(3)$ y Calidad de Demostraciones — [PRIORIDAD INMEDIATA PRE-CAMPAÑA]

*Objetivo: Corregir las discrepancias matemáticas y de ingeniería detectadas en la auditoría técnica antes de ejecutar la recolección masiva de vuelos (Hito 3) y el entrenamiento del modelo (Hito 4).*

#### Bloque A: Correcciones que Afectan la Simulación y Recolección en Gazebo/PX4

##### Tarea 1.5.1: Corrección de Firma Semántica (`profile`) y Ampliación de Catálogo en `task_generator.py` y `world_generator.py` — [ ] PENDIENTE
- **Problema Detectado:**
  1. `task_generator.py` asigna aleatoriamente `profile: B1_LINEAR` o `B2_BSPLINE` y lo incluye en `get_task_signature()`, pero `waypoint_controller.py` ignora ese campo. Dos tareas con idéntica secuencia de objetos y primitivas pero distinto `profile` reciben firmas distintas, provocando una **fuga de datos silenciosa (*data leakage*)** entre Train y Held-Out.
  2. La primitiva `slalom` carece del parámetro `side` en `DEFAULT_ACTION_CATALOG`, por lo que `waypoint_controller.py` ejecuta el 100% de los slaloms por la derecha (`side="right"`).
  3. El tipo `circular_gate` está implementado en `synthetic_lidar.py`, pero nunca se instancia en `world_generator.py`.
- **Acciones de Implementación:**
  * En `task_generator.py`: fijar `profile = "B1_LINEAR"` (o excluirlo de `get_task_signature` hasta que exista un controlador diferenciado) para garantizar unicidad semántica real.
  * Expandir `DEFAULT_ACTION_CATALOG` o parametrizar `slalom` con `side="left"` y `side="right"` (ej. `("pole", "slalom", {"side": "left"})` y `("pole", "slalom", {"side": "right"})`) para que el modelo aprenda desvíos bilaterales en contexto.
  * Añadir soporte opcional en `world_generator.py` (`_create_circular_gate_element` mediante aproximación poligonal o cilindro hueco/toroide en SDF) para poder evaluar generalización a geometrías novedosas en inferencia.
- **Criterio de Aceptación:** `audit_dataset.py` verifica que no existan dos tareas con la misma secuencia de `(landmark, primitiva, side)` en particiones distintas, y los vuelos de `slalom` incluyen pases por izquierda y derecha.

##### Tarea 1.5.2: Suavizado Cinemático en `waypoint_controller.py` y Corrección de Marco en `tracking_controller.py` — [ ] PENDIENTE
- **Problema Detectado:**
  1. En `waypoint_controller.py`, el dron conmuta discretamente entre `pre` $\to$ `center` $\to$ `post` al entrar en una esfera de tolerancia, haciendo que el controlador de posición de PX4 desacelere al llegar a cada punto intermedio y vuelva a acelerar. Según el Apéndice G de *Instant Policy*, las detenciones o cambios bruscos de velocidad a mitad de una maniobra degradan severamente la inferencia.
  2. En `tracking_controller.py` (línea 112), se resta `self.current_pos_ned` (marco NED) directamente de `self.target_gazebo` (marco ENU) sin conversión previa.
- **Acciones de Implementación:**
  * En `waypoint_controller.py`: implementar seguimiento continuo de trayectoria (mediante interpolación suave B-spline / punto guía móvil *carrot-chasing* a velocidad tangencial constante $v_{\text{ref}}$, o aumentando el radio de aceptación de paso *fly-by* en puntos `pre` y `center` acompañado de vector de velocidad feed-forward `msg.velocity` en `TrajectorySetpoint`).
  * En `tracking_controller.py`: convertir `self.current_pos_ned` a ENU (o `self.target_gazebo` a NED) antes de calcular `math.atan2`.
- **Criterio de Aceptación:** El perfil de velocidad $|v_{xy}(t)|$ registrado en `/fmu/out/vehicle_odometry` durante el cruce de un aro o slalom no cae a cero en los puntos `pre` y `center`.

##### Tarea 1.5.3: Unificación de `PX4_GZ_MODEL_POSE` y Variación de Pose Inicial en `data_collector.py` — [ ] PENDIENTE
- **Problema Detectado:**
  1. En `data_collector.py` (línea 613), `"PX4_GZ_MODEL_POSE"` está comentado; en `test_flight.sh` vale `"0,0,0.2,0,0,0"` (valor que `TROUBLESHOOTING.md` advierte que causa explosión física si está la plataforma de salida), y en `TROUBLESHOOTING.md` figura `"0,0,0.8,0,0,0"`.
  2. El dron inicia todos los episodios exactamente en $(0, 0, 1.5, \text{yaw}=0)$, desaprovechando la variación de pose inicial recomendada por *Instant Policy* (Sec. 3.4).
- **Acciones de Implementación:**
  * Unificar la constante `PX4_GZ_MODEL_POSE` en `data_collector.py`, `test_flight.sh` y la documentación.
  * Agregar una perturbación aleatoria acotada y determinista (por `ep_seed`) al waypoint inicial de salida (`takeoff` / `takeoff_exit`): $y_0 \in [-1.0, 1.0]\text{ m}$, $z_0 \in [1.2, 2.0]\text{ m}$, $\psi_0 \in [-0.25, 0.25]\text{ rad}$, guardándola en `waypoints.yaml`.
- **Criterio de Aceptación:** Cada episodio de una misma tarea arranca el circuito desde una posición y orientación inicial ligeramente distinta sin fallos de pre-armado en PX4.

---

#### Bloque B: Correcciones Matemáticas en Observación LiDAR, Grafos `HeteroData` y Empaquetado

##### Tarea 1.5.4: Marco Egocéntrico Estabilizado por Gravedad (*Horizontal FLU*) y Refinamiento de `synthetic_lidar.py` — [ ] PENDIENTE
- **Problema Detectado:**
  1. **Acoplamiento Pitch/Roll en cuadricópteros subactuados:** `px4_odometry_to_gazebo_enu` proyecta la nube de puntos usando la matriz 3D completa $R_{\text{ENU}}^{\text{FLU}}$ (que incluye el *roll* $\phi$ y *pitch* $\theta$ instantáneos del chasis). Al acelerar o frenar a $3\text{ m/s}$, el dron cabecea $\pm 20^\circ$, haciendo que un aro situado a $8\text{ m}$ al frente aparente subir o bajar $\pm 2.7\text{ m}$ en $Z_{\text{FLU}}$, distorsionando la geometría respecto al espacio plano de control $(x, y, z, \text{yaw})$.
  2. **Sobredensidad de `landing_pad`:** La plataforma de aterrizaje se muestrea como una caja de $2.0 \times 2.0 \times 0.1\text{ m}$ ($8\text{ m}^2$ de área total, incluyendo cara inferior), robando gran parte de los 2048 puntos FPS cuando el dron ejecuta la última primitiva a $4\text{–}6\text{ m}$ de ella.
  3. **No-determinismo en pre-submuestreo y colapso en $(0,0,0)$:** `fps_numpy` usa `seed=None` por defecto al reducir nubes de $>4096$ puntos y devuelve `np.zeros((2048, 3))` (el centro de masa del dron) si la nube queda vacía ($N=0$).
- **Acciones de Implementación:**
  * Implementar por defecto la proyección al **marco egocéntrico estabilizado por gravedad (*Horizontal FLU / Gravity-Aligned Frame*)**: extraer el *yaw* $\psi_{\text{ENU}} = \operatorname{atan2}(R_{1,0}, R_{0,0})$ de $R_{\text{ENU}}^{\text{FLU}}$ y rotar la escena exclusivamente con $R_z(\psi_{\text{ENU}})$ (dejando `gravity_aligned: bool = True` como parámetro configurable).
  * En `generate_canonical_landing_pad`: muestrear únicamente la cara superior (o el disco de radio $0.75\text{ m}$ especificado en `ESTRATEGIA_TEORICA.md`) con una cuota máxima acotada de puntos (ej. $\le 1000$ puntos) para no eclipsar a los landmarks principales.
  * Pasar una semilla determinista (`seed = episode_seed * 10000 + step_idx`) a `generate_pointcloud` / `fps_numpy`.
  * En caso de nube vacía ($N=0$), colocar los puntos dummy en la frontera exterior del rango ($R = R_{\text{max}}$) o emitir una máscara de validez `scene_valid_mask`, nunca en $(0,0,0)$.
- **Criterio de Aceptación:** Suite `pytest` en `test_synthetic_lidar.py` ampliada verificando invarianza de la nube ante inclinaciones puras de *pitch/roll* del dron con `gravity_aligned=True`.

##### Tarea 1.5.5: Rediseño Matemático de `bag_to_heterodata.py` acorde a *Instant Policy* ($SE(3)$, Horizonte $T=8$, Keyframes $L=10$ y 6 Keypoints) — [ ] PENDIENTE
- **Problema Detectado:**
  1. `bag_to_heterodata.py` asigna como acción objetivo (`data['agent'].y`) la coordenada global en ENU del setpoint discreto publicado por `WaypointController` (una función escalón que apunta a metros de distancia y salta discontinuamente), contradiciendo el Apéndice H de *Instant Policy*.
  2. `data['agent'].x` y `data['agent'].y` están en el marco global ENU mientras que `data['scene'].pos` está en el marco local del dron, y **no se guarda la orientación del dron ($R_{WE}$ / yaw)** ni el estado binario de vuelo $s_t \in \{0, 1\}$, haciendo imposible reconstruir los desplazamientos relativos en $SE(3)$ o los 6 nodos keypoint del dron.
- **Acciones de Implementación:**
  1. **Re-muestreo y Estado Completo del Agente:**
     * Guardar en `data['agent']`:
       - `pos_enu`: $(T_{\text{steps}}, 3)$ posición en ENU escalada por $s=0.1$.
       - `yaw_enu` y `rot_enu`: $(T_{\text{steps}}, 1)$ y $(T_{\text{steps}}, 3, 3)$ orientación estabilizada $R_z(\psi_t)$ (y cuaternión completo `q_ned` para auditoría).
       - `flight_state`: $(T_{\text{steps}}, 1)$ estado binario $s_t \in \{0, 1\}$ ($1$ en vuelo activo, $0$ durante descenso final/touchdown sobre la plataforma).
       - `keypoints_body`: $(6, 3)$ coordenadas canónicas de los 6 nodos keypoint del UAV en el marco local (centro, proa $+X$, popa $-X$, babor $+Y$, estribor $-Y$, cenit $+Z$) para construir los nodos del agente $\{p_g^i\}_{i=1}^6$ requeridos por el algoritmo SVD de *Instant Policy* (Sec. 3.2 y 3.3).
  2. **Horizonte de Acciones Relativas ($T = 8$) sobre la Trayectoria Ejecutada:**
     * Para cada instante $t$, calcular la secuencia de $T=8$ transformaciones futuras relativas en el marco local estabilizado del dron a partir de la trayectoria realmente volada (con paso uniforme):
       $$T_{EA}^{(t, k)} = (T_{WE}^t)^{-1} T_{WE}^{t+k} \in SE(3), \quad k = 1, \dots, T$$
     * Almacenar tanto las matrices relativas `data['agent'].action_se3` de forma $(T_{\text{steps}}, 8, 4, 4)$ como la descomposición desacoplada $(\Delta p_{t+k}, \Delta \psi_{t+k}, s_{t+k})$ escalada por $s=0.1$.
     * Guardar el setpoint crudo de PX4 en `data['agent'].setpoint_raw` exclusivamente con fines de diagnóstico del controlador.
  3. **Extracción Geométrica de Keyframes de Contexto ($L = 10$):**
     * Implementar la función `extract_trajectory_keyframes(pos_enu, yaw_enu, flight_state, L=10)` que seleccione automáticamente los $L$ índices críticos de una demostración combinando:
       - Inicio ($t=0$) y fin ($t=T_{\text{steps}}-1$) del vuelo.
       - Transición del estado binario de vuelo $s_t$ (inicio del aterrizaje).
       - Máximos locales de curvatura geométrica 3D $\kappa(t)$ y de velocidad angular $|\dot{\psi}(t)|$ (capturando automáticamente los centros de giro en `slalom`, `overfly` y los 4 cuadrantes de `orbit`).
       - Interpolación uniforme por longitud de arco para completar exactamente $L$ keyframes.
     * Almacenar `data['agent'].keyframe_indices` de forma $(L,)$ directamente en el archivo `.pt`.
- **Criterio de Aceptación:** Todo archivo `.pt` generado contiene `scene.pos` $(T_{\text{steps}}, 2048, 3)$, poses $SE(3)$ completas, horizonte de acciones relativas $T=8$ en marco local, estado $s_t$ e índices de $L=10$ keyframes verificados visualmente en `visualize_dataset.py`.

##### Tarea 1.5.6: Sincronización Estructural entre `src/uav_ip/uav_ip/` y `src/uav_ip/scripts/` (Patrón *Thin Wrapper*) — [ ] PENDIENTE
- **Contexto y Justificación:**
  * La carpeta `src/uav_ip/uav_ip/` responde al estándar oficial de paquetes `ament_python` de ROS 2 (instalable vía `colcon build` para importar como `from uav_ip.synthetic_lidar import ...`), mientras que `src/uav_ip/scripts/` permite ejecutar scripts directamente por CLI (`python3 src/uav_ip/scripts/...`) o importarlos desde `bag_to_heterodata.py` dentro del virtualenv de ML (`~/venv_ip`) sin tener ROS 2 cargado.
  * Sin embargo, mantener dos copias completas de 713 líneas (`synthetic_lidar.py`) y 517 líneas (`visualize_dataset.py`) introduce riesgo de divergencia silenciosa al aplicar las mejoras de las Tareas 1.5.4 y 1.5.5.
- **Estrategia de Resolución (*Single Source of Truth + Thin Shim*):**
  1. Mantener el código fuente completo **únicamente** dentro del módulo de Python `src/uav_ip/uav_ip/synthetic_lidar.py` y `src/uav_ip/uav_ip/visualize_dataset.py`.
  2. Reemplazar el contenido de `src/uav_ip/scripts/synthetic_lidar.py` y `src/uav_ip/scripts/visualize_dataset.py` por un *thin wrapper* (~15 líneas) que inyecte dinámicamente `src/uav_ip/` en `sys.path`, re-exporte toda la API (`from uav_ip.synthetic_lidar import *`) y ejecute `main()` si `__name__ == "__main__"`.
  3. De este modo:
     * `ros2 run uav_ip synthetic_lidar` funciona en ROS 2.
     * `from uav_ip.synthetic_lidar import ...` funciona en ROS 2 y en `pytest`.
     * `python3 src/uav_ip/scripts/synthetic_lidar.py` y `import synthetic_lidar` desde `bag_to_heterodata.py` funcionan en cualquier terminal o dentro de `~/venv_ip` sin compilar con `colcon`.
     * Existe **una única copia del código** que mantener.
- **Criterio de Aceptación:** `pytest` y la invocación directa desde `scripts/` y `ros2 run` pasan el 100% de las pruebas sin código duplicado.

---

### Hito 2: Entorno Aislado de ML (`venv_ip`), Aumento de Recuperación y Procesamiento Multiproceso — [PENDIENTE]

*Objetivo: Desacoplar el entorno de Machine Learning respecto de ROS 2 Jazzy (PEP 668), acelerar el postproceso en paralelo e inyectar el aumento de datos correctivo exigido por Instant Policy.*

#### Tarea 2.1: Entorno Virtual Aislado para ML (`~/venv_ip`) — [ ] PENDIENTE
- **Acción:**
  * Crear y configurar `~/venv_ip` con `torch`, `torch_geometric`, `rosbags`, `scipy`, `numpy`, `matplotlib`, `pyyaml`.
  * Instalar el paquete `uav_ip` en modo editable sin dependencias de ROS (`pip install --no-build-isolation --no-deps -e ~/ros2_ws/src/uav_ip`) para que `uav_ip` sea nativamente importable en el venv.
- **Criterio de Aceptación:** `~/venv_ip/bin/python3 -c "import torch, torch_geometric, rosbags, uav_ip; print('OK')"` ejecuta sin errores.

#### Tarea 2.2: Procesamiento por Lotes Multiproceso en `bag_to_heterodata.py` — [ ] PENDIENTE
- **Acción:**
  * Añadir `--dataset-dir datasets/` para descubrir y procesar recursivamente todos los `task_XXXX/episode_YY`.
  * Implementar paralelismo mediante `multiprocessing.Pool` (`--workers N`) para sintetizar nubes LiDAR, calcular FPS y exportar `.pt` en paralelo en todos los núcleos de CPU.
- **Criterio de Aceptación:** Conversión de 100 episodios en $< 30\text{ segundos}$ en máquina multinúcleo estándar.

#### Tarea 2.3: Aumento de Datos Sintético para Recuperación (*30% Tube Augmentation*) — [ ] PENDIENTE
- **Fundamento ([Paper, Apéndices D y E]):**
  * En *Instant Policy*, al 30% de las muestras se les inyectan perturbaciones locales de pose junto con la acción correctiva que devuelve al robot a la trayectoria de referencia, y al 10% se les perturba el estado binario. Sin esto, la política sufre de *covariate shift* en vuelo real cerrado.
- **Acción:**
  * Aprovechando que `SyntheticLidarGenerator` es analítico, implementar en `bag_to_heterodata.py` (o al vuelo en el *Dataset* de PyTorch) un generador de perturbaciones: para una fracción $p_{\text{aug}} = 0.30$ de los instantes $t$, muestrear una desviación $\delta p \sim \mathcal{N}(0, \sigma_p^2)$ ($\sigma_p \approx 0.25\text{ m}$) y $\delta\psi \sim \mathcal{N}(0, \sigma_\psi^2)$ ($\sigma_\psi \approx 5^\circ$).
  * Evaluar `lidar_gen.generate_pointcloud` en la pose perturbada $\tilde{T}_{WE}^t$, y calcular las acciones objetivo hacia las poses nominales futuras:
    $$\tilde{T}_{EA}^{(t, k)} = (\tilde{T}_{WE}^t)^{-1} T_{WE}^{t+k}$$
- **Criterio de Aceptación:** Verificación gráfica de que ante una desviación lateral o vertical respecto al centro de un aro, el flujo de acción objetivo $\tilde{T}_{EA}$ apunta convergentemente de regreso al eje del aro.

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
