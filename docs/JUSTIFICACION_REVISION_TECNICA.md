# Justificación Técnica y Trazabilidad ante la Revisión Experta

*Registro de decisiones sobre `docs/feedback/revision.md`, `docs/feedback/sugerencias.md` (primera revisión) y `docs/feedback/respuesta.md` (segunda revisión, sobre la Fase 1) · Octubre de 2026*

---

## 1. Propósito y regla de redacción

Este documento registra **qué se cambió, por qué, bajo qué hipótesis y qué queda sin resolver**. Lo usamos para defender tanto lo que se hizo como lo que se dejó afuera.

**Regla de redacción (adoptada tras `respuesta.md` §5):** ninguna afirmación de este documento debe ir más allá de lo que el código comprueba y los tests demuestran. En particular:
- El validador es un **filtro conservador bajo hipótesis explícitas**. No es una certificación física.
- "Aceptado" significa que el episodio no violó ninguno de los criterios implementados. **No** significa que se haya demostrado que el vuelo estuvo libre de colisiones en el simulador.
- Cada decisión relevante se documenta en seis partes: **1. Riesgo · 2. Solución elegida · 3. Hipótesis necesarias · 4. Evidencia disponible · 5. Limitación residual · 6. Condición para revisarla**.

---

## 2. Estado actual en una línea por componente

| Componente | Qué hace hoy | Qué **no** hace todavía |
|---|---|---|
| `validation_geometry.py` | Punto único de verdad geométrico (`PROTECTED_VOLUME_MODEL = "sphere_3d_around_base_link"`), mediciones trazables del SDF `x500`/`x500_base` (`UAV_PHYSICAL_RADIUS_XY = 0.25 m`, `UAV_SAFETY_MARGIN_XY = 0.05 m`, `UAV_COLLISION_RADIUS_XY = 0.30 m`), derivación de tolerancias (`derive_validation_tolerances()`) y especificación 1 a 1 de los 16 cilindros de `circular_gate` (`get_circular_gate_cylinders_spec()`). | El radio de `0.25 m` cubre la caja de colisión de `base_link` y los ejes de motores; el barrido completo de las puntas de hélice (`0.3857 m`) se documenta como hipótesis y parámetro revisable antes de la campaña si se modelan contactos de pala. |
| `validate_episode.py` | Valida offline o en línea (`data_collector.py`) `waypoints.yaml`, grabación MCAP (sin degradación silenciosa entre `rosbags` y parser nativo, verificando stream completo y CRC32 de chunks sin comprimir), reloj estructurado con desambiguación analítica de `/fmu/out/timesync_status`, `VehicleOdometry` obligatoria para aceptación y geometría/misión progresiva con cota inferior de discretización $d_{\text{inferior}} = \max(0, d_{\text{muestreada}} - h/2)$. | Integrado con `data_collector.py` (Fase 2 completada); los intentos fallidos se conservan intactos y los reintentos se guardan en `episode_XX_attempt_YY`. |
| `waypoint_controller.py` / `base_pilot.py` | Tolerancias de paso por `center` importadas de `validation_geometry.py` distinguiendo abertura rectangular (`|y| ≤ 0.33 m`, `|z| ≤ 0.38 m`) y circular (`r ≤ 0.3656 m`); aterrizaje con frescura de mensajes, posición XY en pad, `z ≤ 0.25 m`, `landed == True` y persistencia continua (`3.0 s`), saliendo con código `1` propagado al proceso si vence `10.0 s`. | El piloto sigue siendo una máquina de estados cinemática por waypoints; la validación independiente posterior al vuelo sigue siendo indispensable. |
| `world_generator.py` | Usa `get_circular_gate_cylinders_spec()` de `validation_geometry.py` para construir la puerta circular en el SDF y ejecuta el filtro previo de transiciones (`start_pose → pre_1` e interferencia con landmarks de terceros). | No es una garantía de que el vuelo sea factible: es una heurística que descarta escenarios degenerados. |
| Tests | 56 tests en 8 módulos (`test_validation_cli.py`, `test_validation_contract.py`, `test_validation_clock.py`, `test_landing_validation.py`, `test_mcap_reader.py`, `test_validation_geometry.py`, `test_primitive_validation.py`, `test_synthetic_lidar.py`), cubriendo `C1–C9`. | Los tests sobre bags reales se **omiten** (`pytest.skip`) si `datasets/` no está presente. |

---

## 3. Matriz de decisiones (primera y segunda revisión)

| Punto | Clasificación | Decisión | Archivo |
|---|---|---|---|
| Éxito = `pilot.returncode == 0` | Error real crítico | Cinco estados separados (`pilot_exit_code`, `configuration_status`, `recording_status`, `mission_status`, `validation_status`) más `pilot_status_source` y `reader_info` calculados a partir de `waypoints.yaml`, `episode_meta.json` y el `.mcap`. | `validate_episode.py` |
| `cross_track < 1.4 m` en una puerta de ±0,65 m libres | Error real crítico | Geometría compartida en `validation_geometry.py`: `UAV_COLLISION_RADIUS_XY = 0.30 m`; validador exige `|y| ≤ 0.35 m`, `|z| ≤ 0.40 m` (cuadrada) o `r ≤ 0.3856 m` (circular); piloto exige `|y| ≤ 0.33 m`, `|z| ≤ 0.38 m` (cuadrada) o `r ≤ 0.3656 m` (circular). | `validation_geometry.py`, `waypoint_controller.py`, `validate_episode.py` |
| Aterrizaje con temporizador fijo de 4 s | Debilidad real | Exige frescura de mensajes, posición XY en pad, `z ≤ 0.25 m` y `landed == True` con contador de persistencia continua de `3.0 s` (se reinicia si alguna condición cae); si pasan `10.0 s` sin confirmación, termina con código `1` propagado al proceso. | `base_pilot.py`, `waypoint_controller.py` |
| Plugins de contacto de Gazebo | Problema real, solución desproporcionada | Se reemplazaron por un chequeo geométrico conservador fuera de línea con cota inferior de discretización (§5.1). | `validation_geometry.py`, `validate_episode.py` |
| Factibilidad de transiciones | Mejora preventiva | `check_circuit_transition_feasibility` (§4.7). | `world_generator.py` |
| Saltos de reloj PX4 vs `rosbag2` | Hallazgo empírico | `reconstruct_simulation_clock()` devuelve diagnóstico estructurado (`usable_for_validation`, máscaras de reparación/retroceso/ambigüedad); el Patrón B se marca como ambiguo y se rechaza para campaña (§4.5). | `validate_episode.py` |
| *Minimum-snap* / `B2_BSPLINE` | Desproporcionado para esta etapa | Descartado (§5.2). | — |
| `W = 3–4` demos de contexto | Desproporcionado como requisito de recolección | Descartado como requisito; con ≥ 5 episodios aceptados por tarea se puede evaluar `W ∈ {1..4}` (§5.3). | — |

---

## 4. Fase 1.1: correcciones a partir de `respuesta.md` y `plan_f1.md`

La segunda revisión (`respuesta.md`) y el plan quirúrgico de cierre (`plan_f1.md`) identificaron casos donde la Fase 1 todavía permitía aceptaciones o diagnósticos silenciosos. Cada punto se documenta abajo con la estructura de seis partes.

### 4.1 Confirmación de aterrizaje en ventana temporal común y código de salida (`respuesta.md` §3.1, `plan_f1.md` C1/C2/C4)

1. **Riesgo.**
   - Se aceptaba un aterrizaje sin mensajes del detector o con un único flag final, se trataba un timestamp ausente/cero como antigüedad cero (`age = 0.0`), y la ventana terminal se definía por número fijo de muestras (`25 muestras`), acoplando la duración evaluada a la frecuencia de muestreo.
   - En el piloto (`WaypointController`), `elapsed >= 3.0` medía el tiempo desde la entrada a `AUTO_LAND`, no el tiempo continuo cumpliendo las condiciones de aterrizaje, y no se verificaba la frescura de la posición ni del detector ni el error XY al pad.
   - En el CLI (`scripts/validate_episode.py`), no se propagaba `sys.exit(main())`, el descubrimiento por `*.mcap` ocultaba episodios sin bag, y la revalidación sin `pilot_exit_code` usaba `existing_meta.get("status", "success")` fabricando éxito por defecto.
2. **Solución elegida.**
   - **CLI, descubrimiento y procedencia (`C1 + C2`):**
     - `scripts/validate_episode.py` propaga `sys.exit(main())` con contrato de salida `0` (todos los episodios aceptados), `1` (al menos un episodio rechazado) y `2` (error de uso, argumentos mutuamente excluyentes `--episode-dir` / `--dataset-dir`, rutas inexistentes o dataset vacío).
     - `discover_episode_dirs()` descubre todas las carpetas `task_*/episode_*` y reporta `missing_bag` si falta el `.mcap`.
     - `validate_waypoints_config()` valida estructural y semánticamente `waypoints.yaml` antes de evaluar el vuelo (`configuration_status`).
     - Nunca se inventa `pilot_exit_code = 0` cuando falta evidencia (`unknown_pilot_exit_code`); `pilot_status_source` distingue `"explicit_argument"`, `"recorded_exit_code"`, `"recorded_meta"`, `"inferred_from_legacy_status"` y `"unknown"`, y `--update-meta` escribe `episode_meta.json` de forma atómica (`tempfile.mkstemp` + `os.fsync` + `os.replace`) sin sobrescribir JSONs corruptos.
   - **Validador de aterrizaje en ventana temporal común (`C4`):**
     - Sustituye "últimas 25 muestras" por una duración física parametrizable `landing_window_s = 0.50 s` (identificada como elección de ingeniería inicial a calibrar, no garantía física), alineando odometría y detector mediante el reloj de recepción del bag.
     - Exige cobertura completa de odometría en `[t_end - landing_window_s, t_end]` con error XY máximo ≤ `0.75 m`, altura en `[0.0, 0.25] m` y velocidad máxima ≤ `0.25 m/s` en **todas** las muestras de la ventana.
     - Exige persistencia real de `VehicleLandDetected.landed == True` (nunca `ground_contact`): timestamps presentes, finitos y `> 0` (`invalid_land_detector_timestamp` si falta o es cero), al menos 2 mensajes distintos con `landed == True` en la racha final contigua, ningún `landed == False` dentro de la ventana, mensaje `landed == True` vigente al inicio de la ventana (`t_win_start`) con intervalos entre mensajes ≤ `1.5 s` (`LANDING_MAX_DETECTOR_INTERVAL_S`) y antigüedad final ≤ `1.5 s` (`LANDING_MAX_DETECTOR_AGE_S`).
   - **Piloto (`BasePilot` y `WaypointController`):**
     - `BasePilot` registra `last_pos_recv_time` y `last_land_detected_recv_time` al recibir cada mensaje.
     - En `nav_state == 4`, `WaypointController` verifica frescura (`pos_age_s ≤ 0.5 s`, `ld_age_s ≤ 1.5 s`), `xy_err ≤ 0.75 m`, `z ≤ 0.25 m` y `landed_detected == True`. Mantiene un contador de persistencia continua (`landed_persistence_start_time`) que solo avanza mientras **todas** las condiciones se cumplen y se reinicia a `None` si cualquiera deja de cumplirse. Si vence `landing_timeout_s = 10.0 s` antes de acumular `landing_persistence_s = 3.0 s`, termina con código `1` propagado al proceso.
3. **Hipótesis necesarias.** El reloj de recepción del bag (`log_time`) refleja con fidelidad suficiente la alineación entre odometría y `VehicleLandDetected` en el mismo proceso grabador; en PX4 SITL `VehicleLandDetected` se publica a ~1 Hz una vez alcanzado el reposo, por lo que una vigencia máxima de `1.5 s` entre mensajes consecutivos cubre la tasa nominal sin aceptar mensajes obsoletos.
4. **Evidencia disponible.**
   - `test_validation_cli.py` (6 tests en subproceso real) y `test_validation_contract.py` (8 tests de contrato YAML, procedencia y escritura atómica).
   - `test_landing_validation.py` (8 tests): un único `landed=True` rechazado, dos mensajes antiguos rechazados, timestamp cero/ausente rechazado como inválido, `landed=False` dentro de la ventana rechazado, `ground_contact=True` con `landed=False` rechazado, equivalencia de duración entre 50 Hz y 125 Hz, paso cercano al pad seguido de reposo afuera rechazado, y reinicio de persistencia + código de salida `1` en subproceso real de `WaypointController`.
5. **Limitación residual.**
   - La alineación entre odometría y detector en la ventana de aterrizaje usa el tiempo de recepción del bag (`log_time`), no el reloj del estimador.
   - `landing_window_s = 0.50 s` y `LANDING_MAX_FINAL_Z_M = 0.25 m` son parámetros de ingeniería iniciales que deben calibrarse con la altura real del centro del modelo en reposo.
6. **Condición para revisarla.** Durante el piloto pre-campaña (Fase 3), calibrar la altura de reposo del modelo y evaluar si `landing_window_s` debe ampliarse a `≥ 1.0 s`.

### 4.2 Envolvente del UAV, medición del SDF `x500` y módulo compartido `validation_geometry.py` (`respuesta.md` §3.2, `plan_f1.md` C7)

1. **Riesgo.**
   - En la revisión previa, el radio físico (`0.23 m`) era un valor estimado sin medición trazable del SDF del modelo `gz_x500`, las constantes estaban duplicadas entre `world_generator.py`, `waypoint_controller.py` y `validate_episode.py`, y `WaypointController` usaba la misma caja rectangular (`|y| < 0.35 m`, `|z| < 0.40 m`) para `circular_gate`, cuyas esquinas ($\sqrt{0.35^2 + 0.40^2} \approx 0.5315\text{ m}$) salían del radio admisible del anillo circular ($0.3856\text{ m}$).
2. **Solución elegida.**
   - Se creó [`validation_geometry.py`](file:///home/gabrielq/ros2_ws/src/uav_ip/uav_ip/validation_geometry.py) como fuente única de verdad importada por `world_generator.py`, `waypoint_controller.py` y `validate_episode.py`.
   - Se midieron y registraron en `X500_SDF_MEASUREMENT` los archivos `Tools/simulation/gz/models/x500_base/model.sdf` y `Tools/simulation/gz/models/x500/model.sdf` (commit de PX4 `cbc6a6f7abbd06fbb2ff0aed3e57e3e95f88232e`):
     - Caja principal de colisión de `base_link`: `0.353553 × 0.353553 × 0.05 m` $\implies$ semidiagonal XY = `0.2500 m`.
     - Centros de rotores `rotor_0..3` en `(±0.174, ±0.174, 0.06) m` $\implies$ distancia radial al eje del motor = `0.2461 m`.
     - Cilindro de colisión de cada rotor: radio `0.128 m` (pala de longitud `0.1397 m` = 5,5″, diámetro de disco `0.2794 m` = 11″) $\implies$ alcance radial hasta el borde de la caja de colisión del rotor `0.3741 m` y hasta la punta física de la pala `0.3857 m`.
   - Se definió explícitamente el volumen protegido como una **esfera 3D centrada en `base_link`** (`PROTECTED_VOLUME_MODEL = "sphere_3d_around_base_link"`, `PROTECTED_VOLUME_REFERENCE = "x500_base_collision_box_and_motor_hubs"`), adoptando `UAV_PHYSICAL_RADIUS_XY = 0.25 m` (cubre íntegramente la caja de colisión principal `0.2500 m` y los ejes de los 4 motores `0.2461 m`) más `UAV_SAFETY_MARGIN_XY = 0.05 m`, resultando en `UAV_COLLISION_RADIUS_XY = 0.30 m`.
   - Todas las tolerancias del validador y del piloto se derivan mediante `derive_validation_tolerances()`:

     | Constante | Valor | Origen |
     |---|---|---|
     | `UAV_PHYSICAL_RADIUS_XY` | `0.25 m` | Semidiagonal XY de la caja de colisión `base_link` (`0.2500 m`) y ejes de rotores (`0.2461 m`) |
     | `UAV_SAFETY_MARGIN_XY` | `0.05 m` | Margen de seguridad |
     | `UAV_COLLISION_RADIUS_XY` | `0.30 m` | `0.25 + 0.05 m` |
     | Luz libre de `square_gate` | `±0.65 m` (Y) × `±0.70 m` (Z) | Barras en `y = ±0.70 m`, `z = ±0.75 m` de grosor `0.10 m` |
     | `TRAVERSE_MAX_ABS_Y` / `TRAVERSE_MAX_ABS_Z` | `0.35 m` / `0.40 m` | `0.65 - 0.30 m` / `0.70 - 0.30 m` |
     | `PILOT_TRAVERSE_SQUARE_MAX_Y` / `Z` | `0.33 m` / `0.38 m` | Ventana del validador menos buffer del piloto (`0.02 m`) |
     | `TRAVERSE_CIRC_MAX_R` | `0.3856 m` | Apotema interior del polígono de 16 cilindros (`0.75·cos(π/16) - 0.05 = 0.6856 m`) menos `0.30 m` |
     | `PILOT_TRAVERSE_CIRC_MAX_R` | `0.3656 m` | `TRAVERSE_CIRC_MAX_R - 0.02 m` (evaluado radialmente con `is_traverse_center_reached()`) |
     | `SLALOM_MIN_LATERAL` | `0.47 m` | `POLE_RADIUS (0.10) + UAV_COLLISION_RADIUS_XY (0.30) + 0.07 m` |

3. **Hipótesis necesarias.**
   - El modelo de volumen protegido es una esfera 3D de radio `0.30 m` alrededor del origen de `base_link`, invariante a la orientación del vehículo.
   - Esta esfera cubre el chasis/patas de aterrizaje (`0.2500 m`) y los cubos de los motores (`0.2461 m`) con `0.05 m` de margen, pero **no** cubre los últimos `8.6 cm` de las puntas de las palas (`0.3857 m`). Se asume que la ventana del piloto (`0.33 m` en cuadrada, `0.3656 m` en circular) más la verificación de distancia mínima a las barras sobre la trayectoria real mantiene suficiente holgura, y se expone `X500_FULL_ROTOR_TIP_RADIUS_M = 0.385689 m` en `validation_geometry.py` para auditoría o parametrización más estricta.
4. **Evidencia disponible.**
   - `test_validation_geometry.py` (4 tests): verifica las mediciones de `X500_SDF_MEASUREMENT`, la derivación exacta de `derive_validation_tolerances()`, que una esquina `(y=0.32 m, z=0.32 m)` ($r \approx 0.4525\text{ m}$) sea aceptada en `square_gate` pero rechazada por `is_traverse_center_reached("circular_gate", ...)` y por `validate_episode_data()`, y que los 12 episodios de `datasets/` mantengan `min_obstacle_clearance_m ≥ 0.4311 m` (superando tanto `0.30 m` como el radio de punta de pala `0.3857 m`).
5. **Limitación residual.**
   - Con `UAV_COLLISION_RADIUS_XY = 0.30 m`, un paso pegado al borde de la ventana teórica (`|y| = 0.35 m`) dejaría el centro del motor libre pero podría solapar el disco de la hélice (`0.3857 m`) con la barra si el dron vuela plano.
6. **Condición para revisarla.** Si en los vuelos de Fase 2/3 se observan pasos con `min_obstacle_clearance_m < 0.40 m`, elevar `UAV_PHYSICAL_RADIUS_XY` a `0.386 m` en `validation_geometry.py` (lo que recalcula automáticamente todas las ventanas del piloto y del validador).

### 4.3 Colisiones entre muestras, cota inferior de discretización y los 16 cilindros de la puerta circular (`respuesta.md` §3.3, `plan_f1.md` C7)

1. **Riesgo.**
   - Evaluar el despeje únicamente en los puntos muestreados de la polilínea subdividida a paso $h = 0.02\text{ m}$ podía subestimar el acercamiento de un segmento recto a un obstáculo en hasta $h/2 = 0.01\text{ m}$ en el punto medio entre dos muestras consecutivas.
   - Aproximar la `circular_gate` como un toroide analítico ideal menos la sagita en el validador mientras `world_generator.py` instanciaba 16 cilindros rectos orientados producía una discrepancia geométrica entre el obstáculo simulado en Gazebo y el obstáculo auditado.
2. **Solución elegida.**
   - **Cota inferior sobre la polilínea lineal por tramos:** cada segmento entre muestras consecutivas se subdivide a pasos de longitud máxima $h = 0.02\text{ m}$ (`COLLISION_SEGMENT_MAX_STEP_M`). Por desigualdad triangular en $\mathbb{R}^3$, para cualquier punto $p(t)$ sobre el segmento recto entre dos puntos consecutivos de la subdivisión a distancia $\le h$, la distancia euclidiana a cualquier conjunto cerrado $O$ cumple:
     $$d(p(t), O) \ge d_{\text{inferior}} = \max\left(0,\, d_{\text{muestreada}} - \frac{h}{2}\right)$$
     donde $h/2 = 0.01\text{ m}$ (`DISCRETIZATION_MARGIN_M`). El rechazo por colisión se decide comparando $d_{\text{inferior}} < \text{UAV\_COLLISION\_RADIUS\_XY}$ (`0.30 m`), y el reporte expone `min_obstacle_clearance_sampled_m`, `discretization_step_h_m`, `discretization_margin_m` y `min_obstacle_clearance_m` ($= d_{\text{inferior}}$).
   - **Geometría compartida 1 a 1 de los 16 cilindros de `circular_gate`:** `get_circular_gate_cylinders_spec()` en `validation_geometry.py` genera los 16 cilindros rectos (centro local $(0, y_c, z_c)$, eje unitario tangente $(0, -\sin\theta_i, \cos\theta_i)$, longitud `0.3073 m`, radio `0.05 m`) más la columna de soporte inferior. Tanto `world_generator._create_circular_gate_element()` como `compute_landmark_clearance()` consumen exactamente esa especificación, evaluando la distancia exacta punto-a-cilindro finito en 3D contra los 16 cilindros y el soporte.
3. **Hipótesis necesarias.**
   - La cota inferior $d_{\text{inferior}} = \max(0, d_{\text{muestreada}} - h/2)$ es exacta **para la polilínea lineal por tramos** que une las muestras de odometría; entre dos muestras de odometría (a ~125 Hz, con huecos máximos permitidos de `0.15 s` y pasos máximos de `0.35 m`), se asume que la trayectoria física real se desvía de la cuerda lineal en un orden mucho menor que el margen de seguridad (`0.05 m`).
4. **Evidencia disponible.**
   - `test_validation_geometry.py`:
     - Verifica que los 16 `<link name="seg_i">` generados en XML por `world_generator._create_circular_gate_element()` coincidan 1 a 1 en pose, radio y longitud con `get_circular_gate_cylinders_spec()`.
     - Verifica un caso límite donde la distancia muestreada sobre la subdivisión es `0.304 m > 0.30 m`, pero la cota inferior es `0.294 m < 0.30 m`, confirmando que `validate_episode_data()` rechaza el episodio por colisión.
5. **Limitación residual.**
   - $d_{\text{inferior}}$ acota el error de discretización a lo largo de los segmentos rectos entre muestras, pero no acota analíticamente la curvatura de la trayectoria continua real entre dos muestras de odometría.
6. **Condición para revisarla.** Si se reduce la tasa de grabación de odometría o se incrementa la aceleración máxima, sumar al descuento $h/2$ la cota cinemática de desviación respecto de la cuerda ($a_{\max} \Delta t^2 / 8$).

### 4.4 Lectura de MCAP/CDR sin degradación silenciosa y política de odometría (`respuesta.md` §3.4, `plan_f1.md` C5/C6)

1. **Riesgo.**
   - Si `rosbags` estaba instalado y lanzaba una excepción al deserializar un mensaje corrupto o incompatible, `load_episode_telemetry()` capturaba `Exception` y caía silenciosamente al parser nativo `_parse_mcap_native()`, ocultando el fallo del lector primario.
   - `verify_mcap_integrity()` solo comprobaba los bytes iniciales/finales y el registro `Footer`, sin recorrer el stream de registros intermedios ni verificar el CRC32 de los chunks sin comprimir, y el parser nativo ignoraba el texto de definición del esquema y los contadores de reset de `VehicleLocalPosition`.
   - `validate_episode_data()` permitía aceptar un episodio para campaña usando únicamente `VehicleLocalPosition` cuando `VehicleOdometry` estaba vacío, aun sin verificar todos los contadores de reset del estimador.
2. **Solución elegida.**
   - **Prohibición de degradación silenciosa y trazabilidad del lector (`C5`):**
     - `load_episode_telemetry(bag_path, *, backend="auto")` soporta `backend in {"auto", "rosbags", "native"}` y devuelve un `TelemetryBundle` (`dict` con atributo `.reader_info`).
     - En modo `"auto"`, solo se usa `_parse_mcap_native()` cuando `rosbags` **no está instalado** (`ImportError`). Si `rosbags` está disponible y falla al abrir el archivo o deserializar cualquier mensaje, se lanza inmediatamente `ValueError("rosbags_deserialization_error ...")`, provocando el rechazo del episodio con `corrupt_or_unreadable_mcap` sin hacer fallback al parser nativo. En modo `"rosbags"`, la ausencia de la librería lanza `RuntimeError("missing_reader_dependency:rosbags")`.
     - El reporte y `episode_meta.json` registran `reader_info` con `backend` (`"rosbags"` o `"native_cdr"`), `backend_version`, `integrity_checks`, `schema_revision` y `schemas_verified`.
   - **Integridad estructural del stream MCAP y esquemas en el parser nativo (`C5`):**
     - `verify_mcap_integrity()` recorre secuencialmente todos los registros entre el *magic* inicial y el `Footer` (`_verify_mcap_record_stream`), rechazando longitudes truncadas, envolturas de `Footer` duplicadas y verificando `uncompressed_size` y `uncompressed_crc` (`zlib.crc32`) en cada registro `Chunk` (`0x06`) sin comprimir, así como `summary_crc` en el `Footer` cuando es distinto de cero.
     - `_parse_mcap_native()` exige que cada canal tenga un `schema_id != 0` registrado, que el nombre del esquema coincida con `EXPECTED_TOPIC_SCHEMAS`, que el texto del esquema contenga los tokens esperados de la definición (`_verify_schema_definition_compatibility`), que no aparezcan mensajes con `channel_id` desconocido y que el payload CDR tenga encabezado XCDR1 little-endian (`0x00, 0x01`), extrayendo además los 5 contadores de reset de `VehicleLocalPosition` (`xy_reset_counter`, `z_reset_counter`, `vxy_reset_counter`, `vz_reset_counter`, `heading_reset_counter`).
   - **Política de fuente de odometría para campaña (`C6`):**
     - Para aceptar un episodio en campaña (`validation_status == "accepted"`), `/fmu/out/vehicle_odometry` es **obligatorio**. Si `odometry` está vacío, `recording_status` se marca como `"invalid"` con motivo `"missing_required_vehicle_odometry"` (aunque se sigue analizando `local_pos` con fines diagnósticos si está disponible).
     - En el análisis de `local_pos`, si los contadores de reset no están disponibles o no son verificables, nunca se reporta `in_flight_estimator_resets = 0`: se reporta `in_flight_estimator_resets = None`, `estimator_resets_verified = False` y se agrega `"unverified_local_position_reset_counters"`. Si están presentes, se verifican los 5 contadores y cualquier incremento en vuelo rechaza con `"ekf2_estimator_reset_in_flight"`.
3. **Hipótesis necesarias.**
   - Los archivos `.mcap` del pipeline usan serialización CDR little-endian estándar de ROS 2 (`rosbag2`) con las definiciones de `px4_msgs` del workspace. Si los chunks están comprimidos (`zstd`/`lz4`), se requiere `rosbags` para descomprimirlos y verificar sus payloads.
4. **Evidencia disponible.**
   - `test_mcap_reader.py` (5 tests):
     1. Equivalencia campo por campo de `_parse_mcap_native` contra un MCAP real grabado por `rosbag2` (`datasets/task_0001/episode_00/rosbag/rosbag_0.mcap`) y contra `rosbags` cuando está instalado.
     2. Verificación de que un fallo en `rosbags` se propaga como `rosbags_deserialization_error` sin llamar jamás a `_parse_mcap_native`.
     3. Rechazo de truncamiento a mitad de registro, CRC32 inválido en un `Chunk` sin comprimir, canal sin esquema, `channel_id` desconocido, definición de esquema incompatible y encabezado CDR inválido.
     4. Rechazo de campaña cuando falta `VehicleOdometry` (`missing_required_vehicle_odometry`), aun cuando `VehicleLocalPosition` sea válido.
     5. Rechazo cuando `VehicleLocalPosition` carece de contadores de reset (`unverified_local_position_reset_counters`, con `in_flight_estimator_resets is None`) o cuando incrementa `xy_reset_counter` en vuelo (`ekf2_estimator_reset_in_flight`).
5. **Limitación residual.**
   - El parser nativo utiliza offsets fijos verificados para la revisión de `px4_msgs` del repositorio (`px4_msgs_xcdr1_little_endian`) y solo descomprime chunks `"none"`/`""`. No es un deserializador CDR genérico para cualquier versión futura de `px4_msgs`.
6. **Condición para revisarla.** Ejecutar `pytest src/uav_ip/test/test_mcap_reader.py` dentro de `~/venv_ip` (con `rosbags` activo) antes de cualquier campaña o actualización de `px4_msgs`.

### 4.5 Reconstrucción del reloj con diagnóstico estructurado (`respuesta.md` §3.5, `plan_f1.md` C3)

1. **Riesgo.**
   - Reemplazar automáticamente el **Patrón B** ("retraso acumulado $\ge 0.15\text{ s}$ antes, retraso pequeño $[-0.02, +0.05]\text{ s}$ después") por $\Delta t_{\text{bag}}$ confundía una actualización de offset `Timesync` con una entrega en ráfaga tras una congelación o pérdida de muestras de la fuente, borrando huecos reales de hasta $700\text{ ms}$.
   - El **Patrón A** (muestra aislada en tiempo de arranque $< 10^6\text{ s}$ entre muestras UNIX $> 10^8\text{ s}$) reemplazaba ambos intervalos por $\Delta t_{\text{bag}}$ sin verificar si existía un hueco real entre los vecinos UNIX válidos (`raw_px4[k+1] - raw_px4[k-1]`).
   - Cuando faltaba un reloj se fabricaba una serie sintética `i * 0.01`, y los retrocesos o duplicados se reemplazaban por `max(1e-6, dt_bag)` ocultándolos en el eje devuelto por `reconstruct_simulation_clock()`.
2. **Solución elegida.**
   - **Contrato público estructurado (`ClockReconstructionResult`):** `reconstruct_simulation_clock(ts_px4_s, ts_bag_s)` devuelve un diccionario estructurado con `raw_time_s`, `received_time_s`, `reconstructed_time_s`, `repaired_step_mask`, `backstep_or_duplicate_mask`, `ambiguous_step_mask`, `repaired_events`, intervalos máximos/mínimos originales y reconstruidos, `usable_for_validation` y `reasons`.
   - **Retiro de la reparación automática del Patrón B:** el Patrón B se marca en `ambiguous_step_mask`, conserva intacto el intervalo original $\Delta t_{\text{px4}}$ (nunca comprime un hueco en ráfaga) y fija `usable_for_validation = False` con motivo `ambiguous_clock_step_pattern_b`.
   - **Reparación condicionada del Patrón A:** para una muestra aislada en tiempo de arranque $k$ entre vecinos UNIX válidos $k-1$ y $k+1$, se preserva exactamente el intervalo entre vecinos válidos ($\Delta t_{k-1} + \Delta t_k = t_{\text{px4}}[k+1] - t_{\text{px4}}[k-1]$) repartiéndolo proporcionalmente a $(\Delta t_{\text{bag}}[k-1], \Delta t_{\text{bag}}[k])$. Solo se acepta como reparación válida si ambos pasos de recepción están en $(0, 0.10]\text{ s}$, la diferencia entre el span UNIX y el span de recepción es $\le 0.05\text{ s}$ y ningún subpaso supera `MAX_ALLOWED_DT_GAP_S` ($0.15\text{ s}$); de lo contrario se marca como ambiguo (`ambiguous_clock_step_pattern_a`) y se rechaza.
   - **Visibilidad de retrocesos/duplicados y prohibición de fabricar relojes:** los pasos con $\Delta t \le 0$ no se maquillan con `1e-6` y quedan visibles tanto en `reconstructed_time_s` como en `backstep_or_duplicate_mask`. Si falta `timestamp` o `timestamp_sample_us`/`timestamp_us` (o es $\le 0$ o no finito), se rechaza inmediatamente (`missing_or_invalid_bag_timestamps` / `missing_or_invalid_px4_timestamps`) sin fabricar `i * 0.01`.
3. **Hipótesis necesarias.** Es preferible rechazar un episodio con ambigüedad de sincronización (`uxrce_dds_client`) que aceptar en el dataset de entrenamiento una trayectoria con un hueco temporal comprimido.
4. **Evidencia disponible.**
   - `test_validation_clock.py` (6 tests): `test_unix_delivery_backlog_does_not_erase_source_gap` (700 ms en ráfaga no se convierte en 16 ms), visibilidad de duplicados y retrocesos, preservación del intervalo entre vecinos en Patrón A, rechazo de relojes ausentes/no finitos y rechazo de extremo a extremo en `validate_episode_data()`.
   - Al evaluar los 12 episodios históricos de `datasets/task_0000`–`task_0003` con este criterio más estricto, los 12 quedan rechazados en `recording_status` porque contienen 1–2 escalones de offset UNIX de `uxrce_dds_client` de $\sim 0.27\text{–}0.48\text{ s}$ (Patrón B o Patrón A con salto entre vecinos). No se relajaron los umbrales para forzar su aceptación.
5. **Limitación residual y resolución (Fase 2).** En la configuración estándar de `uxrce_dds_client` + PX4 SITL, los reajustes periódicos de offset UNIX durante el vuelo provocaban rechazos conservadores bajo Patrón B porque `VehicleOdometry` no separaba el reloj monotónico del offset UNIX. En la Fase 2 se resolvió formalmente: `data_collector.py` graba `/fmu/out/timesync_status` (`px4_msgs/msg/TimesyncStatus`) y `validate_episode.py` implementa la desambiguación analítica: si un salto en `VehicleOdometry.timestamp` correlaciona en $\pm 0.05\text{ s}$ con un salto de `estimated_offset` de igual magnitud, se repara el reloj (`timesync_offset_adjustment_repaired`) y se acepta para validación (`usable_for_validation = True`).
6. **Condición para revisarla.** Si en vuelos reales de hardware o simulaciones con otra versión de DDS se observan desfases no correlacionados, mantener el rechazo estricto por `ambiguous_clock_step_pattern_b`.

### 4.6 Secuencia temporal progresiva de primitivas y condiciones de validez (`respuesta.md` §3.6, `plan_f1.md` C8)

1. **Riesgo.**
   - Evaluar cada primitiva buscando su primer cruce válido sobre toda la trayectoria `[0, N)` permitía que un episodio con cruces repetidos o en orden invertido (p. ej. `1 → 2 → 1 → 2` o cruzar el landmark 2 antes del landmark 1 y volver a pasar cerca de 2 al final) pareciera tener índices crecientes o espurios sin haber ejecutado la misión como una secuencia progresiva.
   - En `orbit`, el error de guiñada centrípeta se calculaba pero no se exigía, y no se comprobaban los marcos de referencia ni los *flags* de validez.
2. **Solución elegida.**
   - **Búsqueda progresiva con `start_index`, `event_index` y `completion_index` (`C8`):**
     - Cada primitiva $m$ se evalúa únicamente a partir del `completion_index` de la primitiva $m-1$ (`search_start_index = prev_completion_index`), registrando en `primitive_results` los campos `start_index`, `event_index`, `completion_index`, `start_time_s`, `event_time_s` y `completion_time_s`.
     - Además, se inspecciona el prefijo `[0, search_start_index)` de la trayectoria para detectar ejecuciones prematuras fuera de turno: si la primitiva $m$ ya había sido atravesada/ejecutada durante el tramo de una primitiva anterior, el episodio se rechaza con `"primitive_order_violation"` (`premature_execution_before_previous_primitive`).
     - Se exige `start_index[m] >= completion_index[m - 1]` y `event_index[m] > event_index[m - 1]`.
   - **Criterios de `orbit` y marcos de referencia:**
     - En `orbit`: barrido antihorario continuo ≥ `240°` dentro del anillo radial `[1.25, 2.95] m` y error medio de guiñada centrípeta ≤ `45°`, donde `start_index` es la entrada al anillo y `completion_index` es el instante en que se alcanza el barrido de `240°`.
     - Se exige `pose_frame == 1` y `velocity_frame == 1` (NED) en todas las muestras de `VehicleOdometry`, y los *flags* `xy/z/v_xy/v_z_valid` cuando se analiza `VehicleLocalPosition`.
3. **Hipótesis necesarias.** El orden canónico de la misión es el de `x` creciente declarado en `waypoints.yaml`, que comparten el generador, el YAML y el piloto, sin re-visitas intencionales a landmarks posteriores antes de completar los anteriores.
4. **Evidencia disponible.**
   - `test_primitive_validation.py` (5 tests): aceptación de secuencia nominal `1 → 2`, rechazo de orden invertido `2 → 1`, rechazo de cruces repetidos `2 → 1 → 2` (intento de ocultar la inversión inicial con un segundo cruce), rechazo de `orbit` que comienza antes de que termine la primitiva previa o con guiñada hacia afuera, y rechazo de marcos desconocidos (`pose_frame != 1`).
5. **Limitación residual.** El umbral de `240°` en `orbit` se eligió por debajo del arco nominal de `270°` para tolerar la esfera de aceptación del último punto del arco; no se calibró estadísticamente.
6. **Condición para revisarla.** Congelar los umbrales antes de medir el lote de evaluación (Fase 2) y reportar cualquier cambio posterior.

### 4.7 `world_generator.py` (`respuesta.md` §3.7)

1. **Riesgo.** `start_pose` se generaba después de validar el circuito, y los tramos rectos de transición no se comparaban contra landmarks de terceros.
2. **Solución elegida.**
   - `start_pose` se genera primero y se pasa a `check_circuit_transition_feasibility`.
   - Cada tramo `post_i → pre_{i+1}` se muestrea en 14 puntos interiores y se rechaza si se acerca a otro landmark (poste: d_xy < 0,65 m bajo 3,2 m; puerta: d_xy < 0,85 m y |Δz| < 1,0 m).
3. **Hipótesis necesarias.** El piloto vuela aproximadamente en línea recta entre anclas.
4. **Evidencia disponible.** `test_world_generator_transition_feasibility`: un caso factible, un caso de pendiente excesiva y un poste sin visitar sobre el tramo recto.
5. **Limitación residual.** Es un filtro heurístico. No modela el sobrepaso del controlador ni los arcos de `orbit`. La validación real ocurre después del vuelo.
6. **Condición para revisarla.** Si el piloto de Fase 2 muestra rechazos por colisión concentrados en transiciones, agregar un margen dinámico.

---

## 5. Defensa de lo descartado

### 5.1 Por qué no se usaron plugins de contacto de Gazebo

- **Los obstáculos son estáticos y de geometría conocida.** Sus poses y dimensiones salen de `waypoints.yaml` y `validation_geometry.py`, así que la distancia de la trayectoria a cajas, a los 16 cilindros de la puerta circular y a los postes se calcula en forma cerrada en milisegundos con cota inferior de discretización $d_{\text{inferior}} = \max(0, d_{\text{muestreada}} - h/2)$.
- **Un sensor de contacto no detecta fallas semánticas.** No ve que se esquivó la puerta por fuera, que se tomó el flanco equivocado del slalom o que la órbita quedó incompleta. El validador geométrico cubre esas fallas y las colisiones con una sola herramienta.
- **Se evita costo en el simulador.** No hay que agregar sensores ni puentes a `gz_x500`, y el RTF no se ve afectado.

**Lo que esta decisión _no_ ofrece.** No observa contactos reales en el simulador. Depende de que la envolvente esférica adoptada (`0.30 m` sobre el chasis y ejes de motores, §4.2) sea suficiente, de que los obstáculos se modelen como se declaran y de que la trayectoria entre muestras sea aproximadamente rectilínea (§4.3). Un contacto que no viole esas hipótesis pasa desapercibido.

### 5.2 Por qué no se reemplazó el piloto por *minimum-snap* ni se reactivó `B2_BSPLINE`

- **PX4 ya suaviza la referencia.** Recibe posición y velocidad *feed-forward* en `TrajectorySetpoint` y la filtra a través de su control en cascada y de la dinámica del vehículo.
- **Las acciones de entrenamiento salen de la trayectoria volada,** no de los *setpoints* (`bag_to_heterodata.py`).
- **Evitar *data leakage*.** Dos perfiles de control para la misma secuencia semántica volverían a separar tareas equivalentes entre particiones.

**Limitación.** No medimos la suavidad con un criterio formal. El validador registra `p98_acc_mps2` y `rms_jerk_mps3` como métricas descriptivas, sin umbral de rechazo.

### 5.3 Por qué se mantiene `W = 2` en el diseño del modelo

- *Instant Policy* reporta que la ganancia se satura con pocas demostraciones de contexto.
- Recolectar ≥ 5 episodios aceptados por tarea permite muestrear combinaciones de contexto y evaluar `W ∈ {1..4}` sin volver a recolectar.

**Limitación.** La saturación se observó en manipulación. Que se transfiera a UAV es una hipótesis que hay que medir.

### 5.4 Por qué se usa el reloj de PX4 reconstruido y no solo `log_time`

- `log_time` refleja cuándo el proceso grabador recibió el mensaje, con el *jitter* del planificador del sistema operativo.
- `timestamp_sample` refleja el instante de integración de EKF2 en el reloj de simulación.
- Solo se corrigen los patrones verificados de §4.5. Cualquier otra anomalía **se rechaza**, en lugar de repararse en silencio.

---

## 6. Estado de los Componentes y Próximos Pasos

1. **Fase 2 (Hito 2.5) — [COMPLETADO]:**
   - Integración en línea del validador en `data_collector.py` con cierre limpio de bag vía `SIGINT`.
   - Grabación de `/fmu/out/timesync_status` y desambiguación analítica del Patrón B en `validate_episode.py`.
   - Conservación estricta de intentos fallidos y reintentos no destructivos en `episode_XX_attempt_YY`.
   - Congelamiento de versión a `VALIDATOR_VERSION = "1.2.0"` y registro de parámetros en `episode_meta.json`.
   - Remuestreo uniforme a 10 Hz ($\Delta t = 0.1\text{ s}$) con `valid_mask` $(T, 8)$ y Tube Augmentation en esfera euclídea sólida $\|\delta p\|_2 \le 0.60\text{ m}$ con clearance $\ge 0.30\text{ m}$ en `bag_to_heterodata.py`.
   - Auditoría en 3 capas en `audit_dataset.py` (vuelos crudos, tensores .pt, particionado oficial con $\ge 5$ episodios válidos por tarea).
2. **Fase 3: Piloto en Escala de Pre-Campaña (10 Tareas $\times$ 5 Episodios = 50 Vuelos) — [PENDIENTE]:**
   - Ejecutar la campaña piloto preliminar con `data_collector.py`.
   - Reportar la tasa de aceptación por intento original por separado de la tasa post-reintentos.
   - Evaluación de holgura de palas: verificar si `min_obstacle_clearance_m` se mantiene por encima de `X500_FULL_ROTOR_TIP_RADIUS_M = 0.3857 m` (§4.2) o si conviene adoptar directamente ese radio como `UAV_PHYSICAL_RADIUS_XY`.

