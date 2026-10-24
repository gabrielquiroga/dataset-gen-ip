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
| `validate_episode.py` | Valida offline la configuración (`waypoints.yaml`), grabación y misión (CLI `scripts/validate_episode.py` con códigos `0`, `1`, `2`), descubre carpetas completas de episodios (`missing_bag`), preserva la procedencia de `pilot_exit_code` y escribe `episode_meta.json` de forma atómica con `--update-meta`. | `data_collector.py` sigue marcando el éxito por `returncode` del piloto. La integración cierre → lectura → validación → aceptación es **Fase 2**. |
| `waypoint_controller.py` / `base_pilot.py` | Tolerancias de paso por `center` dentro de la ventana segura; el aterrizaje verifica frescura de posición y detector, posición XY en plataforma, `z ≤ 0.25 m`, `landed == True` y un contador de persistencia continua (`landing_persistence_s = 3.0 s`), terminando con código `1` propagado al proceso si vence `landing_timeout_s = 10.0 s`. | El piloto sigue siendo una máquina de estados cinemática por waypoints; la validación independiente posterior al vuelo sigue siendo indispensable. |
| `world_generator.py` | Filtro previo de transiciones (incluye el tramo `start_pose → pre_1` y la interferencia de los tramos rectos con landmarks de terceros). | No es una garantía de que el vuelo sea factible: es una heurística que descarta escenarios degenerados. |
| Tests | 42 tests en 5 módulos (`test_validation_cli.py`, `test_validation_contract.py`, `test_validation_clock.py`, `test_landing_validation.py`, `test_synthetic_lidar.py`), incluidos los contratos CLI/metadatos (`C1–C2`), reloj estructurado (`C3`) y ventana temporal de aterrizaje (`C4`). | Los tests sobre bags reales se **omiten** (`pytest.skip`) si `datasets/` no está presente. |

---

## 3. Matriz de decisiones (primera revisión)

| Punto | Clasificación | Decisión | Archivo |
|---|---|---|---|
| Éxito = `pilot.returncode == 0` | Error real crítico | Cinco estados separados (`pilot_exit_code`, `configuration_status`, `recording_status`, `mission_status`, `validation_status`) más `pilot_status_source` calculados a partir de `waypoints.yaml`, `episode_meta.json` y el `.mcap`. | `validate_episode.py` |
| `cross_track < 1.4 m` en una puerta de ±0,65 m libres | Error real crítico | Paso por `center`: `cross_track < 0.35 m`, `alt_err < 0.40 m`. El validador exige además el cruce por el plano de la puerta dentro de `|y| ≤ 0.37 m`, `|z| ≤ 0.42 m`. | `waypoint_controller.py`, `validate_episode.py` |
| Aterrizaje con temporizador fijo de 4 s | Debilidad real | Exige frescura de mensajes, posición XY en pad, `z ≤ 0.25 m` y `landed == True` con contador de persistencia continua de `3.0 s` (se reinicia si alguna condición cae); si pasan `10.0 s` sin confirmación, termina con código `1` propagado al proceso. | `base_pilot.py`, `waypoint_controller.py` |
| Plugins de contacto de Gazebo | Problema real, solución desproporcionada | Se reemplazaron por un chequeo geométrico conservador fuera de línea (§5.1). | `validate_episode.py` |
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
     - Nunca se inventa `pilot_exit_code = 0` cuando falta evidencia (`unknown_pilot_exit_code`); `pilot_status_source` distingue `"explicit_argument"`, `"recorded_meta"`, `"inferred_from_legacy_status"` y `"unknown"`, y `--update-meta` escribe `episode_meta.json` de forma atómica (`tempfile.mkstemp` + `os.fsync` + `os.replace`) sin sobrescribir JSONs corruptos.
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

### 4.2 Envolvente del UAV y geometría de las puertas (`respuesta.md` §3.2)

1. **Riesgo.** El código usaba 0,18 m y la documentación decía 0,28 m. Las tolerancias del piloto quedaban fuera de la ventana libre real.
2. **Solución elegida.** Todo se deriva de constantes nombradas en `validate_episode.py`:

   | Constante | Valor |
   |---|---|
   | `UAV_PHYSICAL_RADIUS_XY` | 0,23 m |
   | `UAV_SAFETY_MARGIN_XY` | 0,05 m |
   | `UAV_COLLISION_RADIUS_XY` | 0,28 m |
   | Luz libre de la puerta cuadrada | ±0,65 m (Y) × ±0,70 m (Z) |
   | `TRAVERSE_MAX_ABS_Y` | 0,37 m |
   | `TRAVERSE_MAX_ABS_Z` | 0,42 m |
   | `TRAVERSE_CIRC_MAX_R` | 0,4056 m |

   Los umbrales del piloto (0,35 m / 0,40 m) quedan dentro de esa ventana.
3. **Hipótesis necesarias.** **El radio físico de 0,23 m no fue medido contra `x500_base/model.sdf`.** Además, la envolvente es un cilindro en XY: se usa el mismo radio en Z, lo que es conservador para un cuerpo plano.
4. **Evidencia disponible.** Las constantes son coherentes entre sí y con `world_generator._create_gate_element()`. Las distancias mínimas a obstáculos en los 12 episodios están entre 0,43 m y 0,65 m.
5. **Limitación residual (importante).** Según la especificación pública del Holybro X500 (500 mm entre ejes de motores, hélices de 10″), los discos de las hélices podrían llegar a **≈ 0,38 m** del centro. Si se confirma, 0,23 m subestima la envolvente física y tanto `TRAVERSE_MAX_ABS_Y` como la tolerancia del piloto deberían bajar.
6. **Condición para revisarla.** **Antes de la generación masiva:** medir posiciones de rotores y largo de hélice en el SDF y recalcular. Las 12 distancias mínimas observadas (≥ 0,43 m) seguirían pasando con R = 0,39 m. La ventana de `traverse` (|y| ≤ 0,26 m) habría que volver a verificarla.

### 4.3 Colisiones entre muestras y puerta circular poligonal (`respuesta.md` §3.3)

1. **Riesgo.** Dos muestras discretas libres de colisión pueden tener una barra entre ellas. Además, el anillo "circular" se construye con 16 cilindros rectos, no con un toroide.
2. **Solución elegida.**
   - Cada segmento entre muestras consecutivas se subdivide a pasos de ≤ 0,02 m (`_subdivide_trajectory_segments`).
   - Se rechazan huecos temporales de más de 0,15 s y pasos espaciales de más de 0,35 m.
   - En la puerta circular se resta la sagita del polígono, R·(1 − cos(π/16)) ≈ 0,0144 m.
3. **Hipótesis necesarias.** Entre dos muestras el movimiento es aproximadamente rectilíneo. Con 125 Hz y velocidades ≤ ~2 m/s, la desviación respecto de la cuerda es de orden milimétrico.
4. **Evidencia disponible.** El caso 7 de la batería construye dos vértices a 0,2818 m de una barra (libres) cuyo punto medio queda a 0,255 m (colisión). Sin subdivisión no se detecta; con subdivisión, sí, y el episodio se rechaza.
5. **Limitación residual.** La subdivisión lineal no acota la desviación de una trayectoria curva entre muestras. Es una aproximación justificada por la frecuencia de muestreo, no una cota demostrada.
6. **Condición para revisarla.** Si se reduce la frecuencia de odometría grabada o aumenta la velocidad, agregar una cota explícita de error (v_max·Δt²·a_max/8) al radio de colisión.

### 4.4 Lectura de MCAP/CDR y archivos truncados (`respuesta.md` §3.4)

1. **Riesgo.** El parser propio aceptaba un prefijo válido de un archivo truncado, y había offsets CDR incorrectos (`vx`, `heading`, `at_rest`, `nav_state`).
2. **Solución elegida.**
   - `verify_mcap_integrity` exige el *magic* inicial y final y un registro `Footer` (opcode 0x02, 20 bytes) inmediatamente antes del *magic* final; si falta algo, lanza `ValueError`.
   - Se prioriza `rosbags` como lector. El parser nativo queda como respaldo y verifica nombre de esquema, `encoding == cdr` y el encabezado CDR little-endian.
   - Offsets corregidos según `px4_msgs/msg/*.msg`:

     | Mensaje | Campo | Offset |
     |---|---|---|
     | `VehicleLocalPosition` | `vx` | 56 |
     | `VehicleLocalPosition` | `heading` | 104 |
     | `VehicleLandDetected` | `at_rest` | 23 |
     | `VehicleStatus` | `arming_state` | 28 |
     | `VehicleStatus` | `nav_state` | 45 |
     | `VehicleStatus` | `failsafe` | 62 |
3. **Hipótesis necesarias.** Los `.mcap` los escribe `rosbag2` con su *summary* y *footer* estándar. La versión de `px4_msgs` del workspace coincide con la del firmware.
4. **Evidencia disponible.**
   - `test_mcap_native_cdr_offsets_and_footer_verification`: un MCAP binario sintético con valores conocidos en cada offset.
   - Caso 9 de la batería: un MCAP truncado se rechaza tanto en `verify_mcap_integrity` como en `load_episode_telemetry`.
   - `test_real_dataset_mcap_validation`: compara `rosbags` contra el parser nativo, campo por campo, sobre un bag real (si existe).
   - `VehicleLocalPosition.vx` coincide bit a bit con `VehicleOdometry.velocity` en `task_0001/episode_00`.
5. **Limitación residual.**
   - No se verifican CRCs de chunks ni del *summary*.
   - Un archivo dañado en el medio, pero con *footer* intacto, lo detectaría el recorrido de registros del parser nativo, **no** necesariamente `rosbags`.
   - Si `rosbags` falla al deserializar, se vuelve en silencio al parser nativo.
6. **Condición para revisarla.** Si se cambia la versión de PX4/`px4_msgs`, volver a correr la comparación `rosbags` vs nativo antes de grabar.

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
5. **Limitación residual.** En la configuración actual de `uxrce_dds_client` + PX4 SITL, los reajustes de offset UNIX durante el vuelo provocan rechazos conservadores bajo Patrón B porque el mensaje `VehicleOdometry` exportado no separa el reloj monotónico de PX4 del offset UNIX estimado por el puente.
6. **Condición para revisarla.** Si en el piloto de Fase 2/3 los reajustes de `Timesync` de `uxrce_dds_client` siguen provocando rechazos frecuentes en vuelos físicamente limpios, grabar también el tópico `/fmu/out/timesync_status` para verificar analíticamente el cambio de offset antes de reparar un escalón.

### 4.6 Condiciones calculadas que antes no se exigían (`respuesta.md` §3.6)

1. **Riesgo.**
   - El orden de las primitivas no se verificaba.
   - En `orbit`, el error de guiñada centrípeta se calculaba pero no se exigía.
   - No se chequeaban los marcos de referencia ni los *flags* de validez.
2. **Solución elegida.**
   - Los eventos de las primitivas deben tener `event_index` estrictamente creciente, **en el orden declarado en `waypoints.yaml`** (`primitive_order_violation`). `export_waypoints_yaml` escribe las primitivas ordenadas por x, que es el mismo orden que ejecuta el piloto.
   - En `orbit`: barrido antihorario ≥ 240° y error medio de guiñada ≤ 45°.
   - `pose_frame == 1` y `velocity_frame == 1` (NED); si se usa `VehicleLocalPosition`, también los *flags* `xy/z/v_xy/v_z_valid`.
3. **Hipótesis necesarias.** El orden canónico de la misión es el de x creciente, que comparten el generador, el YAML y el piloto.
4. **Evidencia disponible.** Los casos 5 (órbita mirando hacia afuera rechazada, mirando hacia adentro aceptada), 6 (orden incorrecto) y 8 (marco desconocido) de la batería.
5. **Limitación residual.** El umbral de 240° se eligió por debajo del arco nominal de 270° para tolerar la esfera de aceptación final. No se calibró estadísticamente.
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

- **Los obstáculos son estáticos y de geometría conocida.** Sus poses y dimensiones salen de `waypoints.yaml` y `world_generator.py`, así que la distancia de la trayectoria a cajas, cilindros y anillo se calcula en forma cerrada en milisegundos.
- **Un sensor de contacto no detecta fallas semánticas.** No ve que se esquivó la puerta por fuera, que se tomó el flanco equivocado del slalom o que la órbita quedó incompleta. El validador geométrico cubre esas fallas y las colisiones con una sola herramienta.
- **Se evita costo en el simulador.** No hay que agregar sensores ni puentes a `gz_x500`, y el RTF no se ve afectado.

**Lo que esta decisión _no_ ofrece.** No observa contactos reales en el simulador. Depende de que la envolvente del UAV sea correcta (§4.2), de que los obstáculos se modelen como se declaran y de que la trayectoria entre muestras sea aproximadamente rectilínea (§4.3). Un contacto que no viole esas hipótesis pasa desapercibido.

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

## 6. Pendientes antes de la generación masiva

1. **Verificar la envolvente del x500 contra el SDF (§4.2)** y recalcular las tolerancias si corresponde.
2. **Fase 2:** integrar el validador en `data_collector.py`, conservar los intentos fallidos, congelar los umbrales y reportar el éxito por intento original (sin contar reintentos con otra semilla como éxito).
3. Considerar una ventana de reposo final más larga (§4.1).
