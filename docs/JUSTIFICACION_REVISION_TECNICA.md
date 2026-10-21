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
| `validate_episode.py` | Valida offline los episodios grabados (CLI `scripts/validate_episode.py`) y, con `--update-meta`, escribe los resultados en `episode_meta.json`. | `data_collector.py` sigue marcando el éxito por `returncode` del piloto. La integración cierre → lectura → validación → aceptación es **Fase 2**. |
| `waypoint_controller.py` | Tolerancias de paso por `center` dentro de la ventana segura; el aterrizaje exige `landed == True` del detector de PX4 y termina con código 1 si se vence el plazo. | No verifica la posición XY de aterrizaje ni la antigüedad del mensaje del detector; eso queda a cargo del validador. |
| `world_generator.py` | Filtro previo de transiciones (incluye el tramo `start_pose → pre_1` y la interferencia de los tramos rectos con landmarks de terceros). | No es una garantía de que el vuelo sea factible: es una heurística que descarta escenarios degenerados. |
| Tests | 17 tests (`test_synthetic_lidar.py`, `test_validation_clock.py`), incluidos los 9 falsos positivos de `respuesta.md` §6.D y un MCAP binario sintético. | Los tests sobre bags reales se **omiten** (`pytest.skip`) si `datasets/` no está presente. |

---

## 3. Matriz de decisiones (primera revisión)

| Punto | Clasificación | Decisión | Archivo |
|---|---|---|---|
| Éxito = `pilot.returncode == 0` | Error real crítico | Cuatro estados separados (`pilot_exit_code`, `recording_status`, `mission_status`, `validation_status`) calculados a partir del `.mcap` y de `waypoints.yaml`. | `validate_episode.py` |
| `cross_track < 1.4 m` en una puerta de ±0,65 m libres | Error real crítico | Paso por `center`: `cross_track < 0.35 m`, `alt_err < 0.40 m`. El validador exige además el cruce por el plano de la puerta dentro de `|y| ≤ 0.37 m`, `|z| ≤ 0.42 m`. | `waypoint_controller.py`, `validate_episode.py` |
| Aterrizaje con temporizador fijo de 4 s | Debilidad real | Exige `landed == True` y `z ≤ 0.25 m` durante al menos 3 s; si pasan 10 s sin confirmación, termina con código 1 (que se propaga al proceso). | `waypoint_controller.py` |
| Plugins de contacto de Gazebo | Problema real, solución desproporcionada | Se reemplazaron por un chequeo geométrico conservador fuera de línea (§5.1). | `validate_episode.py` |
| Factibilidad de transiciones | Mejora preventiva | `check_circuit_transition_feasibility` (§4.7). | `world_generator.py` |
| Saltos de reloj PX4 vs `rosbag2` | Hallazgo empírico | Solo se reparan los patrones verificados de `Timesync`; cualquier otro retroceso provoca el rechazo (§4.5). | `validate_episode.py` |
| *Minimum-snap* / `B2_BSPLINE` | Desproporcionado para esta etapa | Descartado (§5.2). | — |
| `W = 3–4` demos de contexto | Desproporcionado como requisito de recolección | Descartado como requisito; con ≥ 5 episodios aceptados por tarea se puede evaluar `W ∈ {1..4}` (§5.3). | — |

---

## 4. Fase 1.1: correcciones a partir de `respuesta.md`

La segunda revisión encontró que la Fase 1 dejaba pasar aceptaciones incorrectas. Cada punto se documenta abajo con la estructura de seis partes.

### 4.1 Confirmación de aterrizaje y código de salida (`respuesta.md` §3.1)

1. **Riesgo.** Se aceptaba un aterrizaje sin mensajes del detector, se trataba `ground_contact` como si fuera `landed`, se usaban mínimos históricos (bastaba con pasar cerca de la plataforma una vez) y el piloto podía terminar con código 0 aunque se venciera el plazo.
2. **Solución elegida.**
   - **Validador:**
     - Rechaza si no hay mensajes de `VehicleLandDetected` (`missing_land_detected_messages`).
     - Rechaza si el último mensaje tiene una antigüedad de más de 2,0 s respecto de la última muestra de odometría, medida con el reloj del bag (`stale_land_detector_message`).
     - Exige `landed == True` en los **dos últimos** mensajes del detector; `ground_contact` no alcanza.
     - Sobre la **ventana final** (las últimas 25 muestras de odometría, ≈ 0,2 s a 125 Hz) exige error XY máximo ≤ 0,75 m (plataforma de ±1,0 m), altura máxima ≤ 0,25 m y velocidad máxima ≤ 0,25 m/s.
   - **Piloto:**
     - Considera el aterrizaje cumplido con `landed && z ≤ 0.25 m && t ≥ 3 s`.
     - Si se alcanzan 10 s sin confirmación, lanza `SystemExit(1)`, y `main()` llama a `sys.exit(código)` después de `rclpy.shutdown()`.
3. **Hipótesis necesarias.** El detector de PX4 SITL publica `landed` de forma confiable en `gz_x500`. Una ventana de 25 muestras basta para caracterizar el reposo final.
4. **Evidencia disponible.** Los casos 1–4 de `test_validate_episode_false_positive_battery` cubren: sin detector, contacto sin `landed`, terminó fuera de la plataforma y código de salida ≠ 0. Los 12 episodios locales siguen aceptados.
5. **Limitación residual.**
   - La ventana de reposo es corta (≈ 0,2 s).
   - "Los dos últimos mensajes" no está sincronizado muestra a muestra con la ventana de odometría: solo se exige que sean recientes.
   - El piloto no chequea la posición XY de aterrizaje.
6. **Condición para revisarla.** Si aparecen episodios aceptados con rebote o deriva después del contacto, alargar la ventana a ≥ 1 s y exigir `landed` en todos los mensajes del detector que caigan dentro de ella.

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

### 4.5 Reconstrucción del reloj (`respuesta.md` §3.5 y §6.C)

1. **Riesgo.** La versión anterior reemplazaba por Δt_bag **cualquier** discrepancia de más de 50 ms. Eso podía borrar congelamientos reales de la fuente y "fabricar" un reloj monótono.
2. **Solución elegida.**
   - Se usa `timestamp_sample` (instante de integración de EKF2) en lugar de `timestamp`.
   - Solo se reparan dos patrones verificables de `Timesync`:
     - **(A)** una muestra aislada con tiempo de arranque (< 10⁶ s) entre muestras con tiempo UNIX (> 10⁸ s);
     - **(B)** un escalón que cierra un retraso acumulado de al menos 0,15 s, dejando un retraso final entre −0,02 s y +0,05 s.
   - Los huecos hacia adelante sin explicación **se conservan**, y el chequeo de 0,15 s los rechaza.
   - Los retrocesos sin explicación se regularizan solo para que el eje temporal sea utilizable, pero se cuentan en `unexplained_clock_backsteps` y el episodio se **rechaza**.
3. **Hipótesis necesarias.** Los patrones A y B describen el comportamiento de `uxrce_dds_client` en esta versión de PX4 SITL. Se derivaron de inspeccionar los 12 episodios, no de su código fuente.
4. **Evidencia disponible.**
   - `test_validation_clock.py` incluye los dos tests propuestos por el experto: no borrar un hueco de 700 ms y contabilizar un retroceso.
   - Los 12 episodios locales: Δt mediano de 8,0 ms y ningún retroceso sin explicar.
5. **Limitación residual.** Los umbrales de los patrones son empíricos y pueden no generalizar a otra máquina con más *jitter*. Un patrón nuevo de `Timesync` se rechazaría, lo que es conservador.
6. **Condición para revisarla.** Si la tasa de rechazo por `non_monotonic_or_stalled_simulation_clock` o `telemetry_dropout_gap` es significativa en el piloto de Fase 2, inspeccionar los casos antes de ampliar los patrones.

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
