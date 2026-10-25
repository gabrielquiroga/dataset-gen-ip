**Prepararía el cierre de Fase 1.1 como un conjunto acotado de correcciones: rechazar datos incompletos, preservar los errores hasta la CLI y hacer explícitas las condiciones geométricas y temporales de aceptación.** No hace falta rediseñar el sistema ni adelantar toda la Fase 2.

Verifiqué que `main` sigue en [`24c72e9`](https://github.com/gabrielquiroga/dataset-gen-ip/commit/24c72e93fe921c7c67fb491b6f7d4f438c042d3e), la revisión analizada. **No modifiqué archivos ni ejecuté pruebas.** Lo siguiente es una especificación de trabajo manual; los fragmentos de código son cambios puntuales, no un parche completo de todas las tareas.

# 1. Alcance y orden de aplicación

| ID | Corrección | Prioridad |
|---|---|---|
| C1 | Propagar exit codes y detectar datasets vacíos/incompletos | Bloqueante |
| C2 | Validar configuración y procedencia antes de evaluar el vuelo | Bloqueante |
| C3 | No reparar relojes ambiguos como si fueran sincronización confirmada | Bloqueante |
| C4 | Verificar aterrizaje durante una ventana temporal real | Bloqueante |
| C5 | Evitar fallback silencioso ante errores de lectura | Bloqueante |
| C6 | Completar o deshabilitar la validación alternativa por `local_pos` | Bloqueante |
| C7 | Verificar envolvente del UAV y compartir tolerancias | Antes de campaña masiva |
| C8 | Acotar el error geométrico y corregir el modelo circular | Antes de campaña masiva |
| C9 | Verificar finalización secuencial de primitivas | Cierre funcional |
| C10 | Añadir pruebas independientes y actualizar documentación | Obligatorio para cerrar |

**Regla transversal:** si falta evidencia necesaria, devolver rechazo o estado indeterminado explícito; nunca completar los datos con valores que impliquen éxito.

---

# 2. C1 — CLI y descubrimiento de episodios

## Archivos

- `src/uav_ip/scripts/validate_episode.py`
- `src/uav_ip/uav_ip/validate_episode.py`
- Nuevo: `src/uav_ip/test/test_validation_cli.py`

## Cambios

### A. Propagar el retorno del wrapper

Aplicar:

```diff name=src/uav_ip/scripts/validate_episode.py url=https://github.com/gabrielquiroga/dataset-gen-ip/blob/24c72e93fe921c7c67fb491b6f7d4f438c042d3e/src/uav_ip/scripts/validate_episode.py#L20-L24
@@
 if __name__ == "__main__":
-    main()
+    sys.exit(main())
```

### B. Restaurar argumentos mutuamente excluyentes

En `main()`:

- Usar `add_mutually_exclusive_group(required=True)`.
- Incorporar allí `--episode-dir` y `--dataset-dir`.
- Rechazar la presencia simultánea de ambos.

### C. No omitir episodios sin grabación

En modo dataset:

1. Verificar que el directorio existe.
2. Descubrir **todas** las carpetas `task_*/episode_*`.
3. No filtrar previamente según la existencia de un bag.
4. Pasar cada carpeta al validador.
5. Reportar `missing_bag` cuando corresponda.

Los intentos archivados deberán quedar fuera de ese patrón o tener una estructura documentada.

### D. Contrato de códigos de salida

Adoptar un comportamiento explícito:

| Situación | Código |
|---|---:|
| Episodio individual aceptado | 0 |
| Episodio individual rechazado | 1 |
| Dataset con rechazos y `--strict` | 1 |
| Dataset válido y sin rechazos | 0 |
| Dataset con rechazos, sin `--strict` | 0, con resultados de rechazo en el informe |
| Ruta inexistente, dataset vacío o argumentos inválidos | 2 |

El dataset vacío debe fallar **con y sin** `--strict`.

## Criterios de aceptación

- [ ] El wrapper real termina con código 1 ante un episodio rechazado.
- [ ] Un directorio inexistente termina con código 2.
- [ ] Un directorio vacío termina con código 2.
- [ ] Un episodio sin bag aparece en el informe como rechazado.
- [ ] El total reportado coincide con las carpetas de episodios descubiertas.
- [ ] Pasar ambos argumentos produce error.
- [ ] Las pruebas usan `subprocess.run()` sobre el wrapper, no solamente llamadas a `main()`.

---

# 3. C2 — Configuración válida y procedencia conocida

## Archivos

- `src/uav_ip/uav_ip/validate_episode.py`
- Nuevo: `src/uav_ip/test/test_validation_contract.py`

## Cambios

### A. Añadir validación previa de `waypoints.yaml`

Crear una función independiente, por ejemplo `validate_waypoints_config()`, que devuelva errores estructurados.

Para el formato de campaña actual, exigir:

- Objeto raíz de tipo diccionario.
- Lista no vacía de landmarks.
- IDs presentes, no vacíos y únicos.
- Tipos geométricos soportados.
- Coordenadas y orientación finitas.
- Lista no vacía de primitivas.
- Tipos de primitivas soportados.
- Referencias a landmarks existentes.
- Compatibilidad entre primitiva y objeto.
- `side` válido en slalom.
- Plataforma de aterrizaje con coordenadas finitas.

**No convertir un YAML vacío en una misión vacía aceptable.**

Si se mantiene compatibilidad con el formato histórico `hoops`, hacer una normalización explícita y registrada. No usar defaults silenciosos para suplir partes esenciales de una misión nueva.

### B. Separar configuración inválida de misión fallida

Añadir al informe:

- `configuration_status`
- errores con prefijo o categoría estable, como `invalid_configuration`.

Ante configuración inválida, no calcular métricas físicas que aparenten certificar una misión.

### C. No inventar éxito del piloto

Eliminar el default implícito:

`existing_meta.get("status", "success")`

Comportamiento recomendado:

- Si existe `pilot_exit_code`, usarlo.
- Si solo existe un estado histórico conocido, convertirlo mediante una tabla explícita y registrar que fue inferido.
- Si falta información, conservar `pilot_exit_code = None`.
- No aceptar para campaña un episodio con resultado del piloto desconocido.

El modo de diagnóstico puede seguir calculando geometría de un episodio desconocido, pero sin promoverlo a `accepted`.

### D. Preservar el resultado original

Al actualizar metadatos:

- No borrar `pilot_exit_code` ni causas de fallo originales.
- Registrar versión del validador y parámetros efectivos.
- Escribir mediante archivo temporal y reemplazo atómico.
- No sobrescribir silenciosamente un JSON ilegible.

## Criterios de aceptación

- [ ] YAML vacío → rechazo.
- [ ] Falta `landing` → rechazo.
- [ ] IDs duplicados → rechazo.
- [ ] Coordenada `NaN`/infinita → rechazo.
- [ ] Primitiva o `side` desconocidos → rechazo.
- [ ] Referencia inexistente → rechazo.
- [ ] Metadatos ausentes no se convierten en código cero.
- [ ] La revalidación conserva la procedencia del resultado original.

---

# 4. C3 — Reloj: distinguir reparación, ambigüedad y error

## Archivos

- `src/uav_ip/uav_ip/validate_episode.py`
- `src/uav_ip/test/test_validation_clock.py`

## Cambios

### A. Retirar la aceptación automática del patrón B

La coincidencia “retraso grande antes, retraso pequeño después” no basta para distinguir:

- actualización del offset;
- entrega tardía de una muestra vieja;
- pérdida de muestras intermedias.

**Solución mínima para cerrar esta fase:** marcar el patrón B como ambiguo y rechazarlo para aceptación de campaña, conservando los intervalos originales.

No hace falta implementar ahora una reconstrucción avanzada. Es preferible rechazar un episodio dudoso que comprimir un hueco real.

### B. Tratar el patrón A como reparación condicionada

Para una muestra aislada con tiempo de arranque entre timestamps UNIX:

- Verificar finitud y longitud de las series.
- Comprobar también el intervalo entre los vecinos válidos.
- No sustituir ambos intervalos de forma que desaparezca un hueco entre esos vecinos.
- Registrar el índice afectado, valores originales y método de reparación.
- Si la interpretación sigue siendo ambigua, rechazar.

### C. Cambiar el contrato de la función pública

`reconstruct_simulation_clock()` no debería devolver solamente una serie monótona y un contador.

Devolver un resultado estructurado con:

- tiempo crudo;
- tiempo reconstruido, si existe;
- máscara de pasos reparados;
- máscara de retrocesos/duplicados;
- máscara de ambigüedad;
- máximos intervalos originales y resultantes;
- `usable_for_validation`;
- motivos.

Así el futuro conversor no puede reutilizar el reloj reparado perdiendo las señales de invalidez.

### D. No fabricar timestamps ausentes

Eliminar la sustitución por `i * 0.01` en la ruta de aceptación.

Si falta un reloj obligatorio:

- registrar el dato ausente;
- rechazar la validación temporal;
- permitir diagnóstico parcial sin etiquetar como aceptado.

## Prueba de regresión adicional

Este test expresa el caso del patrón B que falta cubrir. Está diseñado para fallar con la reparación automática actual; adaptarlo a la nueva API cuando se implemente:

```python name=src/uav_ip/test/test_validation_clock.py
def test_unix_delivery_backlog_does_not_erase_source_gap():
    base = 1_700_000_000.0
    source = base + np.array([0.000, 0.008, 0.708, 0.716])
    received = base + np.array([0.692, 0.700, 0.716, 0.724])

    reconstructed, _ = reconstruct_simulation_clock(source, received)

    # Una entrega en ráfaga no demuestra una corrección de offset.
    assert np.max(np.diff(reconstructed)) >= 0.60
```

## Criterios de aceptación

- [ ] El hueco de 700 ms no se transforma silenciosamente en 16 ms.
- [ ] Un patrón B sin evidencia adicional no permite aceptar el episodio.
- [ ] Duplicados y retrocesos quedan visibles en el resultado público.
- [ ] Relojes ausentes o no finitos provocan rechazo.
- [ ] La reparación del patrón A preserva el intervalo entre vecinos válidos.
- [ ] Existe un test de `validate_episode_data()` que comprueba el rechazo final, además de tests de funciones auxiliares.

---

# 5. C4 — Aterrizaje sobre una ventana temporal común

## Archivos

- `src/uav_ip/uav_ip/validate_episode.py`
- `src/uav_ip/uav_ip/base_pilot.py`
- `src/uav_ip/uav_ip/waypoint_controller.py`
- Nuevo: `src/uav_ip/test/test_landing_validation.py`

## Cambios

### A. Sustituir “últimas 25 muestras” por duración

Definir `landing_window_s` como parámetro de validación.

Un valor inicial de **0,5 s** puede servir para el piloto de pruebas, pero debe quedar identificado como elección de ingeniería a calibrar, no como garantía física.

La aceptación exige cobertura durante toda esa ventana, no simplemente contar muestras.

### B. Exigir persistencia real del detector

- Al menos dos mensajes distintos y temporalmente válidos.
- Cobertura del intervalo final mediante una regla explícita de vigencia.
- `landed=True` durante la ventana.
- Ningún intervalo sin evidencia superior al máximo permitido.
- `ground_contact` nunca sustituye `landed`.
- Timestamp ausente/cero → dato inválido, no antigüedad cero.

Para alineación inicial puede usarse el tiempo de recepción de ambos tópicos, dejando documentado ese alcance. No mezclar arbitrariamente tiempos de muestra y recepción.

### C. Evaluar todo en esa misma ventana

Comprobar:

- posición dentro de la plataforma;
- velocidad por debajo del límite;
- altura compatible con la pose de reposo del modelo;
- detector de aterrizaje persistente.

La altura del centro del vehículo en reposo debe calibrarse; no equivale necesariamente a la cota superior de la plataforma.

### D. Añadir frescura al piloto

En `BasePilot`:

- guardar cuándo se recibió la última posición;
- guardar cuándo se recibió el último estado de aterrizaje.

En `WaypointController`:

- exigir posición y detector recientes;
- iniciar un contador de persistencia cuando todas las condiciones son verdaderas;
- reiniciarlo si alguna deja de cumplirse;
- mantener el timeout con código 1;
- conservar la propagación de ese código ya corregida.

**No volver a confundir “pasaron tres segundos desde AUTO_LAND” con “lleva tres segundos aterrizado”.**

## Criterios de aceptación

- [ ] Un solo mensaje `landed=True` no prueba persistencia.
- [ ] Dos mensajes antiguos no permiten aceptar.
- [ ] Un `landed=False` dentro de la ventana invalida la persistencia.
- [ ] El mismo vuelo a 50 y 125 Hz usa una duración equivalente.
- [ ] Un paso cercano a la plataforma seguido de reposo afuera se rechaza.
- [ ] Un timeout real del controlador llega al proceso como código 1.
- [ ] Los fixtures contienen una secuencia temporal del detector, no un único flag final.

---

# 6. C5 — Lectura de bags sin degradación silenciosa

## Archivos

- `src/uav_ip/uav_ip/validate_episode.py`
- Nuevo: `src/uav_ip/test/test_mcap_reader.py`

## Cambios

### A. Separar dependencia ausente de archivo inválido

En `load_episode_telemetry()`:

- Si `rosbags` está disponible, utilizarlo.
- Si falla al leer/deserializar, propagar el error como grabación inválida.
- No capturar cualquier excepción para intentar otro lector.

Si conservar el parser manual es necesario, habilitarlo explícitamente como backend de diagnóstico. Para campaña, la opción más sencilla es exigir el lector mantenido.

### B. Registrar el backend

Añadir al informe:

- lector utilizado;
- versión;
- comprobaciones de integridad realizadas;
- esquema o revisión de tipos utilizada.

### C. Si se conserva el parser manual

Exigir:

- esquema presente;
- encoding esperado;
- layout compatible identificado por contenido/hash o revisión admitida;
- rechazo de payload incompleto;
- rechazo de referencias a canales/esquemas desconocidos en tópicos obligatorios.

Un nombre como `px4_msgs/msg/VehicleStatus` no identifica por sí solo todos los layouts históricos posibles.

### D. Aclarar el alcance de integridad

`magic` y `Footer` comprueban cierre básico, no toda la integridad del archivo.

- Verificar recorrido completo.
- Comprobar CRCs cuando existan y el backend lo permita.
- No afirmar que un footer intacto demuestra integridad del contenido.

## Criterios de aceptación

- [ ] Error de deserialización de `rosbags` → rechazo, sin fallback silencioso.
- [ ] Dependencia ausente → error explícito o backend alternativo solicitado.
- [ ] MCAP truncado → rechazo.
- [ ] Corrupción estructural intermedia → rechazo.
- [ ] Esquema incompatible → rechazo.
- [ ] El informe identifica el lector.
- [ ] La prueba comparativa falla si ambos lectores deberían funcionar y uno no funciona.

**Importante:** generar al menos una fixture con un serializador mantenido. Escribir bytes manualmente en los mismos offsets que usa el lector solo comprueba consistencia interna.

---

# 7. C6 — Ruta alternativa de `local_pos`

## Archivo

- `src/uav_ip/uav_ip/validate_episode.py`

## Decisión recomendada para mantener el alcance pequeño

**Exigir `VehicleOdometry` para aceptar episodios de la campaña actual.**

Si solo existe `local_pos`:

- permitir diagnóstico;
- devolver rechazo de grabación por ausencia de odometría requerida;
- no informar que se verificaron resets que no fueron comprobados.

Esto evita ampliar innecesariamente el parser durante el cierre.

### Alternativa, si el fallback es un requisito real

Preservar y verificar los contadores correspondientes a:

- reset horizontal;
- reset vertical;
- reset de velocidad;
- reset de heading.

Comprobar también todos los flags de validez y finitud.

## Criterios de aceptación

- [ ] Un episodio solo con `local_pos` no se acepta como equivalente sin controles adicionales.
- [ ] No aparece `in_flight_estimator_resets=0` como evidencia verificada cuando faltan los contadores.
- [ ] Si se implementa el fallback completo, un reset sintético provoca rechazo.

---

# 8. C7 — Envolvente del UAV y tolerancias compartidas

## Archivos

- Nuevo: `src/uav_ip/uav_ip/validation_geometry.py`
- `src/uav_ip/uav_ip/validate_episode.py`
- `src/uav_ip/uav_ip/waypoint_controller.py`
- `src/uav_ip/scripts/world_generator.py`
- `docs/JUSTIFICACION_REVISION_TECNICA.md`

## Cambios

### A. Medir el modelo efectivamente utilizado

Sobre la revisión fijada de PX4:

- localizar el modelo incluido por `gz_x500`;
- registrar posiciones de rotores;
- dimensiones de chasis y brazos;
- radio de discos de hélices si forman parte de la envolvente elegida;
- altura y desplazamiento del marco corporal respecto del centro geométrico.

Registrar el commit y los archivos fuente de esa medición.

**No reemplazar el valor de 0,23 m por otro estimado sin comprobarlo.**

### B. Definir qué volumen se protege

Elegir explícitamente:

- geometría de colisión simulada;
- o envolvente física conservadora incluyendo rotores.

Son criterios distintos. El dataset debe indicar cuál utiliza.

El chequeo actual de distancia euclídea a un sólido con un único radio corresponde a una **esfera alrededor del punto de referencia**, no a un cilindro XY. Corregir la descripción o implementar la geometría que se quiera representar.

### C. Compartir la definición

Mover dimensiones, márgenes y límites derivados a un módulo ligero sin ROS ni lectura de bags.

El piloto y el validador deben derivar de allí:

- ventana rectangular para aro cuadrado;
- límite radial para aro circular;
- separación a postes;
- restricciones de plataforma que dependan de la envolvente.

No importar el validador completo desde el piloto solamente para obtener constantes.

## Criterios de aceptación

- [ ] Existe una medición trazable del modelo.
- [ ] Tamaño físico y margen de seguridad están separados.
- [ ] Cambiar el radio actualiza las tolerancias derivadas.
- [ ] El piloto distingue ventana rectangular y circular.
- [ ] Un punto válido para la ventana rectangular pero fuera del radio circular no se acepta para el aro circular.
- [ ] Las tolerancias no se calibraron únicamente para conservar los 12 episodios existentes.

---

# 9. C8 — Error de discretización y geometría circular

## Archivos

- `src/uav_ip/uav_ip/validation_geometry.py`
- `src/uav_ip/uav_ip/validate_episode.py`
- `src/uav_ip/scripts/world_generator.py`
- Nuevo: `src/uav_ip/test/test_validation_geometry.py`

## Cambios

### A. Incorporar una cota de discretización

Mantener la subdivisión de segmentos, pero separar:

- distancia mínima muestreada;
- cota inferior de distancia sobre la polilínea.

Si la separación máxima entre puntos es \(h\), usar, para la polilínea:

\[
d_{\mathrm{inferior}}=\max\left(0,\ d_{\mathrm{muestreada}}-\frac{h}{2}\right).
\]

Comparar la cota inferior con el radio requerido. Esta cota cubre el muestreo de la línea recta; no cubre automáticamente la desviación física respecto de esa línea.

### B. Documentar y controlar la curvatura no observada

Opciones:

- rechazar huecos suficientemente grandes como para impedir una cota útil;
- o introducir un margen basado en una cota de aceleración documentada.

Corregir la fórmula documental: bajo las hipótesis apropiadas de interpolación lineal, el término es del orden de \(a_{\max}\Delta t^2/8\), no `v_max·Δt²·a_max/8`.

### C. Usar los cilindros reales del aro circular

Para evitar una nueva aproximación difícil de justificar:

- extraer la construcción de los 16 segmentos a una función compartida;
- evaluar distancia a la unión de esos cilindros finitos;
- hacer que el generador SDF y el validador usen poses y dimensiones idénticas.

Si se prefiere conservar una envolvente de toroide, demostrar que contiene la geometría creada, incluyendo el factor `1.05`. No utilizar la sagita de un polígono diferente como justificación automática.

## Criterios de aceptación

- [ ] Segmento con extremos libres y obstáculo intermedio → rechazo.
- [ ] Caso tangencial cerca del umbral → resultado conservador.
- [ ] La cota incorpora explícitamente el paso de subdivisión.
- [ ] Geometría circular compartida con el SDF.
- [ ] Pruebas en extremos y uniones de cilindros.
- [ ] El informe distingue distancia estimada y margen/cota utilizados.

---

# 10. C9 — Secuencia temporal de primitivas

## Archivos

- `src/uav_ip/uav_ip/validate_episode.py`
- Nuevo: `src/uav_ip/test/test_primitive_validation.py`

## Cambios

### A. Devolver intervalos, no un índice representativo arbitrario

Cada resultado debería incluir:

- `start_index` o tiempo de inicio;
- `completion_index` o tiempo de finalización;
- evento geométrico principal;
- métricas.

Para órbita, la finalización debe ser el instante en que se cumple el barrido exigido, no el punto medio del segmento de proximidad.

### B. Validar progresivamente

- Buscar la primera primitiva en la ventana admisible.
- Buscar la siguiente después de la finalización anterior.
- No seleccionar globalmente “el mejor cruce” de cada objeto sin considerar la secuencia.

Mantener aparte el chequeo global de colisiones sobre todo el vuelo.

### C. Fijar el alcance semántico

Decidir si radios, alturas y orientación orbital son:

- condiciones de aceptación;
- o métricas descriptivas.

No convertir cada métrica en un requisito nuevo, pero sí evitar anunciar como exigido algo que solo se informa.

## Criterios de aceptación

- [ ] Primitivas invertidas → rechazo.
- [ ] La siguiente comienza antes de completar la órbita → rechazo si la tarea exige ejecución secuencial.
- [ ] Un cruce repetido posterior no oculta una secuencia incorrecta.
- [ ] Órbita mirando hacia afuera → rechazo.
- [ ] Trayectorias nominales válidas siguen aceptándose.

---

# 11. C10 — Pruebas y documentación de cierre

## Organización de pruebas

Mantener `test_synthetic_lidar.py` centrado en el sintetizador. Mover gradualmente la validación a los archivos propuestos:

| Archivo | Responsabilidad |
|---|---|
| `test_validation_cli.py` | Comandos reales y códigos de salida |
| `test_validation_contract.py` | Configuraciones y metadatos |
| `test_validation_clock.py` | Tiempos y anomalías |
| `test_landing_validation.py` | Persistencia y sincronización |
| `test_mcap_reader.py` | Lectura e integridad |
| `test_validation_geometry.py` | Distancias y envolventes |
| `test_primitive_validation.py` | Semántica y orden |

### Cambios obligatorios en los tests existentes

- Separar la batería monolítica en casos independientes o parametrizados.
- Generar secuencias reales de mensajes del detector.
- No capturar excepciones de lectores para omitir comparaciones silenciosamente.
- Usar `pytest.skip` únicamente cuando falte una dependencia o artefacto declarado opcional.
- Probar el timeout del controlador y su salida de proceso, no solo pasar `pilot_exit_code=1` al validador.
- Añadir fixtures autocontenidas de bags; no depender exclusivamente de `datasets/`.

## Documentación

### `README.md`

- Documentar códigos de salida.
- Explicar qué exige `--strict`.
- Aclarar qué ocurre con datos ausentes.
- Mantener explícito que la integración automática pertenece a Fase 2.

### `docs/JUSTIFICACION_REVISION_TECNICA.md`

Actualizar:

- patrón B como ambiguo mientras no exista evidencia adicional;
- persistencia de aterrizaje en segundos;
- lector y fallback;
- medición de la envolvente;
- cota de discretización;
- alcance exacto del orden de primitivas.

### `docs/DECISIONES_DISENO.md`

Eliminar contradicciones de tamaños, geometrías y garantías.

### `docs/PLAN_IMPLEMENTACION.md`

Marcar Fase 1.1 como completada solamente después de adjuntar:

- commit evaluado;
- comando de pruebas;
- entorno;
- cantidad de pruebas pasadas, fallidas y omitidas;
- motivos de omisión;
- resultado de revalidación de episodios locales.

**No declarar que “los 12 episodios deben seguir pasando”.** Si nuevos criterios rechazan alguno, investigar y documentar el motivo antes de cambiar los umbrales.

---

# 12. Qué queda fuera de este cierre

Para evitar ampliar indefinidamente el trabajo, dejar en Fase 2:

- integración automática en `data_collector.py`;
- directorios e identidad de reintentos;
- aceptación tras cierre y validación del bag;
- remuestreo definitivo del conversor a 10 Hz;
- filtrado estricto de episodios para entrenamiento;
- auditoría completa de tensores;
- campaña piloto ampliada.

No obstante, **la Fase 1.1 debe entregar una API y CLI fiables para que esa integración no herede aceptaciones falsas**.

---

# 13. Lista final de aceptación

La Fase 1.1 puede considerarse cerrada cuando:

- [ ] Una validación rechazada llega como fallo al proceso llamador.
- [ ] No se omiten episodios sin bag.
- [ ] Un dataset vacío no produce éxito.
- [ ] Configuración o procedencia incompletas no producen aceptación.
- [ ] Ningún patrón temporal ambiguo borra silenciosamente huecos.
- [ ] El aterrizaje se comprueba durante una ventana temporal cubierta por datos.
- [ ] Los errores del lector principal no activan interpretaciones alternativas silenciosas.
- [ ] La ruta admitida verifica realmente los resets que informa.
- [ ] La envolvente del UAV está medida y sus tolerancias son coherentes.
- [ ] La geometría circular corresponde al mundo simulado.
- [ ] El error de muestreo geométrico se contempla explícitamente.
- [ ] Las primitivas se completan en el orden exigido.
- [ ] Las pruebas cubren tanto las funciones como los comandos reales.
- [ ] La documentación distingue comprobaciones, hipótesis y limitaciones.

**Mi recomendación práctica es agruparlo en cuatro commits revisables:** CLI/contrato; tiempo/aterrizaje; lector; geometría/secuencia/documentación. Así cada corrección puede probarse y revertirse sin mezclarla con la integración de la campaña.