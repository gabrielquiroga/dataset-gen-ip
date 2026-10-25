**Van por el camino correcto en la elección de prioridades y en varias simplificaciones. Pero la Fase 1 commiteada todavía no cumple algunas de las garantías que le atribuyen. Antes de considerarla terminada, corregiría varios falsos positivos del propio validador.**

Revisé el commit [`ec7f011`](https://github.com/gabrielquiroga/dataset-gen-ip/commit/ec7f011f540c390eb5a54b386be40464b7fe004e), incluyendo el nuevo validador, los cambios del piloto y del generador, las pruebas y la justificación técnica.

**Alcance:** revisé la implementación; no ejecuté los vuelos ni tuve acceso a los 12 rosbags locales. Los resultados experimentales mencionados en la documentación son resultados reportados por ustedes, no reproducidos por mí.

Mi devolución tiene dos partes distintas: **el enfoque pragmático me parece mayormente acertado; la implementación necesita una revisión correctiva acotada.**

---

## 1. En qué estoy de acuerdo con el propietario y su agente

### Mantener percepción privilegiada y planificación de cuatro grados de libertad

Es una delimitación sensata para la primera etapa. No hace falta simular un LiDAR comercial ni resolver transferencia al mundo real para investigar aprendizaje en contexto en simulación.

Lo indispensable es describir correctamente qué se está evaluando.

### Usar geometría analítica en lugar de instrumentar todos los contactos

**Es una alternativa perfectamente válida**, especialmente con obstáculos estáticos y sencillos.

De hecho, mi recomendación anterior admitía explícitamente eventos de contacto **y/o comprobación geométrica del volumen barrido**. No considero obligatorio agregar plugins de contacto.

La salvedad es importante:

> Una comprobación geométrica es suficiente si modela adecuadamente el vehículo, los obstáculos, la trayectoria entre muestras y la incertidumbre de la pose.

La implementación actual todavía simplifica varios de esos elementos.

### No reemplazar ahora el piloto completo

Coincido. No hace falta implementar *minimum-snap*, splines o un controlador nuevo si el existente produce demostraciones con calidad suficiente.

Pero la justificación correcta es **la calidad medida del vuelo**, no que la inercia del vehículo garantice automáticamente suavidad adecuada o seguridad.

### Posponer parte del contrato de ejecución aprendido

Acepto esta simplificación **si conservan completos los comandos originales, la respuesta dinámica y sus tiempos**. La prueba con referencias perfectas sigue siendo útil y barata comparada con entrenar el modelo, pero no tiene por qué bloquear toda captura exploratoria.

### Separar desarrollo del validador e integración con el recolector

También es razonable implementarlo en dos fases. Lo importante es no confundir:

- “el validador existe y puede ejecutarse manualmente”;
- con “todos los episodios nuevos pasan obligatoriamente por él”.

Actualmente están en la primera situación.

---

## 2. Lo mejor de la Fase 1 implementada

Hay avances reales:

- **Cruce de aro basado en intersección del plano**, no únicamente en cercanía a un waypoint.
- **Verificación del lado del slalom.**
- **Comprobaciones de separación a obstáculos.**
- **Métricas continuas**, además del resultado binario.
- **Detección de resets del estimador** en la ruta de odometría.
- **Tolerancias del piloto diferenciadas por primitiva.**
- **Suscripción al detector de aterrizaje.**
- **Filtro previo de transiciones problemáticas.**
- **Pruebas negativas** para bypass del aro, colisión y slalom incorrecto.

Esto mejora sustancialmente la capacidad de diagnosticar el dataset.

**No recomendaría tirar este trabajo ni cambiar de arquitectura. Recomendaría fortalecerlo antes de convertirlo en autoridad para aceptar vuelos.**

---

# 3. Hallazgos que corregiría antes de dar por cerrada la Fase 1

## 3.1. El aterrizaje sigue pudiendo aprobarse sin confirmación real

**Prioridad alta. Defecto verificable en el código.**

### En el piloto

En [`waypoint_controller.py`, líneas 363–381](https://github.com/gabrielquiroga/dataset-gen-ip/blob/ec7f011f540c390eb5a54b386be40464b7fe004e/src/uav_ip/uav_ip/waypoint_controller.py#L363-L381):

- se acepta aterrizaje por `landed_detected`;
- **o solamente por altura baja**;
- al llegar al timeout de diez segundos, también se pasa al estado que termina con `SystemExit(0)`.

Por lo tanto, el timeout de confirmación todavía termina como ejecución exitosa.

Además, el `main()` [captura `SystemExit` y no propaga su código](https://github.com/gabrielquiroga/dataset-gen-ip/blob/ec7f011f540c390eb5a54b386be40464b7fe004e/src/uav_ip/uav_ip/waypoint_controller.py#L392-L402). Cambiar únicamente el timeout a `SystemExit(1)` no bastaría: también hay que preservar ese resultado al salir del proceso.

### En el validador

La [comprobación de aterrizaje](https://github.com/gabrielquiroga/dataset-gen-ip/blob/ec7f011f540c390eb5a54b386be40464b7fe004e/src/uav_ip/uav_ip/validate_episode.py#L906-L942) tiene cuatro debilidades:

1. Considera `ground_contact` equivalente a `landed`. El [propio mensaje PX4](https://github.com/gabrielquiroga/dataset-gen-ip/blob/ec7f011f540c390eb5a54b386be40464b7fe004e/src/px4_msgs/msg/VehicleLandDetected.msg#L5-L8) dice que son etapas distintas.
2. Si no hay mensajes del detector, no exige su presencia para aceptar.
3. Usa la **menor distancia a la plataforma en parte de la trayectoria**, no necesariamente la distancia donde finalmente aterrizó.
4. Puede omitir el rechazo por velocidad final alta si encontró algún indicador de contacto/aterrizaje entre los últimos diez mensajes, sin comprobar su frescura.

**Contraejemplo:** pasar cerca de la plataforma a baja altura y luego terminar quieto fuera de ella puede superar la comprobación de posición, porque se conserva el mínimo histórico.

### Corrección pragmática

No requiere más sensores:

- exigir mensajes recientes del detector;
- separar `ground_contact` de `landed`;
- evaluar posición, velocidad y señal de aterrizaje en **la misma ventana temporal final**;
- exigir persistencia breve;
- devolver fallo en timeout;
- propagar correctamente el exit code.

---

## 3.2. El tamaño del vehículo usado para validar contradice la documentación

**Prioridad alta.**

El código usa:

- `UAV_COLLISION_RADIUS_XY = 0.18`;
- distancia mínima al eje del poste de `0.35`;
- límites de cruce distintos de los inicialmente propuestos.

Está en las [constantes del validador](https://github.com/gabrielquiroga/dataset-gen-ip/blob/ec7f011f540c390eb5a54b386be40464b7fe004e/src/uav_ip/uav_ip/validate_episode.py#L57-L88).

Sin embargo, la documentación justifica la seguridad usando un radio de **0,28 m**.

Esto no es una diferencia editorial: cambia qué vuelos se aceptan.

**Ejemplo geométrico:** un cruce centrado verticalmente y desplazado lateralmente 0,45 m deja aproximadamente 0,20 m hasta la cara interna del pilar.

- Supera el umbral de 0,18 m del código.
- No alcanza una envolvente de 0,28 m.

No afirmo que 0,28 m sea el radio físico correcto: **hay que derivarlo del modelo que realmente usan**, distinguiendo chasis, rotores y margen.

Además, la justificación contiene un error aritmético de geometría: compara `0.38 + 0.28 = 0.66` con 0,70 m, cuando el semiancho libre horizontal del aro cuadrado es **0,65 m**.

### Qué haría

- Definir una única envolvente documentada del UAV.
- Derivarla del modelo y las orientaciones relevantes.
- Separar tamaño físico y margen de seguridad.
- Hacer que controlador, validador y documentación usen la misma definición.

**No ajustaría el radio para que los vuelos existentes pasen.**

---

## 3.3. El chequeo de colisiones no es continuo ni “exacto” respecto de toda la simulación

La función [`compute_min_distance_to_landmark()`](https://github.com/gabrielquiroga/dataset-gen-ip/blob/ec7f011f540c390eb5a54b386be40464b7fe004e/src/uav_ip/uav_ip/validate_episode.py#L392-L437) calcula distancias en las posiciones muestreadas. No comprueba el volumen barrido entre ellas.

Eso es útil, pero no permite afirmar inmunidad al *tunneling*.

### Dos límites concretos

**Muestreo temporal:** el validador admite huecos de hasta 0,60 s. Un intervalo así puede esconder un paso completo por un obstáculo. Que la mediana sea 8 ms no resuelve los huecos excepcionales.

**Aro circular:** el validador usa un toroide ideal, mientras que el [SDF se construye con 16 cilindros](https://github.com/gabrielquiroga/dataset-gen-ip/blob/ec7f011f540c390eb5a54b386be40464b7fe004e/src/uav_ip/scripts/world_generator.py#L122-L171). Compartir radio mayor y grosor no hace idénticas ambas superficies.

### Solución proporcionada

- Verificar segmentos con una envolvente conservadora, o subdividir con un límite espacial y una cota explícita del error.
- Rechazar tramos con huecos incompatibles con esa cota.
- Usar la geometría real del aro o un margen que cubra su aproximación.
- Denominar el resultado “validación geométrica bajo estas hipótesis”, no certificación física exacta.

**Sigo sin exigir plugins de contacto. Exijo que la alternativa elegida sea coherente con la garantía anunciada.**

---

## 3.4. El parser binario nuevo introduce errores y complejidad evitables

**Prioridad alta. Es la decisión de implementación que menos comparto.**

Para simplificar dependencias se incorporó un lector MCAP/CDR manual con offsets fijos. Eso agrega una segunda implementación de deserialización precisamente en el componente que decide si los datos son confiables.

Al contrastarlo con las definiciones de mensajes incluidas en el repositorio aparecen discrepancias concretas, bajo el layout CDR little-endian que el propio lector asume:

| Campo | Offset usado | Offset correspondiente a la definición incluida |
|---|---:|---:|
| `VehicleLocalPosition.vx` | 48 | 56 |
| `VehicleLocalPosition.heading` | 88 | 104 |
| `VehicleStatus.nav_state` | 41 | 45 |
| `VehicleLandDetected.at_rest` | 21 | 23 |

Fuentes: [lector manual](https://github.com/gabrielquiroga/dataset-gen-ip/blob/ec7f011f540c390eb5a54b386be40464b7fe004e/src/uav_ip/uav_ip/validate_episode.py#L169-L214), [posición local](https://github.com/gabrielquiroga/dataset-gen-ip/blob/ec7f011f540c390eb5a54b386be40464b7fe004e/src/px4_msgs/msg/VehicleLocalPosition.msg#L6-L46), [estado](https://github.com/gabrielquiroga/dataset-gen-ip/blob/ec7f011f540c390eb5a54b386be40464b7fe004e/src/px4_msgs/msg/VehicleStatus.msg#L5-L31) y [aterrizaje](https://github.com/gabrielquiroga/dataset-gen-ip/blob/ec7f011f540c390eb5a54b386be40464b7fe004e/src/px4_msgs/msg/VehicleLandDetected.msg#L3-L21).

No todos esos campos afectan actualmente la ruta principal, pero la ruta alternativa por posición local sí queda comprometida.

Además:

- no comprueba el esquema antes de interpretar offsets;
- asume endianness;
- no verifica el cierre completo del MCAP;
- ante un registro incompleto puede detener la lectura y devolver el prefijo ya leído;
- ese prefijo puede contener suficientes muestras para continuar la validación.

### Recomendación

**Usar `rosbags` como lector principal en un entorno definido**, o el lector ROS correspondiente. Ejecutarlo en un subproceso del entorno de procesamiento es más sencillo que mantener un deserializador propio.

Si conservan el lector manual, necesita pruebas de equivalencia campo por campo con un lector de referencia, control de esquema y rechazo explícito de archivos truncados.

Esta es una buena oportunidad para aplicar el criterio 80/20 que proponen: **menos infraestructura propia, más confianza en el camino crítico.**

---

## 3.5. La reconstrucción del reloj es una heurística, no una reparación demostrada

La función [`reconstruct_simulation_clock()`](https://github.com/gabrielquiroga/dataset-gen-ip/blob/ec7f011f540c390eb5a54b386be40464b7fe004e/src/uav_ip/uav_ip/validate_episode.py#L668-L699) sustituye un intervalo de PX4 por el del bag cuando difieren más de 50 ms y el bag presenta un intervalo pequeño.

Eso puede reparar el caso observado. **Pero también puede ocultar un hueco real o datos entregados en ráfaga.**

Por ejemplo:

- intervalo PX4: 0,70 s;
- intervalo de recepción: 0,008 s;
- resultado: se reemplaza el hueco por 0,008 s.

El chequeo posterior ya no ve el hueco original.

Tampoco garantiza monotonía: un retroceso pequeño puede conservarse, y la validación solo rechaza retrocesos inferiores a −0,05 s.

### Qué pediría

- Conservar los tiempos crudos.
- Registrar cada reparación y su motivo.
- Contrastar `timestamp_sample`, tiempos de recepción y evidencia de sincronización disponible.
- Separar anomalía temporal de “offset confirmado”.
- Rechazar o marcar indeterminados los casos ambiguos.
- Probar pausas, ráfagas, duplicados, retrocesos y cambios de factor de tiempo real.

**Encontrar una serie monótona después de repararla no demuestra que esa serie conserve el tiempo físico correcto.**

Por otra parte, el hallazgo reportado de **125 Hz** es relevante: si se confirma y se sigue tomando una de cada cinco muestras, la frecuencia derivada sería 25 Hz y el horizonte de ocho pasos abarcaría aproximadamente **0,32 s**, no 0,8 s.

---

## 3.6. El validador calcula algunas condiciones, pero no las exige

### Orden de las primitivas

El [bucle principal](https://github.com/gabrielquiroga/dataset-gen-ip/blob/ec7f011f540c390eb5a54b386be40464b7fe004e/src/uav_ip/uav_ip/validate_episode.py#L863-L894) ordena las primitivas, pero entrega **toda la trayectoria** a cada comprobación.

Eso verifica que cada evento ocurrió en algún momento, no que ocurrió en el orden requerido.

**Corrección pequeña:** devolver índices o tiempos de inicio/fin y exigir progresión temporal entre primitivas.

### Orientación en órbita

Se calcula `mean_centripetal_yaw_err_deg`, pero [no se usa para rechazar](https://github.com/gabrielquiroga/dataset-gen-ip/blob/ec7f011f540c390eb5a54b386be40464b7fe004e/src/uav_ip/uav_ip/validate_episode.py#L640-L660), aunque la documentación describe un límite.

También hay que justificar qué significa aceptar 210° para una maniobra definida como 270°: puede existir una tolerancia válida, pero no debe cambiar silenciosamente el significado de la tarea.

### Marcos y validez

El lector extrae `pose_frame`, `velocity_frame` y flags de validez, pero la [validación](https://github.com/gabrielquiroga/dataset-gen-ip/blob/ec7f011f540c390eb5a54b386be40464b7fe004e/src/uav_ip/uav_ip/validate_episode.py#L765-L800) transforma los vectores como NED sin hacer cumplir esos campos.

**Leer un indicador no equivale a validarlo.**

---

## 3.7. El filtro de transiciones es útil, pero es un filtro heurístico

Me parece bien incorporarlo. Evita algunos casos obviamente difíciles.

No obstante, [`check_circuit_transition_feasibility()`](https://github.com/gabrielquiroga/dataset-gen-ip/blob/ec7f011f540c390eb5a54b386be40464b7fe004e/src/uav_ip/scripts/world_generator.py#L314-L431):

- verifica distancias, pendientes y algunos puntos;
- no comprueba la transición completa contra obstáculos;
- no modela velocidad de entrada, frenado o aceleración;
- no demuestra factibilidad dinámica.

Además, `generate_random_circuit()` [lo llama sin la pose inicial aleatorizada](https://github.com/gabrielquiroga/dataset-gen-ip/blob/ec7f011f540c390eb5a54b386be40464b7fe004e/src/uav_ip/scripts/world_generator.py#L524-L526), que se genera después.

**Lo conservaría**, pero lo describiría como filtro previo de dificultad geométrica. La aceptación final seguiría dependiendo del vuelo y del validador.

---

## 3.8. La validación todavía no está integrada al proceso automático

[`data_collector.py`](https://github.com/gabrielquiroga/dataset-gen-ip/blob/ec7f011f540c390eb5a54b386be40464b7fe004e/src/uav_ip/scripts/data_collector.py#L524-L571) sigue:

1. lanzando el grabador;
2. esperando al piloto;
3. marcando éxito por código cero;
4. cerrando el bag después.

**Esto está previsto para la Fase 2 y no es un reproche por no haber implementado todas las fases.** Pero sí limita lo que hoy puede afirmarse:

> Se implementó una herramienta de validación manual; todavía no se convirtió en requisito obligatorio de aceptación del recolector.

---

# 4. Sobre las decisiones calificadas como “sobreingeniería”

Haría una distinción más precisa.

| Decisión | Mi devolución |
|---|---|
| Posponer LiDAR realista | De acuerdo. |
| No agregar contactos de Gazebo | De acuerdo si la validación geométrica es conservadora y comprobada. |
| No reemplazar el piloto | De acuerdo, sujeto a métricas. |
| Posponer campaña multioperador | De acuerdo; era una recomendación ideal, no un requisito esencial. |
| No medir nunca estimación contra ground truth | No lo daría por resuelto por fijar el spawn en `(0,0)`. |
| Posponer la prueba de ejecución de referencias | Aceptable con captura completa y alcance experimental limitado. |
| Sustituir una librería por un parser binario propio | No me parece la simplificación de menor riesgo. |

### El caso de ground truth

Compartir estimador entre controlador y generador da **consistencia interna**, pero no demuestra coincidencia con la posición física del simulador.

Fijar el spawn horizontal tampoco demuestra ausencia de offset vertical, error de estimación o deriva.

El compromiso pragmático sería **hacer una comprobación de calibración en una pequeña muestra de vuelos**, no necesariamente añadir desde el primer día una infraestructura permanente compleja. Después se puede decidir qué margen de incertidumbre necesita el validador.

### La propuesta de recuperación exponencial

Aunque todavía no corresponde a la Fase 1, no la aprobaría como solución dinámica automática.

Con \(\delta p=0,5\) m y \(\gamma=0,6\), la primera corrección sigue siendo 0,2 m en 0,1 s: una velocidad media correctiva de 2 m/s, sin demostrar compatibilidad con la velocidad inicial ni con la aceleración disponible.

Además, que la pose inicial esté libre no garantiza que toda la recuperación lo esté.

Es una **heurística candidata**, no una garantía de viabilidad.

---

# 5. La documentación necesita menos defensa y más trazabilidad

El documento de justificación contiene diferencias materiales con el código:

- radio de seguridad 0,28 m frente a 0,18 m;
- límites de cruce diferentes;
- orientación centrípeta anunciada como condición, pero solo medida;
- “orden secuencial” no verificado temporalmente;
- “reloj exacto” obtenido mediante una sustitución heurística;
- “inmunidad al tunneling” con chequeos discretos.

También atribuye a mi recomendación anterior exigencias como reemplazar el piloto por *minimum-snap* o aumentar obligatoriamente el contexto. **Esas no eran condiciones que impuse en el plan anterior.**

No lo señalo por defender mi propuesta: el problema es que la trazabilidad pierde precisión si se refutan requisitos más fuertes que los realmente planteados.

Usaría este formato para cada decisión:

1. Riesgo.
2. Solución elegida.
3. Hipótesis necesarias.
4. Evidencia disponible.
5. Limitación residual.
6. Condición para revisarla.

Una simplificación bien delimitada es defendible sin presentarla como superior en todos los casos.

---

# 6. Qué haría ahora, sin volver a un plan enorme

Propondría una **Fase 1.1 corta**, enfocada en validar el validador:

### A. Corregir los puntos de aceptación falsa

- Aterrizaje con señales sincronizadas y recientes.
- Timeout con fallo real.
- Envolvente coherente del UAV.
- Orden temporal de primitivas.
- Validación de marcos y flags.
- Colisiones entre muestras o cota conservadora de error.

### B. Simplificar y verificar la lectura de datos

- Preferir lector mantenido.
- Comparar los resultados del parser actual con él.
- Incluir un MCAP pequeño reproducible y casos truncados.

### C. Poner a prueba la corrección del reloj

Como prueba mínima de regresión, este archivo expresa dos propiedades que la implementación actual no satisface. **No lo ejecuté aquí; son casos derivados de las condiciones del código.**

```python name=src/uav_ip/test/test_validation_clock.py
import numpy as np

from uav_ip.validate_episode import reconstruct_simulation_clock


def test_unexplained_source_gap_is_not_silently_erased():
    # Sin evidencia adicional de offset, no convertir 700 ms en 8 ms.
    source = np.array([0.0, 0.008, 0.708, 0.716])
    received = np.array([0.0, 0.008, 0.016, 0.024])

    reconstructed, _ = reconstruct_simulation_clock(source, received)

    assert np.max(np.diff(reconstructed)) >= 0.60


def test_reconstruction_does_not_claim_a_monotonic_clock_with_backsteps():
    source = np.array([0.0, 0.008, -0.002, 0.016])
    received = np.array([0.0, 0.008, 0.016, 0.024])

    reconstructed, _ = reconstruct_simulation_clock(source, received)

    assert np.all(np.diff(reconstructed) > 0)
```

Una API mejor podría devolver “tiempo indeterminado” o lanzar un error en vez de reconstruir esos casos; los tests se adaptarían a ese contrato.

### D. Agregar una batería pequeña de falsos positivos

| Caso | Resultado esperado |
|---|---|
| Sin mensajes del detector de aterrizaje | No aceptar como aterrizaje confirmado. |
| Contacto sin `landed` | No equipararlo automáticamente a aterrizaje. |
| Pasó cerca de la plataforma y terminó afuera | Rechazar. |
| Timeout de aterrizaje | Código no cero. |
| Barrido orbital suficiente mirando hacia afuera | Rechazar si la orientación es parte de la tarea. |
| Primitivas realizadas en orden incorrecto | Rechazar. |
| Dos muestras libres con una barra entre ellas | Detectar o declarar evidencia insuficiente. |
| Marco de posición desconocido | Rechazar. |
| MCAP truncado con prefijo aparentemente válido | No certificar integridad. |

Los tests nuevos existentes son un buen comienzo, pero el [test de bag real sigue dependiendo de un archivo local opcional](https://github.com/gabrielquiroga/dataset-gen-ip/blob/ec7f011f540c390eb5a54b386be40464b7fe004e/src/uav_ip/test/test_synthetic_lidar.py#L461-L473). Conviene convertirlo en fixture o marcarlo explícitamente como omitido cuando no esté disponible.

### E. Después, avanzar a la Fase 2 y al piloto

- Integrar cierre → lectura → validación → aceptación.
- Preservar intentos fallidos.
- Congelar umbrales antes de medir el lote de evaluación.
- Reportar éxito por intento original, no solamente después de cambiar semillas.

Reintentar con otra semilla puede ser útil para construir un corpus válido, pero **cambia la distribución hacia escenarios que el experto logra resolver**. No debe presentarse como 100% de éxito sobre los escenarios originalmente solicitados.

---

## Conclusión

**La dirección es buena y el trabajo aporta mejoras importantes. Lo que todavía no está justificado es dar por “cerrada” la validación física ni describirla como exacta o garantizada.**

Mi evaluación sería:

- **Priorización:** acertada.
- **Uso de un validador geométrico:** acertado.
- **Conservación del piloto actual:** razonable.
- **Implementación de la Fase 1:** valiosa, pero con defectos que permiten aceptaciones incorrectas.
- **Documentación de garantías:** más fuerte que la implementación.
- **Preparación para generar masivamente:** todavía no; sí para diagnóstico y pruebas controladas.

**El siguiente paso no es agregar más sofisticación. Es asegurarse de que el nuevo criterio de éxito no vuelva a producir la falsa confianza que se intentaba eliminar.**