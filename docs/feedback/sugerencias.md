**Le diría que todavía no lance la campaña masiva. Primero debe demostrar, con una campaña piloto reproducible, que cada demostración aceptada es físicamente válida, está correctamente etiquetada y contiene la información necesaria para entrenar y evaluar la política que realmente quiere investigar.**

El orden importa: **generar más datos antes de corregir los criterios de aceptación puede multiplicar errores silenciosos**, y algunos —como no registrar tiempo de simulación o evidencia de colisiones— no se pueden reparar después.

Este plan se basa en la revisión del commit [`46eff8d`](https://github.com/gabrielquiroga/dataset-gen-ip/tree/46eff8d8245d8e50ce2dafb823c14b6244c55f0d). No presupone que las correcciones ya estén hechas ni que las pruebas hayan sido ejecutadas.

Distingo tres prioridades:

- **P0 — Bloqueante antes de la campaña masiva:** puede invalidar vuelos, etiquetas, evaluación o trazabilidad.
- **P1 — Necesario antes de publicar el dataset o entrenar seriamente:** en general puede resolverse reprocesando datos crudos suficientemente completos.
- **P2 — Investigación posterior:** no hace falta completar toda la política neuronal para empezar a recolectar.

---

# 1. Fijar exactamente qué se pretende demostrar

**Prioridad: P0.**  
**Archivos:** `docs/ESTRATEGIA_TEORICA.md`, `docs/DECISIONES_DISENO.md`, `docs/PLAN_IMPLEMENTACION.md`.

Antes de producir datos, redactar una especificación experimental breve que responda estas preguntas.

## 1.1. ¿El proyecto estudia percepción idealizada o navegación con un LiDAR real?

La versión actual genera superficies a partir de geometría conocida, sin simular oclusiones. Por lo tanto, la descripción defendible es:

> Planificación por imitación en contexto para un UAV simulado, utilizando observaciones geométricas privilegiadas de objetos relevantes.

No presentarlo todavía como validación de navegación con un Livox u otro sensor físico.

Hay dos alternativas legítimas:

| Alternativa | Qué exige antes de recolectar |
|---|---|
| **Percepción privilegiada como primera etapa** | Declarar explícitamente qué información contiene el observador y sus limitaciones. |
| **Aproximación a percepción física** | Definir y validar oclusiones, campo vertical, muestreo angular, ruido, pérdidas y transformación sensor–cuerpo. |

**Mi recomendación:** empezar con la primera, pero preservar datos que permitan generar otras observaciones posteriormente.

## 1.2. ¿Qué predice la política?

Dejar definido:

- Observaciones disponibles.
- Demostraciones de contexto.
- Variables de estado adicionales.
- Acción y unidades.
- Intervalo temporal.
- Horizonte.
- Mecanismo de ejecución en PX4.
- Condición terminal.

La representación actual es esencialmente **posición tridimensional más yaw**, expresada mediante matrices pertenecientes a \(SE(3)\). No equivale a controlar libremente los seis grados de libertad del cuadricóptero.

Describirla como **representación de planificación de cuatro grados de libertad**, con la actitud y el empuje gestionados por el controlador de bajo nivel.

## 1.3. Separar tres afirmaciones que necesitan evidencias distintas

1. El generador produce vuelos expertos válidos.
2. Una política aprende a reproducirlos en lazo cerrado.
3. Esa política generaliza a tareas nuevas usando contexto.

**La primera no demuestra las otras dos.** La campaña debe estar diseñada para que posteriormente puedan evaluarse por separado.

**Entregable:** contrato experimental versionado, con hipótesis, supuestos y métricas.

---

# 2. Reemplazar el criterio actual de éxito por un verificador independiente

**Prioridad: P0, la más importante.**  
**Archivos:** `data_collector.py`, `waypoint_controller.py`, `audit_dataset.py`.  
**Nuevo componente sugerido:** `validate_episode.py`.

Actualmente, el [recolector acepta un código de salida cero](https://github.com/gabrielquiroga/dataset-gen-ip/blob/46eff8d8245d8e50ce2dafb823c14b6244c55f0d/src/uav_ip/scripts/data_collector.py#L535-L561), mientras que el piloto termina después de esperar un tiempo tras ordenar aterrizaje.

Eso debe dejar de ser sinónimo de demostración válida.

## 2.1. Separar ejecución, grabación y validez

Registrar estados independientes:

- `pilot_exit_code`
- `recording_status`
- `mission_status`
- `validation_status`
- `rejection_reasons`

Por ejemplo, un piloto puede terminar normalmente, pero el episodio ser rechazado por atravesar un aro incorrectamente, tener una grabación incompleta o no confirmar aterrizaje.

**Regla:** un fallo del verificador o falta de evidencia produce `unknown` o `rejected`, nunca `accepted`.

## 2.2. Verificar cada primitiva sobre la trayectoria medida

### `traverse`

- Detectar un cruce real del plano del aro en la dirección requerida.
- Interpolar el punto de cruce entre muestras.
- Expresarlo en el marco del aro.
- Verificar que el volumen del vehículo atraviesa la abertura con margen.
- Registrar distancia mínima al marco.

No usar “se acercó al waypoint central” como sustituto.

### `slalom_left` y `slalom_right`

- Verificar el flanco respecto al marco local del poste.
- Verificar que se completa el paso de antes a después del obstáculo.
- Medir separación mínima.
- Evitar que una excursión lateral muy alejada cuente automáticamente como la maniobra pretendida.

### `overfly`

- Usar la altura superior real del objeto.
- Verificar la separación del volumen del dron respecto a esa cota.
- Confirmar el paso sobre la región especificada.

### `orbit`

- Desenvolver el ángulo alrededor del poste.
- Medir barrido angular acumulado y sentido.
- Medir error radial y vertical.
- Verificar orientación centrípeta si forma parte de la tarea.
- Comprobar que no se ha cortado camino a través del poste.

### Aterrizaje

Combinar:

- posición respecto de la plataforma;
- velocidad suficientemente baja;
- señal de aterrizaje;
- estado de armado o condición terminal prevista;
- permanencia durante un intervalo mínimo.

Los umbrales deben calibrarse con el modelo y los datos, no elegirse solamente para elevar la tasa de éxito.

## 2.3. Registrar colisiones explícitamente

Usar eventos de contacto de Gazebo y/o comprobación geométrica independiente del volumen barrido.

Distinguir:

- contacto permitido con la plataforma al aterrizar;
- contacto inicial con el suelo;
- contacto no permitido con obstáculos;
- contacto inesperado con el suelo durante la misión.

La ausencia de un error en el log de PX4 **no demuestra ausencia de colisiones**.

**Criterio de aceptación:** episodios preparados deliberadamente con cruce incorrecto, colisión, aterrizaje incompleto o grabación truncada deben ser rechazados.

---

# 3. Validar físicamente el piloto experto y los circuitos

**Prioridad: P0.**  
**Archivos:** `world_generator.py`, `waypoint_controller.py`.

## 3.1. Revisar las tolerancias de paso

El [controlador actual](https://github.com/gabrielquiroga/dataset-gen-ip/blob/46eff8d8245d8e50ce2dafb823c14b6244c55f0d/src/uav_ip/uav_ip/waypoint_controller.py#L295-L312) tiene tolerancias que no constituyen una comprobación de paso por la abertura.

Acciones:

- Separar el criterio de **avanzar al siguiente setpoint** del criterio de **validar la maniobra**.
- Ajustar tolerancias con la geometría de cada obstáculo.
- Incorporar dimensiones reales del vehículo y margen de seguridad.
- Medir sobrepaso y error de seguimiento en las transiciones.

No basta con reducir todas las tolerancias: eso puede reintroducir detenciones o bloqueos. Hay que evaluar el seguimiento completo.

## 3.2. No asumir que el orden creciente en X garantiza seguridad

La separación entre centros y el orden de landmarks no prueban que:

- la transición entre primitivas sea transitable;
- el vehículo no corte una esquina;
- una órbita no interfiera con otro objeto;
- exista margen con inclinación y velocidad.

Además, una órbita de 270° no es una trayectoria estrictamente monótona en X durante toda la maniobra.

Documentar la restricción como **orden de visita creciente de landmarks**, no como garantía global de monotonía o ausencia de colisiones.

## 3.3. Verificar transiciones entre primitivas

Cubrir particularmente:

- `overfly → traverse` con descenso pronunciado;
- `orbit → traverse`;
- slaloms consecutivos de lados opuestos;
- cambios laterales máximos;
- aros bajos o con yaw extremo;
- última primitiva seguida de aterrizaje.

El generador de mundos actualmente valida principalmente rangos geométricos; necesita una comprobación de factibilidad de la ruta nominal y, después, validación del vuelo ejecutado.

## 3.4. Definir calidad del experto, no solamente supervivencia

Medir:

- velocidad, aceleración y jerk;
- error respecto de la ruta;
- distancia mínima a obstáculos;
- tiempo de misión;
- oscilaciones;
- detenciones innecesarias;
- error de yaw;
- progreso por primitiva.

Un vuelo sin colisiones puede seguir siendo una mala demostración.

**Entregable:** distribución de estas métricas por primitiva y por transición, con ejemplos aceptados y rechazados.

---

# 4. Corregir tiempo, sincronización y horizonte de acciones

**Prioridad: P0 para la captura; P1 para el exportador definitivo.**  
**Archivo:** `bag_to_heterodata.py`.

El [submuestreo actual](https://github.com/gabrielquiroga/dataset-gen-ip/blob/46eff8d8245d8e50ce2dafb823c14b6244c55f0d/src/uav_ip/scripts/bag_to_heterodata.py#L839-L875) opera por número de mensajes, no por tiempo.

## Acciones obligatorias

1. **Registrar y distinguir:**
   - tiempo de recepción del bag;
   - timestamp del mensaje PX4;
   - timestamp de la medición, cuando exista;
   - tiempo de simulación.

2. **Establecer cómo se relacionan esos relojes.**
   - No asumir que tienen el mismo origen.
   - Detectar reinicios, saltos y deriva.
   - Preservar suficiente información para reconstruir la correspondencia.

3. **Remuestrear sobre una grilla temporal explícita.**
   - Por ejemplo, 10 Hz en tiempo de simulación si ese es el contrato elegido.
   - Interpolar posición.
   - Desenvolver yaw antes de interpolarlo, o usar interpolación de rotaciones cuando corresponda.
   - No interpolar a través de huecos grandes sin marcarlos.

4. **Definir el horizonte en segundos.**
   - Guardar los tiempos relativos de cada objetivo.
   - No afirmar que ocho muestras equivalen a 0,8 s sin verificarlo.

5. **Guardar máscaras de validez.**
   - Muestras interpoladas con soporte insuficiente.
   - Horizonte incompleto al final.
   - Datos inválidos.
   - Segmentos posteriores a un reset del estimador.

6. **Elegir una política explícita para el final del episodio.**
   - Descartar objetivos sin horizonte completo;
   - o usar padding terminal documentado con máscara.

La repetición de la última pose puede ser útil, pero no debe confundirse con ocho observaciones futuras reales.

**Pruebas:** misma trayectoria física registrada con frecuencias diferentes debe producir objetivos equivalentes después del remuestreo, dentro de tolerancias numéricas.

---

# 5. Calibrar la relación entre Gazebo y el estimador PX4

**Prioridad: P0.**  
**Archivos:** `base_pilot.py`, `synthetic_lidar.py`, `bag_to_heterodata.py`.

Las conversiones NED/ENU y FRD/FLU son necesarias, pero **cambiar ejes no corrige automáticamente un desplazamiento entre orígenes**.

## Acciones

- Registrar pose ground truth de Gazebo y pose estimada de PX4.
- Medir su diferencia inicial y durante el vuelo.
- Verificar altura de referencia, desplazamiento horizontal y orientación.
- Registrar resets del estimador.
- Comprobar los campos de referencia de odometría y validez de posición.
- Rechazar NaN, cuaterniones inválidos o referencias no soportadas.

La [definición de `VehicleOdometry`](https://github.com/gabrielquiroga/dataset-gen-ip/blob/46eff8d8245d8e50ce2dafb823c14b6244c55f0d/src/px4_msgs/msg/VehicleOdometry.msg#L7-L32) contiene información de marcos y resets que el parser actual no preserva.

### Decisión conceptual importante

Definir qué pose usa cada componente:

| Uso | Pose recomendada según propósito |
|---|---|
| Verificación física | Ground truth del simulador. |
| Observación privilegiada ideal | Ground truth, si así se define el experimento. |
| Estado disponible para la política | Estimación que realmente estaría disponible en despliegue. |
| Diagnóstico | Ambas, con su relación temporal y espacial. |

Si se genera la nube con geometría real del mundo pero con una pose estimada sin calibrar, se obtiene una observación que puede no corresponder ni a un sensor ideal ni a uno real.

## Corregir el error de signo confirmado

Cambio mínimo en la ruta alternativa de recorte:

```diff name=src/uav_ip/scripts/bag_to_heterodata.py url=https://github.com/gabrielquiroga/dataset-gen-ip/blob/46eff8d8245d8e50ce2dafb823c14b6244c55f0d/src/uav_ip/scripts/bag_to_heterodata.py#L318-L328
@@
         else:
-            z = float(item["z"])
+            z = -float(item["z"])
             x = float(item["y"])
             y = float(item["x"])
```

Acompañarlo con un test: una misma secuencia representada como `odometry` y como `local_pos` debe generar el mismo intervalo de recorte.

---

# 6. Resolver la observabilidad y el significado de “estado”

**Prioridad: P0 para decidir qué registrar; P1 para experimentar con entradas del modelo.**

## 6.1. Retirar la garantía de Markov

Sustituirla por una afirmación limitada:

> La observación omnidireccional reduce la pérdida de referencias geométricas; la suficiencia del estado para controlar el UAV debe evaluarse.

## 6.2. Preservar variables dinámicas

Registrar al menos:

- velocidad lineal y su marco;
- orientación completa;
- velocidad angular;
- modo de vuelo;
- estado de armado;
- aterrizaje detectado;
- objetivos enviados al controlador;
- calidad y resets del estimador.

Después podrá compararse:

1. nube + estado binario;
2. nube + velocidad + yaw;
3. nube + estado dinámico más completo;
4. historial corto de observaciones.

No es necesario elegir ahora la arquitectura ganadora. **Sí es necesario no perder esos datos.**

## 6.3. Definir el estado binario correctamente

Actualmente `compute_flight_state()` infiere una fase a partir de altura y proximidad a la plataforma.

Eso no es necesariamente igual a:

- estar armado;
- estar físicamente en vuelo;
- estar aterrizado;
- haber solicitado aterrizaje.

Separar **estado físico medido**, **fase de misión** y **acción discreta objetivo**.

No copiar directamente el significado del estado de una pinza al de un dron.

## 6.4. Examinar la ambigüedad de la tarea

Con objetos repetidos y alcance limitado, dos instantes pueden producir observaciones locales similares pero requerir acciones diferentes según el progreso.

Preparar escenarios que comprueben esto. Si el estado no distingue los casos, las opciones incluyen:

- historial temporal;
- una representación estimada de progreso;
- memoria de la política;
- restricción explícita de la familia de tareas.

Si se incorpora progreso, especificar cómo se obtiene en inferencia. No introducir el índice verdadero de primitiva como una ayuda disponible solamente durante la generación.

---

# 7. Definir cómo se ejecutarán las trayectorias aprendidas

**Prioridad: P0 para el contrato; P2 para el modelo completo.**

La trayectoria futura observada y el comando que la produjo no son lo mismo.

Si el entrenamiento usa posiciones futuras realmente voladas y el despliegue las reenvía como setpoints a PX4, puede aparecer un desajuste por el retardo del controlador.

## Acciones

- Definir si se predicen referencias, desplazamientos deseados o estados futuros.
- Definir si se ejecuta uno, varios o todos los pasos del horizonte.
- Definir frecuencia de replanteamiento.
- Definir tratamiento de velocidad y yaw.
- Registrar los comandos originales y la respuesta real.

### Prueba imprescindible sin red neuronal

Usar la trayectoria futura verdadera como una **predicción perfecta** y ejecutarla con la interfaz propuesta.

Si aun con objetivos perfectos el sistema:

- acumula retardo;
- corta esquinas;
- oscila;
- no aterriza correctamente;

el problema está en el contrato de control, no en el aprendizaje.

**Criterio:** demostrar que la interfaz elegida puede seguir referencias válidas antes de atribuirle esa responsabilidad a la política neuronal.

---

# 8. Validar el observador geométrico sintético

**Prioridad: P0 para consistencia; P1 para realismo adicional.**  
**Archivos:** `world_generator.py`, `synthetic_lidar.py`.

## 8.1. Una sola definición de geometría

Crear una especificación compartida de:

- dimensiones;
- poses;
- radio y altura;
- abertura;
- forma de plataforma;
- geometría de colisión.

Usarla para generar tanto el SDF como la nube.

Ahora hay geometría definida en varios lugares. El aro circular del mundo es una aproximación mediante cilindros, mientras que la nube representa un toroide: no son exactamente la misma superficie.

Se puede aceptar una aproximación, pero debe medirse y documentarse.

## 8.2. Probar correspondencia con el mundo

Para poses conocidas:

- comprobar límites geométricos;
- comprobar centros y normales;
- comparar nube con superficie SDF;
- cuantificar error;
- verificar objetos cerca del límite de alcance.

Incluir las variantes realmente usadas: con y sin plataforma de salida, aros cuadrados y circulares.

## 8.3. Tratar explícitamente la nube vacía

El anillo artificial actual evita colocar puntos en el origen, pero sigue introduciendo geometría inexistente.

Opciones:

- máscara de puntos válidos;
- indicador explícito de observación vacía;
- padding documentado;
- representación especial.

No permitir que el modelo interprete ese anillo como un obstáculo real.

## 8.4. Medir pérdida de objetos por submuestreo

El pre-submuestreo aleatorio a 4096 puntos puede eliminar detalles pequeños antes del FPS.

Medir:

- puntos por objeto antes y después;
- frecuencia con que desaparece un objeto relevante;
- cobertura de barras delgadas;
- sensibilidad a distancia y semilla.

**No asumir que 2048 puntos son suficientes solamente porque aparecen en el método original.**

---

# 9. Revisar el aumento de recuperación

**Prioridad: P1 si se genera offline; P0 si se usa para decidir la captura.**  
**Archivo:** `compute_tube_augmentation()`.

## 9.1. Corregir la cota de perturbación

La documentación exige una cota euclídea; el código limita por componente.

Elegir una distribución precisa:

- gaussiana truncada mediante rechazo dentro de una esfera;
- o proyección radial sobre una esfera.

No son distribuciones idénticas: documentar cuál se usa.

Test obligatorio:

\[
\|\delta p\|_2 \leq r_{\max}+\varepsilon
\]

## 9.2. No llamar “garantía de convergencia” a la composición de poses

Renombrar el test para reflejar lo que verifica, por ejemplo:

- consistencia de reconstrucción de objetivos;
- coherencia de transformaciones relativas.

Reservar “recuperación dinámica” para pruebas ejecutadas con el controlador y la dinámica.

## 9.3. Filtrar perturbaciones inválidas

Verificar:

- pose perturbada fuera de obstáculos;
- volumen del UAV fuera del suelo;
- margen con el aro;
- recuperación transitable;
- viabilidad temporal y dinámica.

Volver a la trayectoria en el siguiente objetivo a 0,1 s puede exigir una maniobra demasiado agresiva para una desviación de varias decenas de centímetros.

Opciones:

- alargar el tiempo de recuperación;
- construir una referencia intermedia;
- generar recuperaciones mediante el experto desde estados perturbados;
- limitar amplitud según velocidad y geometría.

## 9.4. Evaluar si el flip binario tiene sentido

No trasladar automáticamente el ruido del estado de pinza a “vuelo/aterrizaje”.

Definir si representa:

- error de clasificación de un sensor;
- regularización;
- fallo de estado;
- incertidumbre de fase.

Mantener siempre el estado físico verdadero y separar el dato corrompido. Comparar con y sin esa perturbación.

## 9.5. Evitar contaminación de evaluación

- Generar aumentos de entrenamiento después de fijar los splits.
- No modificar las demostraciones nominales de contexto sin declararlo.
- Mantener una evaluación limpia y otra de robustez con perturbaciones controladas.

---

# 10. Corregir catálogo, firmas y protocolo de partición

**Prioridad: P0.**  
**Archivos:** `task_generator.py`, `audit_dataset.py`.

## 10.1. Rechazar el agotamiento del catálogo

El [índice actual usa módulo](https://github.com/gabrielquiroga/dataset-gen-ip/blob/46eff8d8245d8e50ce2dafb823c14b6244c55f0d/src/uav_ip/scripts/task_generator.py#L190-L201). Al superar el tamaño del catálogo, vuelve a generar tareas anteriores.

Debe:

- rechazar índices negativos;
- rechazar índices fuera de rango;
- comprobar `num_tasks <= catalog_size`;
- distinguir claramente modo sin reemplazo y modo aleatorio con reemplazo.

## 10.2. Actualizar la combinatoria

Con el catálogo actual:

- cinco acciones atómicas por posición;
- tres landmarks: \(5^3=125\);
- longitudes 2 a 5: \(25+125+625+3125=3900\).

Con la acción de aro circular opcional, cambia el tamaño.

También aclarar que el intercalado equilibra los primeros tramos del catálogo, pero **no mantiene una distribución uniforme por longitud después de agotar los grupos más pequeños**.

## 10.3. Canonicalizar según comportamiento ejecutado

La firma debe representar lo que hace el controlador, no detalles accidentales del YAML.

Revisar:

- IDs renombrados;
- valores por defecto explícitos o implícitos;
- parámetros que el controlador ignora;
- alias geométricos;
- orden efectivo de ejecución.

Ejemplo: omitir `side` y usar `side: right` deberían ser equivalentes si el controlador ejecuta ambos de la misma forma.

## 10.4. Separar evaluación conocida y composición nueva

La implementación actual separa firmas también para validación. Eso no coincide con describir `val` como “tareas vistas en nuevas poses”.

Elegir explícitamente:

- **Validación dentro de distribución:** mismas firmas, episodios y geometrías nuevos.
- **Generalización composicional:** firmas no vistas.
- **Generalización geométrica:** formas o dimensiones reservadas.
- **Robustez:** perturbaciones de estado, sensor o dinámica.

No hace falta usar todos como benchmark principal, pero no deben confundirse.

## 10.5. Congelar los splits antes de ajustar resultados

Guardar:

- catálogo completo;
- firmas;
- semillas;
- asignación;
- versión del generador;
- hash del manifiesto.

Evitar volver a barajar todo al agregar tareas: eso puede mover antiguas tareas de entrenamiento al test.

## 10.6. Verificar que el contexto realmente importe

Preparar pares de tareas con:

- el mismo mundo;
- distintos comportamientos válidos;
- contextos diferentes.

Por ejemplo, slalom izquierdo versus derecho sobre la misma geometría.

Esto permite evaluar después si el modelo utiliza el contexto o explota correlaciones del escenario.

---

# 11. Fortalecer el recolector y la reanudación

**Prioridad: P0.**  
**Archivo:** `data_collector.py`.

## 11.1. Reanudar sin mezclar datos incompatibles

Antes de reutilizar un episodio:

- verificar manifiesto;
- verificar semilla y firma;
- verificar versión del generador;
- verificar hashes;
- comprobar grabación y validación.

Si cambia la configuración, crear una nueva versión de campaña. No sobrescribir `task_config.yaml` y conservar episodios anteriores incompatibles.

## 11.2. No reutilizar silenciosamente directorios fallidos

Cada intento debe tener identidad propia.

Preservar:

- número de intento;
- causa del fallo;
- logs;
- bag parcial;
- configuración.

Esto también evita intentar grabar sobre una carpeta de rosbag ya existente.

## 11.3. Comprobar la grabación

- Confirmar que el grabador sigue vivo.
- Confirmar llegada de tópicos obligatorios.
- Verificar mensajes mínimos y duración.
- Detener y cerrar el bag.
- Reabrirlo y verificar integridad.
- Solo entonces finalizar el estado del episodio.

**El código cero del piloto no certifica que el bag exista o sea utilizable.**

## 11.4. Reemplazar esperas fijas por condiciones observables

Esperar con timeout hasta que:

- PX4 esté operativo;
- exista odometría válida y reciente;
- DDS esté comunicando;
- el grabador reciba datos;
- Offboard y armado estén confirmados.

## 11.5. Aislar procesos

Los `pkill -9 -f` globales pueden terminar otras simulaciones o grabaciones.

- Gestionar grupos de procesos propios.
- Aislar dominios, puertos y particiones.
- Mantener limpieza global como acción explícita de emergencia.
- No ejecutar campañas simultáneas con recursos compartidos sin aislamiento.

## 11.6. Limitar las relajaciones de seguridad a SITL

El proyecto fuerza armado y relaja comprobaciones. Eso debe:

- estar protegido por un modo explícito de simulación;
- quedar registrado;
- no modificar silenciosamente una instalación compartida;
- restaurarse o ejecutarse en un entorno aislado;
- no trasladarse a hardware.

---

# 12. Convertir la auditoría en una barrera de aceptación

**Prioridad: P0.**  
**Archivo:** `audit_dataset.py`.

La auditoría no debe limitarse a imprimir mensajes de conformidad.

## Acciones

- Salir con código no cero ante errores críticos.
- Distinguir `pass`, `fail` y `unknown`.
- Rechazar configuraciones ilegibles.
- Detectar ausencia de metadatos.
- Verificar integridad del bag.
- Incorporar el informe físico independiente.
- Comprobar compatibilidad de versiones.
- Detectar episodios duplicados y trazas idénticas.
- Medir cobertura de primitivas, lados y longitudes.
- Validar proporciones de splits.

El parámetro `held_out_ratio` debe tener un significado real: actualmente el tamaño se obtiene como resto de las otras particiones.

La variación de waypoints tampoco debe resumirse como “todo varía en 3D” porque haya más de una configuración distinta. Medir variabilidad por componente y por tipo de objeto; algunas coordenadas, como la altura de un poste fijo, pueden permanecer constantes por diseño.

## Métricas separadas

Reportar:

1. tareas solicitadas;
2. mundos generados;
3. intentos de vuelo;
4. vuelos completados;
5. grabaciones válidas;
6. demostraciones aceptadas;
7. rechazos por causa.

No ocultar fallos mediante reintentos hasta obtener éxito.

---

# 13. Congelar entorno y procedencia

**Prioridad: P0.**

## Versiones

Fijar:

- commit de PX4;
- versión compatible de `px4_msgs`;
- ROS 2 y paquetes relevantes;
- Gazebo;
- Python;
- dependencias de procesamiento;
- compilador cuando se use el kernel C.

Evitar que `pip install --upgrade` cambie la campaña entre días.

El entorno creado con `--system-site-packages` no es completamente aislado. Elegir aislamiento estricto o documentar y capturar sus dependencias externas.

## Manifiesto de campaña

Guardar:

- identificador de campaña;
- commit del repositorio;
- estado limpio/sucio;
- configuración completa;
- semillas;
- parámetros PX4;
- modelos y mundo;
- versiones;
- hardware;
- configuración del sensor;
- política de aceptación.

Por episodio, añadir hashes de archivos e identificadores de intento.

## Archivos procesados

Guardar junto a `.pt` o `.npz`:

- `schema_version`;
- factor de escala;
- marcos y unidades;
- timestamps y `dt`;
- horizonte;
- máscaras;
- versión del preprocesador;
- origen del bag;
- configuración del aumento;
- método de keyframes.

Hoy el archivo procesado no es suficientemente autodescriptivo.

## Determinismo

Distinguir:

- misma receta procedural;
- mismo procesamiento offline;
- mismo vuelo bit a bit.

Las semillas no garantizan lo último. Definir tolerancias reproducibles en lugar de prometer igualdad absoluta.

---

# 14. Completar las pruebas que realmente protegen la campaña

**Prioridad: P0 para las rutas críticas; P1 para cobertura restante.**

## Pruebas unitarias mínimas

- NED/ENU y FRD/FLU con diferentes orientaciones.
- Orígenes desplazados.
- Yaw cruzando \(-\pi/\pi\).
- Recorte con odometría y `local_pos`.
- Remuestreo temporal.
- Horizonte terminal.
- NaN y datos inválidos.
- Perturbación con norma acotada.
- Firmas equivalentes.
- Índice fuera del catálogo.
- Proporciones inválidas de splits.
- Geometría SDF/nube.

## Pruebas de integración

- Bag pequeño conocido → exportación → carga → reconstrucción.
- Episodio incompleto → rechazo.
- Grabador muerto → fallo registrado.
- Reanudación con configuración distinta → rechazo.
- Ejecución paralela y serial → resultados equivalentes.
- Carga real de `HeteroData`, no solamente mocks.
- Formación de un batch y de una muestra contexto/consulta.

El objeto actual es principalmente un contenedor de secuencias. Probar que puede alimentar el futuro constructor de grafos; no asumir que guardar en `HeteroData` lo hace automáticamente compatible con la arquitectura de Instant Policy.

## Fixtures autocontenidas

Eliminar tests que dependen de que exista un dataset local y, si no existe, pasan sin comprobar nada.

Crear casos mínimos temporales, deterministas y versionados.

## Integración continua

Añadir una CI propia en la raíz del repositorio:

- tests geométricos sin ROS;
- parser y exportación;
- integridad de fixtures;
- lint;
- smoke test del empaquetado.

La simulación completa puede ejecutarse en un entorno dedicado. No es necesario meter todos los vuelos en cada PR.

---

# 15. Corregir el discurso científico y la documentación

**Prioridad: P0 para el protocolo; P1 para terminar la documentación.**

## Sustituir garantías por afirmaciones verificables

| Afirmación actual o equivalente | Formulación defendible |
|---|---|
| 360° preserva estrictamente Markov | Reduce pérdida de referencias; suficiencia de estado pendiente de evaluar. |
| La identidad \(SE(3)\) garantiza convergencia | Garantiza consistencia geométrica del objetivo. |
| Avance en X garantiza ausencia de colisiones | Restringe el orden; seguridad requiere validación. |
| Escalar elimina aliasing y permite transferencia sin degradación | Normalización elegida; efecto en embeddings y encoder requiere pruebas. |
| Cero fugas de información | Separación según firma definida; otras fugas requieren controles adicionales. |
| 100% de armado exitoso | Resultado medido en una configuración, si existen datos que lo respalden. |

## Revisar las referencias

- Corregir nombres y atribuciones.
- Verificar en fuentes primarias los porcentajes 30% y 10%.
- No presentarlos como confirmados por la búsqueda anterior.
- Separar hiperparámetros originales de adaptaciones al UAV.
- Distinguir evidencia de manipulación, evidencia en UAV y decisiones propias.

También corregir la inconsistencia entre “errores exponenciales” y la cota cuadrática citada para imitación: una cota bajo ciertos supuestos no demuestra divergencia inevitable de este sistema.

## Validar los hiperparámetros heredados

Tratar como hipótesis:

- 2048 puntos;
- 10 keyframes;
- horizonte de ocho pasos;
- escala 0,1;
- proporción de aumentos;
- dimensiones de los keypoints.

Un circuito de cinco landmarks con varias órbitas puede necesitar más información de contexto que uno corto. Conservar las trayectorias completas permite evaluar esto offline.

---

# 16. Ejecutar una campaña piloto antes de escalar

**Prioridad: P0.**

No empezaría directamente con cientos de firmas y miles de vuelos.

## Etapa A — Casos deterministas

- Una prueba por primitiva.
- Ambos lados de slalom.
- Ambos tipos de aro, si se van a usar.
- Diferentes alturas y orientaciones.
- Casos deliberadamente inválidos.

Objetivo: demostrar que el generador, el piloto y el verificador coinciden sobre qué ocurrió.

## Etapa B — Transiciones y extremos

Cubrir:

- pares de primitivas;
- longitudes 2–5;
- extremos de posición y yaw;
- aterrizajes descentrados;
- escenas con poca geometría visible;
- perturbaciones del estado inicial.

## Etapa C — Aleatoriedad controlada

Usar varias semillas por categoría. Como punto de partida operativo podrían ser decenas de vuelos por grupos prioritarios, pero **el tamaño final debe responder a la precisión estadística buscada**, no a un número arbitrario.

Separar:

- casos usados para ajustar el piloto;
- casos nuevos usados para medir su desempeño.

## Etapa D — Repetición independiente

Repetir parte de la campaña:

- en entorno limpio;
- después de reiniciar;
- con otro operador, idealmente;
- verificando versiones y manifiestos.

## Métricas de salida

- éxito físico por primitiva y circuito;
- tasa de aceptación de datos;
- causas de rechazo;
- distribución de márgenes;
- calidad de seguimiento;
- huecos temporales;
- uso de almacenamiento;
- costo por demostración aceptada;
- discrepancia entre verificación automática e inspección independiente.

### Precaución estadística

Cero fallos observados no significa riesgo cero. Como aproximación, con cero fallos en \(n\) ensayos independientes, la “regla de tres” da un límite superior cercano a \(3/n\) para la probabilidad de fallo al 95%.

Las semillas de un mismo generador pueden compartir modos de fallo: también hay que estratificar por maniobra y dificultad.

---

# 17. Planificar almacenamiento y preservar lo irreparable

**Prioridad: P0.**

No calcular almacenamiento solamente con el tamaño del rosbag ligero.

Una nube de 2048 puntos XYZ `float32` ocupa aproximadamente 24 KiB por instante. A 10 Hz durante 60 segundos son alrededor de **14 MiB solamente para las nubes nominales**, antes de aumentos, etiquetas y otros datos.

## Preservar siempre

- telemetría completa elegida;
- tiempo de simulación;
- ground truth;
- evidencia de contactos;
- mundo SDF exacto;
- geometría y configuración;
- comandos;
- metadatos;
- logs e intentos fallidos.

## Mantener como derivados regenerables

- nubes;
- FPS;
- keyframes;
- aumentos;
- grafos;
- animaciones.

Si es posible regenerar las nubes determinísticamente, evaluar almacenamiento bajo demanda o caché. Medir tiempos reales antes de prometer procesamiento de cientos de episodios en segundos.

---

# 18. Criterio final de autorización de la campaña

Yo le pediría al propietario estos entregables antes de aprobar la generación masiva:

| Entregable | Condición mínima |
|---|---|
| Contrato experimental | Observación, acción, tiempo y alcance inequívocos. |
| Verificador independiente | Detecta maniobras incorrectas y colisiones conocidas. |
| Captura completa | Permite reconstruir tiempo, geometría y estado físico. |
| Piloto validado | Métricas por primitiva y transición, no solo exit codes. |
| Calibración de marcos | Error medido y resets gestionados. |
| Catálogo y splits congelados | Sin repetición silenciosa ni cambio retroactivo. |
| Reanudación segura | No mezcla configuraciones ni sobrescribe evidencia. |
| Pruebas automáticas | Rutas críticas verificadas con fixtures autocontenidas. |
| Entorno reproducible | Revisiones y dependencias fijadas. |
| Informe piloto | Resultados, incertidumbre, fallos y decisión de aceptación. |
| Plan de almacenamiento | Datos crudos preservados y derivados regenerables. |

**No exigiría antes de recolectar:**

- terminar la GNN de difusión;
- demostrar transferencia al mundo real;
- implementar todas las variantes sensoriales;
- completar todas las ablaciones;
- desarrollar un controlador avanzado si el actual supera las pruebas necesarias.

**Sí exigiría demostrar que los datos recolectados no comprometen esas investigaciones por falta de información o etiquetas incorrectas.**

---

## Orden recomendado de trabajo

1. **Congelar el contrato experimental y los datos a registrar.**
2. **Implementar verificación física y validez de grabaciones.**
3. **Corregir tiempo, marcos y defectos confirmados.**
4. **Validar piloto, geometría y transiciones.**
5. **Congelar catálogo, splits, versiones y reanudación.**
6. **Ejecutar campaña piloto y revisar rechazos.**
7. **Autorizar producción por lotes pequeños, con auditoría continua.**
8. **Completar preprocesamiento y aumentos offline sobre datos preservados.**

**La recomendación central al propietario sería: “No optimices todavía cuántos vuelos generás por hora. Primero asegurá que podés explicar, verificar y reproducir por qué cada vuelo aceptado es una demostración válida”.**