# Taxonomía de pseudo-tareas y estrategia de dataset: Instant Policy aplicado a drones

*Reporte de diseño para un agente de planificación · Preparado con Claude · 8 de octubre de 2026*

## 1. Resumen ejecutivo

- **Objetivo.** Adaptar Instant Policy (imitación in-context mediante difusión sobre grafos) a la generación de trayectorias de drones. Alcance actual: solo simulación (PX4 + ROS 2 + Gazebo). Sim-to-real y el acople de CGD quedan fuera de este plan.
- **Estado.** Hay un pipeline de captura que funciona para una única trayectoria (3 aros + aterrizaje). Eso es una demostración de una tarea, no un dataset.
- **Unidad de entrenamiento.** El conjunto de demostraciones de una pseudo-tarea: en cada muestra, N−1 demos forman el contexto y la red predice las acciones de la N-ésima.
- **Qué define una tarea.** Una secuencia de waypoints expresados de forma relativa a landmarks, más eventos de cambio de estado. Las demos de una misma tarea solo difieren en las poses de los landmarks y en la pose inicial del dron.
- **Riesgos principales de malinterpretación** (sección 6): confundir escena con tarea; landmarks idénticos o simétricos que vuelven la tarea inobservable; tareas que violan la hipótesis de Markov o la visibilidad de una cámara frontal; la tensión entre una salida en puntos de control de B-spline y lo que el paper reporta que no funcionó; convenciones de frames PX4/ROS/Gazebo; el encoder geométrico preentrenado y la escala métrica.
- **Qué se pide al plan.** Ver sección 11.

## 2. Cómo leer este documento

Cada afirmación lleva una etiqueta de procedencia:

- **\[Paper\]**: afirmado por Instant Policy (Vosylius y Johns, ICLR 2025), con sección o apéndice.
- **\[Inferencia\]**: deducción a partir de la arquitectura del paper, no afirmada por los autores; debe validarse empíricamente.
- **\[Heurística\]**: criterio de ingeniería sin respaldo experimental directo; calibrar.
- **\[Verificar\]**: dato del entorno del usuario que debe confirmarse.
- **\[Decisión del usuario\]**: ya tomada; el plan debe respetarla salvo que se señale un riesgo.

## 3. Contexto y estado actual

- **\[Decisión del usuario\]** Las demostraciones se generan volando con el controlador real de PX4, no moviendo al dron de forma cinemática, para obtener trayectorias dinámicamente factibles. Motivo: en el paper las pseudo-demos no necesitan ser factibles porque la manipulación es cuasi-estática (Sec. 3.4); en un cuadricóptero subactuado sí importa.
- **\[Decisión del usuario\]** Stack: PX4 SITL + ROS 2 + Gazebo, modelo `gz_x500_depth` (x500 con cámara de profundidad frontal modelada sobre la OAK-D, según la documentación de PX4).
- **\[Decisión del usuario\]** La salida de la red serán los puntos de control de una B-spline, inspirada en CGD (arXiv:2405.01758). Sujeta a validación: ver 6.8.
- CGD (guiado por restricciones + QP) se acoplará en una fase posterior. Esta etapa solo prepara el terreno: representación de salida y calidad física de las demos.
- **Captura existente:** topics de ROS 2 con la nube de puntos de la cámara, la odometría (orientación como cuaternión, velocidad lineal y angular) y los setpoints de trayectoria. **\[Verificar\]** que la odometría incluya la posición.
- **Tarea implementada:** atravesar 3 aros secuencialmente y aterrizar.

## 4. Instant Policy: lo relevante para datos

- **\[Paper\] Formulación (Sec. 3.1).** Acción = desplazamiento SE(3) del efector + estado binario del gripper. Observación = nube de puntos segmentada + pose del efector en el mundo + estado binario. Se aprende la distribución de T acciones futuras condicionada a la observación actual y a N demos de longitud L.
- **\[Paper\] Grafo (Sec. 3.2).** Nodos de geometría local (puntos muestreados por FPS con features de un encoder de ocupación preentrenado), 6 nodos de gripper con embedding de su estado, aristas con posiciones relativas. Las demos se conectan con la observación actual a través de los nodos del gripper. Las acciones futuras se representan como nodos 'imaginados'.
- **\[Paper\] Difusión (Sec. 3.3).** Predicción de flujo por nodo, con traslación y rotación desacopladas; DDIM y SVD para recuperar el SE(3).
- **\[Paper\] Datos (Sec. 3.4, Apéndice D).** Pseudo-demostraciones procedurales. Por muestra se toman N pseudo-demos de una misma pseudo-tarea; N−1 forman el contexto. Se generan sobre 2 objetos de ShapeNet colocados al azar sobre un plano; la pseudo-tarea es una secuencia de 2 a 6 waypoints sobre o cerca de los objetos, con uno o más cambios de estado del gripper y distintos estilos de interpolación (lineal, cúbica, esférica). Mover los objetos y cambiar la pose inicial produce varias demos de la misma pseudo-tarea. Espaciación uniforme entre poses (1 cm y 3°). La mitad de las muestras usa waypoints sesgados a tareas comunes y la otra mitad waypoints aleatorios.
- **\[Paper\] Escala (Sec. 4, Apéndice E).** 2.5M pasos de optimización, equivalentes a unas 700K trayectorias únicas generadas continuamente sin repetir; unos 5 días en una RTX 3080 Ti; modelos de 69M a 178M de parámetros. Contexto de 1 a 5 demos en entrenamiento; en test N=2 demos de L=10 waypoints y T=8 acciones.
- **\[Paper\] Procesamiento (Apéndice E).** Demos grabadas a 25 Hz en simulación (10 Hz en real) y reducidas a L=10 keyframes: inicio, fin, cambios de estado del gripper y puntos donde se frena, completando con interpolados. Nubes segmentadas sin robot ni objetos estáticos o distractores, 2048 puntos, expresadas en el frame del efector, 16 nodos de escena.
- **\[Paper\] Limitaciones (Sec. 5).** Requiere nubes segmentadas; sin color ni semántica; horizonte corto con hipótesis de Markov; sensible a la calidad y al submuestreo de las demos; no resuelve evasión de colisiones; sin precisión para tolerancias extremas.
- **\[Paper\] Fallos y lo que no funcionó (Apéndices G y H).** Demos inconsistentes producen contexto contradictorio y la política se frena u oscila; campo de visión insuficiente; segmentación defectuosa. No funcionaron: predecir waypoints dispersos en lugar de acciones pequeñas, entrenar el encoder geométrico desde cero o ajustarlo fino, el grafo totalmente conectado, una única red grande y el grafo homogéneo.

## 5. Definiciones operativas

- **Escena:** los landmarks presentes (aros, pad, postes...). La escena no es la tarea.
- **Pseudo-tarea:** secuencia de waypoints expresados en el frame de cada landmark + eventos de cambio de estado + estilo de interpolación entre waypoints.
- **Demostración:** una ejecución de una pseudo-tarea en un mundo concreto (poses de landmarks + pose inicial).
- **Criterio de misma tarea \[Inferencia\]:** si dos demos se expresan en el frame de cada landmark (alineando los roles) y sus trayectorias se superponen salvo ruido de ejecución, son la misma tarea. Si no, son tareas distintas. Nunca se mezclan demos de tareas distintas en un mismo contexto.
- **Ejemplo base:** tarea = atravesar el aro A, luego el B, luego el C. Demos = la misma estructura con los aros y el inicio en otras poses.

Casos derivados:

- Agregar un aro a la secuencia (atravesar 4): **tarea nueva**.
- Aro presente en la escena pero ignorado por la trayectoria: **misma tarea** con un distractor en la escena. El paper filtra los distractores de la observación (Apéndice E), así que se trata como un experimento aparte.
- Saltear un aro, cambiar el orden, cambiar el comportamiento final o los offsets relativos: **tareas nuevas**.
- Rodear un aro (waypoint de desvío relativo a él): **tarea nueva**. Es imitar un desvío demostrado, no evasión de colisiones.
- Evitar colisiones en general: **no es una tarea**; es un criterio de calidad de todas las demos.
- Cambiar tamaño o forma de un landmark: dentro de una tarea no debe variar (el paper nunca entrenó con geometrías distintas dentro de un contexto; la generalización a geometrías nuevas fue emergente, Sec. 4.3); entre tareas sí puede variar.

## 6. Consideraciones críticas y posibles malinterpretaciones (por impacto)

**6.1 Una trayectoria no es un dataset.** \[Paper\] En entrenamiento se muestrean N pseudo-demos de una misma pseudo-tarea y la red predice las acciones de la última a partir de las N−1 restantes (Sec. 3.4); varias demos de una pseudo-tarea se obtienen moviendo los objetos y la pose inicial (Apéndice D). Sin varias instancias de la misma tarea no existen pares contexto-objetivo.

**6.2 Escena ≠ tarea: el contexto tiene que ser necesario.** \[Inferencia\] Si la escena determina la tarea (3 aros siempre implican los mismos waypoints), la red la resuelve sin leer el contexto y degenera en una política de una sola tarea. La misma escena debe admitir varias tareas. El paper lo logra con waypoints sobre o cerca de los mismos tipos de objetos y con la mitad de las pseudo-demos totalmente aleatorias (Apéndice D). La red no recibe ID ni etiqueta de tarea, solo observación y contexto (Sec. 3.1); la taxonomía sirve para el generador, el muestreo y la evaluación. Evidencia externa (Raventós et al., NeurIPS 2023, regresión lineal): bajo un umbral de diversidad de tareas, el transformer se comporta como un estimador bayesiano atado a las tareas de entrenamiento y no resuelve tareas nuevas; sobre el umbral, más datos mejoran el desempeño en tareas nuevas. El umbral numérico no se traslada a este problema. \[Heurística\] Priorizar el número de tareas distintas sobre el de demos por tarea.

**6.3 Distinguibilidad y simetría de los landmarks.** \[Paper\] El encoder entrega features de geometría local, las nubes no tienen color ni semántica (Sec. 5) y el grafo no tiene IDs de objeto. \[Inferencia\] Con tres aros idénticos, 'el primero, el segundo, el tercero' no es recuperable desde la geometría; con órdenes arbitrarios las demos resultan contradictorias para la red. Opciones: (a) geometría distinta por rol (tamaño o forma), consistente dentro de la tarea y variable entre tareas; (b) un orden definido por una regla espacial consistente en todas las demos. Un aro es simétrico adelante-atrás, así que el 'sentido de pasaje' no es observable salvo con landmarks asimétricos. Validación propuesta: mini-experimento con aros idénticos frente a distinguibles, esperando fallo en el primero.

**6.4 Markov y observabilidad.** \[Paper\] El método apunta a tareas de horizonte corto bajo la hipótesis de Markov (Sec. 5); los bloqueos y oscilaciones se atribuyen a demos contradictorias y a violar esa hipótesis, con la propuesta futura de incorporar observaciones pasadas (Apéndice G). Consecuencia: evitar repeticiones (vueltas múltiples, revisitar un landmark) y 'volver al inicio' salvo que el inicio sea un landmark visible (pad home). \[Paper\] También se atribuyen fallos a un campo de visión insuficiente (Apéndice G, en transferencia entre embodiments) y sus simulaciones usan 3 cámaras (Apéndice D); `gz_x500_depth` tiene una sola cámara frontal. Restricción de muestreo: la pose inicial debe dejar el primer landmark relevante a la vista. Ver la decisión de observación privilegiada o sensorial en la sección 10.

**6.5 Consistencia de las demos.** \[Paper\] Demos con movimientos inconsistentes o arbitrarios producen contexto contradictorio y bloqueo u oscilación (Apéndice G). La diversidad debe venir de las poses del mundo, no de la ejecución. El perfil de velocidad es un parámetro de la tarea o una constante (el paper abstrae el tiempo con espaciación uniforme). QC automático: test de superposición en los frames de los landmarks y descarte de demos con colisión, fallo de tarea o desvío excesivo.

**6.6 Observación: segmentación, frame, escala y encoder geométrico.** \[Paper\] Nubes segmentadas de los objetos de interés, sin robot ni distractores, 2048 puntos, en el frame del efector (Apéndice E). El encoder (red de ocupación) se preentrena aparte con ShapeNet y queda congelado; entrenarlo desde cero no funcionó y ajustarlo fino empeoró (Apéndices A y H). Las posiciones no se normalizan: se confía en embeddings sinusoidales con bandas de frecuencia fijas (Apéndices A y E). \[Inferencia\] Los landmarks de dron (aros, pads, postes, a escala de metros) están lejos del dominio de ShapeNet a escala de mesa. Plan: (1) probar el encoder publicado por los autores sobre muestras de landmarks y medir la calidad de reconstrucción; (2) si es pobre, preentrenar uno propio con el catálogo de landmarks y congelarlo; (3) reescalar coordenadas a un rango compatible. **\[Verificar\]** disponibilidad de código y pesos en https://www.robot-learning.uk/instant-policy.

**6.7 Frames y convenciones.** La odometría de PX4 (`vehicle_odometry`) usa NED en el mundo y FRD en el cuerpo, con la velocidad angular en FRD; la convención ROS (`nav_msgs/Odometry`) usa ENU y FLU; Gazebo usa ENU; `px4_ros_com` incluye utilidades de transformación. Convertir solo NED a ENU sin convertir también FRD a FLU es un error común. La nube de la cámara viene en el frame óptico de la cámara. \[Heurística\] Fijar una convención única para todo el dataset (por ejemplo ENU y FLU), registrarla en la metadata y validar con un movimiento conocido. **\[Verificar\]** que la odometría incluya posición: sin ella no se reconstruye la pose ni la acción.

**6.8 Representación de la acción y de la salida.** \[Paper\] Acciones: desplazamiento SE(3) pequeño por paso (hasta 1 cm y 3°, normalizado a \[−1, 1\]), T=8 acciones, demos a 25 Hz (Apéndice E). El Apéndice H reporta que predecir waypoints dispersos, de mayor magnitud, rindió peor que acciones pequeñas y densas, porque el espacio de acción más grande dificulta la regresión precisa. \[Decisión del usuario\] Salida en puntos de control de B-spline (posición, yaw y tiempo total, como CGD). \[Inferencia\] No es un cambio trivial de cabeza: el denoising del paper opera como flujo por nodo sobre acciones SE(3) relativas y recupera el SE(3) por SVD; pasar a puntos de control exige redefinir los nodos de acción. \[Heurística\] Faseado de riesgo: F1, réplica nativa de Instant Policy con acciones densas pequeñas (paso máximo reajustado a la escala del dron) para validar datos y taxonomía; F2, B-spline ajustada al rollout predicho como post-proceso desacoplado; F3, ablación con salida directa en puntos de control. Grados de libertad \[conocimiento general\]: un cuadricóptero es diferencialmente plano en (x, y, z, yaw) y CGD también parametriza posición y yaw; considerar acciones (Δposición, Δyaw) en lugar de SE(3) completo, dado que roll y pitch no son comandables de forma independiente.

**6.9 Etiqueta de acción.** \[Inferencia\] En el paper las demos son secuencias de poses registradas y las acciones son los desplazamientos entre ellas. \[Heurística\] Etiquetar las acciones con la odometría ejecutada; usar los setpoints para parametrizar las tareas (waypoints relativos) y como diagnóstico del error de seguimiento de PX4. Registrar ambos.

**6.10 Aumentos y recuperación.** \[Paper\] 30% de las trayectorias con perturbaciones locales que incluyen acciones de retorno a la referencia; cambios deliberados del estado binario en el 10% de los puntos para que la política reintente; inversión del estado de entrada con 10% de probabilidad para evitar el sobreajuste; perturbaciones de pose con la nube y las acciones ajustadas en consecuencia (Apéndices D y E). Replicar en el dron manteniendo la consistencia nube-pose-acción.

**6.11 Colisiones y futura integración con CGD.** \[Paper\] Instant Policy no evita colisiones y sus pseudo-demos no las modelan (Sec. 5, Apéndice G). Consecuencias: (a) las demos deben ser libres de colisión por construcción, remuestreando instancias y no cambiando la tarea; (b) 'evitar' un landmark dentro de una tarea es imitar un desvío demostrado; (c) al acoplar CGD, su guiado de repulsión trata todo punto como obstáculo, pero en una política in-context hay landmarks que a veces hay que aproximar (pad, panel): excluir el objetivo del campo de repulsión o modular por distancia a él.

**6.12 Escala y throughput.** \[Paper\] \~700K trayectorias únicas generadas continuamente sin repetir (Apéndice E). \[Heurística\] Con PX4 + Gazebo y cámara de profundidad, \~1 min de simulación por vuelo implica \~170 h de simulación para 10⁴ vuelos: factible con paralelismo y aceleración, no con 10⁵ a 10⁶. Contingencias: reutilizar demos en distintas combinaciones contexto-objetivo con aumentos (el paper evita repetir datos; documentar el riesgo de sobreajuste), o un generador liviano (waypoints, trayectoria suave, nube renderizada sin física) para el volumen, dejando PX4 y Gazebo como set de fidelidad en una mezcla, análoga al co-fine-tuning 50/50 del paper (Sec. 4).

## 7. Taxonomía propuesta

**7.1 Landmarks** (con frames de anclaje; geometrías distinguibles entre roles):

- Aro o gate: centro y normal; catálogo de tamaños y formas (círculo, cuadrado, hexágono).
- Poste o torre: eje vertical.
- Pad: centro de la cara superior y normal; variantes distinguibles (pad objetivo, pad home).
- Panel o pared: punto y normal.
- Caja u obstáculo (opcional): caras y esquinas.

**7.2 Primitivas** (cada una emite uno o más waypoints en el frame del landmark):

- `traverse(aro)`: waypoints de aproximación y de salida sobre la normal, más el centro.
- `overfly` o `underfly(obj, clearance)`.
- `orbit(poste, radio, sentido, ángulo barrido de hasta una vuelta, altura, yaw hacia el centro o tangencial)`.
- `approach_and_hold(panel, standoff, yaw hacia la superficie)`.
- `slalom(postes, lado alterno)`, con postes distinguibles.
- `take_off_to(altura)` y `land_on(pad)`: cambian el estado binario.
- `go_home(pad home)`, con el home como landmark.
- Descartadas por violar Markov: `return_to_start` sin landmark, vueltas múltiples y revisitas.

**7.3 Composición.** Pseudo-tarea = 2 a 6 primitivas sobre los landmarks de la escena, en línea con los 2 a 6 waypoints del paper. La mitad de las pseudo-tareas desde plantillas curadas (carrera, inspección, entrega) y la mitad aleatorias, en el espíritu del 50/50 del paper. Los eventos salientes de la tarea (pasos por landmark, cambios de estado, inicio y fin) deben caber en L keyframes; con aproximación y salida por aro, 3 aros más aterrizaje ya rozan L=10, así que conviene subir L o tratar aproximación y salida como puntos interpolados (el paper nota que tareas más complejas requieren demos más densas).

**7.4 Randomización por nivel.**

- Nivel tarea (fijo en todas sus demos): landmarks y geometrías, orden y subconjunto, parámetros de las primitivas, comportamiento final, estilo de interpolación.
- Nivel demo (varía): poses de cada landmark (posición y yaw, con roll y pitch acotados) y pose inicial del dron, dentro de rangos que cubran lo que se vaya a evaluar, garantizando visibilidad inicial y ausencia de colisiones.

**7.5 Estado binario análogo al gripper.** Elegir uno y mantenerlo en todo el dataset: por ejemplo airborne o landed (natural para la tarea actual). Funciona como señal de fase y entra en los aumentos de 6.10. Es opcional, si no complejiza demasiado en la generación.

**7.6 Semilla mínima desde lo ya implementado.** Escena: 3 aros con geometrías distintas + 1 pad. Tareas: 15 selecciones ordenadas de aros (3 + 6 + 6) × 2 finales (aterrizar en el pad o mantener posición tras el último aro) = 30 pseudo-tareas por conjunto de aros. Con un catálogo de 6 geometrías de aro hay C(6,3) = 20 conjuntos, o sea hasta \~600 pseudo-tareas (cota máxima; descontar combinaciones infactibles). Prueba clave: misma observación, tarea distinta según el contexto.

## 8. Cantidades y fases

- **\[Paper\]** No fija un número de pseudo-tareas; genera continuamente. Cada muestra usa de 2 a 6 demos de una tarea (1 a 5 de contexto más una objetivo). Con \~4 demos por muestra en promedio, 700K trayectorias equivalen a un orden de 10⁵ pseudo-tareas (cuenta de quien redacta, no del paper). El paper no incluye una ablación sobre cantidad de pseudo-tareas frente a demos por tarea.
- **\[Heurística\]** Pool de 8 a 10 demos por pseudo-tarea (al menos 6, el máximo que usa una muestra); en cada paso se sortea objetivo y de 1 a 5 de contexto.
- **Hito 0:** pipeline automatizado de punta a punta (generación, vuelo, registro, postproceso) para una familia, con al menos 20 instancias reproducibles.
- **Hito A:** \~10² pseudo-tareas × \~10 demos (1 a 2.5K vuelos). Pregunta: ¿usa el contexto? Se esperan instancias nuevas de tareas vistas y la ablación de contexto; no se espera generalización a tareas no vistas.
- **Hito B:** \~10³ pseudo-tareas × \~10 demos (\~10⁴ vuelos). Pregunta: generalización a composiciones y tareas no vistas.
- **Hito C:** escalar según las curvas de validación (nº de tareas frente a demos por tarea) sobre pseudo-tareas held-out; el paper usa un hold-out de pseudo-demos para estudiar el escalado (Sec. 4.2).
- Todas las cifras de esta sección salvo las marcadas \[Paper\] son puntos de partida a calibrar.

## 9. Pipeline de datos y evaluación

**9.1 Pipeline a implementar**

1. Configuración del generador (YAML): catálogo de landmarks, plantillas de tarea, rangos de pose, semillas.
2. Generación del mundo por demo: spawn de landmarks y pose inicial.
3. Generador de trayectoria: waypoints relativos, trayectoria suave con límites dinámicos, comandos offboard a PX4; verificación de colisión y de éxito.
4. Registro: rosbag y metadata JSON por demo (`task_id` solo para agrupar, definición de la tarea, configuración de la instancia, semillas, resultados, convención de frames, versiones de software).
5. Postproceso: conversión de frames; observación (privilegiada y, después, sensorial) en el frame del dron con 2048 puntos; keyframes L=10 (inicio, fin, eventos de cambio de estado, mínimos de velocidad, relleno interpolado); acciones desde la odometría; normalización.
6. QC automático: test de superposición y filtros de 6.5.
7. Dataset: índice tarea a demos y sampler de entrenamiento (objetivo y 1 a 5 de contexto, con aumentos).

**9.2 Evaluación**

- **Chequeos:** sobreajuste de una tarea; ablación de contexto (reemplazar el contexto por el de otra tarea de la misma escena: el desempeño debe colapsar si el modelo lo usa); test de identidad de 6.3.
- **Escalera held-out:** (1) poses nuevas de tareas vistas; (2) valores nuevos de parámetros de primitivas conocidas; (3) composiciones nuevas de primitivas conocidas; (4) una primitiva no vista; (5) landmarks con geometría nueva. Los niveles 4 y 5 son análogos a las 24 tareas no vistas de RLBench y al experimento de objetos nuevos del paper.
- **Métricas:** éxito semántico por tarea, error a waypoints, colisiones, tiempo de finalización y pérdida de validación en pseudo-tareas held-out.

## 10. Decisiones abiertas (con default recomendado)

1. **Observación.** Privilegiada (nubes muestreadas de las mallas de los landmarks en poses conocidas, equivalente a las segmentaciones ground truth de las simulaciones del paper, Sec. 4) o sensorial (cámara frontal y segmentación). Default: privilegiada primero, porque aísla el efecto del contexto de FOV, oclusiones y segmentación; seguir capturando la cámara para la fase sensorial.
2. **Acción.** SE(3) completo o (Δposición, Δyaw). Default: (Δposición, Δyaw).
3. **Salida.** Faseado F1, F2, F3 de 6.8.
4. **Estado binario.** airborne o landed frente a carga (7.5).
5. **Paso y frecuencia.** El paper usa 1 cm y 3° por paso a 25 Hz; definir según velocidad nominal del dron, recordando el compromiso con la precisión (Apéndice H).
6. **Encoder geométrico.** Reusar el publicado o preentrenar uno propio (6.6).
7. **Generador.** PX4 y Gazebo, o un generador liviano para el volumen (6.12).
8. **Convención de frames.** Por defecto ENU y FLU (6.7).
9. **Reglas de keyframes** para el dron: eventos de paso por aro, touchdown, mínimos de velocidad, curvatura.

## 11. Entregables esperados del plan de implementación

- Arquitectura y módulos del generador y del postproceso, con interfaces.
- Esquema de configuración, metadata y formato de dataset.
- Especificación del postproceso con tests unitarios y de integración (frames, keyframes, nube de puntos).
- Plan por hitos (0, A, B, C) con criterios de aceptación y estimación de cómputo y tiempo.
- Plan de entrenamiento: sampler, aumentos, encoder y tamaños de modelo de referencia.
- Plan de evaluación (9.2).
- Registro de riesgos con mitigaciones, a partir de la sección 6.
- Lista de preguntas abiertas para el usuario.

## 12. Correcciones a propuestas previas de la conversación

1. Permutar el orden de aros idénticos como fuente de tareas: no aprendible (6.3).
2. 'Sentido de pasaje' por aro simétrico como parámetro de tarea: no observable (6.3).
3. `return_to_start` como primitiva: viola Markov; usar un pad home como landmark (6.4).
4. El conteo de \~234 pseudo-tareas por escena se reemplaza por 30 por conjunto de aros (7.6).
5. La salida en B-spline se había presentado como un cambio simple de representación; tiene una tensión con el Apéndice H (6.8).
6. Se había recomendado Aerial Gym como simulador; el usuario avanzó con PX4 + ROS 2 + Gazebo.

## 13. Fuentes

- Instant Policy: Vosylius y Johns, ICLR 2025. arXiv:2411.12633. Versión de ICLR: https://proceedings.iclr.cc/paper\_files/paper/2025/file/5692c7dbc4abcaa50f9ce609819212e5-Paper-Conference.pdf. Código y videos: https://www.robot-learning.uk/instant-policy
- CGD: Kondo et al., Constraint-Guided Diffusion Policies for UAV Trajectory Planning. arXiv:2405.01758.
- Raventós et al., Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression, NeurIPS 2023. arXiv:2306.15063.
- Aerial Gym: arXiv:2305.16510; repo https://github.com/ntnu-arl/aerial\_gym\_simulator.
- PX4, vehículos de Gazebo (`gz_x500_depth`): https://docs.px4.io/main/en/sim\_gazebo\_gz/vehicles.html
- PX4, mensaje VehicleOdometry: https://docs.px4.io/main/uk/msg\_docs/VehicleOdometry.html
- PX4, conversión de `vehicle_odometry` a `nav_msgs/Odometry` (frames NED/FRD y ENU/FLU): https://discuss.px4.io/t/how-to-convert-uorb-topic-vehicle-odometry-into-nav-msgs-msg-odometry/29801
