**Mi conclusión: es un prototipo de investigación bien encaminado y con trabajo de ingeniería real, pero todavía no es una base científicamente validada ni un generador de datos cuya calidad pueda darse por garantizada.** Su mayor fortaleza es la integración del proceso de generación y procesamiento; su mayor debilidad es que varias afirmaciones teóricas y de confiabilidad son más fuertes que la evidencia que las sostiene.

Revisé el código en el commit [`46eff8d`](https://github.com/gabrielquiroga/dataset-gen-ip/tree/46eff8d8245d8e50ce2dafb823c14b6244c55f0d): documentación, generadores, controladores, recolector, conversor, auditoría, pruebas y configuración. También contrasté bibliografía en la web y consulté código del proyecto original de Instant Policy.

**Alcance:** fue una revisión estática, no ejecuté PX4/Gazebo ni las pruebas. La investigación automatizada requería permisos de escritura que no estaban disponibles, pero sí pude inspeccionar los archivos en modo de lectura.

## 1. Qué está realmente logrado

El repositorio implementa una cadena concreta:

**Tareas semánticas → mundos Gazebo → vuelos con piloto programado → telemetría → observaciones geométricas sintéticas → datos para aprendizaje.**

Utiliza Python, ROS 2 Jazzy, PX4, Gazebo Harmonic, NumPy, PyTorch/PyTorch Geometric y `rosbags`.

La distinción importante es esta:

> **Todavía no implementa ni valida una política Instant Policy para drones. Implementa principalmente la infraestructura para generar sus datos.**

La propia [hoja de ruta](https://github.com/gabrielquiroga/dataset-gen-ip/blob/46eff8d8245d8e50ce2dafb823c14b6244c55f0d/docs/PLAN_IMPLEMENTACION.md#L212-L263) deja pendientes la campaña definitiva, el modelo de difusión, el entrenamiento y la evaluación en lazo cerrado. Eso es transparente y correcto, aunque algunas otras partes de la documentación transmiten una madurez superior.

### Aspectos bien resueltos

- **Separación entre tarea y escenario.** Una secuencia de primitivas permanece fija mientras cambian las posiciones de los objetos entre episodios. Es una decisión pertinente para aprender reglas y no memorizar coordenadas.
- **Particionado por firma semántica.** La auditoría agrupa tareas equivalentes antes de repartirlas. Dentro del formato generado por el propio proyecto, esto reduce una fuente importante de contaminación entre entrenamiento y evaluación.
- **Acciones relativas y marcos explícitos.** `compute_se3_action_horizon()` calcula correctamente, para la representación restringida a posición y yaw, la transformación entre pose presente y futura.
- **Representación estabilizada por gravedad.** Es razonable separar la geometría de planificación de las inclinaciones transitorias del cuadricóptero, siempre que no se confunda eso con conservar toda la información dinámica.
- **Herramientas de inspección.** Hay visualización de trayectorias, horizontes y aumentos de datos, semillas y metadatos por episodio.
- **Pruebas matemáticas relevantes.** Existen tests de cambios de marco, proyección, invariancia ante inclinaciones y composición de transformaciones.

No es solamente un README ambicioso: esas piezas están efectivamente implementadas.

## 2. Trasfondo teórico: razonable como hipótesis, exagerado como garantía

### A. Instant Policy es una referencia válida, pero no valida esta adaptación

La referencia central corresponde a **Vitalis Vosylius y Edward Johns**. El trabajo original estudia imitación en contexto mediante difusión sobre grafos, con experimentos de manipulación robótica.

También lo corrobora su [código de despliegue](https://github.com/vv19/instant_policy/blob/72281736672a555e9c9834eb15c225e2f1baaf93/deploy.py#L35-L64): utiliza poses del efector, nubes segmentadas y estado de la pinza; deja al controlador la ejecución de las transformaciones predichas.

**Adaptarlo a drones es una línea de investigación defendible, no una consecuencia ya demostrada por ese paper.** Cambian la dinámica, la observabilidad, las restricciones de movimiento y el significado del estado binario.

### B. Un LiDAR de 360° no garantiza la propiedad de Markov

La documentación afirma una [“preservación estricta” de la hipótesis de Markov](https://github.com/gabrielquiroga/dataset-gen-ip/blob/46eff8d8245d8e50ce2dafb823c14b6244c55f0d/docs/DECISIONES_DISENO.md#L9-L22).

**Esa afirmación no está justificada.**

Dos drones pueden tener la misma posición, orientación y nube observada, pero velocidades distintas. El que se aproxima rápidamente a un obstáculo necesita una acción diferente del que está quieto. Una instantánea geométrica no distingue esos estados.

Esto tiene correspondencia concreta en el código: el conversor lee la velocidad de la odometría, pero **no la incluye en los atributos exportados del agente**. Además, 360° no elimina limitaciones de alcance, simetrías del entorno o ambigüedad sobre el progreso de una tarea.

El sensor omnidireccional reduce la pérdida de referencias visuales. **No convierte automáticamente una observación parcial en un estado dinámico suficiente.**

### C. Una identidad de matrices no demuestra convergencia del vuelo

El repositorio utiliza:

\[
\widetilde T_t\,(\widetilde T_t^{-1}T_{t+k})=T_{t+k}
\]

La identidad es correcta. Sin embargo, la documentación la presenta como una [garantía de recuperación suave y convergente](https://github.com/gabrielquiroga/dataset-gen-ip/blob/46eff8d8245d8e50ce2dafb823c14b6244c55f0d/docs/ESTRATEGIA_TEORICA.md#L185-L204).

**Lo que demuestra es que las coordenadas del objetivo están bien transformadas, no que el dron pueda alcanzarlo con seguridad.**

No contempla velocidad inicial, aceleración disponible, retardo, saturación, colisiones ni error de la política aprendida. El test llamado `test_tube_augmentation_se3_convergence` verifica composición de matrices y dirección geométrica de recuperación sobre una recta; no simula convergencia dinámica.

El aumento de datos es una idea útil. La garantía atribuida a él no está demostrada.

### D. El escalado `s=0.1` es plausible, pero su justificación es demasiado absoluta

Para la familia sinusoidal escrita en la documentación, la periodicidad que se señala existe. No obstante, de ahí no se desprende que escalar:

- elimine toda ambigüedad de la representación;
- conserve el desempeño del encoder preentrenado;
- evite cualquier necesidad de adaptación.

Eso depende de la codificación completa, las distancias entre nodos y la distribución geométrica. El encoder todavía figura como pendiente de integración.

**El escalado es una elección razonable de normalización, no una garantía de transferencia sin degradación.**

## 3. Problemas concretos que limitan la confiabilidad

### 1. “Episodio exitoso” no significa “misión verificada”

Este es el problema más importante.

El [recolector](https://github.com/gabrielquiroga/dataset-gen-ip/blob/46eff8d8245d8e50ce2dafb823c14b6244c55f0d/src/uav_ip/scripts/data_collector.py#L535-L561) marca éxito cuando el piloto termina con código cero.

El [piloto](https://github.com/gabrielquiroga/dataset-gen-ip/blob/46eff8d8245d8e50ce2dafb823c14b6244c55f0d/src/uav_ip/uav_ip/waypoint_controller.py#L295-L337), después de ordenar aterrizaje, espera cuatro segundos y termina. No exige confirmación de aterrizaje ni verifica contactos o colisiones.

Además, el criterio para aceptar el waypoint central permite errores laterales de hasta 1,4 m en una de sus ramas, mientras que el aro cuadrado tiene una abertura lateral de aproximadamente 1,3 m **en total**, antes de descontar el tamaño del dron.

Esto no prueba que los vuelos habituales fallen. Sí prueba que **el criterio de éxito no certifica que se haya atravesado correctamente el aro ni completado un vuelo libre de colisiones**.

La auditoría cuenta los estados de `episode_meta.json`; no aporta una comprobación física independiente. Por eso, una tasa alta de “success” puede sobreestimar la calidad real del dataset.

### 2. El horizonte temporal no está garantizado

El conversor toma [una muestra cada cinco mensajes](https://github.com/gabrielquiroga/dataset-gen-ip/blob/46eff8d8245d8e50ce2dafb823c14b6244c55f0d/src/uav_ip/scripts/bag_to_heterodata.py#L839-L875), no una muestra cada intervalo fijo de tiempo.

Por lo tanto:

- ocho pasos no equivalen necesariamente a 0,8 segundos;
- cambios de frecuencia o pérdida de mensajes cambian el significado de las etiquetas;
- los timestamps no se conservan en el [archivo exportado](https://github.com/gabrielquiroga/dataset-gen-ip/blob/46eff8d8245d8e50ce2dafb823c14b6244c55f0d/src/uav_ip/scripts/bag_to_heterodata.py#L983-L1060).

Para predecir trayectorias de un sistema dinámico, esta diferencia es sustancial.

### 3. Hay un error de signo en el procesamiento alternativo

En [`find_flight_trim_indices()`](https://github.com/gabrielquiroga/dataset-gen-ip/blob/46eff8d8245d8e50ce2dafb823c14b6244c55f0d/src/uav_ip/scripts/bag_to_heterodata.py#L318-L328):

- con odometría, la altura ENU se calcula como `-position[2]`;
- con `local_pos`, se usa `item["z"]` sin invertirlo, aunque también está en NED.

**Es un defecto verificable por lectura**, que afecta el recorte del vuelo cuando se utiliza esa ruta alternativa.

### 4. El “LiDAR” es una observación privilegiada de superficies, no un sensor realista

La generación [filtra puntos por distancia](https://github.com/gabrielquiroga/dataset-gen-ip/blob/46eff8d8245d8e50ce2dafb823c14b6244c55f0d/src/uav_ip/uav_ip/synthetic_lidar.py#L715-L769), sin resolver oclusiones mediante rayos.

Eso permite incluir superficies que un sensor real no vería. Tampoco reproduce el patrón angular, ruido, retornos o distorsión temporal de un LiDAR físico.

**Esto es aceptable para un experimento con percepción idealizada**, y el proyecto reconoce que utiliza información privilegiada. Lo que no resulta justificado es tratar esa observación como equivalente a un Livox real o extrapolar directamente a vuelo físico.

### 5. El aumento de datos no respeta exactamente su especificación

La teoría establece una perturbación acotada por:

\[
\|\delta p\|\leq0.60\ \text{m}
\]

Pero el [código](https://github.com/gabrielquiroga/dataset-gen-ip/blob/46eff8d8245d8e50ce2dafb823c14b6244c55f0d/src/uav_ip/scripts/bag_to_heterodata.py#L590-L605) limita cada componente individualmente. Eso permite normas de hasta aproximadamente **1,04 m**.

Además, comprueba una altura mínima, pero no que la pose perturbada esté fuera de los obstáculos ni que su recuperación sea transitable. Algunas etiquetas aumentadas pueden ser geométricamente coherentes y físicamente inválidas.

### 6. La reproducibilidad y las pruebas todavía son débiles

Hay buenas semillas, pero reproducibilidad es más que eso:

- El [instalador](https://github.com/gabrielquiroga/dataset-gen-ip/blob/46eff8d8245d8e50ce2dafb823c14b6244c55f0d/src/uav_ip/scripts/setup_venv_ip.sh#L34-L80) instala versiones actualizadas sin fijarlas y crea el entorno con `--system-site-packages`.
- El README depende de PX4 `main`, sin fijar una revisión compatible.
- No hay workflows propios en `.github/workflows` de la raíz; los incluidos están dentro del paquete copiado `src/px4_msgs`.
- El [test de descubrimiento de episodios](https://github.com/gabrielquiroga/dataset-gen-ip/blob/46eff8d8245d8e50ce2dafb823c14b6244c55f0d/src/uav_ip/test/test_synthetic_lidar.py#L236-L254) no hace ninguna comprobación si no existe un dataset local.
- El test de compatibilidad de `HeteroData` verifica un objeto simulado, pero no invoca la función real de carga.
- No encontré en el árbol versionado una campaña de resultados cuantitativos que permita corroborar éxito, recuperación o generalización.

También hay documentación desactualizada: el análisis combinatorio sigue hablando de 64 tareas para tres landmarks, mientras que el catálogo actual distingue slalom izquierdo y derecho y ofrece **\(5^3=125\)** combinaciones.

## 4. Qué tan confiable y bien logrado está

| Dimensión | Evaluación |
|---|---|
| **Idea de investigación** | Buena y defendible. |
| **Integración del proceso de datos** | Bien encaminada, con implementación sustancial. |
| **Geometría y transformaciones** | Base razonable, con pruebas útiles y algunos defectos concretos. |
| **Garantías sobre calidad del dataset** | Insuficientes: éxito operativo no equivale a validez física. |
| **Rigor de la documentación teórica** | Desigual: mezcla decisiones sensatas con garantías no demostradas. |
| **Reproducibilidad independiente** | Limitada por versiones, entorno y falta de artefactos de evaluación. |
| **Instant Policy aprendido para UAV** | Todavía no demostrado en este repositorio. |
| **Uso en drones reales** | No respaldado por la evidencia examinada. |

**Como prototipo académico, está bastante bien logrado. Como infraestructura de investigación confiable, todavía es inmaduro. Como demostración de que Instant Policy funciona para drones, aún no constituye evidencia.**

No usaría estrellas, antigüedad o popularidad para juzgarlo: los motivos de esta evaluación están en el código y en la diferencia entre lo implementado y lo que se afirma.

La síntesis más importante es:

> **Tiene mejores fundamentos de ingeniería que garantías científicas. El riesgo principal no es que no produzca datos, sino que produzca datos visualmente convincentes y etiquetados como exitosos cuya validez física y utilidad para aprendizaje no estén comprobadas.**

## Verificación web: resultados y límites

La búsqueda externa respalda la existencia y pertinencia de las referencias centrales, pero **no permitió corroborar los porcentajes exactos de perturbación atribuidos a los apéndices**. No los considero verificados.

Incluyo los resultados completos de las consultas. **La primera respuesta contiene un error en los nombres de los autores de Instant Policy; la segunda lo corrige: son Vitalis Vosylius y Edward Johns.** Esa discrepancia también muestra por qué una síntesis de buscador no debe tratarse por sí sola como evidencia definitiva.
