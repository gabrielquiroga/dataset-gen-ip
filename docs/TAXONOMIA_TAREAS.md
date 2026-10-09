# Propuesta Extensa: Taxonomía de Pseudo-Tareas para Instant Policy (ICIL)

Este documento detalla la taxonomía extensa para la generación procedimental de tareas en el entorno de simulación de drones. 

## 1. Definiciones Operativas
- **Escena:** Conjunto de *Landmarks* presentes (aros, pads, postes). La escena **no** es la tarea.
- **Pseudo-tarea:** Secuencia de primitivas sobre los *Landmarks*, más eventos de cambio de estado y el estilo de interpolación.
- **Demostración:** Una ejecución de una pseudo-tarea en un mundo concreto (poses de *Landmarks* aleatorias + pose inicial aleatoria).

## 2. Catálogo de Landmarks (Geométricamente Distinguibles)
Para evitar la inobservabilidad (el problema de no saber qué aro es cuál si todos son iguales), el simulador instanciará *Landmarks* de un catálogo con geometrías y escalas distintas:
- **Aro (Gate):** Variantes circular, cuadrada, y hexagonal.
- **Poste (Pole):** Cilindro de altura variable.
- **Pad (Plataforma):** Plataforma de despegue/aterrizaje (Home Pad vs Target Pad).
- **Panel (Pared):** Superficie plana para evitación o inspección.

## 3. Primitivas de Composición
Cada primitiva emite uno o más waypoints en el *frame* local del landmark.
- `traverse(aro)`: Genera waypoints de aproximación, centro y salida sobre la normal del aro.
- `overfly(obj, clearance)` / `underfly(obj, clearance)`: Vuelo por encima o por debajo de un objeto (Slalom Vertical).
- `orbit(poste, radio, sentido, altura)`: Trayectoria circular manteniendo la cámara hacia el centro.
- `slalom(postes, lado_alterno)`: Zig-zag lateral esquivando postes.
- `take_off_to(altura)` y `land_on(pad)`: Modifican el estado binario del dron (Airborne vs Landed).

*(Nota: Primitivas como `return_to_start` sin landmark visible han sido descartadas por violar la propiedad de Markov de visión frontal).*

## 4. Espacio de Parametrización (Ejes Combinatorios)
La "fábrica" de tareas combinará 2 a 6 primitivas alterando los siguientes ejes:

### Eje A: Lógica de Secuenciación y Topología
- `A1_DIRECT`: Cruce secuencial.
- `A2_REVERSE`: Cruce inverso.
- `A3_ALTERNATING`: Patrón alternado.
- `A4_FILTER`: Filtrar por atributo espacial.

### Eje B: Perfil Geométrico y Cinético
- `B1_LINEAR`: Vuelo en línea recta.
- `B2_BSPLINE`: Interpolación suave.
- `B3_TRACKING`: Seguimiento dinámico de objetivos (Feedback continuo).
- `B4_OVERSHOOT`: Pasarse intencionalmente y corregir.

### Eje C: Comportamiento Discreto (Cámara y Velocidad)
- `C1_CRUISE_THROUGH`: Paso a velocidad constante.
- `C2_HOVER_PAUSE`: Freno total de 1s en waypoints clave.
- `C3_YAW_FORWARD`: Yaw tangencial a la ruta (FPV).
- `C4_YAW_TARGET`: Yaw fijado a un landmark de interés.

## 5. Arquitectura del Piloto Experto
El pipeline utilizará el **Patrón Strategy** con controladores dedicados:
1. `base_pilot.py`: Ciclo de vida de ROS 2, armado y QoS.
2. `waypoint_controller.py`: Control *Feed-Forward* para `traverse`, `slalom`, `overfly`.
3. `tracking_controller.py`: Control *Feedback* para `orbit` y primitivas dinámicas.
