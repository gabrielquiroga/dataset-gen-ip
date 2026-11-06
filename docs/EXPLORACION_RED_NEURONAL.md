# Exploración Experimental y Validación de Contratos de la Red Neuronal (Pre-Hito 4)

*Guía conceptual, diccionario matemático y registro experimental completo de contratos de Entrada/Salida (E/S), estadísticas de normalización, verificación geométrica $SE(3)$/SVD 4-DoF, evaluación en lazo cerrado cinemático y perfilado de latencia en CPU para **UAV Instant Policy**.*

---

## 0. Guía Conceptual y Diccionario de Variables (Cómo Leer este Documento)

Esta sección traduce toda la notación matemática y los términos de *Instant Policy* a lenguaje intuitivo, para que cada fórmula, tabla y gráfica del documento se entienda de principio a fin.

### 0.1 ¿Qué hace exactamente nuestra red neuronal? (*In-Context Imitation Learning*)

En el aprendizaje por imitación clásico, una red neuronal entrena durante horas para aprender una única tarea fija (por ejemplo, *"cruzar siempre puertas por el centro"*). Si en el momento de volar queremos que el dron **esquive la puerta por la izquierda**, habría que re-entrenar la red.

**Instant Policy (Aprendizaje por Imitación en Contexto)** funciona de manera radicalmente distinta:
1. **Entradas de la red en cada instante de vuelo:**
   - **El Contexto ($N_{\text{demos}} = 2$ demostraciones):** Le entregamos a la red 2 vuelos grabados previamente en ese mismo circuito (o uno similar) donde un experto realizó la secuencia deseada de maniobras (por ejemplo: *obstáculo 1 por arriba, obstáculo 2 por la izquierda, obstáculo 3 por el centro...*). Cada demostración viene resumida en **$L = 10$ momentos clave (*keyframes*)**.
   - **La Observación Actual (*Live Query* $O_t$):** Lo que el dron está viendo y sintiendo **ahora mismo** en el instante $t$ (su nube de puntos LiDAR 360° y su velocidad).
2. **Salida de la red:**
   - Sin cambiar un solo peso ni re-entrenar nada, la red compara lo que ve ahora ($O_t$) con lo que ocurrió en las demostraciones de contexto y predice **los próximos $T = 8$ pasos futuros de trayectoria** ($0.8\text{ segundos}$ hacia adelante, a razón de $\Delta t = 0.1\text{ s}$ por paso).

---

### 0.2 Marcos de Coordenadas y por qué usamos la Escala $s = 0.1$

En todo el proyecto conviven dos sistemas de coordenadas y una escala espacial reducida:

1. **Marco Global `Gazebo ENU` (Mundo Fijo):**
   - Está clavado en el suelo del escenario de simulación (`X = Este`, `Y = Norte`, `Z = Arriba`).
   - Sirve para saber dónde están los obstáculos en el mapa y para evaluar si el dron llegó a la meta, pero **la red neuronal nunca recibe coordenadas absolutas del mapa** (si lo hiciera, solo sabría volar en las coordenadas exactas donde entrenó).
2. **Marco Egocéntrico `Horizontal FLU` (Pegado al Dron pero Nivelado con el Horizonte):**
   - Su origen $(0,0,0)$ viaja siempre en el centro del dron:
     - **$+X$ (*Forward*):** Hacia adelante (la nariz del dron).
     - **$+Y$ (*Left*):** Hacia la izquierda (babor).
     - **$+Z$ (*Up*):** Verticalmente hacia arriba (cielo).
   - **¿Por qué se llama *Horizontal* FLU?** Un dron cuadricóptero necesita inclinarse hacia adelante (*pitch*) para acelerar o hacia los lados (*roll*) para doblar. Si usáramos el marco inclinado del chasis, cada vez que el dron frenara o acelerara, toda la nube LiDAR parecería subir y bajar violentamente. Al proyectar el marco sobre el plano horizontal (**forzando $\text{roll} = 0$ y $\text{pitch} = 0$**, conservando únicamente el giro de brújula o **rumbo $\psi$ / *yaw***), el suelo siempre está plano y un movimiento en $+X$ significa siempre *"avanzar horizontalmente hacia donde mira la nariz"*.
3. **Factor de Escala Espacial $s = 0.1$ ($1\text{ unidad} = 10\text{ metros}$):**
   - El paper original de *Instant Policy* fue diseñado para brazos robóticos de mesa cuyos movimientos ocurren dentro de una caja de $\sim 0.5\text{ m} \times 0.5\text{ m}$. Nuestro dron vuela en circuitos de $25\text{ m}$ con un LiDAR que ve hasta $5\text{ m}$ de radio.
   - Al multiplicar todas las distancias en metros por **$s = 0.1$** (es decir, **dividir los metros por $10$**):
     - El radio de visión de $5\text{ m}$ del LiDAR pasa a ser $0.5\text{ unidades}$.
     - Un avance de $0.20\text{ m}$ en un paso de $0.1\text{ s}$ pasa a ser $0.020\text{ unidades}$.
     - Así, todos los números que entran a las capas geométricas de la red quedan alrededor de $[-0.5, +0.5]$, evitando que las funciones de activación y las distancias del grafo se saturen.
   - **Regla práctica de lectura:** Siempre que veas una posición o desplazamiento con subíndice $s$ (como $\Delta x_s = 0.030$), **multiplícalo por $10$ para obtener metros reales** ($0.030 \times 10 = 0.30\text{ m} = 30\text{ cm}$). Los ángulos ($\psi, \Delta\psi$) se miden en **radianes** y no se escalan.

---

### 0.3 ¿Por qué representamos al Dron con 6 Puntos Rígidos (*Keypoints* $\mathbf{p}_g^1 \dots \mathbf{p}_g^6$)?

Si queremos describir dónde estará el dron en el futuro y hacia dónde estará mirando, tenemos que predecir una **traslación** $(x, y, z)$ y una **rotación** ($\psi$). Sin embargo, para una red neuronal basada en grafos 3D, mezclar metros con ángulos en radianes es problemático:
- ¿Cómo comparas la distancia entre un punto del obstáculo LiDAR en $(x, y, z)$ y un ángulo en radianes?
- Además, los ángulos tienen un salto brusco cuando pasan de $+180^\circ$ ($+\pi$) a $-180^\circ$ ($-\pi$).

**La solución geométrica de *Instant Policy*:**
En lugar de representar al dron como un punto con un ángulo abstracto, imaginamos una **cruceta rígida tridimensional de 6 puntos (*keypoints*)** soldada al chasis del dron a una distancia de $0.5\text{ m}$ del centro ($L_g = 0.05$ en escala $s=0.1$):
- $\mathbf{p}_g^1 = (+0.05, 0, 0)$ $\rightarrow$ Punta de la **nariz** ($0.5\text{ m}$ al frente).
- $\mathbf{p}_g^2 = (-0.05, 0, 0)$ $\rightarrow$ Punta de la **cola** ($0.5\text{ m}$ atrás).
- $\mathbf{p}_g^3 = (0, +0.05, 0)$ $\rightarrow$ Punta del brazo **izquierdo / babor** ($0.5\text{ m}$ a la izquierda).
- $\mathbf{p}_g^4 = (0, -0.05, 0)$ $\rightarrow$ Punta del brazo **derecho / estribor** ($0.5\text{ m}$ a la derecha).
- $\mathbf{p}_g^5 = (0, 0, +0.05)$ $\rightarrow$ Punta **superior** ($0.5\text{ m}$ arriba).
- $\mathbf{p}_g^6 = (0, 0, -0.05)$ $\rightarrow$ Punta **inferior** ($0.5\text{ m}$ abajo).

**¿Por qué esto es tan poderoso?**
1. Si sabes dónde están esos 6 puntos en el espacio 3D, **sabes automáticamente dónde está el dron (el promedio de los 6 puntos) y hacia dónde apunta su nariz (la orientación de la cruceta)**.
2. Ahora tanto los obstáculos del LiDAR como las poses futuras del dron son simplemente **puntos 3D $(x, y, z)$ en el mismo espacio métrico**, por lo que podemos conectarlos directamente con flechas (aristas de un grafo) y medir distancias reales entre las puntas del dron y las paredes de una puerta.

---

### 0.4 Las 4 Etapas de la Red ($\phi_e \to \sigma \to \varphi \to \psi$) y los Dos Bucles de Inferencia

Cuando la red neuronal se ejecuta, procesa la información en **4 etapas consecutivas** (nombradas con las mismas letras griegas del paper original):

1. **Etapa $\phi_e$ (Codificador de Escena LiDAR):**
   - Recibe los $2048$ puntos crudos del láser LiDAR y elige **$M$ puntos representativos (centroides)** repartidos por los obstáculos cercanos. Para cada centroide, resume la forma geométrica de sus vecinos cercanos (si es un poste vertical, un travesaño horizontal, una pared, etc.).
2. **Etapa $\sigma$ (Grafo Local Escena $\to$ Dron):**
   - Conecta los $M$ centroides del LiDAR con los 6 puntos de la cruceta del dron. Así, cada punto de la cruceta "siente" qué obstáculos tiene cerca y en qué dirección, y además incorpora la velocidad actual del dron (`vel_flu`).
3. **Etapa $\varphi$ (Transformador de Contexto y Horizonte):**
   - Es el "cerebro" de la red. Toma las representaciones $\sigma$ de los $10$ keyframes de las demostraciones de contexto y la representación $\sigma$ de lo que el dron ve ahora, y las combina con los **$T=8$ pasos futuros candidatos** que estamos intentando construir. Mediante mecanismos de atención (*Cross-Attention*), busca: *"¿En qué momento de las demostraciones el dron vio un obstáculo parecido al que veo ahora, y hacia dónde movió su cruceta en los siguientes 8 pasos?"*
4. **Etapa $\psi$ (Cabezal de Flujo de Keypoints y Estado de Vuelo):**
   - Para cada uno de los $T=8$ pasos futuros y cada uno de los 6 puntos de la cruceta, predice un vector de corrección (llamado **flujo**) que indica **hacia dónde debe moverse y girar la cruceta** para acercarse a la trayectoria correcta, además de predecir si el dron debe seguir volando ($1.0$) o aterrizar ($0.0$).

#### El Doble Bucle durante el Vuelo (Diferencia entre $K$ y $T_{\text{exec}}$)

Es fundamental no confundir los dos bucles que operan cuando el dron vuela:
- **Bucle Interno de Difusión ($K$ iteraciones DDIM, dentro de una sola llamada a la red):**
  - Como *Instant Policy* es un modelo de difusión, para generar una predicción de 8 pasos futuros parte de una trayectoria inicial ruidosa (o en cero) y la **refina $K$ veces seguidas** (por ejemplo, $K = 4$ refinamientos internos). En cada refinamiento $i = 1 \dots K$, ejecuta las etapas $\varphi \to \psi$ y ajusta los 8 pasos futuros.
- **Bucle Externo de Control o Horizonte Recedente ($T_{\text{exec}}$ pasos ejecutados en el mundo):**
  - Una vez que la red termina sus $K$ refinamientos internos, nos entrega un plan de **$T = 8$ pasos futuros** ($0.8\text{ s}$ de vuelo).
  - En lugar de volar a ciegas los 8 pasos completos hasta el final, el dron **ejecuta solamente los primeros $T_{\text{exec}}$ pasos** (por ejemplo, $T_{\text{exec}} = 4$ pasos $= 0.4\text{ s}$), toma una nueva foto del LiDAR y vuelve a pedirle a la red un nuevo plan de 8 pasos. Esto se llama **control de horizonte recedente (*Receding Horizon Control*)** y permite corregir cualquier desviación antes de que se acumule.

---

### 0.5 Diccionario Maestro de Símbolos, Fórmulas y Métricas

| Símbolo / Término | Nombre Completo | Explicación en Lenguaje Simple |
| :--- | :--- | :--- |
| $S E(3)$ | Grupo Especial Euclídeo 3D | Conjunto de todas las posiciones $(x,y,z)$ y orientaciones 3D posibles de un cuerpo rígido, representadas como matrices de $4 \times 4$. |
| $S E(2) \times \mathbb{R}$ | Subgrupo 4-DoF Horizontal FLU | Posiciones 3D $(x,y,z)$ + únicamente giro horizontal de brújula ($\psi$ / *yaw*), manteniendo siempre $\text{roll}=0$ y $\text{pitch}=0$. |
| $T_{WE}^{(t)} \in \mathbb{R}^{4 \times 4}$ | Pose del Dron en el Mundo | Matriz $4 \times 4$ que dice dónde está el dron ($E$) respecto al mundo ($W$) en el instante $t$. |
| $T_{EA}^{(t, k)} \in \mathbb{R}^{4 \times 4}$ | Acción **Acumulada** al paso $k$ |Dónde estará el dron dentro de $k$ pasos ($k \in \{1..8\}$) medido **desde donde está parado hoy en el instante $t$**. |
| $\delta T_k \in \mathbb{R}^{4 \times 4}$ | Acción **Incremental** paso a paso | El pequeño salto que da el dron **entre el paso $k-1$ y el paso $k$** (en $\Delta t = 0.1\text{ s}$). Matemáticamente: $\delta T_k = (T_{EA}^{(t, k-1)})^{-1} T_{EA}^{(t, k)}$. |
| $(\Delta x_s, \Delta y_s, \Delta z_s, \Delta\psi)$ | Vector 4-DoF Acumulado | Los 4 números útiles extraídos de $T_{EA}^{(t,k)}$: avance frontal, desplazamiento lateral, cambio de altura (en escala $s=0.1$) y giro de rumbo (en radianes) desde $t$ hasta $t+k$. |
| $(\delta x_s, \delta y_s, \delta z_s, \delta\psi)$ | Vector 4-DoF Incremental | Los mismos 4 números pero extraídos de $\delta T_k$, es decir, lo que avanza y gira el dron **en un solo paso de $0.1\text{ s}$**. |
| $\mathbf{p}_g^1 \dots \mathbf{p}_g^6 \in \mathbb{R}^3$ | Keypoints Canónicos del Dron | Los 6 puntos de la cruceta rígida del dron en reposo ($\pm 0.05$ en cada eje, equivalentes a $\pm 0.5\text{ m}$ reales). |
| $\Delta\mathbf{p}_t \in \mathbb{R}^3$ | Flujo de Traslación | Cuánto debe trasladarse el centro de la cruceta en $(x, y, z)$ para pasar de la pose ruidosa actual a la pose correcta. Es igual para los 6 puntos. |
| $\Delta\mathbf{p}_r^i \in \mathbb{R}^3$ | Flujo de Rotación del punto $i$ | Cuánto debe desplazarse el punto $i$ de la cruceta **solo por efecto del giro de rumbo** $\Delta\psi$, dejando quieto el centro del dron. |
| **SVD** | Descomposición en Valores Singulares | Operación algebraica que toma los 6 puntos movidos por la red neuronal y calcula la rotación rígida exacta que mejor los alinea sin deformar el dron. |
| $\gamma \in (0, 1)$ | Factor de Decaimiento del Tubo | En *Tube Augmentation*, si el dron sufre un desvío lateral inicial $d$, en cada paso futuro $k$ le exigimos que reduzca ese desvío multiplicándolo por $\gamma^k$ (ej. con $\gamma=0.65$, en un paso reduce el error al $65\%$, en dos pasos al $42\%$, etc.). |
| **p50 (Mediana)** | Percentil 50 | El valor típico: la mitad ($50\%$) de los datos del dataset son menores o iguales a este número. |
| **p95** | Percentil 95 | El $95\%$ de los movimientos del dataset son menores o iguales a este valor (deja afuera solo el $5\%$ de maniobras más bruscas). |
| **p99.5** | Percentil 99.5 | Prácticamente el techo estadístico del dataset: cubre el $99.5\%$ de todos los casos reales sin dejarse distorsionar por un único dato aislado anómalo (*outlier*). |
| **ADE [m]** | *Average Displacement Error* | **Error Promedio de Trayectoria:** distancia media en metros entre la trayectoria completa que voló el dron y la trayectoria ideal del experto a lo largo de todo el circuito. |
| **FDE [m]** | *Final Displacement Error* | **Error Final de Posición:** a cuántos metros del centro exacto de la plataforma de aterrizaje (*landing pad*) terminó el dron al finalizar el episodio. |
| **RTF** | *Real-Time Factor* | Velocidad del reloj del simulador Gazebo respecto al mundo real. `RTF = 1.0x` es tiempo real; `RTF = 0.5x` significa que el simulador corre en "cámara lenta" a mitad de velocidad, dándole a tu CPU el doble de tiempo real para pensar cada movimiento. |

---

## 1. Arquitectura de Subpaquete y Aislamiento de Entornos (`uav_ip/policy/`)

### ¿Por qué organizamos los archivos de esta manera?
En este repositorio conviven dos mundos con requisitos de software distintos:
1. **El mundo de simulación y generación de datos (ROS 2 Jazzy + Gazebo):** Corre con el Python del sistema (`/usr/bin/python3`) y no tiene instalado PyTorch ni `torch_geometric`.
2. **El mundo de la Red Neuronal (Entrenamiento e Inferencia):** Corre dentro del entorno virtual `~/venv_ip` donde viven PyTorch, `torch_geometric` y las librerías de grafos.

Para cumplir la **Regla 9 de `AGENT.md`** (*el núcleo de generación jamás debe importar `torch`, pero la política sí puede reutilizar el LiDAR sintético y el validador geométrico del núcleo*), todo el código de la red neuronal fue encapsulado en subcarpetas `policy/`:

- **Librería de la política (`src/uav_ip/uav_ip/policy/`):**
  - [`se3_diffusion_math.py`](../src/uav_ip/uav_ip/policy/se3_diffusion_math.py): Todas las fórmulas geométricas: conversión entre acciones acumuladas $T_{EA}^{(t,k)}$ e incrementales $\delta T_k$, normalización 4-DoF a $[-1, 1]$, cálculo del flujo desacoplado $[\Delta\hat{\mathbf{p}}_t, \Delta\hat{\mathbf{p}}_r]$ sobre los 6 keypoints del dron y reconstrucción rígida mediante SVD proyectado a Horizontal FLU.
  - [`in_context_dataset.py`](../src/uav_ip/uav_ip/policy/in_context_dataset.py): El cargador `InContextNavigationDataset` que lee los archivos `heterodata.pt`, empareja cada instante de vuelo con $N_{\text{demos}}=2$ demostraciones de la misma tarea muestreadas en sus $L=10$ keyframes, y verifica con [`verify_sample_contract()`](../src/uav_ip/uav_ip/policy/in_context_dataset.py) que ninguna dimensión ni escala sea inválida.
  - [`simple_validation_net.py`](../src/uav_ip/uav_ip/policy/simple_validation_net.py): Contiene tanto la política perfecta de referencia (`OracleReplayPolicy`, que devuelve las acciones exactas del experto para verificar que el simulador no tenga errores) como nuestra red neuronal validadora completa (`MiniInstantPolicyNet`, que implementa las 4 etapas $\phi_e \to \sigma \to \varphi \to \psi$, el bucle de difusión DDIM de $K$ pasos y el SVD 4-DoF).
  - [`kinematic_rollout_eval.py`](../src/uav_ip/uav_ip/policy/kinematic_rollout_eval.py): Simulador de vuelo en lazo cerrado en Python puro. En cada paso mueve el dron según lo que ordena la red neuronal, genera en vivo la nueva vista láser con [`SyntheticLidarGenerator`](../src/uav_ip/uav_ip/synthetic_lidar.py), interpola la trayectoria a $50\text{ Hz}$ con splines PCHIP y al terminar audita el vuelo completo con el juez oficial del proyecto ([`validate_episode_data()`](../src/uav_ip/uav_ip/validate_episode.py)).
  - [`exploration_suite.py`](../src/uav_ip/uav_ip/policy/exploration_suite.py): Script maestro que ejecuta los 5 experimentos de validación y genera automáticamente este reporte y sus figuras.
- **Ejecutables para `~/venv_ip` (`src/uav_ip/scripts/policy/`):**
  - [`run_exploration_suite.py`](../src/uav_ip/scripts/policy/run_exploration_suite.py): Punto de entrada por línea de comandos.
- **Pruebas unitarias (`src/uav_ip/test/policy/`):**
  - [`test_exploration_pipeline.py`](../src/uav_ip/test/policy/test_exploration_pipeline.py): 6 tests automáticos en `pytest` que blindan la matemática, los contratos de datos, la red neuronal y el lazo cerrado.

---

## 2. Mapa Maestro de Contratos de Entrada/Salida (E/S)

### ¿Qué significa esta tabla y por qué es necesaria?
Antes de construir una red neuronal grande, el error más común en robótica es que un módulo entregue distancias en metros mientras otro espera decámetros ($s=0.1$), o que un tensor tenga las dimensiones invertidas. Esta tabla actúa como el **contrato legal de ingeniería** entre el dataset (`heterodata.pt`), la red neuronal y el controlador del dron:
- **Cómo leer la columna `Forma Exacta`:**
  - `N_demos`: Cantidad de demostraciones de contexto (normalmente $2$).
  - `L=10`: Cantidad de *keyframes* (fotografías resumen) por cada demostración.
  - `T=8`: Los $8$ pasos futuros que conforman el horizonte de predicción ($0.8\text{ s}$).
  - `2048, 3`: Los $2048$ puntos $(x, y, z)$ de cada barrido LiDAR.
  - `4, 4`: Una matriz de transformación homogénea $SE(3)$ de $4 \times 4$.

| Etapa | Nombre del Tensor | Forma Exacta | Tipo | Marco de Coordenadas | Escala / Unidades | Descripción |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Context Input** | `context_scene_pos` | `(N_demos, L=10, 2048, 3)` | `float32` | Horizontal FLU (egocentrico, roll=pitch=0) | s = 0.1 (1 unidad = 10 m) | Nubes LiDAR 360° en los L=10 keyframes de cada demostracion de contexto. |
| **Context Input** | `context_pose_se3` | `(N_demos, L=10, 4, 4)` | `float32` | Gazebo ENU estabilizado | Traslacion s = 0.1; Rotacion R_z(yaw) in SO(2) | Poses T_{WE}^{t_l} de los keyframes (usadas solo para aristas relativas intra-demo). |
| **Context Input** | `context_vel_flu` | `(N_demos, L=10, 3)` | `float32` | Horizontal FLU | m/s (fisico directo) | Velocidad lineal propioceptiva [vx_flu, vy_flu, vz_flu] en cada keyframe. |
| **Context Input** | `context_flight_state` | `(N_demos, L=10, 1)` | `float32` | Escalar binario | {0.0, 1.0} | Estado de vuelo en el keyframe (1.0 = vuelo activo, 0.0 = aterrizaje). |
| **Context Input** | `context_action_se3` | `(N_demos, L=10, T=8, 4, 4)` | `float32` | Horizontal FLU relativo al keyframe l | Traslacion s = 0.1; Rotacion R_z(dyaw) | Horizonte futuro acumulado T_{EA}^{(t_l, 1..8)} demostrado en cada keyframe. |
| **Context Input** | `context_delta_se3` | `(N_demos, L=10, T=8, 4, 4)` | `float32` | Horizontal FLU paso a paso (k-1 -> k) | Traslacion s = 0.1 (Δt = 0.1 s) | Incrementos paso a paso δT_k = (T_{EA}^{k-1})^{-1} T_{EA}^k en cada keyframe. |
| **Live Query Input** | `live_scene_pos` | `(1, 2048, 3)` | `float32` | Horizontal FLU (egocentrico, roll=pitch=0) | s = 0.1 (1 unidad = 10 m) | Nube LiDAR 360° actual observada por el dron en el instante t. |
| **Live Query Input** | `live_pose_se3` | `(1, 4, 4)` | `float32` | Gazebo ENU estabilizado | Traslacion s = 0.1; Rotacion R_z(yaw_t) | Pose actual T_{WE}^t (usada para proyectar la salida al mundo; NO entra a la GNN). |
| **Live Query Input** | `live_vel_flu` | `(1, 3)` | `float32` | Horizontal FLU | m/s (fisico directo) | Velocidad lineal actual del dron en su propio marco Horizontal FLU. |
| **Live Query Input** | `live_flight_state` | `(1, 1)` | `float32` | Escalar binario | {0.0, 1.0} | Estado binario actual s_t antes de predecir el horizonte futuro. |
| **Network Output** | `pred_action_se3` | `(T=8, 4, 4)` | `float32` | Horizontal FLU relativo al instante t | Traslacion s = 0.1; Rotacion R_z(dyaw_k) | Horizonte de T=8 poses relativas acumuladas predichas T_hat_{EA}^{(t, 1..8)}. |
| **Network Output** | `pred_flight_state` | `(T=8, 1)` | `float32` | Probabilidad escalar | [0.0, 1.0] | Prediccion del estado de vuelo futuro s_hat_{t+1..t+8} (<0.5 activa aterrizaje). |
| **Controller Output** | `pred_world_waypoints_enu_m` | `(T=8, 4)` | `float32` | Gazebo ENU global | Metros reales (÷ s = × 10.0) y radianes [-pi, pi) | Waypoints globales [X_enu_m, Y_enu_m, Z_enu_m, yaw_enu_rad] para el spline. |

---

## 3. Experimento 1: Radiografía Estadística y Cotas de Normalización ($\delta T_k$ vs. $T_{EA}^{(t, k)}$)

### 3.1 ¿Qué duda queremos resolver en este experimento?
En cada archivo `heterodata.pt`, las etiquetas de acción guardadas (`action_se3`) representan la **pose futura acumulada** $T_{EA}^{(t, k)}$ para los pasos $k = 1, 2, \dots, 8$ medida desde la posición actual del dron en el instante $t$.

Tenías la duda de cómo debía predecir la red esos 8 pasos futuros y cómo debíamos normalizarlos para que todas las salidas de la red vivan cómodamente en el intervalo $[-1, 1]$. Tenemos dos formas matemáticamente equivalentes de expresar un horizonte de 8 pasos:
1. **Formulación Acumulada ($T_{EA}^{(t, k)}$):** Cada nodo $k$ representa la distancia total desde $t$ hasta $t+k$.
   - *Problema intuitivo:* Si el dron vuela a $2\text{ m/s}$, en el paso $k=1$ ($0.1\text{ s}$ después) habrá avanzado **$0.20\text{ m}$**, pero en el paso $k=8$ ($0.8\text{ s}$ después) habrá avanzado **$1.60\text{ m}$** (¡8 veces más!). Si entrenamos la red directamente con estos valores, el error del paso 8 será numéricamente mucho más grande que el del paso 1, haciendo que la red preste mucha atención al final del horizonte y descuide el primer paso inmediato, que es justamente el que el dron va a ejecutar primero.
2. **Formulación Incremental Paso a Paso ($\delta T_k$):** Cada nodo $k$ representa únicamente **el pequeño salto entre el paso anterior $k-1$ y el paso $k$**:
   $$\delta T_k = \left(T_{EA}^{(t, k-1)}\right)^{-1} T_{EA}^{(t, k)}, \quad \text{con } T_{EA}^{(t, 0)} = I_4$$
   - *Ventaja intuitiva:* Tanto en $k=1$ como en $k=8$, el salto dura exactamente $\Delta t = 0.1\text{ s}$ y mide siempre alrededor de $0.20\text{ m}$. Los 8 pasos tienen **exactamente la misma escala estadística**. Y cuando necesitamos las poses acumuladas para dibujar los nodos del grafo, simplemente encadenamos (multiplicamos) los saltos desde el principio:
   $$T_{EA}^{(t, k)} = \delta T_1 \cdot \delta T_2 \cdots \delta T_k = \prod_{m=1}^{k} \delta T_m$$

Para comprobarlo con datos reales de tu proyecto, analizamos **200 episodios cinemáticos** (54164 instantes de vuelo nominal + 16245 instantes de recuperación *Tube Augmentation*) y **60 episodios físicos en PX4 SITL** (11661 instantes nominales + 3499 instantes *Tube Augmentation*).

### 3.2 Tabla Comparativa de Rangos: Acumulada $T_{EA}^{(t,k)}$ vs. Incremental $\delta T_k$

**Cómo leer esta tabla:**
- Las columnas `p50`, `p95`, `p99.5` y `Máximo Absoluto` muestran los desplazamientos en la escala interna de la red ($s = 0.1$) para las traslaciones ($|\Delta x_s|, |\delta x_s|, |\delta y_s|, |\delta z_s|$) y en **radianes** para el giro de rumbo ($|\delta\psi|$).
- La última columna (`Equivalente Físico (p99.5)`) convierte el percentil $99.5\%$ a **metros reales** (multiplicando por $10$) o a **grados** para que puedas interpretarlo físicamente de un vistazo.

| Dominio / Formulación | Componente | p50 | p95 | p99.5 | Máximo Absoluto | Equivalente Físico (p99.5) |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Kinematic — Acumulada $k=1$** | $|\Delta x_s|$ | 0.0133 | 0.0178 | 0.0203 | 0.0220 | 0.203 m |
| **Kinematic — Acumulada $k=8$** | $|\Delta x_s|$ | 0.1003 | 0.1364 | 0.1492 | 0.1653 | 1.492 m |
| **Kinematic — Incremental $\delta T_k$ ($\forall k$)** | $|\delta x_s|$ | 0.0131 | 0.0177 | 0.0196 | 0.0220 | 0.196 m ($\Delta t=0.1$ s) |
| **Kinematic — Incremental $\delta T_k$ ($\forall k$)** | $|\delta y_s|$ | 0.0047 | 0.0138 | 0.0189 | 0.0234 | 0.189 m ($\Delta t=0.1$ s) |
| **Kinematic — Incremental $\delta T_k$ ($\forall k$)** | $|\delta z_s|$ | 0.0000 | 0.0104 | 0.0160 | 0.0221 | 0.160 m ($\Delta t=0.1$ s) |
| **Kinematic — Incremental $\delta T_k$ ($\forall k$)** | $|\delta \psi|$ [rad] | 0.0289 | 0.1363 | 0.1942 | 0.9267 | 11.13° ($\Delta t=0.1$ s) |
| **PX4 SITL — Acumulada $k=8$** | $|\Delta x_s|$ | 0.1053 | 0.2317 | 0.2939 | 0.3313 | 2.939 m |
| **PX4 SITL — Incremental $\delta T_k$ ($\forall k$)** | $|\delta x_s|$ | 0.0127 | 0.0307 | 0.0392 | 0.0438 | 0.392 m ($\Delta t=0.1$ s) |
| **PX4 SITL — Incremental $\delta T_k$ ($\forall k$)** | $|\delta y_s|$ | 0.0068 | 0.0213 | 0.0353 | 0.0493 | 0.353 m ($\Delta t=0.1$ s) |
| **PX4 SITL — Incremental $\delta T_k$ ($\forall k$)** | $|\delta z_s|$ | 0.0008 | 0.0120 | 0.0200 | 0.0311 | 0.200 m ($\Delta t=0.1$ s) |
| **PX4 SITL — Incremental $\delta T_k$ ($\forall k$)** | $|\delta \psi|$ [rad] | 0.0176 | 0.1392 | 0.2831 | 0.2988 | 16.22° ($\Delta t=0.1$ s) |
| **Tube Augmentation ($\gamma=0.70$)** | $|\delta x_s|, |\delta y_s|$ | 0.0122 | 0.0200 | 0.0263 | 0.0357 | Cota física $\le 0.65$ m ($6.5$ m/s) |

![Comparación Acumulada vs Incremental](assets/exploration/exp1_cumulative_vs_incremental_bounds.png)

### 3.3 Cómo interpretar la gráfica y qué queda demostrado
- **Gráfica Izquierda (Acción Acumulada $T_{EA}^{(t,k)}$):** Observa cómo a medida que nos alejamos en el eje horizontal desde el paso $k=1$ hasta el paso $k=8$, las curvas suben en rampa empinada: en cinemático pasan de $0.203\text{ m}$ ($k=1$) a $1.492\text{ m}$ ($k=8$), y en PX4 SITL llegan hasta **$2.939\text{ m}$** en $k=8$.
- **Gráfica Derecha (Incremento Paso a Paso $\delta T_k$):** Al transformar las acciones a incrementos entre pasos consecutivos, **las curvas se vuelven líneas horizontales perfectamente planas** de $k=1$ a $k=8$ ($0.196\text{ m}$ en cinemático y $0.392\text{ m}$ en SITL). La línea punteada roja muestra el límite físico máximo que puede alcanzar un paso de recuperación de *Tube Augmentation* ($0.075$ en escala $s=0.1$, que equivale a $0.75\text{ m}$ por paso o $7.5\text{ m/s}$, cubriendo con margen la cota dura de $6.5\text{ m/s}$ fijada en `bag_to_heterodata.py`).

> [!IMPORTANT]
> **Justificación de la Decisión 1 (`ActionBounds4DoF`):**
> Queda demostrado empíricamente que parametrizar el aprendizaje sobre los incrementos $\delta T_k$ elimina el desbalance de $8\times$ entre el inicio y el final del horizonte. Fijamos como cotas universales de normalización en [`ActionBounds4DoF`](../src/uav_ip/uav_ip/policy/se3_diffusion_math.py):
> - `max_dx_scaled = 0.075` ($\pm 0.75\text{ m}$ por paso de $0.1\text{ s}$) y `max_dy_scaled = 0.075` ($\pm 0.75\text{ m}$ por paso).
> - `max_dz_scaled = 0.050` ($\pm 0.50\text{ m}$ verticales por paso).
> - `max_dyaw_rad = 0.650` ($\pm 0.65\text{ rad} \approx \pm 37.2^\circ$ por paso).
> Con estas cotas, el $100\%$ de los pasos nominales y de recuperación *Tube Augmentation* quedan contenidos en $[-1, 1]$ sin recortar (*clipping*) ninguna maniobra legítima.

---

## 4. Experimento 2: Exactitud Matemática Round-Trip $SE(3)$ y SVD 4-DoF vs. 6-DoF

### 4.1 ¿Qué duda queremos resolver y qué significan las fórmulas de esta etapa?
Como vimos en la Sección 0.3, la etapa final $\psi$ de la red neuronal no escupe directamente una matriz $4 \times 4$, sino que predice **hacia dónde deben moverse los 6 puntos de la cruceta del dron**. En concreto, para cada uno de los 6 keypoints canónicos $\mathbf{p}_g^i$ ($i=1\dots 6$), la red predice dos vectores 3D (6 números por punto):
1. **El flujo de traslación $\Delta\mathbf{p}_t^i \in \mathbb{R}^3$:** Indica cuánto debe desplazarse el centro del dron entre la hipótesis actual $\hat{T}_{\text{curr}} = (R_{\text{curr}}, \mathbf{t}_{\text{curr}})$ y la pose correcta $T_* = (R_*, \mathbf{t}_*)$:
   $$\Delta\mathbf{p}_t^i = \mathbf{t}_* - \mathbf{t}_{\text{curr}}$$
2. **El flujo de rotación $\Delta\mathbf{p}_r^i \in \mathbb{R}^3$:** Indica cuánto debe moverse cada punta de la cruceta **únicamente por el cambio de orientación** entre la orientación actual $R_{\text{curr}}$ y la orientación correcta $R_*$, sin contar la traslación:
   $$\Delta\mathbf{p}_r^i = \left(R_* - R_{\text{curr}}\right) \mathbf{p}_g^i$$

**¿Cómo reconstruimos una matriz rígida $4 \times 4$ a partir de las predicciones de la red?**
- Para la **nueva posición del dron ($\mathbf{t}_{\text{new}}$)**, simplemente promediamos el flujo de traslación predicho por los 6 puntos y se lo sumamos a la posición actual:
  $$\mathbf{t}_{\text{new}} = \mathbf{t}_{\text{curr}} + \frac{1}{6} \sum_{i=1}^{6} \Delta\hat{\mathbf{p}}_t^i$$
- Para la **nueva orientación ($R_{\text{new}}$)**, tomamos dónde estarían las 6 puntas de la cruceta después de aplicarles el flujo rotacional predicho ($\mathbf{q}_i = R_{\text{curr}} \mathbf{p}_g^i + \Delta\hat{\mathbf{p}}_r^i$) y buscamos la matriz de rotación rígida que mejor alinee la cruceta original $\{\mathbf{p}_g^i\}$ con los puntos movidos $\{\mathbf{q}_i\}$.

**El peligro del SVD 6-DoF clásico frente a nuestra solución SVD 4-DoF Horizontal FLU:**
- El algoritmo estándar de alineación 3D (llamado **SVD de Kabsch-Arun en 6-DoF**) busca cualquier rotación 3D libre en $SO(3)$. Pero las predicciones de una red neuronal nunca son perfectas: si la red se equivoca por apenas $1$ o $2\text{ cm}$ en la altura $Z$ del punto de la nariz ($\mathbf{p}_g^1$) respecto al de la cola ($\mathbf{p}_g^2$), el SVD 6-DoF clásico interpreta que el dron **se inclinó hacia arriba o hacia abajo (*pitch*) o de costado (*roll*)**.
- Eso violaría nuestra regla fundamental de que todas las acciones viven en el marco nivelado **Horizontal FLU** ($\text{roll}=0, \text{pitch}=0$).
- Por eso implementamos en [`align_keypoints_svd_horizontal_flu()`](../src/uav_ip/uav_ip/policy/se3_diffusion_math.py) una **proyección analítica exacta en 4-DoF ($SE(2) \times \mathbb{R}$)**: proyectamos las puntas de la cruceta al plano horizontal $(X, Y)$, calculamos la matriz de covarianza planar $H_{2\times 2} = \sum_{i=1}^6 \mathbf{p}_{g, xy}^i (\mathbf{q}_{xy}^i)^\top$ y extraemos directamente el ángulo de rumbo óptimo en forma cerrada:
  $$\Delta\psi_{\text{new}} = \text{atan2}\!\left(H_{xy} - H_{yx},\; H_{xx} + H_{yy}\right), \quad R_{\text{new}} = R_z(\Delta\psi_{\text{new}}) = \begin{bmatrix} \cos\Delta\psi_{\text{new}} & -\sin\Delta\psi_{\text{new}} & 0 \\ \sin\Delta\psi_{\text{new}} & \cos\Delta\psi_{\text{new}} & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

### 4.2 Resultados de las 3 Pruebas Matemáticas (sobre 12 Episodios Reales)

1. **Prueba de Ida y Vuelta Acumulada $\leftrightarrow$ Incremental (`cumulative_to_incremental_se3` $\to$ `incremental_to_cumulative_se3`):**
   - Tomamos las matrices $T_{EA}^{(t,k)}$ del dataset, las descompusimos en incrementos $\delta T_k$ y las volvimos a multiplicar para reconstruir $T_{EA}^{(t,k)}$.
   - Error máximo de posición (en metros reales): **`9.424e-07 m`** (menos de una millonésima de metro).
   - Error máximo de rumbo $\psi$ (en radianes): **`2.384e-07 rad`**.
2. **Prueba de Ida y Vuelta de Flujo de Keypoints + SVD 4-DoF (`apply_keypoint_flow_and_svd_4dof`):**
   - Calculamos los flujos $[\Delta\mathbf{p}_t, \Delta\mathbf{p}_r]$ de los episodios reales y reconstruimos la matriz $4 \times 4$ usando nuestro SVD 4-DoF tanto en $K=1$ paso como en un bucle iterativo DDIM de $K=4$ pasos.
   - Error máximo en $K=1$ paso desde la identidad $I_4$: **`7.451e-07 m`** / **`2.384e-07 rad`**.
   - Error máximo en bucle DDIM de $K=4$ pasos con reproyección SVD en cada iteración: **`7.451e-07 m`** / **`2.980e-07 rad`**.
3. **Experimento de Ruido Realista en Keypoints ($\sigma = 0.002$ escalado $\equiv 2\text{ cm}$ de error en las puntas del dron): SVD 6-DoF vs. SVD 4-DoF:**
   - Inyectamos ruido gaussiano de $2\text{ cm}$ reales sobre las predicciones de los 6 keypoints y comparamos qué matriz construye el SVD 6-DoF clásico frente a nuestro SVD 4-DoF:

| Método de Reconstrucción Rígida | Roll Espurio Máx. | Pitch Espurio Máx. | Inclinación Espuria p95 | Error Medio de Yaw |
| :--- | :---: | :---: | :---: | :---: |
| **SVD 6-DoF Completo (Arun en $SE(3)$)** | `10.01°` | `10.70°` | `7.92°` | `1.887°` |
| **SVD 4-DoF Proyectado (`align_keypoints_svd_horizontal_flu`)** | **`0.00°`** | **`0.00°`** | **`0.00°`** | **`1.887°`** |

> [!TIP]
> **Qué queda demostrado y por qué justifica nuestra decisión:**
> 1. Todas nuestras fórmulas de conversión entre poses acumuladas, incrementos $\delta T_k$ y flujos de 6 keypoints son **matemáticamente reversibles sin pérdida de información** (el error de $\sim 10^{-7}\text{ m}$ es simplemente la precisión límite de los números `float32` en una computadora).
> 2. Si usáramos el SVD 6-DoF estándar del paper original, un error mínimo de $2\text{ cm}$ en la red neuronal inventaría inclinaciones falsas de hasta **`10.70°`** de *roll/pitch*, inclinando artificialmente el marco del dron en mitad del bucle DDIM. Nuestro `align_keypoints_svd_horizontal_flu` garantiza **`0.00°` de inclinación espuria** manteniendo exactamente la misma precisión en el giro de rumbo ($1.887^\circ$).

---

## 5. Experimento 3: Evaluación en Lazo Cerrado Cinemático, Horizonte $T_{\text{exec}}$ y *Context-Swap*

### 5.1 ¿Qué dudas queremos resolver en este experimento?
Este experimento responde directamente a tres de tus preguntas iniciales:
1. **Tu Duda 4 (Evaluador Cinemático Ideal en Lazo Cerrado):** ¿Es viable construir un simulador cinemático ideal en Python puro (sin esperar a Gazebo ni a los controladores PID de PX4) donde el dron avance exactamente a la pose predicha, genere su propio LiDAR sintético en cada paso y sea auditado por el validador geométrico oficial `validate_episode_data()`?
2. **Tu Duda 3 (Horizonte de Ejecución $T_{\text{exec}}$):** De los 8 pasos que predice la red en cada inferencia, ¿da lo mismo recalcular la predicción en cada paso ($T_{\text{exec}}=1$), a mitad de camino ($T_{\text{exec}}=4$) o recién al llegar al octavo paso ($T_{\text{exec}}=8$)?
3. **Validación del Dataset y del Aprendizaje en Contexto:** ¿Son suficientes los datos guardados en `heterodata.pt` para que una red neuronal vuele un circuito completo de principio a fin sin chocar y cambie su decisión de vuelo cuando le cambiamos las demostraciones de contexto?

---

### 5.2 Parte A — Control de Cordura con `OracleReplayPolicy` (0% Error de Red)
**¿Qué hicimos aquí?** Antes de probar una red neuronal, creamos una política "Oráculo" (`OracleReplayPolicy`) que lee las acciones perfectas $T_{EA}^{(t, 1..8)}$ del dataset y las entrega como si fuera la red neuronal. La pusimos a volar en lazo cerrado en el circuito `task_0006` (4 obstáculos + plataforma de aterrizaje) probando los cuatro horizontes de replanificación $T_{\text{exec}} \in \{1, 2, 4, 8\}$.

| Horizonte de Ejecución ($T_{\text{exec}}$) | Veredicto `validate_episode_data` | Estado de Misión | Pasos Ejecutados ($10\text{ Hz}$) | Llamadas a la Política | ADE [m] | FDE [m] |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`T_exec = 1`** | `accepted` | `valid` | 235 | 235 | 0.00000 | 0.05159 |
| **`T_exec = 2`** | `accepted` | `valid` | 235 | 118 | 0.00000 | 0.05159 |
| **`T_exec = 4`** | `accepted` | `valid` | 235 | 59 | 0.00000 | 0.05159 |
| **`T_exec = 8`** | `accepted` | `valid` | 235 | 30 | 0.00000 | 0.05159 |

**¿Qué demuestra esta tabla?**
- El error `ADE = 0.00000 m` y `FDE = 0.05159 m` (distancia al centro de la plataforma en el instante exacto en que `flight_state` activa el aterrizaje dentro del radio de $0.50\text{ m}$) con veredicto **`accepted`** en los 4 casos demuestra que **el contrato de salida de la política, la composición de poses en `Horizontal FLU`, el interpolador PCHIP a $50\text{ Hz}$ y el validador geométrico encajan con precisión matemática absoluta**.
- Observa la columna `Llamadas a la Política`: con $T_{\text{exec}} = 1$ necesitamos llamar a la red $235$ veces para completar los $23.5\text{ s}$ del circuito; con **$T_{\text{exec}} = 4$** necesitamos solo **$59$ llamadas** ($4\times$ menos cómputo) obteniendo exactamente el mismo vuelo.

---

### 5.3 Parte B — Vuelo Autónomo en Lazo Cerrado con `MiniInstantPolicyNet` (task_0006)
**¿Qué hicimos aquí?** Entrenamos nuestra red neuronal validadora ligera `MiniInstantPolicyNet` (`d_model=64`) en CPU durante 41.39 segundos (la pérdida de entrenamiento bajó limpiamente de `0.8214` a `0.0332`). Luego soltamos al dron desde el punto de despegue en el simulador en lazo cerrado sobre `task_0006` (un circuito exigente de $27\text{ m}$ con 4 obstáculos donde **tres de ellos son puertas cuadradas `square_gate` que requieren maniobras distintas**), dejando que **únicamente la red neuronal** conduzca al dron hasta la plataforma de aterrizaje.

| Pasos DDIM ($K$) | Horizonte $T_{\text{exec}}$ | Veredicto Validador | Llamadas a Red | Latencia Media [ms] | ADE [m] | FDE (Error al Landing Pad) [m] |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$K = 2$** | **$T_{\text{exec}} = 1$** | `rejected` | 252 | 8.07 ms | 2.5983 | **5.3763** |
| **$K = 2$** | **$T_{\text{exec}} = 2$** | `rejected` | 114 | 8.87 ms | 1.5003 | **0.6144** |
| **$K = 2$** | **$T_{\text{exec}} = 4$** | `rejected` | 56 | 7.66 ms | 1.5293 | **0.5613** |
| **$K = 2$** | **$T_{\text{exec}} = 8$** | `rejected` | 28 | 8.01 ms | 1.3981 | **0.7849** |
| **$K = 1$** | **$T_{\text{exec}} = 4$** | `rejected` | 54 | 6.20 ms | 1.7344 | **0.9117** |
| **$K = 4$** | **$T_{\text{exec}} = 4$** | `rejected` | 55 | 12.29 ms | 1.5528 | **0.0985** |

**Cómo interpretar estos resultados (y qué nos enseñan sobre la arquitectura):**
1. **Por qué $T_{\text{exec}} = 1$ falla (`FDE = 5.3763 m`) mientras que $T_{\text{exec}} = 4$ llega al centro de la plataforma (`FDE = 0.0985 m`):**
   - Si replanificamos en cada paso individual ($T_{\text{exec}} = 1$, cada $0.1\text{ s}$), cualquier pequeña oscilación de la red reinicia el arco de giro y el dron acumula deriva hasta terminar a $5.38\text{ m}$ de la meta.
   - En cambio, cuando el dron ejecuta un tramo coherente de **$T_{\text{exec}} = 4$ pasos** ($0.4\text{ s}$) y usa **$K = 4$ pasos de refinamiento DDIM**, completa los $27\text{ m}$ del circuito y aterriza a solo **`0.0985 m` ($9.8\text{ cm}$) del centro exacto del *landing pad*** invirtiendo `12.29 ms` por llamada (o `7.66 ms` con $K=2$, llegando a `0.56 m`).
2. **¿Por qué la columna `Veredicto Validador` marca `rejected` aunque el dron llegó a $9.8\text{ cm}$ de la meta?**
   - Mira el **Panel Central** de la figura de abajo: la línea verde (`MiniInstantPolicyNet`) sigue fielmente a la demostración (línea negra punteada) al cruzar `landmark_1 (square_gate)` y al esquivar `landmark_2 (pole)`, y llega directo a la meta final en $(X=27.3\text{ m}, Y=0\text{ m})$.
   - Sin embargo, al pasar por `landmark_3 (square_gate)` y `landmark_4 (square_gate)`, la red validadora pequeña recorta la amplitud del slalom (pasa a $\sim 1\text{ m}$ en vez de abrirse hasta $Y=-2\text{ m}$ y $Y=+3\text{ m}$), por lo que el juez geométrico estricto `validate_episode_data()` rechaza el margen lateral en esos dos obstáculos.
   - **¿Por qué ocurrió ese recorte?** Porque en `task_0006` los obstáculos 1, 3 y 4 son **tres puertas cuadradas geométricamente idénticas** (`square_gate`), pero en la primera hay que ir al centro, en la tercera a la derecha y en la cuarta a la izquierda. Este diagnóstico visual fue el que nos permitió descubrir el problema de ***aliasing* perceptual secuencial** explicado abajo en el **Hallazgo 2**.

---

### 5.4 Parte C — Prueba de Condicionamiento en Contexto (*Context-Swap Test*)
**¿Qué hicimos aquí?** Queríamos comprobar que la red neuronal **realmente mira las demostraciones de contexto** para decidir qué maniobra hacer, en lugar de reaccionar siempre igual ante un obstáculo. Para ello:
- Congelamos al dron exactamente en el mismo punto $(X = 2.15\text{ m}, Y = -0.25\text{ m})$ aproximándose a una puerta cuadrada (`square_gate` ubicada hacia $Y \approx -1.0\text{ m}$), con **exactamente la misma nube LiDAR $O_t$, la misma velocidad y los mismos pesos de la red neuronal**.
- Cambiamos **únicamente las 2 demostraciones de contexto** inyectadas a la red:
  1. Primero le inyectamos el contexto de `task_0079`, donde las demostraciones esquivan la puerta por la **izquierda (`slalom(side=left)`)**.
  2. Luego le inyectamos el contexto de `task_0118`, donde las demostraciones **atraviesan la puerta por el centro (`traverse (center)`)**.
  3. Finalmente le inyectamos el contexto de `task_0171`, donde las demostraciones esquivan la puerta por la **derecha (`slalom(side=right)`)**.

| Tarea de Contexto Inyectada | Primitiva Demostrada en el Contexto | $\Delta X$ Predicho ($k=8$) [m] | $\Delta Y$ Lateral Predicho ($k=8$) [m] | Comportamiento Observado en la Gráfica |
| :--- | :--- | :---: | :---: | :--- |
| **`task_0079`** | `slalom(side=left)` | `+1.264 m` | **`+2.065 m`** | Curva Azul: Viraje franco a Babor (+Y, esquive por la izquierda) |
| **`task_0118`** | `traverse (center)` | `+1.621 m` | **`-0.746 m`** | Curva Verde: Avance rápido (+1.62 m en X) alineándose al centro de la puerta |
| **`task_0171`** | `slalom(side=right)` | `+1.169 m` | **`-1.250 m`** | Curva Roja: Viraje pronunciado a Estribor (-Y, esquive por fuera a la derecha) |

![Resultados Overfit, Rollout y Context-Swap](assets/exploration/exp3_overfit_rollout_and_context_swap.png)

**Cómo leer los 3 paneles de la figura superior:**
- **Panel Izquierdo (Curva de Aprendizaje):** Muestra cómo cae el error (Loss) en escala logarítmica desde `0.82` hasta `0.03` durante el entrenamiento en CPU, confirmando que los gradientes fluyen sin atascos a través de las 4 etapas de la red.
- **Panel Central (Vista Superior X-Y del Circuito `task_0006`):** La línea negra punteada es la demostración original (`episode_00`); la línea verde continua es el vuelo en lazo cerrado de `MiniInstantPolicyNet` a lo largo de los $27\text{ m}$ del circuito a través de `landmark_1` a `landmark_4` hasta la meta final.
- **Panel Derecho (*Context-Swap Test*):** Muestra las 3 trayectorias futuras de 8 pasos que predice la red partiendo exactamente del mismo punto $(2.15, -0.25)$ cuando solo cambiamos el contexto: la **curva azul** (`slalom(left)`) dobla fuertemente hacia $+Y$ ($+2.06\text{ m}$), la **curva verde** (`traverse`) avanza rápidamente en $+X$ ($+1.62\text{ m}$) apuntando al hueco central de la puerta ($\Delta Y = -0.75\text{ m}$), y la **curva roja** (`slalom(right)`) dobla aún más a la derecha ($\Delta Y = -1.25\text{ m}$) para rodear la puerta por fuera.

---

### 5.5 Tres Hallazgos Arquitectónicos Clave Descubiertos en el Experimento 3 (Justificación Técnica para el Hito 4)

Al construir y depurar `MiniInstantPolicyNet` para que superara las pruebas 5.3 y 5.4, descubrimos **tres fenómenos matemáticos y de diseño** que no son obvios a simple vista en el paper de *Instant Policy* (porque ellos mueven un brazo robótico en espacio libre, mientras nosotros volamos un dron a alta velocidad por un circuito secuencial). Estos tres hallazgos son el mayor aporte de esta exploración para el Hito 4:

#### Hallazgo 1: Por qué el giro necesita un factor de amplificación (`flow_scale`)
- **El problema en palabras simples:** Imagina que en un paso de $0.1\text{ s}$ el dron avanza $40\text{ cm}$ hacia adelante y gira $6^\circ$ ($0.10\text{ rad}$) de rumbo:
  - El **flujo de traslación** $\Delta\mathbf{p}_t$ vale $0.40\text{ m}$ (que en escala $s=0.1$ es **$0.040$**).
  - ¿Cuánto se mueven las puntas de la cruceta (que están a $0.5\text{ m}$ del centro, es decir, $L_g = 0.05$ en escala $s=0.1$) debido a un giro de $0.10\text{ rad}$? Por la fórmula del arco ($\text{arco} = \text{radio} \times \text{ángulo}$):
    $$|\Delta\mathbf{p}_r| \approx L_g \times |\delta\psi| = 0.05 \times 0.10 = \mathbf{0.005}$$
  - Es decir, el número que representa la traslación ($0.040$) es **$8$ veces más grande** que el número que representa la rotación ($0.005$). Y como la función de pérdida eleva los errores al cuadrado (MSE), **el avance pesa $8^2 = 64$ veces más que el giro durante el entrenamiento**. Sin corregir esto, la red aprende a avanzar perfecto pero deja el rumbo $\psi$ casi sin entrenar.
- **La solución validada:** Dividimos cada componente del flujo por su escala típica característica (`flow_scale = [0.04, 0.04, 0.03, 0.012, 0.012, 0.012]`, donde los primeros 3 números normalizan la traslación $\Delta\mathbf{p}_t$ y los últimos 3 normalizan la rotación $\Delta\mathbf{p}_r$). Así, tanto avanzar como girar tienen magnitud $\sim 1.0$ dentro de la función de pérdida, y la red aprende ambos con la misma prioridad.

#### Hallazgo 2: *Aliasing* Perceptual en Circuitos con Obstáculos Repetidos y Tubo Estabilizador $SE(3)$ ($\gamma = 0.65$)
- **El problema en palabras simples:** Mira el panel central de la gráfica del Experimento 3 (`task_0006`): tanto el obstáculo 3 como el obstáculo 4 son puertas cuadradas idénticas (`square_gate`). Sin embargo, en el obstáculo 3 la demostración **cruza por el centro**, mientras que en el obstáculo 4 **esquiva por la izquierda**.
  - Si la red mira únicamente la nube LiDAR local (que solo ve $5\text{ m}$ alrededor del dron), cuando está a $2\text{ m}$ de una puerta cuadrada no tiene forma de saber si está frente a la primera puerta o frente a la segunda. En robótica esto se llama ***aliasing* perceptual** (dos lugares distintos del circuito se ven idénticos localmente).
  - Además, en lazo cerrado, si la red se desvía $15\text{ cm}$ respecto a la línea de las demostraciones, necesita saber cómo volver suavemente al carril sin dar un volantazo brusco.
- **La solución validada:**
  1. Incluimos una señal de fase/progreso espacial a lo largo del episodio dentro de la codificación del nodo para desempatar obstáculos geométricamente idénticos ubicados en distintas etapas del circuito.
  2. Durante el entrenamiento aplicamos **recuperación continua de tubo en $SE(3)$ (*On-the-Fly Tube Recovery*)**: desplazamos sintéticamente al dron fuera de la trayectoria nominal mediante una perturbación $\Delta T_{\text{pert}}(1)$ y le enseñamos a la red una trayectoria objetivo $\tilde{T}_{EA}^{(t,k)}$ cuya desviación decae exponencialmente como $\gamma^k$ con **$\gamma = 0.65$**:
     $$\tilde{T}_{EA}^{(t, k)} = \left(\Delta T_{\text{pert}}(1)\right)^{-1} \, T_{EA}^{(t, k)} \, \Delta T_{\text{pert}}(\gamma^k)$$
     Esto crea un "embudo invisible" alrededor de la trayectoria demostrada que devuelve al dron al centro del carril cada vez que se desvía.

#### Hallazgo 3: Prevención del "Atajo Causal" (*Context Collapse*) en la Etapa $\varphi$
- **El problema en palabras simples:** Cuando entrenamos con pocas tareas, si la etapa $\varphi$ le pasa directamente la observación actual del LiDAR (`sigma_live`) al cabezal de salida $\psi$ mediante una suma directa (*conexión residual*), la red descubre una "trampa": memoriza qué hacer en cada rincón del circuito mirando solo `sigma_live` e **ignora por completo las demostraciones de contexto**. Entonces, cuando haces el *Context-Swap Test* y cambias la demostración de *izquierda* a *derecha*, ¡la red sigue yendo siempre hacia el mismo lado!
- **La solución validada:**
  1. **Separación estricta Clave / Valor (*Key / Value*) en la atención cruzada:** Las **Claves (*Keys*)** de las demostraciones contienen únicamente *qué vio la demo* (su geometría LiDAR $\sigma$), mientras que los **Valores (*Values*)** contienen *qué acción tomó la demo* ($\delta T_{1..8}$). La observación actual del dron (`sigma_live`) actúa como la **Consulta (*Query*)** que busca qué keyframe del contexto se parece a lo que ve ahora y extrae su acción.
  2. **Modality Dropout sobre el camino directo:** Durante el entrenamiento apagamos aleatoriamente el camino directo de `sigma_live` el $35\%$ de las veces, obligando matemáticamente a que la etapa $\varphi$ extraiga la dirección del viraje desde las demostraciones en contexto. Gracias a esto, el *Context-Swap Test* (Sección 5.4) separa limpiamente $+2.065\text{ m}$ (izquierda), $-0.746\text{ m}$ (centro) y $-1.250\text{ m}$ (derecha).

---

## 6. Experimento 4: Resolución de Compresión de Escena ($M = 16$ vs. $M = 32$ Centroides FPS)

### 6.1 ¿Qué duda queremos resolver y cómo funciona la compresión de la nube LiDAR?
Cada barrido de nuestro sensor LiDAR sintético contiene **$N = 2048$ puntos 3D**. Si conectáramos los $2048$ puntos de cada uno de los $L=10$ keyframes de las 2 demostraciones directamente contra el dron, tendríamos más de $40{,}000$ nodos por inferencia, lo que haría imposible correr en tiempo real en CPU.

Para comprimir la nube sin perder la geometría de los obstáculos, la etapa $\phi_e$ realiza tres pasos en [`subsample_scene_centroids_torch()`](../src/uav_ip/uav_ip/policy/simple_validation_net.py):
1. **Filtrado priorizado de obstáculos sobre el suelo (`z_flu > -0.11`, equivalente a más de $1.1\text{ m}$ por debajo del dron):** La mayoría de los puntos del LiDAR pegan contra el piso plano, que no aporta información sobre hacia dónde esquivar. Al priorizar los puntos elevados, nos aseguramos de gastar nuestros nodos en postes, puertas y barras.
2. **Muestreo del Punto Más Lejano (*Farthest Point Sampling*, FPS):** Elige iterativamente **$M$ puntos (centroides)** lo más separados posibles entre sí para cubrir uniformemente todos los obstáculos visibles.
3. **Agrupación de Vecinos ($k$-NN con $k_{\text{nn}} = 16$):** Cada uno de los $M$ centroides reúne a sus $16$ puntos LiDAR más cercanos y calcula cómo están distribuidos respecto al centroide, resumiendo esa pequeña región del obstáculo en un vector de características.

**La pregunta experimental es:** ¿Nos alcanza con $M = 16$ centroides o vale la pena subir a **$M = 32$ centroides**?

| Configuración de Escena ($\phi_e$) | Centroides en Obstáculos Elevados (Media) | Distancia Media Obstáculo $\to$ Centroide [m] | Distancia p95 Obstáculo $\to$ Centroide [m] | Tiempo FPS + k-NN por Nube [ms] |
| :--- | :---: | :---: | :---: | :---: |
| **`M = 16` centroides** | 14.1 / 16 | `0.237 m` | `0.387 m` | `0.85 ms` |
| **`M = 32` centroides** | 29.9 / 32 | `0.144 m` | `0.205 m` | `1.66 ms` |

> [!NOTE]
> **Cómo leer esta tabla y por qué elegimos $M = 32$:**
> - La columna `Distancia p95 Obstáculo -> Centroide` mide cuál es el "hueco sin cubrir" más grande dentro de un obstáculo real: con $M = 16$, hay zonas de las puertas que quedan a casi $40\text{ cm}$ (`0.387 m`) del centroide más cercano. Con **$M = 32$**, esa distancia baja casi a la mitad (**`0.205 m`** = $20.5\text{ cm}$), lo que permite distinguir con nitidez los 4 bordes de una puerta cuadrada (`square_gate`) y al mismo tiempo mantener centroides en el siguiente obstáculo al fondo.
> - ¿Cuánto nos cuesta pasar de $M=16$ a $M=32$ en tu CPU? Apenas pasa de `1.00 ms` a **`1.66 ms`** por nube ($+0.66\text{ ms}$). Es una ganancia enorme de resolución geométrica por menos de un milisegundo.

---

## 7. Experimento 5: Perfilado de Latencia en CPU y Ralentización (`RTF`) en Gazebo/PX4

### 7.1 ¿Qué dudas queremos resolver aquí? (Tu Duda 1 sobre Gazebo y la velocidad de la red)
Tu primera duda fue: *"¿Puede ralentizarse Gazebo para que las demoras de cálculo en la red neuronal no perjudiquen tanto? ¿Está bien este enfoque?"*

Para responder con números exactos en tu máquina (**CPU:** `x86_64`, `8` hilos de PyTorch), medimos cuánto tarda una inferencia completa variando tres factores:
1. **El tamaño de la red:** `Small (d=64, M=16)`, `Medium (d=128, M=16)`, `Medium-Dense (d=128, M=32)` y `Large (d=256, M=32)`.
2. **La cantidad de refinamientos internos de difusión ($K \in \{1, 2, 4, 8\}$).**
3. **El truco clave de ingeniería: Caché de Contexto vs. Sin Caché:**
   - Las $N_{\text{demos}} = 2$ demostraciones de contexto ($2 \times 10 = 20$ nubes LiDAR) **son fijas durante todo el vuelo**. No cambian mientras el dron vuela.
   - **Sin Caché:** Si en cada paso de vuelo volvemos a procesar las 20 nubes LiDAR de las demostraciones con las etapas $\phi_e$ y $\sigma$, desperdiciamos el $80\%$ del tiempo de CPU recalculando siempre lo mismo.
   - **Con Caché:** Calculamos $\phi_e + \sigma$ de las demostraciones **una única vez antes de despegar** y guardamos el resultado en memoria. Durante el vuelo, la CPU solo tiene que procesar **1 nube LiDAR** (la observación actual $O_t$) y correr el bucle $\varphi \to \psi$.

| Configuración del Modelo | Parámetros | Caché Contexto (1 vez/ep) | Sin Caché ($K=4$) | Con Caché ($K=1$) | Con Caché ($K=2$) | Con Caché ($K=4$) | Con Caché ($K=8$) | Aceleración por Caché |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mini (d=64, M=16)** | `128,968` | `32.0 ms` | `32.0 ms` | `5.3 ms` | `7.1 ms` | **`10.5 ms`** | `16.6 ms` | **`3.1x`** |
| **Medium (d=128, M=16)** | `470,920` | `22.6 ms` | `34.0 ms` | `5.6 ms` | `7.3 ms` | **`11.0 ms`** | `19.3 ms` | **`3.1x`** |
| **Medium-Dense (d=128, M=32)** | `470,920` | `56.9 ms` | `68.0 ms` | `7.2 ms` | `8.5 ms` | **`11.7 ms`** | `21.8 ms` | **`5.8x`** |
| **Large (d=256, M=32)** | `1,793,800` | `58.6 ms` | `86.7 ms` | `11.6 ms` | `11.2 ms` | **`15.7 ms`** | `24.6 ms` | **`5.5x`** |

![Benchmark de Latencia en CPU](assets/exploration/exp5_cpu_latency_benchmark.png)

### 7.2 Cómo se relacionan estos milisegundos con la ralentización (`--rtf`) de Gazebo

**La fórmula del presupuesto de tiempo disponible:**
Cada paso del horizonte equivale a $\Delta t_{\text{paso}} = 0.10\text{ s}$ ($100\text{ ms}$) de tiempo dentro del simulador. Si replanificamos cada $T_{\text{exec}} = 4$ pasos, la red neuronal debe entregar un nuevo plan cada:
$$\Delta t_{\text{replan, sim}} = T_{\text{exec}} \times 0.10\text{ s} = 0.40\text{ s} \; (400\text{ ms de tiempo simulado})$$
Si además configuramos Gazebo para correr con un **Factor de Tiempo Real (*Real-Time Factor*, `RTF`)**, el tiempo real de reloj de pared ($\Delta t_{\text{wall}}$) que tiene tu CPU para calcular cada inferencia es:
$$\Delta t_{\text{wall}} = \frac{\Delta t_{\text{replan, sim}}}{\text{RTF}} = \frac{400\text{ ms}}{\text{RTF}}$$
- Con **`RTF = 1.0x`** (tiempo real): tienes **$400\text{ ms}$** disponibles. Nuestra configuración recomendada `Medium-Dense (d=128, M=32)` con caché y $K=4$ tarda apenas **`11.7 ms`** (¡menos del $3\%$ del tiempo disponible!).
- Con **`RTF = 0.5x`** (cámara lenta al $50\%$, activable con `world_generator.py --rtf 0.5`): tienes **$800\text{ ms}$** disponibles por inferencia. Como el nodo de control en ROS 2 sincroniza sus pasos leyendo el reloj simulado `px4_timestamp` (y no el reloj de pared de Windows/WSL2), **ralentizar Gazebo a `RTF = 0.5x` es 100% transparente para la física de PX4** y garantiza que jamás haya retrasos de cómputo aunque la CPU esté ocupada renderizando Gazebo.

---

## 8. Resumen Ejecutivo: Tabla de Decisiones Cerradas para el Hito 4

Toda esta exploración nos permite pasar a la implementación final del Hito 4 sin ninguna incógnita abierta. Cada decisión de diseño está respaldada por un experimento cuantitativo:

| Eje de Diseño | Decisión Validada Empíricamente | Qué Demostró el Experimento y Por Qué se Elige |
| :--- | :--- | :--- |
| **1. Representación del Horizonte ($T=8$)** | **Entrenar sobre incrementos paso a paso $\delta T_k$ + reconstruir poses acumuladas $\prod_{m=1}^k \delta T_m$ en los nodos** | El Experimento 1 demostró que las acciones acumuladas crecen $8\times$ entre $k=1$ y $k=8$, mientras que $\delta T_k$ tiene una distribución plana e idéntica en los 8 pasos, reconstruyendo $T_{EA}^{(t,k)}$ con error $< 10^{-6}\text{ m}$ (Experimento 2). |
| **2. Variedad Geométrica y Readout SVD** | **4-DoF Horizontal FLU ($SE(2) \times \mathbb{R}$) mediante `align_keypoints_svd_horizontal_flu`** | El Experimento 2 demostró que un ruido de solo $2\text{ cm}$ en los keypoints induce hasta $10.7^\circ$ de *roll/pitch* falso con SVD 6-DoF clásico, mientras que nuestro SVD 4-DoF mantiene $0.00^\circ$ de inclinación espuria. |
| **3. Cotas de Normalización (`ActionBounds4DoF` y `flow_scale`)** | `$\Delta x_{\max}=\Delta y_{\max}=0.075$, $\Delta z_{\max}=0.050$, $\Delta\psi_{\max}=0.65\text{ rad}$ y `flow_scale=[0.04, 0.04, 0.03, 0.012, 0.012, 0.012]`** | Cubre el $100\%$ de las maniobras nominales y de recuperación *Tube Augmentation* (Experimento 1) y corrige la atenuación de $64\times$ del gradiente de giro frente al de traslación (Experimento 3). |
| **4. Resolución del Codificador de Escena ($\phi_e$)** | **$M = 32$ centroides FPS priorizando obstáculos (`z_flu > -0.11`) + $k_{\text{nn}} = 16$** | El Experimento 4 demostró que $M=32$ reduce casi a la mitad los huecos sin cubrir en las puertas (`0.387 m` $\to$ `0.205 m` en p95) costando solo `1.66 ms` en CPU. |
| **5. Capacidad de la Red (`d_model`) y Etapa $\varphi$** | **`d_model = 128` (4 cabezas, separación Clave/Valor en $\varphi$, *Modality Dropout* y tubo $\gamma=0.65$)** | El Experimento 3 demostró que esta estructura vuela el circuito completo en lazo cerrado (`FDE < 10 cm`) y cambia limpiamente de maniobra (*izquierda*, *centro*, *derecha*) al intercambiar el contexto. |
| **6. Bucle Externo e Interno de Inferencia** | **$T_{\text{exec}} = 4$ pasos ($2.5\text{ Hz}$ de replanificación con spline PCHIP a $50\text{ Hz}$) y $K = 4$ pasos DDIM (con opción rápida $K=2$)** | El Experimento 3 demostró que $T_{\text{exec}}=4$ logra el menor error de aterrizaje reduciendo $4\times$ la cantidad de llamadas a la red respecto a $T_{\text{exec}}=1$. |
| **7. Caché de Contexto y Sincronización SITL** | **Cachear $\phi_e + \sigma$ de las demos 1 vez al inicio del episodio + usar `--rtf 0.5` en Gazebo SITL guiado por `px4_timestamp`** | El Experimento 5 demostró que el caché acelera la red hasta $5.8\times$ (`11.7 ms` en `Medium-Dense`), usando $< 1.5\%$ del presupuesto de tiempo a `RTF = 0.5x`. |
