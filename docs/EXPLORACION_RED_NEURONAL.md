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
   - El paper original de *Instant Policy* fue diseñado para brazos robóticos de mesa cuyos movimientos ocurren dentro de una caja de $\sim 0.6\text{–}1.0\text{ m}$. Nuestro dron vuela en circuitos de $25\text{ m}$ con un LiDAR omnidireccional de hasta $10.0\text{ m}$ de radio.
   - Al multiplicar todas las distancias en metros por **$s = 0.1$** (es decir, **dividir los metros por $10$**):
     - El radio máximo de visión de $10.0\text{ m}$ del LiDAR pasa a ser $1.0\text{ unidad}$.
     - Un avance de $0.20\text{ m}$ en un paso de $0.1\text{ s}$ pasa a ser $0.020\text{ unidades}$.
     - Así, las coordenadas relativas que entran a los embeddings sinusoidales de Fourier quedan dentro de $[-1.0, +1.0]$, evitando aliasing periódico en la frecuencia fundamental.
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
  - [`in_context_dataset.py`](../src/uav_ip/uav_ip/policy/in_context_dataset.py): El cargador `InContextNavigationDataset` que lee los archivos `heterodata.pt` y reconstruye al vuelo (sin regenerar `.pt`) cualquier combinación de $N_{\text{demos}}$, cantidad de keyframes $L$ (entero fijo como `10, 16, 24` o `'adaptive'` proporcional a la longitud de arco), horizonte futuro $T$ (ej. `6, 8, 12, 16`) y `action_stride` (`1` o `2`), verificando contratos con [`verify_sample_contract()`](../src/uav_ip/uav_ip/policy/in_context_dataset.py).
  - [`simple_validation_net.py`](../src/uav_ip/uav_ip/policy/simple_validation_net.py): Contiene la política de referencia (`OracleReplayPolicy`, parametrizable en `horizon` y `action_stride`) y una red ligera puramente exploratoria (`MiniInstantPolicyNet`) creada solo para comprobar que las dimensiones, máscaras y gradientes conectan de extremo a extremo (incluye atajos temporales de exploración documentados en la Sección 5.5, que no constituyen un patrón para la red final y quedan marcados para revisión en el Hito 4).
  - [`kinematic_rollout_eval.py`](../src/uav_ip/uav_ip/policy/kinematic_rollout_eval.py): Simulador de vuelo en lazo cerrado en Python puro para cualquier horizonte $T$, ventana $T_{\text{exec}} \in [1, T]$ y `action_stride`. En cada paso mueve el dron según lo que ordena la política, genera en vivo la nueva vista láser con [`SyntheticLidarGenerator`](../src/uav_ip/uav_ip/synthetic_lidar.py), interpola la trayectoria a $50\text{ Hz}$ con splines PCHIP y al terminar audita el vuelo completo con el juez oficial ([`validate_episode_data()`](../src/uav_ip/uav_ip/validate_episode.py)).
  - [`exploration_suite.py`](../src/uav_ip/uav_ip/policy/exploration_suite.py): Script maestro que ejecuta tanto los 5 experimentos canónicos como sondas rápidas con hiperparámetros variables (`run_custom_hyperparameter_probe`).
- **Ejecutables para `~/venv_ip` (`src/uav_ip/scripts/policy/`):**
  - [`run_exploration_suite.py`](../src/uav_ip/scripts/policy/run_exploration_suite.py): Punto de entrada por línea de comandos parametrizable.
- **Pruebas unitarias (`src/uav_ip/test/policy/`):**
  - [`test_exploration_pipeline.py`](../src/uav_ip/test/policy/test_exploration_pipeline.py): Suite automática en `pytest` que verifica la matemática, los contratos variables $(L, T, \text{stride})$, la red exploratoria y el lazo cerrado.

### 1.1 Cómo Invocar el Script con Hiperparámetros Variables (Sin Regenerar `heterodata.pt`)

Como cada archivo `heterodata.pt` almacena la trayectoria completa `pose_se3`, las nubes `scene.pos` y las velocidades `vel_flu` a $10\text{ Hz}$, [`run_exploration_suite.py`](../src/uav_ip/scripts/policy/run_exploration_suite.py) permite variar todos los ejes de diseño directamente desde la terminal:

```bash
# A) Evaluar rápidamente una combinación puntual (sin sobrescribir este reporte):
~/venv_ip/bin/python3 src/uav_ip/scripts/policy/run_exploration_suite.py \
  --custom-eval-only \
  --num-keyframes adaptive --adaptive-arc-m 1.5 \
  --horizon 12 --action-stride 1 \
  --execution-horizon 6 --num-context-demos 2 --num-ddim-steps 4 \
  --num-centroids 32 --d-model 128

# B) Probar paso temporal doble (action_stride=2 -> Δt = 0.2 s, cubriendo 1.6 s con T=8):
~/venv_ip/bin/python3 src/uav_ip/scripts/policy/run_exploration_suite.py \
  --custom-eval-only \
  --num-keyframes 16 --horizon 8 --action-stride 2 --execution-horizon 4

# C) Regenerar únicamente este documento Markdown desde exploration_metrics.json:
~/venv_ip/bin/python3 src/uav_ip/scripts/policy/run_exploration_suite.py --report-only
```

| Parámetro CLI | Default | Valores Admitidos y Efecto |
| :--- | :---: | :--- |
| `--num-keyframes` | `10` | Entero positivo (`10`, `16`, `24`, ...) o `adaptive` ($L_d = \lceil \text{arco}_d / \Delta s \rceil$ acotado en $[10, 28]$). |
| `--adaptive-arc-m` | `1.5` | Metros de recorrido por keyframe cuando `--num-keyframes adaptive`. |
| `--horizon` | `8` | Pasos futuros $T$ predichos por inferencia (`6`, `8`, `12`, `16`, ...). |
| `--action-stride` | `1` | Separación en ticks de $10\text{ Hz}$ entre pasos (`1` $\equiv 0.1\text{ s}$, `2` $\equiv 0.2\text{ s}$). Escala cotas automáticamente. |
| `--num-context-demos` | `2` | Cantidad $N_{\text{demos}}$ de demostraciones de contexto por consulta (`1`, `2`, `3`, `4`). |
| `--execution-horizon` | `4` | Pasos $T_{\text{exec}} \in [1, T]$ ejecutados antes de replanificar en lazo cerrado. |
| `--num-ddim-steps` | `4` | Iteraciones internas $K$ de refinamiento DDIM (`1`, `2`, `4`, `8`). |
| `--num-centroids` / `--d-model` | `16` / `64` | Centroides FPS $M$ del codificador $\phi_e$ y dimensión oculta de `MiniInstantPolicyNet`. |
| `--custom-eval-only` | `False` | Ejecuta validación de contratos + Oráculo + entrenamiento rápido + rollout + latencia en ~20 s sin pisar el reporte. |

---

## 2. Mapa Maestro de Contratos de Entrada/Salida (E/S)

### ¿Qué significa esta tabla y por qué es necesaria?
Antes de construir una red neuronal mayor, un error frecuente en robótica es que un módulo entregue distancias en metros mientras otro espera decámetros ($s=0.1$), o que un tensor tenga las dimensiones invertidas. Esta tabla documenta el **contrato de ingeniería** entre el dataset (`heterodata.pt`), la red neuronal y el controlador del dron:
- **Cómo leer la columna `Forma Exacta`:**
  - `N_demos`: Cantidad de demostraciones de contexto (por defecto $2$, configurable con `--num-context-demos`).
  - `L`: Cantidad de *keyframes* por demostración (por defecto $10$, o variable por longitud de arco con `--num-keyframes adaptive` acompañado de `context_keyframe_mask`).
  - `T`: Cantidad de pasos futuros del horizonte de predicción (por defecto $8$ pasos $= 0.8\text{ s}$ con `action_stride=1`, configurable con `--horizon` y `--action-stride`).
  - `2048, 3`: Los $2048$ puntos $(x, y, z)$ de cada barrido LiDAR.
  - `4, 4`: Una matriz de transformación homogénea $SE(3)$ de $4 \times 4$.

| Etapa | Nombre del Tensor | Forma Exacta | Tipo | Marco de Coordenadas | Escala / Unidades | Descripción |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Context Input** | `context_scene_pos` | `(N_demos, L, 2048, 3)` | `float32` | Horizontal FLU (egocentrico, roll=pitch=0) | s = 0.1 (1 unidad = 10 m) | Nubes LiDAR 360° en los L keyframes (fijo o adaptativo) de cada demostracion. |
| **Context Input** | `context_pose_se3` | `(N_demos, L, 4, 4)` | `float32` | Gazebo ENU estabilizado | Traslacion s = 0.1; Rotacion R_z(yaw) in SO(2) | Poses T_{WE}^{t_l} de los keyframes (usadas solo para aristas relativas intra-demo). |
| **Context Input** | `context_vel_flu` | `(N_demos, L, 3)` | `float32` | Horizontal FLU | m/s (fisico directo) | Velocidad lineal propioceptiva [vx_flu, vy_flu, vz_flu] en cada keyframe. |
| **Context Input** | `context_flight_state` | `(N_demos, L, 1)` | `float32` | Escalar binario | {0.0, 1.0} | Estado de vuelo en el keyframe (1.0 = vuelo activo, 0.0 = aterrizaje). |
| **Context Input** | `context_action_se3` | `(N_demos, L, T, 4, 4)` | `float32` | Horizontal FLU relativo al keyframe l | Traslacion s = 0.1; Rotacion R_z(dyaw) | Horizonte futuro acumulado T_{EA}^{(t_l, 1..T)} demostrado en cada keyframe. |
| **Context Input** | `context_delta_se3` | `(N_demos, L, T, 4, 4)` | `float32` | Horizontal FLU paso a paso (k-1 -> k) | Traslacion s = 0.1 (Δt = 0.1 s * stride) | Incrementos paso a paso δT_k = (T_{EA}^{k-1})^{-1} T_{EA}^k en cada keyframe. |
| **Context Input** | `context_keyframe_mask` | `(N_demos, L)` | `bool` | Mascara booleana | {True, False} | True en keyframes reales; False en slots de relleno cuando L es adaptativo. |
| **Live Query Input** | `live_scene_pos` | `(1, 2048, 3)` | `float32` | Horizontal FLU (egocentrico, roll=pitch=0) | s = 0.1 (1 unidad = 10 m) | Nube LiDAR 360° actual observada por el dron en el instante t. |
| **Live Query Input** | `live_pose_se3` | `(1, 4, 4)` | `float32` | Gazebo ENU estabilizado | Traslacion s = 0.1; Rotacion R_z(yaw_t) | Pose actual T_{WE}^t (para proyectar la salida al mundo; usada en MiniInstantPolicyNet solo como atajo exploratorio, a revisar en el Hito 4). |
| **Live Query Input** | `live_vel_flu` | `(1, 3)` | `float32` | Horizontal FLU | m/s (fisico directo) | Velocidad lineal actual del dron en su propio marco Horizontal FLU. |
| **Live Query Input** | `live_flight_state` | `(1, 1)` | `float32` | Escalar binario | {0.0, 1.0} | Estado binario actual s_t antes de predecir el horizonte futuro. |
| **Network Output** | `pred_action_se3` | `(T, 4, 4)` | `float32` | Horizontal FLU relativo al instante t | Traslacion s = 0.1; Rotacion R_z(dyaw_k) | Horizonte de T poses relativas acumuladas predichas T_hat_{EA}^{(t, 1..T)}. |
| **Network Output** | `pred_flight_state` | `(T, 1)` | `float32` | Probabilidad escalar | [0.0, 1.0] | Prediccion del estado de vuelo futuro s_hat_{t+1..t+T} (<0.5 activa aterrizaje). |
| **Controller Output** | `pred_world_waypoints_enu_m` | `(T, 4)` | `float32` | Gazebo ENU global | Metros reales (÷ s = × 10.0) y radianes [-pi, pi) | Waypoints globales [X_enu_m, Y_enu_m, Z_enu_m, yaw_enu_rad] para el spline. |

---

## 3. Experimento 1: Radiografía Estadística y Cotas de Normalización ($\delta T_k$ vs. $T_{EA}^{(t, k)}$)

### 3.1 ¿Qué duda queremos resolver en este experimento?
En cada archivo `heterodata.pt`, las etiquetas de acción guardadas (`action_se3`) representan la **pose futura acumulada** $T_{EA}^{(t, k)}$ para los pasos $k = 1, 2, \dots, 8$ medida desde la posición actual del dron en el instante $t$.

Tenías la duda de cómo debía predecir la red esos 8 pasos futuros y cómo debíamos normalizarlos para que todas las salidas de la red vivan cómodamente en el intervalo $[-1, 1]$. Tenemos dos formas matemáticamente equivalentes de expresar un horizonte de 8 pasos:
1. **Formulación Acumulada ($T_{EA}^{(t, k)}$):** Cada nodo $k$ representa la distancia total desde $t$ hasta $t+k$.
   - *Problema intuitivo:* Si el dron vuela a $2\text{ m/s}$, en el paso $k=1$ ($0.1\text{ s}$ después) habrá avanzado **$0.20\text{ m}$**, pero en el paso $k=8$ ($0.8\text{ s}$ después) habrá avanzado **$1.60\text{ m}$** (¡8 veces más!). Si entrenamos la red directamente con estos valores, el error del paso 8 será numéricamente mucho más grande que el del paso 1, haciendo que la red preste mucha atención al final del horizonte y descuide el primer paso inmediato, que es justamente el que el dron va a ejecutar primero.
2. **Formulación Incremental Paso a Paso ($\delta T_k$):** Cada nodo $k$ representa únicamente **el pequeño salto entre el paso anterior $k-1$ y el paso $k$**:
   $$\delta T_k = \left(T_{EA}^{(t, k-1)}\right)^{-1} T_{EA}^{(t, k)}, \quad \text{con } T_{EA}^{(t, 0)} = I_4$$
   - *Ventaja intuitiva:* Tanto en $k=1$ como en $k=8$, el salto dura exactamente $\Delta t = 0.1\text{ s}$ y mide siempre alrededor de $0.20\text{ m}$. Los 8 pasos tienen **la misma escala estadística**. Y cuando necesitamos las poses acumuladas para construir los nodos del grafo, encadenamos (multiplicamos) los saltos desde el principio:
   $$T_{EA}^{(t, k)} = \delta T_1 \cdot \delta T_2 \cdots \delta T_k = \prod_{m=1}^{k} \delta T_m$$

Para comprobarlo con datos reales de tu proyecto, analizamos **200 episodios cinemáticos** (54164 instantes de vuelo nominal + 16245 instantes de recuperación *Tube Augmentation*) y **60 episodios físicos en PX4 SITL** (11661 instantes nominales + 3499 instantes *Tube Augmentation*).

### 3.2 Tabla Comparativa de Rangos: Acumulada $T_{EA}^{(t,k)}$ vs. Incremental $\delta T_k$

**Cómo leer esta tabla:**
- Las columnas `p50`, `p95`, `p99.5` y `Máximo Absoluto` muestran los desplazamientos en la escala interna de la red ($s = 0.1$) para las traslaciones ($\vert \Delta x_s|, |\delta x_s|, |\delta y_s|, |\delta z_s \vert $) y en **radianes** para el giro de rumbo ($\vert \delta\psi \vert $).
- La última columna (`Equivalente Físico (p99.5)`) convierte el percentil $99.5\%$ a **metros reales** (multiplicando por $10$) o a **grados** para que puedas interpretarlo físicamente de un vistazo.

| Dominio / Formulación | Componente | p50 | p95 | p99.5 | Máximo Absoluto | Equivalente Físico (p99.5) |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Kinematic - Acumulada $k=1$** | $\vert \Delta x_s \vert$ | 0.0133 | 0.0178 | 0.0203 | 0.0220 | 0.203 m |
| **Kinematic - Acumulada $k=8$** | $\vert \Delta x_s \vert $ | 0.1003 | 0.1364 | 0.1492 | 0.1653 | 1.492 m |
| **Kinematic - Incremental $\delta T_k$ ($\forall k$)** | $\vert \delta x_s \vert $ | 0.0131 | 0.0177 | 0.0196 | 0.0220 | 0.196 m ($\Delta t=0.1$ s) |
| **Kinematic - Incremental $\delta T_k$ ($\forall k$)** | $\vert \delta y_s \vert $ | 0.0047 | 0.0138 | 0.0189 | 0.0234 | 0.189 m ($\Delta t=0.1$ s) |
| **Kinematic - Incremental $\delta T_k$ ($\forall k$)** | $\vert \delta z_s \vert $ | 0.0000 | 0.0104 | 0.0160 | 0.0221 | 0.160 m ($\Delta t=0.1$ s) |
| **Kinematic - Incremental $\delta T_k$ ($\forall k$)** | $\vert \delta \psi \vert $ [rad] | 0.0289 | 0.1363 | 0.1942 | 0.9267 | 11.13° ($\Delta t=0.1$ s) |
| **PX4 SITL - Acumulada $k=8$** | $\vert \Delta x_s \vert $ | 0.1053 | 0.2317 | 0.2939 | 0.3313 | 2.939 m |
| **PX4 SITL - Incremental $\delta T_k$ ($\forall k$)** | $\vert \delta x_s \vert $ | 0.0127 | 0.0307 | 0.0392 | 0.0438 | 0.392 m ($\Delta t=0.1$ s) |
| **PX4 SITL - Incremental $\delta T_k$ ($\forall k$)** | $\vert \delta y_s \vert $ | 0.0068 | 0.0213 | 0.0353 | 0.0493 | 0.353 m ($\Delta t=0.1$ s) |
| **PX4 SITL - Incremental $\delta T_k$ ($\forall k$)** | $\vert \delta z_s \vert $ | 0.0008 | 0.0120 | 0.0200 | 0.0311 | 0.200 m ($\Delta t=0.1$ s) |
| **PX4 SITL - Incremental $\delta T_k$ ($\forall k$)** | $\vert \delta \psi \vert $ [rad] | 0.0176 | 0.1392 | 0.2831 | 0.2988 | 16.22° ($\Delta t=0.1$ s) |
| **Tube Augmentation ($\gamma=0.70$)** | $\vert \delta x_s\vert, \vert\delta y_s \vert $ | 0.0122 | 0.0200 | 0.0263 | 0.0357 | Cota física $\le 0.65$ m ($6.5$ m/s) |

![Comparación Acumulada vs Incremental](assets/exploration/exp1_cumulative_vs_incremental_bounds.png)

### 3.3 Cómo interpretar la gráfica y qué se observa
- **Gráfica Izquierda (Acción Acumulada $T_{EA}^{(t,k)}$):** Observa cómo a medida que nos alejamos en el eje horizontal desde el paso $k=1$ hasta el paso $k=8$, las curvas suben en rampa empinada: en cinemático pasan de $0.203\text{ m}$ ($k=1$) a $1.492\text{ m}$ ($k=8$), y en PX4 SITL llegan hasta **$2.939\text{ m}$** en $k=8$.
- **Gráfica Derecha (Incremento Paso a Paso $\delta T_k$):** Al transformar las acciones a incrementos entre pasos consecutivos, **las curvas se mantienen casi planas** de $k=1$ a $k=8$ ($0.196\text{ m}$ en cinemático y $0.392\text{ m}$ en SITL). La línea punteada roja muestra una referencia exploratoria ($0.075$ en escala $s=0.1$, que equivale a $0.75\text{ m}$ por paso o $7.5\text{ m/s}$, cubriendo la cota de $6.5\text{ m/s}$ fijada en `bag_to_heterodata.py`).

> [!NOTE]
> **Observación sobre `ActionBounds4DoF` (Alcance Exploratorio - A Revisar en el Hito 4):**
> Los datos muestran que trabajar sobre incrementos $\delta T_k$ mantiene una escala uniforme entre $k=1$ y $k=8$, a diferencia de las poses acumuladas $T_{EA}^{(t,k)}$. Para poder ejecutar las pruebas rápidas de este reporte se definieron en [`ActionBounds4DoF`](../src/uav_ip/uav_ip/policy/se3_diffusion_math.py) cotas auxiliares tomadas de este corpus (`0.075` en $X/Y$, `0.050` en $Z$ y `0.65 rad` en $\psi$ por paso de $0.1\text{ s}$).
> **Importante:** Estos valores provienen de la distribución de este dataset concreto y **no representan un patrón fijo para la red final**. Aplicar un recorte duro (`clip=True`) con umbrales ajustados al dataset actual podría limitar maniobras futuras con mayor velocidad de giro o ascenso; en el Hito 4 conviene revisar un esquema más general (por ejemplo, normalización mediante la escala de ruido $\sigma_k$ del *schedule* de difusión, estadísticos calculados dinámicamente del split de entrenamiento sin *clipping* duro en inferencia, o límites físicos del vehículo).

---

## 4. Experimento 2: Exactitud Matemática Round-Trip $SE(3)$ y SVD 4-DoF vs. 6-DoF

### 4.1 ¿Qué duda queremos resolver y qué significan las fórmulas de esta etapa?
Como vimos en la Sección 0.3, la etapa final $\psi$ de la red neuronal no escupe directamente una matriz $4 \times 4$, sino que predice **hacia dónde deben moverse los 6 puntos de la cruceta del dron**. En concreto, para cada uno de los 6 keypoints canónicos $\mathbf{p}_g^i$ ($i=1\dots 6$), la red predice dos vectores 3D (6 números por punto):
1. **El flujo de traslación $\Delta\mathbf{p}_t^i \in \mathbb{R}^3$:** Indica cuánto debe desplazarse el centro del dron entre la hipótesis actual $\hat{T}_{\text{curr}} = (R_{\text{curr}}, \mathbf{t}_{\text{curr}})$ y la pose correcta $T_* = (R_*, \mathbf{t}_*)$:
   $$\Delta\mathbf{p}_t^i = \mathbf{t}_* - \mathbf{t}_{\text{curr}}$$
2. **El flujo de rotación $\Delta\mathbf{p}_r^i \in \mathbb{R}^3$:** Indica cuánto debe moverse cada punta de la cruceta **únicamente por el cambio de orientación** entre la orientación actual $R_{\text{curr}}$ y la orientación correcta $R_*$, sin contar la traslación:
   $$\Delta\mathbf{p}_r^i = \left(R_* - R_{\text{curr}}\right) \mathbf{p}_g^i$$

**¿Cómo reconstruimos una matriz rígida $4 \times 4$ a partir de las predicciones de la red?**
- Para la **nueva posición del dron ($\mathbf{t}_{\text{new}}$)**, promediamos el flujo de traslación predicho por los 6 puntos y se lo sumamos a la posición actual:
  $$\mathbf{t}_{\text{new}} = \mathbf{t}_{\text{curr}} + \frac{1}{6} \sum_{i=1}^{6} \Delta\hat{\mathbf{p}}_t^i$$
- Para la **nueva orientación ($R_{\text{new}}$)**, tomamos dónde estarían las 6 puntas de la cruceta después de aplicarles el flujo rotacional predicho ($\mathbf{q}_i = R_{\text{curr}} \mathbf{p}_g^i + \Delta\hat{\mathbf{p}}_r^i$) y buscamos la matriz de rotación rígida que mejor alinee la cruceta original $\{\mathbf{p}_g^i\}$ con los puntos movidos $\{\mathbf{q}_i\}$.

**Comparación entre SVD 6-DoF libre y SVD 4-DoF proyectado a Horizontal FLU:**
- El algoritmo estándar de alineación 3D (llamado **SVD de Kabsch-Arun en 6-DoF**) busca cualquier rotación 3D libre en $SO(3)$. Si las predicciones de los 6 puntos tienen pequeñas discrepancias independientes (por ejemplo $1$ o $2\text{ cm}$ en la altura $Z$ del punto de la nariz $\mathbf{p}_g^1$ respecto al de la cola $\mathbf{p}_g^2$), el SVD 6-DoF interpreta esa diferencia como una inclinación en *pitch* o *roll*.
- Dado que en este sistema las acciones están definidas sobre el marco nivelado **Horizontal FLU** ($\text{roll}=0, \text{pitch}=0$), implementamos en [`align_keypoints_svd_horizontal_flu()`](../src/uav_ip/uav_ip/policy/se3_diffusion_math.py) una **proyección analítica en 4-DoF ($SE(2) \times \mathbb{R}$)**: proyectamos las puntas de la cruceta al plano horizontal $(X, Y)$, calculamos la matriz de covarianza planar $H_{2\times 2} = \sum_{i=1}^6 \mathbf{p}_{g, xy}^i (\mathbf{q}_{xy}^i)^\top$ y extraemos el ángulo de rumbo óptimo en forma cerrada:
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
> **Lectura de la prueba geométrica:**
> 1. Las conversiones entre poses acumuladas, incrementos $\delta T_k$ y flujos de 6 keypoints son numéricamente reversibles dentro de la precisión `float32` ($\sim 10^{-7}\text{ m}$).
> 2. Mientras las acciones se formulen en el marco nivelado `Horizontal FLU` ($\text{roll}=0, \text{pitch}=0$), el SVD 6-DoF libre traduce pequeños ruidos de predicción ($2\text{ cm}$) en inclinaciones aparentes de hasta **`10.70°`**, mientras que `align_keypoints_svd_horizontal_flu` mantiene la actualización restringida al subgrupo planar + altitud ($SE(2) \times \mathbb{R}$) con el mismo error medio en rumbo.

---

## 5. Experimento 3: Evaluación en Lazo Cerrado Cinemático, Horizonte $T_{\text{exec}}$ y *Context-Swap*

### 5.1 ¿Qué dudas queremos resolver en este experimento?
Este experimento aborda tres preguntas prácticas de integración:
1. **Tu Duda 4 (Evaluador Cinemático Ideal en Lazo Cerrado):** ¿Es viable construir un simulador cinemático ideal en Python puro (sin esperar a Gazebo ni a los controladores PID de PX4) donde el dron avance a la pose predicha, genere su propio LiDAR sintético en cada paso y sea auditado por el validador geométrico `validate_episode_data()`?
2. **Tu Duda 3 (Horizonte de Ejecución $T_{\text{exec}}$):** De los pasos que predice la red en cada inferencia, ¿cómo cambia el comportamiento al recalcular la predicción en cada paso ($T_{\text{exec}}=1$), a mitad de camino ($T_{\text{exec}}=4$) o al final del horizonte ($T_{\text{exec}}=8$)?
3. **Chequeo del Flujo de Datos del Dataset:** ¿Conectan correctamente los tensores guardados en `heterodata.pt` con un bucle de entrenamiento e inferencia en lazo cerrado?

---

### 5.2 Parte A - Control de Cordura con `OracleReplayPolicy` (0% Error de Red)
**¿Qué hicimos aquí?** Antes de probar una red neuronal, creamos una política "Oráculo" (`OracleReplayPolicy`) que lee las acciones de referencia $T_{EA}^{(t, 1..8)}$ del dataset y las entrega con el mismo formato de salida que tendrá la red. La evaluamos en lazo cerrado en el circuito `task_0006` (4 obstáculos + plataforma de aterrizaje) probando cuatro horizontes de replanificación $T_{\text{exec}} \in \{1, 2, 4, 8\}$.

| Horizonte de Ejecución ($T_{\text{exec}}$) | Veredicto `validate_episode_data` | Estado de Misión | Pasos Ejecutados ($10\text{ Hz}$) | Llamadas a la Política | ADE [m] | FDE [m] |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`T_exec = 1`** | `accepted` | `valid` | 235 | 235 | 0.00000 | 0.05159 |
| **`T_exec = 2`** | `accepted` | `valid` | 235 | 118 | 0.00000 | 0.05159 |
| **`T_exec = 4`** | `accepted` | `valid` | 235 | 59 | 0.00000 | 0.05159 |
| **`T_exec = 8`** | `accepted` | `valid` | 235 | 30 | 0.00000 | 0.05159 |

**¿Qué verifica esta tabla?**
- El error `ADE = 0.00000 m` y `FDE = 0.05159 m` (distancia al centro de la plataforma en el instante en que `flight_state` activa el aterrizaje dentro del radio de $0.50\text{ m}$) con veredicto **`accepted`** en los 4 casos confirma que **el formato de salida de la política, la composición de poses en `Horizontal FLU`, el interpolador PCHIP a $50\text{ Hz}$ y el validador geométrico son mutuamente consistentes**.
- En la columna `Llamadas a la Política`: con $T_{\text{exec}} = 1$ se invoca la política $235$ veces para completar los $23.5\text{ s}$ del circuito, frente a **$59$ llamadas** con $T_{\text{exec}} = 4$ ($4\times$ menos invocaciones).

---

### 5.3 Parte B - Prueba Exploratoria de Ajuste Rápido con `MiniInstantPolicyNet` (task_0006)
**¿Qué hicimos aquí?** Como prueba funcional del flujo de gradientes y tensores, sobre-ajustamos (`overfit`) la red validadora ligera `MiniInstantPolicyNet` (`d_model=64`) en CPU durante 41.39 segundos sobre un único episodio de `task_0006` (la pérdida bajó de `0.8214` a `0.0332`). Luego evaluamos ese mismo episodio en el simulador cinemático en lazo cerrado variando $T_{\text{exec}}$ y $K$:

| Pasos DDIM ($K$) | Horizonte $T_{\text{exec}}$ | Veredicto Validador | Llamadas a Red | Latencia Media [ms] | ADE [m] | FDE (Error al Landing Pad) [m] |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$K = 2$** | **$T_{\text{exec}} = 1$** | `rejected` | 252 | 8.07 ms | 2.5983 | **5.3763** |
| **$K = 2$** | **$T_{\text{exec}} = 2$** | `rejected` | 114 | 8.87 ms | 1.5003 | **0.6144** |
| **$K = 2$** | **$T_{\text{exec}} = 4$** | `rejected` | 56 | 7.66 ms | 1.5293 | **0.5613** |
| **$K = 2$** | **$T_{\text{exec}} = 8$** | `rejected` | 28 | 8.01 ms | 1.3981 | **0.7849** |
| **$K = 1$** | **$T_{\text{exec}} = 4$** | `rejected` | 54 | 6.20 ms | 1.7344 | **0.9117** |
| **$K = 4$** | **$T_{\text{exec}} = 4$** | `rejected` | 55 | 12.29 ms | 1.5528 | **0.0985** |

**Lectura exploratoria de esta tabla (sin extrapolar valores óptimos a la red final):**
1. **Sensibilidad observada ante la ventana de replanificación ($T_{\text{exec}}$):**
   - En esta red pequeña, replanificar en cada paso individual ($T_{\text{exec}} = 1$, cada $0.1\text{ s}$) amplificó pequeñas oscilaciones de predicción (`FDE = 5.3763 m`), mientras que ejecutar un tramo intermedio ($T_{\text{exec}} = 4$ pasos $= 0.4\text{ s}$, con $K = 4$) redujo el desvío al final del episodio (`FDE = 0.0985 m`). Cómo se comportará este compromiso en la red final dependerá de su suavidad y de su horizonte $T$, por lo que conviene volver a medirlo en el Hito 4.
2. **Por qué el validador marcó `rejected` a pesar del bajo FDE:**
   - En el **Panel Central** de la figura inferior se observa que la trayectoria de `MiniInstantPolicyNet` cruza `landmark_1 (square_gate)`, esquiva `landmark_2 (pole)` y alcanza la plataforma final, pero recorta la amplitud lateral en `landmark_3` y `landmark_4` (ambas puertas cuadradas con maniobras distintas de slalom), incumpliendo el margen de seguridad de `validate_episode_data()`. Esto refleja las limitaciones de una red exploratoria de 1 capa ajustada rápidamente sobre un solo episodio.

---

### 5.4 Parte C - Sonda Sintética de Condicionamiento en Contexto (*Context-Swap* sobre 3 Muestras)
**¿Qué hicimos aquí?** Para comprobar que el canal de entrada de contexto está cableado de punta a punta con la predicción de waypoints, armamos una sonda mínima con **solo 3 muestras estáticas** frente a una puerta cuadrada desde el mismo punto $(X = 2.15\text{ m}, Y = -0.25\text{ m})$ y la misma nube $O_t$, inyectando contextos correspondientes a tres maniobras (`task_0079` izquierda, `task_0118` centro, `task_0171` derecha):

| Tarea de Contexto Inyectada | Primitiva Demostrada en el Contexto | $\Delta X$ Predicho ($k=8$) [m] | $\Delta Y$ Lateral Predicho ($k=8$) [m] | Comportamiento Observado en la Gráfica |
| :--- | :--- | :---: | :---: | :--- |
| **`task_0079`** | `slalom(side=left)` | `+1.264 m` | **`+2.065 m`** | Curva Azul: Desplazamiento hacia +Y (izquierda) |
| **`task_0118`** | `traverse (center)` | `+1.621 m` | **`-0.746 m`** | Curva Verde: Avance frontal en +X hacia la abertura |
| **`task_0171`** | `slalom(side=right)` | `+1.169 m` | **`-1.250 m`** | Curva Roja: Desplazamiento hacia -Y (derecha) |

![Resultados Overfit, Rollout y Context-Swap](assets/exploration/exp3_overfit_rollout_and_context_swap.png)

**Cómo leer los 3 paneles de la figura superior:**
- **Panel Izquierdo:** Evolución de la pérdida de entrenamiento de `MiniInstantPolicyNet` sobre un episodio, verificando que los gradientes fluyen sin errores numéricos.
- **Panel Central:** Comparación en planta X-Y entre la demostración (`episode_00`, línea negra punteada) y el vuelo de la red exploratoria (línea verde).
- **Panel Derecho:** Salida de los 8 pasos futuros ante una misma observación local al intercambiar los tensores de contexto en la sonda de 3 muestras.

---

### 5.5 Parches Exploratorios Usados en `MiniInstantPolicyNet` - Notas de Revisión para el Hito 4

> [!WARNING]
> **Alcance Estricto de Esta Exploración:**
> El propósito de `MiniInstantPolicyNet` fue comprobar rápidamente en CPU que los tensores de entrada y salida encajan entre sí sobre pruebas reducidas (1 episodio en 5.3 y 3 muestras estáticas en 5.4). Para que esa red simplificada ajustara en pocos segundos se introdujeron **varios parches heurísticos** en `simple_validation_net.py` y `exploration_suite.py`. **Estos parches NO son patrones recomendados para la red final**; se dejan registrados aquí únicamente como aspectos que conviene revisar con cuidado al diseñar el Hito 4.

#### 1. Escala entre Traslación y Rotación (`flow_scale` y `ActionBounds4DoF`)
- **Qué se hizo en la exploración:** Al escalar el espacio por $s = 0.1$, los 6 keypoints del dron ($0.5\text{ m}$ reales) quedan a un radio $L_g = 0.05$, por lo que un giro pequeño desplaza las puntas de la cruceta numéricamente menos que una traslación típica ($\|\Delta\mathbf{p}_r\| \approx L_g |\delta\psi \vert $). En `MiniInstantPolicyNet` se aplicó un vector fijo `flow_scale = [0.04, 0.04, 0.03, 0.012, 0.012, 0.012]` y cotas `ActionBounds4DoF` con `clip=True` sacadas de los percentiles de este dataset.
- **Limitación de ese parche:** Esos coeficientes dependen de las velocidades lineales y de giro presentes en los episodios actuales. En tareas futuras que requieran giros amplios o maniobras verticales rápidas, fijar constantes a mano o aplicar `clip=True` podría sesgar el gradiente o recortar acciones válidas.
- **Qué revisar en el Hito 4:** Evaluar una normalización general no atada manualmente a este subconjunto - por ejemplo, escalar el objetivo por el nivel de ruido $\sigma_k$ propio del *schedule* de difusión en cada paso, o calcular estadísticos dinámicos del split de entrenamiento sin imponer *clipping* duro en inferencia.

#### 2. Señal de Progreso y Coordenadas Globales (`live_pose_se3`)
- **Qué se hizo en la exploración:** En la prueba 5.3 sobre `task_0006` (que atraviesa tres `square_gate` parecidas), `MiniInstantPolicyNet` recibió la posición global en el mapa (`pose_se3[:, :3, 3]`) y ponderó la atención según la diferencia en $X_{\text{ENU}}$ (`ctx_x - live_x`), además de sumar perturbaciones sintéticas extra al vuelo ($\gamma = 0.65$) sobre los datos ya existentes.
- **Limitación de ese parche:** Comparar coordenadas globales de mapa (`Gazebo ENU`) entre la demostración y el dron en vivo ata la red al lugar exacto del mapa donde ocurrió el episodio, por lo que dejaría de corresponderse cuando los obstáculos cambien de ubicación o rumbo entre episodios. En cambio, la idea general de representar el **orden o progreso relativo** sí merece estudiarse diferenciando dónde se aplica: dentro de las demostraciones grabadas (índice relativo o aristas entre keyframes $l = 0 \dots L-1$) no depende del marco global, mientras que para el dron en vivo habría que evaluar alternativas egocéntricas si hicieran falta.
- **Qué revisar en el Hito 4:** Revisar primero cuánto desambigua por sí sola la vista LiDAR 360° de $R_{\max} = 10.0\text{ m}$ (que suele abarcar obstáculos vecinos) junto con los nodos `data['aug']` ($\gamma = 0.70$) ya incluidos en `heterodata.pt`, y, si se observa ambigüedad en circuitos con tramos repetidos, evaluar mecanismos de progreso o memoria que preserven la invariancia espacial respecto al mundo.

#### 3. Separación Manual Clave/Valor, `Modality Dropout` y Anclas Sintéticas en el Test de 3 Muestras
- **Qué ocurrió y qué parche se usó en la exploración:** En la sonda 5.4 (*Context-Swap*) se entrenó `MiniInstantPolicyNet` desde cero con **solo 3 muestras estáticas**. Con tan pocos datos, cualquier red pequeña tiende a memorizar diferencias mínimas entre las 3 entradas e ignorar el contexto. Para forzar que en ese test de juguete mirara el contexto, se separaron manualmente *Keys* y *Values*, se apagó aleatoriamente la rama de la observación actual (`live_modality_dropout`) y en `exploration_suite.py` se sobrescribió el primer keyframe como ancla sintética.
- **Limitación de ese parche:** Tanto el problema observado como esa solución rebuscada provienen del régimen artificial de entrenar con 3 ejemplos sueltos, no necesariamente del comportamiento de la arquitectura real entrenada sobre un corpus amplio.
- **Qué revisar en el Hito 4:** Probar primero la formulación estándar de grafos heterogéneos de *Instant Policy* (donde las acciones demostradas son nodos geométricos 3D dentro del grafo sin particiones forzadas ni apagado manual de ramas) entrenando sobre las tareas variadas del Catálogo $v2$, y recién analizar regularizaciones adicionales si en validación multi-tarea se detectara subutilización del contexto.

#### 4. Otros Atajos Internos de `MiniInstantPolicyNet` (Puente Determinista desde $I_4$ y Cachés por Suma Hash)
- **Puente interpolado determinista:** `MiniInstantPolicyNet` usó una interpolación lineal fija sobre $u \in \{0.25, 0.5, 0.75, 1.0\}$ partiendo siempre desde la identidad $I_4$, únicamente para simplificar el chequeo numérico en CPU. En el Hito 4 corresponde revisar el proceso de difusión estocástico (DDPM/DDIM) y su muestreo de ruido.
- **Cachés por suma hash en entrenamiento:** `_patch_cache` y la deduplicación por suma de tensores en `encode_context` se agregaron solo para acelerar iteraciones repetidas sobre el mismo episodio en CPU y no son apropiados para entrenamiento general (a diferencia de cachear en inferencia la codificación de las demos fijas al comenzar un episodio de vuelo).

---

## 6. Experimento 4: Exploración de Compresión de Escena ($M = 16$ vs. $M = 32$ Centroides FPS)

### 6.1 ¿Qué se midió y qué precaución hay que tener con el prefiltrado usado?
Cada barrido de nuestro sensor LiDAR sintético contiene **$N = 2048$ puntos 3D** pertenecientes a los obstáculos cercanos y a la plataforma `landing_pad` dentro del radio de $10.0\text{ m}$. Para estimar cuánta resolución espacial se pierde al comprimir la nube a $M$ centroides mediante *Farthest Point Sampling* (FPS) y vecindades $k$-NN ($k_{\text{nn}} = 16$), medimos la distancia desde los puntos de los obstáculos cercanos hasta su centroide más próximo con $M = 16$ y $M = 32$:

| Configuración Evaluada | Centroides con `z_flu > -0.10` (Media) | Distancia Media Obstáculo $\to$ Centroide [m] | Distancia p95 Obstáculo $\to$ Centroide [m] | Tiempo FPS + k-NN por Nube [ms] |
| :--- | :---: | :---: | :---: | :---: |
| **`M = 16` centroides** | 14.1 / 16 | `0.237 m` | `0.387 m` | `0.85 ms` |
| **`M = 32` centroides** | 29.9 / 32 | `0.144 m` | `0.205 m` | `1.66 ms` |

> [!NOTE]
> **Lectura exploratoria y advertencia sobre el filtro `z_flu > -0.11`:**
> - Pasar de $M = 16$ a $M = 32$ reduce en esta prueba la distancia p95 sin cubrir de `0.387 m` a `0.205 m` con un incremento moderado de tiempo en CPU (`1.00 ms` $\to$ `1.66 ms` por nube), lo que sugiere considerar $M \in \{16, 32, 64\}$ dentro de las opciones a evaluar en el Hito 4.
> - **Parche exploratorio a revisar:** En [`subsample_scene_centroids_torch()`](../src/uav_ip/uav_ip/policy/simple_validation_net.py) se incluyó un corte heurístico `z_flu > -0.11` ($1.1\text{ m}$ por debajo del dron). Como [`synthetic_lidar.py`](../src/uav_ip/uav_ip/synthetic_lidar.py) **no genera piso infinito** (solo muestrea obstáculos y `landing_pad`), un corte fijo en $Z$ puede descartar puntos legítimos de la plataforma de aterrizaje cuando el dron vuela alto (p. ej. durante un `overfly` a $2.5\text{ m}$). Por ello, este corte fijo fue únicamente un supuesto exploratorio y conviene revisarlo en el Hito 4 (por ejemplo, evaluando FPS / PointNet++ sobre la nube completa sin umbral rígido en $Z$).

---

## 7. Experimento 5: Perfilado de Latencia en CPU y Ralentización (`RTF`) en Gazebo/PX4

### 7.1 ¿Qué dudas queremos resolver aquí? (Tu Duda 1 sobre Gazebo y la velocidad de la red)
Tu primera duda fue: *"¿Puede ralentizarse Gazebo para que las demoras de cálculo en la red neuronal no perjudiquen tanto? ¿Está bien este enfoque?"*

Para tener órdenes de magnitud orientativos en tu máquina (**CPU:** `x86_64`, `8` hilos de PyTorch), medimos tiempos de ejecución sobre `MiniInstantPolicyNet` variando:
1. **Tamaño de referencia:** `Mini (d=64, M=16)`, `Medium (d=128, M=16)`, `Medium-Dense (d=128, M=32)` y `Large (d=256, M=32)`.
2. **Cantidad de pasos internos ($K \in \{1, 2, 4, 8\}$).**
3. **Caché de Contexto en Inferencia vs. Recálculo Completo:**
   - Durante un episodio de vuelo, las $N_{\text{demos}}$ demostraciones de contexto no cambian de un paso al siguiente.
   - **Sin Caché:** Re-ejecutar $\phi_e + \sigma$ sobre todas las nubes de las demostraciones en cada ciclo de control repite trabajo idéntico.
   - **Con Caché en Inferencia:** Codificar las demostraciones una vez al inicio del vuelo deja en el lazo de tiempo real únicamente la nube actual $O_t$ y los pasos de actualización.

| Configuración Evaluada | Parámetros | Caché Contexto (1 vez/ep) | Sin Caché ($K=4$) | Con Caché ($K=1$) | Con Caché ($K=2$) | Con Caché ($K=4$) | Con Caché ($K=8$) | Aceleración por Caché |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mini (d=64, M=16)** | `128,968` | `32.0 ms` | `32.0 ms` | `5.3 ms` | `7.1 ms` | **`10.5 ms`** | `16.6 ms` | **`3.1x`** |
| **Medium (d=128, M=16)** | `470,920` | `22.6 ms` | `34.0 ms` | `5.6 ms` | `7.3 ms` | **`11.0 ms`** | `19.3 ms` | **`3.1x`** |
| **Medium-Dense (d=128, M=32)** | `470,920` | `56.9 ms` | `68.0 ms` | `7.2 ms` | `8.5 ms` | **`11.7 ms`** | `21.8 ms` | **`5.8x`** |
| **Large (d=256, M=32)** | `1,793,800` | `58.6 ms` | `86.7 ms` | `11.6 ms` | `11.2 ms` | **`15.7 ms`** | `24.6 ms` | **`5.5x`** |

![Benchmark de Latencia en CPU](assets/exploration/exp5_cpu_latency_benchmark.png)

### 7.2 Relación entre Latencia de Inferencia y Factor `--rtf` en Gazebo

**Presupuesto temporal según frecuencia de replanificación y `RTF`:**
Si cada paso del horizonte dura $\Delta t_{\text{paso}} = 0.10\text{ s} \times \text{stride}$ de tiempo simulado y se replanifica cada $T_{\text{exec}}$ pasos, el margen de tiempo real de reloj de pared ($\Delta t_{\text{wall}}$) disponible para calcular una inferencia a un `RTF` dado es:
$$\Delta t_{\text{wall}} = \frac{T_{\text{exec}} \times \Delta t_{\text{paso}}}{\text{RTF}}$$
- Por ejemplo, con $T_{\text{exec}} = 4$ y $\Delta t_{\text{paso}} = 0.10\text{ s}$, la ventana simulada es de $400\text{ ms}$ ($400\text{ ms}$ reales a `RTF = 1.0x`, u $800\text{ ms}$ reales a `RTF = 0.5x` usando `world_generator.py --rtf 0.5`).
- Nota: La arquitectura de grafos heterogéneos del Hito 4 tendrá distinto costo computacional que `MiniInstantPolicyNet`, por lo que estos tiempos deberán volver a medirse sobre el modelo final para elegir el par $(K, \text{RTF})$ al evaluar en PX4 SITL sincronizado con `px4_timestamp`.

---

## 8. Síntesis de la Exploración: Qué Verificamos de Interfaz y Qué Queda Abierto para el Hito 4

Para separar lo que corresponde a contratos de datos verificados frente a lo que fueron simplificaciones temporales de la exploración, resumimos ambos grupos por separado:

### 8.1 Interfaces, Geometría y Herramientas Verificadas

| Componente | Qué Se Comprobó en Esta Exploración |
| :--- | :--- |
| **1. Contratos de `heterodata.pt` y Cargador Dinámico** | `InContextNavigationDataset` carga episodios cinemáticos y SITL verificando formas, tipos y escalas (`s=0.1` en posiciones, rad en ángulos, m/s en `vel_flu`), y permite variar al vuelo $(L, T, \text{action\_stride}, N_{\text{demos}})$ - incluyendo `L='adaptive'` con `context_keyframe_mask` - sin regenerar `.pt`. |
| **2. Conversión $\delta T_k \leftrightarrow T_{EA}^{(t,k)}$ y Proyección SVD 4-DoF** | La descomposición incremental mantiene escala uniforme a lo largo del horizonte con error de ida y vuelta $< 10^{-6}\text{ m}$, y `align_keypoints_svd_horizontal_flu` evita introducir *roll/pitch* espurio ante ruido en los keypoints mientras se trabaje en `Horizontal FLU`. |
| **3. Evaluador Cinemático en Lazo Cerrado (`kinematic_rollout_eval.py`)** | Con `OracleReplayPolicy`, la composición de poses, interpolación PCHIP a $50\text{ Hz}$, generación LiDAR en vivo y auditoría con `validate_episode_data()` reproducen los vuelos de referencia (`accepted`, `ADE = 0.000 m`). |
| **4. Soporte `--rtf` y Caché de Demostraciones en Inferencia** | `world_generator.py --rtf <float>` escala la velocidad de reloj de Gazebo manteniendo `max_step_size = 0.004s`, y precalcular la codificación de las demos al inicio del episodio reduce entre $3\times$ y $5.8\times$ el costo por ciclo de control en las pruebas medidas. |

### 8.2 Aspectos Abiertos y Parches Exploratorios a Revisar en el Hito 4

| Aspecto | Atajo Usado en la Exploración (`MiniInstantPolicyNet`) | Qué Revisar / Evaluar en la Red Final (Hito 4) |
| :--- | :--- | :--- |
| **Normalización y Balance Traslación / Giro** | `flow_scale` manual y `ActionBounds4DoF` con `clip=True` ajustados a este dataset. | Revisar una formulación general sin constantes ajustadas a mano ni `clip` duro en inferencia (p. ej. normalización por el *schedule* de ruido $\sigma_k$ de difusión, estadísticos del split de `train` o límites físicos) contemplando tareas futuras con giros o ascensos más intensos. |
| **Invariancia Espacial $SE(3)$ y Señal de Progreso** | Inyección de `live_pose_se3` global, gaussiana longitudinal `ctx_x - live_x` y tubo extra $\gamma=0.65$ al vuelo. | Revisar cómo representar el orden dentro de las demos sin pasar coordenadas globales del mapa en la consulta viva $O_t$, evaluar si `data['aug']` ($\gamma=0.70$) ya resulta suficiente y analizar alternativas egocéntricas si apareciera ambigüedad perceptual. |
| **Propagación de Contexto ($\varphi$) y Difusión ($\psi$)** | Separación manual `Key`/`Value`, `live_modality_dropout` (sonda de 3 muestras) y puente determinista desde $I_4$. | Revisar la arquitectura estándar de grafos heterogéneos de *Instant Policy* (acciones demostradas como nodos geométricos 3D y *schedule* DDPM/DDIM estocástico) entrenando sobre la variedad de tareas del Catálogo $v2$. |
| **Muestreo de Escena ($\phi_e$) e Hiperparámetros** | Corte fijo `z_flu > -0.11` antes de FPS y pruebas puntuales de tamaño. | Revisar el muestreo sobre la nube completa sin el corte rígido en $Z$ (que puede descartar el `landing_pad` al volar alto) y evaluar en grilla $(L, T, \text{stride}, N_{\text{demos}}, T_{\text{exec}}, K, M)$ sobre la arquitectura final (Tarea 4.5 de `PLAN_IMPLEMENTACION.md`). |
