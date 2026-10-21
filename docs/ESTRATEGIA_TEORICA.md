# Estrategia Teórica y Taxonomía de Tareas: Instant Policy Aplicado a Drones

*Documento de especificación metodológica, teórica y matemática · Octubre de 2026*

---

## 1. Resumen Ejecutivo

- **Objetivo.** Adaptar el paradigma de **Instant Policy** (*In-Context Imitation Learning via Graph Diffusion*, Vosylius y Johns, ICLR 2025) a la navegación autónoma y generación de trayectorias 3D para vehículos aéreos no tripulados (UAV). Alcance: simulación robótica de alta fidelidad con **PX4 SITL + ROS 2 Jazzy + Gazebo Harmonic**.
- **Unidad Fundamental de Aprendizaje.** El conjunto de demostraciones de una misma *pseudo-tarea*: en cada muestra de entrenamiento, $N-1$ demostraciones conforman el **Contexto** (el prompt visual-geométrico que define la regla a seguir) y la red predice las acciones futuras de la $N$-ésima demostración (**Query/Target**) en un entorno con diferente configuración espacial.
- **Definición de Tarea.** Una secuencia ordenada de primitivas expresadas de forma relativa en el marco local de *landmarks* físicos (aros, postes, plataformas), más eventos de cambio de estado de vuelo. Las demostraciones de una misma tarea solo difieren en las poses 3D $(x, y, z, \text{yaw})$ de los landmarks en el mundo y en la pose de arranque del dron.
- **Embodiment y Observación.** Conceptualización: Cuadricóptero equipado con un sensor **LiDAR 3D omnidireccional 360°** (análogo a un *Livox Mid-360*), implementado en simulación mediante **observación privilegiada** en modo `range` ($R_{\text{max}} = 10.0\text{ m}$). Esto preserva estrictamente la **Hipótesis de Markov** (eliminando la ceguera retrospectiva tras superar obstáculos) sin requerir redes recurrentes causales en Fase 1.
- **Eficiencia del Pipeline de Simulación.** En Gazebo SITL se utiliza el modelo estándar **`gz_x500`** (sin sensor físico de profundidad), alcanzando un *Real-Time Factor* (RTF) $\ge 1.0\times$. Las nubes de puntos de 2048 puntos no se guardan en disco, sino que se sintetizan bajo demanda mediante `synthetic_lidar.py`. Cada rosbag almacena únicamente telemetría ligera de PX4 ($< 5\text{ MB}$ por vuelo, reducción de $1000\times$ frente al enfoque obsoleto con cámara).
- **Factor de Escala Métrica ($s = 0.1$).** Se aplica de forma estricta un factor fijo $s = 0.1$ a todas las posiciones, deltas y nubes para prevenir el *aliasing espacial* severo que provocaría el encoder sinusoidal de Fourier de Instant Policy (cuyo período base es 2.0).

---

## 2. Convenciones de Procedencia y Lectura

Para preservar la integridad epistemológica del proyecto, cada afirmación técnica lleva una etiqueta explícita:

- **[Paper]**: Afirmado formalmente por Instant Policy (Vosylius y Johns, ICLR 2025).
- **[Inferencia]**: Deducción lógica a partir de la arquitectura del paper; sujeta a validación empírica en el dominio UAV.
- **[Heurística]**: Criterio de ingeniería de control/robótica sin respaldo experimental directo del paper original; a calibrar.
- **[Decisión del Proyecto]**: Decisión arquitectónica adoptada y validada en el repositorio.

---

## 3. Formulación Matemática de Instant Policy en Drones

### 3.1 Espacio de Observación y Acción [Paper, Sec. 3.1]
En cada instante temporal $t$, el estado del dron y su entorno se representan como:
* **Observación ($O_t$):** Nube de puntos egocéntrica segmentada $P_t \in \mathbb{R}^{M \times 3}$ ($M = 2048$), expresada en el marco local FLU del dron, más el estado binario de vuelo $s_t \in \{0, 1\}$ (0: landed, 1: airborne).
* **Acción ($A_t$):** Secuencia de $T$ desplazamientos futuros relativos en $SE(3)$ respecto al marco actual del dron:
  $$A_t = \{ (\Delta p_{t+k}, \Delta R_{t+k}) \}_{k=1}^T, \quad T = 8\dots 10$$
* **Demostraciones de Contexto ($\mathcal{D}_{\text{context}}$):** Conjunto de $N-1$ trayectorias completas de la misma tarea, reducidas a $L$ keyframes ($L = 10$):
  $$\mathcal{D}_{\text{context}} = \{ (O_{\ell}^{(j)}, A_{\ell}^{(j)}) \}_{\ell=1, \dots, L}^{j=1, \dots, N-1}$$
* **Distribución Aprendida:**
  $$P(A_t \mid O_t, \mathcal{D}_{\text{context}})$$

### 3.2 Representación en Grafo Heterogéneo (`HeteroData`) [Paper, Sec. 3.2]
Cada instante se modela como un grafo de paso de mensajes con dos tipos principales de nodos:
1. **Nodos de Escena (`scene`):** Puntos muestreados mediante *Farthest Point Sampling* (FPS) procesados por un encoder de ocupación geométrico preentrenado.
2. **Nodos de Agente (`agent`):** Representan la pose y estado del dron en cada instante.
3. **Aristas Temporales (`temporal`):** Conectan las poses consecutivas del dron a lo largo del tiempo:
   $$\mathcal{E}_{\text{temporal}} = \{ (t, t+1) \mid t = 1, \dots, K-1 \}$$
4. **Aristas Espaciales Relativas:** Conectan el efector con los puntos de la escena mediante embeddings posicionales sinusoidales:
   $$f_k(\Delta p) = \sin(2^k \pi \cdot s \cdot \Delta p), \quad s = 0.1$$

---

## 4. Preservación Estricta de la Hipótesis de Markov

### 4.1 El Dilema de la Cámara Frontal vs. Visión Omnidireccional
Instant Policy formula la política como un mapeo puramente reactivo y condicional del estado presente: la acción futura $A_t$ depende exclusivamente del snapshot actual $(O_t, \mathcal{D}_{\text{context}})$.

* **Fallo con Cámara Frontal (FOV $80^\circ$):**
  Al atravesar un aro o sobrevolar un poste, el obstáculo desaparece hacia la retaguardia del cono de visión. Durante los metros subsiguientes, el dron sufre de *ceguera retrospectiva* (*backward blindness*): no puede ver el landmark que acaba de pasar ni percibe aún el siguiente si está lejos o fuera de ángulo. Esto vacía $O_t$ y viola la propiedad de Markov, forzando a la red a detenerse o a requerir memoria temporal recurrente (RNN/LSTM o Transformer causal).
* **Solución: LiDAR 360° Privilegiado:**
  Al conceptualizar el embodiment como un UAV con sensor LiDAR omnidireccional 360° en modo `range` ($R_{\text{max}} = 10.0\text{ m}$), el snapshot sensorial $O_t$ retiene simultáneamente el landmark recién superado a popa y el siguiente a proa. Toda la información necesaria para inferir la fase de la tarea y la dirección de avance reside en la observación estática de $t$, garantizando que la hipótesis de Markov se cumpla de forma exacta.

---

## 5. Unificación Analítica de Marcos de Coordenadas y Estabilización por Gravedad

Para evitar rotaciones u offsets espaciales sistemáticos y silenciosos entre el simulador y el estimador de vuelo, y desacoplar las inclinaciones de aceleración del dron de la geometría observada, se fija la siguiente convención única:

1. **Marco Global de Simulación:** Gazebo World **ENU** (East-North-Up). Origen $(0,0,0)$ en el centro del mundo.
2. **Marco Local de Cuerpo del Dron:** **FLU** (Forward-Left-Up), centrado en el centro de masa del UAV.
3. **Conversión desde PX4 EKF2 (NED/FRD):**
   * Posición del dron en ENU:
     $$p_{\text{ENU}} = \begin{bmatrix} y_{\text{NED}} \\ x_{\text{NED}} \\ -z_{\text{NED}} \end{bmatrix}$$
   * Matriz de Rotación de Cuerpo FLU a Mundo ENU:
     $$R_{\text{ENU}}^{\text{FLU}} = M_{\text{world}} \cdot R_{\text{NED}}^{\text{FRD}} \cdot M_{\text{body}}$$
     donde:
     $$M_{\text{world}} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & -1 \end{bmatrix}, \quad M_{\text{body}} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \end{bmatrix}$$
4. **Marco Egocéntrico Estabilizado por Gravedad (*Horizontal FLU*):**
   Dado que un cuadricóptero subactuado se inclina en *roll* y *pitch* para acelerar horizontalmente, proyectar la nube rígidamente en el cuerpo del dron provocaría que las estructuras verticales aparezcan inclinadas. Se define el marco *Horizontal FLU* ($H$):
   * Pose de rotación: Conserva la guiñada (*yaw*, $\psi$) del dron pero fuerza $\text{roll} = 0, \text{pitch} = 0$:
     $$R_{\text{ENU}}^H = R_z(\psi_{\text{ENU}})$$
   * De este modo, el eje $+Z_H$ permanece en todo momento estrictamente alineado con la vertical gravitatoria opuesta a la gravedad.
5. **Proyección Egocéntrica de Puntos de la Escena:**
   Cualquier punto en coordenadas del mundo $P_w$ se proyecta al marco egocéntrico Horizontal FLU mediante:
   $$P_H = (P_w - p_{\text{ENU}}) \cdot R_{\text{ENU}}^H$$

---

## 6. Taxonomía Exhaustiva de Pseudo-Tareas

### 6.1 Catálogo de Landmarks Físicos (Geométricamente Distinguibles)
Para que el encoder geométrico local pueda distinguir el rol de cada objeto sin necesidad de IDs semánticos en el grafo [Paper, Sec. 5], los landmarks presentan firmas de forma diferenciadas:
* **Aro / Puerta (`gate`):** Estructura con abertura interior transitable. Normal principal orientada a lo largo del pasillo. Variantes: circular, cuadrada, hexagonal.
* **Poste / Torre (`pole`):** Obstáculo cilíndrico vertical cerrado (sin paso interior). Altura $\ge 3.0\text{ m}$.
* **Plataforma (`pad`):** Base cilíndrica baja en el suelo ($Z \approx 0.1\text{ m}$) para despegue y aterrizaje (*Home Pad* vs *Target Pad*).
* **Panel / Pared (`panel`, opcional):** Superficie vertical plana de bloqueo para maniobras de aproximación frontal.

### 6.2 Catálogo de Acciones Primitivas
Cada primitiva emite una secuencia discreta de waypoints locales relativos al marco del landmark:
1. **`traverse(gate)`:** Cruce perpendicular por el centro del aro.
   * `[pre]`: Aproximación alineada a la normal ($1.5\text{ m}$ antes).
   * `[center]`: Cruce del centro geométrico.
   * `[post]`: Salida sobre la normal ($1.5\text{ m}$ después).
2. **`overfly(obj, clearance=2.5)`:** Sobrepaso vertical por encima del objeto manteniendo margen de seguridad sobre la cota superior.
3. **`slalom(pole, side='right'|'left', offset=1.5)`:** Desvío lateral suave para esquivar el poste por un flanco sin circunvalarlo.
4. **`orbit(pole, radius=1.5, sweep=270°)`:** Maniobra de inspección centrípeta alrededor del poste:
   * Aproximación frontal a radio $R$, mirando hacia el poste ($\text{yaw} = 0^\circ$).
   * Cuadrante derecho a radio $R$, mirando al centro ($\text{yaw} = +90^\circ$).
   * Cuadrante posterior (+X) a radio $R$, mirando al centro ($\text{yaw} = 180^\circ$).
   * Cuadrante izquierdo a radio $R$, mirando al centro ($\text{yaw} = -90^\circ$).
   * Salida tangencial progresiva hacia $+X$ reanudando orientación del pasillo.
5. **`land_on(pad)`:** Descenso vertical controlado sobre el centro de la plataforma con cambio de estado de vuelo a *landed*.

---

## 7. Análisis Combinatorio y Protocolo de Partición Held-Out

### 7.1 Espacio de Firmas Semánticas Únicas
En una configuración canónica de pasillo 1D con 3 landmarks consecutivos ($L_1, L_2, L_3$):
* Cada landmark puede ser de tipo `gate` o `pole` ($2^3 = 8$ distribuciones de objetos).
* Si es `gate`, la primitiva física factible en avance monótono es `traverse` (1 opción).
* Si es `pole`, la primitiva puede ser `overfly`, `slalom` u `orbit` (3 opciones).

El número exacto de combinaciones semánticas únicas se calcula descomponiendo por cantidad de postes presentes:
$$\text{Total Firmas} = \binom{3}{0} \cdot 3^0 + \binom{3}{1} \cdot 3^1 + \binom{3}{2} \cdot 3^2 + \binom{3}{3} \cdot 3^3 = 1 + 9 + 27 + 27 = 64\text{ tareas únicas}$$

Una **Firma Semántica** se formaliza como la tupla inmutable:
$$\mathcal{S} = [(\text{tipo}_1, \text{prim}_1), (\text{tipo}_2, \text{prim}_2), (\text{tipo}_3, \text{prim}_3)]$$

### 7.2 El Riesgo de la Paradoja del Cumpleaños (Muestreo con Reemplazo)
Si un generador procedimental sortea aleatoriamente las primitivas de cada tarea con reemplazo:
* Con $n = 15$ tareas generadas, la probabilidad de que al menos dos tareas compartan la misma firma semántica es de:
  $$P(\text{colisión}) \approx 1 - e^{-\frac{15 \cdot 14}{2 \cdot 64}} \approx 80.6\%$$
* Con $n = 20$ tareas, $P(\text{colisión}) \ge 96.0\%$.

### 7.3 Decisión Metodológica: Muestreo sin Reemplazo y Partición Formal
1. **Muestreo sin Reemplazo:** Las $N_{\text{tasks}}$ tareas a recolectar deben extraerse **sin repetición** del conjunto discreto de las 64 firmas válidas.
2. **Partición Estricta Held-Out:**
   Para evaluar formalmente la generalización *in-context*, el catálogo de firmas se divide a priori:
   * **Entrenamiento (Train Tasks):** $\approx 70\%$ de las firmas (ej. 45 tareas).
   * **Validación (Val Tasks):** $\approx 15\%$ de las firmas (ej. 10 tareas en poses 3D nuevas).
   * **Generalización Fuera de Distribución (Held-Out Tasks):** $\approx 15\%$ de las firmas (ej. 9 combinaciones de primitivas **estrictamente nunca vistas durante el entrenamiento**).
   * *Regla inviolable:* Ninguna firma del conjunto Held-Out puede haber formado parte del entrenamiento bajo ningún conjunto de poses 3D.

---

## 8. Generación de Observaciones Privilegiadas (Sintetizador LiDAR)

### 8.1 Modelo Canónico de Superficie
Para cada clase de landmark se define un modelo canónico de alta resolución $P_{\text{can}}$ muestreado uniformemente por unidad de área:
* **Gate Toroidal/Cuadrado:** Barras tubulares de grosor $0.1\text{ m}$, luz interior de $1.4\text{ m} \times 1.5\text{ m}$.
* **Pole Cilíndrico:** Cilindro vertical de radio $0.1\text{ m}$ y altura $3.0\text{ m}$.
* **Pad:** Cilindro base de radio $0.75\text{ m}$ y grosor $0.1\text{ m}$.

### 8.2 Transformación y Filtrado Dinámico por Alcance (`range`)
En cada instante $t$, dada la pose del dron $T_{w,d}(t)$ y la pose ground-truth del landmark $T_{w,o}$:
1. Se proyectan los puntos del landmark al marco egocéntrico **Horizontal FLU** del dron (estabilizado por gravedad con $\text{roll}=0, \text{pitch}=0$):
   $$P_H^{(k)} = (T_{w,o} \cdot P_{\text{can}}^{(k)} - p_{\text{ENU}}) \cdot R_{\text{ENU}}^H$$
2. **Filtro de Alcance Omnidireccional:** Se seleccionan únicamente los puntos cuya distancia radial euclídea satisfaga:
   $$\| P_H^{(k)} \|_2 \le R_{\text{max}} = 10.0\text{ m}$$
3. Se concatenan los puntos visibles de todos los landmarks en el entorno.
4. **Submuestreo Uniforme por FPS Acelerado:** Se aplica *Farthest Point Sampling* mediante un kernel C vectorizado (SIMD `-O3 -march=native -ffast-math`) para reducir la nube combinada a exactamente $M = 2048$ puntos en $\sim 6\text{ ms}$.
5. **Escalado Métrica:** Se multiplica por el factor $s = 0.1$:
   $$P_{\text{scene}} = 0.1 \cdot P_{\text{FPS}}$$

---

## 9. Esquema de Salida y Representación de la Acción

Siguiendo el análisis de riesgo del Apéndice H de Instant Policy, se implementa una estrategia de fases:
1. **Fase 1 (Nativa Instant Policy):** Predicción de secuencias de $T=8$ acciones relativas en $SE(3)$ ($\Delta p$ escalado por $s = 0.1$, $\Delta q$) a 10 Hz extraídas directamente de la trayectoria de vuelo real.
2. **Fase 2 (Postproceso B-spline):** Ajuste de curvas polinomiales suaves sobre el rollout predicho por la red para alimentar el controlador de seguimiento de PX4.
3. **Fase 3 (CGD / Restricciones QP):** Acople futuro de guiado por difusión con restricciones cinemáticas y dinámicas (arXiv:2405.01758).

---

## 10. Aumento de Datos Sintético para Recuperación (*Tube Augmentation*) y Mitigación del Covariate Shift

### 10.1 Fundamento Teórico: El Desafío del *Covariate Shift* en Aprendizaje por Imitación
En *Behavioral Cloning* (BC) estándar (Ross & Bagnell, 2011), la política se entrena minimizando la pérdida sobre estados extraídos exclusivamente de la distribución del experto:
$$\min_\theta \mathbb{E}_{s \sim d^{\pi^*}} [\mathcal{L}(\pi_\theta(s), \pi^*(s))]$$
Sin embargo, durante la ejecución en bucle cerrado (*closed-loop*), pequeños errores de predicción $\epsilon$ acumulan desviaciones que empujan al UAV hacia estados no vistos $s \notin d^{\pi^*}$. Bajo distribución fuera de soporte, los errores compuestos crecen cuadráticamente con el horizonte temporal:
$$\mathcal{O}(\epsilon T^2)$$
provocando que el dron diverja irremediablemente de la trayectoria segura y colisione con el entorno o landmarks.

### 10.2 Formulación Matemática del Flujo Correctivo en $SE(3)$
Para dotar a la política de difusión de capacidad intrínseca de autocorrección sin requerir bucles interactivos online costosos (DAgger), se sintetiza un conjunto de demostraciones de recuperación dentro de un **tubo de perturbación $\epsilon$** (*Tube Augmentation*):

1. **Muestreo de Estados Perturbados:**
   Para un $30\%$ de los instantes temporales $t$, se aplica una perturbación estocástica a la pose nominal del dron $T_{WE}^t = (p_t, R_t) \in SE(3)$:
   $$\delta p \sim \mathcal{N}(0, \sigma_p^2 I_3), \quad \|\delta p\| \le 0.60\text{ m}, \quad \sigma_p = 0.25\text{ m}$$
   $$\delta \psi \sim \mathcal{N}(0, \sigma_\psi^2), \quad |\delta \psi| \le 15^\circ, \quad \sigma_\psi = 5^\circ$$
   La pose perturbada resultante en coordenadas del mundo ENU es:
   $$\tilde{T}_{WE}^t = \begin{bmatrix} R_z(\psi_t + \delta \psi) & p_t + \delta p \\ 0 & 1 \end{bmatrix}$$

2. **Generación de la Observación Perturbada:**
   Dado el modelo analítico de superficies de los landmarks, la nube de puntos LiDAR egocéntrica $\tilde{P}_{\text{scene}}^t$ se calcula evaluando directamente el sensor desde la pose perturbada $\tilde{T}_{WE}^t$ en el marco Horizontal FLU, aplicando el filtro de rango omnidireccional $R_{\text{max}} = 10.0\text{ m}$ y FPS submuestreado a 2048 puntos.

3. **Demostración de la Acción Correctiva hacia el Horizonte Nominal:**
   Dada la trayectoria futura de referencia del experto $\{ T_{WE}^{t+k} \}_{k=1}^T$ ($T=8$ pasos a $\sim 10\text{ Hz}$), la acción objetivo requerida para corregir el desvío se formula unívocamente mediante la transformación relativa entre la pose perturbada actual y el hito nominal futuro:
   $$\tilde{T}_{EA}^{(t,k)} = (\tilde{T}_{WE}^t)^{-1} \cdot T_{WE}^{t+k} \in SE(3)$$
   
   **Propiedad de Convergencia por Identidad de Grupo:**
   $$\tilde{T}_{WE}^t \cdot \tilde{T}_{EA}^{(t,k)} = \tilde{T}_{WE}^t \cdot (\tilde{T}_{WE}^t)^{-1} \cdot T_{WE}^{t+k} = T_{WE}^{t+k}$$
   Por construcción, la acción relativa etiquetada compensa el desplazamiento $\delta p$ y $\delta \psi$ y apunta de regreso a la trayectoria del experto en el horizonte anticipado. Que la política aprendida reproduzca esa convergencia en lazo cerrado es una hipótesis a evaluar experimentalmente, no una consecuencia garantizada.

4. **Perturbación del Estado Binario de Vuelo ($10\%$ flip):**
   Para robustecer el modelo en las fases críticas de despegue y aterrizaje y evitar que memorice el estado binario $s_t \in \{0, 1\}$ como un oráculo determinista:
   $$\tilde{s}_t = \begin{cases} 1 - s_t & \text{con probabilidad } 0.10 \\ s_t & \text{con probabilidad } 0.90 \end{cases}$$
   Esto fuerza al encoder geométrico a correlacionar la proximidad espacial de los puntos de la escena (distancia al suelo y a la plataforma) con la maniobra de aterrizaje, previniendo fallos si el estimador de aterrizaje de PX4 reporta retardos.

5. **Aislamiento en el Grafo Heterogéneo (`data['aug']`):**
   Para respetar el protocolo riguroso de *In-Context Imitation Learning*, las demostraciones de contexto que definen el prompt de la tarea en la GNN deben provenir estrictamente de ejecuciones nominales limpias. Por ello, los datos aumentados se encapsulan en el tipo de nodo independiente `data['aug']`, permitiendo que el entrenamiento de la difusión sobre el target query utilice recuperación mientras que el contexto $N-1$ se mantiene puro en `data['agent']`.

---

## 11. Fuentes y Bibliografía

1. **Instant Policy:** Vosylius y Johns. *In-Context Imitation Learning via Graph Diffusion*. ICLR 2025. arXiv:2411.12633. https://www.robot-learning.uk/instant-policy
2. **DAgger & Covariate Shift:** Ross, Gordon, Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. AISTATS 2011.
3. **CGD:** Kondo et al. *Constraint-Guided Diffusion Policies for UAV Trajectory Planning*. arXiv:2405.01758 (2024).
4. **In-Context Regression Diversity:** Raventós et al. *Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression*. NeurIPS 2023.
5. **PX4 SITL & Vehicle Odometry:** Dronecode Foundation. *PX4 Architectural Documentation (v1.14 / v1.15)*. https://docs.px4.io
6. **Rosbags Library:** Ternaris. *Pure Python ROS 2 MCAP/CDR Parser*. https://ternaris.gitlab.io/rosbags/
