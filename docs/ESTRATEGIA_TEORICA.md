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

## 5. Unificación Analítica de Marcos de Coordenadas

Para evitar rotaciones u offsets espaciales sistemáticos y silenciosos entre el simulador y el estimador de vuelo, se fija la siguiente convención única para todo el dataset:

1. **Marco Global de Simulación:** Gazebo World **ENU** (East-North-Up). Origen $(0,0,0)$ en el centro del mundo.
2. **Marco Local del Dron:** **FLU** (Forward-Left-Up), centrado en el centro de masa del UAV.
3. **Conversión desde PX4 EKF2 (NED/FRD):**
   * Posición del dron en ENU:
     $$p_{\text{ENU}} = \begin{bmatrix} y_{\text{NED}} \\ x_{\text{NED}} \\ -z_{\text{NED}} \end{bmatrix}$$
   * Matriz de Rotación de Cuerpo FLU a Mundo ENU:
     $$R_{\text{ENU}}^{\text{FLU}} = M_{\text{world}} \cdot R_{\text{NED}}^{\text{FRD}} \cdot M_{\text{body}}$$
     donde:
     $$M_{\text{world}} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & -1 \end{bmatrix}, \quad M_{\text{body}} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \end{bmatrix}$$
4. **Proyección Egocéntrica de Puntos de la Escena:**
   Cualquier punto en coordenadas del mundo $P_w$ se proyecta al marco local del dron FLU mediante:
   $$P_{\text{dron}} = (P_w - p_{\text{ENU}}) \cdot R_{\text{ENU}}^{\text{FLU}}$$

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
1. Se proyectan los puntos del landmark al marco local FLU del dron:
   $$P_{\text{dron}}^{(k)} = T_{w,d}(t)^{-1} \cdot T_{w,o} \cdot P_{\text{can}}^{(k)}$$
2. **Filtro de Alcance Omnidireccional:** Se seleccionan únicamente los puntos cuya distancia radial euclídea satisfaga:
   $$\| P_{\text{dron}}^{(k)} \|_2 \le R_{\text{max}} = 10.0\text{ m}$$
3. Se concatenan los puntos visibles de todos los landmarks en el entorno.
4. **Submuestreo Uniforme por FPS:** Se aplica *Farthest Point Sampling* para reducir la nube combinada a exactamente $M = 2048$ puntos con máxima cobertura espacial.
5. **Escalado Métrica:** Se multiplica por el factor $s = 0.1$:
   $$P_{\text{scene}} = 0.1 \cdot P_{\text{FPS}}$$

---

## 9. Esquema de Salida y Representación de la Acción

Siguiendo el análisis de riesgo del Apéndice H de Instant Policy, se implementa una estrategia de fases:
1. **Fase 1 (Nativa Instant Policy):** Predicción de secuencias densas de pequeños desplazamientos relativos $(\Delta x, \Delta y, \Delta z, \Delta \text{yaw})$ a 10–25 Hz, escalados por $s = 0.1$.
2. **Fase 2 (Postproceso B-spline):** Ajuste de curvas polinomiales suaves sobre el rollout predicho por la red para alimentar el controlador de seguimiento de PX4.
3. **Fase 3 (CGD / Restricciones QP):** Acople futuro de guiado por difusión con restricciones cinemáticas y dinámicas (arXiv:2405.01758).

---

## 10. Fuentes y Bibliografía

1. **Instant Policy:** Vosylius y Johns. *In-Context Imitation Learning via Graph Diffusion*. ICLR 2025. arXiv:2411.12633. https://www.robot-learning.uk/instant-policy
2. **CGD:** Kondo et al. *Constraint-Guided Diffusion Policies for UAV Trajectory Planning*. arXiv:2405.01758 (2024).
3. **In-Context Regression Diversity:** Raventós et al. *Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression*. NeurIPS 2023.
4. **PX4 SITL & Vehicle Odometry:** Dronecode Foundation. *PX4 Architectural Documentation (v1.14 / v1.15)*. https://docs.px4.io
5. **Rosbags Library:** Ternaris. *Pure Python ROS 2 MCAP/CDR Parser*. https://ternaris.gitlab.io/rosbags/
