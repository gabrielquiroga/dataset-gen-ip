# Requisitos Arquitectónicos del Dataset (Instant Policy)

Para que el modelo de Aprendizaje por Imitación en Contexto (ICIL) converja, el pipeline de generación de datos debe cumplir estrictamente con los siguientes requisitos teóricos y matemáticos extraídos de la especificación de Instant Policy.

## 1. Estructura Semántica y Taxonomía de Tareas
- **Pseudo-Task (Tarea):** Una tarea NO es una disposición espacial fija, sino una **regla semántica** (ej. "Cruzar aros 1 y 3 ignorando el 2", o "Cruzar en orden inverso").
- **Episodios (Demostraciones):** Para CADA pseudo-task, se deben grabar múltiples vuelos ($N=3$ a $5$).
- **Variación Espacial Intra-Grupo (Obligatoria):** En cada vuelo de una misma tarea, las posiciones y orientaciones de los aros en 3D DEBEN variar aleatoriamente. Si la disposición 3D es idéntica en las $N$ demostraciones, el dato es inválido.
- **Diversidad Inter-Tareas:** El dataset completo debe contener miles de grupos de pseudo-tareas únicas (1.000-5.000 para POC) mezclando cantidad de aros, orden de paso, perfil de trayectoria, y comportamientos especiales.
- **In-Context Learning:** Durante el entrenamiento, la red consumirá $N-1$ vuelos como "Contexto" (el prompt visual/espacial de cómo cumplir la regla en la disposición X) y predecirá las acciones del vuelo restante ($N$) en una nueva disposición Y como "Query".

## 2. Definición del Ground Truth y Acciones ($SE(3)$)
- **NO usar el Setpoint:** La etiqueta (Target) NO debe ser el comando `TrajectorySetpoint` crudo, ya que ignora la dinámica.
- **USAR Odometría Real:** La acción $A_t$ a predecir es el desplazamiento real ejecutado entre $t$ y $t+1$.
- **Horizonte de Predicción ($T$):** Cada acción objetivo debe empaquetarse como un horizonte de $T$ pasos futuros ($T=8$ a $10$ waypoints).
- **Parametrización Desacoplada en $SE(3)$:** Las acciones se representan como desplazamientos relativos de traslación ($\Delta x, \Delta y, \Delta z$) y deltas de rotación de la cámara.

## 3. Marco de Referencia Espacial
- **Todo es Egocéntrico (Invariancia):** Todas las coordenadas de las nubes de puntos y de las trayectorias futuras DEBEN estar expresadas en el sistema de coordenadas local del dron ($T_{dron, mundo}^{-1} \cdot P$) en cada instante $t$. No se deben almacenar en coordenadas globales de Gazebo.

## 4. Procesamiento de la Observación (Nube de Puntos)
- **Segmentación Estricta:** La observación visual en $t$ debe consistir exclusivamente en la nube de puntos 3D de los objetos de interés (aros/obstáculos). Filtrar suelo, paredes y el propio dron.
- **Submuestreo Fijo (FPS):** Cada observación debe submuestrearse a una densidad constante mediante *Furthest Point Sampling* (ej. $M = 2048$ puntos).

## 5. Dimensión Temporal (Keyframing)
- **Submuestreo Temporal:** Las trayectorias continuas de alta frecuencia (ej. 50Hz) no se guardan como cadenas densas.
- **Extracción de Nodos Temporales:** Cada demostración debe reducirse a un número fijo de $L$ waypoints clave (ej. $L=10$).
- **Heurísticas de Extracción:** Instantes críticos como el punto de inicio/fin, paso por el centro exacto de los aros, cambios bruscos de rumbo o desaceleraciones.

## 6. Estructuración en Grafos Heterogéneos
- Cada muestra compilada del dataset debe permitir construir un grafo (ej. `HeteroData` en PyTorch Geometric) con:
  1. **Nodos de Objeto/Escena:** Características de la nube de puntos.
  2. **Nodos de Agente:** Poses pasadas y actuales del dron.
  3. **Nodos de Acción Futura:** Poses "imaginadas" del dron a predecir.
- **Atributos de Aristas:** Distancias relativas entre nodos codificadas con embeddings posicionales de Fourier (funciones sinusoidales).