# Requisitos Arquitectónicos del Dataset (Instant Policy)
Para que el modelo de Aprendizaje por Imitación en Contexto converja, el pipeline de generación de datos debe cumplir estrictamente con los siguientes requisitos teóricos y matemáticos extraídos del paper original.

## 1. Estructura Semántica: Tareas vs Episodios
- Pseudo-Task (Tarea): Se define como una configuración espacial única del entorno (ej. Posición exacta de los 3 aros y el punto de aterrizaje).
- Episodios (Demostraciones): Para CADA pseudo-task, se deben grabar múltiples vuelos (idealmente entre 2 y 5).
- In-Context Learning: Durante el entrenamiento, la red consumirá $N-1$ vuelos como "Contexto" (el prompt de cómo volar en ese circuito) y predecirá las acciones del vuelo restante ($N$) como "Query".
- Implicación en Código: El randomizador debe mover los aros -> Grabar 3 vuelos completos -> Mover aros -> Grabar 3 vuelos.

## 2. Definición del Ground Truth (La Acción)
- NO usar el Setpoint: La etiqueta (Target) de la red neuronal NO debe ser el comando TrajectorySetpoint que envió el piloto experto. Esto ignoraría la inercia real y la dinámica del viento de PX4.
- USAR la Odometría Real: La acción $A_t$ a predecir se define como el desplazamiento real ejecutado entre el estado actual y el siguiente: $A_t = Poses(T_{t+k}) \ominus Poses(T_t)$.

## 3. Marco de Referencia Espacial
- Todo es Egocéntrico: Las Redes Neuronales no generalizan bien en coordenadas absolutas del mundo de simulación.
- La Nube de Puntos debe estar en el marco del sensor (camera_link).Las acciones ($A_t$) deben expresarse relativas al marco base del dron (base_link). "Avanzar 1 metro", no "Ir a la coordenada X=50 del mapa".

## 4. Procesamiento de la Observación (Nube de Puntos)
- Segmentación Estricta: La red fallará si la nube de puntos contiene elementos distractores masivos (el cielo, paredes lejanas, el plano del suelo completo). Solo deben quedar los puntos pertenecientes a los aros, plataformas o paredes inmediatas de colisión.
- Voxelización: La nube cruda contiene miles de puntos, lo cual es inmanejable para una GNN. Se debe aplicar Voxel Downsampling (vía Open3D) para reducir cada frame a una cantidad manejable (ej. 128 o 256 nodos representativos).

## 5. Dimensión Temporal (Keyframing)
- Sin Flujo Continuo: El paper descarta la alimentación densa a 30Hz. La ventana de contexto colapsaría.
- Extracción de Nodos Temporales: Cada vuelo continuo de 200 segundos debe reducirse a una secuencia de aproximadamente $L=10$ a $15$ "Keyframes".
- Heurísticas de Keyframe:
   - Inicio de vuelo.
   - El instante exacto de paso por el centro de un aro.
   - Instantes de alta desaceleración o cambio drástico de rumbo.
   - El aterrizaje (Touchdown).