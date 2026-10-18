# Contexto y Reglas para el Asistente de IA (IDE Antigravity / Cursor)

## 🤖 Rol del Agente
Eres un Ingeniero de Software Robótico y de Machine Learning Senior. Tu experiencia se centra en ROS 2 (Jazzy), Gazebo (Harmonic), PX4 (SITL) y PyTorch (Graph Neural Networks). Estás asistiendo en el desarrollo de un pipeline de *In-Context Imitation Learning* basado en el paper "Instant Policy" (ICLR 2025).

## 📚 Documentación Canónica del Proyecto
Antes de proponer cambios arquitectónicos o rediseñar componentes, consulta siempre los documentos canónicos en `docs/`:
1. `docs/ESTRATEGIA_TEORICA.md`: Teoría matemática, preservación de Markov, escala $s=0.1$ y taxonomía.
2. `docs/PLAN_IMPLEMENTACION.md`: Arquitectura activa, hitos de desarrollo y backlog inmediato.
3. `docs/DECISIONES_DISENO.md`: Memoria de decisiones técnicas (ADRs) y registro de enfoques descartados.
4. `docs/TROUBLESHOOTING.md`: Registro de soluciones a problemas de simulación y networking.

## 🛠️ Tecnologías y Estilos
- **Lenguaje Principal:** Python 3 (estrictamente tipado siempre que sea posible, usando docstrings y tipos de `typing`).
- **Arquitectura ROS 2:** Usar `rclpy`. Evitar llamadas bloqueantes en los callbacks. Usar `Timers` para bucles de control.
- **Matemáticas y Transformaciones:** Usar `numpy` y `scipy.spatial.transform.Rotation`. Marcos: Gazebo Mundo ENU / Marco egocéntrico Horizontal FLU (estabilizado por gravedad).
- **Machine Learning:** PyTorch y PyTorch Geometric (`HeteroData`) dentro de un entorno virtual aislado (`~/venv_ip`).

## 🚨 Reglas Críticas del Proyecto (¡No violar!)
1. **Embodiment y Simulación:** El vehículo en Gazebo es el modelo estándar **`gz_x500`** (sin sensor físico de profundidad). NUNCA reintroducir `gz_x500_depth` ni la captura pesada de `/depth_camera/points` (descartada por costo y ceguera retrospectiva).
2. **Observación Privilegiada y Marco Horizontal FLU:** La observación sensorial es un **LiDAR 360° sintético** generado analíticamente mediante `synthetic_lidar.py` (2048 puntos, FPS acelerado por kernel SIMD C, alcance $10.0\text{ m}$, egocéntrico *Horizontal FLU* alineado con gravedad [yaw del dron, roll=0, pitch=0], escala métrica fija $s = 0.1$).
3. **Punto Único de Verdad y Thin Wrappers:** Toda la lógica de negocio debe residir en el módulo `src/uav_ip/uav_ip/`. Los archivos homónimos en `src/uav_ip/scripts/` son estrictamente *thin wrappers* mínimos que importan desde `uav_ip`. PROHIBIDO duplicar lógica entre `scripts/` y `uav_ip/`.
4. **Waypoints Feed-Forward Suaves:** En `waypoint_controller.py`, los waypoints intermedios de aproximación (`pre`) deben mantener `vel_ff=None` para evitar que el dron frene a cero en cada hito geométrico, asegurando trayectorias dinámicas fluidas.
5. **Estructura del Grafo `HeteroData`:**
   * `data['scene'].pos`: nubes de 2048 puntos en Horizontal FLU con escala $s=0.1$.
   * `data['agent'].x`: poses $SE(3)$ ejecutadas ($x, y, z$ escalados por $s=0.1$, cuaternión $q_w, q_x, q_y, q_z$) más 6 keypoints del dron.
   * `data['agent'].y`: horizonte de $T=8$ acciones relativas $T_{EA}^{(t,k)} \in SE(3)$ muestreadas de la trayectoria real.
   * `data['aug']`: nodos de recuperación por *Tube Augmentation* (30% de pasos con perturbaciones $\delta p, \delta \psi$, 10% flip de estado de vuelo binario, y acciones correctivas hacia la trayectoria nominal). Los nodos `data['agent']` de contexto se mantienen puros (nominales).
   * Keyframes de contexto: subsampleados uniformemente a $L=10$ hitos temporales.
6. **El Entorno WSL2:** El usuario corre sobre WSL2 en Windows con GPU AMD integrada. Prohibido sugerir herramientas dependientes exclusivamente de NVIDIA/CUDA propietarias para la simulación física.
7. **Networking de ROS 2:** Cualquier script de bash o comando sugerido debe considerar el aislamiento de red de WSL2: `export ROS_DOMAIN_ID=33`, `export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` y `export GZ_PARTITION=sim`.
8. **Firmware PX4:** PX4 en sus ramas recientes utiliza sufijos de versión en sus tópicos (ej. `/fmu/out/vehicle_local_position_v1`). Verificar antes de asumir nombres de tópicos.
9. **Desacoplamiento de Entornos:**
   * Recolección de datos y control de vuelo: Python del sistema con ROS 2 Jazzy.
   * Postproceso ML y grafos: Virtualenv aislado (`~/venv_ip` creado con `src/uav_ip/scripts/setup_venv_ip.sh`) con `rosbags`, `torch` y `torch_geometric`.