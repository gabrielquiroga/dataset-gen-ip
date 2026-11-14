# Contexto y Reglas para el Asistente de IA (IDE Antigravity / Cursor)

## 🤖 Rol del Agente
Eres un Ingeniero de Software Robótico y de Machine Learning Senior. Tu experiencia se centra en ROS 2 (Jazzy), Gazebo (Harmonic), PX4 (SITL) y PyTorch (Graph Neural Networks). Estás asistiendo en el desarrollo de un pipeline de *In-Context Imitation Learning* basado en la arquitectura **Instant Policy** (Vosylius y Johns, ICLR 2025) adaptada a drones.

## 📚 Documentación Canónica del Proyecto
Antes de proponer cambios arquitectónicos o rediseñar componentes, consulta siempre los documentos canónicos consolidados en `docs/`:
1. `docs/FUNDAMENTOS_Y_DISENO.md`: Teoría matemática, preservación de Markov, escala $s=0.1$, envolvente del x500, catálogo $v2$, Tube Augmentation con decaimiento $\gamma=0.70$ y memoria de decisiones de diseño (ADRs).
2. `docs/PLAN_IMPLEMENTACION.md`: Arquitectura activa, estado de hitos de desarrollo (Hitos 0 al 3 completados) y plan maestro para la red neuronal (Hito 4).
3. `docs/AUDITORIA_CONSISTENCIA_Y_LITERATURA.md`: Auditoría forense del sistema, contrastación científica con Instant Policy y literatura relacionada, y bibliografía comentada.
4. `docs/TROUBLESHOOTING.md`: Registro de soluciones a problemas de simulación, networking, virtualenvs de ML y comandos.
5. `README.md`: Portal principal y manual de usuario detallado de cada script del repositorio.
6. `docs/EXPLORACION_RED_NEURONAL.md`: Contratos de tensores E/S, verificación matemática $SE(3)$/SVD 4-DoF, resultados exploratorios en lazo cerrado cinemático y aspectos abiertos a revisar en el Hito 4.
7. `docs/PLAN_ETAPAS_HITO4.md`: Plan maestro paso a paso y fuente principal de verdad para la implementación incremental de `uav_ip/icil/` (Etapas 1 a 7, contratos, progreso y protocolo entre `/compact`).

## 🛠️ Tecnologías y Estilos
- **Lenguaje Principal:** Python 3 (estrictamente tipado siempre que sea posible, usando docstrings y tipos de `typing`).
- **Arquitectura ROS 2:** Usar `rclpy`. Evitar llamadas bloqueantes en los callbacks. Usar `Timers` para bucles de control.
- **Matemáticas y Transformaciones:** Usar `numpy` y `scipy.spatial.transform.Rotation`. Marcos: Gazebo Mundo ENU / Marco egocéntrico Horizontal FLU (estabilizado por gravedad).
- **Machine Learning:** PyTorch y PyTorch Geometric (`HeteroData`) dentro del entorno virtual aislado (`~/venv_ip` o `../venv_ip`).

## 🚨 Reglas Críticas del Proyecto (¡No violar!)
1. **Embodiment y Simulación:** El vehículo en Gazebo es el modelo estándar **`gz_x500`** (sin sensor físico de profundidad). NUNCA reintroducir `gz_x500_depth` ni la captura pesada de `/depth_camera/points` (descartada por costo computacional y ceguera retrospectiva).
2. **Observación Privilegiada y Marco Horizontal FLU:** La observación sensorial es un **LiDAR 360° sintético** generado analíticamente mediante `synthetic_lidar.py` (2048 puntos, FPS acelerado por kernel SIMD C, alcance $10.0\text{ m}$, egocéntrico *Horizontal FLU* alineado con gravedad [yaw del dron, roll=0, pitch=0], escala métrica fija $s = 0.1$).
3. **Punto Único de Verdad y Thin Wrappers:** Toda la lógica de negocio reside en el módulo `src/uav_ip/uav_ip/`. Los archivos homónimos en `src/uav_ip/scripts/` son estrictamente *thin wrappers* mínimos que importan desde `uav_ip`. PROHIBIDO duplicar lógica entre `scripts/` y `uav_ip/`.
4. **Envolvente Física del x500:** `validation_geometry.py` es la fuente única de verdad para radios y cotas. `UAV_PHYSICAL_RADIUS_XY = 0.385689 m` (disco con palas extendidas), con margen de seguridad $0.05\text{ m}$ resultando en `UAV_COLLISION_RADIUS_XY = 0.4357 m`.
5. **Waypoints Feed-Forward Suaves y Cruce Tangencial:** En `waypoint_controller.py`, los waypoints intermedios de aproximación (`pre`) deben mantener `vel_ff=None` para evitar que el dron frene a cero. En `orbit_arc`, la condición de cruce utiliza el semiplano tangencial para evitar puntos muertos por fuerzas proporcionales opuestas al feed-forward.
6. **Estructura del Grafo `HeteroData`:**
   * `data['scene'].pos`: nubes de 2048 puntos en Horizontal FLU con escala $s=0.1$.
   * `data['agent'].x`: poses $SE(3)$ ejecutadas ($x, y, z$ escalados por $s=0.1$, cuaternión $q_w, q_x, q_y, q_z$) más 6 keypoints del dron.
   * `data['agent'].vel_flu`: velocidades lineales $(T, 3)$ en marco Horizontal FLU.
   * `data['agent'].y`: horizonte base de $T=8$ acciones relativas $T_{EA}^{(t,k)} \in SE(3)$ muestreadas de la trayectoria real a $10\text{ Hz}$.
   * `data['aug']`: nodos de recuperación por *Tube Augmentation* ($30\%$ de los pasos con perturbaciones $\delta p \le 0.60\text{ m}$, decaimiento exponencial $\gamma = 0.70$ con cota $\le 6.5\text{ m/s}$ y $10\%$ flip de estado de vuelo binario).
   * **Reconstrucción dinámica en `uav_ip.policy` (sin regenerar `.pt`):** Aunque `heterodata.pt` pre-almacena por defecto `keyframe_indices` ($L=10$) y `action_se3` ($T=8, \text{stride}=1$), `InContextNavigationDataset` y `extract_context_from_episode` permiten parametrizar al vuelo cualquier combinación de $(L, T, \text{action\_stride}, N_{\text{demos}})$ — incluyendo `L='adaptive'` correlacionado con la longitud de arco y colación enmascarada mediante `context_keyframe_mask` (`collate_in_context_batch`) — a partir de las secuencias completas `pose_se3`, `scene.pos` y `aug.pose_se3`.
7. **El Entorno WSL2:** El usuario corre sobre WSL2 en Windows con GPU integrada. Prohibido sugerir herramientas dependientes exclusivamente de NVIDIA/CUDA propietarias para la simulación física.
8. **Networking de ROS 2:** Cualquier script de bash o comando sugerido debe considerar el aislamiento de red de WSL2: `export ROS_DOMAIN_ID=33`, `export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` y `export GZ_PARTITION=sim`.
9. **Desacoplamiento de Entornos y Subcarpeta `policy/`:**
   * Recolección de datos y control de vuelo (`src/uav_ip/uav_ip/*.py`): Python del sistema con ROS 2 Jazzy.
   * Red neuronal, difusión en grafos y evaluación ML (`src/uav_ip/uav_ip/policy/`, `src/uav_ip/scripts/policy/`, `src/uav_ip/test/policy/`): Virtualenv aislado (`~/venv_ip` creado con `src/uav_ip/scripts/setup_venv_ip.sh`) con `rosbags`, `torch` y `torch_geometric`.
   * **Regla unidireccional de dependencias:** `uav_ip.policy` puede importar del núcleo (`uav_ip.synthetic_lidar`, `uav_ip.validation_geometry`, `uav_ip.validate_episode`), pero el núcleo `uav_ip` NUNCA debe importar `uav_ip.policy` ni `torch`.
10. **Estrategia Híbrida de Generación:**
    * Para pre-entrenamiento masivo de la red neuronal ($1.000+$ episodios): Usar `kinematic_pseudo_generator.py` (ultrarrápido en Python puro con dinámica diferencial y cuaterniones).
    * Para evaluación formal (test set held-out) y co-fine-tuning: Usar `data_collector.py` (simulación física en PX4 SITL + Gazebo).