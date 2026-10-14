# Memoria de Estado del Proyecto: UAV Instant Policy

*Índice ejecutivo de estado técnico y punteros de referencia · Actualizado a Octubre de 2026*

---

## 📌 Documentos de Referencia Canónicos (Punto Único de Verdad)

Para evitar desincronizaciones, la información del proyecto se encuentra formalmente centralizada en:
1. **[Estrategia Teórica y Taxonomía](/docs/ESTRATEGIA_TEORICA.md):** Fundamentación matemática, preservación de Markov, escala $s=0.1$, LiDAR 360°, taxonomía de primitivas y análisis combinatorio de tareas.
2. **[Hoja de Ruta y Plan de Implementación](/docs/PLAN_IMPLEMENTACION.md):** Arquitectura por módulos, hitos de desarrollo (Hito 0 al Hito 4) y backlog de tareas operativas inmediatas.
3. **[Registro de Decisiones de Diseño (ADR)](/docs/DECISIONES_DISENO.md):** Registro permanente del "por qué" de cada decisión técnica, incluyendo la memoria histórica de enfoques descartados (cámara de 6 GB, pilotos monolíticos, etc.).
4. **[Guía de Troubleshooting](/docs/TROUBLESHOOTING.md):** Resolución rápida de incidencias en WSL2, PX4 SITL, MicroXRCEAgent y Gazebo.

---

## 🚀 Estado Actual del Stack (Hito 0 Completado)

* **Stack Base:** Ubuntu 24.04 (WSL2), ROS 2 Jazzy, Gazebo Harmonic y PX4 SITL (`v1.14/v1.15`).
* **Vehículo de Simulación:** Modelo estándar **`gz_x500`** (sin cámara física de profundidad), garantizando un RTF $\ge 1.0\times$.
* **Observación y Sensores:** LiDAR 3D omnidireccional 360° privilegiado en modo `range` ($R_{\text{max}} = 10.0\text{ m}$) implementado mediante `synthetic_lidar.py` (2048 puntos, FPS, marco egocéntrico FLU, escala métrica $s = 0.1$).
* **Controladores:** Arquitectura desacoplada bajo patrón Strategy:
  * `base_pilot.py`: Ciclo de vida de ROS 2, armado seguro en 50 Hz, QoS.
  * `waypoint_controller.py`: Control feed-forward interpretando `waypoints.yaml` (`traverse`, `slalom`, `overfly`, `orbit` con yaw centrípeto hacia el landmark).
  * `tracking_controller.py`: Prototipo de seguimiento dinámico en feedback.
* **Orquestación y Recolección:** `data_collector.py` ejecuta campañas automáticas de múltiples tareas y episodios:
  * Semántica fija a nivel tarea (`task_config.yaml`).
  * Aleatorización de poses 3D continua a nivel episodio con semillas deterministas (`waypoints.yaml`).
  * Telemetría ligera en rosbags MCAP ($< 5\text{ MB}$ por vuelo).
* **Postproceso:** `bag_to_heterodata.py` convierte los rosbags en grafos PyG `HeteroData` `.pt`. `visualize_dataset.py` permite validación visual 3D estática y generación de animaciones GIF.

---

## 🛠️ Próximas Tareas Inmediatas (Hitos 1 y 2)

1. **Generación Robusta de Tareas sin Reemplazo (`task_generator.py`):** Muestrear el espacio de las 64 firmas semánticas únicas sin colisiones aleatorias.
2. **Script de Auditoría y Partición (`audit_dataset.py`):** Validar la unicidad de firmas en `datasets/` y emitir `train_tasks.json`, `val_tasks.json` y `held_out_tasks.json`.
3. **Entorno Virtual para Machine Learning (`~/venv_ip`):** Aislar PyTorch, PyTorch Geometric y `rosbags` para cumplir con PEP 668 sin interferir con ROS 2 Jazzy.
4. **Procesamiento por Lotes Multiproceso (`bag_to_heterodata.py`):** Agregar `--dataset-dir` y paralelismo con `multiprocessing.Pool` para convertir cientos de episodios en segundos.
