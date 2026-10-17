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

## 🚀 Estado Actual del Stack (Hitos 0 y 1 Completados)

* **Stack Base:** Ubuntu 24.04 (WSL2), ROS 2 Jazzy, Gazebo Harmonic y PX4 SITL (`v1.14/v1.15`).
* **Vehículo de Simulación:** Modelo estándar **`gz_x500`** (sin cámara física de profundidad), garantizando un RTF $\ge 1.0\times$.
* **Observación y Sensores:** LiDAR 3D omnidireccional 360° privilegiado en modo `range` ($R_{\text{max}} = 10.0\text{ m}$) implementado mediante `synthetic_lidar.py` (2048 puntos, FPS, marco egocéntrico FLU, escala métrica $s = 0.1$).
* **Controladores:** Arquitectura desacoplada bajo patrón Strategy (`base_pilot.py`, `waypoint_controller.py`, `tracking_controller.py`).
* **Generación y Auditoría:** `task_generator.py` muestrea el catálogo combinatorio multi-longitud sin reemplazo; `audit_dataset.py` verifica cero duplicados y exporta particiones `train`, `val` y `held_out` sin fugas.
* **Orquestación y Postproceso:** `data_collector.py` recolecta rosbags MCAP ligeros ($< 5\text{ MB}$/vuelo); `bag_to_heterodata.py` y `visualize_dataset.py` procesan y animan las trayectorias en 3D.

---

## 🛠️ Próximas Tareas Inmediatas (Hito 1.5 y Hito 2 en `docs/PLAN_IMPLEMENTACION.md`)

1. **Hito 1.5 — Bloque A (Simulación y Recolección):**
   * Eliminar el `profile` fantasma de `get_task_signature` e incorporar `slalom` bilateral (`left`/`right`) y `circular_gate` en `task_generator.py` y `world_generator.py`.
   * Suavizar el paso por waypoints en `waypoint_controller.py`, corregir la mezcla de marcos ENU/NED en `tracking_controller.py` y añadir variación en la pose inicial de vuelo.
2. **Hito 1.5 — Bloque B (Postproceso LiDAR, Grafos $SE(3)$ y Empaquetado):**
   * Proyectar las nubes de `synthetic_lidar.py` en el marco estabilizado por gravedad (*Horizontal FLU*), acotar puntos de `landing_pad` y fijar semilla determinista en `fps_numpy`.
   * Rediseñar `bag_to_heterodata.py` para almacenar poses $SE(3)$ completas, los 6 keypoints del UAV, el horizonte $T=8$ de desplazamientos relativos futuros $T_{EA}$ sobre la trayectoria real (en lugar de setpoints escalonados globales) y el extractor de $L=10$ keyframes.
   * Reemplazar las copias duplicadas en `src/uav_ip/scripts/` (`synthetic_lidar.py` y `visualize_dataset.py`) por *thin wrappers* que importen desde `uav_ip/`.
3. **Hito 2 (Infraestructura ML y Aumento de Recuperación):**
   * Configurar `~/venv_ip`, añadir procesamiento por lotes multiproceso en `bag_to_heterodata.py` e implementar el aumento de datos correctivo del 30% (*Tube Augmentation*).
