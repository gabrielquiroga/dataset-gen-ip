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

## 🚀 Estado Actual del Stack (Hitos 0, 1, 1.5 y 2 Completados)

* **Stack Base:** Ubuntu 24.04 (WSL2), ROS 2 Jazzy, Gazebo Harmonic y PX4 SITL (`v1.14/v1.15`).
* **Vehículo de Simulación:** Modelo estándar **`gz_x500`** (sin cámara física de profundidad), garantizando un RTF $\ge 1.0\times$.
* **Observación y Sensores:** LiDAR 3D omnidireccional 360° privilegiado en modo `range` ($R_{\text{max}} = 10.0\text{ m}$) implementado mediante `synthetic_lidar.py` (2048 puntos, FPS acelerado por kernel SIMD C en $\sim 6\text{ ms}$, marco egocéntrico Horizontal FLU estabilizado por gravedad, escala métrica $s = 0.1$).
* **Controladores:** Arquitectura desacoplada bajo patrón Strategy (`base_pilot.py`, `waypoint_controller.py`, `tracking_controller.py`) con paso suave feed-forward en waypoints `pre` (`vel_ff=None`).
* **Generación y Auditoría:** `task_generator.py` con catálogo combinatorio dinámico balanceado, variantes `slalom` bilateral (`left`/`right`) y `circular_gate`; `audit_dataset.py` verifica cero duplicados y exporta particiones `train`, `val` y `held_out`.
* **Orquestación y Procesamiento Batch:** `data_collector.py` orquesta la recolección automática; `bag_to_heterodata.py` convierte rosbags por lotes multiproceso (`--dataset-dir`, `--workers`), almacena poses $SE(3)$, horizonte $T=8$, $L=10$ keyframes y genera aumentos de datos de recuperación (*30% Tube Augmentation*) en `data['aug']`.
* **Visualización y Wrappers:** `visualize_dataset.py` permite inspección estática, animación interactiva/GIF y visualización de recuperación con Tube Augmentation (`--show-aug`); *thin wrappers* en `scripts/` aseguran un punto único de verdad en `uav_ip/`.
* **Infraestructura ML:** Entorno virtual aislado configurable mediante `src/uav_ip/scripts/setup_venv_ip.sh` (`~/venv_ip`).

---

## 🛠️ Próximas Tareas Inmediatas (Hito 3 en `docs/PLAN_IMPLEMENTACION.md`)

1. **Hito 3 — Arquitectura del Modelo de Difusión sobre Grafos (GNN In-Context):**
   * Definir el `Dataset` y `DataLoader` de PyTorch Geometric para estructurar pares (Contexto $N-1$ demos, Query/Target 1 demo) con soporte para Tube Augmentation.
   * Adaptar el backbone de difusión sobre grafos heterogéneos de Instant Policy (MLP/Transformer geométrico + Denoising Diffusion Probabilistic Model / Flow Matching).
   * Implementar la función de pérdida de difusión sobre el horizonte de acción relativo $T=8$ en $SE(3)$.
   * Pipeline de validación offline sobre el conjunto de tareas *Held-Out*.
