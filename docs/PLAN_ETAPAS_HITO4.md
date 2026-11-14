# Plan Paso a Paso de Implementación — Hito 4 (`uav_ip/icil`)

> **Documento fuente principal del Hito 4.** Diseñado para trabajar de forma secuencial, humana e incremental (etapa por etapa) y sobrevivir a compactaciones de contexto (`/compact`) sin perder decisiones, contratos de tensores ni criterios de verificación.

---

## 1. Metodología de Trabajo y Reglas del Proyecto

1. **Workflow humano e incremental:** Se avanza una etapa a la vez. Cada módulo debe ser compacto (~80–220 líneas por archivo), legible de arriba abajo y verificable de forma independiente antes de pasar a la etapa siguiente.
2. **Prioridad funcional primero, robustez después:** Primero aseguramos flujo extremo-a-extremo correcto y tests mínimos claros; mantenemos parametrización limpia en hiperparámetros clave (`L`, `T`, `action_stride`, `N_demos`, `M`, `K`, `T_exec`) sin sobre-ingeniería temprana.
3. **Rol de asistente vs. usuario:**
   - El asistente revisa el repositorio, diseña el código de la etapa actual, propone el código exacto y verifica resultados/errores.
   - **El asistente NO modifica archivos `.py` directamente** salvo pedido explícito: el usuario escribe o pega el código propuesto y ejecuta las pruebas en `~/venv_ip`.
   - El asistente sí mantiene actualizado este documento (`docs/PLAN_ETAPAS_HITO4.md`) al completar o ajustar una etapa.
4. **Protocolo para `/compact`:** Al terminar y verificar cada etapa:
   1. Actualizar la **Tabla Maestra de Progreso** (Sección 2) y anotar valores empíricos observados en la etapa recién terminada (por ejemplo: cotas del dataset, checkpoint generado, métricas de validación).
   2. Ejecutar `/compact`.
   3. En la nueva conversación, indicar simplemente: *"Leé `docs/PLAN_ETAPAS_HITO4.md` y sigamos con la Etapa N"*.
5. **Reglas arquitectónicas heredadas del repo:**
   - **Dependencia unidireccional:** Los módulos base de `uav_ip` nunca deben importar `torch`, `torch_geometric` ni `uav_ip.icil`. Todo el código ML vive en `src/uav_ip/uav_ip/icil/` y `src/uav_ip/scripts/icil/` y se ejecuta con `~/venv_ip/bin/python3`.
   - **Código exploratorio legado (`src/uav_ip/uav_ip/policy/`):** No importar ni reutilizar nada de `policy/`. La nueva implementación es 100% limpia en `icil/`. *(Nota de auditoría: en `policy/in_context_dataset.py` líneas 716–727 quedó un bug con extensiones `.txt` en `split_map` y una condición invertida `if split_file.is_file(): raise FileNotFoundError(...)`; evitarlo por completo en `icil/dataset.py`).*

---

## 2. Tabla Maestra de Progreso

| Etapa | Nombre | Archivos | Criterio de Verificación | Estado |
| :---: | :--- | :--- | :--- | :---: |
| **1** | Geometría $SE(3)$ 4-DoF, Keypoints, Flow y SVD | `src/uav_ip/uav_ip/icil/geometry.py`, `src/uav_ip/test/icil/test_geometry.py` | Round-trips `< 1e-5`, SVD 4-DoF exacto `< 1e-5`, invarianza $SE(3)$ | ✅ Completada (pend. refactor s=0.1) |
| **2** | Cargador de Dataset, Splits JSON y Batching | `src/uav_ip/uav_ip/icil/dataset.py`, `src/uav_ip/test/icil/test_dataset.py` | Splits disjuntos JSON cargados, formas exactas, 0 NaNs, cotas medidas | ⏳ Pendiente |
| **3** | Encoder de Escena $\phi_e$ + Pre-entrenamiento Ocupancia 3D | `src/uav_ip/uav_ip/icil/scene_encoder.py`, `src/uav_ip/scripts/icil/pretrain_scene_encoder.py`, `src/uav_ip/test/icil/test_scene_encoder.py` | Acc. validación > 90% en primitivas 3D + `checkpoints/scene_encoder.pt` | ⏳ Pendiente |
| **4** | Grafo Local $\sigma$ (Escena Congelada + Estado UAV) | `src/uav_ip/uav_ip/icil/scene_encoder.py`, `src/uav_ip/test/icil/test_scene_encoder.py` | Forward/backward `(B, 6, d_model)` OK; $\phi_e$ congelado, $\sigma$ con gradientes | ⏳ Pendiente |
| **5** | Red de Política ($\varphi \to \psi$), Difusión DDPM/DDIM y Test Dorado | `src/uav_ip/uav_ip/icil/policy_net.py`, `src/uav_ip/test/icil/test_policy_net.py` | Overfit en 1 episodio + Test Context-Swap lateral invertido exitoso | ⏳ Pendiente |
| **6** | Evaluador en Lazo Cerrado Cinemático + PCHIP 50 Hz | `src/uav_ip/uav_ip/icil/rollout_eval.py`, `src/uav_ip/test/icil/test_rollout_eval.py` | Oráculo `accepted` (`ADE=0.0`) + rollout del modelo overfiteado exitoso | ⏳ Pendiente |
| **7** | Entrenamiento Completo (Kin + Co-FT SITL), Ablaciones y Piloto ROS 2 | `src/uav_ip/scripts/icil/train_policy.py`, `src/uav_ip/uav_ip/icil/instant_policy_pilot.py` | Val `Success >= 70%`, tabla de ablaciones (Tarea 4.5) y vuelo SITL | ⏳ Pendiente |

---

## 3. Mapa de Directorios y Módulos Objetivo

```text
src/uav_ip/
├── uav_ip/
│   └── icil/
│       ├── __init__.py
│       ├── geometry.py              # Etapa 1: Keypoints, SE(3) 4-DoF, incremental/acumulado, flow + SVD
│       ├── dataset.py               # Etapa 2: Splits JSON, L keyframes, horizonte T/stride, Tube Aug, collate
│       ├── scene_encoder.py         # Etapas 3-4: SinusoidalPosEmb3D, SceneEncoderPhi (phi_e), OccupancyDecoderPsi, LocalStateEncoderSigma (sigma)
│       ├── policy_net.py            # Etapa 5: ContextGraphPhi (varphi), ActionDenoisingGraphPsi (psi), DDPM loss, DDIM sampling
│       ├── rollout_eval.py          # Etapa 6: Lazo cerrado cinemático 10 Hz -> PCHIP 50 Hz -> validate_episode_data
│       └── instant_policy_pilot.py  # Etapa 7B: Nodo ROS 2 (hereda de BasePilot) para PX4 SITL + Gazebo
├── scripts/
│   └── icil/
│       ├── pretrain_scene_encoder.py # Etapa 3: Pre-entrenamiento de phi_e por ocupancia implícita 3D
│       ├── train_policy.py           # Etapa 7A: Pre-entrenamiento kinemático, co-fine-tuning SITL y ablaciones
│       └── eval_ablations.py         # Etapa 7A (opcional): Barrido de hiperparámetros M, L, T/T_exec, K, N_demos
└── test/
    └── icil/                         # Tests pytest por etapa (con pytest.importorskip("torch") para no romper suite ROS 2)
        ├── test_geometry.py          # Etapa 1
        ├── test_dataset.py           # Etapa 2
        ├── test_scene_encoder.py     # Etapas 3-4
        ├── test_policy_net.py        # Etapa 5 (Test Dorado + Context-Swap)
        └── test_rollout_eval.py      # Etapa 6
```

---

## 4. Decisiones Arquitectónicas Críticas y Errores a Evitar

1. **Backbone geométrico $\phi_e$ pre-entrenado y congelado (Tarea 4.2):**
   - La nube sintética de 2048 puntos (Horizontal FLU, escala $s=0.1$) pasa por FPS + $k$-NN (PointNet++ / Set Abstraction) para producir $M$ tokens locales `(B, M, d_model)` con coordenadas centroides `(B, M, 3)`.
   - Se pre-entrena con un decodificador residual $\psi_e$ mediante clasificación binaria de **ocupancia implícita 3D** sobre primitivas sintéticas del dominio UAV (`pole`, `square_gate`, `circular_gate`, `landing_pad`), y luego se congelan todos los pesos de $\phi_e$ (`requires_grad=False`, `eval()`). Evita intentar aprender geometría 3D desde cero usando solo la señal de difusión en pocas tareas.
2. **Nube completa sin filtrado de suelo ($Z$):**
   - **No** descartar puntos con `z_flu > -0.11`. El UAV necesita ver el suelo y el `landing_pad` en todo momento para descender y alinear aterrizaje.
3. **Splits estrictos desde JSON oficiales:**
   - Leer `datasets/<tipo>/splits/{train,val,held_out}_tasks.json` (mapeando `"test"` a `"held_out"`). Exigir existencia explícita (`if not split_file.is_file(): raise FileNotFoundError(...)`) para impedir fuga entre train/val/held_out.
4. **Invarianza $SE(3)$ estricta en entradas de red:**
   - **Nunca** pasar `live_pose_se3` global como entrada a $\sigma$, $\varphi$ o $\psi$, ni restar features manualmente (`ctx_x - live_x`).
   - Todas las observaciones locales viven en el marco Horizontal FLU del instante correspondiente, y la relación espacial entre nodos se expresa mediante coordenadas/aristas relativas codificadas con `SinusoidalPosEmb3D` (`Fourier(p_dst - p_src)`).
5. **Acciones incrementales $\delta T_k$, difusión desacoplada $[\Delta\mathbf{p}_t, \Delta\mathbf{p}_r]$ y sin parches ad-hoc:**
   - Representamos cada paso del horizonte $k=1\dots T$ como transformación incremental paso-a-paso $\delta T_k \in SE(3)$ 4-DoF ($T_k = T_{k-1} \cdot \delta T_k$).
   - **No** introducir multiplicadores `flow_scale=10` ni recortes duros `ActionBounds4DoF` en la salida. En su lugar, normalizar explícitamente las acciones en el espacio geométrico $[\Delta x, \Delta y, \Delta z, \Delta \psi]$ usando cotas físicas medidas en Etapa 2 antes de inyectar ruido DDPM, y decodificar flujo traslacional/rotacional con SVD 4-DoF exacto.
6. **Grafos heterogéneos reales con `TransformerConv`:**
   - Tanto $\sigma$ como $\varphi$ y $\psi$ usan `torch_geometric.nn.TransformerConv` con `edge_attr` Fourier de desplazamientos 3D entre nodos conectados, sin splits manuales `Key/Value`.
7. **Aterrizaje honesto en evaluación (`rollout_eval.py`):**
   - Cuando $\hat{s}_k < 0.5$ (`flight_state`), fijar comando horizontal en la posición actual `(curr_x, curr_y)` mientras se desciende verticalmente, sin leer nunca `landing` de `waypoints.yaml` para corregir $(x, y)$.
8. **Generalización limpia de hiperparámetros y búsqueda de grilla:**
   - Parametrizar todos los hiperparámetros de arquitectura y entrenamiento sin valores cableados, usando como configuración base por defecto: `M=32`, `L=10` (también compatible con `'adaptive'`), `T=8`, `action_stride=1`, `T_exec=4`, `N_demos=2` y `K=8` pasos DDIM, de modo que sea directo variar dimensiones o ejecutar búsqueda de grilla en Etapa 7.
9. **Refactor pendiente del factor de escala $s=0.1$ hacia el codificador de Fourier:**
   - Antes de iniciar la Etapa 2, el usuario moverá el factor de escala `0.1` (utilizado para evitar aliasing geométrico) desde las representaciones geométricas generales hacia el codificador posicional de Fourier (`SinusoidalPosEmb3D` en Etapa 3), manteniendo las coordenadas geométricas y keypoints en escala física consistente. Al retomar después de ese refactor, se actualizará la Etapa 1 (`geometry.py`) y las especificaciones correspondientes de las Etapas 2 a 7.
10. **Pruebas unitarias separadas en `src/uav_ip/test/icil/`:**
    - Las verificaciones de cada etapa se escriben en módulos dedicados `src/uav_ip/test/icil/test_*.py` ejecutables con `pytest` en `~/venv_ip`, evitando dejar bloques `if __name__ == "__main__":` dentro de los módulos productivos de `uav_ip/icil/` y protegiendo la importación de `torch` con `pytest.importorskip("torch")`.

---

## 5. Detalle de Etapas

### Etapa 1 — Geometría $SE(3)$ 4-DoF, Keypoints, Flow y SVD (`geometry.py`)

**Objetivo:** Implementar en PyTorch puro (sin NumPy, sin `torch_geometric`, sin dependencias de dataset) toda la aritmética rígida 4-DoF (actualmente con `DEFAULT_SCALE_FACTOR = 0.1`, pendiente de refactorizar hacia el codificador de Fourier antes de Etapa 2), las transformaciones entre pose acumulada e incremental, el flujo desacoplado $[\Delta\mathbf{p}_t, \Delta\mathbf{p}_r]$ y el alineamiento cerrado SVD/Kabsch 4-DoF.

**Estado actual en código:** Implementado y verificado por el usuario en `src/uav_ip/uav_ip/icil/geometry.py`. Falta únicamente trasladar su verificación a `src/uav_ip/test/icil/test_geometry.py` y aplicar el refactor que mueve el factor `0.1` al codificador de Fourier.

**Contrato de funciones implementado (`src/uav_ip/uav_ip/icil/geometry.py`):**
- Constantes: `DEFAULT_SCALE_FACTOR = 0.1`, `DEFAULT_ARM_LENGTH_M = 0.25`, `DEFAULT_Z_HEIGHT_M = 0.15`.
- `get_uav_keypoints(scale_factor, arm_length, z_height, device, dtype) -> (6, 3)`: 6 keypoints canónicos en Horizontal FLU `[Centro, +X, -X, +Y, -Y, +Z]` escalados por `scale_factor`.
- `wrap_angle(angle) -> Tensor`: envuelve ángulos en radianes mediante `atan2(sin, cos)`.
- `vec4_to_se3(vec4) -> (..., 4, 4)` y `se3_to_vec4(T) -> (..., 4)`: conversión exacta entre `[dx, dy, dz, yaw]` y matrices homogéneas 4-DoF $R_z(\psi)$.
- `invert_se3(T) -> (..., 4, 4)`: inversa analítica rígida $\begin{bmatrix}R^T & -R^T t \\ 0 & 1\end{bmatrix}$.
- `cumulative_to_incremental(action_se3)` y `incremental_to_cumulative(delta_se3)` sobre eje `-3` (`H`):
  - $T_{EA}^{(t, 0)} = I_4$, $T_{EA}^{(t, k)} = \delta T_1 \cdots \delta T_k$ y $\delta T_k = (T_{EA}^{k-1})^{-1} T_{EA}^k$.
- `transform_keypoints(T, keypoints=None) -> (..., 6, 3)`: transforma `keypoints` con matrices $T$ de forma `(..., 4, 4)`.
- `compute_keypoint_flow(T_curr, T_target, keypoints=None) -> (..., 6, 6)`: retorna `[dp_t, dp_r]` concatenados en el último eje:
  - $\Delta\mathbf{p}_t = t_{\text{target}} - t_{\text{curr}}$ (expandido a los 6 puntos).
  - $\Delta\mathbf{p}_r = (R_{\text{target}} - R_{\text{curr}}) p_g^i$ (rotación alrededor del CoM).
- `align_keypoints_svd_4dof(p_src, p_dst) -> (..., 4, 4)`: proyección cerrada Kabsch/Procrustes en `float64` restringida a 4-DoF ($R_z(\psi), \mathbf{t}$).
- `apply_flow_and_svd(T_curr, flow, keypoints=None, step_scale=1.0) -> (..., 4, 4)`: suma $\Delta\mathbf{p}_t + \Delta\mathbf{p}_r$ escalado por `step_scale` sobre `transform_keypoints(T_curr, keypoints)` y recupera la matriz destino mediante `align_keypoints_svd_4dof(keypoints, p_dst)`.

**Test unitario separado (`src/uav_ip/test/icil/test_geometry.py`):**

```python
"""Pruebas unitarias de Etapa 1: geometria SE(3) 4-DoF, flujo desacoplado y SVD."""

from pathlib import Path
import sys

import pytest

PKG_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = PKG_ROOT.parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

torch = pytest.importorskip("torch")

from uav_ip.icil.geometry import (  # noqa: E402
    DEFAULT_ARM_LENGTH_M,
    DEFAULT_SCALE_FACTOR,
    DEFAULT_Z_HEIGHT_M,
    align_keypoints_svd_4dof,
    apply_flow_and_svd,
    compute_keypoint_flow,
    cumulative_to_incremental,
    get_uav_keypoints,
    incremental_to_cumulative,
    invert_se3,
    se3_to_vec4,
    transform_keypoints,
    vec4_to_se3,
    wrap_angle,
)


def test_canonical_keypoints_shape_and_values() -> None:
    kp = get_uav_keypoints()
    assert kp.shape == (6, 3)
    arm = DEFAULT_ARM_LENGTH_M * DEFAULT_SCALE_FACTOR
    z_up = DEFAULT_Z_HEIGHT_M * DEFAULT_SCALE_FACTOR
    assert torch.allclose(kp[0], torch.zeros(3))
    assert torch.isclose(kp[1, 0], torch.tensor(arm))
    assert torch.isclose(kp[2, 0], torch.tensor(-arm))
    assert torch.isclose(kp[3, 1], torch.tensor(arm))
    assert torch.isclose(kp[4, 1], torch.tensor(-arm))
    assert torch.isclose(kp[5, 2], torch.tensor(z_up))


def test_vec4_se3_and_inverse_roundtrip() -> None:
    torch.manual_seed(42)
    vec = torch.empty((16, 8, 4)).uniform_(-0.5, 0.5)
    vec[..., 3] = torch.empty((16, 8)).uniform_(-torch.pi, torch.pi)
    T = vec4_to_se3(vec)
    vec_rec = se3_to_vec4(T)

    assert (vec[..., :3] - vec_rec[..., :3]).abs().max().item() < 1e-6
    assert wrap_angle(vec[..., 3] - vec_rec[..., 3]).abs().max().item() < 1e-6
    assert (torch.matmul(T, invert_se3(T)) - torch.eye(4)).abs().max().item() < 1e-6


def test_cumulative_incremental_roundtrip() -> None:
    pt_file = REPO_ROOT / "datasets/kinematic/task_0000/episode_00/heterodata.pt"
    if pt_file.is_file():
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        action_se3 = data["agent"].action_se3.to(torch.float32)
    else:
        torch.manual_seed(7)
        action_se3 = vec4_to_se3(torch.randn(50, 8, 4) * 0.05)

    deltas = cumulative_to_incremental(action_se3)
    reconstructed = incremental_to_cumulative(deltas)
    err_cum_m = (reconstructed[..., :3, 3] - action_se3[..., :3, 3]).abs().max().item() / DEFAULT_SCALE_FACTOR
    assert err_cum_m < 1e-5


def test_decoupled_flow_and_svd_4dof() -> None:
    torch.manual_seed(123)
    kp = get_uav_keypoints()
    T_curr = vec4_to_se3(torch.randn(32, 8, 4) * torch.tensor([0.3, 0.3, 0.15, 2.5]))
    T_target = vec4_to_se3(torch.randn(32, 8, 4) * torch.tensor([0.3, 0.3, 0.15, 2.5]))

    flow = compute_keypoint_flow(T_curr, T_target, kp)
    assert flow.shape == (32, 8, 6, 6)
    T_svd = apply_flow_and_svd(T_curr, flow, kp, step_scale=1.0)

    v_orig = se3_to_vec4(T_target)
    v_svd = se3_to_vec4(T_svd)
    err_pos_m = (v_svd[..., :3] - v_orig[..., :3]).abs().max().item() / DEFAULT_SCALE_FACTOR
    err_yaw_rad = wrap_angle(v_svd[..., 3] - v_orig[..., 3]).abs().max().item()
    assert err_pos_m < 1e-5
    assert err_yaw_rad < 1e-5


def test_svd_4dof_se3_equivariance() -> None:
    torch.manual_seed(99)
    kp = get_uav_keypoints()
    T_0 = vec4_to_se3(torch.randn(16, 4) * torch.tensor([0.2, 0.2, 0.1, 2.0]))
    T_1 = vec4_to_se3(torch.randn(16, 4) * torch.tensor([0.2, 0.2, 0.1, 2.0]))
    p0 = transform_keypoints(T_0, kp)
    p1 = transform_keypoints(T_1, kp)

    T_corr = align_keypoints_svd_4dof(p0, p1)
    G = vec4_to_se3(torch.tensor([1.2, -0.7, 0.3, 2.1]))
    T_corr_G = align_keypoints_svd_4dof(transform_keypoints(G, p0), transform_keypoints(G, p1))
    expected = torch.matmul(G, torch.matmul(T_corr, invert_se3(G)))
    assert (T_corr_G - expected).abs().max().item() < 1e-5
```

**Verificación rápida Etapa 1:**
```bash
~/venv_ip/bin/pytest src/uav_ip/test/icil/test_geometry.py -v
```
**Criterio de éxito:** Todos los tests en verde (`5 passed`) con errores geométricos `< 1e-5`.

---

### Etapa 2 — Dataset, Splits JSON, Keyframes y Horizonte (`dataset.py`)

**Objetivo:** Cargar episodios `heterodata.pt` de `datasets/kinematic/` y `datasets/sitl/` respetando estrictamente los splits JSON, seleccionar `L` keyframes (fijo o `'adaptive'`), recomputar horizontes con `(T, action_stride)` en formato incremental $\delta T_k$, y producir batches listos para el entrenamiento e inferencia.

**Esquema de splits verificado:**
- Ubicación: `datasets/<kinematic|sitl>/splits/{train,val,held_out}_tasks.json` (mapear `"test"` $\to$ `"held_out"`).
- Estructura JSON observada: `{"tasks": [...], "signatures": [...]}`.
- Datasets actuales en disco: `datasets/kinematic/` (200 tareas, 1000 episodios: 140 train, 30 val, 30 held_out) y `datasets/sitl/` (12 tareas, 60 episodios).

**Funcionalidad a implementar:**
1. Lectura estricta de `splits/{split}_tasks.json` (error explícito si falta el archivo; cero fallback silencioso).
2. Al empezar Etapa 2: inspeccionar la estructura exacta de campos dentro de `datasets/kinematic/task_0000/episode_00/heterodata.pt` (generada en `bag_to_heterodata.py` líneas 1581–1672) para confirmar cada clave del `HeteroData` sin suposiciones.
3. Selector de keyframes parametrizado:
   - `num_keyframes: int | str = 10` (si es entero, re-muestrea/selecciona $L$ frames clave; si es `'adaptive'`, usa extracción adaptativa y devuelve `demo_mask` para padding hasta $L_{\max}$).
4. Recomputación del horizonte futuro a partir de la trayectoria de poses y estados de vuelo:
   - Parametrizable por `horizon: int = 8` (`T`) y `action_stride: int = 1`.
   - Convierte acciones a pasos incrementales $\delta T_k \in SE(3)$ 4-DoF usando `geometry.py`.
   - Soporte uniforme para muestras nominales y muestras de `Tube Augmentation` ($\gamma = 0.70$).
5. `InContextDataset` + `collate_icil_batch`:
   - Muestrea $N_{\text{demos}}$ demostraciones aceptadas del mismo task (excluyendo la trayectoria query cuando sea posible o evitando fuga de índice).
   - Contrato de salida por batch (`dict[str, Tensor]`):
     - `live_cloud`: `(B, 2048, 3)`
     - `live_vel_flu`: `(B, 3)`
     - `live_flight_state`: `(B, 1)`
     - `demo_clouds`: `(B, N_demos, L, 2048, 3)`
     - `demo_poses_se3`: `(B, N_demos, L, 4, 4)` (en marco del episodio demo para obtener transformaciones relativas entre keyframes de esa demo; nunca mezcladas en marco global con `live`)
     - `demo_vel_flu`: `(B, N_demos, L, 3)`
     - `demo_flight_state`: `(B, N_demos, L, 1)`
     - `demo_mask`: `(B, N_demos, L)` booleana (`True` en keyframes válidos)
     - `gt_delta_se3`: `(B, T, 4, 4)` (acciones incrementales $\delta T_k$)
     - `gt_flight_state`: `(B, T, 1)`
     - `action_valid_mask`: `(B, T)` booleana

**Verificación rápida Etapa 2:**
- Verificar 0 intersección entre `train`, `val` y `held_out` task IDs.
- Medir y registrar en este documento los percentiles (99%, max) de $\|\Delta\mathbf{p}\|_t$ y $|\Delta\psi|$ de $\delta T_k$ en `kinematic` y `sitl` para calibrar las escalas de ruido DDPM en Etapa 5.
- Confirmar 0 `NaN`/`Inf` y formas exactas tras `DataLoader(..., collate_fn=collate_icil_batch)`.

---

### Etapa 3 — Encoder de Escena $\phi_e$ y Pre-entrenamiento por Ocupancia 3D

**Objetivo (Tarea 4.2 de `PLAN_IMPLEMENTACION.md`):** Construir la primera mitad de `src/uav_ip/uav_ip/icil/scene_encoder.py` junto con `src/uav_ip/scripts/icil/pretrain_scene_encoder.py`, pre-entrenar el extractor local 3D $\phi_e$ mediante predicción de ocupancia implícita sobre geometrías sintéticas del dominio UAV, y guardar un checkpoint listo para cargarse congelado.

**Componentes:**
1. `SinusoidalPosEmb3D(dim)` (codificación de Fourier 3D para desplazamientos relativos $\Delta\mathbf{x} \in \mathbb{R}^3$).
2. Utilidades FPS (*Farthest Point Sampling*) + $k$-NN local (PyTorch puro o `torch_cluster` si disponible en `~/venv_ip`, verificando disponibilidad al iniciar la etapa) sobre la **nube completa** de 2048 puntos (sin recorte en $Z$).
3. `SceneEncoderPhi` ($\phi_e$):
   - Arquitectura tipo PointNet++ Set Abstraction (2 niveles jerárquicos de agrupación local FPS + MLP + max-pooling):
     - Entrada: nube `(B, 2048, 3)`.
     - Salida: `centroids` `(B, M, 3)` y `features` `(B, M, d_model)` (con $M \in \{16, 32, 64\}$ configurable).
4. `OccupancyDecoderPsi` ($\psi_e$, solo para pre-entrenamiento):
   - Recibe `centroids`, `features` y puntos query 3D `q_pts` `(B, Q, 3)`.
   - Interpola / agrega features de los $k$ centroides más cercanos ponderados por `SinusoidalPosEmb3D(q_pts - c_j)` y pasa por un MLP residual para predecir probabilidad de ocupancia `logit` `(B, Q, 1)`.
5. Generador de primitivas 3D sintéticas (`pole`, `square_gate`, `circular_gate`, `landing_pad`, suelo $z=0$ y combinaciones de 2–4 objetos con rotación/traslación aleatoria a escala $s=0.1$) + muestreo balanceado de puntos query dentro/cerca/lejos de la superficie (`BCEWithLogitsLoss`).

**Verificación rápida Etapa 3:**
- Entrenar $\phi_e + \psi_e$ en `pretrain_scene_encoder.py` hasta superar **90% de accuracy** en validación de ocupancia.
- Guardar `checkpoints/scene_encoder.pt` incluyendo hiperparámetros (`M`, `d_model`).

---

### Etapa 4 — Grafo Local $\sigma$: Fusión Escena Congelada + Estado UAV

**Objetivo:** Completar la segunda mitad de `src/uav_ip/uav_ip/icil/scene_encoder.py` implementando `LocalStateEncoderSigma` ($\sigma$), que conecta los $M$ centroides geométricos congelados de $\phi_e$ con los 6 keypoints canónicos del UAV.

**Contrato de `LocalStateEncoderSigma`:**
- Entradas:
  - `scene_centroids`: `(B, M, 3)` y `scene_feats`: `(B, M, d_model)` (de $\phi_e$ congelado en `eval()` y `torch.no_grad()`).
  - `vel_flu`: `(B, 3)`
  - `flight_state`: `(B, 1)`
- Arquitectura:
  1. Embebe el estado propio del UAV (`vel_flu`, `flight_state`) junto al identificador aprendible de cada uno de los 6 keypoints canónicos `node_emb` (`nn.Embedding(6, d_model)`).
  2. Construye un grafo bipartito $M \to 6$ conectado mediante `torch_geometric.nn.TransformerConv` donde `edge_attr = SinusoidalPosEmb3D(p_kp_j - p_centroid_i)`.
  3. Residual + LayerNorm $\to$ salida `(B, 6, d_model)`.
- Provee también un helper para evaluar batches de demos `(B, N_demos, L, 2048, 3)` de manera eficiente.

**Verificación rápida Etapa 4:**
- Ejecutar forward + backward con pérdida dummy sobre un batch real de Etapa 2.
- Verificar explícitamente que todos los parámetros de $\phi_e$ mantienen `requires_grad == False` y `.grad is None`, mientras todos los parámetros de $\sigma$ reciben gradientes finitos no nulos.

---

### Etapa 5 — Red de Política ($\varphi \to \psi$), Difusión DDPM/DDIM y Test Dorado (`policy_net.py`)

**Objetivo:** Implementar el núcleo In-Context Graph Diffusion siguiendo el paper *Instant Policy* (§3.2–3.4): grafo de contexto $\varphi$, grafo de denoising de acción $\psi$, entrenamiento DDPM sobre acciones incrementales desacopladas y muestreo rápido DDIM en $K$ pasos.

**Componentes:**
1. **`ContextGraphPhi` ($\varphi$):**
   - Toma las $N_{\text{demos}} \times L$ observaciones pasadas por $\sigma$ (`(B, N_demos, L, 6, d_model)`), sus poses dentro de cada demostración y `demo_mask`.
   - En cada demostración, construye aristas relativas entre keypoints dentro del mismo frame y entre keyframes consecutivos/cercanos usando posiciones transformadas por `T_rel = inv(T_l) @ T_{l'}` y `SinusoidalPosEmb3D`, propagando información con `TransformerConv` hacia un cuello de botella de demostración y agregando hacia los 6 nodos del estado actual `live`.
   - Método `encode_context(...)` desacoplado para cachear las representaciones de las demos durante el rollout cerrado (Etapas 6 y 7).
2. **`ActionDenoisingGraphPsi` ($\psi$):**
   - Nodos: los 6 nodos `live` (con features $\sigma + \varphi$) + los $T \times 6$ nodos de acciones futuras ruidosas (cada uno en la pose acumulada ruidosa correspondiente $T_{\text{noisy}, k}$ proyectada al marco `live` actual).
   - Condicionamiento en paso de difusión $t_{\text{diff}}$ (embedding sinusoidal) y en estado de vuelo ruidoso.
   - Aristas espaciales y temporales (`TransformerConv` con `SinusoidalPosEmb3D(p_dst - p_src)`).
   - Cabeza MLP con GELU prediciendo:
     - Flujo traslacional $\Delta\hat{\mathbf{p}}_t$ `(B, T, 6, 3)`
     - Flujo rotacional $\Delta\hat{\mathbf{p}}_r$ `(B, T, 6, 3)`
     - Logit de estado de vuelo $\hat{s}$ `(B, T, 1)` (promedio sobre los 6 nodos de cada paso $k$).
3. **Difusión DDPM (entrenamiento) y DDIM (inferencia):**
   - Perturbación controlada de acciones incrementales $\delta T_k$ en espacio 4-DoF normalizado (usando cotas medidas en Etapa 2) $\to$ cálculo del flujo objetivo exacto mediante `geometry.compute_keypoint_flow`.
   - Pérdida L1/MSE en flujo $[\Delta\mathbf{p}_t, \Delta\mathbf{p}_r]$ + BCE en `flight_state` respetando `action_valid_mask`.
   - Muestreo DDIM en $K \in \{4, 8, 10\}$ pasos aplicando `geometry.apply_flow_and_svd` al final de cada paso sin clipping artificial.

**Verificación rápida Etapa 5 ("Test Dorado"):**
1. **Overfit 1 episodio (~150–300 pasos):** Entrenar `policy_net` sobre las muestras de un solo episodio (`batch_size` pequeño en `~/venv_ip`) verificando caída limpia de loss y error de trayectoria predicha vs GT cercano a 0.
2. **Test Context-Swap (signo lateral):** Pasar exactamente el mismo `live_cloud` + `live_state`, cambiando únicamente las demos de contexto entre una tarea de "rodear por izquierda" y su tarea espejada "rodear por derecha". Verificar que $\Delta y$ / $\Delta \psi$ predichos invierten su signo coherentemente.

---

### Etapa 6 — Evaluador en Lazo Cerrado Cinemático (`rollout_eval.py`)

**Objetivo:** Validar la política prediciendo en lazo cerrado a $10\text{ Hz}$ sin acceso privilegiado a la referencia ni a `waypoints.yaml`.

**Flujo del simulador cinemático:**
1. Inicializar en pose inicial del episodio; pre-calcular contexto `encode_context(demos)` una única vez al comienzo del episodio.
2. En cada ciclo de inferencia (cada `T_exec` pasos de $0.1\text{ s}$):
   - Generar nube sintética actual con `SyntheticLidarGenerator(r_max=10.0, scale_factor=0.1, target_points=2048)` en la pose actual.
   - Ejecutar `predict_actions(..., num_ddim_steps=K)` para obtener $T$ pasos incrementales $\delta T_1 \dots \delta T_T$ y estados $\hat{s}_1 \dots \hat{s}_T$.
   - Avanzar ejecutando los primeros `T_exec` pasos (`T_exec <= T`), acumulando pose en mundo ENU (`scale_factor=0.1`) y estimando `vel_flu`.
   - **Aterrizaje honesto:** Si $\hat{s}_k < 0.5$, mantener $(x, y)$ en la posición actual `(curr_x, curr_y)` mientras desciende hacia $z \le z_{\text{landing}}$; detener al detectar aterrizaje o alcanzar tiempo máximo.
3. Interpolar la trayectoria cerrada resultante a $50\text{ Hz}$ mediante PCHIP y evaluar con `uav_ip.validate_episode.validate_episode_data(...)`.

**Verificación rápida Etapa 6:**
- Modo Oráculo (reemplazando la red por la trayectoria GT): `ADE = 0.0` y `validation_status == 'accepted'`.
- Rollout cerrado del checkpoint sobre-ajustado en el Test Dorado de Etapa 5: episodio `accepted` en su tarea correspondiente.

---

### Etapa 7 — Entrenamiento Real, Ablaciones (Tarea 4.5) y Piloto ROS 2

#### Subetapa 7A — `scripts/icil/train_policy.py` (y `eval_ablations.py`)
1. **Fase 1 (Pre-entrenamiento Kinemático):** Entrenar sobre `datasets/kinematic/splits/train_tasks.json` (con `Tube Augmentation`), evaluando periódicamente con `rollout_eval.py` sobre `val_tasks.json` (objetivo: `Success >= 70%`).
2. **Fase 2 (Co-Fine-Tuning Sim2Real 50/50):** Combinar batches 50% `kinematic` / 50% `sitl` (`splits/train_tasks.json`), midiendo generalización en `datasets/sitl/splits/held_out_tasks.json` y `datasets/kinematic/splits/held_out_tasks.json`.
3. **Grilla de ablaciones (Tarea 4.5 de `PLAN_IMPLEMENTACION.md`):**
   - Densidad de tokens $\phi_e$: `M` $\in \{16, 32, 64\}$
   - Keyframes de contexto: `L=10` fijo vs `'adaptive'`
   - Horizonte / chunk ejecutado: `(T=8, T_exec=4)` vs variantes
   - Pasos DDIM y latencia CPU: `K` $\in \{4, 8, 10\}$
   - Número de demostraciones en contexto: `N_demos` $\in \{1, 2, 4\}$

#### Subetapa 7B — `uav_ip/icil/instant_policy_pilot.py`
- Nodo ROS 2 que hereda de `uav_ip.base_pilot.BasePilot`.
- Carga `scene_encoder.pt` congelado + checkpoint de política entrenada.
- Mantiene buffer de odometría/nube sintética (o LiDAR en vivo cuando corresponda), ejecuta `encode_context()` al inicializar la misión y corre `predict_actions()` en lazo cerrado contra PX4 SITL + Gazebo (`gz_x500`) enviando setpoints en tiempo real.

---

## 6. Registro de Avances y Hallazgos por Etapa

*(Se irá completando al finalizar y verificar cada etapa, antes de ejecutar `/compact`.)*

- **Etapa 1:** `src/uav_ip/uav_ip/icil/geometry.py` implementado y verificado por el usuario.
  - **Decisiones confirmadas:**
    1. Tests unitarios separados por etapa en `src/uav_ip/test/icil/test_*.py` (`pytest.importorskip("torch")`).
    2. Hiperparámetros por defecto acordados y parametrizables para futura búsqueda de grilla: `M=32`, `L=10`, `T=8`, `action_stride=1`, `T_exec=4`, `N_demos=2`, `K=8`.
  - **Pendiente antes de iniciar Etapa 2:** Refactor del usuario moviendo el factor de escala `0.1` desde las representaciones geométricas generales hacia el codificador de Fourier (`SinusoidalPosEmb3D`), tras lo cual se actualizarán `geometry.py` y las Etapas 2–7 en este documento, además de mover el bloque `__main__` de `geometry.py` a `src/uav_ip/test/icil/test_geometry.py`.
