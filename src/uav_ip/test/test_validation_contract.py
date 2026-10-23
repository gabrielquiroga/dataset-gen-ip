import copy
import json
import sys
from pathlib import Path

import yaml

PKG_ROOT = Path(__file__).resolve().parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from uav_ip.validate_episode import (
    VALIDATOR_VERSION,
    validate_episode_data,
    validate_episode_directory,
    validate_waypoints_config,
)
from test_validation_cli import _write_synthetic_valid_mcap


def _base_valid_waypoints() -> dict:
    return {
        "start_pose": {"x": 0.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
        "landmarks": [
            {"id": "landmark_1", "type": "square_gate", "x": 5.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
            {"id": "landmark_2", "type": "pole", "x": 10.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
        ],
        "primitives": [
            {"type": "traverse", "target": "landmark_1"},
            {"type": "slalom", "target": "landmark_2", "side": "left"},
        ],
        "landing": {"x": 15.0, "y": 0.0, "z": 0.05},
    }


def test_empty_waypoints_yaml_is_rejected(tmp_path):
    """C2: Un YAML vacio o diccionario vacio se rechaza sin convertirse en mision vacia aceptable."""
    for empty_cfg in (None, {}, [], ""):
        res = validate_waypoints_config(empty_cfg)
        assert res["valid"] is False
        assert res["configuration_status"] == "invalid"
        assert any("invalid_configuration:waypoints_root_must_be_non_empty_dict" in e for e in res["errors"])

    # Probar tambien en un directorio de episodio real con archivo waypoints.yaml en blanco
    ep_dir = tmp_path / "task_0000" / "episode_00"
    ep_dir.mkdir(parents=True)
    (ep_dir / "waypoints.yaml").write_text("", encoding="utf-8")
    (ep_dir / "episode_meta.json").write_text(
        json.dumps({"status": "success", "pilot_exit_code": 0}), encoding="utf-8"
    )
    _write_synthetic_valid_mcap(ep_dir / "rosbag" / "rosbag_0.mcap")

    rep = validate_episode_directory(ep_dir)
    assert rep["configuration_status"] == "invalid"
    assert rep["mission_status"] == "invalid"
    assert rep["validation_status"] == "rejected"
    assert rep["metrics"] == {}
    assert any("invalid_configuration:" in r for r in rep["rejection_reasons"])


def test_missing_landing_is_rejected():
    """C2: La ausencia de `landing` o coordenadas no finitas en `landing` produce rechazo."""
    cfg_no_landing = _base_valid_waypoints()
    del cfg_no_landing["landing"]

    res = validate_waypoints_config(cfg_no_landing)
    assert res["valid"] is False
    assert any("invalid_configuration:missing_or_invalid_landing" in e for e in res["errors"])

    data_rep = validate_episode_data({"odometry": []}, cfg_no_landing, pilot_exit_code=0)
    assert data_rep["configuration_status"] == "invalid"
    assert data_rep["validation_status"] == "rejected"
    assert data_rep["metrics"] == {}


def test_duplicate_landmark_ids_are_rejected():
    """C2: IDs de landmarks duplicados producen rechazo de configuracion."""
    cfg_dup = _base_valid_waypoints()
    cfg_dup["landmarks"][1]["id"] = "landmark_1"

    res = validate_waypoints_config(cfg_dup)
    assert res["valid"] is False
    assert any("invalid_configuration:duplicate_landmark_id" in e for e in res["errors"])


def test_non_finite_coordinates_or_orientation_are_rejected():
    """C2: Coordenadas o yaw NaN/infinitos en landmarks o landing producen rechazo."""
    for field, bad_val in (("x", float("nan")), ("y", float("inf")), ("z", -float("inf")), ("yaw", float("nan"))):
        cfg_bad = copy.deepcopy(_base_valid_waypoints())
        cfg_bad["landmarks"][0][field] = bad_val
        res = validate_waypoints_config(cfg_bad)
        assert res["valid"] is False
        assert any("invalid_configuration:non_finite_landmark_pose" in e for e in res["errors"])

    cfg_bad_land = copy.deepcopy(_base_valid_waypoints())
    cfg_bad_land["landing"]["x"] = float("nan")
    res_land = validate_waypoints_config(cfg_bad_land)
    assert res_land["valid"] is False
    assert any("invalid_configuration:non_finite_landing_coordinate" in e for e in res_land["errors"])


def test_unknown_primitive_invalid_slalom_side_and_incompatible_landmark_are_rejected():
    """C2: Primitiva desconocida, side invalido/ausente en slalom o incompatibilidad primitiva-objeto se rechazan."""
    # 1. Primitiva desconocida
    cfg_unk_prim = copy.deepcopy(_base_valid_waypoints())
    cfg_unk_prim["primitives"][0]["type"] = "barrel_roll"
    res_unk = validate_waypoints_config(cfg_unk_prim)
    assert res_unk["valid"] is False
    assert any("invalid_configuration:unsupported_primitive_type" in e for e in res_unk["errors"])

    # 2. Slalom sin side explicito
    cfg_no_side = copy.deepcopy(_base_valid_waypoints())
    del cfg_no_side["primitives"][1]["side"]
    res_no_side = validate_waypoints_config(cfg_no_side)
    assert res_no_side["valid"] is False
    assert any("invalid_configuration:invalid_or_missing_slalom_side" in e for e in res_no_side["errors"])

    # 3. Slalom con side desconocido
    cfg_bad_side = copy.deepcopy(_base_valid_waypoints())
    cfg_bad_side["primitives"][1]["side"] = "center"
    res_bad_side = validate_waypoints_config(cfg_bad_side)
    assert res_bad_side["valid"] is False
    assert any("invalid_configuration:invalid_or_missing_slalom_side" in e for e in res_bad_side["errors"])

    # 4. Incompatibilidad entre primitiva y tipo de landmark (orbit sobre un gate, traverse sobre un pole)
    cfg_incompat_1 = copy.deepcopy(_base_valid_waypoints())
    cfg_incompat_1["primitives"][0] = {"type": "orbit", "target": "landmark_1"}
    res_inc_1 = validate_waypoints_config(cfg_incompat_1)
    assert res_inc_1["valid"] is False
    assert any("invalid_configuration:incompatible_primitive_landmark" in e for e in res_inc_1["errors"])

    cfg_incompat_2 = copy.deepcopy(_base_valid_waypoints())
    cfg_incompat_2["primitives"][1] = {"type": "traverse", "target": "landmark_2"}
    res_inc_2 = validate_waypoints_config(cfg_incompat_2)
    assert res_inc_2["valid"] is False
    assert any("invalid_configuration:incompatible_primitive_landmark" in e for e in res_inc_2["errors"])


def test_nonexistent_landmark_reference_is_rejected():
    """C2: Una primitiva que referencia un landmark inexistente produce rechazo de configuracion."""
    cfg_missing_ref = copy.deepcopy(_base_valid_waypoints())
    cfg_missing_ref["primitives"][1]["target"] = "landmark_99"

    res = validate_waypoints_config(cfg_missing_ref)
    assert res["valid"] is False
    assert any("invalid_configuration:unknown_primitive_target (landmark_99)" in e for e in res["errors"])


def test_missing_metadata_does_not_fabricate_pilot_exit_code_zero(tmp_path):
    """
    C2: Si falta `episode_meta.json` (o falta informacion del piloto), `pilot_exit_code` permanece `None`
    y el episodio no se acepta para campana.
    """
    ep_dir = tmp_path / "task_0000" / "episode_00"
    ep_dir.mkdir(parents=True)
    waypoints_cfg = {
        "landmarks": [
            {"id": "landmark_1", "type": "square_gate", "x": 5.0, "y": 0.0, "z": 1.5, "yaw": 0.0}
        ],
        "primitives": [{"type": "traverse", "target": "landmark_1"}],
        "landing": {"x": 10.0, "y": 0.0, "z": 0.05},
    }
    (ep_dir / "waypoints.yaml").write_text(yaml.safe_dump(waypoints_cfg), encoding="utf-8")
    _write_synthetic_valid_mcap(ep_dir / "rosbag" / "rosbag_0.mcap")

    # Sin episode_meta.json en el directorio
    rep = validate_episode_directory(ep_dir, update_meta=True)
    assert rep["pilot_exit_code"] is None
    assert rep["pilot_status_source"] == "unknown"
    assert rep["validation_status"] == "rejected"
    assert "unknown_pilot_exit_code" in rep["rejection_reasons"]

    # Verificar que --update-meta no invento pilot_exit_code=0 en el JSON escrito
    written_meta = json.loads((ep_dir / "episode_meta.json").read_text(encoding="utf-8"))
    assert written_meta["pilot_exit_code"] is None
    assert written_meta["pilot_status_source"] == "unknown"
    assert written_meta["validation_status"] == "rejected"


def test_revalidation_preserves_original_provenance_and_refuses_corrupt_json_overwrite(tmp_path):
    """
    C2:
    - Al revalidar con `--update-meta` un episodio con estado historico `pilot_exit_2` o `timeout`,
      se conserva la procedencia original (`original_status`, `original_pilot_exit_code`, `pilot_status_source`).
    - Se registran `validator_version` y `validation_parameters`.
    - Un `episode_meta.json` corrupto produce rechazo y no es sobrescrito silenciosamente.
    """
    ep_dir = tmp_path / "task_0000" / "episode_00"
    ep_dir.mkdir(parents=True)
    waypoints_cfg = {
        "landmarks": [
            {"id": "landmark_1", "type": "square_gate", "x": 5.0, "y": 0.0, "z": 1.5, "yaw": 0.0}
        ],
        "primitives": [{"type": "traverse", "target": "landmark_1"}],
        "landing": {"x": 10.0, "y": 0.0, "z": 0.05},
    }
    (ep_dir / "waypoints.yaml").write_text(yaml.safe_dump(waypoints_cfg), encoding="utf-8")
    _write_synthetic_valid_mcap(ep_dir / "rosbag" / "rosbag_0.mcap")

    meta_file = ep_dir / "episode_meta.json"
    meta_file.write_text(
        json.dumps({"task_id": "task_0000", "episode_id": "episode_00", "status": "pilot_exit_2"}, indent=2),
        encoding="utf-8",
    )

    # Primera validacion con update_meta=True
    rep1 = validate_episode_directory(ep_dir, update_meta=True)
    assert rep1["pilot_exit_code"] == 2
    assert rep1["pilot_status_source"] == "inferred_from_legacy_status"
    assert rep1["validation_status"] == "rejected"
    assert any("pilot_non_zero_exit_code (2)" in r for r in rep1["rejection_reasons"])

    # Segunda revalidacion con update_meta=True: no debe perder el codigo 2 ni el status original
    rep2 = validate_episode_directory(ep_dir, update_meta=True)
    assert rep2["pilot_exit_code"] == 2
    assert rep2["pilot_status_source"] == "inferred_from_legacy_status"
    assert rep2["validation_status"] == "rejected"

    saved_meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert saved_meta["original_status"] == "pilot_exit_2"
    assert saved_meta["pilot_exit_code"] == 2
    assert saved_meta["validator_version"] == VALIDATOR_VERSION
    assert isinstance(saved_meta["validation_parameters"], dict)
    assert saved_meta["validation_parameters"]["validator_version"] == VALIDATOR_VERSION

    # Ahora probar con un episode_meta.json corrupto (JSON truncado)
    corrupt_dir = tmp_path / "task_0000" / "episode_01"
    corrupt_dir.mkdir(parents=True)
    (corrupt_dir / "waypoints.yaml").write_text(yaml.safe_dump(waypoints_cfg), encoding="utf-8")
    _write_synthetic_valid_mcap(corrupt_dir / "rosbag" / "rosbag_0.mcap")
    corrupt_meta = corrupt_dir / "episode_meta.json"
    corrupt_raw = '{"task_id": "task_0000", "status": '
    corrupt_meta.write_text(corrupt_raw, encoding="utf-8")

    rep_corrupt = validate_episode_directory(corrupt_dir, update_meta=True)
    assert rep_corrupt["validation_status"] == "rejected"
    assert any("corrupted_episode_meta_json" in r for r in rep_corrupt["rejection_reasons"])
    # No debe haberse sobrescrito el archivo corrupto
    assert corrupt_meta.read_text(encoding="utf-8") == corrupt_raw
