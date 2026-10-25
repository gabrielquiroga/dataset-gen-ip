from typing import Any, Dict, List
import math
import numpy as np
import pytest

from uav_ip.validate_episode import (
    reconstruct_simulation_clock,
    validate_episode_data,
)


def _make_clean_flight_fixture(num_samples: int = 300, dt_s: float = 0.02) -> Dict[str, Any]:
    """Construye un vuelo sintetico valido con secuencia persistente de VehicleLandDetected."""
    base_unix_s = 1_700_000_000.0
    ts_s = base_unix_s + np.arange(num_samples, dtype=np.float64) * dt_s

    odom_list: List[Dict[str, Any]] = []
    # 0..19: ascenso, 20..199: crucero atravesando gate_1 en x=5.0, 200..299: aterrizaje y reposo en x=8.0
    for i, t_val in enumerate(ts_s):
        frac = i / float(num_samples - 1)
        if i < 20:
            x = 0.0
            z = 0.1 + (1.4 * (i / 20.0))
            vx, vz = 0.0, 1.0
        elif i < 200:
            alpha = (i - 20) / 180.0
            x = 8.0 * alpha
            z = 1.5
            vx, vz = 8.0 / (180.0 * dt_s), 0.0
        elif i < 240:
            alpha = (i - 200) / 40.0
            x = 8.0
            z = 1.5 * (1.0 - alpha) + 0.05 * alpha
            vx, vz = 0.0, -0.5
        else:
            x = 8.0
            z = 0.05
            vx, vz = 0.0, 0.0

        # ENU (x, 0, z) -> NED (0, x, -z), yaw ENU 0 (Este) -> yaw NED pi/2 => q = [cos(pi/4), 0, 0, sin(pi/4)]
        s = math.sqrt(0.5)
        t_us = int(round(t_val * 1e6))
        t_ns = int(round(t_val * 1e9))
        odom_list.append(
            {
                "timestamp": t_ns,
                "timestamp_us": t_us,
                "timestamp_sample_us": t_us,
                "position": [0.0, float(x), float(-z)],
                "q": [s, 0.0, 0.0, s],
                "velocity": [0.0, float(vx), float(-vz)],
                "pose_frame": 1,
                "velocity_frame": 1,
                "reset_counter": 0,
            }
        )

    # Emitir secuencia temporal de VehicleLandDetected a 5 Hz (False en vuelo, True desde i=240 hasta el final)
    land_list: List[Dict[str, Any]] = []
    step_ld = max(1, int(round(0.20 / dt_s)))
    for i in range(0, num_samples, step_ld):
        is_landed = i >= 240
        t_ns = int(round(ts_s[i] * 1e9))
        t_us = int(round(ts_s[i] * 1e6))
        land_list.append(
            {
                "timestamp": t_ns,
                "timestamp_us": t_us,
                "landed": is_landed,
                "ground_contact": is_landed,
                "maybe_landed": is_landed,
                "freefall": False,
                "in_ground_effect": is_landed,
                "at_rest": is_landed,
            }
        )
    if land_list[-1]["timestamp"] != odom_list[-1]["timestamp"]:
        land_list.append(
            {
                "timestamp": odom_list[-1]["timestamp"],
                "timestamp_us": odom_list[-1]["timestamp_us"],
                "landed": True,
                "ground_contact": True,
                "maybe_landed": True,
                "freefall": False,
                "in_ground_effect": True,
                "at_rest": True,
            }
        )

    waypoints_cfg = {
        "landmarks": [
            {"id": "gate_1", "type": "square_gate", "x": 5.0, "y": 0.0, "z": 1.5, "yaw": 0.0}
        ],
        "primitives": [{"type": "traverse", "target": "gate_1"}],
        "landing": {"x": 8.0, "y": 0.0, "z": 0.05},
    }
    return {
        "telemetry": {"odometry": odom_list, "local_pos": [], "land_detected": land_list},
        "waypoints_cfg": waypoints_cfg,
    }


def test_unexplained_source_gap_is_not_silently_erased():
    # Sin evidencia adicional de offset, no convertir 700 ms en 8 ms.
    base = 1_700_000_000.0
    source = base + np.array([0.0, 0.008, 0.708, 0.716])
    received = base + np.array([0.0, 0.008, 0.016, 0.024])

    res = reconstruct_simulation_clock(source, received)

    assert res["reconstructed_time_s"] is not None
    assert np.max(np.diff(res["reconstructed_time_s"])) >= 0.60
    assert res["usable_for_validation"] is False
    assert "telemetry_dropout_gap" in res["reasons"]


def test_unix_delivery_backlog_does_not_erase_source_gap():
    """
    Regresion C3: una entrega en rafaga tras un hueco de 700 ms en la fuente produce la misma
    firma que el antiguo Patron B ("lag grande antes, lag pequeno despues"). Nunca debe
    comprimirse en 16 ms ni aceptarse para campana.
    """
    base = 1_700_000_000.0
    source = base + np.array([0.000, 0.008, 0.708, 0.716])
    received = base + np.array([0.692, 0.700, 0.716, 0.724])

    res = reconstruct_simulation_clock(source, received)
    reconstructed = res["reconstructed_time_s"]

    assert reconstructed is not None
    assert np.max(np.diff(reconstructed)) >= 0.60
    assert res["usable_for_validation"] is False
    assert bool(res["ambiguous_step_mask"][1]) is True
    assert "ambiguous_clock_step_pattern_b" in res["reasons"]


def test_duplicates_and_backsteps_remain_visible_in_public_result():
    """
    Regresion C3: duplicados (dt == 0) y retrocesos (dt < 0) no explicados por Patron A
    quedan visibles tanto en `reconstructed_time_s` como en `backstep_or_duplicate_mask`.
    """
    base = 1_700_000_000.0
    source_backstep = base + np.array([0.0, 0.020, 0.010, 0.040])
    received = base + np.array([0.0, 0.020, 0.040, 0.060])

    res_backstep = reconstruct_simulation_clock(source_backstep, received)
    assert res_backstep["reconstructed_time_s"] is not None
    dts_back = np.diff(res_backstep["reconstructed_time_s"])
    assert dts_back[1] < 0.0
    assert bool(res_backstep["backstep_or_duplicate_mask"][1]) is True
    assert res_backstep["num_backsteps_or_duplicates"] == 1
    assert res_backstep["usable_for_validation"] is False
    assert "clock_backstep_or_duplicate_timestamp" in res_backstep["reasons"]

    source_dup = base + np.array([0.0, 0.020, 0.020, 0.040])
    res_dup = reconstruct_simulation_clock(source_dup, received)
    assert res_dup["reconstructed_time_s"] is not None
    dts_dup = np.diff(res_dup["reconstructed_time_s"])
    assert dts_dup[1] == 0.0
    assert bool(res_dup["backstep_or_duplicate_mask"][1]) is True
    assert res_dup["usable_for_validation"] is False


def test_pattern_a_conditioned_repair_preserves_neighbor_interval():
    """
    Regresion C3:
      1. Si una muestra aislada cae a tiempo de boot (< 1e6) entre vecinos UNIX validos (> 1e8)
         cuyo intervalo coincide con el de recepcion, se repara preservando el intervalo exacto
         entre los vecinos y registrando el evento.
      2. Si entre los vecinos UNIX validos existe un hueco real (ej. 0.50 s), la reparacion NO
         borra ese hueco y marca el paso como ambiguo e inutilizable para campana.
    """
    ts_bag = np.linspace(1_700_000_000.0, 1_700_000_001.0, 51)  # dt = 0.02 s
    ts_px4 = ts_bag.copy()
    ts_px4[20] = 42.5  # muestra aislada en tiempo de arranque

    res_ok = reconstruct_simulation_clock(ts_px4, ts_bag)
    assert res_ok["usable_for_validation"] is True
    assert res_ok["num_repaired_steps"] == 2
    assert res_ok["num_backsteps_or_duplicates"] == 0
    assert res_ok["num_ambiguous_steps"] == 0
    assert len(res_ok["repaired_events"]) == 1
    ev = res_ok["repaired_events"][0]
    assert ev["sample_index"] == 20
    assert ev["method"] == "pattern_a_proportional_neighbor_span"
    rec = res_ok["reconstructed_time_s"]
    assert rec is not None
    # El span entre los vecinos validos 19 y 21 se preserva exactamente
    assert (rec[21] - rec[19]) == pytest.approx(ts_px4[21] - ts_px4[19], abs=1e-9)
    assert np.allclose(rec, ts_bag - ts_bag[0], atol=1e-6)

    # Ahora simular que entre el vecino 19 y el 21 hubo un hueco real de 0.50 s en el reloj de PX4
    ts_px4_gap = ts_bag.copy()
    ts_px4_gap[20] = 42.5
    ts_px4_gap[21:] += 0.46  # span entre 19 y 21 pasa de 0.04 s a 0.50 s

    res_gap = reconstruct_simulation_clock(ts_px4_gap, ts_bag)
    rec_gap = res_gap["reconstructed_time_s"]
    assert rec_gap is not None
    # El hueco de 0.50 s entre los vecinos validos 19 y 21 NO desaparece
    assert (rec_gap[21] - rec_gap[19]) == pytest.approx(0.50, abs=1e-6)
    assert res_gap["usable_for_validation"] is False
    assert "ambiguous_clock_step_pattern_a" in res_gap["reasons"]
    assert bool(res_gap["ambiguous_step_mask"][19]) is True
    assert bool(res_gap["ambiguous_step_mask"][20]) is True


def test_missing_or_non_finite_clocks_are_rejected():
    """Relojes con NaN, inf, ceros o negativos se rechazan como inutilizables."""
    base = 1_700_000_000.0
    valid = base + np.array([0.0, 0.02, 0.04, 0.06])

    with_nan = valid.copy()
    with_nan[2] = np.nan
    res_nan = reconstruct_simulation_clock(with_nan, valid)
    assert res_nan["usable_for_validation"] is False
    assert res_nan["reconstructed_time_s"] is None
    assert "missing_or_non_finite_px4_clock" in res_nan["reasons"]

    with_zero = valid.copy()
    with_zero[1] = 0.0
    res_zero = reconstruct_simulation_clock(valid, with_zero)
    assert res_zero["usable_for_validation"] is False
    assert res_zero["reconstructed_time_s"] is None
    assert "missing_or_non_finite_bag_clock" in res_zero["reasons"]


def test_validate_episode_data_rejects_pattern_b_and_missing_clocks():
    """
    Regresion C3 de extremo a extremo sobre `validate_episode_data()`:
      - Vuelo limpio -> accepted.
      - Vuelo con escalon Patron B -> rejected con `ambiguous_clock_step_pattern_b`.
      - Vuelo con timestamps ausentes/cero -> rejected sin fabricar `i * 0.01`.
    """
    fx = _make_clean_flight_fixture()
    rep_clean = validate_episode_data(fx["telemetry"], fx["waypoints_cfg"], pilot_exit_code=0)
    assert rep_clean["validation_status"] == "accepted", rep_clean["rejection_reasons"]

    # 1. Inyectar un escalon tipo Patron B (lag_before=0.35 s -> lag_after=0.0 s) en mitad del vuelo
    fx_pb = _make_clean_flight_fixture()
    for i in range(100):
        # Retrasar ts_px4 de las primeras 100 muestras en 0.35 s respecto al bag
        old_us = fx_pb["telemetry"]["odometry"][i]["timestamp_sample_us"]
        fx_pb["telemetry"]["odometry"][i]["timestamp_sample_us"] = old_us - 350_000
        fx_pb["telemetry"]["odometry"][i]["timestamp_us"] = old_us - 350_000

    rep_pb = validate_episode_data(fx_pb["telemetry"], fx_pb["waypoints_cfg"], pilot_exit_code=0)
    assert rep_pb["validation_status"] == "rejected"
    assert rep_pb["recording_status"] == "invalid"
    assert any(r.startswith("ambiguous_clock_step_pattern_b") for r in rep_pb["rejection_reasons"])

    # 2. Eliminar timestamps de recepcion (timestamp=0) para comprobar que NO se fabrican con i * 0.01
    fx_no_bag_ts = _make_clean_flight_fixture()
    fx_no_bag_ts["telemetry"]["odometry"][10]["timestamp"] = 0
    rep_no_bag = validate_episode_data(
        fx_no_bag_ts["telemetry"], fx_no_bag_ts["waypoints_cfg"], pilot_exit_code=0
    )
    assert rep_no_bag["validation_status"] == "rejected"
    assert rep_no_bag["recording_status"] == "invalid"
    assert any(r.startswith("missing_or_invalid_bag_timestamps") for r in rep_no_bag["rejection_reasons"])

    # 3. Eliminar timestamps de PX4 (timestamp_sample_us=0, timestamp_us=0)
    fx_no_px4_ts = _make_clean_flight_fixture()
    fx_no_px4_ts["telemetry"]["odometry"][15]["timestamp_sample_us"] = 0
    fx_no_px4_ts["telemetry"]["odometry"][15]["timestamp_us"] = 0
    rep_no_px4 = validate_episode_data(
        fx_no_px4_ts["telemetry"], fx_no_px4_ts["waypoints_cfg"], pilot_exit_code=0
    )
    assert rep_no_px4["validation_status"] == "rejected"
    assert rep_no_px4["recording_status"] == "invalid"
    assert any(r.startswith("missing_or_invalid_px4_timestamps") for r in rep_no_px4["rejection_reasons"])


def test_timesync_status_analytical_disambiguation_repairs_pattern_b():
    """
    Tarea 2.5.4:
      1. Si un salto de reloj en VehicleOdometry coincide en ±0.05 s con un escalon en
         TimesyncStatus.estimated_offset de identica magnitud (Δt_odom ≈ Δestimated_offset),
         se valida formalmente como reajuste de offset de red, se repara `reconstructed_time_s`
         aplicando el diferencial de offset y se declara `usable_for_validation = True`
         (y `validation_status == "accepted"` en `validate_episode_data`).
      2. Si el salto en VehicleOdometry NO coincide en magnitud o ventana temporal con
         `timesync_status`, se mantiene el rechazo estricto por `ambiguous_clock_step_pattern_b`
         o `telemetry_dropout_gap`.
    """
    fx = _make_clean_flight_fixture()
    odom = fx["telemetry"]["odometry"]
    jump_idx = 100
    offset_jump_us = 350_000  # 0.35 s

    for i in range(jump_idx):
        odom[i]["timestamp_sample_us"] -= offset_jump_us
        odom[i]["timestamp_us"] -= offset_jump_us

    # Caso A: Sin timesync_status -> rechazado por ambiguous_clock_step_pattern_b
    rep_no_ts = validate_episode_data(fx["telemetry"], fx["waypoints_cfg"], pilot_exit_code=0)
    assert rep_no_ts["validation_status"] == "rejected"
    assert any(r.startswith("ambiguous_clock_step_pattern_b") for r in rep_no_ts["rejection_reasons"])

    # Caso B: Con timesync_status coincidente en tiempo (±0.01 s) y magnitud (350_000 us) -> aceptado
    t_jump_ns = odom[jump_idx]["timestamp"]
    t_jump_us = odom[jump_idx]["timestamp_sample_us"]
    fx["telemetry"]["timesync_status"] = [
        {
            "timestamp": t_jump_ns - 500_000_000,
            "timestamp_us": t_jump_us - 500_000 - offset_jump_us,
            "estimated_offset": -offset_jump_us,
            "estimated_offset_us": -offset_jump_us,
        },
        {
            "timestamp": t_jump_ns + 5_000_000,  # +0.005 s dentro de la ventana de ±0.05 s
            "timestamp_us": t_jump_us + 5_000,
            "estimated_offset": 0,
            "estimated_offset_us": 0,
        },
    ]

    rep_with_ts = validate_episode_data(fx["telemetry"], fx["waypoints_cfg"], pilot_exit_code=0)
    assert rep_with_ts["validation_status"] == "accepted", rep_with_ts["rejection_reasons"]
    assert rep_with_ts["recording_status"] == "valid"
    assert rep_with_ts["metrics"]["timesync_offset_jumps"] == 1
    assert rep_with_ts["metrics"]["ambiguous_clock_steps"] == 0
    assert rep_with_ts["metrics"]["max_dt_gap_s"] <= 0.15

    # Caso C: Con timesync_status de magnitud discordante (ej. 100 ms vs salto de 350 ms) -> rechazado
    fx["telemetry"]["timesync_status"][0]["estimated_offset"] = -100_000
    fx["telemetry"]["timesync_status"][0]["estimated_offset_us"] = -100_000
    rep_bad_mag = validate_episode_data(fx["telemetry"], fx["waypoints_cfg"], pilot_exit_code=0)
    assert rep_bad_mag["validation_status"] == "rejected"
    assert any(r.startswith("ambiguous_clock_step_pattern_b") for r in rep_bad_mag["rejection_reasons"])

    # Caso D: Con timesync_status fuera de la ventana temporal (±0.05 s, ej. desplazado 0.25 s) -> rechazado
    fx["telemetry"]["timesync_status"][0]["estimated_offset"] = -offset_jump_us
    fx["telemetry"]["timesync_status"][0]["estimated_offset_us"] = -offset_jump_us
    fx["telemetry"]["timesync_status"][1]["timestamp"] = t_jump_ns + 250_000_000
    fx["telemetry"]["timesync_status"][1]["timestamp_us"] = t_jump_us + 250_000
    rep_bad_time = validate_episode_data(fx["telemetry"], fx["waypoints_cfg"], pilot_exit_code=0)
    assert rep_bad_time["validation_status"] == "rejected"
    assert any(r.startswith("ambiguous_clock_step_pattern_b") for r in rep_bad_time["rejection_reasons"])

