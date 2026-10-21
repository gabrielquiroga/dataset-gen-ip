import numpy as np

from uav_ip.validate_episode import (
    _detect_timesync_and_clock_anomalies,
    reconstruct_simulation_clock,
)


def test_unexplained_source_gap_is_not_silently_erased():
    # Sin evidencia adicional de offset, no convertir 700 ms en 8 ms.
    source = np.array([0.0, 0.008, 0.708, 0.716])
    received = np.array([0.0, 0.008, 0.016, 0.024])

    reconstructed, _ = reconstruct_simulation_clock(source, received)

    assert np.max(np.diff(reconstructed)) >= 0.60


def test_reconstruction_does_not_claim_a_monotonic_clock_with_backsteps():
    source = np.array([0.0, 0.008, -0.002, 0.016])
    received = np.array([0.0, 0.008, 0.016, 0.024])

    reconstructed, _ = reconstruct_simulation_clock(source, received)

    assert np.all(np.diff(reconstructed) > 0)


def test_verified_timesync_boot_drop_is_repaired_while_unexplained_backstep_is_flagged():
    # Patron A: muestra aislada con timestamp crudo desde arranque (< 1e6 s) rodeada de timestamps UNIX (> 1e8 s)
    ts_bag = np.linspace(1700000000.0, 1700000001.0, 50)
    ts_px4 = ts_bag.copy()
    ts_px4[20] = 42.5  # caída aislada a tiempo de boot de PX4 sin offset UNIX

    reconstructed, jumps = reconstruct_simulation_clock(ts_px4, ts_bag)
    _, verified_mask, backstep_mask = _detect_timesync_and_clock_anomalies(ts_px4, ts_bag)
    assert jumps == 2
    assert int(np.sum(verified_mask)) == 2
    assert int(np.sum(backstep_mask)) == 0
    assert np.all(np.diff(reconstructed) > 0)
    assert np.allclose(reconstructed, ts_bag - ts_bag[0], atol=1e-6)

    # Retroceso de reloj NO explicado por Timesync: se regulariza para mantener monotonicidad del eje,
    # pero se contabiliza en `backstep_mask` para que validate_episode_data lo rechace.
    source_bad = np.array([0.0, 0.008, -0.002, 0.016])
    received_bad = np.array([0.0, 0.008, 0.016, 0.024])
    rec_bad, _ = reconstruct_simulation_clock(source_bad, received_bad)
    _, verified_bad, backstep_bad = _detect_timesync_and_clock_anomalies(source_bad, received_bad)
    assert int(np.sum(verified_bad)) == 0
    assert int(np.sum(backstep_bad)) == 1
    assert np.all(np.diff(rec_bad) > 0)
