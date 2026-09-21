from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

from src.coulomb_counting import update_soc
from src.equivalent_circuit_model import ECMParameters, simulate
from src.ocv_soc import ocv_from_soc

script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_simulation.py"
spec = spec_from_file_location("run_simulation", script_path)
run_simulation = module_from_spec(spec)
spec.loader.exec_module(run_simulation)


def test_ocv_clips_outside_soc_range():
    assert ocv_from_soc(-1.0) == pytest.approx(3.0)
    assert ocv_from_soc(2.0) == pytest.approx(4.1)


def test_coulomb_counting_decreases_soc_during_discharge():
    assert update_soc(1.0, 5.0, 360.0, 5.0) == pytest.approx(0.9)


def test_coulomb_counting_clamps_soc():
    assert update_soc(0.1, 100.0, 3600.0, 5.0) == 0.0
    assert update_soc(0.9, -100.0, 3600.0, 5.0) == 1.0


def test_simulation_returns_decreasing_soc_and_loaded_voltage():
    result = simulate(np.array([5.0, 5.0, 5.0]), np.array([0.0, 1.0, 2.0]), ECMParameters())
    assert result.soc[0] > result.soc[-1]
    assert result.terminal_voltage_v[0] < result.ocv_v[0]


def test_simulation_rejects_non_monotonic_time():
    with pytest.raises(ValueError, match="monotonically"):
        simulate(np.ones(2), np.array([1.0, 0.0]), ECMParameters())


def test_calibrate_soc_recovers_the_true_initial_soc():
    parameters = ECMParameters()
    time_s = np.array([0.0, 1.0, 2.0, 3.0])
    current_a = np.array([5.0, 5.0, 5.0, 5.0])
    target_soc = 0.85
    target_voltage = simulate(current_a, time_s, parameters, initial_soc=target_soc).terminal_voltage_v

    best_soc, _, best_rmse = run_simulation.calibrate_soc(
        current_a, time_s, target_voltage, parameters, soc_min=0.0, soc_max=1.0, soc_step=0.01
    )

    assert best_soc == pytest.approx(target_soc, abs=0.02)
    assert best_rmse <= 1e-9