import numpy as np
import pytest

from src.coulomb_counting import update_soc
from src.equivalent_circuit_model import ECMParameters, simulate
from src.ocv_soc import ocv_from_soc


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