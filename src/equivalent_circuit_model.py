"""First-order numerical simulation of a two-RC battery ECM.

This model combines:
- coulomb counting for SOC estimation,
- a lookup-based OCV-SOC curve,
- and two RC branches to capture transient voltage behavior.
"""

from dataclasses import dataclass

import numpy as np

from .coulomb_counting import update_soc
from .ocv_soc import ocv_from_soc


@dataclass(frozen=True)
class ECMParameters:
    """Electrical parameters for the equivalent circuit model.

    These are treated as fixed model parameters for a given cell configuration;
    initial SOC and transient states are handled separately during simulation.
    """

    nominal_capacity_ah: float = 1.6743
    r0_ohm: float = 0.02
    r1_ohm: float = 0.01
    c1_farads: float = 2000.0
    r2_ohm: float = 0.005
    c2_farads: float = 1000.0


@dataclass
class SimulationResult:
    """Time histories returned by :func:`simulate` for one test cycle."""

    time_s: np.ndarray
    current_a: np.ndarray
    terminal_voltage_v: np.ndarray
    soc: np.ndarray
    ocv_v: np.ndarray
    transient_voltage_v: np.ndarray


def _validate_parameters(parameters: ECMParameters) -> None:
    """Validate the physical plausibility of the ECM parameter set."""
    if parameters.nominal_capacity_ah <= 0:
        raise ValueError("nominal capacity must be positive")
    for resistance, capacitance in ((parameters.r1_ohm, parameters.c1_farads), (parameters.r2_ohm, parameters.c2_farads)):
        if resistance <= 0 or capacitance <= 0:
            raise ValueError("RC resistances and capacitances must be positive")


def simulate(current_a: np.ndarray, time_s: np.ndarray, parameters: ECMParameters, initial_soc: float = 1.0) -> SimulationResult:
    """Simulate terminal voltage for a positive-discharge current profile."""
    _validate_parameters(parameters)
    current = np.asarray(current_a, dtype=float)
    time = np.asarray(time_s, dtype=float)
    if current.ndim != 1 or time.ndim != 1 or current.size != time.size or current.size == 0:
        raise ValueError("current and time must be non-empty one-dimensional arrays of equal length")
    if np.any(np.diff(time) < 0):
        raise ValueError("time must be monotonically increasing")
    if not 0.0 <= initial_soc <= 1.00:
        raise ValueError("initial SOC must be between 0 and 1")

    # Pre-allocate outputs so each time step can be filled in-place.
    soc = np.empty_like(time)
    ocv = np.empty_like(time)
    terminal_voltage = np.empty_like(time)
    transient_voltage = np.empty_like(time)

    # The RC branches are assumed to start from a relaxed condition at t=0.
    soc_value, v1, v2 = initial_soc, 0.0, 0.0
    for index, (current_value, timestamp) in enumerate(zip(current, time)):
        # Use zero dt on the first sample to avoid a backward subtraction.
        dt = 0.0 if index == 0 else timestamp - time[index - 1]
        decay_1 = np.exp(-dt / (parameters.r1_ohm * parameters.c1_farads))
        decay_2 = np.exp(-dt / (parameters.r2_ohm * parameters.c2_farads))

        # First-order RC voltage update for each branch.
        v1 = v1 * decay_1 + current_value * parameters.r1_ohm * (1 - decay_1)
        v2 = v2 * decay_2 + current_value * parameters.r2_ohm * (1 - decay_2)

        soc[index] = soc_value
        ocv[index] = ocv_from_soc(soc_value)
        transient_voltage[index] = v1 + v2
        terminal_voltage[index] = ocv[index] - current_value * parameters.r0_ohm - transient_voltage[index]

        # Update SOC after the current sample is applied.
        soc_value = update_soc(soc_value, current_value, dt, parameters.nominal_capacity_ah)
    return SimulationResult(time, current, terminal_voltage, soc, ocv, transient_voltage)
