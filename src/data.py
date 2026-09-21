"""Data-loading utilities for experimental battery test data.

The raw CSV files use the convention where discharge current is negative. The
model internally works with positive discharge current, so the loader converts
that sign at the boundary.
"""

from pathlib import Path

import numpy as np


def load_battery_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a battery-cycle CSV and return time, model current, voltage, and temperature.

    The dataset is expected to contain the measured current and voltage for a
    battery discharge or charge cycle. The conversion to the model convention is:
    model_current = -measured_current
    because the CSV stores discharge as negative current.
    """
    data = np.genfromtxt(path, delimiter=",", names=True)
    required = {"Time", "Current_measured", "Voltage_measured", "Temperature_measured"}
    if not required.issubset(data.dtype.names or ()):
        raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")

    # Keep the original time axis; the ECM can handle irregular sampling.
    time_s = np.asarray(data["Time"], dtype=float)
    # Convert the measured current sign to the model convention used by the ECM.
    current_a = -np.asarray(data["Current_measured"], dtype=float)
    voltage_v = np.asarray(data["Voltage_measured"], dtype=float)
    temperature_c = np.asarray(data["Temperature_measured"], dtype=float)

    # Reject missing or corrupted measurements before the model uses them.
    if not all(np.all(np.isfinite(values)) for values in (time_s, current_a, voltage_v, temperature_c)):
        raise ValueError("CSV contains non-finite values")
    if time_s.size > 1 and not np.all(np.diff(time_s) > 0):
        raise ValueError("Time values must be strictly increasing")

    return time_s, current_a, voltage_v, temperature_c