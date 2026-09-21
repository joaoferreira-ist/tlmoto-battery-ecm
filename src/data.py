"""Load battery test data used by the example scripts."""

from pathlib import Path

import numpy as np


def load_battery_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load time, positive-discharge current, and measured voltage from a CSV."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    required = {"Time", "Current_measured", "Voltage_measured", "Temperature_measured"}
    if not required.issubset(data.dtype.names or ()):
        raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
    
    time_s = np.asarray(data["Time"], dtype=float)
    current_a = -np.asarray(data["Current_measured"], dtype=float)
    voltage_v = np.asarray(data["Voltage_measured"], dtype=float)
    temperature_c = np.asarray(data["Temperature_measured"], dtype=float)

    if not all(
        np.all(np.isfinite(values))
        for values in (time_s, current_a, voltage_v, temperature_c)
    ):
        raise ValueError("CSV contains non-finite values")
    if time_s.size > 1 and not np.all(np.diff(time_s) > 0):
        raise ValueError("Time values must be strictly increasing")

    return time_s, current_a, voltage_v, temperature_c