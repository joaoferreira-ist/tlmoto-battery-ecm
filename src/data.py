"""Load battery test data used by the example scripts."""

from pathlib import Path

import numpy as np


def load_discharge_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load time, positive-discharge current, and measured voltage from a CSV."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    required = {"Time", "Current_measured", "Voltage_measured"}
    if not required.issubset(data.dtype.names or ()):
        raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
    return (
        np.asarray(data["Time"], dtype=float),
        np.maximum(-np.asarray(data["Current_measured"], dtype=float), 0.0),
        np.asarray(data["Voltage_measured"], dtype=float),
    )