"""Open-circuit-voltage versus state-of-charge interpolation."""

import numpy as np

SOC_POINTS = np.linspace(0.0, 1.0, 11)
OCV_POINTS = np.array([3.00, 3.30, 3.45, 3.55, 3.62, 3.68, 3.72, 3.78, 3.85, 3.95, 4.10])


def ocv_from_soc(soc: float | np.ndarray) -> float | np.ndarray:
    """Return interpolated OCV in volts for SOC values from 0 to 1."""
    return np.interp(np.clip(soc, 0.0, 1.0), SOC_POINTS, OCV_POINTS)