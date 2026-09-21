"""Open-circuit-voltage versus state-of-charge interpolation.

This file provides a simple lookup-based OCV model. The values are intentionally
kept lightweight for a prototype, but they represent the battery's voltage as a
function of state of charge.
"""

import numpy as np

# A simple demonstration OCV curve for a lithium-ion cell.
SOC_POINTS = np.linspace(0.0, 1.0, 11)
OCV_POINTS = np.array([3.00, 3.30, 3.45, 3.55, 3.62, 3.68, 3.72, 3.78, 3.85, 3.95, 4.10])


def ocv_from_soc(soc: float | np.ndarray) -> float | np.ndarray:
    """Return interpolated OCV voltage for one or more SOC values.

    The SOC values are clipped to the physically meaningful range [0, 1] before
    interpolation, which prevents invalid inputs from breaking the model.
    """
    return np.interp(np.clip(soc, 0.0, 1.0), SOC_POINTS, OCV_POINTS)