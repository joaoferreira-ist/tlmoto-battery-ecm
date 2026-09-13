"""State-of-charge estimation by current integration."""


def update_soc(soc_previous: float, current: float, dt: float, nominal_capacity_ah: float) -> float:
    """Advance SOC by one timestep.

    Current is positive when the cell is discharging and negative when charging.
    """
    if dt < 0 or nominal_capacity_ah <= 0:
        raise ValueError("dt must be non-negative and capacity must be positive")
    delta_soc = current * dt / (nominal_capacity_ah * 3600.0)
    return min(1.0, max(0.0, soc_previous - delta_soc))