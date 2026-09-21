"""SOC update by coulomb counting.

This is the simplest battery state estimator: it integrates current over time and
converts the result into a state-of-charge fraction in the range [0, 1].
"""


def update_soc(soc_previous: float, current: float, dt: float, nominal_capacity_ah: float) -> float:
    """Advance SOC by one timestep.

    The sign convention is: positive current represents discharge, negative current
    represents charge. The SOC update is therefore:

        SOC_new = SOC_old - current * dt / (capacity * 3600)

    where the current is in amperes and dt is in seconds.
    """
    if dt < 0 or nominal_capacity_ah <= 0:
        raise ValueError("dt must be non-negative and capacity must be positive")
    delta_soc = current * dt / (nominal_capacity_ah * 3600.0)
    return min(1.0, max(0.0, soc_previous - delta_soc))