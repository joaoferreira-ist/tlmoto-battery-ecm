"""Run the battery ECM and save diagnostic figures."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_battery_csv
from src.equivalent_circuit_model import ECMParameters, SimulationResult, simulate


def plot_result(result: SimulationResult, output_path: Path, measured_voltage: np.ndarray | None = None) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(result.time_s, result.terminal_voltage_v, label="Model")
    if measured_voltage is not None:
        axes[0, 0].plot(result.time_s, measured_voltage, label="Measured", alpha=0.7)
        axes[0, 0].legend()
    axes[0, 0].set(xlabel="Time (s)", ylabel="Voltage (V)", title="Terminal voltage")
    axes[0, 1].plot(result.time_s, result.soc * 100.0, color="tab:green")
    axes[0, 1].set(xlabel="Time (s)", ylabel="SOC (%)", title="State of charge")
    axes[1, 0].plot(result.time_s, result.ocv_v, color="tab:orange")
    axes[1, 0].set(xlabel="Time (s)", ylabel="Voltage (V)", title="Open-circuit voltage")
    axes[1, 1].plot(result.time_s, result.transient_voltage_v, color="tab:red")
    axes[1, 1].set(xlabel="Time (s)", ylabel="Voltage (V)", title="RC transient voltage")
    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_error_graph(voltage_error: np.ndarray, output_path: Path, time: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(6, 4)) 
    axis.plot(time, voltage_error, color="tab:red")
    axis.set(xlabel="Time (s)", ylabel="Voltage (V)", title="Error between measured and simulated voltage")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

def calibrate_soc(
    current_a: np.ndarray,
    time_s: np.ndarray,
    measured_voltage: np.ndarray,
    parameters: ECMParameters,
    soc_min: float = 0.0,
    soc_max: float = 1.0,
    soc_step: float = 0.01,
) -> tuple[float, SimulationResult, float]:
    """Find the initial SOC that minimizes voltage RMSE for the measured trace."""
    if measured_voltage.ndim != 1 or measured_voltage.size == 0:
        raise ValueError("measured voltage must be a non-empty one-dimensional array")
    if current_a.size != measured_voltage.size or time_s.size != measured_voltage.size:
        raise ValueError("current, time, and measured voltage must have equal lengths")
    if not 0.0 <= soc_min <= soc_max <= 1.0:
        raise ValueError("SOC bounds must satisfy 0 <= soc_min <= soc_max <= 1")
    if soc_step <= 0:
        raise ValueError("soc_step must be positive")

    candidate_soc = np.arange(soc_min, soc_max + soc_step / 2.0, soc_step)
    best_soc = None
    best_result = None
    best_rmse = np.inf

    for soc in candidate_soc:
        candidate_result = simulate(current_a, time_s, parameters, initial_soc=float(soc))
        error = candidate_result.terminal_voltage_v - measured_voltage
        rmse = float(np.sqrt(np.mean(error**2)))
        if rmse < best_rmse:
            best_soc = float(soc)
            best_result = candidate_result
            best_rmse = rmse

    return best_soc, best_result, best_rmse

def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Optional battery CSV for measured-voltage comparison")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "figures" / "ecm_simulation.png")
    parser.add_argument("--initial-soc", type=float, default=0.93, help="Initial SOC for the fixed simulation run")
    parser.add_argument("--calibrate-soc", action="store_true", help="Fit the initial SOC from the measured data before plotting")
    args = parser.parse_args()

    if args.data:
        time_s, current_a, measured_voltage, temperature_c = load_battery_csv(args.data)
        parameters = ECMParameters(nominal_capacity_ah=1.6743)
    else:
        time_s = np.arange(0.0, 3600.0 * 0.95, 1.0)
        current_a = np.full_like(time_s, 5.0)
        measured_voltage = None
        temperature_c = None
        parameters = ECMParameters()

    if measured_voltage is not None:
        if args.calibrate_soc:
            initial_soc, result, calibrated_rmse = calibrate_soc(
                current_a, time_s, measured_voltage, parameters
            )
            print(f"Calibrated initial SOC: {initial_soc:.2f}")
            print(f"Calibrated voltage RMSE: {calibrated_rmse:.4f} V")
        else:
            initial_soc = args.initial_soc
            result = simulate(current_a, time_s, parameters, initial_soc=initial_soc)
            print(f"Using fixed initial SOC: {initial_soc:.2f}")
    else:
        initial_soc = args.initial_soc
        result = simulate(current_a, time_s, parameters, initial_soc=initial_soc)

    if measured_voltage is not None:
        voltage_error = result.terminal_voltage_v - measured_voltage
        mean_error = np.mean(voltage_error)
        rmse = np.sqrt(np.mean(voltage_error**2))
        print(f"Voltage RMSE: {rmse:.4f} V")
        print(f"Voltage mean error: {mean_error:.4f}V")
        print(f"1º valor da tensão medida é {measured_voltage[0]:.4f} V \n1º valor da tensão simulada é {result.terminal_voltage_v[0]:.4f} V \nDiferença absoluta entre os dois valores é {abs(measured_voltage[0] - result.terminal_voltage_v[0]):.4f} V")
        print(f"Initial SOC used: {initial_soc:.4f}")
        plot_error_graph(voltage_error, args.output.parent / "voltage_error_graph_time.png", time_s)
    plot_result(result, args.output, measured_voltage)
    
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()