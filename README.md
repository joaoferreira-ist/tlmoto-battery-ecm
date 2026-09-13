# Battery Equivalent Circuit Model

Python implementation of a battery equivalent circuit model (ECM) with:

- coulomb counting for state-of-charge estimation;
- OCV-SOC interpolation;
- two RC transient branches;
- terminal-voltage simulation; and
- optional comparison with measured battery data.

## Model

For a positive discharge current, the terminal voltage is modeled as:

```text
V_terminal = OCV(SOC) - I * R0 - V_RC1 - V_RC2
```

The sample data uses negative current for discharge, so the loader converts it
to the model's positive-discharge convention.

## Quick start

```powershell
python -m pip install -r requirements.txt
python scripts/run_simulation.py
python scripts/run_simulation.py --data dataresults/00001.csv --output results/figures/measured_comparison.png
python -m pytest
```

The simulation saves a four-panel diagnostic figure instead of requiring an
interactive plotting window.

## Project structure

```text
src/                         Reusable model and data-loading code
scripts/run_simulation.py    Command-line simulation and plotting entry point
dataresults/                 Example data and data-use notes
tests/                       Focused unit tests
results/                     Reproducible output location
```

## Limitations

The included OCV curve and ECM parameters are demonstration values, not a
calibrated representation of a particular cell. The measured comparison is
therefore a workflow example rather than a model-accuracy claim. Temperature
dependence, automatic parameter fitting, and formal validation metrics remain
future work.

## Future work

- Fit OCV and ECM parameters from measured cycles.
- Add temperature-dependent parameters.
- Report MAE and RMSE for measured-versus-simulated voltage.
- Add charge and regenerative-current operating modes.

## License and data provenance

Add a project license before publishing. Confirm the original battery dataset
license and include its required citation before redistributing raw files.