# QEC Gate Calibration

This repository contains a verified implementation of the ideal three-qubit
bit-flip repetition code in Qiskit Aer and the first noisy experiment: one
encoding CNOT with a coherent target-qubit over-rotation.

## Repository workflow

The project deliberately separates reusable implementation from visual
analysis:

```text
qec-gate-calibration/
├── src/qec_calibration/       # Reusable circuit and simulation functions
├── notebooks/                 # Visual analysis and experiment narratives
├── examples/                  # Small command-line demonstrations
├── tests/                     # Automated correctness checks
├── pyproject.toml             # Dependencies and package configuration
└── README.md
```

The `.py` modules under `src/` are the source of truth. Notebooks import those
functions rather than redefining them. This keeps experiments reproducible and
prevents fixes made in a notebook from diverging from the tested code.

The notebook has a matching `# %%` Python file. It can be opened as an
interactive notebook in VS Code and gives GitHub a clean, line-by-line diff of
the analysis code.

## What the implementation contains

- Three data qubits encoding one logical qubit:
  - `|0_L> = |000>`
  - `|1_L> = |111>`
- Two syndrome ancillas measuring the stabilizers:
  - `S0 = Z0 Z1`
  - `S1 = Z1 Z2`
- Optional insertion of one physical `X` error.
- Mid-circuit syndrome measurement and classically conditioned recovery.
- Automated tests for every correctable single-qubit bit flip.
- A faulty first encoding gate modeled as
  `faulty_CX(0, 1) = Rx_1(epsilon) CX(0, 1)`.
- Analytic and sampled syndrome probabilities for that one-gate error.
- An error-angle sweep that exposes the syndrome-rate calibration landscape.
- An explicit compensating `Rx(-theta)` control and a shot-based,
  coarse-to-fine calibration loop.

The conceptual syndrome ordering used throughout the project is `(s0, s1)`:

| Error | Syndrome | Recovery |
|---|---|---|
| None | `00` | Identity |
| `X0` | `10` | `X0` |
| `X1` | `11` | `X1` |
| `X2` | `01` | `X2` |

Qiskit prints classical bit strings in descending bit-index order. The helper
functions in this project normalize raw Qiskit output back to the conceptual
`s0s1` ordering shown above.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run the demonstration

```bash
python examples/run_ideal_code.py
```

The demonstration prints:

1. The syndrome circuit.
2. The complete ideal syndrome truth table.
3. Recovery results for both logical basis states.

## Run the visual notebook

Launch Jupyter from the repository root:

```bash
jupyter lab
```

Then open:

```text
notebooks/01_ideal_qec_walkthrough.ipynb
notebooks/02_single_gate_error_analysis.ipynb
notebooks/03_single_gate_calibration.ipynb
```

The first notebook verifies ideal syndrome extraction and active recovery. The
second compares one faulty encoding CNOT against
`P(11) = sin(epsilon/2)^2`. The third calibrates a compensating angle by
minimizing the sampled nonzero-syndrome rate.

## Run the tests

```bash
pytest -q
```

## Project stages

- [x] Ideal three-qubit bit-flip code.
- [x] Syndrome truth-table verification.
- [x] Syndrome-conditioned recovery verification.
- [x] Visual ideal-QEC analysis notebook.
- [x] One coherent `Rx` gate error.
- [x] Analytic-versus-Aer syndrome comparison.
- [x] Syndrome-rate calibration objective.
- [x] Closed-loop optimization of one correction angle.
- [ ] Multiple gate errors and multi-parameter optimization.

## Scope of the code

The QEC circuit corrects one `X` error occurring between encoding and syndrome
extraction. The current calibration model targets one coherent `Rx` residual
on the first encoding CNOT. It does not yet calibrate phase errors, multiple
simultaneous gate errors, readout errors, or syndrome-extraction faults.
