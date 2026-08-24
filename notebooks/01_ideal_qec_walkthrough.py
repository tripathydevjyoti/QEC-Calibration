# %% [markdown]
# # Ideal three-qubit bit-flip quantum error correction
#
# This notebook verifies the ideal baseline for the gate-calibration project.
# It encodes one logical bit into three data qubits, extracts the two parity
# syndromes with two ancillas, and applies syndrome-conditioned recovery.

# %% [markdown]
# ## Project structure
#
# Reusable circuit construction and simulation remain in
# `src/qec_calibration/ideal_bit_flip_code.py`. This notebook imports those
# tested functions and is responsible only for analysis, tables, and plots.

# %%
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from qiskit.visualization import plot_histogram


PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qec_calibration import (  # noqa: E402
    EXPECTED_SYNDROMES,
    build_recovery_circuit,
    build_syndrome_circuit,
    run_recovery,
    run_syndrome,
)

SHOTS = 1024
SEED = 42

# %% [markdown]
# ## 1. Build and visualize the syndrome circuit
#
# The example below inserts an `X` error on the middle data qubit. The expected
# conceptual syndrome is `11` because both neighboring parity checks change.

# %%
syndrome_circuit = build_syndrome_circuit(
    error_qubit=1,
    logical_value=0,
)

syndrome_circuit.draw(output="mpl", fold=-1)

# %% [markdown]
# ## 2. Verify the syndrome truth table
#
# The public helper functions return syndromes in conceptual `(s0, s1)` order,
# even though Qiskit displays classical bits in reverse bit-index order.

# %%
def error_label(error_qubit):
    return "None" if error_qubit is None else f"X{error_qubit}"


truth_table_rows = []
syndrome_results = {}

for error_qubit, expected_syndrome in EXPECTED_SYNDROMES.items():
    counts = run_syndrome(
        error_qubit=error_qubit,
        logical_value=0,
        shots=SHOTS,
        seed=SEED,
    )
    syndrome_results[error_label(error_qubit)] = counts
    observed_syndrome = max(counts, key=counts.get)

    truth_table_rows.append(
        {
            "Physical error": error_label(error_qubit),
            "Expected syndrome": expected_syndrome,
            "Observed syndrome": observed_syndrome,
            "Observed counts": counts[observed_syndrome],
            "Passed": observed_syndrome == expected_syndrome,
        }
    )

truth_table = pd.DataFrame(truth_table_rows)
truth_table

# %% [markdown]
# All four rows should have `Passed = True`. Because the baseline simulator is
# ideal, each circuit produces one deterministic syndrome on every shot.

# %% [markdown]
# ## 3. Plot syndrome measurements

# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

for ax, (label, counts) in zip(axes.flat, syndrome_results.items()):
    plot_histogram(counts, ax=ax, title=f"Injected error: {label}")
    ax.set_xlabel("Conceptual syndrome (s0s1)")
    ax.set_ylabel("Counts")

plt.show()

# %% [markdown]
# ## 4. Verify active correction
#
# We test both logical basis states and all four error cases. After recovery,
# the measured data must be `000` for logical zero and `111` for logical one.

# %%
recovery_rows = []

for logical_value in (0, 1):
    expected_data = "000" if logical_value == 0 else "111"

    for error_qubit, expected_syndrome in EXPECTED_SYNDROMES.items():
        counts = run_recovery(
            error_qubit=error_qubit,
            logical_value=logical_value,
            shots=SHOTS,
            seed=SEED,
        )
        (observed_data, observed_syndrome), observed_count = max(
            counts.items(), key=lambda item: item[1]
        )

        recovery_rows.append(
            {
                "Logical input": logical_value,
                "Physical error": error_label(error_qubit),
                "Syndrome": observed_syndrome,
                "Expected data": expected_data,
                "Recovered data": observed_data,
                "Observed counts": observed_count,
                "Passed": (
                    observed_data == expected_data
                    and observed_syndrome == expected_syndrome
                ),
            }
        )

recovery_table = pd.DataFrame(recovery_rows)
recovery_table

# %%
assert truth_table["Passed"].all()
assert recovery_table["Passed"].all()

print("Ideal syndrome extraction: PASS")
print("Ideal active recovery: PASS")

# %% [markdown]
# ## 5. Inspect the recovery circuit

# %%
recovery_circuit = build_recovery_circuit(
    error_qubit=1,
    logical_value=0,
)

recovery_circuit.draw(output="mpl", fold=-1)

# %% [markdown]
# ## Baseline conclusion
#
# The ideal implementation distinguishes no error and all three possible
# single-qubit bit flips. The classically conditioned recovery restores both
# logical basis states. This becomes the reference baseline for the next stage:
# replacing the discrete `X1` error with a coherent `Rx(epsilon)` error and
# estimating its strength from repeated syndrome measurements.

