# %% [markdown]
# # One faulty gate: coherent error and syndrome analysis
#
# We now replace the ideal first encoding CNOT with
#
# $$\widetilde{CX}_{0,1}=R_x^{(1)}(\epsilon)CX_{0,1}.$$
#
# Aer implements this as an ideal `cx(0, 1)` followed by a coherent two-qubit
# noise channel whose only non-identity action is `Rx(epsilon)` on the target.
# Every other gate and both syndrome measurements remain ideal.

# %% [markdown]
# ## 1. Theory
#
# For an arbitrary encoded state
#
# $$|\psi_L\rangle=\alpha|000\rangle+\beta|111\rangle,$$
#
# the residual rotation produces
#
# $$|\widetilde{\psi}_L\rangle=
# \cos(\epsilon/2)|\psi_L\rangle
# -i\sin(\epsilon/2)X_1|\psi_L\rangle.$$
#
# The two branches have different syndromes. The logical branch gives `00`,
# while the middle-qubit bit-flip branch gives `11`. Therefore
#
# $$P(00)=\cos^2(\epsilon/2),\qquad
# P(11)=\sin^2(\epsilon/2).$$

# %%
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit.visualization import plot_histogram


PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qec_calibration import (  # noqa: E402
    build_syndrome_circuit,
    counts_to_probabilities,
    run_error_sweep,
    run_faulty_cx_syndrome,
    theoretical_syndrome_probabilities,
)

EPSILON = 0.20
SHOTS = 20_000
SEED = 42

# %% [markdown]
# ## 2. Inspect the circuit
#
# The circuit drawing is intentionally ideal-looking: the error lives in the
# Aer `NoiseModel`, not as a visible instruction. It is attached only to the
# ordered physical pair `[0, 1]`, which uniquely identifies the first encoding
# CNOT in this circuit.

# %%
circuit = build_syndrome_circuit(error_qubit=None, logical_value=0)
circuit.draw(output="mpl", fold=-1)

# %% [markdown]
# ## 3. Predict the syndrome distribution

# %%
theory = theoretical_syndrome_probabilities(EPSILON)
theory_table = pd.DataFrame(
    {
        "Syndrome": list(theory),
        "Theoretical probability": list(theory.values()),
        "Expected counts": [SHOTS * probability for probability in theory.values()],
    }
)
theory_table

# %% [markdown]
# At `epsilon = 0.20` radians, the expected `11` rate is about one percent, or
# roughly 199 detections in 20,000 shots.

# %% [markdown]
# ## 4. Run the faulty-gate experiment

# %%
counts = run_faulty_cx_syndrome(
    epsilon=EPSILON,
    logical_value=0,
    shots=SHOTS,
    seed=SEED,
)
observed = counts_to_probabilities(counts)

comparison = pd.DataFrame(
    {
        "Syndrome": list(theory),
        "Aer counts": [counts.get(syndrome, 0) for syndrome in theory],
        "Aer probability": [observed[syndrome] for syndrome in theory],
        "Theory": [theory[syndrome] for syndrome in theory],
    }
)
comparison["Absolute difference"] = abs(
    comparison["Aer probability"] - comparison["Theory"]
)
comparison

# %%
plot_histogram(counts, title=f"Faulty CX(0, 1): epsilon = {EPSILON:.2f} rad")
plt.xlabel("Conceptual syndrome (s0s1)")
plt.ylabel("Counts")
plt.show()

# %% [markdown]
# ## 5. Sweep the gate-error angle
#
# The observable calibration signal is the `11` frequency. For now we sweep a
# known injected angle and verify that the samples follow the analytic curve.
# In the next stage, an optimizer will instead update a compensating angle to
# minimize this syndrome rate.

# %%
epsilon_values = np.linspace(0.0, 0.8, 17)
sweep = pd.DataFrame(
    run_error_sweep(
        epsilon_values,
        logical_value=0,
        shots=5_000,
        seed=SEED,
    )
)
sweep

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(
    sweep["epsilon"],
    sweep["p11_theory"],
    label=r"Theory: $\sin^2(\epsilon/2)$",
    linewidth=2,
)
ax.scatter(
    sweep["epsilon"],
    sweep["p11_empirical"],
    label="Aer samples",
    color="tab:orange",
    zorder=3,
)
ax.set(
    xlabel=r"Coherent error angle $\epsilon$ (radians)",
    ylabel="Middle-qubit syndrome probability P(11)",
    title="Syndrome rate reveals the coherent gate-error magnitude",
)
ax.set_ylim(bottom=0)
ax.grid(alpha=0.25)
ax.legend()
plt.show()

# %% [markdown]
# ## 6. Numerical checks
#
# Sampling fluctuates, so we use a five-standard-deviation binomial interval
# rather than requiring exact equality.

# %%
p_11 = theory["11"]
standard_error = np.sqrt(p_11 * (1 - p_11) / SHOTS)

assert set(counts) <= {"00", "11"}
assert abs(observed["11"] - p_11) <= 5 * standard_error + 1 / SHOTS
assert sweep["absolute_error_p11"].max() < 0.03

print(f"Injected angle: {EPSILON:.3f} rad")
print(f"Theory P(11):   {p_11:.6f}")
print(f"Aer P(11):      {observed['11']:.6f}")
print("Analytic-versus-Aer validation: PASS")

# %% [markdown]
# ## Conclusion
#
# A coherent error does not produce a bit flip on every shot. Syndrome
# measurement turns its two coherent branches into a classical distribution:
# `00` with probability `cos²(epsilon/2)` and `11` with probability
# `sin²(epsilon/2)`. This measured `11` rate is our first calibration signal.
