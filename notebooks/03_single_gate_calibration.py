# %% [markdown]
# # Closed-loop calibration of one faulty encoding gate
#
# The faulty CNOT contributes a hidden target-qubit rotation
# `Rx(epsilon)`. We add a tunable control `Rx(-theta)`, leaving
#
# $$R_x(-\theta)R_x(\epsilon)=R_x(\epsilon-\theta).$$
#
# The calibration loop chooses `theta` using only measured syndrome counts.
# The hidden `epsilon` is supplied to Aer only to simulate the physical gate.

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
    build_compensated_syndrome_circuit,
    calibrate_single_gate,
    run_compensated_syndrome,
    theoretical_compensated_loss,
)

TRUE_ERROR = 0.20
SEARCH_BOUNDS = (-0.4, 0.4)
LEVELS = 3
POINTS_PER_LEVEL = 9
SHOTS_PER_POINT = 5_000
VALIDATION_SHOTS = 20_000
SEED = 42

# %% [markdown]
# ## 1. Inspect the compensation circuit
#
# The circuit contains `Rx(-theta)` immediately after the faulty `cx(0, 1)`.
# The coherent `Rx(epsilon)` error itself remains in the Aer noise model and is
# therefore not visible in the drawing.

# %%
trial_circuit = build_compensated_syndrome_circuit(
    correction_angle=0.10,
    logical_value=0,
)
trial_circuit.draw(output="mpl", fold=-1)

# %% [markdown]
# ## 2. Define the measured objective
#
# We minimize the observed probability of any nonzero syndrome:
#
# $$L(\theta)=1-\widehat{P}(00).$$
#
# For the present single-axis error model, only `00` and `11` occur, so this is
# equivalent to $\widehat{P}(11)$. The exact landscape is
#
# $$L_{\mathrm{exact}}(\theta)=
# \sin^2\left(\frac{\epsilon-\theta}{2}\right).$$

# %%
initial_counts = run_compensated_syndrome(
    true_error=TRUE_ERROR,
    correction_angle=0.0,
    shots=VALIDATION_SHOTS,
    seed=SEED,
)

print("Initial counts:", initial_counts)
print(
    "Initial theoretical loss:",
    theoretical_compensated_loss(TRUE_ERROR, 0.0),
)

# %% [markdown]
# ## 3. Run the coarse-to-fine calibration
#
# Each level evaluates nine trial angles. The next interval is centered on the
# smallest sampled syndrome rate and narrowed to one grid spacing on either
# side. Tied zero-count trials are averaged to center the shot-noise plateau.

# %%
result = calibrate_single_gate(
    true_error=TRUE_ERROR,
    search_bounds=SEARCH_BOUNDS,
    levels=LEVELS,
    points_per_level=POINTS_PER_LEVEL,
    shots_per_point=SHOTS_PER_POINT,
    validation_shots=VALIDATION_SHOTS,
    seed=SEED,
)

history = pd.DataFrame(result.evaluations)
level_summary = pd.DataFrame(result.level_summaries)
level_summary

# %%
print(f"Hidden error angle:      {result.true_error:+.6f} rad")
print(f"Estimated correction:   {result.estimated_correction:+.6f} rad")
print(f"Remaining residual:      {result.residual_angle:+.6f} rad")
print(f"Initial expected loss:   {result.initial_theoretical_loss:.6f}")
print(f"Validated measured loss: {result.validation_loss:.6f}")
print("Validation counts:", result.validation_counts)

# %% [markdown]
# ## 4. Visualize the search landscape
#
# The optimizer sees only the sampled points. The analytic curve is plotted
# afterward as a diagnostic—not supplied to the search.

# %%
theta_curve = np.linspace(*SEARCH_BOUNDS, 500)
theory_curve = [
    theoretical_compensated_loss(TRUE_ERROR, theta)
    for theta in theta_curve
]

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(theta_curve, theory_curve, color="black", label="Exact loss")

for level, rows in history.groupby("level"):
    ax.scatter(
        rows["correction_angle"],
        rows["measured_loss"],
        s=55,
        label=f"Measured level {level}",
        zorder=3,
    )

ax.axvline(TRUE_ERROR, color="tab:green", linestyle="--", label="Hidden epsilon")
ax.axvline(
    result.estimated_correction,
    color="tab:red",
    linestyle=":",
    label="Estimated theta",
)
ax.set(
    xlabel=r"Correction angle $\theta$ (radians)",
    ylabel="Nonzero-syndrome rate",
    title="Shot-based calibration landscape",
)
ax.set_ylim(bottom=0)
ax.grid(alpha=0.25)
ax.legend()
plt.show()

# %% [markdown]
# ## 5. Inspect convergence by level

# %%
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(
    level_summary["level"],
    level_summary["level_estimate"],
    marker="o",
    label="Estimated correction",
)
ax.axhline(TRUE_ERROR, color="tab:green", linestyle="--", label="Hidden error")
ax.set(
    xlabel="Search level",
    ylabel="Angle (radians)",
    title="Correction-angle convergence",
    xticks=level_summary["level"],
)
ax.grid(alpha=0.25)
ax.legend()
plt.show()

# %% [markdown]
# ## 6. Compare syndromes before and after calibration

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
plot_histogram(initial_counts, ax=axes[0], title="Before calibration")
plot_histogram(result.validation_counts, ax=axes[1], title="After calibration")

for ax in axes:
    ax.set_xlabel("Conceptual syndrome (s0s1)")
    ax.set_ylabel("Counts")

plt.show()

# %% [markdown]
# ## 7. Validation

# %%
assert abs(result.estimated_correction - TRUE_ERROR) <= 0.05
assert result.validation_loss < result.initial_theoretical_loss
assert len(result.evaluations) == LEVELS * POINTS_PER_LEVEL

print("Single-gate closed-loop calibration: PASS")

# %% [markdown]
# ## Interpretation
#
# The syndrome measurement supplies an experimental cost function without
# directly revealing the gate angle. By scanning, selecting, and narrowing the
# correction range, the loop identifies a control angle that suppresses the
# detectable error. Finite shots limit how precisely angles near the minimum
# can be distinguished; increasing shots narrows that statistical floor.
