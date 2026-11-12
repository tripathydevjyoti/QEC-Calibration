# Syndrome-guided calibration of repetition-code circuits

This compact Qiskit project uses finite-shot stabilizer outcomes to guide
calibration of imperfect CNOT gates. It studies the three-qubit and five-qubit
**bit-flip repetition codes**, not the [[5,1,3]] perfect quantum code.
Repetition codes protect against bit flips, not arbitrary quantum errors.

At each CNOT location, the simulated operation is `CX` followed by
`Rx(theta_g)` on its target. The ideal simulated control is `theta_g=0`.
These angles are directly adjustable simulation coordinates: on real
hardware one would instead vary pulse controls with an unknown relationship
to gate error.

| Notebook | Main question |
|---|---|
| `01_three_bit_single_fault.ipynb` | Does a finite-shot single-fault syndrome sweep match theory? |
| `02_three_bit_spsa_calibration.ipynb` | Can Qiskit SPSA calibrate one through six simultaneous CNOT parameters? |
| `03_five_bit_spsa_calibration.ipynb` | How does the same experiment behave with 12 CNOT parameters? |
| `04_repeated_round_syndromes.ipynb` | How are repeated stabilizer measurements stored while data is retained? |
| `05_repeated_round_spsa_scaling.ipynb` | How do one through six active CNOT angles calibrate with one, three, or five rounds per shot? |

The no-physical-error single-round training loss is `L(theta)=1-P(syndrome=00...0)`.
By default each optimization uses 1,024 shots per objective evaluation and
independent 4,096-shot initial/final validation. Notebook 01 compares sampled
probabilities with `sin²(theta/2)` theory. The SPSA notebooks plot every
angle trajectory in a representative all-gate trial and summarize repeated
trials by loss and angle norm. A low loss demonstrates the chosen circuit
probe works; it need not uniquely identify every gate angle.

Notebook 05 retains the encoded data through repeated syndrome extraction,
including mid-circuit measurement and ancilla reset. Each shot returns its
entire syndrome history and final data word. The optimizer uses **only** the
average fraction of nonzero syndrome rounds, not a history-aware objective.
Full-mode runs use 100 SPSA iterations, 1,024 training shots per evaluation,
and independent 4,096-shot initial/final validation, with three seeded trials
per configuration. It compares R=1 and R=3 for one through six active
parameters and R=5 for sizes 3, 4, and 6 (45 runs total). The notebook
defaults to a shorter `FAST_MODE=True` and writes full validation histories,
angles, and optimizer traces into compressed `notebooks/results/repeated_scaling_*.json.gz`
when executed. The older exploratory two-gate repeated-round run is not part
of this experiment.

The three-qubit circuit has two checks, six CNOTs, and corrects one X error.
The five-qubit circuit has four checks, 12 CNOTs, and corrects up to two X
errors. Basis-state logical recovery is tested separately from the
syndrome-based training objective.

## Install and run

With Python 3.10+:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
jupyter lab
```

Open the notebooks in numerical order. Each imports its tested implementation
from `src/qec_calibration`. Use `FAST_MODE=True` for quick exploratory runs;
use `FAST_MODE=False` for the full finite-shot experiments (notebook 05
defaults to a quick run). Fixed seeds
make runs reproducible, subject to dependency-version changes.

```text
src/qec_calibration/repetition_code.py  # circuits, syndromes, sampling, recovery
src/qec_calibration/calibration.py      # SamplerV2 objective and Qiskit SPSA
src/qec_calibration/repeated_rounds.py  # repeated measurements, histories, SPSA
tests/                                  # ideal, theory, and optimizer checks
notebooks/                              # sequential experiments
```
