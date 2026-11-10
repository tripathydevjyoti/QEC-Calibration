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

The no-physical-error training loss is `L(theta)=1-P(syndrome=00...0)`.
By default each optimization uses 1,024 shots per objective evaluation and
independent 4,096-shot initial/final validation. Notebook 01 compares sampled
probabilities with `sin²(theta/2)` theory. The SPSA notebooks plot every
angle trajectory in a representative all-gate trial and summarize repeated
trials by loss and angle norm. A low loss demonstrates the chosen circuit
probe works; it need not uniquely identify every gate angle.

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
use the default settings for the full finite-shot experiments. Fixed seeds
make runs reproducible, subject to dependency-version changes.

```text
src/qec_calibration/repetition_code.py  # circuits, syndromes, sampling, recovery
src/qec_calibration/calibration.py      # SamplerV2 objective and Qiskit SPSA
tests/                                  # ideal, theory, and optimizer checks
notebooks/                              # three sequential experiments
```
