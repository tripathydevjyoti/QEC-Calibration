"""Finite-shot SPSA on repeated parity measurements of retained data.

Each shot encodes once and measures fresh/reset ancillas for R rounds. Every
occurrence of a syndrome CNOT location shares its calibration parameter.
Full per-shot histories are recorded, but only mean per-round syndrome loss
is passed to the optimizer.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals

from .calibration import SPSAConfig
from .repetition_code import CNOTLocation, _check_n, _check_shots, cnot_locations


@dataclass(frozen=True)
class RepeatedRoundCircuit:
    n: int
    rounds: int
    circuit: QuantumCircuit
    parameters: ParameterVector
    locations: tuple[CNOTLocation, ...]

    @property
    def gate_ids(self) -> tuple[str, ...]:
        return tuple(location.gate_id for location in self.locations)

    def vector(self, angles: Mapping[str, float] | Sequence[float] | None = None) -> np.ndarray:
        if angles is None:
            values = np.zeros(len(self.parameters))
        elif isinstance(angles, Mapping):
            unknown = set(angles) - set(self.gate_ids)
            if unknown:
                raise ValueError(f"unknown gate IDs: {sorted(unknown)}")
            values = np.asarray([angles.get(gate, 0.0) for gate in self.gate_ids], dtype=float)
        else:
            values = np.asarray(angles, dtype=float)
        if values.shape != (len(self.parameters),) or not np.all(np.isfinite(values)):
            raise ValueError(f"expected {len(self.parameters)} finite angles")
        return values.copy()

    def bind(self, angles: Mapping[str, float] | Sequence[float] | None = None) -> QuantumCircuit:
        return self.circuit.assign_parameters(dict(zip(self.parameters, self.vector(angles), strict=True)))


def build_repeated_round_template(n: int = 3, *, rounds: int = 3) -> RepeatedRoundCircuit:
    """Encode |0_L> once; measure adjacent parity R times without data reset.

There is no physical X injection or active recovery in this experiment.
Encoding CNOTs occur once; the same four syndrome CNOT angles recur each round.
"""
    _check_n(n)
    _check_shots(rounds)
    locations = cnot_locations(n)
    theta = ParameterVector("theta", len(locations))
    data = QuantumRegister(n, "data")
    ancilla = QuantumRegister(n - 1, "ancilla")
    syndrome_regs = [ClassicalRegister(n - 1, f"round_{t}") for t in range(rounds)]
    data_out = ClassicalRegister(n, "data_out")
    circuit = QuantumCircuit(data, ancilla, *syndrome_regs, data_out)

    for target in range(1, n):
        circuit.cx(data[0], data[target])
        circuit.rx(theta[target - 1], data[target])
    circuit.barrier()
    for t in range(rounds):
        for check in range(n - 1):
            for side, control in enumerate((check, check + 1)):
                index = (n - 1) + 2 * check + side
                circuit.cx(data[control], ancilla[check])
                circuit.rx(theta[index], ancilla[check])
            circuit.measure(ancilla[check], syndrome_regs[t][check])
            if t < rounds - 1:
                circuit.reset(ancilla[check])
        if t < rounds - 1:
            circuit.barrier()
    circuit.measure(data, data_out)
    return RepeatedRoundCircuit(n, rounds, circuit, theta, locations)


def run_repeated_rounds(
    model: RepeatedRoundCircuit,
    angles: Mapping[str, float] | Sequence[float] | None = None,
    *,
    shots: int = 1024,
    seed: int = 42,
    simulator: AerSimulator | None = None,
    prepared_circuit: QuantumCircuit | None = None,
) -> dict[tuple[str, ...], int]:
    """Counts keyed by (round-0 syndrome, ..., round-(R-1) syndrome, final data).

    Syndrome bits are in conceptual check order. Qiskit's data word is shown
    in register display order, so X on data[1] appears as `010` for n=3.
    """
    _check_shots(shots)
    backend = simulator if simulator is not None else AerSimulator()
    template = prepared_circuit if prepared_circuit is not None else transpile(
        model.circuit, backend, optimization_level=0
    )
    circuit = template.assign_parameters(dict(zip(model.parameters, model.vector(angles), strict=True)))
    raw = backend.run(circuit, shots=shots, seed_simulator=seed).result().get_counts()
    histories: Counter[tuple[str, ...]] = Counter()
    for key, count in raw.items():
        fields = key.split()  # Qiskit displays classical registers in reverse order.
        if len(fields) != model.rounds + 1:
            raise ValueError(f"unexpected classical register layout: {key!r}")
        history = tuple(bits[::-1] for bits in reversed(fields[1:]))
        histories[(*history, fields[0])] += count
    return dict(histories)


def round_syndrome_loss(counts: Mapping[tuple[str, ...], int], model: RepeatedRoundCircuit) -> float:
    """Average fraction of nonzero syndrome rounds per shot; no history weighting."""
    total = sum(counts.values())
    if total < 1 or any(count < 0 for count in counts.values()):
        raise ValueError("counts must contain nonnegative shot counts")
    zero = "0" * (model.n - 1)
    for key in counts:
        if len(key) != model.rounds + 1 or any(len(s) != model.n - 1 for s in key[:-1]):
            raise ValueError("invalid repeated-round history")
    return sum(count * sum(s != zero for s in key[:-1]) for key, count in counts.items()) / (total * model.rounds)


@dataclass(frozen=True)
class RepeatedCalibrationResult:
    active_gate_ids: tuple[str, ...]
    initial_angles: dict[str, float]
    optimized_angles: dict[str, float]
    initial_loss: float
    final_loss: float
    initial_counts: dict[tuple[str, ...], int]
    final_counts: dict[tuple[str, ...], int]
    history: tuple[dict[str, object], ...]
    function_evaluations: int


def calibrate_repeated_rounds(
    model: RepeatedRoundCircuit,
    initial_angles: Mapping[str, float] | Sequence[float],
    *,
    active_gate_ids: Sequence[str] | None = None,
    config: SPSAConfig | None = None,
) -> RepeatedCalibrationResult:
    """Minimize only sampled mean per-round syndrome loss with Qiskit SPSA."""
    settings = config or SPSAConfig()
    initial = model.vector(initial_angles)
    active = tuple(model.gate_ids if active_gate_ids is None else active_gate_ids)
    if not active or len(set(active)) != len(active) or set(active) - set(model.gate_ids):
        raise ValueError("active gate IDs must be unique known locations")
    indices = [model.gate_ids.index(gate) for gate in active]
    backend = AerSimulator()
    prepared = transpile(model.circuit, backend, optimization_level=0)

    def expand(values: Sequence[float]) -> np.ndarray:
        full = initial.copy()
        full[indices] = values
        return full

    evaluations = 0

    def objective(values: Sequence[float]) -> float:
        nonlocal evaluations
        counts = run_repeated_rounds(
            model, expand(values), shots=settings.train_shots,
            seed=settings.seed + evaluations, simulator=backend, prepared_circuit=prepared,
        )
        evaluations += 1
        return round_syndrome_loss(counts, model)

    history: list[dict[str, object]] = []

    def callback(nfev, parameters, value, step_size, accepted):
        history.append({
            "iteration": len(history) + 1,
            "function_evaluations": int(nfev),
            "angles": tuple(map(float, expand(parameters))),
            "loss": float(value),
            "step_size": float(step_size),
            "accepted": bool(accepted),
        })

    initial_counts = run_repeated_rounds(
        model, initial, shots=settings.validation_shots,
        seed=settings.seed + 10_000, simulator=backend, prepared_circuit=prepared,
    )
    algorithm_globals.random_seed = settings.seed
    solution = SPSA(
        maxiter=settings.maxiter, learning_rate=settings.learning_rate,
        perturbation=settings.perturbation, callback=callback,
    ).minimize(fun=objective, x0=initial[indices])
    final = expand(solution.x)
    final_counts = run_repeated_rounds(
        model, final, shots=settings.validation_shots,
        seed=settings.seed + 20_000, simulator=backend, prepared_circuit=prepared,
    )
    return RepeatedCalibrationResult(
        active,
        dict(zip(model.gate_ids, map(float, initial), strict=True)),
        dict(zip(model.gate_ids, map(float, final), strict=True)),
        round_syndrome_loss(initial_counts, model), round_syndrome_loss(final_counts, model),
        initial_counts, final_counts, tuple(history), int(solution.nfev),
    )
