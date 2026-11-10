"""Parameterized odd-length bit-flip repetition codes (n=3 or n=5).

Every physical CNOT is followed by a coherent Rx(theta) on its target.
The theta coordinates are directly adjustable simulation parameters; zero
recovers the ideal circuit. Syndromes are returned in stabilizer order s0... .
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2


def _check_n(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n not in (3, 5):
        raise ValueError("this experiment supports n=3 or n=5")


def _check_shots(shots: int) -> None:
    if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
        raise ValueError("shots must be a positive integer")


@dataclass(frozen=True)
class CNOTLocation:
    gate_id: str
    stage: Literal["encoding", "syndrome"]
    control: str
    target: str
    control_index: int
    target_index: int


def cnot_locations(n: int) -> tuple[CNOTLocation, ...]:
    """Canonical execution/parameter order: encode, then parity checks."""
    _check_n(n)
    locations = [
        CNOTLocation(f"enc_0_{target}", "encoding", "data[0]", f"data[{target}]", 0, target)
        for target in range(1, n)
    ]
    for check in range(n - 1):
        locations.extend(
            (
                CNOTLocation(f"syn_{check}_left", "syndrome", f"data[{check}]", f"ancilla[{check}]", check, n + check),
                CNOTLocation(f"syn_{check}_right", "syndrome", f"data[{check + 1}]", f"ancilla[{check}]", check + 1, n + check),
            )
        )
    return tuple(locations)


def syndrome_for_errors(n: int, errors: Sequence[int] = ()) -> str:
    """Adjacent-parity syndrome after an X on each listed data qubit."""
    _check_n(n)
    if any(isinstance(q, bool) or not isinstance(q, int) or not 0 <= q < n for q in errors):
        raise ValueError("physical error indices must be data qubits in range(n)")
    bits = [0] * n
    for q in errors:
        bits[q] ^= 1
    return "".join(str(bits[i] ^ bits[i + 1]) for i in range(n - 1))


@dataclass(frozen=True)
class ParameterizedCircuit:
    n: int
    circuit: QuantumCircuit
    parameters: ParameterVector
    locations: tuple[CNOTLocation, ...]
    physical_errors: tuple[int, ...]

    @property
    def gate_ids(self) -> tuple[str, ...]:
        return tuple(location.gate_id for location in self.locations)

    def vector(self, angles: Mapping[str, float] | Sequence[float] | None = None) -> np.ndarray:
        """Complete numeric vector; omitted named angles default to zero."""
        if angles is None:
            values = np.zeros(len(self.parameters))
        elif isinstance(angles, Mapping):
            unknown = set(angles) - set(self.gate_ids)
            if unknown:
                raise ValueError(f"unknown gate IDs: {sorted(unknown)}")
            values = np.asarray([angles.get(g, 0.0) for g in self.gate_ids], dtype=float)
        else:
            values = np.asarray(angles, dtype=float)
        if values.ndim != 1 or len(values) != len(self.parameters) or not np.all(np.isfinite(values)):
            raise ValueError(f"angles must contain {len(self.parameters)} finite values")
        return values.copy()

    def named(self, angles: Mapping[str, float] | Sequence[float]) -> dict[str, float]:
        return dict(zip(self.gate_ids, map(float, self.vector(angles)), strict=True))

    def bind(self, angles: Mapping[str, float] | Sequence[float] | None = None) -> QuantumCircuit:
        values = self.vector(angles)
        return self.circuit.assign_parameters(dict(zip(self.parameters, values, strict=True)), inplace=False)

    def sampler_values(self, angles: Mapping[str, float] | Sequence[float]) -> np.ndarray:
        """Match Qiskit's circuit parameter iteration order, including theta[10]."""
        vector = self.vector(angles)
        position = {parameter: i for i, parameter in enumerate(self.parameters)}
        return np.asarray([vector[position[p]] for p in self.circuit.parameters])


def build_syndrome_template(
    n: int,
    *,
    logical_value: int = 0,
    physical_errors: Sequence[int] = (),
    recovery: bool = False,
) -> ParameterizedCircuit:
    """Build one symbolic circuit for syndrome sampling or basis-state recovery."""
    _check_n(n)
    if logical_value not in (0, 1):
        raise ValueError("logical_value must be 0 or 1")
    errors = tuple(physical_errors)
    syndrome_for_errors(n, errors)  # also validates indices
    locations = cnot_locations(n)
    parameters = ParameterVector("theta", len(locations))
    data = QuantumRegister(n, "data")
    ancilla = QuantumRegister(n - 1, "ancilla")
    syndrome = ClassicalRegister(n - 1, "syndrome")
    if recovery:
        data_out = ClassicalRegister(n, "data_out")
        circuit = QuantumCircuit(data, ancilla, syndrome, data_out)
    else:
        circuit = QuantumCircuit(data, ancilla, syndrome)

    def faulty_cx(control, target, index: int) -> None:
        circuit.cx(control, target)
        circuit.rx(parameters[index], target)

    if logical_value:
        circuit.x(data[0])
    for target in range(1, n):
        faulty_cx(data[0], data[target], target - 1)
    circuit.barrier()
    for q in errors:
        circuit.x(data[q])
    circuit.barrier()
    for check in range(n - 1):
        index = (n - 1) + 2 * check
        faulty_cx(data[check], ancilla[check], index)
        faulty_cx(data[check + 1], ancilla[check], index + 1)
        circuit.measure(ancilla[check], syndrome[check])

    if recovery:
        # Unique lowest-weight X pattern for every syndrome when n=3 or n=5.
        for weight in range(1, (n - 1) // 2 + 1):
            for pattern in combinations(range(n), weight):
                bits = syndrome_for_errors(n, pattern)
                register_value = sum(int(bit) << i for i, bit in enumerate(bits))
                with circuit.if_test((syndrome, register_value)):
                    for q in pattern:
                        circuit.x(data[q])
        circuit.measure(data, data_out)
    # Aer SamplerV2 requires an Aer-compatible symbolic circuit before it can
    # bind parameter values (not just an untranspiled QuantumCircuit).
    if not recovery:
        circuit = transpile(circuit, AerSimulator(), optimization_level=0)
    return ParameterizedCircuit(n, circuit, parameters, locations, errors)


def normalize_syndrome_counts(raw_counts: Mapping[str, int], n: int) -> dict[str, int]:
    """Convert Qiskit's s[n-2]...s0 display to conceptual s0...s[n-2]."""
    _check_n(n)
    result: Counter[str] = Counter()
    for key, count in raw_counts.items():
        if len(key) != n - 1 or set(key) - {"0", "1"}:
            raise ValueError(f"invalid raw syndrome: {key!r}")
        result[key[::-1]] += count
    return dict(result)


def counts_to_probabilities(counts: Mapping[str, int], n: int) -> dict[str, float]:
    _check_n(n)
    if any(len(s) != n - 1 or set(s) - {"0", "1"} for s in counts):
        raise ValueError("invalid syndrome key")
    if any(isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in counts.values()):
        raise ValueError("counts must be nonnegative integers")
    total = sum(counts.values())
    if total < 1:
        raise ValueError("counts must contain shots")
    outcomes = (format(i, f"0{n - 1}b") for i in range(2 ** (n - 1)))
    return {outcome: counts.get(outcome, 0) / total for outcome in outcomes}


def syndrome_loss(counts: Mapping[str, int], model: ParameterizedCircuit) -> float:
    expected = syndrome_for_errors(model.n, model.physical_errors)
    return 1.0 - counts_to_probabilities(counts, model.n)[expected]


def run_syndrome(
    model: ParameterizedCircuit,
    angles: Mapping[str, float] | Sequence[float] | None = None,
    *,
    shots: int = 1024,
    seed: int = 42,
    sampler: SamplerV2 | None = None,
) -> dict[str, int]:
    _check_shots(shots)
    sampler = sampler or SamplerV2(default_shots=shots, seed=seed)
    result = sampler.run([(model.circuit, model.sampler_values(angles))], shots=shots).result()[0]
    return normalize_syndrome_counts(result.data.syndrome.get_counts(), model.n)


def run_recovery(
    model: ParameterizedCircuit,
    angles: Mapping[str, float] | Sequence[float] | None = None,
    *,
    shots: int = 1024,
    seed: int = 42,
) -> dict[tuple[str, str], int]:
    """Run a recovery template; return (data bits, conceptual syndrome) counts."""
    _check_shots(shots)
    if not any(reg.name == "data_out" for reg in model.circuit.cregs):
        raise ValueError("model must be built with recovery=True")
    simulator = AerSimulator()
    circuit = transpile(model.bind(angles), simulator, optimization_level=0)
    raw = simulator.run(circuit, shots=shots, seed_simulator=seed).result().get_counts()
    parsed: Counter[tuple[str, str]] = Counter()
    for key, count in raw.items():
        data_out, syndrome = key.split()
        parsed[(data_out, syndrome[::-1])] += count
    return dict(parsed)


def logical_success_probability(counts: Mapping[tuple[str, str], int], logical_value: int) -> float:
    if logical_value not in (0, 1):
        raise ValueError("logical_value must be 0 or 1")
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("counts must contain shots")
    expected = str(logical_value) * len(next(iter(counts))[0])
    return sum(c for (word, _), c in counts.items() if word == expected) / total


def theoretical_single_fault_probabilities(n: int, epsilon: float) -> dict[str, float]:
    """Only enc_0_1 faulty, no physical X: P(0...0)=cos², P(X1)=sin²."""
    _check_n(n)
    angle = float(epsilon)
    if not math.isfinite(angle):
        raise ValueError("epsilon must be finite")
    p = math.sin(angle / 2) ** 2
    zero = "0" * (n - 1)
    signature = syndrome_for_errors(n, (1,))
    distribution = {format(i, f"0{n - 1}b"): 0.0 for i in range(2 ** (n - 1))}
    distribution[zero] = 1.0 - p
    distribution[signature] = p
    return distribution
