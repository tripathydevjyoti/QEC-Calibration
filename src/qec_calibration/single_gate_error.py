"""A coherent error on one encoding CNOT in the three-qubit bit-flip code.

The imperfect operation is

    faulty_CX(0, 1) = Rx_1(epsilon) CX(0, 1),

so the residual rotation acts on the target immediately after the first
encoding CNOT.  Every other instruction remains ideal.  Aer applies this
two-qubit coherent ``QuantumError`` only to a ``cx`` whose ordered physical
qubits are ``[0, 1]``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Final

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, coherent_unitary_error

from .ideal_bit_flip_code import (
    _normalize_syndrome_counts,
    build_syndrome_circuit,
)


SYNDROMES: Final[tuple[str, ...]] = ("00", "10", "11", "01")


def _validate_epsilon(epsilon: float) -> float:
    value = float(epsilon)
    if not math.isfinite(value):
        raise ValueError("epsilon must be finite")
    return value


def _validate_shots(shots: int) -> None:
    if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
        raise ValueError("shots must be a positive integer")


def build_faulty_cx_noise_model(epsilon: float) -> NoiseModel:
    """Return a noise model for an ``Rx(epsilon)`` residual on ``cx(0, 1)``.

    A two-qubit error circuit is used because Aer requires the error dimension
    to match the noisy two-qubit instruction.  Local error-circuit qubit 1
    maps to the target of the ordered physical-qubit pair ``[0, 1]``.
    """
    epsilon = _validate_epsilon(epsilon)

    residual = QuantumCircuit(2, name="cx_01_residual")
    residual.rx(epsilon, 1)
    error = coherent_unitary_error(Operator(residual).data)

    noise_model = NoiseModel()
    noise_model.add_quantum_error(error, ["cx"], [0, 1])
    return noise_model


def theoretical_syndrome_probabilities(epsilon: float) -> dict[str, float]:
    """Return exact syndrome probabilities in conceptual ``s0s1`` order.

    After encoding, the residual rotation gives

    ``cos(epsilon/2)|psi_L> - i sin(epsilon/2) X1|psi_L>``.

    Syndrome measurement separates those orthogonal branches into ``00`` and
    ``11`` respectively.
    """
    epsilon = _validate_epsilon(epsilon)
    p_11 = math.sin(epsilon / 2.0) ** 2
    return {
        "00": 1.0 - p_11,
        "10": 0.0,
        "11": p_11,
        "01": 0.0,
    }


def counts_to_probabilities(counts: Mapping[str, int]) -> dict[str, float]:
    """Convert syndrome counts to a complete four-outcome distribution."""
    unknown = set(counts) - set(SYNDROMES)
    if unknown:
        raise ValueError(f"Unknown syndrome keys: {sorted(unknown)}")

    total = sum(counts.values())
    if total <= 0:
        raise ValueError("counts must contain at least one shot")
    if any(count < 0 for count in counts.values()):
        raise ValueError("counts cannot be negative")

    return {syndrome: counts.get(syndrome, 0) / total for syndrome in SYNDROMES}


def run_faulty_cx_syndrome(
    epsilon: float,
    logical_value: int = 0,
    shots: int = 20_000,
    seed: int = 42,
) -> dict[str, int]:
    """Simulate the faulty encoding CNOT and return conceptual syndrome counts."""
    epsilon = _validate_epsilon(epsilon)
    _validate_shots(shots)

    noise_model = build_faulty_cx_noise_model(epsilon)
    simulator = AerSimulator(noise_model=noise_model)
    circuit = build_syndrome_circuit(
        error_qubit=None,
        logical_value=logical_value,
    )
    compiled = transpile(circuit, simulator, optimization_level=0)
    result = simulator.run(
        compiled,
        shots=shots,
        seed_simulator=seed,
    ).result()
    return _normalize_syndrome_counts(result.get_counts())


def run_error_sweep(
    epsilon_values: Iterable[float],
    logical_value: int = 0,
    shots: int = 5_000,
    seed: int = 42,
) -> list[dict[str, float | int]]:
    """Measure syndrome rates over a sequence of coherent error angles."""
    _validate_shots(shots)
    rows: list[dict[str, float | int]] = []

    for index, epsilon in enumerate(epsilon_values):
        epsilon = _validate_epsilon(epsilon)
        counts = run_faulty_cx_syndrome(
            epsilon=epsilon,
            logical_value=logical_value,
            shots=shots,
            seed=seed + index,
        )
        observed = counts_to_probabilities(counts)
        theory = theoretical_syndrome_probabilities(epsilon)
        rows.append(
            {
                "epsilon": epsilon,
                "shots": shots,
                "count_00": counts.get("00", 0),
                "count_11": counts.get("11", 0),
                "p00_empirical": observed["00"],
                "p11_empirical": observed["11"],
                "p00_theory": theory["00"],
                "p11_theory": theory["11"],
                "absolute_error_p11": abs(observed["11"] - theory["11"]),
            }
        )

    return rows
