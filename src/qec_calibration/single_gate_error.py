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

# Conceptual syndrome s0s1 for each single-qubit X error in the convention
# used by ``ideal_bit_flip_code``.  ``None`` denotes the no-error sector.
ERROR_SYNDROME: Final[dict[int | None, str]] = {
    None: "00",
    0: "10",
    1: "11",
    2: "01",
}


def _validate_epsilon(epsilon: float) -> float:
    value = float(epsilon)
    if not math.isfinite(value):
        raise ValueError("epsilon must be finite")
    return value


def _validate_shots(shots: int) -> None:
    if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
        raise ValueError("shots must be a positive integer")


def _validate_error_qubit(error_qubit: int | None) -> int | None:
    if error_qubit is not None and (
        not isinstance(error_qubit, int)
        or isinstance(error_qubit, bool)
        or error_qubit not in (0, 1, 2)
    ):
        raise ValueError("error_qubit must be None, 0, 1, or 2")
    return error_qubit


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


def theoretical_syndrome_probabilities(
    epsilon: float,
    error_qubit: int | None = None,
) -> dict[str, float]:
    """Return exact syndrome probabilities in conceptual ``s0s1`` order.

    After faulty encoding and an extra physical error ``X_j``, the residual rotation
    gives

    ``cos(epsilon/2) X_j|psi_L> - i sin(epsilon/2) X1 X_j|psi_L>``.

    The first branch has the syndrome of the requested physical error.  The
    second branch has that syndrome XORed with the syndrome of ``X1``.  For
    ``error_qubit=1``, these are ``11`` and ``00`` respectively.
    """
    epsilon = _validate_epsilon(epsilon)
    error_qubit = _validate_error_qubit(error_qubit)

    main_syndrome = ERROR_SYNDROME[error_qubit]
    residual_syndrome = format(
        int(main_syndrome, 2) ^ int(ERROR_SYNDROME[1], 2),
        "02b",
    )
    p_residual = math.sin(epsilon / 2.0) ** 2

    probabilities = {syndrome: 0.0 for syndrome in SYNDROMES}
    probabilities[main_syndrome] = 1.0 - p_residual
    probabilities[residual_syndrome] = p_residual
    return probabilities


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
    error_qubit: int | None = None,
) -> dict[str, int]:
    """Simulate the faulty encoding CNOT and return conceptual syndrome counts.

    ``error_qubit``  injects an ideal physical ``X`` after encoding
    and before syndrome extraction.  The default ``None`` preserves the
    original coherent gate error only experiment.
    """
    epsilon = _validate_epsilon(epsilon)
    _validate_shots(shots)
    error_qubit = _validate_error_qubit(error_qubit)

    noise_model = build_faulty_cx_noise_model(epsilon)
    simulator = AerSimulator(noise_model=noise_model)
    circuit = build_syndrome_circuit(
        error_qubit=error_qubit,
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
    shots: int = 1024,
    seed: int = 42,
    error_qubit: int | None = None,
) -> list[dict[str, float | int | str]]:
    """Measure syndrome rates over a sequence of coherent error angles."""
    _validate_shots(shots)
    error_qubit = _validate_error_qubit(error_qubit)
    rows: list[dict[str, float | int | str]] = []

    for index, epsilon in enumerate(epsilon_values):
        epsilon = _validate_epsilon(epsilon)
        counts = run_faulty_cx_syndrome(
            epsilon=epsilon,
            logical_value=logical_value,
            shots=shots,
            seed=seed + index,
            error_qubit=error_qubit,
        )
        observed = counts_to_probabilities(counts)
        theory = theoretical_syndrome_probabilities(
            epsilon,
            error_qubit=error_qubit,
        )
        expected_syndrome = ERROR_SYNDROME[error_qubit]
        rows.append(
            {
                "epsilon": epsilon,
                "shots": shots,
                "count_00": counts.get("00", 0),
                "count_10": counts.get("10", 0),
                "count_11": counts.get("11", 0),
                "count_01": counts.get("01", 0),
                "p00_empirical": observed["00"],
                "p10_empirical": observed["10"],
                "p11_empirical": observed["11"],
                "p01_empirical": observed["01"],
                "p00_theory": theory["00"],
                "p10_theory": theory["10"],
                "p11_theory": theory["11"],
                "p01_theory": theory["01"],
                "absolute_error_p11": abs(observed["11"] - theory["11"]),
                "expected_syndrome": expected_syndrome,
                "p_expected_empirical": observed[expected_syndrome],
                "p_expected_theory": theory[expected_syndrome],
                "calibration_signal_empirical": 1.0 - observed[expected_syndrome],
                "calibration_signal_theory": 1.0 - theory[expected_syndrome],
            }
        )

    return rows
