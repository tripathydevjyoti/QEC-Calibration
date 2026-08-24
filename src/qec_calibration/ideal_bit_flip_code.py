"""Ideal three-qubit bit-flip code implemented with Qiskit Aer.

The code encodes one logical basis state in three data qubits, optionally
inserts one physical X error, extracts the two parity syndromes with ancillas,
and can apply classically conditioned recovery.

Conceptual syndrome convention
------------------------------

    s0 = q0 XOR q1
    s1 = q1 XOR q2

Public syndrome strings are returned in ``s0s1`` order. Qiskit displays a
two-bit classical register in ``bit1 bit0`` order, so raw count keys are
reversed before being returned by :func:`run_syndrome`.
"""

from __future__ import annotations

from collections import Counter
from typing import Final

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator


EXPECTED_SYNDROMES: Final[dict[int | None, str]] = {
    None: "00",
    0: "10",
    1: "11",
    2: "01",
}


def _validate_inputs(error_qubit: int | None, logical_value: int) -> None:
    if error_qubit not in EXPECTED_SYNDROMES:
        raise ValueError("error_qubit must be None, 0, 1, or 2")
    if logical_value not in (0, 1):
        raise ValueError("logical_value must be 0 or 1")


def _add_encoding(
    circuit: QuantumCircuit,
    data: QuantumRegister,
    logical_value: int,
) -> None:
    """Prepare |0_L> or |1_L> on the three data qubits."""
    if logical_value == 1:
        circuit.x(data[0])

    circuit.cx(data[0], data[1])
    circuit.cx(data[0], data[2])


def _add_physical_error(
    circuit: QuantumCircuit,
    data: QuantumRegister,
    error_qubit: int | None,
) -> None:
    if error_qubit is not None:
        circuit.x(data[error_qubit])


def _add_syndrome_extraction(
    circuit: QuantumCircuit,
    data: QuantumRegister,
    ancilla: QuantumRegister,
    syndrome_bits: ClassicalRegister,
) -> None:
    """Measure Z0Z1 and Z1Z2 parity without measuring the data qubits."""
    circuit.cx(data[0], ancilla[0])
    circuit.cx(data[1], ancilla[0])
    circuit.cx(data[1], ancilla[1])
    circuit.cx(data[2], ancilla[1])

    # Natural storage: classical bit 0 contains s0 and bit 1 contains s1.
    circuit.measure(ancilla[0], syndrome_bits[0])
    circuit.measure(ancilla[1], syndrome_bits[1])


def build_syndrome_circuit(
    error_qubit: int | None = None,
    logical_value: int = 0,
) -> QuantumCircuit:
    """Build an ideal circuit that encodes and measures the error syndrome."""
    _validate_inputs(error_qubit, logical_value)

    data = QuantumRegister(3, "data")
    ancilla = QuantumRegister(2, "ancilla")
    syndrome_bits = ClassicalRegister(2, "syndrome")
    circuit = QuantumCircuit(data, ancilla, syndrome_bits, name="bit_flip_qec")

    _add_encoding(circuit, data, logical_value)
    circuit.barrier()
    _add_physical_error(circuit, data, error_qubit)
    circuit.barrier()
    _add_syndrome_extraction(circuit, data, ancilla, syndrome_bits)
    return circuit


def build_recovery_circuit(
    error_qubit: int | None = None,
    logical_value: int = 0,
) -> QuantumCircuit:
    """Build the ideal code with syndrome-conditioned active recovery.

    The syndrome register stores the integer ``s0 + 2*s1``. Therefore its
    values 1, 3, and 2 correspond to physical errors on q0, q1, and q2.
    """
    _validate_inputs(error_qubit, logical_value)

    data = QuantumRegister(3, "data")
    ancilla = QuantumRegister(2, "ancilla")
    syndrome_bits = ClassicalRegister(2, "syndrome")
    data_bits = ClassicalRegister(3, "data_out")
    circuit = QuantumCircuit(
        data,
        ancilla,
        syndrome_bits,
        data_bits,
        name="bit_flip_qec_recovery",
    )

    _add_encoding(circuit, data, logical_value)
    circuit.barrier()
    _add_physical_error(circuit, data, error_qubit)
    circuit.barrier()
    _add_syndrome_extraction(circuit, data, ancilla, syndrome_bits)

    # Qiskit dynamic-circuit recovery based on the measured syndrome.
    with circuit.if_test((syndrome_bits, 1)):
        circuit.x(data[0])
    with circuit.if_test((syndrome_bits, 3)):
        circuit.x(data[1])
    with circuit.if_test((syndrome_bits, 2)):
        circuit.x(data[2])

    circuit.barrier()
    circuit.measure(data, data_bits)
    return circuit


def _normalize_syndrome_counts(raw_counts: dict[str, int]) -> dict[str, int]:
    """Convert Qiskit's raw ``s1s0`` display into conceptual ``s0s1``."""
    normalized: Counter[str] = Counter()
    for raw_key, count in raw_counts.items():
        compact = raw_key.replace(" ", "")
        if len(compact) != 2:
            raise ValueError(f"Unexpected syndrome count key: {raw_key!r}")
        normalized[compact[::-1]] += count
    return dict(normalized)


def run_syndrome(
    error_qubit: int | None = None,
    logical_value: int = 0,
    shots: int = 1024,
    seed: int = 42,
) -> dict[str, int]:
    """Run the syndrome circuit and return counts in conceptual ``s0s1`` order."""
    if shots <= 0:
        raise ValueError("shots must be positive")

    simulator = AerSimulator()
    circuit = transpile(
        build_syndrome_circuit(error_qubit, logical_value),
        simulator,
        optimization_level=0,
    )
    result = simulator.run(circuit, shots=shots, seed_simulator=seed).result()
    return _normalize_syndrome_counts(result.get_counts())


def _parse_recovery_counts(raw_counts: dict[str, int]) -> dict[tuple[str, str], int]:
    """Return joint ``(data_out, conceptual_syndrome)`` counts."""
    parsed: Counter[tuple[str, str]] = Counter()
    for raw_key, count in raw_counts.items():
        parts = raw_key.split()
        if len(parts) != 2:
            raise ValueError(f"Unexpected recovery count key: {raw_key!r}")

        # Classical registers are displayed in reverse register-creation order:
        # ``data_out syndrome``. Syndrome itself is displayed as ``s1s0``.
        data_out, raw_syndrome = parts
        parsed[(data_out, raw_syndrome[::-1])] += count
    return dict(parsed)


def run_recovery(
    error_qubit: int | None = None,
    logical_value: int = 0,
    shots: int = 1024,
    seed: int = 42,
) -> dict[tuple[str, str], int]:
    """Run active recovery and return joint data/syndrome counts."""
    if shots <= 0:
        raise ValueError("shots must be positive")

    simulator = AerSimulator()
    circuit = transpile(
        build_recovery_circuit(error_qubit, logical_value),
        simulator,
        optimization_level=0,
    )
    result = simulator.run(circuit, shots=shots, seed_simulator=seed).result()
    return _parse_recovery_counts(result.get_counts())

