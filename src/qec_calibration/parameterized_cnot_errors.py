"""Qiskit variational CNOT-error model for the repetition-code experiment.

This is the explicit ``ParameterVector`` implementation used by the SPSA
calibration workflow.  The circuit is constructed once with six symbolic
Qiskit parameters and evaluated repeatedly with numerical parameter vectors.

Each CNOT location has a stable identifier and an independently configurable
angle.  For the first benchmark, location ``g`` implements

    faulty_CX_g(theta_g) = Rx_target(theta_g) CX_g.

The angle itself is the simulated gate-calibration parameter.  There is no
separate correction gate and no hidden error angle in this model.  Setting an
angle to zero recovers the ideal CNOT at that location.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit_aer import AerSimulator

from .ideal_bit_flip_code import (
    EXPECTED_SYNDROMES,
    _normalize_syndrome_counts,
    _parse_recovery_counts,
    _validate_inputs,
)
from .single_gate_error import SYNDROMES, counts_to_probabilities


@dataclass(frozen=True)
class CNOTLocation:
    """One addressable CNOT in the three-qubit repetition-code circuit."""

    gate_id: str
    stage: Literal["encoding", "syndrome"]
    control: str
    target: str
    control_index: int
    target_index: int


CNOT_LOCATIONS: Final[tuple[CNOTLocation, ...]] = (
    CNOTLocation("enc_01", "encoding", "data[0]", "data[1]", 0, 1),
    CNOTLocation("enc_02", "encoding", "data[0]", "data[2]", 0, 2),
    CNOTLocation("syn_0a", "syndrome", "data[0]", "ancilla[0]", 0, 3),
    CNOTLocation("syn_0b", "syndrome", "data[1]", "ancilla[0]", 1, 3),
    CNOTLocation("syn_1a", "syndrome", "data[1]", "ancilla[1]", 1, 4),
    CNOTLocation("syn_1b", "syndrome", "data[2]", "ancilla[1]", 2, 4),
)

CNOT_LOCATION_BY_ID: Final[dict[str, CNOTLocation]] = {
    location.gate_id: location for location in CNOT_LOCATIONS
}
CNOT_GATE_IDS: Final[tuple[str, ...]] = tuple(
    location.gate_id for location in CNOT_LOCATIONS
)
CNOT_PARAMETER_INDEX: Final[dict[str, int]] = {
    gate_id: index for index, gate_id in enumerate(CNOT_GATE_IDS)
}
VARIATIONAL_MODEL_VERSION: Final[str] = "qiskit-parameter-vector-v1"


def make_cnot_parameter_vector(name: str = "theta") -> ParameterVector:
    """Create the six trainable Qiskit parameters in canonical CNOT order.

    The returned vector is

    ``(theta[0], ..., theta[5])``

    with indices defined by :data:`CNOT_PARAMETER_INDEX`.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a nonempty string")
    return ParameterVector(name, len(CNOT_GATE_IDS))


@dataclass(frozen=True)
class ParameterizedCNOTCircuit:
    """A reusable symbolic circuit and its stable CNOT-parameter ordering."""

    circuit: QuantumCircuit
    parameters: ParameterVector
    parameter_index: Mapping[str, int]

    def bind(
        self,
        angles: Mapping[str, float] | None = None,
    ) -> QuantumCircuit:
        """Return a numerical copy with all six CNOT angles assigned."""
        values = angles_to_vector(angles)
        assignments = {
            parameter: float(values[index])
            for index, parameter in enumerate(self.parameters)
        }
        return self.circuit.assign_parameters(assignments, inplace=False)


def _validate_angle(angle: float, gate_id: str) -> float:
    try:
        value = float(angle)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"angle for {gate_id!r} must be a real number") from exc
    if not math.isfinite(value):
        raise ValueError(f"angle for {gate_id!r} must be finite")
    return value


def normalize_cnot_angles(
    angles: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Return a complete validated angle dictionary for all six CNOTs.

    Missing locations are ideal and therefore receive angle zero.  Unknown
    location names are rejected so a spelling error cannot silently create an
    apparently successful calibration run.
    """
    if angles is None:
        return {gate_id: 0.0 for gate_id in CNOT_GATE_IDS}

    unknown = set(angles) - set(CNOT_GATE_IDS)
    if unknown:
        raise ValueError(f"Unknown CNOT gate IDs: {sorted(unknown)}")

    return {
        gate_id: _validate_angle(angles.get(gate_id, 0.0), gate_id)
        for gate_id in CNOT_GATE_IDS
    }


def angles_to_vector(
    angles: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Convert a gate-ID mapping to the stable six-element parameter vector."""
    normalized = normalize_cnot_angles(angles)
    return np.asarray(
        [normalized[gate_id] for gate_id in CNOT_GATE_IDS],
        dtype=float,
    )


def vector_to_angles(values) -> dict[str, float]:
    """Convert a six-element parameter vector back to a gate-ID mapping."""
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or len(vector) != len(CNOT_GATE_IDS):
        raise ValueError(
            f"values must be a one-dimensional vector of length {len(CNOT_GATE_IDS)}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("all CNOT parameter values must be finite")
    return {
        gate_id: float(vector[index])
        for gate_id, index in CNOT_PARAMETER_INDEX.items()
    }


def active_cnot_angles(
    angles: Mapping[str, float] | None = None,
    *,
    atol: float = 0.0,
) -> dict[str, float]:
    """Return only locations whose absolute residual angle exceeds ``atol``."""
    if atol < 0:
        raise ValueError("atol must be nonnegative")
    normalized = normalize_cnot_angles(angles)
    return {
        gate_id: angle
        for gate_id, angle in normalized.items()
        if abs(angle) > atol
    }


def _apply_parameterized_cx(
    circuit: QuantumCircuit,
    control,
    target,
    gate_id: str,
    angles: Mapping[str, float | ParameterExpression],
) -> None:
    """Apply an ideal CNOT followed by its configured target ``Rx`` residual."""
    circuit.cx(control, target)
    angle = angles[gate_id]
    if angle != 0.0:
        circuit.rx(angle, target)


def _add_parameterized_encoding(
    circuit: QuantumCircuit,
    data: QuantumRegister,
    logical_value: int,
    angles: Mapping[str, float | ParameterExpression],
) -> None:
    if logical_value == 1:
        circuit.x(data[0])

    _apply_parameterized_cx(
        circuit,
        data[0],
        data[1],
        "enc_01",
        angles,
    )
    _apply_parameterized_cx(
        circuit,
        data[0],
        data[2],
        "enc_02",
        angles,
    )


def _add_parameterized_syndrome_extraction(
    circuit: QuantumCircuit,
    data: QuantumRegister,
    ancilla: QuantumRegister,
    syndrome_bits: ClassicalRegister,
    angles: Mapping[str, float | ParameterExpression],
) -> None:
    _apply_parameterized_cx(
        circuit,
        data[0],
        ancilla[0],
        "syn_0a",
        angles,
    )
    _apply_parameterized_cx(
        circuit,
        data[1],
        ancilla[0],
        "syn_0b",
        angles,
    )
    _apply_parameterized_cx(
        circuit,
        data[1],
        ancilla[1],
        "syn_1a",
        angles,
    )
    _apply_parameterized_cx(
        circuit,
        data[2],
        ancilla[1],
        "syn_1b",
        angles,
    )

    circuit.measure(ancilla[0], syndrome_bits[0])
    circuit.measure(ancilla[1], syndrome_bits[1])


def build_parameterized_syndrome_template(
    error_qubit: int | None = None,
    logical_value: int = 0,
) -> ParameterizedCNOTCircuit:
    """Build one reusable six-parameter syndrome-measurement circuit."""
    _validate_inputs(error_qubit, logical_value)
    parameters = make_cnot_parameter_vector()
    symbolic_angles = {
        gate_id: parameters[index]
        for gate_id, index in CNOT_PARAMETER_INDEX.items()
    }

    data = QuantumRegister(3, "data")
    ancilla = QuantumRegister(2, "ancilla")
    syndrome_bits = ClassicalRegister(2, "syndrome")
    circuit = QuantumCircuit(
        data,
        ancilla,
        syndrome_bits,
        name="variational_bit_flip_qec",
    )

    _add_parameterized_encoding(circuit, data, logical_value, symbolic_angles)
    circuit.barrier()
    if error_qubit is not None:
        circuit.x(data[error_qubit])
    circuit.barrier()
    _add_parameterized_syndrome_extraction(
        circuit,
        data,
        ancilla,
        syndrome_bits,
        symbolic_angles,
    )
    return ParameterizedCNOTCircuit(
        circuit=circuit,
        parameters=parameters,
        parameter_index=CNOT_PARAMETER_INDEX,
    )


def build_parameterized_recovery_template(
    error_qubit: int | None = None,
    logical_value: int = 0,
) -> ParameterizedCNOTCircuit:
    """Build one reusable six-parameter circuit with active recovery."""
    _validate_inputs(error_qubit, logical_value)
    parameters = make_cnot_parameter_vector()
    symbolic_angles = {
        gate_id: parameters[index]
        for gate_id, index in CNOT_PARAMETER_INDEX.items()
    }

    data = QuantumRegister(3, "data")
    ancilla = QuantumRegister(2, "ancilla")
    syndrome_bits = ClassicalRegister(2, "syndrome")
    data_bits = ClassicalRegister(3, "data_out")
    circuit = QuantumCircuit(
        data,
        ancilla,
        syndrome_bits,
        data_bits,
        name="variational_bit_flip_qec_recovery",
    )

    _add_parameterized_encoding(circuit, data, logical_value, symbolic_angles)
    circuit.barrier()
    if error_qubit is not None:
        circuit.x(data[error_qubit])
    circuit.barrier()
    _add_parameterized_syndrome_extraction(
        circuit,
        data,
        ancilla,
        syndrome_bits,
        symbolic_angles,
    )

    with circuit.if_test((syndrome_bits, 1)):
        circuit.x(data[0])
    with circuit.if_test((syndrome_bits, 3)):
        circuit.x(data[1])
    with circuit.if_test((syndrome_bits, 2)):
        circuit.x(data[2])

    circuit.barrier()
    circuit.measure(data, data_bits)
    return ParameterizedCNOTCircuit(
        circuit=circuit,
        parameters=parameters,
        parameter_index=CNOT_PARAMETER_INDEX,
    )


def build_parameterized_syndrome_circuit(
    cnot_angles: Mapping[str, float] | None = None,
    error_qubit: int | None = None,
    logical_value: int = 0,
) -> QuantumCircuit:
    """Build the syndrome circuit with independently tunable CNOT residuals."""
    _validate_inputs(error_qubit, logical_value)
    angles = normalize_cnot_angles(cnot_angles)

    data = QuantumRegister(3, "data")
    ancilla = QuantumRegister(2, "ancilla")
    syndrome_bits = ClassicalRegister(2, "syndrome")
    circuit = QuantumCircuit(
        data,
        ancilla,
        syndrome_bits,
        name="parameterized_bit_flip_qec",
    )

    _add_parameterized_encoding(circuit, data, logical_value, angles)
    circuit.barrier()
    if error_qubit is not None:
        circuit.x(data[error_qubit])
    circuit.barrier()
    _add_parameterized_syndrome_extraction(
        circuit,
        data,
        ancilla,
        syndrome_bits,
        angles,
    )
    return circuit


def build_parameterized_recovery_circuit(
    cnot_angles: Mapping[str, float] | None = None,
    error_qubit: int | None = None,
    logical_value: int = 0,
) -> QuantumCircuit:
    """Build the faulty circuit followed by standard syndrome-based recovery."""
    _validate_inputs(error_qubit, logical_value)
    angles = normalize_cnot_angles(cnot_angles)

    data = QuantumRegister(3, "data")
    ancilla = QuantumRegister(2, "ancilla")
    syndrome_bits = ClassicalRegister(2, "syndrome")
    data_bits = ClassicalRegister(3, "data_out")
    circuit = QuantumCircuit(
        data,
        ancilla,
        syndrome_bits,
        data_bits,
        name="parameterized_bit_flip_qec_recovery",
    )

    _add_parameterized_encoding(circuit, data, logical_value, angles)
    circuit.barrier()
    if error_qubit is not None:
        circuit.x(data[error_qubit])
    circuit.barrier()
    _add_parameterized_syndrome_extraction(
        circuit,
        data,
        ancilla,
        syndrome_bits,
        angles,
    )

    with circuit.if_test((syndrome_bits, 1)):
        circuit.x(data[0])
    with circuit.if_test((syndrome_bits, 3)):
        circuit.x(data[1])
    with circuit.if_test((syndrome_bits, 2)):
        circuit.x(data[2])

    circuit.barrier()
    circuit.measure(data, data_bits)
    return circuit


def _validate_shots(shots: int) -> None:
    if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
        raise ValueError("shots must be a positive integer")


def run_parameterized_syndrome(
    cnot_angles: Mapping[str, float] | None = None,
    error_qubit: int | None = None,
    logical_value: int = 0,
    shots: int = 4096,
    seed: int = 42,
) -> dict[str, int]:
    """Sample syndrome counts for the configured CNOT parameter vector."""
    _validate_shots(shots)
    simulator = AerSimulator()
    circuit = transpile(
        build_parameterized_syndrome_circuit(
            cnot_angles=cnot_angles,
            error_qubit=error_qubit,
            logical_value=logical_value,
        ),
        simulator,
        optimization_level=0,
    )
    result = simulator.run(
        circuit,
        shots=shots,
        seed_simulator=seed,
    ).result()
    return _normalize_syndrome_counts(result.get_counts())


def run_parameterized_recovery(
    cnot_angles: Mapping[str, float] | None = None,
    error_qubit: int | None = None,
    logical_value: int = 0,
    shots: int = 4096,
    seed: int = 42,
) -> dict[tuple[str, str], int]:
    """Sample joint data-output and syndrome counts after active recovery."""
    _validate_shots(shots)
    simulator = AerSimulator()
    circuit = transpile(
        build_parameterized_recovery_circuit(
            cnot_angles=cnot_angles,
            error_qubit=error_qubit,
            logical_value=logical_value,
        ),
        simulator,
        optimization_level=0,
    )
    result = simulator.run(
        circuit,
        shots=shots,
        seed_simulator=seed,
    ).result()
    return _parse_recovery_counts(result.get_counts())


def expected_syndrome_loss(
    counts: Mapping[str, int],
    error_qubit: int | None = None,
) -> float:
    """Return ``1 - P(expected syndrome)`` for a known probe error."""
    if error_qubit not in EXPECTED_SYNDROMES:
        raise ValueError("error_qubit must be None, 0, 1, or 2")
    probabilities = counts_to_probabilities(counts)
    return 1.0 - probabilities[EXPECTED_SYNDROMES[error_qubit]]


def syndrome_probability_vector(
    counts: Mapping[str, int],
) -> tuple[float, float, float, float]:
    """Return probabilities in the fixed ``(00, 10, 11, 01)`` order."""
    probabilities = counts_to_probabilities(counts)
    return tuple(probabilities[syndrome] for syndrome in SYNDROMES)


def logical_success_probability(
    counts: Mapping[tuple[str, str], int],
    logical_value: int = 0,
) -> float:
    """Return the probability of the exact recovered logical codeword.

    Qiskit displays the three measured data bits as ``q2q1q0``.  The two
    repetition-code basis words are invariant under that reversal, so the
    expected strings are simply ``000`` and ``111``.
    """
    if logical_value not in (0, 1):
        raise ValueError("logical_value must be 0 or 1")
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("counts must contain at least one shot")
    if any(count < 0 for count in counts.values()):
        raise ValueError("counts cannot be negative")

    expected_data = "000" if logical_value == 0 else "111"
    successful = sum(
        count for (data_out, _syndrome), count in counts.items()
        if data_out == expected_data
    )
    return successful / total


def theoretical_enc01_probabilities(theta: float) -> dict[str, float]:
    """Return the exact no-probe distribution for only ``enc_01`` faulty."""
    theta = _validate_angle(theta, "enc_01")
    p_error = math.sin(theta / 2.0) ** 2
    return {
        "00": 1.0 - p_error,
        "10": 0.0,
        "11": p_error,
        "01": 0.0,
    }

