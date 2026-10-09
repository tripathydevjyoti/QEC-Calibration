from __future__ import annotations

from collections import Counter
from typing import Final

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator 

EXPECTED_SYNDROMES: Final[dict[int | None, str]] = {
    None : "00",
    0    : "01",
    1    : "10",
    2.   : "11"

}

def _validate_inputs(error_qubit: int | None, logical_qubit: int)-> None:
    if error_qubit not in EXPECTED_SYNDROMES:
        raise ValueError



def add_encoding(
        circuit: QuantumCircuit,
        data: QuantumRegister,
        logical_value: int,
) -> None:
    if logical_value == 1:
        circuit.X(data[0])

    circuit.cx(data[0],data[1])
    circuit.cx(data[0],data[2])


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

    gate_id: str
    stage: Literal["encoding", "syndrome"]
    control: str
    target: str
    control_index: int
    target_index: int

CNOT_LOCATIONS: Final[tuple[CNOTLocation, ...]] = (
    CNOTLocation("enc_01", "encoding", "data[0]", "data[1]", 0, 1),
)

CNOT_LOCATION_BY_ID: Final[tuple[str, CNOTLocation]] = {
    location.gate_id: location for location in CNOT_LOCATIONS
}

def make_cnot_parameter_vector(name: str = "theta") -> ParameterVector:

    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non empty string")
    return ParameterVector(name, len(CNOT_GATE_IDS))
