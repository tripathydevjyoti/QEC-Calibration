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