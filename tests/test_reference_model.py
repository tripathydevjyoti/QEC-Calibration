"""Dependency-free reference checks for the repetition-code truth table.

These tests validate the intended parity and majority-recovery logic without
depending on Qiskit. They are useful as an independent oracle for the circuit
implementation.
"""

from itertools import product


def syndrome(bits: tuple[int, int, int]) -> tuple[int, int]:
    q0, q1, q2 = bits
    return q0 ^ q1, q1 ^ q2


def correction_for(s: tuple[int, int]) -> int | None:
    return {
        (0, 0): None,
        (1, 0): 0,
        (1, 1): 1,
        (0, 1): 2,
    }[s]


def flip(bits: tuple[int, int, int], qubit: int | None) -> tuple[int, int, int]:
    values = list(bits)
    if qubit is not None:
        values[qubit] ^= 1
    return tuple(values)


def test_all_single_bit_flips_are_corrected_for_both_logical_values():
    for logical_value, error_qubit in product((0, 1), (None, 0, 1, 2)):
        encoded = (logical_value,) * 3
        corrupted = flip(encoded, error_qubit)
        recovered = flip(corrupted, correction_for(syndrome(corrupted)))
        assert recovered == encoded

