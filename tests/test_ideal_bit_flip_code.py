"""Verification tests for the ideal three-qubit bit-flip code."""

import pytest

from qec_calibration import EXPECTED_SYNDROMES, run_recovery, run_syndrome


@pytest.mark.parametrize("error_qubit, expected", EXPECTED_SYNDROMES.items())
@pytest.mark.parametrize("logical_value", [0, 1])
def test_syndrome_truth_table(error_qubit, expected, logical_value):
    shots = 128
    assert run_syndrome(error_qubit, logical_value, shots=shots) == {expected: shots}


@pytest.mark.parametrize("error_qubit, expected", EXPECTED_SYNDROMES.items())
@pytest.mark.parametrize("logical_value", [0, 1])
def test_active_recovery_restores_logical_basis_state(
    error_qubit,
    expected,
    logical_value,
):
    shots = 128
    expected_data = "000" if logical_value == 0 else "111"
    assert run_recovery(error_qubit, logical_value, shots=shots) == {
        (expected_data, expected): shots
    }


@pytest.mark.parametrize("invalid_error", [-1, 3, 9])
def test_invalid_error_qubit_rejected(invalid_error):
    with pytest.raises(ValueError, match="error_qubit"):
        run_syndrome(error_qubit=invalid_error)


def test_invalid_logical_value_rejected():
    with pytest.raises(ValueError, match="logical_value"):
        run_syndrome(logical_value=2)

