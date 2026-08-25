"""Aer tests for one coherent encoding-CNOT error."""

import math

import pytest

from qec_calibration import (
    counts_to_probabilities,
    run_faulty_cx_syndrome,
    theoretical_syndrome_probabilities,
)


@pytest.mark.parametrize("epsilon", [0.0, 0.2, 0.8, math.pi])
def test_theoretical_distribution_is_normalized(epsilon):
    probabilities = theoretical_syndrome_probabilities(epsilon)

    assert set(probabilities) == {"00", "10", "11", "01"}
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities["10"] == 0.0
    assert probabilities["01"] == 0.0


def test_zero_error_has_only_the_no_error_syndrome():
    shots = 256
    assert run_faulty_cx_syndrome(0.0, shots=shots) == {"00": shots}


@pytest.mark.parametrize("logical_value", [0, 1])
def test_sampled_faulty_gate_matches_theory(logical_value):
    epsilon = 0.6
    shots = 20_000
    counts = run_faulty_cx_syndrome(
        epsilon,
        logical_value=logical_value,
        shots=shots,
        seed=123,
    )
    observed = counts_to_probabilities(counts)
    expected = theoretical_syndrome_probabilities(epsilon)

    assert set(counts) <= {"00", "11"}

    # Five binomial standard deviations plus one-shot discretization margin.
    p_11 = expected["11"]
    tolerance = 5 * math.sqrt(p_11 * (1 - p_11) / shots) + 1 / shots
    assert observed["11"] == pytest.approx(p_11, abs=tolerance)


@pytest.mark.parametrize("invalid_epsilon", [math.inf, -math.inf, math.nan])
def test_non_finite_angle_is_rejected(invalid_epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        theoretical_syndrome_probabilities(invalid_epsilon)


@pytest.mark.parametrize("invalid_shots", [0, -1, 1.5, True])
def test_invalid_shot_count_is_rejected(invalid_shots):
    with pytest.raises(ValueError, match="shots"):
        run_faulty_cx_syndrome(0.2, shots=invalid_shots)
