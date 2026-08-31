"""Aer tests for closed-loop calibration of one faulty CNOT."""

import pytest

from qec_calibration import (
    build_compensated_syndrome_circuit,
    calibrate_single_gate,
    run_compensated_syndrome,
    theoretical_compensated_loss,
)


def test_compensating_rotation_is_explicit_in_circuit():
    circuit = build_compensated_syndrome_circuit(0.2)
    assert circuit.count_ops()["rx"] == 1
    assert circuit.count_ops()["cx"] == 6


@pytest.mark.parametrize(
    "true_error, correction_angle, expected",
    [
        (0.2, 0.0, 0.009966711079379185),
        (0.2, 0.2, 0.0),
        (-0.2, -0.2, 0.0),
    ],
)
def test_theoretical_compensation_loss(true_error, correction_angle, expected):
    assert theoretical_compensated_loss(
        true_error,
        correction_angle,
    ) == pytest.approx(expected)


@pytest.mark.parametrize("logical_value", [0, 1])
def test_exact_compensation_removes_the_syndrome(logical_value):
    shots = 512
    assert run_compensated_syndrome(
        true_error=0.2,
        correction_angle=0.2,
        logical_value=logical_value,
        shots=shots,
    ) == {"00": shots}


def test_coarse_to_fine_calibration_recovers_injected_angle():
    result = calibrate_single_gate(
        true_error=0.2,
        levels=2,
        points_per_level=9,
        shots_per_point=4_000,
        validation_shots=10_000,
        seed=123,
    )

    assert result.estimated_correction == pytest.approx(0.2, abs=0.05)
    assert abs(result.residual_angle) <= 0.05
    assert result.validation_loss < result.initial_theoretical_loss
    assert len(result.evaluations) == 18


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"search_bounds": (0.4, -0.4)}, "search_bounds"),
        ({"levels": 0}, "levels"),
        ({"points_per_level": 2}, "points_per_level"),
        ({"shots_per_point": 0}, "shots"),
    ],
)
def test_invalid_calibration_configuration_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        calibrate_single_gate(true_error=0.2, **kwargs)
