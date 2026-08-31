"""Dependency-free checks for the one-parameter calibration landscape."""

import math


def calibration_loss(true_error: float, correction_angle: float) -> float:
    return math.sin((true_error - correction_angle) / 2.0) ** 2


def test_loss_is_minimized_by_matching_the_error_angle():
    for true_error in (-0.3, -0.1, 0.0, 0.2, 0.35):
        candidates = [index / 100 for index in range(-40, 41)]
        estimate = min(
            candidates,
            key=lambda theta: calibration_loss(true_error, theta),
        )
        assert abs(estimate - true_error) <= 0.005


def test_equal_positive_and_negative_residuals_have_equal_loss():
    true_error = 0.2
    offset = 0.07
    assert math.isclose(
        calibration_loss(true_error, true_error - offset),
        calibration_loss(true_error, true_error + offset),
        abs_tol=1e-15,
    )
