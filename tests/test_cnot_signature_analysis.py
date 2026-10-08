import math

import numpy as np
import pytest

from qec_calibration import (
    CNOT_GATE_IDS,
    EXPECTED_SINGLE_FAULT_SYNDROMES,
    SYNDROMES,
    catalog_probability_matrix,
    indistinguishable_signature_groups,
    normalized_single_fault_response_matrix,
    parameter_l2_norm,
    response_matrix_diagnostics,
    run_parameterized_syndrome,
    run_single_fault_catalog,
    theoretical_single_fault_probabilities,
    theoretical_single_fault_probability_matrix,
)


def test_theoretical_single_fault_probabilities_have_expected_support():
    angle = 0.6
    p_residual = math.sin(angle / 2.0) ** 2

    for gate_id in CNOT_GATE_IDS:
        probabilities = theoretical_single_fault_probabilities(gate_id, angle)
        residual = EXPECTED_SINGLE_FAULT_SYNDROMES[gate_id]

        assert tuple(probabilities) == SYNDROMES
        assert sum(probabilities.values()) == pytest.approx(1.0)
        assert probabilities["00"] == pytest.approx(1.0 - p_residual)
        assert probabilities[residual] == pytest.approx(p_residual)


def test_single_fault_probabilities_cannot_resolve_angle_sign():
    for gate_id in CNOT_GATE_IDS:
        positive = theoretical_single_fault_probabilities(gate_id, 0.6)
        negative = theoretical_single_fault_probabilities(gate_id, -0.6)
        assert positive == pytest.approx(negative)


def test_normalized_response_has_rank_three_and_expected_groups():
    response = normalized_single_fault_response_matrix(0.6)
    diagnostics = response_matrix_diagnostics(response)
    groups = indistinguishable_signature_groups(response)

    assert response.shape == (4, 6)
    assert diagnostics.rank == 3
    assert groups == (
        ("enc_01",),
        ("enc_02", "syn_1a", "syn_1b"),
        ("syn_0a", "syn_0b"),
    )


def test_sampled_catalog_matches_theory_within_shot_tolerance():
    angle = 0.6
    shots = 4096
    catalog = run_single_fault_catalog(angle, shots=shots, seed=123)
    empirical = catalog_probability_matrix(catalog)
    theory = theoretical_single_fault_probability_matrix(angle)

    assert empirical.shape == theory.shape == (4, 6)
    assert np.allclose(empirical.sum(axis=0), 1.0)
    assert np.max(np.abs(empirical - theory)) < 0.04


def test_parameter_l2_norm_uses_requested_gate_subset():
    angles = {"enc_01": 3.0, "syn_0a": 4.0, "syn_0b": 12.0}
    assert parameter_l2_norm(angles, ("enc_01", "syn_0a")) == pytest.approx(5.0)


def test_zero_angle_cannot_normalize_response():
    with pytest.raises(ValueError, match="nonzero"):
        normalized_single_fault_response_matrix(0.0)


def test_same_ancilla_faults_have_an_exact_cancellation_direction():
    counts = run_parameterized_syndrome(
        {"syn_0a": 0.6, "syn_0b": -0.6},
        shots=1024,
        seed=99,
    )
    assert counts == {"00": 1024}
