"""Small finite-shot Qiskit SPSA checks; full sweeps live in the notebooks."""

import pytest

from qec_calibration import (
    SPSAConfig,
    SyndromeObjective,
    build_syndrome_template,
    calibrate_spsa,
    parameter_norm,
    single_fault_response_rank,
)


def test_active_subvector_and_rank():
    for n, rank in ((3, 3), (5, 7)):
        model = build_syndrome_template(n)
        assert single_fault_response_rank(model) == rank
        objective = SyndromeObjective(model, {"enc_0_1": 0.4, "enc_0_2": 0.3}, ("enc_0_1",), shots=32, seed=1)
        full = objective.expand([0.1])
        assert full[0] == pytest.approx(0.1)
        assert full[1] == pytest.approx(0.3)
        assert 0.0 <= objective([0.1]) <= 1.0


def test_short_spsa_reduces_single_fault():
    model = build_syndrome_template(3)
    result = calibrate_spsa(
        model, {"enc_0_1": 0.8}, active_gate_ids=("enc_0_1",),
        config=SPSAConfig(maxiter=20, train_shots=512, validation_shots=4096, seed=8),
    )
    assert result.final_loss < result.initial_loss
    assert abs(result.optimized_angles["enc_0_1"]) < 0.25
    assert result.function_evaluations >= 40
    assert result.history
    assert parameter_norm(result.optimized_angles, result.active_gate_ids) < 0.25
