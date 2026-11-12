"""Retained-data parity histories, shared gate angles, and sampled SPSA."""

import math

import pytest

from qec_calibration import (
    SPSAConfig, build_repeated_round_template, calibrate_repeated_rounds,
    round_syndrome_loss, run_repeated_rounds,
)


@pytest.mark.parametrize("n,rounds", [(3, 1), (3, 3), (3, 5), (5, 3)])
def test_ideal_histories(n, rounds):
    model = build_repeated_round_template(n, rounds=rounds)
    assert sum(op.operation.name == "reset" for op in model.circuit.data) == (rounds - 1) * (n - 1)
    counts = run_repeated_rounds(model, shots=32)
    assert counts == {("0" * (n - 1),) * rounds + ("0" * n,): 32}
    assert round_syndrome_loss(counts, model) == 0.0


def test_encoding_fault_persists_across_rounds():
    model = build_repeated_round_template(3, rounds=3)
    counts = run_repeated_rounds(model, {"enc_0_1": 0.8}, shots=4096, seed=12)
    assert set(counts) == {("00", "00", "00", "000"), ("11", "11", "11", "010")}
    assert round_syndrome_loss(counts, model) == pytest.approx(math.sin(0.4) ** 2, abs=0.025)


def test_syndrome_fault_repeated_but_not_persistent():
    model = build_repeated_round_template(3, rounds=3)
    counts = run_repeated_rounds(model, {"syn_0_left": 0.8}, shots=4096, seed=13)
    assert all(key[-1] == "000" for key in counts)
    assert all(all(s in ("00", "10") for s in key[:-1]) for key in counts)
    assert ("00", "10", "00", "000") in counts
    assert round_syndrome_loss(counts, model) == pytest.approx(math.sin(0.4) ** 2, abs=0.02)


def test_same_ancilla_angles_cancel_even_after_multiple_rounds():
    model = build_repeated_round_template(3, rounds=5)
    counts = run_repeated_rounds(model, {"syn_0_left": 0.7, "syn_0_right": -0.7}, shots=128)
    assert counts == {("00",) * 5 + ("000",): 128}


def test_spsa_uses_average_loss_and_retains_histories():
    model = build_repeated_round_template(3, rounds=3)
    result = calibrate_repeated_rounds(
        model, {"enc_0_1": 0.8}, active_gate_ids=("enc_0_1",),
        config=SPSAConfig(maxiter=15, train_shots=256, validation_shots=2048, seed=17),
    )
    assert len(result.history) == 15
    assert result.final_loss == round_syndrome_loss(result.final_counts, model)
    assert sum(result.final_counts.values()) == 2048
    assert result.final_loss < result.initial_loss
    assert abs(result.optimized_angles["enc_0_1"]) < 0.8


def test_invalid_history_is_rejected():
    model = build_repeated_round_template(3, rounds=3)
    with pytest.raises(ValueError, match="invalid"):
        round_syndrome_loss({("00", "000"): 1}, model)
