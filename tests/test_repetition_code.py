"""Structural, ideal-QEC, and analytical checks for both repetition codes."""

import math
from itertools import combinations

import pytest

from qec_calibration import (
    build_syndrome_template,
    cnot_locations,
    counts_to_probabilities,
    logical_success_probability,
    run_recovery,
    run_syndrome,
    syndrome_for_errors,
    theoretical_single_fault_probabilities,
)


@pytest.mark.parametrize("n,total", [(3, 6), (5, 12)])
def test_locations_and_parameter_binding(n, total):
    model = build_syndrome_template(n)
    assert len(cnot_locations(n)) == len(model.parameters) == total
    assert len(set(model.gate_ids)) == total
    values = {"enc_0_1": 0.23}
    assert model.vector(values)[0] == pytest.approx(0.23)
    assert not model.bind(values).parameters
    assert model.sampler_values(range(total)).shape == (total,)


@pytest.mark.parametrize("n", [3, 5])
def test_single_x_truth_table(n):
    for qubit in range(n):
        expected = "".join(str(int(i in (qubit - 1, qubit))) for i in range(n - 1))
        assert syndrome_for_errors(n, (qubit,)) == expected
        model = build_syndrome_template(n, physical_errors=(qubit,))
        assert run_syndrome(model, shots=64) == {expected: 64}
    assert run_syndrome(build_syndrome_template(n), shots=64) == {"0" * (n - 1): 64}


@pytest.mark.parametrize("n", [3, 5])
def test_single_fault_theory(n):
    epsilon = 0.6
    model = build_syndrome_template(n)
    theory = theoretical_single_fault_probabilities(n, epsilon)
    counts = run_syndrome(model, {"enc_0_1": epsilon}, shots=4096, seed=12)
    actual = counts_to_probabilities(counts, n)
    signature = syndrome_for_errors(n, (1,))
    assert sum(theory.values()) == pytest.approx(1.0)
    assert theory[signature] == pytest.approx(math.sin(epsilon / 2) ** 2)
    assert actual[signature] == pytest.approx(theory[signature], abs=0.02)
    assert all(actual[key] == 0 for key in actual if key not in (signature, "0" * (n - 1)))


@pytest.mark.parametrize("n,errors", [(3, (0,)), (5, (0, 3))])
@pytest.mark.parametrize("logical_value", [0, 1])
def test_ideal_basis_recovery(n, errors, logical_value):
    model = build_syndrome_template(n, physical_errors=errors, logical_value=logical_value, recovery=True)
    counts = run_recovery(model, shots=32)
    assert logical_success_probability(counts, logical_value) == 1.0


def test_five_bit_decoder_all_correctable_patterns():
    patterns = [()] + [(q,) for q in range(5)] + list(combinations(range(5), 2))
    assert len({syndrome_for_errors(5, pattern) for pattern in patterns}) == 16
    for pattern in patterns:
        model = build_syndrome_template(5, physical_errors=pattern, recovery=True)
        assert logical_success_probability(run_recovery(model, shots=8), 0) == 1.0


def test_reject_invalid_inputs():
    with pytest.raises(ValueError):
        build_syndrome_template(4)
    with pytest.raises(ValueError):
        build_syndrome_template(3).vector({"typo": 0.1})
    with pytest.raises(ValueError):
        counts_to_probabilities({"00": 0}, 3)
