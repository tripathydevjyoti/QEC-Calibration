"""Syndrome-informed quantum error-correction calibration experiments."""

from .ideal_bit_flip_code import (
    EXPECTED_SYNDROMES,
    build_recovery_circuit,
    build_syndrome_circuit,
    run_recovery,
    run_syndrome,
)
from .single_gate_error import (
    SYNDROMES,
    build_faulty_cx_noise_model,
    counts_to_probabilities,
    run_error_sweep,
    run_faulty_cx_syndrome,
    theoretical_syndrome_probabilities,
)

__all__ = [
    "EXPECTED_SYNDROMES",
    "SYNDROMES",
    "build_faulty_cx_noise_model",
    "build_recovery_circuit",
    "build_syndrome_circuit",
    "counts_to_probabilities",
    "run_error_sweep",
    "run_faulty_cx_syndrome",
    "run_recovery",
    "run_syndrome",
    "theoretical_syndrome_probabilities",
]
