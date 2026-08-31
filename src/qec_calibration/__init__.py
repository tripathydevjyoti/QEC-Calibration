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
from .single_gate_calibration import (
    CalibrationResult,
    build_compensated_syndrome_circuit,
    calibrate_single_gate,
    run_compensated_syndrome,
    syndrome_loss,
    theoretical_compensated_loss,
)

__all__ = [
    "EXPECTED_SYNDROMES",
    "CalibrationResult",
    "SYNDROMES",
    "build_compensated_syndrome_circuit",
    "build_faulty_cx_noise_model",
    "build_recovery_circuit",
    "build_syndrome_circuit",
    "counts_to_probabilities",
    "calibrate_single_gate",
    "run_error_sweep",
    "run_faulty_cx_syndrome",
    "run_compensated_syndrome",
    "run_recovery",
    "run_syndrome",
    "syndrome_loss",
    "theoretical_compensated_loss",
    "theoretical_syndrome_probabilities",
]
