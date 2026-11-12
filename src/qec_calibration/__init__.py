"""Finite-shot syndrome calibration of 3- and 5-qubit repetition codes."""

from .calibration import (
    CalibrationResult, SPSAConfig, SyndromeObjective, calibrate_spsa,
    parameter_norm, single_fault_response_rank,
)
from .repetition_code import (
    CNOTLocation, ParameterizedCircuit, build_syndrome_template,
    cnot_locations, counts_to_probabilities, logical_success_probability,
    run_recovery, run_syndrome, syndrome_for_errors, syndrome_loss,
    theoretical_single_fault_probabilities,
)
from .repeated_rounds import (
    RepeatedCalibrationResult, RepeatedRoundCircuit, build_repeated_round_template,
    calibrate_repeated_rounds, round_syndrome_loss, run_repeated_rounds,
)

__all__ = [
    "CNOTLocation", "ParameterizedCircuit", "CalibrationResult", "SPSAConfig",
    "SyndromeObjective", "build_syndrome_template", "calibrate_spsa",
    "cnot_locations", "counts_to_probabilities", "logical_success_probability",
    "parameter_norm", "run_recovery", "run_syndrome", "single_fault_response_rank",
    "syndrome_for_errors", "syndrome_loss", "theoretical_single_fault_probabilities",
    "RepeatedCalibrationResult", "RepeatedRoundCircuit", "build_repeated_round_template",
    "calibrate_repeated_rounds", "round_syndrome_loss", "run_repeated_rounds",
]
