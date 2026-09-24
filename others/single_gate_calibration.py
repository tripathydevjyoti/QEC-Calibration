"""Shot-based calibration of one coherently faulty encoding CNOT.

The simulated faulty gate contributes ``Rx(epsilon)`` on the target of
``cx(0, 1)``.  The circuit adds a tunable ``Rx(-theta)`` immediately after
that gate, leaving the residual rotation ``Rx(epsilon - theta)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from qiskit import (
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    transpile,
)
from qiskit_aer import AerSimulator

from .ideal_bit_flip_code import (
    _add_syndrome_extraction,
    _normalize_syndrome_counts,
)
from .single_gate_error import (
    _validate_epsilon,
    _validate_shots,
    build_faulty_cx_noise_model,
    counts_to_probabilities,
    theoretical_syndrome_probabilities,
)


@dataclass(frozen=True)
class CalibrationResult:
    """Complete output of a one-parameter coarse-to-fine calibration."""

    true_error: float
    estimated_correction: float
    residual_angle: float
    initial_theoretical_loss: float
    validation_loss: float
    validation_counts: dict[str, int]
    evaluations: tuple[dict[str, float | int], ...]
    level_summaries: tuple[dict[str, float | int], ...]


def _validate_angle(angle: float, name: str) -> float:
    value = float(angle)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def build_compensated_syndrome_circuit(
    correction_angle: float,
    logical_value: int = 0,
) -> QuantumCircuit:
    """Build the QEC syndrome circuit with an explicit ``Rx(-theta)`` control."""
    correction_angle = _validate_angle(correction_angle, "correction_angle")
    if logical_value not in (0, 1):
        raise ValueError("logical_value must be 0 or 1")

    data = QuantumRegister(3, "data")
    ancilla = QuantumRegister(2, "ancilla")
    syndrome_bits = ClassicalRegister(2, "syndrome")
    circuit = QuantumCircuit(
        data,
        ancilla,
        syndrome_bits,
        name="compensated_bit_flip_qec",
    )

    if logical_value == 1:
        circuit.x(data[0])

    # The Aer noise model applies Rx(epsilon) directly after this CX.
    circuit.cx(data[0], data[1])
    circuit.rx(-correction_angle, data[1])
    circuit.cx(data[0], data[2])

    circuit.barrier()
    _add_syndrome_extraction(circuit, data, ancilla, syndrome_bits)
    return circuit


def run_compensated_syndrome(
    true_error: float,
    correction_angle: float,
    logical_value: int = 0,
    shots: int = 5_000,
    seed: int = 42,
) -> dict[str, int]:
    """Run one trial correction against the hidden simulated gate error."""
    true_error = _validate_epsilon(true_error)
    correction_angle = _validate_angle(correction_angle, "correction_angle")
    _validate_shots(shots)

    simulator = AerSimulator(
        noise_model=build_faulty_cx_noise_model(true_error),
    )
    circuit = build_compensated_syndrome_circuit(
        correction_angle=correction_angle,
        logical_value=logical_value,
    )
    compiled = transpile(circuit, simulator, optimization_level=0)
    result = simulator.run(
        compiled,
        shots=shots,
        seed_simulator=seed,
    ).result()
    return _normalize_syndrome_counts(result.get_counts())


def syndrome_loss(counts: dict[str, int]) -> float:
    """Return the measured probability of any nonzero syndrome."""
    probabilities = counts_to_probabilities(counts)
    return 1.0 - probabilities["00"]


def theoretical_compensated_loss(
    true_error: float,
    correction_angle: float,
) -> float:
    """Return ``sin²((epsilon - theta)/2)`` for the current error model."""
    true_error = _validate_epsilon(true_error)
    correction_angle = _validate_angle(correction_angle, "correction_angle")
    residual_angle = true_error - correction_angle
    return theoretical_syndrome_probabilities(residual_angle)["11"]


def _linspace(start: float, stop: float, points: int) -> list[float]:
    step = (stop - start) / (points - 1)
    return [start + index * step for index in range(points)]


def calibrate_single_gate(
    true_error: float,
    search_bounds: tuple[float, float] = (-0.4, 0.4),
    levels: int = 3,
    points_per_level: int = 9,
    shots_per_point: int = 5_000,
    validation_shots: int = 20_000,
    logical_value: int = 0,
    seed: int = 42,
) -> CalibrationResult:
    """Estimate a correction angle using a measured coarse-to-fine scan.

    At each level, trial angles are evaluated using only sampled syndrome
    counts.  The next interval is centered on the mean of all angles tied for
    the smallest measured loss; this behaves sensibly when finite shots create
    a small zero-count plateau near the optimum.
    """
    true_error = _validate_epsilon(true_error)
    _validate_shots(shots_per_point)
    _validate_shots(validation_shots)

    if not isinstance(levels, int) or isinstance(levels, bool) or levels <= 0:
        raise ValueError("levels must be a positive integer")
    if (
        not isinstance(points_per_level, int)
        or isinstance(points_per_level, bool)
        or points_per_level < 3
    ):
        raise ValueError("points_per_level must be an integer of at least 3")

    lower = _validate_angle(search_bounds[0], "lower search bound")
    upper = _validate_angle(search_bounds[1], "upper search bound")
    if lower >= upper:
        raise ValueError("search_bounds must satisfy lower < upper")

    original_lower = lower
    original_upper = upper
    evaluations: list[dict[str, float | int]] = []
    level_summaries: list[dict[str, float | int]] = []
    evaluation_index = 0
    estimated_correction = (lower + upper) / 2.0

    for level in range(1, levels + 1):
        trial_angles = _linspace(lower, upper, points_per_level)
        level_rows: list[dict[str, float | int]] = []

        for correction_angle in trial_angles:
            counts = run_compensated_syndrome(
                true_error=true_error,
                correction_angle=correction_angle,
                logical_value=logical_value,
                shots=shots_per_point,
                seed=seed + evaluation_index,
            )
            measured_loss = syndrome_loss(counts)
            row: dict[str, float | int] = {
                "level": level,
                "correction_angle": correction_angle,
                "count_00": counts.get("00", 0),
                "count_11": counts.get("11", 0),
                "measured_loss": measured_loss,
                "theoretical_loss": theoretical_compensated_loss(
                    true_error,
                    correction_angle,
                ),
            }
            evaluations.append(row)
            level_rows.append(row)
            evaluation_index += 1

        minimum_loss = min(float(row["measured_loss"]) for row in level_rows)
        tied_angles = [
            float(row["correction_angle"])
            for row in level_rows
            if float(row["measured_loss"]) == minimum_loss
        ]
        estimated_correction = sum(tied_angles) / len(tied_angles)
        grid_spacing = (upper - lower) / (points_per_level - 1)

        level_summaries.append(
            {
                "level": level,
                "lower_bound": lower,
                "upper_bound": upper,
                "grid_spacing": grid_spacing,
                "minimum_measured_loss": minimum_loss,
                "level_estimate": estimated_correction,
                "tied_minima": len(tied_angles),
            }
        )

        lower = max(original_lower, estimated_correction - grid_spacing)
        upper = min(original_upper, estimated_correction + grid_spacing)

    validation_counts = run_compensated_syndrome(
        true_error=true_error,
        correction_angle=estimated_correction,
        logical_value=logical_value,
        shots=validation_shots,
        seed=seed + evaluation_index,
    )

    return CalibrationResult(
        true_error=true_error,
        estimated_correction=estimated_correction,
        residual_angle=true_error - estimated_correction,
        initial_theoretical_loss=theoretical_compensated_loss(true_error, 0.0),
        validation_loss=syndrome_loss(validation_counts),
        validation_counts=validation_counts,
        evaluations=tuple(evaluations),
        level_summaries=tuple(level_summaries),
    )
