"""Qiskit SamplerV2 + SPSA calibration of parameterized CNOT residuals.

The production path in this module is a standard Qiskit variational loop:

``ParameterVector circuit -> SamplerV2 -> syndrome loss -> SPSA.minimize``.

Unlike :mod:`single_gate_calibration`, it does not add a compensating gate
against a hidden error.  SPSA directly updates selected entries of the CNOT
residual-angle vector and learns only from sampled syndrome counts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from qiskit_aer.primitives import SamplerV2
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals

from .ideal_bit_flip_code import _normalize_syndrome_counts
from .parameterized_cnot_errors import (
    CNOT_GATE_IDS,
    CNOT_PARAMETER_INDEX,
    ParameterizedCNOTCircuit,
    angles_to_vector,
    build_parameterized_syndrome_template,
    expected_syndrome_loss,
    run_parameterized_syndrome,
    vector_to_angles,
)


SPSA_CALIBRATION_VERSION = "qiskit-algorithms-spsa-v1"


@dataclass(frozen=True)
class DirectCalibrationResult:
    """Complete record of a one-parameter finite-shot calibration run."""

    gate_id: str
    initial_angle: float
    final_angle: float
    initial_loss: float
    validation_loss: float
    initial_counts: dict[str, int]
    validation_counts: dict[str, int]
    iterations: tuple[dict[str, float | int], ...]


@dataclass(frozen=True)
class SPSACalibrationResult:
    """Result of a standard Qiskit SPSA syndrome-calibration run."""

    active_gate_ids: tuple[str, ...]
    initial_angles: dict[str, float]
    optimized_angles: dict[str, float]
    initial_loss: float
    validation_loss: float
    initial_counts: dict[str, int]
    validation_counts: dict[str, int]
    function_evaluations: int
    objective_history: tuple[dict[str, object], ...]
    iteration_history: tuple[dict[str, object], ...]


class SyndromeObjective:
    """Sampler-based syndrome loss callable for Qiskit optimizers.

    The optimizer sees only the selected active parameters.  This object
    expands them into the full six-CNOT vector, submits the same symbolic
    circuit template to ``SamplerV2``, and returns

    ``1 - P(expected syndrome)``.
    """

    def __init__(
        self,
        *,
        active_gate_ids: Sequence[str],
        fixed_angles: Mapping[str, float] | None = None,
        circuit_model: ParameterizedCNOTCircuit | None = None,
        sampler: SamplerV2 | None = None,
        shots: int = 4096,
        seed: int = 42,
        error_qubit: int | None = None,
        logical_value: int = 0,
    ) -> None:
        active = tuple(active_gate_ids)
        if not active:
            raise ValueError("active_gate_ids must contain at least one gate")
        if len(set(active)) != len(active):
            raise ValueError("active_gate_ids cannot contain duplicates")
        unknown = set(active) - set(CNOT_GATE_IDS)
        if unknown:
            raise ValueError(f"Unknown CNOT gate IDs: {sorted(unknown)}")
        _validate_shots(shots, "shots")
        if error_qubit not in (None, 0, 1, 2):
            raise ValueError("error_qubit must be None, 0, 1, or 2")
        if logical_value not in (0, 1):
            raise ValueError("logical_value must be 0 or 1")

        self.active_gate_ids = active
        self.active_indices = tuple(CNOT_PARAMETER_INDEX[name] for name in active)
        self.base_vector = angles_to_vector(fixed_angles)
        self.circuit_model = circuit_model or build_parameterized_syndrome_template(
            error_qubit=error_qubit,
            logical_value=logical_value,
        )
        self.sampler = sampler or SamplerV2(default_shots=shots, seed=seed)
        self.shots = shots
        self.error_qubit = error_qubit
        self.logical_value = logical_value
        self.history: list[dict[str, object]] = []

    def expand(self, active_values) -> np.ndarray:
        """Insert active optimizer values into the complete CNOT vector."""
        values = np.asarray(active_values, dtype=float)
        if values.ndim == 0 and len(self.active_indices) == 1:
            values = values.reshape(1)
        if values.ndim != 1 or len(values) != len(self.active_indices):
            raise ValueError(
                "active_values must be a one-dimensional vector with one "
                "entry per active gate"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("active parameter values must be finite")

        full_vector = self.base_vector.copy()
        for index, value in zip(self.active_indices, values, strict=True):
            full_vector[index] = value
        return full_vector

    def evaluate(
        self,
        active_values,
        *,
        record: bool = True,
    ) -> float:
        """Evaluate the measured syndrome loss at one active parameter vector."""
        full_vector = self.expand(active_values)
        pub_result = self.sampler.run(
            [(self.circuit_model.circuit, full_vector)],
            shots=self.shots,
        ).result()[0]

        try:
            raw_counts = pub_result.data.syndrome.get_counts()
        except AttributeError as exc:
            raise RuntimeError(
                "Sampler result does not contain the expected 'syndrome' register"
            ) from exc

        counts = _normalize_syndrome_counts(raw_counts)
        loss = expected_syndrome_loss(counts, error_qubit=self.error_qubit)
        if record:
            self.history.append(
                {
                    "evaluation": len(self.history) + 1,
                    "active_values": tuple(
                        float(full_vector[index]) for index in self.active_indices
                    ),
                    "full_vector": tuple(float(value) for value in full_vector),
                    "loss": loss,
                    "counts": counts,
                }
            )
        return loss

    def __call__(self, active_values) -> float:
        return self.evaluate(active_values)


def _validate_finite(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_positive(value: float, name: str) -> float:
    result = _validate_finite(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_shots(shots: int, name: str) -> None:
    if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def build_spsa_optimizer(
    *,
    maxiter: int = 24,
    learning_rate: float = 1.2,
    perturbation: float = 0.12,
    callback=None,
) -> SPSA:
    """Construct the standard Qiskit Algorithms SPSA optimizer.

    This public factory makes the variational optimizer explicit and keeps
    the same hyperparameters available to notebooks and reusable workflows.
    """
    if not isinstance(maxiter, int) or isinstance(maxiter, bool) or maxiter <= 0:
        raise ValueError("maxiter must be a positive integer")
    learning_rate = _validate_positive(learning_rate, "learning_rate")
    perturbation = _validate_positive(perturbation, "perturbation")
    return SPSA(
        maxiter=maxiter,
        learning_rate=learning_rate,
        perturbation=perturbation,
        callback=callback,
    )


def calibrate_single_cnot_direct(
    initial_angle: float,
    gate_id: str = "enc_01",
    *,
    iterations: int = 24,
    shots_per_evaluation: int = 4096,
    validation_shots: int = 20_000,
    learning_rate: float = 1.2,
    perturbation: float = 0.12,
    learning_decay: float = 0.602,
    perturbation_decay: float = 0.101,
    stability: float = 2.0,
    angle_bounds: tuple[float, float] = (-math.pi, math.pi),
    logical_value: int = 0,
    seed: int = 42,
) -> DirectCalibrationResult:
    """Calibrate one CNOT angle using finite-difference syndrome feedback.

    For iteration ``k``, the circuit is queried at ``theta_k + c_k`` and
    ``theta_k - c_k``.  Their sampled losses estimate the local derivative,
    and a projected stochastic-gradient update produces ``theta_{k+1}``.

    Only one gate is active in this first benchmark.  The same two-evaluation
    structure will later become vector SPSA when several CNOTs are active.
    """
    if gate_id not in CNOT_GATE_IDS:
        raise ValueError(
            f"gate_id must be one of {CNOT_GATE_IDS}; received {gate_id!r}"
        )
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    _validate_shots(shots_per_evaluation, "shots_per_evaluation")
    _validate_shots(validation_shots, "validation_shots")
    if logical_value not in (0, 1):
        raise ValueError("logical_value must be 0 or 1")

    theta = _validate_finite(initial_angle, "initial_angle")
    learning_rate = _validate_positive(learning_rate, "learning_rate")
    perturbation = _validate_positive(perturbation, "perturbation")
    learning_decay = _validate_positive(learning_decay, "learning_decay")
    perturbation_decay = _validate_positive(
        perturbation_decay,
        "perturbation_decay",
    )
    stability = _validate_finite(stability, "stability")
    if stability < 0:
        raise ValueError("stability must be nonnegative")

    lower = _validate_finite(angle_bounds[0], "lower angle bound")
    upper = _validate_finite(angle_bounds[1], "upper angle bound")
    if lower >= upper:
        raise ValueError("angle_bounds must satisfy lower < upper")
    if not lower <= theta <= upper:
        raise ValueError("initial_angle must lie within angle_bounds")

    initial_counts = run_parameterized_syndrome(
        {gate_id: theta},
        logical_value=logical_value,
        shots=validation_shots,
        seed=seed,
    )
    initial_loss = expected_syndrome_loss(initial_counts)

    history: list[dict[str, float | int]] = []
    evaluation_index = 1

    for iteration in range(iterations):
        step_number = iteration + 1
        a_k = learning_rate / ((step_number + stability) ** learning_decay)
        c_k = perturbation / (step_number**perturbation_decay)

        theta_plus = _clip(theta + c_k, lower, upper)
        theta_minus = _clip(theta - c_k, lower, upper)
        denominator = theta_plus - theta_minus
        if denominator <= 0:
            raise RuntimeError("angle bounds leave no room for a gradient probe")

        plus_counts = run_parameterized_syndrome(
            {gate_id: theta_plus},
            logical_value=logical_value,
            shots=shots_per_evaluation,
            seed=seed + evaluation_index,
        )
        evaluation_index += 1
        minus_counts = run_parameterized_syndrome(
            {gate_id: theta_minus},
            logical_value=logical_value,
            shots=shots_per_evaluation,
            seed=seed + evaluation_index,
        )
        evaluation_index += 1

        loss_plus = expected_syndrome_loss(plus_counts)
        loss_minus = expected_syndrome_loss(minus_counts)
        gradient = (loss_plus - loss_minus) / denominator
        updated_theta = _clip(theta - a_k * gradient, lower, upper)

        history.append(
            {
                "iteration": step_number,
                "angle_before": theta,
                "probe_plus": theta_plus,
                "probe_minus": theta_minus,
                "loss_plus": loss_plus,
                "loss_minus": loss_minus,
                "gradient_estimate": gradient,
                "learning_rate": a_k,
                "perturbation": c_k,
                "angle_after": updated_theta,
            }
        )
        theta = updated_theta

    validation_counts = run_parameterized_syndrome(
        {gate_id: theta},
        logical_value=logical_value,
        shots=validation_shots,
        seed=seed + evaluation_index,
    )
    validation_loss = expected_syndrome_loss(validation_counts)

    return DirectCalibrationResult(
        gate_id=gate_id,
        initial_angle=float(initial_angle),
        final_angle=theta,
        initial_loss=initial_loss,
        validation_loss=validation_loss,
        initial_counts=initial_counts,
        validation_counts=validation_counts,
        iterations=tuple(history),
    )


def calibrate_cnot_parameters_spsa(
    initial_angles: Mapping[str, float],
    active_gate_ids: Sequence[str] = ("enc_01",),
    *,
    maxiter: int = 24,
    shots_per_evaluation: int = 4096,
    validation_shots: int = 20_000,
    learning_rate: float = 1.2,
    perturbation: float = 0.12,
    logical_value: int = 0,
    error_qubit: int | None = None,
    seed: int = 42,
) -> SPSACalibrationResult:
    """Optimize selected CNOT parameters with Qiskit Algorithms SPSA.

    Inactive angles retain the values supplied through ``initial_angles``.
    The optimizer receives only the active subvector and obtains every loss
    value from finite-shot ``SamplerV2`` syndrome measurements.
    """
    if not isinstance(maxiter, int) or isinstance(maxiter, bool) or maxiter <= 0:
        raise ValueError("maxiter must be a positive integer")
    _validate_shots(shots_per_evaluation, "shots_per_evaluation")
    _validate_shots(validation_shots, "validation_shots")
    learning_rate = _validate_positive(learning_rate, "learning_rate")
    perturbation = _validate_positive(perturbation, "perturbation")

    initial_vector = angles_to_vector(initial_angles)
    active = tuple(active_gate_ids)
    if not active:
        raise ValueError("active_gate_ids must contain at least one gate")
    if len(set(active)) != len(active):
        raise ValueError("active_gate_ids cannot contain duplicates")
    unknown = set(active) - set(CNOT_GATE_IDS)
    if unknown:
        raise ValueError(f"Unknown CNOT gate IDs: {sorted(unknown)}")
    active_indices = tuple(CNOT_PARAMETER_INDEX[name] for name in active)
    initial_active = np.asarray(
        [initial_vector[index] for index in active_indices],
        dtype=float,
    )

    initial_angle_map = vector_to_angles(initial_vector)
    initial_counts = run_parameterized_syndrome(
        initial_angle_map,
        error_qubit=error_qubit,
        logical_value=logical_value,
        shots=validation_shots,
        seed=seed,
    )
    initial_loss = expected_syndrome_loss(initial_counts, error_qubit=error_qubit)

    model = build_parameterized_syndrome_template(
        error_qubit=error_qubit,
        logical_value=logical_value,
    )
    objective = SyndromeObjective(
        active_gate_ids=active,
        fixed_angles=initial_angle_map,
        circuit_model=model,
        shots=shots_per_evaluation,
        seed=seed,
        error_qubit=error_qubit,
        logical_value=logical_value,
    )

    iteration_history: list[dict[str, object]] = []

    def callback(
        function_evaluations: int,
        parameters: np.ndarray,
        loss: float,
        step_size: float,
        accepted: bool,
    ) -> None:
        full_vector = objective.expand(parameters)
        iteration_history.append(
            {
                "iteration": len(iteration_history) + 1,
                "function_evaluations": function_evaluations,
                "active_values": tuple(float(value) for value in parameters),
                "full_vector": tuple(float(value) for value in full_vector),
                "loss": float(loss),
                "step_size": float(step_size),
                "accepted": bool(accepted),
            }
        )

    algorithm_globals.random_seed = seed
    optimizer = build_spsa_optimizer(
        maxiter=maxiter,
        learning_rate=learning_rate,
        perturbation=perturbation,
        callback=callback,
    )
    optimizer_result = optimizer.minimize(
        fun=objective,
        x0=initial_active,
    )

    optimized_vector = objective.expand(optimizer_result.x)
    optimized_angles = vector_to_angles(optimized_vector)
    validation_counts = run_parameterized_syndrome(
        optimized_angles,
        error_qubit=error_qubit,
        logical_value=logical_value,
        shots=validation_shots,
        seed=seed + len(objective.history) + 1,
    )
    validation_loss = expected_syndrome_loss(
        validation_counts,
        error_qubit=error_qubit,
    )

    return SPSACalibrationResult(
        active_gate_ids=active,
        initial_angles=initial_angle_map,
        optimized_angles=optimized_angles,
        initial_loss=initial_loss,
        validation_loss=validation_loss,
        initial_counts=initial_counts,
        validation_counts=validation_counts,
        function_evaluations=int(optimizer_result.nfev),
        objective_history=tuple(objective.history),
        iteration_history=tuple(iteration_history),
    )

