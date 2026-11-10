"""Finite-shot syndrome calibration using Qiskit Algorithms SPSA."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from qiskit_aer.primitives import SamplerV2
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals

from .repetition_code import ParameterizedCircuit, _check_shots, run_syndrome, syndrome_loss


@dataclass(frozen=True)
class SPSAConfig:
    maxiter: int = 100
    train_shots: int = 1024
    validation_shots: int = 4096
    learning_rate: float = 0.7
    perturbation: float = 0.12
    seed: int = 42

    def __post_init__(self) -> None:
        _check_shots(self.maxiter)
        _check_shots(self.train_shots)
        _check_shots(self.validation_shots)
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not np.isfinite(self.perturbation) or self.perturbation <= 0:
            raise ValueError("perturbation must be positive and finite")


@dataclass(frozen=True)
class CalibrationResult:
    gate_ids: tuple[str, ...]
    active_gate_ids: tuple[str, ...]
    initial_angles: dict[str, float]
    optimized_angles: dict[str, float]
    initial_loss: float
    final_loss: float
    initial_counts: dict[str, int]
    final_counts: dict[str, int]
    function_evaluations: int
    history: tuple[dict[str, object], ...]


class SyndromeObjective:
    """Adapter from active SPSA coordinates to sampled syndrome loss."""

    def __init__(
        self,
        model: ParameterizedCircuit,
        initial: Mapping[str, float] | Sequence[float],
        active_gate_ids: Sequence[str],
        *,
        shots: int,
        seed: int,
    ) -> None:
        self.model = model
        self.base = model.vector(initial)
        self.active_gate_ids = tuple(active_gate_ids)
        if not self.active_gate_ids or len(set(self.active_gate_ids)) != len(self.active_gate_ids):
            raise ValueError("active gate IDs must be unique and nonempty")
        unknown = set(self.active_gate_ids) - set(model.gate_ids)
        if unknown:
            raise ValueError(f"unknown gate IDs: {sorted(unknown)}")
        self.indices = tuple(model.gate_ids.index(name) for name in self.active_gate_ids)
        _check_shots(shots)
        self.shots = shots
        self.sampler = SamplerV2(default_shots=shots, seed=seed)
        self.evaluations: list[dict[str, object]] = []

    def expand(self, active_values: Sequence[float]) -> np.ndarray:
        values = np.atleast_1d(np.asarray(active_values, dtype=float))
        if values.ndim != 1 or len(values) != len(self.indices) or not np.all(np.isfinite(values)):
            raise ValueError("active values must be a finite vector of the active dimension")
        full = self.base.copy()
        full[list(self.indices)] = values
        return full

    def __call__(self, active_values: Sequence[float]) -> float:
        full = self.expand(active_values)
        counts = run_syndrome(self.model, full, shots=self.shots, sampler=self.sampler)
        loss = syndrome_loss(counts, self.model)
        self.evaluations.append({
            "evaluation": len(self.evaluations) + 1,
            "angles": tuple(map(float, full)),
            "loss": float(loss),
            "counts": counts,
        })
        return loss


def calibrate_spsa(
    model: ParameterizedCircuit,
    initial_angles: Mapping[str, float] | Sequence[float],
    *,
    active_gate_ids: Sequence[str] | None = None,
    config: SPSAConfig | None = None,
) -> CalibrationResult:
    """Minimize 1-P(expected syndrome), then independently validate.

    Pass a syndrome-only model. Inactive angles stay fixed at their initial
    values. The ideal simulated calibration point is theta=0 for all gates.
    """
    if any(reg.name == "data_out" for reg in model.circuit.cregs):
        raise ValueError("SPSA requires a syndrome-only template")
    settings = config or SPSAConfig()
    initial = model.vector(initial_angles)
    active = tuple(model.gate_ids if active_gate_ids is None else active_gate_ids)
    objective = SyndromeObjective(model, initial, active, shots=settings.train_shots, seed=settings.seed)
    initial_counts = run_syndrome(model, initial, shots=settings.validation_shots, seed=settings.seed + 10_000)
    initial_loss = syndrome_loss(initial_counts, model)
    history: list[dict[str, object]] = []

    def callback(nfev, parameters, value, step_size, accepted):
        history.append({
            "iteration": len(history) + 1,
            "function_evaluations": int(nfev),
            "angles": tuple(map(float, objective.expand(parameters))),
            "loss": float(value),
            "step_size": float(step_size),
            "accepted": bool(accepted),
        })

    algorithm_globals.random_seed = settings.seed
    optimizer = SPSA(
        maxiter=settings.maxiter,
        learning_rate=settings.learning_rate,
        perturbation=settings.perturbation,
        callback=callback,
    )
    result = optimizer.minimize(fun=objective, x0=initial[list(objective.indices)])
    final_vector = objective.expand(result.x)
    final_counts = run_syndrome(
        model,
        final_vector,
        shots=settings.validation_shots,
        seed=settings.seed + 20_000,
    )
    return CalibrationResult(
        gate_ids=model.gate_ids,
        active_gate_ids=active,
        initial_angles=model.named(initial),
        optimized_angles=model.named(final_vector),
        initial_loss=initial_loss,
        final_loss=syndrome_loss(final_counts, model),
        initial_counts=initial_counts,
        final_counts=final_counts,
        function_evaluations=int(result.nfev),
        history=tuple(history),
    )


def parameter_norm(angles: Mapping[str, float], active_gate_ids: Sequence[str]) -> float:
    return float(np.linalg.norm([angles[name] for name in active_gate_ids]))


def single_fault_response_rank(
    model: ParameterizedCircuit, gate_ids: Sequence[str] | None = None
) -> int:
    """Rank of exact isolated-fault probability shifts (not a Jacobian)."""
    # Exact single-CNOT syndrome columns for this Rx-on-target model:
    # encoding targets yield data-X syndromes; syndrome CNOTs yield one
    # ancilla measurement-bit flip. Equal signatures imply dependent columns.
    from .repetition_code import syndrome_for_errors

    selected = model.gate_ids if gate_ids is None else tuple(gate_ids)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(model.gate_ids):
        raise ValueError("select one or more distinct known gates")
    columns = []
    for loc in model.locations:
        if loc.gate_id not in selected:
            continue
        if loc.stage == "encoding":
            signature = syndrome_for_errors(model.n, (loc.target_index,))
        else:
            check = loc.target_index - model.n
            signature = "".join("1" if i == check else "0" for i in range(model.n - 1))
        column = np.zeros(2 ** (model.n - 1))
        column[int(signature, 2)] = 1
        column[0] = -1
        columns.append(column)
    return int(np.linalg.matrix_rank(np.stack(columns, axis=1)))
