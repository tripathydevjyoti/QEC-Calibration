"""Syndrome signatures and identifiability diagnostics for faulty CNOTs.

The functions in this module separate two questions:

* static identifiability -- which CNOT locations can be distinguished from a
  fixed syndrome probability vector; and
* active calibratability -- whether controlled parameter perturbations can
  reduce a syndrome-derived objective.

For the current target-``Rx`` model, every isolated CNOT fault transfers
probability from ``00`` to one nonzero syndrome.  Several CNOT locations share
the same transfer direction, producing identical columns in the static
response matrix even though each control parameter can be addressed
independently by an optimizer.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from .parameterized_cnot_errors import (
    CNOT_GATE_IDS,
    run_parameterized_syndrome,
)
from .single_gate_error import SYNDROMES, counts_to_probabilities


EXPECTED_SINGLE_FAULT_SYNDROMES: Final[dict[str, str]] = {
    "enc_01": "11",
    "enc_02": "01",
    "syn_0a": "10",
    "syn_0b": "10",
    "syn_1a": "01",
    "syn_1b": "01",
}


@dataclass(frozen=True)
class SingleFaultSignature:
    """One isolated CNOT fault and its measured syndrome distribution."""

    gate_id: str
    angle: float
    expected_residual_syndrome: str
    counts: dict[str, int]
    probabilities: dict[str, float]


@dataclass(frozen=True)
class ResponseMatrixDiagnostics:
    """Linear-algebra diagnostics for a syndrome response matrix."""

    rank: int
    singular_values: tuple[float, ...]
    nonzero_condition_number: float


def _validate_gate_ids(gate_ids: Sequence[str]) -> tuple[str, ...]:
    result = tuple(gate_ids)
    if not result:
        raise ValueError("gate_ids must contain at least one gate")
    if len(set(result)) != len(result):
        raise ValueError("gate_ids cannot contain duplicates")
    unknown = set(result) - set(CNOT_GATE_IDS)
    if unknown:
        raise ValueError(f"Unknown CNOT gate IDs: {sorted(unknown)}")
    return result


def _validate_angle(angle: float) -> float:
    try:
        value = float(angle)
    except (TypeError, ValueError) as exc:
        raise ValueError("angle must be a real number") from exc
    if not math.isfinite(value):
        raise ValueError("angle must be finite")
    return value


def theoretical_single_fault_probabilities(
    gate_id: str,
    angle: float,
) -> dict[str, float]:
    """Return the exact no-probe syndrome distribution for one faulty CNOT."""
    if gate_id not in EXPECTED_SINGLE_FAULT_SYNDROMES:
        raise ValueError(f"gate_id must be one of {CNOT_GATE_IDS}")
    angle = _validate_angle(angle)
    residual_syndrome = EXPECTED_SINGLE_FAULT_SYNDROMES[gate_id]
    p_residual = math.sin(angle / 2.0) ** 2

    probabilities = {syndrome: 0.0 for syndrome in SYNDROMES}
    probabilities["00"] = 1.0 - p_residual
    probabilities[residual_syndrome] = p_residual
    return probabilities


def run_single_fault_catalog(
    angle: float = 0.6,
    *,
    gate_ids: Sequence[str] = CNOT_GATE_IDS,
    shots: int = 20_000,
    seed: int = 42,
) -> tuple[SingleFaultSignature, ...]:
    """Sample each requested CNOT as the only nonzero calibration parameter."""
    angle = _validate_angle(angle)
    selected = _validate_gate_ids(gate_ids)

    signatures = []
    for offset, gate_id in enumerate(selected):
        counts = run_parameterized_syndrome(
            {gate_id: angle},
            error_qubit=None,
            logical_value=0,
            shots=shots,
            seed=seed + offset,
        )
        signatures.append(
            SingleFaultSignature(
                gate_id=gate_id,
                angle=angle,
                expected_residual_syndrome=(
                    EXPECTED_SINGLE_FAULT_SYNDROMES[gate_id]
                ),
                counts=counts,
                probabilities=counts_to_probabilities(counts),
            )
        )
    return tuple(signatures)


def catalog_probability_matrix(
    catalog: Sequence[SingleFaultSignature],
    *,
    syndrome_order: Sequence[str] = SYNDROMES,
) -> np.ndarray:
    """Return a syndrome-by-gate matrix from a sampled signature catalog."""
    signatures = tuple(catalog)
    if not signatures:
        raise ValueError("catalog must contain at least one signature")
    order = tuple(syndrome_order)
    if len(order) != len(set(order)) or set(order) != set(SYNDROMES):
        raise ValueError(f"syndrome_order must contain each of {SYNDROMES} once")

    return np.asarray(
        [
            [signature.probabilities[syndrome] for signature in signatures]
            for syndrome in order
        ],
        dtype=float,
    )


def theoretical_single_fault_probability_matrix(
    angle: float,
    *,
    gate_ids: Sequence[str] = CNOT_GATE_IDS,
    syndrome_order: Sequence[str] = SYNDROMES,
) -> np.ndarray:
    """Return the exact syndrome-by-gate probability matrix."""
    selected = _validate_gate_ids(gate_ids)
    order = tuple(syndrome_order)
    if len(order) != len(set(order)) or set(order) != set(SYNDROMES):
        raise ValueError(f"syndrome_order must contain each of {SYNDROMES} once")

    columns = [
        theoretical_single_fault_probabilities(gate_id, angle)
        for gate_id in selected
    ]
    return np.asarray(
        [
            [probabilities[syndrome] for probabilities in columns]
            for syndrome in order
        ],
        dtype=float,
    )


def normalized_single_fault_response_matrix(
    angle: float,
    *,
    gate_ids: Sequence[str] = CNOT_GATE_IDS,
    syndrome_order: Sequence[str] = SYNDROMES,
) -> np.ndarray:
    """Return probability transfers normalized by ``sin(angle/2)**2``.

    The ideal probability vector is subtracted from every column before
    normalization.  This produces an angle-independent matrix for the current
    single-fault model and avoids the vanishing first derivative with respect
    to ``angle`` at the ideal point.
    """
    angle = _validate_angle(angle)
    scale = math.sin(angle / 2.0) ** 2
    if math.isclose(scale, 0.0, abs_tol=1e-15):
        raise ValueError(
            "angle must produce nonzero sin(angle/2)**2 for normalization"
        )

    order = tuple(syndrome_order)
    matrix = theoretical_single_fault_probability_matrix(
        angle,
        gate_ids=gate_ids,
        syndrome_order=order,
    )
    ideal = np.asarray(
        [1.0 if syndrome == "00" else 0.0 for syndrome in order],
        dtype=float,
    )
    return (matrix - ideal[:, None]) / scale


def response_matrix_diagnostics(
    matrix,
    *,
    relative_tolerance: float | None = None,
) -> ResponseMatrixDiagnostics:
    """Return rank, singular values, and condition number on the row space."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError("matrix must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix values must be finite")

    singular_values = np.linalg.svd(values, compute_uv=False)
    if relative_tolerance is None:
        tolerance = (
            max(values.shape)
            * np.finfo(float).eps
            * (singular_values[0] if len(singular_values) else 0.0)
        )
    else:
        relative_tolerance = float(relative_tolerance)
        if relative_tolerance < 0 or not math.isfinite(relative_tolerance):
            raise ValueError("relative_tolerance must be finite and nonnegative")
        tolerance = (
            relative_tolerance
            * (singular_values[0] if len(singular_values) else 0.0)
        )

    nonzero = singular_values[singular_values > tolerance]
    rank = int(len(nonzero))
    condition = (
        float(nonzero[0] / nonzero[-1]) if len(nonzero) else math.inf
    )
    return ResponseMatrixDiagnostics(
        rank=rank,
        singular_values=tuple(float(value) for value in singular_values),
        nonzero_condition_number=condition,
    )


def indistinguishable_signature_groups(
    matrix,
    *,
    gate_ids: Sequence[str] = CNOT_GATE_IDS,
    atol: float = 1e-12,
) -> tuple[tuple[str, ...], ...]:
    """Group CNOT locations whose response-matrix columns are equal."""
    values = np.asarray(matrix, dtype=float)
    selected = _validate_gate_ids(gate_ids)
    if values.ndim != 2 or values.shape[1] != len(selected):
        raise ValueError("matrix must have one column per gate_id")
    if atol < 0 or not math.isfinite(atol):
        raise ValueError("atol must be finite and nonnegative")

    unassigned = list(range(len(selected)))
    groups: list[tuple[str, ...]] = []
    while unassigned:
        reference = unassigned.pop(0)
        matches = [
            index
            for index in unassigned
            if np.allclose(
                values[:, reference],
                values[:, index],
                rtol=0.0,
                atol=atol,
            )
        ]
        match_set = set(matches)
        unassigned = [index for index in unassigned if index not in match_set]
        groups.append(
            tuple(selected[index] for index in (reference, *matches))
        )
    return tuple(groups)


def parameter_l2_norm(
    angles: Mapping[str, float],
    gate_ids: Sequence[str] | None = None,
) -> float:
    """Return the Euclidean norm of selected CNOT calibration parameters."""
    selected = CNOT_GATE_IDS if gate_ids is None else _validate_gate_ids(gate_ids)
    unknown = set(angles) - set(CNOT_GATE_IDS)
    if unknown:
        raise ValueError(f"Unknown CNOT gate IDs: {sorted(unknown)}")
    values = np.asarray([float(angles.get(gate_id, 0.0)) for gate_id in selected])
    if not np.all(np.isfinite(values)):
        raise ValueError("all angle values must be finite")
    return float(np.linalg.norm(values))
