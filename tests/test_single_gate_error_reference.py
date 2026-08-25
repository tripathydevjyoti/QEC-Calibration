"""Independent NumPy oracle for the coherent-error syndrome formula."""

import numpy as np


I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
X1 = np.kron(np.kron(I, X), I)


def encoded_state(alpha: complex, beta: complex) -> np.ndarray:
    state = np.zeros(8, dtype=complex)
    state[0] = alpha  # |000>
    state[7] = beta   # |111>
    return state


def syndrome_probabilities_from_state(state: np.ndarray) -> dict[str, float]:
    probabilities = {"00": 0.0, "10": 0.0, "11": 0.0, "01": 0.0}
    for basis_index, amplitude in enumerate(state):
        q0 = (basis_index >> 2) & 1
        q1 = (basis_index >> 1) & 1
        q2 = basis_index & 1
        syndrome = f"{q0 ^ q1}{q1 ^ q2}"
        probabilities[syndrome] += float(abs(amplitude) ** 2)
    return probabilities


def test_statevector_oracle_matches_closed_form_for_arbitrary_logical_state():
    alpha = np.sqrt(0.3)
    beta = np.exp(0.37j) * np.sqrt(0.7)
    logical_state = encoded_state(alpha, beta)

    for epsilon in (0.0, 0.2, 0.8, np.pi):
        faulty_state = (
            np.cos(epsilon / 2) * logical_state
            - 1j * np.sin(epsilon / 2) * (X1 @ logical_state)
        )
        observed = syndrome_probabilities_from_state(faulty_state)
        expected = {
            "00": np.cos(epsilon / 2) ** 2,
            "10": 0.0,
            "11": np.sin(epsilon / 2) ** 2,
            "01": 0.0,
        }

        for syndrome in expected:
            np.testing.assert_allclose(
                observed[syndrome],
                expected[syndrome],
                atol=1e-12,
            )
