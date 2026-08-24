"""Demonstrate ideal syndrome extraction and correction."""

from qec_calibration import (
    EXPECTED_SYNDROMES,
    build_syndrome_circuit,
    run_recovery,
    run_syndrome,
)


def error_label(error_qubit: int | None) -> str:
    return "None" if error_qubit is None else f"X{error_qubit}"


def main() -> None:
    shots = 256

    print("Ideal three-qubit bit-flip syndrome circuit\n")
    print(build_syndrome_circuit(error_qubit=1).draw(output="text"))

    print("\nSyndrome truth table")
    print("error  expected  observed")
    for error_qubit, expected in EXPECTED_SYNDROMES.items():
        observed = run_syndrome(error_qubit=error_qubit, shots=shots)
        print(f"{error_label(error_qubit):<5}  {expected:^8}  {observed}")

    print("\nActive recovery")
    for logical_value in (0, 1):
        for error_qubit in EXPECTED_SYNDROMES:
            observed = run_recovery(
                error_qubit=error_qubit,
                logical_value=logical_value,
                shots=shots,
            )
            print(
                f"logical={logical_value}, error={error_label(error_qubit):<4} "
                f"-> {observed}"
            )


if __name__ == "__main__":
    main()

