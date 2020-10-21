import numpy as np
import matplotlib.pyplot as plt

print("===============================================")
print("Importing qiskit modules...")
from qiskit.providers.aer.noise.errors.standard_errors \
    import coherent_unitary_error
from qiskit.providers.aer.noise import NoiseModel

from qiskit.ignis.characterization.hamiltonian import ZZFitter, zz_circuits

from qiskit.ignis.characterization.gates import (AmpCalFitter,
                                                 ampcal_1Q_circuits,
                                                 AngleCalFitter,
                                                 anglecal_1Q_circuits,
                                                 AmpCalCXFitter,
                                                 ampcal_cx_circuits,
                                                 AngleCalCXFitter,
                                                 anglecal_cx_circuits)
print("Import complete.")
print("===============================================")


## Measuring ZZ

# zz_circuits builds circuits to perform experiment to measure ZZ between a
#  pair of qubits, where ZZ is energy shift on 11 state.

# Experiment: measure coefficient; perform Ramsey experiment on Q0


if __name__ == "__main__":
    pass
