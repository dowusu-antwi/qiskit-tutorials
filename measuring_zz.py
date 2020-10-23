########################### QUESTIONS BEGIN WITH '?' ###########################

import numpy as np
import matplotlib.pyplot as plt

print("===============================================")
print("Importing qiskit modules...")
from qiskit import Aer, execute
from qiskit.providers.aer.noise.errors.standard_errors \
    import coherent_unitary_error
from qiskit.providers.aer.noise import NoiseModel
from qiskit.ignis.characterization.hamiltonian import ZZFitter, zz_circuits
print("Import complete.")

def open_plot():
    '''
    Opens and maximizes the current figure.
    '''
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

## Measuring ZZ

# Note: zz_circuits builds circuits to perform experiment to measure ZZ between 
#  a pair of qubits, where ZZ is energy shift on 11 state.
#
# Experiment: measure 11 state coefficient; perform Ramsey experiment on Q0 (in
#  ground state) and repeat Ramsey with Q1 in excited stae, coefficient given by
#  difference in frequency between experiments
number_of_gates = np.arange(0, 150, 5)    # number of identity gates per circuit
                                          #  ?: how did we get 150? 
gate_time = 0.1                           # time of running on a single gate
                                          # ?: what is the zz rate?

# Selects qubits whose ZZ will be measured
qubits = [0]        # measuring qubit 0?
spectators = [1]    # qubits to "flip the state" (i.e., measure energy shift
                    #  between qubits and spectators)

# Generates experiments.
oscillations_count = 2

circuits, delay_times, oscillation_freq = zz_circuits(number_of_gates,
                                                      gate_time, 
                                                      qubits, spectators,
                                                      nosc = oscillations_count)

## Splits circuits into multiple jobs, giving results to fitter as a list.
# Sets simulator with ZZ
unknown_factor = 0.02 # ?: what is this? ground state energy frequency (E=hw)?
time_evolver = np.exp(1j * 2 * np.pi * unknown_factor * gate_time)
zz_unitary = np.eye(4, dtype=complex)
zz_unitary[3,3] = time_evolver

error = coherent_unitary_error(zz_unitary)    # for applying to qubits in noise
                                              #  model below
my_noise_model = NoiseModel()
error_instructions = 'id'
noise_qubits = qubits + spectators
my_noise_model.add_nonlocal_quantum_error(error, error_instructions, qubits,
                                       noise_qubits)

# Runs the simulator.
backend = Aer.get_backend('qasm_simulator')
trials = 500
print("Running circuits, in two batches of 20 each...")
results = [execute(circuits[:20], backend, shots=trials,
                   noise_model=my_noise_model).result(),
           execute(circuits[20:], backend, shots=trials,
                   noise_model=my_noise_model).result()]

# Fits data to an oscillation
plt.figure(figsize=(10, 6))
a, phi, c = 1, -np.pi/20, 0    #?: what are these?
fitting_parameter_bias = [a, oscillation_freq, phi, c]
parameter_lower_bounds = [-0.5, 0, -np.pi, -0.5]
parameter_upper_bounds = [1.5, 2 * oscillation_freq, np.pi, 1.5]
zz_fit = ZZFitter(results, delay_times, qubits, spectators,
                  fit_p0=fitting_parameter_bias,
                  fit_bounds=(parameter_lower_bounds, parameter_upper_bounds))
zz_fit.plot_ZZ(0, ax=plt.gca())
open_plot()

mhz_energy_shift_rate = zz_fit.ZZ_rate()[0]
print("ZZ rate: %0.2f kHz" % (mhz_energy_shift_rate * 1e3))

if __name__ == "__main__":
    pass
