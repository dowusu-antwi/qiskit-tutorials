print("Importing qiskit materials...")


from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer
from qiskit.providers.aer import noise # imports AER noise model
from qiskit.ignis.mitigation.measurement import (complete_meas_cal,
                                                 CompleteMeasFitter,
                                                 MeasurementFilter)
import qiskit.execute

# Generates noise model for qubits (percentages?).
aer_noise_model = noise.NoiseModel()
error = [[0.75, 0.25], [0.1, 0.9]]
for qb in range(5):
    readout_error = noise.errors.readout_error.ReadoutError(error)
    aer_noise_model.add_readout_error(readout_error, [qb])

# Generates measurement calibration circuits for running measurement error
#  mitigation.
q_register = QuantumRegister(5)
calibration_circuits, state_labels = complete_meas_cal(qubit_list=[2,3,4],
                                                       qr=q_register)


# Executes calibration circuits.
qasm_backend = Aer.get_backend("qasm_simulator") # simulates circuit
job = qiskit.execute(calibration_circuits, backend=qasm_backend, shots=1000,
                     noise_model=aer_noise_model)
calibration_results = job.result()

# Makes calibration matrix.
measure_fitter = CompleteMeasFitter(calibration_results, state_labels)


# Makes a 3-qubit GHZ state (a particular entangled state, at least 3 qubits).
# 1. applies Hadamard, putting qubit into superposition
# 2. applies CX (CNOT), putting qubits into Bell state
# 3. applies CX (CNOT), putting qubits into GHZ state
c_register = ClassicalRegister(3)
ghz = QuantumCircuit(q_register, c_register)
ghz.h(q_register[2]) # ghz.h(qubit); Hadamard gate
ghz.cx(q_register[2], q_register[3]) # ghz.cx(control_qubit, target_qubit); CNOT
ghz.cx(q_register[3], q_register[4])
ghz.measure(q_register[2], c_register[0]) # ghz.measure(qubit, cbit):
                                          #  qubit: quantum register,
                                          #  cbit: classical register, 
                                          # measures qubit into classical bit
ghz.measure(q_register[3], c_register[1])
ghz.measure(q_register[4], c_register[2])


# Executes GHZ circuit (with same noise model).
job = qiskit.execute(ghz, backend=qasm_backend, shots=1000,
                     noise_model=aer_noise_model)
ghz_results = job.result()


# Obtains raw results (i.e., prior to mitigation).
raw_counts = ghz_results.get_counts()
print("Raw results (prior to mitigation): ", raw_counts)


# Creates measurement filter and performs mitigation on measurement errors, 
#  applying to raw counts.
measurement_filter = measure_fitter.filter
mitigated_counts = measurement_filter.apply(raw_counts)
print("Mitigated results: ", {bitstring: int(mitigated_counts[bitstring]) 
                              for bitstring in mitigated_counts})


if __name__ == "__main__":
    pass
