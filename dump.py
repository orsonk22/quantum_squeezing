from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def deutsch_oracle(case='constant'):
    qc = QuantumCircuit(2)    # creates a quantum circuit with 2 qubits, one is input and other is output
    if case == 'balanced':
        qc.cx(0, 1)  # f(x) = x   first one is control qubit and second one is target qubit
    elif case == 'constant':
        # Uncomment one of the following depending on f(x) = 0 or f(x) = 1
        # f(x) = 0 → do nothing
        # f(x) = 1 → flip the second qubit
        qc.x(1)  # this means q2 is always l1> --- could remove this line to return const. l0>
    return qc   # this return is a circuit object

def deutsch_circuit(oracle_gate):
    qc = QuantumCircuit(2, 1)  # 2 qubits and 1 classical bit -- classical bit is used to store the result
    
    # initialise qubits: |0⟩ x |1⟩
    qc.x(1)
    qc.h([0, 1])
    
    # Apply the oracle
    qc.append(oracle_gate.to_gate(), [0, 1])
    
    # Apply Hadamard to the first qubit
    qc.h(0)
    
    # Measure the first qubit
    qc.measure(0, 0)
    
    return qc

def run_circuit(qc):
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator, shots=1024).result()
    counts = result.get_counts()
    plot_histogram(counts)
    plt.show()
    return counts


oracle = deutsch_oracle('constant')
qc = deutsch_circuit(oracle)
counts = run_circuit(qc)
print("Result:", counts)
