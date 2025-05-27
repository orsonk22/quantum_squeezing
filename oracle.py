from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram

def oracle_constant_0(qc):
    # f(x) = 0 → do nothing
    pass

def oracle_constant_1(qc):
    # f(x) = 1 → flip the output qubit
    qc.x(1)

def oracle_i(qc):
    # f(x) = x (Identity)
    qc.cx(0, 1)

def oracle_not(qc):
    # f(x) = ¬x (Negation)
    qc.cx(0, 1)
    qc.x(1)

def deutsch_circuit(oracle_func):
    """Construct the Deutsch circuit with 2 qubits and 2 classical bits."""
    # Create a 2-qubit, 2-bit circuit
    qc = QuantumCircuit(2, 2)

    # Step 1: Initialise |1⟩|1⟩
    qc.x(0)
    qc.x(1)  # Set auxiliary qubit to |1⟩

    # Step 2: Apply Hadamard gates to both qubits
    qc.h(0)
    qc.h(1)

    # Step 3: Apply the oracle
    oracle_func(qc)

    # Step 4: Apply Hadamard gates to both qubits
    qc.h(0)
    qc.h(1)

    # Step 5: Measurement of both qubits
    qc.measure(0, 0)
    qc.measure(1, 1)

    return qc

def run_simulation(oracle_func, label):
    """Run the Deutsch algorithm circuit and display the result."""
    qc = deutsch_circuit(oracle_func)
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator, shots=1024).result()
    counts = result.get_counts()
    print(f"{label}: {counts}")
    plot_histogram(counts).show()

# Define the oracles
oracles = {
    "Constant f(x)=0": oracle_constant_0,
    "Constant f(x)=1": oracle_constant_1,
    "Variable f(x)=x": oracle_i,
    "Variable f(x)=¬x": oracle_not
}

# Run the simulation for each oracle
for name, oracle_func in oracles.items():
    run_simulation(oracle_func, name)


