import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp

# --- Constants ---
T1 = 50e-6
T2 = 20e-6
pulse_duration = 20e-6
pulse_amplitude = 25e6
n_qubits = 2

# --- Pauli Matrices ---
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
P0 = np.array([[1, 0], [0, 0]])
P1 = np.array([[0, 0], [0, 1]])

def kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

def get_operator(pauli, qubit, n):
    ops = [I] * n
    ops[qubit] = pauli
    return kron_all(ops)

def get_hamiltonian(n, pulse_fn, t):
    H = np.zeros((2**n, 2**n), dtype=complex)
    for q in range(n):
        H += pulse_fn(t) * get_operator(X, q, n)
    return H

def lindblad_rhs(t, rho_vec, n, pulse_fn):
    rho = rho_vec.reshape((2**n, 2**n))
    H = get_hamiltonian(n, pulse_fn, t)
    d_rho = -1j * (H @ rho - rho @ H)

    # Decoherence
    for q in range(n):
        sm = get_operator(np.array([[0, 1], [0, 0]]), q, n)
        sz = get_operator(Z, q, n)

        # T1 relaxation
        d_rho += (1/T1) * (sm @ rho @ sm.conj().T - 0.5 * (sm.conj().T @ sm @ rho + rho @ sm.conj().T @ sm))

        # T2 dephasing
        d_rho += (1/T2) * (sz @ rho @ sz - rho)

    return d_rho.reshape(-1)

def partial_trace(rho, keep, dims):
    dim_keep = np.prod([dims[i] for i in keep])
    dim_trace = np.prod([dims[i] for i in range(len(dims)) if i not in keep])
    rho = rho.reshape([dim_keep, dim_trace, dim_keep, dim_trace])
    return np.trace(rho, axis1=1, axis2=3)

def von_neumann_entropy(rho):
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-12]  # avoid log(0)
    return -np.sum(evals * np.log2(evals))

def gaussian_pulse(t):
    t0 = pulse_duration / 2
    sigma = pulse_duration / 6
    return pulse_amplitude * np.exp(-0.5 * ((t - t0) / sigma) ** 2)

def run_sim():
    rho0 = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    rho0[0, 0] = 1  # ground state
    t_span = (0, pulse_duration)
    t_eval = np.linspace(*t_span, 500)

    result = solve_ivp(
        lindblad_rhs,
        t_span,
        rho0.reshape(-1),
        args=(n_qubits, gaussian_pulse),
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8
    )

    entropies = []
    for i in range(len(t_eval)):
        rho = result.y[:, i].reshape((2**n_qubits, 2**n_qubits))
        rho_sub = partial_trace(rho, keep=[0], dims=[2]*n_qubits)
        S = von_neumann_entropy(rho_sub)
        entropies.append(S)

    print("Final entanglement entropy of qubit 0:", entropies[-1])
    print("Max entanglement entropy:", max(entropies))

if __name__ == '__main__':
    run_sim()
