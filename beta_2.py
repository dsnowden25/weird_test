import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import logm, expm
import time
import warnings
warnings.filterwarnings('ignore')

print("QUANTUM ALGORITHM IMPLEMENTATION TEST - ROBUST VERSION")
print("WITH COMPREHENSIVE ERROR CHECKING AND VALIDATION")
print("Testing whether continuous electrical control can actually implement")
print("quantum gates and algorithms better than pulsed control")
print("=" * 70)

class QuantumGateSystem:
    """
    Test quantum gate implementation with different electrical control methods
    Includes comprehensive validation and adjustable parameters
    """
    def __init__(self, qubit_type='superconducting', validation_level='strict'):
        self.qubit_type = qubit_type
        self.validation_level = validation_level
        
        # Platform-specific parameters (CONSERVATIVE, REALISTIC VALUES)
        if qubit_type == 'superconducting':
            self.w_base = 5.0        # 5 GHz
            self.anharmonicity = -0.3  # -300 MHz
            self.g_max = 50.0        # 50 MHz max coupling (reduced from 100)
            self.T1 = 80.0           # 80 μs (realistic)
            self.T2 = 30.0           # 30 μs (realistic T2 < T1)
            self.charge_noise = 0.002 # Increased noise
        elif qubit_type == 'quantum_dot':
            self.w_base = 10.0       # 10 GHz (higher frequency)
            self.anharmonicity = -0.1  # Weaker anharmonicity
            self.g_max = 20.0        # 20 MHz coupling (reduced)
            self.T1 = 500.0          # 500 μs (reduced from 1ms)
            self.T2 = 50.0           # 50 μs (charge noise limited)
            self.charge_noise = 0.01   # Higher charge noise
        elif qubit_type == 'trapped_ion':
            self.w_base = 1.0        # 1 GHz (lower frequency)
            self.anharmonicity = 0.0   # Natural two-level system
            self.g_max = 1.0         # 1 MHz coupling (weaker)
            self.T1 = 1000.0         # 1 ms (reduced from 10ms)
            self.T2 = 200.0          # 200 μs
            self.charge_noise = 0.0005 # Low noise but not zero
        
        # Electrical circuit parameters
        self.R_circuit = 50.0
        self.bandwidth = 1.0  # Control bandwidth in GHz
        
        # Validation parameters
        self.max_coherence_threshold = 0.5  # Flag if coherence > 50%
        self.max_voltage_threshold = 1.0    # Flag if voltage > 1V
        self.max_simulation_time = 200.0    # Stop if simulation takes too long

    def validate_state_vector(self, y, context="unknown"):
        """Comprehensive state vector validation"""
        issues = []
        
        # Check for NaN or infinite values
        if not np.all(np.isfinite(y)):
            issues.append(f"Non-finite values detected in {context}")
        
        # Check electrical voltages
        n_qubits = len([x for x in y[:8:2] if True])  # Count voltage channels
        voltages = y[::2][:n_qubits]
        if np.any(np.abs(voltages) > self.max_voltage_threshold):
            max_v = np.max(np.abs(voltages))
            issues.append(f"Unrealistic voltage {max_v:.3f}V in {context}")
        
        # Check quantum state normalization
        if len(y) > 4:  # Has quantum part
            try:
                rho_vec = y[4:]  # Skip electrical part
                dim = int(np.sqrt(len(rho_vec) + 1))  # Estimate dimension
                if dim**2 - dim == len(rho_vec):  # Correct size for density matrix
                    rho = vector_to_density_matrix(rho_vec, dim)
                    trace = np.trace(rho)
                    if abs(trace - 1.0) > 0.1:
                        issues.append(f"Density matrix trace {trace:.3f} != 1 in {context}")
                    
                    # Check positivity
                    eigenvals = np.linalg.eigvals(rho)
                    if np.any(np.real(eigenvals) < -0.1):
                        min_eval = np.min(np.real(eigenvals))
                        issues.append(f"Negative eigenvalue {min_eval:.3f} in {context}")
            except:
                issues.append(f"Quantum state validation failed in {context}")
        
        return issues

    def validate_physical_parameters(self):
        """Check if system parameters are physically reasonable"""
        issues = []
        
        # Check decoherence hierarchy: T2 ≤ 2*T1
        if self.T2 > 2 * self.T1:
            issues.append(f"Unphysical T2={self.T2} > 2*T1={2*self.T1}")
        
        # Check coupling strength vs frequency
        if self.g_max > self.w_base * 1000:  # MHz vs GHz
            issues.append(f"Coupling {self.g_max}MHz > frequency {self.w_base}GHz")
        
        # Check noise levels
        if self.charge_noise > 0.1:
            issues.append(f"Unrealistically high charge noise {self.charge_noise}")
        
        return issues

    def target_gates(self):
        """Define target quantum gates for testing"""
        # Single qubit gates
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        
        # Rotation gates
        def RX(theta):
            return np.cos(theta/2) * I - 1j * np.sin(theta/2) * X
        
        def RY(theta):
            return np.cos(theta/2) * I - 1j * np.sin(theta/2) * Y
        
        def RZ(theta):
            return np.cos(theta/2) * I - 1j * np.sin(theta/2) * Z
        
        # Two-qubit gates
        CNOT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        
        return {
            'I': I, 'X': X, 'Y': Y, 'Z': Z, 'H': H,
            'RX_pi_2': RX(np.pi/2), 'RY_pi_2': RY(np.pi/2),
            'CNOT': CNOT
        }

def density_matrix_to_vector(rho):
    """Convert density matrix to real vector with validation"""
    dim = rho.shape[0]
    
    # Validate input
    if not np.allclose(rho, rho.conj().T, atol=1e-8):
        print(f"WARNING: Non-Hermitian density matrix, max error = {np.max(np.abs(rho - rho.conj().T)):.2e}")
    
    trace = np.trace(rho)
    if abs(trace - 1.0) > 0.01:
        print(f"WARNING: Density matrix trace = {trace:.6f} ≠ 1")
    
    vec = []
    for i in range(dim):
        for j in range(i, dim):
            if i == j:
                val = np.real(rho[i, j])
                if abs(np.imag(rho[i, j])) > 1e-8:
                    print(f"WARNING: Diagonal element ({i},{j}) has imaginary part {np.imag(rho[i, j]):.2e}")
                vec.append(val)
            else:
                vec.extend([np.real(rho[i, j]), np.imag(rho[i, j])])
    
    return vec

def vector_to_density_matrix(vec, dim):
    """Convert real vector back to density matrix with validation"""
    expected_size = dim + (dim * (dim - 1))  # diagonal + off-diagonal pairs
    if len(vec) != expected_size:
        print(f"ERROR: Vector size {len(vec)} != expected {expected_size} for {dim}x{dim} matrix")
        return np.eye(dim, dtype=complex) / dim  # Return maximally mixed state
    
    rho = np.zeros((dim, dim), dtype=complex)
    idx = 0
    for i in range(dim):
        for j in range(i, dim):
            if i == j:
                rho[i, j] = vec[idx]
                idx += 1
            else:
                rho[i, j] = vec[idx] + 1j * vec[idx + 1]
                rho[j, i] = vec[idx] - 1j * vec[idx + 1]
                idx += 2
    
    return rho

def realistic_noise_model(t, system):
    """Generate realistic time-correlated noise"""
    # 1/f charge noise
    charge_noise = system.charge_noise * (
        np.sin(0.01 * t) / 0.01 +
        np.sin(0.1 * t) / 0.1 + 
        np.sin(1.0 * t) / 1.0
    ) / 3.0
    
    # High frequency noise (thermal, shot noise)
    hf_noise = 0.0001 * np.sin(50.0 * t) * np.exp(-0.01 * t)
    
    return charge_noise + hf_noise

def simplified_gate_dynamics(t, y, system, control_params, n_qubits):
    """
    Simplified dynamics for gate implementation testing with validation
    """
    try:
        dim = 2**n_qubits
        n_electrical = 2 * n_qubits
        
        # Validate input state
        issues = system.validate_state_vector(y, f"t={t:.3f}")
        if issues and system.validation_level == 'strict':
            for issue in issues[:3]:  # Report first 3 issues
                print(f"   VALIDATION WARNING: {issue}")
        
        # Extract states with bounds checking
        if len(y) < n_electrical:
            print(f"ERROR: State vector too short: {len(y)} < {n_electrical}")
            return [0.0] * len(y)
        
        V_channels = y[::2][:n_qubits]
        I_channels = y[1::2][:n_qubits]
        rho_vec = y[n_electrical:]
        
        # Generate control signal with realistic limitations
        if control_params['type'] == 'continuous':
            amp = min(control_params['amplitude'], 0.2)  # Limit max amplitude
            freq = control_params['frequency']
            drive_signal = amp * np.sin(freq * t)
        else:  # pulsed
            amp = min(control_params['amplitude'], 0.3)  # Limit max amplitude
            width = control_params['pulse_width']
            n_pulses = control_params['n_pulses']
            pulse_period = control_params['gate_time'] / n_pulses
            cycle_time = t % pulse_period
            drive_signal = amp if cycle_time < width else 0.0
        
        # Add realistic noise
        noise = realistic_noise_model(t, system)
        drive_signal += noise
        
        # Simple electrical circuit per channel with damping
        electrical_derivatives = []
        for i in range(n_qubits):
            # Add realistic control electronics effects
            bandwidth_factor = np.exp(-abs(drive_signal - 0.05))
            actual_drive = system.bandwidth * drive_signal * bandwidth_factor
            
            # RLC circuit with proper damping
            damping = system.w_base / 1000.0  # Q ~ 1000
            dV_dt = I_channels[i]
            dI_dt = (-system.w_base**2 * V_channels[i] - 
                    damping * I_channels[i] + actual_drive)
            
            electrical_derivatives.extend([dV_dt, dI_dt])
        
        # Quantum evolution with electrical coupling
        try:
            rho = vector_to_density_matrix(rho_vec, dim)
        except Exception as e:
            print(f"ERROR: Density matrix reconstruction failed at t={t:.3f}: {e}")
            return [0.0] * len(y)
        
        # Check for unrealistic coherence
        if dim == 2:  # Single qubit
            coherence = 2 * abs(rho[0, 1])
            if coherence > system.max_coherence_threshold:
                print(f"WARNING: Suspiciously high coherence {coherence:.4f} at t={t:.3f}")
        
        # Build Hamiltonian with electrical coupling
        H = np.zeros((dim, dim), dtype=complex)
        
        # Single qubit terms with realistic coupling
        for i in range(n_qubits):
            # Scale coupling realistically (MHz to energy units)
            coupling_strength = system.g_max * V_channels[i] * 0.0001  # Much smaller scaling
            
            # Build Pauli X operator for qubit i
            sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
            op = np.array([[1]], dtype=complex)
            
            for j in range(n_qubits):
                if j == i:
                    op = np.kron(op, sigma_x)
                else:
                    op = np.kron(op, np.eye(2, dtype=complex))
            
            H += coupling_strength * op
            
            # Add static qubit frequency
            sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
            op_z = np.array([[1]], dtype=complex)
            
            for j in range(n_qubits):
                if j == i:
                    op_z = np.kron(op_z, sigma_z)
                else:
                    op_z = np.kron(op_z, np.eye(2, dtype=complex))
            
            H += (system.w_base / 2) * op_z  # Qubit energy
        
        # Add REALISTIC decoherence (this is critical!)
        gamma_1 = 1.0 / system.T1  # Relaxation rate
        gamma_2 = 1.0 / system.T2  # Dephasing rate
        
        # Enhanced decoherence with noise coupling
        noise_factor = 1.0 + abs(noise) * 10.0  # Noise increases decoherence
        gamma_1_eff = gamma_1 * noise_factor
        gamma_2_eff = gamma_2 * noise_factor
        
        decoherence_term = np.zeros_like(rho)
        
        for i in range(n_qubits):
            # Relaxation operator for qubit i
            sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
            op_minus = np.array([[1]], dtype=complex)
            
            for j in range(n_qubits):
                if j == i:
                    op_minus = np.kron(op_minus, sigma_minus)
                else:
                    op_minus = np.kron(op_minus, np.eye(2, dtype=complex))
            
            # Apply Lindblad term with enhanced rates
            decoherence_term += gamma_1_eff * (
                op_minus @ rho @ op_minus.conj().T - 
                0.5 * (op_minus.conj().T @ op_minus @ rho + rho @ op_minus.conj().T @ op_minus)
            )
            
            # Dephasing
            sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
            op_z = np.array([[1]], dtype=complex)
            
            for j in range(n_qubits):
                if j == i:
                    op_z = np.kron(op_z, sigma_z)
                else:
                    op_z = np.kron(op_z, np.eye(2, dtype=complex))
            
            decoherence_term += gamma_2_eff * (
                op_z @ rho @ op_z.conj().T - 
                0.5 * (op_z.conj().T @ op_z @ rho + rho @ op_z.conj().T @ op_z)
            )
        
        # Total quantum evolution
        drho_dt = -1j * (H @ rho - rho @ H) + decoherence_term
        
        # Convert back to vector with validation
        try:
            drho_dt_vec = density_matrix_to_vector(drho_dt)
        except Exception as e:
            print(f"ERROR: Derivative conversion failed at t={t:.3f}: {e}")
            return [0.0] * len(y)
        
        total_derivatives = electrical_derivatives + drho_dt_vec
        
        # Final validation
        if not np.all(np.isfinite(total_derivatives)):
            print(f"ERROR: Non-finite derivatives at t={t:.3f}")
            return [0.0] * len(y)
        
        # Check for exploding derivatives
        max_deriv = np.max(np.abs(total_derivatives))
        if max_deriv > 1000.0:
            print(f"WARNING: Large derivatives {max_deriv:.2e} at t={t:.3f}")
        
        return total_derivatives
        
    except Exception as e:
        print(f"CRITICAL ERROR in dynamics at t={t:.3f}: {e}")
        import traceback
        traceback.print_exc()
        return [0.0] * len(y)

def gate_fidelity(target_unitary, implemented_unitary):
    """Calculate quantum gate fidelity with validation"""
    try:
        dim = target_unitary.shape[0]
        
        # Validate unitarity of target
        target_check = target_unitary @ target_unitary.conj().T
        if not np.allclose(target_check, np.eye(dim), atol=1e-6):
            print("WARNING: Target gate not unitary")
        
        # Process fidelity: F = |Tr(U_target† U_implemented)|² / d²
        if implemented_unitary is None:
            return 0.0
        
        overlap = np.trace(target_unitary.conj().T @ implemented_unitary)
        fidelity = abs(overlap)**2 / dim**2
        
        # Validate fidelity range
        if not 0 <= fidelity <= 1:
            print(f"WARNING: Fidelity {fidelity:.6f} outside [0,1]")
            return max(0, min(1, fidelity))
        
        return np.real(fidelity)
        
    except Exception as e:
        print(f"Fidelity calculation failed: {e}")
        return 0.0

def simulate_gate_implementation(target_gate, system, control_params):
    """
    Simulate implementing a specific quantum gate with comprehensive checking
    """
    dim = target_gate.shape[0]
    n_qubits = int(np.log2(dim))
    gate_time = control_params['gate_time']
    
    print(f"      Simulating {dim}x{dim} gate over {gate_time:.1f}μs...")
    
    # Validate gate time
    if gate_time > system.T2:
        print(f"      WARNING: Gate time {gate_time:.1f}μs > T2 {system.T2:.1f}μs")
    
    # Test on computational basis states
    implemented_states = []
    success_count = 0
    
    for state_idx in range(min(dim, 2)):  # Test first 2 states for speed
        try:
            print(f"         Testing on basis state |{state_idx:0{n_qubits}b}⟩...")
            
            # Initial state
            initial_state = np.zeros(dim, dtype=complex)
            initial_state[state_idx] = 1.0
            
            # Set up electrical + quantum state
            electrical_initial = [0.001, 0.0] * n_qubits  # Small initial voltage
            rho_initial = np.outer(initial_state, initial_state.conj())
            rho_vec = density_matrix_to_vector(rho_initial)
            full_initial = electrical_initial + rho_vec
            
            # Validate initial state
            init_issues = system.validate_state_vector(full_initial, "initial")
            if init_issues:
                print(f"         Initial state issues: {init_issues}")
            
            # Define dynamics with progress tracking
            call_count = [0]
            last_progress = [0]
            
            def gate_dynamics(t, y):
                call_count[0] += 1
                
                # Progress tracking every 1000 calls
                if call_count[0] % 1000 == 0:
                    progress = int(100 * t / gate_time)
                    if progress > last_progress[0] + 10:
                        print(f"         Progress: {progress}%")
                        last_progress[0] = progress
                
                return simplified_gate_dynamics(t, y, system, control_params, n_qubits)
            
            # Simulate gate operation with conservative tolerances
            sol = solve_ivp(gate_dynamics, [0, gate_time], full_initial,
                           method='LSODA', rtol=1e-6, atol=1e-8,
                           max_step=gate_time/100)  # Limit step size
            
            if sol.success:
                # Extract final state
                final_rho_vec = sol.y[2*n_qubits:, -1]
                final_rho = vector_to_density_matrix(final_rho_vec, dim)
                
                # Validate final state
                final_issues = system.validate_state_vector(sol.y[:, -1], "final")
                if final_issues:
                    print(f"         Final state issues: {final_issues}")
                
                # Extract final state vector (get dominant eigenstate)
                eigenvals, eigenvecs = np.linalg.eigh(final_rho)
                eigenvals = np.real(eigenvals)
                max_idx = np.argmax(eigenvals)
                max_eigenval = eigenvals[max_idx]
                
                if max_eigenval < 0.5:
                    print(f"         WARNING: Largest eigenvalue only {max_eigenval:.3f}")
                
                final_state = eigenvecs[:, max_idx]
                
                # Check state normalization
                norm = np.linalg.norm(final_state)
                if abs(norm - 1.0) > 0.1:
                    print(f"         WARNING: Final state norm = {norm:.3f}")
                    final_state = final_state / norm
                
                implemented_states.append(final_state)
                success_count += 1
                print(f"         ✓ Success (eigenval={max_eigenval:.3f})")
                
            else:
                print(f"         ✗ Integration failed: {sol.message}")
                # Add zero state to maintain indexing
                implemented_states.append(np.zeros(dim, dtype=complex))
                
        except Exception as e:
            print(f"         ✗ Simulation crashed: {e}")
            implemented_states.append(np.zeros(dim, dtype=complex))
    
    if success_count == 0:
        print(f"      All state simulations failed!")
        return 0.0
    
    # Calculate gate fidelity from input-output mapping
    if len(implemented_states) >= 2:
        try:
            # Construct implemented unitary from successful states
            U_implemented = np.column_stack(implemented_states)
            
            # Validate unitary properties
            unitarity_check = U_implemented @ U_implemented.conj().T
            unitarity_error = np.max(np.abs(unitarity_check - np.eye(dim)))
            if unitarity_error > 0.1:
                print(f"      WARNING: Implemented operation not unitary, error={unitarity_error:.3f}")
            
            # Calculate fidelity with target gate
            fidelity = gate_fidelity(target_gate, U_implemented)
            
            # Sanity check fidelity
            if fidelity > 0.99:
                print(f"      WARNING: Suspiciously high fidelity {fidelity:.6f}")
            elif fidelity < 0.01:
                print(f"      WARNING: Very low fidelity {fidelity:.6f}")
            
            print(f"      Gate fidelity: {fidelity:.4f} (unitarity error: {unitarity_error:.3f})")
            return fidelity
            
        except Exception as e:
            print(f"      Fidelity calculation failed: {e}")
            return 0.0
    else:
        return 0.0

def implement_gate_continuous(target_gate, system, gate_time=2.0, optimization_steps=4):
    """
    Find optimal continuous electrical control to implement target gate
    """
    dim = target_gate.shape[0]
    n_qubits = int(np.log2(dim))
    print(f"   Implementing {dim}x{dim} gate with continuous control...")
    
    # Validate system parameters first
    param_issues = system.validate_physical_parameters()
    if param_issues:
        print(f"   System parameter issues: {param_issues}")
    
    best_fidelity = 0.0
    best_params = None
    
    # Conservative parameter search
    amplitudes = np.linspace(0.005, 0.05, optimization_steps)  # Much smaller amplitudes
    frequencies = np.linspace(system.w_base * 0.95, system.w_base * 1.05, optimization_steps)
    
    for i, amp in enumerate(amplitudes):
        for j, freq in enumerate(frequencies):
            print(f"      Testing amp={amp:.4f}, freq={freq:.2f}GHz ({i*len(frequencies)+j+1}/{len(amplitudes)*len(frequencies)})")
            
            try:
                control_params = {
                    'type': 'continuous',
                    'amplitude': amp,
                    'frequency': freq,
                    'gate_time': gate_time
                }
                
                fidelity = simulate_gate_implementation(target_gate, system, control_params)
                
                if fidelity > best_fidelity:
                    best_fidelity = fidelity
                    best_params = control_params.copy()
                    print(f"      ✓ New best: {fidelity:.4f}")
                
            except Exception as e:
                print(f"      ✗ Parameter combination failed: {e}")
                continue
    
    print(f"   Best continuous fidelity: {best_fidelity:.4f}")
    return best_fidelity, best_params

def implement_gate_pulsed(target_gate, system, n_pulses=3):
    """
    Implement target gate using optimized pulse sequence
    """
    dim = target_gate.shape[0]
    print(f"   Implementing {dim}x{dim} gate with pulsed control...")
    
    best_fidelity = 0.0
    best_params = None
    
    # Conservative pulse parameters
    pulse_widths = [0.1, 0.3, 0.5]
    pulse_amplitudes = [0.02, 0.05, 0.08]  # Smaller amplitudes
    
    for i, width in enumerate(pulse_widths):
        for j, amp in enumerate(pulse_amplitudes):
            print(f"      Testing width={width:.1f}μs, amp={amp:.3f} ({i*len(pulse_amplitudes)+j+1}/{len(pulse_widths)*len(pulse_amplitudes)})")
            
            try:
                control_params = {
                    'type': 'pulsed',
                    'amplitude': amp,
                    'pulse_width': width,
                    'n_pulses': n_pulses,
                    'gate_time': n_pulses * width * 3  # Include spacing
                }
                
                fidelity = simulate_gate_implementation(target_gate, system, control_params)
                
                if fidelity > best_fidelity:
                    best_fidelity = fidelity
                    best_params = control_params.copy()
                    print(f"      ✓ New best: {fidelity:.4f}")
                    
            except Exception as e:
                print(f"      ✗ Parameter combination failed: {e}")
                continue
    
    print(f"   Best pulsed fidelity: {best_fidelity:.4f}")
    return best_fidelity, best_params

def test_quantum_gates_across_platforms():
    """
    Test gate implementation across different qubit platforms
    """
    platforms = ['superconducting']  # Start with one platform for debugging
    gate_results = {}
    
    for platform in platforms:
        print(f"\n1. Testing {platform} qubits...")
        system = QuantumGateSystem(platform, validation_level='strict')
        
        # Test single-qubit gates only
        target_gates = system.target_gates()
        platform_results = {}
        
        # Test minimal set of gates
        test_gates = ['X', 'H']  # Just two gates for initial testing
        
        for gate_name in test_gates:
            print(f"   Testing {gate_name} gate:")
            target = target_gates[gate_name]
            
            try:
                # Continuous wave implementation
                print("      Continuous wave optimization...")
                cw_fidelity, cw_params = implement_gate_continuous(target, system)
                
                # Pulsed implementation
                print("      Pulsed sequence optimization...")
                pulse_fidelity, pulse_params = implement_gate_pulsed(target, system)
                
                platform_results[gate_name] = {
                    'continuous': {'fidelity': cw_fidelity, 'params': cw_params},
                    'pulsed': {'fidelity': pulse_fidelity, 'params': pulse_params}
                }
                
                print(f"      FINAL: CW={cw_fidelity:.4f}, Pulsed={pulse_fidelity:.4f}")
                
                # Flag suspicious results
                if cw_fidelity > 0.95 or pulse_fidelity > 0.95:
                    print(f"      WARNING: Suspiciously high fidelity detected!")
                
            except Exception as e:
                print(f"      Gate {gate_name} test FAILED: {e}")
                platform_results[gate_name] = {
                    'continuous': {'fidelity': 0.0, 'params': None},
                    'pulsed': {'fidelity': 0.0, 'params': None}
                }
        
        gate_results[platform] = platform_results
    
    return gate_results

def test_simple_coherence_comparison():
    """
    Simple coherence comparison as sanity check
    """
    print("\n2. Simple Coherence Comparison (Sanity Check)...")
    
    system = QuantumGateSystem('superconducting')
    
    # Simple 2-level system initial state: |0⟩
    electrical_initial = [0.001, 0.0]  # Single qubit
    rho_initial = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)  # |0⟩⟨0|
    rho_vec = density_matrix_to_vector(rho_initial)
    full_initial = electrical_initial + rho_vec
    
    sim_time = 10.0  # 10 μs
    
    def test_coherence(control_type, params):
        try:
            print(f"   Testing {control_type} control...")
            
            def dynamics(t, y):
                return simplified_gate_dynamics(t, y, system, params, 1)
            
            sol = solve_ivp(dynamics, [0, sim_time], full_initial,
                           t_eval=np.linspace(0, sim_time, 100),
                           method='RK45', rtol=1e-6)
            
            if sol.success:
                # Extract coherence evolution
                coherences = []
                for time_idx in range(len(sol.t)):
                    rho_vec = sol.y[2:, time_idx]
                    rho = vector_to_density_matrix(rho_vec, 2)
                    coherence = 2 * abs(rho[0, 1])
                    coherences.append(coherence)
                
                avg_coherence = np.mean(coherences[20:])  # Skip transients
                max_coherence = np.max(coherences)
                
                print(f"      Average coherence: {avg_coherence:.6f}")
                print(f"      Maximum coherence: {max_coherence:.6f}")
                
                # Flag unrealistic values
                if max_coherence > 0.5:
                    print(f"      WARNING: Very high coherence {max_coherence:.4f}")
                
                return avg_coherence, max_coherence
            else:
                print(f"      Integration failed: {sol.message}")
                return 0.0, 0.0
                
        except Exception as e:
            print(f"      Coherence test failed: {e}")
            return 0.0, 0.0
    
    # Test both control methods
    cw_params = {'type': 'continuous', 'amplitude': 0.01, 'frequency': 5.0, 'gate_time': sim_time}
    pulse_params = {'type': 'pulsed', 'amplitude': 0.03, 'pulse_width': 0.2, 'n_pulses': 3, 'gate_time': sim_time}
    
    cw_avg, cw_max = test_coherence('continuous', cw_params)
    pulse_avg, pulse_max = test_coherence('pulsed', pulse_params)
    
    print(f"   COHERENCE COMPARISON:")
    print(f"   Continuous: avg={cw_avg:.6f}, max={cw_max:.6f}")
    print(f"   Pulsed: avg={pulse_avg:.6f}, max={pulse_max:.6f}")
    
    return cw_avg, pulse_avg

def run_comprehensive_gate_test():
    """
    Run complete gate and algorithm testing with validation
    """
    print("COMPREHENSIVE QUANTUM GATE TESTING - ROBUST VERSION")
    print("WITH EXTENSIVE ERROR CHECKING AND VALIDATION")
    print("=" * 60)
    
    start_total = time.time()
    
    try:
        # Start with simple coherence test
        cw_coherence, pulse_coherence = test_simple_coherence_comparison()
        
        # Only proceed with gate tests if coherence test was reasonable
        if cw_coherence < 0.01 and pulse_coherence < 0.01:
            print("WARNING: Very low coherences detected. System may have issues.")
        elif cw_coherence > 0.5 or pulse_coherence > 0.5:
            print("WARNING: Unrealistically high coherences. Check simulation validity.")
        
        # Test gates across platforms
        print("Testing quantum gates...")
        gate_results = test_quantum_gates_across_platforms()
        
        # Analysis and reporting
        print("\n3. Analyzing comprehensive gate test results...")
        
        with open('quantum_algorithm_results.txt', 'w') as f:
            f.write("QUANTUM GATE PERFORMANCE TEST - ROBUST VERSION\n")
            f.write("=" * 55 + "\n\n")
            
            f.write("SIMPLE COHERENCE COMPARISON:\n")
            f.write(f"Continuous Wave Average Coherence: {cw_coherence:.6f}\n")
            f.write(f"Pulsed Drive Average Coherence: {pulse_coherence:.6f}\n")
            if pulse_coherence > 0:
                f.write(f"Coherence Ratio (CW/Pulsed): {cw_coherence/pulse_coherence:.3f}\n")
            f.write("\n")
            
            f.write("QUANTUM GATE FIDELITIES BY PLATFORM:\n")
            
            overall_cw_fidelities = []
            overall_pulse_fidelities = []
            
            for platform, results in gate_results.items():
                f.write(f"\n{platform.upper()} QUBITS:\n")
                total_cw_fidelity = 0.0
                total_pulse_fidelity = 0.0
                gate_count = 0
                
                for gate_name, gate_data in results.items():
                    cw_fid = gate_data['continuous']['fidelity']
                    pulse_fid = gate_data['pulsed']['fidelity']
                    f.write(f"  {gate_name}: CW={cw_fid:.4f}, Pulsed={pulse_fid:.4f}")
                    
                    if cw_fid > 0 and pulse_fid > 0:
                        ratio = cw_fid / pulse_fid
                        f.write(f" (ratio: {ratio:.3f})")
                        
                        # Flag suspicious ratios
                        if ratio > 10 or ratio < 0.1:
                            f.write(" [SUSPICIOUS RATIO]")
                    
                    f.write("\n")
                    
                    total_cw_fidelity += cw_fid
                    total_pulse_fidelity += pulse_fid
                    overall_cw_fidelities.append(cw_fid)
                    overall_pulse_fidelities.append(pulse_fid)
                    gate_count += 1
                
                if gate_count > 0:
                    avg_cw = total_cw_fidelity / gate_count
                    avg_pulse = total_pulse_fidelity / gate_count
                    f.write(f"  Platform Average: CW={avg_cw:.4f}, Pulsed={avg_pulse:.4f}\n")
            
            f.write(f"\nOVERALL ASSESSMENT:\n")
            
            # Calculate overall performance with validation
            valid_cw_scores = [s for s in overall_cw_fidelities if s > 0.001]  # Filter near-zero
            valid_pulse_scores = [s for s in overall_pulse_fidelities if s > 0.001]
            
            if len(valid_cw_scores) > 0 and len(valid_pulse_scores) > 0:
                avg_cw_overall = np.mean(valid_cw_scores)
                avg_pulse_overall = np.mean(valid_pulse_scores)
                
                f.write(f"Average CW Performance: {avg_cw_overall:.4f}\n")
                f.write(f"Average Pulsed Performance: {avg_pulse_overall:.4f}\n")
                f.write(f"Overall Improvement Factor: {avg_cw_overall/avg_pulse_overall:.3f}\n")
                f.write(f"Number of valid CW results: {len(valid_cw_scores)}\n")
                f.write(f"Number of valid Pulsed results: {len(valid_pulse_scores)}\n\n")
                
                # Conservative conclusion criteria
                if avg_cw_overall > avg_pulse_overall * 1.2 and avg_cw_overall > 0.1:  # 20% improvement + minimum threshold
                    f.write("PRELIMINARY CONCLUSION: CONTINUOUS ELECTRICAL CONTROL\n")
                    f.write("SHOWS POTENTIAL ADVANTAGES FOR QUANTUM GATES\n\n")
                    f.write("HOWEVER: These results require experimental validation!\n")
                    f.write("Simulation assumptions may not reflect all physical constraints.\n")
                elif avg_cw_overall > avg_pulse_overall * 0.8:
                    f.write("CONCLUSION: Performance comparable between methods\n")
                else:
                    f.write("CONCLUSION: Pulsed control appears superior\n")
                    
                # Add critical validation notes
                f.write("\nVALIDATION NOTES:\n")
                f.write("- All fidelities should be < 0.99 (perfect gates unrealistic)\n")
                f.write("- Coherence should decay over T1/T2 timescales\n")
                f.write("- Large improvement ratios (>5x) may indicate bugs\n")
                f.write("- Results need experimental verification\n")
                
            else:
                f.write("CONCLUSION: Insufficient valid data for comparison\n")
                f.write(f"Valid CW results: {len(valid_cw_scores)}\n")
                f.write(f"Valid Pulsed results: {len(valid_pulse_scores)}\n")
        
        total_time = time.time() - start_total
        print(f"\nROBUST GATE TESTING COMPLETED in {total_time:.1f} seconds")
        print("Results saved to: quantum_algorithm_results.txt")
        
        # Summary with validation flags
        print(f"\nSUMMARY WITH VALIDATION:")
        if len(valid_cw_scores) > 0 and len(valid_pulse_scores) > 0:
            avg_cw = np.mean(valid_cw_scores)
            avg_pulse = np.mean(valid_pulse_scores)
            print(f"Average Gate Fidelity: CW={avg_cw:.4f} vs Pulsed={avg_pulse:.4f}")
            print(f"Improvement Factor: {avg_cw/avg_pulse:.3f}x")
            print(f"Coherence Ratio: {cw_coherence/pulse_coherence:.3f}x")
            
            # Reality check
            if avg_cw > 0.5:
                print("⚠️  WARNING: CW fidelities seem too high for realistic system")
            if avg_pulse > 0.5:
                print("⚠️  WARNING: Pulsed fidelities seem too high for realistic system")
            if avg_cw/avg_pulse > 5.0:
                print("⚠️  WARNING: Improvement ratio seems unrealistically large")
            
        return gate_results, cw_coherence, pulse_coherence
        
    except Exception as e:
        print(f"Comprehensive testing failed: {e}")
        import traceback
        traceback.print_exc()
        return {}, 0.0, 0.0

# Main execution with safety checks
if __name__ == "__main__":
    print("ROBUST QUANTUM GATE TESTING")
    print("This version includes extensive validation to catch bugs")
    print("and ensure results are physically reasonable")
    print("\nTesting:")
    print("- Conservative parameter ranges")
    print("- Comprehensive error checking")
    print("- Physical validity constraints")
    print("- Progress tracking and debugging")
    print("\nEstimated runtime: 5-10 minutes (reduced scope for debugging)")
    print("=" * 70)
    
    try:
        results = run_comprehensive_gate_test()
        print("\n" + "="*70)
        print("ROBUST QUANTUM GATE TEST COMPLETED!")
        print("\nIf results look reasonable, we can expand the scope.")
        print("If results look suspicious, we need to debug further.")
        
    except KeyboardInterrupt:
        print("\n" + "="*70)
        print("TEST INTERRUPTED BY USER")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        print("\nThis helps us identify where the simulation breaks down.")