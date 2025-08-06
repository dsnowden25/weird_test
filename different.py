import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

class QuantumFieldCouplingSystem:
    """
    PARADIGM SHIFT: Quantum electrical field coupled to quantum matter system
    
    Key Difference from Previous:
    - Before: Classical E-field drives quantum system
    - Now: Quantum E-field entangles with quantum system
    
    This tests the hypothesis that electrical waves are "quantum partners" 
    rather than "classical drivers" - both sides treated as quantum fields.
    """
    
    def __init__(self, n_matter_qubits=3, n_field_modes=4):
        self.n_matter_qubits = n_matter_qubits  # Quantum matter system
        self.n_field_modes = n_field_modes      # Quantum electrical field modes
        
        # Total system: matter ⊗ field
        self.matter_dim = 2**n_matter_qubits
        self.field_dim = n_field_modes + 1  # Fock space: |0⟩, |1⟩, |2⟩, |3⟩, |4⟩
        self.total_dim = self.matter_dim * (self.field_dim**n_field_modes)
        
        # Matter system parameters
        self.omega_matter = 5.0e9  # 5 GHz qubit frequency
        self.J_matter = 20e6       # 20 MHz matter-matter coupling
        
        # Electrical field parameters (treating field as quantum!)
        self.omega_field = 5.1e9   # 5.1 GHz field mode frequency
        self.field_spacing = 100e6 # 100 MHz mode spacing
        
        # QUANTUM FIELD-MATTER COUPLING (the key innovation!)
        self.g_coupling = 50e6     # 50 MHz field-matter coupling strength
        
        # Decoherence (both matter and field can decohere!)
        self.gamma_matter = 1/(100e-6)  # Matter T1
        self.gamma_field = 1/(200e-6)   # Field T1 (longer - electrical fields more stable)
        self.gamma_phi = 1/(50e-6)      # Pure dephasing
        
        print(f"🌊 QUANTUM FIELD-MATTER COUPLING SYSTEM")
        print(f"Matter qubits: {n_matter_qubits}, Field modes: {n_field_modes}")
        print(f"Total Hilbert space dimension: {self.total_dim}")
        print(f"Field-matter coupling: {self.g_coupling/1e6:.0f} MHz")
        print(f"Key insight: Electrical field treated as quantum, not classical!")

def create_matter_operators(n_qubits):
    """Create Pauli operators for matter qubits"""
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    
    # Build multi-qubit operators
    operators = {}
    
    for i in range(n_qubits):
        # X, Y, Z for each qubit
        ops_x = []
        ops_y = []
        ops_z = []
        
        for j in range(n_qubits):
            if j == i:
                ops_x.append(sigma_x)
                ops_y.append(sigma_y)
                ops_z.append(sigma_z)
            else:
                ops_x.append(identity)
                ops_y.append(identity)
                ops_z.append(identity)
        
        # Tensor products
        X_i = ops_x[0]
        Y_i = ops_y[0]
        Z_i = ops_z[0]
        
        for k in range(1, n_qubits):
            X_i = np.kron(X_i, ops_x[k])
            Y_i = np.kron(Y_i, ops_y[k])
            Z_i = np.kron(Z_i, ops_z[k])
        
        operators[f'X_{i}'] = X_i
        operators[f'Y_{i}'] = Y_i
        operators[f'Z_{i}'] = Z_i
    
    return operators

def create_field_operators(n_modes, max_photons):
    """Create quantum field operators (creation/annihilation)"""
    operators = {}
    
    # Single mode operators
    a = np.zeros((max_photons + 1, max_photons + 1), dtype=complex)
    a_dag = np.zeros((max_photons + 1, max_photons + 1), dtype=complex)
    n_op = np.zeros((max_photons + 1, max_photons + 1), dtype=complex)
    
    for n in range(max_photons):
        a[n, n+1] = np.sqrt(n + 1)      # Annihilation
        a_dag[n+1, n] = np.sqrt(n + 1)  # Creation
        n_op[n, n] = n                  # Number operator
    
    # Multi-mode operators
    identity_field = np.eye(max_photons + 1, dtype=complex)
    
    for mode in range(n_modes):
        # Build tensor products for each mode
        ops_a = []
        ops_a_dag = []
        ops_n = []
        
        for m in range(n_modes):
            if m == mode:
                ops_a.append(a)
                ops_a_dag.append(a_dag)
                ops_n.append(n_op)
            else:
                ops_a.append(identity_field)
                ops_a_dag.append(identity_field)
                ops_n.append(identity_field)
        
        # Tensor products
        a_mode = ops_a[0]
        a_dag_mode = ops_a_dag[0]
        n_mode = ops_n[0]
        
        for k in range(1, n_modes):
            a_mode = np.kron(a_mode, ops_a[k])
            a_dag_mode = np.kron(a_dag_mode, ops_a_dag[k])
            n_mode = np.kron(n_mode, ops_n[k])
        
        operators[f'a_{mode}'] = a_mode
        operators[f'a_dag_{mode}'] = a_dag_mode
        operators[f'n_{mode}'] = n_mode
    
    return operators

def build_combined_operators(matter_ops, field_ops, matter_dim, field_dim):
    """Combine matter and field operators in full Hilbert space"""
    combined_ops = {}
    
    # Matter operators (act on matter ⊗ field)
    matter_identity = np.eye(matter_dim, dtype=complex)
    field_identity = np.eye(field_dim, dtype=complex)
    
    for name, matter_op in matter_ops.items():
        combined_op = np.kron(matter_op, field_identity)
        combined_ops[f'matter_{name}'] = combined_op
    
    # Field operators (act on matter ⊗ field)
    for name, field_op in field_ops.items():
        combined_op = np.kron(matter_identity, field_op)
        combined_ops[f'field_{name}'] = combined_op
    
    return combined_ops

def build_quantum_coupling_hamiltonian(system, combined_ops, coupling_type='jaynes_cummings'):
    """
    Build Hamiltonian with QUANTUM field-matter coupling
    
    Key Innovation: Both electrical field and matter are quantum!
    """
    dim = system.total_dim
    H = np.zeros((dim, dim), dtype=complex)
    
    # Matter Hamiltonian (unchanged)
    for i in range(system.n_matter_qubits):
        # Single qubit frequencies
        H += 0.5 * system.omega_matter * combined_ops[f'matter_Z_{i}']
        
        # Matter-matter coupling
        if i < system.n_matter_qubits - 1:
            H += system.J_matter * (combined_ops[f'matter_X_{i}'] @ combined_ops[f'matter_X_{i+1}'] +
                                   combined_ops[f'matter_Y_{i}'] @ combined_ops[f'matter_Y_{i+1}'])
    
    # Quantum electrical field Hamiltonian (NEW!)
    for mode in range(system.n_field_modes):
        field_freq = system.omega_field + mode * system.field_spacing
        H += field_freq * combined_ops[f'field_n_{mode}']
    
    # QUANTUM FIELD-MATTER COUPLING (the breakthrough!)
    if coupling_type == 'jaynes_cummings':
        # Jaynes-Cummings: σ⁺a + σ⁻a† (energy conserving)
        for qubit in range(system.n_matter_qubits):
            for mode in range(system.n_field_modes):
                # σ⁺ = (σₓ + iσᵧ)/2, σ⁻ = (σₓ - iσᵧ)/2
                sigma_plus = 0.5 * (combined_ops[f'matter_X_{qubit}'] + 1j * combined_ops[f'matter_Y_{qubit}'])
                sigma_minus = 0.5 * (combined_ops[f'matter_X_{qubit}'] - 1j * combined_ops[f'matter_Y_{qubit}'])
                
                # Quantum coupling: matter↑ + field photon ↔ matter↓ + field photon
                H += system.g_coupling * (sigma_plus @ combined_ops[f'field_a_{mode}'] +
                                        sigma_minus @ combined_ops[f'field_a_dag_{mode}'])
    
    elif coupling_type == 'dispersive':
        # Dispersive coupling: σᵧn̂ (conditional phase)
        for qubit in range(system.n_matter_qubits):
            for mode in range(system.n_field_modes):
                H += system.g_coupling * combined_ops[f'matter_Z_{qubit}'] @ combined_ops[f'field_n_{mode}']
    
    elif coupling_type == 'parametric':
        # Parametric coupling: (σₓ ⊗ (a + a†)) - field amplitude couples to qubit
        for qubit in range(system.n_matter_qubits):
            for mode in range(system.n_field_modes):
                field_amplitude = combined_ops[f'field_a_{mode}'] + combined_ops[f'field_a_dag_{mode}']
                H += system.g_coupling * combined_ops[f'matter_X_{qubit}'] @ field_amplitude
    
    return H

def quantum_field_master_equation(t, y, system, combined_ops, coupling_type, field_drive_amplitude=0):
    """
    Master equation for quantum field-matter system
    
    Key: Both field and matter evolve quantum mechanically and can entangle!
    """
    try:
        dim = system.total_dim
        rho = y.reshape((dim, dim))
        
        # Build time-dependent Hamiltonian
        H = build_quantum_coupling_hamiltonian(system, combined_ops, coupling_type)
        
        # Optional external drive to pump field modes
        if field_drive_amplitude > 0:
            drive_freq = system.omega_field
            for mode in range(system.n_field_modes):
                # Classical pump creates coherent states in field
                drive_term = field_drive_amplitude * np.cos(drive_freq * t) * (
                    combined_ops[f'field_a_{mode}'] + combined_ops[f'field_a_dag_{mode}'])
                H += drive_term
        
        # Coherent evolution
        coherent_evolution = -1j * (H @ rho - rho @ H)
        
        # Decoherence for matter qubits
        decoherence = np.zeros_like(rho, dtype=complex)
        
        for qubit in range(system.n_matter_qubits):
            # T1 relaxation
            sigma_minus = 0.5 * (combined_ops[f'matter_X_{qubit}'] - 1j * combined_ops[f'matter_Y_{qubit}'])
            sigma_plus = 0.5 * (combined_ops[f'matter_X_{qubit}'] + 1j * combined_ops[f'matter_Y_{qubit}'])
            
            decoherence += system.gamma_matter * (
                sigma_minus @ rho @ sigma_plus - 
                0.5 * (sigma_plus @ sigma_minus @ rho + rho @ sigma_plus @ sigma_minus)
            )
            
            # Dephasing
            decoherence += system.gamma_phi * (
                combined_ops[f'matter_Z_{qubit}'] @ rho @ combined_ops[f'matter_Z_{qubit}'] - rho
            )
        
        # Decoherence for field modes (field can also decohere!)
        for mode in range(system.n_field_modes):
            # Field decay (photon loss)
            a_mode = combined_ops[f'field_a_{mode}']
            a_dag_mode = combined_ops[f'field_a_dag_{mode}']
            
            decoherence += system.gamma_field * (
                a_mode @ rho @ a_dag_mode - 
                0.5 * (a_dag_mode @ a_mode @ rho + rho @ a_dag_mode @ a_mode)
            )
        
        return (coherent_evolution + decoherence).flatten()
        
    except Exception as e:
        print(f"💥 Quantum field coupling error at t={t:.6f}: {e}")
        import traceback
        traceback.print_exc()
        raise

def calculate_field_matter_entanglement(rho, system):
    """
    Calculate entanglement between quantum field and matter
    
    This is the KEY MEASUREMENT: Does electrical field entangle with matter?
    """
    matter_dim = system.matter_dim
    field_dim = system.field_dim**system.n_field_modes
    
    # Reshape density matrix for partial trace
    rho_tensor = rho.reshape((matter_dim, field_dim, matter_dim, field_dim))
    
    # Partial trace over field (get matter reduced state)
    rho_matter = np.trace(rho_tensor, axis1=1, axis3=3)
    
    # Von Neumann entropy of matter subsystem
    eigenvals = np.real(np.linalg.eigvals(rho_matter))
    eigenvals = eigenvals[eigenvals > 1e-12]
    
    if len(eigenvals) <= 1:
        return 0.0
    
    # Entanglement entropy
    entropy = -np.sum(eigenvals * np.log2(eigenvals))
    return entropy

def calculate_field_coherence(rho, system, combined_ops):
    """
    Measure coherence in the quantum electrical field
    """
    field_coherences = []
    
    for mode in range(system.n_field_modes):
        # Field quadrature operators
        X_field = combined_ops[f'field_a_{mode}'] + combined_ops[f'field_a_dag_{mode}']
        P_field = -1j * (combined_ops[f'field_a_{mode}'] - combined_ops[f'field_a_dag_{mode}'])
        
        # Measure field quadratures
        X_expectation = np.real(np.trace(rho @ X_field))
        P_expectation = np.real(np.trace(rho @ P_field))
        
        # Field coherence magnitude
        coherence = np.sqrt(X_expectation**2 + P_expectation**2)
        field_coherences.append(coherence)
    
    return field_coherences

def run_quantum_field_coupling_experiment():
    """
    Test the hypothesis: Quantum electrical fields can entangle with quantum matter
    """
    print("🌊 QUANTUM FIELD-MATTER COUPLING EXPERIMENT")
    print("Testing hypothesis: Electrical waves as quantum partners, not classical drivers")
    print("=" * 70)
    
    # Create smaller system for computational feasibility
    system = QuantumFieldCouplingSystem(n_matter_qubits=2, n_field_modes=2)
    
    # Create operators
    matter_ops = create_matter_operators(system.n_matter_qubits)
    field_ops = create_field_operators(system.n_field_modes, max_photons=3)
    
    field_dim = 4**system.n_field_modes  # 4 photon states per mode
    combined_ops = build_combined_operators(matter_ops, field_ops, 
                                          system.matter_dim, field_dim)
    
    # Update system dimension
    system.total_dim = system.matter_dim * field_dim
    system.field_dim = 4
    
    print(f"Actual system dimension: {system.total_dim}")
    
    # Initial state: matter in ground state, field in coherent superposition
    dim = system.total_dim
    initial_rho = np.zeros((dim, dim), dtype=complex)
    
    # Matter: |00⟩ (both qubits in ground state)
    # Field: |01⟩ (mode 0 has 0 photons, mode 1 has 1 photon)
    matter_state = 0  # |00⟩
    field_state = 1   # |01⟩ in 2-mode 4-level system
    
    initial_state_idx = matter_state * field_dim + field_state
    initial_rho[initial_state_idx, initial_state_idx] = 1.0
    
    print(f"Initial state: matter |00⟩, field |01⟩")
    
    # Test different coupling types
    coupling_types = ['jaynes_cummings', 'dispersive', 'parametric']
    results = {}
    
    for coupling_type in coupling_types:
        print(f"\n🔗 Testing {coupling_type} coupling...")
        start_time = time.time()
        
        try:
            def dynamics(t, y):
                return quantum_field_master_equation(t, y, system, combined_ops, 
                                                   coupling_type, field_drive_amplitude=10e6)
            
            # Simulation parameters
            t_span = (0.0, 5e-6)  # 5 μs
            n_points = 100
            t_eval = np.linspace(0, t_span[1], n_points)
            
            sol = solve_ivp(dynamics, t_span, initial_rho.flatten(),
                           t_eval=t_eval, method='RK45',
                           rtol=1e-6, atol=1e-8, max_step=1e-7)
            
            if sol.success:
                elapsed = time.time() - start_time
                print(f"   ✅ SUCCESS in {elapsed:.1f}s")
                results[coupling_type] = sol
            else:
                print(f"   ❌ FAILED: {sol.message}")
                
        except Exception as e:
            print(f"   💥 ERROR: {e}")
    
    # Analyze results
    if results:
        print(f"\n📊 QUANTUM FIELD-MATTER ENTANGLEMENT ANALYSIS")
        
        analysis = {}
        
        for coupling_type, sol in results.items():
            print(f"\n🔗 {coupling_type.upper()} COUPLING:")
            
            entanglement_evolution = []
            field_coherence_evolution = []
            
            for i, t in enumerate(sol.t):
                rho = sol.y[:, i].reshape((dim, dim))
                
                # Ensure proper density matrix
                rho = (rho + rho.conj().T) / 2
                trace = np.trace(rho)
                if abs(trace) > 1e-10:
                    rho = rho / trace
                
                # Field-matter entanglement
                entanglement = calculate_field_matter_entanglement(rho, system)
                entanglement_evolution.append(entanglement)
                
                # Field coherence
                field_coherences = calculate_field_coherence(rho, system, combined_ops)
                avg_field_coherence = np.mean(field_coherences)
                field_coherence_evolution.append(avg_field_coherence)
            
            # Performance metrics
            peak_entanglement = np.max(entanglement_evolution)
            final_entanglement = entanglement_evolution[-1]
            avg_entanglement = np.mean(entanglement_evolution)
            
            peak_field_coherence = np.max(field_coherence_evolution)
            final_field_coherence = field_coherence_evolution[-1]
            
            analysis[coupling_type] = {
                'peak_entanglement': peak_entanglement,
                'final_entanglement': final_entanglement,
                'avg_entanglement': avg_entanglement,
                'peak_field_coherence': peak_field_coherence,
                'final_field_coherence': final_field_coherence,
                'entanglement_evolution': entanglement_evolution,
                'field_coherence_evolution': field_coherence_evolution,
                'time': sol.t
            }
            
            print(f"   Peak field-matter entanglement: {peak_entanglement:.4f}")
            print(f"   Final field-matter entanglement: {final_entanglement:.4f}")
            print(f"   Peak field coherence: {peak_field_coherence:.4f}")
            print(f"   Final field coherence: {final_field_coherence:.4f}")
        
        # Create comprehensive plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Quantum Field-Matter Coupling Results', fontsize=16)
        
        colors = ['blue', 'red', 'green']
        
        for i, (coupling_type, data) in enumerate(analysis.items()):
            # Entanglement evolution
            axes[0, 0].plot(data['time'] * 1e6, data['entanglement_evolution'], 
                           color=colors[i], label=f'{coupling_type}', linewidth=2)
            
            # Field coherence evolution  
            axes[0, 1].plot(data['time'] * 1e6, data['field_coherence_evolution'],
                           color=colors[i], label=f'{coupling_type}', linewidth=2)
        
        axes[0, 0].set_xlabel('Time (μs)')
        axes[0, 0].set_ylabel('Field-Matter Entanglement')
        axes[0, 0].set_title('Quantum Field-Matter Entanglement')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Time (μs)')
        axes[0, 1].set_ylabel('Field Coherence')
        axes[0, 1].set_title('Quantum Field Coherence')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Performance comparison
        coupling_names = list(analysis.keys())
        peak_entanglements = [analysis[c]['peak_entanglement'] for c in coupling_names]
        final_entanglements = [analysis[c]['final_entanglement'] for c in coupling_names]
        
        x = np.arange(len(coupling_names))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, peak_entanglements, width, label='Peak', alpha=0.8)
        axes[1, 0].bar(x + width/2, final_entanglements, width, label='Final', alpha=0.8)
        axes[1, 0].set_xlabel('Coupling Type')
        axes[1, 0].set_ylabel('Field-Matter Entanglement')
        axes[1, 0].set_title('Entanglement Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(coupling_names)
        axes[1, 0].legend()
        
        # Field coherence comparison
        peak_coherences = [analysis[c]['peak_field_coherence'] for c in coupling_names]
        final_coherences = [analysis[c]['final_field_coherence'] for c in coupling_names]
        
        axes[1, 1].bar(x - width/2, peak_coherences, width, label='Peak', alpha=0.8)
        axes[1, 1].bar(x + width/2, final_coherences, width, label='Final', alpha=0.8)
        axes[1, 1].set_xlabel('Coupling Type')
        axes[1, 1].set_ylabel('Field Coherence')
        axes[1, 1].set_title('Field Coherence Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(coupling_names)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('quantum_field_matter_coupling.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save results
        with open('quantum_field_coupling_results.txt', 'w') as f:
            f.write("QUANTUM FIELD-MATTER COUPLING EXPERIMENT RESULTS\n")
            f.write("Testing: Electrical fields as quantum partners, not classical drivers\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("HYPOTHESIS TEST:\n")
            f.write("Can quantum electrical fields entangle with quantum matter?\n\n")
            
            best_coupling = max(analysis.keys(), key=lambda x: analysis[x]['peak_entanglement'])
            
            f.write("RESULTS SUMMARY:\n")
            for coupling_type, data in analysis.items():
                f.write(f"{coupling_type.upper()} COUPLING:\n")
                f.write(f"  Peak field-matter entanglement: {data['peak_entanglement']:.6f}\n")
                f.write(f"  Final field-matter entanglement: {data['final_entanglement']:.6f}\n")
                f.write(f"  Peak field coherence: {data['peak_field_coherence']:.6f}\n")
                f.write(f"  Final field coherence: {data['final_field_coherence']:.6f}\n\n")
            
            f.write(f"BEST COUPLING TYPE: {best_coupling.upper()}\n")
            f.write(f"Peak entanglement achieved: {analysis[best_coupling]['peak_entanglement']:.6f}\n\n")
            
            if analysis[best_coupling]['peak_entanglement'] > 0.1:
                f.write("✅ HYPOTHESIS CONFIRMED: Quantum field-matter entanglement achieved!\n")
                f.write("Electrical fields can act as quantum partners, not just classical drivers.\n")
            elif analysis[best_coupling]['peak_entanglement'] > 0.01:
                f.write("⚠️  WEAK EVIDENCE: Some field-matter entanglement detected.\n")
                f.write("May need stronger coupling or longer coherence times.\n")
            else:
                f.write("❌ HYPOTHESIS NOT SUPPORTED: No significant field-matter entanglement.\n")
                f.write("Classical treatment may be sufficient for this regime.\n")
        
        print(f"\n🎯 EXPERIMENT CONCLUSION:")
        best_peak = analysis[best_coupling]['peak_entanglement']
        
        if best_peak > 0.1:
            print("✅ HYPOTHESIS CONFIRMED!")
            print("Quantum electrical fields CAN entangle with quantum matter!")
            print("This supports treating electrical waves as 'quantum partners'")
        elif best_peak > 0.01:
            print("⚠️  WEAK EVIDENCE for quantum field-matter entanglement")
            print("Results suggest quantum effects but may need optimization")
        else:
            print("❌ No significant quantum field-matter entanglement detected")
            print("Classical approximation may be sufficient")
        
        print(f"\nBest coupling: {best_coupling} (peak entanglement: {best_peak:.4f})")
        
        return results, analysis
    
    else:
        print("❌ No successful simulations - check system parameters")
        return None, None

if __name__ == "__main__":
    print("🌊 QUANTUM FIELD-MATTER COUPLING EXPERIMENT")
    print("=" * 50)
    print("PARADIGM SHIFT TEST:")
    print("• Before: Classical E-field → Quantum system")  
    print("• Now: Quantum E-field ↔ Quantum system")
    print("• Question: Can electrical fields be quantum partners?")
    print("=" * 50)
    
    try:
        results, analysis = run_quantum_field_coupling_experiment()
        
        if results and analysis:
            print(f"\n🎉 QUANTUM FIELD COUPLING EXPERIMENT COMPLETED!")
            print("Files generated:")
            print("- quantum_field_coupling_results.txt")
            print("- quantum_field_matter_coupling.png")
            print("\nThis tests whether electrical waves can be true")
            print("'quantum partners' rather than classical control fields!")
        
    except Exception as e:
        print(f"💥 Experiment failed: {e}")
        import traceback
        traceback.print_exc()