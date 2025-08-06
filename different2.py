import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

class QuantumFieldCouplingSystem:
    """
    FIXED VERSION: Quantum electrical field coupled to quantum matter system
    
    Key fixes:
    - Reduced system size for computational feasibility
    - Real-valued density matrix representation
    - Proper time scales and coupling strengths
    """
    
    def __init__(self, n_matter_qubits=2, n_field_modes=1):  # REDUCED from 3,4
        self.n_matter_qubits = n_matter_qubits
        self.n_field_modes = n_field_modes
        
        # FIXED: Manageable system size
        self.matter_dim = 2**n_matter_qubits
        self.field_dim = 3  # Only |0⟩, |1⟩, |2⟩ photon states (not 5!)
        self.total_dim = self.matter_dim * (self.field_dim**n_field_modes)
        
        # Matter system parameters
        self.omega_matter = 5.0e9  # 5 GHz qubit frequency
        self.J_matter = 10e6       # REDUCED: 10 MHz matter-matter coupling
        
        # Electrical field parameters
        self.omega_field = 5.0e9   # RESONANT with matter (not detuned!)
        self.field_spacing = 50e6  # REDUCED: 50 MHz mode spacing
        
        # FIXED: Weaker coupling for numerical stability
        self.g_coupling = 15e6     # REDUCED: 15 MHz field-matter coupling
        
        # FIXED: Longer coherence times
        self.gamma_matter = 1/(300e-6)  # 300 μs T1
        self.gamma_field = 1/(600e-6)   # 600 μs field T1
        self.gamma_phi = 1/(150e-6)     # 150 μs dephasing
        
        print(f"🌊 FIXED QUANTUM FIELD-MATTER COUPLING SYSTEM")
        print(f"Matter qubits: {n_matter_qubits}, Field modes: {n_field_modes}")
        print(f"Total Hilbert space dimension: {self.total_dim}")
        print(f"Field-matter coupling: {self.g_coupling/1e6:.0f} MHz")
        print(f"Expected Rabi period: {1e6/(2*self.g_coupling):.3f} μs")

def density_matrix_to_real_vector(rho):
    """Convert complex density matrix to real vector for ODE solver"""
    n = rho.shape[0]
    
    # Real part: diagonal + upper triangular real parts
    real_part = []
    
    # Diagonal elements (always real)
    for i in range(n):
        real_part.append(np.real(rho[i, i]))
    
    # Off-diagonal elements: real and imaginary parts
    for i in range(n):
        for j in range(i+1, n):
            real_part.append(np.real(rho[i, j]))  # Real part
            real_part.append(np.imag(rho[i, j]))  # Imaginary part
    
    return np.array(real_part)

def real_vector_to_density_matrix(vec, n):
    """Convert real vector back to complex density matrix"""
    rho = np.zeros((n, n), dtype=complex)
    
    idx = 0
    
    # Diagonal elements
    for i in range(n):
        rho[i, i] = vec[idx]
        idx += 1
    
    # Off-diagonal elements
    for i in range(n):
        for j in range(i+1, n):
            rho[i, j] = vec[idx] + 1j * vec[idx+1]
            rho[j, i] = vec[idx] - 1j * vec[idx+1]  # Hermitian conjugate
            idx += 2
    
    return rho

def build_simple_jaynes_cummings_hamiltonian(system):
    """
    SIMPLIFIED: Direct Jaynes-Cummings Hamiltonian for small system
    """
    dim = system.total_dim
    H = np.zeros((dim, dim), dtype=complex)
    
    # System: 2 qubits ⊗ 1 field mode with 3 photon states
    # States: |qubit1,qubit2,photons⟩
    
    for i in range(dim):
        # Decode state: i = matter_state * field_dim + photon_number
        matter_state = i // system.field_dim
        photon_number = i % system.field_dim
        
        # Decode matter state: |q1,q2⟩
        qubit1 = matter_state & 1
        qubit2 = (matter_state >> 1) & 1
        
        # Diagonal terms (free evolution)
        matter_energy = 0.5 * system.omega_matter * ((2*qubit1-1) + (2*qubit2-1))
        field_energy = system.omega_field * photon_number
        matter_interaction = system.J_matter * (2*qubit1-1) * (2*qubit2-1)
        
        H[i, i] = matter_energy + field_energy + matter_interaction
        
        # Jaynes-Cummings coupling for each qubit
        for qubit_idx in [0, 1]:
            qubit_mask = 1 << qubit_idx
            current_qubit = (matter_state >> qubit_idx) & 1
            
            # Excitation: |0⟩ + photon → |1⟩
            if current_qubit == 0 and photon_number > 0:
                new_matter = matter_state | qubit_mask  # Excite qubit
                new_photon = photon_number - 1
                j = new_matter * system.field_dim + new_photon
                
                coupling = system.g_coupling * np.sqrt(photon_number)
                H[i, j] += coupling
                H[j, i] += coupling
            
            # Emission: |1⟩ → |0⟩ + photon
            if current_qubit == 1 and photon_number < system.field_dim - 1:
                new_matter = matter_state & (~qubit_mask)  # De-excite qubit
                new_photon = photon_number + 1
                j = new_matter * system.field_dim + new_photon
                
                coupling = system.g_coupling * np.sqrt(photon_number + 1)
                H[i, j] += coupling
                H[j, i] += coupling
    
    return H

def master_equation_real(t, rho_vec, system, H):
    """
    Master equation with real vector representation
    """
    dim = system.total_dim
    
    # Convert back to density matrix
    rho = real_vector_to_density_matrix(rho_vec, dim)
    
    # Coherent evolution
    drho_dt = -1j * (H @ rho - rho @ H)
    
    # Simple decoherence
    for i in range(dim):
        matter_state = i // system.field_dim
        photon_number = i % system.field_dim
        
        # T1 relaxation for excited qubits
        for qubit_idx in [0, 1]:
            if (matter_state >> qubit_idx) & 1:  # Qubit is excited
                ground_matter = matter_state & (~(1 << qubit_idx))
                ground_idx = ground_matter * system.field_dim + photon_number
                
                rate = system.gamma_matter
                drho_dt[ground_idx, ground_idx] += rate * rho[i, i]
                drho_dt[i, i] -= rate * rho[i, i]
                
                # Dephasing
                drho_dt[i, i] -= system.gamma_phi * rho[i, i]
        
        # Field decay
        if photon_number > 0:
            lower_idx = matter_state * system.field_dim + (photon_number - 1)
            rate = system.gamma_field * photon_number
            
            drho_dt[lower_idx, lower_idx] += rate * rho[i, i]
            drho_dt[i, i] -= rate * rho[i, i]
    
    # Convert back to real vector
    return density_matrix_to_real_vector(drho_dt)

def calculate_entanglement_simple(rho, system):
    """Simple entanglement calculation"""
    try:
        # Partial trace over field
        matter_dim = system.matter_dim
        field_dim = system.field_dim
        
        rho_matter = np.zeros((matter_dim, matter_dim), dtype=complex)
        
        for i in range(matter_dim):
            for j in range(matter_dim):
                for k in range(field_dim):
                    idx1 = i * field_dim + k
                    idx2 = j * field_dim + k
                    rho_matter[i, j] += rho[idx1, idx2]
        
        # Von Neumann entropy
        eigenvals = np.real(np.linalg.eigvals(rho_matter))
        eigenvals = eigenvals[eigenvals > 1e-12]
        
        if len(eigenvals) <= 1:
            return 0.0
        
        entropy = -np.sum(eigenvals * np.log2(eigenvals))
        return min(entropy, 2.0)  # Cap at 2 qubits max
        
    except:
        return 0.0

def run_quantum_field_coupling_experiment():
    """
    FIXED experiment that actually runs
    """
    print("🌊 QUANTUM FIELD-MATTER COUPLING EXPERIMENT (FIXED)")
    print("Testing hypothesis: Electrical waves as quantum partners")
    print("=" * 60)
    
    # Create feasible system
    system = QuantumFieldCouplingSystem(n_matter_qubits=2, n_field_modes=1)
    
    # Build Hamiltonian
    H = build_simple_jaynes_cummings_hamiltonian(system)
    
    print(f"System dimension: {system.total_dim}")
    print(f"Hamiltonian size: {H.shape}")
    
    # Initial state: |01,1⟩ (second qubit excited, 1 photon)
    dim = system.total_dim
    initial_rho = np.zeros((dim, dim), dtype=complex)
    
    # State: matter_state=2 (|10⟩), photon_number=1
    matter_state = 2  # |10⟩ (first qubit excited)
    photon_number = 1
    initial_idx = matter_state * system.field_dim + photon_number
    initial_rho[initial_idx, initial_idx] = 1.0
    
    print(f"Initial state: |10,1⟩ at index {initial_idx}")
    
    # Convert to real vector
    initial_vec = density_matrix_to_real_vector(initial_rho)
    print(f"Real vector size: {len(initial_vec)}")
    
    # Simulation parameters
    rabi_period = 1e6 / (2 * system.g_coupling)  # μs
    sim_time = 2 * rabi_period  # 2 Rabi periods
    n_points = 100
    
    print(f"Rabi period: {rabi_period:.3f} μs")
    print(f"Simulation time: {sim_time:.3f} μs")
    
    t_span = (0.0, sim_time * 1e-6)  # Convert to seconds
    t_eval = np.linspace(0, t_span[1], n_points)
    
    # Dynamics function
    def dynamics(t, y):
        return master_equation_real(t, y, system, H)
    
    print("Starting integration...")
    start_time = time.time()
    
    # Use RK45 with real vectors
    sol = solve_ivp(dynamics, t_span, initial_vec,
                   t_eval=t_eval, method='RK45',
                   rtol=1e-5, atol=1e-7, max_step=rabi_period*1e-7)
    
    elapsed = time.time() - start_time
    
    if sol.success:
        print(f"✅ Integration completed in {elapsed:.1f} seconds")
        
        # Analyze results
        entanglement_evolution = []
        population_evolution = []
        photon_evolution = []
        
        for i, t_val in enumerate(sol.t):
            # Convert back to density matrix
            rho = real_vector_to_density_matrix(sol.y[:, i], dim)
            
            # Normalize
            rho = (rho + rho.conj().T) / 2
            trace = np.trace(rho)
            if abs(trace) > 1e-10:
                rho = rho / trace
            
            # Calculate observables
            entanglement = calculate_entanglement_simple(rho, system)
            entanglement_evolution.append(entanglement)
            
            # Excited population
            excited_pop = 0
            total_photons = 0
            for state_idx in range(dim):
                matter_idx = state_idx // system.field_dim
                photon_num = state_idx % system.field_dim
                prob = np.real(rho[state_idx, state_idx])
                
                if matter_idx > 0:  # Any qubit excited
                    excited_pop += prob
                
                total_photons += photon_num * prob
            
            population_evolution.append(excited_pop)
            photon_evolution.append(total_photons)
        
        # Results summary
        peak_entanglement = np.max(entanglement_evolution)
        final_entanglement = entanglement_evolution[-1]
        avg_entanglement = np.mean(entanglement_evolution)
        
        print(f"\n📊 RESULTS:")
        print(f"Peak field-matter entanglement: {peak_entanglement:.4f}")
        print(f"Final field-matter entanglement: {final_entanglement:.4f}")
        print(f"Average field-matter entanglement: {avg_entanglement:.4f}")
        print(f"Initial photons: {photon_evolution[0]:.3f}")
        print(f"Final photons: {photon_evolution[-1]:.3f}")
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Quantum Field-Matter Coupling Results (FIXED)', fontsize=16)
        
        time_us = sol.t * 1e6
        
        # Entanglement evolution
        ax1.plot(time_us, entanglement_evolution, 'b-', linewidth=2, label='Field-Matter Entanglement')
        ax1.set_xlabel('Time (μs)')
        ax1.set_ylabel('Entanglement')
        ax1.set_title('Quantum Field-Matter Entanglement')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Population evolution
        ax2.plot(time_us, population_evolution, 'r-', linewidth=2, label='Excited Population')
        ax2.set_xlabel('Time (μs)')
        ax2.set_ylabel('Population')
        ax2.set_title('Matter Excitation Population')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Photon number evolution
        ax3.plot(time_us, photon_evolution, 'g-', linewidth=2, label='Average Photon Number')
        ax3.set_xlabel('Time (μs)')
        ax3.set_ylabel('Photon Number')
        ax3.set_title('Field Photon Number')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Phase space plot
        ax4.plot(population_evolution, photon_evolution, 'purple', linewidth=2, alpha=0.7)
        ax4.scatter(population_evolution[0], photon_evolution[0], color='green', s=100, label='Start')
        ax4.scatter(population_evolution[-1], photon_evolution[-1], color='red', s=100, label='End')
        ax4.set_xlabel('Excited Population')
        ax4.set_ylabel('Photon Number')
        ax4.set_title('Phase Space Trajectory')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('quantum_field_matter_coupling_fixed.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Assessment
        print(f"\n🎯 EXPERIMENT CONCLUSION:")
        if peak_entanglement > 0.3:
            print("✅ HYPOTHESIS CONFIRMED!")
            print("Strong quantum field-matter entanglement achieved!")
            print("Electrical fields can act as quantum partners!")
        elif peak_entanglement > 0.1:
            print("⚡ MODERATE EVIDENCE for quantum field-matter coupling")
            print("Some entanglement detected - may need optimization")
        else:
            print("📊 WEAK COUPLING detected")
            print("May need stronger interaction or better initial conditions")
        
        # Save results
        with open('quantum_field_coupling_results_fixed.txt', 'w') as f:
            f.write("QUANTUM FIELD-MATTER COUPLING EXPERIMENT RESULTS (FIXED)\n")
            f.write("Testing: Electrical fields as quantum partners\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("SYSTEM PARAMETERS:\n")
            f.write(f"Matter qubits: {system.n_matter_qubits}\n")
            f.write(f"Field modes: {system.n_field_modes}\n")
            f.write(f"Total dimension: {system.total_dim}\n")
            f.write(f"Coupling strength: {system.g_coupling/1e6:.0f} MHz\n")
            f.write(f"Simulation time: {sim_time:.3f} μs\n\n")
            
            f.write("RESULTS:\n")
            f.write(f"Peak field-matter entanglement: {peak_entanglement:.6f}\n")
            f.write(f"Final field-matter entanglement: {final_entanglement:.6f}\n")
            f.write(f"Average field-matter entanglement: {avg_entanglement:.6f}\n\n")
            
            if peak_entanglement > 0.3:
                f.write("CONCLUSION: ✅ HYPOTHESIS CONFIRMED!\n")
                f.write("Strong quantum field-matter entanglement demonstrates that\n")
                f.write("electrical fields can act as quantum partners, not just classical drivers.\n")
            elif peak_entanglement > 0.1:
                f.write("CONCLUSION: ⚡ MODERATE EVIDENCE\n")
                f.write("Some quantum field-matter entanglement detected.\n") 
                f.write("Results suggest quantum effects present.\n")
            else:
                f.write("CONCLUSION: 📊 WEAK COUPLING\n")
                f.write("Limited entanglement detected.\n")
                f.write("May need parameter optimization.\n")
        
        return sol, entanglement_evolution, population_evolution, photon_evolution
        
    else:
        print(f"❌ Integration failed after {elapsed:.1f}s: {sol.message}")
        return None, None, None, None

if __name__ == "__main__":
    print("🌊 QUANTUM FIELD-MATTER COUPLING EXPERIMENT (FIXED VERSION)")
    print("=" * 50)
    print("PARADIGM SHIFT TEST:")
    print("• Before: Classical E-field → Quantum system")  
    print("• Now: Quantum E-field ↔ Quantum system")
    print("• Question: Can electrical fields be quantum partners?")
    print("=" * 50)
    
    try:
        results = run_quantum_field_coupling_experiment()
        
        if results[0] is not None:
            print(f"\n🎉 QUANTUM FIELD COUPLING EXPERIMENT COMPLETED!")
            print("Files generated:")
            print("- quantum_field_coupling_results_fixed.txt")
            print("- quantum_field_matter_coupling_fixed.png")
            print("\nThis tests whether electrical waves can be true")
            print("'quantum partners' rather than classical control fields!")
            print("\nKey improvements:")
            print("• Reduced system size (12 dimensions)")
            print("• Real-valued ODE integration")
            print("• Proper time scale resolution")
            print("• Optimized Hamiltonian construction")
        else:
            print("❌ Experiment failed - check parameters")
        
    except Exception as e:
        print(f"💥 Experiment failed: {e}")
        import traceback
        traceback.print_exc()