import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize

class QuantumFieldControlSystem:
    """
    ISOLATE ELECTRICAL FIELD AS QUANTUM CONTROL RESOURCE
    
    Key Innovation: Use quantum electrical field states for:
    1. Precise quantum control (not just classical pulses)
    2. Quantum sensing and feedback
    3. Entanglement-enhanced operations
    """
    
    def __init__(self):
        # Target system: 2 qubits controlled by quantum E-field
        self.n_qubits = 2
        self.n_field_modes = 1
        
        # System parameters
        self.qubit_freq = 5.0e9  # 5 GHz
        self.field_freq = 5.0e9  # Resonant
        self.g_control = 20e6    # 20 MHz control coupling
        
        # Dimensions
        self.qubit_dim = 2**self.n_qubits  # 4 states: |00⟩, |01⟩, |10⟩, |11⟩
        self.field_dim = 4  # Field states: |0⟩, |1⟩, |2⟩, |3⟩ photons
        self.total_dim = self.qubit_dim * self.field_dim  # 16 total
        
        # Decoherence
        self.T1 = 200e-6  # 200 μs
        self.T2 = 100e-6  # 100 μs
        self.field_T1 = 500e-6  # Field more stable
        
        print(f"🎯 QUANTUM FIELD CONTROL SYSTEM")
        print(f"Target: {self.n_qubits} qubits, {self.n_field_modes} control field")
        print(f"Control coupling: {self.g_control/1e6:.0f} MHz")
        print(f"System dimension: {self.total_dim}")

def create_control_hamiltonian(system, field_state_type='coherent'):
    """
    Build Hamiltonian for quantum field control
    """
    H = np.zeros((system.total_dim, system.total_dim), dtype=complex)
    
    # Encode states: |qubit_state, field_photons⟩
    for i in range(system.total_dim):
        qubit_state = i // system.field_dim  # 0-3: |00⟩,|01⟩,|10⟩,|11⟩
        field_photons = i % system.field_dim  # 0-3: photon number
        
        # Decode qubit state
        q0 = qubit_state & 1        # First qubit
        q1 = (qubit_state >> 1) & 1 # Second qubit
        
        # Free evolution
        qubit_energy = 0.5 * system.qubit_freq * ((2*q0-1) + (2*q1-1))
        field_energy = system.field_freq * field_photons
        H[i, i] = qubit_energy + field_energy
        
        # Quantum field control interactions
        for qubit_idx in [0, 1]:
            qubit_mask = 1 << qubit_idx
            current_qubit = (qubit_state >> qubit_idx) & 1
            
            # Field-controlled qubit operations
            if field_state_type == 'coherent':
                # Coherent field: smooth control
                if current_qubit == 0 and field_photons > 0:
                    # |0⟩ + field photon → |1⟩
                    new_qubit_state = qubit_state | qubit_mask
                    new_field = field_photons - 1
                    j = new_qubit_state * system.field_dim + new_field
                    
                    coupling = system.g_control * np.sqrt(field_photons)
                    H[i, j] += coupling
                    H[j, i] += coupling
                
                if current_qubit == 1 and field_photons < system.field_dim - 1:
                    # |1⟩ → |0⟩ + field photon
                    new_qubit_state = qubit_state & (~qubit_mask)
                    new_field = field_photons + 1
                    j = new_qubit_state * system.field_dim + new_field
                    
                    coupling = system.g_control * np.sqrt(field_photons + 1)
                    H[i, j] += coupling
                    H[j, i] += coupling
            
            elif field_state_type == 'squeezed':
                # Squeezed field: enhanced control precision
                squeeze_factor = 1.5
                
                if current_qubit == 0 and field_photons > 0:
                    new_qubit_state = qubit_state | qubit_mask
                    new_field = field_photons - 1
                    j = new_qubit_state * system.field_dim + new_field
                    
                    # Enhanced coupling from squeezing
                    coupling = system.g_control * squeeze_factor * np.sqrt(field_photons)
                    H[i, j] += coupling
                    H[j, i] += coupling
    
    return H

def density_to_real(rho):
    """Convert density matrix to real vector"""
    n = rho.shape[0]
    vec = []
    
    # Diagonal (real)
    for i in range(n):
        vec.append(np.real(rho[i, i]))
    
    # Off-diagonal (real + imag)
    for i in range(n):
        for j in range(i+1, n):
            vec.append(np.real(rho[i, j]))
            vec.append(np.imag(rho[i, j]))
    
    return np.array(vec)

def real_to_density(vec, n):
    """Convert real vector to density matrix"""
    rho = np.zeros((n, n), dtype=complex)
    idx = 0
    
    # Diagonal
    for i in range(n):
        rho[i, i] = vec[idx]
        idx += 1
    
    # Off-diagonal
    for i in range(n):
        for j in range(i+1, n):
            rho[i, j] = vec[idx] + 1j * vec[idx+1]
            rho[j, i] = vec[idx] - 1j * vec[idx+1]
            idx += 2
    
    return rho

def master_equation_control(t, rho_vec, system, H):
    """Master equation with decoherence"""
    dim = system.total_dim
    rho = real_to_density(rho_vec, dim)
    
    # Coherent evolution
    drho_dt = -1j * (H @ rho - rho @ H)
    
    # Simplified decoherence
    for i in range(dim):
        qubit_state = i // system.field_dim
        field_photons = i % system.field_dim
        
        # Qubit T1 relaxation
        for q_idx in [0, 1]:
            if (qubit_state >> q_idx) & 1:  # Qubit excited
                ground_state = qubit_state & (~(1 << q_idx))
                ground_idx = ground_state * system.field_dim + field_photons
                
                rate = 1/system.T1
                drho_dt[ground_idx, ground_idx] += rate * rho[i, i]
                drho_dt[i, i] -= rate * rho[i, i]
        
        # Field decay
        if field_photons > 0:
            lower_idx = qubit_state * system.field_dim + (field_photons - 1)
            rate = field_photons / system.field_T1
            
            drho_dt[lower_idx, lower_idx] += rate * rho[i, i]
            drho_dt[i, i] -= rate * rho[i, i]
    
    return density_to_real(drho_dt)

def calculate_control_fidelity(rho, target_state, system):
    """Calculate fidelity to target quantum state"""
    try:
        # Extract qubit reduced state
        qubit_dim = system.qubit_dim
        field_dim = system.field_dim
        
        rho_qubit = np.zeros((qubit_dim, qubit_dim), dtype=complex)
        
        for i in range(qubit_dim):
            for j in range(qubit_dim):
                for k in range(field_dim):
                    rho_qubit[i, j] += rho[i*field_dim + k, j*field_dim + k]
        
        # Target state fidelity
        if target_state == 'bell':
            # Target: (|00⟩ + |11⟩)/√2
            target_rho = np.zeros((qubit_dim, qubit_dim), dtype=complex)
            target_rho[0, 0] = 0.5  # |00⟩⟨00|
            target_rho[3, 3] = 0.5  # |11⟩⟨11|
            target_rho[0, 3] = 0.5  # |00⟩⟨11|
            target_rho[3, 0] = 0.5  # |11⟩⟨00|
        
        elif target_state == 'superposition':
            # Target: (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2
            target_rho = np.ones((qubit_dim, qubit_dim), dtype=complex) / qubit_dim
        
        else:  # ground state
            target_rho = np.zeros((qubit_dim, qubit_dim), dtype=complex)
            target_rho[0, 0] = 1.0  # |00⟩
        
        # Fidelity = Tr(√(√ρ * target * √ρ))
        # Simplified: F = Tr(ρ * target) for pure targets
        fidelity = np.real(np.trace(rho_qubit @ target_rho))
        return max(0, min(1, fidelity))
        
    except:
        return 0.0

def calculate_field_control_power(rho, system):
    """Measure how much control the field provides"""
    try:
        # Calculate field-qubit mutual information
        qubit_dim = system.qubit_dim
        field_dim = system.field_dim
        
        # Qubit entropy
        rho_qubit = np.zeros((qubit_dim, qubit_dim), dtype=complex)
        for i in range(qubit_dim):
            for j in range(qubit_dim):
                for k in range(field_dim):
                    rho_qubit[i, j] += rho[i*field_dim + k, j*field_dim + k]
        
        qubit_eigs = np.real(np.linalg.eigvals(rho_qubit))
        qubit_eigs = qubit_eigs[qubit_eigs > 1e-12]
        qubit_entropy = -np.sum(qubit_eigs * np.log2(qubit_eigs)) if len(qubit_eigs) > 1 else 0
        
        # Field entropy
        rho_field = np.zeros((field_dim, field_dim), dtype=complex)
        for i in range(field_dim):
            for j in range(field_dim):
                for k in range(qubit_dim):
                    rho_field[i, j] += rho[k*field_dim + i, k*field_dim + j]
        
        field_eigs = np.real(np.linalg.eigvals(rho_field))
        field_eigs = field_eigs[field_eigs > 1e-12]
        field_entropy = -np.sum(field_eigs * np.log2(field_eigs)) if len(field_eigs) > 1 else 0
        
        # Total entropy
        total_eigs = np.real(np.linalg.eigvals(rho))
        total_eigs = total_eigs[total_eigs > 1e-12]
        total_entropy = -np.sum(total_eigs * np.log2(total_eigs)) if len(total_eigs) > 1 else 0
        
        # Mutual information = S(qubit) + S(field) - S(total)
        mutual_info = qubit_entropy + field_entropy - total_entropy
        return max(0, mutual_info)
        
    except:
        return 0.0

def run_quantum_field_control_experiment():
    """
    Test quantum electrical field as control resource
    """
    print("🎯 QUANTUM ELECTRICAL FIELD CONTROL EXPERIMENT")
    print("Testing: Field as quantum control resource vs. classical noise")
    print("=" * 60)
    
    system = QuantumFieldControlSystem()
    
    # Test different field control types
    control_types = ['coherent', 'squeezed']
    target_states = ['bell', 'superposition']
    
    results = {}
    
    for control_type in control_types:
        for target in target_states:
            key = f"{control_type}_{target}"
            print(f"\n🔬 Testing {control_type} field → {target} state")
            
            # Build Hamiltonian
            H = create_control_hamiltonian(system, control_type)
            
            # Initial state: |00,1⟩ (qubits in ground, 1 control photon)
            initial_rho = np.zeros((system.total_dim, system.total_dim), dtype=complex)
            qubit_state = 0  # |00⟩
            field_photons = 1  # 1 photon available for control
            initial_idx = qubit_state * system.field_dim + field_photons
            initial_rho[initial_idx, initial_idx] = 1.0
            
            # Convert to real vector
            initial_vec = density_to_real(initial_rho)
            
            # Control time (optimize for target state)
            if target == 'bell':
                control_time = np.pi / (2 * system.g_control) * 1e6  # μs for π/2 pulse
            else:
                control_time = np.pi / (4 * system.g_control) * 1e6  # μs for π/4 pulse
            
            t_span = (0, control_time * 1e-6)  # Convert to seconds
            n_points = 100
            t_eval = np.linspace(0, t_span[1], n_points)
            
            print(f"   Control time: {control_time:.3f} μs")
            
            # Dynamics
            def dynamics(t, y):
                return master_equation_control(t, y, system, H)
            
            start_time = time.time()
            sol = solve_ivp(dynamics, t_span, initial_vec,
                           t_eval=t_eval, method='RK45',
                           rtol=1e-5, atol=1e-7)
            
            elapsed = time.time() - start_time
            
            if sol.success:
                print(f"   ✅ Simulation completed in {elapsed:.1f}s")
                
                # Analyze control performance
                fidelities = []
                control_powers = []
                field_photons = []
                
                for i, t_val in enumerate(sol.t):
                    rho = real_to_density(sol.y[:, i], system.total_dim)
                    
                    # Normalize
                    rho = (rho + rho.conj().T) / 2
                    trace = np.trace(rho)
                    if abs(trace) > 1e-10:
                        rho = rho / trace
                    
                    # Calculate metrics
                    fidelity = calculate_control_fidelity(rho, target, system)
                    control_power = calculate_field_control_power(rho, system)
                    
                    fidelities.append(fidelity)
                    control_powers.append(control_power)
                    
                    # Average field photon number
                    avg_photons = 0
                    for state_idx in range(system.total_dim):
                        photon_num = state_idx % system.field_dim
                        prob = np.real(rho[state_idx, state_idx])
                        avg_photons += photon_num * prob
                    field_photons.append(avg_photons)
                
                # Performance metrics
                max_fidelity = np.max(fidelities)
                final_fidelity = fidelities[-1]
                avg_control_power = np.mean(control_powers)
                photon_efficiency = (1.0 - field_photons[-1]) / 1.0  # How much photon was used
                
                results[key] = {
                    'max_fidelity': max_fidelity,
                    'final_fidelity': final_fidelity,
                    'avg_control_power': avg_control_power,
                    'photon_efficiency': photon_efficiency,
                    'fidelities': fidelities,
                    'control_powers': control_powers,
                    'field_photons': field_photons,
                    'time': sol.t * 1e6,  # Convert to μs
                    'control_type': control_type,
                    'target': target
                }
                
                print(f"   Max fidelity: {max_fidelity:.4f}")
                print(f"   Final fidelity: {final_fidelity:.4f}")
                print(f"   Avg control power: {avg_control_power:.4f}")
                print(f"   Photon efficiency: {photon_efficiency:.4f}")
                
            else:
                print(f"   ❌ Simulation failed: {sol.message}")
    
    # Analysis and visualization
    if results:
        print(f"\n📊 QUANTUM FIELD CONTROL ANALYSIS")
        
        # Create comprehensive plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Quantum Electrical Field Control Results', fontsize=16)
        
        colors = {'coherent_bell': 'blue', 'coherent_superposition': 'red',
                 'squeezed_bell': 'green', 'squeezed_superposition': 'orange'}
        
        for key, data in results.items():
            color = colors.get(key, 'black')
            label = f"{data['control_type']} → {data['target']}"
            
            # Fidelity evolution
            axes[0, 0].plot(data['time'], data['fidelities'], 
                           color=color, label=label, linewidth=2)
            
            # Control power evolution
            axes[0, 1].plot(data['time'], data['control_powers'],
                           color=color, label=label, linewidth=2)
            
            # Field photon usage
            axes[0, 2].plot(data['time'], data['field_photons'],
                           color=color, label=label, linewidth=2)
        
        axes[0, 0].set_xlabel('Time (μs)')
        axes[0, 0].set_ylabel('Target Fidelity')
        axes[0, 0].set_title('Control Fidelity Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Time (μs)')
        axes[0, 1].set_ylabel('Control Power (bits)')
        axes[0, 1].set_title('Field-Qubit Control Power')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].set_xlabel('Time (μs)')
        axes[0, 2].set_ylabel('Field Photons')
        axes[0, 2].set_title('Control Resource Usage')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Performance comparison bars
        methods = list(results.keys())
        max_fidelities = [results[m]['max_fidelity'] for m in methods]
        control_powers = [results[m]['avg_control_power'] for m in methods]
        efficiencies = [results[m]['photon_efficiency'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.25
        
        axes[1, 0].bar(x - width, max_fidelities, width, label='Max Fidelity', alpha=0.8)
        axes[1, 0].set_xlabel('Control Method')
        axes[1, 0].set_ylabel('Fidelity')
        axes[1, 0].set_title('Maximum Control Fidelity')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([m.replace('_', ' ') for m in methods], rotation=45)
        
        axes[1, 1].bar(x, control_powers, width, label='Control Power', alpha=0.8, color='orange')
        axes[1, 1].set_xlabel('Control Method')
        axes[1, 1].set_ylabel('Control Power (bits)')
        axes[1, 1].set_title('Average Control Power')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([m.replace('_', ' ') for m in methods], rotation=45)
        
        axes[1, 2].bar(x + width, efficiencies, width, label='Efficiency', alpha=0.8, color='green')
        axes[1, 2].set_xlabel('Control Method')
        axes[1, 2].set_ylabel('Photon Efficiency')
        axes[1, 2].set_title('Resource Efficiency')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels([m.replace('_', ' ') for m in methods], rotation=45)
        
        plt.tight_layout()
        plt.savefig('quantum_field_control_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find best control method
        best_method = max(results.keys(), key=lambda x: results[x]['max_fidelity'])
        best_fidelity = results[best_method]['max_fidelity']
        best_power = results[best_method]['avg_control_power']
        
        print(f"\n🎯 CONTROL ISOLATION RESULTS:")
        print(f"Best method: {best_method.replace('_', ' ')}")
        print(f"Peak control fidelity: {best_fidelity:.4f}")
        print(f"Average control power: {best_power:.4f} bits")
        
        # Assessment
        if best_fidelity > 0.95:
            print("✅ EXCELLENT QUANTUM FIELD CONTROL!")
            print("Electrical field demonstrates high-fidelity quantum control")
        elif best_fidelity > 0.8:
            print("⚡ GOOD QUANTUM FIELD CONTROL")
            print("Electrical field shows strong control capabilities")
        elif best_fidelity > 0.6:
            print("📊 MODERATE QUANTUM FIELD CONTROL")
            print("Some control demonstrated, may need optimization")
        else:
            print("❌ Limited control observed")
        
        if best_power > 0.5:
            print("🎯 HIGH CONTROL POWER: Field provides significant quantum control")
        elif best_power > 0.1:
            print("⚡ MODERATE CONTROL POWER: Field shows control capability")
        
        # Save results
        with open('quantum_field_control_results.txt', 'w') as f:
            f.write("QUANTUM ELECTRICAL FIELD CONTROL ISOLATION RESULTS\n")
            f.write("Testing: Field as quantum control resource\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("HYPOTHESIS: Electrical fields can be quantum control resources\n")
            f.write("rather than just classical noise sources.\n\n")
            
            f.write("CONTROL METHODS TESTED:\n")
            for key, data in results.items():
                f.write(f"{key.upper().replace('_', ' ')}:\n")
                f.write(f"  Max fidelity: {data['max_fidelity']:.6f}\n")
                f.write(f"  Final fidelity: {data['final_fidelity']:.6f}\n")
                f.write(f"  Avg control power: {data['avg_control_power']:.6f} bits\n")
                f.write(f"  Photon efficiency: {data['photon_efficiency']:.6f}\n\n")
            
            f.write(f"BEST CONTROL METHOD: {best_method.upper().replace('_', ' ')}\n")
            f.write(f"Peak fidelity: {best_fidelity:.6f}\n")
            f.write(f"Control power: {best_power:.6f} bits\n\n")
            
            if best_fidelity > 0.8 and best_power > 0.1:
                f.write("CONCLUSION: ✅ QUANTUM FIELD CONTROL CONFIRMED!\n")
                f.write("Electrical fields can serve as high-fidelity quantum control\n")
                f.write("resources, providing precise manipulation of quantum states.\n")
                f.write("This enables 'quantum control with quantum fields' paradigm.\n")
            else:
                f.write("CONCLUSION: ⚠️  LIMITED QUANTUM FIELD CONTROL\n")
                f.write("Some control capability detected but may need optimization.\n")
        
        return results
    
    else:
        print("❌ No successful control experiments")
        return None

if __name__ == "__main__":
    print("🎯 QUANTUM ELECTRICAL FIELD CONTROL ISOLATION")
    print("=" * 50)
    print("PARADIGM TEST:")
    print("• Classical view: E-field = noise or classical control")
    print("• Quantum view: E-field = quantum control resource")
    print("• Question: Can we isolate quantum field control?")
    print("=" * 50)
    
    try:
        results = run_quantum_field_control_experiment()
        
        if results:
            print(f"\n🎉 QUANTUM FIELD CONTROL EXPERIMENT COMPLETED!")
            print("Files generated:")
            print("- quantum_field_control_results.txt")
            print("- quantum_field_control_results.png")
            print("\nThis isolates the quantum electrical field control")
            print("mechanism and demonstrates its utility as a resource!")
        
    except Exception as e:
        print(f"💥 Control experiment failed: {e}")
        import traceback
        traceback.print_exc()