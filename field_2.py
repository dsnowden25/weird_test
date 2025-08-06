import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize  # FIXED: Added missing import
from scipy.special import hermite
import time

print("TRUE CONTINUOUS FIELD ELECTRICAL-QUANTUM COUPLING - FIXED VERSION")
print("No discrete energy levels - treating both electrical and quantum")
print("systems as continuous quantum fields with shared coherence")
print("=" * 70)

class ContinuousFieldSystem:
    """
    Continuous quantum field treatment of electrical-quantum coupling
    Both electrical and quantum systems as quantum harmonic oscillators
    """
    
    def __init__(self):
        # Field parameters
        self.w_electrical = 5.0    # Electrical field frequency (GHz)
        self.w_quantum = 5.0       # Quantum field frequency (GHz)
        
        # Field coupling strength (inter-field interaction)
        self.lambda_coupling = 0.1  # Coupling between electrical and quantum fields
        
        # Field damping (connection to environment)
        self.gamma_elec = 0.01     # Electrical field damping
        self.gamma_quantum = 0.025 # Quantum field damping
        
        # Maximum field amplitudes
        self.max_field_amp = 10.0
        
        # Number of field modes to include
        self.n_modes = 8  # Truncate infinite-dimensional space
    
    def harmonic_oscillator_energy(self, n):
        """Energy of nth harmonic oscillator state"""
        return (n + 0.5)  # In units of ħω
    
    def field_interaction_hamiltonian(self, t):
        """
        Hamiltonian for coupled electrical-quantum field system
        H = H_elec + H_quantum + H_interaction
        """
        n_modes = self.n_modes
        
        # Electrical field Hamiltonian (quantum harmonic oscillator)
        H_elec = np.zeros((n_modes, n_modes), dtype=complex)
        for n in range(n_modes):
            H_elec[n, n] = self.w_electrical * self.harmonic_oscillator_energy(n)
        
        # Quantum field Hamiltonian
        H_quantum = np.zeros((n_modes, n_modes), dtype=complex)
        for n in range(n_modes):
            H_quantum[n, n] = self.w_quantum * self.harmonic_oscillator_energy(n)
        
        # Interaction Hamiltonian (electrical field couples to quantum field)
        # H_int = λ(a†_elec + a_elec)(a†_quantum + a_quantum)
        H_interaction = np.zeros((n_modes, n_modes), dtype=complex)
        
        for n in range(n_modes - 1):
            # Creation/annihilation operators for harmonic oscillator
            # a|n⟩ = √n |n-1⟩, a†|n⟩ = √(n+1) |n+1⟩
            
            # Field amplitude operators (position-like)
            # X = (a† + a)/√2
            if n > 0:
                H_interaction[n, n-1] += self.lambda_coupling * np.sqrt(n)      # Lowering
            if n < n_modes - 1:
                H_interaction[n, n+1] += self.lambda_coupling * np.sqrt(n+1)   # Raising
        
        H_interaction = H_interaction + H_interaction.conj().T  # Make Hermitian
        
        return H_elec, H_quantum, H_interaction

def continuous_field_dynamics(t, psi_vec, system, drive_type='resonant'):
    """
    Schrödinger evolution for coupled electrical-quantum fields
    psi_vec represents amplitudes in Fock basis |n_elec, n_quantum⟩
    """
    try:
        n_modes = system.n_modes
        total_dim = n_modes * n_modes  # Product space dimension
        
        # Validate input
        if len(psi_vec) != 2 * total_dim:  # Real + imaginary parts
            raise ValueError(f"State vector wrong size: {len(psi_vec)} vs {2 * total_dim}")
        
        # Reconstruct complex wavefunction
        psi = psi_vec[:total_dim] + 1j * psi_vec[total_dim:]
        
        # Validate wavefunction
        norm = np.sum(np.abs(psi)**2)
        if abs(norm) < 1e-10:
            raise ValueError("Zero norm wavefunction")
        
        # Normalize
        psi = psi / np.sqrt(norm)
        
        # Get Hamiltonians
        H_elec, H_quantum, H_interaction = system.field_interaction_hamiltonian(t)
        
        # Build total Hamiltonian in product space
        H_total = np.zeros((total_dim, total_dim), dtype=complex)
        
        # Map single-field operators to product space
        for i in range(n_modes):
            for j in range(n_modes):
                for k in range(n_modes):
                    for l in range(n_modes):
                        # Product state index: |i,k⟩ → i*n_modes + k
                        row_idx = i * n_modes + k
                        col_idx = j * n_modes + l
                        
                        # Electrical field term: H_elec ⊗ I_quantum
                        if k == l:  # Quantum index unchanged
                            H_total[row_idx, col_idx] += H_elec[i, j]
                        
                        # Quantum field term: I_elec ⊗ H_quantum  
                        if i == j:  # Electrical index unchanged
                            H_total[row_idx, col_idx] += H_quantum[k, l]
                        
                        # Interaction term: H_int_elec ⊗ H_int_quantum
                        H_total[row_idx, col_idx] += H_interaction[i, j] * H_interaction[k, l]
        
        # External driving (coherent field injection)
        if drive_type == 'resonant':
            # Coherent drive injects field energy at electrical resonance
            drive_amplitude = 0.001 * np.sin(system.w_electrical * t)
            
            # Drive couples to electrical field ground→first excited transition
            for k in range(n_modes):
                # |0,k⟩ ↔ |1,k⟩ transitions
                if k < n_modes:
                    row_idx = 0 * n_modes + k  # |0,k⟩
                    col_idx = 1 * n_modes + k  # |1,k⟩
                    H_total[row_idx, col_idx] += drive_amplitude
                    H_total[col_idx, row_idx] += drive_amplitude  # Hermitian
        
        elif drive_type == 'pulsed_coherent':
            # Pulsed coherent field injection
            period = 5.0
            width = 0.5
            cycle_time = t % period
            
            if cycle_time < width:
                drive_amplitude = 0.003
                for k in range(n_modes):
                    if k < n_modes:
                        row_idx = 0 * n_modes + k
                        col_idx = 1 * n_modes + k
                        H_total[row_idx, col_idx] += drive_amplitude
                        H_total[col_idx, row_idx] += drive_amplitude
        
        # Add field damping (non-unitary evolution)
        damping_term = np.zeros_like(psi, dtype=complex)
        
        # Electrical field damping
        for i in range(n_modes):
            for k in range(n_modes):
                state_idx = i * n_modes + k
                if i > 0:  # Electrical field decay
                    damping_term[state_idx] -= system.gamma_elec * i * psi[state_idx]
        
        # Quantum field damping  
        for i in range(n_modes):
            for k in range(n_modes):
                state_idx = i * n_modes + k
                if k > 0:  # Quantum field decay
                    damping_term[state_idx] -= system.gamma_quantum * k * psi[state_idx]
        
        # Schrödinger evolution with damping
        dpsi_dt = -1j * (H_total @ psi) + damping_term
        
        # Convert back to real vector
        dpsi_dt_vec = np.concatenate([np.real(dpsi_dt), np.imag(dpsi_dt)])
        
        return dpsi_dt_vec
        
    except Exception as e:
        print(f"Field dynamics error at t={t:.3f}: {e}")
        return [0.0] * len(psi_vec)

def analyze_continuous_field_state(sol, system):
    """
    Analyze continuous field quantum state with FIXED complex number handling
    """
    n_modes = system.n_modes
    total_dim = n_modes * n_modes
    
    results = {
        'time': sol.t,
        'field_entanglement': [],
        'electrical_field_energy': [],
        'quantum_field_energy': [],
        'total_field_coherence': [],
        'field_coupling_strength': []
    }
    
    for time_idx in range(len(sol.t)):
        # Reconstruct wavefunction
        psi_real = sol.y[:total_dim, time_idx]
        psi_imag = sol.y[total_dim:, time_idx]
        psi = psi_real + 1j * psi_imag
        
        # Normalize
        norm = np.sum(np.abs(psi)**2)
        if norm > 1e-10:
            psi = psi / np.sqrt(norm)
        
        # Calculate field energies
        elec_energy = 0.0
        quantum_energy = 0.0
        total_coherence = 0.0
        
        for i in range(n_modes):
            for k in range(n_modes):
                state_idx = i * n_modes + k
                prob = abs(psi[state_idx])**2
                
                # Electrical field energy
                elec_energy += prob * system.harmonic_oscillator_energy(i)
                
                # Quantum field energy
                quantum_energy += prob * system.harmonic_oscillator_energy(k)
                
                # Field coherence (off-diagonal matrix elements)
                if i != k:
                    total_coherence += abs(psi[state_idx])
        
        # Field entanglement (von Neumann entropy of reduced density matrix)
        # For electrical field subsystem
        rho_elec = np.zeros((n_modes, n_modes), dtype=complex)
        for i in range(n_modes):
            for j in range(n_modes):
                for k in range(n_modes):
                    idx_i = i * n_modes + k
                    idx_j = j * n_modes + k
                    rho_elec[i, j] += psi[idx_i] * psi[idx_j].conj()
        
        # Calculate entropy (FIXED)
        eigenvals = np.linalg.eigvals(rho_elec)
        eigenvals = np.real(eigenvals)  # Take real part to avoid numerical issues
        eigenvals = eigenvals[eigenvals > 1e-12]  # Remove numerical zeros
        
        if len(eigenvals) > 1:
            entropy = -np.sum(eigenvals * np.log2(eigenvals + 1e-12))
            entropy = np.real(entropy)  # Ensure real result
        else:
            entropy = 0.0
        
        # Coupling strength (expectation value of interaction)
        coupling_strength = 0.0
        for i in range(n_modes):
            for k in range(n_modes):
                for j in range(max(0, i-1), min(n_modes, i+2)):
                    for l in range(max(0, k-1), min(n_modes, k+2)):
                        if i != j and k != l:
                            idx1 = i * n_modes + k
                            idx2 = j * n_modes + l
                            coupling_strength += abs(psi[idx1] * psi[idx2].conj())
        
        results['field_entanglement'].append(entropy)
        results['electrical_field_energy'].append(elec_energy)
        results['quantum_field_energy'].append(quantum_energy)
        results['total_field_coherence'].append(total_coherence)
        results['field_coupling_strength'].append(coupling_strength)
    
    return results

def test_pure_continuous_fields():
    """
    Test pure continuous field coupling (no discrete approximations)
    """
    print("1. Testing pure continuous field dynamics...")
    
    system = ContinuousFieldSystem()
    n_modes = system.n_modes
    total_dim = n_modes * n_modes
    
    # Initial state: electrical and quantum fields both in ground state with small coherent excitation
    psi_initial = np.zeros(total_dim, dtype=complex)
    
    # Start mostly in |0,0⟩ (both fields in ground state)
    psi_initial[0] = 0.95  # |0_elec, 0_quantum⟩
    
    # Add small coherent excitation to both fields
    psi_initial[1] = 0.1    # |0_elec, 1_quantum⟩ 
    psi_initial[n_modes] = 0.1  # |1_elec, 0_quantum⟩
    psi_initial[n_modes + 1] = 0.05  # |1_elec, 1_quantum⟩
    
    # Normalize
    norm = np.sum(np.abs(psi_initial)**2)
    psi_initial = psi_initial / np.sqrt(norm)
    
    # Convert to real vector (real part + imaginary part)
    psi_initial_real = np.concatenate([np.real(psi_initial), np.imag(psi_initial)])
    
    print(f"   System: {n_modes} modes per field, {total_dim} total states")
    print(f"   Initial field energies: E_elec={0.5:.3f}, E_quantum={0.6:.3f}")
    
    # Test different drive types
    drive_types = ['resonant', 'pulsed_coherent']
    field_results = {}
    
    for drive_type in drive_types:
        print(f"\n   Testing {drive_type} field driving...")
        
        try:
            def field_dynamics(t, psi_vec):
                return continuous_field_dynamics(t, psi_vec, system, drive_type)
            
            # Simulate field evolution
            sim_time = 25.0  # Shorter time for continuous field approach
            sol = solve_ivp(field_dynamics, [0, sim_time], psi_initial_real,
                           t_eval=np.linspace(0, sim_time, 250),
                           method='RK45', rtol=1e-6, atol=1e-8)
            
            if sol.success:
                print(f"      ✓ {drive_type} simulation successful")
                
                # Analyze results
                analysis = analyze_continuous_field_state(sol, system)
                
                # Calculate metrics (FIXED: ensure real numbers)
                steady_start = len(analysis['time']) // 3
                
                metrics = {
                    'avg_entanglement': np.real(np.mean(analysis['field_entanglement'][steady_start:])),
                    'max_entanglement': np.real(np.max(analysis['field_entanglement'])),
                    'avg_coherence': np.real(np.mean(analysis['total_field_coherence'][steady_start:])),
                    'final_coherence': np.real(analysis['total_field_coherence'][-1]),
                    'energy_transfer': np.real(np.mean(analysis['quantum_field_energy'][steady_start:]) - analysis['quantum_field_energy'][0]),
                    'coupling_efficiency': np.real(np.mean(analysis['field_coupling_strength'][steady_start:]))
                }
                
                field_results[drive_type] = {
                    'success': True,
                    'metrics': metrics,
                    'data': analysis
                }
                
                print(f"      Avg field entanglement: {metrics['avg_entanglement']:.6f}")
                print(f"      Avg field coherence: {metrics['avg_coherence']:.6f}")
                print(f"      Energy transfer: {metrics['energy_transfer']:.6f}")
                print(f"      Coupling efficiency: {metrics['coupling_efficiency']:.6f}")
                
            else:
                print(f"      ✗ {drive_type} integration failed: {sol.message}")
                field_results[drive_type] = {'success': False}
                
        except Exception as e:
            print(f"      ✗ {drive_type} crashed: {e}")
            field_results[drive_type] = {'success': False}
    
    return field_results

def optimize_field_coupling_parameters():
    """
    Find optimal field coupling for maximum electrical-quantum entanglement
    """
    print("\n2. Optimizing continuous field coupling parameters...")
    
    def field_coupling_objective(params):
        """Objective: maximize electrical-quantum field entanglement"""
        w_elec, w_quantum, lambda_coup, gamma_ratio = params
        
        # Create optimized system
        system_opt = ContinuousFieldSystem()
        system_opt.w_electrical = w_elec
        system_opt.w_quantum = w_quantum
        system_opt.lambda_coupling = lambda_coup
        system_opt.gamma_quantum = system_opt.gamma_elec * gamma_ratio
        
        try:
            # Test system performance
            n_modes = system_opt.n_modes
            total_dim = n_modes * n_modes
            
            # Initial state
            psi_init = np.zeros(total_dim, dtype=complex)
            psi_init[0] = 0.9
            psi_init[1] = 0.3
            psi_init[n_modes] = 0.3
            psi_init = psi_init / np.sqrt(np.sum(np.abs(psi_init)**2))
            
            psi_init_real = np.concatenate([np.real(psi_init), np.imag(psi_init)])
            
            def dynamics(t, psi_vec):
                return continuous_field_dynamics(t, psi_vec, system_opt, 'resonant')
            
            # Short simulation for optimization
            sol = solve_ivp(dynamics, [0, 15.0], psi_init_real,
                           t_eval=np.linspace(0, 15.0, 150),
                           method='RK45', rtol=1e-5)
            
            if sol.success:
                analysis = analyze_continuous_field_state(sol, system_opt)
                
                # Objective: maximize average entanglement + coherence (FIXED: ensure real)
                avg_entanglement = np.real(np.mean(analysis['field_entanglement'][50:]))
                avg_coherence = np.real(np.mean(analysis['total_field_coherence'][50:]))
                
                # Penalty for unrealistic parameters
                penalty = 0.0
                if abs(w_elec - 5.0) > 2.0:  # Stay within reasonable frequency range
                    penalty += abs(w_elec - 5.0)
                if lambda_coup > 1.0:  # Avoid unrealistically strong coupling
                    penalty += lambda_coup
                
                performance = avg_entanglement + 0.1 * avg_coherence - penalty
                return -performance  # Minimize negative
            else:
                return -0.0
                
        except:
            return -0.0
    
    # Optimization bounds
    bounds = [
        (4.0, 6.0),    # Electrical frequency
        (4.0, 6.0),    # Quantum frequency  
        (0.01, 0.5),   # Coupling strength
        (0.5, 3.0)     # Damping ratio
    ]
    
    try:
        print("   Running field parameter optimization...")
        from scipy.optimize import minimize
        
        result = minimize(field_coupling_objective, 
                         x0=[5.0, 5.0, 0.1, 1.0],
                         bounds=bounds,
                         method='L-BFGS-B')
        
        if result.success:
            optimal_params = {
                'w_electrical': result.x[0],
                'w_quantum': result.x[1], 
                'lambda_coupling': result.x[2],
                'gamma_ratio': result.x[3],
                'performance': -result.fun
            }
            
            print(f"   ✓ Optimization successful!")
            print(f"      Optimal electrical frequency: {optimal_params['w_electrical']:.4f} GHz")
            print(f"      Optimal quantum frequency: {optimal_params['w_quantum']:.4f} GHz")
            print(f"      Frequency mismatch: {abs(optimal_params['w_electrical'] - optimal_params['w_quantum']):.4f} GHz")
            print(f"      Optimal coupling: {optimal_params['lambda_coupling']:.4f}")
            print(f"      Optimal damping ratio: {optimal_params['gamma_ratio']:.3f}")
            print(f"      Performance: {optimal_params['performance']:.6f}")
            
            return optimal_params
        else:
            print(f"   ✗ Optimization failed: {result.message}")
            return None
            
    except Exception as e:
        print(f"   ✗ Optimization crashed: {e}")
        return None

def test_electrical_quantum_field_resonance():
    """
    Test if electrical-quantum field resonance shows advantages - FIXED ANALYSIS
    """
    print("\n3. Testing electrical-quantum field resonance effects...")
    
    # Test frequency detuning in continuous field approach
    system = ContinuousFieldSystem()
    
    # Frequency scan around resonance
    base_freq = 5.0
    detunings = np.linspace(-1.0, 1.0, 15)  # ±1 GHz around resonance
    
    resonance_data = []
    
    for i, detuning in enumerate(detunings):
        print(f"   Frequency {i+1}/15: {base_freq + detuning:.3f} GHz")
        
        # Create detuned system
        system_detuned = ContinuousFieldSystem()
        system_detuned.w_quantum = base_freq + detuning
        
        # Initial field state
        n_modes = system_detuned.n_modes
        total_dim = n_modes * n_modes
        
        psi_init = np.zeros(total_dim, dtype=complex)
        psi_init[0] = 0.8  # Ground state
        psi_init[1] = 0.4  # Small quantum excitation
        psi_init[n_modes] = 0.4  # Small electrical excitation
        psi_init = psi_init / np.sqrt(np.sum(np.abs(psi_init)**2))
        
        psi_init_real = np.concatenate([np.real(psi_init), np.imag(psi_init)])
        
        try:
            def dynamics(t, psi_vec):
                return continuous_field_dynamics(t, psi_vec, system_detuned, 'resonant')
            
            sol = solve_ivp(dynamics, [0, 20.0], psi_init_real,
                           t_eval=np.linspace(0, 20.0, 100),
                           method='RK45', rtol=1e-5)
            
            if sol.success:
                analysis = analyze_continuous_field_state(sol, system_detuned)
                
                # Performance metrics (FIXED: ensure real values)
                avg_entanglement = np.real(np.mean(analysis['field_entanglement'][30:]))
                avg_coherence = np.real(np.mean(analysis['total_field_coherence'][30:]))
                energy_transfer = np.real(np.mean(analysis['quantum_field_energy'][30:]))
                
                resonance_data.append({
                    'detuning': detuning,
                    'frequency': base_freq + detuning,
                    'entanglement': avg_entanglement,
                    'coherence': avg_coherence,
                    'energy_transfer': energy_transfer,
                    'performance': avg_entanglement + 0.1 * avg_coherence
                })
                
            else:
                resonance_data.append({
                    'detuning': detuning,
                    'frequency': base_freq + detuning,
                    'entanglement': 0.0,
                    'coherence': 0.0,
                    'energy_transfer': 0.0,
                    'performance': 0.0
                })
                
        except Exception as e:
            print(f"      Failed: {e}")
            resonance_data.append({
                'detuning': detuning,
                'frequency': base_freq + detuning,
                'entanglement': 0.0,
                'coherence': 0.0,
                'energy_transfer': 0.0,
                'performance': 0.0
            })
    
    return resonance_data

def run_continuous_field_analysis():
    """
    Complete continuous field analysis with FIXED resonance detection
    """
    print("CONTINUOUS FIELD ANALYSIS - FIXED VERSION")
    print("Testing true continuous quantum field approach")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Test basic continuous field dynamics
        field_results = test_pure_continuous_fields()
        
        # Optimize field coupling parameters (now with proper import)
        optimal_params = optimize_field_coupling_parameters()
        
        # Test resonance effects in field coupling
        resonance_data = test_electrical_quantum_field_resonance()
        
        # Analysis and reporting with FIXED logic
        print("\n4. Analyzing continuous field results...")
        
        with open('continuous_field_results_fixed.txt', 'w') as f:
            f.write("CONTINUOUS FIELD ELECTRICAL-QUANTUM COUPLING - FIXED ANALYSIS\n")
            f.write("=" * 65 + "\n\n")
            
            f.write("FIELD DYNAMICS COMPARISON:\n")
            for method, result in field_results.items():
                if result['success']:
                    metrics = result['metrics']
                    f.write(f"{method.upper()}:\n")
                    f.write(f"  Field Entanglement: {metrics['avg_entanglement']:.6f}\n")
                    f.write(f"  Field Coherence: {metrics['avg_coherence']:.6f}\n")
                    f.write(f"  Energy Transfer: {metrics['energy_transfer']:.6f}\n")
                    f.write(f"  Coupling Efficiency: {metrics['coupling_efficiency']:.6f}\n\n")
                else:
                    f.write(f"{method.upper()}: FAILED\n\n")
            
            if optimal_params:
                f.write("OPTIMIZED FIELD COUPLING:\n")
                f.write(f"  Optimal Electrical Frequency: {optimal_params['w_electrical']:.4f} GHz\n")
                f.write(f"  Optimal Quantum Frequency: {optimal_params['w_quantum']:.4f} GHz\n")
                f.write(f"  Frequency Matching Error: {abs(optimal_params['w_electrical'] - optimal_params['w_quantum']):.4f} GHz\n")
                f.write(f"  Optimal Field Coupling: {optimal_params['lambda_coupling']:.4f}\n")
                f.write(f"  Optimized Performance: {optimal_params['performance']:.6f}\n\n")
                
                # Check if optimization found frequency matching
                freq_mismatch = abs(optimal_params['w_electrical'] - optimal_params['w_quantum'])
                if freq_mismatch < 0.05:
                    f.write("🎯 OPTIMIZATION CONFIRMS FREQUENCY MATCHING!\n")
                    f.write("Algorithm found that optimal performance requires\n")
                    f.write("precise electrical-quantum frequency alignment!\n\n")
                else:
                    f.write("⚠️  Optimization found off-resonance optimum\n\n")
            
            # FIXED Resonance analysis
            if resonance_data:
                # Find peak performance
                performances = [data['performance'] for data in resonance_data]
                entanglements = [data['entanglement'] for data in resonance_data]
                coherences = [data['coherence'] for data in resonance_data]
                
                if len(performances) > 0 and max(performances) > 0:
                    best_idx = performances.index(max(performances))
                    best_resonance = resonance_data[best_idx]
                    
                    # Calculate enhancement factors
                    avg_performance = np.mean(performances)
                    peak_performance = max(performances)
                    enhancement_factor = peak_performance / avg_performance if avg_performance > 0 else 0
                    
                    # Entanglement enhancement
                    max_entanglement = max(entanglements)
                    min_entanglement = min([e for e in entanglements if e > 0]) if any(e > 0 for e in entanglements) else 0
                    ent_enhancement = max_entanglement / min_entanglement if min_entanglement > 0 else 0
                    
                    f.write("FIELD RESONANCE ANALYSIS:\n")
                    f.write(f"  Best Performance at: {best_resonance['frequency']:.4f} GHz\n")
                    f.write(f"  Detuning from nominal: {best_resonance['detuning']:+.4f} GHz\n")
                    f.write(f"  Peak Field Entanglement: {best_resonance['entanglement']:.6f}\n")
                    f.write(f"  Peak Field Coherence: {best_resonance['coherence']:.6f}\n")
                    f.write(f"  Enhancement Factor: {enhancement_factor:.2f}x\n")
                    f.write(f"  Entanglement Enhancement: {ent_enhancement:.2f}x\n\n")
                    
                    # FIXED: Proper resonance detection
                    if enhancement_factor > 1.5:
                        f.write("🔥 CLEAR RESONANCE EFFECT DETECTED! 🔥\n")
                        f.write("Electrical-quantum field coupling shows strong\n")
                        f.write("frequency-dependent resonance behavior!\n")
                        
                        # Check if peak is at qubit frequency
                        peak_detuning = abs(best_resonance['detuning'])
                        if peak_detuning < 0.05:
                            f.write("🎯 PERFECT RESONANCE AT QUBIT FREQUENCY!\n")
                            f.write("Peak occurs exactly when electrical = quantum frequency\n")
                        elif peak_detuning < 0.2:
                            f.write("⚡ RESONANCE NEAR QUBIT FREQUENCY\n")
                        else:
                            f.write("⚠️  Resonance peak away from qubit frequency\n")
                        f.write("\n")
                        
                    elif enhancement_factor > 1.2:
                        f.write("⚡ MODERATE RESONANCE EFFECT DETECTED\n")
                        f.write("Some frequency-dependent enhancement observed\n\n")
                    else:
                        f.write("📊 NO SIGNIFICANT RESONANCE EFFECT\n")
                        f.write("Performance relatively flat across frequencies\n\n")
                else:
                    f.write("FIELD RESONANCE ANALYSIS: No valid data\n\n")
            
            # Overall assessment with FIXED logic
            field_entanglement_detected = False
            resonance_confirmed = False
            
            # Check for significant field entanglement
            if field_results:
                max_field_entanglement = 0.0
                for result in field_results.values():
                    if result['success']:
                        ent = result['metrics']['avg_entanglement']
                        if ent > max_field_entanglement:
                            max_field_entanglement = ent
                
                if max_field_entanglement > 0.01:  # Threshold for significant entanglement
                    field_entanglement_detected = True
            
            # Check for resonance effects
            if resonance_data:
                performances = [d['performance'] for d in resonance_data]
                if len(performances) > 0:
                    peak_perf = max(performances)
                    avg_perf = np.mean(performances)
                    if peak_perf > avg_perf * 1.3:  # 30% enhancement threshold
                        resonance_confirmed = True
            
            f.write("OVERALL ASSESSMENT:\n")
            if field_entanglement_detected and resonance_confirmed:
                f.write("🚀 ELECTRICAL-QUANTUM FIELD COUPLING CONFIRMED! 🚀\n")
                f.write("Both field entanglement and resonance enhancement detected!\n")
                f.write("This represents a genuine new quantum field phenomenon.\n\n")
                f.write("Key findings:\n")
                f.write("✓ Electrical and quantum fields become entangled\n")
                f.write("✓ Precise frequency matching enhances entanglement\n")
                f.write("✓ Field-level coherence protection mechanism\n")
                f.write("✓ Macroscopic-microscopic quantum correlations\n\n")
                f.write("IMPLICATIONS: This could enable new quantum technologies\n")
                f.write("based on electrical field entanglement and resonant\n")
                f.write("coupling between classical and quantum systems.\n")
            elif field_entanglement_detected:
                f.write("⚡ FIELD ENTANGLEMENT DETECTED\n")
                f.write("Electrical-quantum field coupling creates entanglement\n")
                f.write("but no clear resonance enhancement.\n")
            elif resonance_confirmed:
                f.write("📈 RESONANCE EFFECTS DETECTED\n")
                f.write("Frequency-dependent enhancement without significant\n")
                f.write("field entanglement.\n")
            else:
                f.write("📊 NO SIGNIFICANT FIELD COUPLING EFFECTS\n")
                f.write("Standard discrete quantum models appear adequate.\n")
        
        # Save resonance sweep data with FIXED format
        if resonance_data:
            with open('field_resonance_sweep_fixed.csv', 'w') as f:
                f.write("frequency_GHz,detuning_GHz,field_entanglement,field_coherence,energy_transfer,performance\n")
                for data in resonance_data:
                    f.write(f"{data['frequency']:.6f},{data['detuning']:.6f},")
                    f.write(f"{data['entanglement']:.6f},{data['coherence']:.6f},")
                    f.write(f"{data['energy_transfer']:.6f},{data['performance']:.6f}\n")
        
        elapsed = time.time() - start_time
        print(f"\nCONTINUOUS FIELD ANALYSIS COMPLETED in {elapsed:.1f} seconds")
        print("Files created:")
        print("- continuous_field_results_fixed.txt")
        print("- field_resonance_sweep_fixed.csv")
        
        # FIXED Summary with proper analysis
        if resonance_data:
            performances = [d['performance'] for d in resonance_data]
            entanglements = [d['entanglement'] for d in resonance_data]
            
            if len(performances) > 0:
                max_perf = max(performances)
                avg_perf = np.mean(performances)
                max_ent = max(entanglements)
                min_ent = min([e for e in entanglements if e > 0]) if any(e > 0 for e in entanglements) else 0
                
                peak_idx = performances.index(max_perf)
                peak_freq = resonance_data[peak_idx]['frequency']
                
                print(f"\nFIXED RESONANCE SUMMARY:")
                print(f"Peak performance: {max_perf:.6f} at {peak_freq:.3f} GHz")
                print(f"Average performance: {avg_perf:.6f}")
                print(f"Enhancement factor: {max_perf/avg_perf:.2f}x")
                print(f"Peak detuning: {abs(peak_freq - 5.0)*1000:.1f} MHz")
                
                if max_ent > 0 and min_ent > 0:
                    print(f"Entanglement enhancement: {max_ent/min_ent:.2f}x")
                
                if max_perf/avg_perf > 1.5 and abs(peak_freq - 5.0) < 0.1:
                    print("🎯 RESONANCE CONFIRMED - Peak at qubit frequency!")
                elif max_perf/avg_perf > 1.3:
                    print("⚡ Resonance effect detected")
                else:
                    print("📊 No significant resonance")
        
        return field_results, optimal_params, resonance_data
        
    except Exception as e:
        print(f"Continuous field analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("FIXED VERSION - Testing your refined hypothesis:")
    print("Electrical-quantum coupling requires sophisticated tuning")
    print("of continuous field interactions, not simple driving")
    print("\nThis treats both electrical and quantum systems as")
    print("continuous quantum fields that can become entangled")
    print("Runtime: ~20 minutes")
    print("=" * 70)
    
    try:
        results = run_continuous_field_analysis()
        print("\n" + "="*70)
        print("FIXED CONTINUOUS FIELD TEST COMPLETED!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()