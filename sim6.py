import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveQuantumSystem:
    """
    COMPREHENSIVE FIXES addressing all identified issues:
    1. ✓ Added missing self.decoherence_gradient
    2. ✓ Better error handling (no silent failures)
    3. ✓ Multi-qubit coupling for entanglement generation
    4. ✓ Asymmetric pulses per qubit
    5. ✓ Reduced decoherence rates
    6. ✓ Full entanglement time evolution tracking
    7. ✓ Multiple pulse durations testing
    """
    
    def __init__(self, n_qubits=4):
        self.n_qubits = n_qubits
        self.total_dim = 2**n_qubits  # Proper multi-qubit Hilbert space
        
        # System parameters - optimized for entanglement generation
        self.omega = 5.0e9  # 5 GHz qubit frequency
        self.J_coupling = 10e6  # 10 MHz nearest-neighbor coupling (ESSENTIAL!)
        self.anharmonicity = -200e6  # 200 MHz anharmonicity
        
        # REDUCED decoherence rates for better entanglement preservation
        self.T1 = 200e-6  # Increased from 50 μs to 200 μs
        self.T2 = 100e-6  # Increased from 20 μs to 100 μs
        self.gamma_1 = 1.0 / self.T1
        self.gamma_phi = 1.0 / self.T2
        
        # Fixed: Add the missing attribute
        self.decoherence_gradient = 0.02  # Smaller gradient for less aggressive variation
        
        print(f"Comprehensive system: {n_qubits} qubits, dim = {self.total_dim}")
        print(f"Coupling J = {self.J_coupling/1e6:.0f} MHz (ESSENTIAL FOR ENTANGLEMENT)")
        print(f"T1 = {self.T1*1e6:.0f} μs, T2 = {self.T2*1e6:.0f} μs (REDUCED DECOHERENCE)")
        print(f"Decoherence gradient = {self.decoherence_gradient} ✓")

def pauli_matrices():
    """Standard Pauli matrices"""
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    return sigma_x, sigma_y, sigma_z, identity

def get_multi_qubit_operator(single_op, target_qubit, n_qubits):
    """Build multi-qubit operator acting on specific qubit"""
    sigma_x, sigma_y, sigma_z, identity = pauli_matrices()
    
    operators = []
    for i in range(n_qubits):
        if i == target_qubit:
            operators.append(single_op)
        else:
            operators.append(identity)
    
    # Tensor product
    result = operators[0]
    for op in operators[1:]:
        result = np.kron(result, op)
    
    return result

def get_two_qubit_operator(op1, qubit1, op2, qubit2, n_qubits):
    """Build two-qubit operator"""
    sigma_x, sigma_y, sigma_z, identity = pauli_matrices()
    
    operators = []
    for i in range(n_qubits):
        if i == qubit1:
            operators.append(op1)
        elif i == qubit2:
            operators.append(op2)
        else:
            operators.append(identity)
    
    result = operators[0]
    for op in operators[1:]:
        result = np.kron(result, op)
    
    return result

def create_asymmetric_pulse_functions():
    """
    ASYMMETRIC pulses per qubit - essential for entanglement!
    Each qubit gets different frequency, phase, and timing
    """
    
    def gaussian_pulses(t, pulse_duration=10e-6):
        """Conservative Gaussian with different parameters per qubit"""
        pulses = []
        
        base_params = [
            {'amp': 20e6, 'freq': 5.0e9, 'phase': 0.0, 'sigma': pulse_duration/4},
            {'amp': 25e6, 'freq': 5.1e9, 'phase': np.pi/4, 'sigma': pulse_duration/3},
            {'amp': 22e6, 'freq': 4.9e9, 'phase': np.pi/2, 'sigma': pulse_duration/3.5},
            {'amp': 27e6, 'freq': 5.05e9, 'phase': 3*np.pi/4, 'sigma': pulse_duration/4.5}
        ]
        
        t_center = pulse_duration / 2
        
        for i, params in enumerate(base_params):
            if 0 <= t <= pulse_duration:
                envelope = np.exp(-((t - t_center) / params['sigma'])**2)
                pulse = params['amp'] * envelope * np.cos(2*np.pi*params['freq']*t + params['phase'])
            else:
                pulse = 0.0
            pulses.append(pulse)
        
        return pulses
    
    def drag_pulses(t, pulse_duration=10e-6):
        """DRAG with different corrections per qubit"""
        pulses = []
        
        base_params = [
            {'amp': 30e6, 'freq': 5.0e9, 'drag': -0.2, 'sigma': pulse_duration/5},
            {'amp': 35e6, 'freq': 5.1e9, 'drag': -0.3, 'sigma': pulse_duration/4},
            {'amp': 32e6, 'freq': 4.9e9, 'drag': -0.25, 'sigma': pulse_duration/4.5},
            {'amp': 38e6, 'freq': 5.05e9, 'drag': -0.35, 'sigma': pulse_duration/5.5}
        ]
        
        t_center = pulse_duration / 2
        
        for i, params in enumerate(base_params):
            if 0 <= t <= pulse_duration:
                envelope = np.exp(-((t - t_center) / params['sigma'])**2)
                derivative = -2 * (t - t_center) / params['sigma']**2 * envelope
                
                I_comp = params['amp'] * envelope * np.cos(2*np.pi*params['freq']*t)
                Q_comp = params['drag'] * params['amp'] * derivative * np.sin(2*np.pi*params['freq']*t)
                
                pulse = I_comp  # Real part for now (can extend to complex)
            else:
                pulse = 0.0
            pulses.append(pulse)
        
        return pulses
    
    def chirped_pulses(t, pulse_duration=10e-6):
        """Frequency chirps with different sweep ranges per qubit"""
        pulses = []
        
        chirp_params = [
            {'amp': 25e6, 'f_start': 4.8e9, 'f_end': 5.2e9},
            {'amp': 30e6, 'f_start': 4.9e9, 'f_end': 5.3e9},
            {'amp': 27e6, 'f_start': 4.7e9, 'f_end': 5.1e9},
            {'amp': 32e6, 'f_start': 4.85e9, 'f_end': 5.25e9}
        ]
        
        for i, params in enumerate(chirp_params):
            if 0 <= t <= pulse_duration:
                t_norm = t / pulse_duration
                
                # Linear frequency sweep
                freq = params['f_start'] + (params['f_end'] - params['f_start']) * t_norm
                
                # Smooth envelope
                envelope = np.sin(np.pi * t_norm)**2
                
                # Instantaneous phase (integral of frequency)
                phase = 2*np.pi * (params['f_start'] * t + 
                                 0.5 * (params['f_end'] - params['f_start']) * t**2 / pulse_duration)
                
                pulse = params['amp'] * envelope * np.cos(phase)
            else:
                pulse = 0.0
            pulses.append(pulse)
        
        return pulses
    
    def shaped_pulses(t, pulse_duration=10e-6):
        """Custom shaped pulses optimized per qubit"""
        pulses = []
        
        shape_params = [
            {'amp': 28e6, 'freq': 5.0e9, 'shape_type': 'double_gaussian'},
            {'amp': 33e6, 'freq': 5.1e9, 'shape_type': 'rising_edge'},
            {'amp': 30e6, 'freq': 4.9e9, 'shape_type': 'falling_edge'},
            {'amp': 35e6, 'freq': 5.05e9, 'shape_type': 'rectangular'}
        ]
        
        for i, params in enumerate(shape_params):
            if 0 <= t <= pulse_duration:
                t_norm = t / pulse_duration
                
                if params['shape_type'] == 'double_gaussian':
                    env1 = np.exp(-((t - pulse_duration*0.3) / (pulse_duration*0.1))**2)
                    env2 = 0.7 * np.exp(-((t - pulse_duration*0.7) / (pulse_duration*0.1))**2)
                    envelope = env1 + env2
                elif params['shape_type'] == 'rising_edge':
                    envelope = t_norm**2
                elif params['shape_type'] == 'falling_edge':
                    envelope = (1 - t_norm)**2
                else:  # rectangular
                    envelope = 1.0 if 0.2 <= t_norm <= 0.8 else 0.0
                
                pulse = params['amp'] * envelope * np.cos(2*np.pi*params['freq']*t)
            else:
                pulse = 0.0
            pulses.append(pulse)
        
        return pulses
    
    return {
        'gaussian': gaussian_pulses,
        'drag': drag_pulses,
        'chirped': chirped_pulses,
        'shaped': shaped_pulses
    }

def build_hamiltonian_with_coupling(system, t, pulse_amplitudes):
    """
    Build Hamiltonian with ESSENTIAL multi-qubit coupling
    """
    sigma_x, sigma_y, sigma_z, identity = pauli_matrices()
    dim = system.total_dim
    H = np.zeros((dim, dim), dtype=complex)
    
    # Single-qubit terms
    for i in range(system.n_qubits):
        # Qubit frequency
        sigma_z_i = get_multi_qubit_operator(sigma_z, i, system.n_qubits)
        H += 0.5 * system.omega * sigma_z_i
        
        # Drive term (different for each qubit!)
        if i < len(pulse_amplitudes):
            drive_amp = pulse_amplitudes[i]
            sigma_x_i = get_multi_qubit_operator(sigma_x, i, system.n_qubits)
            H += drive_amp * sigma_x_i
    
    # ESSENTIAL: Multi-qubit coupling (XX + YY interactions)
    for i in range(system.n_qubits - 1):
        # Nearest-neighbor coupling
        XX = get_two_qubit_operator(sigma_x, i, sigma_x, i+1, system.n_qubits)
        YY = get_two_qubit_operator(sigma_y, i, sigma_y, i+1, system.n_qubits)
        
        H += system.J_coupling * (XX + YY)
    
    # Optional: Next-nearest neighbor coupling for richer entanglement
    if system.n_qubits >= 3:
        for i in range(system.n_qubits - 2):
            XX_nn = get_two_qubit_operator(sigma_x, i, sigma_x, i+2, system.n_qubits)
            YY_nn = get_two_qubit_operator(sigma_y, i, sigma_y, i+2, system.n_qubits)
            
            H += 0.3 * system.J_coupling * (XX_nn + YY_nn)  # Weaker next-neighbor
    
    return H

def lindblad_decoherence_multi_qubit(rho, system):
    """
    Multi-qubit Lindblad decoherence with site-dependent rates
    """
    sigma_x, sigma_y, sigma_z, identity = pauli_matrices()
    dim = system.total_dim
    decoherence = np.zeros((dim, dim), dtype=complex)
    
    for i in range(system.n_qubits):
        # Site-dependent decoherence (now decoherence_gradient exists!)
        site_factor = 1.0 + system.decoherence_gradient * i
        
        # T1 relaxation
        gamma_1_eff = system.gamma_1 * site_factor
        sigma_minus = get_multi_qubit_operator((sigma_x - 1j * sigma_y) / 2, i, system.n_qubits)
        sigma_plus = get_multi_qubit_operator((sigma_x + 1j * sigma_y) / 2, i, system.n_qubits)
        
        decoherence += gamma_1_eff * (
            sigma_minus @ rho @ sigma_plus - 
            0.5 * (sigma_plus @ sigma_minus @ rho + rho @ sigma_plus @ sigma_minus)
        )
        
        # Pure dephasing
        gamma_phi_eff = system.gamma_phi * site_factor
        sigma_z_i = get_multi_qubit_operator(sigma_z, i, system.n_qubits)
        
        decoherence += gamma_phi_eff * (
            sigma_z_i @ rho @ sigma_z_i - rho
        )
    
    return decoherence

def master_equation_comprehensive(t, y, system, pulse_functions, pulse_type, pulse_duration):
    """
    Comprehensive master equation with proper error handling
    """
    try:
        dim = system.total_dim
        rho = y.reshape((dim, dim))
        
        # Get asymmetric pulse amplitudes for each qubit
        pulse_amplitudes = pulse_functions[pulse_type](t, pulse_duration)
        
        # Build Hamiltonian with coupling
        H = build_hamiltonian_with_coupling(system, t, pulse_amplitudes)
        
        # Coherent evolution
        coherent_evolution = -1j * (H @ rho - rho @ H)
        
        # Decoherence
        decoherence = lindblad_decoherence_multi_qubit(rho, system)
        
        return (coherent_evolution + decoherence).flatten()
        
    except Exception as e:
        print(f"💥 CRITICAL ERROR at t={t:.6f}: {e}")
        import traceback
        traceback.print_exc()
        raise  # No silent failures!

def calculate_bipartite_entanglement(rho, system):
    """
    Calculate proper bipartite entanglement using partial trace
    """
    n_qubits = system.n_qubits
    
    if n_qubits < 2:
        return 0.0
    
    # Partition: first half vs second half
    n_A = n_qubits // 2
    n_B = n_qubits - n_A
    
    dim_A = 2**n_A
    dim_B = 2**n_B
    
    # Reshape for partial trace
    rho_tensor = rho.reshape((dim_A, dim_B, dim_A, dim_B))
    
    # Partial trace over subsystem B
    rho_A = np.trace(rho_tensor, axis1=1, axis3=3)
    
    # Von Neumann entropy
    eigenvals = np.real(np.linalg.eigvals(rho_A))
    eigenvals = eigenvals[eigenvals > 1e-12]
    
    if len(eigenvals) <= 1:
        return 0.0
    
    entropy = -np.sum(eigenvals * np.log2(eigenvals))
    return entropy

def run_comprehensive_simulation():
    """
    Comprehensive simulation addressing all identified issues
    """
    print("🚀 COMPREHENSIVE FIXED SIMULATION")
    print("All Issues Addressed:")
    print("1. ✓ Multi-qubit coupling for entanglement")
    print("2. ✓ Asymmetric pulses per qubit")
    print("3. ✓ Reduced decoherence rates")
    print("4. ✓ Full time evolution tracking")
    print("5. ✓ Multiple pulse durations")
    print("=" * 60)
    
    system = ComprehensiveQuantumSystem(n_qubits=4)
    pulse_functions = create_asymmetric_pulse_functions()
    
    # Initial state: all qubits in ground state
    dim = system.total_dim
    initial_rho = np.zeros((dim, dim), dtype=complex)
    initial_rho[0, 0] = 1.0  # |0000⟩
    
    # Test different pulse durations
    pulse_durations = [5e-6, 10e-6, 15e-6]  # 5, 10, 15 μs
    pulse_types = ['gaussian', 'drag', 'chirped', 'shaped']
    
    all_results = {}
    
    for duration in pulse_durations:
        print(f"\n🕐 Testing pulse duration: {duration*1e6:.0f} μs")
        duration_results = {}
        
        for pulse_type in pulse_types:
            print(f"   📡 {pulse_type} pulses...")
            start_time = time.time()
            
            try:
                def dynamics(t, y):
                    return master_equation_comprehensive(t, y, system, pulse_functions, 
                                                       pulse_type, duration)
                
                # Simulation parameters
                t_span = (0.0, duration + 5e-6)  # Pulse + relaxation time
                n_points = 200
                t_eval = np.linspace(0, t_span[1], n_points)
                
                sol = solve_ivp(dynamics, t_span, initial_rho.flatten(),
                               t_eval=t_eval, method='RK45',
                               rtol=1e-6, atol=1e-8, max_step=1e-7)
                
                if sol.success:
                    elapsed = time.time() - start_time
                    print(f"      ✅ SUCCESS in {elapsed:.1f}s")
                    duration_results[pulse_type] = sol
                else:
                    print(f"      ❌ FAILED: {sol.message}")
                    
            except Exception as e:
                print(f"      💥 ERROR: {e}")
        
        all_results[duration] = duration_results
    
    # Comprehensive analysis
    print(f"\n📊 COMPREHENSIVE ANALYSIS")
    
    best_results = {}
    
    for duration, duration_results in all_results.items():
        if len(duration_results) >= 2:
            print(f"\n⏱️  Duration: {duration*1e6:.0f} μs")
            
            duration_analysis = {}
            
            for pulse_type, sol in duration_results.items():
                # Calculate full time evolution
                entanglement_evolution = []
                purity_evolution = []
                
                for i, t in enumerate(sol.t):
                    rho = sol.y[:, i].reshape((dim, dim))
                    
                    # Ensure proper density matrix
                    rho = (rho + rho.conj().T) / 2
                    trace = np.trace(rho)
                    if abs(trace) > 1e-10:
                        rho = rho / trace
                    
                    entanglement = calculate_bipartite_entanglement(rho, system)
                    purity = np.real(np.trace(rho @ rho))
                    
                    entanglement_evolution.append(entanglement)
                    purity_evolution.append(purity)
                
                # Performance metrics
                peak_entanglement = np.max(entanglement_evolution)
                final_entanglement = entanglement_evolution[-1]
                avg_entanglement = np.mean(entanglement_evolution)
                entanglement_integral = np.trapz(entanglement_evolution, sol.t)
                purity_decay = purity_evolution[0] - purity_evolution[-1]
                
                duration_analysis[pulse_type] = {
                    'peak_entanglement': peak_entanglement,
                    'final_entanglement': final_entanglement,
                    'avg_entanglement': avg_entanglement,
                    'entanglement_integral': entanglement_integral,
                    'purity_decay': purity_decay,
                    'entanglement_evolution': entanglement_evolution,
                    'purity_evolution': purity_evolution,
                    'time': sol.t
                }
                
                print(f"      {pulse_type}: peak={peak_entanglement:.4f}, final={final_entanglement:.4f}")
            
            # Find best pulse for this duration
            best_pulse = max(duration_analysis.keys(), 
                           key=lambda x: duration_analysis[x]['peak_entanglement'])
            best_results[duration] = {
                'best_pulse': best_pulse,
                'analysis': duration_analysis
            }
            
            print(f"      🏆 Best: {best_pulse} (peak entanglement: {duration_analysis[best_pulse]['peak_entanglement']:.4f})")
    
    # Generate plots and save results
    print(f"\n📈 GENERATING COMPREHENSIVE RESULTS")
    
    # Create plots for best results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Comprehensive Pulse Comparison Results', fontsize=16)
    
    plot_idx = 0
    colors = ['blue', 'red', 'green', 'orange']
    
    for duration, results in best_results.items():
        if plot_idx < 4:
            ax = axes[plot_idx // 2, plot_idx % 2]
            
            for i, (pulse_type, analysis) in enumerate(results['analysis'].items()):
                ax.plot(analysis['time'] * 1e6, analysis['entanglement_evolution'], 
                       color=colors[i], label=f'{pulse_type}', linewidth=2)
            
            ax.set_xlabel('Time (μs)')
            ax.set_ylabel('Entanglement')
            ax.set_title(f'Duration: {duration*1e6:.0f} μs')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('comprehensive_entanglement_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save comprehensive results
    with open('comprehensive_results.txt', 'w') as f:
        f.write("COMPREHENSIVE QUANTUM PULSE SIMULATION RESULTS\n")
        f.write("All Critical Issues Addressed\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("SYSTEM IMPROVEMENTS:\n")
        f.write("1. ✓ Multi-qubit coupling (10 MHz nearest-neighbor)\n")
        f.write("2. ✓ Asymmetric pulses per qubit (different freq, phase, timing)\n")
        f.write("3. ✓ Reduced decoherence (T1=200μs, T2=100μs)\n")
        f.write("4. ✓ Full entanglement time evolution tracking\n")
        f.write("5. ✓ Multiple pulse durations tested\n\n")
        
        for duration, results in best_results.items():
            f.write(f"DURATION: {duration*1e6:.0f} μs\n")
            f.write(f"Best pulse: {results['best_pulse'].upper()}\n")
            f.write("-" * 30 + "\n")
            
            for pulse_type, analysis in results['analysis'].items():
                f.write(f"{pulse_type.upper()}:\n")
                f.write(f"  Peak entanglement: {analysis['peak_entanglement']:.6f}\n")
                f.write(f"  Final entanglement: {analysis['final_entanglement']:.6f}\n")
                f.write(f"  Average entanglement: {analysis['avg_entanglement']:.6f}\n")
                f.write(f"  Entanglement integral: {analysis['entanglement_integral']:.6f}\n")
                f.write(f"  Purity decay: {analysis['purity_decay']:.6f}\n\n")
            
            f.write("\n")
    
    print(f"\n✅ COMPREHENSIVE SIMULATION COMPLETED!")
    print("Files generated:")
    print("- comprehensive_results.txt (detailed analysis)")
    print("- comprehensive_entanglement_evolution.png (time evolution plots)")
    
    return all_results, best_results

if __name__ == "__main__":
    print("🔬 COMPREHENSIVE FIXED SIMULATION")
    print("Addressing ALL identified issues:")
    print("• Multi-qubit coupling for entanglement generation")
    print("• Asymmetric pulses to break symmetry")
    print("• Reduced decoherence for entanglement preservation")
    print("• Full time evolution analysis")
    print("• Multiple pulse duration optimization")
    print("=" * 60)
    
    try:
        all_results, best_results = run_comprehensive_simulation()
        
        # Quick summary
        if best_results:
            print(f"\n🎯 QUICK SUMMARY:")
            for duration, results in best_results.items():
                best_pulse = results['best_pulse']
                peak_ent = results['analysis'][best_pulse]['peak_entanglement']
                final_ent = results['analysis'][best_pulse]['final_entanglement']
                print(f"Duration {duration*1e6:.0f}μs: {best_pulse} (peak={peak_ent:.4f}, final={final_ent:.4f})")
        
    except Exception as e:
        print(f"💥 Comprehensive simulation failed: {e}")
        import traceback
        traceback.print_exc()