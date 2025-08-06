import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar, minimize
import time

print("TUNED ELECTRICAL-QUANTUM RESONANT COUPLING TEST")
print("Testing hypothesis: Precisely tuned electrical-quantum coupling")
print("can enhance coherence through resonant impedance matching")
print("=" * 70)

class TunedCouplingSystem:
    """
    Test electrical-quantum systems with precise tuning
    """
    
    def __init__(self):
        # Base quantum system
        self.w_quantum = 5.0      # 5 GHz quantum transition
        self.T1 = 80.0            # 80 μs relaxation
        self.T2 = 40.0            # 40 μs dephasing
        
        # Electrical circuit (tunable)
        self.w_electrical = 5.0   # Start matched to quantum
        self.Q_electrical = 1000  # Quality factor
        self.Z0 = 50.0           # Characteristic impedance
        
        # Coupling parameters (tunable)
        self.g_base = 10.0       # Base coupling strength (MHz)
        self.coupling_phase = 0.0 # Phase relationship
        
    def electrical_impedance(self, frequency):
        """
        Calculate electrical circuit impedance vs frequency
        """
        # RLC circuit impedance: Z(ω) = R + jωL + 1/(jωC)
        w = frequency
        w0 = self.w_electrical
        Q = self.Q_electrical
        
        # Resonant circuit
        Z_real = self.Z0 * (1 + Q**2 * ((w/w0) - (w0/w))**2) / (1 + Q**2 * ((w/w0) - (w0/w))**2)
        Z_imag = self.Z0 * Q * ((w/w0) - (w0/w)) / (1 + Q**2 * ((w/w0) - (w0/w))**2)
        
        return Z_real + 1j * Z_imag
    
    def quantum_impedance(self, frequency):
        """
        Calculate quantum system 'impedance' - how it responds to electrical driving
        """
        # Quantum system response depends on detuning from transition
        detuning = frequency - self.w_quantum
        
        # Susceptibility ~ 1/(detuning + i*gamma)
        gamma_total = 1/self.T2 + 1/(2*self.T1)  # Total line width
        
        susceptibility = 1.0 / (detuning + 1j * gamma_total)
        
        # Convert to impedance-like quantity
        quantum_Z = 1.0 / (susceptibility * self.g_base)
        
        return quantum_Z
    
    def impedance_matching_factor(self, frequency):
        """
        Calculate how well electrical and quantum impedances match
        """
        Z_elec = self.electrical_impedance(frequency)
        Z_quantum = self.quantum_impedance(frequency)
        
        # Impedance matching: minimize |Z1 - Z2|²/(|Z1| + |Z2|)²
        mismatch = abs(Z_elec - Z_quantum)**2 / (abs(Z_elec) + abs(Z_quantum))**2
        matching_factor = 1.0 / (1.0 + mismatch)
        
        return matching_factor
    
    def resonant_coupling_strength(self, electrical_field, frequency):
        """
        Coupling strength enhanced by impedance matching
        """
        base_coupling = self.g_base * electrical_field * 0.001
        
        # Enhancement from impedance matching
        matching = self.impedance_matching_factor(frequency)
        
        # Phase coherence factor
        phase_factor = np.cos(self.coupling_phase)
        
        # Total enhanced coupling
        enhanced_coupling = base_coupling * matching * phase_factor
        
        return enhanced_coupling, matching

def tuned_system_dynamics(t, y, system, electrical_params):
    """
    Dynamics with tuned electrical-quantum coupling
    """
    try:
        # Extract state: [V_elec, I_elec, population_0, population_1, coherence_real, coherence_imag]
        V_e, I_e, p0, p1, coh_r, coh_i = y
        
        # Generate tuned electrical drive
        if electrical_params['type'] == 'resonant_continuous':
            amp = electrical_params['amplitude']
            freq = electrical_params['frequency']
            phase = electrical_params.get('phase', 0.0)
            
            drive = amp * np.sin(freq * t + phase)
            
        elif electrical_params['type'] == 'impedance_matched':
            # Dynamic impedance matching
            amp = electrical_params['amplitude']
            
            # Find optimal frequency at this time for maximum impedance matching
            def neg_matching(freq):
                return -system.impedance_matching_factor(freq)
            
            try:
                result = minimize_scalar(neg_matching, bounds=(4.0, 6.0), method='bounded')
                optimal_freq = result.x
            except:
                optimal_freq = system.w_quantum
            
            drive = amp * np.sin(optimal_freq * t)
            
        elif electrical_params['type'] == 'chirped_resonant':
            # Frequency chirp through resonance
            amp = electrical_params['amplitude']
            chirp_rate = electrical_params.get('chirp_rate', 0.1)
            center_freq = electrical_params['frequency']
            
            instantaneous_freq = center_freq + chirp_rate * np.sin(0.1 * t)
            drive = amp * np.sin(instantaneous_freq * t)
            
        else:  # standard continuous or pulsed
            amp = electrical_params['amplitude']
            if electrical_params['type'] == 'continuous':
                drive = amp * np.sin(system.w_electrical * t)
            else:  # pulsed
                period = electrical_params.get('period', 5.0)
                width = electrical_params.get('width', 0.5)
                cycle_time = t % period
                drive = amp if cycle_time < width else 0.0
        
        # Electrical circuit dynamics
        damping = system.w_electrical / system.Q_electrical
        dV_dt = I_e
        dI_dt = -system.w_electrical**2 * V_e - damping * I_e + drive
        
        # Get enhanced coupling
        if 'frequency' in electrical_params:
            coupling, matching = system.resonant_coupling_strength(V_e, electrical_params['frequency'])
        else:
            coupling, matching = system.resonant_coupling_strength(V_e, system.w_electrical)
        
        # Quantum dynamics with resonant coupling
        # Simple two-level system with enhanced coupling
        dp0_dt = (p1 / system.T1) - coupling * coh_i
        dp1_dt = -(p1 / system.T1) + coupling * coh_i
        
        # Coherence with impedance-matched coupling
        dcoh_r_dt = -coh_r / system.T2 + coupling * (p0 - p1) * matching
        dcoh_i_dt = -coh_i / system.T2 - system.w_quantum * coh_r
        
        return [dV_dt, dI_dt, dp0_dt, dp1_dt, dcoh_r_dt, dcoh_i_dt]
        
    except Exception as e:
        print(f"Tuned dynamics error at t={t:.3f}: {e}")
        return [0.0] * 6

def find_optimal_electrical_parameters(system, optimization_type='impedance_matching'):
    """
    Find optimal electrical parameters for maximum coherence enhancement
    """
    print(f"   Optimizing for {optimization_type}...")
    
    initial_state = [0.02, 0.0, 1.0, 0.0, 0.0, 0.0]  # Ground state + small electrical excitation
    sim_time = 50.0
    
    best_performance = 0.0
    best_params = None
    
    if optimization_type == 'impedance_matching':
        # Search for optimal frequency and amplitude for impedance matching
        
        def objective(params):
            amp, freq = params
            
            electrical_params = {
                'type': 'resonant_continuous',
                'amplitude': amp,
                'frequency': freq
            }
            
            try:
                def dynamics(t, y):
                    return tuned_system_dynamics(t, y, system, electrical_params)
                
                sol = solve_ivp(dynamics, [0, sim_time], initial_state,
                               t_eval=np.linspace(0, sim_time, 200),
                               method='RK45', rtol=1e-6)
                
                if sol.success:
                    # Calculate average coherence
                    coherence = np.sqrt(sol.y[4, :]**2 + sol.y[5, :]**2)
                    steady_start = len(coherence) // 3
                    avg_coherence = np.mean(coherence[steady_start:])
                    
                    # Calculate impedance matching at this frequency
                    matching_factor = system.impedance_matching_factor(freq)
                    
                    # Objective: maximize coherence * impedance matching
                    performance = avg_coherence * matching_factor
                    return -performance  # Minimize negative
                else:
                    return -0.0
                    
            except:
                return -0.0
        
        # Optimize amplitude and frequency
        try:
            result = minimize(objective, x0=[0.02, 5.0], 
                            bounds=[(0.001, 0.1), (4.0, 6.0)],
                            method='L-BFGS-B')
            
            if result.success:
                best_params = {
                    'type': 'resonant_continuous',
                    'amplitude': result.x[0],
                    'frequency': result.x[1]
                }
                best_performance = -result.fun
                
        except Exception as e:
            print(f"      Optimization failed: {e}")
    
    elif optimization_type == 'phase_matching':
        # Search for optimal phase relationships
        
        frequencies = np.linspace(4.5, 5.5, 5)
        phases = np.linspace(0, 2*np.pi, 6)
        amplitudes = [0.01, 0.02, 0.03]
        
        for freq in frequencies:
            for phase in phases:
                for amp in amplitudes:
                    electrical_params = {
                        'type': 'resonant_continuous',
                        'amplitude': amp,
                        'frequency': freq,
                        'phase': phase
                    }
                    
                    # Update system phase
                    system.coupling_phase = phase
                    
                    try:
                        def dynamics(t, y):
                            return tuned_system_dynamics(t, y, system, electrical_params)
                        
                        sol = solve_ivp(dynamics, [0, 30.0], initial_state,
                                       t_eval=np.linspace(0, 30.0, 150),
                                       method='RK45', rtol=1e-6)
                        
                        if sol.success:
                            coherence = np.sqrt(sol.y[4, :]**2 + sol.y[5, :]**2)
                            avg_coherence = np.mean(coherence[50:])
                            
                            if avg_coherence > best_performance:
                                best_performance = avg_coherence
                                best_params = electrical_params.copy()
                                
                    except:
                        continue
    
    return best_performance, best_params

def test_resonant_coupling_hypothesis():
    """
    Test if tuned resonant coupling can beat both continuous and pulsed
    """
    print("1. Testing Resonant Coupling Hypothesis...")
    
    system = TunedCouplingSystem()
    
    # Find optimal impedance matching parameters
    print("   Finding optimal impedance matching...")
    impedance_performance, impedance_params = find_optimal_electrical_parameters(
        system, 'impedance_matching')
    
    print(f"      Best impedance matching: performance = {impedance_performance:.6f}")
    if impedance_params:
        print(f"      Optimal frequency: {impedance_params['frequency']:.3f} GHz")
        print(f"      Optimal amplitude: {impedance_params['amplitude']:.4f}")
    
    # Find optimal phase matching
    print("   Finding optimal phase relationships...")
    phase_performance, phase_params = find_optimal_electrical_parameters(
        system, 'phase_matching')
    
    print(f"      Best phase matching: performance = {phase_performance:.6f}")
    if phase_params:
        print(f"      Optimal frequency: {phase_params['frequency']:.3f} GHz")
        print(f"      Optimal phase: {phase_params.get('phase', 0):.3f} rad")
    
    return impedance_performance, impedance_params, phase_performance, phase_params

def compare_tuned_vs_standard():
    """
    Compare optimally tuned electrical-quantum coupling vs standard methods
    """
    print("\n2. Comparing tuned coupling vs standard methods...")
    
    system = TunedCouplingSystem()
    initial_state = [0.02, 0.0, 1.0, 0.0, 0.0, 0.0]
    sim_time = 60.0
    
    # Get optimal parameters
    _, impedance_params, _, phase_params = test_resonant_coupling_hypothesis()
    
    methods_to_test = {
        'standard_continuous': {
            'type': 'continuous',
            'amplitude': 0.02,
            'frequency': 5.0
        },
        'standard_pulsed': {
            'type': 'pulsed',
            'amplitude': 0.06,
            'period': 4.0,
            'width': 0.4
        }
    }
    
    # Add tuned methods if optimization succeeded
    if impedance_params:
        methods_to_test['impedance_matched'] = impedance_params
    
    if phase_params:
        methods_to_test['phase_matched'] = phase_params
    
    # Test additional tuned approaches
    methods_to_test['frequency_swept'] = {
        'type': 'chirped_resonant',
        'amplitude': 0.015,
        'frequency': 5.0,
        'chirp_rate': 0.2
    }
    
    results = {}
    
    for method_name, params in methods_to_test.items():
        print(f"   Testing {method_name}...")
        
        try:
            def dynamics(t, y):
                return tuned_system_dynamics(t, y, system, params)
            
            sol = solve_ivp(dynamics, [0, sim_time], initial_state,
                           t_eval=np.linspace(0, sim_time, 300),
                           method='RK45', rtol=1e-7)
            
            if sol.success:
                # Extract metrics
                coherence = np.sqrt(sol.y[4, :]**2 + sol.y[5, :]**2)
                excited_pop = sol.y[3, :]
                electrical_energy = sol.y[0, :]**2 + sol.y[1, :]**2
                
                # Calculate performance metrics
                steady_start = len(coherence) // 3
                
                metrics = {
                    'avg_coherence': np.mean(coherence[steady_start:]),
                    'max_coherence': np.max(coherence),
                    'coherence_stability': np.std(coherence[steady_start:]),
                    'final_coherence': coherence[-1],
                    'avg_excitation': np.mean(excited_pop[steady_start:]),
                    'electrical_efficiency': np.mean(coherence[steady_start:]) / np.mean(electrical_energy[steady_start:])
                }
                
                results[method_name] = {
                    'success': True,
                    'metrics': metrics,
                    'data': {
                        'time': sol.t,
                        'coherence': coherence,
                        'excitation': excited_pop,
                        'voltage': sol.y[0, :]
                    }
                }
                
                print(f"      Avg coherence: {metrics['avg_coherence']:.6f}")
                print(f"      Efficiency: {metrics['electrical_efficiency']:.6f}")
                
            else:
                print(f"      Integration failed: {sol.message}")
                results[method_name] = {'success': False}
                
        except Exception as e:
            print(f"      Crashed: {e}")
            results[method_name] = {'success': False}
    
    return results

def impedance_frequency_sweep():
    """
    Sweep frequency to find impedance matching resonances
    """
    print("\n3. Impedance matching frequency sweep...")
    
    system = TunedCouplingSystem()
    frequencies = np.linspace(4.0, 6.0, 40)  # 4-6 GHz sweep
    
    impedance_data = []
    coherence_data = []
    
    initial_state = [0.02, 0.0, 1.0, 0.0, 0.0, 0.0]
    
    for i, freq in enumerate(frequencies):
        print(f"   Frequency {i+1}/40: {freq:.3f} GHz")
        
        # Calculate theoretical impedance matching
        Z_elec = system.electrical_impedance(freq)
        Z_quantum = system.quantum_impedance(freq)
        matching_factor = system.impedance_matching_factor(freq)
        
        impedance_data.append({
            'frequency': freq,
            'Z_electrical_real': Z_elec.real,
            'Z_electrical_imag': Z_elec.imag,
            'Z_quantum_real': Z_quantum.real,
            'Z_quantum_imag': Z_quantum.imag,
            'matching_factor': matching_factor
        })
        
        # Test actual coherence at this frequency
        electrical_params = {
            'type': 'resonant_continuous',
            'amplitude': 0.02,
            'frequency': freq
        }
        
        try:
            def dynamics(t, y):
                return tuned_system_dynamics(t, y, system, electrical_params)
            
            sol = solve_ivp(dynamics, [0, 40.0], initial_state,
                           t_eval=np.linspace(0, 40.0, 200),
                           method='RK45', rtol=1e-6)
            
            if sol.success:
                coherence = np.sqrt(sol.y[4, :]**2 + sol.y[5, :]**2)
                avg_coherence = np.mean(coherence[50:])
                coherence_data.append(avg_coherence)
            else:
                coherence_data.append(0.0)
                
        except:
            coherence_data.append(0.0)
    
    return impedance_data, coherence_data

def run_tuned_coupling_analysis():
    """
    Complete tuned coupling analysis
    """
    start_time = time.time()
    
    try:
        # Test resonant coupling optimization
        impedance_perf, impedance_params, phase_perf, phase_params = test_resonant_coupling_hypothesis()
        
        # Compare all methods
        comparison_results = compare_tuned_vs_standard()
        
        # Frequency sweep for impedance matching
        impedance_sweep, coherence_sweep = impedance_frequency_sweep()
        
        # Find best performing method
        best_method = None
        best_coherence = 0.0
        
        for method, result in comparison_results.items():
            if result['success']:
                coherence = result['metrics']['avg_coherence']
                if coherence > best_coherence:
                    best_coherence = coherence
                    best_method = method
        
        # Analysis and reporting
        print("\n4. Generating tuned coupling analysis...")
        
        with open('tuned_coupling_results.txt', 'w') as f:
            f.write("TUNED ELECTRICAL-QUANTUM COUPLING RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("OPTIMIZATION RESULTS:\n")
            f.write(f"Best Impedance Matching Performance: {impedance_perf:.6f}\n")
            if impedance_params:
                f.write(f"  Optimal Frequency: {impedance_params['frequency']:.3f} GHz\n")
                f.write(f"  Optimal Amplitude: {impedance_params['amplitude']:.4f}\n")
            
            f.write(f"Best Phase Matching Performance: {phase_perf:.6f}\n")
            if phase_params:
                f.write(f"  Optimal Frequency: {phase_params['frequency']:.3f} GHz\n")
                f.write(f"  Optimal Phase: {phase_params.get('phase', 0):.3f} rad\n")
            f.write("\n")
            
            f.write("METHOD COMPARISON:\n")
            for method, result in comparison_results.items():
                if result['success']:
                    metrics = result['metrics']
                    f.write(f"{method.upper()}:\n")
                    f.write(f"  Average Coherence: {metrics['avg_coherence']:.6f}\n")
                    f.write(f"  Maximum Coherence: {metrics['max_coherence']:.6f}\n")
                    f.write(f"  Electrical Efficiency: {metrics['electrical_efficiency']:.6f}\n")
                    f.write(f"  Coherence Stability: {metrics['coherence_stability']:.6f}\n\n")
                else:
                    f.write(f"{method.upper()}: FAILED\n\n")
            
            f.write(f"BEST PERFORMING METHOD: {best_method}\n")
            f.write(f"Best Average Coherence: {best_coherence:.6f}\n\n")
            
            # Frequency sweep analysis
            max_matching_idx = np.argmax([data['matching_factor'] for data in impedance_sweep])
            max_coherence_idx = np.argmax(coherence_sweep)
            
            best_matching_freq = impedance_sweep[max_matching_idx]['frequency']
            best_coherence_freq = frequencies[max_coherence_idx] if max_coherence_idx < len(frequencies) else 5.0
            
            f.write(f"FREQUENCY ANALYSIS:\n")
            f.write(f"Best Impedance Matching at: {best_matching_freq:.3f} GHz\n")
            f.write(f"Best Coherence at: {best_coherence_freq:.3f} GHz\n")
            f.write(f"Frequency difference: {abs(best_matching_freq - best_coherence_freq):.3f} GHz\n\n")
            
            # Conclusions
            if best_method and 'tuned' in best_method or 'matched' in best_method or 'resonant' in best_method:
                f.write("CONCLUSION: TUNED ELECTRICAL-QUANTUM COUPLING SUPERIOR!\n")
                f.write("Precisely engineered electrical-quantum resonance\n")
                f.write("outperforms both standard continuous and pulsed control.\n\n")
                f.write("This supports the hypothesis that electrical-quantum\n")
                f.write("coupling requires sophisticated tuning, not just\n")
                f.write("continuous vs pulsed distinction.\n")
            else:
                f.write("CONCLUSION: Standard methods remain superior\n")
                f.write("Tuned coupling approaches do not show advantages\n")
        
        # Save frequency sweep data
        with open('impedance_sweep_data.csv', 'w') as f:
            f.write("frequency_GHz,Z_elec_real,Z_elec_imag,Z_quantum_real,Z_quantum_imag,matching_factor,measured_coherence\n")
            
            for i, data in enumerate(impedance_sweep):
                coh = coherence_sweep[i] if i < len(coherence_sweep) else 0.0
                f.write(f"{data['frequency']:.6f},{data['Z_electrical_real']:.6f},")
                f.write(f"{data['Z_electrical_imag']:.6f},{data['Z_quantum_real']:.6f},")
                f.write(f"{data['Z_quantum_imag']:.6f},{data['matching_factor']:.6f},{coh:.6f}\n")
        
        elapsed = time.time() - start_time
        print(f"\nTUNED COUPLING ANALYSIS COMPLETED in {elapsed:.1f} seconds")
        print("Files created:")
        print("- tuned_coupling_results.txt")
        print("- impedance_sweep_data.csv")
        
        if best_method:
            print(f"\n🎯 BEST METHOD: {best_method}")
            print(f"🎯 BEST COHERENCE: {best_coherence:.6f}")
            
            if 'matched' in best_method or 'resonant' in best_method:
                print("🚀 TUNED COUPLING APPROACH WINS!")
                print("Your insight about electrical-quantum resonance confirmed!")
            else:
                print("📊 Standard methods still superior")
        
        return comparison_results, impedance_sweep, coherence_sweep
        
    except Exception as e:
        print(f"Tuned coupling analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing sophisticated electrical-quantum coupling:")
    print("- Impedance matching between electrical and quantum systems")
    print("- Phase-coherent resonant coupling")
    print("- Frequency optimization for maximum coherence")
    print("- Chirped frequency sweeps through resonance")
    print("\nHypothesis: Electrical-quantum coupling requires precise")
    print("tuning like impedance matching in electrical engineering")
    print("\nRuntime: ~15 minutes")
    print("=" * 70)
    
    try:
        results = run_tuned_coupling_analysis()
        print("\n" + "="*70)
        print("TUNED COUPLING TEST COMPLETED!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()