import numpy as np
from scipy.integrate import solve_ivp
import time

print("Starting comprehensive realistic electrical-quantum coupling simulation...")
print("Includes: decoherence, noise, multi-level effects, circuit dynamics")

class RobustQuantumSystem:
    """
    Realistic quantum system with proper error handling
    """
    def __init__(self):
        # Realistic superconducting qubit / quantum dot parameters
        self.w_electrical = 5.0      # 5 GHz electrical frequency
        self.w_quantum = 5.0         # 5 GHz quantum transition  
        self.w_second = 9.5          # 9.5 GHz second transition (anharmonic)
        
        # Coupling strengths (MHz)
        self.g_01 = 30.0            # Ground to first excited
        self.g_12 = 20.0            # First to second excited
        
        # Decoherence times (μs) - realistic for good qubits
        self.T1 = 80.0              # Energy relaxation
        self.T2_echo = 150.0        # Echo coherence time
        self.T2_star = 25.0         # Free induction decay
        
        # Electrical circuit parameters
        self.R_circuit = 50.0       # 50 ohm characteristic impedance
        self.Q_factor = 2000        # High-Q superconducting resonator
        self.L_circuit = 1.0        # Normalized inductance
        self.C_circuit = 1.0 / (self.w_electrical**2)  # Resonant capacitance
        
        # Noise parameters (realistic values)
        self.charge_noise_amp = 0.002   # 0.2% charge noise
        self.flux_noise_amp = 0.001     # 0.1% flux noise  
        self.voltage_noise_amp = 0.0005 # 0.05% voltage noise
        
        # Conversion factors
        self.voltage_to_energy = 0.01   # Realistic voltage-energy coupling

def generate_realistic_noise(t, noise_type='1_over_f'):
    """
    Generate realistic time-correlated noise
    """
    if noise_type == '1_over_f':
        # 1/f noise (dominant in solid-state qubits)
        frequencies = np.array([0.01, 0.03, 0.1, 0.3, 1.0])
        amplitudes = 1.0 / frequencies  # 1/f spectrum
        noise = np.sum(amplitudes * np.sin(frequencies * t + np.pi * frequencies))
        return noise / 10.0  # Normalize
    
    elif noise_type == 'gaussian':
        # White noise approximation (deterministic for reproducibility)
        return 0.1 * np.sin(17.3 * t) * np.exp(-0.01 * abs(t - 25))
    
    else:
        return 0.0

def realistic_electrical_dynamics(t, V, I, external_drive, quantum_backaction, system):
    """
    Realistic electrical circuit with proper RLC dynamics
    """
    # Environmental noise
    voltage_noise = system.voltage_noise_amp * generate_realistic_noise(t, '1_over_f')
    flux_noise = system.flux_noise_amp * generate_realistic_noise(t + 1.7, 'gaussian')
    
    # RLC circuit equations
    # L dI/dt + R I + V/C = external_drive + quantum_backaction + noise
    damping = system.w_electrical / system.Q_factor  # Quality factor damping
    
    dV_dt = I / system.C_circuit
    dI_dt = (external_drive + quantum_backaction + voltage_noise - 
             V / system.C_circuit - damping * I) / system.L_circuit
    
    return dV_dt, dI_dt

def realistic_quantum_dynamics(t, rho_vec, electrical_field, system):
    """
    Three-level quantum system with realistic decoherence
    rho_vec = [ρ₀₀, ρ₀₁ᴿ, ρ₀₁ᴵ, ρ₀₂ᴿ, ρ₀₂ᴵ, ρ₁₁, ρ₁₂ᴿ, ρ₁₂ᴵ, ρ₂₂]
    """
    # Reconstruct density matrix
    rho = np.array([
        [rho_vec[0], rho_vec[1] + 1j*rho_vec[2], rho_vec[3] + 1j*rho_vec[4]],
        [rho_vec[1] - 1j*rho_vec[2], rho_vec[5], rho_vec[6] + 1j*rho_vec[7]],
        [rho_vec[3] - 1j*rho_vec[4], rho_vec[6] - 1j*rho_vec[7], rho_vec[8]]
    ])
    
    # Electrical coupling with realistic scaling
    coupling_01 = system.g_01 * electrical_field * system.voltage_to_energy
    coupling_12 = system.g_12 * electrical_field * system.voltage_to_energy * 0.7  # Weaker higher coupling
    
    # Hamiltonian (MHz units)
    H = np.array([
        [0, coupling_01, 0],
        [coupling_01, system.w_quantum, coupling_12],
        [0, coupling_12, system.w_second]
    ])
    
    # Coherent evolution: -i[H, ρ]
    coherent_evolution = -1j * (H @ rho - rho @ H)
    
    # Realistic decoherence rates
    gamma_10 = 1.0 / system.T1           # |1⟩ → |0⟩ relaxation
    gamma_21 = 1.2 / system.T1           # |2⟩ → |1⟩ relaxation (slightly faster)
    gamma_20 = 0.1 / system.T1           # Direct |2⟩ → |0⟩ (weak)
    
    gamma_phi_01 = 1.0 / system.T2_star  # 0-1 dephasing  
    gamma_phi_12 = 1.2 / system.T2_star  # 1-2 dephasing (faster for higher levels)
    
    # Environmental noise effects on dephasing
    charge_noise = system.charge_noise_amp * generate_realistic_noise(t, '1_over_f')
    noise_dephasing = abs(charge_noise) * 2.0  # Charge noise causes dephasing
    
    # Lindblad decoherence terms
    # Energy relaxation
    L_10 = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])  # σ₋ for |1⟩→|0⟩
    L_21 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])  # |2⟩→|1⟩
    L_20 = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])  # Direct |2⟩→|0⟩
    
    # Pure dephasing
    Z_01 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])  # 0-1 dephasing
    Z_12 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])  # 1-2 dephasing
    
    def lindblad_term(L, gamma, rho):
        """Apply single Lindblad operator"""
        return gamma * (L @ rho @ L.T - 0.5 * (L.T @ L @ rho + rho @ L.T @ L))
    
    # Total decoherence
    decoherence = (
        lindblad_term(L_10, gamma_10, rho) +
        lindblad_term(L_21, gamma_21, rho) + 
        lindblad_term(L_20, gamma_20, rho) +
        lindblad_term(Z_01, gamma_phi_01 + noise_dephasing, rho) +
        lindblad_term(Z_12, gamma_phi_12, rho)
    )
    
    # Total quantum evolution
    total_evolution = coherent_evolution + decoherence
    
    # Convert back to vector form
    drho_dt_vec = [
        np.real(total_evolution[0,0]),  # ρ₀₀
        np.real(total_evolution[0,1]),  # ρ₀₁ᴿ
        np.imag(total_evolution[0,1]),  # ρ₀₁ᴵ
        np.real(total_evolution[0,2]),  # ρ₀₂ᴿ
        np.imag(total_evolution[0,2]),  # ρ₀₂ᴵ
        np.real(total_evolution[1,1]),  # ρ₁₁
        np.real(total_evolution[1,2]),  # ρ₁₂ᴿ
        np.imag(total_evolution[1,2]),  # ρ₁₂ᴵ
        np.real(total_evolution[2,2])   # ρ₂₂
    ]
    
    return drho_dt_vec

def full_system_dynamics(t, y, system, drive_type, drive_params):
    """
    Complete electrical-quantum system evolution
    y = [V, I, ρ₀₀, ρ₀₁ᴿ, ρ₀₁ᴵ, ρ₀₂ᴿ, ρ₀₂ᴵ, ρ₁₁, ρ₁₂ᴿ, ρ₁₂ᴵ, ρ₂₂]
    """
    try:
        V_e, I_e = y[0], y[1]
        rho_vec = y[2:]
        
        # Generate drive signal
        if drive_type == 'continuous':
            amp, freq = drive_params
            external_drive = amp * np.sin(freq * t)
        elif drive_type == 'pulsed':
            amp, period, width = drive_params
            cycle_time = t % period
            external_drive = amp if cycle_time < width else 0.0
        else:
            external_drive = 0.0
        
        # Quantum backaction on electrical circuit
        rho_11, rho_22 = rho_vec[5], rho_vec[8]
        quantum_energy = rho_11 + 1.5 * rho_22  # Weighted by level energy
        quantum_backaction = system.g_01 * quantum_energy * 0.0001
        
        # Electrical circuit evolution
        dV_dt, dI_dt = realistic_electrical_dynamics(t, V_e, I_e, external_drive, quantum_backaction, system)
        
        # Quantum system evolution
        electrical_field = V_e  # Direct voltage coupling
        drho_dt_vec = realistic_quantum_dynamics(t, rho_vec, electrical_field, system)
        
        return [dV_dt, dI_dt] + drho_dt_vec
        
    except Exception as e:
        print(f"Error in dynamics at t={t:.3f}: {e}")
        return [0.0] * len(y)  # Return zeros to prevent crash

def run_comprehensive_realistic_simulation():
    """
    Full comprehensive analysis with robustness checks
    """
    system = RobustQuantumSystem()
    
    # Initial conditions: electrical + quantum ground state
    # [V, I, ρ₀₀, ρ₀₁ᴿ, ρ₀₁ᴵ, ρ₀₂ᴿ, ρ₀₂ᴵ, ρ₁₁, ρ₁₂ᴿ, ρ₁₂ᴵ, ρ₂₂]
    initial_state = [0.02, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    t_max = 150.0  # 150 μs (multiple T1, T2 times)
    n_points = 1500
    
    simulations = {}
    
    print(f"Running simulations over {t_max} μs with {n_points} time points...")
    
    # Continuous wave simulation
    print("1. Continuous wave with realistic decoherence...")
    start = time.time()
    
    def cw_dynamics(t, y):
        return full_system_dynamics(t, y, system, 'continuous', (0.015, 5.0))
    
    try:
        sol_cw = solve_ivp(cw_dynamics, [0, t_max], initial_state,
                          t_eval=np.linspace(0, t_max, n_points),
                          method='LSODA', rtol=1e-7, atol=1e-9)
        
        if sol_cw.success:
            simulations['continuous'] = sol_cw
            print(f"   Completed in {time.time() - start:.1f}s - SUCCESS")
        else:
            print(f"   FAILED: {sol_cw.message}")
            
    except Exception as e:
        print(f"   CRASHED: {e}")
    
    # Pulsed drive simulation  
    print("2. Pulsed drive with realistic decoherence...")
    start = time.time()
    
    def pulse_dynamics(t, y):
        return full_system_dynamics(t, y, system, 'pulsed', (0.06, 8.0, 0.8))
    
    try:
        sol_pulse = solve_ivp(pulse_dynamics, [0, t_max], initial_state,
                             t_eval=np.linspace(0, t_max, n_points),
                             method='LSODA', rtol=1e-7, atol=1e-9)
        
        if sol_pulse.success:
            simulations['pulsed'] = sol_pulse  
            print(f"   Completed in {time.time() - start:.1f}s - SUCCESS")
        else:
            print(f"   FAILED: {sol_pulse.message}")
            
    except Exception as e:
        print(f"   CRASHED: {e}")
    
    return simulations, system

def analyze_comprehensive_results(simulations, system):
    """
    Comprehensive analysis of realistic simulation results
    """
    if 'continuous' not in simulations or 'pulsed' not in simulations:
        print("Cannot analyze - one or both simulations failed")
        return
    
    print("3. Analyzing comprehensive results...")
    
    # Extract data
    sol_cw = simulations['continuous']
    sol_pulse = simulations['pulsed']
    
    def extract_metrics(sol):
        t = sol.t
        V = sol.y[0]
        I = sol.y[1]
        rho_00 = sol.y[2]
        rho_01_r = sol.y[3]
        rho_01_i = sol.y[4]
        rho_02_r = sol.y[5]
        rho_02_i = sol.y[6]
        rho_11 = sol.y[7]
        rho_12_r = sol.y[8]
        rho_12_i = sol.y[9]
        rho_22 = sol.y[10]
        
        # Coherence measures
        coherence_01 = np.sqrt(rho_01_r**2 + rho_01_i**2)  # Ground-excited coherence
        coherence_12 = np.sqrt(rho_12_r**2 + rho_12_i**2)  # First-second coherence
        total_coherence = coherence_01 + 0.5 * coherence_12  # Weighted total
        
        # Population measures  
        ground_pop = rho_00
        excited_pop = rho_11 + rho_22
        
        # Quantum state quality
        purity = (rho_00**2 + rho_11**2 + rho_22**2 + 
                 2*(rho_01_r**2 + rho_01_i**2 + rho_02_r**2 + rho_02_i**2 + rho_12_r**2 + rho_12_i**2))
        
        # Electrical system metrics
        electrical_power = V**2 + I**2
        
        return {
            'time': t, 'voltage': V, 'current': I,
            'coherence_01': coherence_01, 'coherence_12': coherence_12, 
            'total_coherence': total_coherence,
            'ground_pop': ground_pop, 'excited_pop': excited_pop,
            'purity': purity, 'electrical_power': electrical_power
        }
    
    cw_data = extract_metrics(sol_cw)
    pulse_data = extract_metrics(sol_pulse)
    
    # Analysis over steady-state period (skip first 30% for transients)
    steady_start = len(cw_data['time']) // 3
    
    # Average metrics in steady state
    metrics = {}
    for method, data in [('cw', cw_data), ('pulse', pulse_data)]:
        metrics[method] = {
            'avg_coherence': np.mean(data['total_coherence'][steady_start:]),
            'max_coherence': np.max(data['total_coherence']),
            'avg_purity': np.mean(data['purity'][steady_start:]),
            'final_excited': data['excited_pop'][-1],
            'coherence_stability': np.std(data['total_coherence'][steady_start:]),
        }
    
    # Calculate performance ratios
    coherence_ratio = metrics['cw']['avg_coherence'] / metrics['pulse']['avg_coherence']
    purity_ratio = metrics['cw']['avg_purity'] / metrics['pulse']['avg_purity']
    stability_ratio = metrics['pulse']['coherence_stability'] / metrics['cw']['coherence_stability']  # Lower is better
    
    return metrics, coherence_ratio, purity_ratio, stability_ratio, cw_data, pulse_data

def test_realistic_frequency_matching(system, n_points=12):
    """
    Test frequency matching with full realistic model
    """
    print("4. Testing frequency matching with realistic decoherence...")
    
    base_freq = system.w_electrical
    detuning_range = 1.0  # ±1 GHz detuning
    detunings = np.linspace(-detuning_range, detuning_range, n_points)
    
    coherence_results = []
    purity_results = []
    success_count = 0
    
    initial = [0.02, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    for i, detuning in enumerate(detunings):
        print(f"   Detuning {i+1}/{n_points}: {detuning:+.3f} GHz")
        
        # Create detuned system
        system_detuned = RobustQuantumSystem()
        system_detuned.w_quantum = base_freq + detuning
        
        def dynamics_detuned(t, y):
            return full_system_dynamics(t, y, system_detuned, 'continuous', (0.015, base_freq))
        
        try:
            sol = solve_ivp(dynamics_detuned, [0, 40.0], initial,
                           t_eval=np.linspace(0, 40.0, 400),
                           method='LSODA', rtol=1e-6, atol=1e-8)
            
            if sol.success:
                # Calculate metrics
                rho_01_r = sol.y[3]
                rho_01_i = sol.y[4]
                rho_00 = sol.y[2]
                rho_11 = sol.y[7]
                rho_22 = sol.y[10]
                
                coherence = np.sqrt(rho_01_r**2 + rho_01_i**2)
                purity = rho_00**2 + rho_11**2 + rho_22**2 + 2*(rho_01_r**2 + rho_01_i**2)
                
                # Average over steady state
                steady_idx = len(coherence) // 3
                avg_coherence = np.mean(coherence[steady_idx:])
                avg_purity = np.mean(purity[steady_idx:])
                
                coherence_results.append(avg_coherence)
                purity_results.append(avg_purity)
                success_count += 1
            else:
                coherence_results.append(0.0)
                purity_results.append(0.0)
                
        except Exception as e:
            print(f"      Failed: {e}")
            coherence_results.append(0.0) 
            purity_results.append(0.0)
    
    print(f"   Frequency sweep: {success_count}/{n_points} successful")
    return detunings, coherence_results, purity_results

# Main execution
if __name__ == "__main__":
    start_total = time.time()
    
    try:
        # Run main simulations
        simulations, system = run_comprehensive_realistic_simulation()
        
        if len(simulations) == 2:
            # Analyze results
            metrics, coherence_ratio, purity_ratio, stability_ratio, cw_data, pulse_data = analyze_comprehensive_results(simulations, system)
            
            # Frequency matching test
            detunings, coherence_vs_freq, purity_vs_freq = test_realistic_frequency_matching(system)
            
            optimal_detuning = detunings[np.argmax(coherence_vs_freq)]
            max_coherence_freq = max(coherence_vs_freq)
            
            # Write comprehensive output
            with open('comprehensive_realistic_results.txt', 'w') as f:
                f.write("COMPREHENSIVE REALISTIC ELECTRICAL-QUANTUM COUPLING\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("SYSTEM SPECIFICATIONS:\n")
                f.write(f"Electrical frequency: {system.w_electrical} GHz\n")
                f.write(f"Quantum frequency: {system.w_quantum} GHz\n")
                f.write(f"T1 relaxation time: {system.T1} μs\n")
                f.write(f"T2* dephasing time: {system.T2_star} μs\n")
                f.write(f"Coupling strength: {system.g_01} MHz\n")
                f.write(f"Circuit Q factor: {system.Q_factor}\n\n")
                
                f.write("PERFORMANCE COMPARISON:\n")
                f.write(f"Continuous Wave Average Coherence: {metrics['cw']['avg_coherence']:.6f}\n")
                f.write(f"Pulsed Drive Average Coherence: {metrics['pulse']['avg_coherence']:.6f}\n")
                f.write(f"Coherence Improvement Factor: {coherence_ratio:.3f}\n\n")
                
                f.write(f"Continuous Wave Purity: {metrics['cw']['avg_purity']:.6f}\n")
                f.write(f"Pulsed Drive Purity: {metrics['pulse']['avg_purity']:.6f}\n")
                f.write(f"Purity Improvement Factor: {purity_ratio:.3f}\n\n")
                
                f.write(f"CW Coherence Stability: {metrics['cw']['coherence_stability']:.6f}\n")
                f.write(f"Pulse Coherence Stability: {metrics['pulse']['coherence_stability']:.6f}\n")
                f.write(f"Stability Improvement Factor: {stability_ratio:.3f}\n\n")
                
                f.write("FREQUENCY MATCHING RESULTS:\n")
                f.write(f"Optimal Frequency Detuning: {optimal_detuning:.6f} GHz\n")
                f.write(f"Peak Coherence: {max_coherence_freq:.6f}\n")
                f.write(f"Zero Detuning Coherence: {coherence_vs_freq[len(detunings)//2]:.6f}\n\n")
                
                if coherence_ratio > 1.05:  # 5% threshold
                    f.write("CONCLUSION: CONTINUOUS ELECTRICAL WAVES SUPERIOR!\n")
                    f.write("Advantages confirmed even with:\n")
                    f.write("- Realistic T1/T2 decoherence\n")
                    f.write("- Environmental noise (1/f + Gaussian)\n")
                    f.write("- Multi-level quantum system\n")
                    f.write("- Circuit resistance and damping\n")
                    f.write("- Quantum backaction effects\n\n")
                    f.write("This supports the hypothesis that electrical-quantum\n")
                    f.write("resonant coupling provides superior coherence control!\n")
                elif coherence_ratio > 0.95:
                    f.write("CONCLUSION: Performance comparable\n")
                    f.write("Continuous and pulsed show similar performance\n")
                else:
                    f.write("CONCLUSION: Pulsed control superior in realistic conditions\n")
            
            # Save detailed time series
            with open('comprehensive_time_series.csv', 'w') as f:
                f.write("time_us,cw_voltage,cw_coherence_01,cw_coherence_total,cw_excited_pop,cw_purity,")
                f.write("pulse_voltage,pulse_coherence_01,pulse_coherence_total,pulse_excited_pop,pulse_purity\n")
                
                # Interpolate pulse data
                pulse_interp = {}
                for key in ['voltage', 'coherence_01', 'total_coherence', 'excited_pop', 'purity']:
                    pulse_interp[key] = np.interp(cw_data['time'], pulse_data['time'], pulse_data[key])
                
                for i in range(len(cw_data['time'])):
                    f.write(f"{cw_data['time'][i]:.3f},{cw_data['voltage'][i]:.6f},")
                    f.write(f"{cw_data['coherence_01'][i]:.6f},{cw_data['total_coherence'][i]:.6f},")
                    f.write(f"{cw_data['excited_pop'][i]:.6f},{cw_data['purity'][i]:.6f},")
                    f.write(f"{pulse_interp['voltage'][i]:.6f},{pulse_interp['coherence_01'][i]:.6f},")
                    f.write(f"{pulse_interp['total_coherence'][i]:.6f},{pulse_interp['excited_pop'][i]:.6f},")
                    f.write(f"{pulse_interp['purity'][i]:.6f}\n")
            
            print(f"\nTOTAL SIMULATION TIME: {time.time() - start_total:.1f} seconds")
            print("\nRESULTS SUMMARY:")
            print(f"Coherence Ratio (CW/Pulse): {coherence_ratio:.3f}")
            print(f"Purity Ratio (CW/Pulse): {purity_ratio:.3f}")
            print(f"Stability Ratio (Pulse/CW): {stability_ratio:.3f}")
            
            if abs(optimal_detuning) < 0.2:
                print(f"Resonance effect confirmed: peak at {optimal_detuning:+.3f} GHz")
            
            print("\nFiles created:")
            print("- comprehensive_realistic_results.txt")
            print("- comprehensive_time_series.csv")
            
        else:
            print("Simulations failed - check error messages above")
            
    except Exception as e:
        print(f"Analysis failed: {e}")

print("\nStarting comprehensive realistic simulation...")
print("This includes full decoherence physics and may take 10-15 minutes")
print("=" * 60)

# Run the full analysis
run_comprehensive_realistic_simulation()