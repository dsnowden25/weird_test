import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

print("SIMPLE SANITY CHECK TEST")
print("Dead simple implementation to check if effect is real")
print("=" * 60)

def simple_coherence_evolution(t, state, T1, T2, drive_amplitude, drive_frequency):
    """
    Dead simple coherence evolution - no fancy Lindblad, just basic physics
    state = [population_0, population_1, coherence_real, coherence_imag]
    """
    p0, p1, coh_r, coh_i = state
    
    # Drive coupling (simple resonant approximation)
    drive = drive_amplitude * np.sin(drive_frequency * t)
    
    # Population dynamics with T1 relaxation
    dp0_dt = (p1 / T1) - drive * coh_i  # Gain from relaxation, lose to coherent excitation
    dp1_dt = -(p1 / T1) + drive * coh_i  # Lose to relaxation, gain from coherent excitation
    
    # Coherence dynamics with T2 dephasing
    dcoh_r_dt = -coh_r / T2 + drive * (p0 - p1)  # Driven coherence, T2 decay
    dcoh_i_dt = -coh_i / T2  # Pure dephasing
    
    return [dp0_dt, dp1_dt, dcoh_r_dt, dcoh_i_dt]

def test_textbook_decay():
    """
    First: Validate that we can reproduce textbook T1/T2 decay
    """
    print("1. Testing textbook T1/T2 decay (no drive)...")
    
    # Standard parameters
    T1, T2 = 50.0, 25.0  # μs
    
    # Test pure T1 decay: start in excited state
    initial_excited = [0.0, 1.0, 0.0, 0.0]  # Pure |1⟩
    
    def no_drive_dynamics(t, state):
        return simple_coherence_evolution(t, state, T1, T2, 0.0, 5.0)
    
    sol_t1 = solve_ivp(no_drive_dynamics, [0, 200], initial_excited,
                       t_eval=np.linspace(0, 200, 200),
                       method='RK45', rtol=1e-8)
    
    if sol_t1.success:
        # Check T1 decay
        p1_evolution = sol_t1.y[1, :]
        expected_t1 = np.exp(-sol_t1.t / T1)
        
        t1_error = np.mean(np.abs(p1_evolution - expected_t1))
        print(f"   T1 test: average error = {t1_error:.6f}")
        
        if t1_error < 0.01:
            print("   ✓ T1 decay matches textbook!")
        else:
            print("   ✗ T1 decay WRONG - simulation broken")
            return False
    else:
        print("   ✗ T1 simulation failed")
        return False
    
    # Test pure T2 decay: start in superposition
    initial_superpos = [0.5, 0.5, 0.5, 0.0]  # (|0⟩ + |1⟩)/√2
    
    sol_t2 = solve_ivp(no_drive_dynamics, [0, 100], initial_superpos,
                       t_eval=np.linspace(0, 100, 100),
                       method='RK45', rtol=1e-8)
    
    if sol_t2.success:
        # Check T2 decay
        coherence_evolution = np.sqrt(sol_t2.y[2,:]**2 + sol_t2.y[3,:]**2)
        expected_t2 = 0.5 * np.exp(-sol_t2.t / T2)  # Initial coherence was 0.5
        
        t2_error = np.mean(np.abs(coherence_evolution - expected_t2))
        print(f"   T2 test: average error = {t2_error:.6f}")
        
        if t2_error < 0.01:
            print("   ✓ T2 decay matches textbook!")
            return True
        else:
            print("   ✗ T2 decay WRONG - simulation broken")
            return False
    else:
        print("   ✗ T2 simulation failed")
        return False

def simple_continuous_vs_pulsed_test(T1, T2, sim_time=100.0):
    """
    Simple test: does continuous vs pulsed driving make any difference?
    """
    print(f"2. Simple CW vs Pulsed test (T1={T1}μs, T2={T2}μs, time={sim_time}μs)...")
    
    # Start in ground state
    initial_ground = [1.0, 0.0, 0.0, 0.0]
    
    results = {}
    
    # Test continuous wave
    print("   Testing continuous wave...")
    
    def cw_dynamics(t, state):
        return simple_coherence_evolution(t, state, T1, T2, 0.01, 5.0)
    
    sol_cw = solve_ivp(cw_dynamics, [0, sim_time], initial_ground,
                       t_eval=np.linspace(0, sim_time, int(sim_time*2)),
                       method='RK45', rtol=1e-8)
    
    if sol_cw.success:
        # Calculate metrics
        coherence_cw = np.sqrt(sol_cw.y[2, :]**2 + sol_cw.y[3, :]**2)
        excited_pop_cw = sol_cw.y[1, :]
        
        # Average over steady state (skip first 25%)
        steady_start = len(coherence_cw) // 4
        
        results['continuous'] = {
            'final_coherence': coherence_cw[-1],
            'avg_coherence': np.mean(coherence_cw[steady_start:]),
            'max_coherence': np.max(coherence_cw),
            'avg_excitation': np.mean(excited_pop_cw[steady_start:]),
            'time': sol_cw.t,
            'coherence': coherence_cw,
            'population': excited_pop_cw
        }
        
        print(f"      Final coherence: {coherence_cw[-1]:.6f}")
        print(f"      Average coherence: {results['continuous']['avg_coherence']:.6f}")
        print(f"      Max excitation: {np.max(excited_pop_cw):.6f}")
        
    else:
        print("   ✗ Continuous wave simulation failed")
        return None
    
    # Test pulsed drive
    print("   Testing pulsed drive...")
    
    def pulsed_dynamics(t, state):
        # Square pulses: 0.5μs on, 4.5μs off
        cycle_time = t % 5.0
        pulse_amp = 0.03 if cycle_time < 0.5 else 0.0
        return simple_coherence_evolution(t, state, T1, T2, pulse_amp, 5.0)
    
    sol_pulse = solve_ivp(pulsed_dynamics, [0, sim_time], initial_ground,
                          t_eval=np.linspace(0, sim_time, int(sim_time*2)),
                          method='RK45', rtol=1e-8)
    
    if sol_pulse.success:
        # Calculate metrics
        coherence_pulse = np.sqrt(sol_pulse.y[2, :]**2 + sol_pulse.y[3, :]**2)
        excited_pop_pulse = sol_pulse.y[1, :]
        
        steady_start = len(coherence_pulse) // 4
        
        results['pulsed'] = {
            'final_coherence': coherence_pulse[-1],
            'avg_coherence': np.mean(coherence_pulse[steady_start:]),
            'max_coherence': np.max(coherence_pulse),
            'avg_excitation': np.mean(excited_pop_pulse[steady_start:]),
            'time': sol_pulse.t,
            'coherence': coherence_pulse,
            'population': excited_pop_pulse
        }
        
        print(f"      Final coherence: {coherence_pulse[-1]:.6f}")
        print(f"      Average coherence: {results['pulsed']['avg_coherence']:.6f}")
        print(f"      Max excitation: {np.max(excited_pop_pulse):.6f}")
        
    else:
        print("   ✗ Pulsed drive simulation failed")
        return None
    
    return results

def test_multiple_scenarios():
    """
    Test across different T1/T2 values to see if effect is robust
    """
    print("3. Testing multiple T1/T2 scenarios...")
    
    scenarios = [
        {'name': 'excellent_qubit', 'T1': 200, 'T2': 100, 'sim_time': 150},
        {'name': 'good_qubit', 'T1': 80, 'T2': 40, 'sim_time': 100},
        {'name': 'mediocre_qubit', 'T1': 30, 'T2': 15, 'sim_time': 60},
        {'name': 'poor_qubit', 'T1': 10, 'T2': 5, 'sim_time': 30}
    ]
    
    all_results = {}
    
    for scenario in scenarios:
        print(f"\n   Testing {scenario['name']} (T1={scenario['T1']}μs, T2={scenario['T2']}μs)...")
        
        result = simple_continuous_vs_pulsed_test(scenario['T1'], scenario['T2'], scenario['sim_time'])
        
        if result:
            all_results[scenario['name']] = result
            
            # Calculate comparison
            cw_avg = result['continuous']['avg_coherence']
            pulse_avg = result['pulsed']['avg_coherence']
            
            if pulse_avg > 0:
                ratio = cw_avg / pulse_avg
                print(f"      CW/Pulsed ratio: {ratio:.3f}")
                
                # Reality check
                expected_coherence = np.exp(-scenario['sim_time'] / scenario['T2'])
                print(f"      Expected final coherence: {expected_coherence:.6f}")
                print(f"      CW actual: {result['continuous']['final_coherence']:.6f}")
                print(f"      Pulsed actual: {result['pulsed']['final_coherence']:.6f}")
                
                if (result['continuous']['final_coherence'] > expected_coherence * 3 or 
                    result['pulsed']['final_coherence'] > expected_coherence * 3):
                    print("      🚨 COHERENCES TOO HIGH - Physics still broken!")
                elif ratio > 1.5:
                    print("      ⚡ Significant CW advantage detected!")
                elif ratio > 1.1:
                    print("      📈 Modest CW advantage")
                else:
                    print("      📊 No significant advantage")
            else:
                print("      ✗ Pulsed coherence too low for comparison")
        else:
            print(f"      ✗ {scenario['name']} test failed")
    
    return all_results

def run_simple_sanity_check():
    """
    Complete simple sanity check
    """
    print("SIMPLE SANITY CHECK - No complex physics, just basic quantum mechanics")
    print("If this shows unrealistic results, we know there's a fundamental bug")
    print("If this shows realistic results with CW advantage, effect might be real!")
    
    start_time = time.time()
    
    # First: Validate textbook physics
    if not test_textbook_decay():
        print("\n💀 TEXTBOOK PHYSICS FAILED - Simulation fundamentally broken")
        return
    
    print("\n✓ Textbook physics working - proceeding to comparison test")
    
    # Second: Test multiple scenarios
    all_results = test_multiple_scenarios()
    
    # Analysis
    print("\n4. Overall Analysis...")
    
    if all_results:
        with open('simple_sanity_results.txt', 'w') as f:
            f.write("SIMPLE SANITY CHECK RESULTS\n")
            f.write("=" * 40 + "\n\n")
            
            ratios = []
            for scenario_name, result in all_results.items():
                cw_avg = result['continuous']['avg_coherence']
                pulse_avg = result['pulsed']['avg_coherence']
                
                f.write(f"{scenario_name.upper()}:\n")
                f.write(f"  CW Average Coherence: {cw_avg:.8f}\n")
                f.write(f"  Pulsed Average Coherence: {pulse_avg:.8f}\n")
                
                if pulse_avg > 0:
                    ratio = cw_avg / pulse_avg
                    ratios.append(ratio)
                    f.write(f"  Ratio (CW/Pulsed): {ratio:.3f}\n")
                f.write("\n")
            
            if ratios:
                overall_ratio = np.mean(ratios)
                ratio_std = np.std(ratios)
                
                f.write(f"OVERALL RESULTS:\n")
                f.write(f"Average CW/Pulsed Ratio: {overall_ratio:.3f} ± {ratio_std:.3f}\n")
                f.write(f"Consistent advantage: {sum(1 for r in ratios if r > 1.2)}/{len(ratios)} scenarios\n\n")
                
                if overall_ratio > 1.5 and ratio_std < 0.5:
                    f.write("🔥 SIMPLE TEST CONCLUSION: CONTINUOUS ADVANTAGE CONFIRMED!\n")
                    f.write("Effect appears robust across different T1/T2 values\n")
                    f.write("This suggests the effect is NOT just simulation artifacts\n")
                elif overall_ratio > 1.1:
                    f.write("📈 SIMPLE TEST CONCLUSION: Modest CW advantage detected\n")
                    f.write("Worth further investigation with more sophisticated models\n")
                else:
                    f.write("📊 SIMPLE TEST CONCLUSION: No significant advantage\n")
                    f.write("Continuous and pulsed perform similarly\n")
        
        # Save time series for plotting
        with open('simple_time_series.csv', 'w') as f:
            f.write("scenario,time,cw_coherence,pulse_coherence,cw_population,pulse_population\n")
            
            for scenario_name, result in all_results.items():
                cw_data = result['continuous']
                pulse_data = result['pulsed']
                
                # Interpolate to common time base
                common_time = cw_data['time']
                pulse_coherence_interp = np.interp(common_time, pulse_data['time'], pulse_data['coherence'])
                pulse_population_interp = np.interp(common_time, pulse_data['time'], pulse_data['population'])
                
                for i in range(len(common_time)):
                    f.write(f"{scenario_name},{common_time[i]:.3f},")
                    f.write(f"{cw_data['coherence'][i]:.8f},{pulse_coherence_interp[i]:.8f},")
                    f.write(f"{cw_data['population'][i]:.8f},{pulse_population_interp[i]:.8f}\n")
        
        elapsed = time.time() - start_time
        print(f"\nSimple sanity check completed in {elapsed:.1f} seconds")
        print("Results saved to:")
        print("- simple_sanity_results.txt")
        print("- simple_time_series.csv")
        
        # Quick summary
        if ratios:
            print(f"\nQUICK SUMMARY:")
            print(f"Overall CW advantage: {np.mean(ratios):.3f}x")
            print(f"Scenarios with >20% advantage: {sum(1 for r in ratios if r > 1.2)}/{len(ratios)}")
            
            if np.mean(ratios) > 1.3:
                print("🚀 ADVANTAGE DETECTED with simple model!")
                print("This suggests the effect might be REAL, not just simulation bugs")
            else:
                print("📊 No clear advantage with simple model")
                print("Complex simulations may have been showing artifacts")
    
    else:
        print("All scenarios failed - fundamental simulation issues")

def quick_parameter_sweep():
    """
    Quick parameter sweep to see if effect depends on drive strength
    """
    print("\n5. Quick parameter sweep...")
    
    T1, T2 = 50.0, 25.0
    sim_time = 75.0  # 3×T2
    
    # Test different drive amplitudes
    amplitudes = [0.001, 0.005, 0.01, 0.02, 0.05]
    
    cw_results = []
    pulse_results = []
    
    for amp in amplitudes:
        print(f"   Testing amplitude {amp:.3f}...")
        
        # Initial state
        initial = [1.0, 0.0, 0.0, 0.0]
        
        # Continuous wave
        def cw_dynamics(t, state):
            return simple_coherence_evolution(t, state, T1, T2, amp, 5.0)
        
        sol_cw = solve_ivp(cw_dynamics, [0, sim_time], initial,
                          t_eval=np.linspace(0, sim_time, 150),
                          method='RK45', rtol=1e-8)
        
        # Pulsed 
        def pulse_dynamics(t, state):
            cycle_time = t % 4.0
            pulse_amp = amp * 3 if cycle_time < 0.3 else 0.0  # Same average power
            return simple_coherence_evolution(t, state, T1, T2, pulse_amp, 5.0)
        
        sol_pulse = solve_ivp(pulse_dynamics, [0, sim_time], initial,
                             t_eval=np.linspace(0, sim_time, 150),
                             method='RK45', rtol=1e-8)
        
        if sol_cw.success and sol_pulse.success:
            # Extract steady-state coherences
            cw_coherence = np.sqrt(sol_cw.y[2, :]**2 + sol_cw.y[3, :]**2)
            pulse_coherence = np.sqrt(sol_pulse.y[2, :]**2 + sol_pulse.y[3, :]**2)
            
            steady_start = len(cw_coherence) // 3
            cw_avg = np.mean(cw_coherence[steady_start:])
            pulse_avg = np.mean(pulse_coherence[steady_start:])
            
            cw_results.append(cw_avg)
            pulse_results.append(pulse_avg)
            
            ratio = cw_avg / pulse_avg if pulse_avg > 0 else 0
            print(f"      CW: {cw_avg:.6f}, Pulsed: {pulse_avg:.6f}, Ratio: {ratio:.3f}")
        else:
            print(f"      Failed for amplitude {amp:.3f}")
            cw_results.append(0)
            pulse_results.append(0)
    
    # Analysis
    valid_ratios = []
    for i, amp in enumerate(amplitudes):
        if cw_results[i] > 0 and pulse_results[i] > 0:
            valid_ratios.append(cw_results[i] / pulse_results[i])
    
    if valid_ratios:
        avg_ratio = np.mean(valid_ratios)
        print(f"\n   Parameter sweep results:")
        print(f"   Average CW/Pulsed ratio: {avg_ratio:.3f}")
        print(f"   Ratio std deviation: {np.std(valid_ratios):.3f}")
        
        if avg_ratio > 1.3:
            print("   🚀 CONSISTENT ADVANTAGE across drive strengths!")
        elif avg_ratio > 1.05:
            print("   📈 Modest advantage across drive strengths")
        else:
            print("   📊 No consistent advantage")
    
    return valid_ratios

if __name__ == "__main__":
    print("This simple test will:")
    print("1. Validate basic T1/T2 physics work correctly")
    print("2. Test CW vs pulsed with simple, bulletproof model")
    print("3. Check across multiple T1/T2 values")
    print("4. Quick parameter sweep")
    print("5. Reality-check all results")
    print("\nRuns in ~5-10 minutes instead of hours")
    print("If this shows realistic decay + CW advantage, effect is likely real")
    print("If this shows broken physics, we know where the bugs are")
    print("=" * 60)
    
    try:
        run_simple_sanity_check()
        quick_parameter_sweep()
        
        print("\n" + "="*60)
        print("SIMPLE SANITY CHECK COMPLETED!")
        print("Check simple_sanity_results.txt for analysis")
        
    except Exception as e:
        print(f"\nSimple test failed: {e}")
        import traceback
        traceback.print_exc()