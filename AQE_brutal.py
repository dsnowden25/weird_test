#!/usr/bin/env python3
"""
Quick Quantum-Electrical Reality Check
=====================================

Ultra-fast version to test if there's any real physics here
without getting stuck in computational complexity.
"""

import numpy as np
import matplotlib.pyplot as plt

def quick_physics_check():
    """
    Fast analytical check of quantum-electrical coupling
    """
    print("⚡ QUICK PHYSICS REALITY CHECK")
    print("Testing core physical assumptions...")
    
    # Real superconducting circuit parameters
    omega_q = 5e9     # 5 GHz qubit
    omega_c = 5.2e9   # 5.2 GHz cavity  
    g = 50e6          # 50 MHz coupling
    T1 = 80e-6        # 80 μs T1
    T2 = 25e-6        # 25 μs T2*
    kappa = 1e6       # 1 MHz cavity decay
    
    # Key question: Dispersive shift strength
    chi = g**2 / abs(omega_q - omega_c)  # Dispersive coupling
    print(f"   Dispersive shift χ: {chi/1e6:.3f} MHz")
    
    # Signal strength analysis
    max_frequency_shift = chi  # Maximum shift when qubit switches states
    relative_shift = max_frequency_shift / omega_c
    print(f"   Relative frequency shift: {relative_shift*1e6:.3f} ppm")
    
    # SNR estimate  
    measurement_time = 1e-6  # 1 μs measurement
    measurement_photons = 100  # Typical number
    
    # Shot noise limited SNR
    snr_shot_noise = np.sqrt(measurement_photons) * relative_shift
    snr_db = 20 * np.log10(snr_shot_noise)
    print(f"   Shot-noise limited SNR: {snr_db:.1f} dB")
    
    # Thermal noise limit (at 15 mK)
    temperature = 0.015  # 15 mK
    n_thermal = 1/(np.exp(6.626e-34 * omega_c / (1.38e-23 * temperature)) - 1)
    thermal_noise_photons = n_thermal
    print(f"   Thermal photons: {thermal_noise_photons:.6f}")
    
    # Decoherence time scale
    coherence_time = min(T1, T2)
    measurement_cycles = coherence_time / measurement_time
    print(f"   Measurement cycles before decoherence: {measurement_cycles:.0f}")
    
    return {
        'dispersive_shift_MHz': chi/1e6,
        'relative_shift_ppm': relative_shift*1e6, 
        'snr_db': snr_db,
        'thermal_photons': thermal_noise_photons,
        'coherence_cycles': measurement_cycles
    }

def simulate_realistic_correlation():
    """
    Simplified but realistic correlation calculation
    """
    print("\n📊 REALISTIC CORRELATION SIMULATION")
    
    # Parameters
    n_points = 1000
    T2 = 25e-6
    chi = 1e6  # 1 MHz dispersive shift
    readout_fidelity = 0.985
    
    # Time evolution
    t = np.linspace(0, 2*T2, n_points)
    
    # Quantum state evolution with realistic decoherence
    # Start in superposition, decay exponentially
    coherence = np.exp(-t / T2)
    excited_population = 0.5 * (1 + coherence * np.cos(chi * 2 * np.pi * t))
    
    # Cavity response (dispersive coupling)
    cavity_frequency_shift = chi * (2 * excited_population - 1)
    cavity_amplitude = 0.1 * cavity_frequency_shift / chi  # Normalized response
    
    # Add realistic measurement noise
    measurement_noise = 0.02  # 2% noise level
    measured_cavity = cavity_amplitude + measurement_noise * np.random.randn(n_points)
    
    # Add readout errors  
    measured_qubit = excited_population.copy()
    error_indices = np.random.rand(n_points) > readout_fidelity
    measured_qubit[error_indices] = 1 - measured_qubit[error_indices]
    
    # Calculate correlation including decoherence effects
    try:
        correlation = np.corrcoef(measured_qubit, measured_cavity)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    except:
        correlation = 0.0
    
    # SNR calculation
    signal_power = np.var(cavity_amplitude)
    noise_power = measurement_noise**2
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    print(f"   Initial correlation (t=0): {np.corrcoef(excited_population[:100], cavity_amplitude[:100])[0,1]:.3f}")
    print(f"   Final correlation (with decoherence): {abs(correlation):.3f}")
    print(f"   SNR: {snr_db:.1f} dB")
    print(f"   Coherence at end: {coherence[-1]:.3f}")
    
    # Plot the results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Quick Reality Check: Quantum-Electrical Coupling', fontsize=14)
    
    time_us = t * 1e6
    
    # Quantum evolution
    axes[0, 0].plot(time_us, excited_population, 'b-', linewidth=2, label='Excited Population')
    axes[0, 0].plot(time_us, coherence, 'r--', linewidth=2, label='Coherence')
    axes[0, 0].set_xlabel('Time (μs)')
    axes[0, 0].set_ylabel('Population/Coherence')
    axes[0, 0].set_title('Quantum Decoherence')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cavity response
    axes[0, 1].plot(time_us, cavity_amplitude, 'g-', linewidth=2, label='True Signal')
    axes[0, 1].plot(time_us, measured_cavity, 'k--', alpha=0.6, label='Measured')
    axes[0, 1].set_xlabel('Time (μs)')
    axes[0, 1].set_ylabel('Cavity Amplitude')
    axes[0, 1].set_title('Electrical Response')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Correlation
    axes[1, 0].scatter(measured_qubit, measured_cavity, alpha=0.6, s=20)
    axes[1, 0].set_xlabel('Measured Qubit State')
    axes[1, 0].set_ylabel('Measured Cavity Signal')
    axes[1, 0].set_title(f'Correlation: {abs(correlation):.3f}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Reality verdict
    axes[1, 1].axis('off')
    
    if abs(correlation) > 0.15 and snr_db > 3:
        verdict = "🎉 SURVIVES REALITY!"
        verdict_color = 'lightgreen'
    elif abs(correlation) > 0.08 and snr_db > 0:
        verdict = "⚡ WEAK BUT REAL"
        verdict_color = 'lightyellow'
    elif abs(correlation) > 0.03:
        verdict = "📊 BARELY DETECTABLE"
        verdict_color = 'lightcoral'
    else:
        verdict = "💀 KILLED BY REALITY"
        verdict_color = 'lightpink'
    
    verdict_text = f"""
QUICK REALITY CHECK:

Correlation: {abs(correlation):.3f}
SNR: {snr_db:.1f} dB
Decoherence: {coherence[-1]:.3f} remaining

REALISTIC FACTORS:
• T2* decoherence ✓
• Measurement noise ✓  
• Readout errors ✓
• Dispersive coupling ✓

VERDICT: {verdict}

MEANING FOR YOUR EMAIL:
{'Strong enough to discuss seriously' if abs(correlation) > 0.1 else 'Honest about weak signals'}
    """
    
    axes[1, 1].text(0.1, 0.9, verdict_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=verdict_color, alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('quick_reality_check.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'correlation': abs(correlation),
        'snr_db': snr_db,
        'coherence_survival': coherence[-1],
        'verdict': verdict
    }

def analytical_feasibility_check():
    """
    Pure analytical check - no differential equations
    """
    print("\n🧮 ANALYTICAL FEASIBILITY CHECK")
    print("Pure physics analysis - no numerical simulation")
    
    # Standard dispersive measurement parameters
    g = 50e6      # 50 MHz coupling
    omega_q = 5e9
    omega_c = 5.2e9
    T2 = 25e-6
    
    # Dispersive shift
    chi = g**2 / (omega_q - omega_c)
    print(f"   Dispersive shift: {chi/1e6:.1f} MHz")
    
    # Maximum detectable signal
    max_cavity_shift = chi / omega_c  # Fractional frequency shift
    print(f"   Max detectable shift: {max_cavity_shift*1e6:.1f} ppm")
    
    # Decoherence impact
    coherence_after_T2 = 1/np.e  # Standard definition
    signal_after_decoherence = max_cavity_shift * coherence_after_T2
    print(f"   Signal after T2 decay: {signal_after_decoherence*1e6:.2f} ppm")
    
    # Noise floor estimate
    measurement_noise = 0.01  # 1% typical measurement noise
    snr_estimate = signal_after_decoherence / measurement_noise
    snr_db_estimate = 20 * np.log10(snr_estimate)
    
    print(f"   Estimated SNR: {snr_db_estimate:.1f} dB")
    
    # Feasibility assessment
    print(f"\n🎯 ANALYTICAL VERDICT:")
    if snr_db_estimate > 10:
        print("   ✅ EASILY DETECTABLE - strong physics")
    elif snr_db_estimate > 3:
        print("   ⚡ DETECTABLE - real but challenging")
    elif snr_db_estimate > 0:
        print("   📊 MARGINAL - at the edge of detectability")
    else:
        print("   💀 BELOW NOISE FLOOR - not practically detectable")
    
    return snr_db_estimate

if __name__ == "__main__":
    print("⚡" * 40)
    print("⚡ QUICK REALITY CHECK (No Hanging!) ⚡")
    print("⚡" * 40)
    
    # Fast physics check
    physics_results = quick_physics_check()
    
    # Fast correlation simulation  
    correlation_results = simulate_realistic_correlation()
    
    # Pure analytical check
    analytical_snr = analytical_feasibility_check()
    
    print(f"\n⚡ QUICK REALITY SUMMARY:")
    print(f"   Dispersive shift: {physics_results['dispersive_shift_MHz']:.2f} MHz")
    print(f"   Simulated correlation: {correlation_results['correlation']:.3f}")
    print(f"   Simulated SNR: {correlation_results['snr_db']:.1f} dB")
    print(f"   Analytical SNR: {analytical_snr:.1f} dB")
    
    # Final assessment for your email
    if correlation_results['correlation'] > 0.1 and correlation_results['snr_db'] > 0:
        print(f"\n✅ FOR YOUR EMAIL:")
        print(f"   'Realistic simulations show {correlation_results['correlation']:.1%} correlation'")
        print(f"   'SNR analysis suggests {correlation_results['snr_db']:.1f} dB above noise'")
        print(f"   'Worth experimental investigation'")
    else:
        print(f"\n⚠️  FOR YOUR EMAIL:")
        print(f"   'Initial results were promising but realistic analysis shows'")
        print(f"   'correlation drops to {correlation_results['correlation']:.1%} with proper decoherence'")
        print(f"   'Still curious about experimental feasibility'")
    
    print(f"\n🔬 HONEST ASSESSMENT:")
    print(f"Even this quick check shows the challenges of real quantum systems.")
    print(f"But that doesn't mean your idea is wrong - just that quantum physics is hard!")