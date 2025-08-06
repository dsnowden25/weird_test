import numpy as np
from scipy.integrate import solve_ivp

print("Starting electrical-quantum coupling simulation...")

def coupled_dynamics(t, y, w_e, w_q, g):
    """
    Coupled electrical-quantum system
    y = [q_elec, i_elec, c0_real, c0_imag, c1_real, c1_imag]
    """
    q_e, i_e, c0_r, c0_i, c1_r, c1_i = y
    
    # Electrical oscillator with quantum backaction
    quantum_energy = c1_r**2 + c1_i**2  # |c1|^2
    dq_dt = i_e
    di_dt = -w_e**2 * q_e + g * quantum_energy
    
    # Quantum system coupled to electrical field
    coupling_term = g * q_e
    dc0_dt_real = coupling_term * c1_i
    dc0_dt_imag = -coupling_term * c1_r
    dc1_dt_real = -w_q * c1_i + coupling_term * c0_i
    dc1_dt_imag = w_q * c1_r - coupling_term * c0_r
    
    return [dq_dt, di_dt, dc0_dt_real, dc0_dt_imag, dc1_dt_real, dc1_dt_imag]

def continuous_drive_dynamics(t, y, w_e, w_q, g, drive_amp, drive_freq):
    """Same as above but with continuous external drive"""
    q_e, i_e, c0_r, c0_i, c1_r, c1_i = y
    
    # External continuous drive
    external_drive = drive_amp * np.sin(drive_freq * t)
    
    quantum_energy = c1_r**2 + c1_i**2
    dq_dt = i_e
    di_dt = -w_e**2 * q_e + g * quantum_energy + external_drive
    
    coupling_term = g * q_e
    dc0_dt_real = coupling_term * c1_i
    dc0_dt_imag = -coupling_term * c1_r
    dc1_dt_real = -w_q * c1_i + coupling_term * c0_i
    dc1_dt_imag = w_q * c1_r - coupling_term * c0_r
    
    return [dq_dt, di_dt, dc0_dt_real, dc0_dt_imag, dc1_dt_real, dc1_dt_imag]

def pulsed_drive_dynamics(t, y, w_e, w_q, g, pulse_amp, pulse_period, pulse_width):
    """Same but with pulsed drive"""
    q_e, i_e, c0_r, c0_i, c1_r, c1_i = y
    
    # Pulsed external drive
    cycle_time = t % pulse_period
    if cycle_time < pulse_width:
        external_drive = pulse_amp
    else:
        external_drive = 0.0
    
    quantum_energy = c1_r**2 + c1_i**2
    dq_dt = i_e
    di_dt = -w_e**2 * q_e + g * quantum_energy + external_drive
    
    coupling_term = g * q_e
    dc0_dt_real = coupling_term * c1_i
    dc0_dt_imag = -coupling_term * c1_r
    dc1_dt_real = -w_q * c1_i + coupling_term * c0_i
    dc1_dt_imag = w_q * c1_r - coupling_term * c0_r
    
    return [dq_dt, di_dt, dc0_dt_real, dc0_dt_imag, dc1_dt_real, dc1_dt_imag]

# Simulation parameters
w_electrical = 1.0
w_quantum = 1.0  
coupling_strength = 0.1
t_max = 50
t_points = 1000

# Initial conditions: [q, i, c0_real, c0_imag, c1_real, c1_imag]
# Start in ground state with small electrical oscillation
initial = [0.1, 0.0, 1.0, 0.0, 0.0, 0.0]

print("Running continuous wave simulation...")

# Continuous wave simulation
def cw_dynamics(t, y):
    return continuous_drive_dynamics(t, y, w_electrical, w_quantum, coupling_strength, 
                                   drive_amp=0.05, drive_freq=w_electrical)

sol_cw = solve_ivp(cw_dynamics, [0, t_max], initial, 
                   t_eval=np.linspace(0, t_max, t_points), rtol=1e-6)

print("Running pulsed drive simulation...")

# Pulsed drive simulation  
def pulse_dynamics(t, y):
    return pulsed_drive_dynamics(t, y, w_electrical, w_quantum, coupling_strength,
                               pulse_amp=0.2, pulse_period=10, pulse_width=2)

sol_pulse = solve_ivp(pulse_dynamics, [0, t_max], initial,
                     t_eval=np.linspace(0, t_max, t_points), rtol=1e-6)

print("Analyzing results...")

# Analyze continuous wave results
t_cw = sol_cw.t
q_cw, i_cw = sol_cw.y[0], sol_cw.y[1]
c0_cw = sol_cw.y[2] + 1j * sol_cw.y[3]
c1_cw = sol_cw.y[4] + 1j * sol_cw.y[5]

excited_pop_cw = np.abs(c1_cw)**2
coherence_cw = 2 * np.abs(c0_cw * np.conj(c1_cw))

# Analyze pulsed results
t_pulse = sol_pulse.t  
q_pulse, i_pulse = sol_pulse.y[0], sol_pulse.y[1]
c0_pulse = sol_pulse.y[2] + 1j * sol_pulse.y[3]
c1_pulse = sol_pulse.y[4] + 1j * sol_pulse.y[5]

excited_pop_pulse = np.abs(c1_pulse)**2
coherence_pulse = 2 * np.abs(c0_pulse * np.conj(c1_pulse))

print("Writing results to files...")

# Write main results
with open('simulation_output.txt', 'w') as f:
    f.write("ELECTRICAL-QUANTUM COUPLING SIMULATION\n")
    f.write("=" * 45 + "\n\n")
    
    f.write("PARAMETERS:\n")
    f.write(f"Electrical frequency: {w_electrical}\n")
    f.write(f"Quantum frequency: {w_quantum}\n")
    f.write(f"Coupling strength: {coupling_strength}\n")
    f.write(f"Simulation time: {t_max}\n\n")
    
    f.write("CONTINUOUS WAVE RESULTS:\n")
    f.write(f"Average coherence: {np.mean(coherence_cw):.6f}\n")
    f.write(f"Max coherence: {np.max(coherence_cw):.6f}\n")
    f.write(f"Final excited population: {excited_pop_cw[-1]:.6f}\n\n")
    
    f.write("PULSED DRIVE RESULTS:\n")
    f.write(f"Average coherence: {np.mean(coherence_pulse):.6f}\n")
    f.write(f"Max coherence: {np.max(coherence_pulse):.6f}\n")
    f.write(f"Final excited population: {excited_pop_pulse[-1]:.6f}\n\n")
    
    ratio = np.mean(coherence_cw) / np.mean(coherence_pulse)
    f.write("COMPARISON:\n")
    f.write(f"Coherence ratio (CW/Pulsed): {ratio:.3f}\n")
    if ratio > 1.0:
        f.write(">>> CONTINUOUS WAVE WINS! <<<\n")
    else:
        f.write(">>> Pulsed drive performs better\n")

# Write time series data
with open('time_series_data.txt', 'w') as f:
    f.write("time,electrical_field_cw,quantum_excited_cw,coherence_cw,electrical_field_pulse,quantum_excited_pulse,coherence_pulse\n")
    
    # Interpolate pulse data to match cw timepoints for comparison
    excited_pulse_interp = np.interp(t_cw, t_pulse, excited_pop_pulse)
    coherence_pulse_interp = np.interp(t_cw, t_pulse, coherence_pulse)
    q_pulse_interp = np.interp(t_cw, t_pulse, q_pulse)
    
    for i in range(len(t_cw)):
        f.write(f"{t_cw[i]:.6f},{q_cw[i]:.6f},{excited_pop_cw[i]:.6f},{coherence_cw[i]:.6f},")
        f.write(f"{q_pulse_interp[i]:.6f},{excited_pulse_interp[i]:.6f},{coherence_pulse_interp[i]:.6f}\n")

print("Testing resonance hypothesis...")

# Test resonance sweep
freq_ratios = np.linspace(0.5, 2.0, 15)
coherence_results = []

for ratio in freq_ratios:
    w_q_test = w_electrical / ratio
    
    def test_dynamics(t, y):
        return continuous_drive_dynamics(t, y, w_electrical, w_q_test, coupling_strength,
                                       drive_amp=0.05, drive_freq=w_electrical)
    
    sol_test = solve_ivp(test_dynamics, [0, 20], initial, 
                        t_eval=np.linspace(0, 20, 200), rtol=1e-6)
    
    c0_test = sol_test.y[2] + 1j * sol_test.y[3]
    c1_test = sol_test.y[4] + 1j * sol_test.y[5]
    coherence_test = 2 * np.abs(c0_test * np.conj(c1_test))
    
    avg_coherence = np.mean(coherence_test)
    coherence_results.append(avg_coherence)

# Write resonance data
with open('resonance_data.txt', 'w') as f:
    f.write("RESONANCE SWEEP RESULTS\n")
    f.write("=" * 25 + "\n\n")
    f.write("frequency_ratio,average_coherence\n")
    
    for ratio, coherence in zip(freq_ratios, coherence_results):
        f.write(f"{ratio:.3f},{coherence:.6f}\n")
    
    best_idx = np.argmax(coherence_results)
    best_ratio = freq_ratios[best_idx]
    max_coherence = coherence_results[best_idx]
    
    f.write(f"\nBEST RATIO: {best_ratio:.3f}\n")
    f.write(f"MAX COHERENCE: {max_coherence:.6f}\n")
    
    # Check if peak is near resonance (ratio = 1.0)
    resonance_idx = np.argmin(np.abs(freq_ratios - 1.0))
    resonance_coherence = coherence_results[resonance_idx]
    f.write(f"COHERENCE AT RESONANCE (1.0): {resonance_coherence:.6f}\n")
    
    if abs(best_ratio - 1.0) < 0.15:
        f.write("\n>>> RESONANCE EFFECT DETECTED! <<<\n")
    else:
        f.write(f"\n>>> Peak at {best_ratio:.3f}, not at resonance\n")

print("Simulation complete!")
print("Check these files:")
print("- simulation_output.txt (main results)")
print("- time_series_data.txt (full data)")  
print("- resonance_data.txt (frequency sweep)")
print("\nKey question: Does continuous wave show higher coherence than pulsed?")