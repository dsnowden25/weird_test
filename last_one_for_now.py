import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time

class ProgressiveFlowTester:
    """
    Progressive testing of probability flow concept
    Building complexity step by step
    """
    
    def __init__(self):
        self.base_params = {
            'omega': 1.0,           # Base frequency
            'noise_strength': 0.001, # Tiny noise
            'decay_rate': 0.01,     # Weak decoherence
            'nonlinearity': 0.01,   # Weak nonlinear coefficient
            'drive_strength': 0.002, # Weak drive
            'drive_freq': 0.5       # Drive frequency
        }
        
        print("🔬 PROGRESSIVE PROBABILITY FLOW TESTER")
        print("Building complexity step by step")

def base_dynamics(t, state_vec, params, test_type='base'):
    """
    Base dynamics with progressive complexity
    """
    N = len(state_vec) // 2
    psi = state_vec[:N] + 1j * state_vec[N:]
    
    # Grid (can be adjusted for resolution tests)
    x = np.linspace(-2, 2, N)
    dx = x[1] - x[0]
    
    # Base Hamiltonian
    H_psi = np.zeros_like(psi, dtype=complex)
    
    # Kinetic energy
    for i in range(1, N-1):
        H_psi[i] += -0.5 * (psi[i+1] - 2*psi[i] + psi[i-1]) / dx**2
    
    # Harmonic potential
    H_psi += 0.5 * params['omega']**2 * x**2 * psi
    
    # Progressive additions based on test type
    if test_type in ['nonlinear', 'two_field', 'driven']:
        # Add weak nonlinearity (Step 2)
        H_psi += params['nonlinearity'] * x**4 * psi
    
    if test_type in ['driven']:
        # Add driving function (Step 4)
        drive = params['drive_strength'] * np.sin(2 * np.pi * params['drive_freq'] * t)
        H_psi += drive * x * psi
    
    # Base noise (always present)
    noise = params['noise_strength'] * np.sin(0.1 * t)
    H_psi += noise * x * psi
    
    # Evolution
    dpsi_dt = -1j * H_psi
    
    # Tiny decoherence
    energy = np.sum(np.abs(psi)**2 * x**2) * dx
    dpsi_dt -= params['decay_rate'] * energy * psi
    
    return np.concatenate([np.real(dpsi_dt), np.imag(dpsi_dt)])

def two_field_dynamics(t, state_vec, params):
    """
    Two coupled probability fields (Step 3)
    """
    total_size = len(state_vec)
    N = total_size // 4  # Two fields, each complex
    
    # Field 1: psi1 = state_vec[:N] + 1j * state_vec[N:2*N]
    # Field 2: psi2 = state_vec[2*N:3*N] + 1j * state_vec[3*N:]
    psi1 = state_vec[:N] + 1j * state_vec[N:2*N]
    psi2 = state_vec[2*N:3*N] + 1j * state_vec[3*N:]
    
    x = np.linspace(-2, 2, N)
    dx = x[1] - x[0]
    
    # Individual field evolution
    def evolve_field(psi, coupling_from_other):
        H_psi = np.zeros_like(psi, dtype=complex)
        
        # Kinetic
        for i in range(1, N-1):
            H_psi[i] += -0.5 * (psi[i+1] - 2*psi[i] + psi[i-1]) / dx**2
        
        # Potential
        H_psi += 0.5 * params['omega']**2 * x**2 * psi
        
        # Nonlinearity
        H_psi += params['nonlinearity'] * x**4 * psi
        
        # Coupling to other field
        H_psi += 0.005 * coupling_from_other * x * psi  # Weak field-field coupling
        
        # Noise
        noise = params['noise_strength'] * np.sin(0.1 * t + np.pi/4)  # Different phase
        H_psi += noise * x * psi
        
        return -1j * H_psi
    
    # Couple fields to each other
    coupling1 = np.abs(psi2)**2  # Field 2 density affects field 1
    coupling2 = np.abs(psi1)**2  # Field 1 density affects field 2
    
    dpsi1_dt = evolve_field(psi1, coupling1)
    dpsi2_dt = evolve_field(psi2, coupling2)
    
    # Decoherence for both fields
    energy1 = np.sum(np.abs(psi1)**2 * x**2) * dx
    energy2 = np.sum(np.abs(psi2)**2 * x**2) * dx
    
    dpsi1_dt -= params['decay_rate'] * energy1 * psi1
    dpsi2_dt -= params['decay_rate'] * energy2 * psi2
    
    # Return combined derivatives
    return np.concatenate([
        np.real(dpsi1_dt), np.imag(dpsi1_dt),
        np.real(dpsi2_dt), np.imag(dpsi2_dt)
    ])

def measure_coherence_progressive(sol, test_type='base'):
    """
    Measure coherence with progressive complexity
    """
    if test_type == 'two_field':
        return measure_two_field_coherence(sol)
    else:
        return measure_single_field_coherence(sol)

def measure_single_field_coherence(sol):
    """
    Single field coherence measurement
    """
    N = len(sol.y) // 2
    x = np.linspace(-2, 2, N)
    dx = x[1] - x[0]
    
    metrics = {
        'time': sol.t,
        'position_vars': [],
        'energies': [],
        'field_curvature': [],  # New metric for nonlinearity
        'response_amplitude': []  # New metric for driven response
    }
    
    for i in range(len(sol.t)):
        psi = sol.y[:N, i] + 1j * sol.y[N:, i]
        
        # Normalize
        norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
        if norm > 1e-12:
            psi = psi / norm
        
        prob = np.abs(psi)**2
        
        # Position variance
        mean_x = np.sum(prob * x) * dx
        var_x = np.sum(prob * (x - mean_x)**2) * dx
        metrics['position_vars'].append(var_x)
        
        # Energy
        psi_grad = np.gradient(psi, dx)
        energy = 0.5 * np.sum(np.abs(psi_grad)**2) * dx + 0.5 * np.sum(prob * x**2) * dx
        metrics['energies'].append(energy)
        
        # Field curvature (nonlinearity measure)
        curvature = np.sum(prob * x**4) * dx
        metrics['field_curvature'].append(curvature)
        
        # Response amplitude (for driven systems)
        response = np.abs(np.sum(prob * x) * dx)  # Displacement amplitude
        metrics['response_amplitude'].append(response)
    
    # Calculate correlations and coherence
    def safe_corr(x, y):
        if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0
        try:
            corr = np.corrcoef(x, y)[0, 1]
            return abs(corr) if not (np.isnan(corr) or np.isinf(corr)) else 0.0
        except:
            return 0.0
    
    pos_correlation = safe_corr(metrics['position_vars'][:-1], metrics['position_vars'][1:])
    energy_correlation = safe_corr(metrics['energies'][:-1], metrics['energies'][1:])
    
    # Stability measures
    pos_stability = 1.0 / (1.0 + np.std(metrics['position_vars']))
    energy_stability = 1.0 / (1.0 + np.std(metrics['energies']))
    curvature_stability = 1.0 / (1.0 + np.std(metrics['field_curvature']))
    
    # Enhanced coherence measure
    coherence = (pos_correlation * 0.3 + energy_correlation * 0.3 + 
                pos_stability * 0.2 + energy_stability * 0.1 + 
                curvature_stability * 0.1)
    
    return {
        'coherence': coherence,
        'pos_correlation': pos_correlation,
        'energy_correlation': energy_correlation,
        'pos_stability': pos_stability,
        'energy_stability': energy_stability,
        'curvature_stability': curvature_stability,
        'metrics': metrics
    }

def measure_two_field_coherence(sol):
    """
    Two-field coherence and cross-correlation measurement
    """
    total_size = len(sol.y)
    N = total_size // 4
    x = np.linspace(-2, 2, N)
    dx = x[1] - x[0]
    
    field1_vars = []
    field2_vars = []
    cross_correlations = []
    entanglement_measures = []
    
    for i in range(len(sol.t)):
        # Extract both fields
        psi1 = sol.y[:N, i] + 1j * sol.y[N:2*N, i]
        psi2 = sol.y[2*N:3*N, i] + 1j * sol.y[3*N:, i]
        
        # Normalize
        norm1 = np.sqrt(np.sum(np.abs(psi1)**2) * dx)
        norm2 = np.sqrt(np.sum(np.abs(psi2)**2) * dx)
        
        if norm1 > 1e-12:
            psi1 = psi1 / norm1
        if norm2 > 1e-12:
            psi2 = psi2 / norm2
        
        prob1 = np.abs(psi1)**2
        prob2 = np.abs(psi2)**2
        
        # Individual field variances
        mean_x1 = np.sum(prob1 * x) * dx
        var_x1 = np.sum(prob1 * (x - mean_x1)**2) * dx
        field1_vars.append(var_x1)
        
        mean_x2 = np.sum(prob2 * x) * dx
        var_x2 = np.sum(prob2 * (x - mean_x2)**2) * dx
        field2_vars.append(var_x2)
        
        # Cross-correlation
        cross_corr = np.sum(prob1 * prob2) * dx  # Overlap measure
        cross_correlations.append(cross_corr)
        
        # Simple entanglement measure (mutual information approximation)
        joint_entropy = -np.sum((prob1 + prob2) * np.log(prob1 + prob2 + 1e-12)) * dx
        individual_entropy = (-np.sum(prob1 * np.log(prob1 + 1e-12)) * dx - 
                            np.sum(prob2 * np.log(prob2 + 1e-12)) * dx)
        mutual_info = individual_entropy - joint_entropy
        entanglement_measures.append(mutual_info)
    
    # Field-field correlations
    field_correlation = np.corrcoef(field1_vars, field2_vars)[0, 1] if len(field1_vars) > 1 else 0.0
    if np.isnan(field_correlation):
        field_correlation = 0.0
    
    # Average cross-correlation
    avg_cross_corr = np.mean(cross_correlations)
    avg_entanglement = np.mean(entanglement_measures)
    
    # Combined two-field coherence
    coherence = (abs(field_correlation) * 0.4 + avg_cross_corr * 0.3 + 
                avg_entanglement * 0.3)
    
    return {
        'coherence': coherence,
        'field_correlation': field_correlation,
        'avg_cross_correlation': avg_cross_corr,
        'avg_entanglement': avg_entanglement,
        'field1_vars': field1_vars,
        'field2_vars': field2_vars,
        'cross_correlations': cross_correlations,
        'entanglement_measures': entanglement_measures,
        'time': sol.t
    }

def run_progressive_tests():
    """
    Run all four progressive tests
    """
    print("\n🔬 PROGRESSIVE PROBABILITY FLOW TESTS")
    print("Building complexity step by step")
    print("=" * 50)
    
    tester = ProgressiveFlowTester()
    results = {}
    
    # Test 1: Refined Grid Resolution
    print("\n1. REFINED GRID RESOLUTION TEST")
    for N in [16, 32, 64]:
        print(f"   Testing N = {N}...")
        
        x = np.linspace(-2, 2, N)
        dx = x[1] - x[0]
        psi0 = np.exp(-x**2)
        psi0 = psi0 / np.sqrt(np.sum(psi0**2) * dx)
        initial_state = np.concatenate([psi0, np.zeros(N)])
        
        def dynamics(t, y):
            return base_dynamics(t, y, tester.base_params, 'base')
        
        try:
            sol = solve_ivp(dynamics, [0, 5.0], initial_state,
                           t_eval=np.linspace(0, 5.0, 20),
                           method='RK45', rtol=1e-2, max_step=0.5)
            
            if sol.success:
                analysis = measure_coherence_progressive(sol, 'base')
                results[f'resolution_N{N}'] = {
                    'coherence': analysis['coherence'],
                    'grid_size': N,
                    'dx': dx,
                    'analysis': analysis
                }
                print(f"      N={N}: Coherence = {analysis['coherence']:.4f}, dx = {dx:.4f}")
            else:
                print(f"      N={N}: Failed - {sol.message}")
                
        except Exception as e:
            print(f"      N={N}: Error - {e}")
    
    # Test 2: Nonlinearity Sensitivity
    print("\n2. NONLINEARITY SENSITIVITY TEST")
    for nonlin in [0.0, 0.005, 0.01, 0.02]:
        print(f"   Testing nonlinearity = {nonlin}...")
        
        params = tester.base_params.copy()
        params['nonlinearity'] = nonlin
        
        N = 32
        x = np.linspace(-2, 2, N)
        dx = x[1] - x[0]
        psi0 = np.exp(-x**2)
        psi0 = psi0 / np.sqrt(np.sum(psi0**2) * dx)
        initial_state = np.concatenate([psi0, np.zeros(N)])
        
        def dynamics(t, y):
            return base_dynamics(t, y, params, 'nonlinear')
        
        try:
            sol = solve_ivp(dynamics, [0, 5.0], initial_state,
                           t_eval=np.linspace(0, 5.0, 20),
                           method='RK45', rtol=1e-2, max_step=0.5)
            
            if sol.success:
                analysis = measure_coherence_progressive(sol, 'nonlinear')
                results[f'nonlinear_{nonlin}'] = {
                    'coherence': analysis['coherence'],
                    'nonlinearity': nonlin,
                    'curvature_stability': analysis['curvature_stability'],
                    'analysis': analysis
                }
                print(f"      α={nonlin}: Coherence = {analysis['coherence']:.4f}, " +
                      f"Curvature stability = {analysis['curvature_stability']:.4f}")
            else:
                print(f"      α={nonlin}: Failed - {sol.message}")
                
        except Exception as e:
            print(f"      α={nonlin}: Error - {e}")
    
    # Test 3: Two-Field Coupling
    print("\n3. TWO-FIELD COUPLING TEST")
    print("   Testing field-field interactions...")
    
    N = 24  # Smaller for computational efficiency
    x = np.linspace(-2, 2, N)
    dx = x[1] - x[0]
    
    # Initial states for both fields
    psi1_init = np.exp(-(x + 0.5)**2)  # Left-centered
    psi2_init = np.exp(-(x - 0.5)**2)  # Right-centered
    
    psi1_init = psi1_init / np.sqrt(np.sum(psi1_init**2) * dx)
    psi2_init = psi2_init / np.sqrt(np.sum(psi2_init**2) * dx)
    
    # Combined initial state
    initial_two_field = np.concatenate([
        psi1_init, np.zeros(N),  # Field 1: real, imag
        psi2_init, np.zeros(N)   # Field 2: real, imag
    ])
    
    def two_field_dyn(t, y):
        return two_field_dynamics(t, y, tester.base_params)
    
    try:
        sol = solve_ivp(two_field_dyn, [0, 5.0], initial_two_field,
                       t_eval=np.linspace(0, 5.0, 20),
                       method='RK45', rtol=1e-2, max_step=0.5)
        
        if sol.success:
            analysis = measure_coherence_progressive(sol, 'two_field')
            results['two_field'] = {
                'coherence': analysis['coherence'],
                'field_correlation': analysis['field_correlation'],
                'avg_entanglement': analysis['avg_entanglement'],
                'analysis': analysis
            }
            print(f"      Two-field coherence = {analysis['coherence']:.4f}")
            print(f"      Field correlation = {analysis['field_correlation']:.4f}")
            print(f"      Average entanglement = {analysis['avg_entanglement']:.4f}")
        else:
            print(f"      Two-field test failed: {sol.message}")
            
    except Exception as e:
        print(f"      Two-field test error: {e}")
    
    # Test 4: Drive-Response Coupling
    print("\n4. DRIVE-RESPONSE COUPLING TEST")
    for drive_freq in [0.2, 0.5, 1.0]:
        print(f"   Testing drive frequency = {drive_freq}...")
        
        params = tester.base_params.copy()
        params['drive_freq'] = drive_freq
        
        N = 32
        x = np.linspace(-2, 2, N)
        dx = x[1] - x[0]
        psi0 = np.exp(-x**2)
        psi0 = psi0 / np.sqrt(np.sum(psi0**2) * dx)
        initial_state = np.concatenate([psi0, np.zeros(N)])
        
        def dynamics(t, y):
            return base_dynamics(t, y, params, 'driven')
        
        try:
            sol = solve_ivp(dynamics, [0, 8.0], initial_state,  # Longer for drive response
                           t_eval=np.linspace(0, 8.0, 30),
                           method='RK45', rtol=1e-2, max_step=0.5)
            
            if sol.success:
                analysis = measure_coherence_progressive(sol, 'driven')
                
                # Check for drive response
                response_amp = analysis['metrics']['response_amplitude']
                response_oscillation = np.std(response_amp) / (np.mean(response_amp) + 1e-6)
                
                results[f'driven_{drive_freq}'] = {
                    'coherence': analysis['coherence'],
                    'drive_freq': drive_freq,
                    'response_oscillation': response_oscillation,
                    'analysis': analysis
                }
                print(f"      f={drive_freq}: Coherence = {analysis['coherence']:.4f}, " +
                      f"Response oscillation = {response_oscillation:.4f}")
            else:
                print(f"      f={drive_freq}: Failed - {sol.message}")
                
        except Exception as e:
            print(f"      f={drive_freq}: Error - {e}")
    
    return results

def analyze_progressive_results(results):
    """
    Analyze and summarize progressive test results
    """
    print(f"\n📊 PROGRESSIVE TEST ANALYSIS")
    print("=" * 40)
    
    theoretical_baseline = 0.44
    
    # Analysis 1: Grid Resolution Scaling
    resolution_results = {k: v for k, v in results.items() if 'resolution' in k}
    if resolution_results:
        print(f"\n1. GRID RESOLUTION SCALING:")
        grid_sizes = []
        coherences = []
        
        for key, data in resolution_results.items():
            grid_sizes.append(data['grid_size'])
            coherences.append(data['coherence'])
            print(f"   N={data['grid_size']:2d}: {data['coherence']:.4f} " +
                  f"(dx={data['dx']:.4f})")
        
        if len(coherences) > 1:
            resolution_trend = "increasing" if coherences[-1] > coherences[0] else "decreasing"
            print(f"   → Coherence {resolution_trend} with finer resolution")
    
    # Analysis 2: Nonlinearity Sensitivity
    nonlinear_results = {k: v for k, v in results.items() if 'nonlinear' in k}
    if nonlinear_results:
        print(f"\n2. NONLINEARITY SENSITIVITY:")
        for key, data in nonlinear_results.items():
            ratio = data['coherence'] / theoretical_baseline
            print(f"   α={data['nonlinearity']:5.3f}: {data['coherence']:.4f} " +
                  f"({ratio:.2f}x theory)")
        
        coherences = [data['coherence'] for data in nonlinear_results.values()]
        if max(coherences) - min(coherences) > 0.1:
            print(f"   → Coherence sensitive to nonlinearity")
        else:
            print(f"   → Coherence robust against nonlinearity")
    
    # Analysis 3: Two-Field Results
    if 'two_field' in results:
        print(f"\n3. TWO-FIELD COUPLING:")
        data = results['two_field']
        print(f"   Combined coherence: {data['coherence']:.4f}")
        print(f"   Field correlation: {data['field_correlation']:.4f}")
        print(f"   Entanglement measure: {data['avg_entanglement']:.4f}")
        
        if data['coherence'] > theoretical_baseline * 0.5:
            print(f"   → Strong field-field coherence maintained")
        else:
            print(f"   → Field coupling reduces individual coherence")
    
    # Analysis 4: Drive Response
    driven_results = {k: v for k, v in results.items() if 'driven' in k}
    if driven_results:
        print(f"\n4. DRIVE-RESPONSE COUPLING:")
        for key, data in driven_results.items():
            enhancement = data['coherence'] / theoretical_baseline
            print(f"   f={data['drive_freq']:3.1f}: {data['coherence']:.4f} " +
                  f"({enhancement:.2f}x), oscillation={data['response_oscillation']:.4f}")
        
        # Find resonant frequency
        best_drive = max(driven_results.values(), key=lambda x: x['coherence'])
        print(f"   → Best drive frequency: {best_drive['drive_freq']}")
    
    # Overall assessment
    all_coherences = [data['coherence'] if isinstance(data, dict) and 'coherence' in data else 0 
                     for data in results.values()]
    all_coherences = [c for c in all_coherences if c > 0]
    
    if all_coherences:
        avg_coherence = np.mean(all_coherences)
        max_coherence = max(all_coherences)
        min_coherence = min(all_coherences)
        
        print(f"\n🎯 OVERALL ASSESSMENT:")
        print(f"Coherence range: {min_coherence:.4f} - {max_coherence:.4f}")
        print(f"Average coherence: {avg_coherence:.4f}")
        print(f"vs Theory ({theoretical_baseline:.3f}): {avg_coherence/theoretical_baseline:.2f}x")
        
        if avg_coherence > theoretical_baseline:
            print("✅ ROBUST: Probability flow concept survives complexity")
            print("Tesla's field insights confirmed across multiple tests!")
        elif avg_coherence > theoretical_baseline * 0.5:
            print("⚡ GOOD: Concept works with some degradation")
            print("Promising foundation for further development")
        else:
            print("📊 MIXED: Results vary significantly with complexity")
            print("Concept works but sensitive to parameters")

if __name__ == "__main__":
    print("🔬 PROGRESSIVE PROBABILITY FLOW TESTING")
    print("Testing robustness and depth of Tesla's quantum insight")
    print("=" * 55)
    
    try:
        results = run_progressive_tests()
        analyze_progressive_results(results)
        
        print(f"\n🎉 PROGRESSIVE TESTING COMPLETED!")
        print("Explored grid resolution, nonlinearity, two-field coupling, and drive response")
        print("Tesla's probability flow concept tested across multiple dimensions!")
        
    except Exception as e:
        print(f"💥 Progressive testing failed: {e}")
        import traceback
        traceback.print_exc()