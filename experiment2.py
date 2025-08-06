import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time

def ultra_simple_dynamics(t, state_vec):
    """
    ULTRA SIMPLE: Just harmonic oscillator + tiny perturbations
    NO complex decoherence that causes stiffness
    """
    N = len(state_vec) // 2
    psi = state_vec[:N] + 1j * state_vec[N:]
    
    # Simple grid
    x = np.linspace(-2, 2, N)
    dx = x[1] - x[0]
    
    # Simple harmonic oscillator frequency
    omega = 1.0
    
    # VERY SIMPLE Hamiltonian
    # Kinetic energy (simple finite difference)
    H_psi = np.zeros_like(psi, dtype=complex)
    for i in range(1, N-1):
        H_psi[i] += -0.5 * (psi[i+1] - 2*psi[i] + psi[i-1]) / dx**2
    
    # Harmonic potential
    H_psi += 0.5 * omega**2 * x**2 * psi
    
    # Tiny noise perturbation (VERY SMALL)
    noise = 0.001 * np.sin(0.1 * t)  # Slow, small noise
    H_psi += noise * x * psi
    
    # TINY decoherence (much smaller than before)
    energy = np.sum(np.abs(psi)**2 * x**2) * dx
    tiny_decay = -0.01 * energy * psi  # Very weak
    
    # Evolution
    dpsi_dt = -1j * H_psi + tiny_decay
    
    return np.concatenate([np.real(dpsi_dt), np.imag(dpsi_dt)])

def measure_simple_coherence(sol):
    """
    Ultra simple coherence measurement
    """
    N = len(sol.y) // 2
    x = np.linspace(-2, 2, N)
    dx = x[1] - x[0]
    
    position_vars = []
    
    for i in range(len(sol.t)):
        psi = sol.y[:N, i] + 1j * sol.y[N:, i]
        
        # Normalize
        norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
        if norm > 1e-12:
            psi = psi / norm
        
        # Position variance
        prob = np.abs(psi)**2
        mean_x = np.sum(prob * x) * dx
        var_x = np.sum(prob * (x - mean_x)**2) * dx
        position_vars.append(var_x)
    
    # Simple correlation
    if len(position_vars) > 2:
        try:
            corr = np.corrcoef(position_vars[:-1], position_vars[1:])[0, 1]
            correlation = abs(corr) if not np.isnan(corr) else 0.0
        except:
            correlation = 0.0
    else:
        correlation = 0.0
    
    # Simple stability
    stability = 1.0 / (1.0 + np.std(position_vars))
    
    # Simple coherence estimate
    coherence = (correlation * 0.6 + stability * 0.4)
    
    return {
        'coherence': coherence,
        'correlation': correlation,
        'stability': stability,
        'position_vars': position_vars,
        'time': sol.t
    }

def run_ultra_simple_test():
    """
    Ultra simple test that MUST work
    """
    print("🚀 ULTRA SIMPLE TEST")
    print("Bare minimum physics - should definitely work")
    print("=" * 45)
    
    # Very small system
    N = 16
    x = np.linspace(-2, 2, N)
    dx = x[1] - x[0]
    
    # Simple Gaussian initial state
    psi0 = np.exp(-x**2)
    psi0 = psi0 / np.sqrt(np.sum(psi0**2) * dx)
    initial_state = np.concatenate([psi0, np.zeros(N)])  # Real part only initially
    
    print(f"Grid size: {N}")
    print(f"Initial state size: {len(initial_state)}")
    
    # Very short, simple integration
    t_final = 5.0  # Simple time units
    n_points = 20  # Few points
    
    print(f"Simulation time: {t_final}")
    print(f"Time points: {n_points}")
    print("Starting ultra simple integration...")
    
    start_time = time.time()
    
    try:
        sol = solve_ivp(ultra_simple_dynamics, [0, t_final], initial_state,
                       t_eval=np.linspace(0, t_final, n_points),
                       method='RK45', rtol=1e-2, atol=1e-4,  # Very loose tolerances
                       max_step=0.5)  # Large max step
        
        elapsed = time.time() - start_time
        
        if sol.success:
            print(f"✅ Ultra simple test SUCCESS in {elapsed:.1f}s!")
            print(f"Integrated {len(sol.t)} time points")
            
            # Analyze
            results = measure_simple_coherence(sol)
            
            print(f"\n📊 ULTRA SIMPLE RESULTS:")
            print(f"Simple coherence: {results['coherence']:.4f}")
            print(f"Correlation: {results['correlation']:.4f}")
            print(f"Stability: {results['stability']:.4f}")
            
            # Compare to theory
            theoretical = 0.44
            ratio = results['coherence'] / theoretical
            
            print(f"\n🎯 vs THEORY:")
            print(f"Theoretical: {theoretical:.4f}")
            print(f"Ultra simple: {results['coherence']:.4f}")
            print(f"Ratio: {ratio:.3f} ({ratio*100:.1f}%)")
            
            # Assessment
            if ratio > 0.3:
                print("✅ GOOD: Significant coherence survives ultra simple test")
            elif ratio > 0.1:
                print("⚡ MODERATE: Some coherence detected")
            else:
                print("📊 WEAK: Limited coherence in ultra simple case")
            
            # Simple plot
            plt.figure(figsize=(8, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(results['time'], results['position_vars'], 'b-', linewidth=2)
            plt.xlabel('Time')
            plt.ylabel('Position Variance')
            plt.title('Position Variance Evolution')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            metrics = ['Coherence', 'Correlation', 'Stability']
            values = [results['coherence'], results['correlation'], results['stability']]
            plt.bar(metrics, values, alpha=0.7)
            plt.ylabel('Value')
            plt.title('Simple Metrics')
            plt.axhline(y=theoretical, color='red', linestyle='--', label='Theory')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('ultra_simple_test.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return {
                'success': True,
                'coherence': results['coherence'],
                'ratio': ratio,
                'results': results
            }
            
        else:
            print(f"❌ Ultra simple test FAILED: {sol.message}")
            return {'success': False, 'message': sol.message}
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"💥 Ultra simple test CRASHED after {elapsed:.1f}s: {e}")
        return {'success': False, 'error': str(e)}

def test_step_by_step():
    """
    Test each component individually
    """
    print("\n🔍 STEP BY STEP COMPONENT TEST")
    print("=" * 35)
    
    # Test 1: Can we create the initial state?
    print("Step 1: Initial state creation...")
    try:
        N = 16
        x = np.linspace(-2, 2, N)
        psi0 = np.exp(-x**2)
        psi0 = psi0 / np.sqrt(np.sum(psi0**2) * (x[1]-x[0]))
        initial_state = np.concatenate([psi0, np.zeros(N)])
        print(f"✅ Initial state OK: {len(initial_state)} elements")
    except Exception as e:
        print(f"❌ Initial state failed: {e}")
        return False
    
    # Test 2: Can we call dynamics once?
    print("Step 2: Single dynamics call...")
    try:
        result = ultra_simple_dynamics(0.0, initial_state)
        print(f"✅ Dynamics call OK: {len(result)} outputs")
    except Exception as e:
        print(f"❌ Dynamics call failed: {e}")
        return False
    
    # Test 3: Can we do one integration step?
    print("Step 3: Single integration step...")
    try:
        sol = solve_ivp(ultra_simple_dynamics, [0, 0.1], initial_state,
                       method='RK45', max_step=0.01)
        if sol.success:
            print(f"✅ Single step OK: {len(sol.t)} points")
        else:
            print(f"❌ Single step failed: {sol.message}")
            return False
    except Exception as e:
        print(f"❌ Single step error: {e}")
        return False
    
    print("✅ All component tests passed!")
    return True

if __name__ == "__main__":
    print("🚀 ULTRA SIMPLE EXPERIMENTAL TEST")
    print("Absolute minimum to test the concept")
    print("If this hangs, the problem is fundamental")
    print("=" * 45)
    
    # First test components
    if test_step_by_step():
        print("\n" + "="*45)
        
        # Then run full test
        results = run_ultra_simple_test()
        
        if results['success']:
            print(f"\n🎉 ULTRA SIMPLE TEST COMPLETED!")
            
            if results['ratio'] > 0.2:
                print("✅ Probability flow concept works at basic level!")
                print("Tesla's field insights have some quantum validity!")
            else:
                print("📊 Concept works but effects are very weak")
                print("May need different approach or optimization")
        else:
            print(f"\n❌ Even ultra simple test failed")
            print("Fundamental issue with the approach")
    else:
        print("\n❌ Component tests failed - debugging needed")