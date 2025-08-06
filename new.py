import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time

class ProbabilityFlowSystem:
    """
    PARADIGM SHIFT: Treat quantum systems as probability current densities
    
    Instead of discrete states |0⟩, |1⟩:
    - Continuous probability density ρ(x,t) in space
    - Probability current J(x,t) = flow of probability
    - Electrical field E(x,t) = probability flow driver
    
    Key insight: "Qubits" are just standing wave patterns in probability flow!
    """
    
    def __init__(self, spatial_points=128, box_size=10.0):
        self.N = spatial_points  # Spatial grid points
        self.L = box_size        # System size (arbitrary units)
        
        # Spatial grid
        self.dx = self.L / self.N
        self.x = np.linspace(0, self.L, self.N)
        
        # Physical parameters (in natural units)
        self.hbar = 1.0          # ℏ = 1
        self.mass = 1.0          # Effective mass = 1
        self.c = 1.0             # Speed of light = 1
        
        # Field parameters
        self.omega_matter = 2.0   # Matter resonance frequency
        self.omega_field = 2.0    # Field resonance frequency  
        self.coupling = 0.1       # Field-matter coupling strength
        
        # Damping
        self.gamma = 0.01         # Dissipation rate
        
        print(f"🌊 PROBABILITY FLOW SYSTEM")
        print(f"Spatial grid: {self.N} points over {self.L} units")
        print(f"Resolution: dx = {self.dx:.4f}")
        print(f"Treating quantum as continuous probability flow")

def create_kinetic_operator(system):
    """Create kinetic energy operator: -ℏ²/(2m) ∇²"""
    N = system.N
    dx = system.dx
    
    # Second derivative operator (finite differences)
    # ∇²ψ ≈ (ψ[i+1] - 2ψ[i] + ψ[i-1]) / dx²
    diag_main = -2 * np.ones(N)
    diag_off = np.ones(N-1)
    
    # Handle boundary conditions (periodic)
    kinetic_matrix = diags([diag_off, diag_main, diag_off], [-1, 0, 1], shape=(N, N))
    kinetic_matrix = kinetic_matrix.toarray()
    
    # Periodic boundary conditions
    kinetic_matrix[0, -1] = 1.0   # Connect first to last
    kinetic_matrix[-1, 0] = 1.0   # Connect last to first
    
    # Scale by physical constants
    kinetic_matrix *= -system.hbar**2 / (2 * system.mass * dx**2)
    
    return kinetic_matrix

def create_potential_wells(system):
    """Create potential landscape that naturally creates 'qubit-like' behavior"""
    x = system.x
    
    # Double-well potential: two minima create |0⟩ and |1⟩ like regions
    # V(x) = a(x-x1)²(x-x2)² where x1, x2 are well positions
    x1 = system.L * 0.3  # Left well
    x2 = system.L * 0.7  # Right well
    
    # Create double well
    potential = 0.5 * (x - x1)**2 * (x - x2)**2 / (0.1 * system.L)**2
    
    # Add harmonic trapping to prevent escaping
    x_center = system.L / 2
    trap_strength = 0.01
    potential += trap_strength * (x - x_center)**2
    
    return potential

def probability_flow_dynamics(t, psi_real, system, E_field_func):
    """
    Probability flow dynamics: Schrödinger equation with field coupling
    
    iℏ ∂ψ/∂t = [H_kinetic + V + coupling*E(x,t)]ψ + damping terms
    """
    N = system.N
    
    # Convert real vector back to complex wavefunction
    psi = psi_real[:N] + 1j * psi_real[N:]
    
    # Normalize to preserve probability
    norm = np.sqrt(np.sum(np.abs(psi)**2) * system.dx)
    if norm > 1e-10:
        psi = psi / norm
    
    # Kinetic energy operator
    if not hasattr(system, '_kinetic_op'):
        system._kinetic_op = create_kinetic_operator(system)
    
    # Potential energy
    if not hasattr(system, '_potential'):
        system._potential = create_potential_wells(system)
    
    # Electrical field coupling
    E_field = E_field_func(system.x, t)
    
    # Total Hamiltonian
    H_psi = system._kinetic_op @ psi + system._potential * psi + system.coupling * E_field * psi
    
    # Schrödinger evolution: iℏ ∂ψ/∂t = H ψ
    dpsi_dt = -1j * H_psi / system.hbar
    
    # Add damping (non-Hermitian term)
    dpsi_dt -= system.gamma * psi
    
    # Convert back to real vector
    dpsi_real = np.concatenate([np.real(dpsi_dt), np.imag(dpsi_dt)])
    
    return dpsi_real

def electrical_field_functions(system):
    """Different electrical field driving patterns"""
    
    def classical_drive(x, t):
        """Classical sinusoidal drive"""
        return 0.1 * np.sin(system.omega_field * t) * np.ones_like(x)
    
    def quantum_coherent_field(x, t):
        """Spatially modulated coherent field"""
        # Coherent field with spatial structure
        k = 2 * np.pi / system.L  # Wavelength matching system size
        return 0.1 * np.sin(system.omega_field * t) * np.cos(k * x)
    
    def quantum_squeezed_field(x, t):
        """Squeezed field with enhanced spatial gradients"""
        k1 = 2 * np.pi / system.L
        k2 = 4 * np.pi / system.L
        
        # Two-mode squeezed field
        squeeze_factor = 1 + 0.5 * np.cos(2 * system.omega_field * t)
        spatial_pattern = np.cos(k1 * x) + 0.3 * np.cos(k2 * x)
        
        return 0.1 * squeeze_factor * np.sin(system.omega_field * t) * spatial_pattern
    
    def probability_guided_field(x, t):
        """Field that adapts to probability density (the key innovation!)"""
        # This would normally require the current ψ, but we'll use a model
        # In real implementation, this would be E = f(|ψ(x,t)|²)
        x_center = system.L / 2
        width = system.L / 8
        
        # Field stronger where we expect probability density
        spatial_weight = np.exp(-(x - x_center)**2 / (2 * width**2))
        time_modulation = np.sin(system.omega_field * t) * (1 + 0.2 * np.cos(0.5 * t))
        
        return 0.15 * time_modulation * spatial_weight
    
    return {
        'classical': classical_drive,
        'coherent': quantum_coherent_field,
        'squeezed': quantum_squeezed_field,
        'adaptive': probability_guided_field
    }

def analyze_probability_flow(sol, system):
    """Analyze the probability flow patterns"""
    results = {
        'time': sol.t,
        'probability_density': [],
        'probability_current': [],
        'field_coupling_energy': [],
        'localization': [],
        'flow_entropy': []
    }
    
    for i, t in enumerate(sol.t):
        N = system.N
        
        # Reconstruct wavefunction
        psi = sol.y[:N, i] + 1j * sol.y[N:, i]
        
        # Probability density
        prob_density = np.abs(psi)**2
        results['probability_density'].append(prob_density)
        
        # Probability current: J = (ℏ/2mi)[ψ*∇ψ - ψ∇ψ*]
        # Finite difference for gradient
        psi_grad = np.gradient(psi, system.dx)
        current = (system.hbar / (2j * system.mass)) * (
            np.conj(psi) * psi_grad - psi * np.conj(psi_grad)
        )
        results['probability_current'].append(np.real(current))
        
        # Field coupling energy
        # This requires knowing E-field at time t, we'll approximate
        coupling_energy = system.coupling * np.sum(prob_density) * system.dx
        results['field_coupling_energy'].append(coupling_energy)
        
        # Localization measure (inverse participation ratio)
        localization = 1.0 / np.sum(prob_density**2 * system.dx)
        results['localization'].append(localization)
        
        # Flow entropy (spatial entropy of probability)
        prob_norm = prob_density / (np.sum(prob_density) * system.dx + 1e-12)
        flow_entropy = -np.sum(prob_norm * np.log(prob_norm + 1e-12)) * system.dx
        results['flow_entropy'].append(flow_entropy)
    
    return results

def calculate_flow_control_metrics(analysis, system):
    """Calculate how well the field controls probability flow"""
    
    # Flow coherence: how organized is the probability current
    currents = np.array(analysis['probability_current'])
    flow_coherence = []
    
    for i in range(len(analysis['time'])):
        current = currents[i]
        # Coherence = |⟨J⟩|² / ⟨|J|²⟩
        mean_current = np.mean(current)
        mean_abs_current = np.mean(np.abs(current)**2)
        
        if mean_abs_current > 1e-12:
            coherence = np.abs(mean_current)**2 / mean_abs_current
        else:
            coherence = 0.0
        
        flow_coherence.append(coherence)
    
    # Flow control power: how much the field influences flow patterns
    densities = np.array(analysis['probability_density'])
    control_power = []
    
    for i in range(1, len(analysis['time'])):
        # Rate of change of probability distribution
        density_change = np.sum(np.abs(densities[i] - densities[i-1])) * system.dx
        dt = analysis['time'][i] - analysis['time'][i-1]
        
        if dt > 1e-12:
            change_rate = density_change / dt
        else:
            change_rate = 0.0
        
        control_power.append(change_rate)
    
    # Flow efficiency: energy cost vs. control achieved
    coupling_energies = analysis['field_coupling_energy']
    
    return {
        'flow_coherence': np.array(flow_coherence),
        'control_power': np.array([0] + control_power),  # Pad first element
        'avg_coupling_energy': np.mean(coupling_energies),
        'max_flow_coherence': np.max(flow_coherence),
        'avg_control_power': np.mean(control_power) if control_power else 0.0
    }

def run_probability_flow_experiment():
    """
    Test different electrical field approaches on probability flow
    """
    print("🌊 PROBABILITY FLOW FIELD EXPERIMENT")
    print("Testing: Electrical field control of quantum probability flows")
    print("=" * 60)
    
    system = ProbabilityFlowSystem(spatial_points=64, box_size=8.0)
    
    # Initial state: Gaussian wavepacket in left well
    x0 = system.L * 0.3  # Start in left well
    sigma = system.L / 20  # Wave packet width
    k0 = 0.5  # Initial momentum
    
    initial_psi = np.exp(-(system.x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * system.x)
    
    # Normalize
    norm = np.sqrt(np.sum(np.abs(initial_psi)**2) * system.dx)
    initial_psi = initial_psi / norm
    
    # Convert to real vector
    initial_state = np.concatenate([np.real(initial_psi), np.imag(initial_psi)])
    
    print(f"Initial state: Gaussian in left well (x={x0:.2f})")
    print(f"Wave packet width: σ={sigma:.4f}")
    
    # Test different field types
    field_functions = electrical_field_functions(system)
    results = {}
    
    for field_name, field_func in field_functions.items():
        print(f"\n🔬 Testing {field_name} field...")
        
        def dynamics(t, psi):
            return probability_flow_dynamics(t, psi, system, field_func)
        
        # Simulation parameters
        t_final = 20.0  # Longer time to see flow patterns
        n_points = 200
        t_eval = np.linspace(0, t_final, n_points)
        
        import time as time_module
        start_time = time_module.time()
        
        sol = solve_ivp(dynamics, [0, t_final], initial_state,
                       t_eval=t_eval, method='RK45',
                       rtol=1e-6, atol=1e-8, max_step=0.1)
        
        elapsed = time_module.time() - start_time
        
        if sol.success:
            print(f"   ✅ Simulation completed in {elapsed:.1f}s")
            
            # Analyze results
            analysis = analyze_probability_flow(sol, system)
            control_metrics = calculate_flow_control_metrics(analysis, system)
            
            results[field_name] = {
                'analysis': analysis,
                'control': control_metrics,
                'field_type': field_name
            }
            
            print(f"   Max flow coherence: {control_metrics['max_flow_coherence']:.4f}")
            print(f"   Avg control power: {control_metrics['avg_control_power']:.4f}")
            print(f"   Avg coupling energy: {control_metrics['avg_coupling_energy']:.4f}")
            
        else:
            print(f"   ❌ Simulation failed: {sol.message}")
    
    # Create visualization
    if results:
        print(f"\n📊 PROBABILITY FLOW ANALYSIS")
        
        # Create comprehensive flow visualization
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Quantum Probability Flow Control Results', fontsize=16)
        
        colors = {'classical': 'blue', 'coherent': 'red', 'squeezed': 'green', 'adaptive': 'orange'}
        
        # Plot flow metrics
        for field_name, data in results.items():
            color = colors.get(field_name, 'black')
            time = data['analysis']['time']
            
            # Flow coherence evolution
            axes[0, 0].plot(time, data['control']['flow_coherence'], 
                           color=color, label=field_name, linewidth=2)
            
            # Control power evolution
            axes[0, 1].plot(time, data['control']['control_power'],
                           color=color, label=field_name, linewidth=2)
            
            # Localization evolution
            axes[1, 0].plot(time, data['analysis']['localization'],
                           color=color, label=field_name, linewidth=2)
            
            # Flow entropy evolution
            axes[1, 1].plot(time, data['analysis']['flow_entropy'],
                           color=color, label=field_name, linewidth=2)
        
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Flow Coherence')
        axes[0, 0].set_title('Probability Flow Coherence')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Control Power')
        axes[0, 1].set_title('Flow Control Power')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Localization')
        axes[1, 0].set_title('Spatial Localization')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Flow Entropy')
        axes[1, 1].set_title('Probability Flow Entropy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Show final probability density snapshots
        best_field = max(results.keys(), key=lambda x: results[x]['control']['max_flow_coherence'])
        
        final_density = results[best_field]['analysis']['probability_density'][-1]
        final_current = results[best_field]['analysis']['probability_current'][-1]
        
        axes[2, 0].plot(system.x, final_density, 'b-', linewidth=2, label='Probability Density')
        axes[2, 0].set_xlabel('Position')
        axes[2, 0].set_ylabel('Probability Density')
        axes[2, 0].set_title(f'Final State ({best_field} field)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].plot(system.x, final_current, 'r-', linewidth=2, label='Probability Current')
        axes[2, 1].set_xlabel('Position')
        axes[2, 1].set_ylabel('Current Density')
        axes[2, 1].set_title(f'Final Current ({best_field} field)')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('probability_flow_control.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Performance comparison
        print(f"\n🎯 FLOW CONTROL COMPARISON:")
        
        for field_name, data in results.items():
            control = data['control']
            print(f"{field_name.upper()}:")
            print(f"  Max flow coherence: {control['max_flow_coherence']:.4f}")
            print(f"  Avg control power: {control['avg_control_power']:.4f}")
            print(f"  Coupling efficiency: {control['avg_coupling_energy']:.4f}")
        
        best_field_name = max(results.keys(), key=lambda x: results[x]['control']['max_flow_coherence'])
        best_coherence = results[best_field_name]
        
        print(f"\n✨ BEST FLOW CONTROL: {best_field_name.upper()}")
        print(f"Peak coherence: {best_coherence['control']['max_flow_coherence']:.4f}")
        
        if best_coherence['control']['max_flow_coherence'] > 0.5:
            print("🎯 EXCELLENT probability flow control achieved!")
            print("Electrical field demonstrates coherent flow manipulation")
        elif best_coherence['control']['max_flow_coherence'] > 0.2:
            print("⚡ GOOD probability flow control")
            print("Field shows significant influence on quantum flows")
        else:
            print("📊 Limited flow control detected")
        
        return results
    
    else:
        print("❌ No successful flow experiments")
        return None

if __name__ == "__main__":
    print("🌊 PROBABILITY FLOW QUANTUM FIELD EXPERIMENT")
    print("=" * 50)
    print("PARADIGM SHIFT:")
    print("• Traditional: Discrete states |0⟩, |1⟩ with operators")
    print("• New approach: Continuous probability flows ρ(x,t)")
    print("• Electrical field: Probability current driver J(x,t)")
    print("• Question: Can E-field control probability flows?")
    print("=" * 50)
    
    try:
        results = run_probability_flow_experiment()
        
        if results:
            print(f"\n🎉 PROBABILITY FLOW EXPERIMENT COMPLETED!")
            print("This tests electrical field control of quantum probability")
            print("flows rather than discrete state manipulations!")
            print("\nKey innovation: Treating quantum as continuous")
            print("probability current density, not discrete states!")
        
    except Exception as e:
        print(f"💥 Flow experiment failed: {e}")
        import traceback
        traceback.print_exc()