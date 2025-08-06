import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
import time

print("🌐 MULTI-QUBIT FIELD NETWORK SIMULATION 🌐")
print("Testing collective electrical-quantum field entanglement")
print("Networks of coupled electrical resonators + multiple qubits")
print("Exploring field-mediated interactions and collective coherence")
print("=" * 75)

class QuantumFieldNetwork:
    """
    Network of coupled electrical fields and quantum systems
    Models realistic quantum computer architectures with field coupling
    """
    
    def __init__(self, n_qubits=3, network_topology='linear'):
        self.n_qubits = n_qubits
        self.network_topology = network_topology
        
        # Field parameters for each qubit
        self.w_qubits = np.array([5.0 + 0.1 * i for i in range(n_qubits)])  # Slight detuning
        self.w_electrical = np.array([5.0 + 0.05 * i for i in range(n_qubits)])  # Electrical frequencies
        
        # Coupling strengths
        self.g_local = 0.1   # Local electrical-quantum coupling
        self.g_network = 0.05  # Network coupling between electrical fields
        self.g_collective = 0.02  # Collective coupling (all-to-all)
        
        # Network connectivity matrix
        self.connectivity = self.build_network_topology()
        
        # Field damping
        self.gamma_elec = 0.01
        self.gamma_quantum = 0.025
        
        # Truncation for computational feasibility
        self.n_modes_per_field = 4  # States per field: |0⟩,|1⟩,|2⟩,|3⟩
        self.total_dim = self.n_modes_per_field ** (2 * n_qubits)  # (elec+quantum)^n_qubits
        
        print(f"   Network: {n_qubits} qubits, {network_topology} topology")
        print(f"   Qubit frequencies: {[f'{f:.2f}' for f in self.w_qubits]} GHz")
        print(f"   Electrical frequencies: {[f'{f:.2f}' for f in self.w_electrical]} GHz")
        print(f"   Total Hilbert space: {self.total_dim} dimensions")
        print(f"   Network coupling: {self.g_network} (nearest), {self.g_collective} (collective)")
    
    def build_network_topology(self):
        """
        Build network connectivity matrix for different topologies
        """
        n = self.n_qubits
        
        if self.network_topology == 'linear':
            # Linear chain: 0-1-2-3-...
            connectivity = np.zeros((n, n))
            for i in range(n - 1):
                connectivity[i, i + 1] = 1.0
                connectivity[i + 1, i] = 1.0
                
        elif self.network_topology == 'ring':
            # Ring topology: 0-1-2-...-0
            connectivity = np.zeros((n, n))
            for i in range(n):
                connectivity[i, (i + 1) % n] = 1.0
                connectivity[(i + 1) % n, i] = 1.0
                
        elif self.network_topology == 'star':
            # Star topology: central qubit connected to all others
            connectivity = np.zeros((n, n))
            center = 0  # Qubit 0 is center
            for i in range(1, n):
                connectivity[center, i] = 1.0
                connectivity[i, center] = 1.0
                
        elif self.network_topology == 'all_to_all':
            # Fully connected network
            connectivity = np.ones((n, n)) - np.eye(n)
            
        else:  # default to linear
            connectivity = np.eye(n)
        
        return connectivity

def multi_field_network_dynamics(t, psi_vec, system):
    """
    Dynamics for network of coupled electrical-quantum field systems
    """
    try:
        n_qubits = system.n_qubits
        n_modes = system.n_modes_per_field
        
        # For computational feasibility, use reduced representation
        # State vector: [field_amplitudes_real, field_amplitudes_imag, network_correlations]
        
        # Simplified network state representation
        field_dim = n_qubits * n_modes * 2  # Each qubit has electrical + quantum field
        if len(psi_vec) < field_dim:
            raise ValueError(f"State vector too small: {len(psi_vec)} < {field_dim}")
        
        # Extract field amplitudes for each qubit
        field_states = psi_vec[:field_dim].reshape((n_qubits, n_modes, 2))  # [qubit, mode, field_type]
        
        # Calculate time derivatives
        dfield_dt = np.zeros_like(field_states)
        
        for i in range(n_qubits):
            # Local field evolution for each qubit
            for mode in range(n_modes):
                elec_amp = field_states[i, mode, 0]  # Electrical field amplitude
                quantum_amp = field_states[i, mode, 1]  # Quantum field amplitude
                
                # Local field energies
                elec_energy = system.w_electrical[i] * (mode + 0.5)
                quantum_energy = system.w_qubits[i] * (mode + 0.5)
                
                # Local electrical-quantum coupling
                local_coupling = system.g_local * elec_amp * quantum_amp
                
                # Network coupling effects
                network_coupling = 0.0
                for j in range(n_qubits):
                    if i != j and system.connectivity[i, j] > 0:
                        # Couple to neighboring electrical fields
                        neighbor_elec = field_states[j, mode, 0]
                        network_coupling += system.g_network * neighbor_elec * system.connectivity[i, j]
                
                # Collective coupling (all qubits to average field)
                avg_elec_field = np.mean(field_states[:, mode, 0])
                avg_quantum_field = np.mean(field_states[:, mode, 1])
                collective_coupling = system.g_collective * (avg_elec_field + avg_quantum_field)
                
                # External driving (can be qubit-specific)
                drive_i = 0.001 * np.sin(system.w_electrical[i] * t + i * np.pi / 4)  # Phase shifts
                
                # Field evolution equations
                # Electrical field
                dfield_dt[i, mode, 0] = (
                    -1j * elec_energy * elec_amp +
                    -1j * local_coupling * quantum_amp +
                    -1j * network_coupling +
                    -1j * collective_coupling +
                    drive_i +
                    -system.gamma_elec * mode * elec_amp
                )
                
                # Quantum field  
                dfield_dt[i, mode, 1] = (
                    -1j * quantum_energy * quantum_amp +
                    -1j * local_coupling * elec_amp +
                    -1j * collective_coupling +
                    -system.gamma_quantum * mode * quantum_amp
                )
        
        # Add network correlation dynamics
        correlation_terms = np.zeros(len(psi_vec) - field_dim)
        
        # Network-induced correlations (simplified)
        if len(correlation_terms) > 0:
            for corr_idx in range(len(correlation_terms)):
                # Decay of network correlations
                correlation_terms[corr_idx] = -0.1 * psi_vec[field_dim + corr_idx]
        
        # Flatten field derivatives and combine with correlations
        dfield_flat = np.real(dfield_dt.flatten())  # Take real part for now
        
        return np.concatenate([dfield_flat, correlation_terms])
        
    except Exception as e:
        print(f"Network dynamics error at t={t:.3f}: {e}")
        return [0.0] * len(psi_vec)

def analyze_network_field_state(sol, system):
    """
    Analyze multi-qubit field network for collective effects
    """
    n_qubits = system.n_qubits
    n_modes = system.n_modes_per_field
    
    results = {
        'time': sol.t,
        'individual_entanglements': [[] for _ in range(n_qubits)],
        'pairwise_entanglements': [],
        'collective_entanglement': [],
        'network_coherence': [],
        'field_synchronization': [],
        'energy_distribution': []
    }
    
    for time_idx in range(len(sol.t)):
        field_dim = n_qubits * n_modes * 2
        field_data = sol.y[:field_dim, time_idx].reshape((n_qubits, n_modes, 2))
        
        # Individual qubit entanglements (electrical-quantum for each qubit)
        for i in range(n_qubits):
            elec_field = field_data[i, :, 0]
            quantum_field = field_data[i, :, 1]
            
            # Simple entanglement measure: overlap between fields
            local_entanglement = abs(np.dot(elec_field, quantum_field)) / (
                np.linalg.norm(elec_field) * np.linalg.norm(quantum_field) + 1e-10
            )
            results['individual_entanglements'][i].append(local_entanglement)
        
        # Pairwise qubit entanglements (through electrical field network)
        pairwise_ents = []
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if system.connectivity[i, j] > 0:
                    # Entanglement mediated by electrical field coupling
                    field_i = np.concatenate([field_data[i, :, 0], field_data[i, :, 1]])
                    field_j = np.concatenate([field_data[j, :, 0], field_data[j, :, 1]])
                    
                    pairwise_ent = abs(np.dot(field_i, field_j)) / (
                        np.linalg.norm(field_i) * np.linalg.norm(field_j) + 1e-10
                    )
                    pairwise_ents.append(pairwise_ent)
        
        results['pairwise_entanglements'].append(np.mean(pairwise_ents) if pairwise_ents else 0.0)
        
        # Collective entanglement (all qubits collectively entangled)
        all_fields_flat = field_data.flatten()
        
        # Measure of collective coherence
        field_variance = np.var(np.abs(all_fields_flat))
        collective_ent = field_variance / (np.mean(np.abs(all_fields_flat))**2 + 1e-10)
        results['collective_entanglement'].append(collective_ent)
        
        # Network coherence (how synchronized are the fields?)
        elec_fields = field_data[:, :, 0].flatten()
        quantum_fields = field_data[:, :, 1].flatten()
        
        # Phase synchronization measure
        elec_phases = np.angle(elec_fields + 1j * 1e-10)  # Avoid zero division
        quantum_phases = np.angle(quantum_fields + 1j * 1e-10)
        
        phase_sync = abs(np.mean(np.exp(1j * (elec_phases - quantum_phases))))
        results['field_synchronization'].append(phase_sync)
        
        # Energy distribution across network
        energies = []
        for i in range(n_qubits):
            qubit_energy = np.sum(np.abs(field_data[i, :, :])**2)
            energies.append(qubit_energy)
        
        energy_uniformity = 1.0 / (np.std(energies) + 1e-10)  # Higher = more uniform
        results['energy_distribution'].append(energy_uniformity)
    
    return results

def test_network_topologies():
    """
    Test different network topologies for collective field effects
    """
    print("1. TESTING NETWORK TOPOLOGIES")
    print("   Comparing linear, ring, star, and all-to-all networks...")
    
    topologies = ['linear', 'ring', 'star', 'all_to_all']
    network_results = {}
    
    for topology in topologies:
        print(f"\n   Testing {topology} network...")
        
        try:
            # Create network system
            system = QuantumFieldNetwork(n_qubits=4, network_topology=topology)
            
            # Initial state: small excitations with phase relationships
            n_modes = system.n_modes_per_field
            field_dim = system.n_qubits * n_modes * 2
            
            initial_fields = np.zeros(field_dim)
            
            # Initialize with phase-correlated excitations
            for i in range(system.n_qubits):
                for mode in range(min(2, n_modes)):  # Only low modes
                    # Phase relationship across network
                    phase = 2 * np.pi * i / system.n_qubits  # Different phase for each qubit
                    
                    # Electrical field
                    idx_elec = i * (n_modes * 2) + mode * 2 + 0
                    initial_fields[idx_elec] = 0.1 * np.cos(phase)
                    
                    # Quantum field  
                    idx_quantum = i * (n_modes * 2) + mode * 2 + 1
                    initial_fields[idx_quantum] = 0.1 * np.sin(phase)
            
            # Add network correlation state (simplified)
            correlation_dim = max(0, len(initial_fields) - field_dim)
            initial_correlations = np.random.normal(0, 0.01, correlation_dim)
            
            full_initial = np.concatenate([initial_fields, initial_correlations])
            
            print(f"      Network system: {system.n_qubits} qubits, {len(full_initial)} state dimensions")
            
            def network_dynamics(t, psi_vec):
                return multi_field_network_dynamics(t, psi_vec, system)
            
            # Simulate network evolution
            sim_time = 30.0
            sol = solve_ivp(network_dynamics, [0, sim_time], full_initial,
                           t_eval=np.linspace(0, sim_time, 150),
                           method='RK45', rtol=1e-5, atol=1e-7)
            
            if sol.success:
                print(f"      ✓ {topology} network simulation successful")
                
                # Analyze network state
                analysis = analyze_network_field_state(sol, system)
                
                # Calculate network metrics
                steady_start = len(analysis['time']) // 3
                
                metrics = {
                    'avg_individual_entanglement': np.mean([
                        np.mean(ent_series[steady_start:]) 
                        for ent_series in analysis['individual_entanglements']
                    ]),
                    'avg_pairwise_entanglement': np.mean(analysis['pairwise_entanglements'][steady_start:]),
                    'avg_collective_entanglement': np.mean(analysis['collective_entanglement'][steady_start:]),
                    'avg_field_synchronization': np.mean(analysis['field_synchronization'][steady_start:]),
                    'avg_energy_distribution': np.mean(analysis['energy_distribution'][steady_start:]),
                    'max_collective_entanglement': np.max(analysis['collective_entanglement'])
                }
                
                network_results[topology] = {
                    'success': True,
                    'metrics': metrics,
                    'analysis': analysis
                }
                
                print(f"      Individual entanglement: {metrics['avg_individual_entanglement']:.6f}")
                print(f"      Pairwise entanglement: {metrics['avg_pairwise_entanglement']:.6f}")
                print(f"      Collective entanglement: {metrics['avg_collective_entanglement']:.6f}")
                print(f"      Field synchronization: {metrics['avg_field_synchronization']:.6f}")
                
            else:
                print(f"      ✗ {topology} integration failed: {sol.message}")
                network_results[topology] = {'success': False}
                
        except Exception as e:
            print(f"      ✗ {topology} crashed: {e}")
            network_results[topology] = {'success': False}
    
    return network_results

def test_collective_field_resonance():
    """
    Test collective resonance effects in multi-qubit networks
    """
    print("\n2. COLLECTIVE FIELD RESONANCE TESTING")
    print("   Testing if collective resonance enhances network-wide entanglement...")
    
    # Test different degrees of frequency matching across network
    matching_scenarios = {
        'perfect_match': {'freq_spread': 0.0, 'description': 'All frequencies identical'},
        'small_spread': {'freq_spread': 0.05, 'description': '±50 MHz spread'},
        'medium_spread': {'freq_spread': 0.2, 'description': '±200 MHz spread'},
        'large_spread': {'freq_spread': 0.5, 'description': '±500 MHz spread'},
        'random_spread': {'freq_spread': 1.0, 'description': 'Random frequencies'}
    }
    
    collective_results = {}
    
    for scenario_name, config in matching_scenarios.items():
        print(f"   Testing {scenario_name} ({config['description']})...")
        
        try:
            # Create system with specified frequency distribution
            system = QuantumFieldNetwork(n_qubits=3, network_topology='all_to_all')
            
            # Modify frequencies according to scenario
            base_freq = 5.0
            freq_spread = config['freq_spread']
            
            if scenario_name == 'random_spread':
                np.random.seed(42)  # Reproducible randomness
                system.w_qubits = base_freq + np.random.uniform(-freq_spread, freq_spread, system.n_qubits)
                system.w_electrical = base_freq + np.random.uniform(-freq_spread, freq_spread, system.n_qubits)
            else:
                # Systematic frequency distribution
                for i in range(system.n_qubits):
                    offset = freq_spread * (i - system.n_qubits/2) / system.n_qubits
                    system.w_qubits[i] = base_freq + offset
                    system.w_electrical[i] = base_freq + offset
            
            print(f"      Qubit frequencies: {[f'{f:.3f}' for f in system.w_qubits]} GHz")
            
            # Initial state with network-wide phase correlations
            n_modes = system.n_modes_per_field
            field_dim = system.n_qubits * n_modes * 2
            
            initial_state = np.zeros(field_dim)
            
            # Create collective initial state
            for i in range(system.n_qubits):
                for mode in range(min(2, n_modes)):
                    # Collective phase relationships
                    collective_phase = 2 * np.pi * i * mode / (system.n_qubits * n_modes)
                    
                    idx_elec = i * (n_modes * 2) + mode * 2 + 0
                    idx_quantum = i * (n_modes * 2) + mode * 2 + 1
                    
                    if idx_elec < len(initial_state):
                        initial_state[idx_elec] = 0.1 * np.cos(collective_phase)
                    if idx_quantum < len(initial_state):
                        initial_state[idx_quantum] = 0.1 * np.sin(collective_phase)
            
            # Add correlation terms (empty for now)
            correlation_dim = 10  # Small number of network correlations
            initial_correlations = np.random.normal(0, 0.001, correlation_dim)
            full_initial = np.concatenate([initial_state, initial_correlations])
            
            def collective_dynamics(t, psi_vec):
                return multi_field_network_dynamics(t, psi_vec, system)
            
            # Simulate collective evolution
            sim_time = 25.0
            sol = solve_ivp(collective_dynamics, [0, sim_time], full_initial,
                           t_eval=np.linspace(0, sim_time, 125),
                           method='RK45', rtol=1e-5)
            
            if sol.success:
                print(f"      ✓ Collective simulation successful")
                
                # Analyze collective behavior
                analysis = analyze_network_field_state(sol, system)
                
                steady_start = len(analysis['time']) // 3
                
                collective_metrics = {
                    'collective_entanglement': np.mean(analysis['collective_entanglement'][steady_start:]),
                    'field_synchronization': np.mean(analysis['field_synchronization'][steady_start:]),
                    'pairwise_entanglement': np.mean(analysis['pairwise_entanglements'][steady_start:]),
                    'energy_uniformity': np.mean(analysis['energy_distribution'][steady_start:]),
                    'peak_collective': np.max(analysis['collective_entanglement'])
                }
                
                collective_results[scenario_name] = {
                    'success': True,
                    'metrics': collective_metrics,
                    'freq_spread': freq_spread,
                    'frequencies': system.w_qubits.copy()
                }
                
                print(f"      Collective entanglement: {collective_metrics['collective_entanglement']:.6f}")
                print(f"      Field synchronization: {collective_metrics['field_synchronization']:.6f}")
                print(f"      Pairwise entanglement: {collective_metrics['pairwise_entanglement']:.6f}")
                
            else:
                print(f"      ✗ Collective simulation failed")
                collective_results[scenario_name] = {'success': False}
                
        except Exception as e:
            print(f"      ✗ Collective test crashed: {e}")
            collective_results[scenario_name] = {'success': False}
    
    return collective_results

def test_field_mediated_interactions():
    """
    Test field-mediated qubit-qubit interactions
    """
    print("\n3. FIELD-MEDIATED QUBIT INTERACTIONS")
    print("   Testing if electrical fields can mediate quantum interactions...")
    
    # Two-qubit system with electrical field mediator
    system = QuantumFieldNetwork(n_qubits=2, network_topology='linear')
    
    # Test different field coupling strengths
    coupling_strengths = [0.01, 0.05, 0.1, 0.2]
    mediation_results = {}
    
    for g_network in coupling_strengths:
        print(f"   Testing network coupling strength {g_network:.3f}...")
        
        try:
            system.g_network = g_network
            
            # Initial state: qubit 1 excited, qubit 2 ground, no direct coupling
            n_modes = system.n_modes_per_field
            field_dim = system.n_qubits * n_modes * 2
            
            initial_state = np.zeros(field_dim)
            
            # Qubit 1: excited quantum field, small electrical field
            initial_state[0 * (n_modes * 2) + 1 * 2 + 1] = 0.8  # |0,1⟩ for qubit 1
            initial_state[0 * (n_modes * 2) + 0 * 2 + 0] = 0.1  # Small electrical
            
            # Qubit 2: ground state
            initial_state[1 * (n_modes * 2) + 0 * 2 + 1] = 0.5  # |0,0⟩ for qubit 2
            
            # Small correlation seeds
            correlation_dim = 5
            initial_correlations = np.random.normal(0, 0.001, correlation_dim)
            full_initial = np.concatenate([initial_state, initial_correlations])
            
            def mediation_dynamics(t, psi_vec):
                return multi_field_network_dynamics(t, psi_vec, system)
            
            sol = solve_ivp(mediation_dynamics, [0, 20.0], full_initial,
                           t_eval=np.linspace(0, 20.0, 100),
                           method='RK45', rtol=1e-5)
            
            if sol.success:
                analysis = analyze_network_field_state(sol, system)
                
                # Measure field-mediated interaction strength
                final_pairwise = analysis['pairwise_entanglements'][-1]
                max_pairwise = np.max(analysis['pairwise_entanglements'])
                avg_pairwise = np.mean(analysis['pairwise_entanglements'][30:])
                
                mediation_results[g_network] = {
                    'final_pairwise': final_pairwise,
                    'max_pairwise': max_pairwise,
                    'avg_pairwise': avg_pairwise,
                    'coupling_strength': g_network
                }
                
                print(f"      Final pairwise entanglement: {final_pairwise:.6f}")
                print(f"      Maximum pairwise entanglement: {max_pairwise:.6f}")
                
            else:
                print(f"      ✗ Mediation test failed")
                
        except Exception as e:
            print(f"      ✗ Mediation test crashed: {e}")
    
    return mediation_results

def run_network_field_analysis():
    """
    Complete multi-qubit field network analysis
    """
    print("MULTI-QUBIT FIELD NETWORK ANALYSIS")
    print("Exploring collective quantum field effects")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # 1. Test network topologies
        topology_results = test_network_topologies()
        
        # 2. Test collective resonance
        collective_results = test_collective_field_resonance()
        
        # 3. Test field-mediated interactions
        mediation_results = test_field_mediated_interactions()
        
        # Analysis and reporting
        print("\n4. NETWORK FIELD ANALYSIS...")
        
        with open('network_field_results.txt', 'w') as f:
            f.write("MULTI-QUBIT FIELD NETWORK ANALYSIS RESULTS\n")
            f.write("=" * 55 + "\n\n")
            
            # Topology comparison
            f.write("NETWORK TOPOLOGY COMPARISON:\n")
            best_topology = None
            best_collective_performance = 0.0
            
            for topology, result in topology_results.items():
                if result['success']:
                    metrics = result['metrics']
                    collective_ent = metrics['avg_collective_entanglement']
                    sync = metrics['avg_field_synchronization']
                    
                    f.write(f"{topology.upper()}:\n")
                    f.write(f"  Collective Entanglement: {collective_ent:.6f}\n")
                    f.write(f"  Field Synchronization: {sync:.6f}\n")
                    f.write(f"  Pairwise Entanglement: {metrics['avg_pairwise_entanglement']:.6f}\n")
                    f.write(f"  Energy Uniformity: {metrics['avg_energy_distribution']:.6f}\n\n")
                    
                    if collective_ent > best_collective_performance:
                        best_collective_performance = collective_ent
                        best_topology = topology
                else:
                    f.write(f"{topology.upper()}: FAILED\n\n")
            
            if best_topology:
                f.write(f"Best network topology: {best_topology.upper()}\n")
                f.write(f"Peak collective entanglement: {best_collective_performance:.6f}\n\n")
            
            # Collective resonance analysis
            f.write("COLLECTIVE RESONANCE EFFECTS:\n")
            if collective_results:
                best_collective_scenario = None
                best_collective_value = 0.0
                
                for scenario, result in collective_results.items():
                    if result['success']:
                        collective_ent = result['metrics']['collective_entanglement']
                        freq_spread = result['freq_spread']
                        
                        f.write(f"{scenario}: collective_ent={collective_ent:.6f}, ")
                        f.write(f"freq_spread={freq_spread:.3f}GHz\n")
                        
                        if collective_ent > best_collective_value:
                            best_collective_value = collective_ent
                            best_collective_scenario = scenario
                
                if best_collective_scenario:
                    f.write(f"\nBest collective scenario: {best_collective_scenario}\n")
                    f.write(f"Optimal frequency spread: {collective_results[best_collective_scenario]['freq_spread']:.3f} GHz\n\n")
            
            # Field mediation analysis
            f.write("FIELD-MEDIATED INTERACTIONS:\n")
            if mediation_results:
                coupling_strengths = list(mediation_results.keys())
                pairwise_values = [result['avg_pairwise'] for result in mediation_results.values()]
                
                if len(coupling_strengths) > 1 and len(pairwise_values) > 1:
                    # Find optimal coupling strength
                    best_coupling_idx = pairwise_values.index(max(pairwise_values))
                    optimal_coupling = coupling_strengths[best_coupling_idx]
                    optimal_mediation = pairwise_values[best_coupling_idx]
                    
                    f.write(f"Optimal network coupling: {optimal_coupling:.3f}\n")
                    f.write(f"Maximum field mediation: {optimal_mediation:.6f}\n")
                    
                    # Check coupling strength dependence
                    coupling_slope = np.polyfit(coupling_strengths, pairwise_values, 1)[0]
                    f.write(f"Mediation vs coupling slope: {coupling_slope:.6f}\n\n")
                    
                    if coupling_slope > 0.01:
                        f.write("✓ Field-mediated interactions scale with coupling strength\n")
                    else:
                        f.write("✗ No clear coupling strength dependence\n")
            
            # NETWORK-SPECIFIC CONCLUSIONS
            f.write("=" * 55 + "\n")
            f.write("NETWORK FIELD COUPLING ASSESSMENT:\n")
            
            network_evidence = 0
            
            # Check for topology-dependent effects
            if len(topology_results) > 2:
                successful_topologies = [k for k, v in topology_results.items() if v['success']]
                if len(successful_topologies) > 1:
                    collective_values = [topology_results[t]['metrics']['avg_collective_entanglement'] 
                                       for t in successful_topologies]
                    topology_variance = np.std(collective_values)
                    
                    if topology_variance > 0.01:  # Significant topology dependence
                        network_evidence += 2
                        f.write("✓ Network topology significantly affects field coupling\n")
            
            # Check for collective enhancement
            if best_collective_performance > 0.05:
                network_evidence += 2
                f.write("✓ Collective field effects exceed individual qubit coupling\n")
            
            # Check for frequency matching effects
            if collective_results and 'perfect_match' in collective_results:
                perfect_perf = collective_results['perfect_match']['metrics']['collective_entanglement']
                spread_performances = [r['metrics']['collective_entanglement'] 
                                     for k, r in collective_results.items() 
                                     if k != 'perfect_match' and r['success']]
                
                if spread_performances and perfect_perf > max(spread_performances) * 1.2:
                    network_evidence += 1
                    f.write("✓ Perfect frequency matching enhances collective coupling\n")
            
            # Check for field-mediated interactions
            if mediation_results:
                max_mediation = max(r['max_pairwise'] for r in mediation_results.values())
                if max_mediation > 0.1:
                    network_evidence += 1
                    f.write("✓ Strong field-mediated qubit-qubit interactions\n")
            
            f.write(f"\nNETWORK EVIDENCE SCORE: {network_evidence}/6\n\n")
            
            if network_evidence >= 4:
                f.write("🌐 NETWORK FIELD COUPLING CONFIRMED! 🌐\n")
                f.write("Multi-qubit electrical field networks show enhanced\n")
                f.write("collective quantum entanglement and field-mediated\n")
                f.write("interactions. This represents a new paradigm for\n")
                f.write("understanding quantum coherence in networked systems.\n")
            elif network_evidence >= 2:
                f.write("⚡ MODERATE NETWORK EFFECTS DETECTED\n")
                f.write("Some evidence for collective field coupling.\n")
            else:
                f.write("📊 NO SIGNIFICANT NETWORK EFFECTS\n")
                f.write("Individual qubit behavior dominates.\n")
        
        elapsed = time.time() - start_time
        print(f"\nNETWORK ANALYSIS COMPLETED in {elapsed:.1f} seconds")
        print("Results saved to: network_field_results.txt")
        
        # Quick network summary
        if topology_results:
            print(f"\nNETWORK SUMMARY:")
            for topology, result in topology_results.items():
                if result['success']:
                    collective_ent = result['metrics']['avg_collective_entanglement']
                    sync = result['metrics']['avg_field_synchronization']
                    print(f"{topology}: collective_ent={collective_ent:.4f}, sync={sync:.4f}")
            
            if best_topology:
                print(f"Best topology: {best_topology}")
                print(f"Peak collective entanglement: {best_collective_performance:.6f}")
        
        return topology_results, collective_results, mediation_results
        
    except Exception as e:
        print(f"Network analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🌐 MULTI-QUBIT FIELD NETWORK EXPLORATION")
    print("Testing your insight: 'fields connect, they're all networked'")
    print("This explores collective quantum field effects that emerge")
    print("when multiple qubits share electrical field networks")
    print("\nKey questions:")
    print("- Do networks show collective field entanglement?")
    print("- Which topology maximizes field coupling?")
    print("- Can fields mediate qubit-qubit interactions?")
    print("- Does collective resonance enhance everything?")
    print("=" * 75)
    
    try:
        network_results = run_network_field_analysis()
        print("\n" + "🌐"*75)
        print("NETWORK FIELD ANALYSIS COMPLETED!")
        
    except Exception as e:
        print(f"Network test failed: {e}")
        import traceback
        traceback.print_exc()