"""
Markovian Promoter Model for GAL Gene Regulation

This module implements explicit Markovian dynamics for promoter states,
replacing the CME-based promoter binding/unbinding reactions.

Each gene promoter can be in one of three states:
- Empty (DGx): No activator bound
- Active (DGx_G4d): Gal4p dimer bound, can transcribe
- Repressed (DGx_G4d_G80d): Gal4p + Gal80p bound, cannot transcribe

The Markovian model explicitly tracks the probability distribution over
these states and uses stochastic sampling to determine the current state.

Author: Tianyu Wu
"""

import numpy as np

# Python 3.7 compatibility: math.comb only available in Python 3.8+
try:
    from math import comb
except ImportError:
    # Fallback for Python < 3.8
    def comb(n, k):
        """Compute binomial coefficient C(n, k) = n! / (k! * (n-k)!)"""
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)  # Use symmetry
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result


class MarkovianPromoterModel:
    """
    Markovian model for GAL gene promoter dynamics.
    
    For genes with n binding sites, we use a reduced state representation:
    State (k, m): k sites have Gal4p bound, m of those also have Gal80p
    
    Active = at least one site has Gal4p without Gal80p (k > m)
    """
    
    # Promoter configurations: (name, n_sites)
    PROMOTERS = {
        'GAL1': {'n_sites': 4, 'empty': 'DG1', 'active': 'DG1_G4d', 'repressed': 'DG1_G4d_G80d'},
        'GAL2': {'n_sites': 5, 'empty': 'DG2', 'active': 'DG2_G4d', 'repressed': 'DG2_G4d_G80d'},
        'GAL3': {'n_sites': 1, 'empty': 'DG3', 'active': 'DG3_G4d', 'repressed': 'DG3_G4d_G80d'},
        'GAL80': {'n_sites': 1, 'empty': 'DG80', 'active': 'DG80_G4d', 'repressed': 'DG80_G4d_G80d'},
        'reporter': {'n_sites': 4, 'empty': 'DGrep', 'active': 'DGrep_G4d', 'repressed': 'DGrep_G4d_G80d'},
    }
    
    def __init__(self, binding_scale=1.0):
        """
        Initialize the Markovian promoter model.
        
        Parameters:
            binding_scale: Scaling factor for binding kinetics
                          1.0 = normal kinetics
                          >1.0 = faster equilibration
        """
        self.binding_scale = binding_scale
        
        # Kinetic parameters (from Ramsey et al. 2006)
        # These match the original ODE model
        max_R4 = 0.4
        prot_to_mrna_gal4 = 1545
        max_G4d = max_R4 * prot_to_mrna_gal4 / 2  # ~309.1
        
        self.kfp = 6.5 / max_G4d * binding_scale  # Gal4p binding forward rate
        self.krp = 1.0 * binding_scale             # Gal4p unbinding rate
        self.kfr = 5 * self.kfp                    # Gal80p binding forward rate
        self.krr = 1.0 * binding_scale             # Gal80p unbinding rate
        self.qr = 30.0                              # Cooperativity factor
        
        # Equilibrium constants
        self.kp = self.kfp / self.krp
        self.kr = self.kfr / self.krr
        
        # Initialize state probabilities for each promoter
        # Format: {gene_name: np.array of state probabilities}
        self.state_probs = {}
        for name, config in self.PROMOTERS.items():
            n_states = self._get_num_states(config['n_sites'])
            # Initialize all probability in empty state
            p = np.zeros(n_states)
            p[0] = 1.0
            self.state_probs[name] = p
        
        # Activation tracker: {gene_name: {'active_time': float, 'total_time': float}}
        self.activation_tracker = {name: {'active_time': 0.0, 'total_time': 0.0, 'history': []} 
                                   for name in self.PROMOTERS}
    
    def _get_num_states(self, n_sites):
        """Number of states for n binding sites."""
        return (n_sites + 1) * (n_sites + 2) // 2
    
    def _state_to_index(self, k, m):
        """Convert (k, m) to linear index."""
        return k * (k + 1) // 2 + m
    
    def _index_to_state(self, idx, n_sites):
        """Convert linear index to (k, m)."""
        k = 0
        while (k + 1) * (k + 2) // 2 <= idx:
            k += 1
        m = idx - k * (k + 1) // 2
        return k, m
    
    def _compute_rate_matrix(self, n_sites, G4d, G80d):
        """
        Build the rate matrix Q for promoter state transitions.
        
        Returns Q where dp/dt = p @ Q
        """
        n_states = self._get_num_states(n_sites)
        Q = np.zeros((n_states, n_states))
        
        for idx in range(n_states):
            k, m = self._index_to_state(idx, n_sites)
            
            # Gal4p binding: (k,m) -> (k+1,m)
            if k < n_sites:
                j = self._state_to_index(k + 1, m)
                rate = self.kfp * G4d * (n_sites - k)
                Q[idx, j] = rate
            
            # Gal4p unbinding from active site: (k,m) -> (k-1,m)
            if k > m and k > 0:
                j = self._state_to_index(k - 1, m)
                rate = self.krp * (k - m)
                Q[idx, j] = rate
            
            # Gal80p binding: (k,m) -> (k,m+1)
            if m < k:
                j = self._state_to_index(k, m + 1)
                coop = self.qr if m > 0 else 1.0
                rate = self.kfr * G80d * (k - m) * coop
                Q[idx, j] = rate
            
            # Gal80p unbinding: (k,m) -> (k,m-1)
            if m > 0:
                j = self._state_to_index(k, m - 1)
                rate = self.krr * m
                Q[idx, j] = rate
        
        # Diagonal: negative sum of outgoing rates
        for i in range(n_states):
            Q[i, i] = -np.sum(Q[i, :])
        
        return Q
    
    def _compute_equilibrium(self, n_sites, G4d, G80d):
        """
        Compute equilibrium distribution using partition function.
        """
        n_states = self._get_num_states(n_sites)
        weights = np.zeros(n_states)
        
        kpf = self.kp * G4d
        krf = self.kr * G80d
        
        for idx in range(n_states):
            k, m = self._index_to_state(idx, n_sites)
            binom_nk = comb(n_sites, k)
            binom_km = comb(k, m)
            coop_exp = max(0, m - 1)
            coop_factor = self.qr ** coop_exp
            weights[idx] = binom_nk * (kpf ** k) * binom_km * (krf ** m) * coop_factor
        
        Z = np.sum(weights)
        return weights / Z
    
    def step(self, dt, G4d, G80d):
        """
        Advance the Markovian model by time dt.
        
        Parameters:
            dt: time step (minutes)
            G4d: Gal4p dimer concentration (molecules)
            G80d: Gal80p dimer concentration (molecules)
        
        Returns:
            dict: {gene_name: 'empty' | 'active' | 'repressed'}
        """
        gene_states = {}
        
        for name, config in self.PROMOTERS.items():
            n_sites = config['n_sites']
            n_states = self._get_num_states(n_sites)
            
            # Get current probability distribution
            p = self.state_probs[name]
            
            # Build rate matrix
            Q = self._compute_rate_matrix(n_sites, G4d, G80d)
            
            # Evolve probabilities: dp/dt = p @ Q
            # Use matrix exponential for exact solution: p(t+dt) = p(t) @ exp(Q*dt)
            # For small dt, use first-order approximation: p(t+dt) ≈ p(t) + p(t) @ Q * dt
            if dt * np.max(np.abs(Q)) < 0.1:
                # First-order approximation
                p_new = p + p @ Q * dt
            else:
                # Matrix exponential for larger time steps
                from scipy.linalg import expm
                p_new = p @ expm(Q * dt)
            
            # Ensure non-negative and normalized
            p_new = np.maximum(p_new, 0)
            p_new = p_new / np.sum(p_new)
            
            self.state_probs[name] = p_new
            
            # Classify state based on probability distribution
            # Compute active fraction (probability k > m)
            active_prob = 0.0
            repressed_prob = 0.0
            empty_prob = p_new[0]  # State (0,0) = empty
            
            for idx in range(n_states):
                k, m = self._index_to_state(idx, n_sites)
                if k > m:
                    active_prob += p_new[idx]
                elif k == m and k > 0:
                    repressed_prob += p_new[idx]
            
            # Stochastic sampling: determine discrete state
            rand = np.random.random()
            if rand < empty_prob:
                gene_states[name] = 'empty'
            elif rand < empty_prob + active_prob:
                gene_states[name] = 'active'
            else:
                gene_states[name] = 'repressed'
            
            # Update activation tracker
            self.activation_tracker[name]['total_time'] += dt
            if gene_states[name] == 'active':
                self.activation_tracker[name]['active_time'] += dt
            
            # Store history point (every ~1 minute)
            if len(self.activation_tracker[name]['history']) == 0 or \
               self.activation_tracker[name]['total_time'] - \
               (self.activation_tracker[name]['history'][-1][0] if self.activation_tracker[name]['history'] else 0) >= 1.0:
                frac = self.activation_tracker[name]['active_time'] / max(self.activation_tracker[name]['total_time'], 1e-10)
                self.activation_tracker[name]['history'].append((self.activation_tracker[name]['total_time'], frac, active_prob))
        
        return gene_states
    
    def get_active_fraction(self, gene_name, G4d, G80d):
        """
        Compute the instantaneous active fraction for a gene.
        """
        config = self.PROMOTERS[gene_name]
        n_sites = config['n_sites']
        p = self.state_probs[gene_name]
        
        active_prob = 0.0
        n_states = self._get_num_states(n_sites)
        for idx in range(n_states):
            k, m = self._index_to_state(idx, n_sites)
            if k > m:
                active_prob += p[idx]
        
        return active_prob
    
    def get_activation_statistics(self):
        """
        Get activation statistics for all genes.
        
        Returns:
            dict: {gene_name: {'fraction': float, 'active_time': float, 'total_time': float}}
        """
        stats = {}
        for name in self.PROMOTERS:
            tracker = self.activation_tracker[name]
            total = tracker['total_time']
            active = tracker['active_time']
            stats[name] = {
                'fraction': active / max(total, 1e-10),
                'active_time': active,
                'total_time': total,
            }
        return stats
    
    def reset(self):
        """Reset state probabilities and trackers for a new replicate."""
        for name, config in self.PROMOTERS.items():
            n_states = self._get_num_states(config['n_sites'])
            p = np.zeros(n_states)
            p[0] = 1.0
            self.state_probs[name] = p
        
        self.activation_tracker = {name: {'active_time': 0.0, 'total_time': 0.0, 'history': []} 
                                   for name in self.PROMOTERS}
    
    def initialize_to_equilibrium(self, G4d, G80d):
        """Initialize all promoters to their equilibrium distribution."""
        for name, config in self.PROMOTERS.items():
            n_sites = config['n_sites']
            self.state_probs[name] = self._compute_equilibrium(n_sites, G4d, G80d)


class PromoterStateTracker:
    """
    Tracks promoter states over time for analysis and visualization.
    """
    
    def __init__(self, gene_names):
        self.gene_names = gene_names
        self.time_points = []
        self.states = {name: [] for name in gene_names}
        self.active_probs = {name: [] for name in gene_names}
        self.cumulative_active_fraction = {name: [] for name in gene_names}
        self._active_time = {name: 0.0 for name in gene_names}
        self._total_time = 0.0
    
    def record(self, time, gene_states, active_probs, dt):
        """Record state at current time."""
        self.time_points.append(time)
        self._total_time += dt
        
        for name in self.gene_names:
            self.states[name].append(gene_states.get(name, 'empty'))
            self.active_probs[name].append(active_probs.get(name, 0.0))
            
            if gene_states.get(name) == 'active':
                self._active_time[name] += dt
            
            frac = self._active_time[name] / max(self._total_time, 1e-10)
            self.cumulative_active_fraction[name].append(frac)
    
    def get_summary(self):
        """Get summary statistics."""
        summary = {}
        for name in self.gene_names:
            n_active = sum(1 for s in self.states[name] if s == 'active')
            n_repressed = sum(1 for s in self.states[name] if s == 'repressed')
            n_empty = sum(1 for s in self.states[name] if s == 'empty')
            total = len(self.states[name])
            
            summary[name] = {
                'active_fraction': n_active / max(total, 1) if total > 0 else 0,
                'repressed_fraction': n_repressed / max(total, 1) if total > 0 else 0,
                'empty_fraction': n_empty / max(total, 1) if total > 0 else 0,
                'mean_active_prob': np.mean(self.active_probs[name]) if self.active_probs[name] else 0,
                'cumulative_active_fraction': self.cumulative_active_fraction[name][-1] if self.cumulative_active_fraction[name] else 0,
            }
        return summary
    
    def reset(self):
        """Reset tracker for new replicate."""
        self.time_points = []
        self.states = {name: [] for name in self.gene_names}
        self.active_probs = {name: [] for name in self.gene_names}
        self.cumulative_active_fraction = {name: [] for name in self.gene_names}
        self._active_time = {name: 0.0 for name in self.gene_names}
        self._total_time = 0.0

