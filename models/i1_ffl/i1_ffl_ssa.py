
import numpy as np

class I1_FFL_SSA:
    """
    Gillespie SSA ground truth for I1-FFL.
    Discrete molecule counts.
    """
    def __init__(self, params, Omega=1.0):
        self.params = params
        self.Omega = Omega # System volume / scaling factor
        
        # State: [Y, Z] (X is external driven parameter)
        # Counts are integers
        self.Y = 0
        self.Z = 0
        self.time = 0.0
        
        self.history = {'t': [], 'X': [], 'Y': [], 'Z': []}
        
    def propensity(self, X_val):
        p = self.params
        O = self.Omega
        
        # Convert constants to stochastic rates (approx)
        # beta (conc/time) -> beta * Omega (molecules/time)
        b_y = p['beta_y'] * O
        b_z = p['beta_z'] * O
        a_y = p['alpha_y'] # 1/time unchanged
        a_z = p['alpha_z']
        
        # Hill functions using concentrations
        # X_val is conc. Y_conc = Y / O
        Y_conc = self.Y / O
        
        # Y production
        # Prop = b_y * (X / (K + X))
        a1 = b_y * (X_val / (p['K_xy'] + X_val))
        
        # Y degradation
        a2 = a_y * self.Y
        
        # Z production
        # Prop = b_z * (X/(K+X)) * (K/(K+Y))
        term_X = (X_val / (p['K_xz'] + X_val))
        term_Y = (p['K_yz'] / (p['K_yz'] + Y_conc))
        a3 = b_z * term_X * term_Y
        
        # Z degradation
        a4 = a_z * self.Z
        
        return [a1, a2, a3, a4]

    def step(self, X_val, t_end):
        """Advance time until t_end is reached or exceeded."""
        while self.time < t_end:
            props = self.propensity(X_val)
            a_sum = sum(props)
            
            if a_sum == 0:
                self.time = t_end
                break
                
            r1 = np.random.rand()
            tau = -np.log(r1) / a_sum
            
            if self.time + tau > t_end:
                self.time = t_end
                break
                
            self.time += tau
            
            # Determine reaction
            r2 = np.random.rand() * a_sum
            cum = 0
            rxn_idx = -1
            for i, p in enumerate(props):
                cum += p
                if r2 <= cum:
                    rxn_idx = i
                    break
            
            # Execute reaction
            if rxn_idx == 0: # Y prod
                self.Y += 1
            elif rxn_idx == 1: # Y deg
                self.Y -= 1
            elif rxn_idx == 2: # Z prod
                self.Z += 1
            elif rxn_idx == 3: # Z deg
                self.Z -= 1

    def run(self, T, dt, X_func):
        t_grid = np.arange(0, T, dt)
        
        for t in t_grid:
            X_val = X_func(t)
            
            # Record state (concentrations)
            self.history['t'].append(t)
            self.history['X'].append(X_val)
            self.history['Y'].append(self.Y / self.Omega)
            self.history['Z'].append(self.Z / self.Omega)
            
            # Run SSA until next grid point
            self.step(X_val, t + dt)
            
        return self.history
