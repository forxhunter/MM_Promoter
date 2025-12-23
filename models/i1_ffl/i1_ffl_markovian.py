
import numpy as np
import sys
import os

# Add root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from core.multi_input_promoter import MultiInputPromoter

class I1_FFL_Markovian:
    """
    Hybrid Markovian model for I1-FFL.
    Promoters are discrete Markov chains.
    Gene Expression is deterministic ODE regulated by promoter state.
    """
    def __init__(self, params):
        self.params = params
        
        # Initialize Promoters
        
        # Y Promoter: Regulated by X
        # Config: k_on depends on X (handled in step), k_off const
        self.promoter_Y = MultiInputPromoter({
            'X': {'k_on': params['k_on_xy'], 'k_off': params['k_off_xy']}
        })
        
        # Z Promoter: Regulated by X (activator) and Y (repressor)
        self.promoter_Z = MultiInputPromoter({
            'X': {'k_on': params['k_on_xz'], 'k_off': params['k_off_xz']},
            'Y': {'k_on': params['k_on_yz'], 'k_off': params['k_off_yz']}
        }, logic_function=self.z_logic)
        
        # ODE State: [Y, Z] (concentrations)
        self.Y = 0.0
        self.Z = 0.0
        self.time = 0.0
        
        self.history = {'t': [], 'X': [], 'Y': [], 'Z': [], 'Sy': [], 'Sz': []}

    def z_logic(self, state):
        """
        Logic for Z promoter:
        Active if X is bound AND Y is NOT bound.
        """
        x_bound = state.get('X', 0) > 0
        y_bound = state.get('Y', 0) > 0
        return 1.0 if (x_bound and not y_bound) else 0.0

    def step(self, dt, X_val):
        # Update Promoters
        # Concentrations passed map TF names to values
        concs = {'X': X_val, 'Y': self.Y}
        
        # Step promoters
        # Note: MultiInputPromoter returns 1.0 (active) or 0.0 (inactive) based on logic
        act_Y = self.promoter_Y.step(dt, concs) # logic default AND (X bound)
        act_Z = self.promoter_Z.step(dt, concs) # logic custom (X and not Y)
        
        # Update ODEs
        # Production proportional to activity
        p = self.params
        
        prod_Y = p['beta_y'] * act_Y
        deg_Y = p['alpha_y'] * self.Y
        
        prod_Z = p['beta_z'] * act_Z
        deg_Z = p['alpha_z'] * self.Z
        
        # Euler integration
        self.Y += (prod_Y - deg_Y) * dt
        self.Z += (prod_Z - deg_Z) * dt
        
        self.time += dt
        
        return act_Y, act_Z

    def run(self, T, dt, X_func):
        t_grid = np.arange(0, T, dt)
        
        for t in t_grid:
            X_val = X_func(t)
            
            # Record
            sy_state = self.promoter_Y.get_state().get('X', 0)
            sz_state = self.promoter_Z.get_state()
            
            self.history['t'].append(t)
            self.history['X'].append(X_val)
            self.history['Y'].append(self.Y)
            self.history['Z'].append(self.Z)
            self.history['Sy'].append(sy_state) # 1 if X bound
            
            # Z logic state for plotting simplicity
            sz_active = self.z_logic(sz_state)
            self.history['Sz'].append(sz_active)
            
            self.step(dt, X_val)
            
        return self.history
