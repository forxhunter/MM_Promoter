
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class I1_FFL_ODE:
    """
    Deterministic ODE model for Incoherent Feed-Forward Loop (Type 1).
    Structure:
      X -> Y (Activation)
      X -> Z (Activation)
      Y -> Z (Repression)
    """
    def __init__(self, params):
        self.params = params
        # State: [Y, Z] (X is input)
        self.state = [0.0, 0.0]
        self.time = 0.0
        self.history = {'t': [], 'X': [], 'Y': [], 'Z': []}

    def dynamics(self, state, t, X_val):
        Y, Z = state
        p = self.params
        
        # dY/dt = beta_y * (X / (K_xy + X)) - alpha_y * Y
        # Hill activation of Y by X
        prod_Y = p['beta_y'] * (X_val / (p['K_xy'] + X_val))
        dY = prod_Y - p['alpha_y'] * Y
        
        # dZ/dt = beta_z * (X / (K_xz + X)) * (K_yz / (K_yz + Y)) - alpha_z * Z
        # AND logic: X activates AND Y represses
        # Activation term: X/(K+X)
        # Repression term: K/(K+Y)
        act_X = (X_val / (p['K_xz'] + X_val))
        rep_Y = (p['K_yz'] / (p['K_yz'] + Y))
        
        prod_Z = p['beta_z'] * act_X * rep_Y
        dZ = prod_Z - p['alpha_z'] * Z
        
        return [dY, dZ]

    def run(self, T, dt, X_func):
        """
        Run simulation.
        X_func: function of time t returning X concentration.
        """
        t_values = np.arange(0, T, dt)
        
        for t in t_values:
            X_val = X_func(t)
            self.history['t'].append(t)
            self.history['X'].append(X_val)
            self.history['Y'].append(self.state[0])
            self.history['Z'].append(self.state[1])
            
            # Solve for one step
            next_state = odeint(self.dynamics, self.state, [t, t+dt], args=(X_val,))[-1]
            self.state = next_state
            
        return self.history
