import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Goodwin Oscillator - Markovian Hybrid
# Replaces Hill repression of X by Z with Stochastic Promoter Switching.

class MarkovianGoodwin:
    def __init__(self, params):
        self.params = params
        self.time = 0.0
        
        # State:
        # ODE: 3 vars (X, Y, Z)
        # Discrete: 1 Promoter (Sx) \in {0, 1}
        self.ode_state = np.array([0.1, 0.1, 0.1])
        self.promoter = 1 # Start Active
        
        # History
        self.history_t = []
        self.history_ode = []
        self.history_promoter = []
        
    def step(self, dt):
        x, y, z = self.ode_state
        
        # Transition Rate for Promoter Sx
        # Repressed by Z.
        # Hill: 1 / (1 + (z/KM)^n)
        # k_off = k_on * (z/KM)^n
        
        k_burst = self.params.get('k_burst', 1.0)
        n = self.params.get('n', 10.0)
        KM = self.params.get('KM', 1.0)
        
        k_on = k_burst
        k_off = k_burst * (z/KM)**n
        
        # Update Promoter
        if self.promoter == 0: # Repressed
            p_on = 1 - np.exp(-k_on * dt)
            if np.random.rand() < p_on:
                self.promoter = 1
        else: # Active
            p_off = 1 - np.exp(-k_off * dt)
            if np.random.rand() < p_off:
                self.promoter = 0
                
        # Update ODEs
        # dx/dt = k1 * Sx - b1*x
        # dy/dt = k2 * x - b2*y
        # dz/dt = k3 * y - b3*z
        
        def dxdt(state, t):
            sx = self.promoter
            _x, _y, _z = state
            p = self.params
            
            dx = p['k1'] * sx - p['b1'] * _x
            dy = p['k2'] * _x - p['b2'] * _y
            dz = p['k3'] * _y - p['b3'] * _z
            
            return [dx, dy, dz]
            
        next_state = odeint(dxdt, self.ode_state, [0, dt])[-1]
        self.ode_state = next_state
        self.time += dt
        
        self.history_t.append(self.time)
        self.history_ode.append(self.ode_state)
        self.history_promoter.append(self.promoter)
        
    def run(self, T, dt=0.05):
        # Record t=0
        self.history_t.append(self.time)
        self.history_ode.append(self.ode_state)
        self.history_promoter.append(self.promoter)
        
        steps = int(T/dt)
        for _ in range(steps):
            self.step(dt)
        return np.array(self.history_t), np.array(self.history_ode), np.array(self.history_promoter)

# Parameters (Standard Goodwin that oscillates)
params = {
    'k1': 1.0, 'k2': 1.0, 'k3': 1.0,
    'b1': 0.1, 'b2': 0.1, 'b3': 0.1,
    'n': 10.0, 'KM': 1.0,
    'k_burst': 2.0 # Fast switching to approximate Hill
}

if __name__ == "__main__":
    # Run
    model = MarkovianGoodwin(params)
    print("Running Markovian Goodwin...")
    # Smaller dt for accuracy (validating peak hypothesis)
    dt = 0.001 
    t, y, p = model.run(1000, dt=dt)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, y[:, 0], label='X (mRNA)')
    plt.plot(t, y[:, 1], label='Y (Enzyme)')
    plt.plot(t, y[:, 2], label='Z (Repressor)')
    plt.title('Markovian Goodwin Oscillator')
    plt.ylabel('Concentration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.step(t, p, label='Promoter X')
    plt.xlabel('Time')
    plt.title('Promoter Activity')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('goodwin_markovian.png')
    
    # Save Validation Data
    import pandas as pd
    df = pd.DataFrame({'Time': t, 'X': y[:, 0], 'Y': y[:, 1], 'Z': y[:, 2]})
    df.to_csv('../20251218_goodwin/goodwin_markovian_trajectories.csv', index=False)
