import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd

# Goodwin Oscillator - Hybrid V2 (Discrete mRNA)
# Treats mRNA (X) as discrete molecules, Y and Z as continuous ODEs

class MarkovianGoodwinV2:
    def __init__(self, params, Omega=50.0):
        self.params = params
        self.Omega = Omega
        self.time = 0.0
        
        # State:
        # Discrete: Promoter (Sx) ∈ {0, 1}, mRNA count (X) ∈ {0, 1, 2, ...}
        # Continuous: Y, Z (proteins)
        self.promoter = 1  # Start Active
        self.X = int(0.1 * Omega)  # mRNA count (discrete)
        self.ode_state = np.array([0.1, 0.1])  # Y, Z (continuous)
        
        # History
        self.history_t = []
        self.history_promoter = []
        self.history_X = []
        self.history_ode = []
        
    def step(self, dt):
        x_conc = self.X / self.Omega
        y, z = self.ode_state
        
        p = self.params
        k_burst = p.get('k_burst', 2.0)
        n = p.get('n', 10.0)
        KM = p.get('KM', 1.0)
        k1 = p['k1']
        k2 = p['k2']
        k3 = p['k3']
        b1 = p['b1']
        b2 = p['b2']
        b3 = p['b3']
        
        # 1. Promoter switching (same as V1)
        k_on = k_burst
        k_off = k_burst * (z/KM)**n
        
        if self.promoter == 0:
            p_on = 1 - np.exp(-k_on * dt)
            if np.random.rand() < p_on:
                self.promoter = 1
        else:
            p_off = 1 - np.exp(-k_off * dt)
            if np.random.rand() < p_off:
                self.promoter = 0
        
        # 2. mRNA reactions (Discrete, Gillespie-style)
        # Production: Poisson process
        if self.promoter == 1:
            rate_prod = k1 * self.Omega * dt
            n_prod = np.random.poisson(rate_prod)
            self.X += n_prod
        
        # Degradation: Binomial process
        if self.X > 0:
            p_deg = 1 - np.exp(-b1 * dt)
            n_deg = np.random.binomial(self.X, p_deg)
            self.X -= n_deg
        
        # 3. Protein ODEs (with discrete X as input)
        def dxdt(state, t):
            _y, _z = state
            dy = k2 * x_conc - b2 * _y
            dz = k3 * _y - b3 * _z
            return [dy, dz]
        
        next_state = odeint(dxdt, self.ode_state, [0, dt])[-1]
        self.ode_state = next_state
        self.time += dt
        
        # Record
        self.history_t.append(self.time)
        self.history_promoter.append(self.promoter)
        self.history_X.append(self.X)
        self.history_ode.append(self.ode_state.copy())
        
    def run(self, T, dt=0.001):
        # Record initial state
        self.history_t.append(self.time)
        self.history_promoter.append(self.promoter)
        self.history_X.append(self.X)
        self.history_ode.append(self.ode_state.copy())
        
        steps = int(T/dt)
        for _ in range(steps):
            self.step(dt)
        
        return (np.array(self.history_t), 
                np.array(self.history_X) / self.Omega,  # X concentration
                np.array(self.history_ode),  # Y, Z
                np.array(self.history_promoter))

# Parameters (same as V1)
params = {
    'k1': 1.0, 'k2': 1.0, 'k3': 1.0,
    'b1': 0.1, 'b2': 0.1, 'b3': 0.1,
    'n': 9.0, 'KM': 1.0,
    'k_burst': 2.0
}

if __name__ == "__main__":
    print("Running Goodwin Markovian V2 (Discrete mRNA)...")
    model = MarkovianGoodwinV2(params, Omega=50.0)
    t, x, yz, promoter = model.run(1000, dt=0.001)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    axes[0].plot(t, x, 'r-', linewidth=1.0, alpha=0.8, label='X (mRNA, discrete)')
    axes[0].plot(t, yz[:, 0], 'g-', linewidth=1.0, alpha=0.8, label='Y (Enzyme, continuous)')
    axes[0].plot(t, yz[:, 1], 'b-', linewidth=1.0, alpha=0.8, label='Z (Repressor, continuous)')
    axes[0].set_ylabel('Concentration', fontsize=11)
    axes[0].set_title('Goodwin Oscillator - Hybrid V2 (Discrete mRNA)', fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].step(t, promoter, 'k-', linewidth=1.0, alpha=0.7, where='post')
    axes[1].set_ylabel('Promoter State', fontsize=11)
    axes[1].set_xlabel('Time', fontsize=11)
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../20251218_goodwin_v2/goodwin_markovian_v2.png', dpi=150)
    print("Saved ../20251218_goodwin_v2/goodwin_markovian_v2.png")
    
    # Save data
    df = pd.DataFrame({
        'Time': t,
        'X': x,
        'Y': yz[:, 0],
        'Z': yz[:, 1],
        'Promoter': promoter
    })
    df.to_csv('../20251218_goodwin_v2/goodwin_markovian_v2_trajectories.csv', index=False)
    print("Saved ../20251218_goodwin_v2/goodwin_markovian_v2_trajectories.csv")
