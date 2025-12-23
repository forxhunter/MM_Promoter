import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Genetic Toggle Switch - Markovian Hybrid Implementation
# Ref: Gardner et al (2000), Rewrite of ykayuwu/Bistability-Project-Code model

class MarkovianToggle:
    def __init__(self, params):
        self.params = params
        self.time = 0.0
        
        # State: 
        # ODE: 2 vars (U, V)
        # Discrete: 2 Promoters (Su, Sv) \in {0, 1}
        self.ode_state = np.array([0.0, 0.0]) # Start Low
        self.promoters = np.array([1, 1]) # Start Active
        
        # History
        self.history_t = []
        self.history_ode = []
        self.history_promoters = []
        
    def transition_rates(self, u, v):
        # Gene U is repressed by V
        # Effective K_v = 1 (from equation 1/(1+v^beta)) -> K=1
        # Effective K_u = 1 
        
        # Rate U: Repressor = V. 
        # Rate V: Repressor = U.
        
        # k_on (recovery) = constant (k_burst)
        # k_off (repression) = k_on * (Repressor/K)^n
        
        k_burst = self.params.get('k_burst', 0.5)
        
        # Promoter U (Sensitive to V)
        beta = self.params.get('beta', 2.5)
        k_on_u = k_burst
        k_off_u = k_burst * (v)**beta # K=1 implicitly
        
        # Promoter V (Sensitive to U)
        gamma = self.params.get('gamma', 1.0)
        k_on_v = k_burst
        k_off_v = k_burst * (u)**gamma # K=1 implicitly
        
        return [(k_on_u, k_off_u), (k_on_v, k_off_v)]
        
    def step(self, dt):
        u, v = self.ode_state
        rates = self.transition_rates(u, v)
        
        # 1. Update Promoters (CORRECTED: Use exact Poisson formula)
        for i in range(2):
            k_on, k_off = rates[i]
            if self.promoters[i] == 0:  # Repressed
                p_on = 1 - np.exp(-k_on * dt)  # CORRECT Poisson formula
                if np.random.rand() < p_on:
                    self.promoters[i] = 1
            else:  # Active
                p_off = 1 - np.exp(-k_off * dt)  # CORRECT Poisson formula
                if np.random.rand() < p_off:
                    self.promoters[i] = 0
                    
        # 2. Update ODEs
        # equations: du/dt = alpha1 * Su - u
        #            dv/dt = alpha2 * Sv - v
        # Assuming degradation is 1.0 (from -u, -v)
        
        def dxdt(x, t):
            # Production rate matches max rate in Hill: alpha
            # If Su=1 -> alpha. If Su=0 -> 0.
            # Hill function max is alpha.
            return [
                self.params['alpha1'] * self.promoters[0] - x[0],
                self.params['alpha2'] * self.promoters[1] - x[1]
            ]
            
        next_x = odeint(dxdt, self.ode_state, [0, dt])[-1]
        self.ode_state = next_x
        self.time += dt
        
        # Log
        self.history_t.append(self.time)
        self.history_ode.append(self.ode_state)
        self.history_promoters.append(self.promoters.copy())
        
    def run(self, T, dt=0.001):  # REDUCED from 0.01 for better accuracy
        # Record t=0
        self.history_t.append(self.time)
        self.history_ode.append(self.ode_state)
        self.history_promoters.append(self.promoters.copy())
        
        steps = int(T/dt)
        for _ in range(steps):
            self.step(dt)
        return np.array(self.history_t), np.array(self.history_ode), np.array(self.history_promoters)

# Parameters
params = {
    'alpha1': 156.25,
    'alpha2': 15.6,
    'beta': 2.5,
    'gamma': 1.0,
    'k_burst': 1.0 # Tune frequency
}

if __name__ == "__main__":
    # Scenario 1: Start High U
    model1 = MarkovianToggle(params)
    model1.ode_state = np.array([150.0, 0.0])
    model1.promoters = np.array([1, 0]) # U active, V repressed
    print("Running Markovian Toggle (High U)...")
    t1, y1, p1 = model1.run(100, dt=0.05)
    
    # Scenario 2: Start High V
    model2 = MarkovianToggle(params)
    model2.ode_state = np.array([0.0, 15.0])
    model2.promoters = np.array([0, 1])
    print("Running Markovian Toggle (High V)...")
    t2, y2, p2 = model2.run(100, dt=0.05)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t1, y1[:, 0], 'b-', label='U (Start High U)', alpha=0.7)
    plt.plot(t1, y1[:, 1], 'b--', label='V (Start High U)', alpha=0.7)
    plt.plot(t2, y2[:, 0], 'r-', label='U (Start High V)', alpha=0.7)
    plt.plot(t2, y2[:, 1], 'r--', label='V (Start High V)', alpha=0.7)
    plt.title('Markovian Toggle Switch (Noise Induced Switching?)')
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    # Check for switching events
    plt.step(t1, p1[:, 0] + 1.1, 'b-', label='Promoter U (Run 1)', alpha=0.5)
    plt.step(t1, p1[:, 1], 'b--', label='Promoter V (Run 1)', alpha=0.5)
    plt.yticks([0, 1, 1.1, 2.1], ['V Off', 'V On', 'U Off', 'U On'])
    plt.title('Promoter Activity (Run 1)')
    plt.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('toggle_markovian.png')
    
    # Save Data for KS
    import pandas as pd
    df = pd.DataFrame({
        'Time': t1,
        'U_Run1': y1[:, 0], 'V_Run1': y1[:, 1],
        'U_Run2': y2[:, 0], 'V_Run2': y2[:, 1]
    })
    df.to_csv('../20251218_toggle/toggle_markovian_trajectories.csv', index=False)
