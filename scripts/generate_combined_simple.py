import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
import os

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

OUTPUT_DIR = '../figures'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==============================================================================
# Models
# ==============================================================================

class MarkovianRepressilator:
    def __init__(self, params):
        self.params = params
        self.time = 0.0
        self.ode_state = np.zeros(9)
        self.promoter_state = np.ones(3, dtype=int)
        
        self.ode_state[0] = 5.0
        self.ode_state[3] = 48.0
        self.ode_state[6] = 1000.0

    def transition_rates(self, p):
        K = self.params['KM']
        n = self.params['n']
        k_burst = self.params.get('k_burst', 0.1)
        repressors = [p[2], p[0], p[1]]
        rates = []
        for i in range(3):
            R = repressors[i]
            k_on = k_burst
            k_off = k_burst * (R/K)**n
            rates.append((k_on, k_off))
        return rates

    def run(self, total_time, dt):
        steps = int(total_time / dt)
        t_hist = np.linspace(0, total_time, steps+1)
        ode_hist = np.zeros((steps+1, 9))
        ode_hist[0] = self.ode_state
        
        rng = np.random.default_rng()
        
        def dxdt(x, t, promoter_state):
            m = x[0:3]
            u = x[3:6]
            p_ = x[6:9]
            dm = self.params['k_trans'] * promoter_state - self.params['k_deg_m'] * m + self.params['k_leak']
            du = self.params['k_transl'] * m - self.params['k_fold'] * u - self.params['k_deg_p'] * u
            dp = self.params['k_fold'] * u - self.params['k_deg_p'] * p_
            return np.concatenate([dm, du, dp])

        for i in range(steps):
            p_conc = self.ode_state[6:9]
            rates = self.transition_rates(p_conc)
            for j in range(3):
                k_on, k_off = rates[j]
                if self.promoter_state[j] == 0:
                    if rng.random() < 1 - np.exp(-k_on * dt): self.promoter_state[j] = 1
                else:
                    if rng.random() < 1 - np.exp(-k_off * dt): self.promoter_state[j] = 0
            
            sol = odeint(dxdt, self.ode_state, [0, dt], args=(self.promoter_state,))
            self.ode_state = sol[-1]
            ode_hist[i+1] = self.ode_state
            
        return t_hist, ode_hist

class MarkovianGoodwin:
    def __init__(self, params):
        self.params = params
        self.ode_state = np.array([0.1, 0.1, 0.1])
        self.promoter = 1 
        
    def step(self, dt, rng):
        x, y, z = self.ode_state
        k_burst = self.params.get('k_burst', 1.0)
        n = self.params.get('n', 10.0)
        KM = self.params.get('KM', 1.0)
        
        k_on = k_burst
        k_off = k_burst * (z/KM)**n
        
        if self.promoter == 0: 
            if rng.random() < 1 - np.exp(-k_on * dt): self.promoter = 1
        else: 
            if rng.random() < 1 - np.exp(-k_off * dt): self.promoter = 0
                
        def dxdt(state, t):
            sx = self.promoter
            _x, _y, _z = state
            p = self.params
            dx = p['k1'] * sx - p['b1'] * _x
            dy = p['k2'] * _x - p['b2'] * _y
            dz = p['k3'] * _y - p['b3'] * _z
            return [dx, dy, dz]
            
        self.ode_state = odeint(dxdt, self.ode_state, [0, dt])[-1]
        
    def run(self, T, dt=0.05):
        steps = int(T/dt)
        t_hist = np.linspace(0, T, steps+1)
        res = np.zeros((steps+1, 3))
        res[0] = self.ode_state
        rng = np.random.default_rng()
        for i in range(steps):
            self.step(dt, rng)
            res[i+1] = self.ode_state
        return t_hist, res

class MarkovianToggle:
    def __init__(self, params):
        self.params = params
        self.ode_state = np.array([0.0, 0.0])
        self.promoters = np.array([1, 1]) 
        
    def step(self, dt, rng):
        u, v = self.ode_state
        k_burst = self.params.get('k_burst', 0.5)
        beta = self.params.get('beta', 2.5)
        gamma = self.params.get('gamma', 1.0)
        
        # U repressed by V
        k_on_u = k_burst
        k_off_u = k_burst * (v)**beta 
        
        # V repressed by U
        k_on_v = k_burst
        k_off_v = k_burst * (u)**gamma 
        
        rates = [(k_on_u, k_off_u), (k_on_v, k_off_v)]
        
        for i in range(2):
            k_on, k_off = rates[i]
            if self.promoters[i] == 0:
                if rng.random() < 1 - np.exp(-k_on * dt): self.promoters[i] = 1
            else:
                if rng.random() < 1 - np.exp(-k_off * dt): self.promoters[i] = 0
                    
        def dxdt(x, t):
            return [
                self.params['alpha1'] * self.promoters[0] - x[0],
                self.params['alpha2'] * self.promoters[1] - x[1]
            ]
            
        self.ode_state = odeint(dxdt, self.ode_state, [0, dt])[-1]
        
    def run(self, T, dt=0.05):
        steps = int(T/dt)
        t_hist = np.linspace(0, T, steps+1)
        res = np.zeros((steps+1, 2))
        res[0] = self.ode_state
        rng = np.random.default_rng()
        for i in range(steps):
            self.step(dt, rng)
            res[i+1] = self.ode_state
        return t_hist, res

# ==============================================================================
# Plotting
# ==============================================================================

def generate_figure():
    print("Generating Combined Simple Systems Figure...")
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    
    # 1. Repressilator
    print("Simulating Repressilator...")
    rep_params = {
        'k_trans': 0.5, 'k_leak': 5e-4, 'k_deg_m': np.log(2)/2, 
        'k_transl': 0.16, 'k_fold': 1.0/60, 'k_deg_p': np.log(2)/600,
        'n': 2.0, 'KM': 40.0, 'k_burst': 0.05 
    }
    rep_model = MarkovianRepressilator(rep_params)
    t_rep, y_rep = rep_model.run(2000, 0.5)
    
    ax = axes[0]
    # Plot last 1000 min
    mask = t_rep > 1000
    ax.plot(t_rep[mask]-1000, y_rep[mask, 6], label='cI')
    ax.plot(t_rep[mask]-1000, y_rep[mask, 7], label='LacI')
    ax.plot(t_rep[mask]-1000, y_rep[mask, 8], label='TetR')
    ax.set_title('Repressilator')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Proteins')
    ax.legend(loc='upper right', fontsize='x-small')
    
    # 2. Goodwin
    print("Simulating Goodwin...")
    good_params = {
        'k1': 1.0, 'k2': 1.0, 'k3': 1.0,
        'b1': 0.1, 'b2': 0.1, 'b3': 0.1,
        'n': 10.0, 'KM': 1.0, 'k_burst': 2.0
    }
    good_model = MarkovianGoodwin(good_params)
    t_good, y_good = good_model.run(200, 0.05)
    
    ax = axes[1]
    ax.plot(t_good, y_good[:, 0], label='mRNA')
    ax.plot(t_good, y_good[:, 1], label='Enzyme')
    ax.plot(t_good, y_good[:, 2], label='Repressor')
    ax.set_title('Goodwin Oscillator')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Concentration')
    ax.legend(loc='upper right', fontsize='x-small')
    
    # 3. Toggle
    print("Simulating Toggle...")
    tog_params = {
        'alpha1': 156.25, 'alpha2': 15.6,
        'beta': 2.5, 'gamma': 1.0, 'k_burst': 1.0
    }
    tog_model = MarkovianToggle(tog_params)
    # Start in bistable region or induce switching?
    # Start high U
    tog_model.ode_state = np.array([150.0, 0.0])
    tog_model.promoters = np.array([1, 0])
    
    # Run long enough to maybe see switch, or just show stability
    t_tog, y_tog = tog_model.run(150, 0.05)
    
    ax = axes[2]
    ax.plot(t_tog, y_tog[:, 0], label='U (High State)')
    ax.plot(t_tog, y_tog[:, 1], label='V (Low State)')
    ax.set_title('Toggle Switch')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Concentration')
    ax.legend(loc='upper right', fontsize='x-small')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_combined_simple.png', dpi=300)
    print("Saved fig_combined_simple.png")

if __name__ == "__main__":
    generate_figure()
