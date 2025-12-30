import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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
# 1. I1-FFL Promoter Logic Heatmap
# ==============================================================================

def generate_ffl_logic():
    print("Generating FFL Promoter Logic...")
    
    # Grid of inputs
    x_vals = np.logspace(-2, 2, 50) # Activator
    y_vals = np.logspace(-2, 2, 50) # Repressor
    
    # Calculate Activity for Z Promoter
    # Logic: X activates, Y represses.
    # We use a Markovian equilibrium approx for the heatmap.
    # State: Bound/Unbound.
    # States: Empty, X, Y, XY.
    # Active if X only? (AND-NOT) -> Yes. X bound, Y not bound.
    
    # Rates/Weights:
    # W_0 = 1
    # W_x = (X/Kx)^n
    # W_y = (Y/Ky)^m
    # W_xy = W_x * W_y * cooperativity? Assume independent binding for FFL usually?
    # Or competitive?
    # I1-FFL usually implies independent binding sites.
    
    # Active Fraction = W_x / (1 + W_x + W_y + W_xy)
    
    activity = np.zeros((len(y_vals), len(x_vals)))
    
    Kx = 1.0; nx = 2.0
    Ky = 1.0; ny = 2.0
    
    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            wx = (x/Kx)**nx
            wy = (y/Ky)**ny
            wxy = wx * wy
            
            # AND-NOT Logic: Active only if X bound and Y not bound?
            # State "X" is active. State "XY" is repressed.
            
            Z = 1 + wx + wy + wxy
            act = wx / Z
            activity[i, j] = act
            
    # Plot Heatmap
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Log-Log Heatmap trick: plot usually needs linear axes for imshow.
    # We can use pcolormesh with log scale.
    c = ax.pcolormesh(X, Y, activity, cmap='viridis', vmin=0, vmax=1)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Activator [X]')
    ax.set_ylabel('Repressor [Y]')
    ax.set_title('I1-FFL Promoter Logic (AND-NOT)')
    
    # Add colorbar
    plt.colorbar(c, ax=ax, label='Promoter Activity')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_ffl_promoter_logic.png', dpi=300)
    print("Saved fig_ffl_promoter_logic.png")

# ==============================================================================
# 2. p53 Cooperativity
# ==============================================================================

class p53_Simple:
    def __init__(self, params):
        self.params = params
        self.p53 = 1.0
        self.Mdm2 = 1.0
        self.ATM = 0.0
        self.promoter_state = 0 # discrete
        self.n_sites = int(params.get('n', 2))
        
    def step(self, dt, signal):
        p = self.params
        rng = np.random
        
        # Promoter Logic
        k = self.promoter_state
        n = self.n_sites
        coop = p.get('cooperativity', 1.0)
        
        # Rates
        k_off = 10.0 # Fast dynamics for promoter
        k_on_base = 0.1 # Affinity
        
        # Bind
        if k < n:
            c_factor = coop if k > 0 else 1.0
            rate_bind = k_on_base * self.p53 * (n - k) * c_factor
            if rng.rand() < 1 - np.exp(-rate_bind * dt):
                self.promoter_state += 1
        # Unbind
        if k > 0:
            rate_unbind = k_off * k
            if rng.rand() < 1 - np.exp(-rate_unbind * dt):
                self.promoter_state -= 1
                
        # ODEs
        # ATM
        dATM = signal - 0.5 * self.ATM # simplified deact
        
        # Mdm2 (dep on promoter)
        activity = self.promoter_state / n
        dMdm2 = p['k_trans_Mdm2'] * activity - p['k_deg_Mdm2'] * self.Mdm2
        
        # p53 (dep on Mdm2)
        dp53 = p['k_syn_p53'] + p['k_act_ATM'] * self.ATM - p['k_deg_p53'] * self.p53 - p['k_ub'] * self.Mdm2 * self.p53
        
        self.ATM += dATM * dt
        self.Mdm2 += dMdm2 * dt
        self.p53 += dp53 * dt
        
        # Clip
        self.p53 = max(0, self.p53)
        self.Mdm2 = max(0, self.Mdm2)
        self.ATM = max(0, self.ATM)
        
    def run(self, T, dt=0.05):
        steps = int(T/dt)
        res = []
        t_vals = []
        for i in range(steps):
            t = i*dt
            signal = 1.0 if t > 10 else 0.0 # Step input
            self.step(dt, signal)
            res.append(self.p53)
            t_vals.append(t)
        return t_vals, res

def generate_p53_cooperativity():
    print("Generating p53 Cooperativity...")
    
    # Params
    params = {
        'k_trans_Mdm2': 100.0,
        'k_deg_Mdm2': 1.0,
        'k_syn_p53': 10.0,
        'k_act_ATM': 50.0,
        'k_deg_p53': 0.1,
        'k_ub': 0.5,
        'n': 2
    }
    
    coop_values = [1.0, 10.0, 100.0]
    
    fig, ax = plt.subplots(figsize=(6, 3))
    
    colors = ['#CC79A7', '#0072B2', '#D55E00'] # Purple, Blue, Red
    
    for i, c in enumerate(coop_values):
        params['cooperativity'] = c
        model = p53_Simple(params)
        t, p53 = model.run(100, 0.05) # 100 min
        
        ax.plot(t, p53, label=f'Cooperativity={c}', color=colors[i])
        
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('p53 Concentration')
    ax.set_title('Effect of Cooperativity on p53 Oscillations')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_p53_cooperativity.png', dpi=300)
    print("Saved fig_p53_cooperativity.png")

if __name__ == "__main__":
    generate_ffl_logic()
    generate_p53_cooperativity()
