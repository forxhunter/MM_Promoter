import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
import seaborn as sns
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
# 1. SSA Model (Ground Truth)
# ==============================================================================

class SSA_Repressilator:
    def __init__(self, params, Omega=1.0):
        self.params = params
        self.Omega = Omega
        self.state = np.zeros(12, dtype=int)
        self.state[0:3] = 1 
        self.state[3] = int(5.0 * Omega) 
        self.state[6] = int(48.0 * Omega)
        self.state[9] = int(1000.0 * Omega)
        self.time = 0.0
        self.history_p = [] # Just store protein counts for distribution

    def run(self, T, sample_interval=1.0):
        p = self.params
        k_burst = p.get('k_burst', 0.05)
        KM = p.get('KM', 40.0)
        n = p.get('n', 2.0)
        
        k_trans = p['k_trans'] * self.Omega
        k_leak = p['k_leak'] * self.Omega
        k_deg_m = p['k_deg_m']
        k_transl = p['k_transl']
        k_fold = p['k_fold']
        k_deg_p = p['k_deg_p']
        
        repressor_indices = [11, 9, 10]
        
        next_sample = 0.0
        
        while self.time < T:
            rates = []
            reaction_types = []
            
            for i in range(3):
                Si = self.state[i]
                mi = self.state[3+i]
                ui = self.state[6+i]
                pi = self.state[9+i]
                
                rep_idx = repressor_indices[i]
                R_count = self.state[rep_idx]
                R_conc = R_count / self.Omega
                
                # Rates
                r_on = k_burst * (1 - Si)
                r_off = k_burst * (R_conc / KM)**n * Si
                r_trans = k_trans * Si
                r_leak = k_leak
                r_deg_m = k_deg_m * mi
                r_transl = k_transl * mi
                r_fold = k_fold * ui
                r_deg_u = k_deg_p * ui
                r_deg_p = k_deg_p * pi
                
                # Append
                rates.extend([r_on, r_off, r_trans, r_leak, r_deg_m, r_transl, r_fold, r_deg_u, r_deg_p])
                reaction_types.extend([('on',i), ('off',i), ('trans',i), ('leak',i), ('deg_m',i), 
                                       ('transl',i), ('fold',i), ('deg_u',i), ('deg_p',i)])
                
            total = sum(rates)
            if total == 0: break
            
            # Step
            dt = -np.log(np.random.rand()) / total
            self.time += dt
            
            # Sample reaction
            r = np.random.rand() * total
            cum = 0
            for k, rate in enumerate(rates):
                cum += rate
                if r <= cum:
                    rxn_idx = k
                    break
            
            rtype, idx = reaction_types[rxn_idx]
            
            if rtype == 'on': self.state[idx] = 1
            elif rtype == 'off': self.state[idx] = 0
            elif rtype == 'trans': self.state[3+idx] += 1
            elif rtype == 'leak': self.state[3+idx] += 1
            elif rtype == 'deg_m': self.state[3+idx] -= 1
            elif rtype == 'transl': self.state[6+idx] += 1
            elif rtype == 'fold': 
                self.state[6+idx] -= 1
                self.state[9+idx] += 1
            elif rtype == 'deg_u': self.state[6+idx] -= 1
            elif rtype == 'deg_p': self.state[9+idx] -= 1
            
            if self.time >= next_sample:
                self.history_p.append(self.state[9:12].copy()) # Only Proteins
                next_sample += sample_interval
                
        return np.array(self.history_p)/self.Omega

# ==============================================================================
# 2. Hybrid Model (Stochastic Promoter, Deterministic Protein)
# ==============================================================================

class MarkovianRepressilator:
    def __init__(self, params):
        self.params = params
        self.time = 0.0
        self.ode_state = np.zeros(9)
        self.promoter_state = np.ones(3, dtype=int)
        self.history_p = []
        
        # Initialize
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

    def run(self, total_time, dt, sample_interval=1.0):
        steps = int(total_time / dt)
        next_sample = 0.0
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
            # Markovian Step
            p_conc = self.ode_state[6:9]
            rates = self.transition_rates(p_conc)
            for j in range(3):
                k_on, k_off = rates[j]
                if self.promoter_state[j] == 0:
                    if rng.random() < 1 - np.exp(-k_on * dt): self.promoter_state[j] = 1
                else:
                    if rng.random() < 1 - np.exp(-k_off * dt): self.promoter_state[j] = 0
            
            # ODE Step
            sol = odeint(dxdt, self.ode_state, [0, dt], args=(self.promoter_state,))
            self.ode_state = sol[-1]
            self.time += dt
            
            if self.time >= next_sample:
                self.history_p.append(self.ode_state[6:9].copy())
                next_sample += sample_interval
                
        return np.array(self.history_p)

# ==============================================================================
# Analysis
# ==============================================================================

def run_analysis():
    params = {
        'k_trans': 0.5,
        'k_leak': 5e-4,
        'k_deg_m': np.log(2)/2, 
        'k_transl': 0.16,
        'k_fold': 1.0/60,
        'k_deg_p': np.log(2)/600,
        'n': 2.0,
        'KM': 40.0,
        'k_burst': 0.05 
    }
    
    # Run short simulations to save time (but long enough for distribution)
    # T=2000 min should be enough?
    T = 2000
    
    print("Running Hybrid Model...")
    hybrid_model = MarkovianRepressilator(params)
    hybrid_p = hybrid_model.run(T, dt=0.5, sample_interval=1.0) # 2000 points
    
    print("Running SSA Model (Omega=1.0)...")
    ssa_model = SSA_Repressilator(params, Omega=1.0)
    ssa_p = ssa_model.run(T, sample_interval=1.0)
    
    # Analyze (Distribution of One Protein, e.g. p1)
    # Use only second half to ensure steady state (though Repressilator oscillates...)
    # Actually, comparing distributions of oscillating systems is tricky if period/phase differs.
    # But if we look at the distribution of values over many cycles, it represents the "occupancy".
    
    h_vals = hybrid_p[500:, 0] # p1
    s_vals = ssa_p[500:, 0]    # p1
    
    # CV
    cv_h = np.std(h_vals) / np.mean(h_vals)
    cv_s = np.std(s_vals) / np.mean(s_vals)
    
    print(f"Hybrid CV: {cv_h:.4f}")
    print(f"SSA CV: {cv_s:.4f}")
    
    # Plot Histograms / KDE
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    sns.kdeplot(s_vals, fill=True, color='green', alpha=0.3, label=f'SSA (CV={cv_s:.2f})', ax=ax)
    sns.kdeplot(h_vals, fill=True, color='red', alpha=0.3, label=f'Hybrid (CV={cv_h:.2f})', ax=ax)
    
    ax.set_xlabel('Protein Abundance (molecules)')
    ax.set_ylabel('Density')
    ax.set_title('Protein Stochasticity Comparison')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_protein_stochasticity.png', dpi=300)
    print("Saved fig_protein_stochasticity.png")
    
    # Save Data
    df_h = pd.DataFrame({'Protein': h_vals, 'Method': 'Hybrid'})
    df_s = pd.DataFrame({'Protein': s_vals, 'Method': 'SSA'})
    pd.concat([df_h, df_s]).to_csv('protein_stochasticity_data.csv', index=False)

if __name__ == "__main__":
    run_analysis()
