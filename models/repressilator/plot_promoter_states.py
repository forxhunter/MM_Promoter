import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from repressilator_ssa import SSA_Repressilator
from repressilator_ddm_markovian import MarkovianRepressilatorDDM

# Parameters
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

# 1. ODE - Compute Hill functions for all 3 promoters
def repressilator_ode(y, t, params):
    m1, p1, m2, p2, m3, p3 = y
    k_trans = params['k_trans']
    k_deg_m = params['k_deg_m']
    k_deg_p = params['k_deg_p']
    k_transl = params['k_transl']
    K = params['KM']
    n = params['n']
    k_leak = params['k_leak']
    
    # Hill functions (p2 represses p1, p3 represses p2, p1 represses p3)
    H1 = 1 / (1 + (p2/K)**n)
    H2 = 1 / (1 + (p3/K)**n)
    H3 = 1 / (1 + (p1/K)**n)
    
    dm1 = k_trans * (k_leak + (1-k_leak)*H1) - k_deg_m * m1
    dp1 = k_transl * m1 - k_deg_p * p1
    dm2 = k_trans * (k_leak + (1-k_leak)*H2) - k_deg_m * m2
    dp2 = k_transl * m2 - k_deg_p * p2
    dm3 = k_trans * (k_leak + (1-k_leak)*H3) - k_deg_m * m3
    dp3 = k_transl * m3 - k_deg_p * p3
    
    return [dm1, dp1, dm2, dp2, dm3, dp3]

y0 = [5, 1000, 0, 0, 0, 0]
t_ode = np.linspace(0, 5000, 5000)
ode_sol = odeint(repressilator_ode, y0, t_ode, args=(params,))

# Compute Hill functions
p_ode = ode_sol[:, [1, 3, 5]]  # p1, p2, p3
hill1 = 1 / (1 + (p_ode[:, 1] / params['KM'])**params['n'])
hill2 = 1 / (1 + (p_ode[:, 2] / params['KM'])**params['n'])
hill3 = 1 / (1 + (p_ode[:, 0] / params['KM'])**params['n'])

# 2. SSA - Run and extract promoter states
print("Running Repressilator SSA...")
ssa = SSA_Repressilator(params, Omega=50.0)
ssa.history_promoters = []

# Patch run method to track promoters
class SSA_Repressilator_WithPromoters(SSA_Repressilator):
    def __init__(self, params, Omega=50.0):
        super().__init__(params, Omega)
        self.history_promoters = [self.state[0:3].copy()]
    
    def run(self, T):
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
        
        repressor_indices = [11, 9, 10]  # p3->1, p1->2, p2->3
        
        while self.time < T:
            rates = []
            rxn_types = []
            
            for i in range(3):
                Si = self.state[i]
                mi = self.state[3+i]
                ui = self.state[6+i]
                pi = self.state[9+i]
                
                rep_idx = repressor_indices[i]
                R_count = self.state[rep_idx]
                R_conc = R_count / self.Omega
                
                r_on = k_burst * (1 - Si)
                rates.append(r_on)
                rxn_types.append(('on', i))
                
                r_off = k_burst * (R_conc / KM)**n * Si
                rates.append(r_off)
                rxn_types.append(('off', i))
                
                r_trans = k_trans * Si
                rates.append(r_trans)
                rxn_types.append(('trans', i))
                
                r_leak = k_leak
                rates.append(r_leak)
                rxn_types.append(('leak', i))
                
                r_deg_m = k_deg_m * mi
                rates.append(r_deg_m)
                rxn_types.append(('deg_m', i))
                
                r_transl = k_transl * mi
                rates.append(r_transl)
                rxn_types.append(('transl', i))
                
                r_fold = k_fold * ui
                rates.append(r_fold)
                rxn_types.append(('fold', i))
                
                r_deg_u = k_deg_p * ui
                rates.append(r_deg_u)
                rxn_types.append(('deg_u', i))
                
                r_deg_p = k_deg_p * pi
                rates.append(r_deg_p)
                rxn_types.append(('deg_p', i))
            
            total = sum(rates)
            if total == 0: break
            
            tau = -np.log(np.random.rand()) / total
            if self.time + tau > T: break
            self.time += tau
            
            r = np.random.rand() * total
            cum = 0
            for k, rate in enumerate(rates):
                cum += rate
                if r <= cum:
                    rxn_idx = k
                    break
            
            rtype, idx = rxn_types[rxn_idx]
            
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
            
            self.history_t.append(self.time)
            self.history_p.append(self.state[9:12].copy())
            self.history_promoters.append(self.state[0:3].copy())
            
        return np.array(self.history_t), np.array(self.history_p)/self.Omega, np.array(self.history_promoters)

ssa_mod = SSA_Repressilator_WithPromoters(params, Omega=50.0)
t_ssa, p_ssa, promoters_ssa = ssa_mod.run(5000)

# 3. Markovian
print("Running Repressilator Markovian...")
model = MarkovianRepressilatorDDM(params)
model.ode_state[0] = 5.0
model.ode_state[3] = 48.0
model.ode_state[6] = 1000.0
t_mark, ode_mark, promoters_mark = model.run(5000, dt=0.01)

# Plot
fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)

titles = ['Promoter 1 (cI)', 'Promoter 2 (LacI)', 'Promoter 3 (TetR)']
for i in range(3):
    # ODE Hill
    axes[0, i].plot(t_ode, [hill1, hill2, hill3][i], 'k-', linewidth=1.0, alpha=0.8)
    axes[0, i].set_ylabel('Hill Function', fontsize=10)
    axes[0, i].set_title(titles[i], fontsize=11, fontweight='bold')
    axes[0, i].grid(True, alpha=0.3)
    axes[0, i].set_ylim([-0.1, 1.1])
    if i == 0:
        axes[0, i].text(-0.15, 0.5, 'ODE', transform=axes[0, i].transAxes, 
                        fontsize=12, fontweight='bold', va='center', rotation=90)
    
    # SSA
    axes[1, i].step(t_ssa, promoters_ssa[:, i], 'b-', linewidth=0.8, alpha=0.7, where='post')
    axes[1, i].set_ylabel('State (0/1)', fontsize=10)
    axes[1, i].grid(True, alpha=0.3)
    axes[1, i].set_ylim([-0.1, 1.1])
    if i == 0:
        axes[1, i].text(-0.15, 0.5, 'SSA', transform=axes[1, i].transAxes, 
                        fontsize=12, fontweight='bold', va='center', rotation=90)
    
    # Markovian
    axes[2, i].step(t_mark, promoters_mark[:, i], 'r-', linewidth=0.8, alpha=0.7, where='post')
    axes[2, i].set_ylabel('State (0/1)', fontsize=10)
    axes[2, i].set_xlabel('Time (min)', fontsize=10)
    axes[2, i].grid(True, alpha=0.3)
    axes[2, i].set_ylim([-0.1, 1.1])
    if i == 0:
        axes[2, i].text(-0.15, 0.5, 'Markovian', transform=axes[2, i].transAxes, 
                        fontsize=12, fontweight='bold', va='center', rotation=90)

fig.suptitle('Repressilator: Promoter State Comparison', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('../20251218_repressilator/repressilator_promoter_state_comparison.png', dpi=150)
print("Saved ../20251218_repressilator/repressilator_promoter_state_comparison.png")
