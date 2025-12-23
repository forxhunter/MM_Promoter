import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from toggle_ssa import SSA_Toggle
from toggle_markovian import MarkovianToggle

# Parameters
params = {
    'alpha1': 156.25,
    'alpha2': 15.6,
    'beta': 2.5,
    'gamma': 1.0,
    'k_burst': 1.0
}

# 1. ODE - Compute Hill functions for both promoters
def toggle_ode_func(y, t, params):
    u, v = y
    alpha1 = params['alpha1']
    alpha2 = params['alpha2']
    beta = params['beta']
    gamma = params['gamma']
    
    hill_u = 1.0 / (1 + v**beta)
    hill_v = 1.0 / (1 + u**gamma)
    
    du = alpha1 * hill_u - u
    dv = alpha2 * hill_v - v
    return [du, dv]

y0 = [150, 0]
t_ode = np.linspace(0, 100, 1000)
ode_sol = odeint(toggle_ode_func, y0, t_ode, args=(params,))

# Compute Hill functions
u_ode = ode_sol[:, 0]
v_ode = ode_sol[:, 1]
hill_u = 1.0 / (1 + v_ode**params['beta'])
hill_v = 1.0 / (1 + u_ode**params['gamma'])

# 2. SSA - Run and extract promoter states
print("Running Toggle SSA...")

class SSA_Toggle_WithPromoters(SSA_Toggle):
    def __init__(self, params, Omega=1.0):
        super().__init__(params, Omega)
        self.history_promoters = [[self.state[0], self.state[1]]]
    
    def run(self, T):
        alpha1 = self.params.get('alpha1', 156.25)
        alpha2 = self.params.get('alpha2', 15.6)
        beta = self.params.get('beta', 2.5)
        gamma = self.params.get('gamma', 1.0)
        k_burst = self.params.get('k_burst', 1.0)
        
        while self.time < T:
            Su, Sv, U, V = self.state
            
            r_su_on = k_burst * (1 - Su)
            r_su_off = k_burst * ((V/self.Omega)**beta) * Su
            r_sv_on = k_burst * (1 - Sv)
            r_sv_off = k_burst * ((U/self.Omega)**gamma) * Sv
            r_prod_u = alpha1 * self.Omega * Su
            r_prod_v = alpha2 * self.Omega * Sv
            r_deg_u = 1.0 * U
            r_deg_v = 1.0 * V
            
            rates = [r_su_on, r_su_off, r_sv_on, r_sv_off, r_prod_u, r_prod_v, r_deg_u, r_deg_v]
            total_rate = sum(rates)
            
            if total_rate == 0: break
            
            tau = -np.log(np.random.rand()) / total_rate
            if self.time + tau > T: break
            self.time += tau
            
            r = np.random.rand() * total_rate
            cumulative = 0
            for i, rate in enumerate(rates):
                cumulative += rate
                if r <= cumulative:
                    rxn_idx = i
                    break
            
            if rxn_idx == 0: self.state[0] = 1
            elif rxn_idx == 1: self.state[0] = 0
            elif rxn_idx == 2: self.state[1] = 1
            elif rxn_idx == 3: self.state[1] = 0
            elif rxn_idx == 4: self.state[2] += 1
            elif rxn_idx == 5: self.state[3] += 1
            elif rxn_idx == 6: self.state[2] -= 1
            elif rxn_idx == 7: self.state[3] -= 1
            
            self.history_t.append(self.time)
            self.history_u.append(self.state[2])
            self.history_v.append(self.state[3])
            self.history_promoters.append([self.state[0], self.state[1]])
        
        return np.array(self.history_t), np.array(self.history_u)/self.Omega, np.array(self.history_promoters)

ssa_mod = SSA_Toggle_WithPromoters(params, Omega=1.0)
t_ssa, u_ssa, promoters_ssa = ssa_mod.run(100)

# 3. Markovian
print("Running Toggle Markovian...")
model = MarkovianToggle(params)
model.ode_state = np.array([150.0, 0.0])
model.promoters = np.array([1, 0])
t_mark, ode_mark, promoters_mark = model.run(100, dt=0.05)

# Plot
fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)

titles = ['Promoter U', 'Promoter V']
for i in range(2):
    # ODE Hill
    axes[0, i].plot(t_ode, [hill_u, hill_v][i], 'k-', linewidth=1.5, alpha=0.8)
    axes[0, i].set_ylabel('Hill Function', fontsize=11)
    axes[0, i].set_title(titles[i], fontsize=12, fontweight='bold')
    axes[0, i].grid(True, alpha=0.3)
    axes[0, i].set_ylim([-0.1, 1.1])
    if i == 0:
        axes[0, i].text(-0.18, 0.5, 'ODE', transform=axes[0, i].transAxes, 
                        fontsize=13, fontweight='bold', va='center', rotation=90)
    
    # SSA
    axes[1, i].step(t_ssa, promoters_ssa[:, i], 'b-', linewidth=1.0, alpha=0.7, where='post')
    axes[1, i].set_ylabel('State (0/1)', fontsize=11)
    axes[1, i].grid(True, alpha=0.3)
    axes[1, i].set_ylim([-0.1, 1.1])
    if i == 0:
        axes[1, i].text(-0.18, 0.5, 'SSA', transform=axes[1, i].transAxes, 
                        fontsize=13, fontweight='bold', va='center', rotation=90)
    
    # Markovian
    axes[2, i].step(t_mark, promoters_mark[:, i], 'r-', linewidth=1.0, alpha=0.7, where='post')
    axes[2, i].set_ylabel('State (0/1)', fontsize=11)
    axes[2, i].set_xlabel('Time', fontsize=11)
    axes[2, i].grid(True, alpha=0.3)
    axes[2, i].set_ylim([-0.1, 1.1])
    if i == 0:
        axes[2, i].text(-0.18, 0.5, 'Markovian', transform=axes[2, i].transAxes, 
                        fontsize=13, fontweight='bold', va='center', rotation=90)

fig.suptitle('Toggle Switch: Promoter State Comparison', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('../20251218_toggle/toggle_promoter_state_comparison.png', dpi=150)
print("Saved ../20251218_toggle/toggle_promoter_state_comparison.png")

# Compute statistics for both promoters
def compute_promoter_similarity(ssa_trace, mark_trace, ssa_time, mark_time):
    """Compute statistical similarity between SSA and Markovian promoter traces"""
    from sklearn.metrics import cohen_kappa_score
    
    # Interpolate to common time grid
    t_common = np.linspace(max(ssa_time[0], mark_time[0]), 
                          min(ssa_time[-1], mark_time[-1]), 1000)
    
    ssa_interp = np.zeros(len(t_common))
    mark_interp = np.zeros(len(t_common))
    
    for i, t in enumerate(t_common):
        ssa_idx = np.argmin(np.abs(ssa_time - t))
        mark_idx = np.argmin(np.abs(mark_time - t))
        ssa_interp[i] = ssa_trace[ssa_idx]
        mark_interp[i] = mark_trace[mark_idx]
    
    kappa = cohen_kappa_score(ssa_interp, mark_interp)
    agreement = np.mean(ssa_interp == mark_interp)
    correlation = np.corrcoef(ssa_interp, mark_interp)[0, 1]
    
    return kappa, agreement, correlation

print("\n=== Promoter State Statistics ===")
for i, name in enumerate(['U', 'V']):
    kappa, agreement, corr = compute_promoter_similarity(
        promoters_ssa[:, i], promoters_mark[:, i], t_ssa, t_mark)
    print(f"\nPromoter {name}:")
    print(f"  Cohen's Kappa:  {kappa:.3f}")
    print(f"  Time Agreement: {agreement:.1%}")
    print(f"  Correlation:    {corr:.3f}")

# Save statistics
stats_df = pd.DataFrame({
    'Promoter': ['U', 'V'],
    'Cohens_Kappa': [compute_promoter_similarity(promoters_ssa[:, i], promoters_mark[:, i], t_ssa, t_mark)[0] for i in range(2)],
    'Time_Agreement': [compute_promoter_similarity(promoters_ssa[:, i], promoters_mark[:, i], t_ssa, t_mark)[1] for i in range(2)],
    'Correlation': [compute_promoter_similarity(promoters_ssa[:, i], promoters_mark[:, i], t_ssa, t_mark)[2] for i in range(2)]
})
stats_df.to_csv('../20251218_toggle/toggle_promoter_statistics.csv', index=False)
print("\nSaved ../20251218_toggle/toggle_promoter_statistics.csv")
