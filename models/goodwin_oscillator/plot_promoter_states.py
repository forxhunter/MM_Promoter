import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from goodwin_ssa import SSA_Goodwin
from goodwin_markovian import MarkovianGoodwin

# Parameters (same for all models)
params = {
    'k1': 1.0, 'k2': 1.0, 'k3': 1.0,
    'k2': 0.1, 'k4': 0.1, 'k5': 0.5, 'k6': 0.1,
    'b1': 0.1, 'b2': 0.1, 'b3': 0.1,
    'n': 9.0, 'KM': 1.0,
    'k_burst': 2.0
}

# 1. ODE - Compute "Effective Promoter Activity" from Hill function
def goodwin_ode_func(y, t, params):
    x, y_var, z = y
    k1 = params['k1']
    k2 = params['k2']
    k3 = params['k3']
    b1 = params['b1']
    b2 = params['b2']
    b3 = params['b3']
    n = params['n']
    KM = params['KM']
    
    # Hill function for promoter activity
    hill = 1.0 / (1.0 + (z/KM)**n)
    
    dx = k1 * hill - b1 * x
    dy = k2 * x - b2 * y_var
    dz = k3 * y_var - b3 * z
    
    return [dx, dy, dz]

y0 = [0.1, 0.1, 0.1]
t_ode = np.linspace(0, 500, 5000)
ode_sol = odeint(goodwin_ode_func, y0, t_ode, args=(params,))

# Compute Hill function (effective promoter activity) for ODE
z_ode = ode_sol[:, 2]
hill_ode = 1.0 / (1.0 + (z_ode / params['KM'])**params['n'])

# 2. SSA - Extract promoter state
print("Running Goodwin SSA...")
ssa = SSA_Goodwin(params, Omega=50.0)
ssa.history_promoter = []  # Add promoter tracking
ssa_t_raw = []
ssa_promoter_raw = []

# Modify SSA to track promoter (we need to patch the class or re-run with tracking)
# For simplicity, I'll run it and extract from history
# Actually, the SSA class doesn't save promoter history. Let me create a modified version.

class SSA_Goodwin_WithPromoter(SSA_Goodwin):
    def __init__(self, params, Omega=10.0):
        super().__init__(params, Omega)
        self.history_promoter = [self.state[0]]
    
    def run(self, T):
        p = self.params
        k_burst = p.get('k_burst', 2.0)
        n = p.get('n', 10.0)
        KM = p.get('KM', 1.0)
        
        k1 = p['k1'] * self.Omega
        k2 = p['k2']
        k3 = p['k3']
        
        b1 = p['b1']
        b2 = p['b2']
        b3 = p['b3']
        
        while self.time < T:
            Sx, X, Y, Z = self.state
            z_conc = Z / self.Omega
            
            r_on = k_burst * (1 - Sx)
            r_off = k_burst * (z_conc / KM)**n * Sx
            r_trans = k1 * Sx
            r_deg_x = b1 * X
            r_transl = k2 * X
            r_deg_y = b2 * Y
            r_prod_z = k3 * Y
            r_deg_z = b3 * Z
            
            rates = [r_on, r_off, r_trans, r_deg_x, r_transl, r_deg_y, r_prod_z, r_deg_z]
            total = sum(rates)
            
            if total == 0: break
            
            tau = -np.log(np.random.rand()) / total
            if self.time + tau > T: break
            self.time += tau
            
            r = np.random.rand() * total
            cum = 0
            for i, rate in enumerate(rates):
                cum += rate
                if r <= cum:
                    rxn = i
                    break
                    
            if rxn == 0: self.state[0] = 1
            elif rxn == 1: self.state[0] = 0
            elif rxn == 2: self.state[1] += 1
            elif rxn == 3: self.state[1] -= 1
            elif rxn == 4: self.state[2] += 1
            elif rxn == 5: self.state[2] -= 1
            elif rxn == 6: self.state[3] += 1
            elif rxn == 7: self.state[3] -= 1
            
            self.history_t.append(self.time)
            self.history_x.append(self.state[1])
            self.history_promoter.append(self.state[0])
            
        return np.array(self.history_t), np.array(self.history_x)/self.Omega, np.array(self.history_promoter)

ssa_mod = SSA_Goodwin_WithPromoter(params, Omega=50.0)
t_ssa, x_ssa, promoter_ssa = ssa_mod.run(500)

# 3. Markovian - Extract promoter state
print("Running Goodwin Markovian...")
model = MarkovianGoodwin(params)
t_mark, ode_mark, promoter_mark = model.run(500, dt=0.001)

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Plot 1: ODE Hill Function (Effective Promoter Activity)
axes[0].plot(t_ode, hill_ode, 'k-', linewidth=1.5, label='ODE (Hill Function)')
axes[0].set_ylabel('Promoter Activity\n(Hill Function)', fontsize=11)
axes[0].set_title('Goodwin Oscillator: Promoter State Comparison', fontsize=13, fontweight='bold')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([-0.1, 1.1])

# Plot 2: SSA Discrete Promoter
axes[1].step(t_ssa, promoter_ssa, 'b-', linewidth=1.0, alpha=0.7, label='SSA (Discrete State)', where='post')
axes[1].set_ylabel('Promoter State\n(0=OFF, 1=ON)', fontsize=11)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([-0.1, 1.1])

# Plot 3: Markovian Discrete Promoter
axes[2].step(t_mark, promoter_mark, 'r-', linewidth=1.0, alpha=0.7, label='Markovian (Discrete State)', where='post')
axes[2].set_ylabel('Promoter State\n(0=OFF, 1=ON)', fontsize=11)
axes[2].set_xlabel('Time', fontsize=11)
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim([-0.1, 1.1])

plt.tight_layout()
plt.savefig('../20251218_goodwin/goodwin_promoter_state_comparison.png', dpi=150)
print("Saved ../20251218_goodwin/goodwin_promoter_state_comparison.png")

# Compute switching statistics
def compute_switching_stats(promoter_trace, time_trace):
    """Compute ON/OFF durations and switching frequency"""
    switches = np.diff(promoter_trace)
    switch_times = time_trace[1:][switches != 0]
    
    if len(switch_times) < 2:
        return 0, 0, 0
    
    switch_freq = len(switch_times) / (time_trace[-1] - time_trace[0])
    
    # ON durations
    on_starts = time_trace[1:][switches == 1]
    on_ends = time_trace[1:][switches == -1]
    if len(on_ends) > 0 and len(on_starts) > 0:
        if on_ends[0] < on_starts[0]:
            on_ends = on_ends[1:]
        min_len = min(len(on_starts), len(on_ends))
        on_durations = on_ends[:min_len] - on_starts[:min_len]
        mean_on = np.mean(on_durations) if len(on_durations) > 0 else 0
    else:
        mean_on = 0
    
    return switch_freq, mean_on, len(switch_times)

def compute_promoter_similarity(ssa_trace, mark_trace, ssa_time, mark_time):
    """Compute statistical similarity between SSA and Markovian promoter traces"""
    # Interpolate to common time grid
    t_common = np.linspace(max(ssa_time[0], mark_time[0]), 
                          min(ssa_time[-1], mark_time[-1]), 1000)
    
    # Nearest neighbor interpolation (for discrete states)
    ssa_interp = np.zeros(len(t_common))
    mark_interp = np.zeros(len(t_common))
    
    for i, t in enumerate(t_common):
        ssa_idx = np.argmin(np.abs(ssa_time - t))
        mark_idx = np.argmin(np.abs(mark_time - t))
        ssa_interp[i] = ssa_trace[ssa_idx]
        mark_interp[i] = mark_trace[mark_idx]
    
    # Cohen's Kappa (agreement for categorical data)
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(ssa_interp, mark_interp)
    
    # Fraction of time in agreement
    agreement = np.mean(ssa_interp == mark_interp)
    
    # Correlation
    correlation = np.corrcoef(ssa_interp, mark_interp)[0, 1]
    
    return kappa, agreement, correlation

ssa_freq, ssa_on, ssa_n = compute_switching_stats(promoter_ssa, t_ssa)
mark_freq, mark_on, mark_n = compute_switching_stats(promoter_mark, t_mark)

kappa, agreement, correlation = compute_promoter_similarity(promoter_ssa, promoter_mark, t_ssa, t_mark)

print(f"\n=== Promoter Switching Statistics ===")
print(f"SSA:       {ssa_n} switches, freq={ssa_freq:.3f} Hz, mean_ON={ssa_on:.2f}")
print(f"Markovian: {mark_n} switches, freq={mark_freq:.3f} Hz, mean_ON={mark_on:.2f}")
print(f"\n=== Statistical Similarity (SSA vs Markovian) ===")
print(f"Cohen's Kappa:     {kappa:.3f} (>0.6 = substantial agreement)")
print(f"Time Agreement:    {agreement:.1%} (fraction of time in same state)")
print(f"Correlation:       {correlation:.3f}")

# Save statistics to CSV
stats_df = pd.DataFrame({
    'Metric': ['SSA_Switches', 'SSA_Freq_Hz', 'SSA_MeanON', 
               'Mark_Switches', 'Mark_Freq_Hz', 'Mark_MeanON',
               'Cohens_Kappa', 'Time_Agreement', 'Correlation'],
    'Value': [ssa_n, ssa_freq, ssa_on, mark_n, mark_freq, mark_on,
              kappa, agreement, correlation]
})
stats_df.to_csv('../20251218_goodwin/goodwin_promoter_statistics.csv', index=False)
print("\nSaved ../20251218_goodwin/goodwin_promoter_statistics.csv")
