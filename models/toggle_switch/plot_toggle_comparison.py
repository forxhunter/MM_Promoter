
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from toggle_ssa import SSA_Toggle

# 1. Deterministic ODE
def toggle_ode_func(y, t, params):
    # u, v
    u, v = y
    alpha1 = params['alpha1']
    alpha2 = params['alpha2']
    beta = params['beta']
    gamma = params['gamma']
    
    # Simple Hill: 
    # du/dt = alpha1 / (1 + v^beta) - u
    # dv/dt = alpha2 / (1 + u^gamma) - v
    # Note: SSA uses (V/Omega)**beta. Here variables are counts/concentration?
    # Let's assume params are effective rates.
    du = alpha1 / (1 + v**beta) - u
    dv = alpha2 / (1 + u**gamma) - v
    return [du, dv]

params = {
    'alpha1': 156.25,
    'alpha2': 15.6,
    'beta': 2.5,
    'gamma': 1.0, 
    'k_burst': 1.0
}
y0 = [150, 0] # Matches Markovian/SSA (High U)
t_ode = np.linspace(0, 100, 1000)
ode_sol = odeint(toggle_ode_func, y0, t_ode, args=(params,))

# 2. SSA (Ground Truth)
print("Running Toggle SSA...")
# Use Omega=1.0 per original script
ssa = SSA_Toggle(params, Omega=1.0)
t_ssa, u_ssa, v_ssa = ssa.run(100)

# 3. Markovian Hybrid
# Load from 'toggle_markovian_trajectories.csv' if exists, else skip
try:
    mark_df = pd.read_csv('../20251218_toggle/toggle_markovian_trajectories.csv')
    # Use first 100 s
    mark_df = mark_df[mark_df['Time'] <= 100]
    t_mark = mark_df['Time']
    u_mark = mark_df['U_Run1']
except:
    print("Markovian Data not found. Skipping.")
    t_mark, u_mark = [], []

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t_ode, ode_sol[:, 0], 'k-', linewidth=2, label='Deterministic ODE')
# SSA U is normalized by Omega? Yes in script: numpy.array(..)/self.Omega
plt.plot(t_ssa, u_ssa, 'b-', alpha=0.5, label='SSA (Ground Truth)')

if len(t_mark) > 0:
    plt.plot(t_mark, u_mark, 'r-', alpha=0.5, label='Markovian Hybrid')

plt.title('Toggle Switch Trajectory Comparison (Protein U)')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../20251218_toggle/toggle_trajectory_comparison.png', dpi=150)
print("Saved ../20251218_toggle/toggle_trajectory_comparison.png")
