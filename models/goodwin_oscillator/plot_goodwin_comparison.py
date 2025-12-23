
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from goodwin_ssa import SSA_Goodwin

# 1. Deterministic ODE
def goodwin_ode_func(y, t, params):
    # x, y, z
    # dx = k1 / (1 + z^n) - b1 * x
    # dy = k2 * x - b2 * y
    # dz = k3 * y - b3 * z
    X, Y, Z = y
    k1 = params['k1']
    k2 = params['k2']
    k3 = params['k3']
    b1 = params['b1']
    b2 = params['b2']
    b3 = params['b3']
    n = params['n']
    KM = params.get('KM', 1.0)
    
    hill = 1.0 / (1.0 + (Z/KM)**n)
    
    dx = k1 * hill - b1 * X
    dy = k2 * X - b2 * Y
    dz = k3 * Y - b3 * Z
    return [dx, dy, dz]

params = {
    'k1': 1.0,     # Transcription max rate (CORRECTED from 10.0)
    'k2': 1.0,     # Translation (CORRECTED from 0.5)
    'k3': 1.0,     # Enzymatic/Transport (CORRECTED from 0.5)
    'b1': 0.1,     # Deg X
    'b2': 0.1,     # Deg Y
    'b3': 0.1,     # Deg Z
    'n': 10.0,     # Hill Coeff (matches goodwin_markovian.py)
    'k_burst': 2.0 # For Hybrid
}
y0 = [0.1, 0.1, 0.1]
t_ode = np.linspace(0, 500, 1000)
ode_sol = odeint(goodwin_ode_func, y0, t_ode, args=(params,))

# 2. SSA (Ground Truth)
print("Running Goodwin SSA...")
# Use Omega=50 (Optimized)
ssa = SSA_Goodwin(params, Omega=50.0)
t_ssa, x_ssa = ssa.run(500)

# 3. Markovian Hybrid
try:
    mark_df = pd.read_csv('../20251218_goodwin/goodwin_markovian_trajectories.csv')
    mark_df = mark_df[mark_df['Time'] <= 500]
    t_mark = mark_df['Time']
    x_mark = mark_df['X']
except:
    print("Markovian Data not found. Skipping.")
    t_mark, x_mark = [], []

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t_ode, ode_sol[:, 0], 'k-', linewidth=2, label='Deterministic ODE')
plt.plot(t_ssa, x_ssa, 'b-', alpha=0.5, label='SSA (Ground Truth)')

if len(t_mark) > 0:
    plt.plot(t_mark, x_mark, 'r-', alpha=0.5, label='Markovian Hybrid')

plt.title('Goodwin Oscillator Trajectory Comparison (mRNA X)')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../20251218_goodwin/goodwin_trajectory_comparison.png', dpi=150)
print("Saved ../20251218_goodwin/goodwin_trajectory_comparison.png")
