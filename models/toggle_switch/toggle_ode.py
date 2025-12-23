import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Reference: Genetic Toggle Switch
# Source: https://github.com/ykayuwu/Bistability-Project-Code (Accessed Dec 2025)
# Paper: Gardner et al., Nature (2000)
# Model: Two-gene mutual repression.
# equations:
# du/dt = alpha1 / (1 + v^beta) - u
# dv/dt = alpha2 / (1 + u^gamma) - v

def toggle_switch(t, x, params):
    u, v = x
    
    alpha1 = params.get('alpha1', 156.25)
    alpha2 = params.get('alpha2', 15.6)
    beta = params.get('beta', 2.5)
    gamma = params.get('gamma', 1.0)
    
    # Mutual Repression (Hill Functions)
    du = alpha1 / (1 + v**beta) - u
    dv = alpha2 / (1 + u**gamma) - v
    
    return [du, dv]

# Parameters (Bistable Regime)
params = {
    'alpha1': 156.25,
    'alpha2': 15.6,
    'beta': 2.5,
    'gamma': 1.0
}

# Run 1: Start High U
x0_1 = [150, 0]
sol1 = solve_ivp(lambda t, y: toggle_switch(t, y, params), (0, 100), x0_1, method='LSODA', t_eval=np.linspace(0, 100, 1000))

# Run 2: Start High V
x0_2 = [0, 15] # Note alpha2 is lower, so V steady state is lower
sol2 = solve_ivp(lambda t, y: toggle_switch(t, y, params), (0, 100), x0_2, method='LSODA', t_eval=np.linspace(0, 100, 1000))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(sol1.t, sol1.y[0], 'b-', label='U (Init High U)')
plt.plot(sol1.t, sol1.y[1], 'b--', label='V (Init High U)')
plt.plot(sol2.t, sol2.y[0], 'r-', label='U (Init High V)')
plt.plot(sol2.t, sol2.y[1], 'r--', label='V (Init High V)')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Genetic Toggle Switch (Ref: ykayuwu/Bistability-Project-Code)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('toggle_ode.png')

# Save Validation Data
import pandas as pd
df1 = pd.DataFrame({'Time': sol1.t, 'U': sol1.y[0], 'V': sol1.y[1], 'Condition': 'High_U'})
df2 = pd.DataFrame({'Time': sol2.t, 'U': sol2.y[0], 'V': sol2.y[1], 'Condition': 'High_V'})
pd.concat([df1, df2]).to_csv('toggle_ode_trajectories.csv', index=False)
