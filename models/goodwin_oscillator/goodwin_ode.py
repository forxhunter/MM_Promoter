import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Reference: Goodwin Oscillator
# Source: https://github.com/youngmp/strongcoupling (2021)
# Paper: "Higher-order coupling functions of oscillators"
# Model: 3-species negative feedback loop (mRNA -> Enzyme -> Repressor)

def goodwin_oscillator(t, state, params):
    x, y, z = state
    
    # Parameters (Standard Goodwin)
    # Production of X is repressed by Z (Hill)
    # k1/(1 + z^n)
    
    k1 = params.get('k1', 1.0) # mRNA prod
    k2 = params.get('k2', 1.0) # Enzyme prod
    k3 = params.get('k3', 1.0) # Repressor prod
    
    b1 = params.get('b1', 0.1) # Degradation
    b2 = params.get('b2', 0.1)
    b3 = params.get('b3', 0.1)
    
    n = params.get('n', 10.0) # Hill Coeff (High for oscillation)
    KM = params.get('KM', 1.0)
    
    dxdt = k1 / (1 + (z/KM)**n) - b1*x
    dydt = k2 * x - b2*y
    dzdt = k3 * y - b3*z
    
    return [dxdt, dydt, dzdt]

# Parameters that produce oscillations
params = {
    'k1': 1.0, 'k2': 1.0, 'k3': 1.0,
    'b1': 0.1, 'b2': 0.1, 'b3': 0.1,
    'n': 10.0, 'KM': 1.0
}

x0 = [0.1, 0.1, 0.1] # Start low
# Run long enough to settle into limit cycle
t_span = (0, 500)
t_eval = np.linspace(0, 500, 2000)

sol = solve_ivp(lambda t, y: goodwin_oscillator(t, y, params), t_span, x0, t_eval=t_eval, method='LSODA')

# Plot
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='X (mRNA)')
plt.plot(sol.t, sol.y[1], label='Y (Protein)')
plt.plot(sol.t, sol.y[2], label='Z (Repressor)')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Goodwin Oscillator (Ref: youngmp/strongcoupling)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('goodwin_ode.png')

# Save Trajectories
import pandas as pd
df = pd.DataFrame({'Time': sol.t, 'X': sol.y[0], 'Y': sol.y[1], 'Z': sol.y[2]})
df.to_csv('goodwin_ode_trajectories.csv', index=False)
