import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Dynamic Delay Model (DDM) Repressilator (Re-created)
# Model: 3 Genes.
# Pathway: Gene -> mRNA -> Unfolded -> Folded (Active Repressor).

def repressilator_ddm(t, x, params):
    # Unpack state (9 variables)
    # 0,1,2: mRNA (m1, m2, m3)
    # 3,4,5: Unfolded (u1, u2, u3)
    # 6,7,8: Folded (p1, p2, p3)
    m = x[0:3]
    u = x[3:6]
    p = x[6:9]

    k_trans = params.get('k_trans', 0.5)
    k_leak = params.get('k_leak', 5e-4)
    k_deg_m = params.get('k_deg_m', np.log(2)/120)
    
    k_transl = params.get('k_transl', 0.16)
    k_fold = params.get('k_fold', 1.0/60)
    k_deg_p = params.get('k_deg_p', np.log(2)/600)
    
    n = params.get('n', 2.0)
    KM = params.get('KM', 40.0)

    repressors = [p[2], p[0], p[1]]

    dm = np.zeros(3)
    du = np.zeros(3)
    dp = np.zeros(3)

    for i in range(3):
        # Transcription (Hill)
        dm[i] = k_trans / (1 + (repressors[i] / KM)**n) - k_deg_m * m[i] + k_leak
        du[i] = k_transl * m[i] - k_fold * u[i] - k_deg_p * u[i]
        dp[i] = k_fold * u[i] - k_deg_p * p[i]

    return np.concatenate([dm, du, dp])

# Run Simulation
params = {
    'k_trans': 0.5,
    'k_leak': 5e-4,
    'n': 2.0,
    'KM': 40.0,
    'k_fold': 0.1
}

x0 = np.zeros(9)
x0[0] = 5.0 # Start with some mRNA 1

t_eval = np.linspace(0, 10000, 1000)
sol = solve_ivp(lambda t, y: repressilator_ddm(t, y, params), (0, 10000), x0, t_eval=t_eval, method='LSODA')

# Plot
plt.figure(figsize=(10, 6))
plt.plot(sol.t/60, sol.y[6], label='cI (Folded)')
plt.plot(sol.t/60, sol.y[7], label='LacI (Folded)')
plt.plot(sol.t/60, sol.y[8], label='TetR (Folded)')
plt.xlabel('Time (min)')
plt.ylabel('Active Repressor Molecules')
plt.title('Dynamic Delay Model (DDM) Repressilator')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('repressilator_ddm.png')
