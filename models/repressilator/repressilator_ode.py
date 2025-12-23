import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Dynamics for the repressilator (Modified for SciPy)
def repressilator(t, x, params):
    # Unpack state
    # mRNA: m1(cI), m2(lacI), m3(tetR)
    # Protein: p1(cI), p2(lacI), p3(tetR)
    m1, m2, m3, p1, p2, p3 = x

    # Parameters
    k_trans = params.get('k_trans', 0.5)      # Transcription rate
    k_leak  = params.get('k_leak', 5e-4)      # Leakage
    
    # Half-lives -> Degradation rates
    t_half_m = params.get('t_half_m', 120)    # seconds
    k_dm = np.log(2)/t_half_m
    
    t_half_p = params.get('t_half_p', 600)    # seconds
    k_dp = np.log(2)/t_half_p
    
    # Translation
    trans_eff = params.get('trans_eff', 20)   # prot per transcript
    # k_transl = trans_eff * k_dm ? No. average lifespan is 1/k_dm. Use code logic:
    # k_translation = translation_efficiency/average_mRNA_lifespan
    avg_life = 1/k_dm if k_dm > 0 else 0
    k_transl = trans_eff / avg_life if avg_life > 0 else 0

    # Regulation
    n = params.get('n', 2.0)                  # Hill Coeff
    KM = params.get('KM', 40.0)               # Dissociation constant

    # ODEs
    # Structure: 1 inhibits 2, 2 inhibits 3, 3 inhibits 1
    # Check paper:
    # LacI(p2) inhibits TetR(m3)?
    # TetR(p3) inhibits cI(m1)?
    # cI(p1) inhibits LacI(m2)?
    # Reference code:
    # dxdt[0] (m_cI) suppressed by protein_tetR (p3). Correct.
    # dxdt[1] (m_lacI) suppressed by protein_cI (p1). Correct.
    # dxdt[2] (m_tetR) suppressed by protein_lacI (p2). Correct.

    dm1 = k_trans / (1 + (p3 / KM)**n) - k_dm * m1 + k_leak
    dm2 = k_trans / (1 + (p1 / KM)**n) - k_dm * m2 + k_leak
    dm3 = k_trans / (1 + (p2 / KM)**n) - k_dm * m3 + k_leak

    dp1 = k_transl * m1 - k_dp * p1
    dp2 = k_transl * m2 - k_dp * p2
    dp3 = k_transl * m3 - k_dp * p3

    return [dm1, dm2, dm3, dp1, dp2, dp3]

# Parameters
params = {
    'k_trans': 0.5,
    'k_leak': 5e-4,
    't_half_m': 120,
    't_half_p': 600,
    'trans_eff': 20,
    'n': 2.0,
    'KM': 40.0
}

# Initial Conditions (Asymmetric to start oscillation)
x0 = [0, 0, 0, 5, 0, 0] # p1=5, others 0

# Time
t_span = (0, 20000) # seconds (~330 mins)
t_eval = np.linspace(0, 20000, 2000)

# Solve
sol = solve_ivp(lambda t, y: repressilator(t, y, params), t_span, x0, t_eval=t_eval, method='LSODA')

# Plot
plt.figure(figsize=(10, 6))
plt.plot(sol.t/60, sol.y[3], label='cI (p1)')
plt.plot(sol.t/60, sol.y[4], label='LacI (p2)')
plt.plot(sol.t/60, sol.y[5], label='TetR (p3)')
plt.xlabel('Time (min)')
plt.ylabel('Proteins per cell')
plt.title('Reference Repressilator ODE (Murray 2024)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('repressilator_ode.png')
print("Simulation complete. Final values:", sol.y[:, -1])
