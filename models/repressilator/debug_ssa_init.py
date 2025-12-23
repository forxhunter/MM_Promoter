
from repressilator_ssa import SSA_Repressilator
import numpy as np

params = {
    'k_trans': 0.5,
    'k_leak': 5e-4,
    'k_transl': 0.16,
    'k_fold': 1.0/60,
    'k_deg_m': np.log(2)/2,
    'k_deg_p': np.log(2)/600,
    'n': 2.0,
    'KM': 40.0,
    'k_burst': 0.05
}

sim = SSA_Repressilator(params, Omega=50.0)
print(f"SSA Initial State: {sim.state}")
print(f"SSA p1 Index 9: {sim.state[9]}")
print(f"SSA Omega: {sim.Omega}")
print(f"SSA Initial Concentration p1: {sim.state[9] / sim.Omega}")
