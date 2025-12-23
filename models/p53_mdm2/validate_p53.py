
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from p53_ode import p53_ODE
from p53_ssa import p53_SSA
from p53_markovian import p53_Markovian

def damage_pulse(t):
    """Damage pulse at start."""
    return 10.0 if t < 1.0 else 0.0

def run_validation():
    T = 40.0 # Hours
    dt = 0.01
    
    # Parameters (Lahav et al. 2004 approx)
    # Time units: Hours?
    # Period is ~5.5 hours.
    
    # Let's tune for oscillations.
    params = {
        'k_syn_p53': 0.1,      # Synthesis
        'k_act_ATM': 5.0,      # Strong activation by ATM
        'k_deg_p53': 0.1,      # Basal decay
        'k_ub': 5.0,          # Strong degradation by Mdm2
        
        'k_trans_Mdm2': 5.0,   # Synthesis
        'k_deg_Mdm2': 1.0,     # Fast decay needed for oscillations
        
        'k_deact_ATM': 0.5,    # Decay of damage signal
        
        'KF': 1.0,            # Dissociation const
        'n': 3.0,             # Cooperative (at least 3?)
        
        # For Markovian
        'k_on_p53': 10.0,     # Fast binding
        'k_off_p53': 10.0,    # k_off/k_on = 1 = KF
        'cooperativity': 2.0  # Positive coop
    }
    
    # Run ODE
    print("Running ODE...")
    ode_model = p53_ODE(params)
    ode_res = ode_model.run(T, dt, damage_pulse)
    
    # Run SSA
    print("Running SSA...")
    ssa_model = p53_SSA(params, Omega=100.0)
    ssa_res = ssa_model.run(T, dt, damage_pulse)
    
    # Run Markovian
    print("Running Markovian...")
    markov_model = p53_Markovian(params)
    markov_res = markov_model.run(T, dt, damage_pulse)
    
    # Plotting
    plt.figure(figsize=(12, 12))
    
    # ATM
    plt.subplot(4, 1, 1)
    plt.plot(ode_res['t'], ode_res['ATM'], 'k--', label='ATM (Signal)')
    plt.title('Damage Signal')
    plt.ylabel('[ATM]')
    plt.legend()
    
    # p53
    plt.subplot(4, 1, 2)
    plt.plot(ode_res['t'], ode_res['p53'], 'b-', label='ODE')
    plt.plot(ssa_res['t'], ssa_res['p53'], 'g.', alpha=0.3, label='SSA')
    plt.plot(markov_res['t'], markov_res['p53'], 'r-', alpha=0.7, label='Markovian')
    plt.title('p53 Dynamics')
    plt.ylabel('[p53]')
    plt.legend()
    
    # Mdm2
    plt.subplot(4, 1, 3)
    plt.plot(ode_res['t'], ode_res['Mdm2'], 'b-', label='ODE')
    plt.plot(ssa_res['t'], ssa_res['Mdm2'], 'g.', alpha=0.3, label='SSA')
    plt.plot(markov_res['t'], markov_res['Mdm2'], 'r-', alpha=0.7, label='Markovian')
    plt.title('Mdm2 Dynamics')
    plt.ylabel('[Mdm2]')
    plt.legend()
    
    # Phase Plane or Promoter?
    plt.subplot(4, 1, 4)
    plt.plot(markov_res['t'], markov_res['S'], 'm-', label='Promoter State')
    plt.title('Markovian Promoter Activity')
    plt.ylabel('Bound p53')
    plt.xlabel('Time (h)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('p53_comparison.png')
    print("Validation Complete. Saved to p53_comparison.png")

if __name__ == "__main__":
    run_validation()
