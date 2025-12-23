
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from i1_ffl_ode import I1_FFL_ODE
from i1_ffl_ssa import I1_FFL_SSA
from i1_ffl_markovian import I1_FFL_Markovian

def step_input(t):
    """Step input of X at t=10."""
    return 10.0 if t >= 10.0 else 0.0

def run_validation():
    T = 50.0
    dt = 0.05
    
    # Parameters
    # Designed to show Pulse
    # ODE Parameters
    ode_params = {
        'beta_y': 2.0,
        'alpha_y': 0.1,  # Slow decay of Y -> delayed repression
        'K_xy': 1.0,
        
        'beta_z': 2.0,
        'alpha_z': 0.5, # Fast Z dynamics
        'K_xz': 1.0,
        'K_yz': 2.0
    }
    
    # SSA Parameters (Omega=100 for decent counts)
    # Reuse ODE params directly interpretation
    ssa_params = ode_params.copy()
    
    # Markovian Parameters
    # Convert Hill K to k_on/k_off
    # K_d = k_off / k_on
    # Let k_off = 1.0, then k_on = 1/K
    markov_params = {
        'beta_y': 2.0, 'alpha_y': 0.1,
        'beta_z': 2.0, 'alpha_z': 0.5,
        
        # Y promoter
        'k_on_xy': 1.0, # K_xy=1 => k_on=1
        'k_off_xy': 1.0,
        
        # Z promoter
        'k_on_xz': 1.0, # K_xz=1 => k_on=1
        'k_off_xz': 1.0,
        
        'k_on_yz': 0.5, # K_yz=2 => k_on=0.5 (if k_off=1)
        'k_off_yz': 1.0
    }
    
    # Run ODE
    print("Running ODE...")
    ode_model = I1_FFL_ODE(ode_params)
    ode_res = ode_model.run(T, dt, step_input)
    
    # Run SSA
    print("Running SSA...")
    ssa_model = I1_FFL_SSA(ssa_params, Omega=100.0)
    ssa_res = ssa_model.run(T, dt, step_input)
    
    # Run Markovian
    print("Running Markovian...")
    markov_model = I1_FFL_Markovian(markov_params)
    markov_res = markov_model.run(T, dt, step_input)
    
    # Plotting
    plt.figure(figsize=(12, 10))
    
    # Plot X Input
    plt.subplot(3, 1, 1)
    plt.plot(ode_res['t'], ode_res['X'], 'k--', label='Input X')
    plt.title('Input Signal')
    plt.ylabel('[X]')
    plt.legend()
    
    # Plot Y Response
    plt.subplot(3, 1, 2)
    plt.plot(ode_res['t'], ode_res['Y'], 'b-', label='ODE')
    plt.plot(ssa_res['t'], ssa_res['Y'], 'g.', alpha=0.3, label='SSA')
    plt.plot(markov_res['t'], markov_res['Y'], 'r-', alpha=0.7, label='Markovian')
    plt.title('Intermediate Y')
    plt.ylabel('[Y]')
    plt.legend()
    
    # Plot Z Response (Pulse)
    plt.subplot(3, 1, 3)
    plt.plot(ode_res['t'], ode_res['Z'], 'b-', label='ODE')
    plt.plot(ssa_res['t'], ssa_res['Z'], 'g.', alpha=0.3, label='SSA')
    plt.plot(markov_res['t'], markov_res['Z'], 'r-', alpha=0.7, label='Markovian')
    plt.title('Output Z (Pulse)')
    plt.ylabel('[Z]')
    plt.xlabel('Time')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ffl_comparison.png')
    print("Validation Complete. Plot saved to ffl_comparison.png")

if __name__ == "__main__":
    run_validation()
