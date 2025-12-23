
import numpy as np
import matplotlib.pyplot as plt

from nfkb_ode import NFkB_ODE
from nfkb_ssa import NFkB_SSA
from nfkb_markovian import NFkB_Markovian

def signal_pulse(t):
    """TNF Pulse"""
    return 1.0 if t < 15.0 else 0.0 # 15 min pulse

def run_validation():
    T = 240.0 # Minutes (4 hours)
    dt = 0.1
    
    # Parameters (Hoffmann-like, simplified)
    params = {
        'k_deact_IKK': 0.05,
        
        'k_imp_n': 0.1,    # Import N
        'k_imp_i': 0.05,   # Import I
        'k_exp_i': 0.05,   # Export I
        
        'k_bind': 0.5,     # Binding
        'k_unbind': 0.001,
        
        'k_phos': 0.5,     # Strong phosphorylation
        
        'k_trans': 1.0,    # Transcription
        'K_trans': 0.1,    # Hill K
        
        'k_transl': 0.5,
        'k_deg_m': 0.05,
        'k_deg_i': 0.001,
        
        # Markovian
        'k_on_N': 10.0,
        'k_off_N': 1.0, # K ~ 0.1
        'cooperativity': 2.0
    }
    
    # Run ODE
    print("Running ODE...")
    ode_model = NFkB_ODE(params)
    ode_res = ode_model.run(T, dt, signal_pulse)
    
    # Run SSA
    print("Running SSA...")
    ssa_model = NFkB_SSA(params, Omega=50.0)
    ssa_res = ssa_model.run(T, dt, signal_pulse)
    
    # Run Markovian
    print("Running Markovian...")
    markov_model = NFkB_Markovian(params)
    markov_res = markov_model.run(T, dt, signal_pulse)
    
    # Plotting
    plt.figure(figsize=(12, 12))
    
    # NFkB Nuclear
    plt.subplot(3, 1, 1)
    plt.plot(ode_res['t'], ode_res['NFkB_nuc'], 'b-', label='ODE')
    plt.plot(ssa_res['t'], ssa_res['NFkB_nuc'], 'g.', alpha=0.3, label='SSA')
    plt.plot(markov_res['t'], markov_res['NFkB_nuc'], 'r-', alpha=0.7, label='Markovian')
    plt.title('Nuclear NF-kB')
    plt.ylabel('[NFkB_nuc]')
    plt.legend()
    
    # Total IkB
    plt.subplot(3, 1, 2)
    plt.plot(ode_res['t'], ode_res['Total_IkB'], 'b-', label='ODE')
    plt.plot(ssa_res['t'], ssa_res['Total_IkB'], 'g.', alpha=0.3, label='SSA')
    plt.plot(markov_res['t'], markov_res['Total_IkB'], 'r-', alpha=0.7, label='Markovian')
    plt.title('Total IkB (Feedback)')
    plt.ylabel('[IkB]')
    plt.legend()
    
    # Promoter
    plt.subplot(3, 1, 3)
    plt.plot(markov_res['t'], markov_res['S'], 'm-', label='Promoter State')
    plt.title('IkB Promoter Activity')
    plt.xlabel('Time (min)')
    plt.ylabel('Bound Inputs')
    
    plt.tight_layout()
    plt.savefig('nfkb_comparison.png')
    print("Validation Complete. Saved to nfkb_comparison.png")

if __name__ == "__main__":
    run_validation()
