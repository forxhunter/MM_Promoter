
import numpy as np
from scipy.integrate import odeint

class p53_ODE:
    """
    Deterministic ODE model for p53-Mdm2 Oscillator.
    Based on Lahav et al. (2004) / Geva-Zatorsky et al. (2006)
    
    Structure:
    p53 -> Mdm2 (Activation)
    Mdm2 -> p53 (Degradation/Ubiquitination)
    ATM -> p53 (Activation/Stabilization)
    """
    def __init__(self, params):
        self.params = params
        # State: [p53, Mdm2, ATM]
        self.state = [0.0, 0.0, 0.0]
        self.time = 0.0
        self.history = {'t': [], 'p53': [], 'Mdm2': [], 'ATM': []}
        
    def dynamics(self, state, t, signal_func):
        p53, Mdm2, ATM = state
        p = self.params
        
        signal = signal_func(t)
        
        # dATM/dt
        # Activated by signal, deactivates slowly
        dATM = signal - p['k_deact_ATM'] * ATM
        
        # dp53/dt
        # Production: basal + ATM-enhanced?
        # Degradation: Mdm2-dependent
        prod_p53 = p['k_syn_p53'] + p['k_act_ATM'] * ATM
        deg_p53 = p['k_deg_p53'] * p53 + p['k_ub'] * Mdm2 * p53
        dp53 = prod_p53 - deg_p53
        
        # dMdm2/dt
        # Transcription activation by p53 (cooperative Hill)
        # Degradation
        k_trans = p['k_trans_Mdm2']
        KM = p['KF'] # Dissociation constant for p53-DNA
        n = p['n'] # Cooperativity
        
        hill = (p53 / KM)**n / (1 + (p53 / KM)**n)
        prod_Mdm2 = k_trans * hill
        deg_Mdm2 = p['k_deg_Mdm2'] * Mdm2
        
        dMdm2 = prod_Mdm2 - deg_Mdm2
        
        return [dp53, dMdm2, dATM]

    def run(self, T, dt, signal_func):
        t_values = np.arange(0, T, dt)
        
        for t in t_values:
            self.history['t'].append(t)
            self.history['p53'].append(self.state[0])
            self.history['Mdm2'].append(self.state[1])
            self.history['ATM'].append(self.state[2])
            
            # Solve one step
            # Note: signal_func passed properly or handled inside dynamics?
            # dynamics calls signal_func(t).
            next_state = odeint(self.dynamics, self.state, [t, t+dt], args=(signal_func,))[-1]
            # Clip negative values
            next_state = np.maximum(next_state, 0.0)
            self.state = next_state
            
        return self.history
