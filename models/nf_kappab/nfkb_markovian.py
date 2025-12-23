
import numpy as np
from scipy.integrate import odeint

class NFkB_Markovian:
    """
    Markovian Model for NF-kB.
    IkB Promoter is discrete Markov chain.
    """
    def __init__(self, params):
        self.params = params
        
        # Continuous State
        # [N_n, N_c, I_n, I_c, I:N, m, IKK]
        self.state = [0.01, 1.0, 0.01, 0.5, 3.0, 0.1, 0.0] 
        
        # Promoter State
        self.promoter_state = 0 # 0, 1, 2 (NFkB_nuc bound)
        self.n_sites = 2
        
        self.time = 0.0
        self.history = {'t': [], 'NFkB_nuc': [], 'Total_IkB': [], 'S': []}
        
    def step_promoter(self, dt, NFkB_nuc_conc):
        p = self.params
        k = self.promoter_state
        n = self.n_sites
        
        # Constants from params
        k_on_base = p.get('k_on_N', 1.0)
        k_off = p.get('k_off_N', 1.0)
        coop = p.get('cooperativity', 1.0)
        
        # Binding
        if k < n:
            c = coop if k > 0 else 1.0
            rate_bind = k_on_base * NFkB_nuc_conc * (n - k) * c
            if np.random.rand() < 1 - np.exp(-rate_bind * dt):
                self.promoter_state += 1
                
        # Unbinding
        if k > 0:
            rate_unbind = k_off * k
            if np.random.rand() < 1 - np.exp(-rate_unbind * dt):
                self.promoter_state -= 1
                
    def dynamics(self, state, t, signal, promoter_activity):
        # Unpack
        NFkB_nuc, NFkB_cyt, IkB_nuc, IkB_cyt, IkB_NFkB, IkB_mRNA, IKK = state
        p = self.params
        
        dIKK = signal - p['k_deact_IKK'] * IKK
        
        # Fluxes
        J_imp_n = p['k_imp_n'] * NFkB_cyt
        J_imp_i = p['k_imp_i'] * IkB_cyt
        J_exp_i = p['k_exp_i'] * IkB_nuc
        
        # Fast Nuclear Export
        J_form_export = p['k_bind'] * IkB_nuc * NFkB_nuc * 10.0
        
        J_bind = p['k_bind'] * IkB_cyt * NFkB_cyt
        J_unbind = p['k_unbind'] * IkB_NFkB
        J_phos = p['k_phos'] * IKK * IkB_NFkB
        
        # mRNA Synthesis (Markovian Input)
        J_trans = p['k_trans'] * promoter_activity
        
        J_deg_m = p['k_deg_m'] * IkB_mRNA
        J_transl = p['k_transl'] * IkB_mRNA
        J_deg_i = p['k_deg_i'] * IkB_cyt
        
        dNFkB_nuc = J_imp_n - J_form_export
        dNFkB_cyt = -J_imp_n - J_bind + J_unbind + J_phos
        dIkB_nuc = J_imp_i - J_exp_i - J_form_export
        dIkB_cyt = J_transl - J_imp_i + J_exp_i - J_bind + J_unbind - J_deg_i
        dIkB_NFkB = J_bind - J_unbind - J_phos + J_form_export
        dIkB_mRNA = J_trans - J_deg_m
        
        return [dNFkB_nuc, dNFkB_cyt, dIkB_nuc, dIkB_cyt, dIkB_NFkB, dIkB_mRNA, dIKK]

    def run(self, T, dt, signal_func):
        t_grid = np.arange(0, T, dt)
        
        for t in t_grid:
            signal = signal_func(t)
            NFkB_nuc_conc = self.state[0]
            
            # Step Promoter (Stochastic)
            self.step_promoter(dt, NFkB_nuc_conc)
            activity = self.promoter_state / self.n_sites
            
            # Record
            self.history['t'].append(t)
            self.history['NFkB_nuc'].append(NFkB_nuc_conc)
            self.history['Total_IkB'].append(self.state[2]+self.state[3]+self.state[4])
            self.history['S'].append(self.promoter_state)
            
            # Solve ODE (Deterministic step for dt)
            next_state = odeint(self.dynamics, self.state, [t, t+dt], args=(signal, activity))[-1]
            self.state = np.maximum(next_state, 0.0)
            
            self.time += dt
            
        return self.history
