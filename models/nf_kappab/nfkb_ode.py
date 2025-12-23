
import numpy as np
from scipy.integrate import odeint

class NFkB_ODE:
    """
    Deterministic ODE model for NF-kB Pathway (IkappaB-alpha only).
    Based on Hoffmann et al. (2002).
    
    Variables:
    - IKK: IkappaB Kinase (Active)
    - NFkB_cyt: Free NF-kB in cytoplasm
    - NFkB_nuc: Free NF-kB in nucleus
    - IkB_cyt: Free IkappaB in cytoplasm
    - IkB_nuc: Free IkappaB in nucleus (optional, simplified here often just IkB_cyt)
    - IkB_NFkB: Complex in cytoplasm
    - IkB_mRNA: mRNA for IkappaB
    """
    def __init__(self, params):
        self.params = params
        # State: [NFkB_nuc, NFkB_cyt, IkB_cyt, IkB_NFkB, IkB_mRNA, IKK]
        # Ignoring IkB_nuc for simplified model (often negligible or simplified into export)
        # Assuming total NFkB is conserved? No, synthesis/degradation not usually dominant on short scale
        # but Hoffmann model includes them.
        
        # Initial State (approximate resting)
        self.state = [0.001, 0.1, 0.5, 0.1, 0.0, 0.0] 
        # Tuning initial to be closer to steady state is better but let's just start low.
        
        self.time = 0.0
        self.history = {'t': [], 'NFkB_nuc': [], 'Total_IkB': [], 'IKK': []}
        
    def dynamics(self, state, t, signal_func):
        NFkB_nuc, NFkB_cyt, IkB_cyt, IkB_NFkB, IkB_mRNA, IKK = state
        p = self.params
        
        # Stimulus
        # signal = signal_func(t)
        # IKK dynamics: Activation by signal, Deactivation
        # dIKK = signal - k_deact * IKK
        # Or IKK profile given directly? Let's model dynamics.
        signal = signal_func(t)
        dIKK = signal - p['k_deact_IKK'] * IKK
        
        # Reaction Fluxes
        
        # 1. NFkB import/export
        # NFkB_cyt -> NFkB_nuc
        J_imps = p['k_import_NFkB'] * NFkB_cyt
        # NFkB_nuc -> NFkB_cyt (often ignored if IkB absent, but export exists)
        J_exps = p['k_export_NFkB'] * NFkB_nuc
        
        # 2. Complex Formation/Dissociation (Cytoplasm)
        J_bind = p['k_bind'] * IkB_cyt * NFkB_cyt
        J_unbind = p['k_unbind'] * IkB_NFkB
        
        # 3. IKK-mediated degradation of IkB in complex
        # IKK phosphorylates IκB in complex -> releases NF-κB
        J_phos_c = p['k_phos'] * IKK * IkB_NFkB
        
        # 4. IKK-mediated degradation of free IkB
        # J_phos_f = p['k_phos'] * IKK * IkB_cyt
        
        # 5. IkB Transcription (Nuclear NF-kB activation)
        # Hill function
        J_trans = p['k_trans'] * (NFkB_nuc**2) / (p['K_trans']**2 + NFkB_nuc**2)
        
        # 6. IkB Translation
        J_transl = p['k_transl'] * IkB_mRNA
        
        # 7. Degradation (Basal)
        J_deg_mRNA = p['k_deg_mRNA'] * IkB_mRNA
        J_deg_IkB = p['k_deg_IkB'] * IkB_cyt
        # Complex degradation? Usually slow or handled via components.
        
        # Derivatives
        
        # dNFkB_nuc
        dNFkB_nuc = J_imps - J_exps 
        # Wait, binding in nucleus?
        # Hoffmann model often includes IkB_nuc entering and exporting NFkB.
        # Simplification: IkB_cyt can effectively pull NFkB out via "retention" 
        # but real neg feedback is nuclear IkB export.
        # Let's add IkB_nuc for correctness or use effective export.
        # "IKK - NF-kB Signaling Module" usually involves:
        # IkB_alpha enters nucleus, binds NFkB, exports it.
        # Let's stick to the simplified "effective export" if we don't track IkB_nuc.
        # Or add [IkB_nuc] to state?
        # Let's add it for rigor.
        
        # Re-defining state for accuracy:
        # [NFkB_nuc, NFkB_cyt, IkB_nuc, IkB_cyt, IkB_NFkB, IkB_mRNA, IKK]
        pass # Re-writing dynamics below properly
        
        return [0]*6

    def dynamics_full(self, state, t, signal_func):
        # State: [NFkB_nuc, NFkB_cyt, IkB_nuc, IkB_cyt, IkB_NFkB_cyt, IkB_mRNA, IKK]
        # Ignoring Nuclear Complex (IkB:NFkB_nuc) -> assume assumes fast export
        
        NFkB_nuc, NFkB_cyt, IkB_nuc, IkB_cyt, IkB_NFkB, IkB_mRNA, IKK = state
        p = self.params
        
        signal = signal_func(t)
        dIKK = signal - p['k_deact_IKK'] * IKK
        
        # NFkB Fluxes
        J_imp_n = p['k_imp_n'] * NFkB_cyt
        # Export via IkB binding? Or intrinsic? Assume small intrinsic.
        
        # IkB Fluxes
        J_imp_i = p['k_imp_i'] * IkB_cyt
        J_exp_i = p['k_exp_i'] * IkB_nuc
        
        # mRNA
        J_trans = p['k_trans'] * (NFkB_nuc**2)/(p['K_trans']**2 + NFkB_nuc**2)
        J_deg_m = p['k_deg_m'] * IkB_mRNA
        
        # Protein Syn
        J_transl = p['k_transl'] * IkB_mRNA
        
        # Cytoplasmic Interaction
        J_bind = p['k_bind'] * IkB_cyt * NFkB_cyt
        J_unbind = p['k_unbind'] * IkB_NFkB
        J_phos = p['k_phos'] * IKK * IkB_NFkB
        
        # Nuclear Interaction (Export logic)
        # IkB_nuc binds NFkB_nuc -> Complex -> Export
        # Modeled as: k_assoc * IkB_nuc * NFkB_nuc -> Exported Cytoplasmic Complex?
        # Or just: IkB_nuc * NFkB_nuc -> IkB_NFkB_nuc -> (transport) -> IkB_NFkB_cyt
        # Let's assume fast export of complex:
        J_form_export = p['k_bind'] * IkB_nuc * NFkB_nuc * 10.0 # Fast binding/export
        
        # Degradation
        J_deg_i = p['k_deg_i'] * IkB_cyt
        
        # Derivatives
        
        dNFkB_nuc = J_imp_n - J_form_export
        dNFkB_cyt = -J_imp_n - J_bind + J_unbind + J_phos
        
        dIkB_nuc = J_imp_i - J_exp_i - J_form_export
        dIkB_cyt = J_transl - J_imp_i + J_exp_i - J_bind + J_unbind - J_deg_i
        # Note: Complex from nucleus comes here?
        # Creating IkB_NFkB from nuclear export
        
        dIkB_NFkB = J_bind - J_unbind - J_phos + J_form_export
        
        dIkB_mRNA = J_trans - J_deg_m
        
        return [dNFkB_nuc, dNFkB_cyt, dIkB_nuc, dIkB_cyt, dIkB_NFkB, dIkB_mRNA, dIKK]

    def run(self, T, dt, signal_func):
        # Initial State: [0.0, 1.0, 0.0, 0.5, 2.0, 0.0, 0.0]
        # [N_n, N_c, I_n, I_c, I:N, m, IKK]
        state = [0.01, 1.0, 0.01, 0.5, 3.0, 0.1, 0.0] 
        # Approx steady state
        
        t_values = np.arange(0, T, dt)
        
        for t in t_values:
            self.history['t'].append(t)
            self.history['NFkB_nuc'].append(state[0])
            self.history['Total_IkB'].append(state[2]+state[3]+state[4])
            self.history['IKK'].append(state[6])
            
            # Solve
            next_state = odeint(self.dynamics_full, state, [t, t+dt], args=(signal_func,))[-1]
            next_state = np.maximum(next_state, 0.0)
            state = next_state
            
        return self.history
