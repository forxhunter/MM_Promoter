
import numpy as np

class NFkB_SSA:
    """
    SSA for NF-kB Pathway.
    """
    def __init__(self, params, Omega=1.0):
        self.params = params
        self.Omega = Omega
        
        # State counts
        self.NFkB_nuc = 0
        self.NFkB_cyt = int(1.0 * Omega)
        self.IkB_nuc = 0
        self.IkB_cyt = int(0.5 * Omega)
        self.IkB_NFkB = int(3.0 * Omega)
        self.IkB_mRNA = int(0.1 * Omega)
        self.IKK = 0
        
        self.time = 0.0
        self.history = {'t': [], 'NFkB_nuc': [], 'Total_IkB': [], 'IKK': []}
        
    def propensity(self, signal):
        p = self.params
        O = self.Omega
        
        rates = []
        rxns = []
        
        # IKK
        rates.append(signal * O) # Activation
        rxns.append('IKK_up')
        
        rates.append(p['k_deact_IKK'] * self.IKK) # Deactivation
        rxns.append('IKK_down')
        
        # NFkB Import (Cyt -> Nuc)
        rates.append(p['k_imp_n'] * self.NFkB_cyt)
        rxns.append('import_N')
        
        # IkB Import (Cyt -> Nuc)
        rates.append(p['k_imp_i'] * self.IkB_cyt)
        rxns.append('import_I')
        
        # IkB Export (Nuc -> Cyt)
        rates.append(p['k_exp_i'] * self.IkB_nuc)
        rxns.append('export_I')
        
        # Nuclear Binding/Export (Nuc -> Cyt Complex)
        # k_bind/O * I_n * N_n
        rates.append((p['k_bind'] / O) * self.IkB_nuc * self.NFkB_nuc * 10.0) # Fast export
        rxns.append('bind_export_NI')
        
        # Cytoplasmic Binding
        rates.append((p['k_bind'] / O) * self.IkB_cyt * self.NFkB_cyt)
        rxns.append('bind_NI_c')
        
        # Cytoplasmic Unbinding
        rates.append(p['k_unbind'] * self.IkB_NFkB)
        rxns.append('unbind_NI_c')
        
        # IKK-mediated Phos/Degradation of Complex
        # IKK * Complex * k/O?
        rates.append((p['k_phos'] / O) * self.IKK * self.IkB_NFkB)
        rxns.append('phos_NI')
        
        # Transcription (Hill)
        # k * (N_n/O)^2 / (K^2 + (N_n/O)^2) * O = k*O * ...
        N_conc = self.NFkB_nuc / O
        hill = (N_conc**2) / (p['K_trans']**2 + N_conc**2)
        rates.append(p['k_trans'] * O * hill)
        rxns.append('trans')
        
        # Translation
        rates.append(p['k_transl'] * self.IkB_mRNA)
        rxns.append('transl')
        
        # mRNA Deg
        rates.append(p['k_deg_m'] * self.IkB_mRNA)
        rxns.append('deg_m')
        
        # IkB Deg
        rates.append(p['k_deg_i'] * self.IkB_cyt)
        rxns.append('deg_i')
        
        return rates, rxns

    def step(self, t_end, signal):
        while self.time < t_end:
            rates, rxns = self.propensity(signal)
            total = sum(rates)
            
            if total == 0:
                self.time = t_end
                break
                
            dt = -np.log(np.random.rand()) / total
            if self.time + dt > t_end:
                self.time = t_end
                break
                
            self.time += dt
            
            # Execute
            r = np.random.rand() * total
            cum = 0
            for i, rate in enumerate(rates):
                cum += rate
                if r <= cum:
                    rxn = rxns[i]
                    if rxn == 'IKK_up': self.IKK += 1
                    elif rxn == 'IKK_down': self.IKK -= 1
                    elif rxn == 'import_N':
                        self.NFkB_cyt -= 1; self.NFkB_nuc += 1
                    elif rxn == 'import_I':
                        self.IkB_cyt -= 1; self.IkB_nuc += 1
                    elif rxn == 'export_I':
                        self.IkB_nuc -= 1; self.IkB_cyt += 1
                    elif rxn == 'bind_export_NI':
                        self.IkB_nuc -= 1; self.NFkB_nuc -= 1; self.IkB_NFkB += 1
                    elif rxn == 'bind_NI_c':
                        self.IkB_cyt -= 1; self.NFkB_cyt -= 1; self.IkB_NFkB += 1
                    elif rxn == 'unbind_NI_c':
                        self.IkB_NFkB -= 1; self.IkB_cyt += 1; self.NFkB_cyt += 1
                    elif rxn == 'phos_NI':
                        self.IkB_NFkB -= 1; self.NFkB_cyt += 1 # IkB degraded effectively
                    elif rxn == 'trans': self.IkB_mRNA += 1
                    elif rxn == 'transl': self.IkB_cyt += 1
                    elif rxn == 'deg_m': self.IkB_mRNA -= 1
                    elif rxn == 'deg_i': self.IkB_cyt -= 1
                    break

    def run(self, T, dt, signal_func):
        t_grid = np.arange(0, T, dt)
        
        for t in t_grid:
            signal = signal_func(t)
            self.history['t'].append(t)
            self.history['NFkB_nuc'].append(self.NFkB_nuc / self.Omega)
            self.history['Total_IkB'].append((self.IkB_nuc + self.IkB_cyt + self.IkB_NFkB) / self.Omega)
            self.history['IKK'].append(self.IKK / self.Omega)
            
            self.step(t+dt, signal)
            
        return self.history
