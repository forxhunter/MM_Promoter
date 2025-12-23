
import numpy as np

class p53_SSA:
    """
    Gillespie SSA model for p53-Mdm2.
    """
    def __init__(self, params, Omega=1.0):
        self.params = params
        self.Omega = Omega
        
        # Discrete molecule counts
        self.p53 = 0
        self.Mdm2 = 0
        self.ATM = 0
        
        self.time = 0.0
        self.history = {'t': [], 'p53': [], 'Mdm2': [], 'ATM': []}
        
    def propensity(self, signal):
        p = self.params
        O = self.Omega
        
        # Convert Rates
        # 0-order: k * O
        # 1-order: k
        # 2-order: k / O
        
        p53_conc = self.p53 / O
        Mdm2_conc = self.Mdm2 / O
        ATM_conc = self.ATM / O
        
        rates = []
        rxns = [] # (type, species change)
        
        # 1. ATM activation (Source)
        r_atm_on = signal * O # signal is rate of activation? Assume signal is flux.
        rates.append(r_atm_on)
        rxns.append('atm_up')
        
        # 2. ATM deactivation
        r_atm_off = p['k_deact_ATM'] * self.ATM
        rates.append(r_atm_off)
        rxns.append('atm_down')
        
        # 3. p53 synthesis (basal)
        r_syn_p53 = p['k_syn_p53'] * O
        rates.append(r_syn_p53)
        rxns.append('p53_up')
        
        # 4. p53 activation/stabilization by ATM (modeled as extra synthesis or just dependency?)
        # In ODE: prod = k_act * ATM.
        # So it's 1st order in ATM -> p53 production
        r_act_p53 = p['k_act_ATM'] * self.ATM 
        rates.append(r_act_p53)
        rxns.append('p53_up_atm')
        
        # 5. p53 degradation (basal)
        r_deg_p53 = p['k_deg_p53'] * self.p53
        rates.append(r_deg_p53)
        rxns.append('p53_down')
        
        # 6. Mdm2-mediated p53 degradation (ubiquitination)
        # ODE: k_ub * Mdm2 * p53
        # SSA: k_ub/O * Mdm2 * p53
        r_ub = (p['k_ub'] / O) * self.Mdm2 * self.p53
        rates.append(r_ub)
        rxns.append('p53_down_mdm2')
        
        # 7. Mdm2 synthesis
        # Hill function: k_trans * (p53/K)^n / (...)
        # Use concentration for Hill
        hill = (p['k_trans_Mdm2'] * O) * ((p53_conc / p['KF'])**p['n']) / (1 + (p53_conc / p['KF'])**p['n'])
        rates.append(hill)
        rxns.append('mdm2_up')
        
        # 8. Mdm2 degradation
        r_deg_mdm2 = p['k_deg_Mdm2'] * self.Mdm2
        rates.append(r_deg_mdm2)
        rxns.append('mdm2_down')
        
        return rates, rxns

    def step(self, t_end, signal):
        while self.time < t_end:
            rates, rxns = self.propensity(signal)
            rate_sum = sum(rates)
            
            if rate_sum == 0:
                self.time = t_end
                break
                
            r = np.random.rand()
            tau = -np.log(r) / rate_sum
            
            if self.time + tau > t_end:
                self.time = t_end
                break
                
            self.time += tau
            
            # Select rxn
            r2 = np.random.rand() * rate_sum
            cum = 0
            for i, rate in enumerate(rates):
                cum += rate
                if r2 <= cum:
                    rxn_type = rxns[i]
                    if rxn_type == 'atm_up': self.ATM += 1
                    elif rxn_type == 'atm_down': self.ATM -= 1
                    elif rxn_type == 'p53_up': self.p53 += 1
                    elif rxn_type == 'p53_up_atm': self.p53 += 1
                    elif rxn_type == 'p53_down': self.p53 -= 1
                    elif rxn_type == 'p53_down_mdm2': self.p53 -= 1
                    elif rxn_type == 'mdm2_up': self.Mdm2 += 1
                    elif rxn_type == 'mdm2_down': self.Mdm2 -= 1
                    break

    def run(self, T, dt, signal_func):
        t_grid = np.arange(0, T, dt)
        
        for t in t_grid:
            signal = signal_func(t)
            
            self.history['t'].append(t)
            self.history['p53'].append(self.p53 / self.Omega)
            self.history['Mdm2'].append(self.Mdm2 / self.Omega)
            self.history['ATM'].append(self.ATM / self.Omega)
            
            self.step(t + dt, signal)
            
        return self.history
