
import numpy as np

class p53_Markovian:
    """
    Hybrid Markovian model for p53-Mdm2.
    p53, ATM, Mdm2 (protein) are continuous ODEs.
    Mdm2 Promoter is discrete Markov chain regulated by p53.
    """
    def __init__(self, params):
        self.params = params
        
        # State
        self.p53 = 0.0
        self.Mdm2 = 0.0
        self.ATM = 0.0
        
        # Promoter State: Number of p53 bound (0, 1, ..., n)
        # Assume 2 binding sites for simplicity unless specified
        # or n binding sites based on params['n']
        self.promoter_state = 0
        self.n_sites = int(params.get('n', 2))
        
        self.time = 0.0
        self.history = {'t': [], 'p53': [], 'Mdm2': [], 'ATM': [], 'S': []}
        
    def step(self, dt, signal):
        p = self.params
        
        # --- Update Promoter State (Stochastic) ---
        # k: sites bound
        k = self.promoter_state
        n = self.n_sites
        
        # Calculate Rates
        # Binding: k -> k+1
        # Rate = k_on * [p53] * (n - k) * cooperativity?
        # Let's derive effective k_on from Hill K
        # K = k_off / k_on
        # k_on = k_off / K
        
        # Assuming parameters provide k_on, k_off directly or we infer
        k_off = p.get('k_off_p53', 1.0)
        # If not provided, assume K is from Hill
        KF = p.get('KF', 1.0)
        # Then k_on = k_off / KF?
        # Or k_on = 1.0, k_off = KF?
        k_on_base = p.get('k_on_p53', k_off / KF)
        
        # Cooperativity factor c
        coop = p.get('cooperativity', 1.0)
        
        # Transition probabilities
        
        # Forward (Binding)
        if k < n:
            # Cooperative if already bound?
            c_factor = coop if k > 0 else 1.0
            rate_bind = k_on_base * self.p53 * (n - k) * c_factor
            if np.random.rand() < 1 - np.exp(-rate_bind * dt):
                self.promoter_state += 1
                
        # Backward (Unbinding)
        if k > 0:
            rate_unbind = k_off * k
            if np.random.rand() < 1 - np.exp(-rate_unbind * dt):
                self.promoter_state -= 1
                
        # --- Update ODEs ---
        
        # dATM
        dATM = signal - p['k_deact_ATM'] * self.ATM
        
        # dp53
        # Same as ODE
        prod_p53 = p['k_syn_p53'] + p['k_act_ATM'] * self.ATM
        deg_p53 = p['k_deg_p53'] * self.p53 + p['k_ub'] * self.Mdm2 * self.p53
        dp53 = prod_p53 - deg_p53
        
        # dMdm2
        # Transcription depends on promoter state
        # Logic: If bound, active? Or fractional activty?
        # If n sites, maybe proportional to k/n?
        # Or fully active if k >= some threshold?
        # Hill (x/K)^n / (1+(x/K)^n) implies occupancy.
        # Let's assume proportional activity or active if fully bound.
        # Let's use fraction k/n
        
        activity = self.promoter_state / n
        prod_Mdm2 = p['k_trans_Mdm2'] * activity
        deg_Mdm2 = p['k_deg_Mdm2'] * self.Mdm2
        dMdm2 = prod_Mdm2 - deg_Mdm2
        
        # Update state
        self.ATM += dATM * dt
        self.p53 += dp53 * dt
        self.Mdm2 += dMdm2 * dt
        
        # Clip
        self.ATM = max(0.0, self.ATM)
        self.p53 = max(0.0, self.p53)
        self.Mdm2 = max(0.0, self.Mdm2)
        
        self.time += dt

    def run(self, T, dt, signal_func):
        t_values = np.arange(0, T, dt)
        
        for t in t_values:
            signal = signal_func(t)
            
            self.history['t'].append(t)
            self.history['p53'].append(self.p53)
            self.history['Mdm2'].append(self.Mdm2)
            self.history['ATM'].append(self.ATM)
            self.history['S'].append(self.promoter_state)
            
            self.step(dt, signal)
            
        return self.history
