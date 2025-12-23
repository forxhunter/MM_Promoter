
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool
import scipy.interpolate

# SSA Implementation of Repressilator (DDM)
# 4 Species per Gene: M_i, P_unfolded_i, P_folded_i, Promoter_i
# Total 12 variables in state. But actually we track [M0, P0_u, P0_f, S0, M1...]
# Simplified State: 12 elements.
# But for SSA, we need reaction firing.

# Reactions per gene i (i=0,1,2), repressor j = (i-1)%3
# 1. Transcription: S_i=1 -> S_i=1 + M_i (k_trans)
# 2. Translation: M_i -> M_i + P_u_i (k_tl)
# 3. Folding: P_u_i -> P_f_i (k_fold)
# 4. Degradation M: M_i -> 0 (k_deg_m)
# 5. Degradation P_u: P_u_i -> 0 (k_deg_p)
# 6. Degradation P_f: P_f_i -> 0 (k_deg_p)
# 7. Promotion Switching:
#    S_i=0 -> S_i=1 (k_on * (1-S_i))
#    S_i=1 -> S_i=0 (k_off * S_i)
#    Where k_off = k_burst * (P_f_j/Omega / KM)^n

class SSA_Repressilator:
    def __init__(self, params, Omega=50.0):
        self.params = params
        self.Omega = Omega
        self.n_genes = 3
        
        # State: [S0, M0, Pu0, Pf0, S1, M1, Pu1, Pf1, S2, M2, Pu2, Pf2]
        # Indices:
        # Gene i: S=4*i, M=4*i+1, Pu=4*i+2, Pf=4*i+3
        self.state = np.zeros(12, dtype=np.float64)
        
        # Init: High Protein 0, others 0
        self.state[3] = 1000.0 * Omega # P_f_0
        self.state[0] = 1.0 # S0 ON
        self.state[4] = 1.0 # S1 ON
        self.state[8] = 1.0 # S2 ON
        
        self.time = 0.0
        self.history_t = [0.0]
        self.history_p0 = [self.state[3]]
        self.history_p1 = [self.state[7]]
        self.history_p2 = [self.state[11]]
        
    def run(self, T):
        # Unpack Rates
        k_trans = self.params.get('k_trans', 0.5) * 60 # per min
        k_deg_m = self.params.get('k_deg_m', np.log(2)/2) # per min
        k_tl = self.params.get('k_tl', 0.16) * 60 
        k_fold = self.params.get('k_fold', 0.0 * 60) # Delay step. If 0? 
        # Wait, reference says delay=6 min. k_fold approx 1/6?
        # Model uses explicit intermediate? The DDM ODE uses a delay.
        # Here we use Linear Chain Trick step if we want delay.
        # But if k_fold not specified, act as 1 step?
        # Let's assume fast folding if not specified, implies No Delay?
        # No, DDM implies delay. We need k_fold.
        # In DDM Model: tau_p = 6.0 min (folding?).
        # We will use k_fold = 1/6.0
        k_fold = 1.0 / 6.0 
        
        k_deg_p = self.params.get('k_deg_p', np.log(2)/600) # Stable
        
        KM = self.params.get('KM', 40.0)
        n = self.params.get('n', 2.0)
        k_burst = self.params.get('k_burst', 0.05) # Switching freq
        
        while self.time < T:
            rates = []
            rxn_types = [] # (type, species_idx)
            
            # For each gene
            for i in range(3):
                base = 4*i
                S = self.state[base]
                M = self.state[base+1]
                Pu = self.state[base+2]
                Pf = self.state[base+3]
                
                # Repressor Index
                j = (i - 1) % 3
                repressor_conc = self.state[4*j+3] / self.Omega
                
                # 1. Transcription
                r_trans = k_trans * S
                rates.append(r_trans)
                rxn_types.append((0, i))
                
                # 2. Translation
                r_tl = k_tl * M
                rates.append(r_tl)
                rxn_types.append((1, i))
                
                # 3. Folding
                r_fold = k_fold * Pu
                rates.append(r_fold)
                rxn_types.append((2, i))
                
                # 4. Deg M
                r_deg_m = k_deg_m * M
                rates.append(r_deg_m)
                rxn_types.append((3, i))
                
                # 5. Deg Pu
                r_deg_pu = k_deg_p * Pu
                rates.append(r_deg_pu)
                rxn_types.append((4, i))
                
                # 6. Deg Pf
                r_deg_pf = k_deg_p * Pf
                rates.append(r_deg_pf)
                rxn_types.append((5, i))
                
                # 7. Switch ON
                # S=0 -> 1. Rate k_burst * (1-S)
                r_on = k_burst * (1.0 - S)
                rates.append(r_on)
                rxn_types.append((6, i))
                
                # 8. Switch OFF
                # S=1 -> 0. Rate k_burst * Hill * S
                # Hill = (P_j / KM)^n
                r_off = k_burst * ((repressor_conc / KM)**n) * S
                rates.append(r_off)
                rxn_types.append((7, i))
                
            total_rate = sum(rates)
            
            if total_rate == 0: break
            
            # Time Step
            tau = -np.log(np.random.rand()) / total_rate
            if self.time + tau > T: break
            self.time += tau
            
            # Choice
            r = np.random.rand() * total_rate
            cum = 0
            sel_idx = -1
            for idx, rate in enumerate(rates):
                cum += rate
                if r <= cum:
                    sel_idx = idx
                    break
            
            # Execute
            rtype, i = rxn_types[sel_idx]
            base = 4*i
            if rtype == 0: # Trans
                self.state[base+1] += 1
            elif rtype == 1: # TL
                self.state[base+2] += 1
            elif rtype == 2: # Fold
                self.state[base+2] -= 1
                self.state[base+3] += 1
            elif rtype == 3: # Deg M
                self.state[base+1] -= 1
            elif rtype == 4: # Deg Pu
                self.state[base+2] -= 1
            elif rtype == 5: # Deg Pf
                self.state[base+3] -= 1
            elif rtype == 6: # ON
                self.state[base] = 1.0
            elif rtype == 7: # OFF
                self.state[base] = 0.0
                
            # History (Record sparsely to save memory?)
            # Just record every 10 steps or if time delta large?
            # For 1000 reps, full history is heavy.
            # We only need interpolated result. 
            self.history_t.append(self.time)
            self.history_p0.append(self.state[3])
            # self.history_p1.append(self.state[7])
            # self.history_p2.append(self.state[11])

        return np.array(self.history_t), np.array(self.history_p0)/self.Omega

def run_single_worker(seed):
    np.random.seed(seed)
    params = {
        'k_trans': 0.5,
        'k_tl': 0.16,
        'k_deg_m': np.log(2)/2,
        'k_deg_p': np.log(2)/600,
        'n': 2.0,
        'KM': 40.0,
        'k_burst': 0.05
    }
    # Omega = 50.0 (User said "dont increase volume", so we assume 50 is acceptable baseline or 5?)
    # Validating with 50.0 is best compromise.
    sim = SSA_Repressilator(params, Omega=50.0)
    t, p = sim.run(2000) # Shorter run (2000 min) is enough for stationary
    
    # Interp
    t_interp = np.linspace(1000, 2000, 1000) # Compare stationary part
    p_interp = np.interp(t_interp, t, p)
    return p_interp

if __name__ == "__main__":
    N_REPS = 1000
    print(f"Starting {N_REPS} Replicates Parallel SSA...")
    
    with Pool(processes=20) as pool:
        results = pool.map(run_single_worker, range(N_REPS))
        
    data = np.array(results).flatten()
    print(f"Data Shape: {data.shape}")
    
    # Save statistics
    params = {
        'mean': np.mean(data),
        'std': np.std(data),
        'N': len(data)
    }
    print(f"SSA Mean: {params['mean']:.4f}")
    
    # Save raw for KS
    df = pd.DataFrame({'P_folded': data})
    df.to_csv('repressilator_ssa_1000_stats.csv', index=False)
