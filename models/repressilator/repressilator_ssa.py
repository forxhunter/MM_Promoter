import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# SSA for DDM Repressilator
# Serves as Ground Truth for Hybrid Validation.

class SSA_Repressilator:
    def __init__(self, params, Omega=1.0):
        self.params = params
        self.Omega = Omega
        
        # State:
        # Promoters: S1, S2, S3 \in {0, 1}
        # Species: m1, m2, m3, u1, u2, u3, p1, p2, p3
        # Indices:
        # 0-2: S
        # 3-5: m
        # 6-8: u
        # 9-11: p
        self.state = np.zeros(12, dtype=int)
        self.state[0:3] = 1 # Promoters Active
        self.state[3] = int(5.0 * Omega) # Initial m1
        self.state[6] = int(48.0 * Omega) # Initial u1 (Steady State for m=5)
        self.state[9] = int(1000.0 * Omega) # Initial p1 (cI) -> Matches ODE
        
        self.time = 0.0
        self.history_t = [0.0]
        # Just track Folded p1, p2, p3
        self.history_p = [self.state[9:12].copy()]
        
    def run(self, T):
        p = self.params
        k_burst = p.get('k_burst', 0.05)
        KM = p.get('KM', 40.0)
        n = p.get('n', 2.0)
        
        k_trans = p['k_trans'] * self.Omega
        k_leak = p['k_leak'] * self.Omega
        k_deg_m = p['k_deg_m']
        
        k_transl = p['k_transl'] # 1st order m -> m + u (rate * m)
        k_fold = p['k_fold']
        k_deg_p = p['k_deg_p']
        
        # Repressor Map: p3->1, p1->2, p2->3
        # Indices in state: p1=9, p2=10, p3=11.
        repressor_indices = [11, 9, 10]
        
        while self.time < T:
            # Rates
            rates = []
            reaction_types = [] # List of tuples (type, gene_index)
            
            # For each gene i=0,1,2
            for i in range(3):
                Si = self.state[i]
                mi = self.state[3+i]
                ui = self.state[6+i]
                pi = self.state[9+i]
                
                # Repressor
                rep_idx = repressor_indices[i]
                R_count = self.state[rep_idx]
                R_conc = R_count / self.Omega
                
                # 1. Promoter On
                r_on = k_burst * (1 - Si)
                rates.append(r_on)
                reaction_types.append(('on', i))
                
                # 2. Promoter Off
                # k_off = k_on * (R/K)^n
                r_off = k_burst * (R_conc / KM)**n * Si
                rates.append(r_off)
                reaction_types.append(('off', i))
                
                # 3. Transcription (Active)
                r_trans = k_trans * Si
                rates.append(r_trans)
                reaction_types.append(('trans', i))
                
                # 4. Leak
                r_leak = k_leak
                rates.append(r_leak)
                reaction_types.append(('leak', i))
                
                # 5. Deg mRNA
                r_deg_m = k_deg_m * mi
                rates.append(r_deg_m)
                reaction_types.append(('deg_m', i))
                
                # 6. Translation (m -> m + u)
                r_transl = k_transl * mi
                rates.append(r_transl)
                reaction_types.append(('transl', i))
                
                # 7. Folding (u -> p)
                r_fold = k_fold * ui
                rates.append(r_fold)
                reaction_types.append(('fold', i))
                
                # 8. Deg Unfolded
                r_deg_u = k_deg_p * ui
                rates.append(r_deg_u)
                reaction_types.append(('deg_u', i))
                
                # 9. Deg Folded
                r_deg_p = k_deg_p * pi
                rates.append(r_deg_p)
                reaction_types.append(('deg_p', i))
                
            total = sum(rates)
            if total == 0: break
            
            # Step
            tau = -np.log(np.random.rand()) / total
            if self.time + tau > T: break
            self.time += tau
            
            # Select Reaction
            r = np.random.rand() * total
            cum = 0
            for k, rate in enumerate(rates):
                cum += rate
                if r <= cum:
                    rxn_idx = k
                    break
            
            rtype, idx = reaction_types[rxn_idx]
            
            if rtype == 'on': self.state[idx] = 1
            elif rtype == 'off': self.state[idx] = 0
            elif rtype == 'trans': self.state[3+idx] += 1
            elif rtype == 'leak': self.state[3+idx] += 1
            elif rtype == 'deg_m': self.state[3+idx] -= 1
            elif rtype == 'transl': self.state[6+idx] += 1
            elif rtype == 'fold': 
                self.state[6+idx] -= 1
                self.state[9+idx] += 1
            elif rtype == 'deg_u': self.state[6+idx] -= 1
            elif rtype == 'deg_p': self.state[9+idx] -= 1
            
            # Record (sparse)
            # if np.random.rand() < 0.01: # Optimization: Don't record every step
            # Just record time grid later? No, record list then interp.
            pass
        
        # We need trajectory. This implementation stores minimal history? 
        # Actually I didn't append to history in the loop. Fixing.
            # Record significant changes? 
            # Storing every step for long sim is heavy. 
            self.history_t.append(self.time)
            self.history_p.append(self.state[9:12].copy())
            
        return np.array(self.history_t), np.array(self.history_p)/self.Omega

# Parameters (Same as Ref)
params = {
    'k_trans': 0.5,
    'k_leak': 5e-4,
    'k_deg_m': np.log(2)/2, # Matches ODE (2 min half life)
    'k_transl': 0.16,
    'k_fold': 1.0/60,
    'k_deg_p': np.log(2)/600,
    'n': 2.0,
    'KM': 40.0,
    'k_burst': 0.05
}

# Run 5 Replicates (Slow simulation)
if __name__ == "__main__":
    # Run 5 Replicates (Slow simulation)
    data = []
    print("Running SSA Repressilator (Simulating Delay...)")
    for i in range(5):
        print(f"Replicate {i+1}...")
        sim = SSA_Repressilator(params, Omega=200.0) # High volume for convergence proof
        t, p = sim.run(10000) # Longer Run
        
        # Interp
        t_int = np.linspace(0, 5000, 1000)
        p_int = np.zeros((1000, 3))
        for k in range(3):
            p_int[:, k] = np.interp(t_int, t, p[:, k])
            
        for j in range(1000):
            data.append({
                'Time': t_int[j],
                'cI': p_int[j, 0],
                'LacI': p_int[j, 1],
                'TetR': p_int[j, 2],
                'Replicate': i
            })

    df = pd.DataFrame(data)
    df.to_csv('repressilator_ssa_trajectories.csv', index=False)
    print("SSA Complete.")

    plt.figure()
    for i in range(5):
        sub = df[df['Replicate'] == i]
        plt.plot(sub['Time']/60, sub['cI'], alpha=0.3)
    plt.title('SSA Repressilator (Ground Truth)')
    plt.savefig('repressilator_ssa.png')
