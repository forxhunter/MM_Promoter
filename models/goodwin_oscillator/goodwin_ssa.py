import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# SSA (Gillespie) Implementation of Goodwin Oscillator
# To serve as Ground Truth for Markovian Hybrid validation.

# Model:
# 4 Species: Sx (Promoter), X (mRNA), Y (Enzyme), Z (Repressor)
# Sx \in {0, 1}
# X, Y, Z \in {0, 1, 2, ...}

class SSA_Goodwin:
    def __init__(self, params, Omega=10.0):
        self.params = params
        self.Omega = Omega
        
        # State: [Sx, X, Y, Z]
        # Start low/active
        self.state = np.array([1, int(0.1*Omega), int(0.1*Omega), int(0.1*Omega)])
        self.time = 0.0
        self.history_t = [0.0]
        self.history_x = [self.state[1]]
        
    def run(self, T):
        p = self.params
        k_burst = p.get('k_burst', 2.0)
        n = p.get('n', 10.0)
        KM = p.get('KM', 1.0)
        
        k1 = p['k1'] * self.Omega # Max production rate (conc/time * Vol = count/time)
        k2 = p['k2'] # 1st order rate constant (1/time), implies X->X+Y so rate = k2*X
        k3 = p['k3'] # 1st order
        
        b1 = p['b1']
        b2 = p['b2']
        b3 = p['b3']
        
        while self.time < T:
            Sx, X, Y, Z = self.state
            
            # Concentrations for rates
            z_conc = Z / self.Omega
            
            # Rates
            # 1. Promoter On
            r_on = k_burst * (1 - Sx)
            # 2. Promoter Off (Hill Repression Logic converted to Rate)
            # k_off = k_on * (Z/K)^n
            r_off = k_burst * (z_conc / KM)**n * Sx
            
            # 3. Transcription (Requires Sx=1)
            r_trans = k1 * Sx
            # 4. Deg mRNA
            r_deg_x = b1 * X
            
            # 5. Translation
            r_transl = k2 * X
            # 6. Deg Enzyme
            r_deg_y = b2 * Y
            
            # 7. Prod Repressor
            r_prod_z = k3 * Y
            # 8. Deg Repressor
            r_deg_z = b3 * Z
            
            rates = [r_on, r_off, r_trans, r_deg_x, r_transl, r_deg_y, r_prod_z, r_deg_z]
            total = sum(rates)
            
            if total == 0: break
            
            # Time Step
            tau = -np.log(np.random.rand()) / total
            if self.time + tau > T: break
            self.time += tau
            
            # Reaction
            r = np.random.rand() * total
            cum = 0
            for i, rate in enumerate(rates):
                cum += rate
                if r <= cum:
                    rxn = i
                    break
                    
            if rxn == 0: self.state[0] = 1 # On
            elif rxn == 1: self.state[0] = 0 # Off
            elif rxn == 2: self.state[1] += 1 # Prod X
            elif rxn == 3: self.state[1] -= 1 # Deg X
            elif rxn == 4: self.state[2] += 1 # Prod Y
            elif rxn == 5: self.state[2] -= 1 # Deg Y
            elif rxn == 6: self.state[3] += 1 # Prod Z
            elif rxn == 7: self.state[3] -= 1 # Deg Z
            
            # Record (Downsample? Just record X for now)
            self.history_t.append(self.time)
            self.history_x.append(self.state[1])
            
        return np.array(self.history_t), np.array(self.history_x)/self.Omega

# Parameters (Same as Markovian)
params = {
    'k1': 1.0, 'k2': 1.0, 'k3': 1.0,
    'b1': 0.1, 'b2': 0.1, 'b3': 0.1,
    'n': 10.0, 'KM': 1.0,
    'k_burst': 2.0
}

# Run 10 Replicates
if __name__ == "__main__":
    # Run 10 Replicates
    data = []
    for i in range(10):
        print(f"SSA Replicate {i+1}...")
        sim = SSA_Goodwin(params, Omega=50.0) # Larger volume
        t, x = sim.run(1000) # Longer time for stationary dist
        
        # Resample
        t_interp = np.linspace(0, 200, 1000)
        x_interp = np.interp(t_interp, t, x)
        
        for j in range(len(t_interp)):
            data.append({
                'Time': t_interp[j],
                'X': x_interp[j],
                'Replicate': i
            })

    df = pd.DataFrame(data)
    df.to_csv('goodwin_ssa_trajectories.csv', index=False)
    print("SSA Complete.")

    # Plot
    plt.figure()
    for i in range(10):
        subset = df[df['Replicate'] == i]
        plt.plot(subset['Time'], subset['X'], 'b-', alpha=0.1)
    plt.title('SSA Ground Truth (Goodwin X)')
    plt.savefig('goodwin_ssa.png')
