import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# SSA (Gillespie) Implementation of Genetic Toggle Switch
# To serve as Ground Truth for Markovian Hybrid validation.

# Model:
# 4 Species: U, V, Su (Promoter U), Sv (Promoter V)
# Su, Sv \in {0, 1}
# U, V \in {0, 1, 2, ...} (Molecules)

# Reactions:
# 1. Promoter U Switching:
#    Su=0 -> Su=1 (k_on)
#    Su=1 -> Su=0 (k_off = k_on * V^beta)  <- Here V is molecule count. 
#    Wait. If V is molecules, V^beta might be huge.
#    Hill function uses V/K. 
#    We will assume parameters alpha, beta, etc. are for CONCENTRATIONS.
#    We define Volume Omega.
#    Conc = Count / Omega.
#    Rates based on Count:
#    k_off = k_burst * ( (V/Omega) / K )^beta
# 2. Promoter V Switching:
#    Sv=0 -> Sv=1 (k_on)
#    Sv=1 -> Sv=0 (k_off = k_on * ( (U/Omega) / K )^gamma )
# 3. Production:
#    Su=1 -> Su=1 + U (Rate = alpha1 * Omega)
#    Sv=1 -> Sv=1 + V (Rate = alpha2 * Omega)
# 4. Degradation:
#    U -> 0 (Rate = 1.0 * U)
#    V -> 0 (Rate = 1.0 * V)

class SSA_Toggle:
    def __init__(self, params, Omega=1.0):
        self.params = params
        self.Omega = Omega
        
        # State: [Su, Sv, U, V]
        self.state = np.array([1, 0, 150*Omega, 0]) # Default High U
        self.time = 0.0
        self.history_t = [0.0]
        self.history_u = [self.state[2]]
        self.history_v = [self.state[3]]
        
    def run(self, T):
        alpha1 = self.params.get('alpha1', 156.25)
        alpha2 = self.params.get('alpha2', 15.6)
        beta = self.params.get('beta', 2.5)
        gamma = self.params.get('gamma', 1.0)
        k_burst = self.params.get('k_burst', 1.0)
        
        while self.time < T:
            Su, Sv, U, V = self.state
            
            # Rate Calculations
            # 1. Switch U
            # On: Su=0 -> 1. Rate = k_burst.
            # Off: Su=1 -> 0. Rate = k_burst * ( (V/self.Omega)**beta )
            r_su_on = k_burst * (1 - Su)
            r_su_off = k_burst * ((V/self.Omega)**beta) * Su
            
            # 2. Switch V
            # On: Sv=0 -> 1. Rate = k_burst.
            # Off: Sv=1 -> 0. Rate = k_burst * ( (U/self.Omega)**gamma ) * Sv
            r_sv_on = k_burst * (1 - Sv)
            r_sv_off = k_burst * ((U/self.Omega)**gamma) * Sv
            
            # 3. Production
            r_prod_u = alpha1 * self.Omega * Su
            r_prod_v = alpha2 * self.Omega * Sv
            
            # 4. Degradation
            r_deg_u = 1.0 * U
            r_deg_v = 1.0 * V
            
            rates = [r_su_on, r_su_off, r_sv_on, r_sv_off, r_prod_u, r_prod_v, r_deg_u, r_deg_v]
            total_rate = sum(rates)
            
            if total_rate == 0:
                break
                
            # Time Step
            tau = -np.log(np.random.rand()) / total_rate
            if self.time + tau > T:
                break
                
            self.time += tau
            
            # Determine Reaction
            r = np.random.rand() * total_rate
            cumulative = 0
            for i, rate in enumerate(rates):
                cumulative += rate
                if r <= cumulative:
                    rxn_idx = i
                    break
            
            # Execute
            if rxn_idx == 0: self.state[0] = 1 # Su On
            elif rxn_idx == 1: self.state[0] = 0 # Su Off
            elif rxn_idx == 2: self.state[1] = 1 # Sv On
            elif rxn_idx == 3: self.state[1] = 0 # Sv Off
            elif rxn_idx == 4: self.state[2] += 1 # Prod U
            elif rxn_idx == 5: self.state[3] += 1 # Prod V
            elif rxn_idx == 6: self.state[2] -= 1 # Deg U
            elif rxn_idx == 7: self.state[3] -= 1 # Deg V
            
            # Record
            self.history_t.append(self.time)
            self.history_u.append(self.state[2])
            self.history_v.append(self.state[3])
           
        return np.array(self.history_t), np.array(self.history_u)/self.Omega, np.array(self.history_v)/self.Omega

if __name__ == "__main__":
    # Run 10 Replicates
    params = {
        'alpha1': 156.25,
        'alpha2': 15.6,
        'beta': 2.5,
        'gamma': 1.0,
        'k_burst': 1.0
    }

    data = []
    for i in range(10):
        print(f"SSA Replicate {i+1}...")
        sim = SSA_Toggle(params, Omega=1.0) # Unit Volume
        t, u, v = sim.run(100)
        # Resample to common time grid
        t_interp = np.linspace(0, 100, 1000)
        u_interp = np.interp(t_interp, t, u)
        v_interp = np.interp(t_interp, t, v)
        
        for j in range(len(t_interp)):
            data.append({
                'Time': t_interp[j],
                'U': u_interp[j],
                'V': v_interp[j],
                'Replicate': i
            })

    df = pd.DataFrame(data)
    df.to_csv('toggle_ssa_trajectories.csv', index=False)
    print("SSA Complete.")

    # Plot
    plt.figure()
    for i in range(10):
        subset = df[df['Replicate'] == i]
        plt.plot(subset['Time'], subset['U'], 'b-', alpha=0.1)
        
    plt.title('SSA Ground Truth (10 Replicates)')
    plt.savefig('toggle_ssa.png')
