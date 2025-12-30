import numpy as np
import pandas as pd
import os
import sys
import time
from scipy.integrate import odeint, solve_ivp

# ==================================================================================================
# 1. MARKOVIAN PROMOTER MODEL (Discrete State)
# ==================================================================================================

try:
    from math import comb
except ImportError:
    def comb(n, k):
        if k < 0 or k > n: return 0
        if k == 0 or k == n: return 1
        k = min(k, n - k)
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result

class MarkovianPromoter:
    def __init__(self, name, n_sites, rates):
        self.name = name
        self.n_sites = n_sites
        self.kfp = rates['kfp']
        self.krp = rates['krp']
        self.kfr = rates['kfr']
        self.krr = rates['krr']
        self.qr = rates['qr']
        
        self.n_states = (n_sites + 1) * (n_sites + 2) // 2
        
        self.current_state_idx = 0 
        
    def _index_to_state(self, idx):
        k = 0
        while (k + 1) * (k + 2) // 2 <= idx:
            k += 1
        m = idx - k * (k + 1) // 2
        return k, m

    def _state_to_index(self, k, m):
        return k * (k + 1) // 2 + m

    def update(self, dt, G4d, G80d, rng=None):
        if rng is None: rng = np.random
        
        # Build reduced Q matrix: only need outgoing rates from current state
        # Actually, for correctness with varying G4d/G80d, we should compute transition probabilities
        # properly. Since dt is small, P(trans) ~ rate * dt.
        
        k_curr, m_curr = self._index_to_state(self.current_state_idx)
        
        possible_transitions = []
        rates = []
        
        # Gal4p binding: (k,m) -> (k+1,m)
        if k_curr < self.n_sites:
            rate = self.kfp * G4d * (self.n_sites - k_curr)
            possible_transitions.append(self._state_to_index(k_curr + 1, m_curr))
            rates.append(rate)
            
        # Gal4p unbinding: (k,m) -> (k-1,m)
        if k_curr > m_curr and k_curr > 0:
            rate = self.krp * (k_curr - m_curr)
            possible_transitions.append(self._state_to_index(k_curr - 1, m_curr))
            rates.append(rate)
            
        # Gal80p binding: (k,m) -> (k,m+1)
        if m_curr < k_curr:
            coop = self.qr if m_curr > 0 else 1.0
            rate = self.kfr * G80d * (k_curr - m_curr) * coop
            possible_transitions.append(self._state_to_index(k_curr, m_curr + 1))
            rates.append(rate)
            
        # Gal80p unbinding: (k,m) -> (k,m-1)
        if m_curr > 0:
            rate = self.krr * m_curr
            possible_transitions.append(self._state_to_index(k_curr, m_curr - 1))
            rates.append(rate)
            
        rates = np.array(rates)
        total_rate = np.sum(rates)
        
        # Gillespie-style step within dt time window?
        # Or simple probability of switching?
        # If total_rate * dt is not small (>0.1), we might have multiple events.
        # But for hybrid sim, usually we assume at most one event or handle it.
        # Let's use the probability of NO event in dt: exp(-total_rate * dt)
        
        prob_stay = np.exp(-total_rate * dt)
        
        if rng.random() > prob_stay:
            # Event happens
            # Normalize rates to get transition probabilities
            probs = rates / np.sum(rates)
            target = rng.choice(possible_transitions, p=probs)
            self.current_state_idx = target
            
        # Determine activity
        k, m = self._index_to_state(self.current_state_idx)
        is_active = 1 if k > m else 0
        return is_active, (k, m)

# ==================================================================================================
# 2. GAL HYBRID SIMULATOR
# ==================================================================================================

def gal_ode_func(t, y, GAE_mM, active_flags, rates_dict):
    # Unpack y
    R1, R2, R3, R4, RepRNA, R80 = y[0:6]
    G1, G2, G3, G3i, G4, G4d = y[6:12]
    RepProt, G80, G80C, G80d, G80Cd, G80G3i, GAI = y[12:19]
    
    molecTomM = 4.65e-8
    GAI_mM = GAI * molecTomM
    
    # Constants from Ramsey et al.
    mrna_max = {'GAL1': 33, 'GAL2': 33, 'GAL3': 28, 'GAL4': 0.4, 'GAL80': 21, 'reporter': 33}
    kdr = {'GAL1': np.log(2)/31, 'GAL2': np.log(2)/9, 'GAL3': np.log(2)/26, 'GAL4': np.log(2)/28, 'GAL80': np.log(2)/24, 'reporter': np.log(2)/20}
    kdp = {'GAL1': np.log(2)/180, 'GAL2': np.log(2)/180, 'GAL3': np.log(2)/60, 'GAL4': np.log(2)/100, 'GAL80': np.log(2)/100, 'reporter': np.log(2)/60}
    prot_ratio = {'GAL1': 500, 'GAL2': 3500, 'GAL3': 4800, 'GAL4': 1545, 'GAL80': 530, 'reporter': 500}
    
    kir = {g: mrna_max[g] * kdr[g] for g in mrna_max}
    kip = {g: prot_ratio[g] * kdp[g] for g in prot_ratio}
    
    k_TR = 4350.0; Km_TR = 1.0; alpha_TR = 1.0
    kcat_GK = 3350.0; Km_GK = 0.6 / molecTomM
    
    # Interaction constants
    Kfi = 0.000000745; Kri = 890
    Kfd3i80 = 0.025716; Kdr3i80 = 0.0159616
    Kfd = 100; Krd = 0.001
    Kf80 = 50; Kr80 = 50
    
    # Transport
    vTR = k_TR * G2 * (GAE_mM - GAI_mM) / (Km_TR + GAE_mM + GAI_mM + (alpha_TR * GAE_mM * GAI_mM / Km_TR))
    vGK = kcat_GK * G1 * GAI / (Km_GK + GAI)
    
    dydt = np.zeros(19)
    
    # mRNA
    dydt[0] = kir['GAL1'] * active_flags['GAL1'] - kdr['GAL1'] * R1
    dydt[1] = kir['GAL2'] * active_flags['GAL2'] - kdr['GAL2'] * R2
    dydt[2] = 0.57 * kir['GAL3'] * active_flags['GAL3'] - kdr['GAL3'] * R3
    dydt[3] = kir['GAL4'] - kdr['GAL4'] * R4
    dydt[4] = kir['reporter'] * active_flags['reporter'] - kdr['reporter'] * RepRNA
    dydt[5] = kir['GAL80'] * active_flags['GAL80'] - kdr['GAL80'] * R80
    
    # Proteins
    dydt[6] = kip['GAL1'] * R1 - kdp['GAL1'] * G1
    dydt[7] = kip['GAL2'] * R2 - kdp['GAL2'] * G2
    dydt[8] = kip['GAL3'] * R3 - kdp['GAL3'] * G3 - Kfi * G3 * GAI + Kri * G3i
    dydt[9] = Kfi * G3 * GAI - Kri * G3i - kdp['GAL3'] * G3i - Kfd3i80 * G80Cd * G3i + Kdr3i80 * G80G3i
    dydt[10] = kip['GAL4'] * R4 - kdp['GAL4'] * G4 - 2 * Kfd * G4**2 + 2 * Krd * G4d
    dydt[11] = Kfd * G4**2 - Krd * G4d - kdp['GAL4'] * G4d
    dydt[12] = kip['reporter'] * RepRNA - kdp['reporter'] * RepProt
    dydt[13] = kip['GAL80'] * R80 - kdp['GAL80'] * G80 - Kf80 * G80 + Kr80 * G80C - 2 * Kfd * G80**2 + 2 * Krd * G80d
    dydt[14] = Kf80 * G80 - Kr80 * G80C - 2 * Kfd * G80C**2 + 2 * Krd * G80Cd - kdp['GAL80'] * G80C
    dydt[15] = Kfd * G80**2 - Krd * G80d - kdp['GAL80'] * G80d + Kf80 * G80Cd - Kr80 * G80d
    dydt[16] = Kfd * G80C**2 - Krd * G80Cd - kdp['GAL80'] * G80Cd + Kf80 * G80d - Kr80 * G80Cd - Kfd3i80 * G80Cd * G3i + Kdr3i80 * G80G3i
    dydt[17] = Kfd3i80 * G80Cd * G3i - Kdr3i80 * G80G3i - 0.5 * kdp['GAL3'] * G80G3i
    dydt[18] = vTR - vGK
    
    return dydt

class GalHybridSimulator:
    def __init__(self, GAE_mM=0.0):
        self.GAE_mM = GAE_mM
        
        # Kinetic Parameters (Ramsey et al. 2006)
        max_R4 = 0.4
        prot_to_mrna_gal4 = 1545
        max_G4d = max_R4 * prot_to_mrna_gal4 / 2  # ~309.1
        
        self.rates = {
            'kfp': 6.5 / max_G4d,
            'krp': 1.0,
            'kfr': 5 * (6.5 / max_G4d),
            'krr': 1.0,
            'qr': 30.0
        }
        
        # Promoters
        self.promoters = {
            'GAL1': MarkovianPromoter('GAL1', 4, self.rates),
            'GAL2': MarkovianPromoter('GAL2', 5, self.rates),
            'GAL3': MarkovianPromoter('GAL3', 1, self.rates),
            'GAL80': MarkovianPromoter('GAL80', 1, self.rates),
            'reporter': MarkovianPromoter('reporter', 4, self.rates)
        }
        
        # Initial State
        self.y = np.array([
             0.26465, 0.33048, 0.90442, 0.4, 0.26465, 1.1871, 
             132.3267, 1156.7004, 4341.21998, 0, 0.15653, 308.9216, 
             132.3265, 0.11381, 0.1095, 157.2286, 157.2286, 0, 0
        ])
        
    def run(self, duration=200, dt=0.01):
        steps = int(duration / dt)
        time_points = np.linspace(0, duration, steps+1)
        
        history = np.zeros((steps+1, 19 + 5))
        history[0, :19] = self.y
        history[0, 19:] = 0 
        
        rng = np.random.default_rng()
        
        for i in range(steps):
            t_curr = time_points[i]
            
            # 1. Update Promoters
            G4d = self.y[11]
            G80d = self.y[15]
            
            active_flags = {}
            for name, p in self.promoters.items():
                is_active, _ = p.update(dt, G4d, G80d, rng=rng)
                active_flags[name] = is_active
            
            # 2. ODE Step using LSODA or BDF (robust solver) covering [t, t+dt]
            # We assume active_flags are constant over dt (Zero-Order Hold)
            
            sol = solve_ivp(gal_ode_func, [0, dt], self.y, args=(self.GAE_mM, active_flags, self.rates), 
                            method='LSODA', rtol=1e-6, atol=1e-9)
            
            self.y = sol.y[:, -1]
            
            # Record
            history[i+1, :19] = self.y
            history[i+1, 19] = active_flags['GAL1']
            history[i+1, 20] = active_flags['GAL2']
            history[i+1, 21] = active_flags['GAL3']
            history[i+1, 22] = active_flags['GAL80']
            history[i+1, 23] = active_flags['reporter']
            
        return time_points, history

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['dose_response', 'timeseries', 'promoter'], required=True)
    args = parser.parse_args()
    
    if args.mode == 'dose_response':
        doses = [0, 0.1, 1.0, 10.0]
        results = []
        for dose in doses:
            print(f"Simulating dose {dose} mM...")
            sim = GalHybridSimulator(GAE_mM=dose)
            # Run for 300 min
            t, hist = sim.run(duration=300, dt=0.01)
            # Average last 50 min
            ss_data = np.mean(hist[-5000:], axis=0) 
            results.append([dose] + list(ss_data))
        
        cols = ['Dose', 'R1', 'R2', 'R3', 'R4', 'RepRNA', 'R80', 
                'G1', 'G2', 'G3', 'G3i', 'G4', 'G4d', 
                'RepProt', 'G80', 'G80C', 'G80d', 'G80Cd', 'G80G3i', 'GAI',
                'Act_GAL1', 'Act_GAL2', 'Act_GAL3', 'Act_GAL80', 'Act_Rep']
        df = pd.DataFrame(results, columns=cols)
        df.to_csv('gal_dose_response_data.csv', index=False)
        print("Saved gal_dose_response_data.csv")
        
    elif args.mode == 'timeseries':
        print("Simulating step change 0->10 mM...")
        
        sim = GalHybridSimulator(GAE_mM=0.0)
        t1, h1 = sim.run(duration=10, dt=0.01)
        
        sim.GAE_mM = 10.0
        t2, h2 = sim.run(duration=190, dt=0.01)
        
        t2 = t2 + 10.0
        full_t = np.concatenate([t1, t2[1:]])
        full_h = np.concatenate([h1, h2[1:]])
        
        cols = ['R1', 'R2', 'R3', 'R4', 'RepRNA', 'R80', 
                'G1', 'G2', 'G3', 'G3i', 'G4', 'G4d', 
                'RepProt', 'G80', 'G80C', 'G80d', 'G80Cd', 'G80G3i', 'GAI',
                'Act_GAL1', 'Act_GAL2', 'Act_GAL3', 'Act_GAL80', 'Act_Rep']
        df = pd.DataFrame(full_h, columns=cols)
        df.insert(0, 'Time', full_t)
        df.to_csv('gal_timeseries_data.csv', index=False)
        print("Saved gal_timeseries_data.csv")

    elif args.mode == 'promoter':
        sim = GalHybridSimulator(GAE_mM=10.0)
        state_history = []
        dt = 0.01
        steps = int(100/dt)
        rng = np.random.default_rng()
        time_points = np.linspace(0, 100, steps+1)
        
        for i in range(steps):
            t = time_points[i]
            G4d = sim.y[11]
            G80d = sim.y[15]
            
            p = sim.promoters['GAL1']
            is_active, (k, m) = p.update(dt, G4d, G80d, rng=rng)
            state_history.append({'Time': t, 'k': k, 'm': m, 'G4d': G4d, 'G80d': G80d})
            
            for name in ['GAL2', 'GAL3', 'GAL80', 'reporter']:
                sim.promoters[name].update(dt, G4d, G80d, rng=rng)
                
            active_flags = {n: (1 if sim.promoters[n].current_state_idx > 0 else 0) for n in sim.promoters} 
            # Revisit: the update() method already evolved state.
            # We can just check the internal state.
            # actually we returned is_active from update. 
            active_flags['GAL1'] = is_active
            
            sol = solve_ivp(gal_ode_func, [0, dt], sim.y, args=(sim.GAE_mM, active_flags, sim.rates),
                            method='LSODA', rtol=1e-6, atol=1e-9)
            sim.y = sol.y[:, -1]
            
        df = pd.DataFrame(state_history)
        df.to_csv('gal_promoter_states.csv', index=False)
        print("Saved gal_promoter_states.csv")
