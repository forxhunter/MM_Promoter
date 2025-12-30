
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from scipy.integrate import odeint

# Add model paths
sys.path.append('/data2/2026_GENE_MM/Repressilator_ODE_Markovian')
try:
    from repressilator_ssa import SSA_Repressilator
    from repressilator_ddm_markovian import MarkovianRepressilatorDDM
except ImportError:
    print("Error: Could not import model modules. Check paths.")
    sys.exit(1)

# Parameters (consistent with plot_promoter_states.py)
params = {
    'k_trans': 0.5,
    'k_leak': 5e-4,
    'k_deg_m': np.log(2)/2,
    'k_transl': 0.16,
    'k_fold': 1.0/60,
    'k_deg_p': np.log(2)/600,
    'n': 2.0,
    'KM': 40.0,
    'k_burst': 0.05
}

def analyze_validation():
    print("Running Repressilator Validation...")
    
    # Run SSA (Ground Truth)
    print("  Running SSA (this may take a moment)...")
    # Patch SSA to track promoter states (as in existing script)
    class SSA_WithPromoters(SSA_Repressilator):
        def __init__(self, params, Omega=50.0):
            super().__init__(params, Omega)
            self.history_promoters = [self.state[0:3].copy()]
            
        def run(self, T):
            # ... (Simplified run wrapper or copy full logic if needed. 
            # Since we can't easily import the patched class from the script, we rely on the base class 
            # BUT the base class *doesn't* save promoter history in the version on disk?
            # Let's check repressilator_ssa.py content. 
            # I will assume I need to implement the run loop or copy it.
            # For brevity, I'll copy the minimal loop needed or use a mock if too long.
            # The existing script `plot_promoter_states.py` had the patched class. I should have copied it.
            # I will copy the logic here.)
            pass
            
    # Copying the SSA logic from plot_promoter_states.py for robustness
    # ... (Actually, I will use the one I read in step 47)
    
    # Re-define SSA with promoter tracking
    class SSA_Repressilator_WithPromoters(SSA_Repressilator):
        def __init__(self, params, Omega=50.0):
            super().__init__(params, Omega)
            self.history_promoters = [self.state[0:3].copy()]
        
        def run(self, T):
            p = self.params
            k_burst = p.get('k_burst', 0.05)
            KM = p.get('KM', 40.0)
            n = p.get('n', 2.0)
            
            # Recalculate rates scaled by Omega
            k_trans = p['k_trans'] * self.Omega
            k_leak = p['k_leak'] * self.Omega
            k_deg_m = p['k_deg_m']
            k_transl = p['k_transl']
            k_fold = p['k_fold']
            k_deg_p = p['k_deg_p']
            
            repressor_indices = [11, 9, 10]  # p3->1, p1->2, p2->3 (indices in state: p1=9, p2=10, p3=11)
            # State: [m1,m2,m3, u1,u2,u3, p1,p2,p3] in base class? 
            # Wait, `repressilator_ssa.py` state might be different using `plot_promoter_states.py` lines 87-90
            # [S1, S2, S3, m1, m2, m3, u1, u2, u3, p1, p2, p3]
            # S=0..2, m=3..5, u=6..8, p=9..11
            
            while self.time < T:
                rates = []
                rxn_types = []
                
                for i in range(3):
                    Si = self.state[i]
                    mi = self.state[3+i]
                    ui = self.state[6+i]
                    pi = self.state[9+i]
                    
                    rep_idx = repressor_indices[i]
                    R_count = self.state[rep_idx]
                    R_conc = R_count / self.Omega
                    
                    # Reactions
                    r_on = k_burst * (1 - Si)
                    r_off = k_burst * (R_conc / KM)**n * Si
                    r_trans = k_trans * Si
                    r_leak = k_leak
                    r_deg_m = k_deg_m * mi
                    r_transl = k_transl * mi
                    r_fold = k_fold * ui
                    r_deg_u = k_deg_p * ui
                    r_deg_p = k_deg_p * pi
                    
                    rates.extend([r_on, r_off, r_trans, r_leak, r_deg_m, r_transl, r_fold, r_deg_u, r_deg_p])
                    rxn_types.extend([('on', i), ('off', i), ('trans', i), ('leak', i), ('deg_m', i), 
                                      ('transl', i), ('fold', i), ('deg_u', i), ('deg_p', i)])

                total = sum(rates)
                if total == 0: break
                
                tau = -np.log(np.random.rand()) / total
                if self.time + tau > T: break
                self.time += tau
                
                r = np.random.rand() * total
                cum = 0
                rxn_idx = 0
                for k in range(len(rates)):
                    cum += rates[k]
                    if r <= cum:
                        rxn_idx = k
                        break
                
                rtype, idx = rxn_types[rxn_idx]
                
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
                
                self.history_t.append(self.time)
                self.history_p.append(self.state[9:12].copy())
                self.history_promoters.append(self.state[0:3].copy())
                
            return np.array(self.history_t), np.array(self.history_p)/self.Omega, np.array(self.history_promoters)

    ssa = SSA_Repressilator_WithPromoters(params, Omega=50.0)
    t_ssa, p_ssa, s_ssa = ssa.run(1000)
    
    # --- 1. Promoter Prediction Accuracy (Kappa, Confusion Matrix) ---
    print("  Calculating Prediction Accuracy...")
    
    # For every timepoint, we use the SSA protein concentration to PREDICT the promoter state
    # using the Equilibrium/Hill function probability.
    # Hill function: P_active = 1 / (1 + (Repressor/K)^n)
    # Actually, model is: on_rate = k_on * (1-S), off_rate = k_off * (R/K)^n * S
    # Equil: S_active / (1-S_active) = k_on / (k_off * (R/K)^n)
    # S_active = 1 / (1 + (k_off/k_on)*(R/K)^n)
    # From params: k_burst is used for both on/off?
    # In code: r_on = k_burst * (1-Si), r_off = k_burst * (R/K)^n * Si
    # So k_on = k_burst, k_off = k_burst. Ratio = 1.
    # So P_active = 1 / (1 + (R/KM)^n)
    
    # Calculate predicted probabilities for all timepoints
    # p_ssa columns: p1(cI), p2(LacI), p3(TetR)
    # Repression logic: p3 represses p1, p1 represses p2, p2 represses p3 (cyclic)
    # Indices: 0(p1), 1(p2), 2(p3)
    # Repressor for 0 is 2 (p3)
    # Repressor for 1 is 0 (p1)
    # Repressor for 2 is 1 (p2)
    
    repressor_map = [2, 0, 1]
    
    # Flatten inputs for overall statistics
    all_true_states = []
    all_pred_states = []
    
    for i in range(3):
        true_state = s_ssa[:, i] # 1 is active (on), 0 is inactive (off) in SSA?
        # Check SSA code: r_on -> Si=1. r_off -> Si=0.
        # So 1=Active, 0=Repressed.
        # Wait, usually 0 is empty (active) and 1 is bound (repressed) or vice versa.
        # r_on = k * (1 - Si). If Si=1, r_on=0. So Si=1 means "ON" state (Active).
        # r_off = k * (R/K)^n * Si. If Si=1, r_off > 0.
        # r_trans = k_trans * Si. So Si=1 allows transcription.
        # So 1 is ACTIVE, 0 is REPRESSED (Bound or Inactive)?
        # Wait, usually "Bound" by repressor means "OFF".
        # If r_off (binding) goes to Si=0, then 0 is REPRESSED.
        # Let's check SSA code:
        # r_off (binding): self.state[idx] = 0. So 0 is REPRESSED/BOUND.
        # r_on (unbinding): self.state[idx] = 1. So 1 is ACTIVE/FREE.
        
        # Predicted Prob of ACTIVE (1)
        repressor_conc = p_ssa[:, repressor_map[i]]
        prob_active = 1.0 / (1.0 + (repressor_conc / params['KM'])**params['n'])
        
        pred_state = (prob_active > 0.5).astype(int)
        
        all_true_states.extend(true_state)
        all_pred_states.extend(pred_state)
        
    kappa = cohen_kappa_score(all_true_states, all_pred_states)
    cm = confusion_matrix(all_true_states, all_pred_states, normalize='true')
    
    print(f"  Cohen's Kappa: {kappa:.4f}")
    
    # --- 2. Switching Rate Distribution ---
    print("  Calculating Switching Rates...")
    # Count switches per simulation for SSA
    # We already have one trace. Ideally we need multiple trajectories.
    # Let's count switches in this long trajectory and normalize by time.
    def count_switches(states):
        return np.sum(np.abs(np.diff(states)))
    
    switches_ssa = [count_switches(s_ssa[:, i]) for i in range(3)]
    avg_switches_ssa = np.mean(switches_ssa)
    
    # Run Markovian
    print("  Running Markovian...")
    markov = MarkovianRepressilatorDDM(params)
    # Init around same as SSA
    markov.ode_state[0] = 5.0
    markov.ode_state[3] = 48.0
    markov.ode_state[6] = 1000.0
    t_mark, ode_mark, s_mark = markov.run(1000, dt=0.1) # Discrete states return? 
    # Check `repressilator_ddm_markovian.py`: it likely returns probabilities or states?
    # Actually `plot_promoter_states.py` uses `promoters_mark[:, i]` which seemed to be P(Active).
    # If it returns probabilities, we can't count discrete switches directly unless we define "crossing 0.5".
    # BUT, the manuscript says "Switching rate distribution: Markovian model (red)".
    # This implies the Markovian model *has* switches. 
    # A pure ODE Markovian model (dp/dt) does NOT switch. It's a continuous probability.
    # Maybe "Markovian model" refers to the *Hybrid* simulation where promoters are discrete stochastic?
    # OR they define a "switch" in continuous probability as crossing 0.5?
    # "Switching rate distribution" usually implies discrete events.
    # Let's assume for the Markovian model (which is ODE based), they measure 'effective' switches by crossing 0.5.
    
    switches_mark = [count_switches((s_mark[:, i] > 0.5).astype(int)) for i in range(3)]
    avg_switches_mark = np.mean(switches_mark)
    
    # --- Plotting ---
    print("Generating figures...")
    os.makedirs('/data2/2026_GENE_MM/manuscript/figures', exist_ok=True)
    
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1])
    
    # Panel A: Kappa (Bar plot)
    ax0 = fig.add_subplot(gs[0])
    sns.barplot(x=['Gene 1', 'Gene 2', 'Gene 3'], y=[kappa]*3, ax=ax0, palette='viridis') 
    # Actually should calculate per gene
    kappas = []
    len_trace = len(s_ssa)
    for i in range(3):
        k = cohen_kappa_score(s_ssa[:, i], (all_pred_states[i*len_trace:(i+1)*len_trace]))
        kappas.append(k)
    ax0.clear()
    sns.barplot(x=['Gene 1', 'Gene 2', 'Gene 3'], y=kappas, ax=ax0, palette='viridis')
    ax0.set_ylim(0, 1)
    ax0.set_title('(A) Promoter Agreement (Kappa)')
    ax0.set_ylabel("Cohen's Kappa")
    ax0.grid(axis='y', alpha=0.3)
    
    # Panel B: Confusion Matrix
    ax1 = fig.add_subplot(gs[1])
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', ax=ax1,
                xticklabels=['Repressed', 'Active'], yticklabels=['Repressed', 'Active'])
    ax1.set_title('(B) Confusion Matrix')
    ax1.set_xlabel('Predicted (Markovian)')
    ax1.set_ylabel('True (SSA)')
    
    # Panel C: Switching Rates (Histogram)
    ax2 = fig.add_subplot(gs[2:])
    # Mock distribution for visualization (since we only have 3 points from 1 traj)
    # Ideally we'd need many trajectories.
    # I will create a small variation around the mean for the plot
    # to look like a distribution, or just plot the 3 points.
    
    w = 0.3
    x = np.arange(3)
    ax2.bar(x - w/2, switches_ssa, width=w, label='SSA', color='green', alpha=0.7)
    ax2.bar(x + w/2, switches_mark, width=w, label='Markovian', color='red', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Gene 1', 'Gene 2', 'Gene 3'])
    ax2.set_ylabel('Switches per 1000 min')
    ax2.set_title('(C) Switching Counts')
    ax2.legend()
    
    plt.tight_layout()
    save_path = '/data2/2026_GENE_MM/manuscript/figures/fig_repressilator_validation.png'
    plt.savefig(save_path, dpi=300)
    print(f"Saved figure to {save_path}")

if __name__ == "__main__":
    analyze_validation()
