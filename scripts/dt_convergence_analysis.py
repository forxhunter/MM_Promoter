import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from scipy.signal import find_peaks
import os

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

OUTPUT_DIR = '../figures'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==============================================================================
# Model Definition
# ==============================================================================

class MarkovianRepressilator:
    def __init__(self, params):
        self.params = params
        self.time = 0.0
        self.ode_state = np.zeros(9)
        self.promoter_state = np.ones(3, dtype=int)
        self.history_t = []
        self.history_ode = []
        
        # Initialize
        self.ode_state[0] = 5.0
        self.ode_state[3] = 48.0
        self.ode_state[6] = 1000.0

    def transition_rates(self, p):
        # Repressors: p3->1, p1->2, p2->3
        K = self.params['KM']
        n = self.params['n']
        k_burst = self.params.get('k_burst', 0.1)
        repressors = [p[2], p[0], p[1]]
        rates = []
        for i in range(3):
            R = repressors[i]
            k_on = k_burst # Rate of switching ON (repressor unbinding?)
            # Wait, usually k_on is constant?
            # If R binds to promoter to REPRESS it:
            # Empty (Active) + R -> Repressed (Inactive)
            # Rate = k_bind * R
            # Repressed -> Empty
            # Rate = k_unbind
            
            # The params usually give "k_burst" which is related to active state.
            
            # Let's assume:
            # Active (1) -> Inactive (0) rate: k_off = k_bind * (R/K)^n ??
            # Inactive (0) -> Active (1) rate: k_on = const
            
            # In the viewing file:
            # k_on = k_burst
            # k_off = k_on * (R/K)**n
            # This looks like k_on is transition TO 1?
            # And k_off is transition TO 0?
            
            # Correct logic:
            # State 1 (Active) -> 0 (Repressed) happens when R binds. Rate depends on R.
            # State 0 (Repressed) -> 1 (Active) happens when R unbinds. Rate is constant.
            
            # The visible code had:
            # k_on = k_burst 
            # k_off = k_on * (R/K)**n
            
            # And loop:
            # if state == 0: p_on = 1 - exp(-k_on*dt) ... -> state 1
            # if state == 1: p_off = 1 - exp(-k_off*dt) ... -> state 0
            
            # So:
            # 0 -> 1 rate is k_on (constant)
            # 1 -> 0 rate is k_off (dependent on R)
            
            k_on_val = k_burst
            k_off_val = k_burst * (R/K)**n
            
            rates.append((k_on_val, k_off_val))
        return rates

    def run(self, total_time, dt):
        steps = int(total_time / dt)
        
        t_hist = np.zeros(steps+1)
        ode_hist = np.zeros((steps+1, 9))
        
        t_hist[0] = self.time
        ode_hist[0] = self.ode_state
        
        rng = np.random.default_rng()
        
        def dxdt(x, t, promoter_state):
            m = x[0:3]
            u = x[3:6]
            p_ = x[6:9]
            dm = self.params['k_trans'] * promoter_state - self.params['k_deg_m'] * m + self.params['k_leak']
            du = self.params['k_transl'] * m - self.params['k_fold'] * u - self.params['k_deg_p'] * u
            dp = self.params['k_fold'] * u - self.params['k_deg_p'] * p_
            return np.concatenate([dm, du, dp])

        for i in range(steps):
            # 1. Update Promoters (Markovian Step)
            p_conc = self.ode_state[6:9]
            rates = self.transition_rates(p_conc)
            
            for j in range(3):
                k_on, k_off = rates[j]
                if self.promoter_state[j] == 0:
                    if rng.random() < 1 - np.exp(-k_on * dt):
                        self.promoter_state[j] = 1
                else:
                    if rng.random() < 1 - np.exp(-k_off * dt):
                        self.promoter_state[j] = 0
            
            # 2. ODE Step
            # Use odeint over [0, dt]
            # Zero-Order Hold for promoter state
            
            # Optimization: Manual RK4 step might be faster but odeint is safer
            # Given dt is small (e.g. 0.1), maybe explicit method is fine?
            # But let's stick to odeint for accuracy at larger dt
            
            sol = odeint(dxdt, self.ode_state, [0, dt], args=(self.promoter_state,))
            self.ode_state = sol[-1]
            self.time += dt
            
            t_hist[i+1] = self.time
            ode_hist[i+1] = self.ode_state
            
        return t_hist, ode_hist

# ==============================================================================
# Analysis
# ==============================================================================

def analyze_convergence():
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
    
    dt_values = [0.1, 1.0, 5.0, 10.0, 20.0] # 0.01 is too slow for python script in loop?
    # Reference should be small dt. Let's use dt=0.1 as reasonable baseline, maybe check 0.05.
    # The user request mentioned "dt convergence".
    
    results = []
    
    for dt in dt_values:
        print(f"Simulating dt={dt}...")
        periods = []
        amplitudes = []
        
        # Run 3 replicates
        for rep in range(3):
            model = MarkovianRepressilator(params)
            # Run for long time to get stable oscillations
            # ~20000 min
            t, ode = model.run(15000, dt)
            
            # Analyze last 10000 min
            mask = t > 5000
            p1 = ode[mask, 6] # p1 trajectory
            
            # Find peaks
            min_dist = max(1, int(100/dt))
            peaks, _ = find_peaks(p1, distance=min_dist, prominence=10) # Lower prominence
            
            print(f"  Rep {rep}: Found {len(peaks)} peaks")
            
            if len(peaks) > 1:
                peak_times = t[mask][peaks]
                peak_vals = p1[peaks]
                
                # Period
                pers = np.diff(peak_times)
                mean_p = np.mean(pers)
                periods.append(mean_p)
                
                # Amplitude
                mean_amp = np.mean(peak_vals)
                amplitudes.append(mean_amp)
        
        if periods:
            results.append({
                'dt': dt, 
                'period_mean': np.mean(periods), 
                'period_std': np.std(periods),
                'amp_mean': np.mean(amplitudes),
                'amp_std': np.std(amplitudes)
            })
        else:
            print(f"  Warning: No peaks found for dt={dt}")
            # Add dummy row to prevent plotting error if partial results?
            # Or just skip.
            
    df = pd.DataFrame(results)
    if df.empty:
        print("Error: No data to plot.")
        return

    df.to_csv('dt_convergence_results.csv', index=False)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    
    # Period Convergence
    ax = axes[0]
    ax.errorbar(df['dt'], df['period_mean'], yerr=df['period_std'], fmt='o-', capsize=3, color='#0072B2')
    ax.set_xlabel('Communication Step dt (min)')
    ax.set_ylabel('Oscillation Period (min)')
    ax.set_title('Period Convergence')
    ax.set_xscale('log')
    
    # Amplitude Convergence
    ax = axes[1]
    ax.errorbar(df['dt'], df['amp_mean'], yerr=df['amp_std'], fmt='s-', capsize=3, color='#D55E00')
    ax.set_xlabel('Communication Step dt (min)')
    ax.set_ylabel('Peak Amplitude (molecules)')
    ax.set_title('Amplitude Convergence')
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_dt_convergence_repressilator.png', dpi=300)
    print("Saved fig_dt_convergence_repressilator.png")

def plot_trajectory_comparison():
    # Run short trajectories for visual comparison
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
    
    # dt = 0.5 (Good) vs dt = 20 (Bad)
    dt_good = 0.5
    dt_bad = 20.0
    
    print("Simulating visual comparison...")
    model1 = MarkovianRepressilator(params)
    t1, ode1 = model1.run(1000, dt_good)
    
    # Reset seed for fair comparison? Hard with stochasticity.
    model2 = MarkovianRepressilator(params)
    t2, ode2 = model2.run(1000, dt_bad)
    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t1, ode1[:, 6], '-', color='#009E73', label=f'dt={dt_good} min')
    ax.plot(t2, ode2[:, 6], 'o-', color='#D55E00', markersize=3, label=f'dt={dt_bad} min', alpha=0.7)
    
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Repressor 1 (molecules)')
    ax.set_title('Effect of Time Step on Dynamics')
    ax.legend()
    ax.set_xlim(0, 1000)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_dt_comparison_trajectories.png', dpi=300)
    print("Saved fig_dt_comparison_trajectories.png")

if __name__ == "__main__":
    analyze_convergence()
    plot_trajectory_comparison()
