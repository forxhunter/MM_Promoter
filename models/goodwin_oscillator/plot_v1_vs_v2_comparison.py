import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from goodwin_ssa import SSA_Goodwin
from goodwin_markovian import MarkovianGoodwin
from goodwin_markovian_v2 import MarkovianGoodwinV2

# Parameters
params = {
    'k1': 1.0, 'k2': 1.0, 'k3': 1.0,
    'b1': 0.1, 'b2': 0.1, 'b3': 0.1,
    'n': 10.0, 'KM': 1.0,  # n=10 matches goodwin_markovian.py
    'k_burst': 2.0
}

# 1. ODE
def goodwin_ode_func(y, t, params):
    x, y_var, z = y
    k1 = params['k1']
    k2 = params['k2']
    k3 = params['k3']
    b1 = params['b1']
    b2 = params['b2']
    b3 = params['b3']
    n = params['n']
    KM = params['KM']
    
    hill = 1.0 / (1.0 + (z/KM)**n)
    
    dx = k1 * hill - b1 * x
    dy = k2 * x - b2 * y_var
    dz = k3 * y_var - b3 * z
    
    return [dx, dy, dz]

y0 = [0.1, 0.1, 0.1]
t_ode = np.linspace(0, 500, 5000)
ode_sol = odeint(goodwin_ode_func, y0, t_ode, args=(params,))

# 2. SSA
print("Running Goodwin SSA...")
ssa = SSA_Goodwin(params, Omega=50.0)
t_ssa, x_ssa = ssa.run(500)

# 3. Markovian V1
print("Running Goodwin Markovian V1...")
model_v1 = MarkovianGoodwin(params)
t_v1, yz_v1, p_v1 = model_v1.run(500, dt=0.001)
x_v1 = yz_v1[:, 0]

# 4. Markovian V2 (Discrete mRNA)
print("Running Goodwin Markovian V2 (Discrete mRNA)...")
model_v2 = MarkovianGoodwinV2(params, Omega=50.0)
t_v2, x_v2, yz_v2, p_v2 = model_v2.run(500, dt=0.001)

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Full trajectory comparison
axes[0, 0].plot(t_ode, ode_sol[:, 0], 'k-', linewidth=2, label='ODE', alpha=0.8)
axes[0, 0].plot(t_ssa, x_ssa, 'b-', linewidth=1, label='SSA', alpha=0.6)
axes[0, 0].plot(t_v1, x_v1, 'r-', linewidth=1, label='Markovian V1', alpha=0.6)
axes[0, 0].plot(t_v2, x_v2, 'g-', linewidth=1, label='Markovian V2 (Discrete mRNA)', alpha=0.6)
axes[0, 0].set_ylabel('mRNA (X) Concentration', fontsize=11)
axes[0, 0].set_xlabel('Time', fontsize=11)
axes[0, 0].set_title('Full Trajectory Comparison', fontsize=12, fontweight='bold')
axes[0, 0].legend(loc='upper right')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Zoomed view (first 100 time units)
axes[0, 1].plot(t_ode[t_ode <= 100], ode_sol[t_ode <= 100, 0], 'k-', linewidth=2, label='ODE', alpha=0.8)
axes[0, 1].plot(t_ssa[t_ssa <= 100], x_ssa[t_ssa <= 100], 'b-', linewidth=1, label='SSA', alpha=0.6)
axes[0, 1].plot(t_v1[t_v1 <= 100], x_v1[t_v1 <= 100], 'r-', linewidth=1, label='Markovian V1', alpha=0.6)
axes[0, 1].plot(t_v2[t_v2 <= 100], x_v2[t_v2 <= 100], 'g-', linewidth=1, label='Markovian V2', alpha=0.6)
axes[0, 1].set_ylabel('mRNA (X) Concentration', fontsize=11)
axes[0, 1].set_xlabel('Time', fontsize=11)
axes[0, 1].set_title('Zoomed View (0-100)', fontsize=12, fontweight='bold')
axes[0, 1].legend(loc='upper right')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Amplitude comparison (histogram of peaks)
def find_peaks(signal, threshold=0.5):
    """Simple peak finding"""
    peaks = []
    for i in range(1, len(signal)-1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > threshold:
            peaks.append(signal[i])
    return peaks

ode_peaks = find_peaks(ode_sol[:, 0])
ssa_peaks = find_peaks(x_ssa)
v1_peaks = find_peaks(x_v1)
v2_peaks = find_peaks(x_v2)

axes[1, 0].hist([ode_peaks, ssa_peaks, v1_peaks, v2_peaks], 
                bins=15, alpha=0.6, label=['ODE', 'SSA', 'V1', 'V2'])
axes[1, 0].set_xlabel('Peak Amplitude', fontsize=11)
axes[1, 0].set_ylabel('Count', fontsize=11)
axes[1, 0].set_title('Oscillation Peak Amplitude Distribution', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Statistics table
stats_text = f"""
Model Comparison Statistics:

ODE:
  Mean Peak: {np.mean(ode_peaks):.3f}
  Std Peak:  {np.std(ode_peaks):.3f}

SSA:
  Mean Peak: {np.mean(ssa_peaks):.3f}
  Std Peak:  {np.std(ssa_peaks):.3f}

Markovian V1:
  Mean Peak: {np.mean(v1_peaks):.3f}
  Std Peak:  {np.std(v1_peaks):.3f}

Markovian V2 (Discrete mRNA):
  Mean Peak: {np.mean(v2_peaks):.3f}
  Std Peak:  {np.std(v2_peaks):.3f}

Improvement (V2 vs V1):
  Peak Amplitude: {(np.mean(v2_peaks) / np.mean(v1_peaks) - 1) * 100:+.1f}%
  Closer to SSA:  {abs(np.mean(v2_peaks) - np.mean(ssa_peaks)) < abs(np.mean(v1_peaks) - np.mean(ssa_peaks))}
"""

axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=axes[1, 1].transAxes)
axes[1, 1].axis('off')

fig.suptitle('Goodwin Oscillator: V1 vs V2 Comparison', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('../20251218_goodwin_v2/goodwin_v1_vs_v2_comparison.png', dpi=150)
print("Saved ../20251218_goodwin_v2/goodwin_v1_vs_v2_comparison.png")

# Save statistics
stats_df = pd.DataFrame({
    'Model': ['ODE', 'SSA', 'Markovian_V1', 'Markovian_V2'],
    'Mean_Peak': [np.mean(ode_peaks), np.mean(ssa_peaks), np.mean(v1_peaks), np.mean(v2_peaks)],
    'Std_Peak': [np.std(ode_peaks), np.std(ssa_peaks), np.std(v1_peaks), np.std(v2_peaks)],
    'Num_Peaks': [len(ode_peaks), len(ssa_peaks), len(v1_peaks), len(v2_peaks)]
})
stats_df.to_csv('../20251218_goodwin_v2/goodwin_v1_vs_v2_statistics.csv', index=False)
print("Saved ../20251218_goodwin_v2/goodwin_v1_vs_v2_statistics.csv")
print("\n" + stats_text)
