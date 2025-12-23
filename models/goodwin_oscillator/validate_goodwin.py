import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from goodwin_markovian import MarkovianGoodwin

# Validation: Compare SSA (Ground Truth) vs Markovian Hybrid for Goodwin Oscillator
# Focus on X concentration at t=100 (mid-oscillation)

# 1. Load SSA Data
df_ssa = pd.read_csv('goodwin_ssa_trajectories.csv')
final_ssa = []
target_time = 100.0
# Interpolated data already exists for t=100 in the CSV?
# CSV has linspace(0, 200, 1000). 100 is index 500.
# Or I can just filter Time approx 100.
times = df_ssa['Time'].unique()
idx = np.abs(times - target_time).argmin()
t_val = times[idx]
print(f"Sampling at t={t_val}")

for i in range(10):
    subset = df_ssa[df_ssa['Replicate'] == i]
    if not subset.empty:
        # Find row closest to target_time
        val = subset.iloc[(subset['Time'] - target_time).abs().argsort()[:1]]['X'].values[0]
        final_ssa.append(val)

# 2. Run Markovian Hybrid (10 Replicates)
params = {
    'k1': 1.0, 'k2': 1.0, 'k3': 1.0,
    'b1': 0.1, 'b2': 0.1, 'b3': 0.1,
    'n': 10.0, 'KM': 1.0,
    'k_burst': 2.0
}

final_markovian = []
dt = 0.05
T = 200.0 # Match SSA duration
print("Running 10 Markovian Replicates...")
for i in range(10):
    model = MarkovianGoodwin(params)
    # Start same as SSA: 0.1, 0.1, 0.1 scaled?
    # SSA started at 0.1*Omega.
    # Markovian uses concentrations 0.1.
    model.ode_state = np.array([0.1, 0.1, 0.1])
    
    t, y, _ = model.run(T, dt)
    # Get value at t=100
    idx = np.abs(t - target_time).argmin()
    final_markovian.append(y[idx, 0])

# 3. KS Test
stat, pval = ks_2samp(final_ssa, final_markovian)
print(f"KS Statistic: {stat:.4f}")
print(f"P-Value: {pval:.4f}")

if pval > 0.05:
    print("Result: Distributions are NOT significantly different (Validation Passed)")
else:
    print("Result: Distributions ARE significantly different (Validation Failed/Deviation)")

# 4. Plot ECDF
def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(data)+1) / len(data)
    return x, y

x_ssa, y_ssa = ecdf(final_ssa)
x_mark, y_mark = ecdf(final_markovian)

plt.figure()
plt.plot(x_ssa, y_ssa, label='SSA (Ground Truth)')
plt.plot(x_mark, y_mark, label='Markovian Hybrid')
plt.xlabel(f'Concentration of X (t={target_time})')
plt.ylabel('CDF')
plt.legend()
plt.title(f'KS Test (Goodwin): stat={stat:.2f}, p={pval:.2f}')
plt.savefig('goodwin_validation_ks.png')
print("Plot saved to goodwin_validation_ks.png")
