import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from toggle_markovian import MarkovianToggle

# Validation: Compare SSA (Ground Truth) vs Markovian Hybrid
# 1. Load SSA Data
df_ssa = pd.read_csv('toggle_ssa_trajectories.csv')
# Filter for replicates that started with High U (which is all of them in current script)
# Get distributions at t=100
final_ssa = []
for i in range(10):
    subset = df_ssa[df_ssa['Replicate'] == i]
    if not subset.empty:
        final_ssa.append(subset.iloc[-1]['U'])

# 2. Run Markovian Hybrid Data (10 Replicates)
params = {
    'alpha1': 156.25,
    'alpha2': 15.6,
    'beta': 2.5,
    'gamma': 1.0,
    'k_burst': 1.0
}

final_markovian = []
dt = 0.05
T = 100.0
print("Running 10 Markovian Replicates...")
for i in range(10):
    model = MarkovianToggle(params)
    model.ode_state = np.array([150.0, 0.0]) # High U start
    model.promoters = np.array([1, 0])
    
    _, y, _ = model.run(T, dt)
    final_markovian.append(y[-1, 0]) # U at end

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
plt.xlabel('Concentration of U (t=100)')
plt.ylabel('CDF')
plt.legend()
plt.title(f'KS Test: stat={stat:.2f}, p={pval:.2f}')
plt.savefig('toggle_validation_ks.png')
print("Plot saved to toggle_validation_ks.png")
