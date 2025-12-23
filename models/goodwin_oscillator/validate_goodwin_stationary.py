import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# Validation: Goodwin Oscillator Stationary Distribution
# Compares the distribution of [X] values over the long-term simulation to avoid phase issues.

def get_stationary_distribution(csv_file, species='X', start_frac=0.5):
    df = pd.read_csv(csv_file)
    max_time = df['Time'].max()
    # Filter for last 50% of time to ignore transient
    subset = df[df['Time'] > max_time * start_frac]
    return subset[species].values

# 1. Load Data
ssa_vals = get_stationary_distribution('goodwin_ssa_trajectories.csv')
mark_vals = get_stationary_distribution('goodwin_markovian_trajectories.csv')

# 2. Downsample
print(f"SSA Mean: {np.mean(ssa_vals):.4f}, Std: {np.std(ssa_vals):.4f}")
print(f"Mark Mean: {np.mean(mark_vals):.4f}, Std: {np.std(mark_vals):.4f}")

if len(ssa_vals) > 5000: np.random.shuffle(ssa_vals); ssa_vals = ssa_vals[:5000]
if len(mark_vals) > 5000: np.random.shuffle(mark_vals); mark_vals = mark_vals[:5000]

# 3. KS Test
stat, pval = ks_2samp(ssa_vals, mark_vals)
print(f"KS Statistic: {stat:.4f}")
print(f"P-Value: {pval:.4e}")

if pval > 0.05:
    print("Result: Distributions Matches (Validation Passed)")
else:
    print("Result: Distributions Differ (Validation Failed)")

# 4. Plot
def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(data)+1) / len(data)
    return x, y

x_s, y_s = ecdf(ssa_vals)
x_m, y_m = ecdf(mark_vals)

plt.figure(figsize=(8, 6))
plt.plot(x_s, y_s, label='SSA (Ground Truth)', linewidth=2)
plt.plot(x_m, y_m, label='Markovian Hybrid', linewidth=2, linestyle='--')
plt.xlabel('X Concentration')
plt.ylabel('CDF')
plt.legend()
plt.title(f'Goodwin Validation (Stationary Dist)\nKS={stat:.2f}, p={pval:.2e}')
plt.savefig('goodwin_validation_stationary_ks.png')
print("Plot saved to goodwin_validation_stationary_ks.png")
