"""
Validation utilities for comparing ODE, SSA, and Markovian models.
Provides standardized statistical tests and plotting functions.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr
from sklearn.metrics import cohen_kappa_score

def compute_promoter_similarity(ssa_trace, mark_trace, ssa_time, mark_time):
    """
    Compute statistical similarity between SSA and Markovian promoter traces.
    
    Returns:
        dict with keys: kappa, agreement, correlation
    """
    # Interpolate to common time grid
    t_common = np.linspace(max(ssa_time[0], mark_time[0]), 
                          min(ssa_time[-1], mark_time[-1]), 1000)
    
    # Nearest neighbor interpolation (for discrete states)
    ssa_interp = np.zeros(len(t_common))
    mark_interp = np.zeros(len(t_common))
    
    for i, t in enumerate(t_common):
        ssa_idx = np.argmin(np.abs(ssa_time - t))
        mark_idx = np.argmin(np.abs(mark_time - t))
        ssa_interp[i] = ssa_trace[ssa_idx]
        mark_interp[i] = mark_trace[mark_idx]
    
    # Cohen's Kappa (categorical agreement)
    kappa = cohen_kappa_score(ssa_interp, mark_interp)
    
    # Fraction of time in agreement
    agreement = np.mean(ssa_interp == mark_interp)
    
    # Correlation
    correlation = np.corrcoef(ssa_interp, mark_interp)[0, 1]
    
    return {
        'kappa': kappa,
        'agreement': agreement,
        'correlation': correlation
    }

def compute_switching_stats(promoter_trace, time_trace):
    """
    Compute ON/OFF durations and switching frequency.
    
    Returns:
        dict with keys: switch_freq, mean_on_duration, num_switches
    """
    switches = np.diff(promoter_trace)
    switch_times = time_trace[1:][switches != 0]
    
    if len(switch_times) < 2:
        return {'switch_freq': 0, 'mean_on_duration': 0, 'num_switches': 0}
    
    switch_freq = len(switch_times) / (time_trace[-1] - time_trace[0])
    
    # ON durations
    on_starts = time_trace[1:][switches == 1]
    on_ends = time_trace[1:][switches == -1]
    
    if len(on_ends) > 0 and len(on_starts) > 0:
        if on_ends[0] < on_starts[0]:
            on_ends = on_ends[1:]
        min_len = min(len(on_starts), len(on_ends))
        on_durations = on_ends[:min_len] - on_starts[:min_len]
        mean_on = np.mean(on_durations) if len(on_durations) > 0 else 0
    else:
        mean_on = 0
    
    return {
        'switch_freq': switch_freq,
        'mean_on_duration': mean_on,
        'num_switches': len(switch_times)
    }

def find_peaks(signal, threshold=0.5):
    """Simple peak finding algorithm."""
    peaks = []
    for i in range(1, len(signal)-1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > threshold:
            peaks.append(signal[i])
    return np.array(peaks)

def compute_trajectory_similarity(ssa_trace, mark_trace):
    """
    Compute statistical similarity between SSA and Markovian trajectories.
    
    Returns:
        dict with keys: ks_statistic, ks_pvalue, mean_diff, std_diff
    """
    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = ks_2samp(ssa_trace, mark_trace)
    
    # Mean and std difference
    mean_diff = abs(np.mean(ssa_trace) - np.mean(mark_trace))
    std_diff = abs(np.std(ssa_trace) - np.std(mark_trace))
    
    return {
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval,
        'mean_diff': mean_diff,
        'std_diff': std_diff
    }

def compute_peak_statistics(ode_trace, ssa_trace, mark_trace, threshold=0.5):
    """
    Compare peak amplitudes across models.
    
    Returns:
        dict with peak statistics for each model
    """
    ode_peaks = find_peaks(ode_trace, threshold)
    ssa_peaks = find_peaks(ssa_trace, threshold)
    mark_peaks = find_peaks(mark_trace, threshold)
    
    return {
        'ode_mean_peak': np.mean(ode_peaks) if len(ode_peaks) > 0 else np.nan,
        'ode_std_peak': np.std(ode_peaks) if len(ode_peaks) > 0 else np.nan,
        'ode_num_peaks': len(ode_peaks),
        'ssa_mean_peak': np.mean(ssa_peaks) if len(ssa_peaks) > 0 else np.nan,
        'ssa_std_peak': np.std(ssa_peaks) if len(ssa_peaks) > 0 else np.nan,
        'ssa_num_peaks': len(ssa_peaks),
        'mark_mean_peak': np.mean(mark_peaks) if len(mark_peaks) > 0 else np.nan,
        'mark_std_peak': np.std(mark_peaks) if len(mark_peaks) > 0 else np.nan,
        'mark_num_peaks': len(mark_peaks)
    }

def save_statistics(stats_dict, filename):
    """Save statistics dictionary to CSV."""
    df = pd.DataFrame([stats_dict])
    df.to_csv(filename, index=False)
    print(f"Saved statistics to {filename}")
