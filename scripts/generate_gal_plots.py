import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for Nature/Science
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

def plot_dose_response():
    try:
        df = pd.read_csv('gal_dose_response_data.csv')
        
        fig, ax = plt.subplots(figsize=(4, 3))
        
        # Plot Fold Change of RNA vs Dose
        # Normalize to 0 dose (or lowest dose)
        # Actually usually normalize to WT basal or similar. 
        # Here we plot absolute or fold change.
        # Let's plot R1 (GAL1 mRNA) and R3 (GAL3 mRNA)
        
        ax.plot(df['Dose'], df['R1'], 'o-', label='GAL1 mRNA', color='#D55E00')
        ax.plot(df['Dose'], df['R3'], 's-', label='GAL3 mRNA', color='#0072B2')
        
        ax.set_xscale('log')
        ax.set_xlabel('Galactose (mM)')
        ax.set_ylabel('mRNA Molecules')
        ax.set_title('Dose-Response')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/fig_gal_dose_response.png', dpi=300)
        plt.close()
        print("Generated fig_gal_dose_response.png")
    except Exception as e:
        print(f"Could not plot dose response: {e}")

def plot_timeseries():
    try:
        df = pd.read_csv('gal_timeseries_data.csv')
        
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        
        # Panel A: Signals (GAI, G3i, G80d)
        ax = axes[0]
        ax.plot(df['Time'], df['GAI'], label='Int. Gal', color='gray')
        # Scale G3i for visibility?
        ax.plot(df['Time'], df['G3i'], label='Active Gal3', color='#009E73') 
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Molecules (Signal)')
        ax.set_title('Signaling')
        ax.legend()
        
        # Panel B: GAL1 mRNA Induction
        ax = axes[1]
        ax.plot(df['Time'], df['R1'], color='#D55E00')
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('GAL1 mRNA')
        ax.set_title('Transciptional Response')
        
        # Panel C: Promoter Activity
        ax = axes[2]
        # Rolling average for smoother look if it's very stochastic
        window = 50 # 0.5 min
        ax.plot(df['Time'], df['Act_GAL1'].rolling(window).mean(), label='GAL1 Act', color='#D55E00', alpha=0.8)
        ax.plot(df['Time'], df['Act_GAL3'].rolling(window).mean(), label='GAL3 Act', color='#0072B2', alpha=0.6)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Active Fraction')
        ax.set_title('Promoter Activity')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/fig_gal_timeseries.png', dpi=300)
        plt.close()
        print("Generated fig_gal_timeseries.png")
    except Exception as e:
        print(f"Could not plot timeseries: {e}")

def plot_promoter_states():
    try:
        df = pd.read_csv('gal_promoter_states.csv')
        
        fig, ax = plt.subplots(figsize=(6, 2.5))
        
        # Plot discrete state trajectory
        # k=number of Gal4, m=number of Gal80
        # Let's define abstract state index or just separate k and m
        
        # Use steps to show discrete jumps
        ax.step(df['Time'], df['k'], where='post', label='# Gal4 Bound', color='#0072B2', lw=1.5)
        ax.step(df['Time'], df['m'], where='post', label='# Gal80 Bound', color='#D55E00', lw=1.5, linestyle='--')
        
        # Shade active regions (k > m)
        # Use fill_between where k > m
        ax.fill_between(df['Time'], 0, 5, where=(df['k'] > df['m']), color='green', alpha=0.1, label='Active State')
        
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Bound Molecules')
        ax.set_title('GAL1 Promoter State Dynamics (Stochastic)')
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.legend(loc='upper right', ncol=3, fontsize='small')
        ax.set_ylim(-0.5, 4.5)
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/fig_gal_promoter_states.png', dpi=300)
        plt.close()
        print("Generated fig_gal_promoter_states.png")
    except Exception as e:
        print(f"Could not plot promoter states: {e}")

if __name__ == "__main__":
    plot_dose_response()
    plot_timeseries()
    plot_promoter_states()
