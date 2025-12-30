
import os
import subprocess
import time

def run_script(script_name, description):
    print(f"============================================================")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"============================================================")
    
    start_time = time.time()
    result = subprocess.run(['python', script_name], capture_output=False)
    end_time = time.time()
    
    if result.returncode == 0:
        print(f"SUCCESS (Time: {end_time - start_time:.2f} s)")
    else:
        print(f"FAILED (Return Code: {result.returncode})")
        
    print("\n")

def main():
    if not os.path.exists("figures"):
        os.makedirs("figures")
        print("Created figures/ directory")

    # Figure 1: GAL Dose Response
    run_script("scripts/figure_gal_dose_response.py", "Figure 1: GAL Dose-Response Curves")
    
    # Figure 2: GAL Time Series
    run_script("scripts/figure_gal_timeseries.py", "Figure 2: GAL Time Series Dynamics")
    
    # Figure 3: GAL Promoter States
    # Note: This requires generating data first if not present
    if not os.path.exists("gal_promoter_states.csv"):
        print("Generating GAL promoter state data (this may take a few minutes)...")
        run_script("scripts/hybrid_gal_simulator.py --mode promoter", "Generating GAL Promoter Data")
    run_script("scripts/figure_gal_promoter_states.py", "Figure 3: GAL Promoter State Dynamics")

    # Simple Systems Benchmark
    run_script("scripts/generate_combined_simple.py", "Figure 4: Simple Systems Benchmarks (Repressilator, Goodwin, Toggle)")
    
    # Advanced Systems
    run_script("scripts/generate_advanced_plots.py", "Figure 5: Advanced Systems (I1-FFL, p53, NF-kB)")
    
    # Analysis
    run_script("scripts/dt_convergence_analysis.py", "S1 Figure: Time Step Convergence Analysis")
    run_script("scripts/protein_stochasticity_analysis.py", "S2 Figure: Protein Stochasticity Analysis")
    
    print("All tasks completed.")

if __name__ == "__main__":
    main()
