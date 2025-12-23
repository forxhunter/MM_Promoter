
import numpy as np
import pandas as pd
from multiprocessing import Pool
from repressilator_ddm_markovian import MarkovianRepressilatorDDM

def run_markovian_worker(seed):
    np.random.seed(seed)
    params = {
        'k_trans': 0.5,
        'k_leak': 5e-4,
        'k_trans': 0.5,
        'k_leak': 5e-4,
        'k_transl': 0.16,
        'k_fold': 1.0/60,
        'k_deg_m': np.log(2)/2,
        'k_deg_p': np.log(2)/600,
        'n': 2.0,
        'KM': 40.0,
        'k_burst': 0.05
    }
    
    model = MarkovianRepressilatorDDM(params)
    model.ode_state[0] = 5.0
    dt = 0.01
    T = 2000
    
    # Run
    t_hist, ode_hist, p_hist = model.run(T, dt)
    
    # Interp / Sample last half
    # t_hist is list, ode_hist is list of arrays
    t_arr = np.array(t_hist)
    # Extract Protein 0 (index 2 in ODE? No. M0, P0u, P0f... P0f is index 2)
    # ode_state: [M0, P0u, P0f, M1...]
    # indices: 0, 1, 2
    p0_arr = np.array([x[2] for x in ode_hist])
    
    t_interp = np.linspace(1000, 2000, 1000)
    p_interp = np.interp(t_interp, t_arr, p0_arr)
    return p_interp

if __name__ == "__main__":
    N_REPS = 1000
    print(f"Starting {N_REPS} Replicates Markovian...")
    
    with Pool(processes=20) as pool:
        results = pool.map(run_markovian_worker, range(N_REPS))
        
    data = np.array(results).flatten()
    print(f"Data Shape: {data.shape}")
    print(f"Mark Mean: {np.mean(data):.4f}")
    
    df = pd.DataFrame({'P_folded': data})
    df.to_csv('repressilator_markovian_1000_stats.csv', index=False)
