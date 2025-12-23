import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class MarkovianRepressilatorDDM:
    def __init__(self, params):
        self.params = params
        self.time = 0.0
        # ODE: 9 vars (m1,m2,m3, u1,u2,u3, p1,p2,p3)
        # Discrete: 3 vars (S1, S2, S3)
        self.ode_state = np.zeros(9)
        self.promoter_state = np.ones(3, dtype=int)
        self.history_t = []
        self.history_ode = []
        self.history_promoters = []

    def transition_rates(self, p):
        # Repressors: p3->1, p1->2, p2->3
        K = self.params['KM']
        n = self.params['n']
        k_burst = self.params.get('k_burst', 0.1)
        repressors = [p[2], p[0], p[1]]
        rates = []
        for i in range(3):
            R = repressors[i]
            k_on = k_burst
            k_off = k_on * (R/K)**n
            rates.append((k_on, k_off))
        return rates

    def steppers(self, t_step):
        p = self.ode_state[6:9]
        rates = self.transition_rates(p)
        for i in range(3):
            k_on, k_off = rates[i]
            state = self.promoter_state[i]
            if state == 0:
                p_on = 1 - np.exp(-k_on * t_step)
                if np.random.rand() < p_on: self.promoter_state[i] = 1
            else:
                p_off = 1 - np.exp(-k_off * t_step)
                if np.random.rand() < p_off: self.promoter_state[i] = 0
                    
        def dxdt(x, t):
            m = x[0:3]
            u = x[3:6]
            p_ = x[6:9]
            dm = np.zeros(3)
            du = np.zeros(3)
            dp = np.zeros(3)
            for i in range(3):
                Si = self.promoter_state[i]
                dm[i] = self.params['k_trans'] * Si - self.params['k_deg_m'] * m[i] + self.params['k_leak']
                du[i] = self.params['k_transl'] * m[i] - self.params['k_fold'] * u[i] - self.params['k_deg_p'] * u[i]
                dp[i] = self.params['k_fold'] * u[i] - self.params['k_deg_p'] * p_[i]
            return np.concatenate([dm, du, dp])

        next_state = odeint(dxdt, self.ode_state, [0, t_step])[-1]
        self.ode_state = next_state
        self.time += t_step

        self.history_t.append(self.time)
        self.history_ode.append(self.ode_state)
        self.history_promoters.append(self.promoter_state.copy())

    def run(self, total_time, dt):
        # Record Initial State (t=0)
        self.history_t.append(self.time)
        self.history_ode.append(self.ode_state)
        self.history_promoters.append(self.promoter_state.copy())
        
        steps = int(total_time / dt)
        for _ in range(steps):
            self.steppers(dt)
        return np.array(self.history_t), np.array(self.history_ode), np.array(self.history_promoters)

params = {
    'k_trans': 0.5,
    'k_leak': 5e-4,
    'k_deg_m': np.log(2)/2, # Matches ODE
    'k_transl': 0.16,
    'k_fold': 1.0/60,
    'k_deg_p': np.log(2)/600,
    'n': 2.0,
    'KM': 40.0,
    'k_burst': 0.05
}

if __name__ == "__main__":
    model = MarkovianRepressilatorDDM(params)
    model.ode_state[0] = 5.0
    model.ode_state[3] = 48.0 # Initial u1
    model.ode_state[6] = 1000.0 # Initial cI (Matches ODE)
    dt = 0.01
    T = 10000
    print("Running Markovian DDM Repressilator...")
    t, ode, promoters = model.run(T, dt)

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(t/60, ode[:, 6], label='cI')
    plt.plot(t/60, ode[:, 7], label='LacI')
    plt.plot(t/60, ode[:, 8], label='TetR')
    plt.title('Markovian DDM Repressilator')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.step(t/60, promoters[:, 0] + 2.2, label='cI')
    plt.step(t/60, promoters[:, 1] + 1.1, label='LacI')
    plt.step(t/60, promoters[:, 2], label='TetR')
    plt.savefig('repressilator_ddm_markovian.png')

    # Save Data
    import pandas as pd
    df = pd.DataFrame({'Time': t, 'cI': ode[:, 6], 'LacI': ode[:, 7], 'TetR': ode[:, 8]})
    df.to_csv('../20251218_repressilator/repressilator_markovian_trajectories.csv', index=False)
    print("Simulation Complete.")
