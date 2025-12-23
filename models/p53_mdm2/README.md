# p53-Mdm2 Oscillator

## Network Structure

```
DNA Damage → ATM → p53 → Mdm2
                ↑      ↓
                └──────┘
              (ubiquitination)
```

## Models

- **p53_ode.py** - Deterministic ODE model
- **p53_ssa.py** - Stochastic Simulation Algorithm
- **p53_markovian.py** - Hybrid Markovian-ODE model
- **validate_p53.py** - 3-way comparison script

## Key Features

- **Oscillations**: Damped oscillations with ~5-6 hour period
- **Cooperative Binding**: Mdm2 promoter has 2 p53 binding sites
- **Damage Response**: ATM activation triggers p53 pulses

## Running

```bash
export PYTHONPATH=$PYTHONPATH:/data2/2026_GENE_MM
python3 validate_p53.py
```

Output: `p53_comparison.png` (saved to validation/plots/p53/)

## Parameters

- Damage pulse at t=0 (1 hour duration)
- Simulation time: 40 hours
- dt = 0.01
- System volume (SSA): Ω = 100
