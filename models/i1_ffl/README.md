# Incoherent Feed-Forward Loop (I1-FFL)

## Network Structure

```
    X (Master)
   / \
  /   \
 ↓     ↓
Y      Z
 ↘   ↗
(represses)
```

## Models

- **i1_ffl_ode.py** - Deterministic ODE model with Hill functions
- **i1_ffl_ssa.py** - Stochastic Simulation Algorithm (ground truth)
- **i1_ffl_markovian.py** - Hybrid Markovian-ODE model
- **validate_ffl.py** - 3-way comparison script

## Key Features

- **Pulse Generation**: Z shows transient peak despite sustained X input
- **Multi-Input Promoter**: Z promoter regulated by X (activator) and Y (repressor)
- **Logic**: Z active only if X bound AND Y NOT bound

## Running

```bash
export PYTHONPATH=$PYTHONPATH:/data2/2026_GENE_MM
python3 validate_ffl.py
```

Output: `ffl_comparison.png` (saved to validation/plots/ffl/)

## Parameters

- X step input at t=10
- Simulation time: 50 time units
- dt = 0.05
