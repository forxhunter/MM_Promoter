# NF-κB Pathway

## Network Structure

```
TNF → IKK → IκB:NF-κB → NF-κB (free)
                           ↓
                    Nuclear Import
                           ↓
                    NF-κB (nuclear) → IκBα mRNA
                           ↑              ↓
                           └──────────────┘
                        (negative feedback)
```

## Models

- **nfkb_ode.py** - Deterministic ODE model
- **nfkb_ssa.py** - Stochastic Simulation Algorithm
- **nfkb_markovian.py** - Hybrid Markovian-ODE model
- **validate_nfkb.py** - 3-way comparison script

## Key Features

- **Transient Response**: Nuclear NF-κB peaks at ~30 min
- **Negative Feedback**: IκB resynthesis within 60 min
- **Multi-Compartment**: Nuclear-cytoplasmic shuttling
- **Complex Formation**: IκB:NF-κB binding/unbinding

## Running

```bash
export PYTHONPATH=$PYTHONPATH:/data2/2026_GENE_MM
python3 validate_nfkb.py
```

Output: `nfkb_comparison.png` (saved to validation/plots/nfkb/)

## Parameters

- TNF pulse: 15 minutes
- Simulation time: 240 minutes (4 hours)
- dt = 0.1
- System volume (SSA): Ω = 50
